mod gpu;
mod support;

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use crate::backend::cpu::CpuMiningSession;
use crate::backend::types::{
    CpuMiningSessionConfig, GpuMiningSessionConfig, MineBlockResult, MineResult,
};
use crate::backend::{
    BackendKind, ComputeJob, CpuBackend, CudaBackend, MetalBackend, OpenclBackend,
    assign_nonce_ranges,
};
use crate::client::MiningClient;
use crate::error::is_interrupted_error;
use crate::{
    Config, MiningError, RewardKind, StatusResponse, code_type_label, humanize_duration,
    humanize_error, mode_description, preference_label, result_label, reward_benefit_label,
    reward_display_label,
};

use self::support::{
    BenchmarkKey, RoundStatus, SelectedBackend, append_reward_code, check_cancel,
    filter_candidates_for_params, load_persistent_benchmark_cache, save_persistent_benchmark_cache,
    select_backend_workers, sleep_with_cancel,
};

#[derive(Debug, Clone)]
pub(crate) struct Runner {
    config: Config,
    client: MiningClient,
    cancel: Arc<AtomicBool>,
    cpu_backend: CpuBackend,
    cuda_backend: CudaBackend,
    metal_backend: MetalBackend,
    opencl_backend: OpenclBackend,
    benchmark_cache: Arc<Mutex<HashMap<BenchmarkKey, Vec<SelectedBackend>>>>,
    backend_blacklist: Arc<Mutex<HashSet<(BackendKind, String)>>>,
}

#[derive(Debug)]
enum WorkerMessage {
    Result {
        backend: &'static str,
        result: Option<MineResult>,
    },
    Error {
        backend: &'static str,
        error: MiningError,
    },
}

enum BackendSession {
    Cpu(CpuMiningSession),
    Cuda(crate::backend::cuda::CudaMiningSession),
    Metal(crate::backend::metal::MetalMiningSession),
    Opencl(crate::backend::opencl::OpenclMiningSession),
}

impl BackendSession {
    fn mine_until_stop(&mut self) -> Result<MineBlockResult, MiningError> {
        match self {
            Self::Cpu(session) => session.wait_for_result(),
            Self::Cuda(session) => session.mine_until_stop(),
            Self::Metal(session) => session.mine_until_stop(),
            Self::Opencl(session) => session.mine_until_stop(),
        }
    }
}

fn join_candidate_thread(
    label: &str,
    handle: thread::JoinHandle<Result<Vec<SelectedBackend>, MiningError>>,
) -> Result<Vec<SelectedBackend>, MiningError> {
    handle
        .join()
        .map_err(|_| MiningError::Message(format!("{label} 自动调优线程异常退出。")))?
}

impl Runner {
    pub(crate) fn new(config: Config, cancel: Arc<AtomicBool>) -> Result<Self, MiningError> {
        let client = MiningClient::new(&config.base_url, config.http_timeout)?;
        let benchmark_cache =
            load_persistent_benchmark_cache(&config.benchmark_cache_file).unwrap_or_default();
        Ok(Self {
            config,
            client,
            cancel,
            cpu_backend: CpuBackend::new(),
            cuda_backend: CudaBackend::new(),
            metal_backend: MetalBackend::new(),
            opencl_backend: OpenclBackend::new(),
            benchmark_cache: Arc::new(Mutex::new(benchmark_cache)),
            backend_blacklist: Arc::new(Mutex::new(HashSet::new())),
        })
    }

    fn log(&self, args: fmt::Arguments<'_>) {
        self.config.output.line_fmt(args);
    }

    fn log_line(&self, line: impl AsRef<str>) {
        self.config.output.line(line);
    }

    pub(crate) fn run_auto_tuned(&self) -> Result<(), MiningError> {
        self.log_line("开始运行挖矿自动调优模式。");
        self.log(format_args!(
            "当前模式：{}。",
            mode_description(self.config.mode)
        ));
        self.print_output_paths();

        self.run_loop()
    }
    fn print_output_paths(&self) {
        match self.config.mode {
            crate::Mode::InviteOnly => {
                self.log(format_args!(
                    "邀请码会写入：{}",
                    self.config.invite_output_file.display()
                ));
            }
            crate::Mode::BalanceOnly => {
                self.log(format_args!(
                    "余额兑换码会写入：{}",
                    self.config.balance_output_file.display()
                ));
            }
            crate::Mode::InviteThenBalance | crate::Mode::BalanceThenInvite => {
                self.log(format_args!(
                    "邀请码会写入：{}",
                    self.config.invite_output_file.display()
                ));
                self.log(format_args!(
                    "余额兑换码会写入：{}",
                    self.config.balance_output_file.display()
                ));
            }
        }
    }

    fn collect_backend_candidates(
        &self,
        job: &ComputeJob,
    ) -> Result<Vec<SelectedBackend>, MiningError> {
        let cache_key = BenchmarkKey::from(job);
        if let Some(cached) = self
            .benchmark_cache
            .lock()
            .expect("benchmark cache poisoned")
            .get(&cache_key)
            .cloned()
        {
            self.log_line("使用已缓存的自动调优结果。");
            return Ok(self.filter_blacklisted(filter_candidates_for_params(cached, &cache_key)));
        }

        self.log_line("CPU 和 GPU 自动调优并行启动。");
        let cpu_runner = self.clone();
        let cpu_job = job.clone();
        let cpu_handle = thread::spawn(move || {
            let cpu_best = cpu_runner
                .cpu_backend
                .find_best_benchmark_config_with_cancel_and_output(
                    &cpu_job,
                    cpu_runner.config.thread_count,
                    &cpu_runner.cancel,
                    &cpu_runner.config.output,
                )?;
            cpu_runner.log(format_args!(
                "CPU 候选可用：线程数 {}，并发数 {}，预计速度约 {:.2} 次/秒。",
                cpu_best.workers, cpu_best.concurrency, cpu_best.attempts_per_s
            ));
            Ok(vec![SelectedBackend::new(
                cpu_runner.cpu_backend.descriptor(),
                cpu_best,
                BenchmarkKey::from(&cpu_job),
            )])
        });

        let gpu_runner = self.clone();
        let gpu_job = job.clone();
        let gpu_handle = thread::spawn(move || gpu_runner.collect_gpu_backend_candidates(&gpu_job));

        let cpu_candidates = join_candidate_thread("CPU", cpu_handle);
        let gpu_candidates = join_candidate_thread("GPU", gpu_handle);
        let mut candidates = cpu_candidates?;
        candidates.extend(gpu_candidates?);
        let candidates = filter_candidates_for_params(candidates, &cache_key);

        let cache_snapshot = {
            let mut cache = self
                .benchmark_cache
                .lock()
                .expect("benchmark cache poisoned");
            cache.insert(cache_key.clone(), candidates.clone());
            cache.clone()
        };
        if let Err(error) =
            save_persistent_benchmark_cache(&self.config.benchmark_cache_file, &cache_snapshot)
        {
            self.log(format_args!(
                "自动调优磁盘缓存写入失败：{}",
                humanize_error(&error)
            ));
        }

        Ok(self.filter_blacklisted(candidates))
    }

    fn run_loop(&self) -> Result<(), MiningError> {
        loop {
            self.check_cancel()?;
            match self.run_cycle() {
                Ok(()) => {
                    self.log_line("本轮已经命中，等待下一轮开放。");
                    self.sleep_with_cancel(self.config.success_delay)?;
                }
                Err(MiningError::RetryNow) => continue,
                Err(MiningError::DailyLimit) => {
                    self.log(format_args!(
                        "今天的命中次数已经用完，{}后再试。",
                        humanize_duration(self.config.daily_limit_delay)
                    ));
                    self.sleep_with_cancel(self.config.daily_limit_delay)?;
                }
                Err(MiningError::InventoryDepleted) => {
                    self.log(format_args!(
                        "这一轮的目标库存已经发完了，{}后再试。",
                        humanize_duration(self.config.inventory_depleted_delay)
                    ));
                    self.sleep_with_cancel(self.config.inventory_depleted_delay)?;
                }
                Err(error) => {
                    if is_interrupted_error(&error) {
                        return Err(error);
                    }
                    self.log(format_args!(
                        "这一轮没有顺利完成：{}。{}后自动重试。",
                        humanize_error(&error),
                        humanize_duration(self.config.retry_delay)
                    ));
                    self.sleep_with_cancel(self.config.retry_delay)?;
                }
            }
        }
    }

    fn start_backend_session(
        &self,
        backend: &SelectedBackend,
        job: &ComputeJob,
        start_nonce: u64,
        nonce_count: u64,
        stop_mining: &Arc<AtomicBool>,
    ) -> Result<BackendSession, MiningError> {
        match backend.kind {
            BackendKind::Cpu => {
                let start_nonce = usize::try_from(start_nonce)
                    .map_err(|_| MiningError::Message("nonce 超出 CPU 可处理范围。".to_string()))?;
                let nonce_count = usize::try_from(nonce_count)
                    .map_err(|_| MiningError::Message("CPU nonce 范围超出限制。".to_string()))?;
                Ok(BackendSession::Cpu(
                    self.cpu_backend.start_mining_session(
                        job,
                        CpuMiningSessionConfig {
                            workers: backend
                                .profile
                                .workers
                                .max(1)
                                .min(self.config.thread_count.max(1)),
                            concurrency: backend
                                .profile
                                .concurrency
                                .max(1)
                                .min(self.config.thread_count.max(1)),
                            start_nonce,
                            nonce_count,
                        },
                        stop_mining,
                        &self.cancel,
                    )?,
                ))
            }
            BackendKind::Cuda => Ok(BackendSession::Cuda(
                self.cuda_backend.start_mining_session(
                    job,
                    GpuMiningSessionConfig {
                        device_index: backend.device_index.unwrap_or(0),
                        batch_size: backend.profile.concurrency.max(1),
                        session_count: backend.recommended_gpu_session_count(job),
                        by_segment: backend.profile.by_segment,
                        precompute_refs: backend.profile.precompute_refs,
                        start_nonce,
                        nonce_count,
                    },
                    stop_mining,
                    &self.cancel,
                )?,
            )),
            BackendKind::Metal => Ok(BackendSession::Metal(
                self.metal_backend.start_mining_session(
                    job,
                    GpuMiningSessionConfig {
                        device_index: backend.device_index.unwrap_or(0),
                        batch_size: backend.profile.concurrency.max(1),
                        session_count: 1,
                        by_segment: backend.profile.by_segment,
                        precompute_refs: backend.profile.precompute_refs,
                        start_nonce,
                        nonce_count,
                    },
                    stop_mining,
                    &self.cancel,
                )?,
            )),
            BackendKind::Opencl => Ok(BackendSession::Opencl(
                self.opencl_backend.start_mining_session(
                    job,
                    GpuMiningSessionConfig {
                        device_index: backend.device_index.unwrap_or(0),
                        batch_size: backend.profile.concurrency.max(1),
                        session_count: 1,
                        by_segment: backend.profile.by_segment,
                        precompute_refs: backend.profile.precompute_refs,
                        start_nonce,
                        nonce_count,
                    },
                    stop_mining,
                    &self.cancel,
                )?,
            )),
        }
    }

    fn spawn_backend_worker(
        &self,
        backend: &SelectedBackend,
        job: &ComputeJob,
        start_nonce: u64,
        nonce_count: u64,
        stop_mining: &Arc<AtomicBool>,
        sender: &mpsc::Sender<WorkerMessage>,
    ) -> thread::JoinHandle<()> {
        let runner = self.clone();
        let backend = backend.clone();
        let job = job.clone();
        let stop_mining = Arc::clone(stop_mining);
        let sender = sender.clone();
        thread::spawn(move || {
            let label = backend.label;
            match runner.start_backend_session(
                &backend,
                &job,
                start_nonce,
                nonce_count,
                &stop_mining,
            ) {
                Ok(mut session) => match session.mine_until_stop() {
                    Ok(block_result) => {
                        let _ = sender.send(WorkerMessage::Result {
                            backend: label,
                            result: block_result.found,
                        });
                    }
                    Err(error) => {
                        let _ = sender.send(WorkerMessage::Error {
                            backend: label,
                            error,
                        });
                    }
                },
                Err(error) => {
                    let _ = sender.send(WorkerMessage::Error {
                        backend: label,
                        error,
                    });
                }
            }
        })
    }

    fn run_selected_workers(
        &self,
        workers: &[SelectedBackend],
        job: &ComputeJob,
        stop_mining: &Arc<AtomicBool>,
    ) -> Result<Option<MineResult>, MiningError> {
        let ranges = assign_nonce_ranges(workers.len())?;
        let (sender, receiver) = mpsc::channel();
        let mut handles = Vec::new();
        for (worker, (start_nonce, nonce_count)) in workers.iter().zip(ranges) {
            self.run_backend_self_test(worker, job)?;
            self.log(format_args!(
                "启动 {} 持久计算会话：nonce 起点 {}，区间长度 {}，会话数 {}。",
                worker.label,
                start_nonce,
                nonce_count,
                worker.recommended_gpu_session_count(job)
            ));
            handles.push(self.spawn_backend_worker(
                worker,
                job,
                start_nonce,
                nonce_count,
                stop_mining,
                &sender,
            ));
        }
        drop(sender);

        let mut completed = 0usize;
        let mut first_error: Option<MiningError> = None;
        let mut best_result: Option<MineResult> = None;
        while completed < workers.len() {
            match receiver.recv() {
                Ok(WorkerMessage::Result { backend, result }) => {
                    completed += 1;
                    if best_result.is_none()
                        && let Some(result) = result
                    {
                        self.log(format_args!("{} 后端率先命中。", backend));
                        best_result = Some(result);
                        stop_mining.store(true, Ordering::SeqCst);
                    }
                }
                Ok(WorkerMessage::Error { backend, error }) => {
                    completed += 1;
                    if is_interrupted_error(&error) {
                        stop_mining.store(true, Ordering::SeqCst);
                        for handle in handles {
                            let _ = handle.join();
                        }
                        return Err(error);
                    }
                    self.log(format_args!(
                        "{} 后端运行失败：{}",
                        backend,
                        humanize_error(&error)
                    ));
                    if first_error.is_none() {
                        first_error = Some(error);
                    }
                }
                Err(_) => break,
            }
        }

        for handle in handles {
            let _ = handle.join();
        }

        if let Some(result) = best_result {
            return Ok(Some(result));
        }
        if let Some(error) = first_error {
            return Err(error);
        }
        Ok(None)
    }

    fn run_cycle(&self) -> Result<(), MiningError> {
        self.check_cancel()?;
        self.log_line("获取矿池状态...");
        let status = self.client.get_status().map_err(|error| match error {
            MiningError::PoolDisabled | MiningError::NoOpenRound => error,
            other => MiningError::Message(format!("获取状态失败: {}", humanize_error(&other))),
        })?;

        let invite_remaining = status.invite_inventory_remaining();
        let balance_remaining = status.balance_inventory_remaining();
        let round = status
            .current_round
            .as_ref()
            .expect("open round should exist after get_status");
        self.log(format_args!(
            "当前轮次 #{}，难度 {}，剩余邀请码 {}，剩余余额兑换码 {}",
            round.id, round.difficulty_bits, invite_remaining, balance_remaining
        ));

        let reward = self
            .select_reward_kind(&status)
            .ok_or(MiningError::InventoryDepleted)?;
        if reward.preference == "balance"
            && invite_remaining <= 0
            && !matches!(self.config.mode, crate::Mode::BalanceOnly)
        {
            self.log_line("邀请码库存为 0，切换到余额兑换码");
        }
        if reward.preference == "invite"
            && balance_remaining <= 0
            && matches!(self.config.mode, crate::Mode::InviteThenBalance)
        {
            self.log_line("余额兑换码库存为 0，继续尝试邀请码");
        }
        self.log(format_args!("本轮选择：{}", reward.name));
        self.log_line("获取挑战...");

        let challenge = self.client.get_challenge().map_err(|error| match error {
            MiningError::DailyLimit => error,
            other => MiningError::Message(format!("获取挑战失败: {}", humanize_error(&other))),
        })?;
        self.log(format_args!(
            "挑战 #{}，轮次 #{}，难度 {}，time_cost {}，memory_cost {} MiB，parallelism {}",
            challenge.challenge_id,
            challenge.round_id,
            challenge.difficulty_bits,
            challenge.time_cost,
            challenge.memory_cost_mb,
            challenge.parallelism
        ));

        let job = ComputeJob::from(&challenge);
        let run_candidates = self.collect_backend_candidates(&job)?;
        let params_key = BenchmarkKey::from(&job);
        let selected_workers = select_backend_workers(&run_candidates, &params_key);
        if selected_workers.is_empty() {
            return Err(MiningError::Message(
                "当前计算参数下没有可用后端。".to_string(),
            ));
        }
        let has_cpu = selected_workers
            .iter()
            .any(|worker| worker.kind == BackendKind::Cpu);
        let has_gpu = selected_workers
            .iter()
            .any(|worker| worker.kind != BackendKind::Cpu);
        if selected_workers.len() == 1 {
            let selected = &selected_workers[0];
            let memory_detail = selected
                .estimated_gpu_memory_label(&job)
                .map(|label| format!("，预计显存 {}", label))
                .unwrap_or_default();
            self.log(format_args!(
                "本轮使用 {} 后端：{}{}，预计速度约 {} 次/秒。",
                selected.label,
                selected.selection_detail(),
                memory_detail,
                selected.speed_label()
            ));
        } else {
            let labels = selected_workers
                .iter()
                .map(|worker| {
                    let memory_detail = worker
                        .estimated_gpu_memory_label(&job)
                        .map(|label| format!("，预计显存 {}", label))
                        .unwrap_or_default();
                    format!(
                        "{}({}{}，约 {} 次/秒)",
                        worker.label,
                        worker.selection_detail(),
                        memory_detail,
                        worker.speed_label()
                    )
                })
                .collect::<Vec<_>>()
                .join(" + ");
            self.log(format_args!("本轮并发使用后端：{}。", labels));
        }
        if has_cpu && !has_gpu {
            self.log_line("GPU 未参与，本轮仅 CPU 挖矿。");
        }
        if !has_cpu && has_gpu {
            self.log_line("本轮仅使用 GPU 后端挖矿。");
        }

        let stop_mining = Arc::new(AtomicBool::new(false));
        let (status_sender, status_receiver) = mpsc::channel();
        let heartbeat_cancel = Arc::clone(&self.cancel);
        let heartbeat_stop = Arc::clone(&stop_mining);
        let heartbeat_client = self.client.clone();
        let target_name = reward.name.to_string();
        let round_status_poll_interval = self.config.round_status_poll_interval;
        let heartbeat_interval = self.config.heartbeat_interval;
        let reward_for_thread = reward.clone();
        let challenge_for_thread = challenge.clone();
        let output = self.config.output.clone();

        let heartbeat_handle = thread::spawn(move || {
            let mut next_heartbeat = Instant::now() + heartbeat_interval;
            let mut next_status_check = Instant::now() + round_status_poll_interval;
            loop {
                if heartbeat_cancel.load(Ordering::SeqCst) || heartbeat_stop.load(Ordering::SeqCst)
                {
                    return;
                }
                let now = Instant::now();
                if now >= next_heartbeat {
                    match heartbeat_client.heartbeat(
                        challenge_for_thread.challenge_id,
                        challenge_for_thread.round_id,
                    ) {
                        Ok(_) => {}
                        Err(MiningError::RoundClosed) => {
                            let _ = status_sender.send(RoundStatus {
                                round_closed: true,
                                ..RoundStatus::default()
                            });
                            heartbeat_stop.store(true, Ordering::SeqCst);
                            return;
                        }
                        Err(error) => {
                            if !is_interrupted_error(&error) {
                                output
                                    .line_fmt(format_args!("心跳失败：{}", humanize_error(&error)));
                            }
                        }
                    }
                    next_heartbeat = now + heartbeat_interval;
                }
                if now >= next_status_check {
                    match heartbeat_client.get_status_snapshot() {
                        Ok(status) => {
                            let current_round = status.current_round.as_ref();
                            let inventory_remaining = match reward_for_thread.preference {
                                "invite" => status.invite_inventory_remaining(),
                                "balance" => status.balance_inventory_remaining(),
                                _ => 0,
                            };
                            let round_status = if !status.enabled
                                || current_round.is_none_or(|round| !round.is_open())
                                || current_round
                                    .is_some_and(|round| round.id != challenge_for_thread.round_id)
                            {
                                RoundStatus {
                                    round_closed: true,
                                    ..RoundStatus::default()
                                }
                            } else if inventory_remaining <= 0 {
                                RoundStatus {
                                    inventory_depleted: true,
                                    ..RoundStatus::default()
                                }
                            } else if status.daily_limit_reached() {
                                RoundStatus {
                                    daily_limit: true,
                                    ..RoundStatus::default()
                                }
                            } else {
                                RoundStatus::default()
                            };
                            if round_status.inventory_depleted {
                                output
                                    .line_fmt(format_args!("{}已耗尽，停止当前挖矿", target_name));
                                let _ = status_sender.send(round_status);
                                heartbeat_stop.store(true, Ordering::SeqCst);
                                return;
                            }
                            if round_status.round_closed {
                                output.line("轮次已变更，停止挖矿");
                                let _ = status_sender.send(round_status);
                                heartbeat_stop.store(true, Ordering::SeqCst);
                                return;
                            }
                            if round_status.daily_limit {
                                output.line("今日命中次数已达上限");
                                let _ = status_sender.send(round_status);
                                heartbeat_stop.store(true, Ordering::SeqCst);
                                return;
                            }
                        }
                        Err(error) => {
                            if !is_interrupted_error(&error) {
                                output.line_fmt(format_args!(
                                    "轮次状态检查失败：{}",
                                    humanize_error(&error)
                                ));
                            }
                        }
                    }
                    next_status_check = now + round_status_poll_interval;
                }
                thread::sleep(Duration::from_millis(50));
            }
        });

        let mining_result = self.run_selected_workers(&selected_workers, &job, &stop_mining);

        stop_mining.store(true, Ordering::SeqCst);
        let _ = heartbeat_handle.join();

        let status_update = status_receiver.try_iter().last().unwrap_or_default();
        let mining_result = mining_result?;
        let Some(result) = mining_result else {
            if status_update.daily_limit {
                return Err(MiningError::DailyLimit);
            }
            if status_update.inventory_depleted {
                return Err(MiningError::RetryNow);
            }
            if status_update.round_closed {
                self.log_line("轮次已被别人命中，切换新轮次...");
            }
            return Ok(());
        };

        self.log(format_args!(
            "找到解！nonce={}, digest={}, 总尝试 {} 次",
            result.nonce, result.digest, result.attempts
        ));
        self.log_line("提交结果...");
        let submit = self
            .client
            .submit(
                challenge.challenge_id,
                challenge.round_id,
                result.nonce,
                &result.digest,
                reward.preference,
            )
            .map_err(|error| {
                MiningError::Message(format!("提交失败: {}", humanize_error(&error)))
            })?;
        let actual_code_type = code_type_label(&submit.code_type);
        let actual_reward_label =
            reward_display_label(&submit.code_type, submit.balance_amount, submit.concurrency);
        let reward_benefit =
            reward_benefit_label(&submit.code_type, submit.balance_amount, submit.concurrency);
        self.log(format_args!(
            "提交结果已经返回：这次想要的是{}，实际拿到的是{}，结果是{}，余额面额 {:.2}，并发增加 {}，奖励编号 {}",
            preference_label(reward.preference),
            actual_reward_label,
            result_label(&submit.result),
            submit.balance_amount,
            submit.concurrency,
            submit.reward_code_id
        ));
        if submit.result.trim().eq_ignore_ascii_case("late")
            || submit.result.trim().eq_ignore_ascii_case("round_closed")
        {
            self.log_line("提交太晚，轮次已关闭");
            return Ok(());
        }
        if submit
            .result
            .trim()
            .eq_ignore_ascii_case("daily win limit reached")
        {
            return Err(MiningError::DailyLimit);
        }
        if !submit.reward_code.trim().is_empty() {
            if reward_benefit.is_empty() {
                self.log(format_args!(
                    "命中了{}：{}",
                    actual_code_type, submit.reward_code
                ));
            } else {
                self.log(format_args!(
                    "命中了{}：{}，{}",
                    actual_code_type, submit.reward_code, reward_benefit
                ));
            }
            append_reward_code(
                &reward.output_path,
                reward.name,
                &actual_reward_label,
                &submit.reward_code,
            )?;
            self.log(format_args!(
                "{}已保存到 {}: {}",
                actual_reward_label,
                reward.output_path.display(),
                submit.reward_code
            ));
        }
        self.client.reset_session().map_err(|error| {
            MiningError::Message(format!("重建网络客户端失败: {}", humanize_error(&error)))
        })?;
        self.log_line("网络客户端已经重建，旧登录会话不会再用了。");
        if status_update.daily_limit {
            return Err(MiningError::DailyLimit);
        }
        Ok(())
    }

    fn select_reward_kind(&self, status: &StatusResponse) -> Option<RewardKind> {
        self.config
            .reward_kinds()
            .into_iter()
            .find(|kind| match kind.preference {
                "invite" => status.invite_inventory_remaining() > 0,
                "balance" => status.balance_inventory_remaining() > 0,
                _ => false,
            })
    }

    fn check_cancel(&self) -> Result<(), MiningError> {
        check_cancel(&self.cancel)
    }

    fn sleep_with_cancel(&self, wait: Duration) -> Result<(), MiningError> {
        sleep_with_cancel(&self.cancel, wait)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc;

    use super::WorkerMessage;
    use crate::backend::types::MineResult;

    #[test]
    fn worker_message_channel_carries_backend_metadata() {
        let (sender, receiver) = mpsc::channel();

        sender
            .send(WorkerMessage::Result {
                backend: "CPU",
                result: Some(MineResult {
                    nonce: 7,
                    digest: "abc".to_string(),
                    attempts: 11,
                }),
            })
            .expect("send result");
        sender
            .send(WorkerMessage::Error {
                backend: "CUDA",
                error: crate::MiningError::RetryNow,
            })
            .expect("send error");
        drop(sender);

        match receiver.recv().expect("first message") {
            WorkerMessage::Result { backend, result } => {
                assert_eq!(backend, "CPU");
                let result = result.expect("mine result");
                assert_eq!(result.nonce, 7);
                assert_eq!(result.digest, "abc");
                assert_eq!(result.attempts, 11);
            }
            other => panic!("unexpected message: {other:?}"),
        }

        match receiver.recv().expect("second message") {
            WorkerMessage::Error { backend, error } => {
                assert_eq!(backend, "CUDA");
                assert!(matches!(error, crate::MiningError::RetryNow));
            }
            other => panic!("unexpected message: {other:?}"),
        }
    }
}
