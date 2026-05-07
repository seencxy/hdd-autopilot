use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use mining_cuda_sys::{CudaDeviceInfo, CudaSolverConfig};

use mining_cuda_sys as cuda_sys;

use crate::backend::cpu::{ComputeJob, benchmark_job_for_tuning, compute_digest, hex_lower};
use crate::backend::types::{
    BackendDescriptor, BackendKind, BenchmarkResult, GPUAvailability, GpuBenchmarkConfig,
    GpuDeviceProfile, GpuMiningSessionConfig, MineBlockResult, MineResult,
    recommended_gpu_tuning_shapes,
};
use crate::error::{MiningError, interrupted_error};

pub(crate) const GPU_RUNTIME_BENCHMARK_DURATION: Duration = Duration::from_millis(750);
pub(crate) const GPU_DEVICE_SCREENING_DURATION: Duration = Duration::from_millis(250);
pub(crate) const GPU_FINALIST_COUNT: usize = 4;

#[derive(Debug, Clone, Default)]
pub struct CudaBackend;

pub struct CudaMiningSession {
    sessions: Vec<cuda_sys::CudaMiningSession>,
    job: ComputeJob,
    stop: Arc<AtomicBool>,
    cancel: Arc<AtomicBool>,
}

enum CudaWorkerMessage {
    Done,
    Found { nonce: usize, digest: String },
    Error(MiningError),
}

impl CudaMiningSession {
    pub fn mine_until_stop(&mut self) -> Result<MineBlockResult, MiningError> {
        if self.sessions.len() <= 1 {
            return self.mine_single_session();
        }
        self.mine_parallel_sessions()
    }

    fn mine_single_session(&mut self) -> Result<MineBlockResult, MiningError> {
        if self.sessions.is_empty() {
            return Ok(MineBlockResult {
                found: None,
                attempts: 0,
            });
        }
        loop {
            if self.cancel.load(Ordering::SeqCst) {
                self.stop.store(true, Ordering::SeqCst);
                return Err(interrupted_error());
            }
            if self.stop.load(Ordering::SeqCst) {
                return Ok(MineBlockResult {
                    found: None,
                    attempts: 0,
                });
            }
            let result = self
                .sessions
                .get_mut(0)
                .expect("single CUDA session should exist")
                .mine_next_batch()
                .map_err(MiningError::Message)?;
            if result.found {
                let nonce = result.nonce as usize;
                let expected_digest = hex_lower(&compute_digest(&self.job, nonce));
                if result.digest_hex != expected_digest {
                    return Err(MiningError::Message(
                        "CUDA 后端返回的摘要校验失败。".to_string(),
                    ));
                }
                self.stop.store(true, Ordering::SeqCst);
                return Ok(MineBlockResult {
                    found: Some(MineResult {
                        nonce,
                        digest: result.digest_hex,
                        attempts: result.attempts,
                    }),
                    attempts: result.attempts,
                });
            }
        }
    }

    fn mine_parallel_sessions(&mut self) -> Result<MineBlockResult, MiningError> {
        let sessions = std::mem::take(&mut self.sessions);
        let worker_count = sessions.len();
        let attempts = Arc::new(AtomicI64::new(0));
        let (sender, receiver) = mpsc::channel();
        let mut handles = Vec::with_capacity(worker_count);

        for mut session in sessions {
            let sender = sender.clone();
            let job = self.job.clone();
            let stop = Arc::clone(&self.stop);
            let cancel = Arc::clone(&self.cancel);
            let attempts = Arc::clone(&attempts);
            handles.push(thread::spawn(move || {
                let mut last_attempts = 0i64;
                loop {
                    if cancel.load(Ordering::SeqCst) {
                        stop.store(true, Ordering::SeqCst);
                        let _ = sender.send(CudaWorkerMessage::Error(interrupted_error()));
                        return;
                    }
                    if stop.load(Ordering::SeqCst) {
                        let _ = sender.send(CudaWorkerMessage::Done);
                        return;
                    }
                    let result = match session.mine_next_batch() {
                        Ok(result) => result,
                        Err(error) => {
                            let _ =
                                sender.send(CudaWorkerMessage::Error(MiningError::Message(error)));
                            return;
                        }
                    };
                    let delta = result.attempts.saturating_sub(last_attempts);
                    if delta > 0 {
                        attempts.fetch_add(delta, Ordering::Relaxed);
                        last_attempts = result.attempts;
                    }
                    if result.found {
                        let nonce = result.nonce as usize;
                        let expected_digest = hex_lower(&compute_digest(&job, nonce));
                        if result.digest_hex != expected_digest {
                            let _ = sender.send(CudaWorkerMessage::Error(MiningError::Message(
                                "CUDA 后端返回的摘要校验失败。".to_string(),
                            )));
                            return;
                        }
                        stop.store(true, Ordering::SeqCst);
                        let _ = sender.send(CudaWorkerMessage::Found {
                            nonce,
                            digest: result.digest_hex,
                        });
                        return;
                    }
                }
            }));
        }
        drop(sender);

        let mut completed = 0usize;
        let mut found: Option<(usize, String)> = None;
        let mut first_error: Option<MiningError> = None;
        while completed < worker_count {
            match receiver.recv() {
                Ok(CudaWorkerMessage::Done) => {
                    completed += 1;
                }
                Ok(CudaWorkerMessage::Found { nonce, digest }) => {
                    completed += 1;
                    if found.is_none() {
                        found = Some((nonce, digest));
                        self.stop.store(true, Ordering::SeqCst);
                    }
                }
                Ok(CudaWorkerMessage::Error(error)) => {
                    completed += 1;
                    self.stop.store(true, Ordering::SeqCst);
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

        let attempts = attempts.load(Ordering::Relaxed);
        if let Some((nonce, digest)) = found {
            return Ok(MineBlockResult {
                found: Some(MineResult {
                    nonce,
                    digest,
                    attempts,
                }),
                attempts,
            });
        }
        if let Some(error) = first_error {
            return Err(error);
        }
        Ok(MineBlockResult {
            found: None,
            attempts,
        })
    }
}

impl CudaBackend {
    pub fn new() -> Self {
        Self
    }

    pub fn solver_templates_for_descriptor(
        &self,
        descriptor: &BackendDescriptor,
        job: &ComputeJob,
    ) -> Vec<CudaSolverConfig> {
        recommended_gpu_tuning_shapes(descriptor.gpu_profile, job.memory_cost_kib, job.parallelism)
            .into_iter()
            .map(|shape| CudaSolverConfig {
                batch_size: shape.batch_size,
                by_segment: shape.by_segment,
                precompute_refs: shape.precompute_refs,
            })
            .collect()
    }

    pub fn descriptor_for_device(&self, device: &CudaDeviceInfo) -> BackendDescriptor {
        BackendDescriptor {
            kind: BackendKind::Cuda,
            name: device.name.clone(),
            device_id: device.device_id.clone(),
            device_index: Some(device.device_index),
            gpu_profile: Some(GpuDeviceProfile {
                global_memory_bytes: device.global_memory_bytes,
                max_alloc_bytes: device.max_alloc_bytes,
                compute_units: device.compute_units,
                max_threads_per_group: device.max_threads_per_block,
                local_memory_bytes: device.shared_memory_per_block_bytes,
                subgroup_size: device.warp_size,
                unified_memory: false,
                low_power: false,
                removable: false,
            }),
        }
    }

    pub fn list_devices(&self) -> Result<Vec<BackendDescriptor>, MiningError> {
        if !cuda_sys::is_supported_target() {
            return Ok(Vec::new());
        }
        let devices = cuda_sys::list_devices().map_err(MiningError::Message)?;
        Ok(devices
            .iter()
            .map(|device| self.descriptor_for_device(device))
            .collect())
    }

    pub fn detect_availability(&self) -> GPUAvailability {
        if !cuda_sys::is_supported_target() {
            return GPUAvailability {
                available: false,
                reason: "Current platform does not support the CUDA backend.".to_string(),
            };
        }
        match cuda_sys::is_available() {
            Ok(true) => GPUAvailability {
                available: true,
                reason: String::new(),
            },
            Ok(false) => GPUAvailability {
                available: false,
                reason: "当前平台未启用 CUDA 后端。".to_string(),
            },
            Err(error) => GPUAvailability {
                available: false,
                reason: format!("进程内 CUDA 后端不可用：{}", error),
            },
        }
    }

    pub fn default_solver_config_for_job(
        &self,
        descriptor: &BackendDescriptor,
        job: &ComputeJob,
    ) -> Result<CudaSolverConfig, MiningError> {
        let raw_job = cuda_sys::CudaJob {
            seed_bytes: &job.seed_bytes,
            pass_prefix: &job.pass_prefix,
            time_cost: job.time_cost,
            memory_cost_kib: job.memory_cost_kib,
            parallelism: job.parallelism,
            difficulty_bits: job.difficulty_bits,
        };
        cuda_sys::default_solver_config(descriptor.device_index.unwrap_or(0), &raw_job)
            .map_err(MiningError::Message)
    }

    pub fn quick_screen_benchmark_for_descriptor(
        &self,
        descriptor: &BackendDescriptor,
        job: &ComputeJob,
    ) -> Result<BenchmarkResult, MiningError> {
        let benchmark_job = benchmark_job_for_tuning(job);
        let default_config = self.default_solver_config_for_job(descriptor, &benchmark_job)?;
        self.run_runtime_loop_benchmark_for_device(
            descriptor.device_index.unwrap_or(0),
            &benchmark_job,
            default_config.batch_size,
            default_config.by_segment,
            default_config.precompute_refs,
            GPU_DEVICE_SCREENING_DURATION,
        )
    }

    pub fn find_best_benchmark_config_for_descriptor(
        &self,
        descriptor: &BackendDescriptor,
        job: &ComputeJob,
    ) -> Result<BenchmarkResult, MiningError> {
        let device_index = descriptor.device_index.unwrap_or(0);
        let cancel = Arc::new(AtomicBool::new(false));
        let mut best: Option<BenchmarkResult> = None;
        for candidate in self.solver_templates_for_descriptor(descriptor, job) {
            let Ok(result) = self.run_runtime_loop_benchmark_with_cancel(
                job,
                GpuBenchmarkConfig {
                    device_index,
                    batch_size: candidate.batch_size,
                    by_segment: candidate.by_segment,
                    precompute_refs: candidate.precompute_refs,
                    duration: GPU_RUNTIME_BENCHMARK_DURATION,
                },
                &cancel,
            ) else {
                continue;
            };
            if best
                .as_ref()
                .is_none_or(|current| result.attempts_per_s > current.attempts_per_s)
            {
                best = Some(result);
            }
        }
        best.ok_or_else(|| MiningError::Message("CUDA 自动调优没有得到可用结果。".to_string()))
    }

    pub fn run_runtime_loop_benchmark_for_device(
        &self,
        device_index: usize,
        job: &ComputeJob,
        batch_size: usize,
        by_segment: bool,
        precompute_refs: bool,
        duration: Duration,
    ) -> Result<BenchmarkResult, MiningError> {
        self.run_runtime_loop_benchmark_with_cancel(
            job,
            GpuBenchmarkConfig {
                device_index,
                batch_size,
                by_segment,
                precompute_refs,
                duration,
            },
            &Arc::new(AtomicBool::new(false)),
        )
    }

    pub fn run_runtime_loop_benchmark_with_cancel(
        &self,
        job: &ComputeJob,
        config: GpuBenchmarkConfig,
        cancel: &Arc<AtomicBool>,
    ) -> Result<BenchmarkResult, MiningError> {
        let GpuBenchmarkConfig {
            device_index,
            batch_size,
            by_segment,
            precompute_refs,
            duration,
        } = config;
        cuda_sys::validate().map_err(MiningError::Message)?;
        let config = CudaSolverConfig {
            batch_size: batch_size.max(1),
            by_segment,
            precompute_refs,
        };
        let benchmark_job = benchmark_job_for_tuning(job);
        let raw_job = cuda_sys::CudaJob {
            seed_bytes: &benchmark_job.seed_bytes,
            pass_prefix: &benchmark_job.pass_prefix,
            time_cost: benchmark_job.time_cost,
            memory_cost_kib: benchmark_job.memory_cost_kib,
            parallelism: benchmark_job.parallelism,
            difficulty_bits: benchmark_job.difficulty_bits,
        };
        let mut session = cuda_sys::create_session(device_index, &raw_job, config, 1)
            .map_err(MiningError::Message)?;
        let started = std::time::Instant::now();
        let mut attempts = 0i64;
        while started.elapsed() < duration {
            if cancel.load(Ordering::SeqCst) {
                return Err(interrupted_error());
            }
            let result = session.mine_next_batch().map_err(MiningError::Message)?;
            attempts = result.attempts;
        }
        let elapsed = started.elapsed();
        Ok(BenchmarkResult {
            workers: config.batch_size,
            concurrency: config.batch_size,
            by_segment: config.by_segment,
            precompute_refs: config.precompute_refs,
            attempts,
            elapsed,
            attempts_per_s: attempts as f64 / elapsed.as_secs_f64().max(0.001),
        })
    }

    pub fn start_mining_session(
        &self,
        job: &ComputeJob,
        config: GpuMiningSessionConfig,
        stop: &Arc<AtomicBool>,
        cancel: &Arc<AtomicBool>,
    ) -> Result<CudaMiningSession, MiningError> {
        let GpuMiningSessionConfig {
            device_index,
            batch_size,
            session_count,
            by_segment,
            precompute_refs,
            start_nonce,
            nonce_count,
        } = config;
        let session_count = session_count.max(1);
        let mut last_error = None;
        let mut sessions = Vec::new();
        for requested_sessions in (1..=session_count).rev() {
            match create_cuda_sessions(
                device_index,
                job,
                batch_size.max(1),
                requested_sessions,
                by_segment,
                precompute_refs,
                start_nonce,
                nonce_count,
            ) {
                Ok(created) => {
                    sessions = created;
                    break;
                }
                Err(error) => last_error = Some(error),
            }
        }
        if sessions.is_empty() {
            return Err(last_error.unwrap_or_else(|| {
                MiningError::Message("CUDA 持久计算会话创建失败。".to_string())
            }));
        }
        Ok(CudaMiningSession {
            sessions,
            job: job.clone(),
            stop: Arc::clone(stop),
            cancel: Arc::clone(cancel),
        })
    }
}

fn create_cuda_sessions(
    device_index: usize,
    job: &ComputeJob,
    batch_size: usize,
    session_count: usize,
    by_segment: bool,
    precompute_refs: bool,
    start_nonce: u64,
    nonce_count: u64,
) -> Result<Vec<cuda_sys::CudaMiningSession>, MiningError> {
    let raw_job = cuda_sys::CudaJob {
        seed_bytes: &job.seed_bytes,
        pass_prefix: &job.pass_prefix,
        time_cost: job.time_cost,
        memory_cost_kib: job.memory_cost_kib,
        parallelism: job.parallelism,
        difficulty_bits: job.difficulty_bits,
    };
    let raw_config = cuda_sys::CudaSolverConfig {
        batch_size,
        by_segment,
        precompute_refs,
    };
    let session_count = session_count.max(1);
    let span = nonce_count.max(session_count as u64) / session_count as u64;
    let mut sessions = Vec::with_capacity(session_count);
    for index in 0..session_count {
        let offset = span.saturating_mul(index as u64);
        let session_start = start_nonce.saturating_add(offset);
        sessions.push(
            cuda_sys::create_session(device_index, &raw_job, raw_config, session_start)
                .map_err(MiningError::Message)?,
        );
    }
    Ok(sessions)
}
