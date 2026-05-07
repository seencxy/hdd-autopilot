use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::backend::types::estimated_argon2_batch_memory_bytes;
use crate::backend::{BackendDescriptor, BackendKind, BenchmarkResult, ComputeJob};
use crate::{ChallengeResponse, MiningError};

#[derive(Debug, Clone)]
pub(super) struct SelectedBackend {
    pub(super) kind: BackendKind,
    pub(super) label: &'static str,
    pub(super) name: String,
    pub(super) device_id: String,
    pub(super) device_index: Option<usize>,
    pub(super) params_key: BenchmarkKey,
    pub(super) profile: BenchmarkResult,
}

impl SelectedBackend {
    pub(super) fn new(
        descriptor: &BackendDescriptor,
        profile: BenchmarkResult,
        params_key: BenchmarkKey,
    ) -> Self {
        Self {
            kind: descriptor.kind,
            label: match descriptor.kind {
                BackendKind::Cpu => "CPU",
                BackendKind::Cuda => "CUDA",
                BackendKind::Metal => "Metal",
                BackendKind::Opencl => "OpenCL",
            },
            name: descriptor.name.clone(),
            device_id: descriptor.device_id.clone(),
            device_index: descriptor.device_index,
            params_key,
            profile,
        }
    }

    pub(super) fn selection_detail(&self) -> String {
        match self.kind {
            BackendKind::Cpu => format!(
                "线程数 {}，并发数 {}",
                self.profile.workers.max(1),
                self.profile.concurrency.max(1)
            ),
            BackendKind::Cuda | BackendKind::Metal | BackendKind::Opencl => format!(
                "批大小 {}，按分段 {}，预计算参考值 {}",
                self.profile.workers.max(1),
                localized_bool(self.profile.by_segment),
                localized_bool(self.profile.precompute_refs)
            ),
        }
    }

    pub(super) fn speed_label(&self) -> String {
        format!("{:.2}", self.profile.attempts_per_s)
    }

    pub(super) fn estimated_gpu_memory_label(&self, job: &ComputeJob) -> Option<String> {
        if self.kind == BackendKind::Cpu {
            return None;
        }
        Some(format_memory_bytes(estimated_argon2_batch_memory_bytes(
            job.memory_cost_kib,
            self.profile.concurrency,
        )))
    }
}

pub(super) fn select_best_backend_by_kind(
    candidates: &[SelectedBackend],
    kind: BackendKind,
    params_key: &BenchmarkKey,
) -> Option<SelectedBackend> {
    candidates
        .iter()
        .filter(|candidate| candidate.kind == kind && &candidate.params_key == params_key)
        .cloned()
        .max_by(|left, right| {
            left.profile
                .attempts_per_s
                .total_cmp(&right.profile.attempts_per_s)
        })
}

pub(super) fn select_backend_workers(
    candidates: &[SelectedBackend],
    params_key: &BenchmarkKey,
) -> Vec<SelectedBackend> {
    let mut selected = Vec::new();
    if let Some(cpu) = select_best_backend_by_kind(candidates, BackendKind::Cpu, params_key) {
        selected.push(cpu);
    }
    let mut gpu_candidates = candidates
        .iter()
        .filter(|candidate| {
            candidate.kind != BackendKind::Cpu && &candidate.params_key == params_key
        })
        .cloned()
        .collect::<Vec<_>>();
    gpu_candidates.sort_by(|left, right| {
        right
            .profile
            .attempts_per_s
            .total_cmp(&left.profile.attempts_per_s)
    });
    for gpu in gpu_candidates {
        if selected
            .iter()
            .any(|existing| is_duplicate_gpu_backend(existing, &gpu))
        {
            continue;
        }
        selected.push(gpu);
    }
    selected
}

pub(super) fn filter_candidates_for_params(
    candidates: Vec<SelectedBackend>,
    params_key: &BenchmarkKey,
) -> Vec<SelectedBackend> {
    candidates
        .into_iter()
        .filter(|candidate| &candidate.params_key == params_key)
        .collect()
}

fn is_duplicate_gpu_backend(left: &SelectedBackend, right: &SelectedBackend) -> bool {
    if left.kind == BackendKind::Cpu || right.kind == BackendKind::Cpu {
        return false;
    }
    if left.kind == right.kind {
        return left.device_id == right.device_id;
    }
    let left_name = normalized_gpu_name(left);
    let right_name = normalized_gpu_name(right);
    !left_name.is_empty() && left_name == right_name
}

fn normalized_gpu_name(candidate: &SelectedBackend) -> String {
    let raw = candidate.name.trim();
    let without_opencl_wrapper = raw
        .strip_prefix("OpenCL Device '")
        .and_then(|rest| rest.split_once('\'').map(|(name, _)| name))
        .unwrap_or(raw);
    let without_suffix = without_opencl_wrapper
        .split(" [")
        .next()
        .unwrap_or(without_opencl_wrapper)
        .split(" @ ")
        .next()
        .unwrap_or(without_opencl_wrapper);
    without_suffix
        .chars()
        .flat_map(char::to_lowercase)
        .filter(|ch| ch.is_alphanumeric())
        .collect()
}

#[derive(Debug, Clone, Copy, Default)]
pub(super) struct RoundStatus {
    pub(super) round_closed: bool,
    pub(super) daily_limit: bool,
    pub(super) inventory_depleted: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) struct BenchmarkKey {
    pub(super) seed_bytes: Vec<u8>,
    pub(super) pass_prefix: Vec<u8>,
    pub(super) memory_cost_kib: u32,
    pub(super) time_cost: u32,
    pub(super) parallelism: u32,
    pub(super) difficulty_bits: i32,
}

impl From<&ComputeJob> for BenchmarkKey {
    fn from(job: &ComputeJob) -> Self {
        Self {
            seed_bytes: job.seed_bytes.clone(),
            pass_prefix: job.pass_prefix.clone(),
            memory_cost_kib: job.memory_cost_kib,
            time_cost: job.time_cost,
            parallelism: job.parallelism,
            difficulty_bits: job.difficulty_bits,
        }
    }
}

impl From<&ChallengeResponse> for ComputeJob {
    fn from(challenge: &ChallengeResponse) -> Self {
        let parallelism = challenge.parallelism as u8;
        Self {
            seed_bytes: challenge.seed.as_bytes().to_vec(),
            pass_prefix: format!(
                "{}:{}:{}:{}:{}:",
                challenge.seed,
                challenge.round_id,
                challenge.visitor_id,
                challenge.challenge_id,
                challenge.session_salt
            )
            .into_bytes(),
            time_cost: challenge.time_cost as u32,
            memory_cost_kib: (challenge.memory_cost_mb as u32).wrapping_mul(1024),
            parallelism: if parallelism == 0 {
                1
            } else {
                parallelism as u32
            },
            difficulty_bits: challenge.difficulty_bits,
        }
    }
}

pub(super) fn current_unix_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis().min(i64::MAX as u128) as i64)
        .unwrap_or(0)
}

pub(super) fn localized_bool(value: bool) -> &'static str {
    if value { "是" } else { "否" }
}

pub(super) fn format_memory_bytes(bytes: u128) -> String {
    const MIB: u128 = 1024 * 1024;
    const GIB: u128 = 1024 * MIB;
    if bytes >= GIB {
        format!("{:.1} GiB", bytes as f64 / GIB as f64)
    } else {
        format!("{} MiB", bytes.div_ceil(MIB))
    }
}

pub(super) fn append_reward_code(
    path: &Path,
    requested_label: &str,
    actual_label: &str,
    code: &str,
) -> Result<(), MiningError> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    writeln!(
        file,
        "[{}] 已保存{}（实际发放{}）：{}",
        format_log_time(current_unix_ms()),
        requested_label,
        actual_label,
        code
    )?;
    Ok(())
}

fn format_log_time(when_unix_ms: i64) -> String {
    let when = system_local_datetime(when_unix_ms);
    format!(
        "{:04}-{:02}-{:02} {:02}:{:02}:{:02}",
        when.year(),
        u8::from(when.month()),
        when.day(),
        when.hour(),
        when.minute(),
        when.second()
    )
}

fn system_local_datetime(when_unix_ms: i64) -> time::OffsetDateTime {
    let when_unix_ms = if when_unix_ms > 0 {
        when_unix_ms
    } else {
        current_unix_ms()
    };
    let utc = time::OffsetDateTime::from_unix_timestamp_nanos(
        i128::from(when_unix_ms).saturating_mul(1_000_000),
    )
    .unwrap_or_else(|_| time::OffsetDateTime::from_unix_timestamp(0).unwrap());
    let offset = time::UtcOffset::local_offset_at(utc).unwrap_or(time::UtcOffset::UTC);
    utc.to_offset(offset)
}

pub(super) fn check_cancel(cancel: &AtomicBool) -> Result<(), MiningError> {
    if cancel.load(Ordering::SeqCst) {
        return Err(crate::error::interrupted_error());
    }
    Ok(())
}

pub(super) fn sleep_with_cancel(cancel: &AtomicBool, wait: Duration) -> Result<(), MiningError> {
    if wait <= Duration::ZERO {
        return Ok(());
    }
    let started = Instant::now();
    loop {
        check_cancel(cancel)?;
        let elapsed = started.elapsed();
        if elapsed >= wait {
            return Ok(());
        }
        let remaining = wait - elapsed;
        thread::sleep(remaining.min(Duration::from_millis(100)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params_key() -> BenchmarkKey {
        BenchmarkKey {
            seed_bytes: b"seed-a".to_vec(),
            pass_prefix: b"seed-a:1:visitor:1:salt:".to_vec(),
            memory_cost_kib: 64 * 1024,
            time_cost: 1,
            parallelism: 1,
            difficulty_bits: 12,
        }
    }

    fn backend(
        kind: BackendKind,
        device_id: &str,
        attempts_per_s: f64,
        workers: usize,
    ) -> SelectedBackend {
        SelectedBackend {
            kind,
            label: match kind {
                BackendKind::Cpu => "CPU",
                BackendKind::Cuda => "CUDA",
                BackendKind::Metal => "Metal",
                BackendKind::Opencl => "OpenCL",
            },
            name: device_id.to_string(),
            device_id: device_id.to_string(),
            device_index: match kind {
                BackendKind::Cpu => None,
                BackendKind::Cuda | BackendKind::Metal | BackendKind::Opencl => Some(0),
            },
            params_key: params_key(),
            profile: BenchmarkResult {
                workers,
                concurrency: workers,
                by_segment: false,
                precompute_refs: false,
                attempts: 0,
                elapsed: Duration::from_secs(1),
                attempts_per_s,
            },
        }
    }

    #[test]
    fn select_backend_workers_picks_best_cpu_and_all_distinct_gpus() {
        let cpu_slow = backend(BackendKind::Cpu, "cpu:slow", 100.0, 4);
        let cpu_fast = backend(BackendKind::Cpu, "cpu:fast", 180.0, 8);
        let cuda = backend(BackendKind::Cuda, "cuda:0", 250.0, 4096);
        let metal = backend(BackendKind::Metal, "metal:0", 220.0, 2048);

        let selected = select_backend_workers(&[cpu_slow, cpu_fast, cuda, metal], &params_key());

        assert_eq!(selected.len(), 3);
        assert_eq!(selected[0].kind, BackendKind::Cpu);
        assert_eq!(selected[0].device_id, "cpu:fast");
        assert_eq!(selected[1].kind, BackendKind::Cuda);
        assert_eq!(selected[1].device_id, "cuda:0");
        assert_eq!(selected[2].kind, BackendKind::Metal);
        assert_eq!(selected[2].device_id, "metal:0");
    }

    #[test]
    fn select_backend_workers_dedupes_same_gpu_across_apis() {
        let cpu = backend(BackendKind::Cpu, "cpu:fast", 180.0, 8);
        let mut cuda = backend(BackendKind::Cuda, "cuda:0", 260.0, 4096);
        cuda.name = "NVIDIA GeForce RTX 4090".to_string();
        let mut opencl = backend(BackendKind::Opencl, "opencl:0", 230.0, 4096);
        opencl.name =
            "OpenCL Device 'NVIDIA GeForce RTX 4090' (NVIDIA) [GPU] @ NVIDIA CUDA".to_string();

        let selected = select_backend_workers(&[cpu, opencl, cuda], &params_key());

        assert_eq!(selected.len(), 2);
        assert_eq!(selected[0].kind, BackendKind::Cpu);
        assert_eq!(selected[1].kind, BackendKind::Cuda);
    }

    #[test]
    fn select_backend_workers_filters_mismatched_http_params() {
        let cpu = backend(BackendKind::Cpu, "cpu:fast", 180.0, 8);
        let mut stale_gpu = backend(BackendKind::Cuda, "cuda:stale", 1_000.0, 4096);
        stale_gpu.params_key = BenchmarkKey {
            memory_cost_kib: 128 * 1024,
            ..params_key()
        };
        let current_gpu = backend(BackendKind::Opencl, "opencl:current", 220.0, 2048);

        let selected = select_backend_workers(&[stale_gpu, current_gpu], &params_key());
        let filtered = filter_candidates_for_params(vec![cpu], &params_key());

        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].kind, BackendKind::Opencl);
        assert_eq!(filtered.len(), 1);
    }

    #[test]
    fn benchmark_key_uses_difficulty_bits() {
        let left = ComputeJob {
            seed_bytes: b"seed-a".to_vec(),
            pass_prefix: b"prefix-a:".to_vec(),
            time_cost: 1,
            memory_cost_kib: 64 * 1024,
            parallelism: 1,
            difficulty_bits: 12,
        };
        let right = ComputeJob {
            seed_bytes: b"seed-a".to_vec(),
            pass_prefix: b"prefix-a:".to_vec(),
            time_cost: 1,
            memory_cost_kib: 64 * 1024,
            parallelism: 1,
            difficulty_bits: 24,
        };

        assert_ne!(BenchmarkKey::from(&left), BenchmarkKey::from(&right));
    }

    #[test]
    fn benchmark_key_uses_http_seed_and_pass_prefix() {
        let left = ComputeJob {
            seed_bytes: b"seed-a".to_vec(),
            pass_prefix: b"seed-a:1:visitor:1:salt:".to_vec(),
            time_cost: 1,
            memory_cost_kib: 64 * 1024,
            parallelism: 1,
            difficulty_bits: 12,
        };
        let right = ComputeJob {
            seed_bytes: b"seed-b".to_vec(),
            pass_prefix: b"seed-b:1:visitor:2:salt:".to_vec(),
            time_cost: 1,
            memory_cost_kib: 64 * 1024,
            parallelism: 1,
            difficulty_bits: 12,
        };

        assert_ne!(BenchmarkKey::from(&left), BenchmarkKey::from(&right));
    }

    #[test]
    fn benchmark_key_changes_with_memory_cost() {
        let left = ComputeJob {
            seed_bytes: b"seed-a".to_vec(),
            pass_prefix: b"prefix-a:".to_vec(),
            time_cost: 1,
            memory_cost_kib: 64 * 1024,
            parallelism: 1,
            difficulty_bits: 12,
        };
        let right = ComputeJob {
            seed_bytes: b"seed-a".to_vec(),
            pass_prefix: b"prefix-a:".to_vec(),
            time_cost: 1,
            memory_cost_kib: 128 * 1024,
            parallelism: 1,
            difficulty_bits: 12,
        };

        assert_ne!(BenchmarkKey::from(&left), BenchmarkKey::from(&right));
    }

    #[test]
    fn benchmark_key_changes_with_time_cost() {
        let left = ComputeJob {
            seed_bytes: b"seed-a".to_vec(),
            pass_prefix: b"prefix-a:".to_vec(),
            time_cost: 1,
            memory_cost_kib: 64 * 1024,
            parallelism: 1,
            difficulty_bits: 12,
        };
        let right = ComputeJob {
            seed_bytes: b"seed-a".to_vec(),
            pass_prefix: b"prefix-a:".to_vec(),
            time_cost: 2,
            memory_cost_kib: 64 * 1024,
            parallelism: 1,
            difficulty_bits: 12,
        };

        assert_ne!(BenchmarkKey::from(&left), BenchmarkKey::from(&right));
    }

    #[test]
    fn benchmark_key_changes_with_parallelism() {
        let left = ComputeJob {
            seed_bytes: b"seed-a".to_vec(),
            pass_prefix: b"prefix-a:".to_vec(),
            time_cost: 1,
            memory_cost_kib: 64 * 1024,
            parallelism: 1,
            difficulty_bits: 12,
        };
        let right = ComputeJob {
            seed_bytes: b"seed-a".to_vec(),
            pass_prefix: b"prefix-a:".to_vec(),
            time_cost: 1,
            memory_cost_kib: 64 * 1024,
            parallelism: 2,
            difficulty_bits: 12,
        };

        assert_ne!(BenchmarkKey::from(&left), BenchmarkKey::from(&right));
    }

    #[test]
    fn compute_job_from_challenge_matches_go_value_mapping() {
        let challenge = ChallengeResponse {
            ok: true,
            challenge_id: 7,
            round_id: 8,
            difficulty_bits: 9,
            memory_cost_mb: -2,
            parallelism: -1,
            seed: "seed".to_string(),
            session_salt: "salt".to_string(),
            time_cost: -3,
            visitor_id: "visitor".to_string(),
            message: String::new(),
            result: String::new(),
        };

        let job = ComputeJob::from(&challenge);

        assert_eq!(job.seed_bytes, b"seed");
        assert_eq!(job.pass_prefix, b"seed:8:visitor:7:salt:");
        assert_eq!(job.time_cost, (-3i32) as u32);
        assert_eq!(job.memory_cost_kib, ((-2i32) as u32).wrapping_mul(1024));
        assert_eq!(job.parallelism, (-1i32) as u8 as u32);
        assert_eq!(job.difficulty_bits, 9);
    }
}
