mod backend;
mod client;
mod config;
mod error;
mod gpu;
mod messages;
mod protocol;
pub mod rpow2;
mod runner;

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use backend::{BenchmarkResult, CpuBackend, GPUAvailability};
use runner::Runner;

pub use config::{Config, Mode, OutputSink, default_config_for_mode};
pub use error::MiningError;

pub(crate) use config::{DEFAULT_USER_AGENT, RewardKind, default_config};
pub(crate) use messages::{
    code_type_label, humanize_duration, humanize_error, localized_message, mode_description,
    preference_label, result_label, reward_benefit_label, reward_display_label,
};
pub(crate) use protocol::{
    ApiErrorBody, ChallengeResponse, HeartbeatRequest, HeartbeatResponse, StatusResponse,
    SubmitRequest, SubmitResponse,
};

pub fn run_auto_tuned(mode: Mode) -> Result<(), MiningError> {
    let cancel = Arc::new(AtomicBool::new(false));
    run_auto_tuned_with_cancel(mode, &cancel)
}

pub fn find_best_benchmark_config() -> BenchmarkResult {
    CpuBackend::new().find_best_benchmark_config()
}

pub fn run_benchmark_case(workers: usize, duration: Duration) -> BenchmarkResult {
    CpuBackend::new().run_benchmark_case(workers, duration)
}

pub fn run_benchmark_case_with_concurrency(
    workers: usize,
    concurrency: usize,
    duration: Duration,
) -> BenchmarkResult {
    CpuBackend::new().run_benchmark_case_with_concurrency(
        &backend::cpu::default_benchmark_job(),
        workers,
        concurrency,
        duration,
    )
}

pub fn detect_gpu_availability(_mode: Mode) -> GPUAvailability {
    gpu::detect_gpu_availability()
}

pub fn find_best_gpu_benchmark_config() -> Result<BenchmarkResult, MiningError> {
    gpu::find_best_gpu_benchmark_config()
}

pub fn benchmark_best_gpu_runtime() -> Result<(BenchmarkResult, BenchmarkResult), MiningError> {
    gpu::benchmark_best_gpu_runtime()
}

pub fn run_gpu_runtime_loop_benchmark(
    batch_size: usize,
    by_segment: bool,
    precompute_refs: bool,
    duration: Duration,
) -> Result<BenchmarkResult, MiningError> {
    gpu::run_gpu_runtime_loop_benchmark(batch_size, by_segment, precompute_refs, duration)
}

pub fn run_auto_tuned_gpu(mode: Mode) -> Result<(), MiningError> {
    let cancel = Arc::new(AtomicBool::new(false));
    run_auto_tuned_gpu_with_cancel(mode, &cancel)
}

pub fn run_auto_tuned_with_cancel(mode: Mode, cancel: &Arc<AtomicBool>) -> Result<(), MiningError> {
    let config = default_config(mode);
    Runner::new(config, Arc::clone(cancel))?.run_auto_tuned()
}

pub fn run_auto_tuned_with_config_and_cancel(
    config: Config,
    cancel: &Arc<AtomicBool>,
) -> Result<(), MiningError> {
    Runner::new(config, Arc::clone(cancel))?.run_auto_tuned()
}

pub fn run_auto_tuned_gpu_with_cancel(
    mode: Mode,
    cancel: &Arc<AtomicBool>,
) -> Result<(), MiningError> {
    run_auto_tuned_with_cancel(mode, cancel)
}
