use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendKind {
    Cpu,
    Cuda,
    Metal,
    Opencl,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendDescriptor {
    pub kind: BackendKind,
    pub name: String,
    pub device_id: String,
    pub device_index: Option<usize>,
    pub gpu_profile: Option<GpuDeviceProfile>,
}

pub use super::tuning::{
    GpuDeviceProfile, estimated_argon2_batch_memory_bytes, recommended_gpu_tuning_shapes,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BenchmarkResult {
    pub workers: usize,
    pub concurrency: usize,
    pub by_segment: bool,
    pub precompute_refs: bool,
    pub attempts: i64,
    pub elapsed: Duration,
    pub attempts_per_s: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuMiningSessionConfig {
    pub workers: usize,
    pub concurrency: usize,
    pub start_nonce: usize,
    pub nonce_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuBenchmarkConfig {
    pub device_index: usize,
    pub batch_size: usize,
    pub by_segment: bool,
    pub precompute_refs: bool,
    pub duration: Duration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuMiningSessionConfig {
    pub device_index: usize,
    pub batch_size: usize,
    pub session_count: usize,
    pub by_segment: bool,
    pub precompute_refs: bool,
    pub start_nonce: u64,
    pub nonce_count: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GPUAvailability {
    pub available: bool,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MineResult {
    pub nonce: usize,
    pub digest: String,
    pub attempts: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MineBlockResult {
    pub found: Option<MineResult>,
    pub attempts: i64,
}
