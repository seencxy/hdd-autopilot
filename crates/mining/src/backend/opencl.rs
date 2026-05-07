use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use mining_opencl_sys as opencl_sys;
use mining_opencl_sys::{OpenclDeviceInfo, OpenclSolverConfig};

use crate::backend::cpu::{ComputeJob, benchmark_job_for_tuning, compute_digest, hex_lower};
use crate::backend::cuda::{GPU_DEVICE_SCREENING_DURATION, GPU_RUNTIME_BENCHMARK_DURATION};
use crate::backend::types::{
    BackendDescriptor, BackendKind, BenchmarkResult, GPUAvailability, GpuBenchmarkConfig,
    GpuDeviceProfile, GpuMiningSessionConfig, MineBlockResult, MineResult,
    recommended_gpu_tuning_shapes,
};
use crate::error::{MiningError, interrupted_error};

#[derive(Debug, Clone, Default)]
pub struct OpenclBackend;

pub struct OpenclMiningSession {
    session: opencl_sys::OpenclMiningSession,
    job: ComputeJob,
    stop: Arc<AtomicBool>,
    cancel: Arc<AtomicBool>,
}

impl OpenclMiningSession {
    pub fn mine_until_stop(&mut self) -> Result<MineBlockResult, MiningError> {
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
                .session
                .mine_next_batch()
                .map_err(MiningError::Message)?;
            if result.found {
                let nonce = result.nonce as usize;
                let expected_digest = hex_lower(&compute_digest(&self.job, nonce));
                if result.digest_hex != expected_digest {
                    return Err(MiningError::Message(
                        "OpenCL 后端返回的摘要校验失败。".to_string(),
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
}

impl OpenclBackend {
    pub fn new() -> Self {
        Self
    }

    pub fn solver_templates_for_descriptor(
        &self,
        descriptor: &BackendDescriptor,
        job: &ComputeJob,
    ) -> Vec<OpenclSolverConfig> {
        recommended_gpu_tuning_shapes(descriptor.gpu_profile, job.memory_cost_kib, job.parallelism)
            .into_iter()
            .map(|shape| OpenclSolverConfig {
                batch_size: shape.batch_size,
                by_segment: shape.by_segment,
                precompute_refs: shape.precompute_refs,
            })
            .collect()
    }

    pub fn descriptor_for_device(&self, device: &OpenclDeviceInfo) -> BackendDescriptor {
        let name = opencl_device_display_name(device);
        BackendDescriptor {
            kind: BackendKind::Opencl,
            name,
            device_id: device.device_id.clone(),
            device_index: Some(device.device_index),
            gpu_profile: Some(GpuDeviceProfile {
                global_memory_bytes: device.global_memory_bytes,
                max_alloc_bytes: device.max_alloc_bytes,
                compute_units: device.compute_units,
                max_threads_per_group: device.max_work_group_size,
                local_memory_bytes: device.local_memory_bytes,
                subgroup_size: 0,
                unified_memory: device.host_unified_memory,
                low_power: false,
                removable: false,
            }),
        }
    }

    pub fn list_devices(&self) -> Result<Vec<BackendDescriptor>, MiningError> {
        if !opencl_platform_supported() {
            return Ok(Vec::new());
        }
        let devices = opencl_sys::list_devices().map_err(MiningError::Message)?;
        Ok(devices
            .iter()
            .filter(|device| device.is_gpu_like())
            .map(|device| self.descriptor_for_device(device))
            .collect())
    }

    pub fn detect_availability(&self) -> GPUAvailability {
        if !opencl_platform_supported() {
            return GPUAvailability {
                available: false,
                reason: "current platform does not support the OpenCL backend".to_string(),
            };
        }
        match opencl_sys::is_available() {
            Ok(true) => GPUAvailability {
                available: true,
                reason: String::new(),
            },
            Ok(false) => GPUAvailability {
                available: false,
                reason: "当前没有检测到可用的 OpenCL GPU 设备。".to_string(),
            },
            Err(error) => GPUAvailability {
                available: false,
                reason: format!("OpenCL 后端不可用：{}", error),
            },
        }
    }

    pub fn default_solver_config_for_job(
        &self,
        descriptor: &BackendDescriptor,
        job: &ComputeJob,
    ) -> Result<OpenclSolverConfig, MiningError> {
        let raw_job = opencl_sys::OpenclJob {
            seed_bytes: &job.seed_bytes,
            pass_prefix: &job.pass_prefix,
            time_cost: job.time_cost,
            memory_cost_kib: job.memory_cost_kib,
            parallelism: job.parallelism,
            difficulty_bits: job.difficulty_bits,
        };
        opencl_sys::default_solver_config(descriptor.device_index.unwrap_or(0), &raw_job)
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
        best.ok_or_else(|| MiningError::Message("OpenCL 自动调优没有得到可用结果。".to_string()))
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
        #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
        {
            let _ = (
                device_index,
                job,
                batch_size,
                by_segment,
                precompute_refs,
                duration,
                cancel,
            );
            Err(MiningError::Message(
                "当前平台未启用 OpenCL 后端。".to_string(),
            ))
        }
        #[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
        {
            let config = OpenclSolverConfig {
                batch_size: batch_size.max(1),
                by_segment,
                precompute_refs,
            };
            let benchmark_job = benchmark_job_for_tuning(job);
            let raw_job = opencl_sys::OpenclJob {
                seed_bytes: &benchmark_job.seed_bytes,
                pass_prefix: &benchmark_job.pass_prefix,
                time_cost: benchmark_job.time_cost,
                memory_cost_kib: benchmark_job.memory_cost_kib,
                parallelism: benchmark_job.parallelism,
                difficulty_bits: benchmark_job.difficulty_bits,
            };
            let mut session = opencl_sys::create_session(device_index, &raw_job, config, 1)
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
    }

    pub fn start_mining_session(
        &self,
        job: &ComputeJob,
        config: GpuMiningSessionConfig,
        stop: &Arc<AtomicBool>,
        cancel: &Arc<AtomicBool>,
    ) -> Result<OpenclMiningSession, MiningError> {
        let GpuMiningSessionConfig {
            device_index,
            batch_size,
            session_count: _,
            by_segment,
            precompute_refs,
            start_nonce,
            nonce_count: _,
        } = config;
        let session = opencl_sys::create_session(
            device_index,
            &opencl_sys::OpenclJob {
                seed_bytes: &job.seed_bytes,
                pass_prefix: &job.pass_prefix,
                time_cost: job.time_cost,
                memory_cost_kib: job.memory_cost_kib,
                parallelism: job.parallelism,
                difficulty_bits: job.difficulty_bits,
            },
            opencl_sys::OpenclSolverConfig {
                batch_size: batch_size.max(1),
                by_segment,
                precompute_refs,
            },
            start_nonce,
        )
        .map_err(MiningError::Message)?;
        Ok(OpenclMiningSession {
            session,
            job: job.clone(),
            stop: Arc::clone(stop),
            cancel: Arc::clone(cancel),
        })
    }
}

fn opencl_platform_supported() -> bool {
    matches!(std::env::consts::OS, "macos" | "linux" | "windows")
}

fn opencl_device_display_name(device: &OpenclDeviceInfo) -> String {
    let mut name = device.name.trim().to_string();
    if name.is_empty() {
        name = "OpenCL device".to_string();
    }
    name.push_str(" [");
    name.push_str(device.device_type_label());
    name.push(']');
    if !device.platform.trim().is_empty() {
        name.push_str(" @ ");
        name.push_str(device.platform.trim());
    }
    name
}
