use crate::backend::cuda::{GPU_FINALIST_COUNT, GPU_RUNTIME_BENCHMARK_DURATION};
use crate::backend::types::{GpuBenchmarkConfig, estimated_argon2_batch_memory_bytes};
use crate::backend::{BackendDescriptor, BenchmarkResult, ComputeJob};
use crate::error::is_interrupted_error;
use crate::{MiningError, humanize_error};

use super::Runner;
use super::support::{BenchmarkKey, SelectedBackend, format_memory_bytes, localized_bool};

const GPU_LARGE_BATCH_SPEED_FLOOR_RATIO: f64 = 0.90;

impl Runner {
    pub(super) fn collect_gpu_backend_candidates(
        &self,
        job: &ComputeJob,
    ) -> Result<Vec<SelectedBackend>, MiningError> {
        let mut candidates = self.collect_cuda_backend_candidates(job)?;
        let skip_nvidia_opencl = candidates.iter().any(|candidate| {
            candidate.kind == crate::backend::BackendKind::Cuda
                && !candidate.device_id.trim().is_empty()
        });
        candidates.extend(self.collect_opencl_backend_candidates(job, skip_nvidia_opencl)?);
        candidates.extend(self.collect_metal_backend_candidates(job)?);
        Ok(candidates)
    }

    fn collect_cuda_backend_candidates(
        &self,
        job: &ComputeJob,
    ) -> Result<Vec<SelectedBackend>, MiningError> {
        let cuda_availability = self.cuda_backend.detect_availability();
        if !cuda_availability.available {
            if !cuda_availability.reason.trim().is_empty() {
                self.log(format_args!(
                    "CUDA 后端不可用：{}",
                    cuda_availability.reason
                ));
            }
            return Ok(Vec::new());
        }

        self.collect_gpu_candidates_by_device(
            "CUDA",
            self.cuda_backend.list_devices()?,
            job,
            BenchmarkKey::from(job),
            |descriptor| {
                self.cuda_backend
                    .quick_screen_benchmark_for_descriptor(descriptor, job)
            },
            |descriptor| self.tune_cuda_backend(descriptor, job),
        )
    }

    fn collect_opencl_backend_candidates(
        &self,
        job: &ComputeJob,
        skip_nvidia_opencl: bool,
    ) -> Result<Vec<SelectedBackend>, MiningError> {
        let opencl_availability = self.opencl_backend.detect_availability();
        if !opencl_availability.available {
            if !opencl_availability.reason.trim().is_empty() {
                self.log(format_args!(
                    "OpenCL 后端不可用：{}",
                    opencl_availability.reason
                ));
            }
            return Ok(Vec::new());
        }

        let mut devices = self.opencl_backend.list_devices()?;
        if skip_nvidia_opencl {
            let original_count = devices.len();
            devices.retain(|descriptor| !is_nvidia_opencl_descriptor(descriptor));
            let skipped = original_count.saturating_sub(devices.len());
            if skipped > 0 {
                self.log(format_args!(
                    "CUDA 后端已经可用，跳过 {} 个 NVIDIA OpenCL 设备，避免同卡重复调优。",
                    skipped
                ));
            }
        }

        self.collect_gpu_candidates_by_device(
            "OpenCL",
            devices,
            job,
            BenchmarkKey::from(job),
            |descriptor| {
                self.opencl_backend
                    .quick_screen_benchmark_for_descriptor(descriptor, job)
            },
            |descriptor| self.tune_opencl_backend(descriptor, job),
        )
    }

    fn collect_metal_backend_candidates(
        &self,
        job: &ComputeJob,
    ) -> Result<Vec<SelectedBackend>, MiningError> {
        let metal_availability = self.metal_backend.detect_availability();
        if !metal_availability.available {
            if !metal_availability.reason.trim().is_empty() {
                self.log(format_args!(
                    "Metal 后端不可用：{}",
                    metal_availability.reason
                ));
            }
            return Ok(Vec::new());
        }

        self.collect_gpu_candidates_by_device(
            "Metal",
            self.metal_backend.list_devices()?,
            job,
            BenchmarkKey::from(job),
            |descriptor| {
                self.metal_backend
                    .quick_screen_benchmark_for_descriptor(descriptor, job)
            },
            |descriptor| self.tune_metal_backend(descriptor, job),
        )
    }

    fn collect_gpu_candidates_by_device<FScreen, FTune>(
        &self,
        label: &str,
        devices: Vec<BackendDescriptor>,
        job: &ComputeJob,
        params_key: BenchmarkKey,
        screen: FScreen,
        tune: FTune,
    ) -> Result<Vec<SelectedBackend>, MiningError>
    where
        FScreen: Fn(&BackendDescriptor) -> Result<BenchmarkResult, MiningError>,
        FTune: Fn(&BackendDescriptor) -> Result<BenchmarkResult, MiningError>,
    {
        if devices.is_empty() {
            self.log(format_args!(
                "{} 后端可用，但没有检测到可用设备，回退 CPU。",
                label
            ));
            return Ok(Vec::new());
        }

        let mut screened = Vec::new();
        for descriptor in devices {
            self.check_cancel()?;
            match screen(&descriptor) {
                Ok(result) => {
                    self.log(format_args!(
                        "{} 设备初筛完成：设备 {}，默认批大小 {}，按分段 {}，预计算参考值 {}，预计显存 {}，预计速度约 {:.2} 次/秒。",
                        label,
                        descriptor.name,
                        result.workers,
                        localized_bool(result.by_segment),
                        localized_bool(result.precompute_refs),
                        estimated_gpu_memory_label(job, result.concurrency),
                        result.attempts_per_s
                    ));
                    screened.push((descriptor, result));
                }
                Err(error) => {
                    if is_interrupted_error(&error) {
                        return Err(error);
                    }
                    self.log(format_args!(
                        "{} 设备 {} 初筛失败，回退 CPU：{}",
                        label,
                        descriptor.name,
                        humanize_error(&error)
                    ));
                }
            }
        }

        screened.sort_by(|left, right| right.1.attempts_per_s.total_cmp(&left.1.attempts_per_s));
        let finalists = screened
            .into_iter()
            .take(GPU_FINALIST_COUNT)
            .collect::<Vec<_>>();

        let mut candidates = Vec::new();
        for (descriptor, _) in finalists {
            match tune(&descriptor) {
                Ok(result) => {
                    self.log(format_args!(
                        "{} 自动调优完成：设备 {}，推荐批大小 {}，按分段 {}，预计算参考值 {}，预计显存 {}，预计速度约 {:.2} 次/秒。",
                        label,
                        descriptor.name,
                        result.workers,
                        localized_bool(result.by_segment),
                        localized_bool(result.precompute_refs),
                        estimated_gpu_memory_label(job, result.concurrency),
                        result.attempts_per_s
                    ));
                    candidates.push(SelectedBackend::new(
                        &descriptor,
                        result,
                        params_key.clone(),
                    ));
                }
                Err(error) => {
                    if is_interrupted_error(&error) {
                        return Err(error);
                    }
                    self.log(format_args!(
                        "{} 设备 {} 自动调优失败，回退 CPU：{}",
                        label,
                        descriptor.name,
                        humanize_error(&error)
                    ));
                }
            }
        }
        Ok(candidates)
    }

    fn tune_cuda_backend(
        &self,
        descriptor: &BackendDescriptor,
        job: &ComputeJob,
    ) -> Result<BenchmarkResult, MiningError> {
        let templates = self
            .cuda_backend
            .solver_templates_for_descriptor(descriptor, job);
        self.tune_gpu_backend("CUDA", descriptor, job, &templates, |candidate| {
            self.cuda_backend.run_runtime_loop_benchmark_with_cancel(
                job,
                GpuBenchmarkConfig {
                    device_index: descriptor.device_index.unwrap_or(0),
                    batch_size: candidate.batch_size,
                    by_segment: candidate.by_segment,
                    precompute_refs: candidate.precompute_refs,
                    duration: GPU_RUNTIME_BENCHMARK_DURATION,
                },
                &self.cancel,
            )
        })
    }

    fn tune_opencl_backend(
        &self,
        descriptor: &BackendDescriptor,
        job: &ComputeJob,
    ) -> Result<BenchmarkResult, MiningError> {
        let templates = self
            .opencl_backend
            .solver_templates_for_descriptor(descriptor, job);
        self.tune_gpu_backend("OpenCL", descriptor, job, &templates, |candidate| {
            self.opencl_backend.run_runtime_loop_benchmark_with_cancel(
                job,
                GpuBenchmarkConfig {
                    device_index: descriptor.device_index.unwrap_or(0),
                    batch_size: candidate.batch_size,
                    by_segment: candidate.by_segment,
                    precompute_refs: candidate.precompute_refs,
                    duration: GPU_RUNTIME_BENCHMARK_DURATION,
                },
                &self.cancel,
            )
        })
    }

    fn tune_metal_backend(
        &self,
        descriptor: &BackendDescriptor,
        job: &ComputeJob,
    ) -> Result<BenchmarkResult, MiningError> {
        let templates = self
            .metal_backend
            .solver_templates_for_descriptor(descriptor, job);
        self.tune_gpu_backend("Metal", descriptor, job, &templates, |candidate| {
            self.metal_backend.run_runtime_loop_benchmark_with_cancel(
                job,
                GpuBenchmarkConfig {
                    device_index: descriptor.device_index.unwrap_or(0),
                    batch_size: candidate.batch_size,
                    by_segment: candidate.by_segment,
                    precompute_refs: candidate.precompute_refs,
                    duration: GPU_RUNTIME_BENCHMARK_DURATION,
                },
                &self.cancel,
            )
        })
    }

    fn tune_gpu_backend<TConfig, FRun>(
        &self,
        label: &str,
        descriptor: &BackendDescriptor,
        job: &ComputeJob,
        templates: &[TConfig],
        run: FRun,
    ) -> Result<BenchmarkResult, MiningError>
    where
        TConfig: Copy,
        FRun: Fn(TConfig) -> Result<BenchmarkResult, MiningError>,
    {
        let total_cases = templates.len();
        let mut results = Vec::new();
        for (index, candidate) in templates.iter().copied().enumerate() {
            self.check_cancel()?;
            let result = match run(candidate) {
                Ok(result) => result,
                Err(error) if is_interrupted_error(&error) => return Err(error),
                Err(error) => {
                    self.log(format_args!(
                        "{} 自动调优配置 {}/{} 不可用，已跳过：{}",
                        label,
                        index + 1,
                        total_cases,
                        humanize_error(&error)
                    ));
                    continue;
                }
            };
            self.log(format_args!(
                "{} 自动调优结果 {}/{}：设备 {}，批大小 {}，按分段 {}，预计算参考值 {}，预计显存 {}，速度约 {:.2} 次/秒。",
                label,
                index + 1,
                total_cases,
                descriptor.name,
                result.workers,
                localized_bool(result.by_segment),
                localized_bool(result.precompute_refs),
                estimated_gpu_memory_label(job, result.concurrency),
                result.attempts_per_s
            ));
            results.push(result);
        }
        let Some(selected) = select_gpu_tuning_result(&results) else {
            return Err(MiningError::Message(format!(
                "{} 自动调优没有得到可用结果。",
                label
            )));
        };
        if let Some(fastest) = fastest_gpu_tuning_result(&results)
            && selected.concurrency != fastest.concurrency
            && fastest.attempts_per_s > 0.0
        {
            let loss_percent = (fastest.attempts_per_s - selected.attempts_per_s).max(0.0) * 100.0
                / fastest.attempts_per_s;
            self.log(format_args!(
                "{} 自动调优：批大小 {} 与最高速批大小 {} 相差约 {:.1}%，优先使用更大的 GPU 批大小。",
                label, selected.concurrency, fastest.concurrency, loss_percent
            ));
        }
        Ok(selected)
    }

    pub(super) fn filter_blacklisted(
        &self,
        candidates: Vec<SelectedBackend>,
    ) -> Vec<SelectedBackend> {
        let blacklist = self
            .backend_blacklist
            .lock()
            .expect("backend blacklist poisoned");
        candidates
            .into_iter()
            .filter(|candidate| !blacklist.contains(&(candidate.kind, candidate.device_id.clone())))
            .collect()
    }

    pub(super) fn run_backend_self_test(
        &self,
        backend: &SelectedBackend,
        job: &ComputeJob,
    ) -> Result<(), MiningError> {
        let digest = crate::backend::cpu::compute_digest(job, 1);
        if digest.is_empty()
            || !crate::backend::cpu::hex_lower(&digest)
                .chars()
                .all(|ch| ch.is_ascii_hexdigit())
        {
            return Err(MiningError::Message(format!(
                "{} 后端自检失败：摘要格式无效",
                backend.label
            )));
        }
        Ok(())
    }
}

fn is_nvidia_opencl_descriptor(descriptor: &BackendDescriptor) -> bool {
    let text = format!("{} {}", descriptor.name, descriptor.device_id).to_ascii_lowercase();
    text.contains("nvidia") || text.contains("cuda")
}

fn estimated_gpu_memory_label(job: &ComputeJob, batch_size: usize) -> String {
    format_memory_bytes(estimated_argon2_batch_memory_bytes(
        job.memory_cost_kib,
        batch_size,
    ))
}

fn fastest_gpu_tuning_result(results: &[BenchmarkResult]) -> Option<BenchmarkResult> {
    results.iter().copied().max_by(|left, right| {
        left.attempts_per_s
            .total_cmp(&right.attempts_per_s)
            .then_with(|| left.concurrency.cmp(&right.concurrency))
    })
}

fn select_gpu_tuning_result(results: &[BenchmarkResult]) -> Option<BenchmarkResult> {
    let fastest = fastest_gpu_tuning_result(results)?;
    if fastest.attempts_per_s <= 0.0 {
        return Some(fastest);
    }
    let floor = fastest.attempts_per_s * GPU_LARGE_BATCH_SPEED_FLOOR_RATIO;
    results
        .iter()
        .copied()
        .filter(|result| result.attempts_per_s >= floor)
        .max_by(|left, right| {
            left.concurrency
                .cmp(&right.concurrency)
                .then_with(|| left.attempts_per_s.total_cmp(&right.attempts_per_s))
        })
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use crate::backend::BenchmarkResult;

    use super::select_gpu_tuning_result;

    fn result(batch_size: usize, attempts_per_s: f64) -> BenchmarkResult {
        BenchmarkResult {
            workers: batch_size,
            concurrency: batch_size,
            by_segment: true,
            precompute_refs: true,
            attempts: 1,
            elapsed: Duration::from_secs(1),
            attempts_per_s,
        }
    }

    #[test]
    fn select_gpu_tuning_result_prefers_larger_batch_when_speed_is_close() {
        let selected =
            select_gpu_tuning_result(&[result(16, 100.0), result(128, 98.0), result(256, 95.0)])
                .expect("selected tuning result");

        assert_eq!(selected.concurrency, 256);
    }

    #[test]
    fn select_gpu_tuning_result_keeps_fastest_when_larger_batch_is_too_slow() {
        let selected = select_gpu_tuning_result(&[result(16, 100.0), result(128, 89.0)])
            .expect("selected tuning result");

        assert_eq!(selected.concurrency, 16);
    }
}
