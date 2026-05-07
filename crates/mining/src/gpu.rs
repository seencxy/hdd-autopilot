use std::time::Duration;

use crate::MiningError;
use crate::backend::cuda::{GPU_FINALIST_COUNT, GPU_RUNTIME_BENCHMARK_DURATION};
use crate::backend::{
    self, BackendDescriptor, BenchmarkResult, CudaBackend, GPUAvailability, MetalBackend,
    OpenclBackend,
};

pub(crate) fn detect_gpu_availability() -> GPUAvailability {
    let cuda = CudaBackend::new().detect_availability();
    if cuda.available {
        return cuda;
    }
    let opencl = OpenclBackend::new().detect_availability();
    if opencl.available {
        return opencl;
    }
    let metal = MetalBackend::new().detect_availability();
    if metal.available {
        return metal;
    }
    let mut reasons = Vec::new();
    if !cuda.reason.trim().is_empty() {
        reasons.push(format!("CUDA：{}", cuda.reason));
    }
    if !opencl.reason.trim().is_empty() {
        reasons.push(format!("OpenCL：{}", opencl.reason));
    }
    if !metal.reason.trim().is_empty() {
        reasons.push(format!("Metal：{}", metal.reason));
    }
    GPUAvailability {
        available: false,
        reason: reasons.join("；"),
    }
}

pub(crate) fn find_best_gpu_benchmark_config() -> Result<BenchmarkResult, MiningError> {
    best_gpu_benchmark_device().map(|(_, result)| result)
}

pub(crate) fn benchmark_best_gpu_runtime() -> Result<(BenchmarkResult, BenchmarkResult), MiningError>
{
    let selected = best_gpu_runtime_device_with_benchmark()?;
    let best = *selected.benchmark();
    let runtime = selected.run_runtime_loop_benchmark(
        best.workers,
        best.by_segment,
        best.precompute_refs,
        best.generate_passwords_on_gpu,
        GPU_RUNTIME_BENCHMARK_DURATION,
    )?;
    Ok((best, runtime))
}

pub(crate) fn run_gpu_runtime_loop_benchmark(
    batch_size: usize,
    by_segment: bool,
    precompute_refs: bool,
    duration: Duration,
) -> Result<BenchmarkResult, MiningError> {
    let benchmark_job = backend::cpu::default_benchmark_job();
    match best_gpu_runtime_device()? {
        BestGpuRuntimeDevice::Cuda(descriptor, backend) => backend
            .run_runtime_loop_benchmark_for_device(
                descriptor.device_index.unwrap_or(0),
                &benchmark_job,
                batch_size,
                by_segment,
                precompute_refs,
                true,
                duration,
            ),
        BestGpuRuntimeDevice::Metal(descriptor, backend) => backend
            .run_runtime_loop_benchmark_for_device(
                descriptor.device_index.unwrap_or(0),
                &benchmark_job,
                batch_size,
                by_segment,
                precompute_refs,
                duration,
            ),
        BestGpuRuntimeDevice::Opencl(descriptor, backend) => backend
            .run_runtime_loop_benchmark_for_device(
                descriptor.device_index.unwrap_or(0),
                &benchmark_job,
                batch_size,
                by_segment,
                precompute_refs,
                duration,
            ),
    }
}

fn best_gpu_benchmark_device() -> Result<(BackendDescriptor, BenchmarkResult), MiningError> {
    match best_gpu_runtime_device_with_benchmark()? {
        BestGpuRuntimeDeviceWithBenchmark::Cuda(descriptor, result, _) => Ok((descriptor, result)),
        BestGpuRuntimeDeviceWithBenchmark::Metal(descriptor, result, _) => Ok((descriptor, result)),
        BestGpuRuntimeDeviceWithBenchmark::Opencl(descriptor, result, _) => {
            Ok((descriptor, result))
        }
    }
}

fn best_gpu_runtime_device() -> Result<BestGpuRuntimeDevice, MiningError> {
    match best_gpu_runtime_device_with_benchmark()? {
        BestGpuRuntimeDeviceWithBenchmark::Cuda(descriptor, _, backend) => {
            Ok(BestGpuRuntimeDevice::Cuda(descriptor, backend))
        }
        BestGpuRuntimeDeviceWithBenchmark::Metal(descriptor, _, backend) => {
            Ok(BestGpuRuntimeDevice::Metal(descriptor, backend))
        }
        BestGpuRuntimeDeviceWithBenchmark::Opencl(descriptor, _, backend) => {
            Ok(BestGpuRuntimeDevice::Opencl(descriptor, backend))
        }
    }
}

fn best_gpu_runtime_device_with_benchmark() -> Result<BestGpuRuntimeDeviceWithBenchmark, MiningError>
{
    let best_cuda = best_cuda_runtime_device_with_benchmark()?;
    let best_opencl = best_opencl_runtime_device_with_benchmark()?;
    let best_metal = best_metal_runtime_device_with_benchmark()?;

    let mut candidates = Vec::new();
    if let Some((descriptor, result, backend)) = best_cuda {
        candidates.push(BestGpuRuntimeDeviceWithBenchmark::Cuda(
            descriptor, result, backend,
        ));
    }
    if let Some((descriptor, result, backend)) = best_opencl {
        candidates.push(BestGpuRuntimeDeviceWithBenchmark::Opencl(
            descriptor, result, backend,
        ));
    }
    if let Some((descriptor, result, backend)) = best_metal {
        candidates.push(BestGpuRuntimeDeviceWithBenchmark::Metal(
            descriptor, result, backend,
        ));
    }

    candidates
        .into_iter()
        .max_by(|left, right| {
            left.benchmark()
                .attempts_per_s
                .total_cmp(&right.benchmark().attempts_per_s)
        })
        .ok_or_else(|| MiningError::Message("当前环境未检测到可用 GPU 设备。".to_string()))
}

fn best_cuda_runtime_device_with_benchmark()
-> Result<Option<(BackendDescriptor, BenchmarkResult, CudaBackend)>, MiningError> {
    let cuda_backend = CudaBackend::new();
    let benchmark_job = backend::cpu::default_benchmark_job();
    best_runtime_device_with_benchmark(
        cuda_backend.clone(),
        cuda_backend.detect_availability().available,
        || cuda_backend.list_devices(),
        |descriptor| cuda_backend.quick_screen_benchmark_for_descriptor(descriptor, &benchmark_job),
        |descriptor| {
            cuda_backend.find_best_benchmark_config_for_descriptor(descriptor, &benchmark_job)
        },
    )
}

fn best_opencl_runtime_device_with_benchmark()
-> Result<Option<(BackendDescriptor, BenchmarkResult, OpenclBackend)>, MiningError> {
    let opencl_backend = OpenclBackend::new();
    let benchmark_job = backend::cpu::default_benchmark_job();
    best_runtime_device_with_benchmark(
        opencl_backend.clone(),
        opencl_backend.detect_availability().available,
        || opencl_backend.list_devices(),
        |descriptor| {
            opencl_backend.quick_screen_benchmark_for_descriptor(descriptor, &benchmark_job)
        },
        |descriptor| {
            opencl_backend.find_best_benchmark_config_for_descriptor(descriptor, &benchmark_job)
        },
    )
}

fn best_metal_runtime_device_with_benchmark()
-> Result<Option<(BackendDescriptor, BenchmarkResult, MetalBackend)>, MiningError> {
    let metal_backend = MetalBackend::new();
    let benchmark_job = backend::cpu::default_benchmark_job();
    best_runtime_device_with_benchmark(
        metal_backend.clone(),
        metal_backend.detect_availability().available,
        || metal_backend.list_devices(),
        |descriptor| {
            metal_backend.quick_screen_benchmark_for_descriptor(descriptor, &benchmark_job)
        },
        |descriptor| {
            metal_backend.find_best_benchmark_config_for_descriptor(descriptor, &benchmark_job)
        },
    )
}

fn best_runtime_device_with_benchmark<TBackend, FList, FScreen, FTune>(
    backend: TBackend,
    available: bool,
    list_devices: FList,
    quick_screen: FScreen,
    full_tune: FTune,
) -> Result<Option<(BackendDescriptor, BenchmarkResult, TBackend)>, MiningError>
where
    TBackend: Clone,
    FList: Fn() -> Result<Vec<BackendDescriptor>, MiningError>,
    FScreen: Fn(&BackendDescriptor) -> Result<BenchmarkResult, MiningError>,
    FTune: Fn(&BackendDescriptor) -> Result<BenchmarkResult, MiningError>,
{
    if !available {
        return Ok(None);
    }

    let devices = list_devices()?;
    if devices.is_empty() {
        return Ok(None);
    }

    let mut screened = Vec::new();
    for descriptor in devices {
        if let Ok(result) = quick_screen(&descriptor) {
            screened.push((descriptor, result));
        }
    }
    if screened.is_empty() {
        return Ok(None);
    }

    screened.sort_by(|left, right| right.1.attempts_per_s.total_cmp(&left.1.attempts_per_s));

    let mut best: Option<(BackendDescriptor, BenchmarkResult)> = None;
    for (descriptor, _) in screened.into_iter().take(GPU_FINALIST_COUNT) {
        let result = full_tune(&descriptor)?;
        if best
            .as_ref()
            .is_none_or(|(_, current)| result.attempts_per_s > current.attempts_per_s)
        {
            best = Some((descriptor, result));
        }
    }

    Ok(best.map(|(descriptor, result)| (descriptor, result, backend)))
}

#[derive(Debug, Clone)]
enum BestGpuRuntimeDevice {
    Cuda(BackendDescriptor, CudaBackend),
    Metal(BackendDescriptor, MetalBackend),
    Opencl(BackendDescriptor, OpenclBackend),
}

#[derive(Debug, Clone)]
enum BestGpuRuntimeDeviceWithBenchmark {
    Cuda(BackendDescriptor, BenchmarkResult, CudaBackend),
    Metal(BackendDescriptor, BenchmarkResult, MetalBackend),
    Opencl(BackendDescriptor, BenchmarkResult, OpenclBackend),
}

impl BestGpuRuntimeDeviceWithBenchmark {
    fn benchmark(&self) -> &BenchmarkResult {
        match self {
            Self::Cuda(_, result, _) | Self::Metal(_, result, _) | Self::Opencl(_, result, _) => {
                result
            }
        }
    }

    fn run_runtime_loop_benchmark(
        &self,
        batch_size: usize,
        by_segment: bool,
        precompute_refs: bool,
        generate_passwords_on_gpu: bool,
        duration: Duration,
    ) -> Result<BenchmarkResult, MiningError> {
        match self {
            Self::Cuda(descriptor, _, backend) => backend.run_runtime_loop_benchmark_for_device(
                descriptor.device_index.unwrap_or(0),
                &backend::cpu::default_benchmark_job(),
                batch_size,
                by_segment,
                precompute_refs,
                generate_passwords_on_gpu,
                duration,
            ),
            Self::Metal(descriptor, _, backend) => backend.run_runtime_loop_benchmark_for_device(
                descriptor.device_index.unwrap_or(0),
                &backend::cpu::default_benchmark_job(),
                batch_size,
                by_segment,
                precompute_refs,
                duration,
            ),
            Self::Opencl(descriptor, _, backend) => backend.run_runtime_loop_benchmark_for_device(
                descriptor.device_index.unwrap_or(0),
                &backend::cpu::default_benchmark_job(),
                batch_size,
                by_segment,
                precompute_refs,
                duration,
            ),
        }
    }
}
