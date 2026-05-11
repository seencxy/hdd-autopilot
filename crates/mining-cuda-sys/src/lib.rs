#![allow(non_camel_case_types)]
#![cfg_attr(not(mining_cuda_native_enabled), allow(dead_code))]

use std::time::Duration;

#[cfg(mining_cuda_native_enabled)]
use std::ffi::CStr;

#[repr(C)]
pub struct mining_cuda_solver_config {
    pub batch_size: usize,
    pub by_segment: bool,
    pub precompute_refs: bool,
    pub generate_passwords_on_gpu: bool,
}

#[repr(C)]
pub struct mining_cuda_job {
    pub seed_ptr: *const u8,
    pub seed_len: usize,
    pub pass_prefix_ptr: *const u8,
    pub pass_prefix_len: usize,
    pub time_cost: u32,
    pub memory_cost_kib: u32,
    pub parallelism: u32,
    pub difficulty_bits: i32,
}

#[repr(C)]
pub struct mining_cuda_benchmark_result {
    pub batch_size: usize,
    pub by_segment: bool,
    pub precompute_refs: bool,
    pub generate_passwords_on_gpu: bool,
    pub attempts: i64,
    pub elapsed_ms: i64,
    pub attempts_per_second: f64,
}

#[repr(C)]
pub struct mining_cuda_mine_result {
    pub found: bool,
    pub nonce: u64,
    pub attempts: i64,
    pub digest_hex: [u8; 65],
}

#[repr(C)]
pub struct mining_cuda_rpow2_solver_config {
    pub batch_size: u64,
    pub threads_per_block: u32,
    pub nonces_per_thread: u32,
    pub max_blocks: u32,
    pub early_exit: bool,
}

#[repr(C)]
pub struct mining_cuda_rpow2_job {
    pub nonce_prefix_ptr: *const u8,
    pub nonce_prefix_len: usize,
    pub difficulty_bits: u32,
}

#[repr(C)]
pub struct mining_cuda_rpow2_mine_result {
    pub found: bool,
    pub nonce: u64,
    pub attempts: i64,
    pub digest_hex: [u8; 65],
}

#[repr(C)]
pub struct mining_cuda_h98hash_job {
    pub challenge_ptr: *const u8,
    pub challenge_len: usize,
    pub nonce_prefix_ptr: *const u8,
    pub nonce_prefix_len: usize,
    pub difficulty_bits: u32,
}

#[repr(C)]
pub struct mining_cuda_h98hash_mine_result {
    pub found: bool,
    pub nonce_tail: u64,
    pub attempts: i64,
    pub digest_hex: [u8; 65],
}

#[repr(C)]
pub struct mining_cuda_h256hash_job {
    pub challenge_ptr: *const u8,
    pub challenge_len: usize,
    pub target_ptr: *const u8,
    pub target_len: usize,
}

#[repr(C)]
pub struct mining_cuda_h256hash_mine_result {
    pub found: bool,
    pub nonce: u64,
    pub attempts: i64,
    pub digest_hex: [u8; 65],
}

#[repr(C)]
pub struct mining_cuda_rpow2_benchmark_result {
    pub attempts: u64,
    pub batches: u64,
    pub elapsed_ms: f64,
    pub kernel_ms: f64,
    pub empty_launch_us: f64,
    pub kernel_hashrate: f64,
    pub effective_hashrate: f64,
}

#[repr(C)]
pub struct mining_cuda_device_info {
    pub device_index: usize,
    pub global_memory_bytes: u64,
    pub max_alloc_bytes: u64,
    pub compute_units: u32,
    pub max_threads_per_block: u32,
    pub warp_size: u32,
    pub shared_memory_per_block_bytes: u64,
    pub device_id: [u8; 32],
    pub name: [u8; 128],
}

#[repr(C)]
pub struct mining_cuda_session {
    _private: [u8; 0],
}

#[repr(C)]
pub struct mining_cuda_rpow2_session {
    _private: [u8; 0],
}

#[repr(C)]
pub struct mining_cuda_h98hash_session {
    _private: [u8; 0],
}

#[repr(C)]
pub struct mining_cuda_h256hash_session {
    _private: [u8; 0],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaSolverConfig {
    pub batch_size: usize,
    pub by_segment: bool,
    pub precompute_refs: bool,
    pub generate_passwords_on_gpu: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaJob<'a> {
    pub seed_bytes: &'a [u8],
    pub pass_prefix: &'a [u8],
    pub time_cost: u32,
    pub memory_cost_kib: u32,
    pub parallelism: u32,
    pub difficulty_bits: i32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaDeviceInfo {
    pub device_index: usize,
    pub global_memory_bytes: u64,
    pub max_alloc_bytes: u64,
    pub compute_units: u32,
    pub max_threads_per_block: u32,
    pub warp_size: u32,
    pub shared_memory_per_block_bytes: u64,
    pub device_id: String,
    pub name: String,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CudaBenchmarkResult {
    pub config: CudaSolverConfig,
    pub attempts: i64,
    pub elapsed: Duration,
    pub attempts_per_second: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaMineResult {
    pub found: bool,
    pub nonce: u64,
    pub attempts: i64,
    pub digest_hex: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rpow2CudaSolverConfig {
    pub batch_size: u64,
    pub threads_per_block: u32,
    pub nonces_per_thread: u32,
    pub max_blocks: u32,
    pub early_exit: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rpow2CudaJob<'a> {
    pub nonce_prefix: &'a [u8],
    pub difficulty_bits: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rpow2CudaMineResult {
    pub found: bool,
    pub nonce: u64,
    pub attempts: i64,
    pub digest_hex: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct H98HashCudaJob<'a> {
    pub challenge: &'a [u8],
    pub nonce_prefix: &'a [u8],
    pub difficulty_bits: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct H98HashCudaMineResult {
    pub found: bool,
    pub nonce_tail: u64,
    pub attempts: i64,
    pub digest_hex: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct H256HashCudaJob<'a> {
    /// 32-byte challenge (from on-chain `getChallenge(address)`).
    pub challenge: &'a [u8],
    /// 32-byte big-endian uint256 difficulty target. Hash < target wins.
    pub target: &'a [u8],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct H256HashCudaMineResult {
    pub found: bool,
    /// Lower 64 bits of the uint256 nonce. High 24 bytes are always zero.
    pub nonce: u64,
    pub attempts: i64,
    pub digest_hex: String,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rpow2CudaBenchmarkResult {
    pub attempts: u64,
    pub batches: u64,
    pub elapsed: Duration,
    pub kernel_elapsed: Duration,
    pub empty_launch: Duration,
    pub kernel_hashrate: f64,
    pub effective_hashrate: f64,
}

pub struct CudaMiningSession {
    raw: *mut mining_cuda_session,
}

pub struct Rpow2CudaMiningSession {
    raw: *mut mining_cuda_rpow2_session,
}

pub struct H98HashCudaMiningSession {
    raw: *mut mining_cuda_h98hash_session,
}

pub struct H256HashCudaMiningSession {
    raw: *mut mining_cuda_h256hash_session,
}

unsafe impl Send for CudaMiningSession {}
unsafe impl Send for Rpow2CudaMiningSession {}
unsafe impl Send for H98HashCudaMiningSession {}
unsafe impl Send for H256HashCudaMiningSession {}

unsafe extern "C" {
    fn mining_cuda_is_available() -> bool;
    fn mining_cuda_validate() -> bool;
    fn mining_cuda_device_count() -> usize;
    fn mining_cuda_get_device_info(
        device_index: usize,
        result: *mut mining_cuda_device_info,
    ) -> bool;
    fn mining_cuda_last_error_message() -> *const std::ffi::c_char;
    fn mining_cuda_default_solver_config(
        device_index: usize,
        job: *const mining_cuda_job,
        result: *mut mining_cuda_solver_config,
    ) -> bool;
    fn mining_cuda_find_best_benchmark_config(
        device_index: usize,
        result: *mut mining_cuda_benchmark_result,
    ) -> bool;
    fn mining_cuda_mine_batch(
        device_index: usize,
        job: *const mining_cuda_job,
        config: *const mining_cuda_solver_config,
        start_nonce: u64,
        result: *mut mining_cuda_mine_result,
    ) -> bool;
    fn mining_cuda_session_create(
        device_index: usize,
        job: *const mining_cuda_job,
        config: *const mining_cuda_solver_config,
        start_nonce: u64,
    ) -> *mut mining_cuda_session;
    fn mining_cuda_session_mine_next_batch(
        session: *mut mining_cuda_session,
        result: *mut mining_cuda_mine_result,
    ) -> bool;
    fn mining_cuda_session_destroy(session: *mut mining_cuda_session);
    fn mining_cuda_rpow2_mine_batch(
        device_index: usize,
        job: *const mining_cuda_rpow2_job,
        config: *const mining_cuda_rpow2_solver_config,
        start_nonce: u64,
        result: *mut mining_cuda_rpow2_mine_result,
    ) -> bool;
    fn mining_cuda_rpow2_session_create(
        device_index: usize,
        job: *const mining_cuda_rpow2_job,
        config: *const mining_cuda_rpow2_solver_config,
        start_nonce: u64,
    ) -> *mut mining_cuda_rpow2_session;
    fn mining_cuda_rpow2_session_mine_next_batch(
        session: *mut mining_cuda_rpow2_session,
        result: *mut mining_cuda_rpow2_mine_result,
    ) -> bool;
    fn mining_cuda_rpow2_session_destroy(session: *mut mining_cuda_rpow2_session);
    fn mining_cuda_rpow2_benchmark(
        device_index: usize,
        job: *const mining_cuda_rpow2_job,
        config: *const mining_cuda_rpow2_solver_config,
        duration_ms: u32,
        result: *mut mining_cuda_rpow2_benchmark_result,
    ) -> bool;
    fn mining_cuda_h98hash_mine_batch(
        device_index: usize,
        job: *const mining_cuda_h98hash_job,
        config: *const mining_cuda_rpow2_solver_config,
        start_nonce: u64,
        result: *mut mining_cuda_h98hash_mine_result,
    ) -> bool;
    fn mining_cuda_h98hash_session_create(
        device_index: usize,
        job: *const mining_cuda_h98hash_job,
        config: *const mining_cuda_rpow2_solver_config,
        start_nonce: u64,
    ) -> *mut mining_cuda_h98hash_session;
    fn mining_cuda_h98hash_session_mine_next_batch(
        session: *mut mining_cuda_h98hash_session,
        result: *mut mining_cuda_h98hash_mine_result,
    ) -> bool;
    fn mining_cuda_h98hash_session_destroy(session: *mut mining_cuda_h98hash_session);
    fn mining_cuda_h256hash_mine_batch(
        device_index: usize,
        job: *const mining_cuda_h256hash_job,
        config: *const mining_cuda_rpow2_solver_config,
        start_nonce: u64,
        result: *mut mining_cuda_h256hash_mine_result,
    ) -> bool;
    fn mining_cuda_h256hash_session_create(
        device_index: usize,
        job: *const mining_cuda_h256hash_job,
        config: *const mining_cuda_rpow2_solver_config,
        start_nonce: u64,
    ) -> *mut mining_cuda_h256hash_session;
    fn mining_cuda_h256hash_session_mine_next_batch(
        session: *mut mining_cuda_h256hash_session,
        result: *mut mining_cuda_h256hash_mine_result,
    ) -> bool;
    fn mining_cuda_h256hash_session_destroy(session: *mut mining_cuda_h256hash_session);
    fn mining_argon2id_hash_raw(
        password_ptr: *const u8,
        password_len: usize,
        salt_ptr: *const u8,
        salt_len: usize,
        time_cost: u32,
        memory_cost_kib: u32,
        parallelism: u32,
        digest_ptr: *mut u8,
        digest_len: usize,
    ) -> bool;
}

pub fn is_supported_target() -> bool {
    cfg!(mining_cuda_supported_target)
}

pub fn is_available() -> Result<bool, String> {
    #[cfg(not(mining_cuda_supported_target))]
    {
        Ok(false)
    }
    #[cfg(all(mining_cuda_supported_target, not(mining_cuda_native_enabled)))]
    {
        Err("CUDA native backend is not enabled in this build.".to_string())
    }
    #[cfg(mining_cuda_native_enabled)]
    unsafe {
        if mining_cuda_is_available() {
            Ok(true)
        } else {
            Err(last_error_message())
        }
    }
}

pub fn validate() -> Result<(), String> {
    #[cfg(not(mining_cuda_native_enabled))]
    {
        Err("CUDA backend is not enabled on this platform.".to_string())
    }
    #[cfg(mining_cuda_native_enabled)]
    unsafe {
        if mining_cuda_validate() {
            Ok(())
        } else {
            Err(last_error_message())
        }
    }
}

pub fn list_devices() -> Result<Vec<CudaDeviceInfo>, String> {
    #[cfg(not(mining_cuda_native_enabled))]
    {
        Ok(Vec::new())
    }
    #[cfg(mining_cuda_native_enabled)]
    unsafe {
        let count = mining_cuda_device_count();
        let mut devices = Vec::with_capacity(count);
        for device_index in 0..count {
            let mut raw = mining_cuda_device_info {
                device_index,
                global_memory_bytes: 0,
                max_alloc_bytes: 0,
                compute_units: 0,
                max_threads_per_block: 0,
                warp_size: 0,
                shared_memory_per_block_bytes: 0,
                device_id: [0; 32],
                name: [0; 128],
            };
            if !mining_cuda_get_device_info(device_index, &mut raw) {
                return Err(last_error_message());
            }
            devices.push(CudaDeviceInfo {
                device_index: raw.device_index,
                global_memory_bytes: raw.global_memory_bytes,
                max_alloc_bytes: raw.max_alloc_bytes,
                compute_units: raw.compute_units,
                max_threads_per_block: raw.max_threads_per_block,
                warp_size: raw.warp_size,
                shared_memory_per_block_bytes: raw.shared_memory_per_block_bytes,
                device_id: decode_c_string(&raw.device_id),
                name: decode_c_string(&raw.name),
            });
        }
        Ok(devices)
    }
}

pub fn default_solver_config(
    device_index: usize,
    job: &CudaJob<'_>,
) -> Result<CudaSolverConfig, String> {
    #[cfg(not(mining_cuda_native_enabled))]
    {
        let _ = (device_index, job);
        Err("CUDA backend is not enabled on this platform.".to_string())
    }
    #[cfg(mining_cuda_native_enabled)]
    unsafe {
        let raw_job = mining_cuda_job {
            seed_ptr: job.seed_bytes.as_ptr(),
            seed_len: job.seed_bytes.len(),
            pass_prefix_ptr: job.pass_prefix.as_ptr(),
            pass_prefix_len: job.pass_prefix.len(),
            time_cost: job.time_cost,
            memory_cost_kib: job.memory_cost_kib,
            parallelism: job.parallelism,
            difficulty_bits: job.difficulty_bits,
        };
        let mut raw_config = mining_cuda_solver_config {
            batch_size: 0,
            by_segment: false,
            precompute_refs: false,
            generate_passwords_on_gpu: true,
        };
        if !mining_cuda_default_solver_config(device_index, &raw_job, &mut raw_config) {
            return Err(last_error_message());
        }
        Ok(CudaSolverConfig {
            batch_size: raw_config.batch_size,
            by_segment: raw_config.by_segment,
            precompute_refs: raw_config.precompute_refs,
            generate_passwords_on_gpu: raw_config.generate_passwords_on_gpu,
        })
    }
}

pub fn find_best_benchmark_config(device_index: usize) -> Result<CudaBenchmarkResult, String> {
    #[cfg(not(mining_cuda_native_enabled))]
    {
        let _ = device_index;
        Err("CUDA backend is not enabled on this platform.".to_string())
    }
    #[cfg(mining_cuda_native_enabled)]
    unsafe {
        let mut raw = mining_cuda_benchmark_result {
            batch_size: 0,
            by_segment: false,
            precompute_refs: false,
            generate_passwords_on_gpu: true,
            attempts: 0,
            elapsed_ms: 0,
            attempts_per_second: 0.0,
        };
        if !mining_cuda_find_best_benchmark_config(device_index, &mut raw) {
            return Err(last_error_message());
        }
        Ok(CudaBenchmarkResult {
            config: CudaSolverConfig {
                batch_size: raw.batch_size,
                by_segment: raw.by_segment,
                precompute_refs: raw.precompute_refs,
                generate_passwords_on_gpu: raw.generate_passwords_on_gpu,
            },
            attempts: raw.attempts,
            elapsed: Duration::from_millis(raw.elapsed_ms.max(0) as u64),
            attempts_per_second: raw.attempts_per_second,
        })
    }
}

pub fn mine_batch(
    device_index: usize,
    job: &CudaJob<'_>,
    config: CudaSolverConfig,
    start_nonce: u64,
) -> Result<CudaMineResult, String> {
    #[cfg(not(mining_cuda_native_enabled))]
    {
        let _ = (device_index, job, config, start_nonce);
        Err("CUDA backend is not enabled on this platform.".to_string())
    }
    #[cfg(mining_cuda_native_enabled)]
    unsafe {
        let raw_job = mining_cuda_job {
            seed_ptr: job.seed_bytes.as_ptr(),
            seed_len: job.seed_bytes.len(),
            pass_prefix_ptr: job.pass_prefix.as_ptr(),
            pass_prefix_len: job.pass_prefix.len(),
            time_cost: job.time_cost,
            memory_cost_kib: job.memory_cost_kib,
            parallelism: job.parallelism,
            difficulty_bits: job.difficulty_bits,
        };
        let raw_config = mining_cuda_solver_config {
            batch_size: config.batch_size,
            by_segment: config.by_segment,
            precompute_refs: config.precompute_refs,
            generate_passwords_on_gpu: config.generate_passwords_on_gpu,
        };
        let mut raw_result = mining_cuda_mine_result {
            found: false,
            nonce: 0,
            attempts: 0,
            digest_hex: [0; 65],
        };
        if !mining_cuda_mine_batch(
            device_index,
            &raw_job,
            &raw_config,
            start_nonce,
            &mut raw_result,
        ) {
            return Err(last_error_message());
        }
        let digest_len = raw_result
            .digest_hex
            .iter()
            .position(|byte| *byte == 0)
            .unwrap_or(raw_result.digest_hex.len());
        let digest_hex = String::from_utf8_lossy(&raw_result.digest_hex[..digest_len]).to_string();
        Ok(CudaMineResult {
            found: raw_result.found,
            nonce: raw_result.nonce,
            attempts: raw_result.attempts,
            digest_hex,
        })
    }
}

pub fn rpow2_mine_batch(
    device_index: usize,
    job: &Rpow2CudaJob<'_>,
    config: Rpow2CudaSolverConfig,
    start_nonce: u64,
) -> Result<Rpow2CudaMineResult, String> {
    #[cfg(not(mining_cuda_native_enabled))]
    {
        let _ = (device_index, job, config, start_nonce);
        Err("CUDA backend is not enabled on this platform.".to_string())
    }
    #[cfg(mining_cuda_native_enabled)]
    unsafe {
        let raw_job = mining_cuda_rpow2_job {
            nonce_prefix_ptr: job.nonce_prefix.as_ptr(),
            nonce_prefix_len: job.nonce_prefix.len(),
            difficulty_bits: job.difficulty_bits,
        };
        let raw_config = mining_cuda_rpow2_solver_config {
            batch_size: config.batch_size,
            threads_per_block: config.threads_per_block,
            nonces_per_thread: config.nonces_per_thread,
            max_blocks: config.max_blocks,
            early_exit: config.early_exit,
        };
        let mut raw_result = mining_cuda_rpow2_mine_result {
            found: false,
            nonce: 0,
            attempts: 0,
            digest_hex: [0; 65],
        };
        if !mining_cuda_rpow2_mine_batch(
            device_index,
            &raw_job,
            &raw_config,
            start_nonce,
            &mut raw_result,
        ) {
            return Err(last_error_message());
        }
        let digest_len = raw_result
            .digest_hex
            .iter()
            .position(|byte| *byte == 0)
            .unwrap_or(raw_result.digest_hex.len());
        let digest_hex = String::from_utf8_lossy(&raw_result.digest_hex[..digest_len]).to_string();
        Ok(Rpow2CudaMineResult {
            found: raw_result.found,
            nonce: raw_result.nonce,
            attempts: raw_result.attempts,
            digest_hex,
        })
    }
}

pub fn rpow2_create_session(
    device_index: usize,
    job: &Rpow2CudaJob<'_>,
    config: Rpow2CudaSolverConfig,
    start_nonce: u64,
) -> Result<Rpow2CudaMiningSession, String> {
    #[cfg(not(mining_cuda_native_enabled))]
    {
        let _ = (device_index, job, config, start_nonce);
        Err("CUDA backend is not enabled on this platform.".to_string())
    }
    #[cfg(mining_cuda_native_enabled)]
    unsafe {
        let raw_job = mining_cuda_rpow2_job {
            nonce_prefix_ptr: job.nonce_prefix.as_ptr(),
            nonce_prefix_len: job.nonce_prefix.len(),
            difficulty_bits: job.difficulty_bits,
        };
        let raw_config = mining_cuda_rpow2_solver_config {
            batch_size: config.batch_size,
            threads_per_block: config.threads_per_block,
            nonces_per_thread: config.nonces_per_thread,
            max_blocks: config.max_blocks,
            early_exit: config.early_exit,
        };
        let raw =
            mining_cuda_rpow2_session_create(device_index, &raw_job, &raw_config, start_nonce);
        if raw.is_null() {
            Err(last_error_message())
        } else {
            Ok(Rpow2CudaMiningSession { raw })
        }
    }
}

pub fn rpow2_benchmark(
    device_index: usize,
    job: &Rpow2CudaJob<'_>,
    config: Rpow2CudaSolverConfig,
    duration: Duration,
) -> Result<Rpow2CudaBenchmarkResult, String> {
    #[cfg(not(mining_cuda_native_enabled))]
    {
        let _ = (device_index, job, config, duration);
        Err("CUDA backend is not enabled on this platform.".to_string())
    }
    #[cfg(mining_cuda_native_enabled)]
    unsafe {
        let raw_job = mining_cuda_rpow2_job {
            nonce_prefix_ptr: job.nonce_prefix.as_ptr(),
            nonce_prefix_len: job.nonce_prefix.len(),
            difficulty_bits: job.difficulty_bits,
        };
        let raw_config = mining_cuda_rpow2_solver_config {
            batch_size: config.batch_size,
            threads_per_block: config.threads_per_block,
            nonces_per_thread: config.nonces_per_thread,
            max_blocks: config.max_blocks,
            early_exit: config.early_exit,
        };
        let mut raw_result = mining_cuda_rpow2_benchmark_result {
            attempts: 0,
            batches: 0,
            elapsed_ms: 0.0,
            kernel_ms: 0.0,
            empty_launch_us: 0.0,
            kernel_hashrate: 0.0,
            effective_hashrate: 0.0,
        };
        let duration_ms = duration.as_millis().min(u128::from(u32::MAX)) as u32;
        if !mining_cuda_rpow2_benchmark(
            device_index,
            &raw_job,
            &raw_config,
            duration_ms.max(1),
            &mut raw_result,
        ) {
            return Err(last_error_message());
        }
        Ok(Rpow2CudaBenchmarkResult {
            attempts: raw_result.attempts,
            batches: raw_result.batches,
            elapsed: Duration::from_secs_f64((raw_result.elapsed_ms / 1000.0).max(0.0)),
            kernel_elapsed: Duration::from_secs_f64((raw_result.kernel_ms / 1000.0).max(0.0)),
            empty_launch: Duration::from_secs_f64(
                (raw_result.empty_launch_us / 1_000_000.0).max(0.0),
            ),
            kernel_hashrate: raw_result.kernel_hashrate,
            effective_hashrate: raw_result.effective_hashrate,
        })
    }
}

pub fn h98hash_mine_batch(
    device_index: usize,
    job: &H98HashCudaJob<'_>,
    config: Rpow2CudaSolverConfig,
    start_nonce: u64,
) -> Result<H98HashCudaMineResult, String> {
    #[cfg(not(mining_cuda_native_enabled))]
    {
        let _ = (device_index, job, config, start_nonce);
        Err("CUDA backend is not enabled on this platform.".to_string())
    }
    #[cfg(mining_cuda_native_enabled)]
    unsafe {
        let raw_job = mining_cuda_h98hash_job {
            challenge_ptr: job.challenge.as_ptr(),
            challenge_len: job.challenge.len(),
            nonce_prefix_ptr: job.nonce_prefix.as_ptr(),
            nonce_prefix_len: job.nonce_prefix.len(),
            difficulty_bits: job.difficulty_bits,
        };
        let raw_config = mining_cuda_rpow2_solver_config {
            batch_size: config.batch_size,
            threads_per_block: config.threads_per_block,
            nonces_per_thread: config.nonces_per_thread,
            max_blocks: config.max_blocks,
            early_exit: config.early_exit,
        };
        let mut raw_result = mining_cuda_h98hash_mine_result {
            found: false,
            nonce_tail: 0,
            attempts: 0,
            digest_hex: [0; 65],
        };
        if !mining_cuda_h98hash_mine_batch(
            device_index,
            &raw_job,
            &raw_config,
            start_nonce,
            &mut raw_result,
        ) {
            return Err(last_error_message());
        }
        let digest_len = raw_result
            .digest_hex
            .iter()
            .position(|byte| *byte == 0)
            .unwrap_or(raw_result.digest_hex.len());
        let digest_hex = String::from_utf8_lossy(&raw_result.digest_hex[..digest_len]).to_string();
        Ok(H98HashCudaMineResult {
            found: raw_result.found,
            nonce_tail: raw_result.nonce_tail,
            attempts: raw_result.attempts,
            digest_hex,
        })
    }
}

pub fn h98hash_create_session(
    device_index: usize,
    job: &H98HashCudaJob<'_>,
    config: Rpow2CudaSolverConfig,
    start_nonce: u64,
) -> Result<H98HashCudaMiningSession, String> {
    #[cfg(not(mining_cuda_native_enabled))]
    {
        let _ = (device_index, job, config, start_nonce);
        Err("CUDA backend is not enabled on this platform.".to_string())
    }
    #[cfg(mining_cuda_native_enabled)]
    unsafe {
        let raw_job = mining_cuda_h98hash_job {
            challenge_ptr: job.challenge.as_ptr(),
            challenge_len: job.challenge.len(),
            nonce_prefix_ptr: job.nonce_prefix.as_ptr(),
            nonce_prefix_len: job.nonce_prefix.len(),
            difficulty_bits: job.difficulty_bits,
        };
        let raw_config = mining_cuda_rpow2_solver_config {
            batch_size: config.batch_size,
            threads_per_block: config.threads_per_block,
            nonces_per_thread: config.nonces_per_thread,
            max_blocks: config.max_blocks,
            early_exit: config.early_exit,
        };
        let raw =
            mining_cuda_h98hash_session_create(device_index, &raw_job, &raw_config, start_nonce);
        if raw.is_null() {
            Err(last_error_message())
        } else {
            Ok(H98HashCudaMiningSession { raw })
        }
    }
}

pub fn h256hash_mine_batch(
    device_index: usize,
    job: &H256HashCudaJob<'_>,
    config: Rpow2CudaSolverConfig,
    start_nonce: u64,
) -> Result<H256HashCudaMineResult, String> {
    #[cfg(not(mining_cuda_native_enabled))]
    {
        let _ = (device_index, job, config, start_nonce);
        Err("CUDA backend is not enabled on this platform.".to_string())
    }
    #[cfg(mining_cuda_native_enabled)]
    unsafe {
        let raw_job = mining_cuda_h256hash_job {
            challenge_ptr: job.challenge.as_ptr(),
            challenge_len: job.challenge.len(),
            target_ptr: job.target.as_ptr(),
            target_len: job.target.len(),
        };
        let raw_config = mining_cuda_rpow2_solver_config {
            batch_size: config.batch_size,
            threads_per_block: config.threads_per_block,
            nonces_per_thread: config.nonces_per_thread,
            max_blocks: config.max_blocks,
            early_exit: config.early_exit,
        };
        let mut raw_result = mining_cuda_h256hash_mine_result {
            found: false,
            nonce: 0,
            attempts: 0,
            digest_hex: [0; 65],
        };
        if !mining_cuda_h256hash_mine_batch(
            device_index,
            &raw_job,
            &raw_config,
            start_nonce,
            &mut raw_result,
        ) {
            return Err(last_error_message());
        }
        let digest_len = raw_result
            .digest_hex
            .iter()
            .position(|byte| *byte == 0)
            .unwrap_or(raw_result.digest_hex.len());
        let digest_hex = String::from_utf8_lossy(&raw_result.digest_hex[..digest_len]).to_string();
        Ok(H256HashCudaMineResult {
            found: raw_result.found,
            nonce: raw_result.nonce,
            attempts: raw_result.attempts,
            digest_hex,
        })
    }
}

pub fn h256hash_create_session(
    device_index: usize,
    job: &H256HashCudaJob<'_>,
    config: Rpow2CudaSolverConfig,
    start_nonce: u64,
) -> Result<H256HashCudaMiningSession, String> {
    #[cfg(not(mining_cuda_native_enabled))]
    {
        let _ = (device_index, job, config, start_nonce);
        Err("CUDA backend is not enabled on this platform.".to_string())
    }
    #[cfg(mining_cuda_native_enabled)]
    unsafe {
        let raw_job = mining_cuda_h256hash_job {
            challenge_ptr: job.challenge.as_ptr(),
            challenge_len: job.challenge.len(),
            target_ptr: job.target.as_ptr(),
            target_len: job.target.len(),
        };
        let raw_config = mining_cuda_rpow2_solver_config {
            batch_size: config.batch_size,
            threads_per_block: config.threads_per_block,
            nonces_per_thread: config.nonces_per_thread,
            max_blocks: config.max_blocks,
            early_exit: config.early_exit,
        };
        let raw =
            mining_cuda_h256hash_session_create(device_index, &raw_job, &raw_config, start_nonce);
        if raw.is_null() {
            Err(last_error_message())
        } else {
            Ok(H256HashCudaMiningSession { raw })
        }
    }
}

pub fn create_session(
    device_index: usize,
    job: &CudaJob<'_>,
    config: CudaSolverConfig,
    start_nonce: u64,
) -> Result<CudaMiningSession, String> {
    #[cfg(not(mining_cuda_native_enabled))]
    {
        let _ = (device_index, job, config, start_nonce);
        Err("CUDA backend is not enabled on this platform.".to_string())
    }
    #[cfg(mining_cuda_native_enabled)]
    unsafe {
        let raw_job = mining_cuda_job {
            seed_ptr: job.seed_bytes.as_ptr(),
            seed_len: job.seed_bytes.len(),
            pass_prefix_ptr: job.pass_prefix.as_ptr(),
            pass_prefix_len: job.pass_prefix.len(),
            time_cost: job.time_cost,
            memory_cost_kib: job.memory_cost_kib,
            parallelism: job.parallelism,
            difficulty_bits: job.difficulty_bits,
        };
        let raw_config = mining_cuda_solver_config {
            batch_size: config.batch_size,
            by_segment: config.by_segment,
            precompute_refs: config.precompute_refs,
            generate_passwords_on_gpu: config.generate_passwords_on_gpu,
        };
        let raw = mining_cuda_session_create(device_index, &raw_job, &raw_config, start_nonce);
        if raw.is_null() {
            Err(last_error_message())
        } else {
            Ok(CudaMiningSession { raw })
        }
    }
}

impl CudaMiningSession {
    pub fn mine_next_batch(&mut self) -> Result<CudaMineResult, String> {
        #[cfg(not(mining_cuda_native_enabled))]
        {
            Err("CUDA backend is not enabled on this platform.".to_string())
        }
        #[cfg(mining_cuda_native_enabled)]
        unsafe {
            let mut raw_result = mining_cuda_mine_result {
                found: false,
                nonce: 0,
                attempts: 0,
                digest_hex: [0; 65],
            };
            if !mining_cuda_session_mine_next_batch(self.raw, &mut raw_result) {
                return Err(last_error_message());
            }
            let digest_len = raw_result
                .digest_hex
                .iter()
                .position(|byte| *byte == 0)
                .unwrap_or(raw_result.digest_hex.len());
            let digest_hex =
                String::from_utf8_lossy(&raw_result.digest_hex[..digest_len]).to_string();
            Ok(CudaMineResult {
                found: raw_result.found,
                nonce: raw_result.nonce,
                attempts: raw_result.attempts,
                digest_hex,
            })
        }
    }
}

impl Rpow2CudaMiningSession {
    pub fn mine_next_batch(&mut self) -> Result<Rpow2CudaMineResult, String> {
        #[cfg(not(mining_cuda_native_enabled))]
        {
            Err("CUDA backend is not enabled on this platform.".to_string())
        }
        #[cfg(mining_cuda_native_enabled)]
        unsafe {
            let mut raw_result = mining_cuda_rpow2_mine_result {
                found: false,
                nonce: 0,
                attempts: 0,
                digest_hex: [0; 65],
            };
            if !mining_cuda_rpow2_session_mine_next_batch(self.raw, &mut raw_result) {
                return Err(last_error_message());
            }
            let digest_len = raw_result
                .digest_hex
                .iter()
                .position(|byte| *byte == 0)
                .unwrap_or(raw_result.digest_hex.len());
            let digest_hex =
                String::from_utf8_lossy(&raw_result.digest_hex[..digest_len]).to_string();
            Ok(Rpow2CudaMineResult {
                found: raw_result.found,
                nonce: raw_result.nonce,
                attempts: raw_result.attempts,
                digest_hex,
            })
        }
    }
}

impl H98HashCudaMiningSession {
    pub fn mine_next_batch(&mut self) -> Result<H98HashCudaMineResult, String> {
        #[cfg(not(mining_cuda_native_enabled))]
        {
            Err("CUDA backend is not enabled on this platform.".to_string())
        }
        #[cfg(mining_cuda_native_enabled)]
        unsafe {
            let mut raw_result = mining_cuda_h98hash_mine_result {
                found: false,
                nonce_tail: 0,
                attempts: 0,
                digest_hex: [0; 65],
            };
            if !mining_cuda_h98hash_session_mine_next_batch(self.raw, &mut raw_result) {
                return Err(last_error_message());
            }
            let digest_len = raw_result
                .digest_hex
                .iter()
                .position(|byte| *byte == 0)
                .unwrap_or(raw_result.digest_hex.len());
            let digest_hex =
                String::from_utf8_lossy(&raw_result.digest_hex[..digest_len]).to_string();
            Ok(H98HashCudaMineResult {
                found: raw_result.found,
                nonce_tail: raw_result.nonce_tail,
                attempts: raw_result.attempts,
                digest_hex,
            })
        }
    }
}

impl H256HashCudaMiningSession {
    pub fn mine_next_batch(&mut self) -> Result<H256HashCudaMineResult, String> {
        #[cfg(not(mining_cuda_native_enabled))]
        {
            Err("CUDA backend is not enabled on this platform.".to_string())
        }
        #[cfg(mining_cuda_native_enabled)]
        unsafe {
            let mut raw_result = mining_cuda_h256hash_mine_result {
                found: false,
                nonce: 0,
                attempts: 0,
                digest_hex: [0; 65],
            };
            if !mining_cuda_h256hash_session_mine_next_batch(self.raw, &mut raw_result) {
                return Err(last_error_message());
            }
            let digest_len = raw_result
                .digest_hex
                .iter()
                .position(|byte| *byte == 0)
                .unwrap_or(raw_result.digest_hex.len());
            let digest_hex =
                String::from_utf8_lossy(&raw_result.digest_hex[..digest_len]).to_string();
            Ok(H256HashCudaMineResult {
                found: raw_result.found,
                nonce: raw_result.nonce,
                attempts: raw_result.attempts,
                digest_hex,
            })
        }
    }
}

impl Drop for CudaMiningSession {
    fn drop(&mut self) {
        #[cfg(mining_cuda_native_enabled)]
        unsafe {
            if !self.raw.is_null() {
                mining_cuda_session_destroy(self.raw);
                self.raw = std::ptr::null_mut();
            }
        }
    }
}

impl Drop for Rpow2CudaMiningSession {
    fn drop(&mut self) {
        #[cfg(mining_cuda_native_enabled)]
        unsafe {
            if !self.raw.is_null() {
                mining_cuda_rpow2_session_destroy(self.raw);
                self.raw = std::ptr::null_mut();
            }
        }
    }
}

impl Drop for H98HashCudaMiningSession {
    fn drop(&mut self) {
        #[cfg(mining_cuda_native_enabled)]
        unsafe {
            if !self.raw.is_null() {
                mining_cuda_h98hash_session_destroy(self.raw);
                self.raw = std::ptr::null_mut();
            }
        }
    }
}

impl Drop for H256HashCudaMiningSession {
    fn drop(&mut self) {
        #[cfg(mining_cuda_native_enabled)]
        unsafe {
            if !self.raw.is_null() {
                mining_cuda_h256hash_session_destroy(self.raw);
                self.raw = std::ptr::null_mut();
            }
        }
    }
}

pub fn argon2id_hash_raw(
    password: &[u8],
    salt: &[u8],
    time_cost: u32,
    memory_cost_kib: u32,
    parallelism: u32,
    digest: &mut [u8],
) -> Result<(), String> {
    #[cfg(not(mining_cuda_native_enabled))]
    {
        let _ = (
            password,
            salt,
            time_cost,
            memory_cost_kib,
            parallelism,
            digest,
        );
        Err("Native Argon2 backend is not enabled on this platform.".to_string())
    }
    #[cfg(mining_cuda_native_enabled)]
    unsafe {
        if mining_argon2id_hash_raw(
            password.as_ptr(),
            password.len(),
            salt.as_ptr(),
            salt.len(),
            time_cost,
            memory_cost_kib,
            parallelism,
            digest.as_mut_ptr(),
            digest.len(),
        ) {
            Ok(())
        } else {
            Err(last_error_message())
        }
    }
}

#[cfg(mining_cuda_native_enabled)]
fn decode_c_string(bytes: &[u8]) -> String {
    let len = bytes
        .iter()
        .position(|byte| *byte == 0)
        .unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..len]).trim().to_string()
}

#[cfg(mining_cuda_native_enabled)]
unsafe fn last_error_message() -> String {
    let ptr = unsafe { mining_cuda_last_error_message() };
    if ptr.is_null() {
        return "CUDA backend returned an unknown error.".to_string();
    }
    let text = unsafe { CStr::from_ptr(ptr) }
        .to_string_lossy()
        .trim()
        .to_string();
    if text.is_empty() {
        "CUDA backend returned an unknown error.".to_string()
    } else {
        text
    }
}
