#![allow(non_camel_case_types)]
#![cfg_attr(
    not(all(target_os = "macos", mining_metal_native_enabled)),
    allow(dead_code)
)]

use std::time::Duration;

#[cfg(all(target_os = "macos", mining_metal_native_enabled))]
use std::ffi::CStr;

#[repr(C)]
pub struct mining_metal_solver_config {
    pub batch_size: usize,
    pub by_segment: bool,
    pub precompute_refs: bool,
}

#[repr(C)]
pub struct mining_metal_job {
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
pub struct mining_metal_benchmark_result {
    pub batch_size: usize,
    pub by_segment: bool,
    pub precompute_refs: bool,
    pub attempts: i64,
    pub elapsed_ms: i64,
    pub attempts_per_second: f64,
}

#[repr(C)]
pub struct mining_metal_mine_result {
    pub found: bool,
    pub nonce: u64,
    pub attempts: i64,
    pub digest_hex: [u8; 65],
}

#[repr(C)]
pub struct mining_metal_device_info {
    pub device_index: usize,
    pub recommended_working_set_bytes: u64,
    pub max_buffer_bytes: u64,
    pub max_threadgroup_memory_bytes: u64,
    pub max_threads_per_group: u32,
    pub unified_memory: bool,
    pub low_power: bool,
    pub removable: bool,
    pub device_id: [u8; 64],
    pub name: [u8; 128],
}

#[repr(C)]
pub struct mining_metal_rpow2_solver_config {
    pub batch_size: u64,
}

#[repr(C)]
pub struct mining_metal_rpow2_job {
    pub nonce_prefix_ptr: *const u8,
    pub nonce_prefix_len: usize,
    pub difficulty_bits: u32,
}

#[repr(C)]
pub struct mining_metal_rpow2_mine_result {
    pub found: bool,
    pub nonce: u64,
    pub attempts: i64,
    pub digest_hex: [u8; 65],
}

#[repr(C)]
pub struct mining_metal_session {
    _private: [u8; 0],
}

#[repr(C)]
pub struct mining_metal_rpow2_session {
    _private: [u8; 0],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetalSolverConfig {
    pub batch_size: usize,
    pub by_segment: bool,
    pub precompute_refs: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetalJob<'a> {
    pub seed_bytes: &'a [u8],
    pub pass_prefix: &'a [u8],
    pub time_cost: u32,
    pub memory_cost_kib: u32,
    pub parallelism: u32,
    pub difficulty_bits: i32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetalDeviceInfo {
    pub device_index: usize,
    pub recommended_working_set_bytes: u64,
    pub max_buffer_bytes: u64,
    pub max_threadgroup_memory_bytes: u64,
    pub max_threads_per_group: u32,
    pub unified_memory: bool,
    pub low_power: bool,
    pub removable: bool,
    pub device_id: String,
    pub name: String,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MetalBenchmarkResult {
    pub config: MetalSolverConfig,
    pub attempts: i64,
    pub elapsed: Duration,
    pub attempts_per_second: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetalMineResult {
    pub found: bool,
    pub nonce: u64,
    pub attempts: i64,
    pub digest_hex: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rpow2MetalSolverConfig {
    pub batch_size: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rpow2MetalJob<'a> {
    pub nonce_prefix: &'a [u8],
    pub difficulty_bits: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rpow2MetalMineResult {
    pub found: bool,
    pub nonce: u64,
    pub attempts: i64,
    pub digest_hex: String,
}

pub struct MetalMiningSession {
    raw: *mut mining_metal_session,
}

pub struct Rpow2MetalMiningSession {
    raw: *mut mining_metal_rpow2_session,
}

unsafe extern "C" {
    fn mining_metal_is_available() -> bool;
    fn mining_metal_validate() -> bool;
    fn mining_metal_validate_device(device_index: usize) -> bool;
    fn mining_metal_device_count() -> usize;
    fn mining_metal_get_device_info(
        device_index: usize,
        result: *mut mining_metal_device_info,
    ) -> bool;
    fn mining_metal_last_error_message() -> *const std::ffi::c_char;
    fn mining_metal_default_solver_config(
        device_index: usize,
        job: *const mining_metal_job,
        result: *mut mining_metal_solver_config,
    ) -> bool;
    fn mining_metal_find_best_benchmark_config(
        device_index: usize,
        result: *mut mining_metal_benchmark_result,
    ) -> bool;
    fn mining_metal_mine_batch(
        device_index: usize,
        job: *const mining_metal_job,
        config: *const mining_metal_solver_config,
        start_nonce: u64,
        result: *mut mining_metal_mine_result,
    ) -> bool;
    fn mining_metal_session_create(
        device_index: usize,
        job: *const mining_metal_job,
        config: *const mining_metal_solver_config,
        start_nonce: u64,
    ) -> *mut mining_metal_session;
    fn mining_metal_session_mine_next_batch(
        session: *mut mining_metal_session,
        result: *mut mining_metal_mine_result,
    ) -> bool;
    fn mining_metal_session_destroy(session: *mut mining_metal_session);
    fn mining_metal_rpow2_mine_batch(
        device_index: usize,
        job: *const mining_metal_rpow2_job,
        config: *const mining_metal_rpow2_solver_config,
        start_nonce: u64,
        result: *mut mining_metal_rpow2_mine_result,
    ) -> bool;
    fn mining_metal_rpow2_session_create(
        device_index: usize,
        job: *const mining_metal_rpow2_job,
        config: *const mining_metal_rpow2_solver_config,
        start_nonce: u64,
    ) -> *mut mining_metal_rpow2_session;
    fn mining_metal_rpow2_session_mine_next_batch(
        session: *mut mining_metal_rpow2_session,
        result: *mut mining_metal_rpow2_mine_result,
    ) -> bool;
    fn mining_metal_rpow2_session_destroy(session: *mut mining_metal_rpow2_session);
}

pub fn is_available() -> Result<bool, String> {
    #[cfg(not(target_os = "macos"))]
    {
        Ok(false)
    }
    #[cfg(all(target_os = "macos", not(mining_metal_native_enabled)))]
    {
        Err("当前构建未启用 Metal 原生后端。".to_string())
    }
    #[cfg(all(target_os = "macos", mining_metal_native_enabled))]
    unsafe {
        if mining_metal_is_available() {
            Ok(true)
        } else {
            let message = last_error_message_if_any();
            if message.is_empty() {
                Ok(false)
            } else {
                Err(message)
            }
        }
    }
}

pub fn validate() -> Result<(), String> {
    #[cfg(not(all(target_os = "macos", mining_metal_native_enabled)))]
    {
        Err("当前平台未启用 Metal 后端".to_string())
    }
    #[cfg(all(target_os = "macos", mining_metal_native_enabled))]
    unsafe {
        if mining_metal_validate() {
            Ok(())
        } else {
            Err(last_error_message())
        }
    }
}

pub fn validate_device(device_index: usize) -> Result<(), String> {
    #[cfg(not(all(target_os = "macos", mining_metal_native_enabled)))]
    {
        let _ = device_index;
        Err("当前平台未启用 Metal 后端".to_string())
    }
    #[cfg(all(target_os = "macos", mining_metal_native_enabled))]
    unsafe {
        if mining_metal_validate_device(device_index) {
            Ok(())
        } else {
            Err(last_error_message())
        }
    }
}

pub fn list_devices() -> Result<Vec<MetalDeviceInfo>, String> {
    #[cfg(not(all(target_os = "macos", mining_metal_native_enabled)))]
    {
        Ok(Vec::new())
    }
    #[cfg(all(target_os = "macos", mining_metal_native_enabled))]
    unsafe {
        let count = mining_metal_device_count();
        if count == 0 {
            let message = last_error_message_if_any();
            if !message.is_empty() {
                return Err(message);
            }
        }
        let mut devices = Vec::with_capacity(count);
        for device_index in 0..count {
            let mut raw = mining_metal_device_info {
                device_index,
                recommended_working_set_bytes: 0,
                max_buffer_bytes: 0,
                max_threadgroup_memory_bytes: 0,
                max_threads_per_group: 0,
                unified_memory: false,
                low_power: false,
                removable: false,
                device_id: [0; 64],
                name: [0; 128],
            };
            if !mining_metal_get_device_info(device_index, &mut raw) {
                return Err(last_error_message());
            }
            devices.push(MetalDeviceInfo {
                device_index: raw.device_index,
                recommended_working_set_bytes: raw.recommended_working_set_bytes,
                max_buffer_bytes: raw.max_buffer_bytes,
                max_threadgroup_memory_bytes: raw.max_threadgroup_memory_bytes,
                max_threads_per_group: raw.max_threads_per_group,
                unified_memory: raw.unified_memory,
                low_power: raw.low_power,
                removable: raw.removable,
                device_id: decode_c_string(&raw.device_id),
                name: decode_c_string(&raw.name),
            });
        }
        Ok(devices)
    }
}

pub fn default_solver_config(
    device_index: usize,
    job: &MetalJob<'_>,
) -> Result<MetalSolverConfig, String> {
    #[cfg(not(all(target_os = "macos", mining_metal_native_enabled)))]
    {
        let _ = (device_index, job);
        Err("当前平台未启用 Metal 后端".to_string())
    }
    #[cfg(all(target_os = "macos", mining_metal_native_enabled))]
    unsafe {
        let raw_job = mining_metal_job {
            seed_ptr: job.seed_bytes.as_ptr(),
            seed_len: job.seed_bytes.len(),
            pass_prefix_ptr: job.pass_prefix.as_ptr(),
            pass_prefix_len: job.pass_prefix.len(),
            time_cost: job.time_cost,
            memory_cost_kib: job.memory_cost_kib,
            parallelism: job.parallelism,
            difficulty_bits: job.difficulty_bits,
        };
        let mut raw_config = mining_metal_solver_config {
            batch_size: 0,
            by_segment: false,
            precompute_refs: false,
        };
        if !mining_metal_default_solver_config(device_index, &raw_job, &mut raw_config) {
            return Err(last_error_message());
        }
        Ok(MetalSolverConfig {
            batch_size: raw_config.batch_size,
            by_segment: raw_config.by_segment,
            precompute_refs: raw_config.precompute_refs,
        })
    }
}

pub fn find_best_benchmark_config(device_index: usize) -> Result<MetalBenchmarkResult, String> {
    #[cfg(not(all(target_os = "macos", mining_metal_native_enabled)))]
    {
        let _ = device_index;
        Err("当前平台未启用 Metal 后端".to_string())
    }
    #[cfg(all(target_os = "macos", mining_metal_native_enabled))]
    unsafe {
        let mut raw = mining_metal_benchmark_result {
            batch_size: 0,
            by_segment: false,
            precompute_refs: false,
            attempts: 0,
            elapsed_ms: 0,
            attempts_per_second: 0.0,
        };
        if !mining_metal_find_best_benchmark_config(device_index, &mut raw) {
            return Err(last_error_message());
        }
        Ok(MetalBenchmarkResult {
            config: MetalSolverConfig {
                batch_size: raw.batch_size,
                by_segment: raw.by_segment,
                precompute_refs: raw.precompute_refs,
            },
            attempts: raw.attempts,
            elapsed: Duration::from_millis(raw.elapsed_ms.max(0) as u64),
            attempts_per_second: raw.attempts_per_second,
        })
    }
}

pub fn mine_batch(
    device_index: usize,
    job: &MetalJob<'_>,
    config: MetalSolverConfig,
    start_nonce: u64,
) -> Result<MetalMineResult, String> {
    #[cfg(not(all(target_os = "macos", mining_metal_native_enabled)))]
    {
        let _ = (device_index, job, config, start_nonce);
        Err("当前平台未启用 Metal 后端".to_string())
    }
    #[cfg(all(target_os = "macos", mining_metal_native_enabled))]
    unsafe {
        let raw_job = mining_metal_job {
            seed_ptr: job.seed_bytes.as_ptr(),
            seed_len: job.seed_bytes.len(),
            pass_prefix_ptr: job.pass_prefix.as_ptr(),
            pass_prefix_len: job.pass_prefix.len(),
            time_cost: job.time_cost,
            memory_cost_kib: job.memory_cost_kib,
            parallelism: job.parallelism,
            difficulty_bits: job.difficulty_bits,
        };
        let raw_config = mining_metal_solver_config {
            batch_size: config.batch_size,
            by_segment: config.by_segment,
            precompute_refs: config.precompute_refs,
        };
        let mut raw_result = mining_metal_mine_result {
            found: false,
            nonce: 0,
            attempts: 0,
            digest_hex: [0; 65],
        };
        if !mining_metal_mine_batch(
            device_index,
            &raw_job,
            &raw_config,
            start_nonce,
            &mut raw_result,
        ) {
            return Err(last_error_message());
        }
        Ok(to_mine_result(raw_result))
    }
}

pub fn rpow2_mine_batch(
    device_index: usize,
    job: &Rpow2MetalJob<'_>,
    config: Rpow2MetalSolverConfig,
    start_nonce: u64,
) -> Result<Rpow2MetalMineResult, String> {
    #[cfg(not(all(target_os = "macos", mining_metal_native_enabled)))]
    {
        let _ = (device_index, job, config, start_nonce);
        Err("当前平台未启用 Metal 后端".to_string())
    }
    #[cfg(all(target_os = "macos", mining_metal_native_enabled))]
    unsafe {
        let raw_job = mining_metal_rpow2_job {
            nonce_prefix_ptr: job.nonce_prefix.as_ptr(),
            nonce_prefix_len: job.nonce_prefix.len(),
            difficulty_bits: job.difficulty_bits,
        };
        let raw_config = mining_metal_rpow2_solver_config {
            batch_size: config.batch_size,
        };
        let mut raw_result = mining_metal_rpow2_mine_result {
            found: false,
            nonce: 0,
            attempts: 0,
            digest_hex: [0; 65],
        };
        if !mining_metal_rpow2_mine_batch(
            device_index,
            &raw_job,
            &raw_config,
            start_nonce,
            &mut raw_result,
        ) {
            return Err(last_error_message());
        }
        Ok(to_rpow2_mine_result(raw_result))
    }
}

pub fn rpow2_create_session(
    device_index: usize,
    job: &Rpow2MetalJob<'_>,
    config: Rpow2MetalSolverConfig,
    start_nonce: u64,
) -> Result<Rpow2MetalMiningSession, String> {
    #[cfg(not(all(target_os = "macos", mining_metal_native_enabled)))]
    {
        let _ = (device_index, job, config, start_nonce);
        Err("当前平台未启用 Metal 后端".to_string())
    }
    #[cfg(all(target_os = "macos", mining_metal_native_enabled))]
    unsafe {
        let raw_job = mining_metal_rpow2_job {
            nonce_prefix_ptr: job.nonce_prefix.as_ptr(),
            nonce_prefix_len: job.nonce_prefix.len(),
            difficulty_bits: job.difficulty_bits,
        };
        let raw_config = mining_metal_rpow2_solver_config {
            batch_size: config.batch_size,
        };
        let raw =
            mining_metal_rpow2_session_create(device_index, &raw_job, &raw_config, start_nonce);
        if raw.is_null() {
            Err(last_error_message())
        } else {
            Ok(Rpow2MetalMiningSession { raw })
        }
    }
}

pub fn create_session(
    device_index: usize,
    job: &MetalJob<'_>,
    config: MetalSolverConfig,
    start_nonce: u64,
) -> Result<MetalMiningSession, String> {
    #[cfg(not(all(target_os = "macos", mining_metal_native_enabled)))]
    {
        let _ = (device_index, job, config, start_nonce);
        Err("当前平台未启用 Metal 后端".to_string())
    }
    #[cfg(all(target_os = "macos", mining_metal_native_enabled))]
    unsafe {
        let raw_job = mining_metal_job {
            seed_ptr: job.seed_bytes.as_ptr(),
            seed_len: job.seed_bytes.len(),
            pass_prefix_ptr: job.pass_prefix.as_ptr(),
            pass_prefix_len: job.pass_prefix.len(),
            time_cost: job.time_cost,
            memory_cost_kib: job.memory_cost_kib,
            parallelism: job.parallelism,
            difficulty_bits: job.difficulty_bits,
        };
        let raw_config = mining_metal_solver_config {
            batch_size: config.batch_size,
            by_segment: config.by_segment,
            precompute_refs: config.precompute_refs,
        };
        let raw = mining_metal_session_create(device_index, &raw_job, &raw_config, start_nonce);
        if raw.is_null() {
            Err(last_error_message())
        } else {
            Ok(MetalMiningSession { raw })
        }
    }
}

impl MetalMiningSession {
    pub fn mine_next_batch(&mut self) -> Result<MetalMineResult, String> {
        #[cfg(not(all(target_os = "macos", mining_metal_native_enabled)))]
        {
            Err("当前平台未启用 Metal 后端".to_string())
        }
        #[cfg(all(target_os = "macos", mining_metal_native_enabled))]
        unsafe {
            let mut raw_result = mining_metal_mine_result {
                found: false,
                nonce: 0,
                attempts: 0,
                digest_hex: [0; 65],
            };
            if !mining_metal_session_mine_next_batch(self.raw, &mut raw_result) {
                return Err(last_error_message());
            }
            Ok(to_mine_result(raw_result))
        }
    }
}

impl Rpow2MetalMiningSession {
    pub fn mine_next_batch(&mut self) -> Result<Rpow2MetalMineResult, String> {
        #[cfg(not(all(target_os = "macos", mining_metal_native_enabled)))]
        {
            Err("当前平台未启用 Metal 后端".to_string())
        }
        #[cfg(all(target_os = "macos", mining_metal_native_enabled))]
        unsafe {
            let mut raw_result = mining_metal_rpow2_mine_result {
                found: false,
                nonce: 0,
                attempts: 0,
                digest_hex: [0; 65],
            };
            if !mining_metal_rpow2_session_mine_next_batch(self.raw, &mut raw_result) {
                return Err(last_error_message());
            }
            Ok(to_rpow2_mine_result(raw_result))
        }
    }
}

impl Drop for MetalMiningSession {
    fn drop(&mut self) {
        #[cfg(all(target_os = "macos", mining_metal_native_enabled))]
        unsafe {
            if !self.raw.is_null() {
                mining_metal_session_destroy(self.raw);
                self.raw = std::ptr::null_mut();
            }
        }
    }
}

impl Drop for Rpow2MetalMiningSession {
    fn drop(&mut self) {
        #[cfg(all(target_os = "macos", mining_metal_native_enabled))]
        unsafe {
            if !self.raw.is_null() {
                mining_metal_rpow2_session_destroy(self.raw);
                self.raw = std::ptr::null_mut();
            }
        }
    }
}

#[cfg(all(target_os = "macos", mining_metal_native_enabled))]
fn decode_c_string(bytes: &[u8]) -> String {
    let len = bytes
        .iter()
        .position(|byte| *byte == 0)
        .unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..len]).trim().to_string()
}

#[cfg(all(target_os = "macos", mining_metal_native_enabled))]
fn to_mine_result(raw_result: mining_metal_mine_result) -> MetalMineResult {
    let digest_len = raw_result
        .digest_hex
        .iter()
        .position(|byte| *byte == 0)
        .unwrap_or(raw_result.digest_hex.len());
    let digest_hex = String::from_utf8_lossy(&raw_result.digest_hex[..digest_len]).to_string();
    MetalMineResult {
        found: raw_result.found,
        nonce: raw_result.nonce,
        attempts: raw_result.attempts,
        digest_hex,
    }
}

#[cfg(all(target_os = "macos", mining_metal_native_enabled))]
fn to_rpow2_mine_result(raw_result: mining_metal_rpow2_mine_result) -> Rpow2MetalMineResult {
    let digest_len = raw_result
        .digest_hex
        .iter()
        .position(|byte| *byte == 0)
        .unwrap_or(raw_result.digest_hex.len());
    let digest_hex = String::from_utf8_lossy(&raw_result.digest_hex[..digest_len]).to_string();
    Rpow2MetalMineResult {
        found: raw_result.found,
        nonce: raw_result.nonce,
        attempts: raw_result.attempts,
        digest_hex,
    }
}

#[cfg(all(target_os = "macos", mining_metal_native_enabled))]
unsafe fn last_error_message() -> String {
    unsafe { last_error_message_with_fallback("Metal 后端返回了未知错误") }
}

#[cfg(all(target_os = "macos", mining_metal_native_enabled))]
unsafe fn last_error_message_if_any() -> String {
    unsafe { last_error_message_with_fallback("") }
}

#[cfg(all(target_os = "macos", mining_metal_native_enabled))]
unsafe fn last_error_message_with_fallback(fallback: &str) -> String {
    let ptr = unsafe { mining_metal_last_error_message() };
    if ptr.is_null() {
        return fallback.to_string();
    }
    let text = unsafe { CStr::from_ptr(ptr) }
        .to_string_lossy()
        .trim()
        .to_string();
    if text.is_empty() {
        fallback.to_string()
    } else {
        text
    }
}
