use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use reqwest::blocking::Client;
use reqwest::header::{COOKIE, HeaderMap, HeaderValue, USER_AGENT};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{DEFAULT_USER_AGENT, MiningError};

const CPU_ATTEMPT_FLUSH_INTERVAL: i64 = 4096;
const AUTO_GPU_BATCH_SIZE: u64 = 0;
const FALLBACK_GPU_BATCH_SIZE: u64 = 1 << 22;
const GPU_AUTO_TUNE_BATCH_SIZES: [u64; 4] = [1 << 18, 1 << 20, 1 << 22, 1 << 24];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rpow2Job {
    pub nonce_prefix: Vec<u8>,
    pub difficulty_bits: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rpow2Backend {
    Cpu,
    Cuda,
    Metal,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rpow2MineResult {
    pub nonce: u64,
    pub digest_hex: String,
    pub attempts: i64,
    pub backend: Rpow2Backend,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rpow2MineConfig {
    pub start_nonce: u64,
    pub nonce_count: u64,
    pub cpu_threads: usize,
    pub prefer_gpu: bool,
    pub metal_device_index: usize,
    pub metal_batch_size: u64,
}

impl Default for Rpow2MineConfig {
    fn default() -> Self {
        Self {
            start_nonce: 0,
            nonce_count: u64::MAX,
            cpu_threads: std::thread::available_parallelism()
                .map(usize::from)
                .unwrap_or(1),
            prefer_gpu: true,
            metal_device_index: 0,
            metal_batch_size: AUTO_GPU_BATCH_SIZE,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Rpow2PreparedJob {
    nonce_prefix: Vec<u8>,
    difficulty_bits: u32,
    template: Option<Rpow2DigestTemplate>,
}

#[derive(Debug, Clone)]
struct Rpow2DigestTemplate {
    padded: [u8; 128],
    padded_len: usize,
    nonce_offset: usize,
}

impl Rpow2PreparedJob {
    pub fn new(job: &Rpow2Job) -> Self {
        Self {
            nonce_prefix: job.nonce_prefix.clone(),
            difficulty_bits: job.difficulty_bits,
            template: Rpow2DigestTemplate::new(&job.nonce_prefix),
        }
    }

    pub fn digest(&self, nonce: u64) -> [u8; 32] {
        if let Some(template) = &self.template {
            return template.digest(nonce);
        }
        sha256_digest_prefixed_nonce(&self.nonce_prefix, nonce)
    }

    pub fn meets_difficulty(&self, nonce: u64) -> bool {
        trailing_zero_bits(&self.digest(nonce)) >= self.difficulty_bits
    }
}

impl Rpow2DigestTemplate {
    fn new(prefix: &[u8]) -> Option<Self> {
        let message_len = prefix.len().checked_add(8)?;
        let padded_len = ((message_len + 9 + 63) / 64) * 64;
        if padded_len > 128 {
            return None;
        }

        let mut padded = [0u8; 128];
        padded[..prefix.len()].copy_from_slice(prefix);
        padded[message_len] = 0x80;
        padded[padded_len - 8..padded_len]
            .copy_from_slice(&((message_len as u64) * 8).to_be_bytes());
        Some(Self {
            padded,
            padded_len,
            nonce_offset: prefix.len(),
        })
    }

    fn digest(&self, nonce: u64) -> [u8; 32] {
        let mut padded = self.padded;
        padded[self.nonce_offset..self.nonce_offset + 8].copy_from_slice(&nonce.to_le_bytes());
        sha256_digest_padded(&padded[..self.padded_len])
    }
}

impl Rpow2Job {
    pub fn from_nonce_prefix_hex(
        nonce_prefix_hex: &str,
        difficulty_bits: u32,
    ) -> Result<Self, MiningError> {
        Ok(Self {
            nonce_prefix: decode_hex(nonce_prefix_hex)?,
            difficulty_bits,
        })
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Rpow2Challenge {
    pub challenge_id: i64,
    pub nonce_prefix: String,
    pub difficulty_bits: u32,
}

impl Rpow2Challenge {
    pub fn job(&self) -> Result<Rpow2Job, MiningError> {
        Rpow2Job::from_nonce_prefix_hex(&self.nonce_prefix, self.difficulty_bits)
    }
}

#[derive(Debug, Clone)]
pub struct Rpow2Client {
    base_url: String,
    http_client: Client,
}

impl Rpow2Client {
    pub fn new(cookie_header: Option<&str>) -> Result<Self, MiningError> {
        Self::with_base_url("https://api.rpow2.com", cookie_header)
    }

    pub fn with_base_url(base_url: &str, cookie_header: Option<&str>) -> Result<Self, MiningError> {
        let mut headers = HeaderMap::new();
        headers.insert(USER_AGENT, HeaderValue::from_static(DEFAULT_USER_AGENT));
        if let Some(cookie_header) = cookie_header {
            let cookie_header = cookie_header.trim();
            if !cookie_header.is_empty() {
                headers.insert(
                    COOKIE,
                    HeaderValue::from_str(cookie_header).map_err(|error| {
                        MiningError::Message(format!("RPOW2 Cookie 格式无效：{}", error))
                    })?,
                );
            }
        }
        Ok(Self {
            base_url: base_url.trim().trim_end_matches('/').to_string(),
            http_client: Client::builder()
                .default_headers(headers)
                .timeout(Duration::from_secs(30))
                .connect_timeout(Duration::from_secs(10))
                .build()?,
        })
    }

    pub fn me(&self) -> Result<Value, MiningError> {
        self.get_json("/me")
    }

    pub fn ledger(&self) -> Result<Value, MiningError> {
        self.get_json("/ledger")
    }

    pub fn challenge(&self) -> Result<Rpow2Challenge, MiningError> {
        self.post_json_without_body("/challenge")
    }

    pub fn mint(&self, challenge_id: i64, solution_nonce: u64) -> Result<Value, MiningError> {
        #[derive(Serialize)]
        struct MintRequest {
            challenge_id: i64,
            solution_nonce: String,
        }

        self.post_json(
            "/mint",
            &MintRequest {
                challenge_id,
                solution_nonce: solution_nonce.to_string(),
            },
        )
    }

    fn get_json<T: for<'de> Deserialize<'de>>(&self, path: &str) -> Result<T, MiningError> {
        let response = self
            .http_client
            .get(format!("{}{}", self.base_url, path))
            .send()?;
        decode_response(response)
    }

    fn post_json_without_body<T: for<'de> Deserialize<'de>>(
        &self,
        path: &str,
    ) -> Result<T, MiningError> {
        let response = self
            .http_client
            .post(format!("{}{}", self.base_url, path))
            .send()?;
        decode_response(response)
    }

    fn post_json<T: for<'de> Deserialize<'de>, B: Serialize>(
        &self,
        path: &str,
        payload: &B,
    ) -> Result<T, MiningError> {
        let response = self
            .http_client
            .post(format!("{}{}", self.base_url, path))
            .json(payload)
            .send()?;
        decode_response(response)
    }
}

pub fn mine_rpow2(
    job: &Rpow2Job,
    config: Rpow2MineConfig,
    cancel: &Arc<AtomicBool>,
) -> Result<Rpow2MineResult, MiningError> {
    if config.prefer_gpu {
        match mine_rpow2_cuda(job, config, cancel) {
            Ok(Some(result)) => return Ok(result),
            Ok(None) => {}
            Err(error) => {
                if cancel.load(Ordering::SeqCst) {
                    return Err(error);
                }
            }
        }
        match mine_rpow2_metal(job, config, cancel) {
            Ok(Some(result)) => return Ok(result),
            Ok(None) => {}
            Err(error) => {
                if cancel.load(Ordering::SeqCst) {
                    return Err(error);
                }
            }
        }
    }
    mine_rpow2_cpu(job, config, cancel)
}

pub fn mine_rpow2_cpu(
    job: &Rpow2Job,
    config: Rpow2MineConfig,
    cancel: &Arc<AtomicBool>,
) -> Result<Rpow2MineResult, MiningError> {
    if config.nonce_count == 0 {
        return Err(MiningError::Message(
            "RPOW2 nonce 范围不能为空。".to_string(),
        ));
    }
    let worker_count = config.cpu_threads.max(1);
    let prepared = Arc::new(Rpow2PreparedJob::new(job));
    let attempts = Arc::new(AtomicI64::new(0));
    let (sender, receiver) = mpsc::channel();
    let stop = Arc::new(AtomicBool::new(false));
    let mut handles = Vec::with_capacity(worker_count);

    for worker_index in 0..worker_count {
        let prepared = Arc::clone(&prepared);
        let sender = sender.clone();
        let cancel = Arc::clone(cancel);
        let stop = Arc::clone(&stop);
        let attempts = Arc::clone(&attempts);
        handles.push(thread::spawn(move || {
            let mut local_attempts = 0i64;
            let mut nonce = config.start_nonce.saturating_add(worker_index as u64);
            let end_nonce = config.start_nonce.saturating_add(config.nonce_count);
            while nonce < end_nonce
                && !cancel.load(Ordering::Relaxed)
                && !stop.load(Ordering::Relaxed)
            {
                let digest = prepared.digest(nonce);
                local_attempts += 1;
                if local_attempts % CPU_ATTEMPT_FLUSH_INTERVAL == 0 {
                    attempts.fetch_add(CPU_ATTEMPT_FLUSH_INTERVAL, Ordering::Relaxed);
                }
                if trailing_zero_bits(&digest) >= prepared.difficulty_bits {
                    let pending = local_attempts % CPU_ATTEMPT_FLUSH_INTERVAL;
                    let attempt_count = if pending == 0 {
                        attempts.load(Ordering::Relaxed)
                    } else {
                        attempts.fetch_add(pending, Ordering::Relaxed) + pending
                    };
                    stop.store(true, Ordering::SeqCst);
                    let _ = sender.send(Some(Rpow2MineResult {
                        nonce,
                        digest_hex: hex_lower(&digest),
                        attempts: attempt_count,
                        backend: Rpow2Backend::Cpu,
                    }));
                    return;
                }
                nonce = nonce.saturating_add(worker_count as u64);
            }
            attempts.fetch_add(
                local_attempts % CPU_ATTEMPT_FLUSH_INTERVAL,
                Ordering::Relaxed,
            );
            let _ = sender.send(None);
        }));
    }
    drop(sender);

    let mut completed = 0usize;
    let mut found = None;
    while completed < worker_count {
        match receiver.recv() {
            Ok(Some(result)) => {
                found = Some(result);
                break;
            }
            Ok(None) => completed += 1,
            Err(_) => break,
        }
    }
    stop.store(true, Ordering::SeqCst);
    for handle in handles {
        let _ = handle.join();
    }

    if cancel.load(Ordering::SeqCst) {
        return Err(crate::error::interrupted_error());
    }
    if let Some(mut result) = found {
        result.attempts = attempts.load(Ordering::Relaxed);
        return Ok(result);
    }
    Err(MiningError::Message(
        "RPOW2 nonce 范围内没有找到解。".to_string(),
    ))
}

pub fn rpow2_digest(job: &Rpow2Job, nonce: u64) -> [u8; 32] {
    sha256_digest_prefixed_nonce(&job.nonce_prefix, nonce)
}

pub fn rpow2_meets_difficulty(job: &Rpow2Job, nonce: u64) -> bool {
    trailing_zero_bits(&rpow2_digest(job, nonce)) >= job.difficulty_bits
}

pub fn trailing_zero_bits(digest: &[u8]) -> u32 {
    let mut bits = 0u32;
    for byte in digest.iter().rev().copied() {
        if byte == 0 {
            bits += 8;
        } else {
            return bits + byte.trailing_zeros();
        }
    }
    bits
}

pub fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

fn decode_response<T: for<'de> Deserialize<'de>>(
    response: reqwest::blocking::Response,
) -> Result<T, MiningError> {
    let status = response.status();
    let text = response.text()?;
    if !status.is_success() {
        let message = serde_json::from_str::<Value>(&text)
            .ok()
            .and_then(|value| {
                value
                    .get("message")
                    .or_else(|| value.get("error"))
                    .and_then(Value::as_str)
                    .map(str::to_string)
            })
            .filter(|message| !message.trim().is_empty())
            .unwrap_or(text);
        return Err(MiningError::Message(format!(
            "RPOW2 请求失败（状态码 {}）：{}",
            status.as_u16(),
            message
        )));
    }
    Ok(serde_json::from_str(&text)?)
}

fn decode_hex(input: &str) -> Result<Vec<u8>, MiningError> {
    let trimmed = input.trim();
    if !trimmed.len().is_multiple_of(2) {
        return Err(MiningError::Message(
            "RPOW2 nonce_prefix hex 长度必须为偶数。".to_string(),
        ));
    }
    let mut output = Vec::with_capacity(trimmed.len() / 2);
    for index in (0..trimmed.len()).step_by(2) {
        let byte = u8::from_str_radix(&trimmed[index..index + 2], 16).map_err(|_| {
            MiningError::Message(format!(
                "RPOW2 nonce_prefix 包含非法 hex：{}",
                &trimmed[index..index + 2]
            ))
        })?;
        output.push(byte);
    }
    Ok(output)
}

fn mine_rpow2_cuda(
    job: &Rpow2Job,
    config: Rpow2MineConfig,
    cancel: &Arc<AtomicBool>,
) -> Result<Option<Rpow2MineResult>, MiningError> {
    if !mining_cuda_sys::is_available()
        .map_err(MiningError::Message)
        .unwrap_or(false)
    {
        return Ok(None);
    }
    let batch_size = config.metal_batch_size;
    let mut start_nonce = config.start_nonce;
    let mut remaining = config.nonce_count;
    let mut attempts = 0i64;
    let raw_job = mining_cuda_sys::Rpow2CudaJob {
        nonce_prefix: &job.nonce_prefix,
        difficulty_bits: job.difficulty_bits,
    };

    let batch_size = if batch_size == AUTO_GPU_BATCH_SIZE {
        let mut best_batch_size = 0u64;
        let mut best_hashes_per_second = 0.0f64;
        for candidate_batch_size in GPU_AUTO_TUNE_BATCH_SIZES {
            if remaining == 0 {
                break;
            }
            if cancel.load(Ordering::SeqCst) {
                return Err(crate::error::interrupted_error());
            }
            let current_batch_size = candidate_batch_size.min(remaining);
            let started = Instant::now();
            let result = mining_cuda_sys::rpow2_mine_batch(
                config.metal_device_index,
                &raw_job,
                mining_cuda_sys::Rpow2CudaSolverConfig {
                    batch_size: current_batch_size,
                },
                start_nonce,
            )
            .map_err(MiningError::Message)?;
            let elapsed = started.elapsed().as_secs_f64().max(0.001);
            attempts = attempts.saturating_add(result.attempts);
            if let Some(result) = verified_cuda_result(job, result, attempts)? {
                return Ok(Some(result));
            }
            let hashes_per_second = current_batch_size as f64 / elapsed;
            if hashes_per_second > best_hashes_per_second {
                best_hashes_per_second = hashes_per_second;
                best_batch_size = current_batch_size;
            }
            start_nonce = start_nonce.saturating_add(current_batch_size);
            remaining = remaining.saturating_sub(current_batch_size);
        }
        let selected = if best_batch_size == 0 {
            FALLBACK_GPU_BATCH_SIZE
        } else {
            best_batch_size
        };
        selected.min(remaining.max(1))
    } else {
        batch_size.max(1)
    };

    while remaining > 0 {
        if cancel.load(Ordering::SeqCst) {
            return Err(crate::error::interrupted_error());
        }
        let current_batch = batch_size.min(remaining);
        let result = mining_cuda_sys::rpow2_mine_batch(
            config.metal_device_index,
            &raw_job,
            mining_cuda_sys::Rpow2CudaSolverConfig {
                batch_size: current_batch,
            },
            start_nonce,
        )
        .map_err(MiningError::Message)?;
        attempts = attempts.saturating_add(result.attempts);
        if let Some(result) = verified_cuda_result(job, result, attempts)? {
            return Ok(Some(result));
        }
        start_nonce = start_nonce.saturating_add(current_batch);
        remaining = remaining.saturating_sub(current_batch);
    }
    Ok(None)
}

fn verified_cuda_result(
    job: &Rpow2Job,
    result: mining_cuda_sys::Rpow2CudaMineResult,
    attempts: i64,
) -> Result<Option<Rpow2MineResult>, MiningError> {
    if !result.found {
        return Ok(None);
    }
    let expected = hex_lower(&rpow2_digest(job, result.nonce));
    if result.digest_hex != expected {
        return Err(MiningError::Message(
            "CUDA RPOW2 后端返回的摘要校验失败。".to_string(),
        ));
    }
    Ok(Some(Rpow2MineResult {
        nonce: result.nonce,
        digest_hex: result.digest_hex,
        attempts,
        backend: Rpow2Backend::Cuda,
    }))
}

#[cfg(target_os = "macos")]
fn mine_rpow2_metal(
    job: &Rpow2Job,
    config: Rpow2MineConfig,
    cancel: &Arc<AtomicBool>,
) -> Result<Option<Rpow2MineResult>, MiningError> {
    if !mining_metal_sys::is_available()
        .map_err(MiningError::Message)
        .unwrap_or(false)
    {
        return Ok(None);
    }
    let batch_size = config.metal_batch_size;
    let mut start_nonce = config.start_nonce;
    let mut remaining = config.nonce_count;
    let mut attempts = 0i64;
    let raw_job = mining_metal_sys::Rpow2MetalJob {
        nonce_prefix: &job.nonce_prefix,
        difficulty_bits: job.difficulty_bits,
    };

    let batch_size = if batch_size == AUTO_GPU_BATCH_SIZE {
        let mut best_batch_size = 0u64;
        let mut best_hashes_per_second = 0.0f64;
        for candidate_batch_size in GPU_AUTO_TUNE_BATCH_SIZES {
            if remaining == 0 {
                break;
            }
            if cancel.load(Ordering::SeqCst) {
                return Err(crate::error::interrupted_error());
            }
            let current_batch_size = candidate_batch_size.min(remaining);
            let started = Instant::now();
            let result = mining_metal_sys::rpow2_mine_batch(
                config.metal_device_index,
                &raw_job,
                mining_metal_sys::Rpow2MetalSolverConfig {
                    batch_size: current_batch_size,
                },
                start_nonce,
            )
            .map_err(MiningError::Message)?;
            let elapsed = started.elapsed().as_secs_f64().max(0.001);
            attempts = attempts.saturating_add(result.attempts);
            if let Some(result) = verified_metal_result(job, result, attempts, Rpow2Backend::Metal)?
            {
                return Ok(Some(result));
            }
            let hashes_per_second = current_batch_size as f64 / elapsed;
            if hashes_per_second > best_hashes_per_second {
                best_hashes_per_second = hashes_per_second;
                best_batch_size = current_batch_size;
            }
            start_nonce = start_nonce.saturating_add(current_batch_size);
            remaining = remaining.saturating_sub(current_batch_size);
        }
        let selected = if best_batch_size == 0 {
            FALLBACK_GPU_BATCH_SIZE
        } else {
            best_batch_size
        };
        selected.min(remaining.max(1))
    } else {
        batch_size.max(1)
    };

    if remaining >= batch_size {
        let attempts_before_session = attempts;
        let mut session_attempts = 0i64;
        let mut session = mining_metal_sys::rpow2_create_session(
            config.metal_device_index,
            &raw_job,
            mining_metal_sys::Rpow2MetalSolverConfig { batch_size },
            start_nonce,
        )
        .map_err(MiningError::Message)?;
        while remaining >= batch_size {
            if cancel.load(Ordering::SeqCst) {
                return Err(crate::error::interrupted_error());
            }
            let result = session.mine_next_batch().map_err(MiningError::Message)?;
            session_attempts = result.attempts;
            let total_attempts = attempts_before_session.saturating_add(session_attempts);
            if let Some(result) =
                verified_metal_result(job, result, total_attempts, Rpow2Backend::Metal)?
            {
                return Ok(Some(result));
            }
            start_nonce = start_nonce.saturating_add(batch_size);
            remaining = remaining.saturating_sub(batch_size);
        }
        attempts = attempts_before_session.saturating_add(session_attempts);
    }
    while remaining > 0 {
        if cancel.load(Ordering::SeqCst) {
            return Err(crate::error::interrupted_error());
        }
        let current_batch = batch_size.min(remaining);
        let result = mining_metal_sys::rpow2_mine_batch(
            config.metal_device_index,
            &raw_job,
            mining_metal_sys::Rpow2MetalSolverConfig {
                batch_size: current_batch,
            },
            start_nonce,
        )
        .map_err(MiningError::Message)?;
        attempts = attempts.saturating_add(result.attempts);
        if let Some(result) = verified_metal_result(job, result, attempts, Rpow2Backend::Metal)? {
            return Ok(Some(result));
        }
        start_nonce = start_nonce.saturating_add(current_batch);
        remaining = remaining.saturating_sub(current_batch);
    }
    Ok(None)
}

#[cfg(target_os = "macos")]
fn verified_metal_result(
    job: &Rpow2Job,
    result: mining_metal_sys::Rpow2MetalMineResult,
    attempts: i64,
    backend: Rpow2Backend,
) -> Result<Option<Rpow2MineResult>, MiningError> {
    if !result.found {
        return Ok(None);
    }
    let expected = hex_lower(&rpow2_digest(job, result.nonce));
    if result.digest_hex != expected {
        return Err(MiningError::Message(
            "Metal RPOW2 后端返回的摘要校验失败。".to_string(),
        ));
    }
    Ok(Some(Rpow2MineResult {
        nonce: result.nonce,
        digest_hex: result.digest_hex,
        attempts,
        backend,
    }))
}

#[cfg(not(target_os = "macos"))]
fn mine_rpow2_metal(
    job: &Rpow2Job,
    config: Rpow2MineConfig,
    cancel: &Arc<AtomicBool>,
) -> Result<Option<Rpow2MineResult>, MiningError> {
    let _ = (job, config, cancel);
    Ok(None)
}

fn sha256_digest_prefixed_nonce(prefix: &[u8], nonce: u64) -> [u8; 32] {
    let message_len = prefix.len().saturating_add(8);
    let padded_len = ((message_len + 9 + 63) / 64) * 64;
    if padded_len <= 128 {
        let mut padded = [0u8; 128];
        padded[..prefix.len()].copy_from_slice(prefix);
        padded[prefix.len()..message_len].copy_from_slice(&nonce.to_le_bytes());
        padded[message_len] = 0x80;
        padded[padded_len - 8..padded_len]
            .copy_from_slice(&((message_len as u64) * 8).to_be_bytes());
        return sha256_digest_padded(&padded[..padded_len]);
    }

    let mut input = Vec::with_capacity(message_len);
    input.extend_from_slice(prefix);
    input.extend_from_slice(&nonce.to_le_bytes());
    sha256_digest(&input)
}

fn sha256_digest(data: &[u8]) -> [u8; 32] {
    let bit_len = (data.len() as u64).wrapping_mul(8);
    let mut padded = Vec::with_capacity(data.len() + 72);
    padded.extend_from_slice(data);
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());
    sha256_digest_padded(&padded)
}

fn sha256_digest_padded(padded: &[u8]) -> [u8; 32] {
    const H0: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];

    let mut h = H0;
    for chunk in padded.chunks_exact(64) {
        let mut w = [0u32; 64];
        for index in 0..16 {
            let offset = index * 4;
            w[index] = u32::from_be_bytes([
                chunk[offset],
                chunk[offset + 1],
                chunk[offset + 2],
                chunk[offset + 3],
            ]);
        }
        for index in 16..64 {
            let s0 = w[index - 15].rotate_right(7)
                ^ w[index - 15].rotate_right(18)
                ^ (w[index - 15] >> 3);
            let s1 = w[index - 2].rotate_right(17)
                ^ w[index - 2].rotate_right(19)
                ^ (w[index - 2] >> 10);
            w[index] = w[index - 16]
                .wrapping_add(s0)
                .wrapping_add(w[index - 7])
                .wrapping_add(s1);
        }

        let mut a = h[0];
        let mut b = h[1];
        let mut c = h[2];
        let mut d = h[3];
        let mut e = h[4];
        let mut f = h[5];
        let mut g = h[6];
        let mut hh = h[7];
        for index in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[index])
                .wrapping_add(w[index]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }

    let mut output = [0u8; 32];
    for (index, word) in h.into_iter().enumerate() {
        output[index * 4..index * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rpow2_digest_matches_worker_nonce_encoding() {
        let job = Rpow2Job {
            nonce_prefix: Vec::new(),
            difficulty_bits: 0,
        };

        let digest = rpow2_digest(&job, 0);
        let prepared = Rpow2PreparedJob::new(&job);

        assert_eq!(
            hex_lower(&digest),
            "af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc"
        );
        assert_eq!(prepared.digest(0), digest);
        assert_eq!(trailing_zero_bits(&digest), 2);
    }

    #[test]
    fn trailing_zero_bits_counts_from_digest_tail() {
        assert_eq!(trailing_zero_bits(&[0xff, 0b0001_0000]), 4);
        assert_eq!(trailing_zero_bits(&[0xab, 0x00, 0x00]), 16);
    }

    #[test]
    fn prepared_digest_matches_direct_digest_across_two_blocks() {
        let job = Rpow2Job {
            nonce_prefix: vec![0x5a; 57],
            difficulty_bits: 0,
        };
        let prepared = Rpow2PreparedJob::new(&job);

        assert_eq!(
            prepared.digest(0x0102_0304_0506_0708),
            rpow2_digest(&job, 0x0102_0304_0506_0708)
        );
    }

    #[test]
    fn decode_hex_rejects_odd_length_prefixes() {
        assert!(Rpow2Job::from_nonce_prefix_hex("abc", 1).is_err());
    }
}
