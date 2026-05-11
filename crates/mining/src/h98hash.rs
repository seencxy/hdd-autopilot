use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::mpsc;
use std::thread;

use sha2::{Digest, Sha256};

use crate::MiningError;
use crate::rpow2::{
    DEFAULT_CUDA_EARLY_EXIT, DEFAULT_CUDA_NONCES_PER_THREAD, DEFAULT_CUDA_THREADS_PER_BLOCK,
    FULL_CUDA_BATCH_SIZE,
};

const CPU_ATTEMPT_FLUSH_INTERVAL: i64 = 4096;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct H98HashJob {
    pub challenge: [u8; 16],
    pub nonce_prefix: [u8; 8],
    pub difficulty_bits: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum H98HashBackend {
    Cpu,
    Cuda,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct H98HashMineResult {
    pub nonce: [u8; 16],
    pub nonce_tail: u64,
    pub digest_hex: String,
    pub attempts: i64,
    pub backend: H98HashBackend,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct H98HashMineConfig {
    pub start_nonce: u64,
    pub nonce_count: u64,
    pub cpu_threads: usize,
    pub prefer_gpu: bool,
    pub allow_cpu_fallback: bool,
    pub cuda_device_index: usize,
    pub cuda_batch_size: u64,
    pub cuda_threads_per_block: u32,
    pub cuda_nonces_per_thread: u32,
    pub cuda_max_blocks: u32,
    pub cuda_early_exit: bool,
}

impl Default for H98HashMineConfig {
    fn default() -> Self {
        Self {
            start_nonce: 0,
            nonce_count: u64::MAX,
            cpu_threads: std::thread::available_parallelism()
                .map(usize::from)
                .unwrap_or(1),
            prefer_gpu: true,
            allow_cpu_fallback: true,
            cuda_device_index: 0,
            cuda_batch_size: FULL_CUDA_BATCH_SIZE,
            cuda_threads_per_block: DEFAULT_CUDA_THREADS_PER_BLOCK,
            cuda_nonces_per_thread: DEFAULT_CUDA_NONCES_PER_THREAD,
            cuda_max_blocks: 0,
            cuda_early_exit: DEFAULT_CUDA_EARLY_EXIT,
        }
    }
}

impl H98HashJob {
    pub fn from_hex(
        challenge_hex: &str,
        nonce_prefix_hex: &str,
        difficulty_bits: u32,
    ) -> Result<Self, MiningError> {
        Ok(Self {
            challenge: decode_fixed_hex(challenge_hex, "HASH98 challenge")?,
            nonce_prefix: decode_fixed_hex(nonce_prefix_hex, "HASH98 nonce prefix")?,
            difficulty_bits,
        })
    }
}

pub fn mine_h98hash(
    job: &H98HashJob,
    config: H98HashMineConfig,
    cancel: &Arc<AtomicBool>,
) -> Result<H98HashMineResult, MiningError> {
    if config.nonce_count == 0 {
        return Err(MiningError::Message(
            "HASH98 nonce range cannot be empty.".to_string(),
        ));
    }
    if job.difficulty_bits > 256 {
        return Err(MiningError::Message(
            "HASH98 difficulty exceeds SHA-256 digest size.".to_string(),
        ));
    }
    if config.prefer_gpu {
        match mine_h98hash_cuda(job, config, cancel) {
            Ok(Some(result)) => return Ok(result),
            Ok(None) => {
                if !config.allow_cpu_fallback {
                    return Err(MiningError::Message(
                        "CUDA backend is not available for HASH98 mining.".to_string(),
                    ));
                }
            }
            Err(error) => {
                if cancel.load(Ordering::SeqCst) {
                    return Err(error);
                }
                if !config.allow_cpu_fallback {
                    return Err(error);
                }
            }
        }
    }
    mine_h98hash_cpu(job, config, cancel)
}

pub fn mine_h98hash_cpu(
    job: &H98HashJob,
    config: H98HashMineConfig,
    cancel: &Arc<AtomicBool>,
) -> Result<H98HashMineResult, MiningError> {
    let worker_count = config.cpu_threads.max(1);
    let attempts = Arc::new(AtomicI64::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let (sender, receiver) = mpsc::channel();
    let mut handles = Vec::with_capacity(worker_count);

    for worker_index in 0..worker_count {
        let job = job.clone();
        let attempts = Arc::clone(&attempts);
        let cancel = Arc::clone(cancel);
        let stop = Arc::clone(&stop);
        let sender = sender.clone();
        handles.push(thread::spawn(move || {
            let mut local_attempts = 0i64;
            let mut offset = worker_index as u64;
            while offset < config.nonce_count
                && !cancel.load(Ordering::Relaxed)
                && !stop.load(Ordering::Relaxed)
            {
                let nonce_tail = config.start_nonce.wrapping_add(offset);
                let digest = h98hash_digest(&job.challenge, &job.nonce_prefix, nonce_tail);
                local_attempts += 1;
                if local_attempts % CPU_ATTEMPT_FLUSH_INTERVAL == 0 {
                    attempts.fetch_add(CPU_ATTEMPT_FLUSH_INTERVAL, Ordering::Relaxed);
                }
                if h98hash_meets_difficulty_digest(&digest, job.difficulty_bits) {
                    let pending = local_attempts % CPU_ATTEMPT_FLUSH_INTERVAL;
                    let attempt_count = if pending == 0 {
                        attempts.load(Ordering::Relaxed)
                    } else {
                        attempts.fetch_add(pending, Ordering::Relaxed) + pending
                    };
                    stop.store(true, Ordering::SeqCst);
                    let _ = sender.send(Some(H98HashMineResult {
                        nonce: h98hash_nonce(&job.nonce_prefix, nonce_tail),
                        nonce_tail,
                        digest_hex: hex_lower(&digest),
                        attempts: attempt_count,
                        backend: H98HashBackend::Cpu,
                    }));
                    return;
                }
                offset = offset.saturating_add(worker_count as u64);
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
        "No HASH98 proof found in the nonce range.".to_string(),
    ))
}

pub fn h98hash_digest(challenge: &[u8; 16], nonce_prefix: &[u8; 8], nonce_tail: u64) -> [u8; 32] {
    let mut input = [0u8; 32];
    input[..16].copy_from_slice(challenge);
    input[16..24].copy_from_slice(nonce_prefix);
    input[24..32].copy_from_slice(&nonce_tail.to_be_bytes());
    Sha256::digest(input).into()
}

pub fn h98hash_nonce(nonce_prefix: &[u8; 8], nonce_tail: u64) -> [u8; 16] {
    let mut nonce = [0u8; 16];
    nonce[..8].copy_from_slice(nonce_prefix);
    nonce[8..].copy_from_slice(&nonce_tail.to_be_bytes());
    nonce
}

pub fn h98hash_meets_difficulty(job: &H98HashJob, nonce_tail: u64) -> bool {
    h98hash_meets_difficulty_digest(
        &h98hash_digest(&job.challenge, &job.nonce_prefix, nonce_tail),
        job.difficulty_bits,
    )
}

pub fn h98hash_meets_difficulty_digest(digest: &[u8; 32], difficulty_bits: u32) -> bool {
    leading_zero_bits(digest) >= difficulty_bits
}

pub fn leading_zero_bits(digest: &[u8]) -> u32 {
    let mut bits = 0u32;
    for byte in digest.iter().copied() {
        if byte == 0 {
            bits += 8;
        } else {
            return bits + byte.leading_zeros();
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

fn mine_h98hash_cuda(
    job: &H98HashJob,
    config: H98HashMineConfig,
    cancel: &Arc<AtomicBool>,
) -> Result<Option<H98HashMineResult>, MiningError> {
    if !mining_cuda_sys::is_available()
        .map_err(MiningError::Message)
        .unwrap_or(false)
    {
        return Ok(None);
    }

    let batch_size = config.cuda_batch_size.max(1);
    let mut start_nonce = config.start_nonce;
    let mut remaining = config.nonce_count;
    let mut attempts = 0i64;
    let raw_job = mining_cuda_sys::H98HashCudaJob {
        challenge: &job.challenge,
        nonce_prefix: &job.nonce_prefix,
        difficulty_bits: job.difficulty_bits,
    };

    if remaining >= batch_size {
        let attempts_before_session = attempts;
        let mut session_attempts = 0i64;
        let mut session = mining_cuda_sys::h98hash_create_session(
            config.cuda_device_index,
            &raw_job,
            mining_cuda_sys::Rpow2CudaSolverConfig {
                batch_size,
                threads_per_block: config.cuda_threads_per_block,
                nonces_per_thread: config.cuda_nonces_per_thread,
                max_blocks: config.cuda_max_blocks,
                early_exit: config.cuda_early_exit,
            },
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
            if let Some(result) = verified_cuda_result(job, result, total_attempts)? {
                return Ok(Some(result));
            }
            start_nonce = start_nonce.wrapping_add(batch_size);
            remaining = remaining.saturating_sub(batch_size);
        }
        attempts = attempts_before_session.saturating_add(session_attempts);
    }

    while remaining > 0 {
        if cancel.load(Ordering::SeqCst) {
            return Err(crate::error::interrupted_error());
        }
        let current_batch = batch_size.min(remaining);
        let result = mining_cuda_sys::h98hash_mine_batch(
            config.cuda_device_index,
            &raw_job,
            mining_cuda_sys::Rpow2CudaSolverConfig {
                batch_size: current_batch,
                threads_per_block: config.cuda_threads_per_block,
                nonces_per_thread: config.cuda_nonces_per_thread,
                max_blocks: config.cuda_max_blocks,
                early_exit: config.cuda_early_exit,
            },
            start_nonce,
        )
        .map_err(MiningError::Message)?;
        attempts = attempts.saturating_add(result.attempts);
        if let Some(result) = verified_cuda_result(job, result, attempts)? {
            return Ok(Some(result));
        }
        start_nonce = start_nonce.wrapping_add(current_batch);
        remaining = remaining.saturating_sub(current_batch);
    }
    Ok(None)
}

fn verified_cuda_result(
    job: &H98HashJob,
    result: mining_cuda_sys::H98HashCudaMineResult,
    attempts: i64,
) -> Result<Option<H98HashMineResult>, MiningError> {
    if !result.found {
        return Ok(None);
    }
    let digest = h98hash_digest(&job.challenge, &job.nonce_prefix, result.nonce_tail);
    let expected = hex_lower(&digest);
    if result.digest_hex != expected {
        return Err(MiningError::Message(
            "CUDA HASH98 backend returned a digest that failed host verification.".to_string(),
        ));
    }
    if !h98hash_meets_difficulty_digest(&digest, job.difficulty_bits) {
        return Err(MiningError::Message(
            "CUDA HASH98 backend returned a nonce below the requested difficulty.".to_string(),
        ));
    }
    Ok(Some(H98HashMineResult {
        nonce: h98hash_nonce(&job.nonce_prefix, result.nonce_tail),
        nonce_tail: result.nonce_tail,
        digest_hex: result.digest_hex,
        attempts,
        backend: H98HashBackend::Cuda,
    }))
}

pub fn decode_fixed_hex<const N: usize>(
    input: &str,
    field_name: &str,
) -> Result<[u8; N], MiningError> {
    let trimmed = input.trim().strip_prefix("0x").unwrap_or(input.trim());
    if trimmed.len() != N * 2 {
        return Err(MiningError::Message(format!(
            "{field_name} must be {N} bytes / {} hex chars.",
            N * 2
        )));
    }
    let mut output = [0u8; N];
    for index in 0..N {
        output[index] = u8::from_str_radix(&trimmed[index * 2..index * 2 + 2], 16)
            .map_err(|_| MiningError::Message(format!("{field_name} contains invalid hex.")))?;
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nonce_uses_big_endian_tail() {
        let prefix = [0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];
        assert_eq!(
            hex_lower(&h98hash_nonce(&prefix, 0x0102_0304_0506_0708)),
            "11223344556677880102030405060708"
        );
    }

    #[test]
    fn difficulty_counts_leading_bits() {
        assert_eq!(leading_zero_bits(&[0x00, 0x00, 0x10]), 19);
        assert!(h98hash_meets_difficulty_digest(&[0u8; 32], 256));
        assert!(!h98hash_meets_difficulty_digest(&[0x80; 32], 1));
    }

    #[test]
    fn known_digest_is_stable() {
        let challenge =
            decode_fixed_hex::<16>("000102030405060708090a0b0c0d0e0f", "challenge").unwrap();
        let nonce_prefix = decode_fixed_hex::<8>("1011121314151617", "nonce prefix").unwrap();
        let digest = h98hash_digest(&challenge, &nonce_prefix, 0x1819_1a1b_1c1d_1e1f);
        assert_eq!(
            hex_lower(&digest),
            "630dcd2966c4336691125448bbb25b4ff412a49c732db2c8abc1b8581bd710dd"
        );
    }
}
