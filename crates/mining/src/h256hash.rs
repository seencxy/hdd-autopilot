//! HASH256 / hash256.org mining (Keccak-256 PoW on Ethereum mainnet).
//!
//! Reverse engineering summary (see plan file):
//! - Algorithm: `digest = keccak256( challenge[32] ‖ uint256_be(nonce) )` where
//!   the high 24 bytes of the uint256 nonce are always zero, leaving a u64
//!   search space.
//! - Difficulty: 256-bit big-endian unsigned compare, `digest < target`.
//! - On-chain: `mine(uint256 nonce)` to `0xAC7b…A0cc` on chain id 1.
//!
//! This module mirrors `h98hash` but with single-`u64` nonce, 32-byte
//! challenge, 32-byte target, and uint256 less-than difficulty.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::mpsc;
use std::thread;

use ethers_core::utils::keccak256;

use crate::MiningError;

pub use crate::h98hash::{decode_fixed_hex, hex_lower};

/// CPU attempt counter flush interval. Mirrors the h98hash CPU loop.
const CPU_ATTEMPT_FLUSH_INTERVAL: i64 = 4096;

/// Full-load batch size optimized for RTX 3090 (Ampere sm_86, 82 SM):
/// 2³³ ≈ 8.5G nonces per batch. Keccak path is cheaper per nonce than h98hash
/// SHA-256-double, so we scan a much larger window per launch.
pub const H256HASH_FULL_CUDA_BATCH_SIZE: u64 = 1 << 33;

/// Smaller fallback batch when caller does not opt into the full-load path.
pub const H256HASH_DEFAULT_CUDA_BATCH_SIZE: u64 = 1 << 30;

/// 128 threads/block × 16 blocks/SM = 100% Ampere occupancy.
pub const H256HASH_DEFAULT_CUDA_THREADS_PER_BLOCK: u32 = 128;

/// Process 8 nonces per thread. Template specializes on {1, 2, 4, 8}.
pub const H256HASH_DEFAULT_CUDA_NONCES_PER_THREAD: u32 = 8;

/// 0 = auto-detect (host queries SM count and uses SM × 12 blocks).
pub const H256HASH_DEFAULT_CUDA_MAX_BLOCKS: u32 = 0;

/// Exit the kernel as soon as one thread reports a hit.
pub const H256HASH_DEFAULT_CUDA_EARLY_EXIT: bool = true;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct H256HashJob {
    pub challenge: [u8; 32],
    /// Big-endian uint256 difficulty target. A nonce wins iff `digest < target`.
    pub target: [u8; 32],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum H256HashBackend {
    Cpu,
    Cuda,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct H256HashMineResult {
    /// Low 64 bits of the uint256 nonce; high 24 bytes are zero.
    pub nonce: u64,
    /// uint256 big-endian representation of `nonce`, ready for `mine(uint256)`.
    pub nonce_be32: [u8; 32],
    pub digest: [u8; 32],
    pub digest_hex: String,
    pub attempts: i64,
    pub backend: H256HashBackend,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct H256HashMineConfig {
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

impl Default for H256HashMineConfig {
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
            cuda_batch_size: H256HASH_FULL_CUDA_BATCH_SIZE,
            cuda_threads_per_block: H256HASH_DEFAULT_CUDA_THREADS_PER_BLOCK,
            cuda_nonces_per_thread: H256HASH_DEFAULT_CUDA_NONCES_PER_THREAD,
            cuda_max_blocks: H256HASH_DEFAULT_CUDA_MAX_BLOCKS,
            cuda_early_exit: H256HASH_DEFAULT_CUDA_EARLY_EXIT,
        }
    }
}

impl H256HashJob {
    pub fn from_hex(challenge_hex: &str, target_hex: &str) -> Result<Self, MiningError> {
        Ok(Self {
            challenge: decode_fixed_hex(challenge_hex, "HASH256 challenge")?,
            target: decode_fixed_hex(target_hex, "HASH256 target")?,
        })
    }
}

/// Build the 64-byte keccak input that the on-chain verifier expects.
///
/// Layout (from the hash256.org WGSL kernel comment):
/// ```text
/// bytes  0..31  = challenge (32 bytes)
/// bytes 32..55  = 0x00 × 24
/// bytes 56..63  = nonce as big-endian u64 (low 64 bits of uint256 nonce)
/// ```
/// This is exactly `keccak256(challenge ‖ abi.encode(uint256 nonce))`.
#[inline]
pub fn h256hash_input(challenge: &[u8; 32], nonce: u64) -> [u8; 64] {
    let mut input = [0u8; 64];
    input[..32].copy_from_slice(challenge);
    input[56..64].copy_from_slice(&nonce.to_be_bytes());
    input
}

#[inline]
pub fn h256hash_digest(challenge: &[u8; 32], nonce: u64) -> [u8; 32] {
    keccak256(h256hash_input(challenge, nonce))
}

/// Big-endian uint256 representation of a u64 nonce — what `mine(uint256)`
/// receives on chain, and the key used to recompute `usedProofs(bytes32)`.
#[inline]
pub fn h256hash_nonce_be32(nonce: u64) -> [u8; 32] {
    let mut out = [0u8; 32];
    out[24..32].copy_from_slice(&nonce.to_be_bytes());
    out
}

/// Strict big-endian uint256 less-than: `digest < target`.
#[inline]
pub fn h256hash_meets_difficulty(digest: &[u8; 32], target: &[u8; 32]) -> bool {
    for index in 0..32 {
        match digest[index].cmp(&target[index]) {
            std::cmp::Ordering::Less => return true,
            std::cmp::Ordering::Greater => return false,
            std::cmp::Ordering::Equal => continue,
        }
    }
    // Strict less-than: equal does not win.
    false
}

pub fn mine_h256hash(
    job: &H256HashJob,
    config: H256HashMineConfig,
    cancel: &Arc<AtomicBool>,
) -> Result<H256HashMineResult, MiningError> {
    if config.nonce_count == 0 {
        return Err(MiningError::Message(
            "HASH256 nonce range cannot be empty.".to_string(),
        ));
    }
    if job.target.iter().all(|byte| *byte == 0) {
        return Err(MiningError::Message(
            "HASH256 target is zero; no nonce can satisfy `digest < 0`.".to_string(),
        ));
    }
    if config.prefer_gpu {
        match mine_h256hash_cuda(job, config, cancel) {
            Ok(Some(result)) => return Ok(result),
            Ok(None) => {
                if !config.allow_cpu_fallback {
                    return Err(MiningError::Message(
                        "CUDA backend is not available for HASH256 mining.".to_string(),
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
    mine_h256hash_cpu(job, config, cancel)
}

pub fn mine_h256hash_cpu(
    job: &H256HashJob,
    config: H256HashMineConfig,
    cancel: &Arc<AtomicBool>,
) -> Result<H256HashMineResult, MiningError> {
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
                let nonce = config.start_nonce.wrapping_add(offset);
                let digest = h256hash_digest(&job.challenge, nonce);
                local_attempts += 1;
                if local_attempts % CPU_ATTEMPT_FLUSH_INTERVAL == 0 {
                    attempts.fetch_add(CPU_ATTEMPT_FLUSH_INTERVAL, Ordering::Relaxed);
                }
                if h256hash_meets_difficulty(&digest, &job.target) {
                    let pending = local_attempts % CPU_ATTEMPT_FLUSH_INTERVAL;
                    let attempt_count = if pending == 0 {
                        attempts.load(Ordering::Relaxed)
                    } else {
                        attempts.fetch_add(pending, Ordering::Relaxed) + pending
                    };
                    stop.store(true, Ordering::SeqCst);
                    let _ = sender.send(Some(H256HashMineResult {
                        nonce,
                        nonce_be32: h256hash_nonce_be32(nonce),
                        digest,
                        digest_hex: hex_lower(&digest),
                        attempts: attempt_count,
                        backend: H256HashBackend::Cpu,
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
        "No HASH256 proof found in the nonce range.".to_string(),
    ))
}

fn mine_h256hash_cuda(
    job: &H256HashJob,
    config: H256HashMineConfig,
    cancel: &Arc<AtomicBool>,
) -> Result<Option<H256HashMineResult>, MiningError> {
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
    let raw_job = mining_cuda_sys::H256HashCudaJob {
        challenge: &job.challenge,
        target: &job.target,
    };
    let solver_config = mining_cuda_sys::Rpow2CudaSolverConfig {
        batch_size,
        threads_per_block: config.cuda_threads_per_block,
        nonces_per_thread: config.cuda_nonces_per_thread,
        max_blocks: config.cuda_max_blocks,
        early_exit: config.cuda_early_exit,
    };

    if remaining >= batch_size {
        let attempts_before_session = attempts;
        let mut session_attempts = 0i64;
        let mut session = mining_cuda_sys::h256hash_create_session(
            config.cuda_device_index,
            &raw_job,
            solver_config,
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
        let result = mining_cuda_sys::h256hash_mine_batch(
            config.cuda_device_index,
            &raw_job,
            mining_cuda_sys::Rpow2CudaSolverConfig {
                batch_size: current_batch,
                threads_per_block: solver_config.threads_per_block,
                nonces_per_thread: solver_config.nonces_per_thread,
                max_blocks: solver_config.max_blocks,
                early_exit: solver_config.early_exit,
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
    job: &H256HashJob,
    result: mining_cuda_sys::H256HashCudaMineResult,
    attempts: i64,
) -> Result<Option<H256HashMineResult>, MiningError> {
    if !result.found {
        return Ok(None);
    }
    let digest = h256hash_digest(&job.challenge, result.nonce);
    let expected = hex_lower(&digest);
    if result.digest_hex != expected {
        return Err(MiningError::Message(format!(
            "CUDA HASH256 backend returned digest {} that failed host verification (expected {}).",
            result.digest_hex, expected
        )));
    }
    if !h256hash_meets_difficulty(&digest, &job.target) {
        return Err(MiningError::Message(
            "CUDA HASH256 backend returned a nonce that does not meet the target.".to_string(),
        ));
    }
    Ok(Some(H256HashMineResult {
        nonce: result.nonce,
        nonce_be32: h256hash_nonce_be32(result.nonce),
        digest,
        digest_hex: result.digest_hex,
        attempts,
        backend: H256HashBackend::Cuda,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nonce_be32_pads_high_24_bytes_with_zero() {
        let nonce_be = h256hash_nonce_be32(0x0102_0304_0506_0708);
        assert_eq!(
            hex_lower(&nonce_be),
            "0000000000000000000000000000000000000000000000000102030405060708"
        );
    }

    #[test]
    fn keccak_input_matches_uint256_encoding() {
        let challenge = [0xaa_u8; 32];
        let input = h256hash_input(&challenge, 0x42);
        assert_eq!(&input[..32], &challenge);
        assert_eq!(&input[32..56], &[0u8; 24]);
        assert_eq!(&input[56..64], &[0, 0, 0, 0, 0, 0, 0, 0x42]);
    }

    #[test]
    fn digest_matches_reference_keccak_of_zero_buffer() {
        let challenge = [0u8; 32];
        let digest = h256hash_digest(&challenge, 0);
        let expected = ethers_core::utils::keccak256([0u8; 64]);
        assert_eq!(digest, expected);
    }

    #[test]
    fn meets_difficulty_uses_big_endian_less_than() {
        let target_max = [0xff_u8; 32];
        assert!(h256hash_meets_difficulty(&[0u8; 32], &target_max));

        let target = [0x10_u8; 32];
        // Equal is not "less than".
        assert!(!h256hash_meets_difficulty(&[0x10_u8; 32], &target));

        // Last byte 0x0f vs 0x10 → less.
        let mut digest_eq = [0x10_u8; 32];
        digest_eq[31] = 0x0f;
        assert!(h256hash_meets_difficulty(&digest_eq, &target));

        // First differing byte wins regardless of tail.
        let mut digest_high = [0x11_u8; 32];
        digest_high[31] = 0x00;
        assert!(!h256hash_meets_difficulty(&digest_high, &target));
    }

    #[test]
    fn cpu_miner_finds_easy_nonce_immediately() {
        // Target = 2^256 - 1, so digest < target for any digest != 2^256 - 1.
        let challenge = [0u8; 32];
        let target = [0xff_u8; 32];
        let job = H256HashJob { challenge, target };
        let config = H256HashMineConfig {
            prefer_gpu: false,
            cpu_threads: 1,
            nonce_count: 1,
            ..H256HashMineConfig::default()
        };
        let cancel = Arc::new(AtomicBool::new(false));
        let result = mine_h256hash_cpu(&job, config, &cancel).unwrap();
        assert_eq!(result.nonce, 0);
        assert_eq!(result.backend, H256HashBackend::Cpu);
        assert_eq!(result.digest, h256hash_digest(&challenge, 0));
        assert_eq!(&result.nonce_be32[24..32], &[0u8; 8]);
    }

    #[test]
    fn cpu_miner_respects_easy_target_bit_pattern() {
        // Target with MSB byte = 0x01: requires digest[0] == 0x00 (≈1/256).
        let challenge = [0u8; 32];
        let mut target = [0u8; 32];
        target[0] = 0x01;
        let job = H256HashJob { challenge, target };
        let config = H256HashMineConfig {
            prefer_gpu: false,
            cpu_threads: 2,
            nonce_count: 100_000,
            ..H256HashMineConfig::default()
        };
        let cancel = Arc::new(AtomicBool::new(false));
        let result = mine_h256hash_cpu(&job, config, &cancel).unwrap();
        assert_eq!(result.digest[0], 0x00);
        assert_eq!(h256hash_digest(&challenge, result.nonce), result.digest);
        assert!(h256hash_meets_difficulty(&result.digest, &job.target));
    }

    #[test]
    fn cpu_miner_returns_error_when_target_is_zero() {
        let job = H256HashJob {
            challenge: [0u8; 32],
            target: [0u8; 32],
        };
        let cancel = Arc::new(AtomicBool::new(false));
        let result = mine_h256hash(&job, H256HashMineConfig::default(), &cancel);
        assert!(result.is_err());
    }
}
