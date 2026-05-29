//! DigitalWaterCoin (DWC) browser-PoW share miner.
//!
//! Reverse-engineered from `https://digitalwatercoin.com/mine` (`/mine-worker.js`):
//! a valid share is a nonce such that
//! `sha256_hex("{address}|{epoch}|{nonce}")` begins with `difficulty` hex zeros.
//!
//! * `address` — the miner's wallet (0x + 40 hex), also the miner ID.
//! * `epoch`   — `floor(unix_seconds / 300)` as a decimal string; rotates every 5 min.
//! * `nonce`   — any string; the server recomputes the hash. We render it as a
//!   session `salt` (random hex) followed by a zero-padded 16-char hex counter,
//!   so the GPU/CPU only has to vary a 64-bit integer while every share stays
//!   collision-free across restarts.
//!
//! The PoW is plain SHA-256 over an ASCII preimage, so this mirrors the
//! [`crate::rpow2`] miner: a constant prefix plus a per-attempt counter. The two
//! differences are that the counter is written as 16 ASCII hex bytes (it lives
//! inside a UTF-8/JSON string) and that difficulty counts *leading* zero bits.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::mpsc;
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::MiningError;

/// Length of the unix epoch window, in seconds (epoch = floor(unix / 300)).
pub const EPOCH_SECONDS: u64 = 300;
/// Number of ASCII hex characters used to render the 64-bit nonce counter.
pub const HEX_NONCE_LEN: usize = 16;
/// Default public API base.
pub const DEFAULT_API_BASE: &str = "https://digitalwatercoin.com/api";

const CPU_ATTEMPT_FLUSH_INTERVAL: i64 = 4096;
const HEX_DIGITS: &[u8; 16] = b"0123456789abcdef";

/// Compute the current DWC epoch (`floor(unix_seconds / 300)`).
pub fn current_epoch() -> u64 {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    secs / EPOCH_SECONDS
}

/// A fully specified DWC mining job for one (address, epoch, salt) tuple.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DwcJob {
    pub address: String,
    pub epoch: String,
    /// Random per-session hex string prepended to the counter inside the nonce.
    pub salt: String,
    /// Required number of leading zero hex digits (the site default is 5).
    pub difficulty: u32,
}

impl DwcJob {
    pub fn new(
        address: impl Into<String>,
        epoch: impl Into<String>,
        salt: impl Into<String>,
        difficulty: u32,
    ) -> Self {
        Self {
            address: address.into(),
            epoch: epoch.into(),
            salt: salt.into(),
            difficulty,
        }
    }

    /// Required leading zero *bits* (`difficulty` nibbles × 4).
    pub fn difficulty_bits(&self) -> u32 {
        self.difficulty * 4
    }

    /// Constant ASCII bytes that precede the 16-hex counter: `address|epoch|salt`.
    pub fn prefix_bytes(&self) -> Vec<u8> {
        format!("{}|{}|{}", self.address, self.epoch, self.salt).into_bytes()
    }

    /// The nonce string submitted to the server for counter value `counter`.
    pub fn nonce_string(&self, counter: u64) -> String {
        format!("{}{:016x}", self.salt, counter)
    }

    /// The full preimage hashed for counter value `counter`.
    pub fn preimage(&self, counter: u64) -> String {
        format!("{}|{}|{}", self.address, self.epoch, self.nonce_string(counter))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DwcBackend {
    Cpu,
    Metal,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DwcShare {
    /// The 64-bit counter that produced the share.
    pub counter: u64,
    /// The submitted nonce string (`salt` + 16-hex counter).
    pub nonce: String,
    pub digest_hex: String,
    pub attempts: i64,
    pub backend: DwcBackend,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DwcMineConfig {
    pub start_counter: u64,
    pub counter_count: u64,
    pub cpu_threads: usize,
    pub prefer_gpu: bool,
    pub allow_cpu_fallback: bool,
    pub metal_device_index: usize,
    pub metal_batch_size: u64,
}

impl Default for DwcMineConfig {
    fn default() -> Self {
        Self {
            start_counter: 0,
            counter_count: u64::MAX,
            cpu_threads: std::thread::available_parallelism()
                .map(usize::from)
                .unwrap_or(1),
            prefer_gpu: true,
            allow_cpu_fallback: true,
            metal_device_index: 0,
            metal_batch_size: 1 << 22,
        }
    }
}

/// Precomputed SHA-256 padding template so each attempt only rewrites the
/// 16-byte hex counter region. Mirrors [`crate::rpow2`]'s digest template.
#[derive(Debug, Clone)]
pub struct DwcPreparedJob {
    difficulty_bits: u32,
    salt: String,
    buffer: Vec<u8>,
    counter_offset: usize,
}

impl DwcPreparedJob {
    pub fn new(job: &DwcJob) -> Self {
        let mut buffer = job.prefix_bytes();
        let counter_offset = buffer.len();
        buffer.resize(counter_offset + HEX_NONCE_LEN, 0);
        Self {
            difficulty_bits: job.difficulty_bits(),
            salt: job.salt.clone(),
            buffer,
            counter_offset,
        }
    }

    pub fn difficulty_bits(&self) -> u32 {
        self.difficulty_bits
    }

    #[inline]
    fn write_counter(buffer: &mut [u8], offset: usize, counter: u64) {
        // 16 ASCII hex chars, most-significant nibble first (matches `{:016x}`).
        for i in 0..HEX_NONCE_LEN {
            let shift = (HEX_NONCE_LEN - 1 - i) * 4;
            let nibble = ((counter >> shift) & 0xf) as usize;
            buffer[offset + i] = HEX_DIGITS[nibble];
        }
    }

    #[inline]
    pub fn digest(&self, counter: u64) -> [u8; 32] {
        let mut buffer = self.buffer.clone();
        Self::write_counter(&mut buffer, self.counter_offset, counter);
        Sha256::digest(&buffer).into()
    }

    #[inline]
    pub fn meets_difficulty(&self, counter: u64) -> bool {
        leading_zero_bits(&self.digest(counter)) >= self.difficulty_bits
    }

    pub fn nonce_string(&self, counter: u64) -> String {
        format!("{}{:016x}", self.salt, counter)
    }
}

/// SHA-256 of the exact server preimage `address|epoch|nonce` for `counter`.
/// Independent of the prepared-job fast path — use it to re-verify shares.
pub fn share_digest(job: &DwcJob, counter: u64) -> [u8; 32] {
    Sha256::digest(job.preimage(counter).as_bytes()).into()
}

/// True if `counter` yields a valid share for `job`.
pub fn meets_difficulty(job: &DwcJob, counter: u64) -> bool {
    leading_zero_bits(&share_digest(job, counter)) >= job.difficulty_bits()
}

/// Count leading zero *bits* of a big-endian digest (digest[0] is the MSB).
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
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX_DIGITS[(byte >> 4) as usize] as char);
        output.push(HEX_DIGITS[(byte & 0x0f) as usize] as char);
    }
    output
}

/// Mine a single DWC share, preferring the GPU when available.
pub fn mine_dwc(
    job: &DwcJob,
    config: DwcMineConfig,
    cancel: &Arc<AtomicBool>,
) -> Result<DwcShare, MiningError> {
    if config.counter_count == 0 {
        return Err(MiningError::Message(
            "DWC counter range cannot be empty.".to_string(),
        ));
    }
    if config.prefer_gpu {
        match mine_dwc_metal(job, config, cancel) {
            Ok(Some(result)) => return Ok(result),
            Ok(None) => {
                if !config.allow_cpu_fallback {
                    return Err(MiningError::Message(
                        "Metal backend is not available for DWC mining.".to_string(),
                    ));
                }
            }
            Err(error) => {
                if cancel.load(Ordering::SeqCst) || !config.allow_cpu_fallback {
                    return Err(error);
                }
            }
        }
    }
    mine_dwc_cpu(job, config, cancel)
}

pub fn mine_dwc_cpu(
    job: &DwcJob,
    config: DwcMineConfig,
    cancel: &Arc<AtomicBool>,
) -> Result<DwcShare, MiningError> {
    let worker_count = config.cpu_threads.max(1);
    let prepared = Arc::new(DwcPreparedJob::new(job));
    let attempts = Arc::new(AtomicI64::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let (sender, receiver) = mpsc::channel();
    let mut handles = Vec::with_capacity(worker_count);

    for worker_index in 0..worker_count {
        let prepared = Arc::clone(&prepared);
        let attempts = Arc::clone(&attempts);
        let cancel = Arc::clone(cancel);
        let stop = Arc::clone(&stop);
        let sender = sender.clone();
        handles.push(thread::spawn(move || {
            let mut local_attempts = 0i64;
            let mut counter = config.start_counter.saturating_add(worker_index as u64);
            let end = config.start_counter.saturating_add(config.counter_count);
            while counter < end
                && !cancel.load(Ordering::Relaxed)
                && !stop.load(Ordering::Relaxed)
            {
                let digest = prepared.digest(counter);
                local_attempts += 1;
                if local_attempts % CPU_ATTEMPT_FLUSH_INTERVAL == 0 {
                    attempts.fetch_add(CPU_ATTEMPT_FLUSH_INTERVAL, Ordering::Relaxed);
                }
                if leading_zero_bits(&digest) >= prepared.difficulty_bits {
                    let pending = local_attempts % CPU_ATTEMPT_FLUSH_INTERVAL;
                    let attempt_count = if pending == 0 {
                        attempts.load(Ordering::Relaxed)
                    } else {
                        attempts.fetch_add(pending, Ordering::Relaxed) + pending
                    };
                    stop.store(true, Ordering::SeqCst);
                    let _ = sender.send(Some(DwcShare {
                        counter,
                        nonce: prepared.nonce_string(counter),
                        digest_hex: hex_lower(&digest),
                        attempts: attempt_count,
                        backend: DwcBackend::Cpu,
                    }));
                    return;
                }
                counter = counter.saturating_add(worker_count as u64);
            }
            attempts.fetch_add(local_attempts % CPU_ATTEMPT_FLUSH_INTERVAL, Ordering::Relaxed);
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
        "No DWC share found in the counter range.".to_string(),
    ))
}

fn mine_dwc_metal(
    job: &DwcJob,
    config: DwcMineConfig,
    cancel: &Arc<AtomicBool>,
) -> Result<Option<DwcShare>, MiningError> {
    if !mining_metal_sys::dwc_is_available()
        .map_err(MiningError::Message)
        .unwrap_or(false)
    {
        return Ok(None);
    }

    let prepared = DwcPreparedJob::new(job);
    let prefix = job.prefix_bytes();
    let batch_size = config.metal_batch_size.max(1);
    let mut start = config.start_counter;
    let mut remaining = config.counter_count;

    let raw_job = mining_metal_sys::DwcMetalJob {
        prefix: &prefix,
        difficulty_bits: prepared.difficulty_bits(),
    };
    let mut session = mining_metal_sys::dwc_create_session(
        config.metal_device_index,
        &raw_job,
        mining_metal_sys::DwcMetalSolverConfig { batch_size },
        start,
    )
    .map_err(MiningError::Message)?;

    while remaining > 0 {
        if cancel.load(Ordering::SeqCst) {
            return Err(crate::error::interrupted_error());
        }
        let result = session.mine_next_batch().map_err(MiningError::Message)?;
        if result.found {
            // Re-verify on the host so we never submit a bad share.
            let digest = prepared.digest(result.nonce);
            let digest_hex = hex_lower(&digest);
            if leading_zero_bits(&digest) < prepared.difficulty_bits() {
                return Err(MiningError::Message(
                    "Metal DWC backend returned a counter below the requested difficulty."
                        .to_string(),
                ));
            }
            return Ok(Some(DwcShare {
                counter: result.nonce,
                nonce: prepared.nonce_string(result.nonce),
                digest_hex,
                attempts: result.attempts,
                backend: DwcBackend::Metal,
            }));
        }
        start = start.wrapping_add(batch_size);
        remaining = remaining.saturating_sub(batch_size);
    }
    let _ = start;
    Ok(None)
}

// ---------------------------------------------------------------------------
// HTTP client for the public mining API.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct DwcConfig {
    pub epoch: String,
    #[serde(default)]
    pub difficulty: u32,
    #[serde(rename = "epochMs", default)]
    pub epoch_ms: u64,
    #[serde(default)]
    pub prefix: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DwcStats {
    #[serde(default)]
    pub shares: u64,
    #[serde(rename = "unpaidShares", default)]
    pub unpaid_shares: u64,
}

/// Blocking HTTP client for `https://digitalwatercoin.com/api`.
pub struct DwcClient {
    http: reqwest::blocking::Client,
    base: String,
}

impl DwcClient {
    pub fn new() -> Result<Self, MiningError> {
        Self::with_base(DEFAULT_API_BASE)
    }

    pub fn with_base(base: impl Into<String>) -> Result<Self, MiningError> {
        let http = reqwest::blocking::Client::builder()
            .user_agent("Mozilla/5.0 (compatible; dwcmine/0.1)")
            .build()?;
        Ok(Self {
            http,
            base: base.into().trim_end_matches('/').to_string(),
        })
    }

    pub fn config(&self) -> Result<DwcConfig, MiningError> {
        let url = format!("{}/mine/config", self.base);
        decode(self.http.get(url).send()?)
    }

    pub fn stats(&self, address: &str) -> Result<DwcStats, MiningError> {
        let url = format!("{}/mine/stats/{}", self.base, address);
        decode(self.http.get(url).send()?)
    }

    /// Submit a share. Returns the raw JSON body so callers can inspect
    /// acceptance / updated share counts without a fixed schema.
    pub fn submit(
        &self,
        address: &str,
        epoch: &str,
        nonce: &str,
    ) -> Result<serde_json::Value, MiningError> {
        let url = format!("{}/mine/submit", self.base);
        let body = serde_json::json!({
            "address": address,
            "epoch": epoch,
            "nonce": nonce,
        });
        decode(self.http.post(url).json(&body).send()?)
    }
}

fn decode<T: for<'de> Deserialize<'de>>(
    response: reqwest::blocking::Response,
) -> Result<T, MiningError> {
    let status = response.status();
    let text = response.text()?;
    if !status.is_success() {
        let message = serde_json::from_str::<serde_json::Value>(&text)
            .ok()
            .and_then(|value| {
                value
                    .get("error")
                    .or_else(|| value.get("message"))
                    .and_then(|v| v.as_str())
                    .map(str::to_string)
            })
            .filter(|m| !m.trim().is_empty())
            .unwrap_or(text);
        return Err(MiningError::Message(format!(
            "DWC request failed (status {}): {}",
            status.as_u16(),
            message
        )));
    }
    Ok(serde_json::from_str(&text)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn known_job() -> DwcJob {
        DwcJob::new(
            "0x000102030405060708090a0b0c0d0e0f10111213",
            "5933431",
            "7c34a2eec52b6574",
            5,
        )
    }

    #[test]
    fn preimage_and_nonce_layout() {
        let job = known_job();
        // counter 0x3740c -> "000000000003740c"
        assert_eq!(job.nonce_string(0x3740c), "7c34a2eec52b6574000000000003740c");
        assert_eq!(
            job.preimage(0x3740c),
            "0x000102030405060708090a0b0c0d0e0f10111213|5933431|7c34a2eec52b6574000000000003740c"
        );
    }

    #[test]
    fn prepared_digest_matches_oneshot_sha256() {
        let job = known_job();
        let prepared = DwcPreparedJob::new(&job);
        for counter in [0u64, 1, 0x3740c, 0xdead_beef, u64::MAX] {
            let expected: [u8; 32] = Sha256::digest(job.preimage(counter).as_bytes()).into();
            assert_eq!(prepared.digest(counter), expected, "counter {counter:#x}");
        }
    }

    #[test]
    fn known_difficulty5_share_is_valid() {
        // Found by the reference Python miner against the live algorithm.
        let job = known_job();
        let prepared = DwcPreparedJob::new(&job);
        let counter = 0x3740c;
        let digest = prepared.digest(counter);
        assert!(hex_lower(&digest).starts_with("00000"));
        assert!(leading_zero_bits(&digest) >= job.difficulty_bits());
        assert!(prepared.meets_difficulty(counter));
    }

    #[test]
    fn leading_zero_bits_counts_from_msb() {
        assert_eq!(leading_zero_bits(&[0x00, 0x00, 0x08]), 20);
        assert_eq!(leading_zero_bits(&[0xff]), 0);
        assert_eq!(leading_zero_bits(&[0u8; 32]), 256);
    }

    #[test]
    fn cpu_miner_finds_low_difficulty_share() {
        let job = DwcJob::new(
            "0x000102030405060708090a0b0c0d0e0f10111213",
            "5933431",
            "00",
            3,
        );
        let cancel = Arc::new(AtomicBool::new(false));
        let config = DwcMineConfig {
            prefer_gpu: false,
            cpu_threads: 2,
            ..Default::default()
        };
        let share = mine_dwc_cpu(&job, config, &cancel).expect("share");
        assert!(share.digest_hex.starts_with("000"));
        // Independent recompute of the exact server preimage.
        let recomputed: [u8; 32] = Sha256::digest(job.preimage(share.counter).as_bytes()).into();
        assert_eq!(hex_lower(&recomputed), share.digest_hex);
    }
}
