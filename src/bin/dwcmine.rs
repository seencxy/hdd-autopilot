//! DigitalWaterCoin (DWC) GPU/CPU share miner.
//!
//! Reproduces the browser miner at `https://digitalwatercoin.com/mine` but
//! drives the proof-of-work on the GPU (Metal) or all CPU cores instead of a
//! single-threaded WebCrypto worker. A share is a nonce such that
//! `sha256_hex("{address}|{epoch}|{nonce}")` starts with `difficulty` hex zeros.
//!
//! By default it runs a *dry run*: it mines and locally verifies shares but
//! does NOT contact the submit endpoint. Pass `--submit` to actually credit
//! shares to `--address`.

use std::env;
use std::error::Error;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use mining::dwc::{
    DwcBackend, DwcClient, DwcJob, DwcMineConfig, current_epoch, hex_lower, mine_dwc, mine_dwc_cpu,
    share_digest,
};

// Satisfy `unused_crate_dependencies` for deps pulled in by the binary crate.
use crossterm as _;
use ethers_core as _;
use hdd_autopilot as _;
use reqwest as _;
use serde as _;
use serde_json as _;
use time as _;
use tokio as _;
use unicode_width as _;
use url as _;
#[cfg(not(target_os = "macos"))]
use iana_time_zone as _;

use rand::RngCore;

const CONFIG_REFRESH: Duration = Duration::from_secs(30);
const STATS_INTERVAL: Duration = Duration::from_secs(5);

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        let mut source = error.source();
        while let Some(inner) = source {
            eprintln!("caused by: {inner}");
            source = inner.source();
        }
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let args = Args::parse(env::args().skip(1).collect())?;
    if args.help {
        print_usage();
        return Ok(());
    }
    // A single explicit --address keeps the original single-wallet loop;
    // otherwise run the multi-address worker pool.
    if args.address.is_some() {
        run_single(&args)
    } else {
        run_multi(&args)
    }
}

fn run_single(args: &Args) -> Result<(), Box<dyn Error>> {
    let address = args.address.clone().ok_or(
        "missing --address 0x...; your wallet address is your miner ID",
    )?;
    if !(address.starts_with("0x") && address.len() == 42) {
        return Err("--address must be a 0x-prefixed 40-hex-char Ethereum address".into());
    }
    // The server normalizes the miner address to lowercase before hashing
    // sha256("address|epoch|nonce"), so a checksummed (mixed-case) address
    // would mine shares the server rejects with "Hash does not meet difficulty".
    let address = address.to_lowercase();

    let proxies = match &args.proxies {
        Some(path) => mining::dwc::load_proxy_file(path)?,
        None => Vec::new(),
    };
    if args.proxies.is_some() && proxies.is_empty() {
        return Err(format!(
            "--proxies file {:?} had no usable proxy lines",
            args.proxies.as_deref().unwrap_or("")
        )
        .into());
    }
    let client = DwcClient::with_base_and_proxies(&args.api_base, &proxies)?;
    let mut config = client.config()?;
    let mut config_fetched = Instant::now();
    let difficulty = args.difficulty.unwrap_or(config.difficulty.max(1));

    let backend_label = if args.cpu_only { "cpu" } else { "cuda→metal→cpu" };
    println!(
        "address={address} api={} backend={backend_label} difficulty={difficulty} submit={} proxies={}",
        args.api_base,
        args.submit,
        client.proxy_count()
    );
    if client.proxy_count() > 0 {
        println!(
            "submitting shares through a random proxy IP from {} ({} entries); config/stats stay direct",
            args.proxies.as_deref().unwrap_or(""),
            client.proxy_count()
        );
    }
    println!(
        "server epoch={} epochMs={} server difficulty={} prefix=\"{}\"",
        config.epoch, config.epoch_ms, config.difficulty, config.prefix
    );
    if !args.submit {
        println!("(dry run — shares are verified locally but NOT submitted; pass --submit to credit them)");
    }

    let cancel = Arc::new(AtomicBool::new(false));
    let mut rng = rand::rng();

    // Per (epoch,salt) state.
    let mut salt = random_salt(&mut rng);
    let mut counter: u64 = 0;

    let mut shares_found: u64 = 0;
    let mut shares_submitted: u64 = 0;
    let mut total_attempts: i64 = 0;
    let mut mine_secs: f64 = 0.0;
    let run_start = Instant::now();
    let mut last_stats = Instant::now();
    let mut last_submit = Instant::now()
        .checked_sub(Duration::from_secs(3600))
        .unwrap_or_else(Instant::now);

    loop {
        if cancel.load(Ordering::SeqCst) {
            break;
        }
        // Refresh the server config (epoch rotates every 5 min).
        if config_fetched.elapsed() >= CONFIG_REFRESH || epoch_is_stale(&config.epoch) {
            if let Ok(fresh) = client.config() {
                if fresh.epoch != config.epoch {
                    // New epoch: rotate the salt and reset the counter.
                    salt = random_salt(&mut rng);
                    counter = 0;
                    println!("epoch rotated → {}", fresh.epoch);
                }
                config = fresh;
                config_fetched = Instant::now();
            }
        }

        let job = DwcJob::new(&address, &config.epoch, &salt, difficulty);
        let mine_config = DwcMineConfig {
            start_counter: counter,
            counter_count: u64::MAX - counter,
            cpu_threads: args.threads,
            prefer_gpu: !args.cpu_only,
            allow_cpu_fallback: true,
            gpu_device_index: args.device,
            metal_batch_size: args.batch,
            ..DwcMineConfig::default()
        };

        let mine_start = Instant::now();
        let share = if args.cpu_only {
            mine_dwc_cpu(&job, mine_config, &cancel)
        } else {
            mine_dwc(&job, mine_config, &cancel)
        };
        mine_secs += mine_start.elapsed().as_secs_f64();
        let share = match share {
            Ok(share) => share,
            Err(error) => {
                if cancel.load(Ordering::SeqCst) {
                    break;
                }
                return Err(error.into());
            }
        };

        // Independent host re-verification of the exact server preimage.
        let digest = share_digest(&job, share.counter);
        let digest_hex = hex_lower(&digest);
        if !digest_hex.starts_with(&"0".repeat(difficulty as usize)) {
            return Err(format!(
                "internal error: backend share failed verification: {digest_hex}"
            )
            .into());
        }

        shares_found += 1;
        total_attempts = total_attempts.saturating_add(share.attempts);
        let backend = match share.backend {
            DwcBackend::Cuda => "cuda",
            DwcBackend::Metal => "metal",
            DwcBackend::Cpu => "cpu",
        };

        if args.submit {
            // Be polite to the server: optional minimum spacing between submits.
            if args.min_submit_interval_ms > 0 {
                let min = Duration::from_millis(args.min_submit_interval_ms);
                let since = last_submit.elapsed();
                if since < min {
                    std::thread::sleep(min - since);
                }
            }
            match client.submit(&address, &config.epoch, &share.nonce) {
                Ok(body) => {
                    shares_submitted += 1;
                    last_submit = Instant::now();
                    let remaining = body.get("dailyAddrRemaining").and_then(|v| v.as_i64());
                    println!(
                        "share #{shares_found} accepted nonce={} hash={} [{backend}] resp={}",
                        share.nonce, share.digest_hex, body
                    );
                    // The server enforces a per-address daily share cap
                    // (dailyAddrCap, currently 200). Stop once it's exhausted
                    // instead of spamming rejected submits.
                    if remaining == Some(0) {
                        println!("daily per-address share cap reached — stopping.");
                        break;
                    }
                }
                Err(error) => {
                    eprintln!("submit failed for nonce={}: {error}", share.nonce);
                    // Likely an epoch rollover — force a refresh next loop.
                    config_fetched = Instant::now()
                        .checked_sub(CONFIG_REFRESH)
                        .unwrap_or_else(Instant::now);
                }
            }
        } else {
            println!(
                "share #{shares_found} nonce={} hash={} [{backend}] attempts={}",
                share.nonce, share.digest_hex, share.attempts
            );
        }

        // Advance past this counter so the next share is distinct.
        counter = share.counter.wrapping_add(1);

        if last_stats.elapsed() >= STATS_INTERVAL {
            let wall = run_start.elapsed().as_secs_f64().max(0.001);
            let gpu_rate = total_attempts as f64 / mine_secs.max(0.001) / 1.0e6;
            println!(
                "[stats] found={shares_found} submitted={shares_submitted} attempts={total_attempts} gpu_rate={gpu_rate:.1} MH/s mine_time={mine_secs:.1}s wall={wall:.0}s",
            );
            last_stats = Instant::now();
        }

        if args.once {
            break;
        }
        if let Some(max) = args.max_shares {
            if shares_found >= max {
                break;
            }
        }
    }

    let wall = run_start.elapsed().as_secs_f64().max(0.001);
    let gpu_rate = total_attempts as f64 / mine_secs.max(0.001) / 1.0e6;
    println!(
        "done: found={shares_found} submitted={shares_submitted} attempts={total_attempts} gpu_rate={gpu_rate:.1} MH/s mine_time={mine_secs:.1}s wall={wall:.0}s",
    );
    Ok(())
}
/// Multi-address mode: mine auto-generated wallets concurrently and, the
/// instant an address fills its daily cap (or is rate-limited), replace it
/// with a brand-new wallet — running indefinitely. Each new wallet is appended
/// to the store file as it is created (private keys saved so DWC stays
/// spendable). Mining one share at a time per worker wastes no hash power, and
/// the running wallet count is reported.
fn run_multi(args: &Args) -> Result<(), Box<dyn Error>> {
    let proxies = match &args.proxies {
        Some(path) => mining::dwc::load_proxy_file(path)?,
        None => Vec::new(),
    };
    if args.proxies.is_some() && proxies.is_empty() {
        return Err("--proxies file had no usable proxy lines".into());
    }
    let client = Arc::new(DwcClient::with_base_and_proxies(&args.api_base, &proxies)?);

    let config = client.config()?;
    let difficulty = args.difficulty.unwrap_or(config.difficulty.max(1));
    // Align to the server's epoch counter despite any local clock skew.
    let server_epoch: i64 = config
        .epoch
        .parse()
        .unwrap_or_else(|_| current_epoch() as i64);
    let epoch_offset: i64 = server_epoch - current_epoch() as i64;

    // Optional cap on wallets created this run (0 = unlimited). In dry-run we
    // bound it so we don't spin generating wallets forever.
    let mut max_new = args.max_addresses;
    if !args.submit && max_new == 0 {
        max_new = args.concurrency.max(1) as u64;
    }

    let existing = mining::dwc::load_wallets(&args.wallets)?.len() as u64;

    println!(
        "multi-address mode: concurrency={}, difficulty={}, submit={}, proxies={}, new-wallet cap={}",
        args.concurrency,
        difficulty,
        args.submit,
        client.proxy_count(),
        if max_new == 0 {
            "unlimited".to_string()
        } else {
            max_new.to_string()
        }
    );
    println!(
        "wallet store: {} ({} already saved) — ⚠ plaintext private keys; keep it safe & backed up to spend mined DWC",
        args.wallets, existing
    );
    println!(
        "mining backend: {}",
        if args.gpu {
            "gpu (shared)"
        } else {
            "cpu (1 thread/worker)"
        }
    );
    if !args.submit {
        println!("(dry run — shares verified locally but NOT submitted; pass --submit to credit them)");
    }
    if args.submit && client.proxy_count() == 0 {
        println!(
            "note: the server rate-limits ~1 share / 12s PER IP. With no proxies you're capped near 5 shares/min regardless of --concurrency; add working proxies (many IPs) to scale."
        );
    }

    let book = Arc::new(Mutex::new(WalletBook {
        path: args.wallets.clone(),
        created: 0,
    }));
    let cancel = Arc::new(AtomicBool::new(false));
    let total_found = Arc::new(AtomicU64::new(0));
    let total_submitted = Arc::new(AtomicU64::new(0));
    let active = Arc::new(AtomicU64::new(0));
    let created_count = Arc::new(AtomicU64::new(0));
    let run_start = Instant::now();

    let reporter = {
        let total_found = Arc::clone(&total_found);
        let total_submitted = Arc::clone(&total_submitted);
        let created_count = Arc::clone(&created_count);
        let active = Arc::clone(&active);
        let cancel = Arc::clone(&cancel);
        std::thread::spawn(move || {
            while !cancel.load(Ordering::SeqCst) {
                std::thread::sleep(Duration::from_secs(5));
                if cancel.load(Ordering::SeqCst) {
                    break;
                }
                let secs = run_start.elapsed().as_secs_f64().max(0.001);
                let submitted = total_submitted.load(Ordering::Relaxed);
                println!(
                    "[stats] wallets_created={} active={} found={} submitted={} rate={:.1} shares/s elapsed={:.0}s",
                    created_count.load(Ordering::Relaxed),
                    active.load(Ordering::Relaxed),
                    total_found.load(Ordering::Relaxed),
                    submitted,
                    submitted as f64 / secs,
                    secs
                );
            }
        })
    };

    let handles: Vec<_> = (0..args.concurrency.max(1))
        .map(|_| {
            let client = Arc::clone(&client);
            let book = Arc::clone(&book);
            let cancel = Arc::clone(&cancel);
            let total_found = Arc::clone(&total_found);
            let total_submitted = Arc::clone(&total_submitted);
            let active = Arc::clone(&active);
            let created_count = Arc::clone(&created_count);
            let submit = args.submit;
            let prefer_gpu = args.gpu;
            let proxied = client.proxy_count() > 0;
            // The server rate-limits per IP (~1 accepted share / 12s). With one
            // IP we must pace; with a proxy pool each submit uses a random IP so
            // we don't pace proactively and just retry elsewhere on a 429.
            let pace = if args.min_submit_interval_ms > 0 {
                args.min_submit_interval_ms
            } else if proxied {
                0
            } else {
                12_000
            };
            std::thread::spawn(move || {
                let mut rng = rand::rng();
                loop {
                    if cancel.load(Ordering::SeqCst) {
                        break;
                    }
                    if max_new != 0 && created_count.load(Ordering::Relaxed) >= max_new {
                        break;
                    }
                    // Create + persist a fresh wallet under the lock.
                    let (wallet, n) = {
                        let mut book = book.lock().unwrap();
                        if max_new != 0 && book.created >= max_new {
                            break;
                        }
                        match book.create() {
                            Ok(pair) => pair,
                            Err(error) => {
                                eprintln!("wallet create failed: {error}");
                                break;
                            }
                        }
                    };
                    created_count.store(n, Ordering::Relaxed);
                    active.fetch_add(1, Ordering::Relaxed);
                    println!("[wallet #{}] {} — mining", existing + n, wallet.address);

                    let salt = random_salt(&mut rng);
                    let mut counter = 0u64;
                    let mut consec_neterr = 0u32;
                    let mut addr_shares = 0u64;
                    let mut addr_submitted = 0u64;
                    // Safety bound if a submit response ever omits the cap field.
                    const MAX_SHARES_PER_ADDRESS: u64 = 1000;
                    // 429 is per-IP, not per-address, so it never rotates the
                    // address. Only persistent network failures do.
                    const ROTATE_AFTER_NETERR: u32 = 10;
                    loop {
                        if cancel.load(Ordering::SeqCst) {
                            break;
                        }
                        let epoch = ((current_epoch() as i64) + epoch_offset).to_string();
                        let job = DwcJob::new(&wallet.address, &epoch, &salt, difficulty);
                        let mine_config = DwcMineConfig {
                            start_counter: counter,
                            counter_count: u64::MAX - counter,
                            cpu_threads: 1,
                            prefer_gpu,
                            allow_cpu_fallback: true,
                            ..DwcMineConfig::default()
                        };
                        let share = match mine_dwc(&job, mine_config, &cancel) {
                            Ok(share) => share,
                            Err(_) => break,
                        };
                        counter = share.counter.wrapping_add(1);
                        addr_shares += 1;
                        total_found.fetch_add(1, Ordering::Relaxed);
                        if !submit {
                            if addr_shares >= 3 {
                                break;
                            }
                            continue;
                        }
                        match client.submit(&wallet.address, &epoch, &share.nonce) {
                            Ok(body) => {
                                consec_neterr = 0;
                                addr_submitted += 1;
                                total_submitted.fetch_add(1, Ordering::Relaxed);
                                if body.get("dailyAddrRemaining").and_then(|v| v.as_i64())
                                    == Some(0)
                                {
                                    println!(
                                        "[wallet #{}] {} — full ({} submitted today), creating next",
                                        existing + n,
                                        wallet.address,
                                        addr_submitted
                                    );
                                    break;
                                }
                                if pace > 0 {
                                    std::thread::sleep(Duration::from_millis(pace));
                                }
                            }
                            Err(error) => {
                                let msg = error.to_string();
                                let rate_limited = msg.contains("429")
                                    || msg.contains("too fast")
                                    || msg.contains("slow down");
                                if rate_limited {
                                    // Per-IP rate limit — NOT this address being
                                    // full. Keep the address; wait the server's
                                    // requested time (or retry quickly via a
                                    // different proxy IP) and try again.
                                    consec_neterr = 0;
                                    let wait_ms = if proxied {
                                        750
                                    } else {
                                        parse_wait_ms(&msg).unwrap_or(pace.max(1000))
                                    };
                                    std::thread::sleep(Duration::from_millis(wait_ms));
                                } else {
                                    // Transient network/proxy error: retry, do
                                    // NOT burn this address. Surface the reason.
                                    consec_neterr += 1;
                                    if consec_neterr == 1 || consec_neterr % 10 == 0 {
                                        let short: String = msg.chars().take(120).collect();
                                        eprintln!(
                                            "[wallet #{}] submit error ({}x): {}",
                                            existing + n, consec_neterr, short
                                        );
                                    }
                                    if consec_neterr >= ROTATE_AFTER_NETERR {
                                        println!(
                                            "[wallet #{}] {} — submit failing persistently (check proxies/network), creating next",
                                            existing + n, wallet.address
                                        );
                                        break;
                                    }
                                    let backoff = 100u64.saturating_mul(consec_neterr as u64).min(2000);
                                    std::thread::sleep(Duration::from_millis(backoff));
                                }
                            }
                        }
                        if addr_shares >= MAX_SHARES_PER_ADDRESS {
                            break;
                        }
                    }
                    active.fetch_sub(1, Ordering::Relaxed);
                }
            })
        })
        .collect();

    for handle in handles {
        let _ = handle.join();
    }
    cancel.store(true, Ordering::SeqCst);
    let _ = reporter.join();

    let secs = run_start.elapsed().as_secs_f64().max(0.001);
    let submitted = total_submitted.load(Ordering::Relaxed);
    println!(
        "done: wallets_created={} (total in store {}) found={} submitted={} rate={:.1} shares/s elapsed={:.0}s",
        created_count.load(Ordering::Relaxed),
        existing + created_count.load(Ordering::Relaxed),
        total_found.load(Ordering::Relaxed),
        submitted,
        submitted as f64 / secs,
        secs
    );
    Ok(())
}

/// Wallet store guarded across workers; creating a wallet generates a keypair
/// and appends it to the store file before returning.
struct WalletBook {
    path: String,
    created: u64,
}

impl WalletBook {
    fn create(&mut self) -> Result<(mining::dwc::DwcWallet, u64), Box<dyn Error>> {
        let wallet = mining::dwc::generate_wallet();
        mining::dwc::append_wallet(&self.path, &wallet)?;
        self.created += 1;
        Ok((wallet, self.created))
    }
}

/// Parse the seconds from a server message like "… wait 12s before next share"
/// into milliseconds (+0.5s slack).
fn parse_wait_ms(msg: &str) -> Option<u64> {
    let start = msg.find("wait ")? + "wait ".len();
    let digits: String = msg[start..].chars().take_while(|c| c.is_ascii_digit()).collect();
    let secs: u64 = digits.parse().ok()?;
    Some(secs.saturating_mul(1000).saturating_add(500))
}

fn random_salt(rng: &mut impl RngCore) -> String {
    // 8 random bytes → 16 hex chars of session salt.
    let mut bytes = [0u8; 8];
    rng.fill_bytes(&mut bytes);
    hex_lower(&bytes)
}

/// True if our locally-derived epoch has advanced past the one we hold.
fn epoch_is_stale(server_epoch: &str) -> bool {
    match server_epoch.parse::<u64>() {
        Ok(epoch) => current_epoch() > epoch,
        Err(_) => false,
    }
}

#[derive(Debug, Clone)]
struct Args {
    help: bool,
    address: Option<String>,
    api_base: String,
    difficulty: Option<u32>,
    threads: usize,
    cpu_only: bool,
    device: usize,
    batch: u64,
    submit: bool,
    once: bool,
    max_shares: Option<u64>,
    min_submit_interval_ms: u64,
    proxies: Option<String>,
    wallets: String,
    max_addresses: u64,
    concurrency: usize,
    gpu: bool,
}

impl Default for Args {
    fn default() -> Self {
        let defaults = DwcMineConfig::default();
        Self {
            help: false,
            address: None,
            api_base: mining::dwc::DEFAULT_API_BASE.to_string(),
            difficulty: None,
            threads: defaults.cpu_threads,
            cpu_only: false,
            device: 0,
            batch: defaults.metal_batch_size,
            submit: false,
            once: false,
            max_shares: None,
            min_submit_interval_ms: 0,
            proxies: None,
            wallets: "dwc-wallets.jsonl".to_string(),
            max_addresses: 0,
            concurrency: 16,
            gpu: false,
        }
    }
}

impl Args {
    fn parse(raw: Vec<String>) -> Result<Self, Box<dyn Error>> {
        let mut args = Self::default();
        let mut index = 0usize;
        while index < raw.len() {
            match raw[index].as_str() {
                "-h" | "--help" => args.help = true,
                "--address" | "--wallet" => {
                    args.address = Some(next(&raw, &mut index, "--address")?.to_string())
                }
                "--api-base" => args.api_base = next(&raw, &mut index, "--api-base")?.to_string(),
                "--difficulty" => {
                    args.difficulty = Some(parse_u32(next(&raw, &mut index, "--difficulty")?)?)
                }
                "--threads" | "--cpu-threads" => {
                    args.threads = parse_usize(next(&raw, &mut index, "--threads")?)?
                }
                "--cpu" | "--cpu-only" => args.cpu_only = true,
                "--device" => args.device = parse_usize(next(&raw, &mut index, "--device")?)?,
                "--batch" => args.batch = parse_u64(next(&raw, &mut index, "--batch")?)?,
                "--submit" => args.submit = true,
                "--once" => args.once = true,
                "--max-shares" => {
                    args.max_shares = Some(parse_u64(next(&raw, &mut index, "--max-shares")?)?)
                }
                "--min-submit-interval-ms" => {
                    args.min_submit_interval_ms =
                        parse_u64(next(&raw, &mut index, "--min-submit-interval-ms")?)?
                }
                "--proxies" | "--proxy-file" => {
                    args.proxies = Some(next(&raw, &mut index, "--proxies")?.to_string())
                }
                "--wallets" | "--wallets-file" => {
                    args.wallets = next(&raw, &mut index, "--wallets")?.to_string()
                }
                "--max-addresses" | "--addresses" => {
                    args.max_addresses = parse_u64(next(&raw, &mut index, "--max-addresses")?)?
                }
                "--concurrency" | "--workers" => {
                    args.concurrency = parse_usize(next(&raw, &mut index, "--concurrency")?)?
                }
                "--gpu" => args.gpu = true,
                unknown => return Err(format!("unknown argument: {unknown}").into()),
            }
            index += 1;
        }
        Ok(args)
    }
}

fn next<'a>(raw: &'a [String], index: &mut usize, flag: &str) -> Result<&'a str, Box<dyn Error>> {
    *index += 1;
    raw.get(*index)
        .map(String::as_str)
        .ok_or_else(|| format!("{flag} requires a value").into())
}

fn parse_u64(value: &str) -> Result<u64, Box<dyn Error>> {
    let normalized = value.replace('_', "");
    if let Some(hex) = normalized.strip_prefix("0x") {
        return Ok(u64::from_str_radix(hex, 16)?);
    }
    Ok(normalized.parse()?)
}

fn parse_u32(value: &str) -> Result<u32, Box<dyn Error>> {
    Ok(parse_u64(value)?.try_into()?)
}

fn parse_usize(value: &str) -> Result<usize, Box<dyn Error>> {
    Ok(parse_u64(value)?.try_into()?)
}

fn print_usage() {
    println!(
        r#"Usage:
  Multi-address (default): cargo run --release --bin dwcmine -- --submit [--addresses N]
  Single-address:          cargo run --release --bin dwcmine -- --address 0x<wallet> [options]

Mines DigitalWaterCoin shares: sha256("address|epoch|nonce") with <difficulty>
leading hex zeros. The daily share cap and 429 rate-limit are PER ADDRESS, so
multi-address mode farms many auto-generated wallets concurrently, rotating to
the next address as soon as one is full.

Multi-address options (used when --address is omitted):
  Runs forever: mines a wallet until its daily cap (200) is hit, then creates a
  brand-new wallet to replace it. Stop with Ctrl-C.
  --wallets FILE           wallet store, JSON-lines (default dwc-wallets.jsonl);
                           private keys appended here — keep it safe to spend DWC
  --concurrency N          addresses mined at once / worker threads (default 16)
  --max-addresses N        stop after creating N new wallets (default 0 = unlimited)
  --gpu                    mine on the GPU (shared); default is CPU 1-thread/worker

Single-address options:
  --address 0x...          mine one specific wallet (also your miner ID)

Common options:
  --submit                 actually POST shares to /mine/submit (default: dry run)
  --difficulty N           override difficulty (default: server config, currently 5)
  --cpu                    disable GPU; use CPU workers only
  --threads N              CPU worker threads (default: logical cores)
  --device N               Metal device index (default 0)
  --batch N                Metal counters per kernel launch (default 4194304)
  --once                   mine and report a single share, then exit
  --max-shares N           stop after N shares
  --min-submit-interval-ms N  minimum spacing between submits (be polite)
  --proxies FILE           proxy list (one URL/line); each request picks a random one
  --api-base URL           API base (default https://digitalwatercoin.com/api)
"#
    );
}
