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
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
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

/// Multi-address mode: generate/load many wallets and mine them concurrently.
/// Each worker thread owns one address at a time, mines+submits shares for it,
/// and rotates to the next address as soon as that one is full (daily cap) or
/// rate-limited (429). Mining only one share at a time per worker means no
/// hash power is spent on shares that can't be submitted.
fn run_multi(args: &Args) -> Result<(), Box<dyn Error>> {
    let wallets = mining::dwc::load_or_create_wallets(&args.wallets, args.addresses)?;
    let addresses: Vec<String> = wallets.iter().map(|w| w.address.clone()).collect();
    if addresses.is_empty() {
        return Err("no wallets available; set --addresses N (> 0)".into());
    }

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

    println!(
        "multi-address mode: wallets={} (file {}), concurrency={}, difficulty={}, submit={}, proxies={}",
        addresses.len(),
        args.wallets,
        args.concurrency,
        difficulty,
        args.submit,
        client.proxy_count()
    );
    println!(
        "⚠ private keys are stored in {} — keep it safe and backed up; it's the only way to spend mined DWC.",
        args.wallets
    );
    println!("mining backend: {}", if args.gpu { "gpu (shared)" } else { "cpu (1 thread/worker)" });
    if !args.submit {
        println!("(dry run — shares verified locally but NOT submitted; pass --submit to credit them)");
    }

    let addresses = Arc::new(addresses);
    let claim = Arc::new(AtomicUsize::new(0));
    let cancel = Arc::new(AtomicBool::new(false));
    let total_found = Arc::new(AtomicU64::new(0));
    let total_submitted = Arc::new(AtomicU64::new(0));
    let addrs_done = Arc::new(AtomicU64::new(0));
    let run_start = Instant::now();

    let reporter = {
        let total_found = Arc::clone(&total_found);
        let total_submitted = Arc::clone(&total_submitted);
        let addrs_done = Arc::clone(&addrs_done);
        let cancel = Arc::clone(&cancel);
        let total_addrs = addresses.len();
        std::thread::spawn(move || {
            while !cancel.load(Ordering::SeqCst) {
                std::thread::sleep(Duration::from_secs(5));
                if cancel.load(Ordering::SeqCst) {
                    break;
                }
                let secs = run_start.elapsed().as_secs_f64().max(0.001);
                let submitted = total_submitted.load(Ordering::Relaxed);
                println!(
                    "[stats] addrs_done={}/{} found={} submitted={} rate={:.1} shares/s elapsed={:.0}s",
                    addrs_done.load(Ordering::Relaxed),
                    total_addrs,
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
            let addresses = Arc::clone(&addresses);
            let claim = Arc::clone(&claim);
            let cancel = Arc::clone(&cancel);
            let total_found = Arc::clone(&total_found);
            let total_submitted = Arc::clone(&total_submitted);
            let addrs_done = Arc::clone(&addrs_done);
            let submit = args.submit;
            let prefer_gpu = args.gpu;
            let min_interval = args.min_submit_interval_ms;
            std::thread::spawn(move || {
                let mut rng = rand::rng();
                loop {
                    if cancel.load(Ordering::SeqCst) {
                        break;
                    }
                    let idx = claim.fetch_add(1, Ordering::Relaxed);
                    if idx >= addresses.len() {
                        break;
                    }
                    let address = addresses[idx].clone();
                    let salt = random_salt(&mut rng);
                    let mut counter = 0u64;
                    let mut errors = 0u32;
                    let mut addr_shares = 0u64;
                    // Safety bound in case a submit response ever omits the
                    // remaining-cap field, so we never mine one address forever.
                    const MAX_SHARES_PER_ADDRESS: u64 = 1000;
                    loop {
                        if cancel.load(Ordering::SeqCst) {
                            break;
                        }
                        let epoch = ((current_epoch() as i64) + epoch_offset).to_string();
                        let job = DwcJob::new(&address, &epoch, &salt, difficulty);
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
                            // Dry run: prove a few shares per address, then rotate.
                            if addr_shares >= 3 {
                                break;
                            }
                            continue;
                        }
                        match client.submit(&address, &epoch, &share.nonce) {
                            Ok(body) => {
                                errors = 0;
                                total_submitted.fetch_add(1, Ordering::Relaxed);
                                if body.get("dailyAddrRemaining").and_then(|v| v.as_i64())
                                    == Some(0)
                                {
                                    break; // address full → next
                                }
                            }
                            Err(error) => {
                                errors += 1;
                                let msg = error.to_string();
                                // 429 / cap / repeated failure → give up this address.
                                if msg.contains("429") || msg.contains("limit") || errors >= 3 {
                                    break;
                                }
                            }
                        }
                        if addr_shares >= MAX_SHARES_PER_ADDRESS {
                            break;
                        }
                        if min_interval > 0 {
                            std::thread::sleep(Duration::from_millis(min_interval));
                        }
                    }
                    addrs_done.fetch_add(1, Ordering::Relaxed);
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
        "done: addresses={} found={} submitted={} rate={:.1} shares/s elapsed={:.0}s",
        addresses.len(),
        total_found.load(Ordering::Relaxed),
        submitted,
        submitted as f64 / secs,
        secs
    );
    if args.submit {
        println!(
            "all addresses processed; raise --addresses or run again tomorrow for more (daily cap is per address)."
        );
    }
    Ok(())
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
    addresses: usize,
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
            wallets: "dwc-wallets.json".to_string(),
            addresses: 100,
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
                "--addresses" | "--wallet-count" => {
                    args.addresses = parse_usize(next(&raw, &mut index, "--addresses")?)?
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
  --wallets FILE           wallet store JSON (default dwc-wallets.json); private
                           keys saved here — keep it safe to spend mined DWC
  --addresses N            ensure at least N wallets exist (default 100)
  --concurrency N          worker threads / addresses mined at once (default 16)
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
