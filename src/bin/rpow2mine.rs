use std::env;
use std::error::Error;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::thread;
use std::time::{Duration, Instant};

use crossterm as _;
use hdd_autopilot as _;
#[cfg(not(target_os = "macos"))]
use iana_time_zone as _;
use mining::MiningError;
use mining::rpow2::{
    Rpow2Backend, Rpow2Client, Rpow2Job, Rpow2MineConfig, mine_rpow2, rpow2_meets_difficulty,
};
use rand as _;
use reqwest as _;
use serde as _;
use serde_json::{Value, json};
use time as _;
use unicode_width as _;
use url as _;

fn main() {
    if let Err(error) = run() {
        print_error_chain(error.as_ref());
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse(env::args().skip(1).collect())?;
    if args.help {
        print_usage();
        return Ok(());
    }

    if let (Some(prefix), Some(difficulty)) = (args.prefix.as_deref(), args.difficulty_bits) {
        let job = Rpow2Job::from_nonce_prefix_hex(prefix, difficulty)?;
        let result = solve_job(&job, &args)?;
        println!(
            "solution_nonce={} digest={} backend={:?} attempts={}",
            result.nonce, result.digest_hex, result.backend, result.attempts
        );
        if !rpow2_meets_difficulty(&job, result.nonce) {
            return Err("internal verification failed".into());
        }
        return Ok(());
    }

    let cookie = args
        .cookie
        .clone()
        .or_else(|| env::var("RPOW2_COOKIE").ok())
        .ok_or("missing RPOW2 cookie; pass --cookie or set RPOW2_COOKIE")?;
    let client = Rpow2Client::new(Some(&cookie))?;
    if let Ok(me) = client.me() {
        println!("account: {}", me);
    }
    if let Ok(ledger) = client.ledger() {
        println!("ledger: {}", ledger);
    }

    loop {
        let challenge = retry_online("challenge", || client.challenge())?;
        println!(
            "challenge={} difficulty={} prefix={}",
            challenge.challenge_id, challenge.difficulty_bits, challenge.nonce_prefix
        );
        let job = challenge.job()?;
        let result = solve_job(&job, &args)?;
        println!(
            "found solution_nonce={} digest={} backend={:?} attempts={}",
            result.nonce, result.digest_hex, result.backend, result.attempts
        );
        let mint = mint_with_recovery(&client, &challenge.challenge_id, result.nonce)?;
        println!("mint: {}", mint);
        if !args.loop_forever {
            break;
        }
    }
    Ok(())
}

fn solve_job(
    job: &Rpow2Job,
    args: &Args,
) -> Result<mining::rpow2::Rpow2MineResult, Box<dyn std::error::Error>> {
    let started = Instant::now();
    let cancel = Arc::new(AtomicBool::new(false));
    let config = Rpow2MineConfig {
        prefer_gpu: !args.cpu_only,
        metal_device_index: args.gpu_device_index.unwrap_or(0),
        metal_batch_size: args.gpu_batch_size.unwrap_or_else(default_gpu_batch_size),
        cpu_threads: args.cpu_threads.unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(usize::from)
                .unwrap_or(1)
        }),
        ..Rpow2MineConfig::default()
    };
    if !args.cpu_only {
        println!(
            "gpu_device={} gpu_batch_size={}",
            config.metal_device_index,
            gpu_batch_label(config.metal_batch_size)
        );
    }
    let result = mine_rpow2(job, config, &cancel)?;
    let speed = result.attempts as f64 / started.elapsed().as_secs_f64().max(0.001);
    let backend = match result.backend {
        Rpow2Backend::Cpu => "CPU",
        Rpow2Backend::Cuda => "CUDA",
        Rpow2Backend::Metal => "Metal",
    };
    println!("{} speed: {:.2} H/s", backend, speed);
    Ok(result)
}

fn mint_with_recovery(
    client: &Rpow2Client,
    challenge_id: &str,
    solution_nonce: u64,
) -> Result<Value, MiningError> {
    let mut attempt = 1u32;
    loop {
        match client.mint(challenge_id, solution_nonce) {
            Ok(value) => return Ok(value),
            Err(error) if is_already_claimed_error(&error) => {
                eprintln!(
                    "mint already claimed for challenge={}; treating as accepted",
                    challenge_id
                );
                return Ok(json!({
                    "status": "already_claimed",
                    "challenge_id": challenge_id,
                    "solution_nonce": solution_nonce.to_string()
                }));
            }
            Err(error) if is_retryable_online_error(&error) => {
                let delay = retry_delay(attempt);
                eprintln!(
                    "mint request failed on attempt {}: {}; retrying in {}s",
                    attempt,
                    error_chain_to_string(&error),
                    delay.as_secs()
                );
                thread::sleep(delay);
                attempt = attempt.saturating_add(1);
            }
            Err(error) => return Err(error),
        }
    }
}

fn default_gpu_batch_size() -> u64 {
    #[cfg(target_os = "windows")]
    {
        mining::rpow2::FULL_CUDA_BATCH_SIZE
    }
    #[cfg(not(target_os = "windows"))]
    {
        0
    }
}

fn gpu_batch_label(batch_size: u64) -> String {
    if batch_size == 0 {
        "auto".to_string()
    } else {
        batch_size.to_string()
    }
}

fn retry_online<T, F>(label: &str, mut operation: F) -> Result<T, MiningError>
where
    F: FnMut() -> Result<T, MiningError>,
{
    let mut attempt = 1u32;
    loop {
        match operation() {
            Ok(value) => return Ok(value),
            Err(error) if is_retryable_online_error(&error) => {
                let delay = retry_delay(attempt);
                eprintln!(
                    "{} request failed on attempt {}: {}; retrying in {}s",
                    label,
                    attempt,
                    error_chain_to_string(&error),
                    delay.as_secs()
                );
                thread::sleep(delay);
                attempt = attempt.saturating_add(1);
            }
            Err(error) => return Err(error),
        }
    }
}

fn retry_delay(attempt: u32) -> Duration {
    match attempt {
        0 | 1 => Duration::from_secs(1),
        2 => Duration::from_secs(2),
        3 => Duration::from_secs(4),
        4 => Duration::from_secs(8),
        _ => Duration::from_secs(15),
    }
}

fn is_retryable_online_error(error: &MiningError) -> bool {
    match error {
        MiningError::Http(error) => {
            error.is_timeout()
                || error.is_connect()
                || error.is_request()
                || error.is_body()
                || error.status().is_some_and(|status| {
                    status.as_u16() == 408
                        || status.as_u16() == 425
                        || status.as_u16() == 429
                        || status.is_server_error()
                })
        }
        MiningError::Message(message) => retryable_status_message(message),
        _ => false,
    }
}

fn retryable_status_message(message: &str) -> bool {
    [408, 425, 429, 500, 502, 503, 504]
        .into_iter()
        .any(|status| message.contains(&format!("状态码 {status}")))
}

fn is_already_claimed_error(error: &MiningError) -> bool {
    match error {
        MiningError::Message(message) => message.to_ascii_lowercase().contains("already claimed"),
        _ => false,
    }
}

fn print_error_chain(error: &(dyn Error + 'static)) {
    eprintln!("rpow2mine: {}", error_chain_to_string(error));
}

fn error_chain_to_string(error: &(dyn Error + 'static)) -> String {
    let mut parts = vec![error.to_string()];
    let mut source = error.source();
    while let Some(error) = source {
        parts.push(error.to_string());
        source = error.source();
    }
    parts.join(": ")
}

#[derive(Debug, Default)]
struct Args {
    cookie: Option<String>,
    prefix: Option<String>,
    difficulty_bits: Option<u32>,
    cpu_only: bool,
    loop_forever: bool,
    help: bool,
    cpu_threads: Option<usize>,
    gpu_batch_size: Option<u64>,
    gpu_device_index: Option<usize>,
}

impl Args {
    fn parse(raw: Vec<String>) -> Result<Self, Box<dyn std::error::Error>> {
        let mut args = Self::default();
        let mut index = 0;
        while index < raw.len() {
            match raw[index].as_str() {
                "--cookie" => {
                    index += 1;
                    args.cookie = Some(next_value(&raw, index, "--cookie")?.to_string());
                }
                "--prefix" => {
                    index += 1;
                    args.prefix = Some(next_value(&raw, index, "--prefix")?.to_string());
                }
                "--difficulty" => {
                    index += 1;
                    args.difficulty_bits = Some(next_value(&raw, index, "--difficulty")?.parse()?);
                }
                "--cpu-only" => args.cpu_only = true,
                "--loop" => args.loop_forever = true,
                "--cpu-threads" => {
                    index += 1;
                    args.cpu_threads = Some(next_value(&raw, index, "--cpu-threads")?.parse()?);
                }
                "--gpu-batch-size" | "--metal-batch-size" => {
                    index += 1;
                    args.gpu_batch_size =
                        Some(next_value(&raw, index, raw[index - 1].as_str())?.parse()?);
                }
                "--gpu-device" | "--cuda-device" | "--metal-device" => {
                    index += 1;
                    args.gpu_device_index =
                        Some(next_value(&raw, index, raw[index - 1].as_str())?.parse()?);
                }
                "-h" | "--help" => args.help = true,
                other => return Err(format!("unknown argument: {other}").into()),
            }
            index += 1;
        }
        if args.prefix.is_some() ^ args.difficulty_bits.is_some() {
            return Err("--prefix and --difficulty must be provided together".into());
        }
        Ok(args)
    }
}

fn next_value<'a>(
    raw: &'a [String],
    index: usize,
    flag: &str,
) -> Result<&'a str, Box<dyn std::error::Error>> {
    raw.get(index)
        .map(String::as_str)
        .ok_or_else(|| format!("{flag} requires a value").into())
}

fn print_usage() {
    println!(
        "Usage:
  rpow2mine --cookie '<cookie-header>' [--loop] [--cpu-only] [--gpu-device <index>] [--gpu-batch-size <hashes>]
  rpow2mine --prefix <hex> --difficulty <bits> [--cpu-only] [--gpu-device <index>] [--gpu-batch-size <hashes>]

Environment:
  RPOW2_COOKIE   Cookie header copied from an authenticated rpow2.com browser session

Notes:
  Windows GPU mining uses CUDA and defaults to a full-size 2147483648 hash batch
  --gpu-batch-size 0 enables automatic GPU batch tuning
  macOS GPU mining uses Metal"
    );
}
