use std::env;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use crossterm as _;
use hdd_autopilot as _;
#[cfg(not(target_os = "macos"))]
use iana_time_zone as _;
use mining::rpow2::{
    Rpow2Backend, Rpow2Client, Rpow2Job, Rpow2MineConfig, mine_rpow2, rpow2_meets_difficulty,
};
use rand as _;
use reqwest as _;
use serde as _;
use serde_json as _;
use time as _;
use unicode_width as _;
use url as _;

fn main() {
    if let Err(error) = run() {
        eprintln!("rpow2mine: {}", error);
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
        let challenge = client.challenge()?;
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
        let mint = client.mint(challenge.challenge_id, result.nonce)?;
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
        metal_batch_size: args.gpu_batch_size.unwrap_or(0),
        cpu_threads: args.cpu_threads.unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(usize::from)
                .unwrap_or(1)
        }),
        ..Rpow2MineConfig::default()
    };
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
  --gpu-batch-size 0 enables automatic GPU batch tuning and is the default
  Windows GPU mining uses CUDA; macOS GPU mining uses Metal"
    );
}
