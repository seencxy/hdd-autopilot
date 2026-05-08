use std::env;
use std::error::Error;
use std::time::Duration;

use crossterm as _;
use hdd_autopilot as _;
#[cfg(not(target_os = "macos"))]
use iana_time_zone as _;
use mining::rpow2::{
    DEFAULT_CUDA_BATCH_SIZE, DEFAULT_CUDA_EARLY_EXIT, DEFAULT_CUDA_MAX_BLOCKS,
    DEFAULT_CUDA_NONCES_PER_THREAD, DEFAULT_CUDA_THREADS_PER_BLOCK, Rpow2CudaBenchmarkConfig,
    Rpow2Job, benchmark_rpow2_cuda,
};
use rand as _;
use reqwest as _;
use serde as _;
use serde_json as _;
use time as _;
use unicode_width as _;
use url as _;

const DEFAULT_PREFIX_HEX: &str = "000102030405060708090a0b0c0d0e0f";
const DEFAULT_DIFFICULTY_BITS: u32 = 255;

fn main() {
    if let Err(error) = run() {
        eprintln!("rpow2-bench: {}", error_chain_to_string(error.as_ref()));
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let args = Args::parse(env::args().skip(1).collect())?;
    if args.help {
        print_usage();
        return Ok(());
    }

    let prefix = args.prefix.as_deref().unwrap_or(DEFAULT_PREFIX_HEX);
    let difficulty_bits = args.difficulty_bits.unwrap_or(DEFAULT_DIFFICULTY_BITS);
    let job = Rpow2Job::from_nonce_prefix_hex(prefix, difficulty_bits)?;
    let config = Rpow2CudaBenchmarkConfig {
        device_index: args.gpu_device_index.unwrap_or(0),
        batch_size: args.gpu_batch_size.unwrap_or(DEFAULT_CUDA_BATCH_SIZE),
        duration: Duration::from_secs(args.seconds.unwrap_or(30).max(1)),
        threads_per_block: args
            .gpu_threads_per_block
            .unwrap_or(DEFAULT_CUDA_THREADS_PER_BLOCK),
        nonces_per_thread: args
            .gpu_nonces_per_thread
            .unwrap_or(DEFAULT_CUDA_NONCES_PER_THREAD),
        max_blocks: args.gpu_max_blocks.unwrap_or(DEFAULT_CUDA_MAX_BLOCKS),
        early_exit: !args.gpu_no_early_exit,
    };

    println!("prefix={prefix}");
    println!("difficulty_bits={difficulty_bits}");
    println!("gpu_device={}", config.device_index);
    println!("batch_size={}", config.batch_size);
    println!("threads_per_block={}", config.threads_per_block);
    println!("nonces_per_thread={}", config.nonces_per_thread);
    println!("max_blocks={}", gpu_max_blocks_label(config.max_blocks));
    println!("early_exit={}", config.early_exit);
    println!("duration_seconds={:.3}", config.duration.as_secs_f64());

    let report = benchmark_rpow2_cuda(&job, config)?;
    let host_device_overhead_ratio = if report.elapsed.as_secs_f64() > 0.0 {
        report.host_device_overhead.as_secs_f64() / report.elapsed.as_secs_f64()
    } else {
        0.0
    };

    println!("attempts={}", report.attempts);
    println!("batches={}", report.batches);
    println!("elapsed_seconds={:.6}", report.elapsed.as_secs_f64());
    println!("kernel_seconds={:.6}", report.kernel_elapsed.as_secs_f64());
    println!(
        "host_device_overhead_seconds={:.6}",
        report.host_device_overhead.as_secs_f64()
    );
    println!(
        "empty_launch_us={:.3}",
        report.empty_launch.as_secs_f64() * 1_000_000.0
    );
    println!("kernel_hashrate={:.2} H/s", report.kernel_hashrate);
    println!("effective_hashrate={:.2} H/s", report.effective_hashrate);
    println!(
        "host_device_overhead_ratio={:.4}",
        host_device_overhead_ratio
    );
    println!("network_idle_ratio={:.4}", report.network_idle_ratio);
    Ok(())
}

fn gpu_max_blocks_label(max_blocks: u32) -> String {
    if max_blocks == 0 {
        "auto".to_string()
    } else {
        max_blocks.to_string()
    }
}

#[derive(Debug, Default)]
struct Args {
    prefix: Option<String>,
    difficulty_bits: Option<u32>,
    seconds: Option<u64>,
    gpu_batch_size: Option<u64>,
    gpu_device_index: Option<usize>,
    gpu_threads_per_block: Option<u32>,
    gpu_nonces_per_thread: Option<u32>,
    gpu_max_blocks: Option<u32>,
    gpu_no_early_exit: bool,
    help: bool,
}

impl Args {
    fn parse(raw: Vec<String>) -> Result<Self, Box<dyn Error>> {
        let mut args = Self::default();
        let mut index = 0;
        while index < raw.len() {
            match raw[index].as_str() {
                "--prefix" => {
                    index += 1;
                    args.prefix = Some(next_value(&raw, index, "--prefix")?.to_string());
                }
                "--difficulty" => {
                    index += 1;
                    args.difficulty_bits = Some(next_value(&raw, index, "--difficulty")?.parse()?);
                }
                "--seconds" | "--duration" => {
                    index += 1;
                    args.seconds = Some(next_value(&raw, index, raw[index - 1].as_str())?.parse()?);
                }
                "--gpu-batch-size" | "--cuda-batch-size" => {
                    index += 1;
                    args.gpu_batch_size =
                        Some(next_value(&raw, index, raw[index - 1].as_str())?.parse()?);
                }
                "--gpu-device" | "--cuda-device" => {
                    index += 1;
                    args.gpu_device_index =
                        Some(next_value(&raw, index, raw[index - 1].as_str())?.parse()?);
                }
                "--gpu-threads-per-block" | "--cuda-threads-per-block" => {
                    index += 1;
                    args.gpu_threads_per_block =
                        Some(next_value(&raw, index, raw[index - 1].as_str())?.parse()?);
                }
                "--gpu-nonces-per-thread" | "--cuda-nonces-per-thread" => {
                    index += 1;
                    args.gpu_nonces_per_thread =
                        Some(next_value(&raw, index, raw[index - 1].as_str())?.parse()?);
                }
                "--gpu-max-blocks" | "--cuda-max-blocks" => {
                    index += 1;
                    args.gpu_max_blocks =
                        Some(next_value(&raw, index, raw[index - 1].as_str())?.parse()?);
                }
                "--gpu-no-early-exit" | "--cuda-no-early-exit" => {
                    args.gpu_no_early_exit = true;
                }
                "-h" | "--help" => args.help = true,
                other => return Err(format!("unknown argument: {other}").into()),
            }
            index += 1;
        }
        Ok(args)
    }
}

fn next_value<'a>(raw: &'a [String], index: usize, flag: &str) -> Result<&'a str, Box<dyn Error>> {
    raw.get(index)
        .map(String::as_str)
        .ok_or_else(|| format!("{flag} requires a value").into())
}

fn print_usage() {
    println!(
        "Usage:
  rpow2-bench [--gpu-device <index>] [--seconds <n>] [--prefix <hex>] [--difficulty <bits>]
              [--gpu-batch-size <hashes>] [--gpu-threads-per-block <n>]
              [--gpu-nonces-per-thread <n>] [--gpu-max-blocks <n>] [--gpu-no-early-exit]

Defaults:
  prefix                  {DEFAULT_PREFIX_HEX}
  difficulty              {DEFAULT_DIFFICULTY_BITS}
  seconds                 30
  gpu-batch-size          {DEFAULT_CUDA_BATCH_SIZE}
  gpu-threads-per-block   {DEFAULT_CUDA_THREADS_PER_BLOCK}
  gpu-nonces-per-thread   {DEFAULT_CUDA_NONCES_PER_THREAD}
  gpu-max-blocks          auto
  gpu-early-exit          {DEFAULT_CUDA_EARLY_EXIT}"
    );
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
