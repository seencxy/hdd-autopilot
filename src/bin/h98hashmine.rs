use std::env;
use std::error::Error;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use crossterm as _;
use hdd_autopilot as _;
use mining::h98hash::{H98HashJob, H98HashMineConfig, decode_fixed_hex, hex_lower, mine_h98hash};
use mining::h98hash_tx::{
    DEFAULT_HASH98_CHAIN_ID, DEFAULT_HASH98_CONTRACT_ADDRESS, H98HashMintOutcome,
    H98HashMintRequest, H98HashMintValue, parse_h98hash_u256, submit_h98hash_mint,
};
use rand as _;
use reqwest as _;
use serde as _;
use serde_json as _;
use time as _;
use tokio as _;
use unicode_width as _;
use url as _;

#[tokio::main]
async fn main() {
    if let Err(error) = run().await {
        eprintln!("error: {error}");
        let mut source = error.source();
        while let Some(error) = source {
            eprintln!("caused by: {error}");
            source = error.source();
        }
        std::process::exit(1);
    }
}

async fn run() -> Result<(), Box<dyn Error>> {
    let args = Args::parse(env::args().skip(1).collect())?;
    if args.help {
        print_usage();
        return Ok(());
    }

    let challenge = args
        .challenge
        .ok_or("missing --challenge 0x...; use challengeFor(wallet) bytes16")?;
    let difficulty_bits = args.difficulty_bits.ok_or("missing --difficulty bits")?;
    let nonce_prefix = args.nonce_prefix.unwrap_or_else(rand::random);
    let job = H98HashJob {
        challenge,
        nonce_prefix,
        difficulty_bits,
    };
    let config = H98HashMineConfig {
        start_nonce: args.start_nonce,
        nonce_count: args.nonce_count,
        cpu_threads: args.cpu_threads,
        prefer_gpu: !args.cpu_only,
        allow_cpu_fallback: args.allow_cpu_fallback,
        cuda_device_index: args.cuda_device_index,
        cuda_batch_size: args.cuda_batch_size,
        cuda_threads_per_block: args.cuda_threads_per_block,
        cuda_nonces_per_thread: args.cuda_nonces_per_thread,
        cuda_max_blocks: args.cuda_max_blocks,
        cuda_early_exit: args.cuda_early_exit,
    };

    println!(
        "challenge=0x{} difficulty={} nonce_prefix=0x{} backend_preference={}",
        hex_lower(&job.challenge),
        job.difficulty_bits,
        hex_lower(&job.nonce_prefix),
        if config.prefer_gpu { "cuda" } else { "cpu" }
    );
    println!(
        "range_start={} range_count={} cuda_batch={} cuda_device={}",
        config.start_nonce, config.nonce_count, config.cuda_batch_size, config.cuda_device_index
    );

    let cancel = Arc::new(AtomicBool::new(false));
    let result = mine_h98hash(&job, config, &cancel)?;
    println!(
        "solution_nonce=0x{} nonce_tail={} digest={} backend={:?} attempts={}",
        hex_lower(&result.nonce),
        result.nonce_tail,
        result.digest_hex,
        result.backend,
        result.attempts
    );
    println!("mint_arg=0x{}", hex_lower(&result.nonce));
    if args.send_mint || args.dry_run_mint {
        let request = H98HashMintRequest {
            rpc_url: resolve_rpc_url(&args)?,
            private_key: resolve_private_key(&args)?,
            contract_address: args.contract_address.clone(),
            chain_id: args.chain_id,
            nonce: result.nonce,
            value_wei: args.mint_value_wei.clone(),
            gas_limit: args.gas_limit,
            gas_price_wei: args.gas_price_wei,
            verify_before_send: args.verify_before_send,
            dry_run: args.dry_run_mint,
            wait_for_receipt: args.wait_for_receipt,
        };
        println!(
            "mint_submit_mode={} contract={} chain_id={}",
            if request.dry_run { "dry-run" } else { "send" },
            request.contract_address,
            request.chain_id
        );
        let outcome = submit_h98hash_mint(request).await?;
        print_mint_outcome(&outcome);
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct Args {
    help: bool,
    challenge: Option<[u8; 16]>,
    difficulty_bits: Option<u32>,
    nonce_prefix: Option<[u8; 8]>,
    start_nonce: u64,
    nonce_count: u64,
    cpu_threads: usize,
    cpu_only: bool,
    allow_cpu_fallback: bool,
    cuda_device_index: usize,
    cuda_batch_size: u64,
    cuda_threads_per_block: u32,
    cuda_nonces_per_thread: u32,
    cuda_max_blocks: u32,
    cuda_early_exit: bool,
    send_mint: bool,
    dry_run_mint: bool,
    rpc_url: Option<String>,
    private_key: Option<String>,
    private_key_env: Option<String>,
    contract_address: String,
    chain_id: u64,
    mint_value_wei: H98HashMintValue,
    gas_limit: Option<u64>,
    gas_price_wei: Option<ethers_core::types::U256>,
    verify_before_send: bool,
    wait_for_receipt: bool,
}

impl Default for Args {
    fn default() -> Self {
        let defaults = H98HashMineConfig::default();
        Self {
            help: false,
            challenge: None,
            difficulty_bits: None,
            nonce_prefix: None,
            start_nonce: defaults.start_nonce,
            nonce_count: defaults.nonce_count,
            cpu_threads: defaults.cpu_threads,
            cpu_only: false,
            allow_cpu_fallback: false,
            cuda_device_index: defaults.cuda_device_index,
            cuda_batch_size: defaults.cuda_batch_size,
            cuda_threads_per_block: defaults.cuda_threads_per_block,
            cuda_nonces_per_thread: defaults.cuda_nonces_per_thread,
            cuda_max_blocks: defaults.cuda_max_blocks,
            cuda_early_exit: defaults.cuda_early_exit,
            send_mint: false,
            dry_run_mint: false,
            rpc_url: None,
            private_key: None,
            private_key_env: None,
            contract_address: DEFAULT_HASH98_CONTRACT_ADDRESS.to_string(),
            chain_id: DEFAULT_HASH98_CHAIN_ID,
            mint_value_wei: H98HashMintValue::AutoMintPrice,
            gas_limit: None,
            gas_price_wei: None,
            verify_before_send: true,
            wait_for_receipt: false,
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
                "--challenge" => {
                    args.challenge = Some(decode_fixed_hex::<16>(
                        next_value(&raw, &mut index, "--challenge")?,
                        "HASH98 challenge",
                    )?)
                }
                "--difficulty" | "--difficulty-bits" => {
                    args.difficulty_bits =
                        Some(parse_u32(next_value(&raw, &mut index, "--difficulty")?)?)
                }
                "--nonce-prefix" => {
                    args.nonce_prefix = Some(decode_fixed_hex::<8>(
                        next_value(&raw, &mut index, "--nonce-prefix")?,
                        "HASH98 nonce prefix",
                    )?)
                }
                "--start" | "--start-nonce" => {
                    args.start_nonce = parse_u64(next_value(&raw, &mut index, "--start")?)?
                }
                "--count" | "--nonce-count" => {
                    args.nonce_count = parse_u64(next_value(&raw, &mut index, "--count")?)?
                }
                "--cpu-threads" => {
                    args.cpu_threads = parse_usize(next_value(&raw, &mut index, "--cpu-threads")?)?
                }
                "--cpu" | "--cpu-only" => args.cpu_only = true,
                "--allow-cpu-fallback" => args.allow_cpu_fallback = true,
                "--device" | "--cuda-device" => {
                    args.cuda_device_index = parse_usize(next_value(&raw, &mut index, "--device")?)?
                }
                "--batch" | "--cuda-batch" => {
                    args.cuda_batch_size = parse_u64(next_value(&raw, &mut index, "--batch")?)?
                }
                "--threads-per-block" => {
                    args.cuda_threads_per_block =
                        parse_u32(next_value(&raw, &mut index, "--threads-per-block")?)?
                }
                "--nonces-per-thread" => {
                    args.cuda_nonces_per_thread =
                        parse_u32(next_value(&raw, &mut index, "--nonces-per-thread")?)?
                }
                "--max-blocks" => {
                    args.cuda_max_blocks = parse_u32(next_value(&raw, &mut index, "--max-blocks")?)?
                }
                "--no-early-exit" => args.cuda_early_exit = false,
                "--send-mint" | "--submit-mint" => args.send_mint = true,
                "--dry-run-mint" => args.dry_run_mint = true,
                "--rpc-url" => {
                    args.rpc_url = Some(next_value(&raw, &mut index, "--rpc-url")?.to_string())
                }
                "--private-key" => {
                    args.private_key =
                        Some(next_value(&raw, &mut index, "--private-key")?.to_string())
                }
                "--private-key-env" => {
                    args.private_key_env =
                        Some(next_value(&raw, &mut index, "--private-key-env")?.to_string())
                }
                "--contract" | "--contract-address" => {
                    args.contract_address = next_value(&raw, &mut index, "--contract")?.to_string()
                }
                "--chain-id" => {
                    args.chain_id = parse_u64(next_value(&raw, &mut index, "--chain-id")?)?
                }
                "--value-wei" => {
                    let value = next_value(&raw, &mut index, "--value-wei")?;
                    args.mint_value_wei = if value.eq_ignore_ascii_case("auto") {
                        H98HashMintValue::AutoMintPrice
                    } else {
                        H98HashMintValue::Exact(parse_h98hash_u256(value)?)
                    };
                }
                "--gas-limit" => {
                    args.gas_limit = Some(parse_u64(next_value(&raw, &mut index, "--gas-limit")?)?)
                }
                "--gas-price-wei" => {
                    args.gas_price_wei = Some(parse_h98hash_u256(next_value(
                        &raw,
                        &mut index,
                        "--gas-price-wei",
                    )?)?)
                }
                "--no-verify-proof" => args.verify_before_send = false,
                "--wait" | "--wait-receipt" => args.wait_for_receipt = true,
                unknown => return Err(format!("unknown argument: {unknown}").into()),
            }
            index += 1;
        }
        if args.send_mint && args.dry_run_mint {
            return Err("--send-mint and --dry-run-mint cannot be used together".into());
        }
        Ok(args)
    }
}

fn next_value<'a>(
    raw: &'a [String],
    index: &mut usize,
    flag: &str,
) -> Result<&'a str, Box<dyn Error>> {
    *index += 1;
    raw.get(*index)
        .map(String::as_str)
        .ok_or_else(|| format!("{flag} requires a value").into())
}

fn parse_u64(value: &str) -> Result<u64, Box<dyn Error>> {
    let normalized = value.replace('_', "");
    if normalized.eq_ignore_ascii_case("max") {
        return Ok(u64::MAX);
    }
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

fn resolve_rpc_url(args: &Args) -> Result<String, Box<dyn Error>> {
    args.rpc_url
        .clone()
        .or_else(|| env::var("H98HASH_RPC_URL").ok())
        .or_else(|| env::var("ETH_RPC_URL").ok())
        .ok_or_else(|| {
            "--send-mint/--dry-run-mint requires --rpc-url or H98HASH_RPC_URL/ETH_RPC_URL".into()
        })
}

fn resolve_private_key(args: &Args) -> Result<String, Box<dyn Error>> {
    if let Some(private_key) = &args.private_key {
        return Ok(private_key.clone());
    }
    if let Some(env_name) = &args.private_key_env {
        return env::var(env_name)
            .map_err(|_| format!("environment variable {env_name} is not set").into());
    }
    env::var("H98HASH_PRIVATE_KEY")
        .or_else(|_| env::var("ETH_PRIVATE_KEY"))
        .map_err(|_| {
            "--send-mint/--dry-run-mint requires --private-key, --private-key-env, or H98HASH_PRIVATE_KEY/ETH_PRIVATE_KEY".into()
        })
}

fn print_mint_outcome(outcome: &H98HashMintOutcome) {
    println!("mint_from={}", outcome.from_address);
    println!("mint_contract={}", outcome.contract_address);
    println!("mint_nonce={}", outcome.nonce_hex);
    println!("mint_value_wei={}", outcome.value_wei);
    if let Some(verified) = outcome.proof_verified {
        println!("mint_proof_verified={verified}");
    }
    println!("mint_gas_estimate={}", outcome.gas_estimate);
    println!("mint_calldata={}", outcome.calldata_hex);
    if let Some(transaction_hash) = &outcome.transaction_hash {
        println!("mint_tx_hash={transaction_hash}");
    }
    if let Some(status) = outcome.receipt_status {
        println!("mint_receipt_status={status}");
    }
    if let Some(block_number) = outcome.receipt_block_number {
        println!("mint_receipt_block={block_number}");
    }
    if let Some(gas_used) = &outcome.receipt_gas_used {
        println!("mint_receipt_gas_used={gas_used}");
    }
}

fn print_usage() {
    println!(
        r#"Usage:
  cargo run --bin h98hashmine -- --challenge 0x<16-byte challenge> --difficulty 43 [options]

Required:
  --challenge HEX           bytes16 returned by HASH98 challengeFor(wallet)
  --difficulty BITS         current HASH98 difficulty

Options:
  --nonce-prefix HEX        first 8 bytes of nonce; random if omitted
  --start N                starting nonce tail, default 0
  --count N|max            tail count to scan, default max
  --cpu                    disable CUDA and use CPU workers
  --allow-cpu-fallback     use CPU if CUDA is unavailable
  --cpu-threads N          CPU worker count
  --device N               CUDA device index, default 0
  --batch N                CUDA batch size, default 2147483648
  --threads-per-block N    CUDA threads per block, default 512
  --nonces-per-thread N    CUDA nonces per thread: 1, 2, 4, or 8
  --max-blocks N           CUDA launch block cap; 0 means auto
  --no-early-exit          continue a kernel batch after one hit is found
  --dry-run-mint           after a proof is found, verify and estimate mint tx only
  --send-mint              after a proof is found, submit mint(bytes16)
  --rpc-url URL            Ethereum JSON-RPC URL; or H98HASH_RPC_URL/ETH_RPC_URL
  --private-key HEX        Ethereum private key; prefer H98HASH_PRIVATE_KEY env
  --private-key-env NAME   read private key from a custom env var
  --contract ADDRESS       HASH98 contract, default 0x1E5adF70321CA28b3Ead70Eac545E6055E969e6f
  --chain-id N             EVM chain ID, default 1
  --value-wei auto|N       tx value; auto reads getConfig().mintPrice, default auto
  --gas-limit N            optional gas limit override
  --gas-price-wei N        optional legacy gas price override
  --no-verify-proof        skip verifyProof(account, nonce) preflight
  --wait                   wait for receipt after sending

Output:
  mint_arg is the bytes16 nonce you can pass to HASH98 mint(nonce).
"#
    );
}
