//! hash256.org Keccak-256 miner CLI.
//!
//! Default mode: continuous loop that pulls `getChallenge(wallet)` +
//! `miningState()`, runs the GPU miner against `keccak256(challenge ‖ nonce)
//! < difficulty`, calls `usedProofs(hash)` to avoid a wasted broadcast, then
//! submits `mine(uint256 nonce)` via EIP-1559. Use `--once` for a single
//! solve+submit and `--dry-run-mint` to stop before broadcasting.

use std::env;
use std::error::Error;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use crossterm as _;
use ethers_core::types::U256;
use hdd_autopilot as _;
use mining::h256hash::{
    H256HASH_DEFAULT_CUDA_MAX_BLOCKS, H256HASH_DEFAULT_CUDA_NONCES_PER_THREAD,
    H256HASH_DEFAULT_CUDA_THREADS_PER_BLOCK, H256HASH_FULL_CUDA_BATCH_SIZE, H256HashBackend,
    H256HashJob, H256HashMineConfig, hex_lower, mine_h256hash,
};
use mining::h256hash_tx::{
    DEFAULT_HASH256_CHAIN_ID, DEFAULT_HASH256_CONTRACT_ADDRESS, DEFAULT_HASH256_GAS_LIMIT,
    DEFAULT_HASH256_MAX_FEE_HEADROOM_WEI, DEFAULT_HASH256_PRIORITY_TIP_WEI, H256HashMintOutcome,
    H256HashMintRequest, parse_h256hash_u256, read_h256hash_mining_round, read_h256hash_status,
    submit_h256hash_mine,
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
    let rpc_url = resolve_rpc_url(&args)?;
    let wallet_address = args.wallet.clone().ok_or(
        "missing --wallet 0x... (the address that will mint and that getChallenge() is keyed on)",
    )?;
    // Sanity-parse the wallet so we fail loud if it is malformed.
    if !wallet_address.starts_with("0x") || wallet_address.len() != 42 {
        return Err(
            format!("--wallet must be a 0x-prefixed 20-byte hex address: got {wallet_address}")
                .into(),
        );
    }

    let private_key = if args.dry_run_mint {
        // Dry-run does not broadcast, but `submit_h256hash_mine` still wants
        // a signer to derive the from-address. Allow the user to skip and
        // synthesize a throw-away key in that case.
        resolve_private_key(&args).unwrap_or_else(|_| {
            "0x0000000000000000000000000000000000000000000000000000000000000001".to_string()
        })
    } else {
        resolve_private_key(&args)?
    };
    let contract_address = args.contract_address.clone();

    println!(
        "hash256 miner starting · contract={} chain_id={} wallet={} prefer_gpu={} dry_run={} once={}",
        contract_address,
        args.chain_id,
        wallet_address,
        !args.cpu_only,
        args.dry_run_mint,
        args.once
    );

    let block_watch_interval = Duration::from_secs(args.block_watch_secs.max(1));
    let mut total_submits = 0u64;
    let mut first_round = true;

    loop {
        // 1. Refresh challenge + difficulty for this attempt.
        //    First round: full status (5 eth_calls) for display.
        //    Subsequent rounds: lightweight poll (2 eth_calls) to avoid rate limits.
        let (challenge, difficulty) = if first_round {
            first_round = false;
            let status =
                match read_h256hash_status(&rpc_url, &contract_address, &wallet_address).await {
                    Ok(status) => status,
                    Err(error) => {
                        eprintln!("warn: failed to read mining state: {error}; retrying in 12s");
                        tokio::time::sleep(Duration::from_secs(12)).await;
                        first_round = true;
                        continue;
                    }
                };
            let mut target = [0u8; 32];
            status.mining_state.difficulty.to_big_endian(&mut target);
            println!(
                "epoch={} epoch_blocks_left={} reward={} difficulty=0x{} challenge=0x{} balance={}",
                status.mining_state.epoch,
                status.mining_state.epoch_blocks_left,
                status.mining_state.reward,
                hex_lower(&target),
                hex_lower(&status.challenge),
                status.balance
            );
            (status.challenge, status.mining_state.difficulty)
        } else {
            match read_h256hash_mining_round(&rpc_url, &contract_address, &wallet_address).await {
                Ok(pair) => pair,
                Err(error) => {
                    eprintln!("warn: failed to read mining state: {error}; retrying in 12s");
                    tokio::time::sleep(Duration::from_secs(12)).await;
                    continue;
                }
            }
        };

        if difficulty.is_zero() {
            eprintln!("warn: on-chain difficulty is zero; sleeping 12s");
            tokio::time::sleep(Duration::from_secs(12)).await;
            continue;
        }
        let mut target = [0u8; 32];
        difficulty.to_big_endian(&mut target);
        println!(
            "difficulty=0x{} challenge=0x{}",
            hex_lower(&target),
            hex_lower(&challenge)
        );

        let start_nonce = args.start_nonce.unwrap_or_else(rand::random::<u64>);
        let mine_config = H256HashMineConfig {
            start_nonce,
            nonce_count: u64::MAX,
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
        let job = H256HashJob { challenge, target };

        // 2. Start a watcher task that flips the cancel flag if the
        //    challenge or difficulty changes on chain (new block → new
        //    epoch, or other miners advanced the difficulty retarget window).
        let cancel = Arc::new(AtomicBool::new(false));
        let watcher_handle = spawn_chain_watcher(
            cancel.clone(),
            rpc_url.clone(),
            contract_address.clone(),
            wallet_address.clone(),
            challenge,
            difficulty,
            block_watch_interval,
        );

        // 3. Mine on the GPU/CPU in a blocking task to avoid hogging the
        //    tokio worker thread.
        let mine_job = job.clone();
        let mine_cancel = cancel.clone();
        let mine_handle = tokio::task::spawn_blocking(move || {
            mine_h256hash(&mine_job, mine_config, &mine_cancel)
        });
        let mine_outcome = mine_handle.await?;
        cancel.store(true, Ordering::SeqCst);
        let _ = watcher_handle.await;

        let result = match mine_outcome {
            Ok(result) => result,
            Err(error) => {
                if error.to_string().contains("interrupted") {
                    println!(
                        "info: mining cancelled by chain watcher (challenge or difficulty changed)"
                    );
                    if args.once {
                        return Err(error.into());
                    }
                    continue;
                }
                if args.once {
                    return Err(error.into());
                }
                eprintln!("error: mining failed: {error}; backing off 5s");
                tokio::time::sleep(Duration::from_secs(5)).await;
                continue;
            }
        };

        println!(
            "solution backend={:?} nonce=0x{} digest=0x{} attempts={}",
            result.backend,
            hex_lower(&result.nonce_be32),
            result.digest_hex,
            result.attempts
        );
        if matches!(result.backend, H256HashBackend::Cpu) && !args.cpu_only {
            eprintln!(
                "warn: solution came from the CPU fallback; check CUDA availability for full speed"
            );
        }

        // 4. Submit.
        let request = H256HashMintRequest {
            rpc_url: rpc_url.clone(),
            private_key: private_key.clone(),
            contract_address: contract_address.clone(),
            chain_id: args.chain_id,
            nonce: result.nonce,
            gas_limit: Some(args.gas_limit),
            priority_tip_wei: args.priority_tip_wei,
            max_fee_wei: args.max_fee_wei,
            max_fee_headroom_wei: args.max_fee_headroom_wei,
            check_used_proofs: !args.skip_used_proofs_check,
            dry_run: args.dry_run_mint,
            wait_for_receipt: args.wait_for_receipt,
        };
        match submit_h256hash_mine(request).await {
            Ok(outcome) => {
                print_mint_outcome(&outcome);
                if !outcome.dry_run {
                    total_submits += 1;
                }
            }
            Err(error) => {
                if args.once {
                    return Err(error.into());
                }
                eprintln!("error: mine() submit failed: {error}; backing off 5s");
                tokio::time::sleep(Duration::from_secs(5)).await;
            }
        }

        if args.once {
            println!("info: --once specified, exiting after one solution");
            break;
        }

        // Give the chain a beat to advance before re-pulling getChallenge.
        // The challenge can stay the same within an epoch, but usedProofs
        // means we must find a different nonce next round anyway.
        tokio::time::sleep(Duration::from_secs(args.post_submit_sleep_secs)).await;
    }

    println!("hash256 miner stopped · total_submits={total_submits}");
    Ok(())
}

fn spawn_chain_watcher(
    cancel: Arc<AtomicBool>,
    rpc_url: String,
    contract_address: String,
    wallet_address: String,
    baseline_challenge: [u8; 32],
    baseline_difficulty: U256,
    interval: Duration,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(interval).await;
            if cancel.load(Ordering::Relaxed) {
                return;
            }
            match read_h256hash_status(&rpc_url, &contract_address, &wallet_address).await {
                Ok(status) => {
                    if status.challenge != baseline_challenge
                        || status.mining_state.difficulty != baseline_difficulty
                    {
                        cancel.store(true, Ordering::SeqCst);
                        return;
                    }
                }
                Err(_) => { /* transient; keep watching */ }
            }
        }
    })
}

#[derive(Debug, Clone)]
struct Args {
    help: bool,
    wallet: Option<String>,
    rpc_url: Option<String>,
    private_key: Option<String>,
    private_key_env: Option<String>,
    contract_address: String,
    chain_id: u64,
    cpu_only: bool,
    allow_cpu_fallback: bool,
    cpu_threads: usize,
    cuda_device_index: usize,
    cuda_batch_size: u64,
    cuda_threads_per_block: u32,
    cuda_nonces_per_thread: u32,
    cuda_max_blocks: u32,
    cuda_early_exit: bool,
    start_nonce: Option<u64>,
    block_watch_secs: u64,
    post_submit_sleep_secs: u64,
    gas_limit: u64,
    priority_tip_wei: U256,
    max_fee_wei: Option<U256>,
    max_fee_headroom_wei: U256,
    skip_used_proofs_check: bool,
    dry_run_mint: bool,
    wait_for_receipt: bool,
    once: bool,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            help: false,
            wallet: None,
            rpc_url: None,
            private_key: None,
            private_key_env: None,
            contract_address: DEFAULT_HASH256_CONTRACT_ADDRESS.to_string(),
            chain_id: DEFAULT_HASH256_CHAIN_ID,
            cpu_only: false,
            allow_cpu_fallback: true,
            cpu_threads: std::thread::available_parallelism()
                .map(usize::from)
                .unwrap_or(1),
            cuda_device_index: 0,
            cuda_batch_size: H256HASH_FULL_CUDA_BATCH_SIZE,
            cuda_threads_per_block: H256HASH_DEFAULT_CUDA_THREADS_PER_BLOCK,
            cuda_nonces_per_thread: H256HASH_DEFAULT_CUDA_NONCES_PER_THREAD,
            cuda_max_blocks: H256HASH_DEFAULT_CUDA_MAX_BLOCKS,
            cuda_early_exit: true,
            start_nonce: None,
            block_watch_secs: 12,
            post_submit_sleep_secs: 12,
            gas_limit: DEFAULT_HASH256_GAS_LIMIT,
            priority_tip_wei: U256::from(DEFAULT_HASH256_PRIORITY_TIP_WEI),
            max_fee_wei: None,
            max_fee_headroom_wei: U256::from(DEFAULT_HASH256_MAX_FEE_HEADROOM_WEI),
            skip_used_proofs_check: false,
            dry_run_mint: false,
            wait_for_receipt: true,
            once: false,
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
                "--wallet" | "--address" => {
                    args.wallet = Some(next_value(&raw, &mut index, "--wallet")?.to_string())
                }
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
                    args.contract_address =
                        next_value(&raw, &mut index, "--contract")?.to_string()
                }
                "--chain-id" => {
                    args.chain_id = parse_u64(next_value(&raw, &mut index, "--chain-id")?)?
                }
                "--cpu" | "--cpu-only" => args.cpu_only = true,
                "--no-cpu-fallback" => args.allow_cpu_fallback = false,
                "--cpu-threads" => {
                    args.cpu_threads = parse_usize(next_value(&raw, &mut index, "--cpu-threads")?)?
                }
                "--device" | "--cuda-device" => {
                    args.cuda_device_index =
                        parse_usize(next_value(&raw, &mut index, "--device")?)?
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
                    args.cuda_max_blocks =
                        parse_u32(next_value(&raw, &mut index, "--max-blocks")?)?
                }
                "--no-early-exit" => args.cuda_early_exit = false,
                "--start" | "--start-nonce" => {
                    args.start_nonce =
                        Some(parse_u64(next_value(&raw, &mut index, "--start")?)?)
                }
                "--block-watch-secs" => {
                    args.block_watch_secs =
                        parse_u64(next_value(&raw, &mut index, "--block-watch-secs")?)?
                }
                "--post-submit-sleep-secs" => {
                    args.post_submit_sleep_secs =
                        parse_u64(next_value(&raw, &mut index, "--post-submit-sleep-secs")?)?
                }
                "--gas-limit" => {
                    args.gas_limit = parse_u64(next_value(&raw, &mut index, "--gas-limit")?)?
                }
                "--priority-tip-gwei" => {
                    let gwei = next_value(&raw, &mut index, "--priority-tip-gwei")?;
                    args.priority_tip_wei = gwei_to_wei(gwei)?;
                }
                "--priority-tip-wei" => {
                    args.priority_tip_wei = parse_h256hash_u256(next_value(
                        &raw,
                        &mut index,
                        "--priority-tip-wei",
                    )?)?
                }
                "--max-fee-wei" => {
                    args.max_fee_wei = Some(parse_h256hash_u256(next_value(
                        &raw,
                        &mut index,
                        "--max-fee-wei",
                    )?)?)
                }
                "--max-fee-headroom-gwei" => {
                    let gwei = next_value(&raw, &mut index, "--max-fee-headroom-gwei")?;
                    args.max_fee_headroom_wei = gwei_to_wei(gwei)?;
                }
                "--skip-used-proofs-check" => args.skip_used_proofs_check = true,
                "--dry-run-mint" => args.dry_run_mint = true,
                "--no-wait" => args.wait_for_receipt = false,
                "--once" => args.once = true,
                unknown => return Err(format!("unknown argument: {unknown}").into()),
            }
            index += 1;
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

fn gwei_to_wei(value: &str) -> Result<U256, Box<dyn Error>> {
    let trimmed = value.trim().replace('_', "");
    let parsed: f64 = trimmed
        .parse()
        .map_err(|_| format!("invalid gwei value: {value}"))?;
    if !parsed.is_finite() || parsed < 0.0 {
        return Err(format!("invalid gwei value: {value}").into());
    }
    let wei = (parsed * 1_000_000_000.0_f64).round();
    if wei > u128::MAX as f64 {
        return Err(format!("gwei value out of range: {value}").into());
    }
    Ok(U256::from(wei as u128))
}

fn resolve_rpc_url(args: &Args) -> Result<String, Box<dyn Error>> {
    args.rpc_url
        .clone()
        .or_else(|| env::var("HASH256_RPC_URL").ok())
        .or_else(|| env::var("ETH_RPC_URL").ok())
        .ok_or_else(|| "requires --rpc-url or HASH256_RPC_URL/ETH_RPC_URL".into())
}

fn resolve_private_key(args: &Args) -> Result<String, Box<dyn Error>> {
    if let Some(private_key) = &args.private_key {
        return Ok(private_key.clone());
    }
    if let Some(env_name) = &args.private_key_env {
        return env::var(env_name)
            .map_err(|_| format!("environment variable {env_name} is not set").into());
    }
    env::var("HASH256_PRIVATE_KEY")
        .or_else(|_| env::var("ETH_PRIVATE_KEY"))
        .map_err(|_| {
            "requires --private-key, --private-key-env, or HASH256_PRIVATE_KEY/ETH_PRIVATE_KEY"
                .into()
        })
}

fn print_mint_outcome(outcome: &H256HashMintOutcome) {
    println!(
        "mint_submit_mode={} from={} contract={} nonce={} proof_hash={}",
        if outcome.dry_run { "dry-run" } else { "send" },
        outcome.from_address,
        outcome.contract_address,
        outcome.nonce_hex,
        outcome.proof_hash_hex
    );
    if let Some(used) = outcome.proof_used_check {
        println!("mint_proof_used_check={used}");
    }
    println!("mint_gas_estimate={}", outcome.gas_estimate);
    println!("mint_max_fee_per_gas={}", outcome.max_fee_per_gas);
    println!(
        "mint_max_priority_fee_per_gas={}",
        outcome.max_priority_fee_per_gas
    );
    println!("mint_calldata={}", outcome.calldata_hex);
    if let Some(transaction_hash) = &outcome.transaction_hash {
        println!("mint_tx_hash={transaction_hash}");
        println!("mint_tx_explorer=https://etherscan.io/tx/{transaction_hash}");
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
  cargo run --release --bin h256hashmine -- --wallet 0x... --rpc-url $ETH_RPC --private-key 0x...

Default mode is a continuous loop. It will:
  1. read getChallenge(wallet) + miningState() from chain
  2. run the keccak-256 CUDA kernel (RTX 3090 full-load defaults)
  3. on hit, call usedProofs(hash) to avoid a wasted broadcast
  4. submit mine(uint256 nonce) via EIP-1559
  5. wait for receipt, then start the next round

Required:
  --wallet ADDRESS         the address that mints (challenge is keyed on it)
  --rpc-url URL            Ethereum JSON-RPC URL (or HASH256_RPC_URL/ETH_RPC_URL env)
  --private-key HEX        Ethereum private key (or HASH256_PRIVATE_KEY/ETH_PRIVATE_KEY env)

Mining options:
  --cpu                    disable CUDA and use CPU only
  --no-cpu-fallback        fail if CUDA is unavailable instead of falling back
  --cpu-threads N          CPU worker count when running on CPU
  --device N               CUDA device index, default 0
  --batch N                CUDA batch size, default 2^33 (RTX 3090 full load)
  --threads-per-block N    CUDA threads per block, default 128
  --nonces-per-thread N    CUDA nonces per thread: 1, 2, 4, or 8 (default 8)
  --max-blocks N           CUDA launch block cap; 0 means auto = SM × 12
  --no-early-exit          continue a kernel batch after one hit is found
  --start N                fixed starting nonce instead of random
  --block-watch-secs N     re-poll chain every N seconds for new challenge (default 12)
  --post-submit-sleep-secs N   delay between successful submits, default 12

Submit options:
  --contract ADDRESS       HASH256 contract, default {default_contract}
  --chain-id N             EVM chain ID, default 1 (Ethereum mainnet)
  --gas-limit N            gas limit override, default 300000
  --priority-tip-gwei F    EIP-1559 maxPriorityFeePerGas in gwei, default 1.0
  --priority-tip-wei N     same in wei (overrides --priority-tip-gwei)
  --max-fee-wei N          fixed cap on maxFeePerGas (default: base_fee + tip + 5 gwei)
  --max-fee-headroom-gwei F   gwei added on top of (base_fee + tip), default 5
  --skip-used-proofs-check pay gas even if usedProofs(hash) returns true (not recommended)
  --dry-run-mint           verify proof + estimate gas, do NOT broadcast
  --no-wait                broadcast and exit without waiting for receipt
  --once                   exit after the first solve+submit (no loop)
"#,
        default_contract = DEFAULT_HASH256_CONTRACT_ADDRESS
    );
}
