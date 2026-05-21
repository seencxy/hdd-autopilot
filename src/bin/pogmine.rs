//! mintpog.com POG CUDA miner.
//!
//! POG uses the same Keccak-256 proof shape as the existing `h256hash`
//! backend: `keccak256(challenge || uint256_be(nonce)) < difficulty`.
//! This binary only swaps in POG's contract reads and `mine(uint256)` submit.

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
use mining::pog_tx::{
    DEFAULT_POG_CHAIN_ID, DEFAULT_POG_CONTRACT_ADDRESS, DEFAULT_POG_GAS_LIMIT,
    DEFAULT_POG_MAX_FEE_HEADROOM_WEI, DEFAULT_POG_PRIORITY_TIP_WEI, DEFAULT_POG_RPC_URL,
    PogMiningRound, PogMintOutcome, PogMintRequest, parse_pog_u256, pog_address_from_private_key,
    read_pog_mining_round, read_pog_status, submit_pog_mine,
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

    let rpc_url = resolve_rpc_url(&args);
    let private_key = resolve_private_key(&args);
    if !args.no_submit && private_key.is_none() {
        return Err(
            "POG submit mode requires --private-key, --private-key-env, POG_PRIVATE_KEY, or ETH_PRIVATE_KEY"
                .into(),
        );
    }
    let signer_address = match private_key.as_deref() {
        Some(private_key) => Some(pog_address_from_private_key(private_key, args.chain_id)?),
        None => None,
    };
    let wallet_address = args
        .wallet
        .clone()
        .or_else(|| signer_address.clone())
        .ok_or("missing --wallet 0x... (or provide a private key to derive it)")?;
    validate_address(&wallet_address, "--wallet")?;
    if let Some(signer_address) = &signer_address {
        if !same_address(&wallet_address, signer_address) {
            return Err(format!(
                "--wallet ({wallet_address}) does not match private key signer ({signer_address}); POG challenge is keyed to msg.sender"
            )
            .into());
        }
    }

    println!(
        "pog miner starting · contract={} chain_id={} wallet={} rpc={} prefer_gpu={} submit={} dry_run={} once={}",
        args.contract_address,
        args.chain_id,
        wallet_address,
        rpc_url,
        !args.cpu_only,
        !args.no_submit,
        args.dry_run_mint,
        args.once
    );
    if !args.cpu_only {
        match mining::h256hash::cuda_availability_summary() {
            Ok(summary) => println!("cuda: {summary}"),
            Err(reason) => {
                eprintln!("WARN: GPU mining disabled: {reason}");
                if !args.allow_cpu_fallback {
                    return Err(
                        "CUDA is required but not available (use --cpu to force CPU mode)".into(),
                    );
                }
                eprintln!(
                    "WARN: Falling back to CPU mining (add --no-cpu-fallback to abort instead)."
                );
            }
        }
    }

    let block_watch_interval = Duration::from_secs(args.block_watch_secs.max(1));
    let mut first_round = true;
    let mut total_submits = 0u64;

    loop {
        let round = if first_round {
            first_round = false;
            match read_pog_status(&rpc_url, &args.contract_address, &wallet_address).await {
                Ok(status) => {
                    println!(
                        "block={} epoch={} blocks_left={} reward={} difficulty=0x{} total_mints={} total_mined={} remaining={} balance={}",
                        status.block_number,
                        status.epoch,
                        status.epoch_blocks_left,
                        status.reward,
                        u256_hex(&status.difficulty),
                        status.total_mints,
                        status.total_mined,
                        status.mining_remaining,
                        status.balance
                    );
                    PogMiningRound {
                        challenge: status.challenge,
                        difficulty: status.difficulty,
                        block_number: status.block_number,
                        epoch: status.epoch,
                        epoch_length: status.epoch_length,
                        epoch_blocks_left: status.epoch_blocks_left,
                    }
                }
                Err(error) => {
                    eprintln!("warn: failed to read POG status: {error}; retrying in 12s");
                    tokio::time::sleep(Duration::from_secs(12)).await;
                    first_round = true;
                    continue;
                }
            }
        } else {
            match read_pog_mining_round(&rpc_url, &args.contract_address, &wallet_address).await {
                Ok(round) => round,
                Err(error) => {
                    eprintln!("warn: failed to read POG round: {error}; retrying in 12s");
                    tokio::time::sleep(Duration::from_secs(12)).await;
                    continue;
                }
            }
        };

        if round.difficulty.is_zero() {
            eprintln!("warn: POG currentDifficulty is zero; sleeping 12s");
            tokio::time::sleep(Duration::from_secs(12)).await;
            continue;
        }
        if round.epoch_blocks_left <= args.min_blocks_left {
            println!(
                "epoch={} block={} blocks_left={} <= {}; waiting for next epoch",
                round.epoch, round.block_number, round.epoch_blocks_left, args.min_blocks_left
            );
            tokio::time::sleep(Duration::from_secs(args.post_submit_sleep_secs.max(1))).await;
            continue;
        }

        let target = u256_to_be32(&round.difficulty);
        println!(
            "round block={} epoch={} blocks_left={} challenge=0x{} difficulty=0x{}",
            round.block_number,
            round.epoch,
            round.epoch_blocks_left,
            hex_lower(&round.challenge),
            hex_lower(&target)
        );

        let start_nonce = args.start_nonce.unwrap_or_else(rand::random::<u64>);
        let mine_config = H256HashMineConfig {
            start_nonce,
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
        let job = H256HashJob {
            challenge: round.challenge,
            target,
        };

        let cancel = Arc::new(AtomicBool::new(false));
        let watcher_handle = spawn_chain_watcher(
            cancel.clone(),
            rpc_url.clone(),
            args.contract_address.clone(),
            wallet_address.clone(),
            round.challenge,
            round.difficulty,
            block_watch_interval,
            args.min_blocks_left,
        );
        let mine_cancel = cancel.clone();
        let mine_job = job.clone();
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
                    println!("info: mining cancelled by chain watcher");
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
            eprintln!("warn: solution came from CPU fallback; check CUDA availability");
        }
        if args.no_submit {
            println!("submit disabled; mine_arg={}", result.nonce);
            break;
        }

        let request = PogMintRequest {
            rpc_url: rpc_url.clone(),
            private_key: private_key
                .clone()
                .ok_or("private key unexpectedly missing for submit")?,
            contract_address: args.contract_address.clone(),
            chain_id: args.chain_id,
            nonce: result.nonce,
            gas_limit: args.gas_limit,
            priority_tip_wei: args.priority_tip_wei,
            max_fee_wei: args.max_fee_wei,
            max_fee_headroom_wei: args.max_fee_headroom_wei,
            check_used_solutions: !args.skip_used_solutions_check,
            dry_run: args.dry_run_mint,
            wait_for_receipt: args.wait_for_receipt,
        };
        match submit_pog_mine(request).await {
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
                eprintln!("error: POG mine() submit failed: {error}; backing off 5s");
                tokio::time::sleep(Duration::from_secs(5)).await;
            }
        }

        if args.once {
            println!("info: --once specified, exiting after one solution");
            break;
        }
        tokio::time::sleep(Duration::from_secs(args.post_submit_sleep_secs)).await;
    }

    println!("pog miner stopped · total_submits={total_submits}");
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
    min_blocks_left: u64,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(interval).await;
            if cancel.load(Ordering::Relaxed) {
                return;
            }
            match read_pog_mining_round(&rpc_url, &contract_address, &wallet_address).await {
                Ok(round) => {
                    if round.challenge != baseline_challenge
                        || round.difficulty != baseline_difficulty
                        || round.epoch_blocks_left <= min_blocks_left
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
    nonce_count: u64,
    block_watch_secs: u64,
    post_submit_sleep_secs: u64,
    min_blocks_left: u64,
    gas_limit: Option<u64>,
    priority_tip_wei: U256,
    max_fee_wei: Option<U256>,
    max_fee_headroom_wei: U256,
    skip_used_solutions_check: bool,
    no_submit: bool,
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
            contract_address: DEFAULT_POG_CONTRACT_ADDRESS.to_string(),
            chain_id: DEFAULT_POG_CHAIN_ID,
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
            nonce_count: u64::MAX,
            block_watch_secs: 12,
            post_submit_sleep_secs: 3,
            min_blocks_left: 2,
            gas_limit: Some(DEFAULT_POG_GAS_LIMIT),
            priority_tip_wei: U256::from(DEFAULT_POG_PRIORITY_TIP_WEI),
            max_fee_wei: None,
            max_fee_headroom_wei: U256::from(DEFAULT_POG_MAX_FEE_HEADROOM_WEI),
            skip_used_solutions_check: false,
            no_submit: false,
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
                    args.contract_address = next_value(&raw, &mut index, "--contract")?.to_string()
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
                "--start" | "--start-nonce" => {
                    args.start_nonce = Some(parse_u64(next_value(&raw, &mut index, "--start")?)?)
                }
                "--count" | "--nonce-count" => {
                    args.nonce_count = parse_u64(next_value(&raw, &mut index, "--count")?)?
                }
                "--block-watch-secs" => {
                    args.block_watch_secs =
                        parse_u64(next_value(&raw, &mut index, "--block-watch-secs")?)?
                }
                "--post-submit-sleep-secs" => {
                    args.post_submit_sleep_secs =
                        parse_u64(next_value(&raw, &mut index, "--post-submit-sleep-secs")?)?
                }
                "--min-blocks-left" => {
                    args.min_blocks_left =
                        parse_u64(next_value(&raw, &mut index, "--min-blocks-left")?)?
                }
                "--gas-limit" => {
                    let value = next_value(&raw, &mut index, "--gas-limit")?;
                    args.gas_limit = if value.eq_ignore_ascii_case("auto") {
                        None
                    } else {
                        Some(parse_u64(value)?)
                    };
                }
                "--priority-tip-gwei" => {
                    let gwei = next_value(&raw, &mut index, "--priority-tip-gwei")?;
                    args.priority_tip_wei = gwei_to_wei(gwei)?;
                }
                "--priority-tip-wei" => {
                    args.priority_tip_wei =
                        parse_pog_u256(next_value(&raw, &mut index, "--priority-tip-wei")?)?
                }
                "--max-fee-wei" => {
                    args.max_fee_wei = Some(parse_pog_u256(next_value(
                        &raw,
                        &mut index,
                        "--max-fee-wei",
                    )?)?)
                }
                "--max-fee-headroom-gwei" => {
                    let gwei = next_value(&raw, &mut index, "--max-fee-headroom-gwei")?;
                    args.max_fee_headroom_wei = gwei_to_wei(gwei)?;
                }
                "--skip-used-solutions-check" => args.skip_used_solutions_check = true,
                "--no-submit" => args.no_submit = true,
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

fn resolve_rpc_url(args: &Args) -> String {
    args.rpc_url
        .clone()
        .or_else(|| env::var("POG_RPC_URL").ok())
        .or_else(|| env::var("ETH_RPC_URL").ok())
        .unwrap_or_else(|| DEFAULT_POG_RPC_URL.to_string())
}

fn resolve_private_key(args: &Args) -> Option<String> {
    if let Some(private_key) = &args.private_key {
        return Some(private_key.clone());
    }
    if let Some(env_name) = &args.private_key_env {
        return env::var(env_name).ok();
    }
    env::var("POG_PRIVATE_KEY")
        .or_else(|_| env::var("ETH_PRIVATE_KEY"))
        .ok()
}

fn validate_address(value: &str, label: &str) -> Result<(), Box<dyn Error>> {
    if !value.starts_with("0x") || value.len() != 42 {
        return Err(format!("{label} must be a 0x-prefixed 20-byte hex address").into());
    }
    if !value[2..].chars().all(|char| char.is_ascii_hexdigit()) {
        return Err(format!("{label} contains non-hex characters").into());
    }
    Ok(())
}

fn same_address(left: &str, right: &str) -> bool {
    left.eq_ignore_ascii_case(right)
}

fn u256_to_be32(value: &U256) -> [u8; 32] {
    let mut out = [0u8; 32];
    value.to_big_endian(&mut out);
    out
}

fn u256_hex(value: &U256) -> String {
    hex_lower(&u256_to_be32(value))
}

fn print_mint_outcome(outcome: &PogMintOutcome) {
    println!(
        "mint_submit_mode={} from={} contract={} nonce={} proof_hash={}",
        if outcome.dry_run { "dry-run" } else { "send" },
        outcome.from_address,
        outcome.contract_address,
        outcome.nonce_hex,
        outcome.proof_hash_hex
    );
    if let Some(used) = outcome.used_solution_check {
        println!("mint_used_solution_check={used}");
    }
    println!("mint_gas_estimate={}", outcome.gas_estimate);
    println!("mint_gas_limit={}", outcome.gas_limit);
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
  cargo run --release --bin pogmine -- --private-key 0x... [options]
  cargo run --release --bin pogmine -- --wallet 0x... --no-submit --once [options]

Default mode is a continuous solve+submit loop for mintpog.com POG.

Required:
  --private-key HEX        Ethereum private key; or POG_PRIVATE_KEY/ETH_PRIVATE_KEY
  --wallet ADDRESS         optional; derived from private key when omitted

Mining options:
  --cpu                    disable CUDA and use CPU only
  --no-cpu-fallback        fail if CUDA is unavailable instead of falling back
  --cpu-threads N          CPU worker count when running on CPU
  --device N               CUDA device index, default 0
  --batch N                CUDA batch size, default 2^33
  --threads-per-block N    CUDA threads per block, default 128
  --nonces-per-thread N    CUDA nonces per thread: 1, 2, 4, or 8 (default 1)
  --max-blocks N           CUDA launch block cap; 0 means auto
  --no-early-exit          continue a kernel batch after one hit is found
  --start N                fixed starting nonce instead of random
  --count N|max            nonce count to scan, default max
  --block-watch-secs N     re-poll chain every N seconds, default 12
  --min-blocks-left N      cancel/skip when epoch has <= N blocks left, default 2

Submit options:
  --no-submit              only print the first solution; does not broadcast
  --dry-run-mint           verify + estimate gas, do NOT broadcast
  --rpc-url URL            Ethereum JSON-RPC URL; or POG_RPC_URL/ETH_RPC_URL
  --contract ADDRESS       POG contract, default {default_contract}
  --chain-id N             EVM chain ID, default 1
  --gas-limit N|auto       default 300000; auto uses estimate * 1.2
  --priority-tip-gwei F    EIP-1559 maxPriorityFeePerGas in gwei, default 1.0
  --priority-tip-wei N     same in wei
  --max-fee-wei N          fixed cap on maxFeePerGas (default: base_fee + tip + 5 gwei)
  --max-fee-headroom-gwei F   gwei added on top of (base_fee + tip), default 5
  --skip-used-solutions-check  skip usedSolutions(hash) preflight
  --no-wait                broadcast and continue without waiting for receipt
  --once                   exit after the first solve+submit
"#,
        default_contract = DEFAULT_POG_CONTRACT_ADDRESS
    );
}
