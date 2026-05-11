//! Standalone hash256.org `mine(uint256)` submit CLI.
//!
//! Useful when you already have a winning nonce (for example because the
//! miner was killed mid-flight) and just want to broadcast or dry-run the
//! transaction without re-running the GPU search.

use std::env;
use std::error::Error;

use crossterm as _;
use ethers_core::types::U256;
use hdd_autopilot as _;
use mining::h256hash::hex_lower;
use mining::h256hash_tx::{
    DEFAULT_HASH256_CHAIN_ID, DEFAULT_HASH256_CONTRACT_ADDRESS, DEFAULT_HASH256_GAS_LIMIT,
    DEFAULT_HASH256_MAX_FEE_HEADROOM_WEI, DEFAULT_HASH256_PRIORITY_TIP_WEI, H256HashMintOutcome,
    H256HashMintRequest, parse_h256hash_u256, submit_h256hash_mine,
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
    let nonce = args
        .nonce
        .ok_or("missing --nonce <u64> (or 0x-prefixed hex of low 64 bits)")?;
    let request = H256HashMintRequest {
        rpc_url: resolve_rpc_url(&args)?,
        private_key: resolve_private_key(&args)?,
        contract_address: args.contract_address,
        chain_id: args.chain_id,
        nonce,
        gas_limit: Some(args.gas_limit),
        priority_tip_wei: args.priority_tip_wei,
        max_fee_wei: args.max_fee_wei,
        max_fee_headroom_wei: args.max_fee_headroom_wei,
        check_used_proofs: !args.skip_used_proofs_check,
        dry_run: !args.send,
        wait_for_receipt: args.wait_for_receipt,
    };

    let mut nonce_be32 = [0u8; 32];
    nonce_be32[24..32].copy_from_slice(&nonce.to_be_bytes());
    println!(
        "mint_submit_mode={} nonce=0x{} contract={} chain_id={}",
        if request.dry_run { "dry-run" } else { "send" },
        hex_lower(&nonce_be32),
        request.contract_address,
        request.chain_id
    );

    let outcome = submit_h256hash_mine(request).await?;
    print_mint_outcome(&outcome);
    Ok(())
}

#[derive(Debug, Clone)]
struct Args {
    help: bool,
    nonce: Option<u64>,
    send: bool,
    rpc_url: Option<String>,
    private_key: Option<String>,
    private_key_env: Option<String>,
    contract_address: String,
    chain_id: u64,
    gas_limit: u64,
    priority_tip_wei: U256,
    max_fee_wei: Option<U256>,
    max_fee_headroom_wei: U256,
    skip_used_proofs_check: bool,
    wait_for_receipt: bool,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            help: false,
            nonce: None,
            send: false,
            rpc_url: None,
            private_key: None,
            private_key_env: None,
            contract_address: DEFAULT_HASH256_CONTRACT_ADDRESS.to_string(),
            chain_id: DEFAULT_HASH256_CHAIN_ID,
            gas_limit: DEFAULT_HASH256_GAS_LIMIT,
            priority_tip_wei: U256::from(DEFAULT_HASH256_PRIORITY_TIP_WEI),
            max_fee_wei: None,
            max_fee_headroom_wei: U256::from(DEFAULT_HASH256_MAX_FEE_HEADROOM_WEI),
            skip_used_proofs_check: false,
            wait_for_receipt: true,
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
                "--nonce" => {
                    args.nonce = Some(parse_u64(next_value(&raw, &mut index, "--nonce")?)?)
                }
                "--send" => args.send = true,
                "--dry-run" => args.send = false,
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
                "--no-wait" => args.wait_for_receipt = false,
                "--wait" | "--wait-receipt" => args.wait_for_receipt = true,
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
    if let Some(hex) = normalized.strip_prefix("0x") {
        return Ok(u64::from_str_radix(hex, 16)?);
    }
    Ok(normalized.parse()?)
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
    println!("mint_from={}", outcome.from_address);
    println!("mint_contract={}", outcome.contract_address);
    println!("mint_nonce={}", outcome.nonce_hex);
    println!("mint_proof_hash={}", outcome.proof_hash_hex);
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
  cargo run --release --bin h256hashsend -- --nonce 0x<u64 hex> [--send] [options]

Required:
  --nonce N|0xN          Low 64 bits of the uint256 nonce (high 24 bytes are zero)

Submit options:
  --send                 broadcast the transaction; default is dry-run
  --dry-run              verify + estimate only (default)
  --rpc-url URL          Ethereum JSON-RPC URL; or HASH256_RPC_URL/ETH_RPC_URL
  --private-key HEX      Ethereum private key; or HASH256_PRIVATE_KEY/ETH_PRIVATE_KEY
  --private-key-env NAME read private key from a custom env var
  --contract ADDRESS     HASH256 contract, default {default_contract}
  --chain-id N           EVM chain ID, default 1
  --gas-limit N          gas limit override, default 300000
  --priority-tip-gwei F  EIP-1559 maxPriorityFeePerGas in gwei, default 1.0
  --priority-tip-wei N   same in wei
  --max-fee-wei N        fixed cap on maxFeePerGas (default: base_fee + tip + headroom)
  --max-fee-headroom-gwei F   gwei added on top of (base_fee + tip), default 5
  --skip-used-proofs-check  pay gas even if usedProofs(hash) returns true
  --no-wait              broadcast and exit without waiting for receipt
"#,
        default_contract = DEFAULT_HASH256_CONTRACT_ADDRESS
    );
}
