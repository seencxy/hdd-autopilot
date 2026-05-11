use std::env;
use std::error::Error;

use crossterm as _;
use ethers_core::types::U256;
use hdd_autopilot as _;
use mining::h98hash::{decode_fixed_hex, hex_lower};
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
    let nonce = args
        .nonce
        .ok_or("missing --nonce 0x... bytes16 from h98hashmine mint_arg")?;
    let request = H98HashMintRequest {
        rpc_url: resolve_rpc_url(&args)?,
        private_key: resolve_private_key(&args)?,
        contract_address: args.contract_address,
        chain_id: args.chain_id,
        nonce,
        value_wei: args.mint_value_wei,
        gas_limit: args.gas_limit,
        gas_price_wei: args.gas_price_wei,
        verify_before_send: args.verify_before_send,
        dry_run: !args.send,
        wait_for_receipt: args.wait_for_receipt,
    };

    println!(
        "mint_submit_mode={} nonce=0x{} contract={} chain_id={}",
        if request.dry_run { "dry-run" } else { "send" },
        hex_lower(&request.nonce),
        request.contract_address,
        request.chain_id
    );
    let outcome = submit_h98hash_mint(request).await?;
    print_mint_outcome(&outcome);
    Ok(())
}

#[derive(Debug, Clone)]
struct Args {
    help: bool,
    nonce: Option<[u8; 16]>,
    send: bool,
    rpc_url: Option<String>,
    private_key: Option<String>,
    private_key_env: Option<String>,
    contract_address: String,
    chain_id: u64,
    mint_value_wei: H98HashMintValue,
    gas_limit: Option<u64>,
    gas_price_wei: Option<U256>,
    verify_before_send: bool,
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
                "--nonce" | "--mint-arg" => {
                    args.nonce = Some(decode_fixed_hex::<16>(
                        next_value(&raw, &mut index, "--nonce")?,
                        "HASH98 mint nonce",
                    )?)
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

fn resolve_rpc_url(args: &Args) -> Result<String, Box<dyn Error>> {
    args.rpc_url
        .clone()
        .or_else(|| env::var("H98HASH_RPC_URL").ok())
        .or_else(|| env::var("ETH_RPC_URL").ok())
        .ok_or_else(|| "requires --rpc-url or H98HASH_RPC_URL/ETH_RPC_URL".into())
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
            "requires --private-key, --private-key-env, or H98HASH_PRIVATE_KEY/ETH_PRIVATE_KEY"
                .into()
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
  cargo run --bin h98hashsend -- --nonce 0x<16-byte nonce> [--send] [options]

Required:
  --nonce HEX              bytes16 nonce from h98hashmine mint_arg

Options:
  --send                   broadcast the transaction; default is dry-run
  --dry-run                verify and estimate only
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
"#
    );
}
