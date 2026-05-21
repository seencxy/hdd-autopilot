//! On-chain read and submit path for mintpog.com POG mining.
//!
//! The public `/mine` page derives each round as:
//! `challenge = keccak256(abi.encode(chainId, contract, miner, epoch))`,
//! where `epoch = blockNumber / 100`, then searches
//! `keccak256(challenge || uint256_be(nonce)) < currentDifficulty()`.
//! The contract also exposes `challengeFor(address)`, so runtime code uses the
//! contract as the source of truth and keeps the local derivation for tests and
//! documentation.

use std::fmt;
use std::str::FromStr;
use std::sync::Arc;

use ethers_core::abi::{Function, Param, ParamType, StateMutability, Token, encode};
use ethers_core::types::transaction::eip2718::TypedTransaction;
use ethers_core::types::{
    Address, BlockId, BlockNumber, Bytes, Eip1559TransactionRequest, TransactionReceipt, U256,
};
use ethers_core::utils::keccak256;
use ethers_middleware::SignerMiddleware;
use ethers_providers::{Http, Middleware, Provider};
use ethers_signers::{LocalWallet, Signer};

use crate::error::MiningError;
use crate::h256hash::{h256hash_input, hex_lower};

pub const DEFAULT_POG_CONTRACT_ADDRESS: &str = "0x214748fC525C1b001e5d4EeB16A3F0b7eaB042B3";
pub const DEFAULT_POG_CHAIN_ID: u64 = 1;
pub const DEFAULT_POG_RPC_URL: &str = "https://eth-mainnet.g.alchemy.com/v2/ZB5t3a2OxubjbmrVK3tA5";
pub const DEFAULT_POG_EPOCH_LENGTH: u64 = 100;
pub const DEFAULT_POG_GAS_LIMIT: u64 = 300_000;
pub const DEFAULT_POG_PRIORITY_TIP_WEI: u64 = 1_000_000_000;
pub const DEFAULT_POG_LOW_PRIORITY_TIP_FLOOR_WEI: u64 = 100_000_000;
pub const DEFAULT_POG_MAX_FEE_HEADROOM_WEI: u64 = 5_000_000_000;

#[derive(Clone)]
pub struct PogMintRequest {
    pub rpc_url: String,
    pub private_key: String,
    pub contract_address: String,
    pub chain_id: u64,
    pub nonce: u64,
    pub gas_limit: Option<u64>,
    pub priority_tip: PogPriorityTip,
    pub max_fee_wei: Option<U256>,
    pub max_fee_headroom_wei: U256,
    pub check_used_solutions: bool,
    pub dry_run: bool,
    pub wait_for_receipt: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PogPriorityTip {
    Fixed(U256),
    Low,
    Auto,
}

impl fmt::Debug for PogMintRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PogMintRequest")
            .field("rpc_url", &self.rpc_url)
            .field("private_key", &"<redacted>")
            .field("contract_address", &self.contract_address)
            .field("chain_id", &self.chain_id)
            .field("nonce", &format_args!("0x{:016x}", self.nonce))
            .field("gas_limit", &self.gas_limit)
            .field("priority_tip", &self.priority_tip)
            .field("max_fee_wei", &self.max_fee_wei)
            .field("max_fee_headroom_wei", &self.max_fee_headroom_wei)
            .field("check_used_solutions", &self.check_used_solutions)
            .field("dry_run", &self.dry_run)
            .field("wait_for_receipt", &self.wait_for_receipt)
            .finish()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PogChainStatus {
    pub account_address: String,
    pub contract_address: String,
    pub block_number: u64,
    pub epoch: U256,
    pub epoch_length: u64,
    pub epoch_blocks_left: u64,
    pub challenge: [u8; 32],
    pub difficulty: U256,
    pub reward: U256,
    pub balance: U256,
    pub total_mined: U256,
    pub mining_remaining: U256,
    pub total_mints: U256,
    pub mints_this_block: U256,
    pub mints_per_block_cap: U256,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PogMiningRound {
    pub challenge: [u8; 32],
    pub difficulty: U256,
    pub block_number: u64,
    pub epoch: U256,
    pub epoch_length: u64,
    pub epoch_blocks_left: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PogMintOutcome {
    pub dry_run: bool,
    pub from_address: String,
    pub contract_address: String,
    pub nonce: u64,
    pub nonce_hex: String,
    pub proof_hash_hex: String,
    pub calldata_hex: String,
    pub used_solution_check: Option<bool>,
    pub gas_estimate: String,
    pub gas_limit: String,
    pub max_fee_per_gas: String,
    pub max_priority_fee_per_gas: String,
    pub transaction_hash: Option<String>,
    pub receipt_status: Option<u64>,
    pub receipt_block_number: Option<u64>,
    pub receipt_gas_used: Option<String>,
}

pub fn pog_address_from_private_key(
    private_key: &str,
    chain_id: u64,
) -> Result<String, MiningError> {
    let wallet = private_key
        .parse::<LocalWallet>()
        .map_err(|error| message(format!("invalid POG private key: {error}")))?
        .with_chain_id(chain_id);
    Ok(format_address(wallet.address()))
}

pub fn pog_mine_calldata(nonce: u64) -> Result<Vec<u8>, MiningError> {
    mine_function()
        .encode_input(&[Token::Uint(U256::from(nonce))])
        .map_err(|error| message(format!("failed to ABI-encode POG mine: {error}")))
}

pub fn pog_nonce_to_be32(nonce: u64) -> [u8; 32] {
    let mut out = [0u8; 32];
    out[24..32].copy_from_slice(&nonce.to_be_bytes());
    out
}

pub fn parse_pog_u256(value: &str) -> Result<U256, MiningError> {
    let normalized = value.trim().replace('_', "");
    if normalized.is_empty() {
        return Err(message("empty uint256 value"));
    }
    if let Some(hex) = normalized
        .strip_prefix("0x")
        .or_else(|| normalized.strip_prefix("0X"))
    {
        return U256::from_str_radix(hex, 16)
            .map_err(|error| message(format!("invalid hex uint256 value: {error}")));
    }
    U256::from_dec_str(&normalized)
        .map_err(|error| message(format!("invalid decimal uint256 value: {error}")))
}

pub fn derive_pog_challenge(
    chain_id: u64,
    contract_address: &str,
    miner_address: &str,
    epoch: U256,
) -> Result<[u8; 32], MiningError> {
    let contract = parse_address(contract_address)?;
    let miner = parse_address(miner_address)?;
    Ok(keccak256(encode(&[
        Token::Uint(U256::from(chain_id)),
        Token::Address(contract),
        Token::Address(miner),
        Token::Uint(epoch),
    ])))
}

pub async fn read_pog_status(
    rpc_url: &str,
    contract_address: &str,
    account_address: &str,
) -> Result<PogChainStatus, MiningError> {
    let contract_address = parse_address(contract_address)?;
    let account_address = parse_address(account_address)?;
    let provider = Provider::<Http>::try_from(rpc_url)
        .map_err(|error| message(format!("invalid POG RPC URL: {error}")))?;
    let block_number = provider
        .get_block_number()
        .await
        .map_err(|error| message(format!("POG blockNumber RPC call failed: {error}")))?
        .as_u64();
    let epoch_length = read_uint_function_with_client(&provider, contract_address, "EPOCH_LENGTH")
        .await
        .ok()
        .and_then(|value| u256_to_u64(value).ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_POG_EPOCH_LENGTH);
    let challenge =
        read_pog_challenge_with_client(&provider, contract_address, account_address).await?;
    let difficulty =
        read_uint_function_with_client(&provider, contract_address, "currentDifficulty").await?;
    let reward =
        read_uint_function_with_client(&provider, contract_address, "currentReward").await?;
    let balance = read_balance_with_client(&provider, contract_address, account_address)
        .await
        .unwrap_or(U256::zero());
    let total_mined = read_uint_function_with_client(&provider, contract_address, "totalMined")
        .await
        .unwrap_or(U256::zero());
    let mining_remaining =
        read_uint_function_with_client(&provider, contract_address, "miningRemaining")
            .await
            .unwrap_or(U256::zero());
    let total_mints = read_uint_function_with_client(&provider, contract_address, "totalMints")
        .await
        .unwrap_or(U256::zero());
    let mints_this_block =
        read_uint_function_with_client(&provider, contract_address, "mintsThisBlock")
            .await
            .unwrap_or(U256::zero());
    let mints_per_block_cap =
        read_uint_function_with_client(&provider, contract_address, "MINTS_PER_BLOCK_CAP")
            .await
            .unwrap_or(U256::zero());
    let epoch = U256::from(block_number / epoch_length);
    Ok(PogChainStatus {
        account_address: format_address(account_address),
        contract_address: format_address(contract_address),
        block_number,
        epoch,
        epoch_length,
        epoch_blocks_left: blocks_left_in_epoch(block_number, epoch_length),
        challenge,
        difficulty,
        reward,
        balance,
        total_mined,
        mining_remaining,
        total_mints,
        mints_this_block,
        mints_per_block_cap,
    })
}

pub async fn read_pog_mining_round(
    rpc_url: &str,
    contract_address: &str,
    account_address: &str,
) -> Result<PogMiningRound, MiningError> {
    let contract_address = parse_address(contract_address)?;
    let account_address = parse_address(account_address)?;
    let provider = Provider::<Http>::try_from(rpc_url)
        .map_err(|error| message(format!("invalid POG RPC URL: {error}")))?;
    let block_number = provider
        .get_block_number()
        .await
        .map_err(|error| message(format!("POG blockNumber RPC call failed: {error}")))?
        .as_u64();
    let epoch_length = read_uint_function_with_client(&provider, contract_address, "EPOCH_LENGTH")
        .await
        .ok()
        .and_then(|value| u256_to_u64(value).ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_POG_EPOCH_LENGTH);
    let challenge =
        read_pog_challenge_with_client(&provider, contract_address, account_address).await?;
    let difficulty =
        read_uint_function_with_client(&provider, contract_address, "currentDifficulty").await?;
    Ok(PogMiningRound {
        challenge,
        difficulty,
        block_number,
        epoch: U256::from(block_number / epoch_length),
        epoch_length,
        epoch_blocks_left: blocks_left_in_epoch(block_number, epoch_length),
    })
}

pub async fn submit_pog_mine(request: PogMintRequest) -> Result<PogMintOutcome, MiningError> {
    let contract_address = parse_address(&request.contract_address)?;
    let provider = Provider::<Http>::try_from(request.rpc_url.as_str())
        .map_err(|error| message(format!("invalid POG RPC URL: {error}")))?;
    let wallet = request
        .private_key
        .parse::<LocalWallet>()
        .map_err(|error| message(format!("invalid POG private key: {error}")))?
        .with_chain_id(request.chain_id);
    let from_address = wallet.address();
    let client = Arc::new(SignerMiddleware::new(provider, wallet));

    let challenge =
        read_pog_challenge_with_client(client.as_ref(), contract_address, from_address).await?;
    let proof_hash = keccak256(h256hash_input(&challenge, request.nonce));
    let used_solution_check = if request.check_used_solutions {
        let already_used =
            check_used_solution_with_client(client.as_ref(), contract_address, &proof_hash).await?;
        if already_used {
            return Err(message(
                "POG usedSolutions reports this digest is already consumed; refusing to submit",
            ));
        }
        Some(false)
    } else {
        None
    };

    let priority_tip = resolve_priority_tip(client.as_ref(), &request.priority_tip).await?;
    let max_fee = match request.max_fee_wei {
        Some(value) => value,
        None => {
            let block = client
                .get_block(BlockNumber::Latest)
                .await
                .map_err(|error| message(format!("POG latest block fetch failed: {error}")))?
                .ok_or_else(|| message("POG RPC returned no latest block"))?;
            let base_fee = block.base_fee_per_gas.unwrap_or(U256::zero());
            base_fee
                .saturating_add(priority_tip)
                .saturating_add(request.max_fee_headroom_wei)
        }
    };

    let nonce_be32 = pog_nonce_to_be32(request.nonce);
    let calldata = pog_mine_calldata(request.nonce)?;
    let mut transaction = Eip1559TransactionRequest::new()
        .to(contract_address)
        .from(from_address)
        .data(Bytes::from(calldata.clone()))
        .chain_id(request.chain_id)
        .max_priority_fee_per_gas(priority_tip)
        .max_fee_per_gas(max_fee);
    if let Some(gas_limit) = request.gas_limit {
        transaction = transaction.gas(U256::from(gas_limit));
    }
    let typed_estimate_transaction: TypedTransaction = transaction.clone().into();
    let gas_estimate = client
        .estimate_gas(&typed_estimate_transaction, None)
        .await
        .map_err(|error| message(format!("POG gas estimate failed: {error}")))?;
    let gas_limit = request
        .gas_limit
        .map(U256::from)
        .unwrap_or_else(|| gas_estimate.saturating_mul(U256::from(12u64)) / U256::from(10u64));
    transaction = transaction.gas(gas_limit);

    let mut outcome = PogMintOutcome {
        dry_run: request.dry_run,
        from_address: format_address(from_address),
        contract_address: format_address(contract_address),
        nonce: request.nonce,
        nonce_hex: format!("0x{}", hex_lower(&nonce_be32)),
        proof_hash_hex: format!("0x{}", hex_lower(&proof_hash)),
        calldata_hex: format!("0x{}", hex_lower(&calldata)),
        used_solution_check,
        gas_estimate: gas_estimate.to_string(),
        gas_limit: gas_limit.to_string(),
        max_fee_per_gas: max_fee.to_string(),
        max_priority_fee_per_gas: priority_tip.to_string(),
        transaction_hash: None,
        receipt_status: None,
        receipt_block_number: None,
        receipt_gas_used: None,
    };

    if request.dry_run {
        return Ok(outcome);
    }

    let typed_transaction: TypedTransaction = transaction.into();
    let pending_transaction = client
        .send_transaction(typed_transaction, None)
        .await
        .map_err(|error| message(format!("POG mine() transaction send failed: {error}")))?;
    let transaction_hash = pending_transaction.tx_hash();
    outcome.transaction_hash = Some(format!("{transaction_hash:#x}"));
    if request.wait_for_receipt {
        if let Some(receipt) = pending_transaction
            .await
            .map_err(|error| message(format!("POG mine() receipt wait failed: {error}")))?
        {
            fill_receipt_fields(&mut outcome, &receipt);
        }
    }
    Ok(outcome)
}

async fn resolve_priority_tip<M>(
    client: &M,
    priority_tip: &PogPriorityTip,
) -> Result<U256, MiningError>
where
    M: Middleware,
{
    match priority_tip {
        PogPriorityTip::Fixed(value) => Ok(*value),
        PogPriorityTip::Low => resolve_low_priority_tip(client).await,
        PogPriorityTip::Auto => {
            let (_, tip) = client
                .estimate_eip1559_fees(None)
                .await
                .map_err(|error| message(format!("POG EIP-1559 fee estimate failed: {error}")))?;
            Ok(tip)
        }
    }
}

async fn resolve_low_priority_tip<M>(client: &M) -> Result<U256, MiningError>
where
    M: Middleware,
{
    let floor = U256::from(DEFAULT_POG_LOW_PRIORITY_TIP_FLOOR_WEI);
    let fee_history = match client.fee_history(8u64, BlockNumber::Latest, &[10.0]).await {
        Ok(fee_history) => fee_history,
        Err(_) => return Ok(floor),
    };
    let mut rewards = fee_history
        .reward
        .iter()
        .filter_map(|block| block.first().copied())
        .filter(|value| !value.is_zero())
        .collect::<Vec<_>>();
    rewards.sort_unstable();
    let tip = rewards.first().copied().unwrap_or(floor);
    if tip < floor { Ok(floor) } else { Ok(tip) }
}

fn blocks_left_in_epoch(block_number: u64, epoch_length: u64) -> u64 {
    epoch_length.saturating_sub(block_number % epoch_length)
}

fn u256_to_u64(value: U256) -> Result<u64, MiningError> {
    if value > U256::from(u64::MAX) {
        return Err(message("uint256 value exceeds u64"));
    }
    Ok(value.as_u64())
}

async fn read_pog_challenge_with_client<M>(
    client: &M,
    contract_address: Address,
    account_address: Address,
) -> Result<[u8; 32], MiningError>
where
    M: Middleware,
{
    let function = challenge_for_function();
    let tokens = call_contract_function(
        client,
        contract_address,
        &function,
        &[Token::Address(account_address)],
    )
    .await?;
    match tokens.as_slice() {
        [Token::FixedBytes(bytes)] if bytes.len() == 32 => {
            let mut out = [0u8; 32];
            out.copy_from_slice(bytes);
            Ok(out)
        }
        _ => Err(message("POG challengeFor returned unexpected ABI output")),
    }
}

async fn read_balance_with_client<M>(
    client: &M,
    contract_address: Address,
    account_address: Address,
) -> Result<U256, MiningError>
where
    M: Middleware,
{
    let function = uint_address_function("balanceOf");
    let tokens = call_contract_function(
        client,
        contract_address,
        &function,
        &[Token::Address(account_address)],
    )
    .await?;
    match tokens.as_slice() {
        [Token::Uint(value)] => Ok(*value),
        _ => Err(message("POG balanceOf returned unexpected ABI output")),
    }
}

async fn read_uint_function_with_client<M>(
    client: &M,
    contract_address: Address,
    name: &str,
) -> Result<U256, MiningError>
where
    M: Middleware,
{
    let function = uint_function(name);
    let tokens = call_contract_function(client, contract_address, &function, &[]).await?;
    match tokens.as_slice() {
        [Token::Uint(value)] => Ok(*value),
        _ => Err(message(format!(
            "POG {name} returned unexpected ABI output"
        ))),
    }
}

async fn check_used_solution_with_client<M>(
    client: &M,
    contract_address: Address,
    digest: &[u8; 32],
) -> Result<bool, MiningError>
where
    M: Middleware,
{
    let function = used_solutions_function();
    let tokens = call_contract_function(
        client,
        contract_address,
        &function,
        &[Token::FixedBytes(digest.to_vec())],
    )
    .await?;
    match tokens.as_slice() {
        [Token::Bool(value)] => Ok(*value),
        _ => Err(message("POG usedSolutions returned unexpected ABI output")),
    }
}

async fn call_contract_function<M>(
    client: &M,
    contract_address: Address,
    function: &Function,
    inputs: &[Token],
) -> Result<Vec<Token>, MiningError>
where
    M: Middleware,
{
    let data = function.encode_input(inputs).map_err(|error| {
        message(format!(
            "failed to ABI-encode POG {} call: {error}",
            function.name
        ))
    })?;
    let transaction: TypedTransaction = Eip1559TransactionRequest::new()
        .to(contract_address)
        .data(Bytes::from(data))
        .into();
    let block = Some(BlockId::Number(BlockNumber::Latest));
    let output = client
        .call(&transaction, block)
        .await
        .map_err(|error| message(format!("POG {} RPC call failed: {error}", function.name)))?;
    function.decode_output(output.as_ref()).map_err(|error| {
        message(format!(
            "failed to ABI-decode POG {} output: {error}",
            function.name
        ))
    })
}

fn fill_receipt_fields(outcome: &mut PogMintOutcome, receipt: &TransactionReceipt) {
    outcome.receipt_status = receipt.status.map(|status| status.as_u64());
    outcome.receipt_block_number = receipt.block_number.map(|block| block.as_u64());
    outcome.receipt_gas_used = receipt.gas_used.map(|gas| gas.to_string());
}

fn mine_function() -> Function {
    abi_function(
        "mine",
        vec![param("nonce", ParamType::Uint(256))],
        vec![],
        StateMutability::NonPayable,
    )
}

fn challenge_for_function() -> Function {
    abi_function(
        "challengeFor",
        vec![param("miner", ParamType::Address)],
        vec![param("", ParamType::FixedBytes(32))],
        StateMutability::View,
    )
}

fn used_solutions_function() -> Function {
    abi_function(
        "usedSolutions",
        vec![param("", ParamType::FixedBytes(32))],
        vec![param("", ParamType::Bool)],
        StateMutability::View,
    )
}

fn uint_function(name: &str) -> Function {
    abi_function(
        name,
        vec![],
        vec![param("", ParamType::Uint(256))],
        StateMutability::View,
    )
}

fn uint_address_function(name: &str) -> Function {
    abi_function(
        name,
        vec![param("account", ParamType::Address)],
        vec![param("", ParamType::Uint(256))],
        StateMutability::View,
    )
}

#[allow(deprecated)]
fn abi_function(
    name: &str,
    inputs: Vec<Param>,
    outputs: Vec<Param>,
    state_mutability: StateMutability,
) -> Function {
    Function {
        name: name.to_string(),
        inputs,
        outputs,
        constant: None,
        state_mutability,
    }
}

fn param(name: &str, kind: ParamType) -> Param {
    Param {
        name: name.to_string(),
        kind,
        internal_type: None,
    }
}

fn parse_address(value: &str) -> Result<Address, MiningError> {
    Address::from_str(value.trim())
        .map_err(|error| message(format!("invalid Ethereum address: {error}")))
}

fn format_address(address: Address) -> String {
    format!("{address:#x}")
}

fn message(text: impl Into<String>) -> MiningError {
    MiningError::Message(text.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mine_calldata_encodes_uint256_nonce() {
        let nonce = 0x0102_0304_0506_0708u64;
        let calldata = pog_mine_calldata(nonce).unwrap();

        assert_eq!(calldata.len(), 36);
        assert_eq!(&calldata[4..28], &[0u8; 24]);
        assert_eq!(&calldata[28..36], &nonce.to_be_bytes());
    }

    #[test]
    fn nonce_be32_pads_high_bytes() {
        assert_eq!(
            hex_lower(&pog_nonce_to_be32(0x42)),
            "0000000000000000000000000000000000000000000000000000000000000042"
        );
    }

    #[test]
    fn challenge_derivation_is_stable() {
        let challenge = derive_pog_challenge(
            DEFAULT_POG_CHAIN_ID,
            DEFAULT_POG_CONTRACT_ADDRESS,
            "0x0000000000000000000000000000000000000001",
            U256::from(12345u64),
        )
        .unwrap();

        assert_eq!(
            hex_lower(&challenge),
            "cf428c48320f27db983bb618d1991d144075cd4c6d882b8da7b1c1d06ccfd2b5"
        );
    }

    #[test]
    fn parses_uint256_values() {
        assert_eq!(parse_pog_u256("0x2a").unwrap(), U256::from(42u64));
        assert_eq!(parse_pog_u256("42").unwrap(), U256::from(42u64));
    }

    #[test]
    fn priority_tip_debug_does_not_panic() {
        let mode = PogPriorityTip::Fixed(U256::from(DEFAULT_POG_PRIORITY_TIP_WEI));
        assert_eq!(
            format!("{mode:?}"),
            format!("Fixed({})", DEFAULT_POG_PRIORITY_TIP_WEI)
        );
    }
}
