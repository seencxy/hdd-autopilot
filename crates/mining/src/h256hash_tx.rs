//! On-chain submit path for hash256.org `HASH` token.
//!
//! ABI was reverse-engineered from the public mining page's Wagmi/viem
//! configuration (`l.abi` in chunk `7476-bce50aba3843f162.js`). See the plan
//! file for the full set of selectors. This module only implements the calls
//! the miner actually needs:
//! - `mine(uint256 nonce)` (the write path, EIP-1559).
//! - `getChallenge(address miner) -> bytes32`.
//! - `miningState() -> (era, reward, difficulty, minted, remaining, epoch,
//!   epochBlocksLeft)` — all `uint256`.
//! - `usedProofs(bytes32 hash) -> bool` (anti-replay).
//! - `currentDifficulty / currentReward / epochBlocksLeft / totalMints /
//!   MAX_MINTS_PER_BLOCK / EPOCH_BLOCKS / MINING_SUPPLY / balanceOf` (read).

use std::fmt;
use std::str::FromStr;
use std::sync::Arc;

use ethers_core::abi::{Function, Param, ParamType, StateMutability, Token};
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

pub const DEFAULT_HASH256_CONTRACT_ADDRESS: &str = "0xAC7b5d06fa1e77D08aea40d46cB7C5923A87A0cc";
pub const DEFAULT_HASH256_CHAIN_ID: u64 = 1;
/// Gas limit the front-end uses for `mine(uint256)`. Plenty of headroom.
pub const DEFAULT_HASH256_GAS_LIMIT: u64 = 300_000;
/// Default priority tip in wei (1 gwei). Frontend slider defaults to 1.0 gwei.
pub const DEFAULT_HASH256_PRIORITY_TIP_WEI: u64 = 1_000_000_000;
/// Headroom added on top of `base_fee + priority_tip` for `maxFeePerGas`.
/// The frontend uses +5 gwei; matching here avoids being underpriced when
/// base fee spikes between block N and broadcast.
pub const DEFAULT_HASH256_MAX_FEE_HEADROOM_WEI: u64 = 5_000_000_000;

#[derive(Clone)]
pub struct H256HashMintRequest {
    pub rpc_url: String,
    pub private_key: String,
    pub contract_address: String,
    pub chain_id: u64,
    /// The low 64 bits of the uint256 nonce; high 24 bytes are zero.
    pub nonce: u64,
    pub gas_limit: Option<u64>,
    /// Priority fee in wei (`maxPriorityFeePerGas`).
    pub priority_tip_wei: U256,
    /// Optional cap on `maxFeePerGas`. If `None`, computed from the latest
    /// block's `baseFeePerGas + priority_tip_wei + headroom`.
    pub max_fee_wei: Option<U256>,
    /// Additional headroom added on top of `base_fee + priority_tip` when
    /// `max_fee_wei` is `None`. Defaults to +5 gwei (matches the frontend).
    pub max_fee_headroom_wei: U256,
    /// If true, call `usedProofs(hash) -> bool` before broadcasting.
    pub check_used_proofs: bool,
    pub dry_run: bool,
    pub wait_for_receipt: bool,
}

impl fmt::Debug for H256HashMintRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("H256HashMintRequest")
            .field("rpc_url", &self.rpc_url)
            .field("private_key", &"<redacted>")
            .field("contract_address", &self.contract_address)
            .field("chain_id", &self.chain_id)
            .field("nonce", &format_args!("0x{:016x}", self.nonce))
            .field("gas_limit", &self.gas_limit)
            .field("priority_tip_wei", &self.priority_tip_wei)
            .field("max_fee_wei", &self.max_fee_wei)
            .field("max_fee_headroom_wei", &self.max_fee_headroom_wei)
            .field("check_used_proofs", &self.check_used_proofs)
            .field("dry_run", &self.dry_run)
            .field("wait_for_receipt", &self.wait_for_receipt)
            .finish()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct H256HashMiningState {
    pub era: U256,
    pub reward: U256,
    pub difficulty: U256,
    pub minted: U256,
    pub remaining: U256,
    pub epoch: U256,
    pub epoch_blocks_left: U256,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct H256HashChainStatus {
    pub account_address: String,
    pub contract_address: String,
    pub challenge: [u8; 32],
    pub balance: U256,
    pub mining_state: H256HashMiningState,
    pub max_mints_per_block: U256,
    pub epoch_blocks: U256,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct H256HashMintOutcome {
    pub dry_run: bool,
    pub from_address: String,
    pub contract_address: String,
    pub nonce: u64,
    pub nonce_hex: String,
    pub proof_hash_hex: String,
    pub calldata_hex: String,
    /// Was `usedProofs(hash)` checked before broadcasting?
    pub proof_used_check: Option<bool>,
    pub gas_estimate: String,
    pub max_fee_per_gas: String,
    pub max_priority_fee_per_gas: String,
    pub transaction_hash: Option<String>,
    pub receipt_status: Option<u64>,
    pub receipt_block_number: Option<u64>,
    pub receipt_gas_used: Option<String>,
}

pub async fn submit_h256hash_mine(
    request: H256HashMintRequest,
) -> Result<H256HashMintOutcome, MiningError> {
    let contract_address = parse_address(&request.contract_address)?;
    let provider = Provider::<Http>::try_from(request.rpc_url.as_str())
        .map_err(|error| message(format!("invalid HASH256 RPC URL: {error}")))?;
    let wallet = request
        .private_key
        .parse::<LocalWallet>()
        .map_err(|error| message(format!("invalid HASH256 private key: {error}")))?
        .with_chain_id(request.chain_id);
    let from_address = wallet.address();
    let client = Arc::new(SignerMiddleware::new(provider, wallet));

    // Fetch the challenge once and reuse for both the optional usedProofs
    // anti-replay check and the displayed proof hash.
    let challenge =
        read_h256hash_challenge_with_client(client.as_ref(), contract_address, from_address)
            .await?;
    let proof_hash = keccak256(h256hash_input(&challenge, request.nonce));

    // Anti-replay: hash256.org keys `usedProofs` by the keccak result, not
    // the nonce. If the same hash was already accepted on chain, our tx will
    // revert. Optional but strongly recommended before paying gas.
    let proof_used_check = if request.check_used_proofs {
        let already_used =
            check_used_proofs_with_client(client.as_ref(), contract_address, &proof_hash).await?;
        if already_used {
            return Err(message(
                "HASH256 usedProofs reports this digest is already consumed; refusing to submit",
            ));
        }
        Some(false)
    } else {
        None
    };

    let nonce_be32 = nonce_to_be32(request.nonce);
    let calldata = h256hash_mine_calldata(request.nonce)?;

    // EIP-1559 fees, mirroring the hash256.org frontend.
    let priority_tip = request.priority_tip_wei;
    let max_fee = match request.max_fee_wei {
        Some(value) => value,
        None => {
            let block = client
                .get_block(BlockNumber::Latest)
                .await
                .map_err(|error| message(format!("HASH256 latest block fetch failed: {error}")))?
                .ok_or_else(|| message("HASH256 RPC returned no latest block"))?;
            let base_fee = block.base_fee_per_gas.unwrap_or(U256::zero());
            base_fee
                .saturating_add(priority_tip)
                .saturating_add(request.max_fee_headroom_wei)
        }
    };

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

    let typed_transaction: TypedTransaction = transaction.clone().into();
    let gas_estimate = client
        .estimate_gas(&typed_transaction, None)
        .await
        .map_err(|error| message(format!("HASH256 gas estimate failed: {error}")))?;

    let mut outcome = H256HashMintOutcome {
        dry_run: request.dry_run,
        from_address: format_address(from_address),
        contract_address: format_address(contract_address),
        nonce: request.nonce,
        nonce_hex: format!("0x{}", hex_lower(&nonce_be32)),
        proof_hash_hex: format!("0x{}", hex_lower(&proof_hash)),
        calldata_hex: format!("0x{}", hex_lower(&calldata)),
        proof_used_check,
        gas_estimate: gas_estimate.to_string(),
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

    let pending_transaction = client
        .send_transaction(typed_transaction, None)
        .await
        .map_err(|error| message(format!("HASH256 mine() transaction send failed: {error}")))?;
    let transaction_hash = pending_transaction.tx_hash();
    outcome.transaction_hash = Some(format!("{transaction_hash:#x}"));
    if request.wait_for_receipt {
        if let Some(receipt) = pending_transaction
            .await
            .map_err(|error| message(format!("HASH256 mine() receipt wait failed: {error}")))?
        {
            fill_receipt_fields(&mut outcome, &receipt);
        }
    }
    Ok(outcome)
}

pub async fn read_h256hash_status(
    rpc_url: &str,
    contract_address: &str,
    account_address: &str,
) -> Result<H256HashChainStatus, MiningError> {
    let contract_address = parse_address(contract_address)?;
    let account_address = parse_address(account_address)?;
    let provider = Provider::<Http>::try_from(rpc_url)
        .map_err(|error| message(format!("invalid HASH256 RPC URL: {error}")))?;
    let challenge =
        read_h256hash_challenge_with_client(&provider, contract_address, account_address).await?;
    let balance = read_balance_with_client(&provider, contract_address, account_address).await?;
    let mining_state = read_mining_state_with_client(&provider, contract_address).await?;
    let max_mints_per_block =
        read_uint_constant_with_client(&provider, contract_address, "MAX_MINTS_PER_BLOCK").await?;
    let epoch_blocks =
        read_uint_constant_with_client(&provider, contract_address, "EPOCH_BLOCKS").await?;
    Ok(H256HashChainStatus {
        account_address: format_address(account_address),
        contract_address: format_address(contract_address),
        challenge,
        balance,
        mining_state,
        max_mints_per_block,
        epoch_blocks,
    })
}

pub fn h256hash_mine_calldata(nonce: u64) -> Result<Vec<u8>, MiningError> {
    mine_function()
        .encode_input(&[Token::Uint(U256::from(nonce))])
        .map_err(|error| message(format!("failed to ABI-encode HASH256 mine: {error}")))
}

pub fn nonce_to_be32(nonce: u64) -> [u8; 32] {
    let mut out = [0u8; 32];
    out[24..32].copy_from_slice(&nonce.to_be_bytes());
    out
}

pub fn parse_h256hash_u256(value: &str) -> Result<U256, MiningError> {
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

async fn read_h256hash_challenge_with_client<M>(
    client: &M,
    contract_address: Address,
    account_address: Address,
) -> Result<[u8; 32], MiningError>
where
    M: Middleware,
{
    let function = get_challenge_function();
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
        _ => Err(message("HASH256 getChallenge returned unexpected ABI output")),
    }
}

async fn read_mining_state_with_client<M>(
    client: &M,
    contract_address: Address,
) -> Result<H256HashMiningState, MiningError>
where
    M: Middleware,
{
    let function = mining_state_function();
    let tokens = call_contract_function(client, contract_address, &function, &[]).await?;
    match tokens.as_slice() {
        [
            Token::Uint(era),
            Token::Uint(reward),
            Token::Uint(difficulty),
            Token::Uint(minted),
            Token::Uint(remaining),
            Token::Uint(epoch),
            Token::Uint(epoch_blocks_left),
        ] => Ok(H256HashMiningState {
            era: *era,
            reward: *reward,
            difficulty: *difficulty,
            minted: *minted,
            remaining: *remaining,
            epoch: *epoch,
            epoch_blocks_left: *epoch_blocks_left,
        }),
        _ => Err(message(
            "HASH256 miningState returned unexpected ABI output",
        )),
    }
}

async fn check_used_proofs_with_client<M>(
    client: &M,
    contract_address: Address,
    digest: &[u8; 32],
) -> Result<bool, MiningError>
where
    M: Middleware,
{
    let function = used_proofs_function();
    let tokens = call_contract_function(
        client,
        contract_address,
        &function,
        &[Token::FixedBytes(digest.to_vec())],
    )
    .await?;
    match tokens.as_slice() {
        [Token::Bool(value)] => Ok(*value),
        _ => Err(message("HASH256 usedProofs returned unexpected ABI output")),
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
    let function = balance_of_function();
    let tokens = call_contract_function(
        client,
        contract_address,
        &function,
        &[Token::Address(account_address)],
    )
    .await?;
    match tokens.as_slice() {
        [Token::Uint(value)] => Ok(*value),
        _ => Err(message("HASH256 balanceOf returned unexpected ABI output")),
    }
}

async fn read_uint_constant_with_client<M>(
    client: &M,
    contract_address: Address,
    name: &str,
) -> Result<U256, MiningError>
where
    M: Middleware,
{
    let function = uint_constant_function(name);
    let tokens = call_contract_function(client, contract_address, &function, &[]).await?;
    match tokens.as_slice() {
        [Token::Uint(value)] => Ok(*value),
        _ => Err(message(format!(
            "HASH256 {name} returned unexpected ABI output"
        ))),
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
            "failed to ABI-encode HASH256 {} call: {error}",
            function.name
        ))
    })?;
    let transaction: TypedTransaction = Eip1559TransactionRequest::new()
        .to(contract_address)
        .data(Bytes::from(data))
        .into();
    let block = Some(BlockId::Number(BlockNumber::Latest));
    let output = client.call(&transaction, block).await.map_err(|error| {
        message(format!(
            "HASH256 {} RPC call failed: {error}",
            function.name
        ))
    })?;
    function.decode_output(output.as_ref()).map_err(|error| {
        message(format!(
            "failed to ABI-decode HASH256 {} output: {error}",
            function.name
        ))
    })
}

fn fill_receipt_fields(outcome: &mut H256HashMintOutcome, receipt: &TransactionReceipt) {
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

fn get_challenge_function() -> Function {
    abi_function(
        "getChallenge",
        vec![param("miner", ParamType::Address)],
        vec![param("", ParamType::FixedBytes(32))],
        StateMutability::View,
    )
}

fn mining_state_function() -> Function {
    abi_function(
        "miningState",
        vec![],
        vec![
            param("era", ParamType::Uint(256)),
            param("reward", ParamType::Uint(256)),
            param("difficulty", ParamType::Uint(256)),
            param("minted", ParamType::Uint(256)),
            param("remaining", ParamType::Uint(256)),
            param("epoch", ParamType::Uint(256)),
            param("epochBlocksLeft", ParamType::Uint(256)),
        ],
        StateMutability::View,
    )
}

fn used_proofs_function() -> Function {
    abi_function(
        "usedProofs",
        vec![param("", ParamType::FixedBytes(32))],
        vec![param("", ParamType::Bool)],
        StateMutability::View,
    )
}

fn balance_of_function() -> Function {
    abi_function(
        "balanceOf",
        vec![param("account", ParamType::Address)],
        vec![param("", ParamType::Uint(256))],
        StateMutability::View,
    )
}

fn uint_constant_function(name: &str) -> Function {
    abi_function(
        name,
        vec![],
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
    fn mine_calldata_encodes_uint256_with_low_64_bits() {
        let nonce: u64 = 0x0102_0304_0506_0708;
        let calldata = h256hash_mine_calldata(nonce).unwrap();
        // selector (4 bytes) + 32-byte word
        assert_eq!(calldata.len(), 36);
        // High 24 bytes of the uint256 word are zero, low 8 = BE nonce.
        assert_eq!(&calldata[4..28], &[0u8; 24]);
        assert_eq!(&calldata[28..36], &nonce.to_be_bytes());
    }

    #[test]
    fn nonce_be32_only_uses_low_eight_bytes() {
        let nonce_be = nonce_to_be32(0x42);
        assert_eq!(&nonce_be[..24], &[0u8; 24]);
        assert_eq!(&nonce_be[24..32], &[0u8, 0, 0, 0, 0, 0, 0, 0x42]);
    }

    #[test]
    fn parses_decimal_and_hex_u256_values() {
        assert_eq!(
            parse_h256hash_u256("1000000000000000000").unwrap(),
            U256::exp10(18)
        );
        assert_eq!(
            parse_h256hash_u256("0xde0b6b3a7640000").unwrap(),
            U256::exp10(18)
        );
    }

    #[test]
    fn default_contract_and_chain_id_match_hash256_org() {
        assert_eq!(
            DEFAULT_HASH256_CONTRACT_ADDRESS,
            "0xAC7b5d06fa1e77D08aea40d46cB7C5923A87A0cc"
        );
        assert_eq!(DEFAULT_HASH256_CHAIN_ID, 1);
    }
}
