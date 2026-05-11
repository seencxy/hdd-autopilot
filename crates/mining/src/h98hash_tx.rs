use std::fmt;
use std::str::FromStr;
use std::sync::Arc;

use ethers_core::abi::{Function, Param, ParamType, StateMutability, Token};
use ethers_core::types::transaction::eip2718::TypedTransaction;
use ethers_core::types::{Address, Bytes, TransactionReceipt, TransactionRequest, U256};
use ethers_middleware::SignerMiddleware;
use ethers_providers::{Http, Middleware, Provider};
use ethers_signers::{LocalWallet, Signer};

use crate::error::MiningError;
use crate::h98hash::hex_lower;

pub const DEFAULT_HASH98_CONTRACT_ADDRESS: &str = "0x1E5adF70321CA28b3Ead70Eac545E6055E969e6f";
pub const DEFAULT_HASH98_CHAIN_ID: u64 = 1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum H98HashMintValue {
    AutoMintPrice,
    Exact(U256),
}

#[derive(Clone)]
pub struct H98HashMintRequest {
    pub rpc_url: String,
    pub private_key: String,
    pub contract_address: String,
    pub chain_id: u64,
    pub nonce: [u8; 16],
    pub value_wei: H98HashMintValue,
    pub gas_limit: Option<u64>,
    pub gas_price_wei: Option<U256>,
    pub verify_before_send: bool,
    pub dry_run: bool,
    pub wait_for_receipt: bool,
}

impl fmt::Debug for H98HashMintRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("H98HashMintRequest")
            .field("rpc_url", &self.rpc_url)
            .field("private_key", &"<redacted>")
            .field("contract_address", &self.contract_address)
            .field("chain_id", &self.chain_id)
            .field("nonce", &format_args!("0x{}", hex_lower(&self.nonce)))
            .field("value_wei", &self.value_wei)
            .field("gas_limit", &self.gas_limit)
            .field("gas_price_wei", &self.gas_price_wei)
            .field("verify_before_send", &self.verify_before_send)
            .field("dry_run", &self.dry_run)
            .field("wait_for_receipt", &self.wait_for_receipt)
            .finish()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct H98HashConfig {
    pub mint_open: bool,
    pub market_open: bool,
    pub listing_open: bool,
    pub buying_open: bool,
    pub batch_open: bool,
    pub market_mode: u8,
    pub difficulty_bits: u32,
    pub mint_price_wei: U256,
    pub mint_amount: U256,
    pub max_public_mints: U256,
    pub treasury_reserve_mints: U256,
    pub reserve_minted: U256,
    pub total_supply: U256,
    pub active_listings: U256,
    pub market_fee_bps: U256,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct H98HashChainStatus {
    pub account_address: String,
    pub contract_address: String,
    pub challenge: [u8; 16],
    pub mint_nonce: U256,
    pub config: H98HashConfig,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct H98HashMintOutcome {
    pub dry_run: bool,
    pub from_address: String,
    pub contract_address: String,
    pub nonce_hex: String,
    pub calldata_hex: String,
    pub value_wei: String,
    pub proof_verified: Option<bool>,
    pub gas_estimate: String,
    pub transaction_hash: Option<String>,
    pub receipt_status: Option<u64>,
    pub receipt_block_number: Option<u64>,
    pub receipt_gas_used: Option<String>,
}

pub async fn submit_h98hash_mint(
    request: H98HashMintRequest,
) -> Result<H98HashMintOutcome, MiningError> {
    let contract_address = parse_address(&request.contract_address)?;
    let provider = Provider::<Http>::try_from(request.rpc_url.as_str())
        .map_err(|error| mining_message(format!("invalid HASH98 RPC URL: {error}")))?;
    let wallet = request
        .private_key
        .parse::<LocalWallet>()
        .map_err(|error| mining_message(format!("invalid HASH98 private key: {error}")))?
        .with_chain_id(request.chain_id);
    let from_address = wallet.address();
    let client = Arc::new(SignerMiddleware::new(provider, wallet));

    let value_wei = match request.value_wei {
        H98HashMintValue::AutoMintPrice => {
            let config = read_h98hash_config_with_client(client.as_ref(), contract_address).await?;
            if !config.mint_open {
                return Err(mining_message(
                    "HASH98 getConfig().mintOpen is false; refusing to submit mint transaction",
                ));
            }
            config.mint_price_wei
        }
        H98HashMintValue::Exact(value) => value,
    };

    let proof_verified = if request.verify_before_send {
        let verified = verify_h98hash_proof_with_client(
            client.as_ref(),
            contract_address,
            from_address,
            request.nonce,
        )
        .await?;
        if !verified {
            return Err(mining_message(
                "HASH98 verifyProof returned false; refusing to submit mint transaction",
            ));
        }
        Some(true)
    } else {
        None
    };

    let calldata = h98hash_mint_calldata(&request.nonce)?;
    let mut transaction = TransactionRequest::new()
        .to(contract_address)
        .from(from_address)
        .data(Bytes::from(calldata.clone()))
        .value(value_wei);
    if let Some(gas_limit) = request.gas_limit {
        transaction = transaction.gas(U256::from(gas_limit));
    }
    if let Some(gas_price_wei) = request.gas_price_wei {
        transaction = transaction.gas_price(gas_price_wei);
    }

    let typed_transaction: TypedTransaction = transaction.clone().into();
    let gas_estimate = client
        .estimate_gas(&typed_transaction, None)
        .await
        .map_err(|error| mining_message(format!("HASH98 gas estimate failed: {error}")))?;

    let mut outcome = H98HashMintOutcome {
        dry_run: request.dry_run,
        from_address: format_address(from_address),
        contract_address: format_address(contract_address),
        nonce_hex: format!("0x{}", hex_lower(&request.nonce)),
        calldata_hex: format!("0x{}", hex_lower(&calldata)),
        value_wei: value_wei.to_string(),
        proof_verified,
        gas_estimate: gas_estimate.to_string(),
        transaction_hash: None,
        receipt_status: None,
        receipt_block_number: None,
        receipt_gas_used: None,
    };

    if request.dry_run {
        return Ok(outcome);
    }

    let pending_transaction = client
        .send_transaction(transaction, None)
        .await
        .map_err(|error| mining_message(format!("HASH98 mint transaction send failed: {error}")))?;
    let transaction_hash = pending_transaction.tx_hash();
    outcome.transaction_hash = Some(format!("{transaction_hash:#x}"));
    if request.wait_for_receipt {
        if let Some(receipt) = pending_transaction
            .await
            .map_err(|error| mining_message(format!("HASH98 mint receipt wait failed: {error}")))?
        {
            fill_receipt_fields(&mut outcome, &receipt);
        }
    }
    Ok(outcome)
}

pub async fn read_h98hash_status(
    rpc_url: &str,
    contract_address: &str,
    account_address: &str,
) -> Result<H98HashChainStatus, MiningError> {
    let contract_address = parse_address(contract_address)?;
    let account_address = parse_address(account_address)?;
    let provider = Provider::<Http>::try_from(rpc_url)
        .map_err(|error| mining_message(format!("invalid HASH98 RPC URL: {error}")))?;
    let challenge =
        read_h98hash_challenge_with_client(&provider, contract_address, account_address).await?;
    let mint_nonce =
        read_h98hash_mint_nonce_with_client(&provider, contract_address, account_address).await?;
    let config = read_h98hash_config_with_client(&provider, contract_address).await?;
    Ok(H98HashChainStatus {
        account_address: format_address(account_address),
        contract_address: format_address(contract_address),
        challenge,
        mint_nonce,
        config,
    })
}

pub fn h98hash_mint_calldata(nonce: &[u8; 16]) -> Result<Vec<u8>, MiningError> {
    mint_function()
        .encode_input(&[Token::FixedBytes(nonce.to_vec())])
        .map_err(|error| mining_message(format!("failed to ABI-encode HASH98 mint: {error}")))
}

pub fn parse_h98hash_u256(value: &str) -> Result<U256, MiningError> {
    let normalized = value.trim().replace('_', "");
    if normalized.is_empty() {
        return Err(mining_message("empty uint256 value"));
    }
    if let Some(hex) = normalized
        .strip_prefix("0x")
        .or_else(|| normalized.strip_prefix("0X"))
    {
        return U256::from_str_radix(hex, 16)
            .map_err(|error| mining_message(format!("invalid hex uint256 value: {error}")));
    }
    U256::from_dec_str(&normalized)
        .map_err(|error| mining_message(format!("invalid decimal uint256 value: {error}")))
}

async fn verify_h98hash_proof_with_client<M>(
    client: &M,
    contract_address: Address,
    account_address: Address,
    nonce: [u8; 16],
) -> Result<bool, MiningError>
where
    M: Middleware,
{
    let function = verify_proof_function();
    let tokens = call_contract_function(
        client,
        contract_address,
        &function,
        &[
            Token::Address(account_address),
            Token::FixedBytes(nonce.to_vec()),
        ],
    )
    .await?;
    match tokens.as_slice() {
        [Token::Bool(verified)] => Ok(*verified),
        _ => Err(mining_message(
            "HASH98 verifyProof returned unexpected ABI output",
        )),
    }
}

async fn read_h98hash_challenge_with_client<M>(
    client: &M,
    contract_address: Address,
    account_address: Address,
) -> Result<[u8; 16], MiningError>
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
        [Token::FixedBytes(bytes)] if bytes.len() == 16 => {
            let mut challenge = [0u8; 16];
            challenge.copy_from_slice(bytes);
            Ok(challenge)
        }
        _ => Err(mining_message(
            "HASH98 challengeFor returned unexpected ABI output",
        )),
    }
}

async fn read_h98hash_mint_nonce_with_client<M>(
    client: &M,
    contract_address: Address,
    account_address: Address,
) -> Result<U256, MiningError>
where
    M: Middleware,
{
    let function = mint_nonce_function();
    let tokens = call_contract_function(
        client,
        contract_address,
        &function,
        &[Token::Address(account_address)],
    )
    .await?;
    match tokens.as_slice() {
        [Token::Uint(value)] => Ok(*value),
        _ => Err(mining_message(
            "HASH98 mintNonce returned unexpected ABI output",
        )),
    }
}

async fn read_h98hash_config_with_client<M>(
    client: &M,
    contract_address: Address,
) -> Result<H98HashConfig, MiningError>
where
    M: Middleware,
{
    let function = get_config_function();
    let tokens = call_contract_function(client, contract_address, &function, &[]).await?;
    let items = match tokens.as_slice() {
        [Token::Tuple(items)] => items.as_slice(),
        _ => {
            return Err(mining_message(
                "HASH98 getConfig returned unexpected ABI output",
            ));
        }
    };
    if items.len() != 15 {
        return Err(mining_message(format!(
            "HASH98 getConfig returned {} fields; expected 15",
            items.len()
        )));
    }
    Ok(H98HashConfig {
        mint_open: token_bool(&items[0], "mintOpen")?,
        market_open: token_bool(&items[1], "marketOpen")?,
        listing_open: token_bool(&items[2], "listingOpen")?,
        buying_open: token_bool(&items[3], "buyingOpen")?,
        batch_open: token_bool(&items[4], "batchOpen")?,
        market_mode: token_u8(&items[5], "marketMode")?,
        difficulty_bits: token_u32(&items[6], "difficulty")?,
        mint_price_wei: token_uint(&items[7], "mintPrice")?,
        mint_amount: token_uint(&items[8], "mintAmount")?,
        max_public_mints: token_uint(&items[9], "maxPublicMints")?,
        treasury_reserve_mints: token_uint(&items[10], "treasuryReserveMints")?,
        reserve_minted: token_uint(&items[11], "reserveMinted")?,
        total_supply: token_uint(&items[12], "totalSupply")?,
        active_listings: token_uint(&items[13], "activeListings")?,
        market_fee_bps: token_uint(&items[14], "marketFeeBps")?,
    })
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
        mining_message(format!(
            "failed to ABI-encode HASH98 {} call: {error}",
            function.name
        ))
    })?;
    let transaction: TypedTransaction = TransactionRequest::new()
        .to(contract_address)
        .data(Bytes::from(data))
        .into();
    let output = client.call(&transaction, None).await.map_err(|error| {
        mining_message(format!("HASH98 {} RPC call failed: {error}", function.name))
    })?;
    function.decode_output(output.as_ref()).map_err(|error| {
        mining_message(format!(
            "failed to ABI-decode HASH98 {} output: {error}",
            function.name
        ))
    })
}

fn fill_receipt_fields(outcome: &mut H98HashMintOutcome, receipt: &TransactionReceipt) {
    outcome.receipt_status = receipt.status.map(|status| status.as_u64());
    outcome.receipt_block_number = receipt.block_number.map(|block| block.as_u64());
    outcome.receipt_gas_used = receipt.gas_used.map(|gas| gas.to_string());
}

fn verify_proof_function() -> Function {
    abi_function(
        "verifyProof",
        vec![
            param("account", ParamType::Address),
            param("nonce", ParamType::FixedBytes(16)),
        ],
        vec![param("", ParamType::Bool)],
        StateMutability::View,
    )
}

fn challenge_for_function() -> Function {
    abi_function(
        "challengeFor",
        vec![param("account", ParamType::Address)],
        vec![param("", ParamType::FixedBytes(16))],
        StateMutability::View,
    )
}

fn mint_nonce_function() -> Function {
    abi_function(
        "mintNonce",
        vec![param("", ParamType::Address)],
        vec![param("", ParamType::Uint(256))],
        StateMutability::View,
    )
}

fn get_config_function() -> Function {
    abi_function(
        "getConfig",
        vec![],
        vec![param(
            "",
            ParamType::Tuple(vec![
                ParamType::Bool,
                ParamType::Bool,
                ParamType::Bool,
                ParamType::Bool,
                ParamType::Bool,
                ParamType::Uint(8),
                ParamType::Uint(256),
                ParamType::Uint(256),
                ParamType::Uint(256),
                ParamType::Uint(256),
                ParamType::Uint(256),
                ParamType::Uint(256),
                ParamType::Uint(256),
                ParamType::Uint(256),
                ParamType::Uint(256),
            ]),
        )],
        StateMutability::View,
    )
}

fn mint_function() -> Function {
    abi_function(
        "mint",
        vec![param("nonce", ParamType::FixedBytes(16))],
        vec![param("mintIndex", ParamType::Uint(256))],
        StateMutability::Payable,
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

fn token_bool(token: &Token, name: &str) -> Result<bool, MiningError> {
    match token {
        Token::Bool(value) => Ok(*value),
        _ => Err(mining_message(format!(
            "HASH98 getConfig field {name} is not bool"
        ))),
    }
}

fn token_uint(token: &Token, name: &str) -> Result<U256, MiningError> {
    match token {
        Token::Uint(value) => Ok(*value),
        _ => Err(mining_message(format!(
            "HASH98 getConfig field {name} is not uint"
        ))),
    }
}

fn token_u8(token: &Token, name: &str) -> Result<u8, MiningError> {
    let value = token_uint(token, name)?;
    if value > U256::from(u8::MAX) {
        return Err(mining_message(format!(
            "HASH98 getConfig field {name} exceeds u8"
        )));
    }
    Ok(value.as_u32() as u8)
}

fn token_u32(token: &Token, name: &str) -> Result<u32, MiningError> {
    let value = token_uint(token, name)?;
    if value > U256::from(u32::MAX) {
        return Err(mining_message(format!(
            "HASH98 getConfig field {name} exceeds u32"
        )));
    }
    Ok(value.as_u32())
}

fn parse_address(value: &str) -> Result<Address, MiningError> {
    Address::from_str(value.trim())
        .map_err(|error| mining_message(format!("invalid Ethereum address: {error}")))
}

fn format_address(address: Address) -> String {
    format!("{address:#x}")
}

fn mining_message(message: impl Into<String>) -> MiningError {
    MiningError::Message(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mint_calldata_encodes_bytes16_as_left_aligned_word() {
        let nonce = [
            0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d,
            0x1e, 0x1f,
        ];

        let calldata = h98hash_mint_calldata(&nonce).unwrap();

        assert_eq!(calldata.len(), 36);
        assert_eq!(&calldata[4..20], nonce);
        assert_eq!(&calldata[20..36], [0u8; 16]);
    }

    #[test]
    fn parses_decimal_and_hex_u256_values() {
        assert_eq!(
            parse_h98hash_u256("1000000000000000000").unwrap(),
            U256::exp10(18)
        );
        assert_eq!(
            parse_h98hash_u256("0xde0b6b3a7640000").unwrap(),
            U256::exp10(18)
        );
    }
}
