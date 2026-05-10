use std::collections::{HashSet, VecDeque};
use std::env;
use std::error::Error;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use crossterm as _;
use hdd_autopilot as _;
#[cfg(not(target_os = "macos"))]
use iana_time_zone as _;
use mining::MiningError;
use mining::rpow2::{
    Rpow2Backend, Rpow2Challenge, Rpow2Client, Rpow2Job, Rpow2MineConfig, Rpow2RoutedChallenge,
    Rpow4Signer, load_rpow2_proxy_file, mine_rpow2, rpow2_meets_difficulty,
};
use rand as _;
use reqwest as _;
use serde as _;
use serde_json::{Value, json};
use time as _;
use unicode_width as _;
use url::Url;

const DEFAULT_PROXY_FILE: &str = "rpow2-proxies.txt";
const RECENT_CHALLENGE_LIMIT: usize = 4096;

fn main() {
    if let Err(error) = run() {
        print_error_chain(error.as_ref());
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

    let target = ApiTarget::from_args(&args)?;
    let cookie = load_cookie(&args, target.network)?;
    let mint_mode = MintMode::from_args(&args, target.network)?;
    let proxy_concurrency = load_proxy_concurrency(&args, target.network)?;
    let (proxy_urls, proxy_file) = load_proxy_pool(&args, target.network)?;
    let client = Rpow2Client::with_base_url_origin_and_proxy_pool(
        &target.api_base,
        Some(&cookie),
        target.origin.as_deref(),
        target.referer.as_deref(),
        &proxy_urls,
    )?;
    println!(
        "network={} api_base={}",
        target.network.label(),
        target.api_base
    );
    if client.proxy_pool_size() > 0 {
        println!(
            "proxy_pool={} proxy_file={}",
            client.proxy_pool_size(),
            proxy_file
        );
    }
    if let Ok(me) = client.me() {
        validate_account_matches_mint_mode(&me, &mint_mode)?;
        println!("account: {}", me);
    }
    if let Ok(ledger) = client.ledger() {
        println!("ledger: {}", ledger);
    }

    if args.loop_forever {
        if args.serial_api {
            run_serial_loop(&client, &args, &mint_mode)?;
        } else {
            run_pipelined_loop(&client, &args, mint_mode, proxy_concurrency)?;
        }
    } else {
        run_single_round(&client, &args, &mint_mode)?;
    }
    Ok(())
}

fn load_cookie(args: &Args, network: RpowNetwork) -> Result<String, Box<dyn std::error::Error>> {
    args.cookie
        .clone()
        .or_else(|| env::var(network.cookie_env()).ok())
        .or_else(|| env::var("RPOW_COOKIE").ok())
        .ok_or_else(|| {
            format!(
                "missing RPOW cookie; pass --cookie or set {}/RPOW_COOKIE",
                network.cookie_env()
            )
            .into()
        })
}

fn load_proxy_pool(
    args: &Args,
    network: RpowNetwork,
) -> Result<(Vec<String>, String), Box<dyn std::error::Error>> {
    let proxy_file = args
        .proxy_file
        .clone()
        .or_else(|| env::var(network.proxy_file_env()).ok())
        .or_else(|| env::var("RPOW_PROXY_FILE").ok())
        .unwrap_or_else(|| network.default_proxy_file().to_string());
    let mut proxies = load_rpow2_proxy_file(&proxy_file)?;
    if let Some(proxy) = args
        .proxy
        .clone()
        .or_else(|| env::var(network.proxy_env()).ok())
        .or_else(|| env::var("RPOW_PROXY").ok())
        .map(|proxy| proxy.trim().to_string())
        .filter(|proxy| !proxy.is_empty())
    {
        proxies.push(proxy);
    }
    Ok((proxies, proxy_file))
}

fn load_proxy_concurrency(
    args: &Args,
    network: RpowNetwork,
) -> Result<Option<usize>, Box<dyn std::error::Error>> {
    if let Some(concurrency) = args.proxy_concurrency {
        return Ok(Some(concurrency));
    }
    if let Ok(value) =
        env::var(network.proxy_concurrency_env()).or_else(|_| env::var("RPOW_PROXY_CONCURRENCY"))
    {
        let value = value.trim();
        if !value.is_empty() {
            return Ok(Some(value.parse()?));
        }
    }
    Ok(None)
}

#[derive(Debug, Clone)]
struct ApiTarget {
    network: RpowNetwork,
    api_base: String,
    origin: Option<String>,
    referer: Option<String>,
}

impl ApiTarget {
    fn from_args(args: &Args) -> Result<Self, Box<dyn std::error::Error>> {
        let network = args.network.unwrap_or(RpowNetwork::Rpow2);
        let api_base = args
            .api_base
            .clone()
            .or_else(|| env::var(network.api_base_env()).ok())
            .or_else(|| env::var("RPOW_API_BASE").ok())
            .unwrap_or_else(|| network.default_api_base().to_string());
        Url::parse(&api_base).map_err(|error| format!("invalid API base URL: {error}"))?;
        let origin = args
            .origin
            .clone()
            .or_else(|| env::var(network.origin_env()).ok())
            .or_else(|| env::var("RPOW_ORIGIN").ok())
            .or_else(|| Some(network.default_origin().to_string()));
        let referer = args
            .referer
            .clone()
            .or_else(|| env::var(network.referer_env()).ok())
            .or_else(|| env::var("RPOW_REFERER").ok())
            .or_else(|| {
                origin
                    .as_ref()
                    .map(|origin| format!("{}/", origin.trim_end_matches('/')))
            });
        Ok(Self {
            network,
            api_base,
            origin,
            referer,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RpowNetwork {
    Rpow2,
    Rpow3,
    Rpow4,
}

impl RpowNetwork {
    fn parse(value: &str) -> Result<Self, Box<dyn std::error::Error>> {
        match value {
            "rpow2" | "2" => Ok(Self::Rpow2),
            "rpow3" | "3" => Ok(Self::Rpow3),
            "rpow4" | "4" => Ok(Self::Rpow4),
            _ => Err(format!("unknown network: {value}; expected rpow2, rpow3, or rpow4").into()),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Rpow2 => "rpow2",
            Self::Rpow3 => "rpow3",
            Self::Rpow4 => "rpow4",
        }
    }

    fn default_api_base(self) -> &'static str {
        match self {
            Self::Rpow2 => "https://api.rpow2.com",
            Self::Rpow3 => "https://api.rpow3.com",
            Self::Rpow4 => "https://rpow4.com",
        }
    }

    fn default_origin(self) -> &'static str {
        match self {
            Self::Rpow2 => "https://rpow2.com",
            Self::Rpow3 => "https://rpow3.com",
            Self::Rpow4 => "https://rpow4.com",
        }
    }

    fn default_proxy_file(self) -> &'static str {
        match self {
            Self::Rpow2 => DEFAULT_PROXY_FILE,
            Self::Rpow3 => "rpow3-proxies.txt",
            Self::Rpow4 => "rpow4-proxies.txt",
        }
    }

    fn cookie_env(self) -> &'static str {
        match self {
            Self::Rpow2 => "RPOW2_COOKIE",
            Self::Rpow3 => "RPOW3_COOKIE",
            Self::Rpow4 => "RPOW4_COOKIE",
        }
    }

    fn api_base_env(self) -> &'static str {
        match self {
            Self::Rpow2 => "RPOW2_API_BASE",
            Self::Rpow3 => "RPOW3_API_BASE",
            Self::Rpow4 => "RPOW4_API_BASE",
        }
    }

    fn origin_env(self) -> &'static str {
        match self {
            Self::Rpow2 => "RPOW2_ORIGIN",
            Self::Rpow3 => "RPOW3_ORIGIN",
            Self::Rpow4 => "RPOW4_ORIGIN",
        }
    }

    fn referer_env(self) -> &'static str {
        match self {
            Self::Rpow2 => "RPOW2_REFERER",
            Self::Rpow3 => "RPOW3_REFERER",
            Self::Rpow4 => "RPOW4_REFERER",
        }
    }

    fn proxy_env(self) -> &'static str {
        match self {
            Self::Rpow2 => "RPOW2_PROXY",
            Self::Rpow3 => "RPOW3_PROXY",
            Self::Rpow4 => "RPOW4_PROXY",
        }
    }

    fn proxy_file_env(self) -> &'static str {
        match self {
            Self::Rpow2 => "RPOW2_PROXY_FILE",
            Self::Rpow3 => "RPOW3_PROXY_FILE",
            Self::Rpow4 => "RPOW4_PROXY_FILE",
        }
    }

    fn proxy_concurrency_env(self) -> &'static str {
        match self {
            Self::Rpow2 => "RPOW2_PROXY_CONCURRENCY",
            Self::Rpow3 => "RPOW3_PROXY_CONCURRENCY",
            Self::Rpow4 => "RPOW4_PROXY_CONCURRENCY",
        }
    }
}

#[derive(Debug, Clone)]
enum MintMode {
    Legacy,
    Rpow4 { signer: Rpow4Signer },
}

impl MintMode {
    fn from_args(args: &Args, network: RpowNetwork) -> Result<Self, Box<dyn std::error::Error>> {
        if network != RpowNetwork::Rpow4 {
            return Ok(Self::Legacy);
        }
        Ok(Self::Rpow4 {
            signer: rpow4_signer_from_args(args)?,
        })
    }

    fn mint(
        &self,
        client: &Rpow2Client,
        challenge: &Rpow2Challenge,
        solution_nonce: u64,
        routed_challenge: &Rpow2RoutedChallenge,
    ) -> Result<Value, MiningError> {
        match self {
            Self::Legacy => client.mint_via_route(
                &challenge.challenge_id,
                solution_nonce,
                routed_challenge.route(),
            ),
            Self::Rpow4 { signer } => {
                let signature = signer.sign_mint(&challenge.challenge_id, solution_nonce)?;
                client.mint_rpow4_via_route(
                    challenge,
                    solution_nonce,
                    &signature,
                    routed_challenge.route(),
                )
            }
        }
    }
}

fn rpow4_signer_from_args(args: &Args) -> Result<Rpow4Signer, Box<dyn std::error::Error>> {
    if let Some(private_key) = args
        .rpow4_private_key
        .clone()
        .or_else(|| env::var("RPOW4_PRIVATE_KEY").ok())
        .or_else(|| env::var("RPOW_PRIVATE_KEY").ok())
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
    {
        return Ok(Rpow4Signer::from_private_key(&private_key)?);
    }
    if let Some(mnemonic) = args
        .rpow4_mnemonic
        .clone()
        .or_else(|| env::var("RPOW4_MNEMONIC").ok())
        .or_else(|| env::var("RPOW_MNEMONIC").ok())
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
    {
        return Ok(Rpow4Signer::from_mnemonic(&mnemonic)?);
    }
    Err("rpow4 mint requires wallet signing secret; set RPOW4_PRIVATE_KEY/RPOW_PRIVATE_KEY or RPOW4_MNEMONIC/RPOW_MNEMONIC".into())
}

fn validate_account_matches_mint_mode(
    me: &Value,
    mint_mode: &MintMode,
) -> Result<(), Box<dyn std::error::Error>> {
    let MintMode::Rpow4 { signer } = mint_mode else {
        return Ok(());
    };
    let Some(account_pubkey) = me.get("pubkey").and_then(Value::as_str) else {
        return Ok(());
    };
    if account_pubkey != signer.public_key_base58() {
        return Err(format!(
            "RPOW4 signer pubkey {} does not match session pubkey {}",
            signer.public_key_base58(),
            account_pubkey
        )
        .into());
    }
    Ok(())
}

fn run_single_round(
    client: &Rpow2Client,
    args: &Args,
    mint_mode: &MintMode,
) -> Result<(), Box<dyn std::error::Error>> {
    let challenge = retry_online("challenge", || client.challenge_with_route())?;
    let result = mine_challenge(challenge.challenge(), args)?;
    let mint = mint_with_recovery(client, &challenge, result.nonce, mint_mode)?;
    println!("mint: {}", mint);
    Ok(())
}

fn run_serial_loop(
    client: &Rpow2Client,
    args: &Args,
    mint_mode: &MintMode,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("api_mode=serial");
    loop {
        run_single_round(client, args, mint_mode)?;
    }
}

fn run_pipelined_loop(
    client: &Rpow2Client,
    args: &Args,
    mint_mode: MintMode,
    proxy_concurrency: Option<usize>,
) -> Result<(), Box<dyn std::error::Error>> {
    let prefetch_depth = args
        .challenge_prefetch
        .or(proxy_concurrency)
        .unwrap_or_else(default_challenge_prefetch)
        .max(1);
    let mint_workers = args
        .mint_workers
        .or(proxy_concurrency)
        .unwrap_or_else(default_mint_workers)
        .max(1);
    println!(
        "pipeline challenge_prefetch={} mint_workers={}",
        prefetch_depth, mint_workers
    );

    let challenges = spawn_challenge_prefetchers(client.clone(), prefetch_depth);
    let mint_sender = spawn_mint_workers(client.clone(), mint_workers, mint_mode);

    loop {
        let challenge = challenges
            .recv()
            .map_err(|_| "challenge prefetchers stopped unexpectedly")?;
        let result = mine_challenge(challenge.challenge(), args)?;
        mint_sender
            .send(MintJob {
                challenge,
                solution_nonce: result.nonce,
            })
            .map_err(|_| "mint workers stopped unexpectedly")?;
    }
}

fn mine_challenge(
    challenge: &Rpow2Challenge,
    args: &Args,
) -> Result<mining::rpow2::Rpow2MineResult, Box<dyn std::error::Error>> {
    println!(
        "challenge={} difficulty={} prefix={}",
        challenge.challenge_id, challenge.difficulty_bits, challenge.nonce_prefix
    );
    let job = challenge.job()?;
    let result = solve_job(&job, args)?;
    println!(
        "found challenge={} solution_nonce={} digest={} backend={:?} attempts={}",
        challenge.challenge_id, result.nonce, result.digest_hex, result.backend, result.attempts
    );
    Ok(result)
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
        metal_batch_size: args.gpu_batch_size.unwrap_or_else(default_gpu_batch_size),
        cuda_threads_per_block: args
            .gpu_threads_per_block
            .unwrap_or(mining::rpow2::DEFAULT_CUDA_THREADS_PER_BLOCK),
        cuda_nonces_per_thread: args
            .gpu_nonces_per_thread
            .unwrap_or(mining::rpow2::DEFAULT_CUDA_NONCES_PER_THREAD),
        cuda_max_blocks: args
            .gpu_max_blocks
            .unwrap_or(mining::rpow2::DEFAULT_CUDA_MAX_BLOCKS),
        cuda_early_exit: !args.gpu_no_early_exit,
        cpu_threads: args.cpu_threads.unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(usize::from)
                .unwrap_or(1)
        }),
        ..Rpow2MineConfig::default()
    };
    if !args.cpu_only {
        println!(
            "gpu_device={} gpu_batch_size={} threads_per_block={} nonces_per_thread={} max_blocks={} early_exit={}",
            config.metal_device_index,
            gpu_batch_label(config.metal_batch_size),
            config.cuda_threads_per_block,
            config.cuda_nonces_per_thread,
            gpu_max_blocks_label(config.cuda_max_blocks),
            config.cuda_early_exit
        );
    }
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

#[derive(Debug, Clone)]
struct MintJob {
    challenge: Rpow2RoutedChallenge,
    solution_nonce: u64,
}

fn spawn_challenge_prefetchers(
    client: Rpow2Client,
    prefetch_depth: usize,
) -> Receiver<Rpow2RoutedChallenge> {
    let (sender, receiver) = sync_channel(prefetch_depth);
    let recent_challenges = Arc::new(Mutex::new(RecentChallengeIds::new(RECENT_CHALLENGE_LIMIT)));
    for worker_index in 0..prefetch_depth {
        let client = client.clone();
        let sender = sender.clone();
        let recent_challenges = Arc::clone(&recent_challenges);
        thread::spawn(move || {
            let label = format!("challenge-prefetch-{worker_index}");
            loop {
                match retry_online(&label, || client.challenge_with_route()) {
                    Ok(challenge) => {
                        let is_new_challenge = {
                            let mut recent_challenges = recent_challenges
                                .lock()
                                .expect("recent challenge lock poisoned");
                            recent_challenges.insert(challenge.challenge().challenge_id.clone())
                        };
                        if !is_new_challenge {
                            eprintln!(
                                "{} skipped duplicate challenge={}",
                                label,
                                challenge.challenge().challenge_id
                            );
                            thread::sleep(Duration::from_secs(1));
                            continue;
                        }
                        if sender.send(challenge).is_err() {
                            break;
                        }
                    }
                    Err(error) => {
                        eprintln!(
                            "{} stopped after non-retryable error: {}; retrying worker in 15s",
                            label,
                            error_chain_to_string(&error)
                        );
                        thread::sleep(Duration::from_secs(15));
                    }
                }
            }
        });
    }
    drop(sender);
    receiver
}

#[derive(Debug)]
struct RecentChallengeIds {
    limit: usize,
    order: VecDeque<String>,
    ids: HashSet<String>,
}

impl RecentChallengeIds {
    fn new(limit: usize) -> Self {
        Self {
            limit: limit.max(1),
            order: VecDeque::new(),
            ids: HashSet::new(),
        }
    }

    fn insert(&mut self, challenge_id: String) -> bool {
        if self.ids.contains(&challenge_id) {
            return false;
        }
        self.ids.insert(challenge_id.clone());
        self.order.push_back(challenge_id);
        while self.order.len() > self.limit {
            if let Some(expired) = self.order.pop_front() {
                self.ids.remove(&expired);
            }
        }
        true
    }
}

fn spawn_mint_workers(
    client: Rpow2Client,
    worker_count: usize,
    mint_mode: MintMode,
) -> SyncSender<MintJob> {
    let (sender, receiver) = sync_channel::<MintJob>(worker_count.saturating_mul(4).max(1));
    let receiver = Arc::new(Mutex::new(receiver));
    for worker_index in 0..worker_count {
        let client = client.clone();
        let receiver = Arc::clone(&receiver);
        let mint_mode = mint_mode.clone();
        thread::spawn(move || {
            loop {
                let job = {
                    let receiver = receiver.lock().expect("mint receiver lock poisoned");
                    receiver.recv()
                };
                let Ok(job) = job else {
                    break;
                };
                match mint_with_recovery(&client, &job.challenge, job.solution_nonce, &mint_mode) {
                    Ok(value) => println!("mint: {}", value),
                    Err(error) => eprintln!(
                        "mint worker {} failed for challenge={}: {}",
                        worker_index,
                        job.challenge.challenge().challenge_id,
                        error_chain_to_string(&error)
                    ),
                }
            }
        });
    }
    sender
}

fn mint_with_recovery(
    client: &Rpow2Client,
    routed_challenge: &Rpow2RoutedChallenge,
    solution_nonce: u64,
    mint_mode: &MintMode,
) -> Result<Value, MiningError> {
    let mut attempt = 1u32;
    let challenge = routed_challenge.challenge();
    loop {
        match mint_mode.mint(client, challenge, solution_nonce, routed_challenge) {
            Ok(value) => return Ok(value),
            Err(error) if is_already_claimed_error(&error) => {
                eprintln!(
                    "mint already claimed for challenge={}; treating as accepted",
                    challenge.challenge_id
                );
                return Ok(json!({
                    "status": "already_claimed",
                    "challenge_id": challenge.challenge_id,
                    "solution_nonce": solution_nonce.to_string()
                }));
            }
            Err(error) if is_retryable_online_error(&error) => {
                let delay = retry_delay(attempt);
                eprintln!(
                    "mint request failed on attempt {}: {}; retrying in {}s",
                    attempt,
                    error_chain_to_string(&error),
                    delay.as_secs()
                );
                thread::sleep(delay);
                attempt = attempt.saturating_add(1);
            }
            Err(error) => return Err(error),
        }
    }
}

fn default_gpu_batch_size() -> u64 {
    #[cfg(target_os = "windows")]
    {
        mining::rpow2::DEFAULT_CUDA_BATCH_SIZE
    }
    #[cfg(not(target_os = "windows"))]
    {
        0
    }
}

fn default_challenge_prefetch() -> usize {
    #[cfg(target_os = "windows")]
    {
        2
    }
    #[cfg(not(target_os = "windows"))]
    {
        2
    }
}

fn default_mint_workers() -> usize {
    #[cfg(target_os = "windows")]
    {
        1
    }
    #[cfg(not(target_os = "windows"))]
    {
        1
    }
}

fn gpu_batch_label(batch_size: u64) -> String {
    if batch_size == 0 {
        "auto".to_string()
    } else {
        batch_size.to_string()
    }
}

fn gpu_max_blocks_label(max_blocks: u32) -> String {
    if max_blocks == 0 {
        "auto".to_string()
    } else {
        max_blocks.to_string()
    }
}

fn retry_online<T, F>(label: &str, mut operation: F) -> Result<T, MiningError>
where
    F: FnMut() -> Result<T, MiningError>,
{
    let mut attempt = 1u32;
    loop {
        match operation() {
            Ok(value) => return Ok(value),
            Err(error) if is_retryable_online_error(&error) => {
                let delay = retry_delay(attempt);
                eprintln!(
                    "{} request failed on attempt {}: {}; retrying in {}s",
                    label,
                    attempt,
                    error_chain_to_string(&error),
                    delay.as_secs()
                );
                thread::sleep(delay);
                attempt = attempt.saturating_add(1);
            }
            Err(error) => return Err(error),
        }
    }
}

fn retry_delay(attempt: u32) -> Duration {
    match attempt {
        0 | 1 => Duration::from_secs(1),
        2 => Duration::from_secs(2),
        3 => Duration::from_secs(4),
        4 => Duration::from_secs(8),
        _ => Duration::from_secs(15),
    }
}

fn is_retryable_online_error(error: &MiningError) -> bool {
    match error {
        MiningError::Http(error) => {
            error.is_timeout()
                || error.is_connect()
                || error.is_request()
                || error.is_body()
                || error.status().is_some_and(|status| {
                    status.as_u16() == 408
                        || status.as_u16() == 425
                        || status.as_u16() == 429
                        || status.is_server_error()
                })
        }
        MiningError::Message(message) => retryable_status_message(message),
        _ => false,
    }
}

fn retryable_status_message(message: &str) -> bool {
    [408, 425, 429, 500, 502, 503, 504]
        .into_iter()
        .any(|status| message.contains(&format!("状态码 {status}")))
}

fn is_already_claimed_error(error: &MiningError) -> bool {
    match error {
        MiningError::Message(message) => message.to_ascii_lowercase().contains("already claimed"),
        _ => false,
    }
}

fn print_error_chain(error: &(dyn Error + 'static)) {
    eprintln!("rpow2mine: {}", error_chain_to_string(error));
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

#[derive(Debug, Default)]
struct Args {
    cookie: Option<String>,
    rpow4_private_key: Option<String>,
    rpow4_mnemonic: Option<String>,
    network: Option<RpowNetwork>,
    api_base: Option<String>,
    origin: Option<String>,
    referer: Option<String>,
    proxy: Option<String>,
    proxy_file: Option<String>,
    prefix: Option<String>,
    difficulty_bits: Option<u32>,
    cpu_only: bool,
    loop_forever: bool,
    help: bool,
    cpu_threads: Option<usize>,
    gpu_batch_size: Option<u64>,
    gpu_device_index: Option<usize>,
    gpu_threads_per_block: Option<u32>,
    gpu_nonces_per_thread: Option<u32>,
    gpu_max_blocks: Option<u32>,
    gpu_no_early_exit: bool,
    challenge_prefetch: Option<usize>,
    mint_workers: Option<usize>,
    proxy_concurrency: Option<usize>,
    serial_api: bool,
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
                "--rpow4-private-key" => {
                    index += 1;
                    args.rpow4_private_key =
                        Some(next_value(&raw, index, "--rpow4-private-key")?.to_string());
                }
                "--rpow4-mnemonic" => {
                    index += 1;
                    args.rpow4_mnemonic =
                        Some(next_value(&raw, index, "--rpow4-mnemonic")?.to_string());
                }
                "--network" => {
                    index += 1;
                    args.network = Some(RpowNetwork::parse(next_value(&raw, index, "--network")?)?);
                }
                "--rpow2" => args.network = Some(RpowNetwork::Rpow2),
                "--rpow3" => args.network = Some(RpowNetwork::Rpow3),
                "--rpow4" => args.network = Some(RpowNetwork::Rpow4),
                "--api-base" => {
                    index += 1;
                    args.api_base = Some(next_value(&raw, index, "--api-base")?.to_string());
                }
                "--origin" => {
                    index += 1;
                    args.origin = Some(next_value(&raw, index, "--origin")?.to_string());
                }
                "--referer" => {
                    index += 1;
                    args.referer = Some(next_value(&raw, index, "--referer")?.to_string());
                }
                "--proxy" => {
                    index += 1;
                    args.proxy = Some(next_value(&raw, index, "--proxy")?.to_string());
                }
                "--proxy-file" => {
                    index += 1;
                    args.proxy_file = Some(next_value(&raw, index, "--proxy-file")?.to_string());
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
                "--serial-api" | "--no-pipeline" => {
                    args.serial_api = true;
                }
                "--challenge-prefetch" => {
                    index += 1;
                    args.challenge_prefetch =
                        Some(next_value(&raw, index, "--challenge-prefetch")?.parse()?);
                }
                "--mint-workers" => {
                    index += 1;
                    args.mint_workers = Some(next_value(&raw, index, "--mint-workers")?.parse()?);
                }
                "--proxy-concurrency" => {
                    index += 1;
                    args.proxy_concurrency =
                        Some(next_value(&raw, index, "--proxy-concurrency")?.parse()?);
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
  rpow2mine --cookie '<cookie-header>' [--network rpow2|rpow3|rpow4] [--rpow4-private-key <key>|--rpow4-mnemonic <words>] [--api-base <url>] [--proxy <url>] [--proxy-file <path>] [--proxy-concurrency <n>] [--loop] [--serial-api] [--cpu-only] [--gpu-device <index>] [--gpu-batch-size <hashes>] [--gpu-threads-per-block <n>] [--gpu-nonces-per-thread <n>] [--gpu-max-blocks <n>] [--gpu-no-early-exit] [--challenge-prefetch <n>] [--mint-workers <n>]
  rpow2mine --prefix <hex> --difficulty <bits> [--cpu-only] [--gpu-device <index>] [--gpu-batch-size <hashes>]

Environment:
  RPOW2_COOKIE   Cookie header copied from an authenticated rpow2.com browser session
  RPOW3_COOKIE   Cookie header copied from an authenticated rpow3.com browser session
  RPOW4_COOKIE   Cookie header copied from an authenticated rpow4.com browser session
  RPOW_COOKIE    Compatible cookie alias
  RPOW4_PRIVATE_KEY/RPOW_PRIVATE_KEY RPOW4 wallet private key or seed, base58 or hex
  RPOW4_MNEMONIC/RPOW_MNEMONIC       RPOW4 wallet mnemonic
  RPOW2_API_BASE/RPOW3_API_BASE/RPOW4_API_BASE/RPOW_API_BASE override API base URL
  RPOW2_PROXY/RPOW3_PROXY/RPOW4_PROXY/RPOW_PROXY HTTP/HTTPS proxy URL for API requests
  RPOW2_PROXY_FILE/RPOW3_PROXY_FILE/RPOW4_PROXY_FILE/RPOW_PROXY_FILE proxy pool file
  RPOW2_PROXY_CONCURRENCY/RPOW3_PROXY_CONCURRENCY/RPOW4_PROXY_CONCURRENCY/RPOW_PROXY_CONCURRENCY proxy API concurrency

Notes:
  --network rpow3 defaults to https://api.rpow3.com with Origin/Referer https://rpow3.com
  --network rpow4 defaults to https://rpow4.com with Origin/Referer https://rpow4.com
  RPOW4 mint requires both an authenticated Cookie and the matching wallet signing secret
  --proxy-concurrency sets challenge prefetch and mint workers unless they are provided explicitly
  proxy files accept one proxy URL per line; empty lines and # comments are ignored
  Windows GPU mining uses CUDA and defaults to batch 268435456, 512 threads/block, 4 nonces/thread
  --gpu-batch-size 0 enables automatic GPU batch tuning
  --gpu-max-blocks 0 uses an SM-based automatic grid size
  --gpu-no-early-exit disables cross-thread stop checks inside a CUDA batch
  --serial-api disables challenge prefetch and async mint submission for strict one-request-at-a-time API use
  --loop defaults to challenge prefetch 2 and mint workers 1 on Windows
  macOS GPU mining uses Metal"
    );
}
