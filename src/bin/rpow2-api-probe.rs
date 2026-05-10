use std::collections::BTreeMap;
use std::env;
use std::error::Error;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use crossterm as _;
use hdd_autopilot as _;
#[cfg(not(target_os = "macos"))]
use iana_time_zone as _;
use mining::MiningError;
use mining::rpow2::{Rpow2Client, load_rpow2_proxy_file};
use rand as _;
use reqwest as _;
use serde as _;
use serde_json::{Value, json};
use time as _;
use unicode_width as _;
use url::Url;

const DEFAULT_PROXY_FILE: &str = "rpow2-proxies.txt";

fn main() {
    if let Err(error) = run() {
        eprintln!("rpow2-api-probe: {}", error_chain_to_string(error.as_ref()));
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let args = Args::parse(env::args().skip(1).collect())?;
    if args.help {
        print_usage();
        return Ok(());
    }

    let target = ApiTarget::from_args(&args)?;
    let endpoint = args.endpoint.unwrap_or(Endpoint::Me);
    let cookie = load_cookie(&args, target.network);
    if endpoint.requires_cookie() && cookie.is_none() {
        return Err(format!(
            "missing RPOW cookie; pass --cookie or set {}/RPOW_COOKIE",
            target.network.cookie_env()
        )
        .into());
    }
    let (proxy_urls, _) = load_proxy_pool(&args, target.network)?;
    let requests = args.requests.unwrap_or(5).max(1);
    let concurrency = args.concurrency.unwrap_or(1).max(1).min(requests);
    let client = Arc::new(Rpow2Client::with_base_url_origin_and_proxy_pool(
        &target.api_base,
        cookie.as_deref(),
        target.origin.as_deref(),
        target.referer.as_deref(),
        &proxy_urls,
    )?);

    println!(
        "network={} endpoint={} requests={} concurrency={} proxy={}",
        target.network.label(),
        endpoint.label(),
        requests,
        concurrency,
        if client.proxy_pool_size() > 0 {
            "enabled"
        } else {
            "disabled"
        }
    );
    if endpoint == Endpoint::Challenge {
        eprintln!("note: /challenge creates challenges; keep request counts small for diagnostics");
    }

    let started = Instant::now();
    let next_request = Arc::new(AtomicUsize::new(0));
    let (sender, receiver) = mpsc::channel();
    for _ in 0..concurrency {
        let client = Arc::clone(&client);
        let next_request = Arc::clone(&next_request);
        let sender = sender.clone();
        thread::spawn(move || {
            loop {
                let request_index = next_request.fetch_add(1, Ordering::SeqCst);
                if request_index >= requests {
                    break;
                }
                let request_started = Instant::now();
                let status = match endpoint.call(&client) {
                    Ok(_) => "ok".to_string(),
                    Err(error) => classify_error(&error),
                };
                let elapsed = request_started.elapsed();
                if sender
                    .send(ProbeResult {
                        request_number: request_index + 1,
                        status,
                        elapsed,
                    })
                    .is_err()
                {
                    break;
                }
            }
        });
    }
    drop(sender);

    let mut counts = BTreeMap::<String, usize>::new();
    for result in receiver {
        println!(
            "request={} status={} elapsed_ms={:.1}",
            result.request_number,
            result.status,
            result.elapsed.as_secs_f64() * 1000.0
        );
        *counts.entry(result.status).or_default() += 1;
    }

    println!("elapsed_seconds={:.3}", started.elapsed().as_secs_f64());
    for (status, count) in counts {
        println!("summary {}={}", status, count);
    }
    Ok(())
}

fn load_cookie(args: &Args, network: RpowNetwork) -> Option<String> {
    args.cookie
        .clone()
        .or_else(|| env::var(network.cookie_env()).ok())
        .or_else(|| env::var("RPOW_COOKIE").ok())
}

fn load_proxy_pool(
    args: &Args,
    network: RpowNetwork,
) -> Result<(Vec<String>, String), Box<dyn Error>> {
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

#[derive(Debug, Clone)]
struct ApiTarget {
    network: RpowNetwork,
    api_base: String,
    origin: Option<String>,
    referer: Option<String>,
}

impl ApiTarget {
    fn from_args(args: &Args) -> Result<Self, Box<dyn Error>> {
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
    fn parse(value: &str) -> Result<Self, Box<dyn Error>> {
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
}

#[derive(Debug)]
struct ProbeResult {
    request_number: usize,
    status: String,
    elapsed: Duration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Endpoint {
    Me,
    Ledger,
    Challenge,
}

impl Endpoint {
    fn parse(value: &str) -> Result<Self, Box<dyn Error>> {
        match value {
            "me" => Ok(Self::Me),
            "ledger" => Ok(Self::Ledger),
            "challenge" => Ok(Self::Challenge),
            _ => {
                Err(format!("unknown endpoint: {value}; expected me, ledger, or challenge").into())
            }
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Me => "me",
            Self::Ledger => "ledger",
            Self::Challenge => "challenge",
        }
    }

    fn requires_cookie(self) -> bool {
        !matches!(self, Self::Ledger)
    }

    fn call(self, client: &Rpow2Client) -> Result<Value, MiningError> {
        match self {
            Self::Me => client.me(),
            Self::Ledger => client.ledger(),
            Self::Challenge => {
                let challenge = client.challenge()?;
                Ok(json!({
                    "challenge_id": challenge.challenge_id,
                    "nonce_prefix": challenge.nonce_prefix,
                    "difficulty_bits": challenge.difficulty_bits
                }))
            }
        }
    }
}

#[derive(Debug, Default)]
struct Args {
    cookie: Option<String>,
    network: Option<RpowNetwork>,
    api_base: Option<String>,
    origin: Option<String>,
    referer: Option<String>,
    proxy: Option<String>,
    proxy_file: Option<String>,
    endpoint: Option<Endpoint>,
    requests: Option<usize>,
    concurrency: Option<usize>,
    help: bool,
}

impl Args {
    fn parse(raw: Vec<String>) -> Result<Self, Box<dyn Error>> {
        let mut args = Self::default();
        let mut index = 0;
        while index < raw.len() {
            match raw[index].as_str() {
                "--cookie" => {
                    index += 1;
                    args.cookie = Some(next_value(&raw, index, "--cookie")?.to_string());
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
                "--endpoint" => {
                    index += 1;
                    args.endpoint = Some(Endpoint::parse(next_value(&raw, index, "--endpoint")?)?);
                }
                "--requests" | "-n" => {
                    index += 1;
                    args.requests =
                        Some(next_value(&raw, index, raw[index - 1].as_str())?.parse()?);
                }
                "--concurrency" | "-c" => {
                    index += 1;
                    args.concurrency =
                        Some(next_value(&raw, index, raw[index - 1].as_str())?.parse()?);
                }
                "-h" | "--help" => args.help = true,
                other => return Err(format!("unknown argument: {other}").into()),
            }
            index += 1;
        }
        Ok(args)
    }
}

fn classify_error(error: &MiningError) -> String {
    let message = error_chain_to_string(error);
    for status in [400, 401, 403, 408, 425, 429, 500, 502, 503, 504] {
        if message.contains(&format!("状态码 {status}")) {
            return format!("http_{status}");
        }
    }
    let lower = message.to_ascii_lowercase();
    if lower.contains("timed out") || lower.contains("operation timed out") {
        "timeout".to_string()
    } else if lower.contains("connect") {
        "connect_error".to_string()
    } else {
        "error".to_string()
    }
}

fn next_value<'a>(raw: &'a [String], index: usize, flag: &str) -> Result<&'a str, Box<dyn Error>> {
    raw.get(index)
        .map(String::as_str)
        .ok_or_else(|| format!("{flag} requires a value").into())
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

fn print_usage() {
    println!(
        "Usage:
  rpow2-api-probe [--cookie '<cookie-header>'] [--network rpow2|rpow3|rpow4] [--api-base <url>] [--proxy <url>] [--proxy-file <path>] [--endpoint me|ledger|challenge] [--requests <n>] [--concurrency <n>]

Environment:
  RPOW2_COOKIE   Cookie header copied from an authenticated rpow2.com browser session
  RPOW3_COOKIE   Cookie header copied from an authenticated rpow3.com browser session
  RPOW4_COOKIE   Cookie header copied from an authenticated rpow4.com browser session
  RPOW_COOKIE    Compatible cookie env alias
  RPOW2_API_BASE/RPOW3_API_BASE/RPOW4_API_BASE/RPOW_API_BASE override API base URL
  RPOW2_PROXY/RPOW3_PROXY/RPOW4_PROXY/RPOW_PROXY HTTP/HTTPS proxy URL for API requests
  RPOW2_PROXY_FILE/RPOW3_PROXY_FILE/RPOW4_PROXY_FILE/RPOW_PROXY_FILE proxy pool file

Defaults:
  endpoint       me
  requests       5
  concurrency    1"
    );
}
