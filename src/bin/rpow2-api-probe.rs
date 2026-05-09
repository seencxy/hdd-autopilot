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
use url as _;

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

    let endpoint = args.endpoint.unwrap_or(Endpoint::Me);
    let cookie = args
        .cookie
        .clone()
        .or_else(|| env::var("RPOW2_COOKIE").ok())
        .or_else(|| env::var("RPOW_COOKIE").ok());
    if endpoint.requires_cookie() && cookie.is_none() {
        return Err("missing RPOW2 cookie; pass --cookie or set RPOW2_COOKIE/RPOW_COOKIE".into());
    }
    let (proxy_urls, _) = load_proxy_pool(&args)?;
    let requests = args.requests.unwrap_or(5).max(1);
    let concurrency = args.concurrency.unwrap_or(1).max(1).min(requests);
    let client = Arc::new(Rpow2Client::new_with_proxy_pool(
        cookie.as_deref(),
        &proxy_urls,
    )?);

    println!(
        "endpoint={} requests={} concurrency={} proxy={}",
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

fn load_proxy_pool(args: &Args) -> Result<(Vec<String>, String), Box<dyn Error>> {
    let proxy_file = args
        .proxy_file
        .clone()
        .or_else(|| env::var("RPOW2_PROXY_FILE").ok())
        .unwrap_or_else(|| DEFAULT_PROXY_FILE.to_string());
    let mut proxies = load_rpow2_proxy_file(&proxy_file)?;
    if let Some(proxy) = args
        .proxy
        .clone()
        .or_else(|| env::var("RPOW2_PROXY").ok())
        .map(|proxy| proxy.trim().to_string())
        .filter(|proxy| !proxy.is_empty())
    {
        proxies.push(proxy);
    }
    Ok((proxies, proxy_file))
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
  rpow2-api-probe [--cookie '<cookie-header>'] [--proxy <url>] [--proxy-file <path>] [--endpoint me|ledger|challenge] [--requests <n>] [--concurrency <n>]

Environment:
  RPOW2_COOKIE   Cookie header copied from an authenticated rpow2.com browser session
  RPOW_COOKIE    Compatible cookie env alias
  RPOW2_PROXY    HTTP/HTTPS proxy URL for RPOW2 API requests
  RPOW2_PROXY_FILE  Proxy pool file; defaults to rpow2-proxies.txt

Defaults:
  endpoint       me
  requests       5
  concurrency    1"
    );
}
