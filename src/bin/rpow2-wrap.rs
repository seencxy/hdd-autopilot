use std::env;
use std::thread;
use std::time::{Duration, Instant};

use crossterm as _;
use hdd_autopilot as _;
#[cfg(not(target_os = "macos"))]
use iana_time_zone as _;
use mining as _;
use rand::Rng;
use reqwest::blocking::Client;
use reqwest::header::{
    ACCEPT, ACCEPT_LANGUAGE, CONTENT_TYPE, HeaderMap, HeaderValue, ORIGIN, REFERER, USER_AGENT,
};
use serde as _;
use serde_json::{Value, json};
use time as _;
use unicode_width as _;
use url as _;

const API_URL: &str = "https://api.rpow2.com/srpow/wrap";
const ORIGIN_URL: &str = "https://rpow2.com";
const REFERER_URL: &str = "https://rpow2.com/";
const USER_AGENT_STR: &str = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36";
const AMOUNT: &str = "4300000000000";
const INTERVAL_SECS: u64 = 60;

fn main() {
    if let Err(e) = run() {
        eprintln!("rpow2-wrap: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_usage();
        return Ok(());
    }

    let cookie = parse_cookie(&args).ok_or(
        "missing cookie; pass --cookie '<value>' or set RPOW_COOKIE / RPOW2_COOKIE env var",
    )?;

    let amount = args
        .windows(2)
        .find(|w| w[0] == "--amount")
        .map(|w| w[1].as_str())
        .unwrap_or(AMOUNT)
        .to_string();

    let client = build_client(&cookie)?;

    println!("rpow2-wrap: starting, calling {API_URL} every {INTERVAL_SECS}s");
    println!("rpow2-wrap: amount={amount}  Ctrl-C to stop");

    let mut attempt: u64 = 0;
    loop {
        attempt += 1;
        let idempotency_key = new_uuid_v4();
        let start = Instant::now();
        let result = call_wrap(&client, &amount, &idempotency_key);
        let elapsed = start.elapsed();

        match result {
            Ok(body) => {
                if let Some(err_code) = body.get("error").and_then(|v| v.as_str()) {
                    let status = body
                        .get("status")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown");
                    let reason = body
                        .get("failure_reason")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    eprintln!(
                        "[{attempt}] FAILED ({elapsed:.0?}) key={idempotency_key} error={err_code} status={status} reason={reason}",
                        elapsed = elapsed
                    );
                } else {
                    println!(
                        "[{attempt}] OK ({elapsed:.0?}) key={idempotency_key} response={}",
                        body,
                        elapsed = elapsed
                    );
                }
            }
            Err(e) => {
                eprintln!(
                    "[{attempt}] ERROR ({elapsed:.0?}) key={idempotency_key} {e}",
                    elapsed = elapsed
                );
            }
        }

        thread::sleep(Duration::from_secs(INTERVAL_SECS));
    }
}

fn call_wrap(
    client: &Client,
    amount: &str,
    idempotency_key: &str,
) -> Result<Value, Box<dyn std::error::Error>> {
    let body = json!({
        "amount_base_units": amount,
        "idempotency_key": idempotency_key,
    });
    let resp = client.post(API_URL).json(&body).send()?;
    let text = resp.text()?;
    let value: Value = serde_json::from_str(&text).unwrap_or_else(|_| json!({"raw": text}));
    Ok(value)
}

fn build_client(cookie: &str) -> Result<Client, Box<dyn std::error::Error>> {
    let mut headers = HeaderMap::new();
    headers.insert(ACCEPT, HeaderValue::from_static("*/*"));
    headers.insert(ACCEPT_LANGUAGE, HeaderValue::from_static("zh-CN,zh;q=0.9"));
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    headers.insert(ORIGIN, HeaderValue::from_static(ORIGIN_URL));
    headers.insert(REFERER, HeaderValue::from_static(REFERER_URL));
    headers.insert(USER_AGENT, HeaderValue::from_str(USER_AGENT_STR)?);
    headers.insert("sec-fetch-dest", HeaderValue::from_static("empty"));
    headers.insert("sec-fetch-mode", HeaderValue::from_static("cors"));
    headers.insert("sec-fetch-site", HeaderValue::from_static("same-site"));
    headers.insert("priority", HeaderValue::from_static("u=1, i"));
    headers.insert("sec-ch-ua-mobile", HeaderValue::from_static("?0"));
    headers.insert("sec-ch-ua-platform", HeaderValue::from_static("\"macOS\""));
    headers.insert(
        "sec-ch-ua",
        HeaderValue::from_static("\"Not/A)Brand\";v=\"99\", \"Chromium\";v=\"148\""),
    );

    let cookie_header = format!("rpow_session={cookie}");
    headers.insert(
        reqwest::header::COOKIE,
        HeaderValue::from_str(&cookie_header)?,
    );

    let client = Client::builder()
        .default_headers(headers)
        .timeout(Duration::from_secs(30))
        .build()?;
    Ok(client)
}

fn parse_cookie(args: &[String]) -> Option<String> {
    // --cookie <value>
    if let Some(val) = args
        .windows(2)
        .find(|w| w[0] == "--cookie")
        .map(|w| w[1].clone())
    {
        return Some(val);
    }
    // env vars
    env::var("RPOW2_COOKIE")
        .or_else(|_| env::var("RPOW_COOKIE"))
        .ok()
}

fn new_uuid_v4() -> String {
    let mut rng = rand::rng();
    let a: u32 = rng.random();
    let b: u16 = rng.random();
    let c: u16 = (rng.random::<u16>() & 0x0fff) | 0x4000;
    let d: u16 = (rng.random::<u16>() & 0x3fff) | 0x8000;
    let e: u64 = rng.random::<u64>() & 0x0000_ffff_ffff_ffff;
    format!("{a:08x}-{b:04x}-{c:04x}-{d:04x}-{e:012x}")
}

fn print_usage() {
    println!(
        "Usage: rpow2-wrap --cookie '<session-cookie>' [--amount <base-units>]

Options:
  --cookie <value>   rpow_session cookie value (or set RPOW2_COOKIE / RPOW_COOKIE)
  --amount <value>   amount_base_units to send (default: {AMOUNT})
  -h, --help         show this help

The program calls {API_URL} once per minute with a fresh
idempotency_key, printing OK or the error details for each attempt."
    );
}
