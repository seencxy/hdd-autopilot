use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use crate::backend::CpuBackend;

pub(crate) const DEFAULT_BASE_URL: &str = "https://sub.hdd.sb";
pub(crate) const DEFAULT_USER_AGENT: &str =
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36";
pub(crate) const DEFAULT_INVITE_OUTPUT_FILE: &str = "var/data/mining/invite-codes.txt";
pub(crate) const DEFAULT_BALANCE_OUTPUT_FILE: &str = "var/data/mining/balance-codes.txt";
pub(crate) const DEFAULT_BENCHMARK_CACHE_FILE: &str = "var/data/mining/benchmark-cache.json";

#[derive(Clone)]
pub struct OutputSink {
    write_line: Arc<dyn Fn(String) + Send + Sync>,
}

impl OutputSink {
    pub fn stdout() -> Self {
        Self::new(|line| println!("{}", line))
    }

    pub fn new<F>(write_line: F) -> Self
    where
        F: Fn(String) + Send + Sync + 'static,
    {
        Self {
            write_line: Arc::new(write_line),
        }
    }

    pub fn line(&self, line: impl AsRef<str>) {
        let text = line.as_ref();
        if text.is_empty() {
            self.write(String::new());
            return;
        }
        for line in text.lines() {
            self.write(line.to_string());
        }
    }

    pub(crate) fn line_fmt(&self, args: fmt::Arguments<'_>) {
        self.line(args.to_string());
    }

    fn write(&self, line: String) {
        (self.write_line)(line);
    }
}

impl fmt::Debug for OutputSink {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("OutputSink")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    InviteThenBalance,
    BalanceThenInvite,
    InviteOnly,
    BalanceOnly,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub base_url: String,
    pub invite_output_file: PathBuf,
    pub balance_output_file: PathBuf,
    pub benchmark_cache_file: PathBuf,
    pub thread_count: usize,
    pub http_timeout: Duration,
    pub heartbeat_interval: Duration,
    pub progress_interval: Duration,
    pub retry_delay: Duration,
    pub success_delay: Duration,
    pub daily_limit_delay: Duration,
    pub inventory_depleted_delay: Duration,
    pub round_status_poll_interval: Duration,
    pub output: OutputSink,
    pub mode: Mode,
}

#[derive(Debug, Clone)]
pub(crate) struct RewardKind {
    pub(crate) name: &'static str,
    pub(crate) preference: &'static str,
    pub(crate) output_path: PathBuf,
}

impl Config {
    pub(crate) fn reward_kinds(&self) -> Vec<RewardKind> {
        let invite = RewardKind {
            name: "邀请码",
            preference: "invite",
            output_path: self.invite_output_file.clone(),
        };
        let balance = RewardKind {
            name: "余额兑换码",
            preference: "balance",
            output_path: self.balance_output_file.clone(),
        };
        match self.mode {
            Mode::BalanceThenInvite => vec![balance, invite],
            Mode::InviteOnly => vec![invite],
            Mode::BalanceOnly => vec![balance],
            Mode::InviteThenBalance => vec![invite, balance],
        }
    }
}

pub fn default_config_for_mode(mode: Mode) -> Config {
    default_config(mode)
}

pub(crate) fn default_config(mode: Mode) -> Config {
    Config {
        base_url: DEFAULT_BASE_URL.to_string(),
        invite_output_file: PathBuf::from(DEFAULT_INVITE_OUTPUT_FILE),
        balance_output_file: PathBuf::from(DEFAULT_BALANCE_OUTPUT_FILE),
        benchmark_cache_file: PathBuf::from(DEFAULT_BENCHMARK_CACHE_FILE),
        thread_count: CpuBackend::default_thread_count(),
        http_timeout: Duration::from_secs(30),
        heartbeat_interval: Duration::from_secs(4),
        progress_interval: Duration::from_secs(10),
        retry_delay: Duration::from_secs(3),
        success_delay: Duration::from_secs(3),
        daily_limit_delay: Duration::from_secs(60),
        inventory_depleted_delay: Duration::from_secs(60),
        round_status_poll_interval: Duration::from_millis(500),
        output: OutputSink::stdout(),
        mode,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_writes_codes_to_var_data() {
        let config = default_config(Mode::InviteThenBalance);

        assert_eq!(
            config.invite_output_file,
            PathBuf::from(DEFAULT_INVITE_OUTPUT_FILE)
        );
        assert_eq!(
            config.balance_output_file,
            PathBuf::from(DEFAULT_BALANCE_OUTPUT_FILE)
        );
    }
}
