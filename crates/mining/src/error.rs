use std::io;

#[derive(Debug)]
pub enum MiningError {
    Io(io::Error),
    Http(reqwest::Error),
    Json(serde_json::Error),
    PoolDisabled,
    NoOpenRound,
    InventoryDepleted,
    RoundClosed,
    DailyLimit,
    RetryNow,
    ChallengeRejected(String),
    Message(String),
}

impl std::fmt::Display for MiningError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => error.fmt(f),
            Self::Http(error) => error.fmt(f),
            Self::Json(error) => error.fmt(f),
            Self::PoolDisabled => f.write_str("矿池当前未开放"),
            Self::NoOpenRound => f.write_str("当前没有开放轮次"),
            Self::InventoryDepleted => f.write_str("当前邀请码和余额兑换码库存都已耗尽"),
            Self::RoundClosed => f.write_str("当前轮次已关闭"),
            Self::DailyLimit => f.write_str("今日命中次数已达上限"),
            Self::RetryNow => f.write_str("立即重试"),
            Self::ChallengeRejected(message) => write!(f, "挑战被矿池拒绝：{}", message),
            Self::Message(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for MiningError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Http(error) => Some(error),
            Self::Json(error) => Some(error),
            _ => None,
        }
    }
}

impl From<io::Error> for MiningError {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<reqwest::Error> for MiningError {
    fn from(value: reqwest::Error) -> Self {
        Self::Http(value)
    }
}

impl From<serde_json::Error> for MiningError {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}

pub(crate) fn interrupted_error() -> MiningError {
    MiningError::Io(io::Error::new(io::ErrorKind::Interrupted, "interrupted"))
}

pub(crate) fn is_interrupted_error(error: &MiningError) -> bool {
    matches!(error, MiningError::Io(io_error) if io_error.kind() == io::ErrorKind::Interrupted)
}
