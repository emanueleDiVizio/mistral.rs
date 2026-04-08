use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error(transparent)]
    Build(#[from] anyhow::Error),
    #[error(transparent)]
    Inference(#[from] mistralrs::error::Error),
    #[error(transparent)]
    Stateful(#[from] mistralrs::core::MistralRsError),
    #[error("{0}")]
    Runtime(String),
    #[error("tokenization produced no prompt tokens")]
    EmptyPrompt,
    #[error("stateful scheduler is unavailable")]
    SchedulerUnavailable,
    #[error("stateful scheduler dropped the response")]
    SchedulerDropped,
}

pub type Result<T> = std::result::Result<T, Error>;
