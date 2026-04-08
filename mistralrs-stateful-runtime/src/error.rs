use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error(transparent)]
    Build(#[from] anyhow::Error),
    #[error(transparent)]
    Inference(#[from] mistralrs::error::Error),
    #[error(transparent)]
    Stateful(#[from] mistralrs::core::MistralRsError),
    #[error("tokenization produced no prompt tokens")]
    EmptyPrompt,
}

pub type Result<T> = std::result::Result<T, Error>;
