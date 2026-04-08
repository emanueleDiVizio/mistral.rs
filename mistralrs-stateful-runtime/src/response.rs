use crate::{FinishReason, SessionStats};

#[derive(Debug, Clone)]
pub struct StatefulChoice {
    pub text: String,
    pub finish_reason: FinishReason,
}

#[derive(Debug, Clone)]
pub struct StatefulCompletion {
    pub model: String,
    pub choice: StatefulChoice,
    pub usage: SessionStats,
}

#[derive(Debug, Clone)]
pub struct StatefulBatchCompletion {
    pub model: String,
    pub choices: Vec<StatefulChoice>,
    pub usage: SessionStats,
}
