use crate::{FinishReason, SessionStats};

#[derive(Debug, Clone)]
pub struct DeltaEvent {
    pub token: u32,
    pub text_delta: Option<String>,
    pub is_done: bool,
}

#[derive(Debug, Clone)]
pub struct StatefulStreamOutput {
    pub deltas: Vec<DeltaEvent>,
    pub tokens: Vec<u32>,
    pub text: String,
    pub stats: SessionStats,
    pub finish_reason: FinishReason,
}

