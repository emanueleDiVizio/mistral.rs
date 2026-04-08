#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
}

#[derive(Debug, Clone, Default)]
pub struct SessionStats {
    pub prompt_tokens: usize,
    pub cached_prompt_tokens: usize,
    pub completion_tokens: usize,
}

