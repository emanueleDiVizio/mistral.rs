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

impl SessionStats {
    pub fn total_tokens(&self) -> usize {
        self.prompt_tokens + self.completion_tokens
    }

    pub fn merge_all(stats: impl IntoIterator<Item = SessionStats>) -> Self {
        let mut merged = Self::default();
        for stat in stats {
            merged.prompt_tokens += stat.prompt_tokens;
            merged.cached_prompt_tokens += stat.cached_prompt_tokens;
            merged.completion_tokens += stat.completion_tokens;
        }
        merged
    }
}
