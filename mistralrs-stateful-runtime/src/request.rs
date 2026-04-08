#[derive(Debug, Clone)]
pub struct StatefulRequest {
    pub messages: mistralrs::TextMessages,
    pub session_config: mistralrs::DecodeSessionConfig,
    pub add_special_tokens: bool,
    pub add_generation_prompt: bool,
    pub enable_thinking: Option<bool>,
    pub max_generated_tokens: Option<usize>,
    pub skip_special_tokens: bool,
}

impl StatefulRequest {
    pub fn new(messages: mistralrs::TextMessages) -> Self {
        Self {
            messages,
            session_config: mistralrs::DecodeSessionConfig::default(),
            add_special_tokens: true,
            add_generation_prompt: true,
            enable_thinking: None,
            max_generated_tokens: None,
            skip_special_tokens: true,
        }
    }

    pub fn with_session_config(mut self, session_config: mistralrs::DecodeSessionConfig) -> Self {
        self.session_config = session_config;
        self
    }

    pub fn with_max_generated_tokens(mut self, max_generated_tokens: usize) -> Self {
        self.max_generated_tokens = Some(max_generated_tokens);
        self
    }

    pub fn with_add_special_tokens(mut self, add_special_tokens: bool) -> Self {
        self.add_special_tokens = add_special_tokens;
        self
    }

    pub fn with_add_generation_prompt(mut self, add_generation_prompt: bool) -> Self {
        self.add_generation_prompt = add_generation_prompt;
        self
    }

    pub fn with_enable_thinking(mut self, enable_thinking: bool) -> Self {
        self.enable_thinking = Some(enable_thinking);
        self
    }

    pub fn with_skip_special_tokens(mut self, skip_special_tokens: bool) -> Self {
        self.skip_special_tokens = skip_special_tokens;
        self
    }
}

