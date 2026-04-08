use either::Either;

use crate::{
    DeltaEvent, Error, FinishReason, Result, RuntimeFeatures, SessionStats, StatefulRequest,
    StatefulStreamOutput,
};

#[derive(Clone)]
pub struct StatefulRuntime {
    model: mistralrs::Model,
    stateful: mistralrs::StatefulModel,
}

#[derive(Debug, Clone)]
pub struct SingleSeqOutput {
    pub text: String,
    pub tokens: Vec<u32>,
    pub stats: SessionStats,
    pub finish_reason: FinishReason,
}

impl StatefulRuntime {
    pub fn new(model: mistralrs::Model, stateful: mistralrs::StatefulModel) -> Self {
        Self { model, stateful }
    }

    pub async fn from_text_builder(builder: mistralrs::TextModelBuilder) -> Result<Self> {
        let model = builder.clone().build().await?;
        let stateful = builder.build_stateful().await?;
        Ok(Self::new(model, stateful))
    }

    pub fn features(&self) -> RuntimeFeatures {
        self.stateful.backend_features().into()
    }

    pub fn model(&self) -> &mistralrs::Model {
        &self.model
    }

    pub fn stateful_model(&self) -> &mistralrs::StatefulModel {
        &self.stateful
    }

    pub async fn run(&self, request: StatefulRequest) -> Result<SingleSeqOutput> {
        let output = self.run_streaming(request).await?;
        Ok(SingleSeqOutput {
            text: output.text,
            tokens: output.tokens,
            stats: output.stats,
            finish_reason: output.finish_reason,
        })
    }

    pub async fn run_streaming(&self, request: StatefulRequest) -> Result<StatefulStreamOutput> {
        let prompt_tokens = self
            .model
            .tokenize(
                Either::Left(request.messages.clone()),
                None,
                request.add_special_tokens,
                request.add_generation_prompt,
                request.enable_thinking,
            )
            .await?;

        if prompt_tokens.is_empty() {
            return Err(Error::EmptyPrompt);
        }

        let mut session = self
            .stateful
            .new_decode_session(request.session_config.clone())?;
        let prefill = self.stateful.prefill(&mut session, &prompt_tokens)?;

        let mut next_token = *prompt_tokens.last().expect("checked non-empty");
        let max_generated_tokens = request.max_generated_tokens.unwrap_or(
            request
                .session_config
                .sampling_params
                .max_len
                .unwrap_or(256),
        );

        let mut tokens = Vec::new();
        let mut deltas = Vec::new();
        let mut finish_reason = FinishReason::Length;

        for _ in 0..max_generated_tokens {
            let step = self.stateful.decode_step(&mut session, next_token)?;
            let Some(token) = step.token else {
                finish_reason = FinishReason::Stop;
                break;
            };

            let text_delta = match step.text_delta {
                Some(delta) => Some(delta),
                None => {
                    let detok = self
                        .model
                        .detokenize(vec![token], request.skip_special_tokens)
                        .await?;
                    if detok.is_empty() {
                        None
                    } else {
                        Some(detok)
                    }
                }
            };

            deltas.push(DeltaEvent {
                token,
                text_delta,
                is_done: step.is_done,
            });
            tokens.push(token);
            next_token = token;

            if step.is_done {
                finish_reason = FinishReason::Stop;
                break;
            }
        }

        let text = if tokens.is_empty() {
            String::new()
        } else {
            self.model
                .detokenize(tokens.clone(), request.skip_special_tokens)
                .await?
        };

        let stats = SessionStats {
            prompt_tokens: prefill.prompt_tokens,
            cached_prompt_tokens: prefill.cached_prompt_tokens,
            completion_tokens: tokens.len(),
        };

        Ok(StatefulStreamOutput {
            deltas,
            tokens,
            text,
            stats,
            finish_reason,
        })
    }
}

