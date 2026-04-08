use either::Either;

use crate::{
    DeltaEvent, Error, FinishReason, PreparedStatefulRequest, Result, RuntimeFeatures,
    SessionStats, StatefulRequest, StatefulStreamOutput,
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
        let prepared = self.prepare_text_request(request).await?;
        let output = self.run_prepared_streaming(prepared).await?;
        Ok(SingleSeqOutput {
            text: output.text,
            tokens: output.tokens,
            stats: output.stats,
            finish_reason: output.finish_reason,
        })
    }

    pub async fn run_streaming(&self, request: StatefulRequest) -> Result<StatefulStreamOutput> {
        let prepared = self.prepare_text_request(request).await?;
        self.run_prepared_streaming(prepared).await
    }

    pub async fn prepare_text_request(
        &self,
        request: StatefulRequest,
    ) -> Result<PreparedStatefulRequest> {
        let input_ids = self
            .model
            .tokenize(
                Either::Left(request.messages.clone()),
                None,
                request.add_special_tokens,
                request.add_generation_prompt,
                request.enable_thinking,
            )
            .await?;

        if input_ids.is_empty() {
            return Err(Error::EmptyPrompt);
        }

        let max_generated_tokens = request.max_generated_tokens.unwrap_or(
            request
                .session_config
                .sampling_params
                .max_len
                .unwrap_or(256),
        );

        Ok(PreparedStatefulRequest {
            session_config: request.session_config,
            input_ids,
            max_tokens: max_generated_tokens,
            skip_special_tokens: request.skip_special_tokens,
        })
    }

    pub async fn run_prepared(
        &self,
        prepared: PreparedStatefulRequest,
    ) -> Result<SingleSeqOutput> {
        let output = self.run_prepared_streaming(prepared).await?;
        Ok(SingleSeqOutput {
            text: output.text,
            tokens: output.tokens,
            stats: output.stats,
            finish_reason: output.finish_reason,
        })
    }

    pub async fn run_prepared_streaming(
        &self,
        prepared: PreparedStatefulRequest,
    ) -> Result<StatefulStreamOutput> {
        let mut session = self
            .stateful
            .new_decode_session(prepared.session_config)?;
        let prefill = self.stateful.prefill(&mut session, &prepared.input_ids)?;

        let mut next_token = *prepared.input_ids.last().expect("checked non-empty");
        let mut tokens = Vec::new();
        let mut deltas = Vec::new();
        let mut visible_text = String::new();
        let mut finish_reason = FinishReason::Length;

        for _ in 0..prepared.max_tokens {
            let step = self.stateful.decode_step(&mut session, next_token)?;
            let Some(token) = step.token else {
                finish_reason = FinishReason::Stop;
                break;
            };

            let text_delta = match step.text_delta {
                Some(delta) => {
                    if let Some(suffix) = delta.strip_prefix(&visible_text) {
                        visible_text.push_str(suffix);
                        if suffix.is_empty() {
                            None
                        } else {
                            Some(suffix.to_string())
                        }
                    } else {
                        visible_text = delta.clone();
                        if delta.is_empty() {
                            None
                        } else {
                            Some(delta)
                        }
                    }
                }
                None => {
                    let detok = self
                        .model
                        .detokenize(vec![token], prepared.skip_special_tokens)
                        .await?;
                    if detok.is_empty() {
                        None
                    } else {
                        visible_text.push_str(&detok);
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
                .detokenize(tokens.clone(), prepared.skip_special_tokens)
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
