use either::Either;

use crate::{
    DeltaEvent, Error, FinishReason, PreparedStatefulRequest, Result, RuntimeFeatures,
    SessionStats, StatefulBatchCompletion, StatefulChoice, StatefulCompletion, StatefulRequest,
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

struct ActiveSequence {
    index: usize,
    session: mistralrs::DecodeSession,
    last_token: u32,
    remaining_tokens: usize,
    tokens: Vec<u32>,
    deltas: Vec<DeltaEvent>,
    visible_text: String,
    stats: SessionStats,
    finish_reason: FinishReason,
    done: bool,
    skip_special_tokens: bool,
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

    pub fn model_name(&self) -> &str {
        self.stateful.model_id()
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

    pub async fn complete(&self, request: StatefulRequest) -> Result<StatefulCompletion> {
        let output = self.run(request).await?;
        Ok(StatefulCompletion {
            model: self.model_name().to_string(),
            choice: StatefulChoice {
                text: output.text,
                finish_reason: output.finish_reason,
            },
            usage: output.stats,
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

    pub async fn run_batch(
        &self,
        requests: Vec<StatefulRequest>,
    ) -> Result<Vec<SingleSeqOutput>> {
        let mut prepared = Vec::with_capacity(requests.len());
        for request in requests {
            prepared.push(self.prepare_text_request(request).await?);
        }
        self.run_prepared_batch(prepared).await
    }

    pub async fn complete_batch(
        &self,
        requests: Vec<StatefulRequest>,
    ) -> Result<StatefulBatchCompletion> {
        let outputs = self.run_batch(requests).await?;
        let usage = SessionStats::merge_all(outputs.iter().map(|out| out.stats.clone()));
        let choices = outputs
            .into_iter()
            .map(|out| StatefulChoice {
                text: out.text,
                finish_reason: out.finish_reason,
            })
            .collect();
        Ok(StatefulBatchCompletion {
            model: self.model_name().to_string(),
            choices,
            usage,
        })
    }

    pub async fn run_prepared_batch(
        &self,
        prepared: Vec<PreparedStatefulRequest>,
    ) -> Result<Vec<SingleSeqOutput>> {
        let mut active = Vec::with_capacity(prepared.len());
        let mut completed = vec![None; prepared.len()];

        for (index, prepared) in prepared.into_iter().enumerate() {
            if prepared.input_ids.is_empty() {
                return Err(Error::EmptyPrompt);
            }

            let mut session = self
                .stateful
                .new_decode_session(prepared.session_config)?;
            let prefill = self.stateful.prefill(&mut session, &prepared.input_ids)?;

            active.push(ActiveSequence {
                index,
                session,
                last_token: *prepared.input_ids.last().expect("checked non-empty"),
                remaining_tokens: prepared.max_tokens,
                tokens: Vec::new(),
                deltas: Vec::new(),
                visible_text: String::new(),
                stats: SessionStats {
                    prompt_tokens: prefill.prompt_tokens,
                    cached_prompt_tokens: prefill.cached_prompt_tokens,
                    completion_tokens: 0,
                },
                finish_reason: FinishReason::Length,
                done: false,
                skip_special_tokens: prepared.skip_special_tokens,
            });
        }

        while !active.is_empty() {
            let token_ids = active.iter().map(|seq| seq.last_token).collect::<Vec<_>>();
            let mut sessions = active
                .iter_mut()
                .map(|seq| &mut seq.session)
                .collect::<Vec<_>>();
            let batch = self.stateful.decode_batch(&mut sessions, &token_ids)?;

            if batch.outputs.len() != active.len() {
                return Err(Error::Build(anyhow::anyhow!(
                    "decode_batch returned {} outputs for {} active sessions",
                    batch.outputs.len(),
                    active.len()
                )));
            }

            for (sequence, output) in active.iter_mut().zip(batch.outputs.into_iter()) {
                let Some(token) = output.token else {
                    sequence.finish_reason = FinishReason::Stop;
                    sequence.done = true;
                    continue;
                };

                sequence.stats.completion_tokens += 1;
                sequence.remaining_tokens = sequence.remaining_tokens.saturating_sub(1);

                let text_delta = match output.text_delta {
                    Some(delta) => {
                        if let Some(suffix) = delta.strip_prefix(&sequence.visible_text) {
                            sequence.visible_text.push_str(suffix);
                            if suffix.is_empty() {
                                None
                            } else {
                                Some(suffix.to_string())
                            }
                        } else {
                            sequence.visible_text = delta.clone();
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
                            .detokenize(vec![token], sequence.skip_special_tokens)
                            .await?;
                        if detok.is_empty() {
                            None
                        } else {
                            sequence.visible_text.push_str(&detok);
                            Some(detok)
                        }
                    }
                };

                sequence.deltas.push(DeltaEvent {
                    token,
                    text_delta,
                    is_done: output.is_done,
                });
                sequence.tokens.push(token);
                sequence.last_token = token;

                if output.is_done {
                    sequence.finish_reason = FinishReason::Stop;
                    sequence.done = true;
                } else if sequence.remaining_tokens == 0 {
                    sequence.finish_reason = FinishReason::Length;
                    sequence.done = true;
                }
            }

            let mut i = 0usize;
            while i < active.len() {
                if active[i].done {
                    let sequence = active.swap_remove(i);
                    completed[sequence.index] = Some(SingleSeqOutput {
                        text: sequence.visible_text,
                        tokens: sequence.tokens,
                        stats: sequence.stats,
                        finish_reason: sequence.finish_reason,
                    });
                } else {
                    i += 1;
                }
            }
        }

        completed
            .into_iter()
            .map(|output| {
                output.ok_or_else(|| Error::Build(anyhow::anyhow!("batch output missing completion")))
            })
            .collect()
    }
}
