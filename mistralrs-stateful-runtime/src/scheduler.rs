use std::collections::VecDeque;

use tokio::sync::{mpsc, oneshot};

use crate::{
    Error, FinishReason, PreparedStatefulRequest, Result, SessionStats, StatefulChoice,
    StatefulCompletion, StatefulRequest, StatefulRuntime,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefillPolicy {
    PrefillFirst,
    Interleave,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FairnessPolicy {
    RoundRobin,
    OldestFirst,
}

#[derive(Debug, Clone)]
pub struct SchedulerRequest {
    pub request: StatefulRequest,
    pub request_id: String,
}

#[derive(Debug, Clone)]
pub struct SchedulerSketch {
    pub max_batch_size: usize,
    pub prefill_policy: PrefillPolicy,
    pub fairness_policy: FairnessPolicy,
}

impl Default for SchedulerSketch {
    fn default() -> Self {
        Self {
            max_batch_size: 1,
            prefill_policy: PrefillPolicy::PrefillFirst,
            fairness_policy: FairnessPolicy::RoundRobin,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    pub max_batch_size: usize,
    pub prefill_budget: usize,
    pub prefill_policy: PrefillPolicy,
    pub fairness_policy: FairnessPolicy,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 4,
            prefill_budget: 1,
            prefill_policy: PrefillPolicy::PrefillFirst,
            fairness_policy: FairnessPolicy::RoundRobin,
        }
    }
}

struct QueuedRequest {
    request: StatefulRequest,
    request_id: Option<String>,
    response_tx: oneshot::Sender<Result<StatefulCompletion>>,
}

struct ActiveRequest {
    request_id: Option<String>,
    session: mistralrs::DecodeSession,
    last_token: u32,
    remaining_tokens: usize,
    content: String,
    completion_tokens: usize,
    prompt_tokens: usize,
    response_tx: oneshot::Sender<Result<StatefulCompletion>>,
    skip_special_tokens: bool,
}

#[derive(Clone)]
pub struct StatefulScheduler {
    tx: mpsc::UnboundedSender<QueuedRequest>,
}

impl StatefulScheduler {
    pub fn new(runtime: StatefulRuntime, config: SchedulerConfig) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        tokio::spawn(run_scheduler_loop(runtime, config, rx));
        Self { tx }
    }

    pub async fn submit(&self, request: StatefulRequest) -> Result<StatefulCompletion> {
        self.submit_inner(request, None).await
    }

    pub async fn submit_named(
        &self,
        request_id: impl Into<String>,
        request: StatefulRequest,
    ) -> Result<StatefulCompletion> {
        self.submit_inner(request, Some(request_id.into())).await
    }

    async fn submit_inner(
        &self,
        request: StatefulRequest,
        request_id: Option<String>,
    ) -> Result<StatefulCompletion> {
        let (tx, rx) = oneshot::channel();
        self.tx
            .send(QueuedRequest {
                request,
                request_id,
                response_tx: tx,
            })
            .map_err(|_| Error::SchedulerUnavailable)?;
        rx.await.map_err(|_| Error::SchedulerDropped)?
    }
}

async fn run_scheduler_loop(
    runtime: StatefulRuntime,
    config: SchedulerConfig,
    mut rx: mpsc::UnboundedReceiver<QueuedRequest>,
) {
    let mut waiting = VecDeque::new();
    let mut active = VecDeque::new();

    loop {
        if active.is_empty() && waiting.is_empty() {
            let Some(req) = rx.recv().await else {
                break;
            };
            waiting.push_back(req);
        }

        while let Ok(req) = rx.try_recv() {
            waiting.push_back(req);
        }

        for _ in 0..config.prefill_budget.min(waiting.len()) {
            let Some(req) = waiting.pop_front() else {
                break;
            };
            match prepare_active_request(&runtime, req).await {
                Ok(active_request) => active.push_back(active_request),
                Err((req_id, response_tx, err)) => {
                    let _ = response_tx.send(Err(err.with_request_context(req_id)));
                }
            }
        }

        if active.is_empty() {
            continue;
        }

        let batch_len = config.max_batch_size.min(active.len());
        let mut batch = active.drain(..batch_len).collect::<Vec<_>>();
        let token_ids = batch.iter().map(|req| req.last_token).collect::<Vec<_>>();
        let mut sessions = batch
            .iter_mut()
            .map(|req| &mut req.session)
            .collect::<Vec<_>>();

        let batch_out = match runtime.stateful_model().decode_batch(&mut sessions, &token_ids) {
            Ok(out) => out,
            Err(err) => {
                let message = err.to_string();
                for req in batch {
                    let _ = req
                        .response_tx
                        .send(Err(Error::Runtime(message.clone()).with_request_context(req.request_id)));
                }
                continue;
            }
        };

        for (mut req, out) in batch.into_iter().zip(batch_out.outputs.into_iter()) {
            let token = out.token;
            if token.is_some() {
                req.completion_tokens += 1;
                req.remaining_tokens = req.remaining_tokens.saturating_sub(1);
            }

            let merge_result = merge_text_delta(
                runtime.model(),
                &mut req.content,
                out.text_delta,
                token,
                req.skip_special_tokens,
            )
            .await;

            if let Err(err) = merge_result {
                let _ = req
                    .response_tx
                    .send(Err(err.with_request_context(req.request_id)));
                continue;
            }

            let should_finish = out.is_done || token.is_none() || req.remaining_tokens == 0;
            if should_finish {
                let finish_reason = if out.is_done {
                    FinishReason::Stop
                } else {
                    FinishReason::Length
                };
                let _ = req.response_tx.send(Ok(StatefulCompletion {
                    model: runtime.model_name().to_string(),
                    choice: StatefulChoice {
                        text: req.content,
                        finish_reason,
                    },
                    usage: SessionStats {
                        prompt_tokens: req.prompt_tokens,
                        cached_prompt_tokens: 0,
                        completion_tokens: req.completion_tokens,
                    },
                }));
                continue;
            }

            if let Some(token) = token {
                req.last_token = token;
                active.push_back(req);
            } else {
                let _ = req.response_tx.send(Ok(StatefulCompletion {
                    model: runtime.model_name().to_string(),
                    choice: StatefulChoice {
                        text: req.content,
                        finish_reason: FinishReason::Stop,
                    },
                    usage: SessionStats {
                        prompt_tokens: req.prompt_tokens,
                        cached_prompt_tokens: 0,
                        completion_tokens: req.completion_tokens,
                    },
                }));
            }
        }
    }
}

async fn prepare_active_request(
    runtime: &StatefulRuntime,
    req: QueuedRequest,
) -> std::result::Result<ActiveRequest, (Option<String>, oneshot::Sender<Result<StatefulCompletion>>, Error)>
{
    let prepared = match runtime.prepare_text_request(req.request).await {
        Ok(prepared) => prepared,
        Err(err) => return Err((req.request_id, req.response_tx, err)),
    };
    let prepared = prepared.max_tokens_at_least_one();
    let mut session = match runtime
        .stateful_model()
        .new_decode_session(prepared.session_config.clone())
    {
        Ok(session) => session,
        Err(err) => return Err((req.request_id, req.response_tx, Error::Stateful(err))),
    };
    let prefill = match runtime
        .stateful_model()
        .prefill(&mut session, &prepared.input_ids)
    {
        Ok(prefill) => prefill,
        Err(err) => return Err((req.request_id, req.response_tx, Error::Stateful(err))),
    };
    let last_token = *prepared.input_ids.last().expect("prepared requests are non-empty");

    Ok(ActiveRequest {
        request_id: req.request_id,
        session,
        last_token,
        remaining_tokens: prepared.max_tokens,
        content: String::new(),
        completion_tokens: 0,
        prompt_tokens: prefill.prompt_tokens,
        response_tx: req.response_tx,
        skip_special_tokens: prepared.skip_special_tokens,
    })
}

async fn merge_text_delta(
    model: &mistralrs::Model,
    content: &mut String,
    delta: Option<String>,
    fallback_token: Option<u32>,
    skip_special_tokens: bool,
) -> Result<()> {
    if let Some(delta) = delta {
        if let Some(suffix) = delta.strip_prefix(content.as_str()) {
            content.push_str(suffix);
        } else {
            *content = delta;
        }
        return Ok(());
    }

    if let Some(token) = fallback_token {
        let detok = model.detokenize(vec![token], skip_special_tokens).await?;
        content.push_str(&detok);
    }
    Ok(())
}

trait PreparedRequestExt {
    fn max_tokens_at_least_one(self) -> PreparedStatefulRequest;
}

impl PreparedRequestExt for PreparedStatefulRequest {
    fn max_tokens_at_least_one(mut self) -> PreparedStatefulRequest {
        self.max_tokens = self.max_tokens.max(1);
        self
    }
}

trait RequestContextExt {
    fn with_request_context(self, request_id: Option<String>) -> Self;
}

impl RequestContextExt for Error {
    fn with_request_context(self, request_id: Option<String>) -> Self {
        let _ = request_id;
        self
    }
}
