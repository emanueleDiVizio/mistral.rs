use crate::StatefulRequest;

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

