#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RuntimeFeatures {
    pub supports_stateful_decode: bool,
    pub supports_external_scheduler: bool,
    pub supports_decode_batch: bool,
}

impl From<mistralrs::BackendFeatures> for RuntimeFeatures {
    fn from(value: mistralrs::BackendFeatures) -> Self {
        Self {
            supports_stateful_decode: value.supports_stateful_decode,
            supports_external_scheduler: value.supports_external_scheduler,
            supports_decode_batch: false,
        }
    }
}

