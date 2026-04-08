//! Experimental runtime utilities built around the `mistral.rs` stateful decode
//! API.
//!
//! The crate is intentionally small. It focuses on:
//!
//! - a narrow single-sequence runtime loop
//! - a streaming-oriented delta output path
//! - lightweight capability reporting
//! - scheduler sketches for future batch stepping work

mod error;
mod features;
mod request;
mod scheduler;
mod session;
mod single_seq;
mod stream;

pub use error::{Error, Result};
pub use features::RuntimeFeatures;
pub use request::{PreparedStatefulRequest, StatefulRequest};
pub use scheduler::{FairnessPolicy, PrefillPolicy, SchedulerRequest, SchedulerSketch};
pub use session::{FinishReason, SessionStats};
pub use single_seq::{SingleSeqOutput, StatefulRuntime};
pub use stream::{DeltaEvent, StatefulStreamOutput};
