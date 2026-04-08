# mistralrs-stateful-runtime

Experimental runtime utilities for the `mistral.rs` stateful decode API.

Current scope:

- single-sequence stateful execution loop
- streaming-oriented delta output path
- capability reporting
- scheduler sketches for future `decode_batch()` work

This crate is intentionally narrow. It is meant to make the low-level
stateful API easier to drive directly, without pulling broader serving or
deployment concerns into the public surface.

## Current Entry Points

- `StatefulRuntime::from_text_builder(...)`
- `StatefulRuntime::run(...)`
- `StatefulRuntime::run_streaming(...)`

## Examples

- `cargo run -p mistralrs-stateful-runtime --example stateful_single_seq`
- `cargo run -p mistralrs-stateful-runtime --example stateful_stream`
