## Stateful Runtime WIP

This note captures the next layer of work around the new stateful decode API.

The current branch already provides:

- `StatefulModel`
- `DecodeSession`
- `prefill()`
- `decode_step()`
- Metal smoke validation

The next question is how far Project Aster should extend the open
`mistral.rs` stateful runtime line.

### Current Split

What belongs in `mistral.rs`:

- stateful execution primitives
- backend feature reporting
- model/session lifecycle
- batch stepping primitives
- backend-specific MLX / Metal / Qwen3.5 / GDN support

What does not belong in `mistral.rs`:

- cluster routing
- node discovery
- company connectors
- internal tools/prompts/workflows
- product-specific serving behavior

### Project Aster Shape

Project Aster focuses on the runtime shape built around the stateful API.

The target pieces are:

- capability reporting for scheduler-aware execution
- the external lifecycle:
  - create session
  - prefill
  - iterative decode
  - later decode-batch
- a narrow single-sequence reference path
- a minimal streaming bridge for iterative decode
- later, a generic continuous batching scheduler

### Suggested Open-Source Shape

The best next step is small and concrete.

#### 1. Add a narrow single-sequence example

Add an example that:

- tokenizes chat input
- creates a `DecodeSession`
- runs `prefill()`
- loops `decode_step()`
- detokenizes the response

This should be a reference path for users of the stateful API.

#### 2. Add a streaming example

Before a full scheduler lands, provide a small streaming example that emits
token deltas from `decode_step()`.

That gives a clean migration path from whole-request inference to externally
owned stepping.

#### 3. Add `decode_batch()`

The real scheduler boundary becomes useful once the backend exposes batch
stepping over externally owned sessions.

That should stay backend-focused:

- no policy
- no cluster logic
- just batched stepping primitives

#### 4. Add a minimal scheduler layer later

If a reusable scheduler is added, it should be deliberately small:

- request queue
- batch assembly
- token-step loop
- cancellation
- fairness knobs

It should not include:

- cluster concerns
- product-specific routes
- internal integrations

### Project Aster Scope

Project Aster is centered on:

- capability reporting
- a narrow stateful single-sequence path
- a future streaming bridge
- later, token interleaving policy

Those pieces should be expressed in generic form as first-class public runtime
work, not as product-specific behavior.

### End State

The intended end state is:

- `mistral.rs` owns the execution substrate
- examples show how to drive the stateful API directly
- an optional lightweight runtime layer can exist for generic scheduling
- broader private product/runtime concerns stay outside the public branch

This keeps the open work useful and reviewable without dragging private system
architecture into the public branch.
