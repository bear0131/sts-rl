# Learner latency instrumentation handoff

## Original intention

Instrument the NOSL training Learner before choosing an optimization.  The
requested baseline must separate replay sampling, CPU materialization, CPU
metric aggregation, host-to-device transfer, forward, backward, distributed
gradient synchronization, and optimizer work from the complete Learner wall
time.  This stage is instrumentation and measurement only: it must not change
the model, search, replay distribution, effective batch, microbatch, AMP,
loss, optimizer, or checkpoint policy.

The originally named baseline was
`origin/codex/combat-rule-baseline-20260918` at `f3b1f86`.  On recovery, the
old clean checkout under `runs/local/training-fixes-20260921` still pointed to
that commit.  The main checkout was clean, had no `MERGE_HEAD`, and had
advanced on the same branch to `0f9819a`; `f3b1f86` is its ancestor.  The
post-`f3b1f86` commits contain the 2026-09-22 controlled training geometry and
Learner-boundary diagnosis, so implementation proceeds from the clean current
main checkout while retaining the old checkout as a fixed historical
reference.

## Scope and timing contract

The distributed round record and W&B payload will expose these additive
per-round fields:

- `learner_total_sec`: monotonic wall time around the complete synchronized
  Learner call, including replay preparation, scalar collectives, gradient
  synchronization, and optimizer work.
- `learner_replay_sample_sec`: monotonic CPU wall time used to choose replay
  indices.
- `learner_materialize_sec`: monotonic CPU wall time used to gather replay
  arrays and construct/pack the host microbatches.
- `learner_cpu_metrics_sec`: monotonic CPU wall time used for host diagnostic
  reductions after the blocking diagnostic copy has completed.
- `learner_h2d_sec`, `learner_forward_sec`, and `learner_backward_sec`: CUDA
  event time on CUDA; monotonic CPU wall time for the corresponding compute
  stages on the CPU fallback.  H2D is zero on CPU.
- `learner_gradient_sync_sec`: existing transport wall time for gradient
  synchronization.
- `learner_optimizer_sec`: CUDA event time on CUDA and monotonic wall time on
  CPU.
- `learner_unattributed_sec`: non-negative residual of total wall time after
  subtracting only the non-overlapping main-thread wall spans.  It includes
  uncategorized dispatch/waiting such as the blocking diagnostic D2H boundary.

CUDA events are recorded per microbatch but read only after the existing
update-level diagnostic barrier; instrumentation must not add a device-wide
synchronization per microbatch.  CPU materialization can overlap queued GPU
work, so CUDA event fields are an overlay and are deliberately not subtracted
again from `learner_unattributed_sec`.  The complete list is therefore not an
exclusive partition and must not be summed as one.

The historical `backward_sec`, `sync_sec`, and `optimizer_step_sec` fields stay
available for log compatibility.  `backward_sec` retains its historical broad
local-Learner wall boundary; the new `learner_backward_sec` is the actual
backward device/CPU stage.

All timing accumulators are local variables returned in the update result.
They are not attached to the model and therefore cannot enter SB3 or recovery
checkpoints.

## Acceptance

- Preserve the fixed training/search/replay/batch/microbatch/AMP/loss/
  optimizer/checkpoint contract and numerical behavior.
- Preserve CPU/no-CUDA execution.
- Exercise at least two optimizer updates and verify every field is finite,
  non-negative, reset between calls, and does not alter parameters, optimizer
  state, RNG progression, or sample counts.
- Aggregate once per update/round; do not print per microbatch.
- Run focused unit/integration tests and `git diff --check`.
- Run a bounded fixed-seed, fixed-workload, full-replay baseline with an
  explicit timeout and ignored `runs/local` output.  Report replay sampling,
  CPU materialization, H2D, forward, backward, gradient synchronization,
  optimizer, total, and unattributed time without presenting overlapping
  CUDA and CPU wall spans as mutually exclusive.
- Update `TODO.md` and `worklog/2026-09-22.md`, then create a local commit.
  Do not push, deploy, or start an unbounded training run.

## Status at the 2026-09-23 local checkpoint

The implementation has passed its local CPU acceptance boundary in a
workspace-local Python 3.12 environment.  The focused learner/distributed
tests passed with two optimizer updates and explicit reset/state assertions;
the shared-update, recovery, and training-smoke integrations passed; and the
two production lifecycle variants that actually perform NOSL optimizer work
passed through collector, learner, payload, checkpoint, and resume.  The full
lifecycle parameterization still has one pre-existing
`rules-transformer`/no-update startup-occupancy assertion failure, recorded
separately in `worklog/2026-09-23.md` rather than attributed to PERF09.

The CUDA-only temporary-host-source test and the bounded remote full-replay
baseline remain pending.  A CUDA Torch environment could not be downloaded
after the network interruption, and key-only remote access was not available
in the current session.  No password, key material, deployment, remote
mutation, or training run was attempted.  Resume by deploying the exact local
instrumentation commit through a key-only SSH path, then run a fixed-seed,
fixed-workload, timeout-bounded full-replay baseline into ignored
`runs/local/` output and report the overlap-aware fields defined above.
