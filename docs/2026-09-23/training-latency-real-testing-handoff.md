# PERF09 real GPU baseline handoff

## Objective and immutable scope

Run the first controlled GPU/full-replay baseline for the NOSL Learner timing
instrumentation before choosing an optimization.  This measurement must not
change the model, search, replay distribution, effective batch, microbatch,
AMP, loss, optimizer, or checkpoint policy.

Use exactly this runtime implementation unless a defect is found and
separately fixed:

- branch: `codex/learner-latency-instrumentation-20260923`
- commit: `9f6d3ad8e3fc75fb9746c2fd5810bf059b843ada`
- parent: `0f9819a53a43260b82c573763a4c5dfe44796af8`
- remote: `origin` / `git@github.com:bear0131/sts-rl.git`

The branch was pushed and the local checkout was clean and tracking the
remote branch at handoff.  Documentation-only descendants of `9f6d3ad` are
expected on that branch; they do not change the tested runtime.  Read the
design and overlap contract in
`docs/2026-09-22/training-latency-instrumentation-handoff.md` before running
anything.

## Accepted implementation state

The implementation exposes these per-round fields:

- monotonic host wall: `learner_total_sec`,
  `learner_replay_sample_sec`, `learner_materialize_sec`,
  `learner_cpu_metrics_sec`, `learner_gradient_sync_sec`, and
  `learner_optimizer_wall_sec`;
- CUDA events on CUDA, monotonic wall on CPU: `learner_h2d_sec`,
  `learner_forward_sec`, `learner_backward_sec`, and
  `learner_optimizer_sec`;
- host residual: `learner_unattributed_sec`.

CUDA events are recorded per microbatch but resolved only at the update
boundary.  They can overlap host materialization and dispatch, so they are an
overlay rather than mutually exclusive wall-time buckets.  Do not sum every
field or subtract CUDA event time from `learner_unattributed_sec`.  The legacy
`backward_sec` remains the old broad Learner boundary and is not equivalent to
`learner_backward_sec`.  Timing state is local to calls and is not persisted
in model or recovery checkpoints.

Local CPU acceptance already passed:

- focused learner/distributed suite: 30 passed, 1 CUDA-only skipped, 2
  subtests passed;
- shared-update/recovery/training-smoke integrations: 14 passed;
- two real NOSL lifecycle variants: 2 passed through collection, learner,
  payload, checkpoint, and resume;
- syntax compilation and `git diff --check`: passed.

One full lifecycle parameterization has a pre-existing
`rules-transformer`/no-update startup-occupancy assertion failure.  It did not
perform optimizer work and is not a PERF09 failure.  See
`worklog/2026-09-23.md` for the exact results.

## Remote access and last known state

The standard endpoints are `connect.bjb2.seetacloud.com` ports
`22324`, `50418`, `25377`, and `17011`, user `root`, remote repository
`/root/autodl-tmp/sts-rl`.  The supported Windows entry is
`experiments/launchers/manage_nosl_replay_4way.ps1`; it expects a key at
`%USERPROFILE%\.ssh\sts_dist_shared_ed25519` and never stores passwords.

At this handoff that key path did not exist, so the 2026-09-23 read-only
status command stopped locally before connecting.  Remote process, disk,
source, and recovery state are therefore **unknown now**.  The last historical
observation in `worklog/2026-09-22.md` was the TRAIN08 run
`u40x4-ruleupper-v1-reduced-align-768x16-20260922` / W&B
`u40x4tr100a22`, but it must not be treated as current.

Do not paste a password or private key into chat, Git, a command line, or a
worklog.  Ask the user to install/authorize the key outside the repository if
it is still missing.  A password may be used interactively by the user only
to bootstrap the public key.

## Mandatory startup checks

1. Read `AGENTS.md`, `TODO.md`, `worklog/2026-09-22.md`,
   `worklog/2026-09-23.md`, and both PERF09 handoffs.
2. Verify the local checkout before edits:

   ```powershell
   git status --short --branch
   git rev-parse HEAD
   git log -2 --oneline --decorate
   git merge-base --is-ancestor 9f6d3ad HEAD
   git diff --name-only 9f6d3ad HEAD
   ```

   Expect a clean instrumentation branch, with `9f6d3ad` as an ancestor and
   only `TODO.md`, `docs/`, or `worklog/` changed after it.  Do not silently
   test a later runtime/code change.
3. Verify key availability, then run only read-only remote checks:

   ```powershell
   Test-Path "$env:USERPROFILE\.ssh\sts_dist_shared_ed25519"
   $ports = 22324,50418,25377,17011
   .\experiments\launchers\manage_nosl_replay_4way.ps1 `
     -Action plan -Ports $ports
   .\experiments\launchers\manage_nosl_replay_4way.ps1 `
     -Action status -Ports $ports `
     -RunId u40x4-ruleupper-v1-reduced-align-768x16-20260922
   ```

4. On every host record the trainer/GPU process state, free data-disk space,
   Git HEAD/status, Python/Torch/CUDA versions, GPU model/driver, native build
   path and SHA-256.  If production is still running or another process owns
   a GPU, do not deploy, create `STOP`, or start the baseline without explicit
   coordination.
5. Use an isolated checkout/worktree for commit `9f6d3ad`; never pull over a
   live production checkout.  Reuse a native build only after its source and
   ABI identity are verified.  Run the focused CUDA test, including the
   temporary host-source lifecycle test, before the baseline.

## Controlled baseline protocol

The exact source generation must be selected from current read-only evidence,
not guessed from this document.

1. Select one common schema-2 generation present on all four ranks.  Run
   `rl.training.checkpoints.read_bundle` for each rank and record run/W&B ID,
   global step `S`, and hashes for `model.zip`, `learner.pt`, and
   `environment.pt`.
2. Verify `learner.pt` on every rank has a non-null replay with
   `replay['size'] == replay['capacity'] == 524288`.  This is the required
   **full replay** gate.  Record the check; do not copy the large tensor data
   into Git or the worklog.
3. Fork to a new run/W&B identity so the source bundle and production run are
   untouched.  Use the production contract from
   `configs/train/nosl_ruleupper_v1_reduced_align_768x16_30m.args` unless the
   selected source manifest proves a different active contract.  The expected
   contract is seed 0, 4 ranks, 768 environments/rank, 16 steps,
   32 environments/worker, BF16, strict Full-Gumbel 128/depth 0, replay
   capacity 524288 with old fraction 0.90, batch 98304, microbatch 2048,
   40 optimizer updates, LR 2.5e-4, and the existing checkpoint cadence.
   Any mismatch must stop the run rather than be normalized silently.
4. Fix the workload at exactly two collection rounds: one explicit warmup
   round and one measured round.  With the expected geometry, one round is
   `4 * 768 * 16 = 49152` global transitions, so the absolute target is
   `S + 98304`.  Put only the computed `--total-steps` override in a
   run-specific ignored file; do not edit the production args preset.
5. Add an external wall deadline.  The manager's `-TimeoutSeconds` only stops
   waiting and leaves trainers running, so it is not by itself a training
   timeout.  Use a documented watchdog that requests the common run's `STOP`
   at an update boundary and is cancelled after normal completion.  Do not
   hard-kill one rank independently.  Record deadline, watchdog PID/state,
   exit codes, and whether STOP fired.
6. Keep raw logs, status captures, resource samples, and analysis under a new
   ignored `runs/local/perf09-learner-baseline-*` directory.  Do not use the
   source bundle directory as output and do not commit large recovery files.
7. Copy the four complete rank logs after termination.  Verify both common
   steps have `finite_fraction=1`, replay size 524288, batch used 98304,
   updates/optimizer steps 40, and every new timing field finite and
   non-negative.  Verify the final recovery manifests remotely; copying all
   multi-GiB bundles locally is optional if remote checksum receipts are
   retained.
8. Summarize the copied logs with:

   ```powershell
   python -m tools.analysis.compare_nosl_timing `
     --run baseline=runs/local/perf09-learner-baseline-YYYYMMDD `
     --steps STEP1,STEP2 --world-size 4
   ```

   Save the JSON output under the same ignored directory.  Use the slowest
   rank's `iteration_wall_sec` as end-to-end time.  For Learner stages report
   per-step rank min/mean/max, not an invented exclusive sum.

## Required report

Report the fixed source commit/generation and all workload/config identities,
then separate warmup and measured rounds.  At minimum include:

- end-to-end and `learner_total_sec`;
- replay sampling, CPU materialization, and CPU metric aggregation;
- H2D, forward, backward, gradient synchronization, optimizer CUDA event
  time, optimizer host wall, and unattributed host time;
- collection/checkpoint time kept separate from Learner time;
- process-tree CPU time/utilization, RSS with the shared-page caveat, system
  available memory, Torch allocated/reserved/peak, device VRAM and GPU util,
  with unavailable values reported as `null`, not zero;
- exit/timeout/STOP state, field/reset checks, replay/fullness contract, and
  cross-rank policy/Adam consistency where a final bundle is written.

Do not recommend or implement an optimization until this baseline is complete
and its overlap semantics are stated correctly.  Update `TODO.md` and the
current-date worklog, run `git diff --check`, and commit only the small report
and documentation.  Do not commit credentials, raw multi-GiB bundles, or
`runs/local` outputs.
