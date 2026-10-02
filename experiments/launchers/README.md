# Four-Rank NOSL Startup: Authoritative Procedure

This is the visible, authoritative startup record for the four-machine NOSL
trainer. The manager is the only supported production entry point:
`manage_nosl_replay_4way.ps1`.

## Connection settings

The manager and Bash launcher default to the replacement rank 0–3 endpoints
recorded in `worklog/2026-10-02.md#train29-clone-resume-running`.
Override them with manager `-Ports` or launcher `NOSL_PEER_PORTS` when machines
change. Connection defaults do not select a training run: pass its current
`-RemoteRepo` and `-RunId` explicitly for `-Action status`.
The manager accepts `-KnownHostsFile PATH` to keep SSH host identities in a
workspace-local file. New identities use OpenSSH `accept-new`; changed keys
are still rejected. Do not commit private keys or machine-local connection files.

## Required Sequence

1. Stop and verify the previous run. Every rank must have no live
   `rl.training.distributed_sb3` process.
2. Run manager preflight for all four ranks. It must validate the exact run,
   checkpoint, recovery identity, and accepted native extension.
3. Let the manager clean the exact run-specific transport root once on every
   host. The manager then exports `NOSL_SKIP_TRANSPORT_CLEANUP=1` before any
   rank launcher starts.
4. Launch ranks through the manager in its normal 0, 1, 2, 3 order. The rank
   launchers may only create directories; they must not delete the shared
   transport root.
5. Confirm all four ring messages, then confirm rank 0 prints a W&B `View run
   at` URL, and only then report the training as started. Finally confirm a
   common recovery generation and a metric line on all ranks.

The cleanup and skip behavior is implemented in
`launch_nosl_replay.sh` and `manage_nosl_replay_4way.ps1`. A run is not
considered started merely because a PID file or parent process exists.

## Forbidden Startup Sequences

- Do not let every rank execute `rm -rf` on the shared transport root.
- Do not start ranks 2/3, wait for partial ready markers, and then start
  ranks 0/1. The barrier requires all ranks and this order can erase markers.
- Do not report success before W&B initialization and a first metric/checkpoint
  are observed.

## Incident Record: 2026-09-24

The first two fresh attempts were **failed startup attempts**, not training:
their shared-root cleanup race destroyed ring ready markers, so all ranks
exited before `wandb.init`. They produced no W&B runs and no recovery state.
Those attempts must not be described as successful in later logs.

The corrected run is:

- Run: `u40x4-ruleupper-v1-original-256-align-96x128-noclip-mb12288-shared-unfrozen-retry3-20260924`
- W&B: `u40x4orig256shared24c`
- URL: `https://wandb.ai/1283802076-tsinghua-university/sts-rl/runs/u40x4orig256shared24c`
- Source: original step-0 checkpoint, not an old run's final parameters
- First common checkpoint verified: `step-000000049152`

## Manual Health Check

Every 30 minutes, perform a direct human check using manager `status` and a
direct `df -h /root/autodl-tmp` query on all four hosts. Record process state,
latest common checkpoint, latest metric, free space, and any traceback in the
dated worklog. Do not replace this with a resident monitoring script.
