#!/usr/bin/env bash
set -euo pipefail

rank=${1:?usage: launch_nosl_replay.sh rank}
case "$rank" in 0|1|2|3) ;; *) echo 'rank must be 0..3' >&2; exit 2 ;; esac
repo=${NOSL_REPO:-/root/autodl-tmp/sts-rl}
run_id=${NOSL_RUN_ID:?set a persistent transport run ID}
wandb_id=${NOSL_WANDB_ID:?set a persistent W&B run ID}
[[ "$run_id" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'invalid run ID' >&2; exit 2; }
[[ "$wandb_id" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'invalid W&B run ID' >&2; exit 2; }
run_name=${NOSL_RUN_NAME:-nosl-rootpaper-graph-replay}
args_file=${NOSL_ARGS_FILE:-$repo/configs/train/nosl_rootpaper_replay.args}
python="$repo/.venv/bin/python"
# Use the accepted native build by default. An explicit override remains
# available for bounded experiments, but deleted/unknown builds must not be
# selected implicitly by a production launch.
build=${STS_LIGHTSPEED_BUILD_DIR:-$repo/runs/local/search-perf04-20260919/accepted-build-latest-rules}
checkpoint=${NOSL_CHECKPOINT:-$repo/migration/nosl-az/nosl_az_random_start_s512_boundary_step0.zip}
save_every_updates=${NOSL_SAVE_EVERY_UPDATES:-1}
[[ "$save_every_updates" =~ ^[1-9][0-9]*$ ]] || {
  echo 'NOSL_SAVE_EVERY_UPDATES must be a positive integer' >&2
  exit 2
}
run_dir="$repo/runs/$run_id"
remote_root="$repo/.tmp-dist-sb3-$run_id"
stop_file="$run_dir/STOP"
max_checkpoints=${NOSL_MAX_CHECKPOINTS:-2}
log_max_bytes=${NOSL_LOG_MAX_BYTES:-67108864}
[[ "$max_checkpoints" =~ ^[2-9][0-9]*$ ]] || {
  echo 'NOSL_MAX_CHECKPOINTS must be an integer >= 2' >&2
  exit 2
}
[[ "$log_max_bytes" =~ ^[1-9][0-9]*$ ]] || {
  echo 'NOSL_LOG_MAX_BYTES must be a positive integer' >&2
  exit 2
}
if pgrep -f -- "--run-id $run_id( |$)" >/dev/null 2>&1; then
  echo "run $run_id already has a trainer on this machine" >&2; exit 3
fi
test ! -e "$stop_file" || { echo 'clear the completed stop request before resuming' >&2; exit 3; }
test -x "$python"
test -f "$checkpoint"
test -f "$args_file"
mkdir -p "$run_dir" /tmp/sts-launch
# The transport root is shared by all four ranks through SCP. It must be
# cleaned exactly once before any rank starts; deleting it here from every
# rank races with ready-marker exchange and can erase another rank's marker.
if [[ "${NOSL_SKIP_TRANSPORT_CLEANUP:-0}" != 1 ]]; then
  rm -rf -- "$remote_root"
fi
mkdir -p "$remote_root"
# W&B keeps a small local directory per process even when the cloud run ID is
# resumed.  The cloud history is durable, so old local copies for this exact
# run ID can be discarded before a restart.
wandb_root="$repo/wandb"
if [ -d "$wandb_root" ]; then
  find "$wandb_root" -mindepth 1 -maxdepth 1 -type d \
    -name "run-*-$wandb_id" -exec rm -rf -- {} +
fi
read -r -a ports <<< "${NOSL_PEER_PORTS:-12009 17944 13500 44892}"
[ "${#ports[@]}" -eq 4 ] || {
  echo 'NOSL_PEER_PORTS must contain four space-separated ports' >&2
  exit 2
}
read -r -a hosts <<< "${NOSL_PEER_HOSTS:-connect.bjb2.seetacloud.com connect.bjb2.seetacloud.com connect.bjb2.seetacloud.com connect.bjb2.seetacloud.com}"
[ "${#hosts[@]}" -eq 4 ] || {
  echo 'NOSL_PEER_HOSTS must contain four space-separated hosts' >&2
  exit 2
}
peers=()
for peer in 0 1 2 3; do
  [ "$peer" = "$rank" ] && continue
  peers+=(--peer "${peer}=root@${hosts[$peer]}:${ports[$peer]}")
done
extra=()
if [ -n "${NOSL_OVERRIDES_FILE:-}" ]; then extra+=("@$NOSL_OVERRIDES_FILE"); fi
if [ -n "${NOSL_RESUME_STEP:-}" ]; then
  printf -v generation 'step-%012d' "$NOSL_RESUME_STEP"
  if [ -n "${NOSL_RESUME_ROOT:-}" ]; then
    resume_state="$NOSL_RESUME_ROOT/rank$rank/$generation"
  else
    resume_state="$run_dir/recovery/rank$rank/$generation"
  fi
  if [ "${NOSL_FORK_RESUME:-0}" = 1 ]; then
    extra+=(--fork-state "$resume_state")
    if [ "${NOSL_FORK_PRESERVE_ENVIRONMENTS:-0}" != 1 ]; then
      # Geometry-changing forks discard source VecEnv state by default.  An
      # explicit opt-in may preserve it when n_envs is unchanged; the runtime
      # validates the restored state count before training starts.
      extra+=(--fork-reset-environments)
    fi
  else
    extra+=(--resume-state "$resume_state")
    if [ "${NOSL_RESUME_RESET_ENVIRONMENTS:-0}" = 1 ]; then
      extra+=(--resume-reset-environments)
    fi
  fi
fi
cmd=("$python" -u -m rl.training.distributed_sb3
  --rank "$rank" --world-size 4 --run-id "$run_id"
  --remote-root "$remote_root" --ssh-key /root/.ssh/sts_dist_shared_ed25519
  --ssh-control-dir /tmp/sts-launch --checkpoint "$checkpoint"
  "@$args_file" "${extra[@]}"
  --gradient-transport ssh-ring --ring-port "${NOSL_RING_PORT:-29789}"
  --reasoning-batch-size 1 --reasoning-updates 0 --reasoning-coef 0
  --reasoning-gradient-accumulation-steps 1 --no-interactive-reasoning
  --leaf-inference "${NOSL_LEAF_INFERENCE:-cuda-graph}"
  --wandb --wandb-project sts-rl --wandb-id "$wandb_id" --wandb-name "$run_name"
  --save-every-updates "$save_every_updates" --max-checkpoints "$max_checkpoints" --training-state-dir "$run_dir/recovery"
  --stop-file "$stop_file" --barrier-timeout 900 --float32-matmul-precision high
  "${peers[@]}")
cd "$repo"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export STS_LIGHTSPEED_BUILD_DIR="$build" PYTHONPATH="$build:$repo"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
printf '\n=== NOSL launch utc=%s rank=%s resume_step=%s ===\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rank" "${NOSL_RESUME_STEP:-fresh}" \
  >> "$run_dir/rank$rank.log"
# Keep the diagnostic log bounded.  W&B remains the durable metric history;
# retaining only the newest half keeps local disk use independent of run age.
if [ -f "$run_dir/rank$rank.log" ]; then
  log_bytes=$(stat -c '%s' "$run_dir/rank$rank.log")
  if [ "$log_bytes" -gt "$log_max_bytes" ]; then
    tail -c "$((log_max_bytes / 2))" "$run_dir/rank$rank.log" > "$run_dir/.rank$rank.log.trim"
    mv -f -- "$run_dir/.rank$rank.log.trim" "$run_dir/rank$rank.log"
  fi
fi
nohup "${cmd[@]}" >> "$run_dir/rank$rank.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$run_dir/rank$rank.pid"
sleep 1
if ! kill -0 "$pid" 2>/dev/null; then
  echo "rank=$rank exited during startup; inspect $run_dir/rank$rank.log" >&2
  tail -n 20 "$run_dir/rank$rank.log" >&2
  exit 4
fi
echo "started rank=$rank pid=$pid run=$run_id wandb=$wandb_id log=$run_dir/rank$rank.log"
