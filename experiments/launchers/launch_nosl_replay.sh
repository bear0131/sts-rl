#!/usr/bin/env bash
set -euo pipefail

rank=${1:?usage: launch_nosl_replay.sh rank}
case "$rank" in 0|1|2|3) ;; *) echo 'rank must be 0..3' >&2; exit 2 ;; esac
repo=${NOSL_REPO:-/root/autodl-tmp/sts-rl}
run_id=${NOSL_RUN_ID:?set a persistent transport run ID}
wandb_id=${NOSL_WANDB_ID:?set a persistent W&B run ID}
[[ "$run_id" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'invalid run ID' >&2; exit 2; }
run_name=${NOSL_RUN_NAME:-nosl-rootpaper-graph-replay}
args_file=${NOSL_ARGS_FILE:-$repo/configs/train/nosl_rootpaper_replay.args}
python="$repo/.venv/bin/python"
build=${STS_LIGHTSPEED_BUILD_DIR:-$repo/sim/sts_lightspeed/build-gumbel-current}
checkpoint=${NOSL_CHECKPOINT:-$repo/migration/nosl-az/nosl_az_random_start_s512_boundary_step0.zip}
run_dir="$repo/runs/$run_id"
remote_root="$repo/.tmp-dist-sb3-$run_id"
stop_file="$run_dir/STOP"
if pgrep -f -- "--run-id $run_id( |$)" >/dev/null 2>&1; then
  echo "run $run_id already has a trainer on this machine" >&2; exit 3
fi
test ! -e "$stop_file" || { echo 'clear the completed stop request before resuming' >&2; exit 3; }
test -x "$python"
test -f "$checkpoint"
test -f "$args_file"
mkdir -p "$run_dir" "$remote_root" /tmp/sts-launch
read -r -a ports <<< "${NOSL_PEER_PORTS:-12009 17944 13500 44892}"
[ "${#ports[@]}" -eq 4 ] || {
  echo 'NOSL_PEER_PORTS must contain four space-separated ports' >&2
  exit 2
}
peers=()
for peer in 0 1 2 3; do
  [ "$peer" = "$rank" ] && continue
  peers+=(--peer "${peer}=root@connect.bjb2.seetacloud.com:${ports[$peer]}")
done
extra=()
if [ -n "${NOSL_OVERRIDES_FILE:-}" ]; then extra+=("@$NOSL_OVERRIDES_FILE"); fi
if [ -n "${NOSL_RESUME_STEP:-}" ]; then
  printf -v generation 'step-%012d' "$NOSL_RESUME_STEP"
  if [ -n "${NOSL_RESUME_ROOT:-}" ]; then
    resume_state="$NOSL_RESUME_ROOT/rank$rank"
  else
    resume_state="$run_dir/recovery/rank$rank/$generation"
  fi
  if [ "${NOSL_FORK_RESUME:-0}" = 1 ]; then
    extra+=(--fork-state "$resume_state")
  else
    extra+=(--resume-state "$resume_state")
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
  --save-every-updates 1 --max-checkpoints "${NOSL_MAX_CHECKPOINTS:-3}" --training-state-dir "$run_dir/recovery"
  --stop-file "$stop_file" --barrier-timeout 900 --float32-matmul-precision high
  "${peers[@]}")
cd "$repo"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export STS_LIGHTSPEED_BUILD_DIR="$build" PYTHONPATH="$build:$repo"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
printf '\n=== NOSL launch utc=%s rank=%s resume_step=%s ===\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rank" "${NOSL_RESUME_STEP:-fresh}" \
  >> "$run_dir/rank$rank.log"
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
