#!/usr/bin/env bash
set -euo pipefail

rank=${1:?usage: $0 rank}
case "$rank" in
  0) self_port=12009 ;;
  1) self_port=17944 ;;
  2) self_port=13500 ;;
  3) self_port=44892 ;;
  *) echo "invalid rank: $rank" >&2; exit 2 ;;
esac
ring_port=29786

repo=/root/autodl-tmp/sts-rl
python="$repo/.venv/bin/python"
build="$repo/sim/sts_lightspeed/build-gumbel-current"
run_id=fgs128u40noetlnr288fix
run_name=nosl-paper-full-gumbel-4way-s128-randomstart-boundary-lr1e4-noreplay-u40-no-end-turn-leaf-e288-u40fix-20260913
checkpoint="$repo/migration/nosl-az/nosl_az_random_start_s512_boundary_step0.zip"
remote_root="$repo/.tmp-dist-sb3-$run_id"
log_dir="$repo/logs/$run_name"
checkpoint_dir="$repo/checkpoints/$run_name"
ssh_key=/root/.ssh/sts_dist_shared_ed25519
control_dir=/tmp/sts-launch

if pgrep -af 'python.*-m rl\.distributed_sb3' >/dev/null 2>&1; then
  echo "a distributed_sb3 process is already running on rank $rank" >&2
  exit 3
fi

mkdir -p \
  "$remote_root/$run_id/rank$rank" \
  "$log_dir" \
  "$checkpoint_dir" \
  "$control_dir"
test -x "$python"
test -f "$checkpoint"
test -f "$build"/slaythespire*.so

peer_args=()
for peer in 0 1 2 3; do
  [ "$peer" = "$rank" ] && continue
  case "$peer" in
    0) port=12009 ;;
    1) port=17944 ;;
    2) port=13500 ;;
    3) port=44892 ;;
  esac
  peer_args+=(--peer "${peer}=root@connect.bjb2.seetacloud.com:${port}")
done

cmd=(env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  STS_ENABLE_RULE_TRACE=ON \
  STS_LIGHTSPEED_BUILD_DIR="$build" \
  PYTHONPATH="$build:$repo" \
  "$python" -u -m rl.distributed_sb3 \
  --rank "$rank" --world-size 4 \
  --ssh-key "$ssh_key" --ssh-control-dir "$control_dir" \
  --remote-root "$remote_root" --run-id "$run_id" \
  --checkpoint "$checkpoint" --no-interactive-reasoning \
  --nosl-is-az-simulations 128 --nosl-is-az-depth 0 \
  --no-nosl-is-az-end-turn-as-leaf \
  --nosl-is-az-prior-temperature 1 \
  --nosl-is-az-search-mode full_gumbel --nosl-is-az-paper-full-gumbel \
  --nosl-is-az-batch-size 24576 --nosl-is-az-updates 40 \
  --nosl-is-az-micro-batch-size 2048 --no-nosl-is-az-replay \
  --nosl-is-az-dirichlet-alpha 0 --nosl-is-az-dirichlet-epsilon 0 \
  --nosl-is-az-leaf-batch-size 1 \
  --total-steps 10000000 \
  --n-envs 288 --envs-per-worker 12 --n-steps 128 --batch-size 1536 \
  --gradient-accumulation-steps 1 --n-epochs 1 --learning-rate 0.0001 \
  --amp-mode bf16 --gradient-transport ssh-ring --ring-port "$ring_port" \
  --reasoning-batch-size 1 --reasoning-updates 0 \
  --reasoning-gradient-accumulation-steps 1 --reasoning-coef 0.0 \
  --wandb --wandb-project sts-rl --wandb-name "$run_name" \
  --save-every-updates 25 --checkpoint-dir "$checkpoint_dir" \
  --checkpoint-prefix nosl_az --max-checkpoints 2 \
  --barrier-timeout 900 --float32-matmul-precision high \
  "${peer_args[@]}")

log_file="$log_dir/rank$rank.log"
nohup "${cmd[@]}" > "$log_file" 2>&1 < /dev/null &
pid=$!
echo "$pid" > "$log_dir/rank$rank.pid"
echo "started rank=$rank pid=$pid log=$log_file"
