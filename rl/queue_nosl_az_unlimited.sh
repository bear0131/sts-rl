#!/usr/bin/env bash
set -euo pipefail

# Remote-only launcher for the unlimited-depth experiment. Worker mode starts
# one rank with fixed experiment args; supervise mode is retained for manual
# queueing but is not needed when the depth-limited run is stopped explicitly.
MODE=${1:-supervise}
RANK=${2:-0}
REPO=/root/autodl-tmp/sts-rl
PYTHON="$REPO/.venv/bin/python"
BUILD="$REPO/sim/sts_lightspeed/build-handonly-v4"
REMOTE_ROOT="$REPO/.tmp-dist-sb3-numberencoder-262144"
SSH_KEY=/root/.ssh/sts_dist_shared_ed25519
CONTROL_DIR=/tmp/sts-launch
INIT_CHECKPOINT="$REPO/migration/nosl-az/nosl_az_random_start_step0.zip"
DEPTH_RUN_ID=naz4w-cont10m-r3
DEPTH_PID=498182
DEPTH_CHECKPOINT_DIR="$REPO/checkpoints/nosl-az-4way-s128d3-b16-cont10m-20260905"
UNLIMITED_RUN_ID=naz4w-unlimited
UNLIMITED_LOG_DIR="$REPO/logs/nosl-az-4way-s128-unlimited-b16-20260905"
UNLIMITED_CHECKPOINT_DIR="$REPO/checkpoints/nosl-az-4way-s128-unlimited-b16-20260905"
UNLIMITED_WANDB_NAME=nosl-cpp-az-4way-s128-unlimited-b16
UNLIMITED_RING_PORT=29764

port_for_rank() {
    case "$1" in
        0) printf '%s' 21496 ;;
        1) printf '%s' 20862 ;;
        2) printf '%s' 15557 ;;
        3) printf '%s' 46144 ;;
        *) return 1 ;;
    esac
}

peer_args_for_rank() {
    local rank=$1 peer args=""
    for peer in 0 1 2 3; do
        if [[ "$peer" != "$rank" ]]; then
            args+=" --peer ${peer}=root@connect.bjb2.seetacloud.com:$(port_for_rank "$peer")"
        fi
    done
    printf '%s' "$args"
}

launch_worker() {
    local rank=$1
    local log_file="$UNLIMITED_LOG_DIR/rank${rank}.log"
    local pid_file="$UNLIMITED_LOG_DIR/rank${rank}.pid"
    mkdir -p "$UNLIMITED_LOG_DIR" "$UNLIMITED_CHECKPOINT_DIR" \
        "$REMOTE_ROOT/$UNLIMITED_RUN_ID/rank${rank}"
    if [[ -s "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
        echo "rank${rank} already running pid=$(cat "$pid_file")"
        return 0
    fi
    local peers
    peers=$(peer_args_for_rank "$rank")
    # shellcheck disable=SC2086
    nohup env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        STS_ENABLE_RULE_TRACE=ON STS_LIGHTSPEED_BUILD_DIR="$BUILD" \
        PYTHONPATH="$BUILD:$REPO" "$PYTHON" -u -m rl.distributed_sb3 \
        --rank "$rank" --world-size 4 \
        --ssh-key "$SSH_KEY" --ssh-control-dir "$CONTROL_DIR" \
        --remote-root "$REMOTE_ROOT" --run-id "$UNLIMITED_RUN_ID" \
        --checkpoint "$INIT_CHECKPOINT" --no-interactive-reasoning \
        --nosl-is-az-depth 0 --total-steps 10000000 \
        --n-envs 24 --n-steps 128 --batch-size 1536 \
        --gradient-accumulation-steps 1 --n-epochs 1 --learning-rate 0.0003 \
        --amp-mode bf16 --gradient-transport ssh-ring \
        --ring-port "$UNLIMITED_RING_PORT" \
        --reasoning-batch-size 1 --reasoning-updates 0 \
        --reasoning-gradient-accumulation-steps 1 --reasoning-coef 0.0 \
        --wandb --wandb-project sts-rl --wandb-name "$UNLIMITED_WANDB_NAME" \
        --save-every-updates 25 --checkpoint-dir "$UNLIMITED_CHECKPOINT_DIR" \
        --checkpoint-prefix nosl_az --max-checkpoints 2 \
        --barrier-timeout 900 --float32-matmul-precision high $peers \
        >"$log_file" 2>&1 < /dev/null &
    echo "$!" > "$pid_file"
    echo "started rank${rank} pid=$(cat "$pid_file") log=$log_file"
}

if [[ "$MODE" == worker ]]; then
    launch_worker "$RANK"
    exit 0
fi

if [[ "$MODE" != supervise || "$RANK" != 0 ]]; then
    echo "usage: $0 [supervise|worker] [rank]" >&2
    exit 2
fi

peer_depth_pids=("" 123280 125772 132349)
peer_ports=(21496 20862 15557 46144)

depth_process_alive() {
    local rank=$1 pid=$2 port=${peer_ports[$1]}
    if [[ "$rank" == 0 ]]; then
        ps -p "$pid" -o args= 2>/dev/null \
            | grep -F -- "--run-id $DEPTH_RUN_ID" >/dev/null
    else
        ssh -q -i "$SSH_KEY" -o IdentitiesOnly=yes \
            -o BatchMode=yes -o StrictHostKeyChecking=no -p "$port" \
            root@connect.bjb2.seetacloud.com \
            "ps -eo args= | grep -E '[r]l.distributed_sb3.*$DEPTH_RUN_ID' >/dev/null"
    fi
}

echo "waiting for depth-limited run $DEPTH_RUN_ID to finish" >&2
while depth_process_alive 0 "$DEPTH_PID" || \
    depth_process_alive 1 "${peer_depth_pids[1]}" || \
    depth_process_alive 2 "${peer_depth_pids[2]}" || \
    depth_process_alive 3 "${peer_depth_pids[3]}"; do
    sleep 60
done

# 10,000,000 is reached on the next complete 12,288-step global update.
final_checkpoint="$DEPTH_CHECKPOINT_DIR/nosl_az_10004952_steps.zip"
if [[ ! -s "$final_checkpoint" ]]; then
    echo "depth run exited without final checkpoint: $final_checkpoint" >&2
    exit 1
fi

echo "depth-limited run finished; starting unlimited-depth run" >&2
for rank in 1 2 3; do
    port=${peer_ports[$rank]}
    ssh -q -i "$SSH_KEY" -o IdentitiesOnly=yes \
        -o BatchMode=yes -o StrictHostKeyChecking=no -p "$port" \
        root@connect.bjb2.seetacloud.com \
        "$REPO/rl/queue_nosl_az_unlimited.sh worker $rank"
done
launch_worker 0
echo "unlimited-depth run queued: $UNLIMITED_RUN_ID" >&2
