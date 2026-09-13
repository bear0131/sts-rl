#!/usr/bin/env bash
set -euo pipefail

# Compatibility entry point for the current schema-v7 executor contract 10.
# Training and validation records are generated online and consumed once.

repo_dir="${REPO_DIR:-/root/autodl-tmp/sts-rl}"
python_bin="${PYTHON_BIN:-.venv/bin/python}"
run_name="${RUN_NAME:-vector-rule-v7-contract-v10-register-values-20260901}"
run_dir="${RUN_DIR:-.tmp-${run_name}}"
vocab_data="${VOCAB_DATA:-.tmp-vector-rule-v7-contract-v10-vocab.jsonl}"
vocab_samples="${VOCAB_SAMPLES:-4000}"
vocab_seed="${VOCAB_SEED:-9100}"
train_seed="${TRAIN_SEED:-9101}"
validation_seed="${VALIDATION_SEED:-9102}"

stream_steps="${STREAM_STEPS:-100000}"
batch_size="${BATCH_SIZE:-128}"
gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS:-8}"
validation_batch_size="${VALIDATION_BATCH_SIZE:-256}"
validation_stream_steps="${VALIDATION_STREAM_STEPS:-2}"
validation_every_steps="${VALIDATION_EVERY_STEPS:-50}"
learning_rate="${LEARNING_RATE:-0.0003}"
delta_loss_weight="${DELTA_LOSS_WEIGHT:-1.0}"
log_every="${LOG_EVERY:-20}"
checkpoint_every_steps="${CHECKPOINT_EVERY_STEPS:-500}"
resume_checkpoint="${RESUME_CHECKPOINT:-}"

wandb_enabled="${WANDB_ENABLED:-1}"
wandb_project="${WANDB_PROJECT:-sts-rl-vector-v7}"
wandb_entity="${WANDB_ENTITY:-1283802076-tsinghua-university}"
wandb_name="${WANDB_NAME:-$run_name}"
wandb_id="${WANDB_ID:-}"
wandb_mode="${WANDB_MODE:-online}"
wandb_log_every="${WANDB_LOG_EVERY:-20}"
wandb_sample_offset="${WANDB_SAMPLE_OFFSET:-0}"

# Every module remains independently configurable. These defaults use the
# requested four-layer outer decoder and the established baseline width.
card_dim="${CARD_DIM:-128}"
relic_dim="${RELIC_DIM:-128}"
hand_dim="${HAND_DIM:-128}"
world_dim="${WORLD_DIM:-256}"
decoder_dim="${DECODER_DIM:-256}"
card_token_dim="${CARD_TOKEN_DIM:-64}"
relic_token_dim="${RELIC_TOKEN_DIM:-64}"
card_heads="${CARD_HEADS:-4}"
relic_heads="${RELIC_HEADS:-4}"
hand_heads="${HAND_HEADS:-4}"
world_heads="${WORLD_HEADS:-8}"
decoder_heads="${DECODER_HEADS:-8}"
card_layers="${CARD_LAYERS:-2}"
relic_layers="${RELIC_LAYERS:-2}"
hand_layers="${HAND_LAYERS:-2}"
world_layers="${WORLD_LAYERS:-3}"
decoder_layers="${DECODER_LAYERS:-4}"
card_feedforward="${CARD_FEEDFORWARD:-256}"
relic_feedforward="${RELIC_FEEDFORWARD:-256}"
hand_hidden="${HAND_HIDDEN:-256}"
hand_feedforward="${HAND_FEEDFORWARD:-512}"
world_feedforward="${WORLD_FEEDFORWARD:-1024}"
decoder_feedforward="${DECODER_FEEDFORWARD:-1024}"
delta_hidden="${DELTA_HIDDEN:-512}"
delta_layers="${DELTA_LAYERS:-2}"

cd "$repo_dir"
mkdir -p logs "$run_dir"
train_log="logs/${run_name}.log"
pid_file="${run_dir}/train.pid"

if [[ -f "$pid_file" ]]; then
  old_pid="$(<"$pid_file")"
  if kill -0 "$old_pid" 2>/dev/null; then
    echo "training is already running: pid=$old_pid"
    exit 0
  fi
fi

vocab_ok=0
if [[ -s "$vocab_data" ]]; then
  if "$python_bin" - "$vocab_data" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    record = json.loads(next(stream))
if record.get("schema_version") != 7 or record.get("executor_contract_version") != 10:
    raise SystemExit(1)
PY
  then
    vocab_ok=1
  fi
fi
if [[ "$vocab_ok" -ne 1 ]]; then
  "$python_bin" -m rl.make_vector_rule_v7_data \
    --out "$vocab_data" --samples "$vocab_samples" --seed "$vocab_seed"
fi

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

extra_args=()
if [[ -n "$resume_checkpoint" ]]; then
  extra_args+=(--resume "$resume_checkpoint")
fi
if [[ "$wandb_enabled" == "1" ]]; then
  extra_args+=(
    --wandb
    --wandb-project "$wandb_project"
    --wandb-name "$wandb_name"
    --wandb-mode "$wandb_mode"
    --wandb-log-every "$wandb_log_every"
    --wandb-sample-offset "$wandb_sample_offset"
  )
  if [[ -n "$wandb_entity" ]]; then
    extra_args+=(--wandb-entity "$wandb_entity")
  fi
  if [[ -n "$wandb_id" ]]; then
    extra_args+=(--wandb-id "$wandb_id")
  fi
fi

nohup "$python_bin" -m rl.train_rule_vectors_v7 \
  --vocab-data "$vocab_data" \
  --stream-command "$python_bin -m rl.make_vector_rule_v7_data --stdout --stream --seed $train_seed" \
  --stream-cwd "$repo_dir" \
  --stream-steps "$stream_steps" \
  --batch-size "$batch_size" \
  --gradient-accumulation-steps "$gradient_accumulation_steps" \
  --out "$run_dir" \
  --device cuda \
  --learning-rate "$learning_rate" \
  --delta-loss-weight "$delta_loss_weight" \
  --max-cards 10 \
  --max-relics 32 \
  --max-tokens 2048 \
  --decoder-max-tokens 512 \
  --validation-stream-command "$python_bin -m rl.make_vector_rule_v7_data --stdout --stream --seed $validation_seed" \
  --validation-stream-steps "$validation_stream_steps" \
  --validation-batch-size "$validation_batch_size" \
  --validation-every-steps "$validation_every_steps" \
  --log-every "$log_every" \
  --checkpoint-every-steps "$checkpoint_every_steps" \
  --card-dim "$card_dim" \
  --relic-dim "$relic_dim" \
  --hand-dim "$hand_dim" \
  --world-dim "$world_dim" \
  --decoder-dim "$decoder_dim" \
  --card-token-dim "$card_token_dim" \
  --relic-token-dim "$relic_token_dim" \
  --card-heads "$card_heads" \
  --relic-heads "$relic_heads" \
  --hand-heads "$hand_heads" \
  --world-heads "$world_heads" \
  --decoder-heads "$decoder_heads" \
  --card-layers "$card_layers" \
  --relic-layers "$relic_layers" \
  --hand-layers "$hand_layers" \
  --world-layers "$world_layers" \
  --decoder-layers "$decoder_layers" \
  --card-feedforward "$card_feedforward" \
  --relic-feedforward "$relic_feedforward" \
  --hand-hidden "$hand_hidden" \
  --hand-feedforward "$hand_feedforward" \
  --world-feedforward "$world_feedforward" \
  --decoder-feedforward "$decoder_feedforward" \
  --delta-hidden "$delta_hidden" \
  --delta-layers "$delta_layers" \
  "${extra_args[@]}" \
  >"$train_log" 2>&1 < /dev/null &
train_pid=$!
printf '%s\n' "$train_pid" >"$pid_file"

sleep 5
if ! kill -0 "$train_pid" 2>/dev/null; then
  tail -n 100 "$train_log"
  exit 1
fi

echo "run_name=$run_name"
echo "run_dir=$run_dir"
echo "train_log=$train_log"
echo "train_pid=$train_pid"
echo "batch_size=$batch_size"
echo "gradient_accumulation_steps=$gradient_accumulation_steps"
echo "effective_batch_size=$((batch_size * gradient_accumulation_steps))"
echo "stream_steps=$stream_steps"
echo "decoder_layers=$decoder_layers"
echo "resume_checkpoint=${resume_checkpoint:-none}"
echo "wandb_enabled=$wandb_enabled"
echo "wandb_id=${wandb_id:-auto}"
