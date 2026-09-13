#!/usr/bin/env bash
set -euo pipefail

# Reproducible detached launcher for schema-v7 executor contract v5.
# Every training/validation example is generated once by an independent stream.

repo_dir="${REPO_DIR:-/root/autodl-tmp/sts-rl}"
python_bin="${PYTHON_BIN:-.venv/bin/python}"
run_name="${RUN_NAME:-vector-rule-v7-contract-v5-relic-counter-20260831}"
run_dir="${RUN_DIR:-.tmp-${run_name}}"
vocab_data="${VOCAB_DATA:-.tmp-vector-rule-v7-contract-v5-relic-counter-vocab.jsonl}"
vocab_samples="${VOCAB_SAMPLES:-4000}"
train_seed="${TRAIN_SEED:-4901}"
validation_seed="${VALIDATION_SEED:-4902}"

stream_steps="${STREAM_STEPS:-100000}"
batch_size="${BATCH_SIZE:-1024}"
gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS:-1}"
validation_batch_size="${VALIDATION_BATCH_SIZE:-256}"
validation_stream_steps="${VALIDATION_STREAM_STEPS:-2}"
validation_every_steps="${VALIDATION_EVERY_STEPS:-50}"
learning_rate="${LEARNING_RATE:-0.0003}"
delta_loss_weight="${DELTA_LOSS_WEIGHT:-1.0}"
resume_checkpoint="${RESUME_CHECKPOINT:-}"
wandb_enabled="${WANDB_ENABLED:-0}"
wandb_project="${WANDB_PROJECT:-sts-rl-vector-v7}"
wandb_entity="${WANDB_ENTITY:-}"
wandb_name="${WANDB_NAME:-$run_name}"
wandb_id="${WANDB_ID:-}"
wandb_mode="${WANDB_MODE:-online}"
wandb_log_every="${WANDB_LOG_EVERY:-20}"
wandb_sample_offset="${WANDB_SAMPLE_OFFSET:-0}"

card_dim="${CARD_DIM:-128}"
relic_dim="${RELIC_DIM:-128}"
hand_dim="${HAND_DIM:-128}"
world_dim="${WORLD_DIM:-256}"
decoder_dim="${DECODER_DIM:-256}"
card_token_dim="${CARD_TOKEN_DIM:-64}"
relic_token_dim="${RELIC_TOKEN_DIM:-64}"
card_layers="${CARD_LAYERS:-2}"
relic_layers="${RELIC_LAYERS:-2}"
hand_layers="${HAND_LAYERS:-2}"
world_layers="${WORLD_LAYERS:-3}"
decoder_layers="${DECODER_LAYERS:-4}"
delta_hidden="${DELTA_HIDDEN:-512}"
delta_layers="${DELTA_LAYERS:-2}"

cd "$repo_dir"
mkdir -p logs
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
if record.get("schema_version") != 7 or record.get("executor_contract_version") != 5:
    raise SystemExit(1)
PY
  then
    vocab_ok=1
  fi
fi
if [[ "$vocab_ok" -ne 1 ]]; then
  "$python_bin" -m rl.make_vector_rule_v7_data \
    --out "$vocab_data" --samples "$vocab_samples" --seed 4900
fi

mkdir -p "$run_dir"
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
  --log-every 20 \
  --checkpoint-every-steps 500 \
  --card-dim "$card_dim" \
  --relic-dim "$relic_dim" \
  --hand-dim "$hand_dim" \
  --world-dim "$world_dim" \
  --decoder-dim "$decoder_dim" \
  --card-token-dim "$card_token_dim" \
  --relic-token-dim "$relic_token_dim" \
  --card-layers "$card_layers" \
  --relic-layers "$relic_layers" \
  --hand-layers "$hand_layers" \
  --world-layers "$world_layers" \
  --decoder-layers "$decoder_layers" \
  --delta-hidden "$delta_hidden" \
  --delta-layers "$delta_layers" \
  "${extra_args[@]}" \
  >"$train_log" 2>&1 < /dev/null &
train_pid=$!
printf '%s\n' "$train_pid" >"$pid_file"

sleep 5
if ! kill -0 "$train_pid" 2>/dev/null; then
  tail -n 80 "$train_log"
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
echo "resume_checkpoint=${resume_checkpoint:-none}"
echo "wandb_enabled=$wandb_enabled"
echo "wandb_id=${wandb_id:-auto}"
