#!/usr/bin/env bash
set -euo pipefail

# Long-running campfire probe entry point.  It selects the newest completed
# whole-run checkpoint by the numeric step suffix and writes every invocation
# to a fresh timestamped directory, so repeated probes never overwrite data.
REPO="${STS_REPO:-/root/autodl-tmp/sts-rl}"
PYTHON="${STS_PYTHON:-$REPO/.venv/bin/python}"
CHECKPOINT_DIR="${STS_CAMPFIRE_CHECKPOINT_DIR:-$REPO/checkpoints/wholerun-cardvec-step10k-random-bs262144-e10-n128-hierhp}"
OUTPUT_ROOT="${STS_CAMPFIRE_OUTPUT_ROOT:-$REPO/evaluations/campfire}"
EPISODES="${STS_CAMPFIRE_EPISODES:-80}"
SEED="${STS_CAMPFIRE_SEED:-30000}"
ASCENSION="${STS_CAMPFIRE_ASCENSION:-0}"
DEVICE="${STS_CAMPFIRE_DEVICE:-cpu}"
BATCH_SIZE="${STS_CAMPFIRE_BATCH_SIZE:-16}"
MAX_STEPS="${STS_CAMPFIRE_MAX_STEPS:-6000}"
PROGRESS_EVERY="${STS_CAMPFIRE_PROGRESS_EVERY:-5}"

if [[ ! -x "$PYTHON" ]]; then
  echo "missing Python interpreter: $PYTHON" >&2
  exit 2
fi
if [[ ! -d "$CHECKPOINT_DIR" ]]; then
  echo "missing checkpoint directory: $CHECKPOINT_DIR" >&2
  exit 2
fi
cd "$REPO"

checkpoint="$($PYTHON - "$CHECKPOINT_DIR" <<'PY'
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
paths = []
for path in root.glob("hier_hp_ppo_*_steps.zip"):
    match = re.search(r"_(\d+)_steps$", path.stem)
    if match:
        paths.append((int(match.group(1)), path))
if not paths:
    raise SystemExit(f"no completed checkpoint found in {root}")
print(max(paths, key=lambda item: item[0])[1])
PY
)"

step="$(basename "$checkpoint" | sed -E 's/^.*_([0-9]+)_steps\.zip$/\1/')"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
output_dir="$OUTPUT_ROOT/${stamp}-step${step}"
mkdir -p "$output_dir"

echo "[campfire-longterm] checkpoint=$checkpoint"
echo "[campfire-longterm] output_dir=$output_dir"
exec "$PYTHON" -u -m rl.probe_campfire \
  "$checkpoint" \
  --episodes "$EPISODES" \
  --seed "$SEED" \
  --ascension "$ASCENSION" \
  --device "$DEVICE" \
  --batch-size "$BATCH_SIZE" \
  --max-steps "$MAX_STEPS" \
  --progress-every "$PROGRESS_EVERY" \
  --output-dir "$output_dir"
