#!/usr/bin/env bash
# Pull the newest checkpoint from the autodl box to the same run-id path locally.
# Usage:  bash rl/pull_latest_ckpt.sh         # newest .zip across all runs
#         bash rl/pull_latest_ckpt.sh final   # newest file matching *final*.zip
set -euo pipefail

REMOTE="root@connect.westc.seetacloud.com"
PORT=22780
RUNS="~/autodl-tmp/sts-rl/rl/runs"
PAT="${1:-}"

# newest matching .zip on the remote, by mtime (path is relative to the runs dir)
rel=$(ssh -p "$PORT" "$REMOTE" \
    "cd $RUNS && find . -name '*${PAT}*.zip' -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-")
[ -n "$rel" ] || { echo "no checkpoint matching '*${PAT}*.zip' found on remote"; exit 1; }

dst="rl/runs/${rel#./}"
mkdir -p "$(dirname "$dst")"
scp -P "$PORT" "$REMOTE:$RUNS/${rel#./}" "$dst"
echo "$dst"
