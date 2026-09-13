#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
exec bash "$repo/experiments/launchers/launch_nosl_paper_replay_fresh10_20260913.sh" "$@"
