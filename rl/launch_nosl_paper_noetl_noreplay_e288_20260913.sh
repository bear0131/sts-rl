#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
exec bash "$repo/experiments/launchers/launch_nosl_paper_noetl_noreplay_e288_20260913.sh" "$@"
