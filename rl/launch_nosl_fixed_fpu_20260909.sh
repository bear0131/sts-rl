#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
exec bash "$repo/experiments/launchers/launch_nosl_fixed_fpu_20260909.sh" "$@"
