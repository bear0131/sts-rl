#!/usr/bin/env bash
set -euo pipefail

# Canonical contract-10 entry point. Keep the implementation in the previous
# launcher path so existing remote automation continues to work.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$script_dir/launch_vector_rule_v7_contract_v9.sh" "$@"
