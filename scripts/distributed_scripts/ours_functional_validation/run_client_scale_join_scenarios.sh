#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCENARIO_RUNNER="${SCRIPT_DIR}/../ggeur_headonly_system_officehome_vit/run_loopback_multiip_scenarios.sh"

if [[ ! -f "${SCENARIO_RUNNER}" ]]; then
  echo "missing scenario runner: ${SCENARIO_RUNNER}" >&2
  exit 1
fi

export GEN_NUM="${GEN_NUM:-1}"
export TOTAL_ROUNDS="${TOTAL_ROUNDS:-2}"
export CLIENT_START_GAP="${CLIENT_START_GAP:-0}"

exec bash "${SCENARIO_RUNNER}"
