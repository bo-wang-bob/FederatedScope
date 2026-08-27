#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 3 ]]; then
  echo "用法: run_case_step.sh T-编号 步骤号 root [--plan-only]" >&2
  exit 2
fi

CASE_ID="$1"
STEP="$2"
ROLE="$3"
MODE="${4:-}"
PROJECT_DIR="/root/autodl-tmp/FederatedScope"
PYTHON_BIN="/root/.local/share/mamba/envs/GGEUR/bin/python"
ARGS=(
  "$PROJECT_DIR/scripts/test_outline_validation/case_step_runner.py"
  --project-dir "$PROJECT_DIR"
  --case-id "$CASE_ID"
  --step "$STEP"
  --role "$ROLE"
)
if [[ "$MODE" == "--plan-only" ]]; then
  ARGS+=(--plan-only)
fi
"$PYTHON_BIN" "${ARGS[@]}"
