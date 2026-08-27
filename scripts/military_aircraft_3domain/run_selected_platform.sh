#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/autodl-tmp/FederatedScope}"
RUN_ROOT="${RUN_ROOT:-exp/military_aircraft_3domain}"
PYTHON_BIN="/root/.local/share/mamba/envs/GGEUR/bin/python"
export PYTHONUNBUFFERED=1

cd "$REPO_DIR"
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/formal/platform"
log="$RUN_ROOT/logs/platform_formal_console.log"
start_epoch="$(date +%s)"
PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON_BIN" -m federatedscope.main \
  --cfg scripts/military_aircraft_3domain/configs/platform_pilot.yaml \
  federate.total_round_num 100 \
  outdir "$RUN_ROOT/formal/platform" \
  expname military_aircraft_vit_mlp_platform_formal \
  2>&1 | tee "$log"
"$PYTHON_BIN" scripts/military_aircraft_3domain/summarize_accuracy.py \
  --method platform --log "$log" \
  --output "$RUN_ROOT/formal/platform/accuracy_summary.json"
end_epoch="$(date +%s)"
echo "SELECTED_PLATFORM_COMPLETE=true ELAPSED_SECONDS=$((end_epoch-start_epoch))"
