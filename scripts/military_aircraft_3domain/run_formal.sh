#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/autodl-tmp/FederatedScope}"
RUN_ROOT="${RUN_ROOT:-exp/military_aircraft_3domain}"
export PYTHONUNBUFFERED=1

if [[ -x /root/.local/share/mamba/envs/GGEUR/bin/python ]]; then
  PYTHON_BIN=/root/.local/share/mamba/envs/GGEUR/bin/python
else
  PYTHON_BIN=python
fi

cd "$REPO_DIR"
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/formal"

for method in fedavg fedprox platform; do
  config="scripts/military_aircraft_3domain/configs/${method}_pilot.yaml"
  log="$RUN_ROOT/logs/${method}_formal_console.log"
  echo "START_FORMAL_METHOD=$method"
  start_epoch="$(date +%s)"
  PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON_BIN" -m federatedscope.main --cfg "$config" \
    federate.total_round_num 100 \
    outdir "$RUN_ROOT/formal/$method" \
    expname "military_aircraft_vit_mlp_${method}_formal" \
    2>&1 | tee "$log"
  "$PYTHON_BIN" scripts/military_aircraft_3domain/summarize_accuracy.py \
    --method "$method" \
    --log "$log" \
    --output "$RUN_ROOT/formal/$method/accuracy_summary.json"
  end_epoch="$(date +%s)"
  echo "FORMAL_METHOD_COMPLETE=$method ELAPSED_SECONDS=$((end_epoch-start_epoch))"
done

echo "MILITARY_AIRCRAFT_FORMAL_COMPLETE=true"
