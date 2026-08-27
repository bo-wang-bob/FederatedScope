#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/autodl-tmp/FederatedScope}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/datasets/MilitaryAircraft3D}"
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-/root/autodl-tmp/datasets/MilitaryAircraft3D_downloads}"
RUN_ROOT="${RUN_ROOT:-exp/military_aircraft_3domain}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTHONUNBUFFERED=1

if [[ -x /root/miniconda3/envs/fs/bin/python ]]; then
  PYTHON_BIN=/root/miniconda3/envs/fs/bin/python
elif [[ -x /root/.local/share/mamba/envs/GGEUR/bin/python ]]; then
  PYTHON_BIN=/root/.local/share/mamba/envs/GGEUR/bin/python
else
  PYTHON_BIN=python
fi

cd "$REPO_DIR"
mkdir -p "$RUN_ROOT/logs"

if [[ ! -f "$DATA_ROOT/dataset_manifest.json" ]]; then
  prepare_options=()
  if [[ -d "$DATA_ROOT" ]]; then
    prepare_options+=(--overwrite)
  fi
  "$PYTHON_BIN" scripts/military_aircraft_3domain/prepare_dataset.py \
    --output-root "$DATA_ROOT" \
    --download-root "$DOWNLOAD_ROOT" \
    --samples-per-class 50 \
    "${prepare_options[@]}" \
    2>&1 | tee "$RUN_ROOT/logs/prepare_dataset.log"
else
  echo "DATASET_CACHE_HIT=$DATA_ROOT"
fi

if [[ ! -f "$RUN_ROOT/central_sanity.json" ]]; then
  "$PYTHON_BIN" scripts/military_aircraft_3domain/validate_vit_mlp.py \
    --data-root "$DATA_ROOT" \
    --output "$RUN_ROOT/central_sanity.json" \
    2>&1 | tee "$RUN_ROOT/logs/central_sanity.log"
else
  echo "CENTRAL_SANITY_CACHE_HIT=$RUN_ROOT/central_sanity.json"
fi

for method in fedavg fedprox platform; do
  config="scripts/military_aircraft_3domain/configs/${method}_pilot.yaml"
  echo "START_METHOD=$method"
  start_epoch="$(date +%s)"
  PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON_BIN" -m federatedscope.main --cfg "$config" \
    2>&1 | tee "$RUN_ROOT/logs/${method}_pilot_console.log"
  end_epoch="$(date +%s)"
  echo "METHOD_COMPLETE=$method ELAPSED_SECONDS=$((end_epoch-start_epoch))"
done

echo "MILITARY_AIRCRAFT_PILOT_COMPLETE=true"
echo "LOG_DIR=$RUN_ROOT/logs"
