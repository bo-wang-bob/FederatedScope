#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/autodl-tmp/FederatedScope}"
RUN_ROOT="${RUN_ROOT:-exp/military_aircraft_3domain}"
PYTHON_BIN="/root/.local/share/mamba/envs/GGEUR/bin/python"
export PYTHONUNBUFFERED=1

cd "$REPO_DIR"
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/tuning"

for spec in \
  t20_s000:20:0.0 \
  t20_s005:20:0.05 \
  t20_s010:20:0.1 \
  t40_s005:40:0.05 \
  t40_s010:40:0.1 \
  t40_s025:40:0.25; do
  IFS=: read -r name target scale <<< "$spec"
  log="$RUN_ROOT/logs/platform_tune_${name}.log"
  out="$RUN_ROOT/tuning/$name"
  cache="$RUN_ROOT/platform_cache_${name}"
  echo "START_PLATFORM_TUNING=$name TARGET=$target COVARIANCE_SCALE=$scale"
  start_epoch="$(date +%s)"
  PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON_BIN" -m federatedscope.main \
    --cfg scripts/military_aircraft_3domain/configs/platform_pilot.yaml \
    federate.total_round_num 60 \
    ggeur.num_generated_per_sample 1 \
    ggeur.num_generated_per_prototype 20 \
    ggeur.target_size_per_class "$target" \
    ggeur.generation_covariance_scale "$scale" \
    ggeur.reuse_augmented_feature_cache False \
    ggeur.augmented_feature_cache_dir "$cache" \
    outdir "$out" \
    expname "military_aircraft_platform_${name}" \
    2>&1 | tee "$log"
  "$PYTHON_BIN" scripts/military_aircraft_3domain/summarize_accuracy.py \
    --method "platform_${name}" --log "$log" \
    --output "$out/accuracy_summary.json"
  end_epoch="$(date +%s)"
  echo "PLATFORM_TUNING_COMPLETE=$name ELAPSED_SECONDS=$((end_epoch-start_epoch))"
done

echo "MILITARY_AIRCRAFT_PLATFORM_TUNING_COMPLETE=true"
