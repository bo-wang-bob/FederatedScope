#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/autodl-tmp/FederatedScope}"
RUN_ROOT="${RUN_ROOT:-exp/military_aircraft_3domain}"
PYTHON_BIN="/root/.local/share/mamba/envs/GGEUR/bin/python"
export PYTHONUNBUFFERED=1

cd "$REPO_DIR"
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/multiproto_tuning"

for spec in \
  p64_t20_s000:64:20:0.0 \
  p64_t40_s000:64:40:0.0 \
  p64_t80_s000:64:80:0.0 \
  p64_t20_s001:64:20:0.01 \
  p64_t40_s001:64:40:0.01 \
  p64_t80_s001:64:80:0.01; do
  IFS=: read -r name proto target scale <<< "$spec"
  log="$RUN_ROOT/logs/platform_multiproto_${name}.log"
  out="$RUN_ROOT/multiproto_tuning/$name"
  cache="$RUN_ROOT/platform_cache_${name}"
  echo "START_MULTIPROTO_TUNING=$name PROTOTYPES=$proto TARGET=$target SCALE=$scale"
  start_epoch="$(date +%s)"
  PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON_BIN" -m federatedscope.main \
    --cfg scripts/military_aircraft_3domain/configs/platform_pilot.yaml \
    federate.total_round_num 100 \
    ggeur.local_prototypes_per_class "$proto" \
    ggeur.max_cross_client_prototypes_per_class 0 \
    ggeur.num_generated_per_sample 0 \
    ggeur.num_generated_per_prototype 1 \
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
  echo "MULTIPROTO_TUNING_COMPLETE=$name ELAPSED_SECONDS=$((end_epoch-start_epoch))"
done

echo "MILITARY_AIRCRAFT_MULTIPROTO_TUNING_COMPLETE=true"
