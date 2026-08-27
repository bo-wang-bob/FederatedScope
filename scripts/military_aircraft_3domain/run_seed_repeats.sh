#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/autodl-tmp/FederatedScope}"
RUN_ROOT="${RUN_ROOT:-exp/military_aircraft_3domain/seed_repeats}"
PYTHON_BIN="${PYTHON_BIN:-/root/.local/share/mamba/envs/GGEUR/bin/python}"
export PYTHONUNBUFFERED=1

cd "$REPO_DIR"
mkdir -p "$RUN_ROOT"

for seed in 43 44; do
  seed_root="$RUN_ROOT/seed_${seed}"
  mkdir -p "$seed_root/logs"
  for method in fedavg fedprox platform; do
    config="scripts/military_aircraft_3domain/configs/${method}_pilot.yaml"
    log="$seed_root/logs/${method}_100round.log"
    output="$seed_root/${method}/accuracy_summary.json"
    start_epoch="$(date +%s)"
    echo "SEED_RUN_START seed=$seed method=$method rounds=100"

    extra_args=()
    if [[ "$method" == "platform" ]]; then
      extra_args+=(
        ggeur.reuse_augmented_feature_cache False
        ggeur.augmented_feature_cache_dir "$seed_root/platform_augmented_cache"
      )
    fi

    PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}" \
      "$PYTHON_BIN" -m federatedscope.main --cfg "$config" \
      seed "$seed" \
      ggeur.lds_seed "$seed" \
      federate.total_round_num 100 \
      outdir "$seed_root/$method" \
      expname "military_aircraft_${method}_seed_${seed}" \
      "${extra_args[@]}" \
      2>&1 | tee "$log"

    "$PYTHON_BIN" scripts/military_aircraft_3domain/summarize_accuracy.py \
      --method "$method" \
      --log "$log" \
      --output "$output"
    end_epoch="$(date +%s)"
    echo "SEED_RUN_COMPLETE seed=$seed method=$method elapsed_seconds=$((end_epoch-start_epoch))"
  done
done

echo "MILITARY_AIRCRAFT_SEED_REPEATS_COMPLETE=true"
