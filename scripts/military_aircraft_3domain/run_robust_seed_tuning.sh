#!/usr/bin/env bash
set -uo pipefail

REPO_DIR="${REPO_DIR:-/root/autodl-tmp/FederatedScope}"
RUN_ROOT="${RUN_ROOT:-exp/military_aircraft_3domain/robust_seed_tuning}"
PYTHON_BIN="${PYTHON_BIN:-/root/.local/share/mamba/envs/GGEUR/bin/python}"
export PYTHONUNBUFFERED=1

cd "$REPO_DIR" || exit 90
mkdir -p "$RUN_ROOT/logs"
rm -f "$RUN_ROOT/exit_code"

finish() {
  code="$1"
  trap - EXIT
  printf '%s\n' "$code" >"$RUN_ROOT/exit_code"
  exit "$code"
}
trap 'finish $?' EXIT

# Shared FL settings for all three methods. All 15 clients participate. The
# same frozen configuration is evaluated for every seed below.
scenario="formal_lr5e5_platform_uncapped"
for seed in 42 43 44; do
for method in fedavg fedprox platform; do
  config="scripts/military_aircraft_3domain/configs/${method}_pilot.yaml"
  run_dir="$RUN_ROOT/scenarios/$scenario/seed_${seed}/$method"
  log="$RUN_ROOT/logs/${scenario}_seed_${seed}_${method}.log"
  summary="$run_dir/accuracy_summary.json"
  mkdir -p "$run_dir"
  extra_args=()
  if [[ "$method" == "platform" ]]; then
    extra_args+=(
      ggeur.reuse_augmented_feature_cache False
      ggeur.augmented_feature_cache_dir \
        "$RUN_ROOT/cache/$scenario/seed_${seed}/platform"
      ggeur.target_size_per_class 0
      ggeur.generation_covariance_scale 0.01
    )
  fi
  start_epoch="$(date +%s)"
  printf 'ROBUST_SCENARIO_START scenario=%s seed=%s method=%s\n' \
    "$scenario" "$seed" "$method"
  PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON_BIN" -m federatedscope.main --cfg "$config" \
      seed "$seed" \
      ggeur.lds_seed "$seed" \
      ggeur.lds_alpha 0.1 \
      train.local_update_steps 3 \
      train.optimizer.lr 0.00005 \
      federate.sample_client_num 0 \
      federate.total_round_num 100 \
      ggeur.domain_personalized_head_epochs 0 \
      ggeur.use_fedproto False \
      outdir "$run_dir" \
      expname "military_aircraft_${scenario}_${method}_seed_${seed}" \
      "${extra_args[@]}" \
      >"$log" 2>&1
  status="$?"
  if [[ "$status" -ne 0 ]]; then
    printf 'ROBUST_SCENARIO_FAILED scenario=%s seed=%s method=%s exit_code=%s log=%s\n' \
      "$scenario" "$seed" "$method" "$status" "$log"
    continue
  fi
  "$PYTHON_BIN" scripts/military_aircraft_3domain/summarize_accuracy.py \
    --method "$method" --log "$log" --output "$summary"
  elapsed="$(( $(date +%s) - start_epoch ))"
  final_accuracy="$($PYTHON_BIN -c \
    "import json; print(json.load(open('$summary'))['final']['average'])")"
  printf 'ROBUST_SCENARIO_COMPLETE scenario=%s seed=%s method=%s final_accuracy=%s elapsed_seconds=%s\n' \
    "$scenario" "$seed" "$method" "$final_accuracy" "$elapsed"
done
done

printf 'ROBUST_SHARED_SCENARIO_COMPLETE=true\n'
trap - EXIT
printf '0\n' >"$RUN_ROOT/exit_code"
