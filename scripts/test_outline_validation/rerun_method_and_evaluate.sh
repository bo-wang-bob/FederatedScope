#!/usr/bin/env bash
set -euo pipefail

case_id="$1"
method="$2"
gpu_index="$3"
run_tag="${4:-outline_rerun_20260824}"
repo="/root/autodl-tmp/FederatedScope"
python_bin="/root/.local/share/mamba/envs/GGEUR/bin/python"
control_dir="$repo/exp/outline_run_20260824/$run_tag/${case_id}_${method}_retry"
mkdir -p "$control_dir"
exec > >(tee -a "$control_dir/orchestration.log") 2>&1
started_at="$(date +%s)"

on_exit() {
  status=$?
  elapsed=$(( $(date +%s) - started_at ))
  if [[ $status -eq 0 ]]; then
    printf 'METHOD_RETRY_COMPLETE=true\nELAPSED_SECONDS=%s\n' "$elapsed" \
      > "$control_dir/COMPLETE.txt"
  else
    printf 'METHOD_RETRY_COMPLETE=false\nEXIT_CODE=%s\nELAPSED_SECONDS=%s\n' \
      "$status" "$elapsed" > "$control_dir/FAILED.txt"
  fi
}
trap on_exit EXIT

cd "$repo"
echo "METHOD_RETRY_START case=$case_id method=$method gpu=$gpu_index"
CUDA_VISIBLE_DEVICES="$gpu_index" "$python_bin" \
  scripts/test_outline_validation/run_accuracy_case.py \
  --case "$case_id" --method "$method" --seeds 42 43 44 --rounds 100
CUDA_VISIBLE_DEVICES="$gpu_index" "$python_bin" \
  scripts/test_outline_validation/evaluate_saved_mlp.py \
  --case "$case_id" --seeds 42 43 44 --device cuda:0
echo "METHOD_RETRY_AND_EVALUATION_COMPLETE case=$case_id method=$method"
