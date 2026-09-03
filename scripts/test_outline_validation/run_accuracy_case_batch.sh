#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 CASE_ID GPU_INDEX [RUN_TAG]" >&2
  exit 2
fi

case_id="$1"
gpu_index="$2"
run_tag="${3:-outline_rerun_20260824}"
repo="/root/autodl-tmp/FederatedScope"
python_bin="/root/.local/share/mamba/envs/GGEUR/bin/python"
case_dir="$repo/exp/test_outline_validation/$case_id"
control_dir="$repo/exp/outline_run_20260824/$run_tag/$case_id"
archive_root="$repo/exp/test_outline_validation/_archive_20260824"

mkdir -p "$control_dir" "$archive_root"
exec > >(tee -a "$control_dir/orchestration.log") 2>&1

started_at="$(date +%s)"
echo "CASE_BATCH_START case=$case_id gpu=$gpu_index started_at_unix=$started_at"

on_exit() {
  status=$?
  finished_at="$(date +%s)"
  elapsed=$((finished_at - started_at))
  if [[ $status -eq 0 ]]; then
    printf 'CASE_BATCH_COMPLETE=true\nELAPSED_SECONDS=%s\n' "$elapsed" \
      > "$control_dir/COMPLETE.txt"
    echo "CASE_BATCH_COMPLETE case=$case_id elapsed_seconds=$elapsed"
  else
    printf 'CASE_BATCH_COMPLETE=false\nEXIT_CODE=%s\nELAPSED_SECONDS=%s\n' \
      "$status" "$elapsed" > "$control_dir/FAILED.txt"
    echo "CASE_BATCH_FAILED case=$case_id exit_code=$status elapsed_seconds=$elapsed"
  fi
}
trap on_exit EXIT

if [[ -d "$case_dir" ]] && find "$case_dir" -mindepth 1 -print -quit | grep -q .; then
  archive="$archive_root/${case_id}_$(date +%Y%m%d_%H%M%S)"
  mv "$case_dir" "$archive"
  echo "PREVIOUS_RESULTS_ARCHIVED=$archive"
fi
mkdir -p "$case_dir"

cd "$repo"
for method in fedavg fedprox platform; do
  echo "METHOD_START case=$case_id method=$method"
  CUDA_VISIBLE_DEVICES="$gpu_index" "$python_bin" \
    scripts/test_outline_validation/run_accuracy_case.py \
    --case "$case_id" --method "$method" --seeds 42 43 44 --rounds 100
  echo "METHOD_COMPLETE case=$case_id method=$method"
done

CUDA_VISIBLE_DEVICES="$gpu_index" "$python_bin" \
  scripts/test_outline_validation/evaluate_saved_mlp.py \
  --case "$case_id" --seeds 42 43 44 --device cuda:0

echo "INDEPENDENT_EVALUATION_COMPLETE case=$case_id"
