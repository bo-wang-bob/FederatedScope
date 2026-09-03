#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/root/autodl-tmp/FederatedScope}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/fs/bin/python}"

cd "$PROJECT_DIR"
mkdir -p exp/ggeur_remote_logs exp/ggeur_final_5models/mdsent_summary

timestamp="$(date +%Y%m%d_%H%M%S)"
queue_log="exp/ggeur_remote_logs/mdsent_lstm_resume_${timestamp}.log"
summary_file="exp/ggeur_final_5models/mdsent_summary/summary_${timestamp}.csv"

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

run_case() {
  local method="$1"
  local cfg="scripts/example_configs/ggeur_final_5models/mdsent_lstm/${method}.yaml"
  local case_log="exp/ggeur_remote_logs/mdsent_lstm_${method}_${timestamp}.log"

  echo "[$(date '+%F %T')] START mdsent_lstm/${method}" | tee -a "$queue_log"
  "$PYTHON_BIN" federatedscope/main.py --cfg "$cfg" 2>&1 | tee "$case_log"
  echo "[$(date '+%F %T')] DONE mdsent_lstm/${method}" | tee -a "$queue_log"
}

run_case fedavg
run_case fedprox
run_case fedproto
run_case fedopt
run_case ggeur

"$PYTHON_BIN" scripts/summarize_mdsent_results.py \
  --root exp \
  --output "$summary_file" | tee -a "$queue_log"

echo "[$(date '+%F %T')] SUMMARY ${summary_file}" | tee -a "$queue_log"
