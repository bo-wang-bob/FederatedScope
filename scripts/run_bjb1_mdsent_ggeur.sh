#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/root/autodl-tmp/FederatedScope}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/fs/bin/python}"

cd "$PROJECT_DIR"
mkdir -p exp/ggeur_remote_logs

timestamp="$(date +%Y%m%d_%H%M%S)"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

"$PYTHON_BIN" federatedscope/main.py \
  --cfg scripts/example_configs/ggeur_baseline_rnn/ggeur_rnn_mdsent_rating4.yaml \
  2>&1 | tee "exp/ggeur_remote_logs/rnn_ggeur_${timestamp}.log"

"$PYTHON_BIN" federatedscope/main.py \
  --cfg scripts/example_configs/ggeur_baseline_lstm/ggeur_lstm_mdsent_rating4.yaml \
  2>&1 | tee "exp/ggeur_remote_logs/lstm_ggeur_${timestamp}.log"
