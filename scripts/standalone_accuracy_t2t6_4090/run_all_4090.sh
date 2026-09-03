#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/autodl-tmp/FederatedScope}"
if [[ -x /root/miniconda3/envs/fs/bin/python ]]; then
  PYTHON_BIN=/root/miniconda3/envs/fs/bin/python
elif [[ -x /root/.local/share/mamba/envs/GGEUR/bin/python ]]; then
  PYTHON_BIN=/root/.local/share/mamba/envs/GGEUR/bin/python
else
  PYTHON_BIN=python
fi

cd "$REPO_DIR"
"$PYTHON_BIN" scripts/standalone_accuracy_t2t6_4090/preflight_4090.py
"$PYTHON_BIN" scripts/standalone_accuracy_t2t6_4090/run_t2t6_4090.py \
  --python "$PYTHON_BIN" \
  --run-id t2t6_4090_final
"$PYTHON_BIN" scripts/standalone_accuracy_t2t6_4090/summarize_results.py \
  exp/t2t6_4090_final/summary.json
