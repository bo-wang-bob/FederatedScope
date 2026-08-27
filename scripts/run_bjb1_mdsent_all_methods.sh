#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/root/autodl-tmp/FederatedScope}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/fs/bin/python}"
WAIT_SESSION="${WAIT_SESSION:-ggeur_mdsent}"

cd "$PROJECT_DIR"
mkdir -p exp/ggeur_remote_logs exp/ggeur_final_5models/mdsent_summary

timestamp="$(date +%Y%m%d_%H%M%S)"
queue_log="exp/ggeur_remote_logs/mdsent_all_methods_${timestamp}.log"
summary_file="exp/ggeur_final_5models/mdsent_summary/summary_${timestamp}.csv"

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$queue_log"
}

has_final_result() {
  local outdir="$1"
  find "$outdir" -path '*/eval_results.log' -type f 2>/dev/null \
    -exec grep -l "'Round': 'Final'" {} \; | grep -q .
}

run_case() {
  local group="$1"
  local method="$2"
  local cfg="$3"
  local outdir="$4"
  shift 4
  local fallback_outdirs=("$@")

  if has_final_result "$outdir"; then
    log "SKIP completed ${group}/${method}: ${outdir}"
    return
  fi

  for fallback in "${fallback_outdirs[@]}"; do
    if [[ -n "$fallback" ]] && has_final_result "$fallback"; then
      log "SKIP completed ${group}/${method}: ${fallback}"
      return
    fi
  done

  local case_log="exp/ggeur_remote_logs/${group}_${method}_${timestamp}.log"
  log "START ${group}/${method} cfg=${cfg}"
  "$PYTHON_BIN" federatedscope/main.py --cfg "$cfg" 2>&1 | tee "$case_log"
  log "DONE ${group}/${method}"
}

if [[ -n "$WAIT_SESSION" ]]; then
  while screen -ls 2>/dev/null | grep -q "[.]${WAIT_SESSION}[[:space:]]"; do
    log "Waiting for existing screen session ${WAIT_SESSION} to finish..."
    sleep 300
  done
fi

run_case mdsent_rnn fedavg \
  scripts/example_configs/ggeur_final_5models/mdsent_rnn/fedavg.yaml \
  exp/ggeur_final_5models/mdsent_rnn/fedavg
run_case mdsent_rnn fedprox \
  scripts/example_configs/ggeur_final_5models/mdsent_rnn/fedprox.yaml \
  exp/ggeur_final_5models/mdsent_rnn/fedprox
run_case mdsent_rnn fedproto \
  scripts/example_configs/ggeur_final_5models/mdsent_rnn/fedproto.yaml \
  exp/ggeur_final_5models/mdsent_rnn/fedproto
run_case mdsent_rnn fedopt \
  scripts/example_configs/ggeur_final_5models/mdsent_rnn/fedopt.yaml \
  exp/ggeur_final_5models/mdsent_rnn/fedopt
run_case mdsent_rnn ggeur \
  scripts/example_configs/ggeur_final_5models/mdsent_rnn/ggeur.yaml \
  exp/ggeur_final_5models/mdsent_rnn/ggeur \
  exp/ggeur_rnn_sentiment/ggeur_mdsent_rating4_rnn

run_case mdsent_lstm fedavg \
  scripts/example_configs/ggeur_final_5models/mdsent_lstm/fedavg.yaml \
  exp/ggeur_final_5models/mdsent_lstm/fedavg
run_case mdsent_lstm fedprox \
  scripts/example_configs/ggeur_final_5models/mdsent_lstm/fedprox.yaml \
  exp/ggeur_final_5models/mdsent_lstm/fedprox
run_case mdsent_lstm fedproto \
  scripts/example_configs/ggeur_final_5models/mdsent_lstm/fedproto.yaml \
  exp/ggeur_final_5models/mdsent_lstm/fedproto
run_case mdsent_lstm fedopt \
  scripts/example_configs/ggeur_final_5models/mdsent_lstm/fedopt.yaml \
  exp/ggeur_final_5models/mdsent_lstm/fedopt
run_case mdsent_lstm ggeur \
  scripts/example_configs/ggeur_final_5models/mdsent_lstm/ggeur.yaml \
  exp/ggeur_final_5models/mdsent_lstm/ggeur \
  exp/ggeur_lstm_sentiment/ggeur_mdsent_rating4_lstm

"$PYTHON_BIN" scripts/summarize_mdsent_results.py \
  --root exp \
  --output "$summary_file" | tee -a "$queue_log"

log "SUMMARY ${summary_file}"
