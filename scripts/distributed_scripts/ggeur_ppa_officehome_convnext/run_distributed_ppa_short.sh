#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/zqq/.conda/envs/fs_zqq/bin/python}"
RUN_ID="${RUN_ID:-officehome_60c_no_defense_$(date +%Y%m%d_%H%M%S)}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-51251}"
CLIENT_NUM=60
TOTAL_ROUNDS="${TOTAL_ROUNDS:-100}"
SEED="${SEED:-12345}"
PPA_SAVE_INTERVAL="${PPA_SAVE_INTERVAL:-10}"
PPA_PROBES_PER_CLASS="${PPA_PROBES_PER_CLASS:-16}"
PPA_PROBE_EPOCHS="${PPA_PROBE_EPOCHS:-3}"
PPA_CLASSIFIER="${PPA_CLASSIFIER:-svm}"
CLIENT_START_GAP="${CLIENT_START_GAP:-2}"
SERVER_START_WAIT="${SERVER_START_WAIT:-8}"
RUN_TIMEOUT="${RUN_TIMEOUT:-172800}"
USE_GPU="${USE_GPU:-0}"
DEVICE="${DEVICE:-0}"

RUN_DIR="$ROOT_DIR/exp/headonly_system/runs/$RUN_ID"
CONFIG_DIR="$RUN_DIR/configs"
LOG_DIR="$RUN_DIR/logs"
PID_DIR="$RUN_DIR/pids"
METRIC_DIR="$RUN_DIR/metrics"
SYSTEM_DIR="$RUN_DIR/system"
FEATURE_CACHE_DIR="${FEATURE_CACHE_DIR:-$ROOT_DIR/exp/headonly_system/cache/officehome_60c_no_defense}"

mkdir -p "$CONFIG_DIR" "$LOG_DIR" "$PID_DIR" "$METRIC_DIR" "$SYSTEM_DIR"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

"$PYTHON_BIN" - "$CONFIG_DIR" "$SERVER_HOST" "$SERVER_PORT" \
  "$CLIENT_NUM" "$TOTAL_ROUNDS" "$SEED" "$PPA_SAVE_INTERVAL" \
  "$PPA_PROBES_PER_CLASS" "$PPA_PROBE_EPOCHS" "$PPA_CLASSIFIER" \
  "$FEATURE_CACHE_DIR" "$USE_GPU" "$DEVICE" <<'PY'
import sys
from pathlib import Path

config_dir = Path(sys.argv[1])
run_dir = config_dir.parent
server_host = sys.argv[2]
server_port = int(sys.argv[3])
client_num = int(sys.argv[4])
total_rounds = int(sys.argv[5])
seed = int(sys.argv[6])
save_interval = int(sys.argv[7])
probes_per_class = int(sys.argv[8])
probe_epochs = int(sys.argv[9])
classifier = sys.argv[10]
feature_cache_dir = sys.argv[11]
use_gpu = sys.argv[12] not in {"0", "false", "False"}
device = int(sys.argv[13])
bool_yaml = lambda value: "True" if value else "False"

if classifier not in {'svm', 'randomforest', 'lr'}:
    raise ValueError('PPA_CLASSIFIER must be svm, randomforest, or lr')

common = f"""
use_gpu: {bool_yaml(use_gpu)}
device: {device}
seed: {seed}
verbose: 1

early_stop:
  patience: 0

federate:
  method: 'ggeur'
  mode: 'distributed'
  client_num: {client_num}
  sample_client_num: {client_num}
  total_round_num: {total_rounds}
  make_global_eval: False
  online_aggr: False

data:
  type: 'office-home'
  root: '/root/autodl-tmp/zqq/OfficeHomeDataset_10072016'
  splits: [0.7, 0.0, 0.3]

dataloader:
  batch_size: 8
  num_workers: 0

model:
  type: 'ggeur_mlp'
  num_classes: 65

train:
  local_update_steps: 1
  batch_or_epoch: epoch
  optimizer:
    type: 'Adam'
    lr: 0.001
    weight_decay: 0.0

criterion:
  type: CrossEntropyLoss

eval:
  freq: 5
  metrics: ['acc', 'correct']
  split: ['test']
  best_res_update_round_wise_key: 'test_acc'

trainer:
  type: 'ggeur'

attack:
  attack_method: ''
  distributed_fedmia: False
  distributed_ppa: True
  classifier_PIA: '{classifier}'
  meta_ppa_save_interval: {save_interval}
  meta_ppa_probe_samples_per_class: {probes_per_class}
  meta_ppa_probe_epochs: {probe_epochs}
  meta_ppa_probe_lr: 0.001
  meta_ppa_probe_batch_size: 16
  meta_ppa_max_attack_rounds: 10
  meta_ppa_max_clients: 0
  meta_ppa_target_layers: []

ggeur:
  use: True
  head_only_mode: True
  head_only_after_round0: True
  headonly_cache_version: 'officehome_60c_no_defense'
  headonly_eval_mode: 'server'
  feature_extractor: 'cnn'
  cnn_backbone: 'convnext_base'
  cnn_pretrained: True
  embedding_dim: 1024
  freeze_backbone: True
  use_feature_cache: True
  unload_extractor_after_cache: True
  use_fp16_extraction: True
  extract_batch_size: 16
  feature_cache_dir: '{feature_cache_dir}'
  num_generated_per_sample: 0
  num_generated_per_prototype: 0
  target_size_per_class: 0
  mlp_hidden_dim: 0
  mlp_dropout: 0.0
  use_cross_client_prototypes: True
  statistics_round: 0
  distributed_stage_timeout: 14400
  use_fedproto: False
  use_lds: True
  lds_alpha: 0.1
  lds_seed: 42
  use_cnn_distillation: False
  use_feature_alignment: False
  use_separated_training: False
  use_end_to_end_finetune: False
  use_promptfl: False
"""

server = common + f"""
distribute:
  use: True
  role: 'server'
  join_timeout_seconds: 900
  server_host: '{server_host}'
  server_port: {server_port}

outdir: '{run_dir / 'outputs' / 'server'}'
expname: 'headonly_system_distributed_ppa_server'
"""
config_dir.joinpath('server.yaml').write_text(server, encoding='utf-8')

for client_id in range(1, client_num + 1):
    client = common + f"""
distribute:
  use: True
  role: 'client'
  join_timeout_seconds: 900
  server_host: '{server_host}'
  server_port: {server_port}
  client_host: '127.0.0.1'
  client_port: {server_port + client_id}
  data_idx: {client_id}

outdir: '{run_dir / 'outputs' / f'client_{client_id}'}'
expname: 'headonly_system_distributed_ppa_client_{client_id}'
"""
    config_dir.joinpath(f'client_{client_id}.yaml').write_text(
        client, encoding='utf-8')
PY

{
  echo "run_id=$RUN_ID"
  echo "root_dir=$ROOT_DIR"
  echo "run_dir=$RUN_DIR"
  echo "server_host=$SERVER_HOST"
  echo "server_port=$SERVER_PORT"
  echo "client_num=$CLIENT_NUM"
  echo "total_rounds=$TOTAL_ROUNDS"
  echo "ppa_save_interval=$PPA_SAVE_INTERVAL"
  echo "ppa_probes_per_class=$PPA_PROBES_PER_CLASS"
  echo "ppa_probe_epochs=$PPA_PROBE_EPOCHS"
  echo "ppa_classifier=$PPA_CLASSIFIER"
  echo "started_at=$(date '+%Y-%m-%d %H:%M:%S %z')"
  nvidia-smi || true
} | tee "$SYSTEM_DIR/run_info.log"

if [ "${GENERATE_ONLY:-0}" = "1" ]; then
  echo "$RUN_DIR"
  exit 0
fi

cleanup() {
  for pid_file in "$PID_DIR"/*.pid; do
    [ -f "$pid_file" ] || continue
    pid="$(cat "$pid_file")"
    kill "$pid" >/dev/null 2>&1 || true
  done
}
trap cleanup INT TERM

nohup "$PYTHON_BIN" "$ROOT_DIR/federatedscope/main.py" \
  --cfg "$CONFIG_DIR/server.yaml" > "$LOG_DIR/server.log" 2>&1 &
echo "$!" > "$PID_DIR/server.pid"
sleep "$SERVER_START_WAIT"

for client_id in $(seq 1 "$CLIENT_NUM"); do
  nohup "$PYTHON_BIN" "$ROOT_DIR/federatedscope/main.py" \
    --cfg "$CONFIG_DIR/client_${client_id}.yaml" \
    > "$LOG_DIR/client_${client_id}.log" 2>&1 &
  echo "$!" > "$PID_DIR/client_${client_id}.pid"
  sleep "$CLIENT_START_GAP"
done

deadline=$(( $(date +%s) + RUN_TIMEOUT ))
while true; do
  alive=0
  for pid_file in "$PID_DIR"/*.pid; do
    [ -f "$pid_file" ] || continue
    pid="$(cat "$pid_file")"
    if ps -p "$pid" >/dev/null 2>&1; then
      alive=1
      break
    fi
  done
  [ "$alive" -eq 0 ] && break
  if [ "$(date +%s)" -gt "$deadline" ]; then
    echo "timeout" | tee -a "$SYSTEM_DIR/run_info.log"
    cleanup
    break
  fi
  sleep 5
done

"$PYTHON_BIN" "$ROOT_DIR/scripts/parse_headonly_system_metrics.py" \
  "$LOG_DIR" --output "$METRIC_DIR/metrics_summary.json" \
  > "$METRIC_DIR/metrics_summary.pretty.json" || true

{
  echo "finished_at=$(date '+%Y-%m-%d %H:%M:%S %z')"
  nvidia-smi || true
} | tee -a "$SYSTEM_DIR/run_info.log"

grep -q "Training finished" "$LOG_DIR/server.log"
