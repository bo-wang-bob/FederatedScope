#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/zqq/.conda/envs/fs_zqq/bin/python}"
RUN_ID="${RUN_ID:-ggeur_ppa_4c_100r_aligned_$(date +%Y%m%d_%H%M%S)}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-51251}"
CLIENT_NUM="${CLIENT_NUM:-4}"
TOTAL_ROUNDS="${TOTAL_ROUNDS:-100}"
SEED="${SEED:-12345}"
CLIENT_START_GAP="${CLIENT_START_GAP:-10}"
SERVER_START_WAIT="${SERVER_START_WAIT:-8}"
RUN_TIMEOUT="${RUN_TIMEOUT:-86400}"

RUN_DIR="$ROOT_DIR/exp/headonly_system/runs/$RUN_ID"
CONFIG_DIR="$RUN_DIR/configs"
LOG_DIR="$RUN_DIR/logs"
PID_DIR="$RUN_DIR/pids"
METRIC_DIR="$RUN_DIR/metrics"
SYSTEM_DIR="$RUN_DIR/system"

mkdir -p "$CONFIG_DIR" "$LOG_DIR" "$PID_DIR" "$METRIC_DIR" "$SYSTEM_DIR"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

# Generate one server config and four client configs. Training/model/PPA
# parameters follow editable/scripts/attack_exp_scripts/privacy_attack/
# ggeur_meta_ppa_example_60.yaml. Only standalone->distributed and 60->4 are
# changed, plus the distributed PPA switch and process addresses.
"$PYTHON_BIN" - "$CONFIG_DIR" "$SERVER_HOST" "$SERVER_PORT" \
  "$CLIENT_NUM" "$TOTAL_ROUNDS" "$SEED" <<'PY'
import sys
from pathlib import Path

config_dir = Path(sys.argv[1])
run_dir = config_dir.parent
server_host = sys.argv[2]
server_port = int(sys.argv[3])
client_num = int(sys.argv[4])
total_rounds = int(sys.argv[5])
seed = int(sys.argv[6])

if client_num != 4:
    raise ValueError('The aligned distributed PPA experiment uses 4 clients')

common = f"""
use_gpu: True
device: 0
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
    type: Adam
    lr: 0.001
    weight_decay: 0.0

criterion:
  type: CrossEntropyLoss

trainer:
  type: ggeur

eval:
  freq: 5
  metrics: ['acc', 'correct']
  split: ['test']
  best_res_update_round_wise_key: test_acc

attack:
  attack_method: ggeur_ppa
  use_ggeur: True
  modular_attacks: True
  attack_plugins:
    - meta_ppa
  distributed_fedmia: False
  distributed_ppa: True
  classifier_PIA: svm
  meta_ppa_save_interval: 10
  meta_ppa_probe_samples_per_class: 16
  meta_ppa_probe_epochs: 3
  meta_ppa_probe_lr: 0.001
  meta_ppa_probe_batch_size: 16
  meta_ppa_max_attack_rounds: 10
  meta_ppa_max_clients: 0
  meta_ppa_target_layers: []

ggeur:
  use: True
  head_only_mode: True
  head_only_after_round0: True
  headonly_eval_mode: server
  headonly_cache_version: 'officehome_convnext_ppa4_aligned'
  headonly_skip_round0_if_augmented_cache_exists: False
  feature_extractor: cnn
  cnn_backbone: convnext_base
  cnn_pretrained: True
  freeze_backbone: True
  embedding_dim: 1024
  use_feature_cache: False
  feature_cache_dir: ''
  reuse_augmented_feature_cache: False
  unload_extractor_after_cache: True
  num_generated_per_sample: 0
  num_generated_per_prototype: 0
  target_size_per_class: 0
  mlp_hidden_dim: 0
  mlp_dropout: 0
  mlp_weight_decay: 0
  mlp_label_smoothing: 0
  mlp_l2_reg: 0.0
  use_cross_client_prototypes: True
  statistics_round: 0
  distributed_stage_timeout: 3600
  use_fedproto: False
  use_lds: True
  lds_alpha: 0.1
  lds_seed: 42
  use_cnn_distillation: False
  use_feature_alignment: False
  use_separated_training: False
  use_end_to_end_finetune: False
  cnn_model: convnext_base
  cnn_lr: 0.01
  cnn_local_epochs: 1
  cnn_warmup_rounds: 0
  local_decoy:
    use: False
    train_with_decoy: True
    num_per_class: 60
    noise_scale: 0.4
    min_std: 0.0001
    max_total: 0
    seed_offset: 1701

"""

server = common + f"""
distribute:
  use: True
  role: server
  server_host: '{server_host}'
  server_port: {server_port}

outdir: '{run_dir / 'outputs' / 'server'}'
expname: 'headonly_system_distributed_ppa_aligned_server'
"""
config_dir.joinpath('server.yaml').write_text(server, encoding='utf-8')

for client_id in range(1, client_num + 1):
    client = common + f"""
distribute:
  use: True
  role: client
  server_host: '{server_host}'
  server_port: {server_port}
  client_host: '127.0.0.1'
  client_port: {server_port + client_id}
  data_idx: {client_id}

outdir: '{run_dir / 'outputs' / f'client_{client_id}'}'
expname: 'headonly_system_distributed_ppa_aligned_client_{client_id}'
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
  echo "model=cnn/convnext_base"
  echo "feature_cache=False"
  echo "ppa_classifier=svm"
  echo "ppa_save_interval=10"
  echo "ppa_probes_per_class=16"
  echo "ppa_probe_epochs=3"
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
  find "$RUN_DIR/outputs/server" -name distributed_ppa_results.json \
    -type f -print 2>/dev/null || true
} | tee -a "$SYSTEM_DIR/run_info.log"

grep -q "Training finished" "$LOG_DIR/server.log"
