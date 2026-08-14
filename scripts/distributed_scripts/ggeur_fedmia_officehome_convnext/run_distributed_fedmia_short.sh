#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
TASK_DIR="$ROOT_DIR/scripts/distributed_scripts/ggeur_fedmia_officehome_convnext"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/zqq/.conda/envs/fs_zqq/bin/python}"
RUN_ID="${RUN_ID:-officehome_60c_no_defense_$(date +%Y%m%d_%H%M%S)}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-51251}"
CLIENT_NUM=60
TOTAL_ROUNDS="${TOTAL_ROUNDS:-100}"
GEN_NUM="${GEN_NUM:-0}"
SEED="${SEED:-12345}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
FEATURE_EXTRACTOR="${FEATURE_EXTRACTOR:-cnn}"
FEATURE_CACHE_DIR="${FEATURE_CACHE_DIR:-exp/headonly_system/cache/officehome_60c_no_defense}"
FEDMIA_ENABLED="${FEDMIA_ENABLED:-1}"
FEDMIA_TARGET_CLIENT_ID="${FEDMIA_TARGET_CLIENT_ID:-3}"
FEDMIA_PROBE_SIZE="${FEDMIA_PROBE_SIZE:-200}"
FEDMIA_NONMEMBER_SIZE="${FEDMIA_NONMEMBER_SIZE:-0}"
FEDMIA_SAVE_INTERVAL="${FEDMIA_SAVE_INTERVAL:-10}"
FEDMIA_STORE_GRAD_COS="${FEDMIA_STORE_GRAD_COS:-1}"
FEDMIA_ROUND_AGG="${FEDMIA_ROUND_AGG:-mean}"
FEDMIA_SHADOW_STAT_MODE="${FEDMIA_SHADOW_STAT_MODE:-indexed}"
FEDMIA_VAR_FLOOR="${FEDMIA_VAR_FLOOR:-1e-8}"
FEDMIA_COMPUTE_ALL_CLIENTS="${FEDMIA_COMPUTE_ALL_CLIENTS:-0}"
FEDMIA_MODE="${FEDMIA_MODE:-mix}"
FEDMIA_MIX_LENGTH="${FEDMIA_MIX_LENGTH:-1000}"
FEDMIA_CROSS_EVAL="${FEDMIA_CROSS_EVAL:-1}"
GGEUR_NORMALIZE_FIELDS="${GGEUR_NORMALIZE_FIELDS:-train_losses}"
CLIENT_START_GAP="${CLIENT_START_GAP:-2}"
SERVER_START_WAIT="${SERVER_START_WAIT:-8}"
RUN_TIMEOUT="${RUN_TIMEOUT:-172800}"
USE_GPU="${USE_GPU:-0}"
DEVICE="${DEVICE:-0}"

RUN_DIR="${RUN_DIR:-$ROOT_DIR/exp/headonly_system/runs/$RUN_ID}"
CONFIG_DIR="$RUN_DIR/configs"
LOG_DIR="$RUN_DIR/logs"
PID_DIR="$RUN_DIR/pids"
METRIC_DIR="$RUN_DIR/metrics"
SYSTEM_DIR="$RUN_DIR/system"

mkdir -p "$CONFIG_DIR" "$LOG_DIR" "$PID_DIR" "$METRIC_DIR" "$SYSTEM_DIR"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

"$PYTHON_BIN" - "$CONFIG_DIR" "$SERVER_HOST" "$SERVER_PORT" "$CLIENT_NUM" "$TOTAL_ROUNDS" "$GEN_NUM" "$SEED" "$BATCH_SIZE" "$LEARNING_RATE" "$FEATURE_EXTRACTOR" "$FEATURE_CACHE_DIR" "$FEDMIA_ENABLED" "$FEDMIA_TARGET_CLIENT_ID" "$FEDMIA_PROBE_SIZE" "$FEDMIA_NONMEMBER_SIZE" "$FEDMIA_SAVE_INTERVAL" "$FEDMIA_STORE_GRAD_COS" "$FEDMIA_ROUND_AGG" "$FEDMIA_SHADOW_STAT_MODE" "$FEDMIA_VAR_FLOOR" "$FEDMIA_COMPUTE_ALL_CLIENTS" "$FEDMIA_MODE" "$FEDMIA_MIX_LENGTH" "$FEDMIA_CROSS_EVAL" "$GGEUR_NORMALIZE_FIELDS" "$USE_GPU" "$DEVICE" <<'PY'
import sys
from pathlib import Path

config_dir = Path(sys.argv[1])
run_dir = config_dir.parent
server_host = sys.argv[2]
server_port = int(sys.argv[3])
client_num = int(sys.argv[4])
total_rounds = int(sys.argv[5])
gen_num = int(sys.argv[6])
seed = int(sys.argv[7])
batch_size = int(sys.argv[8])
learning_rate = float(sys.argv[9])
feature_extractor = sys.argv[10].lower()
feature_cache_dir = sys.argv[11]
fedmia_enabled = sys.argv[12] not in {'0', 'false', 'False'}
fedmia_target_client_id = int(sys.argv[13])
fedmia_probe_size = int(sys.argv[14])
fedmia_nonmember_size = int(sys.argv[15])
fedmia_save_interval = int(sys.argv[16])
fedmia_store_grad_cos = sys.argv[17] not in {'0', 'false', 'False'}
fedmia_round_agg = sys.argv[18]
fedmia_shadow_stat_mode = sys.argv[19]
fedmia_var_floor = float(sys.argv[20])
fedmia_compute_all_clients = sys.argv[21] not in {'0', 'false', 'False'}
fedmia_mode = sys.argv[22]
fedmia_mix_length = int(sys.argv[23])
fedmia_cross_eval = sys.argv[24] not in {'0', 'false', 'False'}
ggeur_normalize_fields = [
    item.strip() for item in sys.argv[25].split(',') if item.strip()
]
use_gpu = sys.argv[26] not in {'0', 'false', 'False'}
device = int(sys.argv[27])
normalize_fields_yaml = ', '.join(
    repr(item) for item in ggeur_normalize_fields)

if feature_extractor not in {'cnn', 'clip'}:
    raise ValueError('FEATURE_EXTRACTOR must be cnn or clip')
if not 1 <= fedmia_target_client_id <= client_num:
    raise ValueError(
        f'FEDMIA_TARGET_CLIENT_ID={fedmia_target_client_id} must be in '
        f'[1, CLIENT_NUM={client_num}]')

bool_yaml = lambda value: 'True' if value else 'False'
if feature_extractor == 'cnn':
    extractor_cfg = """  feature_extractor: 'cnn'
  cnn_backbone: 'convnext_base'
  cnn_pretrained: True
  embedding_dim: 1024"""
else:
    extractor_cfg = """  feature_extractor: 'clip'
  clip_model: 'ViT-B-16'
  clip_pretrained: 'openai'
  clip_model_path: '/root/autodl-tmp/models/open_clip_vitb16.bin'
  embedding_dim: 512"""

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
  total_round_num: {total_rounds}
  sample_client_num: {client_num}
  make_global_eval: False
  online_aggr: False

data:
  type: 'office-home'
  root: '/root/autodl-tmp/zqq/OfficeHomeDataset_10072016'
  splits: [0.7, 0.0, 0.3]

dataloader:
  batch_size: {batch_size}
  num_workers: 0

model:
  type: 'ggeur_mlp'
  num_classes: 65

train:
  local_update_steps: 1
  batch_or_epoch: epoch
  optimizer:
    type: 'Adam'
    lr: {learning_rate}
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
  distributed_fedmia: {bool_yaml(fedmia_enabled)}
  fedmia_target_client_id: {fedmia_target_client_id}
  fedmia_probe_size: {fedmia_probe_size}
  fedmia_nonmember_size: {fedmia_nonmember_size}
  fedmia_save_interval: {fedmia_save_interval}
  fedmia_store_grad_cos: {bool_yaml(fedmia_store_grad_cos)}
  fedmia_round_agg: '{fedmia_round_agg}'
  fedmia_i_round_agg: '{fedmia_round_agg}'
  fedmia_shadow_stat_mode: '{fedmia_shadow_stat_mode}'
  fedmia_var_floor: {fedmia_var_floor}
  fedmia_compute_all_clients: {bool_yaml(fedmia_compute_all_clients)}
  mode: '{fedmia_mode}'
  mix_length: {fedmia_mix_length}
  ggeur_target_size: {fedmia_probe_size}
  ggeur_normalize_fields: [{normalize_fields_yaml}]
  fedmia_use_augmented_nonmember: False
  fedmia_use_image_aug_member: False
  fedmia_cross_eval: {bool_yaml(fedmia_cross_eval)}

ggeur:
  use: True
  head_only_mode: True
  head_only_after_round0: True
  headonly_cache_version: 'officehome_60c_no_defense'
  headonly_eval_mode: 'server'
{extractor_cfg}
  freeze_backbone: True
  use_feature_cache: True
  unload_extractor_after_cache: True
  use_fp16_extraction: True
  extract_batch_size: 16
  feature_cache_dir: '{feature_cache_dir}'
  num_generated_per_sample: {gen_num}
  num_generated_per_prototype: {gen_num}
  target_size_per_class: {gen_num}
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
expname: 'headonly_system_distributed_fedmia_server'
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
expname: 'headonly_system_distributed_fedmia_client_{client_id}'
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
  echo "gen_num=$GEN_NUM"
  echo "seed=$SEED"
  echo "batch_size=$BATCH_SIZE"
  echo "learning_rate=$LEARNING_RATE"
  echo "feature_extractor=$FEATURE_EXTRACTOR"
  echo "fedmia_enabled=$FEDMIA_ENABLED"
  echo "fedmia_target_client_id=$FEDMIA_TARGET_CLIENT_ID"
  echo "fedmia_probe_size=$FEDMIA_PROBE_SIZE"
  echo "fedmia_nonmember_size=$FEDMIA_NONMEMBER_SIZE"
  echo "fedmia_save_interval=$FEDMIA_SAVE_INTERVAL"
  echo "fedmia_shadow_stat_mode=$FEDMIA_SHADOW_STAT_MODE"
  echo "fedmia_mode=$FEDMIA_MODE"
  echo "fedmia_mix_length=$FEDMIA_MIX_LENGTH"
  echo "fedmia_cross_eval=$FEDMIA_CROSS_EVAL"
  echo "ggeur_normalize_fields=$GGEUR_NORMALIZE_FIELDS"
  echo "client_start_gap=$CLIENT_START_GAP"
  echo "run_timeout=$RUN_TIMEOUT"
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

nohup "$PYTHON_BIN" "$ROOT_DIR/federatedscope/main.py" --cfg "$CONFIG_DIR/server.yaml" \
  > "$LOG_DIR/server.log" 2>&1 &
echo "$!" > "$PID_DIR/server.pid"
sleep "$SERVER_START_WAIT"

for client_id in $(seq 1 "$CLIENT_NUM"); do
  nohup "$PYTHON_BIN" "$ROOT_DIR/federatedscope/main.py" --cfg "$CONFIG_DIR/client_${client_id}.yaml" \
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
  "$LOG_DIR" \
  --output "$METRIC_DIR/metrics_summary.json" \
  > "$METRIC_DIR/metrics_summary.pretty.json" || true

{
  echo "finished_at=$(date '+%Y-%m-%d %H:%M:%S %z')"
  nvidia-smi || true
} | tee -a "$SYSTEM_DIR/run_info.log"

grep -q "Training finished" "$LOG_DIR/server.log"
echo "$RUN_DIR"
