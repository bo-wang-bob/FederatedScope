#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BASE_SCRIPT="$ROOT_DIR/scripts/distributed_scripts/ggeur_fedmia_officehome_convnext/run_distributed_fedmia_short.sh"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ID="${RUN_ID:-officehome_60c_no_defense}"
RUN_DIR="${RUN_DIR:-$ROOT_DIR/exp/distributed/$RUN_ID}"
CONFIG_DIR="$RUN_DIR/configs"
LOG_DIR="$RUN_DIR/logs"
PID_DIR="$RUN_DIR/pids"
SYSTEM_DIR="$RUN_DIR/system"

SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-51251}"
CLIENT_NUM=60
TOTAL_ROUNDS="${TOTAL_ROUNDS:-100}"
SEED="${SEED:-12345}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
FEATURE_CACHE_DIR="${FEATURE_CACHE_DIR:-exp/headonly_system/cache/officehome_60c_no_defense}"
CLIENT_START_GAP="${CLIENT_START_GAP:-2}"
SERVER_START_WAIT="${SERVER_START_WAIT:-8}"
RUN_TIMEOUT="${RUN_TIMEOUT:-172800}"
JOIN_TIMEOUT_SECONDS="${JOIN_TIMEOUT_SECONDS:-900}"
DISTRIBUTED_STAGE_TIMEOUT="${DISTRIBUTED_STAGE_TIMEOUT:-14400}"
USE_GPU="${USE_GPU:-1}"
DEVICE="${DEVICE:-0}"
EXTRACT_BATCH_SIZE="${EXTRACT_BATCH_SIZE:-16}"
FEATURE_EXTRACTION_LOCK_FILE="${FEATURE_EXTRACTION_LOCK_FILE:-$SYSTEM_DIR/gpu_feature_extraction.lock}"

DP_ENABLED=0
DP_EPSILON="${DP_EPSILON:-6.0}"
DP_DELTA="${DP_DELTA:-1e-5}"
DP_NOISE_MULTIPLIER="${DP_NOISE_MULTIPLIER:-0.05}"
DP_INITIAL_CLIP="${DP_INITIAL_CLIP:-1.0}"
DP_TARGET_QUANTILE="${DP_TARGET_QUANTILE:-0.7}"
DP_EMA="${DP_EMA:-0.9}"
DP_MIN_CLIP="${DP_MIN_CLIP:-0.05}"
DP_MAX_CLIP="${DP_MAX_CLIP:-10.0}"

LOCAL_DECOY_USE=0
LOCAL_DECOY_NUM_PER_CLASS=0
LOCAL_DECOY_NOISE_SCALE="${LOCAL_DECOY_NOISE_SCALE:-0.5}"
LOCAL_DECOY_MAX_TOTAL="${LOCAL_DECOY_MAX_TOTAL:-0}"
LOCAL_DECOY_SEED_OFFSET="${LOCAL_DECOY_SEED_OFFSET:-1234}"

mkdir -p "$CONFIG_DIR" "$LOG_DIR" "$PID_DIR" "$SYSTEM_DIR"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

echo "[prepare] generate 1 server + 60 client configs"
GENERATE_ONLY=1 \
PYTHON_BIN="$PYTHON_BIN" \
RUN_ID="$RUN_ID" \
RUN_DIR="$RUN_DIR" \
SERVER_HOST="$SERVER_HOST" \
SERVER_PORT="$SERVER_PORT" \
CLIENT_NUM="$CLIENT_NUM" \
TOTAL_ROUNDS="$TOTAL_ROUNDS" \
SEED="$SEED" \
BATCH_SIZE="$BATCH_SIZE" \
LEARNING_RATE="$LEARNING_RATE" \
FEATURE_EXTRACTOR=cnn \
FEATURE_CACHE_DIR="$FEATURE_CACHE_DIR" \
GEN_NUM=0 \
FEDMIA_ENABLED=0 \
CLIENT_START_GAP="$CLIENT_START_GAP" \
RUN_TIMEOUT="$RUN_TIMEOUT" \
bash "$BASE_SCRIPT" >"$SYSTEM_DIR/config_generator.log" 2>&1

"$PYTHON_BIN" - \
  "$CONFIG_DIR" "$USE_GPU" "$DEVICE" \
  "$DP_EPSILON" "$DP_DELTA" "$DP_NOISE_MULTIPLIER" \
  "$DP_INITIAL_CLIP" "$DP_TARGET_QUANTILE" "$DP_EMA" \
  "$DP_MIN_CLIP" "$DP_MAX_CLIP" \
  "$JOIN_TIMEOUT_SECONDS" \
  "$LOCAL_DECOY_USE" "$LOCAL_DECOY_NUM_PER_CLASS" \
  "$LOCAL_DECOY_NOISE_SCALE" "$LOCAL_DECOY_MAX_TOTAL" \
  "$LOCAL_DECOY_SEED_OFFSET" \
  "$DISTRIBUTED_STAGE_TIMEOUT" "$EXTRACT_BATCH_SIZE" \
  "$FEATURE_EXTRACTION_LOCK_FILE" "$DP_ENABLED" <<'PY'
import sys
from pathlib import Path

import yaml

config_dir = Path(sys.argv[1])
use_gpu = str(sys.argv[2]).lower() not in {'0', 'false', 'no'}
device = int(sys.argv[3])
epsilon = float(sys.argv[4])
delta = float(sys.argv[5])
noise_multiplier = float(sys.argv[6])
initial_clip = float(sys.argv[7])
target_quantile = float(sys.argv[8])
ema = float(sys.argv[9])
min_clip = float(sys.argv[10])
max_clip = float(sys.argv[11])
join_timeout_seconds = int(sys.argv[12])
local_decoy_use = str(sys.argv[13]).lower() not in {
    '0', 'false', 'no'
}
local_decoy_num_per_class = int(sys.argv[14])
local_decoy_noise_scale = float(sys.argv[15])
local_decoy_max_total = int(sys.argv[16])
local_decoy_seed_offset = int(sys.argv[17])
distributed_stage_timeout = int(sys.argv[18])
extract_batch_size = int(sys.argv[19])
feature_extraction_lock_file = sys.argv[20]
dp_enabled = str(sys.argv[21]).lower() not in {'0', 'false', 'no'}

paths = [config_dir / 'server.yaml']
paths.extend(config_dir / f'client_{idx}.yaml' for idx in range(1, 61))
for path in paths:
    with path.open('r', encoding='utf-8') as stream:
        cfg = yaml.safe_load(stream)

    cfg['use_gpu'] = use_gpu
    cfg['device'] = device
    cfg.setdefault('distribute', {})['join_timeout_seconds'] = \
        join_timeout_seconds
    cfg['attack'] = {
        'attack_method': '',
        'distributed_fedmia': False,
    }
    cfg['dp'] = {
        'enabled': dp_enabled,
        'level': 'client_update',
        'mechanism': 'gaussian',
        'baseline': 'adaptive',
        'accountant': 'empirical',
        'epsilon': epsilon,
        'delta': delta,
        'noise_multiplier': noise_multiplier,
        'max_grad_norm': initial_clip,
        'clip_percentile': 75.0,
        'param_level': True,
        'protect_ggeur_update': dp_enabled,
        'private_clip_update': True,
        'upload_private_stats': False,
        'log_private_stats': True,
        'eps': 1e-12,
        'seed': 12345,
        'clipping': {
            'type': 'adaptive',
            'initial_clip': initial_clip,
            'target_quantile': target_quantile,
            'ema': ema,
            'min_clip': min_clip,
            'max_clip': max_clip,
        },
    }

    ggeur = cfg.setdefault('ggeur', {})
    ggeur['headonly_cache_version'] = (
        'officehome_60c_adaptive_defense' if dp_enabled
        else 'officehome_60c_no_defense')
    ggeur['distributed_stage_timeout'] = distributed_stage_timeout
    ggeur['extract_batch_size'] = extract_batch_size
    ggeur['feature_extraction_lock_file'] = \
        feature_extraction_lock_file
    ggeur['local_decoy'] = {
        'use': local_decoy_use and dp_enabled,
        'train_with_decoy': True,
        'num_per_class': local_decoy_num_per_class,
        'noise_scale': local_decoy_noise_scale,
        'min_std': 0.0001,
        'max_total': local_decoy_max_total,
        'seed_offset': local_decoy_seed_offset,
    }

    role = cfg.get('distribute', {}).get('role', 'unknown')
    variant = 'adaptive_defense' if dp_enabled else 'no_defense'
    cfg['expname'] = f'officehome_60c_{variant}_{role}'
    with path.open('w', encoding='utf-8') as stream:
        yaml.safe_dump(cfg, stream, sort_keys=False)
PY

{
  echo "run_id=$RUN_ID"
  echo "run_dir=$RUN_DIR"
  echo "attack_enabled=0"
  echo "dp_enabled=$DP_ENABLED"
  echo "client_num=$CLIENT_NUM"
  echo "total_rounds=$TOTAL_ROUNDS"
  echo "server_host=$SERVER_HOST"
  echo "server_port=$SERVER_PORT"
  echo "join_timeout_seconds=$JOIN_TIMEOUT_SECONDS"
  echo "distributed_stage_timeout=$DISTRIBUTED_STAGE_TIMEOUT"
  echo "use_gpu=$USE_GPU"
  echo "device=$DEVICE"
  echo "extract_batch_size=$EXTRACT_BATCH_SIZE"
  echo "feature_extraction_lock_file=$FEATURE_EXTRACTION_LOCK_FILE"
  echo "dp_epsilon=$DP_EPSILON"
  echo "dp_delta=$DP_DELTA"
  echo "dp_noise_multiplier=$DP_NOISE_MULTIPLIER"
  echo "dp_initial_clip=$DP_INITIAL_CLIP"
  echo "dp_target_quantile=$DP_TARGET_QUANTILE"
  echo "dp_ema=$DP_EMA"
  echo "local_decoy_use=$LOCAL_DECOY_USE"
  echo "local_decoy_num_per_class=$LOCAL_DECOY_NUM_PER_CLASS"
  echo "local_decoy_noise_scale=$LOCAL_DECOY_NOISE_SCALE"
  echo "local_decoy_max_total=$LOCAL_DECOY_MAX_TOTAL"
  echo "started_at=$(date '+%Y-%m-%d %H:%M:%S %z')"
} | tee "$SYSTEM_DIR/run_info.log"

if [ "${GENERATE_ONLY:-0}" = "1" ]; then
  echo "$RUN_DIR"
  exit 0
fi

stale_pids="$(pgrep -f "$ROOT_DIR/federatedscope/main.py.*exp/distributed/officehome_60c_" || true)"
if [ -n "$stale_pids" ]; then
  echo "[cleanup] stop stale Office-Home 60-client processes"
  for pid in $stale_pids; do
    kill -TERM "$pid" >/dev/null 2>&1 || true
  done
  sleep 3
  for pid in $stale_pids; do
    if kill -0 "$pid" >/dev/null 2>&1; then
      kill -KILL "$pid" >/dev/null 2>&1 || true
    fi
  done
fi

cleanup() {
  for pid_file in "$PID_DIR"/*.pid; do
    [ -f "$pid_file" ] || continue
    pid="$(cat "$pid_file")"
    kill -TERM "$pid" >/dev/null 2>&1 || true
  done
  sleep 2
  for pid_file in "$PID_DIR"/*.pid; do
    [ -f "$pid_file" ] || continue
    pid="$(cat "$pid_file")"
    if kill -0 "$pid" >/dev/null 2>&1; then
      kill -KILL "$pid" >/dev/null 2>&1 || true
    fi
  done
}
trap cleanup INT TERM

echo "[launch] server"
nohup "$PYTHON_BIN" "$ROOT_DIR/federatedscope/main.py" \
  --cfg "$CONFIG_DIR/server.yaml" >"$LOG_DIR/server.log" 2>&1 &
echo "$!" >"$PID_DIR/server.pid"
sleep "$SERVER_START_WAIT"

echo "[launch] 60 clients"
server_pid="$(cat "$PID_DIR/server.pid")"
for client_id in $(seq 1 "$CLIENT_NUM"); do
  if ! ps -p "$server_pid" >/dev/null 2>&1; then
    echo "[error] server exited while clients were still starting" | \
      tee -a "$SYSTEM_DIR/run_info.log"
    cleanup
    exit 1
  fi
  nohup "$PYTHON_BIN" "$ROOT_DIR/federatedscope/main.py" \
    --cfg "$CONFIG_DIR/client_${client_id}.yaml" \
    >"$LOG_DIR/client_${client_id}.log" 2>&1 &
  echo "$!" >"$PID_DIR/client_${client_id}.pid"
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

  server_pid="$(cat "$PID_DIR/server.pid")"
  if ! ps -p "$server_pid" >/dev/null 2>&1 && \
     ! grep -q "Training finished" "$LOG_DIR/server.log"; then
    echo "[error] server exited before training finished" | \
      tee -a "$SYSTEM_DIR/run_info.log"
    cleanup
    exit 1
  fi

  if [ "$(date +%s)" -gt "$deadline" ]; then
    echo "[error] timeout" | tee -a "$SYSTEM_DIR/run_info.log"
    cleanup
    exit 124
  fi
  sleep 5
done

{
  echo "finished_at=$(date '+%Y-%m-%d %H:%M:%S %z')"
} | tee -a "$SYSTEM_DIR/run_info.log"

grep -q "Training finished" "$LOG_DIR/server.log"
echo "$RUN_DIR"

