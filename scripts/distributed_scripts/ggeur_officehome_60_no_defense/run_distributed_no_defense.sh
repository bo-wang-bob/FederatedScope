#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ONCE_SCRIPT="$ROOT_DIR/scripts/distributed_scripts/ggeur_officehome_60_no_defense/run_distributed_no_defense_once.sh"
FEATURE_CACHE_DIR="${FEATURE_CACHE_DIR:-exp/headonly_system/cache/officehome_60c_no_defense}"
CACHE_WARMUP_TIMEOUT="${CACHE_WARMUP_TIMEOUT:-7200}"
CLIENT_NUM=60

if [[ "$FEATURE_CACHE_DIR" = /* ]]; then
  CACHE_ROOT="$FEATURE_CACHE_DIR"
else
  CACHE_ROOT="$ROOT_DIR/$FEATURE_CACHE_DIR"
fi

cache_count() {
  find "$CACHE_ROOT/headonly_augmented" -type f \
    -name 'office-home_client_*.pt' 2>/dev/null | wc -l
}

warmup_pid=""
warmup_id=""
cleanup_warmup() {
  if [ -n "$warmup_pid" ] && kill -0 "$warmup_pid" 2>/dev/null; then
    kill -TERM "$warmup_pid" 2>/dev/null || true
    wait "$warmup_pid" 2>/dev/null || true
  fi
  if [ -n "$warmup_id" ]; then
    for pid in $(pgrep -f "$warmup_id" || true); do
      [ "$pid" = "$$" ] || kill -TERM "$pid" 2>/dev/null || true
    done
    sleep 2
    for pid in $(pgrep -f "$warmup_id" || true); do
      [ "$pid" = "$$" ] || kill -KILL "$pid" 2>/dev/null || true
    done
  fi
}
trap cleanup_warmup INT TERM

if [ "${GENERATE_ONLY:-0}" = "1" ]; then
  export FEATURE_CACHE_DIR
  exec bash "$ONCE_SCRIPT"
fi

current_cache_count="$(cache_count)"
if [ "$current_cache_count" -lt "$CLIENT_NUM" ]; then
  warmup_id="officehome_60c_no_defense_cache_warmup_$(date +%Y%m%d_%H%M%S)"
  warmup_dir="$ROOT_DIR/exp/distributed/$warmup_id"
  mkdir -p "$warmup_dir"
  echo "[warmup] no-defense cache incomplete: $current_cache_count/$CLIENT_NUM"
  echo "[warmup] generate client feature caches, then restart clean processes"

  RUN_ID="$warmup_id" \
  FEATURE_CACHE_DIR="$FEATURE_CACHE_DIR" \
  TOTAL_ROUNDS=1 \
  RUN_TIMEOUT="$CACHE_WARMUP_TIMEOUT" \
  bash "$ONCE_SCRIPT" >"$warmup_dir/launcher.log" 2>&1 &
  warmup_pid=$!

  deadline=$(( $(date +%s) + CACHE_WARMUP_TIMEOUT ))
  last_report=0
  while true; do
    current_cache_count="$(cache_count)"
    if [ "$current_cache_count" -ge "$CLIENT_NUM" ]; then
      break
    fi
    if ! kill -0 "$warmup_pid" 2>/dev/null; then
      echo "[error] cache warmup exited at $current_cache_count/$CLIENT_NUM"
      tail -100 "$warmup_dir/launcher.log" || true
      cleanup_warmup
      exit 1
    fi
    now="$(date +%s)"
    if [ "$now" -gt "$deadline" ]; then
      echo "[error] cache warmup timeout at $current_cache_count/$CLIENT_NUM"
      cleanup_warmup
      exit 124
    fi
    if [ $((now - last_report)) -ge 30 ]; then
      echo "[warmup] cached clients: $current_cache_count/$CLIENT_NUM"
      last_report="$now"
    fi
    sleep 5
  done

  echo "[warmup] all $CLIENT_NUM client caches are ready"
  cleanup_warmup
  warmup_pid=""
  warmup_id=""
fi

trap - INT TERM
export FEATURE_CACHE_DIR
export RUN_ID="${RUN_ID:-officehome_60c_no_defense}"
echo "[train] start clean no-defense 60-client training: $RUN_ID"
exec bash "$ONCE_SCRIPT"
