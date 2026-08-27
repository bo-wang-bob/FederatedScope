#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/autodl-tmp/FederatedScope}"
PYTHON_BIN="${PYTHON_BIN:-/root/.local/share/mamba/envs/GGEUR/bin/python}"
RUN_ID="${RUN_ID:?Set RUN_ID}"
PAYLOAD_FILE="${PAYLOAD_FILE:?Set PAYLOAD_FILE}"
BASE_PORT="${BASE_PORT:-62010}"
SUBSERVERS="${SUBSERVERS:-5}"
CLIENTS_PER_SUBSERVER="${CLIENTS_PER_SUBSERVER:-1000}"
EXPECTED_PROBES="${EXPECTED_PROBES:-1}"
EXPECTED_TRAINING_CLIENTS="${EXPECTED_TRAINING_CLIENTS:-0}"
REQUEST_DURATION_SEC="${REQUEST_DURATION_SEC:-0}"
REQUESTS_PER_CLIENT="${REQUESTS_PER_CLIENT:-0}"
REQUEST_START_DELAY_SEC="${REQUEST_START_DELAY_SEC:-2}"
START_AT_UNIX="${START_AT_UNIX:?Set START_AT_UNIX}"
OBSERVE_AT_UNIX="${OBSERVE_AT_UNIX:?Set OBSERVE_AT_UNIX}"
ACTIVE_UNTIL_UNIX="${ACTIVE_UNTIL_UNIX:?Set ACTIVE_UNTIL_UNIX}"
OUT_ROOT="${OUT_ROOT:-exp/concurrent_availability}"
NOFILE_LIMIT="${NOFILE_LIMIT:-1048576}"

cd "$REPO_DIR"
RUN_DIR="$OUT_ROOT/$RUN_ID/root4090"
mkdir -p "$RUN_DIR"
ulimit -n "$NOFILE_LIMIT" 2>/dev/null || true

if ss -ltn 2>/dev/null | grep -Eq ':(60050|61000|61001|61002|61003)[[:space:]]'; then
  echo "Refusing to overlap the formal accuracy queue." >&2
  exit 20
fi

{
  echo "role=availability_server"
  echo "host_role=root4090"
  echo "run_id=$RUN_ID"
  echo "business_endpoint=10.112.81.135:$BASE_PORT"
  echo "subservers=$SUBSERVERS"
  echo "clients_per_subserver=$CLIENTS_PER_SUBSERVER"
  echo "expected_probes=$EXPECTED_PROBES"
  echo "expected_training_clients=$EXPECTED_TRAINING_CLIENTS"
  echo "request_duration_sec=$REQUEST_DURATION_SEC"
  echo "requests_per_client=$REQUESTS_PER_CLIENT"
  echo "start_at_unix=$START_AT_UNIX"
  echo "observe_at_unix=$OBSERVE_AT_UNIX"
  echo "active_until_unix=$ACTIVE_UNTIL_UNIX"
  echo "payload_file=$PAYLOAD_FILE"
  sha256sum "$PAYLOAD_FILE"
  echo "nofile_limit=$(ulimit -n)"
  echo "started_at=$(date --iso-8601=seconds)"
} >"$RUN_DIR/command_audit.log"

"$PYTHON_BIN" scripts/benchmark_headonly_concurrent_availability.py server \
  --payload-file "$PAYLOAD_FILE" \
  --listen-host 0.0.0.0 \
  --base-port "$BASE_PORT" \
  --subservers "$SUBSERVERS" \
  --clients-per-subserver "$CLIENTS_PER_SUBSERVER" \
  --expected-probes "$EXPECTED_PROBES" \
  --expected-training-clients "$EXPECTED_TRAINING_CLIENTS" \
  --request-duration-sec "$REQUEST_DURATION_SEC" \
  --requests-per-client "$REQUESTS_PER_CLIENT" \
  --request-start-delay-sec "$REQUEST_START_DELAY_SEC" \
  --start-at-unix "$START_AT_UNIX" \
  --observe-at-unix "$OBSERVE_AT_UNIX" \
  --active-until-unix "$ACTIVE_UNTIL_UNIX" \
  --ready-timeout 900 \
  --test-timeout 1800 \
  --io-timeout 900 \
  --output "$RUN_DIR/server_summary.json"

echo "finished_at=$(date --iso-8601=seconds)" \
  >>"$RUN_DIR/command_audit.log"
