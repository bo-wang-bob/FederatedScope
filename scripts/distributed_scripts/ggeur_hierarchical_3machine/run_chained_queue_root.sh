#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/autodl-tmp/FederatedScope}"
PYTHON_BIN="${PYTHON_BIN:-/root/.local/share/mamba/envs/GGEUR/bin/python}"
SCRIPT_DIR="$REPO_DIR/scripts/distributed_scripts/ggeur_hierarchical_3machine"
PREDECESSOR_RUN_ID="${PREDECESSOR_RUN_ID:-mdsent_reference_distributed_20260730_v1}"
CHAIN_RUN_IDS="${CHAIN_RUN_IDS:-mdsent_baselines_corrected_20260731_v1,domainnet_remaining_20260731_v1}"
HANDOFF_GRACE_SECONDS="${HANDOFF_GRACE_SECONDS:-120}"
CHAIN_ID="${CHAIN_ID:-remaining_20260731_v1}"
CHAIN_STATE_DIR="$REPO_DIR/exp/hierarchical/$CHAIN_ID"
CHAIN_LOG="$CHAIN_STATE_DIR/root_chain.tsv"
CHAIN_COMPLETE="$CHAIN_STATE_DIR/root_chain.complete"
mkdir -p "$CHAIN_STATE_DIR"

IFS=',' read -r -a chain_runs <<<"$CHAIN_RUN_IDS"
if [[ "${#chain_runs[@]}" -eq 0 ]]; then
  printf 'No chained run IDs were supplied\n' >&2
  exit 2
fi

record() {
  printf '%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "$1" "$2" >>"$CHAIN_LOG"
}

wait_for_pid_exit() {
  local pid_file="$1" label="$2" pid=""
  pid="$(cat "$pid_file" 2>/dev/null || true)"
  if [[ -z "$pid" ]]; then
    return 0
  fi
  while kill -0 "$pid" 2>/dev/null; do
    record "$label" "WAIT_QUEUE_EXIT:pid=$pid"
    sleep 10
  done
}

ensure_control_server() {
  local run_id="$1" run_root state_file pid_file log_file pid=""
  run_root="$SCRIPT_DIR/runs/$run_id"
  state_file="$run_root/queue_state/control.json"
  pid_file="$run_root/queue_state/control_server.pid"
  log_file="$run_root/queue_state/control_server.log"
  if timeout 1 bash -c ">/dev/tcp/127.0.0.1/60049" 2>/dev/null; then
    return 0
  fi
  test -f "$state_file"
  pid="$(cat "$pid_file" 2>/dev/null || true)"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    record "$run_id" "CONTROL_PID_ALIVE_BUT_PORT_DOWN:pid=$pid"
    return 1
  fi
  record "$run_id" START_RECOVERY_CONTROL
  nohup "$PYTHON_BIN" "$SCRIPT_DIR/queue_control.py" serve \
    --state-file "$state_file" --host 0.0.0.0 --port 60049 \
    >>"$log_file" 2>&1 &
  pid="$!"
  printf '%s\n' "$pid" >"$pid_file"
  for _ in $(seq 1 40); do
    if timeout 1 bash -c ">/dev/tcp/127.0.0.1/60049" 2>/dev/null; then
      record "$run_id" "RECOVERY_CONTROL_READY:pid=$pid"
      return 0
    fi
    kill -0 "$pid" 2>/dev/null || break
    sleep 0.25
  done
  record "$run_id" RECOVERY_CONTROL_FAILED
  return 2
}

wait_for_run_complete() {
  local run_id="$1" run_root marker
  run_root="$SCRIPT_DIR/runs/$run_id"
  marker="$run_root/queue_state/root_queue.complete"
  while [[ ! -f "$marker" ]]; do
    ensure_control_server "$run_id"
    record "$run_id" "WAIT_FILE:$marker"
    sleep 60
  done
}

stop_exact_control_server() {
  local run_id="$1" run_root pid_file pid="" cmdline=""
  run_root="$SCRIPT_DIR/runs/$run_id"
  pid_file="$run_root/queue_state/control_server.pid"
  pid="$(cat "$pid_file" 2>/dev/null || true)"
  if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
    return 0
  fi
  cmdline="$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null || true)"
  if [[ "$cmdline" != *"queue_control.py serve"* ]] || \
      [[ "$cmdline" != *"$run_root/queue_state/control.json"* ]]; then
    record "$run_id" "REFUSE_UNVERIFIED_CONTROL_PID:pid=$pid:cmd=$cmdline"
    return 3
  fi
  record "$run_id" "STOP_OLD_CONTROL:pid=$pid"
  kill -TERM "$pid"
  for _ in $(seq 1 30); do
    kill -0 "$pid" 2>/dev/null || return 0
    sleep 1
  done
  record "$run_id" "CONTROL_DID_NOT_EXIT:pid=$pid"
  return 4
}

wait_for_control_port_down() {
  while timeout 1 bash -c ">/dev/tcp/127.0.0.1/60049" 2>/dev/null; do
    record CONTROL "WAIT_PORT_DOWN:60049"
    sleep 5
  done
}

finish_run_handoff() {
  local run_id="$1" run_root
  run_root="$SCRIPT_DIR/runs/$run_id"
  wait_for_run_complete "$run_id"
  wait_for_pid_exit "$run_root/queue_state/root_queue.driver.pid" "$run_id"
  # Windows role queues consume the all_complete state over HTTP.  Keep it
  # visible for a bounded grace period before replacing the control server.
  record "$run_id" "HANDOFF_GRACE:${HANDOFF_GRACE_SECONDS}s"
  sleep "$HANDOFF_GRACE_SECONDS"
  stop_exact_control_server "$run_id"
  wait_for_control_port_down
  record "$run_id" HANDOFF_READY
}

if [[ -f "$CHAIN_COMPLETE" ]]; then
  record ALL ALREADY_COMPLETE
  exit 0
fi

previous_run="$PREDECESSOR_RUN_ID"
for run_id in "${chain_runs[@]}"; do
  run_id="${run_id//[[:space:]]/}"
  [[ -n "$run_id" ]] || continue
  run_root="$SCRIPT_DIR/runs/$run_id"
  test -f "$run_root/matrix_manifest.json"

  if [[ -f "$run_root/queue_state/root_queue.complete" ]]; then
    record "$run_id" SKIP_COMPLETE
    previous_run="$run_id"
    continue
  fi

  finish_run_handoff "$previous_run"

  if pgrep -af '[f]ederatedscope.main|[r]un_remaining_queue_root.sh' \
      | grep -q .; then
    record "$run_id" REFUSE_ACTIVE_TRAINING_ROLE
    exit 5
  fi

  mkdir -p "$run_root/queue_state"
  record "$run_id" START_ROOT_QUEUE
  env RUN_ID="$run_id" \
    PYTHON_BIN="$PYTHON_BIN" \
    REPO_DIR="$REPO_DIR" \
    bash "$SCRIPT_DIR/run_remaining_queue_root.sh" \
    >>"$run_root/queue_state/root_queue.driver.log" 2>&1 &
  queue_pid="$!"
  printf '%s\n' "$queue_pid" >"$run_root/queue_state/root_queue.driver.pid"
  wait "$queue_pid"
  test -f "$run_root/queue_state/root_queue.complete"
  record "$run_id" ROOT_QUEUE_COMPLETE
  previous_run="$run_id"
done

finish_run_handoff "$previous_run"
touch "$CHAIN_COMPLETE"
record ALL COMPLETE
