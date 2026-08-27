#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/autodl-tmp/FederatedScope}"
RUN_ID="${RUN_ID:-final_remaining_20260723_v1}"
PYTHON_BIN="${PYTHON_BIN:-/root/.local/share/mamba/envs/GGEUR/bin/python}"
SCRIPT_DIR="$REPO_DIR/scripts/distributed_scripts/ggeur_hierarchical_3machine"
RUN_ROOT="$SCRIPT_DIR/runs/$RUN_ID"
STATE_DIR="$RUN_ROOT/queue_state"
STATE_LOG="$STATE_DIR/root_queue.tsv"
CONTROL_STATE="$STATE_DIR/control.json"
CONTROL_PID_FILE="$STATE_DIR/control_server.pid"
CONTROL_LOG="$STATE_DIR/control_server.log"
CONTROL_PORT="${QUEUE_CONTROL_PORT:-60049}"
mkdir -p "$STATE_DIR"

manifest="$RUN_ROOT/matrix_manifest.json"
if [[ -f "$manifest" ]]; then
  mapfile -t cases < <(
    "$PYTHON_BIN" -c \
      'import json,sys; print(*[case["case"] for case in json.load(open(sys.argv[1], encoding="utf-8"))["cases"]], sep="\n")' \
      "$manifest"
  )
else
  cases=(
    officehome_cnn_fedavg officehome_cnn_fedprox
    officehome_cnn_fedproto officehome_cnn_fedopt
    officehome_cnn_moon officehome_cnn_ggeur
    officehome_mixer_fedavg officehome_mixer_fedprox
    officehome_mixer_fedproto officehome_mixer_fedopt
    officehome_mixer_moon officehome_mixer_ggeur
    mdsent_rnn_fedavg mdsent_rnn_fedprox mdsent_rnn_fedproto
    mdsent_rnn_fedopt mdsent_rnn_ggeur
    mdsent_lstm_fedavg mdsent_lstm_fedprox mdsent_lstm_fedproto
    mdsent_lstm_fedopt mdsent_lstm_ggeur
    domainnet_vit_fedavg domainnet_vit_fedprox domainnet_vit_fedproto
    domainnet_vit_fedopt domainnet_vit_moon domainnet_vit_ggeur
    domainnet_cnn_fedavg domainnet_cnn_fedprox domainnet_cnn_fedproto
    domainnet_cnn_fedopt domainnet_cnn_moon domainnet_cnn_ggeur
    domainnet_mixer_fedavg domainnet_mixer_fedprox
    domainnet_mixer_fedproto domainnet_mixer_fedopt
    domainnet_mixer_moon domainnet_mixer_ggeur
  )
fi
if [[ "${#cases[@]}" -eq 0 ]]; then
  printf 'No cases found for run %s\n' "$RUN_ID" >&2
  exit 2
fi
total_cases="${#cases[@]}"
completed_count="$(find "$RUN_ROOT" -mindepth 2 -maxdepth 2 \
  -name .formal_complete 2>/dev/null | wc -l)"
current_case=""
current_index=-1

record() {
  printf '%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "$1" "$2" >>"$STATE_LOG"
}

publish_state() {
  local phase="$1" message="${2:-}" accuracy_rounds="${3:-0}"
  "$PYTHON_BIN" "$SCRIPT_DIR/queue_control.py" write \
    --state-file "$CONTROL_STATE" \
    --run-id "$RUN_ID" \
    --case "${current_case:-none}" \
    --case-index "$current_index" \
    --total-cases "$total_cases" \
    --completed-count "$completed_count" \
    --phase "$phase" \
    --accuracy-rounds "$accuracy_rounds" \
    --root-port 60050 \
    --message "$message" >>"$CONTROL_LOG" 2>&1
}

start_control_server() {
  local pid="" ready=0
  if [[ -f "$CONTROL_PID_FILE" ]]; then
    pid="$(cat "$CONTROL_PID_FILE" 2>/dev/null || true)"
  fi
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    return 0
  fi
  if timeout 1 bash -c ">/dev/tcp/127.0.0.1/$CONTROL_PORT" 2>/dev/null; then
    record CONTROL "FAILED_PORT_IN_USE:$CONTROL_PORT"
    return 1
  fi
  nohup "$PYTHON_BIN" "$SCRIPT_DIR/queue_control.py" serve \
    --state-file "$CONTROL_STATE" --host 0.0.0.0 --port "$CONTROL_PORT" \
    >>"$CONTROL_LOG" 2>&1 &
  pid="$!"
  printf '%s\n' "$pid" >"$CONTROL_PID_FILE"
  for _ in $(seq 1 40); do
    if ! kill -0 "$pid" 2>/dev/null; then
      break
    fi
    if timeout 1 bash -c ">/dev/tcp/127.0.0.1/$CONTROL_PORT" 2>/dev/null; then
      ready=1
      break
    fi
    sleep 0.25
  done
  [[ "$ready" -eq 1 ]]
}

archive_incomplete_attempt() {
  local case_dir="$1" log_file="$2" hash="" previous="" stamp="" attempt_dir=""
  [[ -s "$log_file" ]] || return 0
  [[ ! -f "$case_dir/.formal_complete" ]] || return 0
  hash="$(sha256sum "$log_file" | awk '{print $1}')"
  previous="$(cat "$case_dir/attempts/.last_archived_sha256" 2>/dev/null || true)"
  [[ "$hash" != "$previous" ]] || return 0
  stamp="$(date +%Y%m%d_%H%M%S)"
  attempt_dir="$case_dir/attempts/attempt_$stamp"
  mkdir -p "$attempt_dir"
  cp "$log_file" "$attempt_dir/root.stdout.log"
  cp "$case_dir/configs/root_server.yaml" "$attempt_dir/root_server.yaml"
  printf '%s\n' \
    "status=INCOMPLETE" \
    "archived_at=$(date --iso-8601=seconds)" \
    "root_log_sha256=$hash" >"$attempt_dir/attempt_status.txt"
  printf '%s\n' "$hash" >"$case_dir/attempts/.last_archived_sha256"
  record "$current_case" "ARCHIVE_INCOMPLETE:$attempt_dir:$hash"
}

on_exit() {
  local status="$?"
  if [[ "$status" -ne 0 && -n "$current_case" ]]; then
    set +e
    record "$current_case" "QUEUE_EXIT_FAILED:status=$status"
    publish_state failed "root queue exited with status $status"
  fi
  # The control server is scoped to this run's state file.  Leaving it alive
  # after the queue exits keeps port 60049 occupied and prevents the next
  # accuracy queue from starting.
  if [[ -f "$CONTROL_PID_FILE" ]]; then
    control_pid="$(cat "$CONTROL_PID_FILE" 2>/dev/null || true)"
    if [[ -n "$control_pid" ]]; then
      kill "$control_pid" 2>/dev/null || true
    fi
    rm -f "$CONTROL_PID_FILE"
  fi
}
trap on_exit EXIT

wait_tcp() {
  local host="$1" port="$2" expected="$3"
  while true; do
    if timeout 2 bash -c ">/dev/tcp/$host/$port" 2>/dev/null; then
      state=up
    else
      state=down
    fi
    [[ "$state" == "$expected" ]] && return 0
    sleep 5
  done
}

wait_resources() {
  local case_name="$1" group
  group="$($PYTHON_BIN -c \
    'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8-sig"))["group"])' \
    "$RUN_ROOT/$case_name/manifest.json")"
  if [[ "$group" == domainnet_* ]]; then
    while [[ ! -f "$REPO_DIR/exp/distributed_feature_cache/$group/.ggeur_feature_cache_ready.json" ]]; do
      record "$case_name" "WAIT_DOMAINNET_CACHE:$group"
      sleep 60
    done
    manifest="$REPO_DIR/exp/distributed_manifests/domainnet_4domains/domainnet_manifest.json"
    while [[ ! -f "$manifest" ]]; do sleep 30; done
  fi
}

start_control_server

for index in "${!cases[@]}"; do
  case_name="${cases[$index]}"
  current_case="$case_name"
  current_index="$index"
  case_dir="$RUN_ROOT/$case_name"
  pid_file="$case_dir/pids/root.pid"
  log_file="$case_dir/logs/root.stdout.log"
  expected_rounds="$($PYTHON_BIN -c \
    'import sys,yaml; print(int(yaml.safe_load(open(sys.argv[1], encoding="utf-8-sig"))["federate"]["total_round_num"]))' \
    "$case_dir/configs/root_server.yaml")"
  expected_accuracy_round=$((expected_rounds - 1))
  eval_frequency="$($PYTHON_BIN -c \
    'import sys,yaml; print(int(yaml.safe_load(open(sys.argv[1], encoding="utf-8-sig")).get("eval", {}).get("freq", 1)))' \
    "$case_dir/configs/root_server.yaml")"
  terminal_client_eval_only="$($PYTHON_BIN -c \
    'import sys,yaml; g=yaml.safe_load(open(sys.argv[1], encoding="utf-8-sig")).get("ggeur", {}); print("1" if str(g.get("headonly_eval_mode", "server")).lower()=="both" and bool(g.get("terminal_client_eval_only", False)) else "0")' \
    "$case_dir/configs/root_server.yaml")"
  expected_updates="$($PYTHON_BIN -c \
    'import sys,yaml; print(int(yaml.safe_load(open(sys.argv[1], encoding="utf-8-sig"))["ggeur"].get("hierarchical_subserver_num", 1)))' \
    "$case_dir/configs/root_server.yaml")"
  test -d "$case_dir"
  if [[ -f "$case_dir/.formal_complete" ]]; then
    record "$case_name" SKIP_COMPLETE
    continue
  fi
  wait_resources "$case_name"
  publish_state starting "preparing root and waiting for matching roles"

  running=0
  if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    running=1
    record "$case_name" "ADOPT_ROOT:pid=$(cat "$pid_file")"
  fi
  if [[ "$running" -eq 0 ]]; then
    archive_incomplete_attempt "$case_dir" "$log_file"
    record "$case_name" START_ROOT
    PYTHON_BIN="$PYTHON_BIN" bash "$SCRIPT_DIR/launch_root.sh" "$case_dir" \
      >>"$STATE_DIR/root_launch.log" 2>&1
  fi
  publish_state running "root is listening"

  # Text cases own client IDs 61-120 on this host.  Wait until all four
  # subservers on the third machine are listening before launching them.
  if [[ "$case_name" == mdsent_* ]] && \
      [[ -d "$case_dir/configs/clients_4090" ]]; then
    root_client_pid="$case_dir/pids/client_000061.pid"
    if [[ ! -f "$root_client_pid" ]] || \
        ! kill -0 "$(cat "$root_client_pid")" 2>/dev/null; then
      for port in 61000 61001 61002 61003; do
        wait_tcp 10.129.248.111 "$port" up
      done
      record "$case_name" START_ROOT_CLIENTS
      # The original 600 MiB/process estimate is intentionally conservative.
      # Measurements from the same cached MDSent clients were <=170 MiB RSS;
      # retain a >2x margin while fitting the host's RAM+existing swap.
      MEMORY_PER_CLIENT_KB="${ROOT_CLIENT_MEMORY_PER_CLIENT_KB:-409600}" \
        START_GAP_SECONDS=0.1 LAUNCH_BATCH_SIZE=12 PYTHON_BIN="$PYTHON_BIN" \
        bash "$SCRIPT_DIR/launch_root_clients.sh" "$case_dir" \
        >>"$STATE_DIR/root_clients_launch.log" 2>&1
    fi
  fi

  pid="$(cat "$pid_file")"
  last_report=0
  while kill -0 "$pid" 2>/dev/null; do
    now="$(date +%s)"
    if (( now - last_report >= 300 )); then
      rounds="$(grep -c 'Round [0-9][0-9]* MLP Test Accuracy' "$log_file" 2>/dev/null || true)"
      record "$case_name" "RUNNING:pid=$pid:accuracy_rounds=$rounds"
      publish_state running "training" "$rounds"
      last_report="$now"
    fi
    sleep 20
  done

  if ! grep -q "Training finished after $expected_rounds rounds" "$log_file"; then
    record "$case_name" FAILED_ROOT
    exit 20
  fi
  validation_args=(
    "$log_file"
    --output "$case_dir/completion_validation.json"
    --expected-rounds "$expected_rounds"
    --expected-accuracy-rounds "$expected_accuracy_round"
    --eval-frequency "$eval_frequency"
    --expected-updates "$expected_updates"
  )
  if [[ "$terminal_client_eval_only" == 1 ]]; then
    validation_args+=(--terminal-client-eval-only)
  fi
  if ! "$PYTHON_BIN" "$SCRIPT_DIR/validate_case_completion.py" \
      "${validation_args[@]}" \
      >>"$STATE_DIR/validation.log" 2>&1; then
    record "$case_name" FAILED_VALIDATION
    exit 21
  fi
  "$PYTHON_BIN" "$SCRIPT_DIR/summarize_accuracy.py" "$log_file" \
    --output "$case_dir/accuracy_summary.json" \
    >>"$STATE_DIR/summarize.log" 2>&1
  record "$case_name" COMPLETE
  marker_tmp="$case_dir/.formal_complete.tmp.$$"
  printf '{"run_id":"%s","case":"%s","completed_at":"%s","validation":"completion_validation.json","summary":"accuracy_summary.json"}\n' \
    "$RUN_ID" "$case_name" "$(date --iso-8601=seconds)" >"$marker_tmp"
  mv "$marker_tmp" "$case_dir/.formal_complete"
  completed_count=$((index + 1))
  publish_state complete "completion contract passed" "$expected_accuracy_round"

  # Give both Windows queues time to observe the old port going down before
  # the next root reuses port 60050.
  wait_tcp 127.0.0.1 60050 down
  sleep 45
done

record ALL COMPLETE
touch "$STATE_DIR/root_queue.complete"
current_case=ALL
current_index="$total_cases"
completed_count="$total_cases"
publish_state all_complete "all formal cases completed" 0
# Successful queues bypass the EXIT trap below, so release the run-scoped
# control service explicitly.  Otherwise port 60049 remains occupied and the
# next queued test cannot start.
if [[ -f "$CONTROL_PID_FILE" ]]; then
  control_pid="$(cat "$CONTROL_PID_FILE" 2>/dev/null || true)"
  if [[ -n "$control_pid" ]]; then
    kill "$control_pid" 2>/dev/null || true
  fi
  rm -f "$CONTROL_PID_FILE"
fi
trap - EXIT
