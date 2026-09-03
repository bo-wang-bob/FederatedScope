#!/usr/bin/env bash
set -euo pipefail

CASE_DIR="${1:?usage: launch_root.sh CASE_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
CASE_DIR="$(cd "${REPO_DIR}" && realpath "${CASE_DIR}")"
CONFIG="${CASE_DIR}/configs/root_server.yaml"
LOG_DIR="${CASE_DIR}/logs"
PID_DIR="${CASE_DIR}/pids"

mkdir -p "${LOG_DIR}" "${PID_DIR}"
test -f "${CONFIG}"
PORT="$(awk '/^[[:space:]]*server_port:/{print $2; exit}' "${CONFIG}" | tr -d '\r')"
test -n "${PORT}"
if [[ -f "${PID_DIR}/root.pid" ]] && kill -0 "$(cat "${PID_DIR}/root.pid")" 2>/dev/null; then
  echo "root already running pid=$(cat "${PID_DIR}/root.pid")"
  exit 1
fi
if timeout 1 bash -c ">/dev/tcp/127.0.0.1/${PORT}" 2>/dev/null; then
  echo "root port ${PORT} is already in use" >&2
  exit 1
fi

cd "${REPO_DIR}"
export FEDERATEDSCOPE_GGEUR_LIGHTWEIGHT=1
nohup "${PYTHON_BIN}" -m federatedscope.main --cfg "${CONFIG}" \
  >"${LOG_DIR}/root.stdout.log" 2>&1 &
echo "$!" >"${PID_DIR}/root.pid"
pid="$!"
ready=0
for _ in $(seq 1 60); do
  if ! kill -0 "${pid}" 2>/dev/null; then
    break
  fi
  if timeout 1 bash -c ">/dev/tcp/127.0.0.1/${PORT}" 2>/dev/null; then
    ready=1
    break
  fi
  sleep 0.5
done
if [[ "${ready}" -ne 1 ]]; then
  echo "root failed to listen on port ${PORT}; see ${LOG_DIR}/root.stdout.log" >&2
  kill "${pid}" 2>/dev/null || true
  wait "${pid}" 2>/dev/null || true
  exit 1
fi
echo "root_pid=$!"
echo "root_port=${PORT} ready=true"
echo "root_log=${LOG_DIR}/root.stdout.log"
