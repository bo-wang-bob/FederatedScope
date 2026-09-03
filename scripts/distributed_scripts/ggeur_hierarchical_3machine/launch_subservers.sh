#!/usr/bin/env bash
set -euo pipefail

CASE_DIR="${1:?usage: launch_subservers.sh CASE_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
SCRIPT="${REPO_DIR}/scripts/distributed_scripts/ggeur_hierarchical_3machine/hierarchical_subserver.py"
CASE_DIR="$(cd "${REPO_DIR}" && realpath "${CASE_DIR}")"
LOG_DIR="${CASE_DIR}/logs"
PID_DIR="${CASE_DIR}/pids"

mkdir -p "${LOG_DIR}" "${PID_DIR}"
shopt -s nullglob
CONFIGS=("${CASE_DIR}"/configs/subserver_*.json)
if [[ ${#CONFIGS[@]} -eq 0 ]]; then
  echo "no subserver configs under ${CASE_DIR}/configs" >&2
  exit 1
fi

for config in "${CONFIGS[@]}"; do
  name="$(basename "${config}" .json)"
  pid_file="${PID_DIR}/${name}.pid"
  port="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["listen_port"])' "${config}")"
  if [[ -f "${pid_file}" ]] && kill -0 "$(cat "${pid_file}")" 2>/dev/null; then
    echo "${name} already running pid=$(cat "${pid_file}")" >&2
    exit 1
  fi
  if timeout 1 bash -c ">/dev/tcp/127.0.0.1/${port}" 2>/dev/null; then
    echo "${name} port ${port} is already in use" >&2
    exit 1
  fi
done

cd "${REPO_DIR}"
for config in "${CONFIGS[@]}"; do
  name="$(basename "${config}" .json)"
  log="${LOG_DIR}/${name}.stdout.log"
  nohup "${PYTHON_BIN}" "${SCRIPT}" --config "${config}" --log "${LOG_DIR}/${name}.log" \
    >"${log}" 2>&1 &
  pid="$!"
  echo "${pid}" >"${PID_DIR}/${name}.pid"
  port="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["listen_port"])' "${config}")"
  ready=0
  for _ in $(seq 1 60); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      break
    fi
    if timeout 1 bash -c ">/dev/tcp/127.0.0.1/${port}" 2>/dev/null; then
      ready=1
      break
    fi
    sleep 0.5
  done
  if [[ "${ready}" -ne 1 ]]; then
    echo "${name} failed to listen on port ${port}; see ${log}" >&2
    kill "${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
    exit 1
  fi
  echo "${name}_pid=${pid} port=${port} ready=true log=${log}"
done
