#!/usr/bin/env bash
set -euo pipefail

CASE_DIR="${1:?usage: status_role.sh CASE_DIR}"
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
CASE_DIR="$(cd "${REPO_DIR}" && realpath "${CASE_DIR}")"
PID_DIR="${CASE_DIR}/pids"
LOG_DIR="${CASE_DIR}/logs"

shopt -s nullglob
for pid_file in "${PID_DIR}"/*.pid; do
  name="$(basename "${pid_file}" .pid)"
  pid="$(cat "${pid_file}")"
  command_line="$(ps -p "${pid}" -o args= 2>/dev/null || true)"
  if kill -0 "${pid}" 2>/dev/null && [[ "${command_line}" == *"${CASE_DIR}"* ]]; then
    state=running
  elif kill -0 "${pid}" 2>/dev/null; then
    state=pid_reused
  else
    state=stopped
  fi
  echo "${name} pid=${pid} state=${state}"
  log="${LOG_DIR}/${name}.stdout.log"
  [[ -f "${log}" ]] && tail -n 5 "${log}"
done
