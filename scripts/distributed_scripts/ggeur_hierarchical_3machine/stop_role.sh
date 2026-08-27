#!/usr/bin/env bash
set -euo pipefail

CASE_DIR="${1:?usage: stop_role.sh CASE_DIR root|subserver|client}"
ROLE="${2:?usage: stop_role.sh CASE_DIR root|subserver|client}"
FORCE="${3:-false}"
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
CASE_DIR="$(cd "${REPO_DIR}" && realpath "${CASE_DIR}")"
PID_DIR="${CASE_DIR}/pids"

case "${ROLE}" in
  root) pattern="root.pid" ;;
  subserver) pattern="subserver_*.pid" ;;
  client) pattern="client_*.pid" ;;
  *) echo "role must be root, subserver, or client" >&2; exit 2 ;;
esac

shopt -s nullglob
for pid_file in "${PID_DIR}"/${pattern}; do
  pid="$(cat "${pid_file}")"
  if kill -0 "${pid}" 2>/dev/null; then
    command_line="$(ps -p "${pid}" -o args= 2>/dev/null || true)"
    if [[ "${command_line}" != *"${CASE_DIR}"* ]]; then
      echo "refuse to stop reused/unrelated pid=${pid}: ${command_line}" >&2
      continue
    fi
    kill "${pid}"
    for _ in $(seq 1 60); do
      kill -0 "${pid}" 2>/dev/null || break
      sleep 0.5
    done
    if kill -0 "${pid}" 2>/dev/null; then
      if [[ "${FORCE}" == "true" ]]; then
        kill -KILL "${pid}"
      else
        echo "pid=${pid} did not stop; rerun with third argument true" >&2
        continue
      fi
    fi
    echo "stopped $(basename "${pid_file}" .pid) pid=${pid}"
  fi
  rm -f "${pid_file}"
done
