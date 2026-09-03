#!/usr/bin/env bash
set -euo pipefail

CASE_DIR="${1:?usage: launch_root_clients.sh CASE_DIR [CONFIG_SET]}"
CONFIG_SET="${2:-clients_4090}"
PYTHON_BIN="${PYTHON_BIN:-/root/.local/share/mamba/envs/GGEUR/bin/python}"
START_GAP_SECONDS="${START_GAP_SECONDS:-1}"
LAUNCH_BATCH_SIZE="${LAUNCH_BATCH_SIZE:-12}"
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
CASE_DIR="$(cd "${REPO_DIR}" && realpath "${CASE_DIR}")"
CONFIG_DIR="${CASE_DIR}/configs/${CONFIG_SET}"
LOG_DIR="${CASE_DIR}/logs"
PID_DIR="${CASE_DIR}/pids"

mkdir -p "${LOG_DIR}" "${PID_DIR}"
test -x "${PYTHON_BIN}"
test -d "${CONFIG_DIR}"
if ! [[ "${LAUNCH_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] ||
   (( LAUNCH_BATCH_SIZE > 32 )); then
  echo "invalid LAUNCH_BATCH_SIZE=${LAUNCH_BATCH_SIZE}" >&2
  exit 1
fi
shopt -s nullglob
configs=("${CONFIG_DIR}"/client_*.yaml)
if [[ "${#configs[@]}" -eq 0 ]]; then
  echo "no client configs under ${CONFIG_DIR}" >&2
  exit 1
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export FEDERATEDSCOPE_GGEUR_LIGHTWEIGHT=1
export CUDA_VISIBLE_DEVICES=""

runtime="$(${PYTHON_BIN} -c 'import torch; print(str(torch.__version__) + "|" + str(torch.cuda.is_available()))')"
echo "client_runtime=${runtime} cuda_visible_devices=empty"

cache_dir="$(awk -F': ' '/^[[:space:]]*feature_cache_dir:/{print $2; exit}' "${configs[0]}" | tr -d "'\"\r")"
if [[ -z "${cache_dir}" || ! -f "${cache_dir}/.ggeur_feature_cache_ready.json" ]]; then
  echo "feature cache is not validated: ${cache_dir}/.ggeur_feature_cache_ready.json" >&2
  exit 1
fi

available_kb="$(awk '/MemAvailable:/{m=$2}/SwapFree:/{s=$2}END{print m+s}' /proc/meminfo)"
memory_per_client_kb="${MEMORY_PER_CLIENT_KB:-614400}"
if ! [[ "${memory_per_client_kb}" =~ ^[1-9][0-9]*$ ]]; then
  echo "invalid MEMORY_PER_CLIENT_KB=${memory_per_client_kb}" >&2
  exit 1
fi
required_kb="$(( ${#configs[@]} * memory_per_client_kb + 2 * 1024 * 1024 ))"
echo "memory_gate available_kb=${available_kb} required_kb=${required_kb} per_client_kb=${memory_per_client_kb}"
if (( available_kb < required_kb )); then
  echo "insufficient memory+swap: available_kb=${available_kb} required_kb=${required_kb}" >&2
  exit 1
fi

for config in "${configs[@]}"; do
  name="$(basename "${config}" .yaml)"
  pid_file="${PID_DIR}/${name}.pid"
  if [[ -f "${pid_file}" ]] && kill -0 "$(cat "${pid_file}")" 2>/dev/null; then
    echo "${name} is already running pid=$(cat "${pid_file}")" >&2
    exit 1
  fi
  port="$(awk '/^[[:space:]]*client_port:/{print $2; exit}' "${config}" | tr -d '\r')"
  test -n "${port}"
  if timeout 1 bash -c ">/dev/tcp/127.0.0.1/${port}" 2>/dev/null; then
    echo "client port ${port} is already in use" >&2
    exit 1
  fi
done

cd "${REPO_DIR}"
started=0
pending_pids=()
pending_names=()
pending_ports=()
pending_logs=()

confirm_pending() {
  local i pid name port log ready
  for i in "${!pending_pids[@]}"; do
    pid="${pending_pids[$i]}"
    name="${pending_names[$i]}"
    port="${pending_ports[$i]}"
    log="${pending_logs[$i]}"
    ready=0
    for _ in $(seq 1 120); do
      if ! kill -0 "${pid}" 2>/dev/null; then
        break
      fi
      if timeout 1 bash -c ">/dev/tcp/127.0.0.1/${port}" 2>/dev/null; then
        ready=1
        break
      fi
      sleep 0.25
    done
    if [[ "${ready}" -ne 1 ]]; then
      for pid in "${pending_pids[@]}"; do
        kill "${pid}" 2>/dev/null || true
      done
      echo "${name} failed to listen on ${port}; see ${log}" >&2
      return 1
    fi
    echo "${pid}" >"${PID_DIR}/${name}.pid"
    echo "${name} pid=${pid} port=${port} ready=true"
    started=$((started + 1))
  done
  pending_pids=()
  pending_names=()
  pending_ports=()
  pending_logs=()
}

for config in "${configs[@]}"; do
  name="$(basename "${config}" .yaml)"
  port="$(awk '/^[[:space:]]*client_port:/{print $2; exit}' "${config}" | tr -d '\r')"
  log="${LOG_DIR}/${name}.stdout.log"
  nohup "${PYTHON_BIN}" -m federatedscope.main --cfg "${config}" \
    >"${log}" 2>&1 &
  pid="$!"
  pending_pids+=("${pid}")
  pending_names+=("${name}")
  pending_ports+=("${port}")
  pending_logs+=("${log}")
  sleep "${START_GAP_SECONDS}"
  if (( ${#pending_pids[@]} >= LAUNCH_BATCH_SIZE )); then
    confirm_pending
  fi
done

if (( ${#pending_pids[@]} > 0 )); then
  confirm_pending
fi

echo "started_clients=${started}"
