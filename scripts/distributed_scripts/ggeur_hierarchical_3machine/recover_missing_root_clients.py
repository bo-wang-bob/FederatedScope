#!/usr/bin/env python3
"""Relaunch explicitly missing 4090 clients without restarting a formal case."""

import argparse
import os
import re

import paramiko


ROOT_REPO = "/root/autodl-tmp/FederatedScope"
SCRIPT_REL = "scripts/distributed_scripts/ggeur_hierarchical_3machine"
NAME_PATTERN = re.compile(r"^[a-z0-9_]+$")


def shell_quote(value):
    return "'" + value.replace("'", "'\"'\"'") + "'"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--client-ids", type=int, nargs="+", required=True)
    parser.add_argument(
        "--key", default=os.path.expanduser(r"~\.ssh\id_ed25519")
    )
    args = parser.parse_args()

    if not NAME_PATTERN.fullmatch(args.run_id):
        parser.error("unsupported --run-id")
    if not NAME_PATTERN.fullmatch(args.case):
        parser.error("unsupported --case")
    client_ids = sorted(set(args.client_ids))
    if not client_ids or any(value < 61 or value > 120 for value in client_ids):
        parser.error("--client-ids must be within the 4090 range 61..120")

    client_list = " ".join(str(value) for value in client_ids)
    run_root = f"{ROOT_REPO}/{SCRIPT_REL}/runs/{args.run_id}"
    case_dir = f"{run_root}/{args.case}"
    python_bin = "/root/.local/share/mamba/envs/GGEUR/bin/python"
    command = f"""
set -euo pipefail
run_root={shell_quote(run_root)}
case_dir={shell_quote(case_dir)}
case_name={shell_quote(args.case)}
python_bin={shell_quote(python_bin)}
client_ids=({client_list})
control_case="$(curl -fsS http://127.0.0.1:60049/state |
  "$python_bin" -c 'import json,sys; print(json.load(sys.stdin)["case"])')"
[[ "$control_case" == "$case_name" ]]
root_pid="$(cat "$case_dir/pids/root.pid")"
kill -0 "$root_pid"
grep -Fq "$case_dir/configs/root_server.yaml" "/proc/$root_pid/cmdline"

for client_id in "${{client_ids[@]}}"; do
  name="$(printf 'client_%06d' "$client_id")"
  config="$case_dir/configs/clients_4090/$name.yaml"
  test -f "$config"
  if pgrep -f "$config" >/dev/null; then
    echo "$name is already running" >&2
    exit 20
  fi
  if grep -Fq "Client $client_id augmentation ready" \
      "$case_dir/logs/root.stdout.log"; then
    echo "$name already completed augmentation" >&2
    exit 21
  fi
done

stamp="$(date +%Y%m%d_%H%M%S)"
attempt="$case_dir/attempts/attempt_${{stamp}}_missing_root_clients"
mkdir -p "$attempt"
cp -a "$case_dir/logs/root.stdout.log" "$attempt/" 2>/dev/null || true
for client_id in "${{client_ids[@]}}"; do
  name="$(printf 'client_%06d' "$client_id")"
  cp -a "$case_dir/configs/clients_4090/$name.yaml" "$attempt/" \
    2>/dev/null || true
  cp -a "$case_dir/logs/$name.stdout.log" "$attempt/" \
    2>/dev/null || true
  cp -a "$case_dir/pids/$name.pid" "$attempt/" 2>/dev/null || true
done
printf '%s\n' \
  'status=INCOMPLETE_MISSING_ROOT_CLIENTS' \
  "archived_at=$(date --iso-8601=seconds)" \
  "case=$case_name" \
  "client_ids={','.join(str(value) for value in client_ids)}" \
  'accuracy_rounds=0' >"$attempt/attempt_status.txt"
printf '%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "$case_name" \
  "RECOVERY_MISSING_ROOT_CLIENTS:{','.join(str(value) for value in client_ids)}:$attempt" \
  >>"$run_root/queue_state/root_queue.tsv"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export FEDERATEDSCOPE_GGEUR_LIGHTWEIGHT=1
export CUDA_VISIBLE_DEVICES=""
cd {shell_quote(ROOT_REPO)}
for client_id in "${{client_ids[@]}}"; do
  name="$(printf 'client_%06d' "$client_id")"
  config="$case_dir/configs/clients_4090/$name.yaml"
  port="$(awk '/^[[:space:]]*client_port:/{{print $2; exit}}' "$config" |
    tr -d '\\r')"
  test -n "$port"
  if timeout 1 bash -c ">/dev/tcp/127.0.0.1/$port" 2>/dev/null; then
    echo "$name port $port is already in use" >&2
    exit 22
  fi
  nohup "$python_bin" -m federatedscope.main --cfg "$config" \
    >"$case_dir/logs/$name.stdout.log" 2>&1 &
  pid="$!"
  ready=0
  for _ in $(seq 1 120); do
    if ! kill -0 "$pid" 2>/dev/null; then
      break
    fi
    if timeout 1 bash -c ">/dev/tcp/127.0.0.1/$port" 2>/dev/null; then
      ready=1
      break
    fi
    sleep 0.25
  done
  if [[ "$ready" -ne 1 ]]; then
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    echo "$name failed to listen; see its stdout log" >&2
    exit 23
  fi
  printf '%s\n' "$pid" >"$case_dir/pids/$name.pid"
  printf 'RELAUNCHED=%s PID=%s PORT=%s\n' "$name" "$pid" "$port"
done
printf 'ATTEMPT_ARCHIVE=%s\n' "$attempt"
"""

    key = paramiko.Ed25519Key.from_private_key_file(args.key)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect("::1", port=41322, username="root", pkey=key, timeout=20)
    try:
        _, stdout, stderr = client.exec_command(command, timeout=120)
        output = stdout.read().decode("utf-8", "replace").strip()
        error = stderr.read().decode("utf-8", "replace").strip()
        status = stdout.channel.recv_exit_status()
    finally:
        client.close()
    if status:
        raise RuntimeError(
            f"remote recovery failed ({status})\nstdout:\n{output}\nstderr:\n{error}"
        )
    print(output)
    if error:
        print(error)


if __name__ == "__main__":
    main()
