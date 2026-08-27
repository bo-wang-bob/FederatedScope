#!/usr/bin/env python3
"""Manage MilitaryAircraft3D experiments on the lab 4090."""

import argparse
import importlib.util
import json
import os
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
MANAGER_PATH = (
    REPO / "scripts" / "distributed_scripts" /
    "ggeur_concurrent_availability" / "manage_three_machine.py")
ROOT_REPO = "/root/autodl-tmp/FederatedScope"
REMOTE_RUN = "exp/military_aircraft_3domain/robust_seed_tuning"


def load_manager():
    spec = importlib.util.spec_from_file_location(
        "military_remote_manager", MANAGER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def connect(args):
    manager = load_manager()
    namespace = argparse.Namespace(
        key=args.key, root_banner_timeout=args.root_banner_timeout)
    return manager, manager.connect_root_only(namespace)


def inspect(args):
    manager, hosts = connect(args)
    command = rf"""
set -euo pipefail
cd {ROOT_REPO}
printf 'training_roles=%s\n' "$(pgrep -af '[f]ederatedscope.main|[f]ederatedscope/main.py' | wc -l)"
nvidia-smi --query-gpu=index,name,memory.free,utilization.gpu --format=csv,noheader
/root/.local/share/mamba/envs/GGEUR/bin/python - <<'PY'
import glob, json
paths = sorted(set(
    glob.glob('exp/military_aircraft_3domain/**/accuracy_summary.json', recursive=True)
    + glob.glob('exp/military_aircraft_3domain/**/*accuracy*100round.json', recursive=True)
))
for path in paths:
    try:
        data = json.load(open(path, encoding='utf-8'))
    except Exception:
        continue
    final = data.get('final', {{}})
    average = final.get('average', data.get('final_accuracy'))
    if average is not None:
        print(json.dumps({{
            'path': path,
            'final_average': average,
            'best_average': data.get('best_average'),
        }}, sort_keys=True))
PY
"""
    try:
        result = manager.run(
            hosts["root4090"], command, timeout=180, check=False)
    finally:
        manager.close_hosts(hosts)
    print(result["stdout"])
    if result["stderr"]:
        print(result["stderr"])
    return result["status"]


def sync(args):
    manager, hosts = connect(args)
    local_dir = REPO / "scripts" / "military_aircraft_3domain"
    files = [
        path for path in local_dir.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    ]
    files.append(
        REPO / "federatedscope" / "contrib" / "worker" /
        "ggeur_server.py")
    try:
        for path in files:
            relative = path.relative_to(REPO).as_posix()
            remote = f"{ROOT_REPO}/{relative}"
            manager.run(
                hosts["root4090"],
                "mkdir -p '{}'".format(remote.rsplit("/", 1)[0]))
            manager.upload_file(
                hosts["root4090"], path, remote, newline="lf")
    finally:
        manager.close_hosts(hosts)
    print(json.dumps({"synced_files": len(files)}, indent=2))
    return 0


def start(args):
    manager, hosts = connect(args)
    script = (
        "scripts/military_aircraft_3domain/"
        "run_robust_seed_tuning.sh")
    command = rf"""
set -euo pipefail
cd {ROOT_REPO}
if pgrep -af '[f]ederatedscope.main|[f]ederatedscope/main.py' >/dev/null; then
  echo 'another_training_role_is_active' >&2
  exit 20
fi
mkdir -p {REMOTE_RUN}
nohup bash {script} >{REMOTE_RUN}/launcher.log 2>&1 </dev/null &
printf 'PID=%s\n' "$!"
"""
    try:
        result = manager.run(hosts["root4090"], command, check=False)
    finally:
        manager.close_hosts(hosts)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result["status"]


def status(args):
    manager, hosts = connect(args)
    command = rf"""
cd {ROOT_REPO}
if [[ -f {REMOTE_RUN}/exit_code ]]; then
  printf 'state=FINISHED\nexit_code=%s\n' "$(cat {REMOTE_RUN}/exit_code)"
elif pgrep -af '[f]ederatedscope.main|[f]ederatedscope/main.py' >/dev/null; then
  printf 'state=RUNNING\n'
else
  printf 'state=UNKNOWN\n'
fi
printf 'result_files=%s\n' "$(find {REMOTE_RUN} -name accuracy_summary.json -type f 2>/dev/null | wc -l)"
printf '%s\n' '-- launcher tail --'
tail -n 50 {REMOTE_RUN}/launcher.log 2>/dev/null || true
printf '%s\n' '-- current accuracy tail --'
find {REMOTE_RUN} -name '*.log' -type f -printf '%T@ %p\n' 2>/dev/null |
  sort -nr | head -n 1 | cut -d' ' -f2- | while read -r log; do
    printf 'log=%s\n' "$log"
    grep -E 'Server: Round .* Test Accuracy|ROBUST_' "$log" | tail -n 12 || true
  done
printf '%s\n' '-- failed log tail --'
find {REMOTE_RUN}/logs -name '*.log' -type f -printf '%T@ %p\n' 2>/dev/null |
  sort -nr | head -n 1 | cut -d' ' -f2- | while read -r log; do
    printf 'failed_log=%s\n' "$log"
    tail -n 35 "$log" || true
  done
"""
    try:
        result = manager.run(
            hosts["root4090"], command, timeout=120, check=False)
    finally:
        manager.close_hosts(hosts)
    print(result["stdout"])
    if result["stderr"]:
        print(result["stderr"])
    return result["status"]


def fetch(args):
    manager, hosts = connect(args)
    remote_archive = "/tmp/military_robust_seed_tuning.tar.gz"
    local = Path(args.output).resolve()
    local.parent.mkdir(parents=True, exist_ok=True)
    formal_scenario = (
        f"{REMOTE_RUN}/scenarios/formal_lr5e5_platform_uncapped")
    command = rf"""
set -euo pipefail
cd {ROOT_REPO}
find {formal_scenario} -type f -name accuracy_summary.json -print0 > /tmp/military_final_files.list
find {REMOTE_RUN}/logs -type f -name 'formal_lr5e5_platform_uncapped_*.log' -print0 >> /tmp/military_final_files.list
printf '%s\0' {REMOTE_RUN}/launcher.log >> /tmp/military_final_files.list
tar --null -czf {remote_archive} -T /tmp/military_final_files.list
"""
    try:
        manager.run(hosts["root4090"], command, timeout=300)
        manager.download_file(hosts["root4090"], remote_archive, local)
    finally:
        manager.close_hosts(hosts)
    print(json.dumps({"output": str(local)}, ensure_ascii=False, indent=2))
    return 0


def fetch_samples(args):
    """Download one deterministic image for every domain/class pair."""
    manager, hosts = connect(args)
    remote_archive = "/tmp/military_aircraft_dataset_samples.tar.gz"
    local = Path(args.output).resolve()
    local.parent.mkdir(parents=True, exist_ok=True)
    dataset_root = "/root/autodl-tmp/datasets/MilitaryAircraft3D"
    command = rf"""
set -euo pipefail
list=/tmp/military_aircraft_dataset_samples.list
: > "$list"
for domain in aerial natural recon; do
  for class_name in B-52 C-130 C-17 F-15 F-16; do
    find "{dataset_root}/$domain/$class_name" -type f \
      \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) \
      | sort | head -n 1 >> "$list"
  done
done
test "$(wc -l < "$list")" -eq 15
tar -czf {remote_archive} -T "$list"
"""
    try:
        manager.run(hosts["root4090"], command, timeout=300)
        manager.download_file(hosts["root4090"], remote_archive, local)
    finally:
        manager.close_hosts(hosts)
    print(json.dumps({"output": str(local)}, ensure_ascii=False, indent=2))
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action", choices=(
            "inspect", "sync", "start", "status", "fetch",
            "fetch_samples"))
    parser.add_argument(
        "--key", default=os.path.expanduser(r"~\.ssh\id_ed25519"))
    parser.add_argument("--root-banner-timeout", type=int, default=90)
    parser.add_argument(
        "--output",
        default="docs/test_logs/military_aircraft_robust_seed_tuning.tar.gz")
    args = parser.parse_args()
    return globals()[args.action](args)


if __name__ == "__main__":
    raise SystemExit(main())
