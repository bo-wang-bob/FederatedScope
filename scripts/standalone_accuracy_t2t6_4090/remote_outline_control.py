#!/usr/bin/env python3
"""Control the outline T2-T6 commands on the lab 4090 through the 8G jump."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shlex
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
MANAGER_PATH = (
    REPO / "scripts" / "distributed_scripts" /
    "ggeur_concurrent_availability" / "manage_three_machine.py"
)
ROOT_REPO = "/root/autodl-tmp/FederatedScope"
CONTROL_ROOT = f"{ROOT_REPO}/exp/t2t6_outline_rerun_20260818/control"

COMMANDS = {
    ("T-02", "fedavg"): (
        "python federatedscope/main.py --cfg "
        "scripts/example_configs/ggeur_final_5models/digit3_vit/fedavg.yaml"
    ),
    ("T-02", "fedprox"): (
        "python federatedscope/main.py --cfg "
        "scripts/example_configs/ggeur_final_5models/digit3_vit/fedprox.yaml"
    ),
    ("T-02", "platform"): (
        "python federatedscope/main.py --cfg "
        "scripts/example_configs/ggeur_final_5models/digit3_vit/ggeur.yaml "
        "ggeur.task_adaptation_file "
        "scripts/test_outline_validation/task_profiles/"
        "mddigits_class_targets.json"
    ),
    ("T-03", "fedavg"): (
        "python federatedscope/main.py --cfg "
        "scripts/example_configs/ggeur_final_5models/officehome_cnn/fedavg.yaml"
    ),
    ("T-03", "fedprox"): (
        "python federatedscope/main.py --cfg "
        "scripts/example_configs/ggeur_final_5models/officehome_cnn/fedprox.yaml"
    ),
    ("T-03", "platform"): (
        "python federatedscope/main.py --cfg "
        "scripts/example_configs/ggeur_final_5models/officehome_cnn/ggeur.yaml "
        "ggeur.task_adaptation_file "
        "scripts/test_outline_validation/task_profiles/"
        "officehome_accuracy_class_targets.json "
        "ggeur.training_distribution_dir "
        "exp/test_outline_validation/T-03/training_distributions"
    ),
    ("T-04", "fedavg"): (
        "python federatedscope/main.py --cfg "
        "scripts/example_configs/ggeur_final_5models/officehome_mixer/fedavg.yaml"
    ),
    ("T-04", "fedprox"): (
        "python federatedscope/main.py --cfg "
        "scripts/example_configs/ggeur_final_5models/officehome_mixer/fedprox.yaml"
    ),
    ("T-04", "platform"): (
        "python federatedscope/main.py --cfg "
        "scripts/example_configs/ggeur_final_5models/officehome_mixer/ggeur.yaml "
        "ggeur.task_adaptation_file "
        "scripts/test_outline_validation/task_profiles/"
        "officehome_accuracy_class_targets.json "
        "ggeur.training_distribution_dir "
        "exp/test_outline_validation/T-04/training_distributions"
    ),
    ("T-05", "fedavg"): (
        "python federatedscope/main.py --cfg "
        "scripts/example_configs/ggeur_final_5models/mdsent_rnn/fedavg.yaml "
        "federate.total_round_num 100"
    ),
    ("T-05", "fedprox"): (
        "python federatedscope/main.py --cfg "
        "scripts/example_configs/ggeur_final_5models/mdsent_rnn/fedprox.yaml "
        "federate.total_round_num 100"
    ),
    ("T-05", "platform"): (
        "python federatedscope/main.py --cfg "
        "scripts/example_configs/ggeur_final_5models/mdsent_rnn/ggeur.yaml "
        "federate.total_round_num 100 "
        "ggeur.platform_target_samples_per_client 0 "
        "ggeur.task_adaptation_file "
        "scripts/test_outline_validation/task_profiles/mdsent_class_targets.json "
        "ggeur.training_distribution_dir "
        "exp/test_outline_validation/T-05/training_distributions"
    ),
    ("T-06", "fedavg"): (
        "python federatedscope/main.py --cfg "
        "scripts/example_configs/ggeur_final_5models/mdsent_lstm/fedavg.yaml "
        "federate.total_round_num 100"
    ),
    ("T-06", "fedprox"): (
        "python federatedscope/main.py --cfg "
        "scripts/example_configs/ggeur_final_5models/mdsent_lstm/fedprox.yaml "
        "federate.total_round_num 100"
    ),
    ("T-06", "platform"): (
        "python federatedscope/main.py --cfg "
        "scripts/example_configs/ggeur_final_5models/mdsent_lstm/ggeur.yaml "
        "federate.total_round_num 100 "
        "ggeur.platform_target_samples_per_client 0 "
        "ggeur.task_adaptation_file "
        "scripts/test_outline_validation/task_profiles/mdsent_class_targets.json "
        "ggeur.training_distribution_dir "
        "exp/test_outline_validation/T-06/training_distributions"
    ),
}

LOGS = {
    (case, method): "{}_{}_formal_{}.log".format(
        {
            "T-02": "t2",
            "T-03": "t3",
            "T-04": "t4",
            "T-05": "t5",
            "T-06": "t6",
        }[case],
        {
            "T-02": "digit3_vit",
            "T-03": "officehome_cnn",
            "T-04": "officehome_mlp",
            "T-05": "mdsent_rnn",
            "T-06": "mdsent_lstm",
        }[case],
        method,
    )
    for case, method in COMMANDS
}


def load_manager():
    spec = importlib.util.spec_from_file_location("outline_remote_manager", MANAGER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def connect(args):
    manager = load_manager()
    ns = argparse.Namespace(
        key=args.key,
        root_banner_timeout=args.root_banner_timeout,
    )
    return manager, manager.connect_root_only(ns)


def inspect(args):
    manager, hosts = connect(args)
    command = rf"""
set -euo pipefail
cd {ROOT_REPO}
export PATH=/root/.local/share/mamba/envs/GGEUR/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
printf 'hostname=%s\n' "$(hostname)"
printf 'python=%s\n' "$(command -v python || true)"
python -V
python -c 'import torch; print("torch=" + torch.__version__); print("cuda=" + str(torch.cuda.is_available())); print("gpu_count=" + str(torch.cuda.device_count()))'
nvidia-smi --query-gpu=index,name,memory.total,memory.free,utilization.gpu --format=csv,noheader
printf 'training_roles=%s\n' "$(pgrep -af '[f]ederatedscope/main.py|[r]un_t2t6_4090.py' | wc -l)"
find /root/.cache /root/autodl-tmp -type f \
  \( -iname '*convnext*' -o -name 'convnext_base-6075fbad.pth' \) \
  -print 2>/dev/null | head -n 30 || true
python - <<'PY'
import os
paths = [
    'exp/distributed_feature_cache/digit3_vit',
    'exp/cache_warmup/officehome_cnn/cache_warmup',
    'exp/cache_warmup/officehome_mixer/cache_warmup',
    'exp/cache_warmup/mdsent_rnn/cache_warmup',
    'exp/aug/p814a/mdsent_rnn_platform/augmented_features',
    'exp/aug/p814a/mdsent_lstm_platform/augmented_features',
    'data/clip_feature_cache',
    '/root/autodl-tmp/datasets/clip_feature_cache',
    '/root/autodl-tmp/FederatedScope_t2t6_20260814/data/clip_feature_cache',
    'exp/t2t6_4090_platform_cache/t2_digit3_vit',
    'exp/t2t6_4090_platform_cache/t3_officehome_cnn',
    'exp/t2t6_4090_platform_cache/t4_officehome_mlp',
    'exp/t2t6_4090_platform_cache/t5_mdsent_rnn',
    'exp/t2t6_4090_platform_cache/t6_mdsent_lstm',
]
for path in paths:
    if not os.path.isdir(path):
        print(f'cache={{path}} MISSING')
        continue
    files = 0
    size = 0
    for root, _, names in os.walk(path):
        for name in names:
            files += 1
            try:
                size += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    print(f'cache={{path}} files={{files}} bytes={{size}}')
    if path.startswith('exp/cache_warmup') or path.endswith('digit3_vit'):
        shown = 0
        for root, _, names in os.walk(path):
            for name in sorted(names):
                print('cache_file=' + os.path.relpath(
                    os.path.join(root, name), path))
                shown += 1
                if shown >= 20:
                    break
            if shown >= 20:
                break
patterns = ('office', 'art_', 'clipart_', 'product_', 'real_world_', 'mdsent',
            'books_', 'dvd_', 'electronics_', 'kitchen_')
matched_dirs = set()
for root, dirs, names in os.walk('/root/autodl-tmp'):
    depth = root[len('/root/autodl-tmp'):].count(os.sep)
    if depth > 8:
        dirs[:] = []
        continue
    if any(name.endswith(('.npz', '.pt')) and
           any(token in name.lower() for token in patterns)
           for name in names):
        matched_dirs.add(root)
for path in sorted(matched_dirs):
    print('feature_file_candidate=' + path)
for root, dirs, names in os.walk('/root/autodl-tmp'):
    depth = root[len('/root/autodl-tmp'):].count(os.sep)
    if depth > 8:
        dirs[:] = []
        continue
    for name in names:
        lowered = name.lower()
        if (name == 'dataset_manifest.json' or
                ('convnext' in lowered and name.endswith(('.pth', '.pt')))):
            print('required_file_candidate=' + os.path.join(root, name))
    if os.path.basename(root).lower() in {'sentiment', 'mdsent', 'mdsent_cache'}:
        print('required_dir_candidate=' + root)
PY
"""
    try:
        result = manager.run(hosts["root4090"], command, timeout=180, check=False)
    finally:
        manager.close_hosts(hosts)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result["status"]


def search_legacy_cache(args):
    manager, hosts = connect(args)
    command = rf"""
set -uo pipefail
printf '%s\n' '-- exact legacy cache namespaces --'
find /root /root/autodl-tmp -type d \
  -name '*c5719ddf8d5139ea*' -print 2>/dev/null || true
printf '%s\n' '-- archives that may contain MDSent caches --'
find /root /root/autodl-tmp -type f \
  \( -name '*.tar' -o -name '*.tar.gz' -o -name '*.tgz' -o -name '*.zip' \) \
  -size +1M -print 2>/dev/null | head -n 200 || true
"""
    try:
        result = manager.run(
            hosts["root4090"], command, timeout=300, check=False)
    finally:
        manager.close_hosts(hosts)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result["status"]


def locate_evidence(args):
    manager, hosts = connect(args)
    command = rf"""
cd {ROOT_REPO}
printf '%s\n' '-- training distribution evidence --'
find exp -type f -path '*training_distributions/client_*.json' \
  -print 2>/dev/null | sort || true
printf '%s\n' '-- formal logs --'
ls -l t[2-6]_*_formal_*.log 2>/dev/null || true
"""
    try:
        result = manager.run(
            hosts["root4090"], command, timeout=120, check=False)
    finally:
        manager.close_hosts(hosts)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result["status"]


def upload(args):
    if not args.archive:
        raise SystemExit("--archive is required")
    archive = Path(args.archive).resolve()
    if not archive.is_file():
        raise SystemExit(f"archive does not exist: {archive}")
    manager, hosts = connect(args)
    is_digit_manifests = args.upload_target == "digit-manifests"
    remote = (
        "/tmp/t2t6_digit_manifests.tar.gz"
        if is_digit_manifests else "/tmp/t2t6_outline_sync.tar.gz"
    )
    try:
        manager.upload_file(hosts["root4090"], archive, remote)
        if is_digit_manifests:
            result = manager.run(
                hosts["root4090"],
                "set -euo pipefail; "
                "mkdir -p /root/autodl-tmp/datasets/digit_three_domain; "
                f"tar -xzf {remote} -C "
                "/root/autodl-tmp/datasets/digit_three_domain; "
                "count=$(find /root/autodl-tmp/datasets/digit_three_domain/"
                "manifests -type f -name client_manifest.json | wc -l); "
                "printf 'DIGIT_CLIENT_MANIFESTS=%s\\n' \"$count\"; "
                "test \"$count\" -eq 60",
                timeout=300,
                check=False,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return result["status"]
        result = manager.run(
            hosts["root4090"],
            "set -euo pipefail; "
            "export PATH=/root/.local/share/mamba/envs/GGEUR/bin:"
            "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; "
            f"mkdir -p {ROOT_REPO}; "
            f"tar -xzf {remote} -C {ROOT_REPO}; "
            f"cd {ROOT_REPO}; "
            "python -m py_compile federatedscope/contrib/worker/ggeur_client.py "
            "federatedscope/contrib/worker/ggeur_server.py "
            "federatedscope/core/auxiliaries/logging.py "
            "scripts/standalone_accuracy_t2t6_4090/run_t2t6_4090.py; "
            "echo SYNC_OK",
            timeout=300,
            check=False,
        )
    finally:
        manager.close_hosts(hosts)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result["status"]


def preflight(args):
    manager, hosts = connect(args)
    command = (
        f"cd {ROOT_REPO}; "
        "export PATH=/root/.local/share/mamba/envs/GGEUR/bin:"
        "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; "
        f"export PYTHONPATH={ROOT_REPO}; "
        "export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; "
        "python scripts/standalone_accuracy_t2t6_4090/preflight_4090.py"
    )
    try:
        result = manager.run(
            hosts["root4090"], command, timeout=300, check=False)
    finally:
        manager.close_hosts(hosts)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result["status"]


def prepare(args):
    manager, hosts = connect(args)
    command = rf"""
set -euo pipefail
mkdir -p {ROOT_REPO}/data
if [[ ! -e {ROOT_REPO}/data/digit_three_domain ]]; then
  ln -s /root/autodl-tmp/datasets/digit_three_domain \
    {ROOT_REPO}/data/digit_three_domain
fi
if [[ ! -e {ROOT_REPO}/data/sentiment ]]; then
  ln -s /root/autodl-tmp/datasets/sentiment {ROOT_REPO}/data/sentiment
fi
mkdir -p {ROOT_REPO}/pretrained_models
if [[ ! -e {ROOT_REPO}/pretrained_models/convnext_base-6075fbad.pth ]]; then
  ln -s /root/.cache/torch/hub/checkpoints/convnext_base-6075fbad.pth \
    {ROOT_REPO}/pretrained_models/convnext_base-6075fbad.pth
fi
test -f {ROOT_REPO}/data/digit_three_domain/dataset_manifest.json
test -d {ROOT_REPO}/data/sentiment
test -f {ROOT_REPO}/pretrained_models/convnext_base-6075fbad.pth
printf 'DIGIT_DATA_READY=%s\n' "$(readlink -f {ROOT_REPO}/data/digit_three_domain)"
printf 'MDSENT_DATA_READY=%s\n' "$(readlink -f {ROOT_REPO}/data/sentiment)"
printf 'CONVNEXT_READY=%s\n' "$(readlink -f {ROOT_REPO}/pretrained_models/convnext_base-6075fbad.pth)"
"""
    try:
        result = manager.run(hosts["root4090"], command, timeout=120, check=False)
    finally:
        manager.close_hosts(hosts)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result["status"]


def run_key(case: str, method: str, warmup: bool, trial_tag: str = "") -> str:
    suffix = "warmup" if warmup else "formal"
    if trial_tag:
        suffix = f"trial_{trial_tag}"
    return f"{case.lower().replace('-', '')}_{method}_{suffix}"


def stable_log_for(args) -> str:
    if not args.trial_tag:
        return LOGS[(args.case, args.method)]
    base = LOGS[(args.case, args.method)].removesuffix(".log")
    return f"{base}_trial_{args.trial_tag}.log"


def start(args):
    key = (args.case, args.method)
    if key not in COMMANDS:
        raise SystemExit(f"unsupported run: {key}")
    command = COMMANDS[key]
    if args.rounds is not None:
        command += f" federate.total_round_num {args.rounds}"
    if args.local_steps is not None:
        command += f" train.local_update_steps {args.local_steps}"
    if args.sample_clients is not None:
        command += f" federate.sample_client_num {args.sample_clients}"
    if args.device is not None:
        command += f" device {args.device}"
    if args.seed is not None:
        command += f" seed {args.seed}"
    if args.model_hidden is not None:
        command += f" model.hidden {args.model_hidden}"
    if args.model_layer is not None:
        command += f" model.layer {args.model_layer}"
    if args.model_dropout is not None:
        command += f" model.dropout {args.model_dropout}"
    if args.fedprox_mu is not None:
        if args.method != "fedprox":
            raise SystemExit("--fedprox-mu is only valid for FedProx")
        command += f" fedprox.mu {args.fedprox_mu}"
    if args.baseline_samples is not None:
        if args.method not in {"fedavg", "fedprox"}:
            raise SystemExit(
                "--baseline-samples is only valid for FedAvg/FedProx")
        command += (
            " ggeur.baseline_target_samples_per_client "
            f"{args.baseline_samples}"
        )
    if args.platform_samples is not None:
        if args.method != "platform":
            raise SystemExit("--platform-samples is only valid for platform")
        command += (
            " ggeur.platform_target_samples_per_client "
            f"{args.platform_samples}"
        )
    if args.platform_auto_samples is not None:
        if args.method != "platform":
            raise SystemExit(
                "--platform-auto-samples is only valid for platform")
        command += (
            " ggeur.platform_auto_target_samples_per_client "
            f"{args.platform_auto_samples}"
        )
    platform_only_overrides = {
        "task_profile_path": "ggeur.task_adaptation_file",
        "aug_cache_dir": "ggeur.augmented_feature_cache_dir",
        "aug_cache_version": "ggeur.augmented_feature_cache_version",
        "generated_per_sample": "ggeur.num_generated_per_sample",
        "generated_per_prototype": "ggeur.num_generated_per_prototype",
        "target_per_class": "ggeur.target_size_per_class",
    }
    for attr, config_key in platform_only_overrides.items():
        value = getattr(args, attr)
        if value is None:
            continue
        if args.method != "platform":
            raise SystemExit(f"--{attr.replace('_', '-')} is platform-only")
        command += f" {config_key} {shlex.quote(str(value))}"
    if args.trial_tag:
        command += f" log_file {stable_log_for(args)}"
    if args.without_task_profile:
        if args.method != "platform" or not args.warmup:
            raise SystemExit(
                "--without-task-profile is only valid for platform warmup")
        command = command.split(" ggeur.task_adaptation_file ", 1)[0]
    if args.full_client_cache:
        if args.method != "platform" or not args.warmup:
            raise SystemExit(
                "--full-client-cache is only valid for platform warmup")
        command += (
            " federate.sample_client_num 120 "
            "ggeur.min_statistics_clients 120 "
            "ggeur.min_augmentation_clients 120 "
            "ggeur.min_train_updates 120"
        )
    if args.warmup:
        command += (
            " federate.total_round_num 1 "
            "ggeur.require_complete_feature_cache False"
        )
    name = run_key(args.case, args.method, args.warmup, args.trial_tag)
    stable_log = stable_log_for(args)
    manager, hosts = connect(args)
    script_path = f"{CONTROL_ROOT}/{name}.sh"
    console_path = f"{CONTROL_ROOT}/{name}.console.log"
    exit_path = f"{CONTROL_ROOT}/{name}.exit"
    timing_path = f"{CONTROL_ROOT}/{name}.timing.json"
    archive_dir = f"{CONTROL_ROOT}/archive"
    script = f"""#!/usr/bin/env bash
set -uo pipefail
cd {shlex.quote(ROOT_REPO)}
export PATH=/root/.local/share/mamba/envs/GGEUR/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export PYTHONPATH={shlex.quote(ROOT_REPO)}${{PYTHONPATH:+:$PYTHONPATH}}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p {shlex.quote(CONTROL_ROOT)} {shlex.quote(archive_dir)}
if [[ -f {shlex.quote(console_path)} || -f {shlex.quote(exit_path)} || -f {shlex.quote(timing_path)} ]]; then
  stamp=$(date +%Y%m%d_%H%M%S)
  for previous in {shlex.quote(console_path)} {shlex.quote(exit_path)} {shlex.quote(timing_path)} {shlex.quote(script_path)}; do
    if [[ -f "$previous" ]]; then
      mv "$previous" {shlex.quote(archive_dir)}/"$(basename "$previous")".$stamp
    fi
  done
fi
if [[ -f {shlex.quote(stable_log)} ]]; then
  stamp=$(date +%Y%m%d_%H%M%S)
  mv {shlex.quote(stable_log)} {shlex.quote(archive_dir)}/{shlex.quote(stable_log)}.$stamp
fi
started=$(date +%s.%N)
printf '{{"name":"%s","case":"%s","method":"%s","phase":"%s","started_at_unix":%s,"command":%s}}\n' \
  {shlex.quote(name)} {shlex.quote(args.case)} {shlex.quote(args.method)} \
  {shlex.quote('warmup' if args.warmup else 'formal')} "$started" \
  {shlex.quote(json.dumps(command))} > {shlex.quote(timing_path)}
set +e
{command} > {shlex.quote(console_path)} 2>&1
code=$?
set -e
finished=$(date +%s.%N)
printf '%s\n' "$code" > {shlex.quote(exit_path)}
python - {shlex.quote(timing_path)} "$finished" "$code" <<'PY'
import json, sys
p, finished, code = sys.argv[1:]
d = json.load(open(p, encoding='utf-8'))
elapsed = float(finished) - float(d['started_at_unix'])
d.update(finished_at_unix=float(finished), elapsed_seconds=round(elapsed, 3), exit_code=int(code))
json.dump(d, open(p, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
PY
exit "$code"
"""
    try:
        active = manager.run(
            hosts["root4090"],
            "pgrep -af '[f]ederatedscope/main.py' || true",
        )["stdout"]
        if active:
            raise RuntimeError(f"an accuracy run is already active:\n{active}")
        manager.run(hosts["root4090"], f"mkdir -p {CONTROL_ROOT}")
        manager.upload_bytes(
            hosts["root4090"], script.encode("utf-8"), script_path)
        launch = manager.run(
            hosts["root4090"],
            f"chmod 755 {script_path}; "
            f"nohup bash {script_path} >{CONTROL_ROOT}/{name}.launcher.log "
            f"2>&1 </dev/null & echo PID=$!",
        )
    finally:
        manager.close_hosts(hosts)
    print(json.dumps({
        "name": name,
        "command": command,
        "stable_log": stable_log,
        "launch": launch,
    }, ensure_ascii=False, indent=2))
    return 0


def status(args):
    name = run_key(args.case, args.method, args.warmup, args.trial_tag)
    stable_log = stable_log_for(args)
    manager, hosts = connect(args)
    command = rf"""
cd {ROOT_REPO}
export PATH=/root/.local/share/mamba/envs/GGEUR/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
printf 'name={name}\n'
if [[ -f {CONTROL_ROOT}/{name}.exit ]]; then
  printf 'state=FINISHED\nexit_code=%s\n' "$(cat {CONTROL_ROOT}/{name}.exit)"
elif pgrep -af '[f]ederatedscope/main.py' >/dev/null; then
  printf 'state=RUNNING\n'
else
  printf 'state=UNKNOWN\n'
fi
if [[ -f {CONTROL_ROOT}/{name}.timing.json ]]; then
  if [[ -f {CONTROL_ROOT}/{name}.exit ]] && \
     ! grep -q 'elapsed_seconds' {CONTROL_ROOT}/{name}.timing.json; then
    python - {CONTROL_ROOT}/{name}.timing.json \
      {CONTROL_ROOT}/{name}.console.log {CONTROL_ROOT}/{name}.exit <<'PY'
import json, os, sys
timing_path, console_path, exit_path = sys.argv[1:]
data = json.load(open(timing_path, encoding='utf-8'))
finished = os.path.getmtime(console_path)
elapsed = finished - float(data['started_at_unix'])
data.update(
    finished_at_unix=finished,
    elapsed_seconds=round(elapsed, 3),
    exit_code=int(open(exit_path, encoding='utf-8').read().strip()),
    finish_time_source='console_log_mtime_repair',
)
json.dump(data, open(timing_path, 'w', encoding='utf-8'),
          ensure_ascii=False, indent=2)
PY
  fi
  cat {CONTROL_ROOT}/{name}.timing.json
fi
printf '\nround_accuracy_lines=%s\n' "$(grep -c 'Server: Round .* Test Accuracy' {stable_log} 2>/dev/null || true)"
printf '%s\n' '-- stable log tail --'
grep -E 'Starting training \(Round|Server: Round .* Test Accuracy|Training finished after|average: final=|Loaded augmented feature cache|Saved augmented feature cache|Ignore augmented cache metadata|Loaded task-adaptive generation profile' {stable_log} 2>/dev/null | tail -n 18 || true
printf '%s\n' '-- console tail --'
tail -n 20 {CONTROL_ROOT}/{name}.console.log 2>/dev/null || true
printf '%s\n' '-- launcher tail --'
tail -n 10 {CONTROL_ROOT}/{name}.launcher.log 2>/dev/null || true
"""
    try:
        result = manager.run(hosts["root4090"], command, timeout=120, check=False)
    finally:
        manager.close_hosts(hosts)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result["status"]


def fetch(args):
    manager, hosts = connect(args)
    remote = "/tmp/t2t6_outline_rerun_20260818.tar.gz"
    local = Path(args.output).resolve()
    local.parent.mkdir(parents=True, exist_ok=True)
    command = (
        f"cd {ROOT_REPO}; tar -czf {remote} "
        "exp/t2t6_outline_rerun_20260818 "
        "exp/digit_three_domain/training_distributions "
        "exp/test_outline_validation/T-03 "
        "exp/test_outline_validation/T-04 "
        "exp/test_outline_validation/T-05 "
        "exp/test_outline_validation/T-06 "
        "t2_digit3_vit_formal_*.log t3_officehome_cnn_formal_*.log "
        "t4_officehome_mlp_formal_*.log t5_mdsent_rnn_formal_*.log "
        "t6_mdsent_lstm_formal_*.log"
    )
    try:
        manager.run(hosts["root4090"], command, timeout=300)
        manager.download_file(hosts["root4090"], remote, local)
    finally:
        manager.close_hosts(hosts)
    print(json.dumps({"output": str(local)}, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=(
            "inspect", "search_legacy_cache", "locate_evidence",
            "upload", "prepare", "preflight",
            "start", "status", "fetch",
        ),
    )
    parser.add_argument("--key", default=os.path.expanduser(r"~\.ssh\id_ed25519"))
    parser.add_argument("--root-banner-timeout", type=int, default=90)
    parser.add_argument("--archive")
    parser.add_argument(
        "--upload-target",
        choices=("sync", "digit-manifests"),
        default="sync",
    )
    parser.add_argument("--case", choices=("T-02", "T-03", "T-04", "T-05", "T-06"))
    parser.add_argument("--method", choices=("fedavg", "fedprox", "platform"))
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--local-steps", type=int)
    parser.add_argument("--sample-clients", type=int)
    parser.add_argument("--device", type=int, choices=(0, 1))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--model-hidden", type=int)
    parser.add_argument("--model-layer", type=int)
    parser.add_argument("--model-dropout", type=float)
    parser.add_argument("--fedprox-mu", type=float)
    parser.add_argument("--baseline-samples", type=int)
    parser.add_argument("--platform-samples", type=int)
    parser.add_argument("--platform-auto-samples", type=int)
    parser.add_argument("--task-profile-path")
    parser.add_argument("--aug-cache-dir")
    parser.add_argument("--aug-cache-version")
    parser.add_argument("--generated-per-sample", type=int)
    parser.add_argument("--generated-per-prototype", type=int)
    parser.add_argument("--target-per-class", type=int)
    parser.add_argument("--trial-tag", default="")
    parser.add_argument("--without-task-profile", action="store_true")
    parser.add_argument("--full-client-cache", action="store_true")
    parser.add_argument("--output", default="docs/test_logs/t2t6_outline_rerun_20260818.tar.gz")
    args = parser.parse_args()
    if args.trial_tag and not re.fullmatch(r"[A-Za-z0-9_-]+", args.trial_tag):
        parser.error("--trial-tag may contain only letters, numbers, '_' and '-'")
    if args.rounds is not None and args.rounds <= 0:
        parser.error("--rounds must be positive")
    if args.local_steps is not None and args.local_steps <= 0:
        parser.error("--local-steps must be positive")
    if args.sample_clients is not None and args.sample_clients <= 0:
        parser.error("--sample-clients must be positive")
    if args.model_hidden is not None and args.model_hidden <= 0:
        parser.error("--model-hidden must be positive")
    if args.model_layer is not None and args.model_layer <= 0:
        parser.error("--model-layer must be positive")
    if (args.model_dropout is not None and
            not 0.0 <= args.model_dropout < 1.0):
        parser.error("--model-dropout must be in [0, 1)")
    if args.baseline_samples is not None and args.baseline_samples <= 0:
        parser.error("--baseline-samples must be positive")
    if args.platform_samples is not None and args.platform_samples <= 0:
        parser.error("--platform-samples must be positive")
    if (args.platform_auto_samples is not None and
            args.platform_auto_samples < 0):
        parser.error("--platform-auto-samples must be non-negative")
    for attr in ("generated_per_sample", "generated_per_prototype",
                 "target_per_class"):
        value = getattr(args, attr)
        if value is not None and value < 0:
            parser.error(f"--{attr.replace('_', '-')} must be non-negative")
    if args.action in {"start", "status"} and not (args.case and args.method):
        parser.error("--case and --method are required")
    return globals()[args.action](args)


if __name__ == "__main__":
    raise SystemExit(main())
