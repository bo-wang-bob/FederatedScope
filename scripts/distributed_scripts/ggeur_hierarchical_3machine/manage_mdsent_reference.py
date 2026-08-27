#!/usr/bin/env python3
"""Deploy, start, and inspect the corrected distributed MDSent run.

Management always enters through the 8G host and then uses SSH direct-tcpip
to reach the internal third and 4090 hosts.  Training traffic is configured
separately in the generated YAML/JSON files and uses only the real 10.x
addresses.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import time


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
CONCURRENT_DIR = THIS_DIR.parent / "ggeur_concurrent_availability"
sys.path.insert(0, str(CONCURRENT_DIR))

from manage_three_machine import (  # noqa: E402
    CLIENT_REPO,
    ROOT_REPO,
    THIRD_REPO,
    close_hosts,
    connect_hosts,
    download_file,
    encoded_powershell,
    run,
    upload_file,
)


DEFAULT_RUN_ID = "mdsent_reference_distributed_20260730_v1"
HIER_REL = "scripts/distributed_scripts/ggeur_hierarchical_3machine"

# Deploy the complete lightweight import/runtime surface used by the corrected
# MDSent clients.  This intentionally excludes unrelated datasets and any
# standalone-reference launcher.
CODE_FILES = [
    "scripts/prepare_digit_three_domain_vit_cache.py",
    "federatedscope/contrib/data/__init__.py",
    "federatedscope/contrib/data/ggeur_data.py",
    "federatedscope/contrib/data/mdsent_data.py",
    "federatedscope/contrib/model/__init__.py",
    "federatedscope/contrib/model/ggeur_cnn.py",
    "federatedscope/contrib/model/ggeur_cnn_extractor.py",
    "federatedscope/contrib/model/ggeur_mlp.py",
    "federatedscope/contrib/model/ggeur_timm_extractor.py",
    "federatedscope/contrib/model/ggeur_text_rnn.py",
    "federatedscope/contrib/trainer/__init__.py",
    "federatedscope/contrib/worker/__init__.py",
    "federatedscope/contrib/worker/ggeur_client.py",
    "federatedscope/contrib/worker/ggeur_server.py",
    "federatedscope/core/auxiliaries/criterion_builder.py",
    "federatedscope/core/auxiliaries/data_builder.py",
    "federatedscope/core/auxiliaries/metric_builder.py",
    "federatedscope/core/auxiliaries/model_builder.py",
    "federatedscope/core/auxiliaries/optimizer_builder.py",
    "federatedscope/core/auxiliaries/scheduler_builder.py",
    "federatedscope/core/auxiliaries/trainer_builder.py",
    "federatedscope/core/auxiliaries/utils.py",
    "federatedscope/core/communication.py",
    "federatedscope/core/configs/cfg_fl_setting.py",
    "federatedscope/core/configs/cfg_ggeur.py",
    "federatedscope/core/data/base_data.py",
    "federatedscope/core/fed_runner.py",
    "federatedscope/core/monitors/metric_calculator.py",
    "federatedscope/core/trainers/__init__.py",
    "federatedscope/core/workers/__init__.py",
    "federatedscope/cv/dataset/digit_three_domain.py",
    "federatedscope/cv/dataset/domainnet.py",
]

RUNTIME_FILES = [
    f"{HIER_REL}/prepare_digit3_clip_vitb16_cache.ps1",
    f"{HIER_REL}/hierarchical_subserver.py",
    f"{HIER_REL}/launch_clients.ps1",
    f"{HIER_REL}/launch_root.sh",
    f"{HIER_REL}/launch_root_clients.sh",
    f"{HIER_REL}/launch_subservers.ps1",
    f"{HIER_REL}/queue_control.py",
    f"{HIER_REL}/queue_control_client.ps1",
    f"{HIER_REL}/run_chained_queue_root.sh",
    f"{HIER_REL}/run_chained_queue_windows.ps1",
    f"{HIER_REL}/run_remaining_queue_clients.ps1",
    f"{HIER_REL}/run_remaining_queue_root.sh",
    f"{HIER_REL}/run_remaining_queue_subservers.ps1",
    f"{HIER_REL}/summarize_accuracy.py",
    f"{HIER_REL}/validate_case_completion.py",
]


def remote_path(repo, relative):
    return f"{repo.rstrip('/')}/{relative}".replace("\\", "/")


def payload_files(run_id):
    files = [REPO_ROOT / relative for relative in CODE_FILES + RUNTIME_FILES]
    run_root = THIS_DIR / "runs" / run_id
    for path in sorted(run_root.rglob("*")):
        if path.is_file() and not any(
                part in {"logs", "pids", "queue_state", "attempts"}
                for part in path.parts):
            files.append(path)
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing deployment files: " + ", ".join(missing))
    return files


def relative_payload_path(path):
    return path.relative_to(REPO_ROOT).as_posix()


def ensure_linux_directories(client, repo, files):
    directories = sorted({
        str(Path(remote_path(repo, relative_payload_path(path))).parent)
        .replace("\\", "/")
        for path in files
    })
    # Keep remote command lines bounded for full matrices with thousands of
    # generated client configuration directories.
    for start in range(0, len(directories), 200):
        batch = directories[start:start + 200]
        run(client, "mkdir -p " + " ".join(batch), timeout=120)


def ensure_windows_directories(client, repo, files):
    directories = sorted({
        str(Path(remote_path(repo, relative_payload_path(path))).parent)
        .replace("/", "\\")
        for path in files
    })
    # Encoded PowerShell is passed through cmd.exe on the 8G jump host.
    # Batch paths to stay below the Windows command-line length limit.
    for start in range(0, len(directories), 8):
        quoted = ",\n  ".join(
            f"'{path}'" for path in directories[start:start + 8])
        script = f"""
$ErrorActionPreference = 'Stop'
$dirs = @(
  {quoted}
)
foreach ($dir in $dirs) {{
  New-Item -ItemType Directory -Force -Path $dir | Out-Null
}}
"""
        run(client, encoded_powershell(script), timeout=180)


def deploy(args):
    files = payload_files(args.run_id)
    hosts = connect_hosts(args, root_required=True)
    result = {"run_id": args.run_id, "file_count": len(files), "hosts": {}}
    try:
        # A complete matrix contains thousands of small YAML files.  Package
        # them once so the 8G management hop performs one SFTP transfer per
        # host instead of thousands of latency-bound transfers.
        with tempfile.TemporaryDirectory(
                prefix=f"ggeur_deploy_{args.run_id}_") as temp_dir:
            archive_path = Path(temp_dir) / f"{args.run_id}.tar.gz"
            with tarfile.open(archive_path, "w:gz") as archive:
                for path in files:
                    archive.add(
                        path,
                        arcname=relative_payload_path(path),
                        recursive=False,
                    )

            archive_sha256 = hashlib.sha256(
                archive_path.read_bytes()).hexdigest()
            result["archive"] = {
                "bytes": archive_path.stat().st_size,
                "sha256": archive_sha256,
            }

            for host_name, repo in (
                    ("root4090", ROOT_REPO),
                    ("third", THIRD_REPO),
                    ("client8g", CLIENT_REPO)):
                remote_archive = remote_path(
                    repo, f"exp/hierarchical/.deploy_{args.run_id}.tar.gz")
                if host_name == "root4090":
                    run(
                        hosts[host_name],
                        f"mkdir -p {repo}/exp/hierarchical",
                        timeout=60,
                    )
                else:
                    prepare = rf"""
$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Force `
  -Path '{repo}\exp\hierarchical' | Out-Null
"""
                    run(
                        hosts[host_name],
                        encoded_powershell(prepare),
                        timeout=60,
                    )
                upload_file(
                    hosts[host_name], archive_path, remote_archive)
                if host_name == "root4090":
                    extract = (
                        f"tar -xzf {remote_archive} -C {repo} && "
                        f"rm -f {remote_archive}"
                    )
                    run(hosts[host_name], extract, timeout=300)
                else:
                    extract = rf"""
$ErrorActionPreference = 'Stop'
& tar.exe -xzf '{remote_archive}' -C '{repo}'
if ($LASTEXITCODE -ne 0) {{
  throw "tar extraction failed with exit code $LASTEXITCODE"
}}
Remove-Item -LiteralPath '{remote_archive}' -Force
"""
                    run(
                        hosts[host_name],
                        encoded_powershell(extract),
                        timeout=300,
                    )
                result["hosts"][host_name] = {
                    "uploaded": len(files),
                    "transport": "tar.gz",
                }

        linux_scripts = " ".join(
            remote_path(ROOT_REPO, relative)
            for relative in RUNTIME_FILES if relative.endswith(".sh"))
        run(hosts["root4090"], f"chmod 755 {linux_scripts}")
        syntax = run(
            hosts["root4090"],
            "bash -n " + remote_path(
                ROOT_REPO, f"{HIER_REL}/run_remaining_queue_root.sh"
            ) + " " + remote_path(
                ROOT_REPO, f"{HIER_REL}/run_chained_queue_root.sh"
            ) + " " + remote_path(
                ROOT_REPO, f"{HIER_REL}/launch_root.sh"
            ) + " " + remote_path(
                ROOT_REPO, f"{HIER_REL}/launch_root_clients.sh"
            ),
        )
        result["root_bash_syntax"] = syntax
    finally:
        close_hosts(hosts)
    print(json.dumps(result, ensure_ascii=True, indent=2))


def preflight(args):
    hosts = connect_hosts(args, root_required=True)
    run_rel = f"{HIER_REL}/runs/{args.run_id}"
    root_script = rf"""
set -euo pipefail
repo={ROOT_REPO}
python=/root/.local/share/mamba/envs/GGEUR/bin/python
printf 'hostname=%s\n' "$(hostname)"
printf 'training_roles=%s\n' "$(pgrep -af '[f]ederatedscope.main|[h]ierarchical_subserver.py|[r]un_remaining_queue_root.sh' | wc -l)"
printf 'python=%s\n' "$("$python" -c 'import sys; print(sys.executable)')"
printf 'torch=%s\n' "$("$python" -c 'import torch; print(torch.__version__)')"
printf 'mem_available_kb=%s\n' "$(awk '/MemAvailable/ {{print $2}}' /proc/meminfo)"
printf 'swap_free_kb=%s\n' "$(awk '/SwapFree/ {{print $2}}' /proc/meminfo)"
printf 'nofile=%s\n' "$(ulimit -n)"
printf 'dataset_domains=%s\n' "$(timeout 20 find /root/autodl-tmp/datasets/sentiment -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)"
printf 'bert_config=%s\n' "$([ -f /root/autodl-tmp/models/nlptown_bert_base_multilingual_uncased_senti/config.json ] && echo YES || echo NO)"
printf 'domainnet_manifest=%s\n' "$([ -f "$repo/exp/distributed_manifests/domainnet_4domains/domainnet_manifest.json" ] && echo YES || echo NO)"
printf 'domainnet_dataset_domains=%s\n' "$(timeout 20 find /root/autodl-tmp/datasets/DomainNet -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)"
printf 'clip_checkpoint=%s\n' "$([ -f /root/.cache/clip/ViT-B-16.pt ] && echo YES || echo NO)"
printf 'mixer_checkpoint=%s\n' "$([ -f /root/autodl-tmp/models/mixer_b16_224_complete.pth ] && echo YES || echo NO)"
for group in mdsent_rnn mdsent_lstm; do
  cache="$repo/exp/distributed_feature_cache/$group"
  printf '%s_marker=%s\n' "$group" "$([ -f "$cache/.ggeur_feature_cache_ready.json" ] && echo YES || echo NO)"
  printf '%s_npz=%s\n' "$group" "$(timeout 20 find "$cache" -maxdepth 1 -type f -name '*.npz' 2>/dev/null | wc -l)"
done
for group in domainnet_vit domainnet_cnn domainnet_mixer; do
  cache="$repo/exp/distributed_feature_cache/$group"
  printf '%s_marker=%s\n' "$group" "$([ -f "$cache/.ggeur_feature_cache_ready.json" ] && echo YES || echo NO)"
  printf '%s_npz=%s\n' "$group" "$(timeout 20 find "$cache" -maxdepth 1 -type f -name '*.npz' 2>/dev/null | wc -l)"
done
printf 'run_manifest=%s\n' "$([ -f "$repo/{run_rel}/matrix_manifest.json" ] && echo YES || echo NO)"
printf 'port_60049=%s\n' "$(ss -ltn | grep -Ec ':60049[[:space:]]' || true)"
printf 'port_60050=%s\n' "$(ss -ltn | grep -Ec ':60050[[:space:]]' || true)"
"""
    third_script = rf"""
$ErrorActionPreference = 'Stop'
$repo = '{THIRD_REPO}'
$python = 'C:\Users\pc\miniconda3\envs\cerp\python.exe'
$roles = @(Get-CimInstance Win32_Process | Where-Object {{
  $_.CommandLine -like '*federatedscope.main*' -or
  $_.CommandLine -like '*hierarchical_subserver.py*' -or
  $_.CommandLine -like '*run_remaining_queue_subservers.ps1*'
}})
"hostname=$env:COMPUTERNAME"
"training_roles=$($roles.Count)"
"python_exists=$(Test-Path -LiteralPath $python)"
if (Test-Path -LiteralPath $python) {{
  $runtime = @(& $python -c 'import sys,torch,grpc; print(sys.executable); print(torch.__version__)')
  "runtime=$($runtime -join '|')"
}}
$os = Get-CimInstance Win32_OperatingSystem
"mem_available_kb=$($os.FreePhysicalMemory)"
"run_manifest=$(Test-Path -LiteralPath (Join-Path $repo '{run_rel.replace("/", chr(92))}\matrix_manifest.json'))"
foreach ($port in 61000..61003) {{
  "port_$port=$([bool](Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue))"
}}
"""
    client_script = rf"""
$ErrorActionPreference = 'Stop'
$repo = '{CLIENT_REPO}'
$roles = @(Get-CimInstance Win32_Process | Where-Object {{
  $_.CommandLine -like '*federatedscope.main*' -or
  $_.CommandLine -like '*run_remaining_queue_clients.ps1*'
}})
"hostname=$env:COMPUTERNAME"
"training_roles=$($roles.Count)"
$os = Get-CimInstance Win32_OperatingSystem
"mem_available_kb=$($os.FreePhysicalMemory)"
"cpu_site_packages=$(Test-Path -LiteralPath (Join-Path $repo '.venv_client_cpu\Lib\site-packages'))"
"dataset_domains=$(@(Get-ChildItem -LiteralPath (Join-Path $repo 'data\sentiment') -Directory -ErrorAction SilentlyContinue).Count)"
"bert_config=$(Test-Path -LiteralPath (Join-Path $repo 'pretrained_models\nlptown_bert_base_multilingual_uncased_senti\config.json'))"
"domainnet_manifest=$(Test-Path -LiteralPath (Join-Path $repo 'exp\distributed_manifests\domainnet_4domains\domainnet_manifest.json'))"
"domainnet_dataset_domains=$(@(Get-ChildItem -LiteralPath (Join-Path $repo 'data\DomainNet') -Directory -ErrorAction SilentlyContinue).Count)"
"clip_checkpoint=$(Test-Path -LiteralPath (Join-Path $repo 'pretrained_models\ViT-B-16.pt'))"
"mixer_checkpoint=$(Test-Path -LiteralPath (Join-Path $repo 'pretrained_models\mixer_b16_224_complete.pth'))"
foreach ($group in @('mdsent_rnn', 'mdsent_lstm')) {{
  $cache = Join-Path $repo "exp\distributed_feature_cache\$group"
  "${{group}}_marker=$(Test-Path -LiteralPath (Join-Path $cache '.ggeur_feature_cache_ready.json'))"
  "${{group}}_npz=$(@(Get-ChildItem -LiteralPath $cache -File -Filter '*.npz' -ErrorAction SilentlyContinue).Count)"
  if (Test-Path -LiteralPath (Join-Path $cache '.ggeur_feature_cache_ready.json')) {{
    "${{group}}_marker_json=$((Get-Content -LiteralPath (Join-Path $cache '.ggeur_feature_cache_ready.json') -Raw).Replace("`r", '').Replace("`n", ''))"
  }}
}}
foreach ($group in @('domainnet_vit', 'domainnet_cnn', 'domainnet_mixer')) {{
  $cache = Join-Path $repo "exp\distributed_feature_cache\$group"
  "${{group}}_marker=$(Test-Path -LiteralPath (Join-Path $cache '.ggeur_feature_cache_ready.json'))"
  "${{group}}_npz=$(@(Get-ChildItem -LiteralPath $cache -File -Filter '*.npz' -ErrorAction SilentlyContinue).Count)"
  if (Test-Path -LiteralPath (Join-Path $cache '.ggeur_feature_cache_ready.json')) {{
    "${{group}}_marker_json=$((Get-Content -LiteralPath (Join-Path $cache '.ggeur_feature_cache_ready.json') -Raw).Replace("`r", '').Replace("`n", ''))"
  }}
}}
"run_manifest=$(Test-Path -LiteralPath (Join-Path $repo '{run_rel.replace("/", chr(92))}\matrix_manifest.json'))"
foreach ($port in 20000..20059) {{
  if (Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue) {{
    "occupied_client_port=$port"
  }}
}}
"""
    try:
        result = {
            "root4090": run(
                hosts["root4090"], root_script, timeout=180, check=False),
            "third": run(
                hosts["third"], encoded_powershell(third_script),
                timeout=180, check=False),
            "client8g": run(
                hosts["client8g"], encoded_powershell(client_script),
                timeout=180, check=False),
        }
    finally:
        close_hosts(hosts)
    print(json.dumps(result, ensure_ascii=True, indent=2))


def _register_queue_task(
        client, task_name, script_path, arguments, repo,
        execution_time_limit_days=7):
    ps_script = rf"""
$ErrorActionPreference = 'Stop'
$taskName = '{task_name}'
$scriptPath = '{script_path}'
$repo = '{repo}'
$cmdPath = Join-Path $repo "exp\hierarchical\{task_name}.run.cmd"
New-Item -ItemType Directory -Force -Path (Split-Path $cmdPath) | Out-Null
$commandLine = 'powershell -NoProfile -NonInteractive ' +
  '-ExecutionPolicy Bypass -File "' + $scriptPath + '" {arguments}'
@(
  '@echo off',
  $commandLine
) | Set-Content -LiteralPath $cmdPath -Encoding ASCII
$old = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($old -and $old.State -eq 'Running') {{
  Stop-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
  Start-Sleep -Seconds 2
}}
if ($old) {{
  Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
}}
$action = New-ScheduledTaskAction `
  -Execute "$env:SystemRoot\System32\cmd.exe" `
  -Argument "/d /c `"$cmdPath`"" `
  -WorkingDirectory $repo
$identity = [Security.Principal.WindowsIdentity]::GetCurrent().Name
$principal = New-ScheduledTaskPrincipal `
  -UserId $identity -LogonType S4U -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet `
  -ExecutionTimeLimit (New-TimeSpan -Days {execution_time_limit_days}) `
  -MultipleInstances IgnoreNew `
  -AllowStartIfOnBatteries `
  -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName $taskName -Action $action `
  -Principal $principal -Settings $settings -Force | Out-Null
Start-ScheduledTask -TaskName $taskName
Start-Sleep -Seconds 2
$info = Get-ScheduledTaskInfo -TaskName $taskName
"task=$taskName state=$((Get-ScheduledTask -TaskName $taskName).State) result=$($info.LastTaskResult)"
"cmd=$cmdPath"
"""
    return run(client, encoded_powershell(ps_script), timeout=90)


def start(args):
    hosts = connect_hosts(args, root_required=True)
    run_rel = f"{HIER_REL}/runs/{args.run_id}"
    root_queue = remote_path(
        ROOT_REPO, f"{HIER_REL}/run_remaining_queue_root.sh")
    root_run = remote_path(ROOT_REPO, run_rel)
    root_command = rf"""
set -euo pipefail
cd {ROOT_REPO}
test -f {root_run}/matrix_manifest.json
pid="$(pgrep -f '^bash .*run_remaining_queue_root\.sh' | head -1 || true)"
if [[ -n "$pid" ]]; then
  control_run="$(
    /root/.local/share/mamba/envs/GGEUR/bin/python -c \
      'import json; print(json.load(open("{root_run}/queue_state/control.json"))["run_id"])'
  )"
  test "$control_run" = "{args.run_id}"
  printf 'root_queue_adopted_pid=%s\n' "$pid"
else
  active_roles="$(
    ps -eo args= |
      awk '
        /^\/root\/[^ ]*\/python .*federatedscope\.main/ ||
        /^python[0-9.]* .*federatedscope\.main/ {{count++}}
        END {{print count + 0}}
      '
  )"
  test "$active_roles" -eq 0
  test ! -e {root_run}/queue_state/root_queue.complete
  mkdir -p {root_run}/queue_state
  nohup env RUN_ID={args.run_id} \
    PYTHON_BIN=/root/.local/share/mamba/envs/GGEUR/bin/python \
    REPO_DIR={ROOT_REPO} \
    bash {root_queue} \
    >{root_run}/queue_state/root_queue.driver.log 2>&1 &
  pid=$!
  printf '%s\n' "$pid" >{root_run}/queue_state/root_queue.driver.pid
  sleep 3
  kill -0 "$pid"
  printf 'root_queue_pid=%s\n' "$pid"
fi
"""
    third_script = (
        f"{THIRD_REPO}\\{HIER_REL.replace('/', chr(92))}\\"
        "run_remaining_queue_subservers.ps1"
    )
    client_script = (
        f"{CLIENT_REPO}\\{HIER_REL.replace('/', chr(92))}\\"
        "run_remaining_queue_clients.ps1"
    )
    try:
        result = {
            "root4090": (
                run(hosts["root4090"], root_command, timeout=60)
                if hosts["root4090"] is not None else {
                    "status": -1,
                    "stdout": "",
                    "stderr": hosts["root_error"],
                }
            ),
            "third": _register_queue_task(
                hosts["third"],
                f"GGEUR-{args.run_id}-subservers",
                third_script,
                (
                    f'-RunId "{args.run_id}" '
                    '-PythonBin "C:\\Users\\pc\\miniconda3\\envs\\cerp\\python.exe" '
                    '-ControlUrl "http://10.112.81.135:60049"'
                ),
                THIRD_REPO,
            ),
            "client8g": _register_queue_task(
                hosts["client8g"],
                f"GGEUR-{args.run_id}-clients",
                client_script,
                (
                    f'-RunId "{args.run_id}" '
                    '-ControlUrl "http://10.112.81.135:60049"'
                ),
                CLIENT_REPO,
            ),
        }
    finally:
        close_hosts(hosts)
    print(json.dumps(result, ensure_ascii=True, indent=2))


def chain(args):
    run_ids = [
        item.strip() for item in args.chain_run_ids.split(",")
        if item.strip()
    ]
    if not run_ids:
        raise ValueError("--chain-run-ids must contain at least one run ID")
    for run_id in run_ids:
        manifest = THIS_DIR / "runs" / run_id / "matrix_manifest.json"
        if not manifest.is_file():
            raise FileNotFoundError(f"Missing local manifest: {manifest}")

    # Register the two Windows waiters even if the busy 4090 temporarily
    # refuses a new SSH banner.  A later idempotent retry adopts/restarts only
    # these dedicated chain waiters and installs the root handoff driver.
    hosts = connect_hosts(args, root_required=False)
    root_script = remote_path(
        ROOT_REPO, f"{HIER_REL}/run_chained_queue_root.sh")
    chain_csv = ",".join(run_ids)
    root_state = remote_path(
        ROOT_REPO, f"exp/hierarchical/{args.chain_id}")
    third_script = (
        f"{THIRD_REPO}\\{HIER_REL.replace('/', chr(92))}\\"
        "run_chained_queue_windows.ps1"
    )
    client_script = (
        f"{CLIENT_REPO}\\{HIER_REL.replace('/', chr(92))}\\"
        "run_chained_queue_windows.ps1"
    )
    local_root_script = THIS_DIR / "run_chained_queue_root.sh"
    local_windows_script = THIS_DIR / "run_chained_queue_windows.ps1"
    upload_file(
        hosts["third"], local_windows_script,
        remote_path(
            THIRD_REPO, f"{HIER_REL}/run_chained_queue_windows.ps1"))
    upload_file(
        hosts["client8g"], local_windows_script,
        remote_path(
            CLIENT_REPO, f"{HIER_REL}/run_chained_queue_windows.ps1"))
    root_script_validation = None
    if hosts["root4090"] is not None:
        upload_file(
            hosts["root4090"], local_root_script, root_script, newline="lf")
        root_script_validation = run(
            hosts["root4090"],
            f"chmod 755 {root_script} && bash -n {root_script}",
            timeout=60,
        )
    root_command = rf"""
set -euo pipefail
test -x {root_script}
for run_id in {chain_csv.replace(",", " ")}; do
  test -f {ROOT_REPO}/{HIER_REL}/runs/$run_id/matrix_manifest.json
done
mkdir -p {root_state}
pid_file={root_state}/root_chain.driver.pid
pid="$(cat "$pid_file" 2>/dev/null || true)"
if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
  printf 'root_chain_adopted_pid=%s\n' "$pid"
else
  nohup env \
    PREDECESSOR_RUN_ID={args.predecessor_run_id} \
    CHAIN_RUN_IDS={chain_csv} \
    CHAIN_ID={args.chain_id} \
    REPO_DIR={ROOT_REPO} \
    PYTHON_BIN=/root/.local/share/mamba/envs/GGEUR/bin/python \
    bash {root_script} \
    >{root_state}/root_chain.driver.log 2>&1 &
  pid=$!
  printf '%s\n' "$pid" >"$pid_file"
  sleep 3
  kill -0 "$pid"
  printf 'root_chain_pid=%s\n' "$pid"
fi
"""
    try:
        result = {
            "chain_id": args.chain_id,
            "predecessor_run_id": args.predecessor_run_id,
            "run_ids": run_ids,
            "root_script_validation": root_script_validation,
            "root4090": (
                run(hosts["root4090"], root_command, timeout=60)
                if hosts["root4090"] is not None else {
                    "status": -1,
                    "stdout": "",
                    "stderr": hosts["root_error"],
                }
            ),
            "third": _register_queue_task(
                hosts["third"],
                f"GGEUR-{args.chain_id}-subservers-chain",
                third_script,
                (
                    '-Role "subservers" '
                    f'-RunIds "{chain_csv}" '
                    '-ControlUrl "http://10.112.81.135:60049" '
                    '-PythonBin "C:\\Users\\pc\\miniconda3\\envs\\cerp\\python.exe" '
                    f'-ChainId "{args.chain_id}"'
                ),
                THIRD_REPO,
                execution_time_limit_days=30,
            ),
            "client8g": _register_queue_task(
                hosts["client8g"],
                f"GGEUR-{args.chain_id}-clients-chain",
                client_script,
                (
                    '-Role "clients" '
                    f'-RunIds "{chain_csv}" '
                    '-ControlUrl "http://10.112.81.135:60049" '
                    f'-ChainId "{args.chain_id}"'
                ),
                CLIENT_REPO,
                execution_time_limit_days=30,
            ),
        }
    finally:
        close_hosts(hosts)
    print(json.dumps(result, ensure_ascii=True, indent=2))


def chain_status(args):
    hosts = connect_hosts(args, root_required=False)
    root_state = remote_path(
        ROOT_REPO, f"exp/hierarchical/{args.chain_id}")
    root_script = rf"""
set -u
state={root_state}
printf 'chain_pid='
cat "$state/root_chain.driver.pid" 2>/dev/null || true
printf 'chain_alive=%s\n' "$(
  pid="$(cat "$state/root_chain.driver.pid" 2>/dev/null || true)"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null && echo YES || echo NO
)"
printf 'chain_complete=%s\n' "$([ -f "$state/root_chain.complete" ] && echo YES || echo NO)"
printf 'queue_processes=%s\n' "$(pgrep -af '[r]un_remaining_queue_root.sh' | wc -l)"
printf 'training_processes=%s\n' "$(pgrep -af '[f]ederatedscope.main' | wc -l)"
printf 'port_60049=%s\n' "$(ss -ltn | grep -Ec ':60049[[:space:]]' || true)"
printf 'port_60050=%s\n' "$(ss -ltn | grep -Ec ':60050[[:space:]]' || true)"
printf '%s\n' '-- root chain state --'
tail -30 "$state/root_chain.tsv" 2>/dev/null || true
printf '%s\n' '-- root chain driver --'
tail -30 "$state/root_chain.driver.log" 2>/dev/null || true
"""
    windows_template = r"""
$repo = '__REPO__'
$chainId = '__CHAIN_ID__'
$role = '__ROLE__'
$taskName = "GGEUR-$chainId-$role-chain"
$stateDir = Join-Path $repo "exp\hierarchical\$chainId"
$task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
$info = Get-ScheduledTaskInfo -TaskName $taskName -ErrorAction SilentlyContinue
"task=$taskName state=$($task.State) result=$($info.LastTaskResult)"
"chain_complete=$(Test-Path -LiteralPath (Join-Path $stateDir "$($role)_chain.complete"))"
Get-Content -LiteralPath (Join-Path $stateDir "$($role)_chain.tsv") `
  -Tail 30 -ErrorAction SilentlyContinue
if ($role -eq 'clients') {
  foreach ($port in @(22, 60049, 60050)) {
    $tcp = [Net.Sockets.TcpClient]::new()
    try {
      $connected = $tcp.ConnectAsync('10.112.81.135', $port).Wait(3000) -and
        $tcp.Connected
      "root_port_$port=$connected"
    } catch {
      "root_port_$port=False"
    } finally {
      $tcp.Dispose()
    }
  }
  try {
    $control = Invoke-RestMethod `
      -Uri 'http://10.112.81.135:60049/state' -TimeoutSec 5
    "root_control=$($control | ConvertTo-Json -Compress)"
  } catch {
    "root_control=UNAVAILABLE"
  }
}
"""
    try:
        result = {
            "root4090": (
                run(hosts["root4090"], root_script, timeout=60, check=False)
                if hosts["root4090"] is not None else {
                    "status": -1,
                    "stdout": "",
                    "stderr": hosts["root_error"],
                }
            ),
            "third": run(
                hosts["third"],
                encoded_powershell(
                    windows_template
                    .replace("__REPO__", THIRD_REPO)
                    .replace("__CHAIN_ID__", args.chain_id)
                    .replace("__ROLE__", "subservers")),
                timeout=60,
                check=False,
            ),
            "client8g": run(
                hosts["client8g"],
                encoded_powershell(
                    windows_template
                    .replace("__REPO__", CLIENT_REPO)
                    .replace("__CHAIN_ID__", args.chain_id)
                    .replace("__ROLE__", "clients")),
                timeout=60,
                check=False,
            ),
        }
    finally:
        close_hosts(hosts)
    print(json.dumps(result, ensure_ascii=True, indent=2))


def status(args):
    hosts = connect_hosts(args, root_required=False)
    root_run = remote_path(
        ROOT_REPO, f"{HIER_REL}/runs/{args.run_id}")
    root_script = rf"""
set -u
run={root_run}
printf 'time=%s\n' "$(date --iso-8601=seconds)"
printf 'queue=%s\n' "$(pgrep -af '[r]un_remaining_queue_root.sh' | wc -l)"
printf 'root=%s\n' "$(pgrep -af '[f]ederatedscope.main.*{args.run_id}.*root_server.yaml' | wc -l)"
printf 'root_clients=%s\n' "$(pgrep -af '[f]ederatedscope.main.*{args.run_id}.*clients_4090' | wc -l)"
printf 'listeners=%s\n' "$(ss -ltn | grep -Ec ':(60049|60050|200[0-5][0-9])[[:space:]]' || true)"
printf 'control='
tr -d '\r\n' <"$run/queue_state/control.json" 2>/dev/null || true
printf '\n'
printf 'manifest=%s\n' "$([ -f "$run/matrix_manifest.json" ] && echo YES || echo NO)"
printf 'active_guard_count=%s\n' "$(pgrep -af '[f]ederatedscope.main|[h]ierarchical_subserver.py|[r]un_remaining_queue_root.sh' | wc -l)"
printf 'complete_marker=%s\n' "$([ -e "$run/queue_state/root_queue.complete" ] && echo YES || echo NO)"
printf 'run_dir=%s\n' "$([ -d "$run" ] && echo YES || echo NO)"
printf 'state_dir=%s\n' "$([ -d "$run/queue_state" ] && echo YES || echo NO)"
printf '%s\n' '-- root queue driver tail --'
tail -30 "$run/queue_state/root_queue.driver.log" 2>/dev/null || true
printf '%s\n' '-- root queue nohup tail --'
tail -30 "$run/queue_state/root_queue.nohup.log" 2>/dev/null || true
printf '%s\n' '-- root queue state tail --'
tail -20 "$run/queue_state/root_queue.tsv" 2>/dev/null || true
printf '%s\n' '-- root launch tail --'
tail -30 "$run/queue_state/root_launch.log" 2>/dev/null || true
case_name="$(/root/.local/share/mamba/envs/GGEUR/bin/python -c 'import json; p="{root_run}/queue_state/control.json"; print(json.load(open(p)).get("case", ""))' 2>/dev/null || true)"
if [[ -n "$case_name" && "$case_name" != none ]]; then
  log="$run/$case_name/logs/root.stdout.log"
  printf 'case=%s\n' "$case_name"
  printf 'accuracy_rounds=%s\n' "$(grep -c 'Round [0-9][0-9]* MLP Test Accuracy' "$log" 2>/dev/null || true)"
  printf 'valid4=%s\n' "$(grep -c 'valid_updates=4/4' "$log" 2>/dev/null || true)"
  printf 'errors=%s\n' "$(grep -c 'ERROR' "$log" 2>/dev/null || true)"
  printf 'finished=%s\n' "$(grep -c 'Training finished after 100 rounds' "$log" 2>/dev/null || true)"
  tail -12 "$log" 2>/dev/null || true
fi
"""
    windows_template = r"""
$runId = '__RUN_ID__'
$kind = '__KIND__'
$case = '__CASE__'
$repo = '__REPO__'
$queueKind = '__QUEUE_KIND__'
$taskName = "GGEUR-$runId-$queueKind"
$roles = @(Get-CimInstance Win32_Process | Where-Object {
  $_.CommandLine -like "*$runId*" -and (
    $_.CommandLine -like '*federatedscope.main*' -or
    $_.CommandLine -like '*hierarchical_subserver.py*' -or
    $_.CommandLine -like '*run_remaining_queue_*.ps1*'
  )
})
"kind=$kind roles=$($roles.Count)"
$roles | Select-Object -First 8 | ForEach-Object {
  "pid=$($_.ProcessId) ws=$($_.WorkingSetSize) cmd=$($_.CommandLine)"
}
$conns = @(Get-NetTCPConnection -State Established -ErrorAction SilentlyContinue |
  Where-Object {
    ($_.LocalPort -ge 20000 -and $_.LocalPort -le 20059) -or
    ($_.LocalPort -ge 61000 -and $_.LocalPort -le 61003) -or
    ($_.RemotePort -ge 60050 -and $_.RemotePort -le 61003)
  })
"established=$($conns.Count)"
"real10=$(@($conns | Where-Object { $_.LocalAddress -like '10.*' -and $_.RemoteAddress -like '10.*' }).Count)"
"non10=$(@($conns | Where-Object { $_.LocalAddress -notlike '10.*' -or $_.RemoteAddress -notlike '10.*' }).Count)"
$conns | Where-Object { $_.LocalAddress -notlike '10.*' -or $_.RemoteAddress -notlike '10.*' } | ForEach-Object {
  "non10_endpoint=$($_.LocalAddress):$($_.LocalPort)->$($_.RemoteAddress):$($_.RemotePort) pid=$($_.OwningProcess)"
}
$task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
$taskInfo = Get-ScheduledTaskInfo -TaskName $taskName -ErrorAction SilentlyContinue
"task=$taskName state=$($task.State) result=$($taskInfo.LastTaskResult)"
$run = Join-Path $repo "scripts\distributed_scripts\ggeur_hierarchical_3machine\runs\$runId"
$stateFile = Join-Path $run "queue_state\$($queueKind.TrimEnd('s'))_queue.tsv"
if (-not (Test-Path -LiteralPath $stateFile)) {
  $stateFile = if ($queueKind -eq 'subservers') {
    Join-Path $run 'queue_state\subserver_queue.tsv'
  } else {
    Join-Path $run 'queue_state\client_queue.tsv'
  }
}
"state_file=$stateFile exists=$(Test-Path -LiteralPath $stateFile)"
Get-Content -LiteralPath $stateFile -Tail 12 -ErrorAction SilentlyContinue
$cmdPath = Join-Path $repo "exp\hierarchical\$taskName.run.cmd"
"cmd_file=$cmdPath"
Get-Content -LiteralPath $cmdPath -ErrorAction SilentlyContinue
"""
    try:
        root_result = (
            run(hosts["root4090"], root_script, timeout=90, check=False)
            if hosts["root4090"] is not None else {
                "status": -1,
                "stdout": "",
                "stderr": hosts["root_error"],
            }
        )
        case_name = args.case or ""
        try:
            for line in root_result["stdout"].splitlines():
                if line.startswith("case="):
                    case_name = line.split("=", 1)[1]
        except Exception:
            pass
        result = {
            "root4090": root_result,
            "third": run(
                hosts["third"],
                encoded_powershell(
                    windows_template
                    .replace("__RUN_ID__", args.run_id)
                    .replace("__KIND__", "third")
                    .replace("__CASE__", case_name)
                    .replace("__REPO__", THIRD_REPO)
                    .replace("__QUEUE_KIND__", "subservers")),
                timeout=90,
                check=False,
            ),
            "client8g": run(
                hosts["client8g"],
                encoded_powershell(
                    windows_template
                    .replace("__RUN_ID__", args.run_id)
                    .replace("__KIND__", "client8g")
                    .replace("__CASE__", case_name)
                    .replace("__REPO__", CLIENT_REPO)
                    .replace("__QUEUE_KIND__", "clients")),
                timeout=90,
                check=False,
            ),
        }
    finally:
        close_hosts(hosts)
    print(json.dumps(result, ensure_ascii=True, indent=2))


def case_probe(args):
    if not args.case:
        raise ValueError("--case is required for case-probe")
    manifest_path = THIS_DIR / "runs" / args.run_id / "matrix_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    allowed_cases = {item["case"] for item in manifest["cases"]}
    if args.case not in allowed_cases:
        raise ValueError(
            f"case {args.case!r} is not in run {args.run_id!r}")

    root_run = remote_path(
        ROOT_REPO, f"{HIER_REL}/runs/{args.run_id}")
    root_probe = rf"""
run_id='{args.run_id}'
run_dir='{root_run}'
echo '-- matching root processes --'
ps -eo pid=,ppid=,stat=,comm=,args= | awk -v token="$run_id" \
  'index($0, token) {{print}}'
echo '-- reserved listeners --'
ss -ltnpH 2>/dev/null | awk \
  '$4 ~ /:(60049|60050|200[0-5][0-9])$/ {{print}}'
echo '-- pid files --'
find "$run_dir" -maxdepth 5 -path '*/pids/*.pid' -type f \
  -exec sh -c 'printf "%s=" "$1"; cat "$1"' _ {{}} \; 2>/dev/null || true
"""
    remote_run = remote_path(
        THIRD_REPO, f"{HIER_REL}/runs/{args.run_id}")
    script = rf"""
$run = '{remote_run}'
$case = '{args.case}'
$logDir = Join-Path $run "$case\logs"
"case=$case log_dir=$logDir exists=$(Test-Path -LiteralPath $logDir)"
foreach ($id in 1..4) {{
  $needle = "subserver_$id.json"
  $count = @(Get-CimInstance Win32_Process | Where-Object {{
    $_.CommandLine -like '*hierarchical_subserver.py*' -and
    $_.CommandLine -like "*$needle*"
  }}).Count
  "subserver_$id=$count"
  foreach ($suffix in @('.log', '.stdout.log', '.stderr.log')) {{
    $path = Join-Path $logDir "subserver_$id$suffix"
    $item = Get-Item -LiteralPath $path -ErrorAction SilentlyContinue
    if ($null -ne $item) {{
      "file=$($item.Name) bytes=$($item.Length) mtime=$($item.LastWriteTime.ToString('o'))"
    }}
  }}
}}
"-- subserver_1 tail --"
Get-Content -LiteralPath (Join-Path $logDir 'subserver_1.log') -Tail 12 -ErrorAction SilentlyContinue
"-- nonempty stderr tails --"
$errors = @(Get-ChildItem -LiteralPath $logDir -Filter '*.stderr.log' -File -ErrorAction SilentlyContinue | Where-Object {{ $_.Length -gt 0 }} | Sort-Object LastWriteTime -Descending)
foreach ($errorFile in $errors) {{
  "error_file=$($errorFile.Name) bytes=$($errorFile.Length) mtime=$($errorFile.LastWriteTime.ToString('o'))"
  Get-Content -LiteralPath $errorFile.FullName -Tail 30 -ErrorAction SilentlyContinue
}}
"""
    client_run = remote_path(
        CLIENT_REPO, f"{HIER_REL}/runs/{args.run_id}")
    client_probe = rf"""
function Test-Port([string]$HostName, [int]$Port) {{
  $tcp = [Net.Sockets.TcpClient]::new()
  try {{
    return $tcp.ConnectAsync($HostName, $Port).Wait(750) -and $tcp.Connected
  }} catch {{
    return $false
  }} finally {{
    $tcp.Dispose()
  }}
}}
$open = @()
$closed = @()
$control = $null
try {{
  $control = Invoke-RestMethod -Uri 'http://10.112.81.135:60049/state' `
    -TimeoutSec 5
  "control_state=$($control | ConvertTo-Json -Compress)"
}} catch {{
  "control_state_error=$($_.Exception.Message)"
}}
foreach ($port in 20000..20059) {{
  if (Test-Port '10.112.81.135' $port) {{
    $open += $port
  }} else {{
    $closed += $port
  }}
}}
"root_client_ports_open=$($open.Count)/60"
"root_client_ports_closed=$($closed -join ',')"
"root_port_60050=$(Test-Port '10.112.81.135' 60050)"
$run = '{client_run}'
$case = '{args.case}'
$logDir = Join-Path $run "$case\logs"
$caseTasks = @(Get-ScheduledTask `
  -TaskName "GGEUR-*-{args.case}-client_*" -ErrorAction SilentlyContinue)
"client_task_count=$($caseTasks.Count)"
$taskStates = @{{}}
foreach ($task in $caseTasks) {{
  $taskInfo = Get-ScheduledTaskInfo -TaskName $task.TaskName
  $key = "$($task.State):$($taskInfo.LastTaskResult)"
  if (-not $taskStates.ContainsKey($key)) {{ $taskStates[$key] = 0 }}
  $taskStates[$key]++
}}
foreach ($key in ($taskStates.Keys | Sort-Object)) {{
  "client_task_state=$key count=$($taskStates[$key])"
}}
$clientProcesses = @(Get-CimInstance Win32_Process | Where-Object {{
  $_.CommandLine -like '*federatedscope.main*' -and
  $_.CommandLine -like "*$case*"
}})
"client_process_count=$($clientProcesses.Count)"
"-- client queue tail --"
Get-Content -LiteralPath (Join-Path $run 'queue_state\client_queue.tsv') `
  -Tail 20 -ErrorAction SilentlyContinue
"-- nonempty client stderr tails --"
$errors = @(Get-ChildItem -LiteralPath $logDir -Filter '*.stderr.log' `
  -File -ErrorAction SilentlyContinue | Where-Object {{ $_.Length -gt 0 }} |
  Sort-Object LastWriteTime -Descending)
foreach ($errorFile in ($errors | Select-Object -First 8)) {{
  "error_file=$($errorFile.Name) bytes=$($errorFile.Length) mtime=$($errorFile.LastWriteTime.ToString('o'))"
  Get-Content -LiteralPath $errorFile.FullName -Tail 18 `
    -ErrorAction SilentlyContinue
}}
"""
    hosts = connect_hosts(args, root_required=False)
    try:
        result = {
            "root4090": (
                run(hosts["root4090"], root_probe, timeout=90, check=False)
                if hosts["root4090"] is not None else {
                    "status": -1,
                    "stdout": "",
                    "stderr": hosts["root_error"],
                }
            ),
            "third": run(
                hosts["third"],
                encoded_powershell(script),
                timeout=90,
                check=False,
            ),
            "client8g": run(
                hosts["client8g"],
                encoded_powershell(client_probe),
                timeout=90,
                check=False,
            ),
        }
    finally:
        close_hosts(hosts)
    print(json.dumps(result, ensure_ascii=True, indent=2))


def client_log_probe(args):
    """Read a small set of terminal-client logs without enumerating tasks."""
    if not args.case:
        raise ValueError("--case is required for client-log-probe")
    case_dir = (
        f"{CLIENT_REPO}\\{HIER_REL.replace('/', chr(92))}\\runs\\"
        f"{args.run_id}\\{args.case}"
    )
    script = rf"""
$caseDir = '{case_dir}'
$logDir = Join-Path $caseDir 'logs'
"log_dir=$logDir exists=$(Test-Path -LiteralPath $logDir)"
foreach ($client in @('client_000001','client_000015','client_000031','client_000045','client_000060')) {{
  foreach ($suffix in @('stdout.log','stderr.log')) {{
    $path = Join-Path $logDir ($client + '.' + $suffix)
    if (Test-Path -LiteralPath $path) {{
      $item = Get-Item -LiteralPath $path
      "--- $($item.Name) bytes=$($item.Length) mtime=$($item.LastWriteTime.ToString('o')) ---"
      Get-Content -LiteralPath $path -Tail 35
    }} else {{
      "missing=$path"
    }}
  }}
}}
$processes = @(Get-CimInstance Win32_Process | Where-Object {{
  $_.CommandLine -and $_.CommandLine -like ('*' + '{args.case}' + '*') -and
  $_.CommandLine -like '*federatedscope.main*'
}})
"client_process_count=$($processes.Count)"
"""
    hosts = connect_hosts(args, root_required=False)
    try:
        result = run(hosts["client8g"], encoded_powershell(script),
                     timeout=90, check=False)
    finally:
        close_hosts(hosts)
    print(json.dumps(
        {"run_id": args.run_id, "case": args.case, "client8g": result},
        ensure_ascii=True, indent=2))


def sync_officehome_eval_cache(args):
    """Copy the four server-held OfficeHome test caches to the client host."""
    if args.group not in {"officehome_cnn", "officehome_mixer"}:
        raise ValueError(
            "--group must be officehome_cnn or officehome_mixer")
    root_dir = remote_path(
        ROOT_REPO, f"exp/distributed_feature_cache/{args.group}")
    client_dir = remote_path(
        CLIENT_REPO, f"exp/distributed_feature_cache/{args.group}")
    hosts = connect_hosts(args, root_required=True)
    copied = []
    try:
        listing = run(
            hosts["root4090"],
            f"find '{root_dir}' -maxdepth 1 -type f -name '*_test_*.npz' "
            "-printf '%f\\t%s\\n' | sort",
            timeout=90,
        )
        entries = []
        for line in listing["stdout"].splitlines():
            if not line.strip():
                continue
            name, size = line.rsplit("\t", 1)
            entries.append((name, int(size)))
        domains = ("Art", "Clipart", "Product", "Real_World")
        selected = []
        for domain in domains:
            matches = [item for item in entries
                       if f"_{domain}_test_" in item[0]
                       and "terminal_client" not in item[0]]
            if len(matches) != 1:
                raise RuntimeError(
                    f"Expected one root test cache for {domain} in "
                    f"{root_dir}, found {len(matches)}: {matches}")
            selected.append(matches[0])
        run(
            hosts["client8g"],
            encoded_powershell(
                f"New-Item -ItemType Directory -Force -Path "
                f"'{client_dir.replace('/', chr(92))}' | Out-Null"),
            timeout=90,
        )
        with tempfile.TemporaryDirectory(
                prefix=f"{args.group}_eval_cache_") as temp_dir:
            for name, expected_size in selected:
                local_path = Path(temp_dir) / name
                source = remote_path(root_dir, name)
                destination = remote_path(client_dir, name)
                download_file(hosts["root4090"], source, local_path)
                if local_path.stat().st_size != expected_size:
                    raise RuntimeError(
                        f"Downloaded size mismatch for {name}: "
                        f"{local_path.stat().st_size} != {expected_size}")
                upload_file(hosts["client8g"], local_path, destination)
                copied.append({
                    "name": name,
                    "bytes": expected_size,
                    "source": source,
                    "destination": destination,
                })
        verify_script = rf"""
$ErrorActionPreference = 'Stop'
$cacheDir = '{client_dir.replace('/', chr(92))}'
Get-ChildItem -LiteralPath $cacheDir -Filter '*_test_*.npz' -File |
  Sort-Object Name | ForEach-Object {{
    "$($_.Name)`t$($_.Length)"
  }}
"""
        verification = run(
            hosts["client8g"], encoded_powershell(verify_script), timeout=90)
    finally:
        close_hosts(hosts)
    print(json.dumps({
        "group": args.group,
        "copied": copied,
        "client_verification": verification,
    }, ensure_ascii=False, indent=2))


def resume_clients(args):
    """Relaunch only the current case's terminal clients, with captured output."""
    if not args.case:
        raise ValueError("--case is required for resume-clients")
    manifest_path = THIS_DIR / "runs" / args.run_id / "matrix_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    allowed_cases = {item["case"] for item in manifest["cases"]}
    if args.case not in allowed_cases:
        raise ValueError(
            f"case {args.case!r} is not in run {args.run_id!r}")
    case_relative = (
        f"{HIER_REL}/runs/{args.run_id}/{args.case}"
        .replace("/", "\\")
    )
    launcher = (
        f"{CLIENT_REPO}\\{HIER_REL.replace('/', chr(92))}\\"
        "launch_clients.ps1"
    )
    task_name = f"GGEUR-{args.run_id}-clients"
    script = rf"""
$ErrorActionPreference = 'Stop'
$state = Invoke-RestMethod -Uri 'http://10.112.81.135:60049/state' `
  -TimeoutSec 10
if ([string]$state.run_id -ne '{args.run_id}' -or
    [string]$state.case -ne '{args.case}' -or
    [string]$state.phase -notin @('starting', 'running')) {{
  throw "Root control state does not match requested active case: $($state | ConvertTo-Json -Compress)"
}}
foreach ($port in 61000..61003) {{
  $tcp = [Net.Sockets.TcpClient]::new()
  try {{
    $connected = $tcp.ConnectAsync('10.129.248.111', $port).Wait(1500) -and $tcp.Connected
    if (-not $connected) {{ throw "Subserver port $port is not reachable" }}
  }} finally {{
    $tcp.Dispose()
  }}
}}
& '{launcher}' `
  -CaseDir '{case_relative}' `
  -UseScheduledTasks `
  -AllowConfigSubset `
  -LaunchBatchSize 12 `
  -StartGapMilliseconds 75
if ($LASTEXITCODE -ne 0) {{ throw "launch_clients.ps1 failed: $LASTEXITCODE" }}
Start-ScheduledTask -TaskName '{task_name}'
"client_queue_restarted=$((Get-ScheduledTask -TaskName '{task_name}').State)"
"resume_clients=SUCCESS"
"""
    hosts = connect_hosts(args, root_required=False)
    try:
        result = run(
            hosts["client8g"],
            encoded_powershell(script),
            timeout=900,
            check=False,
        )
    finally:
        close_hosts(hosts)
    print(json.dumps(
        {"run_id": args.run_id, "case": args.case, "client8g": result},
        ensure_ascii=True,
        indent=2,
    ))


def resume_subservers(args):
    """Relaunch missing subservers for the active case and resume its queue."""
    if not args.case:
        raise ValueError("--case is required for resume-subservers")
    manifest_path = THIS_DIR / "runs" / args.run_id / "matrix_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    allowed_cases = {item["case"] for item in manifest["cases"]}
    if args.case not in allowed_cases:
        raise ValueError(
            f"case {args.case!r} is not in run {args.run_id!r}")
    case_relative = (
        f"{HIER_REL}/runs/{args.run_id}/{args.case}"
        .replace("/", "\\")
    )
    launcher = (
        f"{THIRD_REPO}\\{HIER_REL.replace('/', chr(92))}\\"
        "launch_subservers.ps1"
    )
    python_bin = r"C:\Users\pc\miniconda3\envs\cerp\python.exe"
    queue_task = f"GGEUR-{args.run_id}-subservers"
    script = rf"""
$ErrorActionPreference = 'Stop'
$state = Invoke-RestMethod -Uri 'http://10.112.81.135:60049/state' `
  -TimeoutSec 10
if ([string]$state.run_id -ne '{args.run_id}' -or
    [string]$state.case -ne '{args.case}' -or
    [string]$state.phase -notin @('starting', 'running')) {{
  throw "Root control state does not match requested active case: $($state | ConvertTo-Json -Compress)"
}}
& '{launcher}' -CaseDir '{case_relative}' `
  -PythonBin '{python_bin}' -UseScheduledTasks -StartGapMilliseconds 100
Start-ScheduledTask -TaskName '{queue_task}'
"subserver_queue_restarted=$((Get-ScheduledTask -TaskName '{queue_task}').State)"
"resume_subservers=SUCCESS"
"""
    hosts = connect_hosts(args, root_required=False)
    try:
        result = run(
            hosts["third"], encoded_powershell(script), timeout=300,
            check=False)
    finally:
        close_hosts(hosts)
    print(json.dumps(
        {"run_id": args.run_id, "case": args.case, "third": result},
        ensure_ascii=True,
        indent=2,
    ))


def restart_windows_queues(args):
    """Reload only the two Windows queue drivers without stopping workers."""
    hosts = connect_hosts(args, root_required=False)
    template = r"""
$ErrorActionPreference = 'Stop'
$taskName = 'GGEUR-__RUN_ID__-__ROLE__'
$task = Get-ScheduledTask -TaskName $taskName -ErrorAction Stop
Stop-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
Start-Sleep -Seconds 1
Start-ScheduledTask -TaskName $taskName
"queue=$taskName state=$((Get-ScheduledTask -TaskName $taskName).State)"
"""
    try:
        results = {
            "third": run(
                hosts["third"], encoded_powershell(
                    template.replace("__RUN_ID__", args.run_id)
                    .replace("__ROLE__", "subservers")), timeout=60,
                check=False),
            "client8g": run(
                hosts["client8g"], encoded_powershell(
                    template.replace("__RUN_ID__", args.run_id)
                    .replace("__ROLE__", "clients")), timeout=60,
                check=False),
        }
    finally:
        close_hosts(hosts)
    print(json.dumps(results, ensure_ascii=True, indent=2))


def stop(args):
    """Stop only processes and scheduled tasks belonging to one run ID."""
    # Keep Windows cleanup available even when the Linux root is temporarily
    # unreachable (for example, when an evaluation process exhausts its RAM).
    hosts = connect_hosts(args, root_required=False)
    root_run = remote_path(ROOT_REPO, f"{HIER_REL}/runs/{args.run_id}")
    root_script = rf"""
set -euo pipefail
run_id='{args.run_id}'
run_dir='{root_run}'
control_run="$(curl -fsS http://127.0.0.1:60049/state 2>/dev/null |
  python3 -c 'import json,sys; print(json.load(sys.stdin).get("run_id", ""))' 2>/dev/null || true)"
if [[ -n "$control_run" && "$control_run" != "$run_id" ]]; then
  echo "refusing: active control run is $control_run" >&2
  exit 3
fi
queue_pid="$(cat "$run_dir/queue_state/root_queue.driver.pid" 2>/dev/null || true)"
if [[ "$queue_pid" =~ ^[0-9]+$ ]]; then
  queue_args="$(ps -o args= -p "$queue_pid" 2>/dev/null || true)"
  queue_pgid="$(ps -o pgid= -p "$queue_pid" 2>/dev/null | tr -d ' ' || true)"
  current_pgid="$(ps -o pgid= -p $$ 2>/dev/null | tr -d ' ' || true)"
  if [[ "$queue_args" == *run_remaining_queue_root.sh* &&
        "$queue_pgid" =~ ^[0-9]+$ && "$queue_pgid" != "$current_pgid" ]]; then
    kill -TERM -- "-$queue_pgid" 2>/dev/null || true
    sleep 2
    kill -KILL -- "-$queue_pgid" 2>/dev/null || true
  fi
fi
pids="$({{
  cat "$run_dir/queue_state/root_queue.driver.pid" 2>/dev/null || true
  find "$run_dir" -path '*/pids/*.pid' -type f -maxdepth 5 \
    -exec cat {{}} \; 2>/dev/null || true
  ps -eo pid=,comm=,args= | awk -v token="$run_id" '
    $2 ~ /^python/ && index($0, token) {{print $1}}
  '
  ss -ltnpH 2>/dev/null | awk '
    $4 ~ /:(60049|60050|200[0-5][0-9])$/ {{
      line=$0
      while (match(line, /pid=[0-9]+/)) {{
        print substr(line, RSTART + 4, RLENGTH - 4)
        line=substr(line, RSTART + RLENGTH)
      }}
    }}
  '
}} | awk 'NF && $1 ~ /^[0-9]+$/ && !seen[$1]++')"
for pid in $pids; do
  kill -TERM "$pid" 2>/dev/null || true
done
sleep 3
for pid in $pids; do
  kill -KILL "$pid" 2>/dev/null || true
done
control_pids="$(ps -eo pid=,comm=,args= | awk -v state="$run_dir/queue_state/control.json" '
  $2 ~ /^python/ && index($0, "queue_control.py serve") &&
    index($0, state) {{print $1}}
')"
for pid in $control_pids; do
  kill -KILL "$pid" 2>/dev/null || true
done
sleep 1
if ss -ltnH 2>/dev/null | awk '$4 ~ /:60049$/ {{found=1}} END {{exit !found}}'; then
  echo "control_port_still_in_use=60049" >&2
  exit 4
fi
echo "stopped_root_pids=$(echo "$pids" | awk 'NF' | wc -l)"
"""
    windows_template = r"""
$runId = '__RUN_ID__'
$queueTask = 'GGEUR-' + $runId + '-__QUEUE__'
Stop-ScheduledTask -TaskName $queueTask -ErrorAction SilentlyContinue
# Per-client tasks use unique run/case names and never repeat.  Enumerating and
# unregistering hundreds of completed tasks here can take several minutes on
# Windows, so keep that maintenance outside the training critical path.  Any
# still-running client process is terminated by the exact run-id filter below.
$processes = @(Get-CimInstance Win32_Process | Where-Object {
  $_.CommandLine -and $_.CommandLine -like ('*' + $runId + '*') -and
  ($_.CommandLine -like '*federatedscope.main*' -or
   $_.CommandLine -like '*hierarchical_subserver.py*' -or
   $_.CommandLine -like '*run_remaining_queue_*')
})
foreach ($process in $processes) {
  Stop-Process -Id $process.ProcessId -Force -ErrorAction SilentlyContinue
}
"queue_task_stopped=true stopped_processes=$($processes.Count)"
"""
    try:
        root = hosts.get("root4090")
        result = {
            "root4090": (
                run(root, root_script, timeout=90, check=False)
                if root is not None
                else {
                    "status": -1,
                    "stdout": "",
                    "stderr": "root4090 is unreachable; Windows cleanup continued",
                }
            ),
            "third": run(
                hosts["third"],
                encoded_powershell(
                    windows_template
                    .replace("__RUN_ID__", args.run_id)
                    .replace("__QUEUE__", "subservers")),
                timeout=300,
                check=False,
            ),
            "client8g": run(
                hosts["client8g"],
                encoded_powershell(
                    windows_template
                    .replace("__RUN_ID__", args.run_id)
                    .replace("__QUEUE__", "clients")),
                timeout=300,
                check=False,
            ),
        }
    finally:
        close_hosts(hosts)
    print(json.dumps(result, ensure_ascii=True, indent=2))


def archive(args):
    if not args.case:
        raise ValueError("--case is required for archive")
    manifest_path = THIS_DIR / "runs" / args.run_id / "matrix_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    allowed_cases = {item["case"] for item in manifest["cases"]}
    if args.case not in allowed_cases:
        raise ValueError(
            f"case {args.case!r} is not in run {args.run_id!r}")

    destination_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else REPO_ROOT / "docs" / "test_logs" / args.run_id
    )
    destination = destination_root / args.case
    remote_case = remote_path(
        ROOT_REPO, f"{HIER_REL}/runs/{args.run_id}/{args.case}")
    files = {
        "logs/root.stdout.log": "root.stdout.log",
        "accuracy_summary.json": "accuracy_summary.json",
        "completion_validation.json": "completion_validation.json",
    }
    hosts = connect_hosts(args, root_required=True)
    archived = {}
    try:
        for remote_relative, local_name in files.items():
            remote_file = remote_path(remote_case, remote_relative)
            local_file = destination / local_name
            download_file(hosts["root4090"], remote_file, local_file)
            archived[local_name] = {
                "remote_path": remote_file,
                "bytes": local_file.stat().st_size,
            }
    finally:
        close_hosts(hosts)
    print(json.dumps(
        {
            "run_id": args.run_id,
            "case": args.case,
            "destination": str(destination),
            "files": archived,
        },
        ensure_ascii=False,
        indent=2,
    ))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=[
            "preflight", "deploy", "start", "chain", "chain-status",
            "status", "case-probe", "resume-clients", "resume-subservers",
            "client-log-probe", "sync-officehome-eval-cache",
            "restart-windows-queues", "stop", "archive"])
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--case")
    parser.add_argument("--output-root")
    parser.add_argument("--group", default="officehome_cnn")
    parser.add_argument(
        "--predecessor-run-id",
        default="mdsent_reference_distributed_20260730_v1")
    parser.add_argument(
        "--chain-run-ids",
        default=(
            "mdsent_baselines_corrected_20260731_v1,"
            "domainnet_remaining_20260731_v1"))
    parser.add_argument("--chain-id", default="remaining_20260731_v1")
    parser.add_argument(
        "--key", default=os.path.expanduser(r"~\.ssh\id_ed25519"))
    parser.add_argument(
        "--third-password", default=os.environ.get("GGEUR_THIRD_PASSWORD"))
    parser.add_argument("--root-banner-timeout", type=int, default=60)
    args = parser.parse_args()
    started = time.time()
    try:
        {
            "preflight": preflight,
            "deploy": deploy,
            "start": start,
            "chain": chain,
            "chain-status": chain_status,
            "status": status,
            "case-probe": case_probe,
            "client-log-probe": client_log_probe,
            "sync-officehome-eval-cache": sync_officehome_eval_cache,
            "resume-clients": resume_clients,
            "resume-subservers": resume_subservers,
            "restart-windows-queues": restart_windows_queues,
            "stop": stop,
            "archive": archive,
        }[args.action](args)
    finally:
        elapsed = time.time() - started
        print(f"elapsed_seconds={elapsed:.3f}", file=sys.stderr)


if __name__ == "__main__":
    main()
