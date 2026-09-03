#!/usr/bin/env python3
"""Manage the two-host deployment of the three-layer GGEUR hierarchy.

The physical hosts are the 8G Windows machine and the third Windows machine.
Management enters only through the 8G SSH endpoint.  The 4090 host is never
contacted by this helper.
"""

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
CONCURRENT_DIR = THIS_DIR.parent / "ggeur_concurrent_availability"
sys.path.insert(0, str(CONCURRENT_DIR))

from manage_three_machine import (  # noqa: E402
    CLIENT_BIND_IPV6,
    CLIENT_IPV6,
    CLIENT_REPO,
    THIRD_REPO,
    encoded_powershell,
    jump_client,
    key_client,
    run,
    upload_file,
    download_file,
)


def connect_dual(args):
    client8g = key_client(
        CLIENT_IPV6,
        22,
        "fsuser",
        args.key,
        bind=CLIENT_BIND_IPV6,
    )
    try:
        third = jump_client(
            client8g,
            "10.129.248.111",
            22,
            "pc",
            key_path=args.key,
        )
    except Exception:
        if not args.third_password:
            client8g.close()
            raise
        third = jump_client(
            client8g,
            "10.129.248.111",
            22,
            "pc",
            password=args.third_password,
        )
    return {"client8g": client8g, "third": third}


def close_dual(hosts):
    for name in ("third", "client8g"):
        client = hosts.get(name)
        if client is not None:
            client.close()


def probe(args):
    template = r"""
$ErrorActionPreference = 'Continue'
$repo = '__REPO__'
$computer = Get-CimInstance Win32_ComputerSystem
$os = Get-CimInstance Win32_OperatingSystem
"computer=$env:COMPUTERNAME"
"repo=$repo exists=$(Test-Path -LiteralPath $repo)"
"logical_processors=$($computer.NumberOfLogicalProcessors)"
"memory_total_gib=$([math]::Round($computer.TotalPhysicalMemory / 1GB, 2))"
"memory_free_gib=$([math]::Round($os.FreePhysicalMemory * 1KB / 1GB, 2))"
"open_files=$((Get-Process | Measure-Object HandleCount -Sum).Sum)"
foreach ($path in @(
  '.venv_client_cpu\pyvenv.cfg',
  'data\sentiment',
  'data\digit_three_domain',
  'data\DomainNet',
  'exp\distributed_manifests\domainnet_4domains\domainnet_manifest.json',
  'pretrained_models\nlptown_bert_base_multilingual_uncased_senti',
  'pretrained_models\ViT-B-16.pt',
  'pretrained_models\mixer_b16_224_complete.pth'
)) {
  $full = Join-Path $repo $path
  $files = @(Get-ChildItem -LiteralPath $full -Recurse -File -ErrorAction SilentlyContinue)
  $bytes = ($files | Measure-Object Length -Sum).Sum
  "resource=$path exists=$(Test-Path -LiteralPath $full) files=$($files.Count) bytes=$bytes"
}
foreach ($group in @(
  'mdsent_rnn', 'mdsent_lstm',
  'digit3_vit',
  'domainnet_vit', 'domainnet_cnn', 'domainnet_mixer'
)) {
  $marker = Join-Path $repo "exp\distributed_feature_cache\$group\.ggeur_feature_cache_ready.json"
  if (Test-Path -LiteralPath $marker) {
    $item = Get-Item -LiteralPath $marker
    "cache=$group ready=true bytes=$($item.Length) mtime=$($item.LastWriteTime.ToString('o'))"
    Get-Content -LiteralPath $marker -Raw
    Get-ChildItem -LiteralPath (Split-Path -Parent $marker) -File |
      Sort-Object Name | ForEach-Object {
        "cache_file=$group/$($_.Name) bytes=$($_.Length)"
      }
  } else {
    "cache=$group ready=false"
  }
}
$roles = @(Get-CimInstance Win32_Process | Where-Object {
  $_.CommandLine -like '*ggeur_hierarchical_3machine*' -or
  $_.CommandLine -like '*federatedscope.main*'
})
"roles=$($roles.Count)"
$roles | Select-Object ProcessId, WorkingSetSize, CommandLine | ConvertTo-Json -Compress
foreach ($port in @(60049, 60050, 61000, 61001, 61002, 61003)) {
  $listeners = @(Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue)
  "listen_$port=$($listeners.Count)"
}
"""
    hosts = connect_dual(args)
    try:
        result = {}
        for name, repo in (("client8g", CLIENT_REPO), ("third", THIRD_REPO)):
            script = template.replace("__REPO__", repo)
            result[name] = run(
                hosts[name], encoded_powershell(script), timeout=180,
                check=False)
    finally:
        close_dual(hosts)
    print(json.dumps(result, ensure_ascii=True, indent=2))


def audit_forbidden_legacy(args):
    """Report obsolete roles that still reference the forbidden 4090 host."""
    template = r'''
$ErrorActionPreference = 'Continue'
$forbidden = '10.112.81.135'
$legacyRuns = @(
  'mdsent_reference_distributed_20260730_v1',
  'mdsent_baselines_corrected_20260731_v1',
  'domainnet_remaining_20260731_v1',
  'remaining_20260731_v1'
)
$tasks = @()
foreach ($task in @(Get-ScheduledTask -ErrorAction SilentlyContinue)) {
  $definition = (($task.Actions | ForEach-Object {
    "$($_.Execute) $($_.Arguments) $($_.WorkingDirectory)"
  }) -join ' ')
  $matched = $definition -like "*$forbidden*"
  foreach ($run in $legacyRuns) {
    if ($definition -like "*$run*") { $matched = $true }
  }
  if ($matched) { $tasks += $task }
}
"forbidden_tasks=$($tasks.Count)"
foreach ($task in $tasks | Sort-Object TaskName) {
  $info = Get-ScheduledTaskInfo -TaskName $task.TaskName
  $arguments = (($task.Actions | ForEach-Object { $_.Arguments }) -join ' ')
  "task=$($task.TaskName) state=$($task.State) result=$($info.LastTaskResult) arguments=$arguments"
}
$processes = @(Get-CimInstance Win32_Process | Where-Object {
  $command = [string]$_.CommandLine
  $matched = $command -like "*$forbidden*"
  foreach ($run in $legacyRuns) {
    if ($command -like "*$run*") { $matched = $true }
  }
  $matched
})
"forbidden_processes=$($processes.Count)"
foreach ($process in $processes | Sort-Object ProcessId) {
  "pid=$($process.ProcessId) parent=$($process.ParentProcessId) name=$($process.Name) command=$($process.CommandLine)"
}
$connections = @(Get-NetTCPConnection -ErrorAction SilentlyContinue |
  Where-Object { $_.RemoteAddress -eq $forbidden })
"forbidden_connections=$($connections.Count)"
foreach ($connection in $connections | Sort-Object OwningProcess) {
  "connection_pid=$($connection.OwningProcess) state=$($connection.State) local=$($connection.LocalAddress):$($connection.LocalPort) remote=$($connection.RemoteAddress):$($connection.RemotePort)"
}
'''
    hosts = connect_dual(args)
    result = {}
    try:
        for name in ("client8g", "third"):
            result[name] = run(
                hosts[name], encoded_powershell(template),
                timeout=300, check=False)
    finally:
        close_dual(hosts)
    print(json.dumps(result, ensure_ascii=True, indent=2))


def quarantine_forbidden_legacy(args):
    """Preserve and stop only obsolete roles tied to the forbidden 4090."""
    template = r'''
$ErrorActionPreference = 'Stop'
$repo = '__REPO__'
$forbidden = '10.112.81.135'
$legacyRuns = @(
  'mdsent_reference_distributed_20260730_v1',
  'mdsent_baselines_corrected_20260731_v1',
  'domainnet_remaining_20260731_v1',
  'remaining_20260731_v1'
)
function Test-Legacy([string]$command) {
  if ($command -like "*$forbidden*") { return $true }
  foreach ($run in $legacyRuns) {
    if ($command -like "*$run*") { return $true }
  }
  return $false
}
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$audit = Join-Path $repo "exp\dual_topology_quarantine\$stamp"
New-Item -ItemType Directory -Force -Path $audit | Out-Null
$tasks = @()
foreach ($task in @(Get-ScheduledTask -ErrorAction SilentlyContinue)) {
  $definition = (($task.Actions | ForEach-Object {
    "$($_.Execute) $($_.Arguments) $($_.WorkingDirectory)"
  }) -join ' ')
  if (Test-Legacy $definition) { $tasks += $task }
}
$taskEvidence = foreach ($task in $tasks | Sort-Object TaskName) {
  $info = Get-ScheduledTaskInfo -TaskName $task.TaskName
  $arguments = (($task.Actions | ForEach-Object { $_.Arguments }) -join ' ')
  "$($task.TaskName)`t$($task.State)`t$($info.LastTaskResult)`t$arguments"
}
$taskEvidence | Set-Content -LiteralPath (Join-Path $audit 'tasks.tsv') -Encoding UTF8
$processes = @(Get-CimInstance Win32_Process | Where-Object {
  Test-Legacy ([string]$_.CommandLine)
})
$processEvidence = foreach ($process in $processes | Sort-Object ProcessId) {
  "$($process.ProcessId)`t$($process.ParentProcessId)`t$($process.Name)`t$($process.CommandLine)"
}
$processEvidence | Set-Content -LiteralPath (Join-Path $audit 'processes.tsv') -Encoding UTF8
$connections = @(Get-NetTCPConnection -ErrorAction SilentlyContinue |
  Where-Object { $_.RemoteAddress -eq $forbidden })
$connectionEvidence = foreach ($connection in $connections | Sort-Object OwningProcess) {
  "$($connection.OwningProcess)`t$($connection.State)`t$($connection.LocalAddress):$($connection.LocalPort)`t$($connection.RemoteAddress):$($connection.RemotePort)"
}
$connectionEvidence | Set-Content -LiteralPath (Join-Path $audit 'connections.tsv') -Encoding UTF8
foreach ($task in $tasks) {
  if ($task.State -eq 'Running') {
    Stop-ScheduledTask -TaskName $task.TaskName -ErrorAction SilentlyContinue
  }
  Disable-ScheduledTask -TaskName $task.TaskName -ErrorAction SilentlyContinue | Out-Null
}
Start-Sleep -Seconds 2
$stopped = 0
foreach ($process in @(Get-CimInstance Win32_Process | Where-Object {
  Test-Legacy ([string]$_.CommandLine)
}) | Sort-Object ProcessId -Descending) {
  Stop-Process -Id $process.ProcessId -Force -ErrorAction SilentlyContinue
  $stopped += 1
}
Start-Sleep -Seconds 3
$remainingProcesses = @(Get-CimInstance Win32_Process | Where-Object {
  Test-Legacy ([string]$_.CommandLine)
})
$remainingConnections = @(Get-NetTCPConnection -ErrorAction SilentlyContinue |
  Where-Object { $_.RemoteAddress -eq $forbidden })
"audit=$audit"
"quarantined_tasks=$($tasks.Count)"
"stopped_process_attempts=$stopped"
"remaining_legacy_processes=$($remainingProcesses.Count)"
"remaining_forbidden_connections=$($remainingConnections.Count)"
foreach ($taskName in @(
  'GGEUR-domainnet-resources-download-third',
  'GGEUR-domainnet-resource-handoff-third',
  'GGEUR-domainnet-eval-cache-third'
)) {
  $task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
  if ($task) {
    $info = Get-ScheduledTaskInfo -TaskName $taskName
    "protected_task=$taskName state=$($task.State) result=$($info.LastTaskResult)"
  } else {
    "protected_task=$taskName state=MISSING"
  }
}
if ($remainingProcesses.Count -ne 0 -or $remainingConnections.Count -ne 0) {
  throw "legacy topology quarantine incomplete"
}
'''
    hosts = connect_dual(args)
    result = {}
    try:
        for name, repo in (("client8g", CLIENT_REPO),
                           ("third", THIRD_REPO)):
            result[name] = run(
                hosts[name], encoded_powershell(
                    template.replace("__REPO__", repo)),
                timeout=600, check=False)
    finally:
        close_dual(hosts)
    print(json.dumps(result, ensure_ascii=True, indent=2))


def quarantine_forbidden_legacy_from_file(args):
    """Run the topology quarantine from a parsed on-disk script."""
    local_script = THIS_DIR / "quarantine_forbidden_legacy.ps1"
    if not local_script.is_file():
        raise FileNotFoundError(local_script)
    remote_script = (
        f"{THIRD_REPO}/scripts/distributed_scripts/"
        "ggeur_hierarchical_3machine/quarantine_forbidden_legacy.ps1"
    ).replace("\\", "/")
    client_script = (
        f"{CLIENT_REPO}/scripts/distributed_scripts/"
        "ggeur_hierarchical_3machine/quarantine_forbidden_legacy.ps1"
    ).replace("\\", "/")
    hosts = connect_dual(args)
    result = {}
    try:
        for name, repo, remote in (
            ("client8g", CLIENT_REPO, client_script),
            ("third", THIRD_REPO, remote_script),
        ):
            prepare = rf'''
$repo = '{repo}'
$directory = Join-Path $repo `
  'scripts\distributed_scripts\ggeur_hierarchical_3machine'
New-Item -ItemType Directory -Force -Path $directory | Out-Null
'''
            run(hosts[name], encoded_powershell(prepare), timeout=120)
            upload_file(hosts[name], local_script, remote)
            execute = rf'''
$ErrorActionPreference = 'Stop'
& '{remote.replace('/', chr(92))}' -RepoDir '{repo}'
'''
            result[name] = run(
                hosts[name], encoded_powershell(execute),
                timeout=600, check=False)
    finally:
        close_dual(hosts)
    print(json.dumps(result, ensure_ascii=True, indent=2))


def deploy(args):
    bundle = Path(args.bundle).resolve()
    if not bundle.is_file():
        raise FileNotFoundError(bundle)
    hosts = connect_dual(args)
    result = {}
    try:
        for name, repo in (("client8g", CLIENT_REPO),
                           ("third", THIRD_REPO)):
            setup = rf"""
$repo = '{repo}'
New-Item -ItemType Directory -Force -Path `
  (Join-Path $repo 'exp\dual_deploy') | Out-Null
"""
            run(hosts[name], encoded_powershell(setup))
            remote_bundle = (
                f"{repo}/exp/dual_deploy/{bundle.name}".replace("\\", "/"))
            upload_file(hosts[name], bundle, remote_bundle)
            extract = rf"""
$ErrorActionPreference = 'Stop'
$repo = '{repo}'
$bundle = '{remote_bundle.replace('/', chr(92))}'
& tar.exe -xf $bundle -C $repo
if ($LASTEXITCODE -ne 0) {{ throw "tar extraction failed: $LASTEXITCODE" }}
"deployed=$repo bundle=$bundle"
"""
            result[name] = run(
                hosts[name], encoded_powershell(extract), timeout=300)
    finally:
        close_dual(hosts)
    print(json.dumps(result, ensure_ascii=True, indent=2))


def sync_mdsent(args):
    """Copy only MDSent data/caches from 8G to the third host."""
    transfer = Path(args.transfer_file).resolve()
    transfer.parent.mkdir(parents=True, exist_ok=True)
    remote_name = "dual_mdsent_resources.tar"
    remote8 = f"{CLIENT_REPO}/exp/dual_deploy/{remote_name}".replace(
        "\\", "/")
    remote3 = f"{THIRD_REPO}/exp/dual_deploy/{remote_name}".replace(
        "\\", "/")
    hosts = connect_dual(args)
    result = {}
    try:
        create = rf"""
$ErrorActionPreference = 'Stop'
$repo = '{CLIENT_REPO}'
$archive = '{remote8.replace('/', chr(92))}'
New-Item -ItemType Directory -Force -Path (Split-Path $archive) | Out-Null
& tar.exe -cf $archive -C $repo `
  'data\sentiment' `
  'exp\distributed_feature_cache\mdsent_rnn' `
  'exp\distributed_feature_cache\mdsent_lstm'
if ($LASTEXITCODE -ne 0) {{ throw "tar creation failed: $LASTEXITCODE" }}
$item = Get-Item -LiteralPath $archive
"archive=$archive bytes=$($item.Length)"
"""
        result["create"] = run(
            hosts["client8g"], encoded_powershell(create), timeout=900)
        download_file(hosts["client8g"], remote8, transfer)
        prepare3 = rf"""
New-Item -ItemType Directory -Force -Path `
  '{THIRD_REPO}\exp\dual_deploy' | Out-Null
"""
        run(hosts["third"], encoded_powershell(prepare3))
        upload_file(hosts["third"], transfer, remote3)
        extract = rf"""
$ErrorActionPreference = 'Stop'
$repo = '{THIRD_REPO}'
$archive = '{remote3.replace('/', chr(92))}'
& tar.exe -xf $archive -C $repo
if ($LASTEXITCODE -ne 0) {{ throw "tar extraction failed: $LASTEXITCODE" }}
foreach ($group in @('mdsent_rnn', 'mdsent_lstm')) {{
  $marker = Join-Path $repo `
    "exp\distributed_feature_cache\$group\.ggeur_feature_cache_ready.json"
  $state = Get-Content -LiteralPath $marker -Raw | ConvertFrom-Json
  $state.validated_on = $env:COMPUTERNAME
  $state.require_complete_feature_cache = $true
  $state | ConvertTo-Json -Depth 20 | `
    Set-Content -LiteralPath $marker -Encoding UTF8
}}
$sentiment = @(Get-ChildItem -LiteralPath `
  (Join-Path $repo 'data\sentiment') -Recurse -File)
"sentiment_files=$($sentiment.Count)"
foreach ($group in @('mdsent_rnn', 'mdsent_lstm')) {{
  $dir = Join-Path $repo "exp\distributed_feature_cache\$group"
  $files = @(Get-ChildItem -LiteralPath $dir -File)
  "cache=$group files=$($files.Count) bytes=$(($files | Measure-Object Length -Sum).Sum)"
}}
"""
        result["extract"] = run(
            hosts["third"], encoded_powershell(extract), timeout=900)
    finally:
        close_dual(hosts)
    print(json.dumps(result, ensure_ascii=True, indent=2))


def transition(args):
    """Archive process evidence, then stop only stale formal GGEUR roles."""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    template = r"""
$ErrorActionPreference = 'Continue'
$repo = '__REPO__'
$archive = Join-Path $repo 'exp\dual_transition_archive\__STAMP__'
New-Item -ItemType Directory -Force -Path $archive | Out-Null
$roles = @(Get-CimInstance Win32_Process | Where-Object {
  $_.CommandLine -like '*federatedscope.main*' -or
  $_.CommandLine -like '*hierarchical_subserver.py*' -or
  $_.CommandLine -like '*run_remaining_queue*' -or
  $_.CommandLine -like '*run_chained_queue*'
})
$roles | Select-Object ProcessId,ParentProcessId,CreationDate,
  ExecutablePath,CommandLine | ConvertTo-Json -Depth 4 |
  Set-Content -LiteralPath (Join-Path $archive 'processes.json') -Encoding UTF8
$tasks = @(Get-ScheduledTask -ErrorAction SilentlyContinue | Where-Object {
  $_.TaskName -like 'GGEUR-*'
})
$tasks | Select-Object TaskName,State,TaskPath | ConvertTo-Json -Depth 4 |
  Set-Content -LiteralPath (Join-Path $archive 'scheduled_tasks.json') -Encoding UTF8
foreach ($task in $tasks) {
  Stop-ScheduledTask -TaskName $task.TaskName -ErrorAction SilentlyContinue
  Unregister-ScheduledTask -TaskName $task.TaskName -Confirm:$false `
    -ErrorAction SilentlyContinue
}
foreach ($role in $roles) {
  Stop-Process -Id $role.ProcessId -Force -ErrorAction SilentlyContinue
}
"archive=$archive stopped_roles=$($roles.Count) removed_tasks=$($tasks.Count)"
"""
    hosts = connect_dual(args)
    result = {}
    try:
        for name, repo in (("client8g", CLIENT_REPO),
                           ("third", THIRD_REPO)):
            script = template.replace("__REPO__", repo).replace(
                "__STAMP__", stamp)
            result[name] = run(
                hosts[name], encoded_powershell(script), timeout=180)
    finally:
        close_dual(hosts)
    print(json.dumps(result, ensure_ascii=True, indent=2))


def configure_firewall(args):
    hosts = connect_dual(args)
    result = {}
    try:
        for name, peer in (("client8g", "10.129.248.111"),
                           ("third", "10.129.222.189")):
            display = f"GGEUR Dual Exact {name}"
            script = rf"""
$ErrorActionPreference = 'Stop'
$display = '{display}'
Get-NetFirewallRule -DisplayName $display -ErrorAction SilentlyContinue |
  Remove-NetFirewallRule
New-NetFirewallRule -DisplayName $display -Direction Inbound -Action Allow `
  -Protocol TCP -Profile Any -RemoteAddress '{peer}' `
  -LocalPort 60049,60050,62000-62003,30000-30119 | Out-Null
$rule = Get-NetFirewallRule -DisplayName $display
$filter = $rule | Get-NetFirewallPortFilter
$address = $rule | Get-NetFirewallAddressFilter
"rule=$($rule.DisplayName) enabled=$($rule.Enabled) action=$($rule.Action) ports=$($filter.LocalPort) remote=$($address.RemoteAddress)"
"""
            result[name] = run(
                hosts[name], encoded_powershell(script), timeout=120)
    finally:
        close_dual(hosts)
    print(json.dumps(result, ensure_ascii=True, indent=2))


def _queue_task_script(repo, run_id, host):
    script_dir = (
        f"{repo}\\scripts\\distributed_scripts\\"
        "ggeur_hierarchical_3machine")
    if host == "client8g":
        root_driver_stdout = (
            f"{repo}\\exp\\dual_queue_commands\\"
            f"GGEUR-dual-{run_id}-root.stdout.log")
        root_driver_stderr = (
            f"{repo}\\exp\\dual_queue_commands\\"
            f"GGEUR-dual-{run_id}-root.stderr.log")
        return rf"""
$ErrorActionPreference = 'Stop'
$repo = '{repo}'
$scriptDir = '{script_dir}'
$venv = Join-Path $repo '.venv_client_cpu'
$homeLine = Get-Content -LiteralPath (Join-Path $venv 'pyvenv.cfg') |
  Where-Object {{ $_ -match '^\s*home\s*=\s*(.+)\s*$' }} |
  Select-Object -First 1
$pythonHome = ([regex]::Match(
  $homeLine, '^\s*home\s*=\s*(.+)\s*$')).Groups[1].Value.Trim()
$python = Join-Path $pythonHome 'python.exe'
$site = Join-Path $venv 'Lib\site-packages'
$commands = [ordered]@{{
  'GGEUR-dual-{run_id}-root' = "& '$scriptDir\run_dual_queue_root.ps1' -RunId '{run_id}' -PythonBin '$python' -SitePackages '$site' 1>> '{root_driver_stdout}' 2>> '{root_driver_stderr}'"
  'GGEUR-dual-{run_id}-subservers-8g' = "& '$scriptDir\run_dual_queue_role.ps1' -Role subservers -RunId '{run_id}' -ConfigSet subservers_8g -PythonBin '$python' -SitePackages '$site'"
  'GGEUR-dual-{run_id}-clients-8g' = "& '$scriptDir\run_dual_queue_role.ps1' -Role clients -RunId '{run_id}' -ConfigSet clients_8g -PythonBin '$python' -SitePackages '$site'"
}}
"""
    return rf"""
$ErrorActionPreference = 'Stop'
$repo = '{repo}'
$scriptDir = '{script_dir}'
$python = 'C:\Users\pc\miniconda3\envs\cerp\python.exe'
$site = 'C:\Users\pc\miniconda3\envs\cerp\Lib\site-packages'
$commands = [ordered]@{{
  'GGEUR-dual-{run_id}-subservers-third' = "& '$scriptDir\run_dual_queue_role.ps1' -Role subservers -RunId '{run_id}' -ConfigSet subservers_third -PythonBin '$python' -SitePackages '$site'"
  'GGEUR-dual-{run_id}-clients-third' = "& '$scriptDir\run_dual_queue_role.ps1' -Role clients -RunId '{run_id}' -ConfigSet clients_third -PythonBin '$python' -SitePackages '$site' -AllowCudaClientRuntime"
}}
"""


def start(args):
    hosts = connect_dual(args)
    result = {}
    try:
        for name, repo in (("client8g", CLIENT_REPO),
                           ("third", THIRD_REPO)):
            header = _queue_task_script(repo, args.run_id, name)
            install = header + r"""
$queueDir = Join-Path $repo 'exp\dual_queue_commands'
New-Item -ItemType Directory -Force -Path $queueDir | Out-Null
$identity = [Security.Principal.WindowsIdentity]::GetCurrent().Name
foreach ($item in $commands.GetEnumerator()) {
  $taskName = [string]$item.Key
  $queueScript = Join-Path $queueDir "$taskName.ps1"
  [string]$item.Value | Set-Content -LiteralPath $queueScript -Encoding UTF8
  $old = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
  if ($old) {
    Stop-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
  }
  $action = New-ScheduledTaskAction `
    -Execute "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" `
    -Argument "-NoProfile -NonInteractive -ExecutionPolicy Bypass -File `"$queueScript`"" `
    -WorkingDirectory $repo
  $principal = New-ScheduledTaskPrincipal -UserId $identity `
    -LogonType S4U -RunLevel Limited
  $settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Days 14) `
    -MultipleInstances IgnoreNew -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries
  Register-ScheduledTask -TaskName $taskName -Action $action `
    -Principal $principal -Settings $settings -Force | Out-Null
  Start-ScheduledTask -TaskName $taskName
  "started=$taskName"
}
"""
            result[name] = run(
                hosts[name], encoded_powershell(install), timeout=180)
    finally:
        close_dual(hosts)
    print(json.dumps(result, ensure_ascii=True, indent=2))


def status(args):
    template = r"""
$repo = '__REPO__'
$run = '__RUN__'
$stateDir = Join-Path $repo `
  "scripts\distributed_scripts\ggeur_hierarchical_3machine\runs\$run\queue_state"
"computer=$env:COMPUTERNAME"
$tasks = @(Get-ScheduledTask -TaskName "GGEUR-dual-$run-*" `
  -ErrorAction SilentlyContinue)
$tasks | ForEach-Object {
  $info = Get-ScheduledTaskInfo -TaskName $_.TaskName
  "task=$($_.TaskName) state=$($_.State) result=$($info.LastTaskResult)"
}
$driverDir = Join-Path $repo 'exp\dual_queue_commands'
Get-ChildItem -LiteralPath $driverDir -File -ErrorAction SilentlyContinue |
  Where-Object { $_.Name -like "GGEUR-dual-$run-*.log" } |
  ForEach-Object {
    "driver_log=$($_.Name) bytes=$($_.Length)"
    Get-Content -LiteralPath $_.FullName -Tail 20 -ErrorAction SilentlyContinue |
      ForEach-Object { "driver_tail=$($_)" }
  }
if (Test-Path -LiteralPath (Join-Path $stateDir 'control.json')) {
  "control=" + (Get-Content -LiteralPath `
    (Join-Path $stateDir 'control.json') -Raw)
}
Get-ChildItem -LiteralPath $stateDir -File -ErrorAction SilentlyContinue |
  ForEach-Object {
    "state_file=$($_.Name) bytes=$($_.Length)"
    if ($_.Extension -in @('.tsv', '.log')) {
      Get-Content -LiteralPath $_.FullName -Tail 8 -ErrorAction SilentlyContinue |
        ForEach-Object { "tail=$($_)" }
    }
  }
$roles = @(Get-CimInstance Win32_Process | Where-Object {
  $_.CommandLine -like "*$run*" -and (
    $_.CommandLine -like '*federatedscope.main*' -or
    $_.CommandLine -like '*hierarchical_subserver.py*')
})
"roles=$($roles.Count)"
function Test-RemoteTcp([string]$HostName, [int]$Port) {
  $client = [Net.Sockets.TcpClient]::new()
  try {
    $task = $client.ConnectAsync($HostName, $Port)
    return $task.Wait(1500) -and $client.Connected
  } catch { return $false } finally { $client.Dispose() }
}
"peer_root_control=$(Test-RemoteTcp '10.129.222.189' 60049)"
"peer_root_training=$(Test-RemoteTcp '10.129.222.189' 60050)"
"peer_third_subserver=$(Test-RemoteTcp '10.129.248.111' 62001)"
foreach ($port in @(60049,60050,61000,61001,61002,61003,
                    62000,62001,62002,62003)) {
  $listeners = @(Get-NetTCPConnection -State Listen -LocalPort $port `
    -ErrorAction SilentlyContinue)
  "listen_$port=$($listeners.Count)"
}
"""
    hosts = connect_dual(args)
    result = {}
    try:
        for name, repo in (("client8g", CLIENT_REPO),
                           ("third", THIRD_REPO)):
            script = template.replace("__REPO__", repo).replace(
                "__RUN__", args.run_id)
            result[name] = run(
                hosts[name], encoded_powershell(script), timeout=180,
                check=False)
    finally:
        close_dual(hosts)
    print(json.dumps(result, ensure_ascii=True, indent=2))


def _decode_remote_text(data):
    for encoding in ("utf-8-sig", "utf-16", "gb18030"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", "replace")


def inspect_run(args):
    """Read queue and case logs over SFTP without executing remote code."""
    hosts = connect_dual(args)
    result = {}
    try:
        for name, repo in (("client8g", CLIENT_REPO),
                           ("third", THIRD_REPO)):
            sftp = hosts[name].open_sftp()
            try:
                run_root = repo + (
                    r"\scripts\distributed_scripts"
                    r"\ggeur_hierarchical_3machine\runs" + "\\" +
                    args.run_id)
                case_root = run_root + "\\" + args.case_name
                state_root = run_root + r"\queue_state"
                logs_root = case_root + r"\logs"
                entry = {"state": {}, "logs": {}}
                for remote_root, bucket in ((state_root, "state"),
                                            (logs_root, "logs")):
                    try:
                        attrs = sftp.listdir_attr(remote_root)
                    except OSError:
                        continue
                    if bucket == "logs":
                        entry["log_counts"] = {
                            "client_stdout": sum(
                                attr.filename.startswith("client_") and
                                attr.filename.endswith(".stdout.log")
                                for attr in attrs),
                            "client_stderr": sum(
                                attr.filename.startswith("client_") and
                                attr.filename.endswith(".stderr.log")
                                for attr in attrs),
                            "nonempty_client_stderr": sum(
                                attr.filename.startswith("client_") and
                                attr.filename.endswith(".stderr.log") and
                                attr.st_size > 0 for attr in attrs),
                        }
                    selected = []
                    for attr in attrs:
                        filename = attr.filename
                        if bucket == "state":
                            wanted = filename.endswith((".tsv", ".log", ".json"))
                        else:
                            wanted = (
                                filename == "root.stdout.log" or
                                filename == "root.stderr.log" or
                                (filename.startswith("subserver_") and
                                 filename.endswith(".stderr.log") and
                                 attr.st_size > 0) or
                                (filename.startswith("client_") and
                                 filename.endswith(".stderr.log") and
                                 attr.st_size > 0))
                        if wanted:
                            selected.append((filename, attr.st_size))
                    if bucket == "logs":
                        selected.sort(
                            key=lambda item: (
                                0 if item[0] == "root.stderr.log" else
                                1 if item[0] == "root.stdout.log" else
                                2 if item[0].startswith("subserver_") else 3,
                                item[0],
                            ))
                    for filename, size in selected[:12]:
                        path = remote_root + "\\" + filename
                        try:
                            with sftp.file(path, "rb") as handle:
                                data = handle.read()
                            lines = _decode_remote_text(data).splitlines()
                            entry[bucket][filename] = {
                                "bytes": size,
                                "tail": lines[-30:],
                            }
                        except OSError as error:
                            entry[bucket][filename] = {
                                "bytes": size,
                                "read_error": str(error),
                            }
                result[name] = entry
            finally:
                sftp.close()
    finally:
        close_dual(hosts)
    print(json.dumps(result, ensure_ascii=True, indent=2))


def inspect_root(args):
    """Read only the current root logs so failures are never output-truncated."""
    hosts = connect_dual(args)
    result = {}
    try:
        sftp = hosts["client8g"].open_sftp()
        try:
            logs_root = CLIENT_REPO + (
                r"\scripts\distributed_scripts"
                r"\ggeur_hierarchical_3machine\runs" + "\\" +
                args.run_id + "\\" + args.case_name + r"\logs")
            for filename in ("root.stderr.log", "root.stdout.log"):
                path = logs_root + "\\" + filename
                try:
                    with sftp.file(path, "rb") as handle:
                        data = handle.read()
                    lines = _decode_remote_text(data).splitlines()
                    result[filename] = {
                        "bytes": len(data),
                        "tail": lines[-120:],
                    }
                except OSError as error:
                    result[filename] = {"read_error": str(error)}
        finally:
            sftp.close()
    finally:
        close_dual(hosts)
    print(json.dumps(result, ensure_ascii=True, indent=2))


def domainnet_cache_status(args):
    """Audit DomainNet evaluation caches and manifest-backed raw paths."""
    cache_code = r'''import glob
import os
import numpy as np
import open_clip
import timm
import torch
import torchvision
from torchvision.models import ConvNeXt_Base_Weights
root = os.environ['GGEUR_CACHE_AUDIT_REPO']
print(f'python_torch={torch.__version__} torchvision={torchvision.__version__} timm={timm.__version__}')
open_clip_version = getattr(open_clip, '__version__', 'unknown')
open_clip_cfg = open_clip.get_pretrained_cfg('ViT-B-16', 'openai')
print(f'open_clip={open_clip_version} vit_b16_openai={open_clip_cfg}')
print(f'convnext_base_weights_url={ConvNeXt_Base_Weights.IMAGENET1K_V1.url}')
try:
    model = timm.create_model('mixer_b16_224', pretrained=False)
    print(f'mixer_default_cfg={model.default_cfg}')
    del model
except Exception as error:
    print(f'mixer_default_cfg_error={type(error).__name__}:{error}')
for group in ('domainnet_vit', 'domainnet_cnn', 'domainnet_mixer'):
    pattern = os.path.join(root, 'exp', 'distributed_feature_cache', group, '*.npz')
    for path in sorted(glob.glob(pattern)):
        try:
            with np.load(path, allow_pickle=True) as data:
                shapes = ','.join(f'{key}:{tuple(data[key].shape)}' for key in data.files)
            print(f'npz={path} shapes={shapes}')
        except Exception as error:
            print(f'npz={path} read_error={type(error).__name__}:{error}')
'''
    cache_code_b64 = base64.b64encode(
        cache_code.encode("utf-8")).decode("ascii")
    template = r'''
$ErrorActionPreference = 'Continue'
$repo = '__REPO__'
$manifest = Join-Path $repo 'exp\distributed_manifests\domainnet_4domains\domainnet_manifest.json'
"computer=$env:COMPUTERNAME"
$repoRoot = ([IO.Path]::GetPathRoot($repo) -replace '[:\\]+$','')
$repoDrive = Get-PSDrive -Name $repoRoot -ErrorAction SilentlyContinue
if ($repoDrive) {
  "repo_drive=$($repoDrive.Name) used_bytes=$($repoDrive.Used) free_bytes=$($repoDrive.Free)"
}
try {
  $gpuLines = @(& nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits 2>$null)
  "gpu_count=$($gpuLines.Count)"
  foreach ($line in $gpuLines) { "gpu=$line" }
} catch {
  "gpu_count=0"
}
foreach ($relative in @(
  'pretrained_models\\ViT-B-16.pt',
  'pretrained_models\\open_clip_vitb16.bin',
  'pretrained_models\\mixer_b16_224_complete.pth'
)) {
  $modelPath = Join-Path $repo $relative
  $modelItem = Get-Item -LiteralPath $modelPath -ErrorAction SilentlyContinue
  $modelBytes = if ($modelItem) { $modelItem.Length } else { 0 }
  "model=$modelPath exists=$(Test-Path -LiteralPath $modelPath) bytes=$modelBytes"
}
"manifest=$manifest exists=$(Test-Path -LiteralPath $manifest)"
if (Test-Path -LiteralPath $manifest) {
  $raw = Get-Content -LiteralPath $manifest -Raw
  "manifest_bytes=$([Text.Encoding]::UTF8.GetByteCount($raw))"
  $obj = $raw | ConvertFrom-Json
  foreach ($property in $obj.PSObject.Properties) {
    $value = $property.Value
    $count = if ($value -is [System.Collections.ICollection]) { $value.Count } else { -1 }
    "manifest_property=$($property.Name) type=$($value.GetType().FullName) count=$count"
  }
  "manifest_source_root=$($obj.source_root) exists=$(Test-Path -LiteralPath $obj.source_root)"
  $matches = [regex]::Matches($raw, '[^"'']+\.(jpg|jpeg|png)')
  "manifest_image_strings=$($matches.Count)"
  foreach ($match in @($matches | Select-Object -First 8)) {
    $sample = $match.Value.Replace('/', '\')
    $candidate = if ([IO.Path]::IsPathRooted($sample)) { $sample } else { Join-Path $repo $sample }
    "manifest_image=$sample repo_candidate_exists=$(Test-Path -LiteralPath $candidate) candidate=$candidate"
  }
}
$testCaches = @(Get-ChildItem -LiteralPath (Join-Path $repo 'exp') -Recurse -File -Filter 'domainnet_*_test_*.npz' -ErrorAction SilentlyContinue)
"test_cache_files=$($testCaches.Count)"
foreach ($file in $testCaches) {
  "test_cache=$($file.FullName) bytes=$($file.Length)"
}
foreach ($candidate in @(
  (Join-Path $repo 'data\DomainNet'),
  (Join-Path $repo 'data\domainnet'),
  'D:\data\DomainNet', 'D:\datasets\DomainNet',
  'C:\data\DomainNet', 'C:\datasets\DomainNet'
)) {
  $files = @(Get-ChildItem -LiteralPath $candidate -Recurse -File -ErrorAction SilentlyContinue)
  "raw_candidate=$candidate exists=$(Test-Path -LiteralPath $candidate) files=$($files.Count)"
}
'''
    shape_template = r'''
$ErrorActionPreference = 'Continue'
$repo = '__REPO__'
$python = Join-Path $repo '.venv_client_cpu\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
  $python = 'C:\Users\pc\miniconda3\envs\cerp\python.exe'
}
if (Test-Path -LiteralPath $python) {
  $cacheCode = [Text.Encoding]::UTF8.GetString(
    [Convert]::FromBase64String('__CACHE_CODE_B64__'))
  $env:GGEUR_CACHE_AUDIT_REPO = $repo
  & $python -c $cacheCode
  Remove-Item Env:\GGEUR_CACHE_AUDIT_REPO -ErrorAction SilentlyContinue
}
'''
    hosts = connect_dual(args)
    result = {}
    try:
        for name, repo in (("client8g", CLIENT_REPO), ("third", THIRD_REPO)):
            status = run(
                hosts[name],
                encoded_powershell(template.replace("__REPO__", repo)),
                timeout=300,
                check=False,
            )
            shapes = run(
                hosts[name],
                encoded_powershell(shape_template.replace(
                    "__REPO__", repo).replace(
                        "__CACHE_CODE_B64__", cache_code_b64)),
                timeout=300,
                check=False,
            )
            status["stdout"] = "\n".join(filter(None, [
                status.get("stdout", ""), shapes.get("stdout", "")]))
            status["stderr"] = "\n".join(filter(None, [
                status.get("stderr", ""), shapes.get("stderr", "")]))
            if shapes.get("status"):
                status["status"] = shapes["status"]
            result[name] = status
    finally:
        close_dual(hosts)
    print(json.dumps(result, ensure_ascii=True, indent=2))


def compare_domainnet_cnn_caches(args):
    """Compare the active and previously validated DomainNet CNN caches."""
    local_script = THIS_DIR / "compare_domainnet_cnn_caches.py"
    remote_script = (CLIENT_REPO +
                     r"\scripts\distributed_scripts"
                     r"\ggeur_hierarchical_3machine"
                     r"\compare_domainnet_cnn_caches.py")
    template = r'''
$ErrorActionPreference = 'Stop'
$repo = '__REPO__'
$python = Join-Path $repo '.venv_client_cpu\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
  $python = 'C:\Users\pc\miniconda3\envs\cerp\python.exe'
}
& $python '__SCRIPT__' --repo $repo
'''
    hosts = connect_dual(args)
    try:
        upload_file(hosts["client8g"], local_script, remote_script)
        result = run(
            hosts["client8g"],
            encoded_powershell(template.replace(
                "__REPO__", CLIENT_REPO).replace(
                    "__SCRIPT__", remote_script)),
            timeout=600,
            check=False,
        )
    finally:
        close_dual(hosts)
    print(json.dumps(result, ensure_ascii=True, indent=2))


def start_domainnet_resource_download(args):
    """Install a durable official DomainNet resource download on the third host."""
    local_script = THIS_DIR / "download_domainnet_resources.ps1"
    if not local_script.is_file():
        raise FileNotFoundError(local_script)
    transfer_manifest = (REPO_ROOT / "exp" / "dual_deploy" /
                         "domainnet_manifest_from_8g.json")
    transfer_manifest.parent.mkdir(parents=True, exist_ok=True)
    remote_manifest_8g = (
        f"{CLIENT_REPO}/exp/distributed_manifests/domainnet_4domains/"
        "domainnet_manifest.json").replace("\\", "/")
    remote_manifest_third = (
        f"{THIRD_REPO}/exp/distributed_manifests/domainnet_4domains/"
        "domainnet_manifest.json").replace("\\", "/")
    remote_script = (
        f"{THIRD_REPO}/scripts/distributed_scripts/"
        "ggeur_hierarchical_3machine/download_domainnet_resources.ps1"
    ).replace("\\", "/")
    task_name = "GGEUR-domainnet-resources-download-third"

    hosts = connect_dual(args)
    result = {}
    try:
        download_file(hosts["client8g"], remote_manifest_8g,
                      transfer_manifest)
        prepare = rf'''
$ErrorActionPreference = 'Stop'
$repo = '{THIRD_REPO}'
New-Item -ItemType Directory -Force -Path `
  (Join-Path $repo 'scripts\distributed_scripts\ggeur_hierarchical_3machine'), `
  (Join-Path $repo 'exp\distributed_manifests\domainnet_4domains') | Out-Null
'''
        result["prepare"] = run(
            hosts["third"], encoded_powershell(prepare), timeout=120)
        upload_file(hosts["third"], local_script, remote_script)
        upload_file(hosts["third"], transfer_manifest,
                    remote_manifest_third)
        install = rf'''
$ErrorActionPreference = 'Stop'
$repo = '{THIRD_REPO}'
$taskName = '{task_name}'
$script = '{remote_script.replace('/', chr(92))}'
$state = Join-Path $repo 'exp\domainnet_resource_download_dual'
New-Item -ItemType Directory -Force -Path $state | Out-Null
$driverOut = Join-Path $state 'driver.stdout.log'
$driverErr = Join-Path $state 'driver.stderr.log'
$command = "& '$script' -RepoDir '$repo' " + `
  "1>> '$driverOut' 2>> '$driverErr'"
$driver = Join-Path $state 'run_download.ps1'
$command | Set-Content -LiteralPath $driver -Encoding UTF8
$old = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($old -and $old.State -eq 'Running') {{
  "already_running=$taskName"
  exit 0
}}
if ($old) {{
  Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
}}
$identity = [Security.Principal.WindowsIdentity]::GetCurrent().Name
$action = New-ScheduledTaskAction `
  -Execute "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" `
  -Argument "-NoProfile -NonInteractive -ExecutionPolicy Bypass -File `"$driver`"" `
  -WorkingDirectory $repo
$principal = New-ScheduledTaskPrincipal -UserId $identity `
  -LogonType S4U -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet `
  -ExecutionTimeLimit (New-TimeSpan -Days 7) `
  -MultipleInstances IgnoreNew -AllowStartIfOnBatteries `
  -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName $taskName -Action $action `
  -Principal $principal -Settings $settings -Force | Out-Null
Start-ScheduledTask -TaskName $taskName
"started=$taskName state_dir=$state"
'''
        result["start"] = run(
            hosts["third"], encoded_powershell(install), timeout=180)
    finally:
        close_dual(hosts)
    print(json.dumps(result, ensure_ascii=True, indent=2))


def domainnet_resource_download_status(args):
    """Report the durable DomainNet resource download on the third host."""
    task_name = "GGEUR-domainnet-resources-download-third"
    script = rf'''
$ErrorActionPreference = 'Continue'
$repo = '{THIRD_REPO}'
$state = Join-Path $repo 'exp\domainnet_resource_download_dual'
$taskName = '{task_name}'
$task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($task) {{
  $info = Get-ScheduledTaskInfo -TaskName $taskName
  "task=$taskName state=$($task.State) result=$($info.LastTaskResult)"
}} else {{
  "task=$taskName state=MISSING"
}}
foreach ($name in @('pipeline.log', 'driver.stdout.log', 'driver.stderr.log')) {{
  $path = Join-Path $state $name
  if (Test-Path -LiteralPath $path) {{
    $item = Get-Item -LiteralPath $path
    "log=$name bytes=$($item.Length) mtime=$($item.LastWriteTime.ToString('o'))"
    Get-Content -LiteralPath $path -Tail 30
  }}
}}
$downloads = Join-Path $state 'downloads'
Get-ChildItem -LiteralPath $downloads -ErrorAction SilentlyContinue |
  Where-Object {{ -not $_.PSIsContainer }} | Sort-Object Name |
  ForEach-Object {{ "download=$($_.Name) bytes=$($_.Length)" }}
foreach ($domain in @('clipart', 'painting', 'real', 'sketch')) {{
  $dir = Join-Path $repo "data\DomainNet\$domain"
  $marker = Join-Path $state "$domain.extracted.json"
  "domain=$domain dir=$(Test-Path -LiteralPath $dir) marker=$(Test-Path -LiteralPath $marker)"
}}
"complete=$(Test-Path -LiteralPath (Join-Path $state 'completed.json'))"
'''
    hosts = connect_dual(args)
    try:
        result = run(hosts["third"], encoded_powershell(script),
                     timeout=180, check=False)
    finally:
        close_dual(hosts)
    print(json.dumps(result, ensure_ascii=True, indent=2))


def install_domainnet_resource_handoff(args):
    """Install a non-disruptive handoff to the latest resource pipeline."""
    local_resource = THIS_DIR / "download_domainnet_resources.ps1"
    local_handoff = THIS_DIR / "run_domainnet_resource_handoff.ps1"
    for path in (local_resource, local_handoff):
        if not path.is_file():
            raise FileNotFoundError(path)
    remote_dir = (
        f"{THIRD_REPO}/scripts/distributed_scripts/"
        "ggeur_hierarchical_3machine"
    ).replace("\\", "/")
    remote_resource = f"{remote_dir}/download_domainnet_resources.ps1"
    remote_handoff = f"{remote_dir}/run_domainnet_resource_handoff.ps1"
    task_name = "GGEUR-domainnet-resource-handoff-third"
    hosts = connect_dual(args)
    result = {}
    try:
        prepare = rf'''
$ErrorActionPreference = 'Stop'
$repo = '{THIRD_REPO}'
New-Item -ItemType Directory -Force -Path `
  (Join-Path $repo 'scripts\distributed_scripts\ggeur_hierarchical_3machine'), `
  (Join-Path $repo 'exp\domainnet_resource_download_dual') | Out-Null
'''
        result["prepare"] = run(
            hosts["third"], encoded_powershell(prepare), timeout=120)
        upload_file(hosts["third"], local_resource, remote_resource)
        upload_file(hosts["third"], local_handoff, remote_handoff)
        install = rf'''
$ErrorActionPreference = 'Stop'
$repo = '{THIRD_REPO}'
$taskName = '{task_name}'
$driver = '{remote_handoff.replace('/', chr(92))}'
$state = Join-Path $repo 'exp\domainnet_resource_download_dual'
$old = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($old -and $old.State -eq 'Running') {{
  "already_running=$taskName"
  exit 0
}}
if (Test-Path -LiteralPath (Join-Path $state 'handoff.completed.json')) {{
  "already_complete=$taskName"
  exit 0
}}
if ($old) {{
  $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
  $attempt = Join-Path $state "handoff_attempts\$stamp"
  New-Item -ItemType Directory -Force -Path $attempt | Out-Null
  foreach ($name in @('handoff.pipeline.log', 'handoff.stdout.log', 'handoff.stderr.log')) {{
    $path = Join-Path $state $name
    if (Test-Path -LiteralPath $path) {{
      Copy-Item -LiteralPath $path -Destination $attempt
    }}
  }}
  Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
}}
$argument = "-NoProfile -NonInteractive -ExecutionPolicy Bypass -File `"$driver`" -RepoDir `"$repo`""
$identity = [Security.Principal.WindowsIdentity]::GetCurrent().Name
$action = New-ScheduledTaskAction `
  -Execute "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" `
  -Argument $argument -WorkingDirectory $repo
$principal = New-ScheduledTaskPrincipal -UserId $identity `
  -LogonType S4U -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet `
  -ExecutionTimeLimit (New-TimeSpan -Days 7) `
  -MultipleInstances IgnoreNew -AllowStartIfOnBatteries `
  -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName $taskName -Action $action `
  -Principal $principal -Settings $settings -Force | Out-Null
Start-ScheduledTask -TaskName $taskName
"started=$taskName state_dir=$state"
'''
        result["start"] = run(
            hosts["third"], encoded_powershell(install), timeout=180)
    finally:
        close_dual(hosts)
    print(json.dumps(result, ensure_ascii=True, indent=2))


def domainnet_resource_handoff_status(args):
    """Report the latest-resource handoff task on the third host."""
    task_name = "GGEUR-domainnet-resource-handoff-third"
    script = rf'''
$ErrorActionPreference = 'Continue'
$repo = '{THIRD_REPO}'
$state = Join-Path $repo 'exp\domainnet_resource_download_dual'
$taskName = '{task_name}'
$task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($task) {{
  $info = Get-ScheduledTaskInfo -TaskName $taskName
  "task=$taskName state=$($task.State) result=$($info.LastTaskResult)"
}} else {{
  "task=$taskName state=MISSING"
}}
foreach ($name in @('handoff.pipeline.log', 'handoff.stdout.log', 'handoff.stderr.log')) {{
  $path = Join-Path $state $name
  if (Test-Path -LiteralPath $path) {{
    $item = Get-Item -LiteralPath $path
    "log=$name bytes=$($item.Length) mtime=$($item.LastWriteTime.ToString('o'))"
    Get-Content -LiteralPath $path -Tail 24
  }}
}}
"complete=$(Test-Path -LiteralPath (Join-Path $state 'handoff.completed.json'))"
'''
    hosts = connect_dual(args)
    try:
        result = run(hosts["third"], encoded_powershell(script),
                     timeout=180, check=False)
    finally:
        close_dual(hosts)
    print(json.dumps(result, ensure_ascii=True, indent=2))


def diagnose_domainnet_resource_download(args):
    """Diagnose the failed third-host downloader without changing its state."""
    task_name = "GGEUR-domainnet-resources-download-third"
    script = rf'''
$ErrorActionPreference = 'Continue'
$repo = '{THIRD_REPO}'
$state = Join-Path $repo 'exp\domainnet_resource_download_dual'
$taskName = '{task_name}'
"curl_command=$((Get-Command curl.exe -ErrorAction SilentlyContinue).Source)"
& curl.exe --version
"curl_version_exit=$LASTEXITCODE"
& curl.exe --help all 2>&1 | Select-String -SimpleMatch 'retry-all-errors'
"curl_retry_all_errors_help_exit=$LASTEXITCODE"
$driver = Join-Path $state 'run_download.ps1'
if (Test-Path -LiteralPath $driver) {{
  "driver_begin"
  Get-Content -LiteralPath $driver -Raw
  "driver_end"
}}
$task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($task) {{
  $info = Get-ScheduledTaskInfo -TaskName $taskName
  "task_state=$($task.State) task_result=$($info.LastTaskResult)"
  $task.Actions | Select-Object Execute, Arguments, WorkingDirectory |
    ConvertTo-Json -Compress
  $task.Principal | Select-Object UserId, LogonType, RunLevel |
    ConvertTo-Json -Compress
}}
$curlProcesses = @(Get-CimInstance Win32_Process -Filter "Name='curl.exe'" `
  -ErrorAction SilentlyContinue)
"curl_process_count=$($curlProcesses.Count)"
foreach ($curlProcess in $curlProcesses) {{
  $curlProcess | Select-Object ProcessId, ParentProcessId, CreationDate, CommandLine |
    ConvertTo-Json -Compress
  Get-NetTCPConnection -OwningProcess $curlProcess.ProcessId `
    -ErrorAction SilentlyContinue |
    Select-Object State, LocalAddress, LocalPort, RemoteAddress, RemotePort |
    ConvertTo-Json -Compress
}}
"head_probe_begin"
& curl.exe --location --fail --show-error --ssl-no-revoke --connect-timeout 30 `
  --max-time 45 --head `
  'https://csr.bu.edu/ftp/visda/2019/multi-source/clipart.zip' 2>&1
"head_probe_exit=$LASTEXITCODE"
"head_probe_end"
'''
    hosts = connect_dual(args)
    try:
        result = run(hosts["third"], encoded_powershell(script),
                     timeout=180, check=False)
    finally:
        close_dual(hosts)
    print(json.dumps(result, ensure_ascii=True, indent=2))


def probe_domainnet_resource_task_network(args):
    """Run a bounded HEAD probe under the same S4U identity as the downloader."""
    script = rf'''
$ErrorActionPreference = 'Stop'
$repo = '{THIRD_REPO}'
$state = Join-Path $repo 'exp\domainnet_resource_download_dual'
$stamp = (Get-Date).ToString('yyyyMMdd_HHmmss')
$taskName = "GGEUR-domainnet-resource-network-probe-$stamp"
    $probeOut = Join-Path $state "network_probe_$stamp.range.bin"
$probeErr = Join-Path $state "network_probe_$stamp.stderr.log"
$identity = [Security.Principal.WindowsIdentity]::GetCurrent().Name
$arguments = "--location --fail --silent --show-error --ssl-no-revoke " +
  "--connect-timeout 30 --max-time 45 --range 0-1048575 " +
  "--output `"$probeOut`" --stderr `"$probeErr`" " +
  "https://csr.bu.edu/ftp/visda/2019/multi-source/clipart.zip"
$action = New-ScheduledTaskAction -Execute "$env:SystemRoot\System32\curl.exe" `
  -Argument $arguments -WorkingDirectory $repo
$principal = New-ScheduledTaskPrincipal -UserId $identity `
  -LogonType S4U -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet `
  -ExecutionTimeLimit (New-TimeSpan -Minutes 2) `
  -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName $taskName -Action $action `
  -Principal $principal -Settings $settings -Force | Out-Null
Start-ScheduledTask -TaskName $taskName
$deadline = (Get-Date).AddSeconds(60)
do {{
  Start-Sleep -Seconds 2
  $task = Get-ScheduledTask -TaskName $taskName
}} while ($task.State -eq 'Running' -and (Get-Date) -lt $deadline)
$info = Get-ScheduledTaskInfo -TaskName $taskName
"probe_task=$taskName state=$($task.State) result=$($info.LastTaskResult)"
if (Test-Path -LiteralPath $probeOut) {{
  "probe_payload=$probeOut exists=True bytes=$((Get-Item -LiteralPath $probeOut).Length)"
}} else {{
  "probe_payload=$probeOut exists=False bytes=0"
}}
"probe_stderr=$probeErr exists=$(Test-Path -LiteralPath $probeErr)"
if (Test-Path -LiteralPath $probeErr) {{ Get-Content -LiteralPath $probeErr -Raw }}
'''
    hosts = connect_dual(args)
    try:
        result = run(hosts["third"], encoded_powershell(script),
                     timeout=180, check=False)
    finally:
        close_dual(hosts)
    print(json.dumps(result, ensure_ascii=True, indent=2))


def probe_domainnet_resource_lengths(args):
    """Read final response lengths for the exact approved resource URLs."""
    script = r'''
$ErrorActionPreference = 'Continue'
$urls = @(
  'https://csr.bu.edu/ftp/visda/2019/multi-source/clipart.zip',
  'https://csr.bu.edu/ftp/visda/2019/multi-source/painting.zip',
  'https://csr.bu.edu/ftp/visda/2019/multi-source/real.zip',
  'https://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip',
  'https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt',
  'https://download.pytorch.org/models/convnext_base-6075fbad.pth',
  'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_mixer_b16_224-76587d61.pth'
)
foreach ($url in $urls) {
  $value = & curl.exe --location --fail --silent --show-error `
    --ssl-no-revoke --connect-timeout 30 --max-time 60 --head `
    --output NUL --write-out '%{url_effective}|%header{content-length}' $url
  "resource_length=$value source=$url exit=$LASTEXITCODE"
}
'''
    hosts = connect_dual(args)
    try:
        result = run(hosts["third"], encoded_powershell(script),
                     timeout=480, check=False)
    finally:
        close_dual(hosts)
    print(json.dumps(result, ensure_ascii=True, indent=2))


def pause_domainnet_resource_download(args):
    """Stop only the exact resource task while preserving all partial files."""
    task_name = "GGEUR-domainnet-resources-download-third"
    script = rf'''
$ErrorActionPreference = 'Stop'
$repo = '{THIRD_REPO}'
$state = Join-Path $repo 'exp\domainnet_resource_download_dual'
$taskName = '{task_name}'
$task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($null -eq $task) {{
  "task=$taskName state=MISSING"
  exit 0
}}
if ($task.State -eq 'Running') {{
  Stop-ScheduledTask -TaskName $taskName
  $deadline = (Get-Date).AddSeconds(30)
  do {{
    Start-Sleep -Seconds 1
    $task = Get-ScheduledTask -TaskName $taskName
  }} while ($task.State -eq 'Running' -and (Get-Date) -lt $deadline)
}}
$resourceProcesses = @(Get-CimInstance Win32_Process | Where-Object {{
  ($_.Name -eq 'curl.exe' -and
    $_.CommandLine -like '*domainnet_resource_download_dual\downloads\*') -or
  ($_.Name -eq 'powershell.exe' -and
    $_.CommandLine -like '*domainnet_resource_download_dual\run_download.ps1*')
}})
foreach ($resourceProcess in $resourceProcesses) {{
  "stopping_resource_pid=$($resourceProcess.ProcessId) name=$($resourceProcess.Name)"
  Stop-Process -Id $resourceProcess.ProcessId -Force -ErrorAction SilentlyContinue
}}
Start-Sleep -Seconds 2
$info = Get-ScheduledTaskInfo -TaskName $taskName
"task=$taskName state=$($task.State) result=$($info.LastTaskResult) preserved=$state"
Get-ChildItem -LiteralPath (Join-Path $state 'downloads') `
  -ErrorAction SilentlyContinue | Where-Object {{ -not $_.PSIsContainer }} |
  Sort-Object Name | ForEach-Object {{ "preserved_file=$($_.Name) bytes=$($_.Length)" }}
'''
    hosts = connect_dual(args)
    try:
        result = run(hosts["third"], encoded_powershell(script),
                     timeout=180, check=False)
    finally:
        close_dual(hosts)
    print(json.dumps(result, ensure_ascii=True, indent=2))


def sync_domainnet_verification_resources(args):
    """Copy the existing CLIP weights and compact cache references to third."""
    build_script = THIS_DIR / "build_domainnet_reference_samples.py"
    cache_script = THIS_DIR / "prepare_domainnet_eval_cache.py"
    for path in (build_script, cache_script):
        if not path.is_file():
            raise FileNotFoundError(path)
    transfer_dir = (REPO_ROOT / "exp" / "dual_deploy" /
                    "domainnet_verification_resources")
    transfer_dir.mkdir(parents=True, exist_ok=True)
    local_clip = transfer_dir / "open_clip_vitb16.bin"
    local_references = transfer_dir / "domainnet_reference_samples.tar"
    remote_build = (
        f"{CLIENT_REPO}/scripts/distributed_scripts/"
        "ggeur_hierarchical_3machine/build_domainnet_reference_samples.py"
    ).replace("\\", "/")
    remote_cache_script = (
        f"{THIRD_REPO}/scripts/distributed_scripts/"
        "ggeur_hierarchical_3machine/prepare_domainnet_eval_cache.py"
    ).replace("\\", "/")
    remote_clip_8g = (
        f"{CLIENT_REPO}/pretrained_models/open_clip_vitb16.bin"
    ).replace("\\", "/")
    remote_clip_third = (
        f"{THIRD_REPO}/pretrained_models/open_clip_vitb16.bin"
    ).replace("\\", "/")
    remote_references_8g = (
        f"{CLIENT_REPO}/exp/dual_deploy/domainnet_reference_samples.tar"
    ).replace("\\", "/")
    remote_references_third = (
        f"{THIRD_REPO}/exp/dual_deploy/domainnet_reference_samples.tar"
    ).replace("\\", "/")

    hosts = connect_dual(args)
    result = {}
    try:
        prepare8 = rf'''
$ErrorActionPreference = 'Stop'
$repo = '{CLIENT_REPO}'
$scriptDir = Join-Path $repo 'scripts\distributed_scripts\ggeur_hierarchical_3machine'
New-Item -ItemType Directory -Force -Path $scriptDir, `
  (Join-Path $repo 'exp\dual_deploy') | Out-Null
$clip = Join-Path $repo 'pretrained_models\open_clip_vitb16.bin'
if (-not (Test-Path -LiteralPath $clip)) {{ throw "missing CLIP checkpoint: $clip" }}
"clip_bytes=$((Get-Item -LiteralPath $clip).Length)"
'''
        result["prepare8"] = run(
            hosts["client8g"], encoded_powershell(prepare8), timeout=180)
        upload_file(hosts["client8g"], build_script, remote_build)
        build = rf'''
$ErrorActionPreference = 'Stop'
$repo = '{CLIENT_REPO}'
$python = Join-Path $repo '.venv_client_cpu\Scripts\python.exe'
& $python '{remote_build.replace('/', chr(92))}' --repo $repo --samples 16
if ($LASTEXITCODE -ne 0) {{ throw "reference builder failed: $LASTEXITCODE" }}
$archive = '{remote_references_8g.replace('/', chr(92))}'
& tar.exe -cf $archive -C $repo 'exp\domainnet_reference_samples'
if ($LASTEXITCODE -ne 0) {{ throw "reference tar failed: $LASTEXITCODE" }}
"reference_archive_bytes=$((Get-Item -LiteralPath $archive).Length)"
'''
        result["build_references"] = run(
            hosts["client8g"], encoded_powershell(build), timeout=600)
        download_file(hosts["client8g"], remote_references_8g,
                      local_references)

        clip_size_text = result["prepare8"]["stdout"].strip().split("=")[-1]
        expected_clip_bytes = int(clip_size_text)
        if (not local_clip.is_file() or
                local_clip.stat().st_size != expected_clip_bytes):
            download_file(hosts["client8g"], remote_clip_8g, local_clip)
        if local_clip.stat().st_size != expected_clip_bytes:
            raise RuntimeError("local CLIP checkpoint transfer size mismatch")
        clip_digest = hashlib.sha256()
        with local_clip.open("rb") as handle:
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                clip_digest.update(chunk)

        prepare3 = rf'''
$ErrorActionPreference = 'Stop'
$repo = '{THIRD_REPO}'
New-Item -ItemType Directory -Force -Path `
  (Join-Path $repo 'scripts\distributed_scripts\ggeur_hierarchical_3machine'), `
  (Join-Path $repo 'pretrained_models'), `
  (Join-Path $repo 'exp\dual_deploy') | Out-Null
'''
        result["prepare3"] = run(
            hosts["third"], encoded_powershell(prepare3), timeout=180)
        upload_file(hosts["third"], cache_script, remote_cache_script)
        upload_file(hosts["third"], local_references,
                    remote_references_third)
        third_clip_status = run(
            hosts["third"], encoded_powershell(rf'''
$path = '{remote_clip_third.replace('/', chr(92))}'
if (Test-Path -LiteralPath $path) {{
  "clip_bytes=$((Get-Item -LiteralPath $path).Length)"
}} else {{
  "clip_bytes=0"
}}
'''), timeout=180)
        third_clip_bytes = int(
            third_clip_status["stdout"].strip().split("=")[-1])
        if third_clip_bytes != expected_clip_bytes:
            upload_file(hosts["third"], local_clip, remote_clip_third)
        install3 = rf'''
$ErrorActionPreference = 'Stop'
$repo = '{THIRD_REPO}'
$archive = '{remote_references_third.replace('/', chr(92))}'
& tar.exe -xf $archive -C $repo
if ($LASTEXITCODE -ne 0) {{ throw "reference extraction failed: $LASTEXITCODE" }}
$clip = '{remote_clip_third.replace('/', chr(92))}'
$hash = (Get-FileHash -LiteralPath $clip -Algorithm SHA256).Hash.ToLowerInvariant()
"clip_bytes=$((Get-Item -LiteralPath $clip).Length)"
"clip_sha256=$hash"
Get-ChildItem -LiteralPath (Join-Path $repo 'exp\domainnet_reference_samples') |
  Where-Object {{ -not $_.PSIsContainer }} | Sort-Object Name |
  ForEach-Object {{ "reference=$($_.Name) bytes=$($_.Length)" }}
'''
        result["install3"] = run(
            hosts["third"], encoded_powershell(install3), timeout=600)
        remote_hash = ""
        for line in result["install3"]["stdout"].splitlines():
            if line.startswith("clip_sha256="):
                remote_hash = line.split("=", 1)[1].strip()
        if remote_hash != clip_digest.hexdigest():
            raise RuntimeError("third-host CLIP checkpoint hash mismatch")
        result["local"] = {
            "clip": str(local_clip),
            "clip_bytes": expected_clip_bytes,
            "clip_sha256": clip_digest.hexdigest(),
            "references": str(local_references),
            "references_bytes": local_references.stat().st_size,
        }
    finally:
        close_dual(hosts)
    print(json.dumps(result, ensure_ascii=True, indent=2))


def verify_domainnet_extractors(args):
    """Verify exact extractor/preprocessing parity against formal train caches."""
    groups = args.groups or ["domainnet_vit"]
    allowed = {"domainnet_vit", "domainnet_cnn", "domainnet_mixer"}
    invalid = [group for group in groups if group not in allowed]
    if invalid:
        raise ValueError(f"invalid DomainNet groups: {invalid}")
    group_args = " ".join(groups)
    script = rf'''
$ErrorActionPreference = 'Stop'
$repo = '{THIRD_REPO}'
$python = 'C:\Users\pc\miniconda3\envs\cerp\python.exe'
$tool = Join-Path $repo `
  'scripts\distributed_scripts\ggeur_hierarchical_3machine\prepare_domainnet_eval_cache.py'
& $python $tool --repo $repo --groups {group_args} --verify-only --num-workers 0
if ($LASTEXITCODE -ne 0) {{ throw "extractor verification failed: $LASTEXITCODE" }}
'''
    hosts = connect_dual(args)
    try:
        result = run(hosts["third"], encoded_powershell(script),
                     timeout=1800, check=False)
    finally:
        close_dual(hosts)
    print(json.dumps(result, ensure_ascii=True, indent=2))


def probe_domainnet_extractor_resources(args):
    """Report exact installed extractor metadata and checkpoint readability."""
    code = r'''
import json
from pathlib import Path
import open_clip
import timm
import torch

repo = Path(r"C:\Users\pc\FederatedScope")
cfg = open_clip.get_pretrained_cfg("ViT-B-16", "openai")
print(json.dumps({
    "open_clip_version": getattr(open_clip, "__version__", "unknown"),
    "open_clip_cfg": cfg,
    "timm_version": getattr(timm, "__version__", "unknown"),
}, default=str))
for name in ("open_clip_vitb16.bin", "convnext_base-6075fbad.pth",
             "mixer_b16_224_complete.pth"):
    path = repo / "pretrained_models" / name
    item = {"name": name, "exists": path.is_file(),
            "bytes": path.stat().st_size if path.is_file() else 0}
    if path.is_file():
        try:
            torch.load(str(path), map_location="cpu", weights_only=True)
            item["torch_load"] = "pass"
        except Exception as error:
            item["torch_load"] = f"fail:{type(error).__name__}:{error}"
    print(json.dumps(item))
'''
    code_b64 = base64.b64encode(code.encode("utf-8")).decode("ascii")
    script = rf'''
$python = 'C:\Users\pc\miniconda3\envs\cerp\python.exe'
$probe = '{THIRD_REPO}\exp\dual_deploy\probe_domainnet_extractors.py'
[IO.File]::WriteAllBytes(
  $probe, [Convert]::FromBase64String('{code_b64}'))
& $python $probe
'''
    hosts = connect_dual(args)
    try:
        result = run(hosts["third"], encoded_powershell(script),
                     timeout=600, check=False)
    finally:
        close_dual(hosts)
    print(json.dumps(result, ensure_ascii=True, indent=2))


def start_domainnet_eval_cache(args):
    """Install a durable third-host job for verified held-out caches."""
    local_script = THIS_DIR / "prepare_domainnet_eval_cache.py"
    local_driver = THIS_DIR / "run_domainnet_eval_cache.ps1"
    local_extractor = (
        REPO_ROOT / "federatedscope" / "contrib" / "model" /
        "ggeur_timm_extractor.py"
    )
    for path in (local_script, local_driver, local_extractor):
        if not path.is_file():
            raise FileNotFoundError(path)
    remote_script = (
        f"{THIRD_REPO}/scripts/distributed_scripts/"
        "ggeur_hierarchical_3machine/prepare_domainnet_eval_cache.py"
    ).replace("\\", "/")
    remote_driver = (
        f"{THIRD_REPO}/scripts/distributed_scripts/"
        "ggeur_hierarchical_3machine/run_domainnet_eval_cache.ps1"
    ).replace("\\", "/")
    remote_extractor = (
        f"{THIRD_REPO}/federatedscope/contrib/model/"
        "ggeur_timm_extractor.py"
    ).replace("\\", "/")
    task_name = "GGEUR-domainnet-eval-cache-third"

    hosts = connect_dual(args)
    result = {}
    try:
        prepare = rf'''
$ErrorActionPreference = 'Stop'
$repo = '{THIRD_REPO}'
$scriptDir = Join-Path $repo 'scripts\distributed_scripts\ggeur_hierarchical_3machine'
$state = Join-Path $repo 'exp\domainnet_eval_cache_dual'
New-Item -ItemType Directory -Force -Path $scriptDir, $state | Out-Null
'''
        result["prepare"] = run(
            hosts["third"], encoded_powershell(prepare), timeout=120)
        upload_file(hosts["third"], local_script, remote_script)
        upload_file(hosts["third"], local_driver, remote_driver)
        upload_file(hosts["third"], local_extractor, remote_extractor)
        install = rf'''
$ErrorActionPreference = 'Stop'
$repo = '{THIRD_REPO}'
$state = Join-Path $repo 'exp\domainnet_eval_cache_dual'
$driver = '{remote_driver.replace('/', chr(92))}'
$taskName = '{task_name}'
"install_stage=inspect_existing task=$taskName"
$old = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($old -and $old.State -eq 'Running') {{
  "already_running=$taskName"
  exit 0
}}
if (Test-Path -LiteralPath (Join-Path $state 'completed.json')) {{
  "already_complete=$taskName"
  exit 0
}}
if ($old) {{
  $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
  $attempt = Join-Path $state "attempts\$stamp"
  New-Item -ItemType Directory -Force -Path $attempt | Out-Null
  foreach ($name in @('pipeline.log', 'task.stdout.log', 'task.stderr.log', 'cache.stdout.log', 'cache.stderr.log')) {{
    $path = Join-Path $state $name
    if (Test-Path -LiteralPath $path) {{ Copy-Item -LiteralPath $path -Destination $attempt }}
  }}
  Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
}}
$argument = "-NoProfile -NonInteractive -ExecutionPolicy Bypass -File `"$driver`" -RepoDir `"$repo`""
"install_stage=build_definition identity=$([Security.Principal.WindowsIdentity]::GetCurrent().Name)"
$identity = [Security.Principal.WindowsIdentity]::GetCurrent().Name
$action = New-ScheduledTaskAction `
  -Execute "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" `
  -Argument $argument -WorkingDirectory $repo
$principal = New-ScheduledTaskPrincipal -UserId $identity `
  -LogonType S4U -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet `
  -ExecutionTimeLimit (New-TimeSpan -Days 7) `
  -MultipleInstances IgnoreNew -AllowStartIfOnBatteries `
  -DontStopIfGoingOnBatteries
"install_stage=register task=$taskName"
Register-ScheduledTask -TaskName $taskName -Action $action `
  -Principal $principal -Settings $settings -Force | Out-Null
"install_stage=start task=$taskName"
Start-ScheduledTask -TaskName $taskName
"started=$taskName state_dir=$state"
'''
        result["start"] = run(
            hosts["third"], encoded_powershell(install), timeout=180)
    finally:
        close_dual(hosts)
    print(json.dumps(result, ensure_ascii=True, indent=2))


def domainnet_eval_cache_status(args):
    """Report the durable held-out cache job on the third host."""
    task_name = "GGEUR-domainnet-eval-cache-third"
    script = rf'''
$ErrorActionPreference = 'Continue'
$repo = '{THIRD_REPO}'
$state = Join-Path $repo 'exp\domainnet_eval_cache_dual'
$taskName = '{task_name}'
$task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($task) {{
  $info = Get-ScheduledTaskInfo -TaskName $taskName
  "task=$taskName state=$($task.State) result=$($info.LastTaskResult)"
}} else {{
  "task=$taskName state=MISSING"
}}
foreach ($name in @('pipeline.log', 'task.stdout.log', 'task.stderr.log', 'cache.stdout.log', 'cache.stderr.log')) {{
  $path = Join-Path $state $name
  if (Test-Path -LiteralPath $path) {{
    $item = Get-Item -LiteralPath $path
    "log=$name bytes=$($item.Length) mtime=$($item.LastWriteTime.ToString('o'))"
    Get-Content -LiteralPath $path -Tail 24
  }}
}}
foreach ($group in @('domainnet_vit', 'domainnet_cnn', 'domainnet_mixer')) {{
  $path = Join-Path $repo "exp\distributed_feature_cache\$group"
  $files = @(Get-ChildItem -LiteralPath $path -File -Filter 'domainnet_*_test_*.npz' -ErrorAction SilentlyContinue)
  $bytes = ($files | Measure-Object Length -Sum).Sum
  "group=$group test_files=$($files.Count) bytes=$bytes"
}}
$archive = Join-Path $state 'domainnet_eval_cache.tar'
if (Test-Path -LiteralPath $archive) {{
  "archive=$archive bytes=$((Get-Item -LiteralPath $archive).Length)"
}}
"complete=$(Test-Path -LiteralPath (Join-Path $state 'completed.json'))"
'''
    hosts = connect_dual(args)
    try:
        result = run(hosts["third"], encoded_powershell(script),
                     timeout=180, check=False)
    finally:
        close_dual(hosts)
    print(json.dumps(result, ensure_ascii=True, indent=2))


def sync_domainnet_eval_cache(args):
    """Transfer the verified held-out cache archive from third to 8G."""
    transfer_dir = (REPO_ROOT / "exp" / "dual_deploy" /
                    "domainnet_eval_cache_transfer")
    transfer_dir.mkdir(parents=True, exist_ok=True)
    local_archive = transfer_dir / "domainnet_eval_cache.tar"
    local_validator = THIS_DIR / "validate_domainnet_eval_cache.py"
    if not local_validator.is_file():
        raise FileNotFoundError(local_validator)
    third_archive = (
        f"{THIRD_REPO}/exp/domainnet_eval_cache_dual/"
        "domainnet_eval_cache.tar"
    ).replace("\\", "/")
    client_archive = (
        f"{CLIENT_REPO}/exp/dual_deploy/domainnet_eval_cache.tar"
    ).replace("\\", "/")
    client_validator = (
        f"{CLIENT_REPO}/exp/dual_deploy/"
        "validate_domainnet_eval_cache.py"
    ).replace("\\", "/")

    hosts = connect_dual(args)
    result = {}
    try:
        third_status = run(
            hosts["third"], encoded_powershell(rf'''
$ErrorActionPreference = 'Stop'
$state = '{THIRD_REPO}\exp\domainnet_eval_cache_dual'
$complete = Join-Path $state 'completed.json'
if (-not (Test-Path -LiteralPath $complete)) {{
  throw "eval cache is not complete: $complete"
}}
$archive = Join-Path $state 'domainnet_eval_cache.tar'
if (-not (Test-Path -LiteralPath $archive)) {{
  throw "eval cache archive is missing: $archive"
}}
$item = Get-Item -LiteralPath $archive
$hash = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
"archive_bytes=$($item.Length)"
"archive_sha256=$hash"
Get-Content -LiteralPath $complete -Raw
'''), timeout=1800)
        result["third"] = third_status
        third_bytes = None
        third_sha256 = ""
        for line in third_status["stdout"].splitlines():
            if line.startswith("archive_bytes="):
                third_bytes = int(line.split("=", 1)[1])
            elif line.startswith("archive_sha256="):
                third_sha256 = line.split("=", 1)[1].strip()
        if not third_bytes or not re.fullmatch(r"[0-9a-f]{64}", third_sha256):
            raise RuntimeError("invalid third-host eval-cache archive metadata")

        local_valid = False
        if local_archive.is_file() and local_archive.stat().st_size == third_bytes:
            digest = hashlib.sha256()
            with local_archive.open("rb") as handle:
                for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                    digest.update(chunk)
            local_valid = digest.hexdigest() == third_sha256
        if not local_valid:
            if local_archive.exists():
                stamp = time.strftime("%Y%m%d_%H%M%S")
                preserved = local_archive.with_name(
                    f"domainnet_eval_cache.{stamp}.invalid.tar")
                os.replace(local_archive, preserved)
                result["preserved_local"] = str(preserved)
            download_file(hosts["third"], third_archive, local_archive)
        if local_archive.stat().st_size != third_bytes:
            raise RuntimeError("local eval-cache archive size mismatch")
        digest = hashlib.sha256()
        with local_archive.open("rb") as handle:
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
        local_sha256 = digest.hexdigest()
        if local_sha256 != third_sha256:
            raise RuntimeError("local eval-cache archive hash mismatch")
        result["local"] = {
            "archive": str(local_archive),
            "bytes": third_bytes,
            "sha256": local_sha256,
            "reused": local_valid,
        }

        prepare8 = run(
            hosts["client8g"], encoded_powershell(rf'''
$ErrorActionPreference = 'Stop'
$repo = '{CLIENT_REPO}'
$archive = '{client_archive.replace('/', chr(92))}'
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $archive) | Out-Null
if (Test-Path -LiteralPath $archive) {{
  $item = Get-Item -LiteralPath $archive
  $hash = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
  "archive_bytes=$($item.Length)"
  "archive_sha256=$hash"
}} else {{
  "archive_bytes=0"
  "archive_sha256=missing"
}}
'''), timeout=1800)
        result["prepare8"] = prepare8
        remote_bytes = 0
        remote_sha256 = ""
        for line in prepare8["stdout"].splitlines():
            if line.startswith("archive_bytes="):
                remote_bytes = int(line.split("=", 1)[1])
            elif line.startswith("archive_sha256="):
                remote_sha256 = line.split("=", 1)[1].strip()
        if remote_bytes != third_bytes or remote_sha256 != third_sha256:
            preserve8 = run(
                hosts["client8g"], encoded_powershell(rf'''
$archive = '{client_archive.replace('/', chr(92))}'
if (Test-Path -LiteralPath $archive) {{
  $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
  $preserved = Join-Path (Split-Path -Parent $archive) `
    "domainnet_eval_cache.$stamp.previous.tar"
  Move-Item -LiteralPath $archive -Destination $preserved
  "preserved=$preserved"
}}
'''), timeout=180)
            result["preserve8"] = preserve8
            upload_file(hosts["client8g"], local_archive, client_archive)

        upload_file(hosts["client8g"], local_validator, client_validator)
        install8 = run(
            hosts["client8g"], encoded_powershell(rf'''
$ErrorActionPreference = 'Stop'
$repo = '{CLIENT_REPO}'
$archive = '{client_archive.replace('/', chr(92))}'
$expectedBytes = {third_bytes}
$expectedHash = '{third_sha256}'
"install_stage=archive_inspect path=$archive"
$item = Get-Item -LiteralPath $archive
if ($item.Length -ne $expectedBytes) {{
  throw "8G archive size mismatch: $($item.Length)/$expectedBytes"
}}
$hash = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
if ($hash -ne $expectedHash) {{ throw "8G archive hash mismatch: $hash" }}
"install_stage=archive_verified bytes=$($item.Length) sha256=$hash"
$existing = @()
foreach ($group in @('domainnet_vit', 'domainnet_cnn', 'domainnet_mixer')) {{
  $path = Join-Path $repo "exp\distributed_feature_cache\$group"
  $existing += @(Get-ChildItem -LiteralPath $path -File `
    -Filter 'domainnet_*_test_*.npz' -ErrorAction SilentlyContinue)
}}
if ($existing.Count -gt 0) {{
  "install_stage=preserve_existing count=$($existing.Count)"
  $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
  $attempt = Join-Path $repo "exp\domainnet_eval_cache_attempts\$stamp"
  foreach ($file in $existing) {{
    $group = Split-Path -Leaf (Split-Path -Parent $file.FullName)
    $target = Join-Path $attempt $group
    New-Item -ItemType Directory -Force -Path $target | Out-Null
    Move-Item -LiteralPath $file.FullName -Destination $target
  }}
  "preserved_existing=$($existing.Count) attempt=$attempt"
}}
"install_stage=extract_start"
& tar.exe -xf $archive -C $repo
if ($LASTEXITCODE -ne 0) {{ throw "8G cache extraction failed: $LASTEXITCODE" }}
"install_stage=extract_complete"
$validator = Join-Path $repo 'exp\dual_deploy\validate_domainnet_eval_cache.py'
$python = Join-Path $repo '.venv_client_cpu\Scripts\python.exe'
$savedErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
"install_stage=validation_start python=$python"
try {{
  & $python $validator $repo
  $validationExitCode = $LASTEXITCODE
}} finally {{
  $ErrorActionPreference = $savedErrorActionPreference
}}
if ($validationExitCode -ne 0) {{ throw "8G cache validation failed: $validationExitCode" }}
"install_stage=validation_complete"
'''), timeout=3600, check=False)
        result["install8"] = install8
        if install8["status"] != 0:
            print(json.dumps(result, ensure_ascii=True, indent=2))
            raise RuntimeError("8G eval-cache install/validation failed")
    finally:
        close_dual(hosts)
    print(json.dumps(result, ensure_ascii=True, indent=2))


def prepare_domainnet_terminal_shards(args):
    """Create memory-bounded held-out DomainNet shards on both client hosts."""
    cache_relative = (args.groups[0] if args.groups else
                      r"exp\distributed_feature_cache\domainnet_cnn")
    if not cache_relative.lower().endswith("domainnet_cnn"):
        raise ValueError(
            "prepare-domainnet-terminal-shards --groups must contain one "
            "cache path ending in domainnet_cnn")
    hosts = connect_dual(args)
    try:
        script8 = rf'''
$ErrorActionPreference = 'Stop'
$repo = '{CLIENT_REPO}'
$python = Join-Path $repo '.venv_client_cpu\Scripts\python.exe'
$tool = Join-Path $repo `
  'scripts\distributed_scripts\ggeur_hierarchical_3machine\prepare_domainnet_terminal_eval_shards.py'
$cache = Join-Path $repo '{cache_relative}'
& $python $tool --cache-dir $cache --client-num 60
if ($LASTEXITCODE -ne 0) {{
  throw "terminal eval shard preparation failed: $LASTEXITCODE"
}}
'''
        script_third = rf'''
$ErrorActionPreference = 'Stop'
$repo = '{THIRD_REPO}'
$python = 'C:\Users\pc\miniconda3\envs\cerp\python.exe'
$tool = Join-Path $repo `
  'scripts\distributed_scripts\ggeur_hierarchical_3machine\prepare_domainnet_terminal_eval_shards.py'
$cache = Join-Path $repo '{cache_relative}'
& $python $tool --cache-dir $cache --client-num 60
if ($LASTEXITCODE -ne 0) {{
  throw "terminal eval shard preparation failed: $LASTEXITCODE"
}}
'''
        result8 = run(hosts["client8g"], encoded_powershell(script8),
                      timeout=1800, check=False)
        result_third = run(hosts["third"], encoded_powershell(script_third),
                           timeout=1800, check=False)
    finally:
        close_dual(hosts)
    print(json.dumps({"client8g": result8, "third": result_third},
                     ensure_ascii=True, indent=2))


def prepare_domainnet_original_terminal_shards_8g(args):
    """Create terminal shards in the validated original-4 cache on 8G."""
    remote_tool = (CLIENT_REPO +
                   r"\scripts\distributed_scripts"
                   r"\ggeur_hierarchical_3machine"
                   r"\prepare_domainnet_terminal_eval_shards.py")
    local_tool = THIS_DIR / "prepare_domainnet_terminal_eval_shards.py"
    script = rf'''
$ErrorActionPreference = 'Stop'
$repo = '{CLIENT_REPO}'
$python = Join-Path $repo '.venv_client_cpu\Scripts\python.exe'
$cache = Join-Path $repo `
  'exp\distributed_feature_cache_domainnet_original_20260804\domainnet_cnn'
& $python '{remote_tool}' --cache-dir $cache --client-num 60
if ($LASTEXITCODE -ne 0) {{
  throw "original terminal shard preparation failed: $LASTEXITCODE"
}}
'''
    hosts = connect_dual(args)
    try:
        upload_file(hosts["client8g"], local_tool, remote_tool)
        result = run(hosts["client8g"], encoded_powershell(script),
                     timeout=1800, check=False)
    finally:
        close_dual(hosts)
    print(json.dumps({"client8g": result}, ensure_ascii=True, indent=2))


def resume_third_clients(args):
    """Resume only the third-host client queue for the current case."""
    hosts = connect_dual(args)
    try:
        task_name = f"GGEUR-dual-{args.run_id}-clients-third"
        script = rf"""
$ErrorActionPreference = 'Stop'
$taskName = '{task_name}'
$task = Get-ScheduledTask -TaskName $taskName -ErrorAction Stop
if ($task.State -eq 'Running') {{
  "already_running=$taskName"
}} else {{
  Start-ScheduledTask -TaskName $taskName
  "resumed=$taskName"
}}
"""
        result = run(
            hosts["third"], encoded_powershell(script), timeout=60)
    finally:
        close_dual(hosts)
    print(json.dumps(result, ensure_ascii=True, indent=2))


def resume_8g_clients(args):
    """Resume only the 8G-host client queue for the current case."""
    hosts = connect_dual(args)
    try:
        task_name = f"GGEUR-dual-{args.run_id}-clients-8g"
        script = rf"""
$ErrorActionPreference = 'Stop'
$taskName = '{task_name}'
$task = Get-ScheduledTask -TaskName $taskName -ErrorAction Stop
if ($task.State -eq 'Running') {{
  "already_running=$taskName"
}} else {{
  Start-ScheduledTask -TaskName $taskName
  "resumed=$taskName"
}}
"""
        result = run(
            hosts["client8g"], encoded_powershell(script), timeout=60)
    finally:
        close_dual(hosts)
    print(json.dumps(result, ensure_ascii=True, indent=2))


def resume_third_subservers(args):
    """Resume only the third-host subserver queue for the current case."""
    hosts = connect_dual(args)
    try:
        task_name = f"GGEUR-dual-{args.run_id}-subservers-third"
        script = rf"""
$ErrorActionPreference = 'Stop'
$taskName = '{task_name}'
$task = Get-ScheduledTask -TaskName $taskName -ErrorAction Stop
if ($task.State -eq 'Running') {{
  "already_running=$taskName"
}} else {{
  Start-ScheduledTask -TaskName $taskName
  "resumed=$taskName"
}}
"""
        result = run(
            hosts["third"], encoded_powershell(script), timeout=60)
    finally:
        close_dual(hosts)
    print(json.dumps(result, ensure_ascii=True, indent=2))


def restart_third_client_driver(args):
    """Restart only the PowerShell queue driver; client tasks stay intact."""
    hosts = connect_dual(args)
    try:
        task_name = f"GGEUR-dual-{args.run_id}-clients-third"
        script = rf"""
$ErrorActionPreference = 'Stop'
$taskName = '{task_name}'
$task = Get-ScheduledTask -TaskName $taskName -ErrorAction Stop
if ($task.State -eq 'Running') {{
  Stop-ScheduledTask -TaskName $taskName
  foreach ($attempt in 1..60) {{
    if ((Get-ScheduledTask -TaskName $taskName).State -ne 'Running') {{
      break
    }}
    Start-Sleep -Milliseconds 250
  }}
}}
Start-ScheduledTask -TaskName $taskName
"restarted_driver=$taskName"
"""
        result = run(
            hosts["third"], encoded_powershell(script), timeout=60)
    finally:
        close_dual(hosts)
    print(json.dumps(result, ensure_ascii=True, indent=2))


def diagnose_8g_clients(args):
    """Report non-running 8G client tasks and their log tails."""
    task_prefix = f"GGEUR-{args.run_id}-{args.case_name}-client_"
    logs_root = CLIENT_REPO + (
        r"\scripts\distributed_scripts\ggeur_hierarchical_3machine\runs"
        + "\\" + args.run_id + "\\" + args.case_name + r"\logs")
    script = rf"""
$ErrorActionPreference = 'Continue'
$prefix = '{task_prefix}'
$logs = '{logs_root}'
$tasks = @(Get-ScheduledTask -ErrorAction SilentlyContinue | Where-Object {{
  $_.TaskName.StartsWith($prefix)
}} | Sort-Object TaskName)
"matched_clients=$($tasks.Count)"
foreach ($task in $tasks) {{
  $info = Get-ScheduledTaskInfo -TaskName $task.TaskName
  if ($task.State -ne 'Running' -or $info.LastTaskResult -ne 267009) {{
    $client = $task.TaskName.Substring($prefix.Length)
    "client=$client state=$($task.State) result=$($info.LastTaskResult)"
    foreach ($stream in @('stderr', 'stdout')) {{
      $path = Join-Path $logs ('client_' + $client + '.' + $stream + '.log')
      if (Test-Path -LiteralPath $path) {{
        $item = Get-Item -LiteralPath $path
        "log=$($item.Name) bytes=$($item.Length) mtime=$($item.LastWriteTime.ToString('o'))"
        Get-Content -LiteralPath $path -Tail 16
      }}
    }}
  }}
}}
"listening_client_ports=$(@(Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue | Where-Object {{
  $_.LocalPort -ge 30000 -and $_.LocalPort -le 30059
}}).Count)"
"""
    hosts = connect_dual(args)
    try:
        result = run(hosts["client8g"], encoded_powershell(script),
                     timeout=180, check=False)
    finally:
        close_dual(hosts)
    print(json.dumps({"client8g": result}, ensure_ascii=True, indent=2))


def diagnose_active_domainnet_8g(args):
    """Read-only health snapshot for active DomainNet clients on the 8G host."""
    logs_root = CLIENT_REPO + (
        r"\scripts\distributed_scripts\ggeur_hierarchical_3machine\runs"
        + "\\" + args.run_id + "\\" + args.case_name + r"\logs")
    script = rf"""
$ErrorActionPreference = 'Continue'
$logs = '{logs_root}'
$needle = '{args.run_id}-{args.case_name}'
$procs = @(Get-CimInstance Win32_Process | Where-Object {{
  $_.CommandLine -and $_.CommandLine.Contains($needle)
}})
"matched_processes=$($procs.Count)"
$working = 0L
$private = 0L
$cpu = 0.0
foreach ($proc in $procs) {{
  $p = Get-Process -Id $proc.ProcessId -ErrorAction SilentlyContinue
  if ($p) {{
    $working += $p.WorkingSet64
    $private += $p.PrivateMemorySize64
    if ($null -ne $p.CPU) {{ $cpu += $p.CPU }}
  }}
}}
"working_set_bytes=$working"
"private_bytes=$private"
"cpu_seconds_total=$([math]::Round($cpu, 3))"
$os = Get-CimInstance Win32_OperatingSystem
"memory_free_kib=$($os.FreePhysicalMemory)"
"memory_total_kib=$($os.TotalVisibleMemorySize)"
foreach ($id in 31..60) {{
  $name = 'client_' + $id.ToString('000000') + '.stderr.log'
  $path = Join-Path $logs $name
  if (Test-Path -LiteralPath $path) {{
    $item = Get-Item -LiteralPath $path
    $last = @(Get-Content -LiteralPath $path -Tail 1)
    "client=$id bytes=$($item.Length) mtime=$($item.LastWriteTime.ToString('o')) last=$last"
  }} else {{
    "client=$id log=missing"
  }}
}}
$errorFiles = @(Get-ChildItem -LiteralPath $logs -Filter 'client_*.stderr.log' `
  -ErrorAction SilentlyContinue | Select-String `
  -Pattern 'ERROR|Traceback|MemoryError|OutOfMemory|missing feature|cache is incomplete' `
  -List -ErrorAction SilentlyContinue)
"error_log_files=$($errorFiles.Count)"
foreach ($hit in $errorFiles | Select-Object -First 12) {{
  "error_file=$($hit.Path) line=$($hit.LineNumber) text=$($hit.Line)"
}}
"listening_client_ports=$(@(Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue | Where-Object {{
  $_.LocalPort -ge 30000 -and $_.LocalPort -le 30059
}}).Count)"
"""
    hosts = connect_dual(args)
    try:
        result = run(hosts["client8g"], encoded_powershell(script),
                     timeout=300, check=False)
    finally:
        close_dual(hosts)
    print(json.dumps({"client8g": result}, ensure_ascii=True, indent=2))


def recover_8g_clients(args):
    """Preserve failed logs, then restart only the 8G client queue driver."""
    task_prefix = f"GGEUR-{args.run_id}-{args.case_name}-client_"
    driver_task = f"GGEUR-dual-{args.run_id}-clients-8g"
    run_root = CLIENT_REPO + (
        r"\scripts\distributed_scripts\ggeur_hierarchical_3machine\runs"
        + "\\" + args.run_id)
    case_root = run_root + "\\" + args.case_name
    script = rf"""
$ErrorActionPreference = 'Stop'
$prefix = '{task_prefix}'
$driver = '{driver_task}'
$caseRoot = '{case_root}'
$logs = Join-Path $caseRoot 'logs'
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$attempt = Join-Path $caseRoot ('attempts\\' + $stamp + '_8g_client_recovery')
New-Item -ItemType Directory -Force -Path $attempt | Out-Null
$audit = Join-Path $attempt 'failed_tasks.tsv'
$failed = @()
foreach ($task in @(Get-ScheduledTask -ErrorAction SilentlyContinue | Where-Object {{
  $_.TaskName.StartsWith($prefix)
}} | Sort-Object TaskName)) {{
  $info = Get-ScheduledTaskInfo -TaskName $task.TaskName
  if ($task.State -ne 'Running' -or $info.LastTaskResult -ne 267009) {{
    $failed += $task
    $client = $task.TaskName.Substring($prefix.Length)
    Add-Content -LiteralPath $audit -Value `
      "$client`t$($task.State)`t$($info.LastTaskResult)"
    foreach ($stream in @('stderr', 'stdout')) {{
      $source = Join-Path $logs ('client_' + $client + '.' + $stream + '.log')
      if (Test-Path -LiteralPath $source) {{
        Copy-Item -LiteralPath $source -Destination $attempt -Force
      }}
    }}
  }}
}}
if ($failed.Count -eq 0) {{
  "no_failed_clients=true"
  "attempt=$attempt"
  exit 0
}}
$queueAudit = Join-Path '{run_root}' 'queue_state\clients_clients_8g_queue.tsv'
Add-Content -LiteralPath $queueAudit -Value `
  "$(Get-Date -Format o)`t{args.case_name}`tRECOVER_CLIENTS_8G:failed=$($failed.Count):attempt=$attempt"
$driverState = (Get-ScheduledTask -TaskName $driver -ErrorAction Stop).State
if ($driverState -eq 'Running') {{
  Stop-ScheduledTask -TaskName $driver
  foreach ($attemptNumber in 1..60) {{
    if ((Get-ScheduledTask -TaskName $driver).State -ne 'Running') {{ break }}
    Start-Sleep -Milliseconds 250
  }}
}}
Start-ScheduledTask -TaskName $driver
"preserved_failed_clients=$($failed.Count)"
"attempt=$attempt"
"restarted_driver=$driver"
"""
    hosts = connect_dual(args)
    try:
        result = run(hosts["client8g"], encoded_powershell(script),
                     timeout=180, check=False)
    finally:
        close_dual(hosts)
    print(json.dumps({"client8g": result}, ensure_ascii=True, indent=2))


def stop_failed_case(args):
    """Archive and stop only one irrecoverably stalled formal case."""
    safe = r"[A-Za-z0-9_.-]{8,128}"
    safe_reason = r"[A-Za-z0-9_.-]{1,128}"
    if not re.fullmatch(safe, args.run_id):
        raise ValueError(f"unsafe run id: {args.run_id!r}")
    if not re.fullmatch(safe, args.case_name):
        raise ValueError(f"unsafe case name: {args.case_name!r}")
    if not re.fullmatch(safe_reason, args.failure_reason):
        raise ValueError(f"unsafe failure reason: {args.failure_reason!r}")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    template = r"""
$ErrorActionPreference = 'Continue'
$repo = '__REPO__'
$run = '__RUN__'
$case = '__CASE__'
$stamp = '__STAMP__'
$runRoot = Join-Path $repo `
  "scripts\distributed_scripts\ggeur_hierarchical_3machine\runs\$run"
$caseRoot = Join-Path $runRoot $case
$attempt = Join-Path $caseRoot `
  ("attempts\full_restart_" + $stamp + "_" + $env:COMPUTERNAME)
New-Item -ItemType Directory -Force -Path $attempt | Out-Null
$logs = Join-Path $caseRoot 'logs'
if (Test-Path -LiteralPath $logs) {
  Copy-Item -LiteralPath $logs -Destination (Join-Path $attempt 'logs') `
    -Recurse -Force
}
$rootConfig = Join-Path $caseRoot 'configs\root_server.yaml'
if (Test-Path -LiteralPath $rootConfig) {
  Copy-Item -LiteralPath $rootConfig -Destination $attempt -Force
}
$taskPrefix = "GGEUR-$run-$case-"
$tasks = @(Get-ScheduledTask -ErrorAction SilentlyContinue | Where-Object {
  $_.TaskName.StartsWith($taskPrefix)
})
$taskRows = foreach ($task in $tasks) {
  $info = Get-ScheduledTaskInfo -TaskName $task.TaskName
  [pscustomobject]@{
    name = $task.TaskName
    state = [string]$task.State
    result = $info.LastTaskResult
  }
}
$taskRows | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath `
  (Join-Path $attempt 'tasks.json') -Encoding UTF8
$caseNeedle = "\runs\$run\$case\"
$processes = @(Get-CimInstance Win32_Process | Where-Object {
  $_.CommandLine -and $_.CommandLine.Contains($caseNeedle)
})
$processes | Select-Object ProcessId,ParentProcessId,CreationDate,
  ExecutablePath,CommandLine | ConvertTo-Json -Depth 4 |
  Set-Content -LiteralPath (Join-Path $attempt 'processes.json') -Encoding UTF8
@(
  "status=FAILED_RESTART",
  "archived_at=$(Get-Date -Format o)",
  "reason=__REASON__",
  "matched_tasks=$($tasks.Count)",
  "matched_processes=$($processes.Count)"
) | Set-Content -LiteralPath (Join-Path $attempt 'attempt_status.txt')
$queueAudit = Join-Path $runRoot 'queue_state\recovery_audit.tsv'
Add-Content -LiteralPath $queueAudit -Value `
  "$(Get-Date -Format o)`t$case`tFULL_RESTART_ARCHIVED:$attempt:tasks=$($tasks.Count):processes=$($processes.Count)"
foreach ($task in $tasks) {
  Stop-ScheduledTask -TaskName $task.TaskName -ErrorAction SilentlyContinue
}
Start-Sleep -Seconds 2
foreach ($original in $processes) {
  $current = Get-CimInstance Win32_Process -Filter `
    "ProcessId=$($original.ProcessId)" -ErrorAction SilentlyContinue
  if ($current -and $current.CommandLine -and
      $current.CommandLine.Contains($caseNeedle)) {
    Stop-Process -Id $current.ProcessId -Force -ErrorAction SilentlyContinue
  }
}
"attempt=$attempt"
"stopped_case_tasks=$($tasks.Count)"
"stopped_case_processes=$($processes.Count)"
"""
    hosts = connect_dual(args)
    result = {}
    try:
        for name, repo in (("client8g", CLIENT_REPO),
                           ("third", THIRD_REPO)):
            script = (template.replace("__REPO__", repo)
                      .replace("__RUN__", args.run_id)
                      .replace("__CASE__", args.case_name)
                      .replace("__REASON__", args.failure_reason)
                      .replace("__STAMP__", stamp))
            result[name] = run(hosts[name], encoded_powershell(script),
                               timeout=300, check=False)
    finally:
        close_dual(hosts)
    print(json.dumps(result, ensure_ascii=True, indent=2))


def progress_run(args):
    """Return compact file-backed progress for polling long queues."""
    hosts = connect_dual(args)
    result = {}
    try:
        for name, repo in (("client8g", CLIENT_REPO),
                           ("third", THIRD_REPO)):
            sftp = hosts[name].open_sftp()
            try:
                run_root = repo + (
                    r"\scripts\distributed_scripts"
                    r"\ggeur_hierarchical_3machine\runs" + "\\" +
                    args.run_id)
                case_root = run_root + "\\" + args.case_name
                item = {}
                for label, path in (
                    ("control", run_root + r"\queue_state\control.json"),
                    ("client_queue", run_root +
                     r"\queue_state\clients_clients_third_queue.tsv"),
                    ("root_queue", run_root + r"\queue_state\root_queue.tsv"),
                ):
                    try:
                        with sftp.file(path, "rb") as handle:
                            lines = _decode_remote_text(
                                handle.read()).splitlines()
                        item[label] = lines[-8:]
                    except OSError:
                        pass
                try:
                    logs = sftp.listdir_attr(case_root + r"\logs")
                    client_stderr = [
                        attr for attr in logs
                        if attr.filename.startswith("client_") and
                        attr.filename.endswith(".stderr.log")]
                    item["client_log_count"] = len(client_stderr)
                    item["client_nonempty_log_count"] = sum(
                        attr.st_size > 0 for attr in client_stderr)
                    item["highest_client_log"] = max(
                        (attr.filename for attr in client_stderr),
                        default=None)
                except OSError:
                    pass
                root_text = ""
                for filename in ("root.stdout.log", "root.stderr.log"):
                    try:
                        with sftp.file(
                                case_root + r"\logs" + "\\" + filename,
                                "rb") as handle:
                            root_text += "\n" + _decode_remote_text(
                                handle.read())
                    except OSError:
                        pass
                if root_text:
                    stat_counts = [int(value) for value in re.findall(
                        r"buffer_clients=(\d+)", root_text)]
                    item["root_metrics"] = {
                        "statistics_clients_max": max(stat_counts,
                                                       default=0),
                        "accuracy_rounds": len(re.findall(
                            r"Round \d+ MLP Test Accuracy", root_text)),
                        "valid_update_lines": len(re.findall(
                            r"valid_updates=\d+/\d+", root_text)),
                        "training_finished": (
                            "Training finished after" in root_text),
                        "error_lines": sum(
                            "ERROR" in line for line in
                            root_text.splitlines()),
                        "tail": root_text.splitlines()[-8:],
                    }
                try:
                    pids = sftp.listdir_attr(case_root + r"\pids")
                    item["pid_file_count"] = sum(
                        attr.filename.endswith(".pid") for attr in pids)
                except OSError:
                    pass
                result[name] = item
            finally:
                sftp.close()
    finally:
        close_dual(hosts)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def archive_completed(args):
    """Download the formal evidence for every completed case from the 8G root."""
    local_run_root = THIS_DIR / "runs" / args.run_id
    manifest_path = local_run_root / "matrix_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"matrix manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    case_names = [item["case"] for item in manifest["cases"]]
    local_archive = REPO_ROOT / "docs" / "test_logs" / args.run_id
    remote_run_root = CLIENT_REPO + (
        r"\scripts\distributed_scripts\ggeur_hierarchical_3machine\runs"
        + "\\" + args.run_id)
    hosts = connect_dual(args)
    archived = []
    pending = []
    try:
        root = hosts["client8g"]
        sftp = root.open_sftp()
        try:
            for case_name in case_names:
                remote_case = remote_run_root + "\\" + case_name
                try:
                    sftp.stat(remote_case + r"\.formal_complete")
                except OSError:
                    pending.append(case_name)
                    continue
                destination = local_archive / case_name
                destination.mkdir(parents=True, exist_ok=True)
                files = (
                    (remote_case + r"\logs\root.stdout.log",
                     destination / "root.stdout.log"),
                    (remote_case + r"\accuracy_summary.json",
                     destination / "accuracy_summary.json"),
                    (remote_case + r"\completion_validation.json",
                     destination / "completion_validation.json"),
                )
                for remote_path, local_path in files:
                    download_file(root, remote_path, local_path)
                log_text = (destination / "root.stdout.log").read_text(
                    encoding="utf-8", errors="replace")
                evidence_matches = re.findall(
                    r"Client model accuracy evidence written to\s+"
                    r"([^\r\n]*?client_model_accuracy_round_\d+\.json)",
                    log_text)
                if evidence_matches:
                    terminal_evidence = evidence_matches[-1].replace(
                        "/", "\\")
                    terminal_name = terminal_evidence.rsplit("\\", 1)[-1]
                    try:
                        sftp.stat(terminal_evidence)
                    except OSError:
                        pass
                    else:
                        download_file(
                            root, terminal_evidence,
                            destination / terminal_name)
                archived.append(case_name)
        finally:
            sftp.close()
    finally:
        close_dual(hosts)
    print(json.dumps({
        "run_id": args.run_id,
        "archive_root": str(local_archive),
        "archived_count": len(archived),
        "archived": archived,
        "pending_count": len(pending),
        "pending": pending,
    }, ensure_ascii=False, indent=2))


def finalize_completed_case(args):
    """Validate and mark a finished root log after a validator-only failure."""
    safe = r"[A-Za-z0-9_.-]{8,128}"
    if not re.fullmatch(safe, args.run_id):
        raise ValueError(f"unsafe run id: {args.run_id!r}")
    if not re.fullmatch(safe, args.case_name):
        raise ValueError(f"unsafe case name: {args.case_name!r}")
    hosts = connect_dual(args)
    try:
        root = hosts["client8g"]
        remote_case = CLIENT_REPO + (
            r"\scripts\distributed_scripts\ggeur_hierarchical_3machine"
            r"\runs" + "\\" + args.run_id + "\\" + args.case_name)
        local_attempt = (
            REPO_ROOT / "exp" / "dual_finalize" / args.run_id /
            args.case_name)
        local_attempt.mkdir(parents=True, exist_ok=True)
        remote_log = remote_case + r"\logs\root.stdout.log"
        remote_cfg = remote_case + r"\configs\root_server.yaml"
        local_log = local_attempt / "root.stdout.log"
        local_cfg = local_attempt / "root_server.yaml"
        download_file(root, remote_log, local_log)
        download_file(root, remote_cfg, local_cfg)

        import subprocess
        import yaml
        config = yaml.safe_load(local_cfg.read_text(encoding="utf-8-sig"))
        rounds = int(config["federate"]["total_round_num"])
        last_round = max(0, rounds - 1)
        eval_frequency = int(config.get("eval", {}).get("freq", 1))
        eval_mode = str(config.get("ggeur", {}).get(
            "headonly_eval_mode", "server")).lower()
        terminal_client_eval_only = bool(config.get("ggeur", {}).get(
            "terminal_client_eval_only", False))
        records_per_eval = (
            2 if eval_mode == "both" and not terminal_client_eval_only else 1)
        expected_updates = 4 if args.case_name.startswith(
            "mdsent_") else 2
        terminal_local = None
        if eval_mode in {"client", "both"}:
            terminal_name = (
                f"client_model_accuracy_round_{last_round}.json")
            log_text = local_log.read_text(
                encoding="utf-8", errors="replace")
            evidence_matches = re.findall(
                r"Client model accuracy evidence written to\s+"
                r"([^\r\n]*?client_model_accuracy_round_\d+\.json)",
                log_text,
            )
            if not evidence_matches:
                raise RuntimeError(
                    "terminal evidence path is absent from root log")
            terminal_remote = evidence_matches[-1].replace("/", "\\")
            terminal_local = local_attempt / terminal_name
            download_file(root, terminal_remote, terminal_local)

        validation_local = local_attempt / "completion_validation.json"
        command = [
            sys.executable,
            str(THIS_DIR / "validate_case_completion.py"),
            str(local_log),
            "--output", str(validation_local),
            "--expected-rounds", str(rounds),
            "--expected-accuracy-rounds", str(last_round),
            "--eval-frequency", str(eval_frequency),
            "--accuracy-records-per-eval", str(records_per_eval),
            "--expected-updates", str(expected_updates),
        ]
        if terminal_client_eval_only:
            command.append("--terminal-client-eval-only")
        if terminal_local is not None:
            command.extend([
                "--client-evidence", str(terminal_local),
                "--expected-clients",
                str(int(config["federate"]["client_num"])),
            ])
        subprocess.run(command, check=True)
        summary_local = local_attempt / "accuracy_summary.json"
        subprocess.run([
            sys.executable, str(THIS_DIR / "summarize_accuracy.py"),
            str(local_log), "--output", str(summary_local),
        ], check=True, capture_output=True, text=True)

        remote_validation = remote_case + r"\completion_validation.json"
        remote_summary = remote_case + r"\accuracy_summary.json"
        upload_file(root, validation_local, remote_validation)
        upload_file(root, summary_local, remote_summary)
        if terminal_local is not None:
            upload_file(
                root, terminal_local,
                remote_case + "\\" + terminal_local.name)
        marker = {
            "run_id": args.run_id,
            "case": args.case_name,
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "validation": "completion_validation.json",
            "summary": "accuracy_summary.json",
            "finalized_after_validator_fix": True,
        }
        marker_bytes = json.dumps(
            marker, ensure_ascii=False).encode("utf-8")
        sftp = root.open_sftp()
        try:
            with sftp.file(remote_case + r"\.formal_complete", "wb") as h:
                h.write(marker_bytes)
        finally:
            sftp.close()
        result = {
            "status": 0,
            "stdout": (
                f"finalized={args.case_name}\n"
                f"validation={remote_validation}\n"
                f"summary={remote_summary}"),
            "stderr": "",
        }
    finally:
        close_dual(hosts)
    print(json.dumps({"client8g": result}, ensure_ascii=True, indent=2))


def runtime_status(args):
    template = r"""
$run = '__RUN__'
$computer = Get-CimInstance Win32_ComputerSystem
$os = Get-CimInstance Win32_OperatingSystem
$roles = @(Get-CimInstance Win32_Process | Where-Object {
  $_.CommandLine -like "*$run*" -and (
    $_.CommandLine -like '*federatedscope.main*' -or
    $_.CommandLine -like '*hierarchical_subserver.py*')
})
"computer=$env:COMPUTERNAME"
"logical_processors=$($computer.NumberOfLogicalProcessors)"
"memory_total_gib=$([math]::Round($computer.TotalPhysicalMemory / 1GB, 2))"
"memory_free_gib=$([math]::Round($os.FreePhysicalMemory * 1KB / 1GB, 2))"
"system_handle_count=$((Get-Process | Measure-Object HandleCount -Sum).Sum)"
"run_roles=$($roles.Count)"
"run_working_set_gib=$([math]::Round((($roles | Measure-Object WorkingSetSize -Sum).Sum) / 1GB, 2))"
$tasks = @(Get-ScheduledTask -TaskName "GGEUR-dual-$run-*" `
  -ErrorAction SilentlyContinue)
$tasks | ForEach-Object {
  $info = Get-ScheduledTaskInfo -TaskName $_.TaskName
  "task=$($_.TaskName) state=$($_.State) result=$($info.LastTaskResult)"
}
$roleTasks = @(Get-ScheduledTask -TaskName "GGEUR-$run-*" `
  -ErrorAction SilentlyContinue)
"role_tasks=$($roleTasks.Count)"
$roleTasks | Group-Object State | ForEach-Object {
  "role_task_state=$($_.Name) count=$($_.Count)"
}
$roleTasks | Sort-Object TaskName -Descending | Select-Object -First 4 |
  ForEach-Object {
    $info = Get-ScheduledTaskInfo -TaskName $_.TaskName
    "role_task=$($_.TaskName) state=$($_.State) result=$($info.LastTaskResult)"
  }
"""
    hosts = connect_dual(args)
    result = {}
    try:
        for name in ("client8g", "third"):
            result[name] = run(
                hosts[name], encoded_powershell(
                    template.replace("__RUN__", args.run_id)),
                timeout=120, check=False)
    finally:
        close_dual(hosts)
    print(json.dumps(result, ensure_ascii=True, indent=2))


def cache_audit(args):
    """Report feature-cache key coverage without contacting the 4090 host."""
    template = r"""
$repo = '__REPO__'
"computer=$env:COMPUTERNAME"
$model = Join-Path $repo `
  'pretrained_models\nlptown_bert_base_multilingual_uncased_senti'
"bert_model_exists=$(Test-Path -LiteralPath $model)"
if (Test-Path -LiteralPath $model) {
  $modelFiles = @(Get-ChildItem -LiteralPath $model -Recurse -File `
    -ErrorAction SilentlyContinue)
  "bert_model_files=$($modelFiles.Count)"
  "bert_model_bytes=$(($modelFiles | Measure-Object Length -Sum).Sum)"
}
$hfRoots = @(
  (Join-Path $env:USERPROFILE `
    '.cache\huggingface\hub\models--nlptown--bert-base-multilingual-uncased-sentiment'),
  (Join-Path $env:USERPROFILE `
    '.cache\huggingface\hub\models--nlptown--bert-base-multilingual-uncased-senti')
)
foreach ($hfRoot in $hfRoots) {
  if (Test-Path -LiteralPath $hfRoot) {
    $hfFiles = @(Get-ChildItem -LiteralPath $hfRoot -Recurse -File `
      -ErrorAction SilentlyContinue)
    "hf_model=$hfRoot files=$($hfFiles.Count) bytes=$(($hfFiles | Measure-Object Length -Sum).Sum)"
    Get-ChildItem -LiteralPath $hfRoot -Recurse -File |
      Where-Object { $_.Name -match 'config|token|vocab|model|pytorch|safetensors' } |
      Select-Object -First 20 | ForEach-Object {
        "hf_file=$($_.FullName) bytes=$($_.Length)"
      }
  }
}
$pythonCandidates = @(
  (Join-Path $repo '.venv_client_cpu\Scripts\python.exe'),
  'C:\Users\pc\miniconda3\envs\cerp\python.exe',
  'D:\ProgramData\anaconda3\envs\pi_fmd_gpu_py39\python.exe'
)
$py = $pythonCandidates | Where-Object { Test-Path -LiteralPath $_ } |
  Select-Object -First 1
"python=$py"
if ($py) {
  & $py -c "import importlib.util;print('transformers='+str(importlib.util.find_spec('transformers') is not None));print('torch='+str(importlib.util.find_spec('torch') is not None))"
}
$cacheDir = Join-Path $repo 'exp\distributed_feature_cache\mdsent_lstm'
$code = "import sys,numpy as np;d=np.load(sys.argv[1],allow_pickle=True);p=[str(x).replace('\\','/').casefold() for x in d['paths']];print('npz='+sys.argv[1]);print('count='+str(len(p)));print('feature_shape='+str(d['features'].shape));print('first='+repr(p[:5]));print('last='+repr(p[-5:]));print('books='+str(sum(x.startswith('books/') for x in p)));print('books_test_probe='+str(sum(x in {'books/negative.review:900','books/negative.review:258','books/positive.review:912'} for x in p)))"
if ($py -and (Test-Path -LiteralPath $cacheDir)) {
  Get-ChildItem -LiteralPath $cacheDir -Filter '*.npz' -File |
    Sort-Object Name | ForEach-Object { & $py -c $code $_.FullName }
}
"""
    hosts = connect_dual(args)
    result = {}
    try:
        for name, repo in (("client8g", CLIENT_REPO),
                           ("third", THIRD_REPO)):
            result[name] = run(
                hosts[name], encoded_powershell(
                    template.replace("__REPO__", repo)),
                timeout=180, check=False)
    finally:
        close_dual(hosts)
    print(json.dumps(result, ensure_ascii=True, indent=2))


def start_domainnet_train_cache(args):
    """Build a complete raw DomainNet training cache on the permitted GPU host."""
    groups = args.groups or ["domainnet_vit"]
    allowed = {"domainnet_vit", "domainnet_cnn", "domainnet_mixer"}
    if len(groups) != 1 or groups[0] not in allowed:
        raise ValueError(
            "start-domainnet-train-cache requires exactly one DomainNet group")
    group = groups[0]
    local_launcher = THIS_DIR / "prepare_feature_cache.ps1"
    remote_launcher = (THIRD_REPO +
                       r"\scripts\distributed_scripts\ggeur_hierarchical_3machine"
                       r"\prepare_feature_cache.ps1")
    task_name = f"GGEUR-domainnet-train-cache-{group}-third"
    script = rf"""
$ErrorActionPreference = 'Stop'
$repo = '{THIRD_REPO}'
$launcher = '{remote_launcher}'
$taskName = '{task_name}'
$state = Join-Path $repo 'exp\domainnet_train_cache_dual\{group}'
New-Item -ItemType Directory -Force -Path $state | Out-Null
$task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($task -and $task.State -eq 'Running') {{
  "already_running_task=$taskName"
  "state=$state"
  exit 0
}}
if ($task) {{
  $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
  $attempt = Join-Path $state ("attempt_" + $stamp)
  New-Item -ItemType Directory -Force -Path $attempt | Out-Null
  $logRoot = Join-Path $repo 'exp\cache_warmup\{group}'
  foreach ($name in @('prepare_feature_cache.log',
                       'prepare_feature_cache.log.stderr')) {{
    $source = Join-Path $logRoot $name
    if (Test-Path -LiteralPath $source) {{
      Copy-Item -LiteralPath $source -Destination $attempt -Force
    }}
  }}
  $info = Get-ScheduledTaskInfo -TaskName $taskName
  @(
    "preserved_at=$(Get-Date -Format o)",
    "task_state=$($task.State)",
    "last_result=$($info.LastTaskResult)"
  ) | Set-Content -LiteralPath (Join-Path $attempt 'attempt_status.txt')
  "preserved_attempt=$attempt"
  Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
}}
$powerShell = "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe"
$arguments = ('-NoProfile -NonInteractive -ExecutionPolicy Bypass ' +
  '-File "' + $launcher + '" ' +
  '-Group "{group}" ' +
  '-PythonBin "C:\Users\pc\miniconda3\envs\cerp\python.exe" ' +
  '-FeatureCacheRoot "C:/Users/pc/FederatedScope/exp/distributed_feature_cache" ' +
  '-DomainNetRoot "C:/Users/pc/FederatedScope/data/DomainNet" ' +
  '-DomainNetManifestPath "C:/Users/pc/FederatedScope/exp/distributed_manifests/domainnet_4domains/domainnet_manifest.json" ' +
  '-ClipModelPath "C:/Users/pc/FederatedScope/pretrained_models/ViT-B-16.pt" ' +
  '-MixerCheckpointPath "C:/Users/pc/FederatedScope/pretrained_models/mixer_b16_224_complete.pth"')
$action = New-ScheduledTaskAction -Execute $powerShell `
  -Argument $arguments -WorkingDirectory $repo
$identity = [Security.Principal.WindowsIdentity]::GetCurrent().Name
$principal = New-ScheduledTaskPrincipal -UserId $identity `
  -LogonType S4U -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet `
  -ExecutionTimeLimit (New-TimeSpan -Hours 4) `
  -MultipleInstances IgnoreNew -AllowStartIfOnBatteries `
  -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName $taskName -Action $action `
  -Principal $principal -Settings $settings -Force | Out-Null
Start-ScheduledTask -TaskName $taskName
"started_task=$taskName"
"state=$state"
"launcher=$launcher"
"cache=$(Join-Path $repo 'exp\distributed_feature_cache\{group}')"
"""
    hosts = connect_dual(args)
    try:
        upload_file(hosts["third"], local_launcher, remote_launcher)
        result = run(hosts["third"], encoded_powershell(script),
                     timeout=120, check=False)
    finally:
        close_dual(hosts)
    print(json.dumps({"third": result}, ensure_ascii=True, indent=2))


def domainnet_train_cache_status(args):
    """Report one DomainNet raw training-cache job and its output coverage."""
    groups = args.groups or ["domainnet_vit"]
    allowed = {"domainnet_vit", "domainnet_cnn", "domainnet_mixer"}
    if len(groups) != 1 or groups[0] not in allowed:
        raise ValueError(
            "domainnet-train-cache-status requires exactly one DomainNet group")
    group = groups[0]
    task_name = f"GGEUR-domainnet-train-cache-{group}-third"
    script = rf"""
$ErrorActionPreference = 'Continue'
$repo = '{THIRD_REPO}'
$group = '{group}'
$taskName = '{task_name}'
$task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($task) {{
  $info = Get-ScheduledTaskInfo -TaskName $taskName
  "task=$taskName state=$($task.State) result=$($info.LastTaskResult)"
}} else {{
  "task=$taskName state=MISSING"
}}
$logRoot = Join-Path $repo "exp\cache_warmup\$group"
foreach ($name in @('prepare_feature_cache.log',
                     'prepare_feature_cache.log.stderr')) {{
  $path = Join-Path $logRoot $name
  if (Test-Path -LiteralPath $path) {{
    $item = Get-Item -LiteralPath $path
    "log=$name bytes=$($item.Length) mtime=$($item.LastWriteTime.ToString('o'))"
    Get-Content -LiteralPath $path -Tail 20
  }}
}}
$cache = Join-Path $repo "exp\distributed_feature_cache\$group"
$marker = Join-Path $cache '.ggeur_feature_cache_ready.json'
"marker_exists=$(Test-Path -LiteralPath $marker)"
if (Test-Path -LiteralPath $marker) {{ Get-Content -LiteralPath $marker -Raw }}
Get-ChildItem -LiteralPath $cache -Filter '*.npz' -File `
  -ErrorAction SilentlyContinue | Sort-Object Name | ForEach-Object {{
    "cache_file=$($_.Name) bytes=$($_.Length) mtime=$($_.LastWriteTime.ToString('o'))"
  }}
"""
    hosts = connect_dual(args)
    try:
        result = run(hosts["third"], encoded_powershell(script),
                     timeout=180, check=False)
    finally:
        close_dual(hosts)
    print(json.dumps({"third": result}, ensure_ascii=True, indent=2))


def start_domainnet_train_cache_direct(args):
    """Replace the warm-up run with direct, verified batch extraction."""
    groups = args.groups or ["domainnet_vit"]
    allowed = {"domainnet_vit", "domainnet_cnn", "domainnet_mixer"}
    if len(groups) != 1 or groups[0] not in allowed:
        raise ValueError(
            "start-domainnet-train-cache-direct requires one DomainNet group")
    group = groups[0]
    uploads = (
        (THIS_DIR / "prepare_domainnet_train_cache.py",
         THIRD_REPO + r"\scripts\distributed_scripts\ggeur_hierarchical_3machine\prepare_domainnet_train_cache.py"),
        (THIS_DIR / "prepare_domainnet_eval_cache.py",
         THIRD_REPO + r"\scripts\distributed_scripts\ggeur_hierarchical_3machine\prepare_domainnet_eval_cache.py"),
        (THIS_DIR / "run_domainnet_train_cache.ps1",
         THIRD_REPO + r"\scripts\distributed_scripts\ggeur_hierarchical_3machine\run_domainnet_train_cache.ps1"),
    )
    task_name = f"GGEUR-domainnet-train-cache-direct-{group}-third"
    legacy_task = f"GGEUR-domainnet-train-cache-{group}-third"
    runner = uploads[2][1]
    script = rf"""
$ErrorActionPreference = 'Stop'
$repo = '{THIRD_REPO}'
$group = '{group}'
$taskName = '{task_name}'
$legacyTask = '{legacy_task}'
$runner = '{runner}'
$state = Join-Path $repo "exp\domainnet_train_cache_dual\$group"
New-Item -ItemType Directory -Force -Path $state | Out-Null
$active = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($active -and $active.State -eq 'Running') {{
  "already_running_task=$taskName"
  exit 0
}}
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$legacy = Get-ScheduledTask -TaskName $legacyTask -ErrorAction SilentlyContinue
if ($legacy) {{
  $attempt = Join-Path $state ("attempt_" + $stamp + "_warmup_replaced")
  New-Item -ItemType Directory -Force -Path $attempt | Out-Null
  if ($legacy.State -eq 'Running') {{
    Stop-ScheduledTask -TaskName $legacyTask
    foreach ($n in 1..60) {{
      if ((Get-ScheduledTask -TaskName $legacyTask).State -ne 'Running') {{ break }}
      Start-Sleep -Milliseconds 250
    }}
  }}
  $logRoot = Join-Path $repo "exp\cache_warmup\$group"
  foreach ($name in @('prepare_feature_cache.log',
                       'prepare_feature_cache.log.stderr')) {{
    $source = Join-Path $logRoot $name
    if (Test-Path -LiteralPath $source) {{
      Copy-Item -LiteralPath $source -Destination $attempt -Force
    }}
  }}
  @(
    "preserved_at=$(Get-Date -Format o)",
    "reason=replaced_slow_augmentation_warmup_with_direct_extractor",
    "task_state=$($legacy.State)"
  ) | Set-Content -LiteralPath (Join-Path $attempt 'attempt_status.txt')
  Unregister-ScheduledTask -TaskName $legacyTask -Confirm:$false
  "preserved_replaced_attempt=$attempt"
}}
if ($active) {{ Unregister-ScheduledTask -TaskName $taskName -Confirm:$false }}
$powerShell = "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe"
$arguments = ('-NoProfile -NonInteractive -ExecutionPolicy Bypass ' +
  '-File "' + $runner + '" -Group "' + $group + '" ' +
  '-Repo "' + $repo + '" ' +
  '-Python "C:/Users/pc/miniconda3/envs/cerp/python.exe"')
$action = New-ScheduledTaskAction -Execute $powerShell `
  -Argument $arguments -WorkingDirectory $repo
$identity = [Security.Principal.WindowsIdentity]::GetCurrent().Name
$principal = New-ScheduledTaskPrincipal -UserId $identity `
  -LogonType S4U -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet `
  -ExecutionTimeLimit (New-TimeSpan -Hours 4) `
  -MultipleInstances IgnoreNew -AllowStartIfOnBatteries `
  -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName $taskName -Action $action `
  -Principal $principal -Settings $settings -Force | Out-Null
Start-ScheduledTask -TaskName $taskName
"started_task=$taskName"
"state=$state"
"""
    hosts = connect_dual(args)
    try:
        for local_path, remote_path in uploads:
            upload_file(hosts["third"], local_path, remote_path)
        result = run(hosts["third"], encoded_powershell(script),
                     timeout=180, check=False)
    finally:
        close_dual(hosts)
    print(json.dumps({"third": result}, ensure_ascii=True, indent=2))


def domainnet_train_cache_direct_status(args):
    """Report direct DomainNet training-cache extraction progress."""
    groups = args.groups or ["domainnet_vit"]
    allowed = {"domainnet_vit", "domainnet_cnn", "domainnet_mixer"}
    if len(groups) != 1 or groups[0] not in allowed:
        raise ValueError(
            "domainnet-train-cache-direct-status requires one group")
    group = groups[0]
    task_name = f"GGEUR-domainnet-train-cache-direct-{group}-third"
    script = rf"""
$ErrorActionPreference = 'Continue'
$repo = '{THIRD_REPO}'
$group = '{group}'
$taskName = '{task_name}'
$task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($task) {{
  $info = Get-ScheduledTaskInfo -TaskName $taskName
  "task=$taskName state=$($task.State) result=$($info.LastTaskResult)"
}} else {{ "task=$taskName state=MISSING" }}
$state = Join-Path $repo "exp\domainnet_train_cache_dual\$group"
foreach ($name in @('direct.stdout.log', 'direct.stderr.log')) {{
  $path = Join-Path $state $name
  if (Test-Path -LiteralPath $path) {{
    $item = Get-Item -LiteralPath $path
    "log=$name bytes=$($item.Length) mtime=$($item.LastWriteTime.ToString('o'))"
    Get-Content -LiteralPath $path -Tail 24
  }}
}}
$cache = Join-Path $repo "exp\distributed_feature_cache\$group"
$marker = Join-Path $cache '.ggeur_feature_cache_ready.json'
"marker_exists=$(Test-Path -LiteralPath $marker)"
if (Test-Path -LiteralPath $marker) {{ Get-Content -LiteralPath $marker -Raw }}
Get-ChildItem -LiteralPath $cache -Filter '*.npz' -File `
  -ErrorAction SilentlyContinue | Where-Object {{ $_.Name -notlike '*_test_*' }} |
  Sort-Object Name | ForEach-Object {{
    "train_cache=$($_.Name) bytes=$($_.Length) mtime=$($_.LastWriteTime.ToString('o'))"
  }}
"""
    hosts = connect_dual(args)
    try:
        result = run(hosts["third"], encoded_powershell(script),
                     timeout=180, check=False)
    finally:
        close_dual(hosts)
    print(json.dumps({"third": result}, ensure_ascii=True, indent=2))


def sync_domainnet_train_cache(args):
    """Relay one verified raw training cache from the third host to 8G."""
    groups = args.groups or ["domainnet_vit"]
    allowed = {"domainnet_vit", "domainnet_cnn", "domainnet_mixer"}
    if len(groups) != 1 or groups[0] not in allowed:
        raise ValueError("sync-domainnet-train-cache requires one group")
    group = groups[0]
    remote_rel = rf"\exp\distributed_feature_cache\{group}"
    third_cache = THIRD_REPO + remote_rel
    client_cache = CLIENT_REPO + remote_rel
    marker_name = ".ggeur_feature_cache_ready.json"
    local_dir = (REPO_ROOT / "exp" / "dual_deploy" /
                 "domainnet_train_cache_transfer" / group)
    local_dir.mkdir(parents=True, exist_ok=True)

    def digest(path):
        value = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                value.update(chunk)
        return value.hexdigest()

    hosts = connect_dual(args)
    result = {"group": group, "files": []}
    try:
        third_sftp = hosts["third"].open_sftp()
        try:
            with third_sftp.file(third_cache + "\\" + marker_name,
                                 "rb") as handle:
                marker_bytes = handle.read()
            marker = json.loads(marker_bytes.decode("utf-8-sig"))
        finally:
            third_sftp.close()
        if (marker.get("group") != group or
                marker.get("require_complete_feature_cache") is not True or
                set(marker.get("domains") or {}) !=
                {"clipart", "painting", "real", "sketch"}):
            raise RuntimeError(f"invalid third-host completion marker: {marker}")

        expected = {}
        for domain in ("clipart", "painting", "real", "sketch"):
            item = marker["domains"][domain]
            filename = Path(str(item["path"]).replace("\\", "/")).name
            if not re.fullmatch(
                    rf"domainnet_{domain}_[A-Za-z0-9_]+_d\d+\.npz",
                    filename):
                raise RuntimeError(f"unsafe cache filename: {filename!r}")
            expected[filename] = {
                "sha256": str(item["sha256"]).lower(),
                "bytes": int(item["bytes"]),
            }
        marker_path = local_dir / marker_name
        marker_path.write_bytes(marker_bytes)
        expected[marker_name] = {
            "sha256": digest(marker_path), "bytes": len(marker_bytes)}

        for filename, item in expected.items():
            local_path = local_dir / filename
            if filename != marker_name:
                if (not local_path.is_file() or
                        local_path.stat().st_size != item["bytes"] or
                        digest(local_path) != item["sha256"]):
                    download_file(hosts["third"],
                                  third_cache + "\\" + filename,
                                  local_path)
            if (local_path.stat().st_size != item["bytes"] or
                    digest(local_path) != item["sha256"]):
                raise RuntimeError(f"local relay verification failed: {filename}")
            result["files"].append({"name": filename, **item})

        stamp = time.strftime("%Y%m%d_%H%M%S")
        client_sftp = hosts["client8g"].open_sftp()
        try:
            for filename in expected:
                client_sftp.put(
                    str(local_dir / filename),
                    client_cache + "\\" + filename + ".incoming_" + stamp)
        finally:
            client_sftp.close()

        expected_ps = ";".join(
            f"'{name}'='{item['sha256']}'"
            for name, item in expected.items())
        names_ps = ",".join(f"'{name}'" for name in expected)
        script = rf"""
$ErrorActionPreference = 'Stop'
$cache = '{client_cache}'
$stamp = '{stamp}'
$expected = @{{{expected_ps}}}
$names = @({names_ps})
foreach ($name in $names) {{
  $incoming = Join-Path $cache ($name + '.incoming_' + $stamp)
  if (-not (Test-Path -LiteralPath $incoming)) {{
    throw "missing incoming cache: $incoming"
  }}
  $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $incoming).Hash.ToLower()
  if ($actual -ne $expected[$name]) {{
    throw "incoming SHA256 mismatch: $name actual=$actual expected=$($expected[$name])"
  }}
}}
$attempt = Join-Path '{CLIENT_REPO}' `
  ('exp\domainnet_train_cache_attempts\' + $stamp + '_{group}_previous')
New-Item -ItemType Directory -Force -Path $attempt | Out-Null
foreach ($name in $names) {{
  $target = Join-Path $cache $name
  if (Test-Path -LiteralPath $target) {{
    Move-Item -LiteralPath $target -Destination $attempt -Force
  }}
}}
foreach ($name in $names) {{
  $incoming = Join-Path $cache ($name + '.incoming_' + $stamp)
  Move-Item -LiteralPath $incoming -Destination (Join-Path $cache $name) -Force
}}
"installed_group={group}"
"preserved_previous=$attempt"
foreach ($name in $names) {{
  $target = Join-Path $cache $name
  $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $target).Hash.ToLower()
  "installed_file=$name bytes=$((Get-Item -LiteralPath $target).Length) sha256=$hash"
}}
"""
        result["client8g"] = run(
            hosts["client8g"], encoded_powershell(script),
            timeout=300, check=False)
        if result["client8g"].get("status") != 0:
            raise RuntimeError(
                f"8G cache install failed: {result['client8g']}")
    finally:
        close_dual(hosts)
    print(json.dumps(result, ensure_ascii=True, indent=2))


def validate_domainnet_train_cache_8g(args):
    """Verify a synced DomainNet cache and bind its marker to the 8G host."""
    groups = args.groups or ["domainnet_vit"]
    if len(groups) != 1 or not groups[0].startswith("domainnet_"):
        raise RuntimeError(
            "validate-domainnet-train-cache-8g requires one DomainNet group")
    group = groups[0]
    client8g = key_client(
        CLIENT_IPV6, 22, "fsuser", args.key, bind=CLIENT_BIND_IPV6)
    try:
        script = rf'''
$ErrorActionPreference = 'Stop'
$group = '{group}'
$cache = Join-Path '{CLIENT_REPO}' "exp\distributed_feature_cache\$group"
$marker = Join-Path $cache '.ggeur_feature_cache_ready.json'
$state = Get-Content -LiteralPath $marker -Raw | ConvertFrom-Json
if ([string]$state.group -ne $group) {{
  throw "marker group mismatch: $($state.group)"
}}
if (-not $state.require_complete_feature_cache) {{
  throw 'marker does not require a complete feature cache'
}}
foreach ($domain in @('clipart', 'painting', 'real', 'sketch')) {{
  $entry = $state.domains.$domain
  if ($null -eq $entry) {{ throw "missing marker domain: $domain" }}
  $name = Split-Path -Leaf ([string]$entry.path)
  $path = Join-Path $cache $name
  $item = Get-Item -LiteralPath $path
  $sha = (Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash.ToLowerInvariant()
  if ([int64]$item.Length -ne [int64]$entry.bytes) {{
    throw "byte mismatch for $domain"
  }}
  if ($sha -ne ([string]$entry.sha256).ToLowerInvariant()) {{
    throw "sha256 mismatch for $domain"
  }}
  "verified=$domain count=$($entry.count) bytes=$($item.Length) sha256=$sha"
}}
$state | Add-Member -NotePropertyName validated_on `
  -NotePropertyValue $env:COMPUTERNAME -Force
$state | Add-Member -NotePropertyName validated_at `
  -NotePropertyValue (Get-Date -Format o) -Force
$state | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $marker -Encoding UTF8
"marker=$marker validated_on=$env:COMPUTERNAME require_complete_feature_cache=$($state.require_complete_feature_cache)"
'''
        result = run(client8g, encoded_powershell(script), timeout=900)
    finally:
        client8g.close()
    print(json.dumps({"client8g": result}, ensure_ascii=True, indent=2))


def validate_domainnet_train_cache_both(args):
    """Bind one verified DomainNet cache contract to both client hosts."""
    groups = args.groups or ["domainnet_vit"]
    if len(groups) != 1 or not groups[0].startswith("domainnet_"):
        raise RuntimeError(
            "validate-domainnet-train-cache-both requires one DomainNet group")
    group = groups[0]
    hosts = connect_dual(args)
    results = {"group": group}
    try:
        marker_path = (
            f"{CLIENT_REPO}\\exp\\distributed_feature_cache\\{group}\\"
            ".ggeur_feature_cache_ready.json")
        sftp = hosts["client8g"].open_sftp()
        try:
            with sftp.file(marker_path, "rb") as handle:
                marker_bytes = handle.read()
        finally:
            sftp.close()
        marker = json.loads(marker_bytes.decode("utf-8-sig"))
        if (marker.get("group") != group or
                marker.get("require_complete_feature_cache") is not True):
            raise RuntimeError(f"invalid source marker: {marker}")
        expected_counts = {
            "clipart": 34183,
            "painting": 53031,
            "real": 122728,
            "sketch": 49270,
        }
        if set(marker.get("domains") or {}) != set(expected_counts):
            counts_ps = ";".join(
                f"'{domain}'={count}"
                for domain, count in expected_counts.items())
            inventory_script = rf'''
$ErrorActionPreference = 'Stop'
$cache = Join-Path '{CLIENT_REPO}' `
  'exp\distributed_feature_cache\{group}'
$counts = @{{{counts_ps}}}
$domains = [ordered]@{{}}
foreach ($domain in @('clipart', 'painting', 'real', 'sketch')) {{
  $files = @(Get-ChildItem -LiteralPath $cache `
    -Filter "domainnet_${{domain}}_*_d1024.npz" -File |
    Where-Object {{ $_.Name -notlike '*_test_*' }})
  if ($files.Count -ne 1) {{
    throw "expected one raw training cache for $domain, got $($files.Count)"
  }}
  $file = $files[0]
  $domains[$domain] = [ordered]@{{
    path = $file.FullName
    count = [int]$counts[$domain]
    bytes = [int64]$file.Length
    sha256 = (Get-FileHash -Algorithm SHA256 `
      -LiteralPath $file.FullName).Hash.ToLowerInvariant()
  }}
}}
([ordered]@{{domains=$domains}} | ConvertTo-Json -Depth 10 -Compress)
'''
            inventory_result = run(
                hosts["client8g"], encoded_powershell(inventory_script),
                timeout=1200)
            inventory = json.loads(inventory_result["stdout"].strip())
            marker["domains"] = inventory["domains"]
            marker["cache_files"] = 4
        marker_bytes = (json.dumps(
            marker, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        marker_b64 = base64.b64encode(marker_bytes).decode("ascii")
        for name, repo in (("client8g", CLIENT_REPO),
                           ("third", THIRD_REPO)):
            script = rf'''
$ErrorActionPreference = 'Stop'
$group = '{group}'
$cache = Join-Path '{repo}' "exp\distributed_feature_cache\$group"
$marker = Join-Path $cache '.ggeur_feature_cache_ready.json'
$json = [Text.Encoding]::UTF8.GetString(
  [Convert]::FromBase64String('{marker_b64}'))
$state = $json | ConvertFrom-Json
foreach ($domain in @('clipart', 'painting', 'real', 'sketch')) {{
  $entry = $state.domains.$domain
  $fileName = Split-Path -Leaf ([string]$entry.path)
  $path = Join-Path $cache $fileName
  $item = Get-Item -LiteralPath $path
  $sha = (Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash.ToLowerInvariant()
  if ([int64]$item.Length -ne [int64]$entry.bytes) {{
    throw "byte mismatch for $domain"
  }}
  if ($sha -ne ([string]$entry.sha256).ToLowerInvariant()) {{
    throw "sha256 mismatch for $domain"
  }}
  "verified=$domain count=$($entry.count) bytes=$($item.Length) sha256=$sha"
}}
$state | Add-Member -NotePropertyName validated_on `
  -NotePropertyValue $env:COMPUTERNAME -Force
$state | Add-Member -NotePropertyName validated_at `
  -NotePropertyValue (Get-Date -Format o) -Force
$state | ConvertTo-Json -Depth 20 | Set-Content `
  -LiteralPath $marker -Encoding UTF8
"marker=$marker validated_on=$env:COMPUTERNAME client_num=$($state.client_num)"
'''
            results[name] = run(hosts[name], encoded_powershell(script),
                                timeout=1200, check=False)
            if results[name].get("status") != 0:
                raise RuntimeError(
                    f"{name} DomainNet cache validation failed: "
                    f"{results[name]}")
    finally:
        close_dual(hosts)
    print(json.dumps(results, ensure_ascii=True, indent=2))


def sync_domainnet_train_cache_8g_to_third(args):
    """Relay a verified raw DomainNet training cache from 8G to third."""
    groups = args.groups or ["domainnet_cnn"]
    if len(groups) != 1 or groups[0] != "domainnet_cnn":
        raise RuntimeError(
            "sync-domainnet-train-cache-8g-to-third currently requires "
            "domainnet_cnn")
    group = groups[0]
    expected_counts = {
        "clipart": 34183,
        "painting": 53031,
        "real": 122728,
        "sketch": 49270,
    }
    counts_ps = ";".join(
        f"'{domain}'={count}" for domain, count in expected_counts.items())
    source_inventory = rf'''
$ErrorActionPreference = 'Stop'
$cache = Join-Path '{CLIENT_REPO}' `
  'exp\distributed_feature_cache\{group}'
$counts = @{{{counts_ps}}}
$domains = [ordered]@{{}}
foreach ($domain in @('clipart', 'painting', 'real', 'sketch')) {{
  $files = @(Get-ChildItem -LiteralPath $cache `
    -Filter "domainnet_${{domain}}_*_d1024.npz" -File |
    Where-Object {{ $_.Name -notlike '*_test_*' }})
  if ($files.Count -ne 1) {{
    throw "expected one raw training cache for $domain, got $($files.Count)"
  }}
  $file = $files[0]
  $domains[$domain] = [ordered]@{{
    path = $file.FullName
    name = $file.Name
    count = [int]$counts[$domain]
    dim = 1024
    bytes = [int64]$file.Length
    sha256 = (Get-FileHash -Algorithm SHA256 `
      -LiteralPath $file.FullName).Hash.ToLowerInvariant()
  }}
}}
([ordered]@{{domains=$domains}} | ConvertTo-Json -Depth 10 -Compress)
'''
    relay = (REPO_ROOT / "exp" / "dual_deploy" /
             "domainnet_train_cache_8g_to_third" / group)
    relay.mkdir(parents=True, exist_ok=True)

    def digest(path):
        value = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                value.update(chunk)
        return value.hexdigest()

    hosts = connect_dual(args)
    result = {"group": group, "files": []}
    try:
        inventory_result = run(
            hosts["client8g"], encoded_powershell(source_inventory),
            timeout=1200)
        inventory = json.loads(inventory_result["stdout"].strip())
        source_sftp = hosts["client8g"].open_sftp()
        target_sftp = hosts["third"].open_sftp()
        try:
            for domain, item in inventory["domains"].items():
                local = relay / item["name"]
                if (not local.is_file() or
                        local.stat().st_size != int(item["bytes"]) or
                        digest(local) != item["sha256"]):
                    partial = Path(str(local) + ".part")
                    source_sftp.get(item["path"], str(partial))
                    partial.replace(local)
                if (local.stat().st_size != int(item["bytes"]) or
                        digest(local) != item["sha256"]):
                    raise RuntimeError(
                        f"local relay verification failed for {domain}")
                remote = (f"{THIRD_REPO}\\exp\\distributed_feature_cache\\"
                          f"{group}\\{item['name']}")
                incoming = remote + ".incoming_20260813"
                try:
                    remote_size = target_sftp.stat(incoming).st_size
                except (FileNotFoundError, IOError, OSError):
                    remote_size = -1
                if remote_size != int(item["bytes"]):
                    target_sftp.put(str(local), incoming)
                result["files"].append({
                    "domain": domain,
                    "name": item["name"],
                    "bytes": int(item["bytes"]),
                    "sha256": item["sha256"],
                    "incoming": incoming,
                })
        finally:
            source_sftp.close()
            target_sftp.close()

        expected_ps = ";".join(
            f"'{item['name']}'='{item['sha256']}'"
            for item in result["files"])
        bytes_ps = ";".join(
            f"'{item['name']}'={item['bytes']}"
            for item in result["files"])
        marker = {
            "group": group,
            "method": "verified_cache_relay",
            "client_num": 60,
            "cache_files": 4,
            "require_complete_feature_cache": True,
            "domains": inventory["domains"],
        }
        manifest_path = relay / "install_manifest.json"
        manifest_path.write_text(
            json.dumps(marker, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        installer_path = THIS_DIR / "install_domainnet_train_cache.py"
        remote_manifest = (
            f"{THIRD_REPO}\\exp\\distributed_feature_cache\\{group}\\"
            ".install_manifest_20260813.json")
        remote_installer = (
            f"{THIRD_REPO}\\scripts\\distributed_scripts\\"
            "ggeur_hierarchical_3machine\\"
            "install_domainnet_train_cache.py")
        upload_file(hosts["third"], manifest_path, remote_manifest)
        upload_file(hosts["third"], installer_path, remote_installer)
        install = rf'''
$ErrorActionPreference = 'Stop'
$cache = Join-Path '{THIRD_REPO}' `
  'exp\distributed_feature_cache\{group}'
$python = 'C:\Users\pc\miniconda3\envs\cerp\python.exe'
& $python '{remote_installer}' --cache-dir $cache `
  --manifest '{remote_manifest}'
if ($LASTEXITCODE -ne 0) {{
  throw "cache installer failed: $LASTEXITCODE"
}}
'''
        result["third"] = run(
            hosts["third"], encoded_powershell(install), timeout=1800,
            check=False)
        if result["third"].get("status") != 0:
            print(json.dumps(result, ensure_ascii=True, indent=2))
            raise RuntimeError("target cache install failed; see result above")
    finally:
        close_dual(hosts)
    print(json.dumps(result, ensure_ascii=True, indent=2))


def sync_domainnet_original_cnn_cache(args):
    """Relay the previously validated original-4 CNN cache to the third host."""
    def file_digest(path):
        value = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                value.update(chunk)
        return value.hexdigest()

    relative = (r"exp\distributed_feature_cache_domainnet_original_20260804"
                r"\domainnet_cnn")
    local_archive = (REPO_ROOT / "exp" / "dual_deploy" /
                     "domainnet_original4_cnn_cache_to_third.tar")
    remote8 = (CLIENT_REPO +
               r"\exp\dual_deploy\domainnet_original4_cnn_cache_to_third.tar")
    remote3 = (THIRD_REPO +
               r"\exp\dual_deploy\domainnet_original4_cnn_cache_to_third.tar")
    create = rf'''
$ErrorActionPreference = 'Stop'
$repo = '{CLIENT_REPO}'
$archive = '{remote8}'
$cache = Join-Path $repo '{relative}'
if (-not (Test-Path -LiteralPath $cache)) {{
  throw "validated DomainNet original cache is missing: $cache"
}}
New-Item -ItemType Directory -Force -Path (Split-Path $archive) | Out-Null
& tar.exe -cf $archive -C $repo '{relative}'
if ($LASTEXITCODE -ne 0) {{ throw "archive creation failed: $LASTEXITCODE" }}
$item = Get-Item -LiteralPath $archive
$sha = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
"archive=$archive bytes=$($item.Length) sha256=$sha"
'''
    prepare3 = rf'''
$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Force -Path `
  '{THIRD_REPO}\exp\dual_deploy' | Out-Null
'''
    install3 = rf'''
$ErrorActionPreference = 'Stop'
$repo = '{THIRD_REPO}'
$archive = '{remote3}'
& tar.exe -xf $archive -C $repo
if ($LASTEXITCODE -ne 0) {{ throw "archive extraction failed: $LASTEXITCODE" }}
$cache = Join-Path $repo '{relative}'
$files = @(Get-ChildItem -LiteralPath $cache -File -Filter '*.npz')
"installed_cache=$cache npz_files=$($files.Count)"
'''
    hosts = connect_dual(args)
    result = {}
    try:
        result["create"] = run(
            hosts["client8g"], encoded_powershell(create),
            timeout=1800, check=False)
        if result["create"].get("status") != 0:
            raise RuntimeError("original cache archive creation failed")
        download_file(hosts["client8g"], remote8, local_archive)
        result["local_archive"] = {
            "path": str(local_archive),
            "bytes": local_archive.stat().st_size,
            "sha256": file_digest(local_archive),
        }
        run(hosts["third"], encoded_powershell(prepare3), timeout=120)
        upload_file(hosts["third"], local_archive, remote3)
        result["install"] = run(
            hosts["third"], encoded_powershell(install3),
            timeout=1800, check=False)
        if result["install"].get("status") != 0:
            raise RuntimeError("original cache install failed")
    finally:
        close_dual(hosts)
    print(json.dumps(result, ensure_ascii=True, indent=2))


def prepare_mdsent_cache(args):
    """Start an auditable cache-completion job on the 8G management host."""
    script = rf"""
$ErrorActionPreference = 'Stop'
$repo = '{CLIENT_REPO}'
$py = Join-Path $repo '.venv_client_cpu\Scripts\python.exe'
$launcher = Join-Path $repo `
  'scripts\distributed_scripts\ggeur_hierarchical_3machine\prepare_mdsent_eval_cache.ps1'
$state = Join-Path $repo 'exp\distributed_feature_cache\mdsent_cache_prepare'
New-Item -ItemType Directory -Force -Path $state | Out-Null
$stdout = Join-Path $state 'prepare.stdout.log'
$stderr = Join-Path $state 'prepare.stderr.log'
$taskName = 'GGEUR-prepare-mdsent-eval-cache'
$task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($task -and $task.State -eq 'Running') {{
  "already_running_task=$taskName"
}} else {{
  if ($task) {{
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
  }}
  $powerShell = "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe"
  $arguments = ('-NoProfile -NonInteractive -ExecutionPolicy Bypass ' +
    '-File "' + $launcher + '" -Repo "' + $repo +
    '" -Python "' + $py + '"')
  $action = New-ScheduledTaskAction -Execute $powerShell `
    -Argument $arguments -WorkingDirectory $repo
  $identity = [Security.Principal.WindowsIdentity]::GetCurrent().Name
  $principal = New-ScheduledTaskPrincipal -UserId $identity `
    -LogonType S4U -RunLevel Limited
  $settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Hours 6) `
    -MultipleInstances IgnoreNew -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries
  Register-ScheduledTask -TaskName $taskName -Action $action `
    -Principal $principal -Settings $settings -Force | Out-Null
  Start-ScheduledTask -TaskName $taskName
  "started_task=$taskName"
}}
"stdout=$stdout"
"stderr=$stderr"
"""
    hosts = connect_dual(args)
    try:
        result = run(hosts["client8g"], encoded_powershell(script),
                     timeout=120, check=False)
    finally:
        close_dual(hosts)
    print(json.dumps({"client8g": result}, ensure_ascii=True, indent=2))


def mdsent_cache_status(args):
    template = r"""
$repo = '__REPO__'
$state = Join-Path $repo 'exp\distributed_feature_cache\mdsent_cache_prepare'
$active = @(Get-CimInstance Win32_Process | Where-Object {
  $_.CommandLine -like '*prepare_mdsent_eval_cache.py*'
})
"active_pids=$($active.ProcessId -join ',')"
$taskName = 'GGEUR-prepare-mdsent-eval-cache'
$task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($task) {
  $info = Get-ScheduledTaskInfo -TaskName $taskName
  "task=$taskName state=$($task.State) result=$($info.LastTaskResult) last=$($info.LastRunTime.ToString('o'))"
}
foreach ($name in @('prepare.stdout.log', 'prepare.stderr.log',
                    'launch_audit.tsv', 'exit_code.txt')) {
  $path = Join-Path $state $name
  if (Test-Path -LiteralPath $path) {
    $item = Get-Item -LiteralPath $path
    "file=$name bytes=$($item.Length) mtime=$($item.LastWriteTime.ToString('o'))"
    Get-Content -LiteralPath $path -Tail 30
  }
}
$marker = Join-Path $repo `
  'exp\distributed_feature_cache\mdsent_eval_cache_completion.json'
"marker_exists=$(Test-Path -LiteralPath $marker)"
if (Test-Path -LiteralPath $marker) {
  Get-Content -LiteralPath $marker -Raw
}
"""
    hosts = connect_dual(args)
    try:
        result = run(
            hosts["client8g"], encoded_powershell(
                template.replace("__REPO__", CLIENT_REPO)),
            timeout=120, check=False)
    finally:
        close_dual(hosts)
    print(json.dumps({"client8g": result}, ensure_ascii=True, indent=2))


def pause_queue(args):
    """Stop only the dual queue guards for one run, preserving all tasks and logs."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]{8,128}", args.run_id):
        raise ValueError(f"unsafe run id: {args.run_id!r}")
    template = r"""
$ErrorActionPreference = 'Continue'
$repo = '__REPO__'
$run = '__RUN__'
$stateDir = Join-Path $repo `
  "scripts\distributed_scripts\ggeur_hierarchical_3machine\runs\$run\queue_state"
New-Item -ItemType Directory -Force -Path $stateDir | Out-Null
$audit = Join-Path $stateDir 'queue_pause_audit.tsv'
$timestamp = Get-Date -Format o
$tasks = @(Get-ScheduledTask -TaskName "GGEUR-dual-$run-*" `
  -ErrorAction SilentlyContinue)
$running = @($tasks | Where-Object { $_.State -eq 'Running' })
foreach ($task in $running) {
  Stop-ScheduledTask -TaskName $task.TaskName -ErrorAction SilentlyContinue
  Add-Content -LiteralPath $audit -Value `
    "$timestamp`tSTOP_QUEUE_TASK`t$($task.TaskName)"
}
"matched_queue_tasks=$($tasks.Count)"
"stopped_running_queue_tasks=$($running.Count)"
"audit=$audit"
"""
    hosts = connect_dual(args)
    result = {}
    try:
        for name, repo in (("client8g", CLIENT_REPO),
                           ("third", THIRD_REPO)):
            result[name] = run(
                hosts[name], encoded_powershell(
                    template.replace("__REPO__", repo).replace(
                        "__RUN__", args.run_id)),
                timeout=120, check=False)
    finally:
        close_dual(hosts)
    print(json.dumps(result, ensure_ascii=True, indent=2))


def cleanup_failed_run(args):
    """Stop only processes/tasks belonging to one failed dual run."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]{8,128}", args.run_id):
        raise ValueError(f"unsafe run id: {args.run_id!r}")
    template = r"""
$ErrorActionPreference = 'Continue'
$repo = '__REPO__'
$run = '__RUN__'
$runNeedle = "\runs\$run\"
$auditDir = Join-Path $repo `
  "scripts\distributed_scripts\ggeur_hierarchical_3machine\runs\$run\queue_state"
New-Item -ItemType Directory -Force -Path $auditDir | Out-Null
$audit = Join-Path $auditDir 'cleanup_audit.tsv'
$timestamp = Get-Date -Format o
$tasks = @(Get-ScheduledTask -ErrorAction SilentlyContinue | Where-Object {
  $_.TaskName -like "GGEUR-$run-*" -or
  $_.TaskName -like "GGEUR-dual-$run-*"
})
$processes = @(Get-CimInstance Win32_Process | Where-Object {
  $_.CommandLine -and $_.CommandLine.Contains($runNeedle)
})
"matched_tasks=$($tasks.Count)"
"matched_processes=$($processes.Count)"
Add-Content -LiteralPath $audit -Value `
  "$timestamp`tMATCHED`ttasks=$($tasks.Count)`tprocesses=$($processes.Count)"
foreach ($task in $tasks) {
  Stop-ScheduledTask -TaskName $task.TaskName -ErrorAction SilentlyContinue
}
Start-Sleep -Seconds 2
foreach ($original in $processes) {
  $current = Get-CimInstance Win32_Process -Filter `
    "ProcessId=$($original.ProcessId)" -ErrorAction SilentlyContinue
  if ($current -and $current.CommandLine -and
      $current.CommandLine.Contains($runNeedle)) {
    Stop-Process -Id $current.ProcessId -Force -ErrorAction SilentlyContinue
    Add-Content -LiteralPath $audit -Value `
      "$timestamp`tSTOP_PID`t$($current.ProcessId)"
  }
}
foreach ($task in $tasks) {
  Unregister-ScheduledTask -TaskName $task.TaskName -Confirm:$false `
    -ErrorAction SilentlyContinue
  Add-Content -LiteralPath $audit -Value `
    "$timestamp`tUNREGISTER_TASK`t$($task.TaskName)"
}
"cleanup_complete=$run"
"audit=$audit"
"""
    hosts = connect_dual(args)
    result = {}
    try:
        for name, repo in (("client8g", CLIENT_REPO),
                           ("third", THIRD_REPO)):
            result[name] = run(
                hosts[name], encoded_powershell(
                    template.replace("__REPO__", repo).replace(
                        "__RUN__", args.run_id)),
                timeout=300, check=False)
    finally:
        close_dual(hosts)
    print(json.dumps(result, ensure_ascii=True, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("probe", "audit-forbidden-legacy",
                 "quarantine-forbidden-legacy", "deploy", "sync-mdsent", "transition",
                 "configure-firewall", "start", "status", "inspect",
                 "inspect-root",
                 "domainnet-cache-status",
                 "compare-domainnet-cnn-caches",
                 "start-domainnet-resource-download",
                 "domainnet-resource-download-status",
                 "install-domainnet-resource-handoff",
                 "domainnet-resource-handoff-status",
                 "diagnose-domainnet-resource-download",
                 "probe-domainnet-resource-task-network",
                 "probe-domainnet-resource-lengths",
                 "pause-domainnet-resource-download",
                 "sync-domainnet-verification-resources",
                 "verify-domainnet-extractors",
                 "probe-domainnet-extractor-resources",
                 "start-domainnet-eval-cache",
                 "domainnet-eval-cache-status",
                 "sync-domainnet-eval-cache",
                 "prepare-domainnet-terminal-shards",
                 "prepare-domainnet-original-terminal-shards-8g",
                 "resume-clients", "resume-clients-8g", "resume-subservers",
                 "restart-client-driver", "diagnose-clients-8g",
                 "diagnose-active-domainnet-8g",
                 "recover-clients-8g", "stop-failed-case", "progress",
                 "archive-completed", "finalize-completed-case", "runtime",
                 "cache-audit",
                 "start-domainnet-train-cache",
                 "domainnet-train-cache-status",
                 "start-domainnet-train-cache-direct",
                 "domainnet-train-cache-direct-status",
                 "validate-domainnet-train-cache-8g",
                 "validate-domainnet-train-cache-both",
                 "sync-domainnet-train-cache-8g-to-third",
                 "sync-domainnet-original-cnn-cache",
                 "sync-domainnet-train-cache",
                 "prepare-mdsent-cache",
                 "mdsent-cache-status", "pause-queue", "cleanup-failed-run"))
    parser.add_argument(
        "--key", default=os.path.expanduser(r"~\.ssh\id_ed25519"))
    parser.add_argument(
        "--third-password", default=os.environ.get("GGEUR_THIRD_PASSWORD"))
    parser.add_argument("--bundle", default="")
    parser.add_argument(
        "--transfer-file",
        default=str(REPO_ROOT / "exp" / "dual_deploy" /
                    "dual_mdsent_resources.tar"))
    parser.add_argument("--run-id", default="remaining_dual_simulated_20260802_v1")
    parser.add_argument("--case-name", default="mdsent_lstm_ggeur")
    parser.add_argument(
        "--failure-reason",
        default="unspecified_formal_case_failure")
    parser.add_argument("--groups", nargs="*", default=[])
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "probe":
        probe(args)
    elif args.command == "audit-forbidden-legacy":
        audit_forbidden_legacy(args)
    elif args.command == "quarantine-forbidden-legacy":
        quarantine_forbidden_legacy_from_file(args)
    elif args.command == "deploy":
        deploy(args)
    elif args.command == "sync-mdsent":
        sync_mdsent(args)
    elif args.command == "transition":
        transition(args)
    elif args.command == "configure-firewall":
        configure_firewall(args)
    elif args.command == "start":
        start(args)
    elif args.command == "status":
        status(args)
    elif args.command == "inspect":
        inspect_run(args)
    elif args.command == "inspect-root":
        inspect_root(args)
    elif args.command == "domainnet-cache-status":
        domainnet_cache_status(args)
    elif args.command == "compare-domainnet-cnn-caches":
        compare_domainnet_cnn_caches(args)
    elif args.command == "start-domainnet-resource-download":
        start_domainnet_resource_download(args)
    elif args.command == "domainnet-resource-download-status":
        domainnet_resource_download_status(args)
    elif args.command == "install-domainnet-resource-handoff":
        install_domainnet_resource_handoff(args)
    elif args.command == "domainnet-resource-handoff-status":
        domainnet_resource_handoff_status(args)
    elif args.command == "diagnose-domainnet-resource-download":
        diagnose_domainnet_resource_download(args)
    elif args.command == "probe-domainnet-resource-task-network":
        probe_domainnet_resource_task_network(args)
    elif args.command == "probe-domainnet-resource-lengths":
        probe_domainnet_resource_lengths(args)
    elif args.command == "pause-domainnet-resource-download":
        pause_domainnet_resource_download(args)
    elif args.command == "sync-domainnet-verification-resources":
        sync_domainnet_verification_resources(args)
    elif args.command == "verify-domainnet-extractors":
        verify_domainnet_extractors(args)
    elif args.command == "probe-domainnet-extractor-resources":
        probe_domainnet_extractor_resources(args)
    elif args.command == "start-domainnet-eval-cache":
        start_domainnet_eval_cache(args)
    elif args.command == "domainnet-eval-cache-status":
        domainnet_eval_cache_status(args)
    elif args.command == "sync-domainnet-eval-cache":
        sync_domainnet_eval_cache(args)
    elif args.command == "prepare-domainnet-terminal-shards":
        prepare_domainnet_terminal_shards(args)
    elif args.command == "prepare-domainnet-original-terminal-shards-8g":
        prepare_domainnet_original_terminal_shards_8g(args)
    elif args.command == "resume-clients":
        resume_third_clients(args)
    elif args.command == "resume-clients-8g":
        resume_8g_clients(args)
    elif args.command == "resume-subservers":
        resume_third_subservers(args)
    elif args.command == "restart-client-driver":
        restart_third_client_driver(args)
    elif args.command == "diagnose-clients-8g":
        diagnose_8g_clients(args)
    elif args.command == "diagnose-active-domainnet-8g":
        diagnose_active_domainnet_8g(args)
    elif args.command == "recover-clients-8g":
        recover_8g_clients(args)
    elif args.command == "stop-failed-case":
        stop_failed_case(args)
    elif args.command == "progress":
        progress_run(args)
    elif args.command == "archive-completed":
        archive_completed(args)
    elif args.command == "finalize-completed-case":
        finalize_completed_case(args)
    elif args.command == "runtime":
        runtime_status(args)
    elif args.command == "cache-audit":
        cache_audit(args)
    elif args.command == "start-domainnet-train-cache":
        start_domainnet_train_cache(args)
    elif args.command == "domainnet-train-cache-status":
        domainnet_train_cache_status(args)
    elif args.command == "start-domainnet-train-cache-direct":
        start_domainnet_train_cache_direct(args)
    elif args.command == "domainnet-train-cache-direct-status":
        domainnet_train_cache_direct_status(args)
    elif args.command == "validate-domainnet-train-cache-8g":
        validate_domainnet_train_cache_8g(args)
    elif args.command == "validate-domainnet-train-cache-both":
        validate_domainnet_train_cache_both(args)
    elif args.command == "sync-domainnet-train-cache-8g-to-third":
        sync_domainnet_train_cache_8g_to_third(args)
    elif args.command == "sync-domainnet-original-cnn-cache":
        sync_domainnet_original_cnn_cache(args)
    elif args.command == "sync-domainnet-train-cache":
        sync_domainnet_train_cache(args)
    elif args.command == "prepare-mdsent-cache":
        prepare_mdsent_cache(args)
    elif args.command == "mdsent-cache-status":
        mdsent_cache_status(args)
    elif args.command == "pause-queue":
        pause_queue(args)
    elif args.command == "cleanup-failed-run":
        cleanup_failed_run(args)


if __name__ == "__main__":
    main()
