#!/usr/bin/env python3
"""Read-only status probe for the three-machine formal experiment queue."""

import argparse
import base64
import gzip
import ipaddress
import json
import os
import re
import socket

import paramiko


def ipv6_bind_candidates(preferred):
    """Return the preferred source plus currently assigned global IPv6s."""
    candidates = []
    for address in [preferred] + [
            item[4][0] for item in socket.getaddrinfo(
                socket.gethostname(), None, socket.AF_INET6)]:
        try:
            is_global = ipaddress.ip_address(address).is_global
        except ValueError:
            is_global = False
        if is_global and address not in candidates:
            candidates.append(address)
    return candidates


def decode_remote_output(payload):
    for encoding in ("utf-8", "gb18030", "cp1252"):
        try:
            return payload.decode(encoding).strip()
        except UnicodeDecodeError:
            continue
    return payload.decode("utf-8", "replace").strip()


def run(client, command):
    _, stdout, stderr = client.exec_command(command, timeout=45)
    return {
        "stdout": decode_remote_output(stdout.read()),
        "stderr": decode_remote_output(stderr.read()),
    }


def key_client(host, port, user, key_path, bind=None):
    key = paramiko.Ed25519Key.from_private_key_file(key_path)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    if bind is None:
        client.connect(host, port=port, username=user, pkey=key, timeout=15)
        return client
    sock = None
    for source in ipv6_bind_candidates(bind):
        candidate = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        candidate.settimeout(15)
        try:
            candidate.bind((source, 0))
            candidate.connect((host, port))
            sock = candidate
            break
        except OSError:
            candidate.close()
    if sock is None:
        # Last resort for environments whose system route is already direct.
        client.connect(
            host, port=port, username=user, pkey=key, timeout=15)
        return client
    transport = paramiko.Transport(sock)
    transport.start_client(timeout=15)
    transport.auth_publickey(user, key)
    client._transport = transport
    return client


def jump_client(jump, host, port, user, *, key_path=None, password=None):
    """Open management SSH through an existing jump-host SSH session."""
    channel = jump.get_transport().open_channel(
        "direct-tcpip",
        (host, port),
        ("127.0.0.1", 0),
    )
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    connect_args = {
        "hostname": host,
        "port": port,
        "username": user,
        "sock": channel,
        "timeout": 15,
        "banner_timeout": 15,
        "auth_timeout": 15,
    }
    if key_path is not None:
        connect_args["pkey"] = \
            paramiko.Ed25519Key.from_private_key_file(key_path)
    else:
        connect_args["password"] = password
    client.connect(**connect_args)
    return client


def encoded_powershell(script):
    compressed = base64.b64encode(
        gzip.compress(script.encode("utf-8"))).decode("ascii")
    loader = (
        f"$b=[Convert]::FromBase64String('{compressed}');"
        "$m=New-Object IO.MemoryStream(,$b);"
        "$g=New-Object IO.Compression.GzipStream("
        "$m,[IO.Compression.CompressionMode]::Decompress);"
        "$r=New-Object IO.StreamReader($g,[Text.Encoding]::UTF8);"
        "& ([ScriptBlock]::Create($r.ReadToEnd()))"
    )
    payload = base64.b64encode(loader.encode("utf-16-le")).decode("ascii")
    return "powershell -NoProfile -NonInteractive -EncodedCommand " + payload


def parse_case(root_output):
    match = re.search(r"\bCASE=([a-z0-9_]+)", root_output)
    if not match:
        raise RuntimeError("Could not determine the active case from root output")
    return match.group(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="final_remaining_20260723_v1")
    parser.add_argument(
        "--key", default=os.path.expanduser(r"~\.ssh\id_ed25519")
    )
    parser.add_argument(
        "--third-password", default=os.environ.get("GGEUR_THIRD_PASSWORD")
    )
    parser.add_argument(
        "--case",
        help="Known active case, used only if the 4090 management endpoint is unavailable",
    )
    parser.add_argument(
        "--root-management-port",
        type=int,
        default=41322,
        help="Local SSH management forward for the 4090 root host",
    )
    parser.add_argument(
        "--skip-root-management",
        action="store_true",
        help="Skip a known-unavailable root SSH endpoint and use --case",
    )
    args = parser.parse_args()
    if not args.third_password:
        parser.error(
            "--third-password or GGEUR_THIRD_PASSWORD is required for management SSH"
        )

    root_command = rf"""
cd /root/autodl-tmp/FederatedScope
run=scripts/distributed_scripts/ggeur_hierarchical_3machine/runs/{args.run_id}
case_name=$(ps -eo args | sed -n 's#.*run_ggeur_fs.py.*runs/{args.run_id}/\([^/ ]*\)/configs/root_server.yaml.*#\1#p' | head -1)
[ -n "$case_name" ] || case_name=$(tail -1 "$run/queue_state/root_queue.tsv" | cut -f2)
case_dir="$run/$case_name"
log="$case_dir/logs/root.stdout.log"
queue=$(ps -eo args | grep run_remaining_queue_root.sh | grep -v grep | wc -l)
root=$(ps -eo args | grep "$case_name/configs/root_server.yaml" | grep -v grep | wc -l)
clients=$(ps -eo args | grep "$case_name/configs/clients_4090/client_" | grep -v grep | wc -l)
acc=$(grep -c 'Round [0-9][0-9]* MLP Test Accuracy' "$log" 2>/dev/null || true)
augmentation=$(grep -c 'augmentation ready' "$log" 2>/dev/null || true)
valid2=$(grep -c 'valid_updates=2/2' "$log" 2>/dev/null || true)
valid4=$(grep -c 'valid_updates=4/4' "$log" 2>/dev/null || true)
bad=$(grep 'valid_updates=' "$log" 2>/dev/null | grep -Evc 'valid_updates=(2/2|4/4)' || true)
err=$(grep -c 'ERROR' "$log" 2>/dev/null || true)
finished=$(grep -c 'Training finished after 100 rounds' "$log" 2>/dev/null || true)
summary=$([ -f "$case_dir/accuracy_summary.json" ] && echo YES || echo NO)
marker=$([ -f "$case_dir/.formal_complete" ] && echo YES || echo NO)
done_count=$(find "$run" -mindepth 2 -maxdepth 2 -name .formal_complete | wc -l)
ready=0
for group in domainnet_vit domainnet_cnn domainnet_mixer; do
  [ -f "exp/distributed_feature_cache/$group/.ggeur_feature_cache_ready.json" ] && ready=$((ready+1))
done
pipeline=$([ -f exp/domainnet_resource_prepare/pipeline.complete ] && echo YES || echo NO)
printf 'CASE=%s QUEUE=%s ROOT=%s CLIENTS=%s ACC=%s AUG=%s VALID2=%s VALID4=%s BAD=%s ERR=%s FINISHED=%s SUMMARY=%s MARKER=%s DONE=%s READY=%s PIPELINE=%s\n' "$case_name" "$queue" "$root" "$clients" "$acc" "$augmentation" "$valid2" "$valid4" "$bad" "$err" "$finished" "$summary" "$marker" "$done_count" "$ready" "$pipeline"
grep 'Round [0-9][0-9]* MLP Test Accuracy' "$log" | tail -1
tail -3 "$run/queue_state/root_queue.tsv"
printf 'CONTROL_STATE='
tr -d '\r\n' <"$run/queue_state/control.json" 2>/dev/null || true
printf '\n'
if [[ "$clients" -ne 60 || "$err" -gt 0 || \
      ( "$case_name" == *_ggeur && "$acc" -eq 0 && \
        "$augmentation" -lt 120 ) ]]; then
  printf 'DIAG_ROOT_RUNTIME\n'
  ps -p "$(pgrep -f "$case_name/configs/root_server.yaml" | head -1)" \
    -o pid=,etime=,stat=,pcpu=,pmem=,args= 2>/dev/null || true
  uptime || true
  printf 'DIAG_MISSING_ROOT_CLIENTS='
  missing=""
  for client_id in $(seq 61 120); do
    name="$(printf 'client_%06d' "$client_id")"
    if ! pgrep -f "$case_name/configs/clients_4090/${{name}}.yaml" \
        >/dev/null; then
      missing="${{missing}}${{client_id}},"
    fi
  done
  printf '%s\n' "${{missing%,}}"
  printf 'DIAG_ROOT_CLIENT_COMMAND_SAMPLE\n'
  ps -eo args | grep "$case_name/configs/clients_4090/client_" \
    | grep -v grep | head -3
  printf 'DIAG_ROOT_CLIENT_CONFIG_SAMPLE\n'
  find "$case_dir/configs/clients_4090" -maxdepth 1 -type f \
    -printf '%f\n' 2>/dev/null | sort -V | head -5
  missing_augmentation="$(
    awk 'NR==FNR {{seen[$1]=1; next}} !($1 in seen)' \
    <(grep -o 'Client [0-9][0-9]* augmentation ready' "$log" 2>/dev/null \
      | awk '{{print $2}}' | sort -nu) \
      <(seq 1 120) | paste -sd, -
  )"
  printf 'DIAG_MISSING_AUGMENTATION_IDS=%s\n' "$missing_augmentation"
  for client_id in $(printf '%s' "$missing_augmentation" | tr ',' ' '); do
    [[ "$client_id" =~ ^[0-9]+$ ]] || continue
    if (( client_id < 61 || client_id > 120 )); then
      continue
    fi
    name="$(printf 'client_%06d' "$client_id")"
    config="$case_dir/configs/clients_4090/$name.yaml"
    client_pid="$(pgrep -f "$config" | head -1 || true)"
    printf 'DIAG_CLIENT_%s_RUNTIME\n' "$client_id"
    if [[ -n "$client_pid" ]]; then
      ps -p "$client_pid" -o pid=,etime=,stat=,pcpu=,pmem=,args= \
        2>/dev/null || true
    else
      printf 'NOT_RUNNING\n'
    fi
    printf 'DIAG_CLIENT_%s_LOG_TAIL\n' "$client_id"
    tail -20 "$case_dir/logs/$name.stdout.log" 2>/dev/null || true
  done
  printf 'DIAG_ROOT_LOG_TAIL\n'
  tail -30 "$log" 2>/dev/null || true
  printf 'DIAG_NONEMPTY_ROOT_STDERR\n'
  find "$case_dir/logs" -maxdepth 1 -type f -name '*.stderr.log' -size +0c \
    -printf '%f\t%s bytes\n' 2>/dev/null | sort | tail -20
fi
"""

    # Keep one direct management session to the 8G host.  It is also the
    # fallback jump host when the optional local 41222/41322 forwards expire.
    client = key_client(
        "2001:da8:215:3c0a:f51:4d71:80:8075",
        22,
        "fsuser",
        args.key,
        bind="2408:8207:1a25:5b70:3c24:419a:cbdf:5187",
    )

    result = {}
    if args.skip_root_management:
        if not args.case or not re.fullmatch(r"[a-z0-9_]+", args.case):
            parser.error("--skip-root-management requires a valid --case")
        result["root4090"] = {
            "stdout": "",
            "stderr": "MANAGEMENT_CHECK_SKIPPED: using 8G real-network control probe",
        }
        case_name = args.case
    else:
        try:
            try:
                root = key_client(
                    "::1", args.root_management_port, "root", args.key)
                root_route = "local_forward"
            except Exception:
                root = jump_client(
                    client,
                    "10.112.81.135",
                    22,
                    "root",
                    key_path=args.key,
                )
                root_route = "8g_jump_to_10.112.81.135"
            try:
                result["root4090"] = run(root, root_command)
                result["root4090"]["management_route"] = root_route
            finally:
                root.close()
            case_name = parse_case(result["root4090"]["stdout"])
        except Exception as error:
            result["root4090"] = {
                "stdout": "",
                "stderr": (
                    f"MANAGEMENT_CHECK_FAILED: {type(error).__name__}: {error}"
                ),
            }
            if not args.case or not re.fullmatch(r"[a-z0-9_]+", args.case):
                print(json.dumps(result, ensure_ascii=True, indent=2))
                raise
            case_name = args.case

    third_script = rf"""
$ProgressPreference = 'SilentlyContinue'
$repo = 'C:\Users\pc\FederatedScope'
$case = '{case_name}'
$taskObject = Get-ScheduledTask -TaskName 'GGEUR-final-subserver-queue' -ErrorAction SilentlyContinue
$task = $taskObject.State
$taskInfo = Get-ScheduledTaskInfo -TaskName 'GGEUR-final-subserver-queue' -ErrorAction SilentlyContinue
$procs = @(Get-CimInstance Win32_Process | Where-Object {{ $_.CommandLine }})
$queueProcs = @($procs | Where-Object {{
    $_.ProcessId -ne $PID -and
    $_.CommandLine -like '*run_remaining_queue_subservers.ps1*'
}}).Count
$roles = @($procs | Where-Object {{ $_.CommandLine -like "*$case*configs\subserver_*" }}).Count
$other = @($procs | Where-Object {{ $_.CommandLine -like '*{args.run_id}*configs\subserver_*' -and $_.CommandLine -notlike "*$case*" }}).Count
$standalone = @($procs | Where-Object {{ $_.CommandLine -match 'standalone' }}).Count
$logs = Join-Path $repo "scripts\distributed_scripts\ggeur_hierarchical_3machine\runs\{args.run_id}\$case\logs"
$errors = if (Test-Path $logs) {{ @(Get-ChildItem $logs -File | Select-String -SimpleMatch 'ERROR').Count }} else {{ 0 }}
$conns = @(Get-NetTCPConnection -State Established -ErrorAction SilentlyContinue | Where-Object {{ ($_.LocalPort -ge 61000 -and $_.LocalPort -le 61003) -or ($_.RemotePort -ge 61000 -and $_.RemotePort -le 61003) }})
$real10 = @($conns | Where-Object {{ $_.LocalAddress -like '10.*' -and $_.RemoteAddress -like '10.*' }}).Count
$non10 = @($conns | Where-Object {{ $_.LocalAddress -notlike '10.*' -or $_.RemoteAddress -notlike '10.*' }}).Count
"QUEUE=$task QPROCS=$queueProcs TASK_RESULT=$($taskInfo.LastTaskResult) ROLES=$roles OTHER=$other REAL10=$real10 NON10=$non10 ERR=$errors STANDALONE=$standalone"
"TASK_ACTION=$($taskObject.Actions.Execute) $($taskObject.Actions.Arguments)"
"SUBSERVER_AGGREGATION_TAIL"
if (Test-Path $logs) {{
    Get-ChildItem $logs -File -Filter 'subserver_*.stdout.log' |
        Select-String -SimpleMatch 'locally aggregated clients=' |
        Select-Object -Last 8 |
        ForEach-Object {{ "$($_.Path):$($_.LineNumber):$($_.Line)" }}
}}
"QUEUE_STATE_TAIL"
Get-Content -LiteralPath (Join-Path $repo "scripts\distributed_scripts\ggeur_hierarchical_3machine\runs\{args.run_id}\queue_state\subserver_queue.tsv") `
    -Tail 2 -ErrorAction SilentlyContinue
if ($other -gt 0 -or $errors -gt 0) {{
    "DIAG_SUBSERVER_PROCESSES"
    $procs | Where-Object {{
        $_.CommandLine -like '*{args.run_id}*configs\subserver_*'
    }} | Select-Object ProcessId, CreationDate, CommandLine |
        ConvertTo-Json -Compress
    "DIAG_QUEUE_TAIL"
    Get-Content -LiteralPath (Join-Path $repo "scripts\distributed_scripts\ggeur_hierarchical_3machine\runs\{args.run_id}\queue_state\subserver_queue.tsv") `
        -Tail 12 -ErrorAction SilentlyContinue
    "DIAG_CURRENT_CASE_ERRORS"
    if (Test-Path $logs) {{
        Get-ChildItem $logs -File |
            Select-String -SimpleMatch 'ERROR' |
            Select-Object -Last 12 |
            ForEach-Object {{ "$($_.Path):$($_.LineNumber):$($_.Line)" }}
    }}
}}
"""
    third = None
    try:
        try:
            third = paramiko.SSHClient()
            third.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            third.connect(
                "::1",
                port=41222,
                username="pc",
                password=args.third_password,
                timeout=15,
            )
            third_route = "local_forward"
        except Exception:
            third = jump_client(
                client,
                "10.129.248.111",
                22,
                "pc",
                password=args.third_password,
            )
            third_route = "8g_jump_to_10.129.248.111"
        result["third"] = run(third, encoded_powershell(third_script))
        result["third"]["management_route"] = third_route
    finally:
        if third is not None:
            third.close()

    client_script = rf"""
$ProgressPreference = 'SilentlyContinue'
$repo = 'D:\Projects\FederatedScope'
$case = '{case_name}'
$taskObject = Get-ScheduledTask -TaskName 'GGEUR-final-client-queue' -ErrorAction SilentlyContinue
$task = $taskObject.State
$taskInfo = Get-ScheduledTaskInfo -TaskName 'GGEUR-final-client-queue' -ErrorAction SilentlyContinue
$procs = @(Get-CimInstance Win32_Process | Where-Object {{ $_.CommandLine }})
$queueProcs = @($procs | Where-Object {{
    $_.ProcessId -ne $PID -and
    $_.CommandLine -like '*run_remaining_queue_clients.ps1*'
}}).Count
$roles = @($procs | Where-Object {{ $_.CommandLine -like "*$case*configs\clients_8g\client_*" }}).Count
$other = @($procs | Where-Object {{ $_.CommandLine -like '*{args.run_id}*configs\clients_8g\client_*' -and $_.CommandLine -notlike "*$case*" }}).Count
$standalone = @($procs | Where-Object {{ $_.CommandLine -match 'standalone' }}).Count
$logs = Join-Path $repo "scripts\distributed_scripts\ggeur_hierarchical_3machine\runs\{args.run_id}\$case\logs"
$errors = if (Test-Path $logs) {{ @(Get-ChildItem $logs -File | Select-String -SimpleMatch 'ERROR').Count }} else {{ 0 }}
$conns = @(Get-NetTCPConnection -State Established -ErrorAction SilentlyContinue | Where-Object {{ ($_.RemotePort -ge 61000 -and $_.RemotePort -le 61003) -or $_.RemotePort -eq 60050 }})
$real10 = @($conns | Where-Object {{ $_.LocalAddress -like '10.*' -and $_.RemoteAddress -like '10.*' }}).Count
$non10 = @($conns | Where-Object {{ $_.LocalAddress -notlike '10.*' -or $_.RemoteAddress -notlike '10.*' }}).Count
$ready = @('domainnet_vit','domainnet_cnn','domainnet_mixer' | Where-Object {{ Test-Path (Join-Path $repo "exp\distributed_feature_cache\$_\.ggeur_feature_cache_ready.json") }}).Count
$pipeline = Test-Path (Join-Path $repo 'exp\domainnet_resource_prepare\pipeline.complete')
$root22 = Test-NetConnection -ComputerName '10.112.81.135' -Port 22 -InformationLevel Quiet
$root60050 = Test-NetConnection -ComputerName '10.112.81.135' -Port 60050 -InformationLevel Quiet
"QUEUE=$task QPROCS=$queueProcs TASK_RESULT=$($taskInfo.LastTaskResult) ROLES=$roles OTHER=$other REAL10=$real10 NON10=$non10 ERR=$errors STANDALONE=$standalone READY=$ready PIPELINE=$pipeline ROOT22=$root22 ROOT60050=$root60050"
$controlState = try {{
    Invoke-RestMethod -Uri 'http://10.112.81.135:60049/state' -TimeoutSec 8 |
        ConvertTo-Json -Compress
}} catch {{
    "UNAVAILABLE:$($_.Exception.Message)"
}}
"ROOT_CONTROL=$controlState"
"TASK_ACTION=$($taskObject.Actions.Execute) $($taskObject.Actions.Arguments)"
"CLIENT_TRAINING_TAIL"
$clientOneLog = Join-Path $logs 'client_000001.stdout.log'
if (Test-Path $clientOneLog) {{
    Select-String -LiteralPath $clientOneLog `
        -SimpleMatch 'Training on augmented data' |
        Select-Object -Last 5 |
        ForEach-Object {{ "$($_.Path):$($_.LineNumber):$($_.Line)" }}
}}
"QUEUE_STATE_TAIL"
Get-Content -LiteralPath (Join-Path $repo "scripts\distributed_scripts\ggeur_hierarchical_3machine\runs\{args.run_id}\queue_state\client_queue.tsv") `
    -Tail 2 -ErrorAction SilentlyContinue
"""
    result["client8g"] = run(client, encoded_powershell(client_script))
    client.close()

    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
