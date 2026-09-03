#!/usr/bin/env python3
"""Recover a formal three-machine queue from a cross-case transition skew.

This helper is intentionally scoped to one run/case pair. It preserves the
incomplete attempt, stops only matching formal-run roles, deploys the
root-authoritative queue handshake, and restarts the three persistent queues.
"""

import argparse
import base64
import ipaddress
import json
import os
import socket
import time
from pathlib import Path

import paramiko


ROOT_REPO = "/root/autodl-tmp/FederatedScope"
THIRD_REPO = r"C:\Users\pc\FederatedScope"
CLIENT_REPO = r"D:\Projects\FederatedScope"
SCRIPT_REL = "scripts/distributed_scripts/ggeur_hierarchical_3machine"


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


def key_client(host, port, user, key_path, bind=None):
    key = paramiko.Ed25519Key.from_private_key_file(key_path)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    if bind is None:
        client.connect(host, port=port, username=user, pkey=key, timeout=20)
        return client
    sock = None
    for source in ipv6_bind_candidates(bind):
        candidate = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        candidate.settimeout(20)
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
            host, port=port, username=user, pkey=key, timeout=20)
        return client
    transport = paramiko.Transport(sock)
    transport.start_client(timeout=20)
    transport.auth_publickey(user, key)
    client._transport = transport
    return client


def jump_client(jump, host, port, user, *, key_path=None, password=None):
    """Open a management SSH session through the already connected 8G host."""
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
        "timeout": 20,
        "banner_timeout": 20,
        "auth_timeout": 20,
    }
    if key_path is not None:
        connect_args["pkey"] = \
            paramiko.Ed25519Key.from_private_key_file(key_path)
    else:
        connect_args["password"] = password
    client.connect(**connect_args)
    return client


def run(client, command, timeout=120):
    _, stdout, stderr = client.exec_command(command, timeout=timeout)
    out = stdout.read().decode("utf-8", "replace").strip()
    err = stderr.read().decode("utf-8", "replace").strip()
    status = stdout.channel.recv_exit_status()
    if status:
        raise RuntimeError(
            f"remote command failed ({status})\nstdout:\n{out}\nstderr:\n{err}"
        )
    return out


def encoded_powershell(script):
    payload = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    return "powershell -NoProfile -NonInteractive -EncodedCommand " + payload


def upload(client, local_path, remote_path, *, newline=None):
    data = Path(local_path).read_bytes()
    if newline == "lf":
        data = data.replace(b"\r\n", b"\n")
    sftp = client.open_sftp()
    try:
        with sftp.file(remote_path, "wb") as handle:
            handle.write(data)
    finally:
        sftp.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="final_remaining_20260723_v1")
    parser.add_argument("--case", default="mdsent_rnn_fedprox")
    parser.add_argument(
        "--action",
        choices=(
            "recover",
            "resume-windows",
            "restart-current-case",
            "resume-current-case",
            "start-windows-queues",
        ),
        default="recover",
    )
    parser.add_argument(
        "--key", default=os.path.expanduser(r"~\.ssh\id_ed25519")
    )
    parser.add_argument(
        "--third-password", default=os.environ.get("GGEUR_THIRD_PASSWORD")
    )
    parser.add_argument(
        "--third-queue-task",
        default="GGEUR-final-subserver-queue",
        help="Scheduled task that owns the persistent third-host queue",
    )
    parser.add_argument(
        "--client-queue-task",
        default="GGEUR-final-client-queue",
        help="Scheduled task that owns the persistent 8G client queue",
    )
    args = parser.parse_args()
    if args.action == "recover" and args.case != "mdsent_rnn_fedprox":
        parser.error("this recovery helper is scoped to mdsent_rnn_fedprox")

    local_dir = Path(__file__).resolve().parent
    repo_root = local_dir.parents[2]
    root = None
    third = None
    client = None

    try:
        client = key_client(
            "2001:da8:215:3c0a:f51:4d71:80:8075",
            22,
            "fsuser",
            args.key,
            bind="2408:8207:1a25:5b70:3c24:419a:cbdf:5187",
        )
        root = jump_client(
            client,
            "10.112.81.135",
            22,
            "root",
            key_path=args.key,
        )
        try:
            third = jump_client(
                client,
                "10.129.248.111",
                22,
                "pc",
                key_path=args.key,
            )
        except paramiko.AuthenticationException:
            if not args.third_password:
                raise
            third = jump_client(
                client,
                "10.129.248.111",
                22,
                "pc",
                password=args.third_password,
            )
        if args.action == "start-windows-queues":
            start_third = rf"""
$ErrorActionPreference = 'Stop'
Start-ScheduledTask -TaskName '{args.third_queue_task}'
Start-Sleep -Seconds 2
"THIRD_QUEUE=$((Get-ScheduledTask -TaskName '{args.third_queue_task}').State)"
"""
            start_client = rf"""
$ErrorActionPreference = 'Stop'
Start-ScheduledTask -TaskName '{args.client_queue_task}'
Start-Sleep -Seconds 2
"CLIENT_QUEUE=$((Get-ScheduledTask -TaskName '{args.client_queue_task}').State)"
"""
            print(run(third, encoded_powershell(start_third)))
            print(run(client, encoded_powershell(start_client)))
            return
        if args.action == "resume-windows":
            stop_third_queue = (
                "$self=$PID; "
                "Stop-ScheduledTask -TaskName "
                "'GGEUR-final-subserver-queue' "
                "-ErrorAction SilentlyContinue; "
                "Get-CimInstance Win32_Process | Where-Object { "
                "$_.ProcessId -ne $self -and "
                "$_.CommandLine -like "
                "'*run_remaining_queue_subservers.ps1*' "
                "} | ForEach-Object { "
                "Stop-Process -Id $_.ProcessId -Force "
                "-ErrorAction SilentlyContinue }"
            )
            stop_client_queue = (
                "$self=$PID; "
                "Stop-ScheduledTask -TaskName "
                "'GGEUR-final-client-queue' "
                "-ErrorAction SilentlyContinue; "
                "Get-CimInstance Win32_Process | Where-Object { "
                "$_.ProcessId -ne $self -and "
                "$_.CommandLine -like "
                "'*run_remaining_queue_clients.ps1*' "
                "} | ForEach-Object { "
                "Stop-Process -Id $_.ProcessId -Force "
                "-ErrorAction SilentlyContinue }"
            )
            run(third, encoded_powershell(stop_third_queue))
            run(client, encoded_powershell(stop_client_queue))
            for name in [
                "run_remaining_queue_subservers.ps1",
                "queue_control_client.ps1",
                "launch_subservers.ps1",
            ]:
                upload(
                    third,
                    local_dir / name,
                    f"{THIRD_REPO}/{SCRIPT_REL}/{name}".replace("\\", "/"),
                )
            for name in [
                "run_remaining_queue_clients.ps1",
                "queue_control_client.ps1",
                "launch_clients.ps1",
            ]:
                upload(
                    client,
                    local_dir / name,
                    f"{CLIENT_REPO}/{SCRIPT_REL}/{name}".replace("\\", "/"),
                )
            start_third_queue = r"""
Start-ScheduledTask -TaskName 'GGEUR-final-subserver-queue'
Start-Sleep -Seconds 2
$taskState = (Get-ScheduledTask -TaskName 'GGEUR-final-subserver-queue').State
if ($taskState -ne 'Running') {
    $repo = 'C:\Users\pc\FederatedScope'
    $script = Join-Path $repo 'scripts\distributed_scripts\ggeur_hierarchical_3machine\run_remaining_queue_subservers.ps1'
    $state = Join-Path $repo 'scripts\distributed_scripts\ggeur_hierarchical_3machine\runs\final_remaining_20260723_v1\queue_state'
    $process = Start-Process -FilePath 'powershell.exe' -ArgumentList @(
        '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $script
    ) -WorkingDirectory $repo -WindowStyle Hidden `
      -RedirectStandardOutput (Join-Path $state 'subserver_queue.direct.stdout.log') `
      -RedirectStandardError (Join-Path $state 'subserver_queue.direct.stderr.log') `
      -PassThru
    "THIRD_QUEUE=DIRECT PID=$($process.Id)"
} else {
    "THIRD_QUEUE=SCHEDULED_TASK"
}
"""
            start_client_queue = r"""
Start-ScheduledTask -TaskName 'GGEUR-final-client-queue'
Start-Sleep -Seconds 2
$taskState = (Get-ScheduledTask -TaskName 'GGEUR-final-client-queue').State
if ($taskState -ne 'Running') {
    $repo = 'D:\Projects\FederatedScope'
    $script = Join-Path $repo 'scripts\distributed_scripts\ggeur_hierarchical_3machine\run_remaining_queue_clients.ps1'
    $state = Join-Path $repo 'scripts\distributed_scripts\ggeur_hierarchical_3machine\runs\final_remaining_20260723_v1\queue_state'
    $process = Start-Process -FilePath 'powershell.exe' -ArgumentList @(
        '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $script
    ) -WorkingDirectory $repo -WindowStyle Hidden `
      -RedirectStandardOutput (Join-Path $state 'client_queue.direct.stdout.log') `
      -RedirectStandardError (Join-Path $state 'client_queue.direct.stderr.log') `
      -PassThru
    "CLIENT_QUEUE=DIRECT PID=$($process.Id)"
} else {
    "CLIENT_QUEUE=SCHEDULED_TASK"
}
"""
            print(run(third, encoded_powershell(start_third_queue)))
            print(run(client, encoded_powershell(start_client_queue)))
            return

        expected_case_index = 13
        recovery_suffix = "cross_case_skew"
        recovery_status = "INCOMPLETE_CROSS_CASE_QUEUE_SKEW"
        recovery_record = "RECOVERY_CROSS_CASE_SKEW"
        third_archive_names = f"@('{args.case}', 'mdsent_rnn_fedproto')"
        if args.action in {"restart-current-case", "resume-current-case"}:
            state_text = run(
                root, "curl -fsS http://127.0.0.1:60049/state"
            )
            state = json.loads(state_text)
            if state.get("case") != args.case:
                raise RuntimeError(
                    f"root current case is {state.get('case')}, not {args.case}"
                )
            if state.get("phase") not in {"starting", "running", "failed"}:
                raise RuntimeError(f"unsafe root phase: {state.get('phase')}")
            expected_case_index = int(state["case_index"])
            recovery_suffix = "same_case_restart"
            recovery_status = "INCOMPLETE_SAME_CASE_RESTART"
            recovery_record = "RECOVERY_SAME_CASE_RESTART"
            third_archive_names = f"@('{args.case}')"

        root_stop = f"""
set -euo pipefail
repo={ROOT_REPO}
run="$repo/{SCRIPT_REL}/runs/{args.run_id}"
case_dir="$run/{args.case}"
stamp="$(date +%Y%m%d_%H%M%S)"
attempt="$case_dir/attempts/attempt_${{stamp}}_{recovery_suffix}"
mkdir -p "$attempt"
for pid in $(pgrep -f '[r]un_remaining_queue_root.sh' || true); do
  kill "$pid" 2>/dev/null || true
done
sleep 2
for pid_file in "$case_dir"/pids/root.pid "$case_dir"/pids/client_*.pid; do
  [[ -f "$pid_file" ]] || continue
  pid="$(cat "$pid_file" 2>/dev/null || true)"
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  if kill -0 "$pid" 2>/dev/null && \
     tr '\\0' ' ' <"/proc/$pid/cmdline" | grep -Fq "$case_dir"; then
    kill "$pid" 2>/dev/null || true
  fi
done
sleep 5
for pid_file in "$case_dir"/pids/root.pid "$case_dir"/pids/client_*.pid; do
  [[ -f "$pid_file" ]] || continue
  pid="$(cat "$pid_file" 2>/dev/null || true)"
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  if kill -0 "$pid" 2>/dev/null && \
     tr '\\0' ' ' <"/proc/$pid/cmdline" | grep -Fq "$case_dir"; then
    kill -9 "$pid" 2>/dev/null || true
  fi
done
cp -a "$case_dir/logs" "$attempt/" 2>/dev/null || true
cp -a "$case_dir/configs/root_server.yaml" "$attempt/" 2>/dev/null || true
accuracy_rounds="$(grep -a -c 'MLP Test Accuracy' "$case_dir/logs/root.stdout.log" 2>/dev/null || true)"
printf '%s\n' \
  'status={recovery_status}' \
  "archived_at=$(date --iso-8601=seconds)" \
  "accuracy_rounds=$accuracy_rounds" >"$attempt/attempt_status.txt"
printf '%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "{args.case}" \
  "{recovery_record}:$attempt" >>"$run/queue_state/root_queue.tsv"
echo "ROOT_STOPPED_ATTEMPT=$attempt"
"""
        if args.action != "resume-current-case":
            print(run(root, root_stop))

        third_stop = rf"""
$ErrorActionPreference = 'Stop'
$repo = '{THIRD_REPO}'
$run = Join-Path $repo '{SCRIPT_REL.replace("/", chr(92))}\runs\{args.run_id}'
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
Stop-ScheduledTask -TaskName '{args.third_queue_task}' -ErrorAction SilentlyContinue
$procs = @(Get-CimInstance Win32_Process | Where-Object {{
    $_.CommandLine -like '*{args.run_id}*' -and
    $_.CommandLine -like '*hierarchical_subserver.py*'
}})
foreach ($proc in $procs) {{
    Stop-Process -Id $proc.ProcessId -Force -ErrorAction SilentlyContinue
}}
$caseTasks = @(Get-ScheduledTask `
    -TaskName 'GGEUR-{args.case}-subserver_*' `
    -ErrorAction SilentlyContinue)
foreach ($task in $caseTasks) {{
    Stop-ScheduledTask -TaskName $task.TaskName `
        -ErrorAction SilentlyContinue
    Unregister-ScheduledTask -TaskName $task.TaskName `
        -Confirm:$false -ErrorAction SilentlyContinue
}}
Start-Sleep -Seconds 2
foreach ($name in {third_archive_names}) {{
    $caseDir = Join-Path $run $name
    $attempt = Join-Path $caseDir "attempts\attempt_${{stamp}}_{recovery_suffix}"
    New-Item -ItemType Directory -Force -Path $attempt | Out-Null
    $logs = Join-Path $caseDir 'logs'
    if (Test-Path $logs) {{
        Copy-Item -LiteralPath $logs -Destination $attempt -Recurse -Force
    }}
    @(
        'status={recovery_status}',
        "archived_at=$((Get-Date).ToString('o'))"
    ) | Set-Content -LiteralPath (Join-Path $attempt 'attempt_status.txt')
}}
Add-Content -LiteralPath (Join-Path $run 'queue_state\subserver_queue.tsv') `
    -Value "$((Get-Date).ToString('o'))`t{args.case}`t{recovery_record}"
"THIRD_STOPPED_ROLES=$($procs.Count)"
"""
        if args.action != "resume-current-case":
            print(run(third, encoded_powershell(third_stop)))

        client_stop = rf"""
$ErrorActionPreference = 'Stop'
$repo = '{CLIENT_REPO}'
$run = Join-Path $repo '{SCRIPT_REL.replace("/", chr(92))}\runs\{args.run_id}'
$caseDir = Join-Path $run '{args.case}'
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
Stop-ScheduledTask -TaskName '{args.client_queue_task}' -ErrorAction SilentlyContinue
$procs = @(Get-CimInstance Win32_Process | Where-Object {{
    $_.CommandLine -like '*{args.run_id}*' -and
    $_.CommandLine -like '*{args.case}*' -and
    $_.CommandLine -like '*federatedscope.main*'
}})
foreach ($proc in $procs) {{
    Stop-Process -Id $proc.ProcessId -Force -ErrorAction SilentlyContinue
}}
$caseTasks = @(Get-ScheduledTask `
    -TaskName 'GGEUR-{args.case}-client_*' `
    -ErrorAction SilentlyContinue)
foreach ($task in $caseTasks) {{
    Stop-ScheduledTask -TaskName $task.TaskName `
        -ErrorAction SilentlyContinue
    Unregister-ScheduledTask -TaskName $task.TaskName `
        -Confirm:$false -ErrorAction SilentlyContinue
}}
Start-Sleep -Seconds 2
$attempt = Join-Path $caseDir "attempts\attempt_${{stamp}}_{recovery_suffix}"
New-Item -ItemType Directory -Force -Path $attempt | Out-Null
$logs = Join-Path $caseDir 'logs'
if (Test-Path $logs) {{
    Copy-Item -LiteralPath $logs -Destination $attempt -Recurse -Force
}}
@(
    'status={recovery_status}',
    "archived_at=$((Get-Date).ToString('o'))"
) | Set-Content -LiteralPath (Join-Path $attempt 'attempt_status.txt')
Add-Content -LiteralPath (Join-Path $run 'queue_state\client_queue.tsv') `
    -Value "$((Get-Date).ToString('o'))`t{args.case}`t{recovery_record}"
"CLIENT_STOPPED_ROLES=$($procs.Count)"
"""
        if args.action != "resume-current-case":
            print(run(client, encoded_powershell(client_stop)))

        root_files = [
            "run_remaining_queue_root.sh",
            "queue_control.py",
            "validate_case_completion.py",
        ]
        for name in root_files:
            upload(
                root,
                local_dir / name,
                f"{ROOT_REPO}/{SCRIPT_REL}/{name}",
                newline="lf",
            )
        upload(
            root,
            repo_root / "federatedscope/contrib/worker/ggeur_server.py",
            f"{ROOT_REPO}/federatedscope/contrib/worker/ggeur_server.py",
            newline="lf",
        )
        upload(
            root,
            repo_root / "federatedscope/contrib/worker/ggeur_client.py",
            f"{ROOT_REPO}/federatedscope/contrib/worker/ggeur_client.py",
            newline="lf",
        )
        for name in [
            "run_remaining_queue_subservers.ps1",
            "queue_control_client.ps1",
        ]:
            upload(
                third,
                local_dir / name,
                f"{THIRD_REPO}/{SCRIPT_REL}/{name}".replace("\\", "/"),
            )
        upload(
            third,
            repo_root / "federatedscope/contrib/worker/ggeur_client.py",
            f"{THIRD_REPO}/federatedscope/contrib/worker/ggeur_client.py"
            .replace("\\", "/"),
        )
        for name in [
            "run_remaining_queue_clients.ps1",
            "queue_control_client.ps1",
        ]:
            upload(
                client,
                local_dir / name,
                f"{CLIENT_REPO}/{SCRIPT_REL}/{name}".replace("\\", "/"),
            )
        upload(
            client,
            repo_root / "federatedscope/contrib/worker/ggeur_client.py",
            f"{CLIENT_REPO}/federatedscope/contrib/worker/ggeur_client.py"
            .replace("\\", "/"),
        )

        start_root = f"""
set -euo pipefail
cd {ROOT_REPO}
nohup env RUN_ID={args.run_id} \
  PYTHON_BIN=/root/.local/share/mamba/envs/GGEUR/bin/python \
  REPO_DIR={ROOT_REPO} \
  bash {SCRIPT_REL}/run_remaining_queue_root.sh \
  >>{SCRIPT_REL}/runs/{args.run_id}/queue_state/root_queue.nohup.log 2>&1 &
echo "ROOT_QUEUE_PID=$!"
"""
        print(run(root, start_root))

        deadline = time.time() + 90
        state = ""
        while time.time() < deadline:
            state = run(
                root,
                f"curl -fsS http://127.0.0.1:60049/state || true",
            )
            try:
                parsed = json.loads(state)
            except json.JSONDecodeError:
                parsed = {}
            if (parsed.get("case") == args.case
                    and parsed.get("case_index") == expected_case_index
                    and parsed.get("phase") == "running"):
                    break
            time.sleep(3)
        else:
            raise RuntimeError(f"root control state did not converge: {state}")
        print("ROOT_CONTROL_STATE=" + state)

        connectivity = r"""
$state = Invoke-RestMethod -Uri 'http://10.112.81.135:60049/state' -TimeoutSec 10
$rootReady = $false
for ($attempt = 1; $attempt -le 18 -and -not $rootReady; $attempt++) {
    $tcp = [Net.Sockets.TcpClient]::new()
    try {
        $task = $tcp.ConnectAsync('10.112.81.135', 60050)
        try {
            $rootReady = $task.Wait(5000) -and $tcp.Connected
        } catch {
            $rootReady = $false
        }
    } finally {
        $tcp.Dispose()
    }
    if (-not $rootReady) {
        Start-Sleep -Seconds 5
    }
}
if (-not $rootReady) {
    throw 'root business endpoint 10.112.81.135:60050 is not reachable after 18 attempts'
}
"CONTROL_CASE=$($state.case) INDEX=$($state.case_index) PHASE=$($state.phase) ROOT60050=$rootReady"
"""
        print("THIRD_" + run(third, encoded_powershell(connectivity)))
        print("CLIENT_" + run(client, encoded_powershell(connectivity)))

        start_third = rf"""
Start-ScheduledTask -TaskName '{args.third_queue_task}'
Start-Sleep -Seconds 2
"THIRD_QUEUE=$((Get-ScheduledTask -TaskName '{args.third_queue_task}').State)"
"""
        start_client = rf"""
Start-ScheduledTask -TaskName '{args.client_queue_task}'
Start-Sleep -Seconds 2
"CLIENT_QUEUE=$((Get-ScheduledTask -TaskName '{args.client_queue_task}').State)"
"""
        print(run(third, encoded_powershell(start_third)))
        print(run(client, encoded_powershell(start_client)))
    finally:
        if root is not None:
            root.close()
        if third is not None:
            third.close()
        if client is not None:
            client.close()


if __name__ == "__main__":
    main()
