#!/usr/bin/env python3
"""Deploy and manage the three-machine concurrent-availability validation."""

import argparse
import base64
import ipaddress
import json
import math
import os
import posixpath
import socket
import stat
import struct
import time
from pathlib import Path

import paramiko


ROOT_REPO = "/root/autodl-tmp/FederatedScope"
THIRD_REPO = r"C:\Users\pc\FederatedScope"
CLIENT_REPO = r"D:\Projects\FederatedScope"
REL_DIR = "scripts/distributed_scripts/ggeur_concurrent_availability"
CLIENT_IPV6 = os.environ.get(
    "GGEUR_CLIENT_IPV6",
    "2001:da8:215:3c0a:f51:4d71:80:8075",
)
CLIENT_BIND_IPV6 = "2408:8207:1a25:5b70:3c24:419a:cbdf:5187"


def ipv6_bind_candidates(preferred):
    candidates = []
    discovered = [
        item[4][0] for item in socket.getaddrinfo(
            socket.gethostname(), None, socket.AF_INET6)
    ]
    for address in [preferred] + discovered:
        try:
            valid = ipaddress.ip_address(address).is_global
        except ValueError:
            valid = False
        if valid and address not in candidates:
            candidates.append(address)
    return candidates


def recv_exact(sock, size):
    chunks = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise EOFError("SOCKS5 connection closed early")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def socks5_socket(proxy_host, proxy_port, host, port):
    sock = socket.create_connection((proxy_host, proxy_port), timeout=20)
    sock.sendall(b"\x05\x01\x00")
    if recv_exact(sock, 2) != b"\x05\x00":
        raise RuntimeError("SOCKS5 proxy rejected no-authentication mode")
    address = host.encode("idna")
    sock.sendall(
        b"\x05\x01\x00\x03" + bytes([len(address)]) + address
        + struct.pack("!H", port))
    version, status, _, address_type = recv_exact(sock, 4)
    if version != 5 or status != 0:
        raise RuntimeError(f"SOCKS5 connect failed with status {status}")
    if address_type == 1:
        recv_exact(sock, 4)
    elif address_type == 3:
        recv_exact(sock, recv_exact(sock, 1)[0])
    elif address_type == 4:
        recv_exact(sock, 16)
    else:
        raise RuntimeError(f"unexpected SOCKS5 address type {address_type}")
    recv_exact(sock, 2)
    sock.settimeout(None)
    return sock


def load_ed25519_key(key_path):
    try:
        return paramiko.Ed25519Key.from_private_key_file(key_path)
    except paramiko.PasswordRequiredException:
        return paramiko.Ed25519Key.from_private_key_file(
            key_path, password='""')


def key_client(host, port, user, key_path, bind=None, sock=None):
    key = load_ed25519_key(key_path)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    if sock is not None:
        client.connect(
            host,
            port=port,
            username=user,
            pkey=key,
            sock=sock,
            timeout=20,
            banner_timeout=30,
            auth_timeout=20,
        )
        return client
    if bind is None:
        client.connect(
            host,
            port=port,
            username=user,
            pkey=key,
            timeout=20,
            banner_timeout=30,
            auth_timeout=20,
        )
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
        client.connect(
            host,
            port=port,
            username=user,
            pkey=key,
            timeout=20,
            banner_timeout=30,
            auth_timeout=20,
        )
        return client
    transport = paramiko.Transport(sock)
    transport.banner_timeout = 30
    transport.start_client(timeout=30)
    transport.auth_publickey(user, key)
    client._transport = transport
    return client


def jump_client(
        jump, host, port, user, *, key_path=None, password=None,
        banner_timeout=60):
    channel = jump.get_transport().open_channel(
        "direct-tcpip", (host, port), ("127.0.0.1", 0))
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    options = {
        "hostname": host,
        "port": port,
        "username": user,
        "sock": channel,
        "timeout": 30,
        "banner_timeout": banner_timeout,
        "auth_timeout": 30,
    }
    if key_path:
        options["pkey"] = load_ed25519_key(key_path)
    else:
        options["password"] = password
    client.connect(**options)
    return client


def run(client, command, timeout=120, check=True):
    _, stdout, stderr = client.exec_command(command, timeout=timeout)
    out = stdout.read().decode("utf-8", "replace").strip()
    err = stderr.read().decode("utf-8", "replace").strip()
    status = stdout.channel.recv_exit_status()
    result = {"status": status, "stdout": out, "stderr": err}
    if check and status:
        raise RuntimeError(
            f"remote command failed ({status})\nstdout:\n{out}\nstderr:\n{err}")
    return result


def encoded_powershell(script):
    payload = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    return "powershell -NoProfile -NonInteractive -EncodedCommand " + payload


def upload_bytes(client, data, remote_path):
    sftp = client.open_sftp()
    try:
        with sftp.file(remote_path, "wb") as handle:
            handle.write(data)
    finally:
        sftp.close()


def upload_file(client, local_path, remote_path, newline=None):
    data = Path(local_path).read_bytes()
    if newline == "lf":
        data = data.replace(b"\r\n", b"\n")
    upload_bytes(client, data, remote_path)


def download_file(client, remote_path, local_path):
    destination = Path(local_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    sftp = client.open_sftp()
    try:
        sftp.get(remote_path, str(temporary))
    finally:
        sftp.close()
    temporary.replace(destination)


def download_optional_file(client, remote_path, local_path):
    try:
        download_file(client, remote_path, local_path)
        return True
    except (FileNotFoundError, IOError, OSError):
        destination = Path(local_path)
        temporary = destination.with_suffix(destination.suffix + ".part")
        temporary.unlink(missing_ok=True)
        return False


def download_tree(client, remote_dir, local_dir):
    """Recursively archive a remote evidence directory over management SSH."""
    destination = Path(local_dir)
    destination.mkdir(parents=True, exist_ok=True)
    sftp = client.open_sftp()
    try:
        def visit(source, target):
            Path(target).mkdir(parents=True, exist_ok=True)
            for entry in sftp.listdir_attr(source):
                remote_item = posixpath.join(source, entry.filename)
                local_item = Path(target) / entry.filename
                if stat.S_ISDIR(entry.st_mode):
                    visit(remote_item, local_item)
                elif stat.S_ISREG(entry.st_mode):
                    temporary = local_item.with_suffix(
                        local_item.suffix + ".part")
                    temporary.parent.mkdir(parents=True, exist_ok=True)
                    sftp.get(remote_item, str(temporary))
                    temporary.replace(local_item)

        visit(remote_dir.replace("\\", "/"), destination)
    finally:
        sftp.close()


def _connect_hosts_once(args, root_required=True):
    if args.bore_port:
        outer = socks5_socket(
            args.socks_host, args.socks_port, args.bore_host, args.bore_port)
        client8g = key_client(
            args.bore_host, args.bore_port, "fsuser", args.key, sock=outer)
    else:
        client8g = key_client(
            CLIENT_IPV6,
            22,
            "fsuser",
            args.key,
            bind=CLIENT_BIND_IPV6,
        )
    root = None
    root_error = ""
    try:
        root = jump_client(
            client8g,
            "10.112.81.135",
            22,
            "root",
            key_path=args.key,
            banner_timeout=args.root_banner_timeout,
        )
    except Exception as error:
        root_error = repr(error)
        if root_required:
            client8g.close()
            raise
    third = None
    third_errors = []
    # Prefer the original direct campus-LAN path.  Some WLAN sessions cannot
    # route from the client machine to the third host even though the root
    # server can, so transparently fall back to client -> root -> third.
    for jump, route_name in ((client8g, "client8g"), (root, "root4090")):
        if jump is None:
            continue
        try:
            try:
                third = jump_client(
                    jump,
                    "10.129.248.111",
                    22,
                    "pc",
                    key_path=args.key,
                )
            except paramiko.AuthenticationException:
                if not args.third_password:
                    raise
                third = jump_client(
                    jump,
                    "10.129.248.111",
                    22,
                    "pc",
                    password=args.third_password,
                )
            break
        except Exception as error:
            third_errors.append(f"{route_name}: {error!r}")
    if third is None:
        if root is not None:
            root.close()
        client8g.close()
        raise RuntimeError(
            "third host is unreachable through all management routes: "
            + "; ".join(third_errors))
    return {
        "client8g": client8g,
        "third": third,
        "root4090": root,
        "root_error": root_error,
    }


def connect_hosts(args, root_required=True):
    attempts = 8 if args.bore_port else 1
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            return _connect_hosts_once(args, root_required=root_required)
        except Exception as error:
            last_error = error
            if attempt < attempts:
                time.sleep(min(attempt * 1.5, 8))
    raise last_error


def connect_root_only(args):
    client8g = key_client(
        CLIENT_IPV6,
        22,
        "fsuser",
        args.key,
        bind=CLIENT_BIND_IPV6,
    )
    try:
        root = jump_client(
            client8g,
            "10.112.81.135",
            22,
            "root",
            key_path=args.key,
            banner_timeout=args.root_banner_timeout,
        )
    except Exception:
        client8g.close()
        raise
    return {"client8g": client8g, "root4090": root}


def clean_root_memory(args):
    """Release reclaimable Linux page cache without killing user processes."""
    hosts = connect_root_only(args)
    command = r"""
set -euo pipefail
training_roles="$( { pgrep -af \
  '[r]un_ggeur|[r]un_remaining_queue_root|[f]ederatedscope' || true; } |
  wc -l)"
availability_roles="$( { pgrep -af \
  '[b]enchmark_headonly_concurrent_availability' || true; } | wc -l)"
printf 'training_roles=%s\n' "$training_roles"
printf 'availability_roles=%s\n' "$availability_roles"
if [[ "$training_roles" -ne 0 || "$availability_roles" -ne 0 ]]; then
  echo 'refuse_cache_drop_while_experiment_roles_are_active' >&2
  exit 20
fi
echo '-- before --'
free -m
printf 'cached_kb=%s\n' "$(awk '/^Cached:/ {print $2}' /proc/meminfo)"
printf 'swap_cached_kb=%s\n' "$(awk '/^SwapCached:/ {print $2}' /proc/meminfo)"
sync
if sh -c 'echo 3 >/proc/sys/vm/drop_caches' 2>/dev/null; then
  echo 'cache_drop=completed'
else
  echo 'cache_drop=blocked_read_only'
fi
sleep 2
echo '-- after --'
free -m
printf 'cached_kb=%s\n' "$(awk '/^Cached:/ {print $2}' /proc/meminfo)"
printf 'swap_cached_kb=%s\n' "$(awk '/^SwapCached:/ {print $2}' /proc/meminfo)"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader 2>/dev/null || true
fi
"""
    try:
        result = run(hosts["root4090"], command, timeout=60, check=False)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if result["status"]:
            raise SystemExit(result["status"])
    finally:
        close_hosts(hosts)


def close_hosts(hosts):
    for name in ("root4090", "third", "client8g"):
        client = hosts.get(name)
        if client is not None:
            client.close()


def root_check(args):
    client8g = key_client(
        CLIENT_IPV6,
        22,
        "fsuser",
        args.key,
        bind=CLIENT_BIND_IPV6,
    )
    root = None
    third = None
    try:
        probe_script = r"""
$ProgressPreference = 'SilentlyContinue'
function Test-RemotePort([string]$HostName, [int]$Port) {
    $tcp = [System.Net.Sockets.TcpClient]::new()
    try {
        $task = $tcp.ConnectAsync($HostName, $Port)
        if (-not $task.Wait(3000)) { return $false }
        return $tcp.Connected
    } catch {
        return $false
    } finally {
        $tcp.Dispose()
    }
}
"client8g_status=OK"
"client8g_hostname=$env:COMPUTERNAME"
$client8gRoles = @(Get-CimInstance Win32_Process |
  Where-Object {
    $_.CommandLine -like '*federatedscope.main*' -or
    $_.CommandLine -like '*hierarchical_subserver.py*' -or
    $_.CommandLine -like '*benchmark_headonly_concurrent_availability.py*'
  })
$client8gOs = Get-CimInstance Win32_OperatingSystem
"client8g_roles=$($client8gRoles.Count)"
"client8g_mem_available_kb=$($client8gOs.FreePhysicalMemory)"
Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
  Where-Object {
    $_.IPAddress -notlike '127.*' -and
    $_.AddressState -eq 'Preferred'
  } |
  Sort-Object InterfaceIndex |
  ForEach-Object {
    "client8g_ipv4=$($_.IPAddress)/$($_.PrefixLength) if=$($_.InterfaceAlias)"
  }
foreach ($target in @('10.112.81.135', '10.129.248.111')) {
  try {
    $route = Find-NetRoute -RemoteIPAddress $target -ErrorAction Stop
    "client8g_route=$target via=$($route.NextHop) if=$($route.InterfaceAlias)"
  } catch {
    "client8g_route=$target UNAVAILABLE"
  }
}
"root_port_22=$(Test-RemotePort '10.112.81.135' 22)"
"root_port_60049=$(Test-RemotePort '10.112.81.135' 60049)"
"root_port_60050=$(Test-RemotePort '10.112.81.135' 60050)"
"root_port_62010=$(Test-RemotePort '10.112.81.135' 62010)"
"third_port_22=$(Test-RemotePort '10.129.248.111' 22)"
"third_port_61000=$(Test-RemotePort '10.129.248.111' 61000)"
"third_port_62020=$(Test-RemotePort '10.129.248.111' 62020)"
try {
    $control = Invoke-RestMethod -Uri 'http://10.112.81.135:60049/state' `
        -TimeoutSec 5
    "root_control=$($control | ConvertTo-Json -Compress)"
} catch {
    "root_control=UNAVAILABLE:$($_.Exception.Message)"
}
"""
        result = {
            "management_route": "local_to_8g_then_10.112.81.135",
            "client8g_probe": run(
                client8g,
                encoded_powershell(probe_script),
                timeout=30,
                check=False,
            ),
        }
        if "root_port_22=True" not in result["client8g_probe"]["stdout"]:
            result["root4090"] = {
                "status": -2,
                "stdout": "",
                "stderr": "TCP port 22 is unreachable from client8g",
            }
        else:
            try:
                root = jump_client(
                    client8g,
                    "10.112.81.135",
                    22,
                    "root",
                    key_path=args.key,
                    banner_timeout=args.root_banner_timeout,
                )
                result["root4090"] = run(
                    root,
                    (
                        "printf 'root_status=OK\\n'; "
                        "printf 'hostname=%s\\n' \"$(hostname)\"; "
                        "printf 'uptime=%s\\n' \"$(uptime -p)\"; "
                        "printf 'training_roles=%s\\n' "
                        "\"$(pgrep -af '[r]un_ggeur|"
                        "[r]un_remaining_queue_root|[f]ederatedscope' | "
                        "wc -l)\"; "
                        "printf 'availability_roles=%s\\n' "
                        "\"$(pgrep -af "
                        "'[b]enchmark_headonly_concurrent_availability' | "
                        "wc -l)\"; "
                        "free -m | awk 'NR==2 {printf "
                        "\"mem_available_mb=%s\\\\n\", $7}'; "
                        "nvidia-smi --query-gpu=index,name,memory.total,"
                        "memory.free,utilization.gpu "
                        "--format=csv,noheader; "
                        f"test -d {ROOT_REPO} && "
                        "printf 'project_status=OK\\n' || "
                        "printf 'project_status=MISSING\\n'; "
                        "if test -x /root/.local/share/mamba/envs/GGEUR/bin/python; "
                        "then /root/.local/share/mamba/envs/GGEUR/bin/python "
                        "-c 'import torch; print(\"torch_status=OK\", "
                        "torch.__version__, torch.cuda.is_available(), "
                        "torch.cuda.device_count())'; "
                        "else printf 'python_env=MISSING\\n'; fi"
                    ),
                    timeout=30,
                )
            except Exception as error:
                result["root4090"] = {
                    "status": -1,
                    "stdout": "",
                    "stderr": f"{type(error).__name__}: {error}",
                }
        if "third_port_22=True" in result["client8g_probe"]["stdout"]:
            try:
                third = jump_client(
                    client8g,
                    "10.129.248.111",
                    22,
                    "pc",
                    key_path=args.key,
                    banner_timeout=args.root_banner_timeout,
                )
                third_script = r"""
$roles = @(Get-CimInstance Win32_Process | Where-Object {
  $_.CommandLine -like '*federatedscope.main*' -or
  $_.CommandLine -like '*hierarchical_subserver.py*' -or
  $_.CommandLine -like '*benchmark_headonly_concurrent_availability.py*'
})
$os = Get-CimInstance Win32_OperatingSystem
"third_status=OK"
"third_hostname=$env:COMPUTERNAME"
"third_roles=$($roles.Count)"
"third_mem_available_kb=$($os.FreePhysicalMemory)"
"""
                result["third"] = run(
                    third,
                    encoded_powershell(third_script),
                    timeout=30,
                )
            except Exception as error:
                result["third"] = {
                    "status": -1,
                    "stdout": "",
                    "stderr": f"{type(error).__name__}: {error}",
                }
        print(json.dumps(result, ensure_ascii=True, indent=2))
    finally:
        if third is not None:
            third.close()
        if root is not None:
            root.close()
        client8g.close()


def deploy(args):
    hosts = connect_hosts(args, root_required=False)
    repo_root = Path(__file__).resolve().parents[3]
    local_rel = repo_root / "scripts/distributed_scripts" / \
        "ggeur_concurrent_availability"
    benchmark = repo_root / "scripts" / \
        "benchmark_headonly_concurrent_availability.py"
    payload = Path(args.payload_file).resolve()
    result = {"deployed": [], "root_error": hosts["root_error"]}
    try:
        third_dirs = rf"""
$dirs = @(
  '{THIRD_REPO}\scripts',
  '{THIRD_REPO}\{REL_DIR.replace("/", chr(92))}',
  '{THIRD_REPO}\exp\concurrent_availability\artifacts'
)
foreach ($dir in $dirs) {{
  New-Item -ItemType Directory -Force -Path $dir | Out-Null
}}
"""
        client_dirs = rf"""
$dirs = @(
  '{CLIENT_REPO}\scripts',
  '{CLIENT_REPO}\{REL_DIR.replace("/", chr(92))}',
  '{CLIENT_REPO}\exp\concurrent_availability\artifacts'
)
foreach ($dir in $dirs) {{
  New-Item -ItemType Directory -Force -Path $dir | Out-Null
}}
"""
        run(hosts["third"], encoded_powershell(third_dirs))
        run(hosts["client8g"], encoded_powershell(client_dirs))

        common_name = "benchmark_headonly_concurrent_availability.py"
        artifact_name = payload.name
        upload_file(
            hosts["third"],
            benchmark,
            f"{THIRD_REPO}/scripts/{common_name}".replace("\\", "/"),
        )
        for name in (
                "run_server_windows.ps1",
                "run_probe_windows.ps1",
                "run_load_windows.ps1"):
            upload_file(
                hosts["third"],
                local_rel / name,
                f"{THIRD_REPO}/{REL_DIR}/{name}".replace("\\", "/"),
            )
        upload_file(
            hosts["third"],
            payload,
            f"{THIRD_REPO}/exp/concurrent_availability/artifacts/"
            f"{artifact_name}".replace("\\", "/"),
        )
        result["deployed"].append("third")

        upload_file(
            hosts["client8g"],
            benchmark,
            f"{CLIENT_REPO}/scripts/{common_name}".replace("\\", "/"),
        )
        for name in ("run_load_windows.ps1", "run_probe_windows.ps1"):
            upload_file(
                hosts["client8g"],
                local_rel / name,
                f"{CLIENT_REPO}/{REL_DIR}/{name}".replace("\\", "/"),
            )
        upload_file(
            hosts["client8g"],
            local_rel / "observe_connections_tcpvcon.ps1",
            f"{CLIENT_REPO}/{REL_DIR}/observe_connections_tcpvcon.ps1"
            .replace("\\", "/"),
        )
        upload_file(
            hosts["client8g"],
            payload,
            f"{CLIENT_REPO}/exp/concurrent_availability/artifacts/"
            f"{artifact_name}".replace("\\", "/"),
        )
        result["deployed"].append("client8g")

        if hosts["root4090"] is not None:
            run(
                hosts["root4090"],
                f"mkdir -p {ROOT_REPO}/scripts "
                f"{ROOT_REPO}/{REL_DIR} "
                f"{ROOT_REPO}/exp/concurrent_availability/artifacts",
            )
            upload_file(
                hosts["root4090"],
                benchmark,
                f"{ROOT_REPO}/scripts/{common_name}",
                newline="lf",
            )
            upload_file(
                hosts["root4090"],
                local_rel / "run_server_linux.sh",
                f"{ROOT_REPO}/{REL_DIR}/run_server_linux.sh",
                newline="lf",
            )
            upload_file(
                hosts["root4090"],
                payload,
                f"{ROOT_REPO}/exp/concurrent_availability/artifacts/"
                f"{artifact_name}",
            )
            run(
                hosts["root4090"],
                f"chmod 755 {ROOT_REPO}/{REL_DIR}/run_server_linux.sh",
            )
            result["deployed"].append("root4090")
    finally:
        close_hosts(hosts)
    print(json.dumps(result, ensure_ascii=True, indent=2))
    if "root4090" not in result["deployed"]:
        raise SystemExit(3)


def preflight(args):
    # The performance test is a three-machine test.  Treat an unreachable root
    # server as a failed preflight instead of silently reporting success.
    hosts = connect_hosts(args, root_required=True)
    linux = r"""
printf 'time_unix=%s\n' "$(date +%s)"
printf 'gpu_name=%s\n' "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
printf 'gpu_memory_free_mib=%s\n' "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)"
printf 'training_roles=%s\n' "$(pgrep -af '[m]dsent_lstm_fedavg|[r]un_ggeur_fs.py' | wc -l)"
printf 'queue_drivers=%s\n' "$(pgrep -af '[r]un_remaining_queue_root.sh' | wc -l)"
printf 'nofile=%s\n' "$(ulimit -n)"
printf 'mem_available_kb=%s\n' "$(awk '/MemAvailable/ {print $2}' /proc/meminfo)"
printf 'swap_free_kb=%s\n' "$(awk '/SwapFree/ {print $2}' /proc/meminfo)"
printf 'formal_ports=%s\n' "$(ss -ltn | grep -Ec ':(60050|61000|61001|61002|61003)[[:space:]]' || true)"
printf 'availability_ports=%s\n' "$(ss -ltn | grep -Ec ':(6201[0-4])[[:space:]]' || true)"
"""
    windows = rf"""
$training = @(Get-CimInstance Win32_Process | Where-Object {{
  $_.CommandLine -like '*federatedscope.main*' -or
  $_.CommandLine -like '*hierarchical_subserver.py*'
}})
$formal = @(Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue |
  Where-Object {{ $_.LocalPort -in @(60050,61000,61001,61002,61003) }})
$availability = @(Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue |
  Where-Object {{ $_.LocalPort -ge 62010 -and $_.LocalPort -le 62039 }})
$os = Get-CimInstance Win32_OperatingSystem
function Test-RemotePort([string]$RemoteHost, [int]$RemotePort) {{
  $tcp = [Net.Sockets.TcpClient]::new()
  try {{
    $task = $tcp.ConnectAsync($RemoteHost, $RemotePort)
    return $task.Wait(2000) -and $tcp.Connected
  }} catch {{
    return $false
  }} finally {{
    $tcp.Dispose()
  }}
}}
"time_unix=$([DateTimeOffset]::UtcNow.ToUnixTimeSeconds())"
"training_roles=$($training.Count)"
"mem_available_kb=$($os.FreePhysicalMemory)"
"formal_ports=$($formal.Count)"
"availability_ports=$($availability.Count)"
"root_port_22=$(Test-RemotePort '10.112.81.135' 22)"
"root_port_60049=$(Test-RemotePort '10.112.81.135' 60049)"
"root_port_60050=$(Test-RemotePort '10.112.81.135' 60050)"
try {{
  $control = Invoke-RestMethod -Uri 'http://10.112.81.135:60049/state' `
    -TimeoutSec 4
  "root_control=$($control | ConvertTo-Json -Compress)"
}} catch {{
  "root_control=UNAVAILABLE:$($_.Exception.Message)"
}}
"-- training process sample --"
$training | Sort-Object ProcessId | Select-Object -First 8 | ForEach-Object {{
  $cmd = [string]$_.CommandLine
  if ($cmd.Length -gt 240) {{ $cmd = $cmd.Substring(0, 240) }}
  "pid=$($_.ProcessId) created=$($_.CreationDate) working_set=$($_.WorkingSetSize) cmd=$cmd"
}}
"-- queue tasks --"
Get-ScheduledTask -ErrorAction SilentlyContinue | Where-Object {{
  $_.TaskName -like 'GGEUR-*' -or $_.TaskName -like '*remaining*'
}} | ForEach-Object {{
  $info = Get-ScheduledTaskInfo -TaskName $_.TaskName
  "task=$($_.TaskName) state=$($_.State) result=$($info.LastTaskResult)"
}}
"-- current case recent logs --"
$candidateRoots = @(
  '{THIRD_REPO}\scripts\distributed_scripts\ggeur_hierarchical_3machine\runs\{args.accuracy_run_id}\mdsent_lstm_fedavg',
  '{CLIENT_REPO}\scripts\distributed_scripts\ggeur_hierarchical_3machine\runs\{args.accuracy_run_id}\mdsent_lstm_fedavg'
)
foreach ($candidateRoot in $candidateRoots) {{
  if (Test-Path -LiteralPath $candidateRoot) {{
    Get-ChildItem -LiteralPath $candidateRoot -Recurse -File `
      -ErrorAction SilentlyContinue |
      Where-Object {{ $_.Name -match '\.(log|json|tsv)$' }} |
      Sort-Object LastWriteTime -Descending |
      Select-Object -First 12 |
      ForEach-Object {{
        "file=$($_.FullName) bytes=$($_.Length) modified=$($_.LastWriteTime.ToString('o'))"
      }}
  }}
}}
netsh int ipv4 show dynamicport tcp
"""
    result = {
        "root4090": run(hosts["root4090"], linux, check=False),
        "third": run(
            hosts["third"], encoded_powershell(windows), check=False),
        "client8g": run(
            hosts["client8g"], encoded_powershell(windows), check=False),
    }
    close_hosts(hosts)
    print(json.dumps(result, ensure_ascii=True, indent=2))
    failed = {
        name: payload for name, payload in result.items()
        if payload.get("status") != 0
    }
    if failed:
        raise RuntimeError(
            "三机环境预检失败：{}".format(
                ", ".join(sorted(failed))))


def suspend_queue(args):
    hosts = connect_hosts(args, root_required=True)
    command = rf"""
set -euo pipefail
run={ROOT_REPO}/scripts/distributed_scripts/ggeur_hierarchical_3machine/runs/{args.accuracy_run_id}
pid="$(pgrep -f '[r]un_remaining_queue_root.sh' | head -1)"
[[ "$pid" =~ ^[0-9]+$ ]]
state="$(awk '{{print $3}}' /proc/$pid/stat)"
if [[ "$state" != T && "$state" != t ]]; then
  kill -STOP "$pid"
fi
printf '%s\n' "$pid" >"$run/queue_state/root_queue.suspended_for_availability.pid"
printf '%s\t%s\t%s\n' "$(date --iso-8601=seconds)" \
  "QUEUE" "SUSPENDED_FOR_CONCURRENT_AVAILABILITY:pid=$pid" \
  >>"$run/queue_state/root_queue.tsv"
printf 'queue_pid=%s\nstate=%s\n' "$pid" "$(awk '{{print $3}}' /proc/$pid/stat)"
"""
    try:
        print(json.dumps(
            run(hosts["root4090"], command), ensure_ascii=False, indent=2))
    finally:
        close_hosts(hosts)


def resume_queue(args):
    hosts = connect_hosts(args, root_required=True)
    command = rf"""
set -euo pipefail
run={ROOT_REPO}/scripts/distributed_scripts/ggeur_hierarchical_3machine/runs/{args.accuracy_run_id}
pid="$(cat "$run/queue_state/root_queue.suspended_for_availability.pid")"
[[ "$pid" =~ ^[0-9]+$ ]]
kill -CONT "$pid"
rm -f "$run/queue_state/root_queue.suspended_for_availability.pid"
printf '%s\t%s\t%s\n' "$(date --iso-8601=seconds)" \
  "QUEUE" "RESUMED_AFTER_CONCURRENT_AVAILABILITY:pid=$pid" \
  >>"$run/queue_state/root_queue.tsv"
printf 'queue_pid=%s\nstate=%s\n' "$pid" "$(awk '{{print $3}}' /proc/$pid/stat)"
"""
    try:
        print(json.dumps(
            run(hosts["root4090"], command), ensure_ascii=False, indent=2))
    finally:
        close_hosts(hosts)


def pause_accuracy(args):
    hosts = connect_hosts(args, root_required=False)
    case_name = args.accuracy_case

    def windows_pause(repo_dir, queue_task, host_label):
        case_dir = (
            f"{repo_dir}\\scripts\\distributed_scripts\\"
            f"ggeur_hierarchical_3machine\\runs\\"
            f"{args.accuracy_run_id}\\{case_name}"
        )
        script = rf"""
$ErrorActionPreference = 'Stop'
$caseDir = '{case_dir}'
$stamp = (Get-Date).ToString('yyyyMMdd_HHmmss')
$attemptDir = Join-Path $caseDir `
  "attempts\attempt_${{stamp}}_paused_for_concurrent_availability\{host_label}"
New-Item -ItemType Directory -Force -Path $attemptDir | Out-Null
$roles = @(Get-CimInstance Win32_Process | Where-Object {{
  $_.CommandLine -like '*{args.accuracy_run_id}*' -and
  $_.CommandLine -like '*{case_name}*' -and
  ($_.CommandLine -like '*federatedscope.main*' -or
   $_.CommandLine -like '*hierarchical_subserver.py*')
}})
$tasks = @(Get-ScheduledTask -ErrorAction SilentlyContinue | Where-Object {{
  $_.TaskName -eq '{queue_task}' -or
  $_.TaskName -like 'GGEUR-{case_name}-*'
}})
@{{
  paused_at = (Get-Date).ToString('o')
  reason = 'PAUSED_FOR_CONCURRENT_AVAILABILITY'
  host_role = '{host_label}'
  run_id = '{args.accuracy_run_id}'
  case_name = '{case_name}'
  processes = @($roles | Select-Object ProcessId, CreationDate,
    WorkingSetSize, CommandLine)
  tasks = @($tasks | Select-Object TaskName, State)
}} | ConvertTo-Json -Depth 6 |
  Set-Content -LiteralPath (Join-Path $attemptDir 'pause_record.json') `
    -Encoding UTF8
if (Test-Path -LiteralPath (Join-Path $caseDir 'logs')) {{
  Copy-Item -LiteralPath (Join-Path $caseDir 'logs') `
    -Destination (Join-Path $attemptDir 'logs') -Recurse -Force
}}
foreach ($task in $tasks) {{
  if ($task.State -eq 'Running') {{
    Stop-ScheduledTask -TaskName $task.TaskName -ErrorAction Continue
  }}
}}
Start-Sleep -Seconds 3
$remaining = @(Get-CimInstance Win32_Process | Where-Object {{
  $_.CommandLine -like '*{args.accuracy_run_id}*' -and
  $_.CommandLine -like '*{case_name}*' -and
  ($_.CommandLine -like '*federatedscope.main*' -or
   $_.CommandLine -like '*hierarchical_subserver.py*')
}})
foreach ($process in $remaining) {{
  Stop-Process -Id $process.ProcessId -Force -ErrorAction Continue
}}
Start-Sleep -Seconds 2
$left = @(Get-CimInstance Win32_Process | Where-Object {{
  $_.CommandLine -like '*{args.accuracy_run_id}*' -and
  $_.CommandLine -like '*{case_name}*' -and
  ($_.CommandLine -like '*federatedscope.main*' -or
   $_.CommandLine -like '*hierarchical_subserver.py*')
}})
$markerDir = Split-Path -Parent $caseDir
$stateDir = Join-Path $markerDir 'queue_state'
New-Item -ItemType Directory -Force -Path $stateDir | Out-Null
"$((Get-Date).ToString('o'))`t{case_name}`tPAUSED_FOR_CONCURRENT_AVAILABILITY:{host_label}" |
  Add-Content -LiteralPath (Join-Path $stateDir 'availability_pause.tsv')
"attempt_dir=$attemptDir"
"roles_before=$($roles.Count)"
"roles_after=$($left.Count)"
"tasks_stopped=$($tasks.Count)"
if ($left.Count -ne 0) {{ throw "accuracy roles remain after pause" }}
"""
        return script

    results = {
        "third": run(
            hosts["third"],
            encoded_powershell(windows_pause(
                THIRD_REPO,
                "GGEUR-final-subserver-queue",
                "third",
            )),
            timeout=300,
            check=False,
        ),
        "client8g": run(
            hosts["client8g"],
            encoded_powershell(windows_pause(
                CLIENT_REPO,
                "GGEUR-final-client-queue",
                "client8g",
            )),
            timeout=300,
            check=False,
        ),
    }

    if results["third"]["status"] or results["client8g"]["status"]:
        close_hosts(hosts)
        print(json.dumps(results, ensure_ascii=False, indent=2))
        raise SystemExit(4)

    root = hosts.get("root4090")
    root_errors = []
    if root is None:
        for _ in range(args.root_recovery_attempts):
            time.sleep(args.root_recovery_interval_sec)
            try:
                root = jump_client(
                    hosts["client8g"],
                    "10.112.81.135",
                    22,
                    "root",
                    key_path=args.key,
                    banner_timeout=args.root_banner_timeout,
                )
                hosts["root4090"] = root
                break
            except Exception as error:
                root_errors.append(repr(error))

    if root is not None:
        root_case = (
            f"{ROOT_REPO}/scripts/distributed_scripts/"
            f"ggeur_hierarchical_3machine/runs/"
            f"{args.accuracy_run_id}/{case_name}"
        )
        root_pause = rf"""
set -euo pipefail
case_dir={root_case}
run_dir={ROOT_REPO}/scripts/distributed_scripts/ggeur_hierarchical_3machine/runs/{args.accuracy_run_id}
stamp="$(date +%Y%m%d_%H%M%S)"
attempt_dir="$case_dir/attempts/attempt_${{stamp}}_paused_for_concurrent_availability/root4090"
mkdir -p "$attempt_dir" "$run_dir/queue_state"
date --iso-8601=seconds >"$attempt_dir/paused_at.txt"
printf '%s\n' PAUSED_FOR_CONCURRENT_AVAILABILITY \
  >"$attempt_dir/reason.txt"
pgrep -af '[m]dsent_lstm_fedavg|[r]un_ggeur_fs.py' \
  >"$attempt_dir/processes.before.txt" || true
if [[ -d "$case_dir/logs" ]]; then
  cp -a "$case_dir/logs" "$attempt_dir/logs"
fi
queue_pid="$(pgrep -f '[r]un_remaining_queue_root.sh' | head -1 || true)"
if [[ "$queue_pid" =~ ^[0-9]+$ ]]; then
  kill -STOP "$queue_pid"
  printf '%s\n' "$queue_pid" \
    >"$run_dir/queue_state/root_queue.suspended_for_availability.pid"
fi
printf '%s\t%s\t%s\n' "$(date --iso-8601=seconds)" \
  "{case_name}" "PAUSED_FOR_CONCURRENT_AVAILABILITY:root4090" \
  >>"$run_dir/queue_state/root_queue.tsv"
pids="$(pgrep -f '[m]dsent_lstm_fedavg' || true)"
if [[ -n "$pids" ]]; then
  kill -TERM $pids || true
  for _ in $(seq 1 10); do
    sleep 1
    pids="$(pgrep -f '[m]dsent_lstm_fedavg' || true)"
    [[ -z "$pids" ]] && break
  done
  [[ -z "$pids" ]] || kill -KILL $pids || true
fi
sleep 2
printf 'attempt_dir=%s\nqueue_pid=%s\nremaining_roles=%s\n' \
  "$attempt_dir" "${{queue_pid:-}}" \
  "$(pgrep -af '[m]dsent_lstm_fedavg' | wc -l)"
"""
        results["root4090"] = run(
            root, root_pause, timeout=300, check=False)
    else:
        results["root4090"] = {
            "status": -1,
            "stdout": "",
            "stderr": (
                "root management did not recover after Windows-side pause; "
                + "; ".join(root_errors)
            ),
        }
    close_hosts(hosts)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    if results["root4090"]["status"]:
        raise SystemExit(5)


def ps_task_script(task_name, wrapper, arguments, repo_dir, log_dir):
    safe_task = task_name.replace("'", "''")
    safe_wrapper = wrapper.replace("'", "''")
    safe_repo = repo_dir.replace("'", "''")
    safe_log = log_dir.replace("'", "''")
    command = (
        f'powershell.exe -NoProfile -ExecutionPolicy Bypass '
        f'-File "{safe_wrapper}" {arguments} '
        f'1>"{safe_log}\\task.stdout.log" '
        f'2>"{safe_log}\\task.stderr.log"'
    )
    command = command.replace("'", "''")
    return rf"""
$taskName = '{safe_task}'
$runDir = '{safe_log}'
New-Item -ItemType Directory -Force -Path $runDir | Out-Null
$old = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($old) {{
  Stop-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
  Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
}}
$cmdPath = Join-Path $runDir 'task.run.cmd'
@('@echo off', '{command}') |
  Set-Content -LiteralPath $cmdPath -Encoding ASCII
$action = New-ScheduledTaskAction `
  -Execute "$env:SystemRoot\System32\cmd.exe" `
  -Argument "/d /c `"$cmdPath`"" `
  -WorkingDirectory '{safe_repo}'
$identity = [Security.Principal.WindowsIdentity]::GetCurrent().Name
$principal = New-ScheduledTaskPrincipal `
  -UserId $identity -LogonType S4U -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet `
  -ExecutionTimeLimit (New-TimeSpan -Hours 2) `
  -MultipleInstances IgnoreNew `
  -AllowStartIfOnBatteries `
  -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName $taskName -Action $action `
  -Principal $principal -Settings $settings -Force | Out-Null
Start-ScheduledTask -TaskName $taskName
"TASK=$taskName STATE=$((Get-ScheduledTask -TaskName $taskName).State)"
"""


def start_run(args):
    hosts = connect_hosts(args, root_required=True)
    run_id = args.validation_run_id
    phase = args.start_phase
    local_repo = Path(__file__).resolve().parents[3]
    schedule_path = (
        local_repo / "exp" / "concurrent_availability" / run_id /
        "launch_schedule.json")
    payload_name = Path(args.payload_file).name
    root_registration_connections = (
        args.root_subservers * args.root_clients_per_subserver
    )
    edge_clients = args.third_subservers * args.third_clients_per_subserver
    third_training_clients = (
        args.third_subservers
        * args.third_training_clients_per_subserver
    )
    third_regular_clients_per_subserver = (
        args.third_clients_per_subserver
        - args.third_training_clients_per_subserver
    )
    total_clients = edge_clients
    if total_clients != 10_000:
        close_hosts(hosts)
        raise ValueError(
            f"formal validation requires exactly 10,000 edge clients, got "
            f"{total_clients}")
    if args.third_subservers != 10:
        close_hosts(hosts)
        raise ValueError(
            "formal validation requires exactly 10 child servers, got "
            f"{args.third_subservers}")
    if root_registration_connections != args.third_subservers:
        close_hosts(hosts)
        raise ValueError(
            "root registration count must equal child-server count: "
            f"{root_registration_connections} != {args.third_subservers}")
    if third_regular_clients_per_subserver < 0:
        close_hosts(hosts)
        raise ValueError(
            "third training clients per subserver cannot exceed the "
            "third total clients per subserver")
    requests_per_client = (
        math.ceil(args.request_duration_sec / args.poll_interval_sec)
        if args.request_duration_sec > 0 else 0
    )
    expected_successful_requests = edge_clients * requests_per_client
    safety_linux = r"""
set -euo pipefail
if pgrep -af '[m]dsent_lstm_fedavg|[r]un_ggeur_fs.py' >/dev/null; then
  echo 'formal training role still active' >&2
  exit 20
fi
if pgrep -af '[b]enchmark_headonly_concurrent_availability.py|[r]un_server_linux.sh' >/dev/null; then
  echo 'concurrent availability role still active' >&2
  exit 22
fi
if ss -ltn | grep -Eq ':(60050|61000|61001|61002|61003)[[:space:]]'; then
  echo 'formal training port still active' >&2
  exit 21
fi
"""
    safety_windows = r"""
$roles = @(Get-CimInstance Win32_Process | Where-Object {
  $_.CommandLine -like '*federatedscope.main*' -or
  $_.CommandLine -like '*hierarchical_subserver.py*' -or
  $_.CommandLine -like '*benchmark_headonly_concurrent_availability.py*'
})
$formal = @(Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue |
  Where-Object { $_.LocalPort -in @(60050,61000,61001,61002,61003) })
if ($roles.Count -gt 0 -or $formal.Count -gt 0) {
  throw "formal accuracy work is still active: roles=$($roles.Count) ports=$($formal.Count)"
}
"""
    archive_results = {}
    remote_times = []
    if phase in {"all", "root"}:
        run(hosts["root4090"], safety_linux)
        run(hosts["third"], encoded_powershell(safety_windows))
        run(hosts["client8g"], encoded_powershell(safety_windows))
        archive_stamp = time.strftime(
            "attempt_%Y%m%d_%H%M%S_prestart_archive")
    else:
        archive_stamp = "not_applicable"

    archive_linux = rf"""
set -euo pipefail
run_root={ROOT_REPO}/exp/concurrent_availability/{run_id}
source="$run_root/root4090"
target="$run_root/attempts/{archive_stamp}/root4090"
if [[ -d "$source" ]]; then
  mkdir -p "$(dirname "$target")"
  mv "$source" "$target"
  printf 'archived=%s\n' "$target"
else
  printf 'archived=none\n'
fi
"""
    archive_windows = rf"""
$ErrorActionPreference = 'Stop'
$repo = if ($env:COMPUTERNAME -eq 'LKX') {{
  '{CLIENT_REPO}'
}} else {{
  '{THIRD_REPO}'
}}
$roles = if ($env:COMPUTERNAME -eq 'LKX') {{
  @('client8g', 'client_probe')
}} else {{
  @('third', 'third_root_registration', 'third_root_registration_task',
    'third_training_clients', 'third_training_task')
}}
$runRoot = Join-Path $repo 'exp\concurrent_availability\{run_id}'
$attemptRoot = Join-Path $runRoot 'attempts\{archive_stamp}'
$archived = @()
foreach ($role in $roles) {{
  $source = Join-Path $runRoot $role
  if (Test-Path -LiteralPath $source) {{
    New-Item -ItemType Directory -Force -Path $attemptRoot | Out-Null
    Move-Item -LiteralPath $source -Destination (Join-Path $attemptRoot $role)
    $archived += $role
  }}
}}
"archived=$($archived -join ',')"
"""
    if phase in {"all", "root"}:
        archive_results = {
            "root4090": run(hosts["root4090"], archive_linux),
            "third": run(
                hosts["third"], encoded_powershell(archive_windows)),
            "client8g": run(
                hosts["client8g"], encoded_powershell(archive_windows)),
        }
        remote_times = [
            int(run(hosts["root4090"], "date +%s")["stdout"]),
            int(run(
                hosts["third"],
                encoded_powershell(
                    "[DateTimeOffset]::UtcNow.ToUnixTimeSeconds()"),
            )["stdout"].splitlines()[-1]),
            int(run(
                hosts["client8g"],
                encoded_powershell(
                    "[DateTimeOffset]::UtcNow.ToUnixTimeSeconds()"),
            )["stdout"].splitlines()[-1]),
        ]
        start_at = max(remote_times) + args.start_delay_sec
        observe_at = start_at + args.probe_delay_sec
        probe_at = observe_at
        active_until = probe_at + args.active_after_probe_sec
        schedule_path.parent.mkdir(parents=True, exist_ok=True)
        schedule_path.write_text(json.dumps({
            "run_id": run_id,
            "remote_times": remote_times,
            "start_at_unix": start_at,
            "observe_at_unix": observe_at,
            "probe_at_unix": probe_at,
            "active_until_unix": active_until,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        if not schedule_path.is_file():
            close_hosts(hosts)
            raise RuntimeError(
                f"launch schedule is missing; run root phase first: {schedule_path}")
        schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
        if schedule.get("run_id") != run_id:
            close_hosts(hosts)
            raise RuntimeError("launch schedule run id does not match")
        remote_times = schedule["remote_times"]
        start_at = int(schedule["start_at_unix"])
        observe_at = int(schedule["observe_at_unix"])
        probe_at = int(schedule["probe_at_unix"])
        active_until = int(schedule["active_until_unix"])

    root_payload = (
        f"{ROOT_REPO}/exp/concurrent_availability/artifacts/{payload_name}")
    root_command = f"""
set -euo pipefail
cd {ROOT_REPO}
mkdir -p exp/concurrent_availability/{run_id}/root4090
RUN_ID={run_id} PAYLOAD_FILE={root_payload} \
BASE_PORT=62010 SUBSERVERS={args.root_subservers} \
CLIENTS_PER_SUBSERVER={args.root_clients_per_subserver} \
EXPECTED_PROBES=0 EXPECTED_TRAINING_CLIENTS=0 \
START_AT_UNIX={start_at} \
OBSERVE_AT_UNIX={observe_at} \
ACTIVE_UNTIL_UNIX={active_until} \
nohup bash {REL_DIR}/run_server_linux.sh \
  >exp/concurrent_availability/{run_id}/root4090/task.stdout.log \
  2>exp/concurrent_availability/{run_id}/root4090/task.stderr.log &
echo "PID=$!"
"""
    root_started = None
    if phase in {"all", "root"}:
        root_started = run(hosts["root4090"], root_command)

    third_payload = (
        f"{THIRD_REPO}\\exp\\concurrent_availability\\artifacts\\"
        f"{payload_name}")
    third_wrapper = (
        f"{THIRD_REPO}\\{REL_DIR.replace('/', chr(92))}\\"
        "run_server_windows.ps1")
    third_args = (
        f'-RunId "{run_id}" -PayloadFile "{third_payload}" '
        f"-StartAtUnix {start_at} -ObserveAtUnix {observe_at} "
        f"-ActiveUntilUnix {active_until} "
        f"-BasePort 62020 -Subservers {args.third_subservers} "
        f"-ClientsPerSubserver {args.third_clients_per_subserver} "
        f"-ExpectedProbes 1 "
        f"-ExpectedTrainingClients {third_training_clients}")
    third_args += (
        f" -RequestDurationSec {args.request_duration_sec}"
        f" -RequestsPerClient {requests_per_client}"
        f" -PollIntervalSec {args.poll_interval_sec}"
        f" -PollWorkerConnections {args.poll_worker_connections}"
        f" -RequestStartDelaySec {args.request_start_delay_sec}")
    third_task = ps_task_script(
        f"GGEUR-availability-{run_id}-third-server",
        third_wrapper,
        third_args,
        THIRD_REPO,
        f"{THIRD_REPO}\\exp\\concurrent_availability\\{run_id}\\third",
    )
    third_started = None
    if phase in {"all", "third"}:
        third_started = run(
            hosts["third"], encoded_powershell(third_task))

    client_payload = (
        f"{CLIENT_REPO}\\exp\\concurrent_availability\\artifacts\\"
        f"{payload_name}")
    load_wrapper = (
        f"{CLIENT_REPO}\\{REL_DIR.replace('/', chr(92))}\\"
        "run_load_windows.ps1")
    third_load_wrapper = (
        f"{THIRD_REPO}\\{REL_DIR.replace('/', chr(92))}\\"
        "run_load_windows.ps1")
    root_registration_args = (
        f'-RunId "{run_id}" -PayloadFile "{third_payload}" '
        f'-ConnectHost "10.112.81.135" -BasePort 62010 '
        f"-Subservers {args.root_subservers} "
        f"-ClientsPerSubserver {args.root_clients_per_subserver} "
        f'-ClientIdOffset 1 -HostLabel "root_registration" '
        f'-RunDirLabel "third_root_registration" '
        f'-PythonBin "C:\\Users\\pc\\miniconda3\\envs\\cerp\\python.exe" '
        f'-RepoDir "{THIRD_REPO}"')
    child_load_args = (
        f'-RunId "{run_id}" -PayloadFile "{client_payload}" '
        f'-ConnectHost "10.129.248.111" -BasePort 62020 '
        f"-Subservers {args.third_subservers} "
        f"-ClientsPerSubserver "
        f"{third_regular_clients_per_subserver} "
        f'-ClientIdOffset 1 -HostLabel "child_servers"')
    root_registration_task = ps_task_script(
        f"GGEUR-availability-{run_id}-root-registration",
        third_load_wrapper,
        root_registration_args,
        THIRD_REPO,
        f"{THIRD_REPO}\\exp\\concurrent_availability\\{run_id}\\"
        "third_root_registration_task",
    )
    child_load_task = ps_task_script(
        f"GGEUR-availability-{run_id}-load-children",
        load_wrapper,
        child_load_args,
        CLIENT_REPO,
        f"{CLIENT_REPO}\\exp\\concurrent_availability\\{run_id}\\"
        "client8g\\child_task",
    )
    client_started = []
    if phase in {"all", "clients"}:
        if int(time.time()) >= start_at:
            close_hosts(hosts)
            raise RuntimeError(
                "launch schedule expired before clients were started; "
                "rerun the root, third and clients steps")
        client_started = [run(
            hosts["client8g"], encoded_powershell(child_load_task))]
        run(hosts["third"], encoded_powershell(root_registration_task))

    third_training_started = None
    if third_training_clients and phase in {"all", "clients"}:
        third_training_offset = (
            args.third_subservers
            * third_regular_clients_per_subserver
            + 1
        )
        third_training_args = (
            f'-RunId "{run_id}" -PayloadFile "{third_payload}" '
            f'-ConnectHost "10.129.248.111" -BasePort 62020 '
            f"-Subservers {args.third_subservers} "
            f"-ClientsPerSubserver "
            f"{args.third_training_clients_per_subserver} "
            f"-ClientIdOffset {third_training_offset} "
            f"-TrainClientCount {third_training_clients} "
            f"-TrainSteps {args.training_steps} "
            f"-TrainBatchSize {args.training_batch_size} "
            f"-TrainLearningRate {args.training_learning_rate} "
            f"-TrainingConcurrency {args.training_concurrency} "
            f'-HostLabel "third-local-train" '
            f'-RunDirLabel "third_training_clients" '
            f'-PythonBin "C:\\Users\\pc\\miniconda3\\envs\\cerp\\python.exe" '
            f'-RepoDir "{THIRD_REPO}"')
        third_training_task = ps_task_script(
            f"GGEUR-availability-{run_id}-third-training-clients",
            third_load_wrapper,
            third_training_args,
            THIRD_REPO,
            f"{THIRD_REPO}\\exp\\concurrent_availability\\{run_id}\\"
            "third_training_task",
        )
        third_training_started = run(
            hosts["third"],
            encoded_powershell(third_training_task),
        )

    probe_wrapper = (
        f"{CLIENT_REPO}\\{REL_DIR.replace('/', chr(92))}\\"
        "run_probe_windows.ps1")
    probe_args = (
        f'-RunId "{run_id}" -PayloadFile "{client_payload}" '
        f'-ConnectHost "10.129.248.111" -ConnectAtUnix {probe_at} '
        f'-Port 62020 -ProbeId "late-edge-probe" '
        f'-RunDirLabel "client_probe" -HostRole "client" '
        f'-PythonBin "{CLIENT_REPO}\\.venv_client_cpu\\Scripts\\python.exe" '
        f'-RepoDir "{CLIENT_REPO}"')
    probe_task = ps_task_script(
        f"GGEUR-availability-{run_id}-late-probe",
        probe_wrapper,
        probe_args,
        CLIENT_REPO,
        f"{CLIENT_REPO}\\exp\\concurrent_availability\\{run_id}\\"
        "client_probe",
    )
    probe_started = None
    if phase in {"all", "clients"}:
        probe_started = run(
            hosts["client8g"], encoded_powershell(probe_task))
    close_hosts(hosts)
    print(json.dumps({
        "run_id": run_id,
        "phase": phase,
        "prestart_archive": archive_results,
        "remote_times": remote_times,
        "start_at_unix": start_at,
        "observe_at_unix": observe_at,
        "probe_at_unix": probe_at,
        "active_until_unix": active_until,
        "root_registration_connections": root_registration_connections,
        "child_server_count": args.third_subservers,
        "edge_clients": edge_clients,
        "third_regular_clients": (
            args.third_subservers
            * third_regular_clients_per_subserver),
        "third_training_clients": third_training_clients,
        "total_clients": total_clients,
        "request_duration_sec": args.request_duration_sec,
        "poll_interval_sec": args.poll_interval_sec,
        "poll_worker_connections": args.poll_worker_connections,
        "requests_per_client": requests_per_client,
        "expected_successful_requests": expected_successful_requests,
        "target_qps": args.target_qps,
        "root": root_started,
        "third": third_started,
        "third_training_load": third_training_started,
        "client8g": client_started,
        "probe": probe_started,
    }, ensure_ascii=True, indent=2))


def status(args):
    hosts = connect_hosts(args, root_required=False)
    run_id = args.validation_run_id
    linux = rf"""
printf 'server_processes=%s\n' "$(pgrep -af '{run_id}.*run_server_linux|benchmark_headonly_concurrent_availability.py server' | wc -l)"
ss -ltn | grep -E ':(6201[0-4])[[:space:]]' || true
find {ROOT_REPO}/exp/concurrent_availability/{run_id} -maxdepth 3 -type f \
  -printf '%p\t%s bytes\n' 2>/dev/null | sort
echo '-- root stdout tail --'
tail -n 40 {ROOT_REPO}/exp/concurrent_availability/{run_id}/root4090/task.stdout.log 2>/dev/null || true
echo '-- root stderr tail --'
tail -n 80 {ROOT_REPO}/exp/concurrent_availability/{run_id}/root4090/task.stderr.log 2>/dev/null || true
"""
    windows = rf"""
$procs = @(Get-CimInstance Win32_Process | Where-Object {{
  $_.CommandLine -like '*{run_id}*' -or
  $_.CommandLine -like '*benchmark_headonly_concurrent_availability.py*'
}})
"processes=$($procs.Count)"
Get-NetTCPConnection -ErrorAction SilentlyContinue |
  Where-Object {{
    ($_.LocalPort -ge 62010 -and $_.LocalPort -le 62039) -or
    ($_.RemotePort -ge 62010 -and $_.RemotePort -le 62039)
  }} |
  Group-Object State | ForEach-Object {{ "tcp_$($_.Name)=$($_.Count)" }}
Get-ScheduledTask -TaskName 'GGEUR-availability-{run_id}-*' `
  -ErrorAction SilentlyContinue |
  ForEach-Object {{
    $info = Get-ScheduledTaskInfo -TaskName $_.TaskName
    "task=$($_.TaskName) state=$($_.State) result=$($info.LastTaskResult)"
  }}
$runRoot = if ($env:COMPUTERNAME -eq 'LKX') {{
  '{CLIENT_REPO}\exp\concurrent_availability\{run_id}'
}} else {{
  '{THIRD_REPO}\exp\concurrent_availability\{run_id}'
}}
Get-ChildItem -LiteralPath $runRoot -Recurse -File -ErrorAction SilentlyContinue |
  Where-Object {{ $_.Name -match '(task|server|load|probe)\.(stdout|stderr)\.log$' }} |
  Sort-Object FullName |
  ForEach-Object {{
    "log=$($_.FullName) bytes=$($_.Length)"
    Get-Content -LiteralPath $_.FullName -Tail 12 -ErrorAction SilentlyContinue
  }}
"""
    result = {
        "root4090": (
            run(hosts["root4090"], linux, check=False)
            if hosts["root4090"] is not None else {
                "status": -1,
                "stdout": "",
                "stderr": hosts["root_error"],
            }
        ),
        "third": run(
            hosts["third"], encoded_powershell(windows), check=False),
        "client8g": run(
            hosts["client8g"], encoded_powershell(windows), check=False),
    }
    close_hosts(hosts)
    print(json.dumps(result, ensure_ascii=True, indent=2))


def _metric_value(output, name):
    prefix = name + "="
    for line in output.splitlines():
        if line.startswith(prefix):
            return int(line[len(prefix):].strip())
    raise RuntimeError(
        "missing network metric {} in output:\n{}".format(name, output))


def network_monitor(args):
    """Observe the real T1 TCP connections from the SSH controller."""
    hosts = connect_hosts(args, root_required=True)
    run_id = args.validation_run_id
    project_dir = Path(__file__).resolve().parents[3]
    output_dir = (
        project_dir
        / "docs" / "test_logs" / "concurrent_availability" / run_id
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = output_dir / "network_connection_monitor.tsv"
    summary_path = output_dir / "network_connection_monitor.json"
    client_command = encoded_powershell(r"""
$items = @(Get-NetTCPConnection -State Established -ErrorAction SilentlyContinue |
  Where-Object {
    $_.RemotePort -ge 62020 -and $_.RemotePort -le 62029
  })
"NETWORK_TOOL=Get-NetTCPConnection"
"LOAD_CLIENT_ESTABLISHED=$($items.Count)"
""".strip())
    root_command = r"""
count="$(ss -Htan state established |
  grep -Ec ':(6201[0-4])[[:space:]]' || true)"
printf 'NETWORK_TOOL=ss\nROOT_SERVER_ESTABLISHED=%s\n' "$count"
""".strip()
    third_command = encoded_powershell(r"""
$items = @(Get-NetTCPConnection -State Established -ErrorAction SilentlyContinue |
  Where-Object {
    $_.LocalPort -ge 62020 -and $_.LocalPort -le 62029
  })
"NETWORK_TOOL=Get-NetTCPConnection"
"THIRD_SERVER_ESTABLISHED=$($items.Count)"
""".strip())
    deadline = time.time() + args.monitor_timeout_sec
    samples = []
    reached = False
    consecutive_target_samples = 0
    try:
        while True:
            sampled_at = time.time()
            client_result = run(
                hosts["client8g"], client_command, timeout=60, check=False)
            root_result = run(
                hosts["root4090"], root_command, timeout=60, check=False)
            third_result = run(
                hosts["third"], third_command, timeout=60, check=False)
            if any(item["status"] for item in (
                    client_result, root_result, third_result)):
                raise RuntimeError(
                    "network observation command failed:\n{}".format(
                        json.dumps({
                            "client8g": client_result,
                            "root4090": root_result,
                            "third": third_result,
                        }, ensure_ascii=False, indent=2)))
            load_count = _metric_value(
                client_result["stdout"], "LOAD_CLIENT_ESTABLISHED")
            root_count = _metric_value(
                root_result["stdout"], "ROOT_SERVER_ESTABLISHED")
            third_count = _metric_value(
                third_result["stdout"], "THIRD_SERVER_ESTABLISHED")
            sample = {
                "sampled_at_unix": sampled_at,
                "load_client_established": load_count,
                "root_server_established": root_count,
                "third_server_established": third_count,
            }
            samples.append(sample)
            print(
                "网络连接观测：客户端机已建立连接数={}，"
                "根服务器机={}，子服务器机={}，目标={}".format(
                    load_count, root_count, third_count,
                    args.target_connections),
                flush=True,
            )
            if (load_count >= args.target_connections and
                    root_count + third_count >= args.target_connections):
                consecutive_target_samples += 1
            else:
                consecutive_target_samples = 0
            if consecutive_target_samples >= args.monitor_confirm_samples:
                reached = True
                break
            if time.time() >= deadline:
                break
            time.sleep(args.monitor_interval_sec)
    finally:
        close_hosts(hosts)
    with samples_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(
            "sampled_at_unix\tload_client_established\t"
            "root_server_established\tthird_server_established\n")
        for sample in samples:
            handle.write(
                "{sampled_at_unix:.6f}\t{load_client_established}\t"
                "{root_server_established}\t{third_server_established}\n".
                format(**sample))
    summary = {
        "run_id": run_id,
        "network_observation_tool": "Get-NetTCPConnection/ss",
        "target_connections": args.target_connections,
        "target_reached": reached,
        "required_consecutive_samples": args.monitor_confirm_samples,
        "max_load_client_established": max(
            (item["load_client_established"] for item in samples),
            default=0),
        "samples_file": samples_path.relative_to(project_dir).as_posix(),
        "samples": samples,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8")
    if not reached:
        raise RuntimeError(
            "network monitor observed at most {} established connections; "
            "target is {}".format(
                summary["max_load_client_established"],
                args.target_connections))
    print("万级客户端连接观测成功：已建立连接数={}。".format(
        summary["max_load_client_established"]))
    print("NETWORK_MONITOR=PASS")
    print("NETWORK_TOOL=Get-NetTCPConnection/ss")
    print("ESTABLISHED_CONNECTIONS={}".format(
        summary["max_load_client_established"]))
    print("NETWORK_MONITOR_SUMMARY={}".format(
        summary_path.relative_to(project_dir).as_posix()))
    print("NETWORK_MONITOR_SAMPLES={}".format(
        samples_path.relative_to(project_dir).as_posix()))


def third_party_monitor(args):
    """Observe T1 connections with an independent Sysinternals tool."""
    hosts = connect_hosts(args, root_required=False)
    run_id = args.validation_run_id
    project_dir = Path(__file__).resolve().parents[3]
    output_dir = (
        project_dir / "docs" / "test_logs" /
        "concurrent_availability" / run_id
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    remote_script = (
        f"{CLIENT_REPO}/{REL_DIR}/observe_connections_tcpvcon.ps1"
    ).replace("/", "\\")
    remote_summary = (
        f"{CLIENT_REPO}/exp/concurrent_availability/{run_id}/"
        "network_observation/tcpvcon_connection_summary.json"
    ).replace("/", "\\")
    remote_csv = (
        f"{CLIENT_REPO}/exp/concurrent_availability/{run_id}/"
        "network_observation/tcpvcon_connections.csv"
    ).replace("/", "\\")
    command = encoded_powershell(rf"""
& '{remote_script}' -RunId '{run_id}' `
  -TargetConnections {int(args.target_connections)}
""".strip())
    try:
        deadline = time.time() + args.monitor_timeout_sec
        result = None
        while True:
            result = run(
                hosts["client8g"], command, timeout=300, check=False)
            if result["status"] == 0:
                break
            if time.time() >= deadline:
                raise RuntimeError(
                    "Tcpvcon did not observe the target within {} seconds:\n{}"
                    .format(
                        args.monitor_timeout_sec,
                        json.dumps(result, ensure_ascii=False, indent=2)))
            time.sleep(args.monitor_interval_sec)
        download_file(
            hosts["client8g"], remote_summary,
            output_dir / "tcpvcon_connection_summary.json")
        download_file(
            hosts["client8g"], remote_csv,
            output_dir / "tcpvcon_connections.csv")
    finally:
        close_hosts(hosts)
    summary_path = output_dir / "tcpvcon_connection_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    observed = int(summary["observed_established_connections"])
    if not bool(summary["target_reached"]):
        raise RuntimeError(
            "Tcpvcon observed {} connections; target is {}".format(
                observed, args.target_connections))
    print("第三方网络观测成功：Tcpvcon显示已建立连接数={}。".format(observed))
    print("THIRD_PARTY_NETWORK_OBSERVATION=PASS")
    print("THIRD_PARTY_NETWORK_TOOL=Microsoft Sysinternals Tcpvcon")
    print("OBSERVED_ESTABLISHED_CONNECTIONS={}".format(observed))
    print("TCPVCON_CONNECTION_CSV={}".format(
        (output_dir / "tcpvcon_connections.csv").relative_to(
            project_dir).as_posix()))
    print("TCPVCON_SUMMARY={}".format(
        summary_path.relative_to(project_dir).as_posix()))


def recover_failed_run(args):
    """Stop the failed availability attempt and preserve all raw evidence."""
    hosts = connect_hosts(args, root_required=True)
    run_id = args.validation_run_id
    reason = "".join(
        char if char.isalnum() else "_" for char in args.recovery_reason
    ).strip("_") or "recovery"
    stamp = time.strftime("attempt_%Y%m%d_%H%M%S_") + reason
    linux = rf"""
set -euo pipefail
pkill -f '[b]enchmark_headonly_concurrent_availability.py server' || true
pkill -f '[r]un_server_linux.sh' || true
sleep 2
run_root={ROOT_REPO}/exp/concurrent_availability/{run_id}
attempt_root="$run_root/attempts/{stamp}"
mkdir -p "$attempt_root"
if [[ -d "$run_root/root4090" ]]; then
  mv "$run_root/root4090" "$attempt_root/root4090"
fi
printf 'attempt=%s\n' "$attempt_root/root4090"
printf 'remaining_roles=%s\n' \
  "$(pgrep -af '[b]enchmark_headonly_concurrent_availability' | wc -l)"
"""
    windows = rf"""
$ErrorActionPreference = 'Stop'
$runId = '{run_id}'
$taskPattern = "GGEUR-availability-$runId-*"
Get-ScheduledTask -TaskName $taskPattern -ErrorAction SilentlyContinue |
  ForEach-Object {{
    Stop-ScheduledTask -TaskName $_.TaskName -ErrorAction SilentlyContinue
  }}
Start-Sleep -Seconds 3
$roles = @(Get-CimInstance Win32_Process | Where-Object {{
  $_.Name -match '^(python|pythonw|cmd|powershell)\.exe$' -and
  ($_.CommandLine -like "*$runId*" -or
   $_.CommandLine -like '*benchmark_headonly_concurrent_availability.py*')
}})
$roles | ForEach-Object {{
  Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
}}
Start-Sleep -Seconds 2
$repo = if ($env:COMPUTERNAME -eq 'LKX') {{
  '{CLIENT_REPO}'
}} else {{
  '{THIRD_REPO}'
}}
$runRoot = Join-Path $repo "exp\concurrent_availability\$runId"
$attemptRoot = Join-Path $runRoot "attempts\{stamp}"
New-Item -ItemType Directory -Force -Path $attemptRoot | Out-Null
$roleDirs = if ($env:COMPUTERNAME -eq 'LKX') {{
  @('client8g', 'client_probe')
}} else {{
  @('third', 'third_root_registration', 'third_root_registration_task',
    'third_training_clients', 'third_training_task')
}}
foreach ($role in $roleDirs) {{
  $source = Join-Path $runRoot $role
  if (Test-Path -LiteralPath $source) {{
    Move-Item -LiteralPath $source -Destination (Join-Path $attemptRoot $role)
  }}
}}
$remaining = @(Get-CimInstance Win32_Process | Where-Object {{
  $_.Name -match '^python(w)?\.exe$' -and
  $_.CommandLine -like '*benchmark_headonly_concurrent_availability.py*'
}})
"attempt=$attemptRoot"
"remaining_roles=$($remaining.Count)"
"""
    try:
        result = {
            "attempt": stamp,
            "root4090": run(
                hosts["root4090"], linux, timeout=60, check=False),
            "third": run(
                hosts["third"], encoded_powershell(windows),
                timeout=90, check=False),
            "client8g": run(
                hosts["client8g"], encoded_powershell(windows),
                timeout=90, check=False),
        }
    finally:
        close_hosts(hosts)
    print(json.dumps(result, ensure_ascii=True, indent=2))


def open_third_business_ports(args):
    """Allow only the 8G load generator into the third host test ports."""
    hosts = connect_hosts(args, root_required=False)
    script = r"""
$ErrorActionPreference = 'Stop'
$name = 'GGEUR-Concurrent-Availability-62020-62029'
$existing = Get-NetFirewallRule -DisplayName $name -ErrorAction SilentlyContinue
if (-not $existing) {
  New-NetFirewallRule `
    -DisplayName $name `
    -Direction Inbound `
    -Action Allow `
    -Protocol TCP `
    -LocalPort 62020-62029 `
    -RemoteAddress 10.129.222.189 `
    -Profile Any | Out-Null
} else {
  Enable-NetFirewallRule -DisplayName $name | Out-Null
}
$rule = Get-NetFirewallRule -DisplayName $name
$port = $rule | Get-NetFirewallPortFilter
$address = $rule | Get-NetFirewallAddressFilter
"rule=$($rule.DisplayName)"
"enabled=$($rule.Enabled)"
"action=$($rule.Action)"
"local_port=$($port.LocalPort -join ',')"
"remote_address=$($address.RemoteAddress -join ',')"
"""
    try:
        result = run(
            hosts["third"], encoded_powershell(script),
            timeout=60, check=False)
    finally:
        close_hosts(hosts)
    print(json.dumps(result, ensure_ascii=True, indent=2))
    if result["status"]:
        raise SystemExit(result["status"])


def collect(args):
    hosts = connect_hosts(args, root_required=True)
    run_id = args.validation_run_id
    output = Path(args.output_dir).resolve() / run_id
    root_remote = (
        f"{ROOT_REPO}/exp/concurrent_availability/{run_id}/"
        "root4090/server_summary.json")
    third_remote = (
        f"{THIRD_REPO}/exp/concurrent_availability/{run_id}/"
        "third/server_summary.json").replace("\\", "/")
    probe_remote = (
        f"{CLIENT_REPO}/exp/concurrent_availability/{run_id}/"
        "client_probe/probe_summary.json").replace("\\", "/")
    root_registration_remote = (
        f"{THIRD_REPO}/exp/concurrent_availability/{run_id}/"
        "third_root_registration/root_registration_load_summary.json"
    ).replace("\\", "/")
    client_child_remote = (
        f"{CLIENT_REPO}/exp/concurrent_availability/{run_id}/"
        "client8g/child_servers_load_summary.json").replace("\\", "/")
    third_training_remote = (
        f"{THIRD_REPO}/exp/concurrent_availability/{run_id}/"
        "third_training_clients/"
        "third-local-train_load_summary.json").replace("\\", "/")
    root_run_remote = (
        f"{ROOT_REPO}/exp/concurrent_availability/{run_id}")
    third_run_remote = (
        f"{THIRD_REPO}/exp/concurrent_availability/{run_id}"
    ).replace("\\", "/")
    client_run_remote = (
        f"{CLIENT_REPO}/exp/concurrent_availability/{run_id}"
    ).replace("\\", "/")
    missing_required = []
    try:
        if not download_optional_file(
            hosts["root4090"], root_remote,
                output / "root4090_server_summary.json"):
            missing_required.append("root4090_server_summary.json")
        if not download_optional_file(
            hosts["third"], third_remote,
                output / "third_server_summary.json"):
            missing_required.append("third_server_summary.json")
        if not download_optional_file(
            hosts["client8g"], probe_remote,
                output / "client_probe_summary.json"):
            missing_required.append("client_probe_summary.json")
        if not download_optional_file(
            hosts["third"], root_registration_remote,
                output / "third_root_registration_summary.json"):
            missing_required.append("third_root_registration_summary.json")
        if not download_optional_file(
            hosts["client8g"], client_child_remote,
                output / "client8g_child_load_summary.json"):
            missing_required.append("client8g_child_load_summary.json")
        download_optional_file(
            hosts["third"], third_training_remote,
            output / "third_training_load_summary.json")
        download_tree(
            hosts["root4090"], root_run_remote,
            output / "raw" / "root4090")
        download_tree(
            hosts["third"], third_run_remote,
            output / "raw" / "third")
        download_tree(
            hosts["client8g"], client_run_remote,
            output / "raw" / "client8g")
    finally:
        close_hosts(hosts)
    print(json.dumps({
        "run_id": run_id,
        "output": str(output),
        "missing_required": missing_required,
    }, ensure_ascii=False, indent=2))
    if missing_required:
        raise RuntimeError(
            "required evidence files are missing after available files were "
            "archived: " + ", ".join(missing_required))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=(
            "root-check",
            "clean-root-memory",
            "deploy",
            "preflight",
            "suspend-accuracy-queue",
            "resume-accuracy-queue",
            "pause-accuracy",
            "start",
            "status",
            "network-monitor",
            "third-party-monitor",
            "recover-failed-run",
            "open-third-business-ports",
            "collect",
        ),
    )
    parser.add_argument("--payload-file", required=True)
    parser.add_argument(
        "--key", default=os.path.expanduser(r"~\.ssh\id_ed25519"))
    parser.add_argument("--bore-host", default="bore.pub")
    parser.add_argument(
        "--bore-port", type=int,
        default=int(os.environ.get("GGEUR_BORE_PORT", "0")))
    parser.add_argument("--socks-host", default="127.0.0.1")
    parser.add_argument(
        "--socks-port", type=int,
        default=int(os.environ.get("GGEUR_SOCKS_PORT", "7897")))
    parser.add_argument(
        "--third-password",
        default=os.environ.get("GGEUR_THIRD_PASSWORD"),
    )
    parser.add_argument(
        "--accuracy-run-id", default="final_remaining_20260723_v1")
    parser.add_argument(
        "--accuracy-case", default="mdsent_lstm_fedavg")
    parser.add_argument("--root-recovery-attempts", type=int, default=5)
    parser.add_argument(
        "--root-recovery-interval-sec", type=int, default=8)
    parser.add_argument("--root-banner-timeout", type=int, default=60)
    parser.add_argument(
        "--validation-run-id",
        default="concurrent_availability_20260730_v1",
    )
    parser.add_argument("--start-delay-sec", type=int, default=180)
    parser.add_argument(
        "--start-phase",
        choices=("all", "root", "third", "clients"),
        default="all",
    )
    parser.add_argument("--probe-delay-sec", type=int, default=2)
    parser.add_argument("--active-after-probe-sec", type=int, default=10)
    parser.add_argument("--target-connections", type=int, default=10000)
    parser.add_argument("--target-qps", type=float, default=10000.0)
    parser.add_argument("--request-duration-sec", type=float, default=600.0)
    parser.add_argument("--poll-interval-sec", type=float, default=0.8)
    parser.add_argument("--poll-worker-connections", type=int, default=200)
    parser.add_argument("--request-start-delay-sec", type=float, default=2.0)
    parser.add_argument("--monitor-timeout-sec", type=int, default=300)
    parser.add_argument("--monitor-interval-sec", type=float, default=2.0)
    parser.add_argument("--monitor-confirm-samples", type=int, default=3)
    parser.add_argument("--root-subservers", type=int, default=1)
    parser.add_argument(
        "--root-clients-per-subserver", type=int, default=10,
        help=("Root-side registration connections.  The formal T1 run uses "
              "one root service and one registration per child server."),
    )
    parser.add_argument("--third-subservers", type=int, default=10)
    parser.add_argument("--third-clients-per-subserver", type=int, default=1000)
    parser.add_argument(
        "--third-training-clients-per-subserver",
        type=int,
        default=0,
        help=(
            "Clients per third-host subserver that run on the third host and "
            "perform a real local MLP SGD update; they are included in the "
            "third-clients-per-subserver total."
        ),
    )
    parser.add_argument("--training-steps", type=int, default=1)
    parser.add_argument("--training-batch-size", type=int, default=8)
    parser.add_argument("--training-learning-rate", type=float, default=0.01)
    parser.add_argument("--training-concurrency", type=int, default=4)
    parser.add_argument(
        "--recovery-reason", default="asyncio_loop_recovery")
    parser.add_argument(
        "--output-dir",
        default="docs/test_logs/concurrent_availability",
    )
    args = parser.parse_args()
    actions = {
        "root-check": root_check,
        "clean-root-memory": clean_root_memory,
        "deploy": deploy,
        "preflight": preflight,
        "suspend-accuracy-queue": suspend_queue,
        "resume-accuracy-queue": resume_queue,
        "pause-accuracy": pause_accuracy,
        "start": start_run,
        "status": status,
        "network-monitor": network_monitor,
        "third-party-monitor": third_party_monitor,
        "recover-failed-run": recover_failed_run,
        "open-third-business-ports": open_third_business_ports,
        "collect": collect,
    }
    actions[args.action](args)


if __name__ == "__main__":
    main()
