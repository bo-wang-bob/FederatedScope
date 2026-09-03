#!/usr/bin/env python3
"""Stage the user's canonical four-domain DomainNet ZIPs on the dual hosts.

The local machine connects only to the 8G management host.  Access to the
third host is opened by ``manage_dual_machine.connect_dual`` through an SSH
``direct-tcpip`` channel on the 8G host.  Uploads are streamed and resumable;
existing data and caches are never overwritten.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from urllib.parse import quote
import zipfile

from manage_dual_machine import (
    CLIENT_REPO,
    THIRD_REPO,
    close_dual,
    connect_dual,
    encoded_powershell,
    run,
)


DOMAINS = ("clipart", "painting", "real", "sketch")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
STAGE_NAME = "domainnet_original_4domains_20260804"
CHUNK_BYTES = 4 * 1024 * 1024
RELAY_CHUNK_BYTES = 512 * 1024
RELAY_FLUSH_BYTES = 16 * 1024 * 1024
PROGRESS_BYTES = 256 * 1024 * 1024
# Reuse the already-approved third-host DomainNet subserver port while the
# experiment queue is stopped.  The temporary server is closed before smoke.
HTTP_CACHE_PORT = 62001


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("audit", "status", "transfer", "extract",
                            "start-cache", "cache-status", "sync-cache",
                            "sync-cache-http",
                            "sync-cache-local-relay",
                            "bind-client8g-cache-markers",
                            "resume-client8g-driver",
                            "sync-cache-status", "sync-live-status"))
    parser.add_argument(
        "--source-dir", default=r"C:\Users\Dbook\Downloads")
    parser.add_argument(
        "--key", default=os.path.expanduser(r"~\.ssh\id_ed25519"))
    parser.add_argument(
        "--third-password", default=os.environ.get("GGEUR_THIRD_PASSWORD"))
    parser.add_argument(
        "--hosts", nargs="*", choices=("client8g", "third"),
        default=["client8g", "third"])
    parser.add_argument("--skip-crc", action="store_true")
    parser.add_argument("--run-id")
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_archives(source_dir, verify_crc):
    source_dir = Path(source_dir)
    result = {}
    shared_classes = None
    for domain in DOMAINS:
        path = source_dir / f"{domain}.zip"
        if not path.is_file():
            raise FileNotFoundError(path)
        with zipfile.ZipFile(path) as archive:
            files = [item for item in archive.infolist()
                     if not item.is_dir() and
                     Path(item.filename).suffix.lower() in IMAGE_SUFFIXES]
            unsafe = [item.filename for item in archive.infolist()
                      if Path(item.filename).is_absolute() or
                      ".." in Path(item.filename).parts]
            if unsafe:
                raise RuntimeError(
                    f"unsafe paths in {path}: {unsafe[:5]}")
            roots = {Path(item.filename).parts[0] for item in files}
            if roots != {domain}:
                raise RuntimeError(
                    f"unexpected roots in {path}: {sorted(roots)}")
            classes = sorted({Path(item.filename).parts[1] for item in files
                              if len(Path(item.filename).parts) >= 3})
            if len(classes) != 345:
                raise RuntimeError(
                    f"expected 345 classes in {path}, got {len(classes)}")
            if shared_classes is None:
                shared_classes = classes
            elif classes != shared_classes:
                raise RuntimeError(
                    f"class list differs for {domain}")
            bad_member = archive.testzip() if verify_crc else None
            if bad_member:
                raise RuntimeError(
                    f"CRC failure in {path}: {bad_member}")
            result[domain] = {
                "path": str(path),
                "archive_bytes": path.stat().st_size,
                "archive_sha256": sha256_file(path),
                "image_count": len(files),
                "uncompressed_image_bytes": sum(item.file_size for item in files),
                "class_count": len(classes),
                "first": files[0].filename if files else "",
                "last": files[-1].filename if files else "",
                "crc_verified": bool(verify_crc),
            }
    result["summary"] = {
        "domains": list(DOMAINS),
        "class_count": len(shared_classes or []),
        "record_count": sum(result[item]["image_count"] for item in DOMAINS),
        "archive_bytes": sum(result[item]["archive_bytes"] for item in DOMAINS),
        "uncompressed_image_bytes": sum(
            result[item]["uncompressed_image_bytes"] for item in DOMAINS),
    }
    return result


def stage_root(host_name):
    repo = CLIENT_REPO if host_name == "client8g" else THIRD_REPO
    return f"{repo}/data_staging/{STAGE_NAME}".replace("\\", "/")


def ps_literal(value):
    return str(value).replace("'", "''")


def remote_prepare(client, host_name, required_bytes):
    root = stage_root(host_name)
    script = rf"""
$ErrorActionPreference = 'Stop'
$root = '{ps_literal(root)}'
$archives = Join-Path $root 'archives'
$dataset = Join-Path $root 'dataset'
$driveName = ([IO.Path]::GetPathRoot($root) -replace '[:\\]+$','')
$drive = Get-PSDrive -Name $driveName
[IO.Directory]::CreateDirectory($archives) | Out-Null
"computer=$env:COMPUTERNAME"
"stage_root=$root"
"archive_dir=$archives"
"dataset_dir=$dataset"
"drive_free_bytes=$($drive.Free)"
"required_bytes={required_bytes}"
if ($drive.Free -lt {required_bytes}) {{
  throw "insufficient free space: free=$($drive.Free) required={required_bytes}"
}}
"""
    return run(client, encoded_powershell(script), timeout=120)


def remote_hash(client, path, timeout=3600):
    script = rf"""
$ErrorActionPreference = 'Stop'
$path = '{ps_literal(path)}'
if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {{
  throw "missing file: $path"
}}
$item = Get-Item -LiteralPath $path
$hash = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
"bytes=$($item.Length)"
"sha256=$hash"
"""
    result = run(client, encoded_powershell(script), timeout=timeout)
    parsed = {}
    for line in result["stdout"].splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            parsed[key] = value
    return int(parsed["bytes"]), parsed["sha256"]


def preserve_remote(client, path, reason):
    stamp = time.strftime("%Y%m%d_%H%M%S")
    destination = f"{path}.invalid_{stamp}"
    script = rf"""
$ErrorActionPreference = 'Stop'
$path = '{ps_literal(path)}'
$destination = '{ps_literal(destination)}'
if (Test-Path -LiteralPath $path) {{
  Move-Item -LiteralPath $path -Destination $destination
  "preserved=$destination"
}}
"reason={ps_literal(reason)}"
"""
    result = run(client, encoded_powershell(script), timeout=120)
    print(json.dumps({"event": "preserve_remote", "path": path,
                      "reason": reason, "result": result},
                     ensure_ascii=True), flush=True)


def finalize_remote(client, partial, final):
    script = rf"""
$ErrorActionPreference = 'Stop'
$partial = '{ps_literal(partial)}'
$final = '{ps_literal(final)}'
if (Test-Path -LiteralPath $final) {{
  throw "refusing to overwrite final archive: $final"
}}
Move-Item -LiteralPath $partial -Destination $final
"final=$final"
"""
    return run(client, encoded_powershell(script), timeout=120)


def upload_resumable(client, host_name, source, expected_sha256):
    source = Path(source)
    archive_dir = f"{stage_root(host_name)}/archives"
    final = f"{archive_dir}/{source.name}"
    partial = final + ".part"
    expected_size = source.stat().st_size

    sftp = client.open_sftp()
    try:
        try:
            final_size = sftp.stat(final).st_size
        except OSError:
            final_size = None
    finally:
        sftp.close()
    if final_size is not None:
        if final_size == expected_size:
            size, digest = remote_hash(client, final)
            if size == expected_size and digest == expected_sha256:
                print(json.dumps({"event": "skip_verified", "host": host_name,
                                  "file": source.name, "bytes": size,
                                  "sha256": digest}), flush=True)
                return
        preserve_remote(client, final, "final archive size/hash mismatch")

    sftp = client.open_sftp()
    try:
        try:
            offset = sftp.stat(partial).st_size
        except OSError:
            offset = 0
        if offset > expected_size:
            sftp.close()
            preserve_remote(client, partial, "partial larger than source")
            sftp = client.open_sftp()
            offset = 0
        if offset == expected_size:
            sftp.close()
            size, digest = remote_hash(client, partial)
            if size != expected_size or digest != expected_sha256:
                preserve_remote(client, partial, "complete partial hash mismatch")
                sftp = client.open_sftp()
                offset = 0
            else:
                finalize_remote(client, partial, final)
                print(json.dumps({"event": "finalized_existing_partial",
                                  "host": host_name, "file": source.name,
                                  "bytes": size, "sha256": digest}), flush=True)
                return

        started = time.time()
        next_report = offset + PROGRESS_BYTES
        with source.open("rb") as local_stream:
            local_stream.seek(offset)
            remote_stream = sftp.open(partial, "ab")
            try:
                remote_stream.set_pipelined(True)
                while True:
                    chunk = local_stream.read(CHUNK_BYTES)
                    if not chunk:
                        break
                    remote_stream.write(chunk)
                    offset += len(chunk)
                    if offset >= next_report:
                        elapsed = max(time.time() - started, 0.001)
                        print(json.dumps({
                            "event": "upload_progress", "host": host_name,
                            "file": source.name, "bytes": offset,
                            "total": expected_size,
                            "percent": round(offset * 100 / expected_size, 2),
                            "session_mib_s": round(
                                (offset - (next_report - PROGRESS_BYTES)) /
                                elapsed / (1024 * 1024), 2),
                        }), flush=True)
                        next_report = offset + PROGRESS_BYTES
            finally:
                remote_stream.close()
    finally:
        try:
            sftp.close()
        except Exception:
            pass

    size, digest = remote_hash(client, partial)
    if size != expected_size or digest != expected_sha256:
        raise RuntimeError(
            f"remote verification failed for {host_name}:{partial}: "
            f"size={size} sha256={digest}")
    finalize_remote(client, partial, final)
    print(json.dumps({"event": "upload_complete", "host": host_name,
                      "file": source.name, "bytes": size,
                      "sha256": digest}), flush=True)


def remote_status(client, host_name, audit):
    root = stage_root(host_name)
    result = {"host": host_name, "stage_root": root, "archives": {}}
    sftp = client.open_sftp()
    try:
        for domain in DOMAINS:
            expected = audit[domain]["archive_bytes"]
            final = f"{root}/archives/{domain}.zip"
            partial = final + ".part"
            entry = {"expected_bytes": expected}
            for key, path in (("final_bytes", final),
                              ("partial_bytes", partial)):
                try:
                    entry[key] = sftp.stat(path).st_size
                except OSError:
                    entry[key] = None
            result["archives"][domain] = entry
    finally:
        sftp.close()
    return result


def extract_remote(client, host_name, audit):
    root = stage_root(host_name)
    dataset = f"{root}/dataset"
    expected = {domain: audit[domain]["image_count"] for domain in DOMAINS}
    expected_json = json.dumps(expected, separators=(",", ":"))
    repo = CLIENT_REPO if host_name == "client8g" else THIRD_REPO
    python_candidates = (
        [f"{repo}/.venv_client_cpu/Scripts/python.exe",
         r"D:/ProgramData/anaconda3/envs/pi_fmd_gpu_py39/python.exe"]
        if host_name == "client8g" else
        [f"{repo}/.venv_client_cpu/Scripts/python.exe",
         r"C:/Users/pc/miniconda3/envs/cerp/python.exe"])
    candidates_ps = ",".join(
        f"'{ps_literal(item)}'" for item in python_candidates)
    generator = (
        f"{repo}/scripts/distributed_scripts/ggeur_hierarchical_3machine/"
        "generate_domainnet_manifest.py").replace("\\", "/")
    script = rf"""
$ErrorActionPreference = 'Stop'
$root = '{ps_literal(root)}'
$archives = Join-Path $root 'archives'
$dataset = '{ps_literal(dataset)}'
[IO.Directory]::CreateDirectory($dataset) | Out-Null
foreach ($domain in @('clipart','painting','real','sketch')) {{
  $archive = Join-Path $archives ($domain + '.zip')
  if (-not (Test-Path -LiteralPath $archive -PathType Leaf)) {{
    throw "missing archive: $archive"
  }}
  "extract_start=$domain archive=$archive"
  & tar.exe -xf $archive -C $dataset
  if ($LASTEXITCODE -ne 0) {{ throw "tar failed for $($archive): $LASTEXITCODE" }}
  "extract_done=$domain"
}}
$expected = '{ps_literal(expected_json)}' | ConvertFrom-Json
$validation = [ordered]@{{
  stage = $root
  dataset = $dataset
  validated_at = (Get-Date).ToString('o')
  domains = [ordered]@{{}}
}}
foreach ($domain in @('clipart','painting','real','sketch')) {{
  $domainPath = Join-Path $dataset $domain
  $files = @(Get-ChildItem -LiteralPath $domainPath -Recurse -File)
  $classes = @(Get-ChildItem -LiteralPath $domainPath -Directory)
  $expectedCount = [int]$expected.$domain
  if ($classes.Count -ne 345 -or $files.Count -ne $expectedCount) {{
    throw "validation failed for $($domain): classes=$($classes.Count) files=$($files.Count) expected=$expectedCount"
  }}
  $validation.domains[$domain] = [ordered]@{{
    classes = $classes.Count
    files = $files.Count
    bytes = ($files | Measure-Object Length -Sum).Sum
  }}
}}
$validation | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $root 'extraction_validation.json') -Encoding UTF8
$python = @({candidates_ps}) | Where-Object {{ Test-Path -LiteralPath $_ -PathType Leaf }} | Select-Object -First 1
if (-not $python) {{ throw 'no Python interpreter found for manifest generation' }}
$generator = '{ps_literal(generator)}'
$manifest = Join-Path $root 'domainnet_manifest.json'
& $python $generator --root $dataset --output $manifest --domains 'clipart,painting,real,sketch'
if ($LASTEXITCODE -ne 0) {{ throw "manifest generation failed: $LASTEXITCODE" }}
$obj = Get-Content -LiteralPath $manifest -Raw | ConvertFrom-Json
if ($obj.record_count -ne 370305 -or $obj.classes.Count -ne 345) {{
  throw "manifest validation failed: records=$($obj.record_count) classes=$($obj.classes.Count)"
}}
"manifest=$manifest"
"record_count=$($obj.record_count)"
"records_sha256=$($obj.records_sha256)"
"""
    return run(client, encoded_powershell(script), timeout=7200)


def start_cache_chain(hosts):
    uploads = (
        "prepare_domainnet_reference_samples.py",
        "prepare_domainnet_train_cache.py",
        "prepare_domainnet_eval_cache.py",
        "run_domainnet_original_cache_chain.ps1",
    )
    remote_dir = (
        f"{THIRD_REPO}/scripts/distributed_scripts/"
        "ggeur_hierarchical_3machine").replace("\\", "/")
    prepare = rf"""
$ErrorActionPreference = 'Stop'
[IO.Directory]::CreateDirectory('{ps_literal(remote_dir)}') | Out-Null
"remote_dir={ps_literal(remote_dir)}"
"""
    run(hosts["third"], encoded_powershell(prepare), timeout=120)
    sftp = hosts["third"].open_sftp()
    try:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        for name in uploads:
            local = Path(__file__).resolve().parent / name
            remote = f"{remote_dir}/{name}"
            temporary = remote + ".uploading"
            sftp.put(str(local), temporary)
            try:
                sftp.stat(remote)
            except OSError:
                pass
            else:
                sftp.rename(remote, f"{remote}.previous_{stamp}")
            sftp.rename(temporary, remote)
    finally:
        sftp.close()

    task = "GGEUR-domainnet-original-cache-chain-third"
    runner = f"{remote_dir}/run_domainnet_original_cache_chain.ps1"
    state = f"{THIRD_REPO}/exp/domainnet_original_cache_20260804".replace(
        "\\", "/")
    script = rf"""
$ErrorActionPreference = 'Stop'
$taskName = '{task}'
$runner = '{ps_literal(runner)}'
$state = '{ps_literal(state)}'
[IO.Directory]::CreateDirectory($state) | Out-Null
$existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existing -and $existing.State -eq 'Running') {{
  "already_running=$taskName"
  exit 0
}}
if ($existing) {{
  $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
  $attempt = Join-Path $state ("attempt_" + $stamp)
  [IO.Directory]::CreateDirectory($attempt) | Out-Null
  Get-ChildItem -LiteralPath $state -File -ErrorAction SilentlyContinue |
    Where-Object {{ $_.Name -match '\.(log|json)$' }} |
    Copy-Item -Destination $attempt
  Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
}}
$powerShell = "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe"
$arguments = '-NoProfile -NonInteractive -ExecutionPolicy Bypass -File "' +
  $runner + '"'
$action = New-ScheduledTaskAction -Execute $powerShell -Argument $arguments `
  -WorkingDirectory '{ps_literal(THIRD_REPO)}'
$identity = [Security.Principal.WindowsIdentity]::GetCurrent().Name
$principal = New-ScheduledTaskPrincipal -UserId $identity `
  -LogonType S4U -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet `
  -ExecutionTimeLimit (New-TimeSpan -Hours 48) `
  -MultipleInstances IgnoreNew -AllowStartIfOnBatteries `
  -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName $taskName -Action $action `
  -Principal $principal -Settings $settings -Force | Out-Null
Start-ScheduledTask -TaskName $taskName
"started=$taskName"
"state=$state"
"runner=$runner"
"""
    return run(hosts["third"], encoded_powershell(script),
               timeout=180, check=False)


def cache_chain_status(hosts):
    task = "GGEUR-domainnet-original-cache-chain-third"
    state = f"{THIRD_REPO}/exp/domainnet_original_cache_20260804".replace(
        "\\", "/")
    cache = (
        f"{THIRD_REPO}/exp/"
        "distributed_feature_cache_domainnet_original_20260804").replace(
            "\\", "/")
    script = rf"""
$ErrorActionPreference = 'Continue'
$task = Get-ScheduledTask -TaskName '{task}' -ErrorAction SilentlyContinue
if ($task) {{
  $info = Get-ScheduledTaskInfo -TaskName '{task}'
  "task_state=$($task.State) last_result=$($info.LastTaskResult) last_run=$($info.LastRunTime.ToString('o'))"
}} else {{ "task_missing=true" }}
$state = '{ps_literal(state)}'
$cache = '{ps_literal(cache)}'
"completed=$(Test-Path -LiteralPath (Join-Path $state 'completed.json'))"
$active = Get-ChildItem -LiteralPath $state -File -Filter '*.stdout.log' `
  -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending |
  Select-Object -First 1
if ($active) {{
  "active_file=$($active.Name) bytes=$($active.Length) mtime=$($active.LastWriteTime.ToString('o'))"
  Get-Content -LiteralPath $active.FullName -Tail 10 `
    -ErrorAction SilentlyContinue
}}
$failures = @(Get-ChildItem -LiteralPath $state -File `
  -ErrorAction SilentlyContinue | Select-String `
  -Pattern 'Traceback|(^|\W)ERROR(\W|$)|RuntimeError' `
  -ErrorAction SilentlyContinue)
"failure_matches=$($failures.Count)"
if ($failures.Count -gt 0) {{
  $failures | Select-Object -Last 10 | ForEach-Object {{
    "failure=$($_.Path):$($_.LineNumber):$($_.Line)"
  }}
}}
foreach ($group in @('domainnet_vit','domainnet_cnn','domainnet_mixer')) {{
  $dir = Join-Path $cache $group
  $files = @(Get-ChildItem -LiteralPath $dir -File -ErrorAction SilentlyContinue)
  "cache_group=$group files=$($files.Count) bytes=$(($files | Measure-Object Length -Sum).Sum) ready=$(Test-Path -LiteralPath (Join-Path $dir '.ggeur_feature_cache_ready.json'))"
}}
"eval_complete=$(Test-Path -LiteralPath (Join-Path $cache 'domainnet_eval_cache_completion.json'))"
"""
    return run(hosts["third"], encoded_powershell(script),
               timeout=300, check=False)


def cache_root(host_name):
    repo = CLIENT_REPO if host_name == "client8g" else THIRD_REPO
    return (f"{repo}/exp/"
            "distributed_feature_cache_domainnet_original_20260804").replace(
                "\\", "/")


def remote_tree_manifest(client, root):
    script = rf"""
$ErrorActionPreference = 'Stop'
$root = '{ps_literal(root)}'
if (-not (Test-Path -LiteralPath $root -PathType Container)) {{
  '[]'
  exit 0
}}
$root = (Get-Item -LiteralPath $root).FullName.TrimEnd('\')
$items = @(Get-ChildItem -LiteralPath $root -Recurse -File |
  Where-Object {{ -not $_.Name.EndsWith('.part') }} | ForEach-Object {{
  [pscustomobject]@{{
    path = $_.FullName.Substring($root.Length + 1).Replace('\','/')
    bytes = [int64]$_.Length
    sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
  }}
}})
ConvertTo-Json -InputObject $items -Depth 4 -Compress
"""
    result = run(client, encoded_powershell(script), timeout=7200)
    content = result["stdout"].strip()
    parsed = json.loads(content or "[]")
    if isinstance(parsed, dict):
        parsed = [parsed]
    return sorted(parsed, key=lambda item: item["path"].casefold())


def cache_completion_guard(hosts):
    root = cache_root("third")
    required = [
        f"{root}/domainnet_eval_cache_completion.json",
        *(f"{root}/{group}/.ggeur_feature_cache_ready.json"
          for group in ("domainnet_vit", "domainnet_cnn",
                        "domainnet_mixer")),
    ]
    state = (f"{THIRD_REPO}/exp/domainnet_original_cache_20260804/"
             "completed.json").replace("\\", "/")
    checks = ",".join(
        f"'{ps_literal(item)}'" for item in [state, *required])
    script = rf"""
$ErrorActionPreference = 'Stop'
$required = @({checks})
$missing = @($required | Where-Object {{
  -not (Test-Path -LiteralPath $_ -PathType Leaf)
}})
if ($missing.Count -gt 0) {{
  throw ('cache chain is incomplete; missing: ' + ($missing -join ', '))
}}
"cache_chain_complete=true"
"""
    return run(hosts["third"], encoded_powershell(script), timeout=120)


def cache_sync_order(item):
    name = item["path"].replace("\\", "/").casefold()
    if name.endswith(".npz"):
        return 0, name
    if name.endswith(".ggeur_feature_cache_ready.json"):
        return 2, name
    if name.endswith("domainnet_eval_cache_completion.json"):
        return 3, name
    return 1, name


def ensure_cache_directories(client, destination_root, source_manifest):
    directories = sorted({
        str(Path(item["path"]).parent).replace("\\", "/")
        for item in source_manifest
        if str(Path(item["path"]).parent) not in ("", ".")
    })
    values = ",".join(
        f"'{ps_literal(destination_root.rstrip('/') + '/' + item)}'"
        for item in directories)
    script = rf"""
$ErrorActionPreference = 'Stop'
[IO.Directory]::CreateDirectory('{ps_literal(destination_root)}') | Out-Null
foreach ($path in @({values})) {{
  [IO.Directory]::CreateDirectory($path) | Out-Null
}}
"directories_ready=true"
"""
    return run(client, encoded_powershell(script), timeout=300)


def relay_cache_file(hosts, item, destination_manifest):
    relative = item["path"].replace("\\", "/")
    expected_size = int(item["bytes"])
    expected_sha = item["sha256"].casefold()
    source = f"{cache_root('third').rstrip('/')}/{relative}"
    final = f"{cache_root('client8g').rstrip('/')}/{relative}"
    # Keep interrupted relay attempts outside the verified cache manifest.
    # A distinct suffix also avoids reusing an SFTP handle that Windows may
    # briefly retain after the relay process is terminated.
    partial = final + ".relay.part"
    existing = destination_manifest.get(relative.casefold())
    if (existing and int(existing["bytes"]) == expected_size and
            existing["sha256"].casefold() == expected_sha):
        print(json.dumps({"event": "cache_sync_skip", "path": relative,
                          "bytes": expected_size, "sha256": expected_sha}),
              flush=True)
        return
    if existing:
        preserve_remote(hosts["client8g"], final,
                        "destination cache size/hash mismatch")

    source_sftp = hosts["third"].open_sftp()
    destination_sftp = hosts["client8g"].open_sftp()
    try:
        try:
            offset = destination_sftp.stat(partial).st_size
        except OSError:
            offset = 0
        if offset > expected_size:
            destination_sftp.close()
            preserve_remote(hosts["client8g"], partial,
                            "partial cache larger than source")
            destination_sftp = hosts["client8g"].open_sftp()
            offset = 0
        if offset == expected_size and offset > 0:
            destination_sftp.close()
            size, digest = remote_hash(hosts["client8g"], partial)
            if size == expected_size and digest == expected_sha:
                finalize_remote(hosts["client8g"], partial, final)
                print(json.dumps({"event": "cache_sync_finalize",
                                  "path": relative, "bytes": size,
                                  "sha256": digest}), flush=True)
                return
            preserve_remote(hosts["client8g"], partial,
                            "complete partial cache hash mismatch")
            destination_sftp = hosts["client8g"].open_sftp()
            offset = 0

        started = time.time()
        next_report = offset + PROGRESS_BYTES
        source_stream = source_sftp.open(source, "rb", bufsize=0)
        source_stream.seek(offset)
        destination_stream = destination_sftp.open(
            partial, "ab" if offset else "wb", bufsize=0)
        next_flush = offset + RELAY_FLUSH_BYTES
        try:
            while True:
                chunk = source_stream.read(RELAY_CHUNK_BYTES)
                if not chunk:
                    break
                destination_stream.write(chunk)
                offset += len(chunk)
                if offset >= next_flush:
                    destination_stream.flush()
                    next_flush = offset + RELAY_FLUSH_BYTES
                if offset >= next_report:
                    elapsed = max(time.time() - started, 0.001)
                    print(json.dumps({
                        "event": "cache_sync_progress", "path": relative,
                        "bytes": offset, "total": expected_size,
                        "percent": round(offset * 100 / expected_size, 2),
                        "session_mib_s": round(
                            offset / elapsed / (1024 * 1024), 2),
                    }), flush=True)
                    next_report = offset + PROGRESS_BYTES
        finally:
            destination_stream.close()
            source_stream.close()
    finally:
        try:
            destination_sftp.close()
        except Exception:
            pass
        source_sftp.close()

    size, digest = remote_hash(hosts["client8g"], partial)
    if size != expected_size or digest != expected_sha:
        raise RuntimeError(
            f"cache relay verification failed for {relative}: "
            f"size={size} sha256={digest}")
    finalize_remote(hosts["client8g"], partial, final)
    print(json.dumps({"event": "cache_sync_complete", "path": relative,
                      "bytes": size, "sha256": digest}), flush=True)


def sync_cache(hosts):
    cache_completion_guard(hosts)
    source = remote_tree_manifest(hosts["third"], cache_root("third"))
    if not source:
        raise RuntimeError("source cache tree is empty")
    ensure_cache_directories(hosts["client8g"], cache_root("client8g"),
                             source)
    destination = {
        item["path"].casefold(): item
        for item in remote_tree_manifest(
            hosts["client8g"], cache_root("client8g"))
    }
    for item in sorted(source, key=cache_sync_order):
        relay_cache_file(hosts, item, destination)

    target = remote_tree_manifest(hosts["client8g"],
                                  cache_root("client8g"))
    source_map = {item["path"].casefold(): item for item in source}
    target_map = {item["path"].casefold(): item for item in target}
    mismatches = []
    for key, expected in source_map.items():
        actual = target_map.get(key)
        if not actual or int(actual["bytes"]) != int(expected["bytes"]) or \
                actual["sha256"].casefold() != expected["sha256"].casefold():
            mismatches.append(expected["path"])
    if mismatches:
        raise RuntimeError(f"cache tree verification failed: {mismatches}")
    return {
        "source_files": len(source),
        "source_bytes": sum(int(item["bytes"]) for item in source),
        "verified_files": len(source) - len(mismatches),
        "source_root": cache_root("third"),
        "destination_root": cache_root("client8g"),
    }


def start_cache_http_server(hosts):
    root = cache_root("third")
    script = rf"""
$ErrorActionPreference = 'Stop'
$python = 'C:\Users\pc\miniconda3\envs\cerp\python.exe'
$root = '{ps_literal(root)}'
$port = {HTTP_CACHE_PORT}
$state = Join-Path '{ps_literal(THIRD_REPO)}' 'exp\domainnet_original_cache_http_20260804'
[IO.Directory]::CreateDirectory($state) | Out-Null
$listeners = @(Get-NetTCPConnection -State Listen -LocalPort $port `
  -ErrorAction SilentlyContinue)
if ($listeners.Count -gt 0) {{
  throw "refusing to reuse occupied HTTP port $port"
}}
$stamp = (Get-Date).ToString('yyyyMMdd_HHmmss')
$stdout = Join-Path $state "http_$stamp.stdout.log"
$stderr = Join-Path $state "http_$stamp.stderr.log"
$arguments = @('-m', 'http.server', "$port", '--bind',
  '10.129.248.111', '--directory', $root)
$process = Start-Process -FilePath $python -ArgumentList $arguments `
  -WindowStyle Hidden -RedirectStandardOutput $stdout `
  -RedirectStandardError $stderr -PassThru
Start-Sleep -Seconds 2
if ($process.HasExited) {{
  throw "cache HTTP server exited early with code $($process.ExitCode)"
}}
"pid=$($process.Id)"
"stdout=$stdout"
"stderr=$stderr"
"root=$root"
"port=$port"
"""
    result = run(hosts["third"], encoded_powershell(script), timeout=120)
    values = {}
    for line in result["stdout"].splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    if "pid" not in values:
        raise RuntimeError(f"HTTP server did not report a PID: {result}")
    return int(values["pid"]), result


def stop_cache_http_server(hosts, pid):
    script = rf"""
$ErrorActionPreference = 'Stop'
$pidValue = {int(pid)}
$process = Get-Process -Id $pidValue -ErrorAction SilentlyContinue
if ($null -ne $process) {{
  Stop-Process -Id $pidValue -Force
  Wait-Process -Id $pidValue -Timeout 30 -ErrorAction SilentlyContinue
}}
"stopped_pid=$pidValue"
"""
    return run(hosts["third"], encoded_powershell(script), timeout=60,
               check=False)


def download_cache_file_http(hosts, item, destination_manifest):
    relative = item["path"].replace("\\", "/")
    expected_size = int(item["bytes"])
    expected_sha = item["sha256"].casefold()
    final = f"{cache_root('client8g').rstrip('/')}/{relative}"
    partial = final + ".http.part"
    existing = destination_manifest.get(relative.casefold())
    if (existing and int(existing["bytes"]) == expected_size and
            existing["sha256"].casefold() == expected_sha):
        print(json.dumps({"event": "cache_http_skip", "path": relative,
                          "bytes": expected_size, "sha256": expected_sha}),
              flush=True)
        return
    if existing:
        preserve_remote(hosts["client8g"], final,
                        "destination cache size/hash mismatch")

    url_path = quote(relative, safe="/._-")
    url = f"http://10.129.248.111:{HTTP_CACHE_PORT}/{url_path}"
    script = rf"""
$ErrorActionPreference = 'Stop'
$partial = '{ps_literal(partial)}'
$url = '{ps_literal(url)}'
& "$env:SystemRoot\System32\curl.exe" --fail --show-error --silent `
  --retry 12 --retry-delay 3 --retry-all-errors `
  --connect-timeout 15 --max-time 7200 `
  --output $partial $url
if ($LASTEXITCODE -ne 0) {{
  throw "curl cache download failed with exit code $LASTEXITCODE: $url"
}}
"downloaded=$partial"
"bytes=$((Get-Item -LiteralPath $partial).Length)"
"""
    started = time.time()
    result = run(hosts["client8g"], encoded_powershell(script), timeout=7500)
    size, digest = remote_hash(hosts["client8g"], partial)
    if size != expected_size or digest != expected_sha:
        preserve_remote(hosts["client8g"], partial,
                        "HTTP cache size/hash mismatch")
        raise RuntimeError(
            f"HTTP cache verification failed for {relative}: "
            f"size={size} sha256={digest}")
    finalize_remote(hosts["client8g"], partial, final)
    print(json.dumps({
        "event": "cache_http_complete", "path": relative,
        "bytes": size, "sha256": digest,
        "elapsed_seconds": round(time.time() - started, 2),
        "download": result,
    }, ensure_ascii=True), flush=True)


def sync_cache_http(hosts):
    cache_completion_guard(hosts)
    source = remote_tree_manifest(hosts["third"], cache_root("third"))
    if not source:
        raise RuntimeError("source cache tree is empty")
    ensure_cache_directories(hosts["client8g"], cache_root("client8g"),
                             source)
    destination = {
        item["path"].casefold(): item
        for item in remote_tree_manifest(
            hosts["client8g"], cache_root("client8g"))
    }
    pid, server = start_cache_http_server(hosts)
    print(json.dumps({"event": "cache_http_server_started", "pid": pid,
                      "result": server}, ensure_ascii=True), flush=True)
    try:
        probe = rf"""
$ErrorActionPreference = 'Stop'
$url = 'http://10.129.248.111:{HTTP_CACHE_PORT}/domainnet_eval_cache_completion.json'
& "$env:SystemRoot\System32\curl.exe" --fail --silent --show-error `
  --connect-timeout 10 --max-time 30 --head $url | Out-Null
if ($LASTEXITCODE -ne 0) {{ throw "cache HTTP probe failed: $LASTEXITCODE" }}
"http_probe=ok"
"""
        run(hosts["client8g"], encoded_powershell(probe), timeout=60)
        for item in sorted(source, key=cache_sync_order):
            download_cache_file_http(hosts, item, destination)
    finally:
        stopped = stop_cache_http_server(hosts, pid)
        print(json.dumps({"event": "cache_http_server_stopped",
                          "pid": pid, "result": stopped},
                         ensure_ascii=True), flush=True)

    target = remote_tree_manifest(hosts["client8g"],
                                  cache_root("client8g"))
    source_map = {item["path"].casefold(): item for item in source}
    target_map = {item["path"].casefold(): item for item in target}
    mismatches = []
    for key, expected in source_map.items():
        actual = target_map.get(key)
        if not actual or int(actual["bytes"]) != int(expected["bytes"]) or \
                actual["sha256"].casefold() != expected["sha256"].casefold():
            mismatches.append(expected["path"])
    if mismatches:
        raise RuntimeError(f"HTTP cache tree verification failed: {mismatches}")
    return {
        "transport": "direct-http-10.x",
        "source_files": len(source),
        "source_bytes": sum(int(item["bytes"]) for item in source),
        "verified_files": len(source_map),
        "verified_bytes": sum(int(item["bytes"])
                              for item in source_map.values()),
    }


def download_cache_file_local(hosts, item, local_path):
    relative = item["path"].replace("\\", "/")
    expected_size = int(item["bytes"])
    expected_sha = item["sha256"].casefold()
    source = f"{cache_root('third').rstrip('/')}/{relative}"
    partial = Path(str(local_path) + ".part")
    if local_path.is_file():
        if (local_path.stat().st_size == expected_size and
                sha256_file(local_path) == expected_sha):
            return local_path
        invalid = Path(str(local_path) +
                       time.strftime(".invalid_%Y%m%d_%H%M%S"))
        local_path.replace(invalid)
    if partial.exists() and partial.stat().st_size > expected_size:
        invalid = Path(str(partial) +
                       time.strftime(".invalid_%Y%m%d_%H%M%S"))
        partial.replace(invalid)
    offset = partial.stat().st_size if partial.exists() else 0
    started = time.time()
    next_report = offset + PROGRESS_BYTES
    sftp = hosts["third"].open_sftp()
    try:
        source_stream = sftp.open(source, "rb")
        source_stream.seek(offset)
        try:
            with partial.open("ab" if offset else "wb") as destination:
                while True:
                    chunk = source_stream.read(RELAY_CHUNK_BYTES)
                    if not chunk:
                        break
                    destination.write(chunk)
                    offset += len(chunk)
                    if offset >= next_report:
                        print(json.dumps({
                            "event": "cache_local_download_progress",
                            "path": relative, "bytes": offset,
                            "total": expected_size,
                            "percent": round(offset * 100 / expected_size, 2),
                        }), flush=True)
                        next_report = offset + PROGRESS_BYTES
        finally:
            source_stream.close()
    finally:
        sftp.close()
    if partial.stat().st_size != expected_size:
        raise RuntimeError(
            f"local relay size mismatch for {relative}: "
            f"{partial.stat().st_size} != {expected_size}")
    digest = sha256_file(partial)
    if digest != expected_sha:
        invalid = Path(str(partial) +
                       time.strftime(".invalid_%Y%m%d_%H%M%S"))
        partial.replace(invalid)
        raise RuntimeError(
            f"local relay hash mismatch for {relative}: {digest}")
    partial.replace(local_path)
    print(json.dumps({
        "event": "cache_local_download_complete", "path": relative,
        "bytes": expected_size, "sha256": digest,
        "elapsed_seconds": round(time.time() - started, 2),
    }), flush=True)
    return local_path


def upload_cache_file_from_local(hosts, item, local_path,
                                 destination_manifest):
    relative = item["path"].replace("\\", "/")
    expected_size = int(item["bytes"])
    expected_sha = item["sha256"].casefold()
    final = f"{cache_root('client8g').rstrip('/')}/{relative}"
    partial = final + ".localrelay.part"
    existing = destination_manifest.get(relative.casefold())
    if (existing and int(existing["bytes"]) == expected_size and
            existing["sha256"].casefold() == expected_sha):
        print(json.dumps({"event": "cache_local_relay_skip",
                          "path": relative, "bytes": expected_size,
                          "sha256": expected_sha}), flush=True)
        return
    if existing:
        preserve_remote(hosts["client8g"], final,
                        "destination cache size/hash mismatch")

    sftp = hosts["client8g"].open_sftp()
    try:
        try:
            offset = sftp.stat(partial).st_size
        except OSError:
            offset = 0
        if offset > expected_size:
            sftp.close()
            preserve_remote(hosts["client8g"], partial,
                            "local relay partial larger than source")
            sftp = hosts["client8g"].open_sftp()
            offset = 0
        if offset == expected_size and offset > 0:
            sftp.close()
            size, digest = remote_hash(hosts["client8g"], partial)
            if digest == expected_sha:
                finalize_remote(hosts["client8g"], partial, final)
                return
            preserve_remote(hosts["client8g"], partial,
                            "local relay complete partial hash mismatch")
            sftp = hosts["client8g"].open_sftp()
            offset = 0

        started = time.time()
        next_report = offset + PROGRESS_BYTES
        with local_path.open("rb") as source:
            source.seek(offset)
            destination = sftp.open(partial, "ab" if offset else "wb")
            destination.set_pipelined(True)
            try:
                while True:
                    chunk = source.read(CHUNK_BYTES)
                    if not chunk:
                        break
                    destination.write(chunk)
                    offset += len(chunk)
                    if offset >= next_report:
                        print(json.dumps({
                            "event": "cache_local_upload_progress",
                            "path": relative, "bytes": offset,
                            "total": expected_size,
                            "percent": round(offset * 100 / expected_size, 2),
                        }), flush=True)
                        next_report = offset + PROGRESS_BYTES
            finally:
                destination.close()
    finally:
        try:
            sftp.close()
        except Exception:
            pass
    size, digest = remote_hash(hosts["client8g"], partial)
    if size != expected_size or digest != expected_sha:
        raise RuntimeError(
            f"local relay upload verification failed for {relative}: "
            f"size={size} sha256={digest}")
    finalize_remote(hosts["client8g"], partial, final)
    print(json.dumps({
        "event": "cache_local_upload_complete", "path": relative,
        "bytes": size, "sha256": digest,
        "elapsed_seconds": round(time.time() - started, 2),
    }), flush=True)


def sync_cache_local_relay(hosts):
    cache_completion_guard(hosts)
    source = remote_tree_manifest(hosts["third"], cache_root("third"))
    if not source:
        raise RuntimeError("source cache tree is empty")
    ensure_cache_directories(hosts["client8g"], cache_root("client8g"),
                             source)
    destination = {
        item["path"].casefold(): item
        for item in remote_tree_manifest(
            hosts["client8g"], cache_root("client8g"))
    }
    relay_root = (Path(__file__).resolve().parents[3] / "exp" /
                  "dual_deploy" /
                  "domainnet_cache_local_relay_20260804")
    relay_root.mkdir(parents=True, exist_ok=True)
    for item in sorted(source, key=cache_sync_order):
        relative = item["path"].replace("\\", "/")
        existing = destination.get(relative.casefold())
        if (existing and int(existing["bytes"]) == int(item["bytes"]) and
                existing["sha256"].casefold() == item["sha256"].casefold()):
            print(json.dumps({"event": "cache_local_relay_skip",
                              "path": relative,
                              "bytes": int(item["bytes"]),
                              "sha256": item["sha256"]}), flush=True)
            continue
        local_name = f"{item['sha256']}_{Path(relative).name}"
        local_path = download_cache_file_local(
            hosts, item, relay_root / local_name)
        upload_cache_file_from_local(hosts, item, local_path, destination)
        local_path.unlink()

    target = remote_tree_manifest(hosts["client8g"],
                                  cache_root("client8g"))
    source_map = {item["path"].casefold(): item for item in source}
    target_map = {item["path"].casefold(): item for item in target}
    mismatches = []
    for key, expected in source_map.items():
        actual = target_map.get(key)
        if not actual or int(actual["bytes"]) != int(expected["bytes"]) or \
                actual["sha256"].casefold() != expected["sha256"].casefold():
            mismatches.append(expected["path"])
    if mismatches:
        raise RuntimeError(
            f"local relay cache tree verification failed: {mismatches}")
    return {
        "transport": "ssh-local-relay-via-8g",
        "source_files": len(source),
        "source_bytes": sum(int(item["bytes"]) for item in source),
        "verified_files": len(source_map),
        "verified_bytes": sum(int(item["bytes"])
                              for item in source_map.values()),
    }


def bind_client8g_cache_markers(hosts):
    source = remote_tree_manifest(hosts["third"], cache_root("third"))
    target = remote_tree_manifest(hosts["client8g"],
                                  cache_root("client8g"))
    source_map = {item["path"].casefold(): item for item in source}
    target_map = {item["path"].casefold(): item for item in target}
    mismatches = []
    for key, expected in source_map.items():
        actual = target_map.get(key)
        if not actual or int(actual["bytes"]) != int(expected["bytes"]) or \
                actual["sha256"].casefold() != expected["sha256"].casefold():
            mismatches.append(expected["path"])
    if mismatches:
        raise RuntimeError(
            f"refusing to bind unverified client8g cache: {mismatches}")

    root = cache_root("client8g")
    script = rf"""
$ErrorActionPreference = 'Stop'
$root = '{ps_literal(root)}'
foreach ($group in @('domainnet_vit','domainnet_cnn','domainnet_mixer')) {{
  $dir = Join-Path $root $group
  $marker = Join-Path $dir '.ggeur_feature_cache_ready.json'
  $state = Get-Content -LiteralPath $marker -Raw | ConvertFrom-Json
  if ([string]$state.group -ne $group) {{
    throw "marker group mismatch: expected=$group actual=$($state.group)"
  }}
  if (-not $state.require_complete_feature_cache) {{
    throw "marker does not require a complete cache: $group"
  }}
  $sourceSha = (Get-FileHash -LiteralPath $marker -Algorithm SHA256).Hash.ToLowerInvariant()
  $sourceHost = [string]$state.validated_on
  foreach ($domain in @('clipart','painting','real','sketch')) {{
    $entry = $state.domains.$domain
    if ($null -eq $entry) {{ throw "missing marker domain: $group/$domain" }}
    $name = Split-Path -Leaf ([string]$entry.path)
    $path = Join-Path $dir $name
    $item = Get-Item -LiteralPath $path
    $sha = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ([int64]$item.Length -ne [int64]$entry.bytes -or
        $sha -ne ([string]$entry.sha256).ToLowerInvariant()) {{
      throw "marker cache mismatch: $group/$domain"
    }}
    $entry.path = $path
  }}
  $state | Add-Member -NotePropertyName source_marker_sha256 `
    -NotePropertyValue $sourceSha -Force
  $state | Add-Member -NotePropertyName source_validated_on `
    -NotePropertyValue $sourceHost -Force
  $state | Add-Member -NotePropertyName validated_on `
    -NotePropertyValue $env:COMPUTERNAME -Force
  $state | Add-Member -NotePropertyName validated_at `
    -NotePropertyValue (Get-Date -Format o) -Force
  $state | ConvertTo-Json -Depth 20 |
    Set-Content -LiteralPath $marker -Encoding UTF8
  "bound=$group validated_on=$env:COMPUTERNAME source_marker_sha256=$sourceSha"
}}
"""
    result = run(hosts["client8g"], encoded_powershell(script), timeout=1800)
    return {
        "prebind_verified_files": len(source_map),
        "prebind_verified_bytes": sum(int(item["bytes"])
                                      for item in source_map.values()),
        "client8g": result,
    }


def resume_client8g_driver(hosts, run_id):
    if not run_id or not all(char.isalnum() or char in "_-" for char in run_id):
        raise RuntimeError("a safe --run-id is required")
    task_name = f"GGEUR-dual-{run_id}-clients-8g"
    script = rf"""
$ErrorActionPreference = 'Stop'
$taskName = '{ps_literal(task_name)}'
$task = Get-ScheduledTask -TaskName $taskName -ErrorAction Stop
if ($task.State -eq 'Running') {{
  "already_running=$taskName"
}} else {{
  Start-ScheduledTask -TaskName $taskName
  Start-Sleep -Seconds 2
  $task = Get-ScheduledTask -TaskName $taskName -ErrorAction Stop
  "started=$taskName state=$($task.State)"
}}
"""
    return run(hosts["client8g"], encoded_powershell(script), timeout=60)


def cache_sync_status(hosts):
    result = {}
    for name in ("third", "client8g"):
        manifest = remote_tree_manifest(hosts[name], cache_root(name))
        result[name] = {
            "root": cache_root(name),
            "files": len(manifest),
            "bytes": sum(int(item["bytes"]) for item in manifest),
            "ready_groups": sorted({
                item["path"].split("/", 1)[0]
                for item in manifest
                if item["path"].endswith(".ggeur_feature_cache_ready.json")
            }),
            "eval_complete": any(
                item["path"] == "domainnet_eval_cache_completion.json"
                for item in manifest),
        }
    return result


def cache_sync_live_status(hosts):
    result = {}
    for name in ("third", "client8g"):
        root = cache_root(name)
        script = rf"""
$ErrorActionPreference = 'Continue'
$root = '{ps_literal(root)}'
$files = @(Get-ChildItem -LiteralPath $root -Recurse -File `
  -ErrorAction SilentlyContinue)
"root=$root"
"files=$($files.Count) bytes=$(($files | Measure-Object Length -Sum).Sum)"
$files | Where-Object {{ $_.Name -like '*.part' }} |
  Sort-Object LastWriteTime -Descending | Select-Object -First 5 |
  ForEach-Object {{
    "partial=$($_.FullName.Substring($root.Length + 1)) bytes=$($_.Length) mtime=$($_.LastWriteTime.ToString('o'))"
  }}
$files | Where-Object {{ $_.Name -notlike '*.part' }} |
  Sort-Object LastWriteTime -Descending | Select-Object -First 3 |
  ForEach-Object {{
    "recent=$($_.FullName.Substring($root.Length + 1)) bytes=$($_.Length) mtime=$($_.LastWriteTime.ToString('o'))"
  }}
"""
        result[name] = run(hosts[name], encoded_powershell(script),
                           timeout=120, check=False)
    return result


def main():
    args = parse_args()
    cache_commands = {
        "start-cache", "cache-status", "sync-cache", "sync-cache-http",
        "sync-cache-local-relay", "bind-client8g-cache-markers",
        "resume-client8g-driver", "sync-cache-status", "sync-live-status",
    }
    audit = None
    if args.command not in cache_commands:
        verify_crc = args.command == "audit" and not args.skip_crc
        audit = inspect_archives(args.source_dir, verify_crc=verify_crc)
    if args.command == "audit":
        print(json.dumps(audit, ensure_ascii=True, indent=2))
        return

    hosts = connect_dual(args)
    try:
        if args.command == "status":
            result = {name: remote_status(hosts[name], name, audit)
                      for name in args.hosts}
            print(json.dumps(result, ensure_ascii=True, indent=2))
            return

        if args.command == "start-cache":
            print(json.dumps({"third": start_cache_chain(hosts)},
                             ensure_ascii=True, indent=2))
            return

        if args.command == "cache-status":
            print(json.dumps({"third": cache_chain_status(hosts)},
                             ensure_ascii=True, indent=2))
            return

        if args.command == "sync-cache":
            print(json.dumps(sync_cache(hosts), ensure_ascii=True, indent=2))
            return

        if args.command == "sync-cache-http":
            print(json.dumps(sync_cache_http(hosts), ensure_ascii=True,
                             indent=2))
            return

        if args.command == "sync-cache-local-relay":
            print(json.dumps(sync_cache_local_relay(hosts),
                             ensure_ascii=True, indent=2))
            return

        if args.command == "bind-client8g-cache-markers":
            print(json.dumps(bind_client8g_cache_markers(hosts),
                             ensure_ascii=True, indent=2))
            return

        if args.command == "resume-client8g-driver":
            print(json.dumps(resume_client8g_driver(hosts, args.run_id),
                             ensure_ascii=True, indent=2))
            return

        if args.command == "sync-cache-status":
            print(json.dumps(cache_sync_status(hosts), ensure_ascii=True,
                             indent=2))
            return

        if args.command == "sync-live-status":
            print(json.dumps(cache_sync_live_status(hosts),
                             ensure_ascii=True, indent=2))
            return

        required = (audit["summary"]["archive_bytes"] +
                    audit["summary"]["uncompressed_image_bytes"] +
                    5 * 1024 * 1024 * 1024)
        for name in args.hosts:
            prepared = remote_prepare(hosts[name], name, required)
            print(json.dumps({"event": "remote_prepare", "host": name,
                              "result": prepared}, ensure_ascii=True),
                  flush=True)

        if args.command == "transfer":
            for name in args.hosts:
                for domain in DOMAINS:
                    metadata = audit[domain]
                    upload_resumable(
                        hosts[name], name, metadata["path"],
                        metadata["archive_sha256"])
            return

        if args.command == "extract":
            result = {}
            for name in args.hosts:
                result[name] = extract_remote(hosts[name], name, audit)
            print(json.dumps(result, ensure_ascii=True, indent=2))
            return
    finally:
        close_dual(hosts)


if __name__ == "__main__":
    main()
