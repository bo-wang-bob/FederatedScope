param(
    [Parameter(Mandatory = $true)][string]$RunId,
    [Parameter(Mandatory = $true)][string]$PayloadFile,
    [Parameter(Mandatory = $true)][string]$ConnectHost,
    [Parameter(Mandatory = $true)][double]$ConnectAtUnix,
    [int]$Port = 62010,
    [string]$ProbeId = "third-late-edge-probe",
    [string]$RunDirLabel = "third_probe",
    [string]$HostRole = "third",
    [string]$PythonBin = "C:\Users\pc\miniconda3\envs\cerp\python.exe",
    [string]$RepoDir = "C:\Users\pc\FederatedScope"
)

$ErrorActionPreference = "Stop"
$runDir = Join-Path $RepoDir "exp\concurrent_availability\$RunId\$RunDirLabel"
New-Item -ItemType Directory -Force -Path $runDir | Out-Null
@(
    "role=late_edge_probe",
    "host_role=$HostRole",
    "run_id=$RunId",
    "business_endpoint=${ConnectHost}:$Port",
    "connect_at_unix=$ConnectAtUnix",
    "payload_file=$PayloadFile",
    "payload_sha256=$((Get-FileHash -Algorithm SHA256 -LiteralPath $PayloadFile).Hash.ToLowerInvariant())",
    "started_at=$((Get-Date).ToString('o'))"
) | Set-Content -LiteralPath (Join-Path $runDir "command_audit.log")

& $PythonBin `
    (Join-Path $RepoDir "scripts\benchmark_headonly_concurrent_availability.py") `
    probe `
    --payload-file $PayloadFile `
    --connect-host $ConnectHost `
    --port $Port `
    --connect-at-unix $ConnectAtUnix `
    --probe-id $ProbeId `
    --connect-timeout 600 `
    --test-timeout 1800 `
    --io-timeout 900 `
    --output (Join-Path $runDir "probe_summary.json") `
    2>&1 | Tee-Object -FilePath (Join-Path $runDir "probe.stdout.log")
if ($LASTEXITCODE -ne 0) {
    throw "late edge probe exited with $LASTEXITCODE"
}
Add-Content -LiteralPath (Join-Path $runDir "command_audit.log") `
    -Value "finished_at=$((Get-Date).ToString('o'))"
