param(
    [Parameter(Mandatory = $true)][string]$RunId,
    [Parameter(Mandatory = $true)][string]$PayloadFile,
    [Parameter(Mandatory = $true)][double]$StartAtUnix,
    [Parameter(Mandatory = $true)][double]$ObserveAtUnix,
    [Parameter(Mandatory = $true)][double]$ActiveUntilUnix,
    [int]$BasePort = 62020,
    [int]$Subservers = 5,
    [int]$ClientsPerSubserver = 1000,
    [int]$ExpectedProbes = 0,
    [int]$ExpectedTrainingClients = 0,
    [double]$RequestDurationSec = 0,
    [int]$RequestsPerClient = 0,
    [double]$PollIntervalSec = 0.8,
    [int]$PollWorkerConnections = 200,
    [double]$RequestStartDelaySec = 2,
    [string]$PythonBin = "C:\Users\pc\miniconda3\envs\cerp\python.exe",
    [string]$RepoDir = "C:\Users\pc\FederatedScope"
)

$ErrorActionPreference = "Stop"
$runDir = Join-Path $RepoDir "exp\concurrent_availability\$RunId\third"
New-Item -ItemType Directory -Force -Path $runDir | Out-Null

$training = @(Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -like "*hierarchical_subserver.py*" -or
    $_.CommandLine -like "*federatedscope.main*"
})
if ($training.Count -gt 0) {
    throw "Refusing to overlap $($training.Count) formal training processes"
}

@(
    "role=availability_server",
    "host_role=third",
    "run_id=$RunId",
    "business_endpoint=10.129.248.111:$BasePort",
    "subservers=$Subservers",
    "clients_per_subserver=$ClientsPerSubserver",
    "expected_probes=$ExpectedProbes",
    "expected_training_clients=$ExpectedTrainingClients",
    "request_duration_sec=$RequestDurationSec",
    "requests_per_client=$RequestsPerClient",
    "poll_interval_sec=$PollIntervalSec",
    "poll_worker_connections=$PollWorkerConnections",
    "start_at_unix=$StartAtUnix",
    "observe_at_unix=$ObserveAtUnix",
    "active_until_unix=$ActiveUntilUnix",
    "payload_file=$PayloadFile",
    "payload_sha256=$((Get-FileHash -Algorithm SHA256 -LiteralPath $PayloadFile).Hash.ToLowerInvariant())",
    "started_at=$((Get-Date).ToString('o'))"
) | Set-Content -LiteralPath (Join-Path $runDir "command_audit.log")

& $PythonBin `
    (Join-Path $RepoDir "scripts\benchmark_headonly_concurrent_availability.py") `
    server `
    --payload-file $PayloadFile `
    --listen-host 0.0.0.0 `
    --base-port $BasePort `
    --subservers $Subservers `
    --clients-per-subserver $ClientsPerSubserver `
    --expected-probes $ExpectedProbes `
    --expected-training-clients $ExpectedTrainingClients `
    --request-duration-sec $RequestDurationSec `
    --requests-per-client $RequestsPerClient `
    --poll-interval-sec $PollIntervalSec `
    --poll-worker-connections $PollWorkerConnections `
    --request-start-delay-sec $RequestStartDelaySec `
    --start-at-unix $StartAtUnix `
    --observe-at-unix $ObserveAtUnix `
    --active-until-unix $ActiveUntilUnix `
    --ready-timeout 900 `
    --test-timeout 1800 `
    --io-timeout 900 `
    --output (Join-Path $runDir "server_summary.json") `
    2>&1 | Tee-Object -FilePath (Join-Path $runDir "server.stdout.log")
if ($LASTEXITCODE -ne 0) {
    throw "availability server exited with $LASTEXITCODE"
}
Add-Content -LiteralPath (Join-Path $runDir "command_audit.log") `
    -Value "finished_at=$((Get-Date).ToString('o'))"
