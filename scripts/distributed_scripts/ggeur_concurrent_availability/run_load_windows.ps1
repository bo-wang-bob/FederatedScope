param(
    [Parameter(Mandatory = $true)][string]$RunId,
    [Parameter(Mandatory = $true)][string]$PayloadFile,
    [Parameter(Mandatory = $true)][string]$ConnectHost,
    [int]$BasePort = 62010,
    [int]$Subservers = 5,
    [int]$ClientsPerSubserver = 1000,
    [int]$ClientIdOffset = 1,
    [int]$TrainClientCount = 0,
    [int]$TrainSteps = 1,
    [int]$TrainBatchSize = 8,
    [double]$TrainLearningRate = 0.01,
    [int]$TrainingConcurrency = 4,
    [string]$HostLabel = "root4090",
    [string]$RunDirLabel = "client8g",
    [string]$PythonBin = "D:\ProgramData\anaconda3\envs\pi_fmd_gpu_py39\python.exe",
    [string]$RepoDir = "D:\Projects\FederatedScope"
)

$ErrorActionPreference = "Stop"
$runDir = Join-Path $RepoDir "exp\concurrent_availability\$RunId\$RunDirLabel"
New-Item -ItemType Directory -Force -Path $runDir | Out-Null

$training = @(Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -like "*federatedscope.main*"
})
if ($training.Count -gt 0) {
    throw "Refusing to overlap $($training.Count) formal training processes"
}

$audit = Join-Path $runDir "${HostLabel}_command_audit.log"
@(
    "role=availability_load",
    "host_role=$RunDirLabel",
    "run_id=$RunId",
    "business_endpoint=${ConnectHost}:$BasePort",
    "subservers=$Subservers",
    "clients_per_subserver=$ClientsPerSubserver",
    "client_id_offset=$ClientIdOffset",
    "train_client_count=$TrainClientCount",
    "train_steps=$TrainSteps",
    "train_batch_size=$TrainBatchSize",
    "train_learning_rate=$TrainLearningRate",
    "training_concurrency=$TrainingConcurrency",
    "payload_file=$PayloadFile",
    "payload_sha256=$((Get-FileHash -Algorithm SHA256 -LiteralPath $PayloadFile).Hash.ToLowerInvariant())",
    "started_at=$((Get-Date).ToString('o'))"
) | Set-Content -LiteralPath $audit

& $PythonBin `
    (Join-Path $RepoDir "scripts\benchmark_headonly_concurrent_availability.py") `
    load `
    --payload-file $PayloadFile `
    --connect-host $ConnectHost `
    --base-port $BasePort `
    --subservers $Subservers `
    --clients-per-subserver $ClientsPerSubserver `
    --client-id-offset $ClientIdOffset `
    --client-id-prefix "$HostLabel-" `
    --train-client-count $TrainClientCount `
    --train-steps $TrainSteps `
    --train-batch-size $TrainBatchSize `
    --train-learning-rate $TrainLearningRate `
    --training-concurrency $TrainingConcurrency `
    --connect-timeout 600 `
    --test-timeout 1800 `
    --io-timeout 900 `
    --output (Join-Path $runDir "${HostLabel}_load_summary.json") `
    2>&1 | Tee-Object -FilePath (Join-Path $runDir "${HostLabel}_load.stdout.log")
if ($LASTEXITCODE -ne 0) {
    throw "availability load exited with $LASTEXITCODE"
}
Add-Content -LiteralPath $audit `
    -Value "finished_at=$((Get-Date).ToString('o'))"
