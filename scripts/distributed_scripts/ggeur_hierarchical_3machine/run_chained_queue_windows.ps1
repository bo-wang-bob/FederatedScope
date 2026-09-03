param(
    [Parameter(Mandatory = $true)][ValidateSet("subservers", "clients")]
    [string]$Role,
    [Parameter(Mandatory = $true)][string]$RunIds,
    [string]$ControlUrl = "http://10.112.81.135:60049",
    [string]$PythonBin = "C:\Users\pc\miniconda3\envs\cerp\python.exe",
    [string]$ChainId = "remaining_20260731_v1"
)

$ErrorActionPreference = "Stop"
$RepoDir = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$StateDir = Join-Path $RepoDir "exp\hierarchical\$ChainId"
$StateLog = Join-Path $StateDir "$($Role)_chain.tsv"
$CompleteMarker = Join-Path $StateDir "$($Role)_chain.complete"
New-Item -ItemType Directory -Force -Path $StateDir | Out-Null
. (Join-Path $PSScriptRoot "queue_control_client.ps1")

function Write-State([string]$RunId, [string]$State) {
    Add-Content -LiteralPath $StateLog `
        -Value "$(Get-Date -Format o)`t$RunId`t$State"
}

function Wait-ForRun([string]$RunId) {
    while ($true) {
        $state = Get-QueueControlState -ControlUrl $ControlUrl
        if ($null -ne $state -and [string]$state.run_id -eq $RunId) {
            if ([string]$state.phase -eq "failed") {
                throw "Root queue failed before $Role joined ${RunId}: $($state.message)"
            }
            return $state
        }
        Start-Sleep -Seconds 5
    }
}

if (Test-Path -LiteralPath $CompleteMarker) {
    Write-State ALL ALREADY_COMPLETE
    return
}

$QueueScript = if ($Role -eq "subservers") {
    Join-Path $PSScriptRoot "run_remaining_queue_subservers.ps1"
} else {
    Join-Path $PSScriptRoot "run_remaining_queue_clients.ps1"
}

$Runs = @($RunIds.Split(",") | ForEach-Object { $_.Trim() } |
    Where-Object { $_ })
if ($Runs.Count -eq 0) {
    throw "RunIds must contain at least one run ID"
}

foreach ($RunId in $Runs) {
    $RunRoot = Join-Path $PSScriptRoot "runs\$RunId"
    if (-not (Test-Path -LiteralPath (Join-Path $RunRoot "matrix_manifest.json"))) {
        throw "Missing matrix manifest for $RunId"
    }
    $RoleComplete = Join-Path $RunRoot "queue_state\$(
        if ($Role -eq 'subservers') { 'subserver_queue.complete' }
        else { 'client_queue.complete' })"
    if (Test-Path -LiteralPath $RoleComplete) {
        Write-State $RunId SKIP_COMPLETE
        continue
    }

    Write-State $RunId WAIT_ROOT_CONTROL
    $state = Wait-ForRun $RunId
    Write-State $RunId "START_ROLE_QUEUE:phase=$($state.phase)"
    if ($Role -eq "subservers") {
        & $QueueScript -RunId $RunId -PythonBin $PythonBin `
            -ControlUrl $ControlUrl -ControlWaitTimeoutSec 0
    } else {
        & $QueueScript -RunId $RunId -ControlUrl $ControlUrl `
            -ControlWaitTimeoutSec 0
    }
    if ($LASTEXITCODE -ne 0) {
        throw "$Role queue returned $LASTEXITCODE for $RunId"
    }
    if (-not (Test-Path -LiteralPath $RoleComplete)) {
        throw "$Role queue exited without completion marker for $RunId"
    }
    Write-State $RunId COMPLETE
}

New-Item -ItemType File -Force -Path $CompleteMarker | Out-Null
Write-State ALL COMPLETE
