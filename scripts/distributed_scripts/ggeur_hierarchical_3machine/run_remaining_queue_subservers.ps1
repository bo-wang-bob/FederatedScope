param(
    [string]$RunId = "final_remaining_20260723_v1",
    [string]$PythonBin = "C:\Users\pc\miniconda3\envs\cerp\python.exe",
    [string]$ControlUrl = "http://10.112.81.135:60049",
    [int]$ControlWaitTimeoutSec = 172800
)

$ErrorActionPreference = "Stop"
$RepoDir = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$RunRoot = Join-Path $PSScriptRoot "runs\$RunId"
$StateDir = Join-Path $RunRoot "queue_state"
$StateLog = Join-Path $StateDir "subserver_queue.tsv"
New-Item -ItemType Directory -Force -Path $StateDir | Out-Null
. (Join-Path $PSScriptRoot "queue_control_client.ps1")

$ManifestPath = Join-Path $RunRoot "matrix_manifest.json"
if (Test-Path -LiteralPath $ManifestPath) {
    $Cases = @(
        (Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json).cases |
            ForEach-Object { [string]$_.case }
    )
} else {
    $Cases = @(
        "officehome_cnn_fedavg", "officehome_cnn_fedprox", "officehome_cnn_fedproto", "officehome_cnn_fedopt", "officehome_cnn_moon", "officehome_cnn_ggeur",
        "officehome_mixer_fedavg", "officehome_mixer_fedprox", "officehome_mixer_fedproto", "officehome_mixer_fedopt", "officehome_mixer_moon", "officehome_mixer_ggeur",
        "mdsent_rnn_fedavg", "mdsent_rnn_fedprox", "mdsent_rnn_fedproto", "mdsent_rnn_fedopt", "mdsent_rnn_ggeur",
        "mdsent_lstm_fedavg", "mdsent_lstm_fedprox", "mdsent_lstm_fedproto", "mdsent_lstm_fedopt", "mdsent_lstm_ggeur",
        "domainnet_vit_fedavg", "domainnet_vit_fedprox", "domainnet_vit_fedproto", "domainnet_vit_fedopt", "domainnet_vit_moon", "domainnet_vit_ggeur",
        "domainnet_cnn_fedavg", "domainnet_cnn_fedprox", "domainnet_cnn_fedproto", "domainnet_cnn_fedopt", "domainnet_cnn_moon", "domainnet_cnn_ggeur",
        "domainnet_mixer_fedavg", "domainnet_mixer_fedprox", "domainnet_mixer_fedproto", "domainnet_mixer_fedopt", "domainnet_mixer_moon", "domainnet_mixer_ggeur"
    )
}
if ($Cases.Count -eq 0) {
    throw "No cases found for run $RunId"
}

function Write-State([string]$Case, [string]$State) {
    Add-Content -LiteralPath $StateLog -Value "$(Get-Date -Format o)`t$Case`t$State"
}

function Test-Tcp([string]$HostName, [int]$Port) {
    $client = [Net.Sockets.TcpClient]::new()
    try {
        $task = $client.ConnectAsync($HostName, $Port)
        return $task.Wait(1500) -and $client.Connected
    } catch { return $false } finally { $client.Dispose() }
}

function Wait-Tcp([string]$HostName, [int]$Port, [bool]$Expected) {
    while ((Test-Tcp $HostName $Port) -ne $Expected) { Start-Sleep -Seconds 5 }
}

$initialRootState = $null
while ($null -eq $initialRootState -or
       [string]$initialRootState.run_id -ne $RunId) {
    $initialRootState = Get-QueueControlState -ControlUrl $ControlUrl
    if ($null -eq $initialRootState -or
        [string]$initialRootState.run_id -ne $RunId) {
        Start-Sleep -Seconds 5
    }
}
$startIndex = [Math]::Max(0, [int]$initialRootState.case_index)
if ($startIndex -ge $Cases.Count) {
    Write-State ALL COMPLETE
    New-Item -ItemType File -Force `
        -Path (Join-Path $StateDir "subserver_queue.complete") | Out-Null
    return
}

$completedInThisRun = $false
foreach ($index in $startIndex..($Cases.Count - 1)) {
    $caseName = $Cases[$index]
    $caseDir = Join-Path $RunRoot $caseName
    $caseRelative = "scripts\distributed_scripts\ggeur_hierarchical_3machine\runs\$RunId\$caseName"
    $lastCaseState = Get-Content -LiteralPath $StateLog `
        -ErrorAction SilentlyContinue |
        Where-Object { $_ -match "`t$([regex]::Escape($caseName))`t" } |
        Select-Object -Last 1
    $rootState = Wait-QueueControlState -ControlUrl $ControlUrl `
        -RunId $RunId -CaseName $caseName -CaseIndex $index `
        -DesiredPhases @("starting", "running", "complete") `
        -TimeoutSec $ControlWaitTimeoutSec
    if ([int]$rootState.completed_count -gt $index -or
        ([int]$rootState.case_index -eq $index -and
         [string]$rootState.phase -eq "complete")) {
        if ($lastCaseState -notmatch "`t$([regex]::Escape($caseName))`tCOMPLETE$") {
            Write-State $caseName "ROOT_CONFIRMED_COMPLETE"
        }
        continue
    }
    if ($lastCaseState -match "`t$([regex]::Escape($caseName))`tCOMPLETE$") {
        Write-State $caseName "REPLAY_STALE_LOCAL_COMPLETE"
    }
    $related = @(Get-CimInstance Win32_Process | Where-Object {
        $_.CommandLine -like "*$caseName*hierarchical_subserver.py*" -or
        ($_.CommandLine -like "*hierarchical_subserver.py*" -and $_.CommandLine -like "*$caseName*")
    })
    if ($related.Count -gt 0) {
        Write-State $caseName "ADOPT_SUBSERVERS:count=$($related.Count)"
    } else {
        Wait-Tcp "10.112.81.135" 60050 $true
        Write-State $caseName "START_SUBSERVERS"
        & (Join-Path $PSScriptRoot "launch_subservers.ps1") `
            -CaseDir $caseRelative -PythonBin $PythonBin -UseScheduledTasks
    }

    do {
        Start-Sleep -Seconds 10
        $related = @(Get-CimInstance Win32_Process | Where-Object {
            $_.CommandLine -like "*hierarchical_subserver.py*" -and
            $_.CommandLine -like "*$caseName*"
        })
    } while ($related.Count -gt 0)

    $caseTasks = @(Get-ScheduledTask `
        -TaskName "GGEUR-$caseName-subserver_*" `
        -ErrorAction SilentlyContinue)
    $failedTasks = @($caseTasks | Where-Object {
        (Get-ScheduledTaskInfo -TaskName $_.TaskName).LastTaskResult -ne 0
    })
    if ($failedTasks.Count -gt 0) {
        $failedNames = ($failedTasks.TaskName -join ",")
        Write-State $caseName "FAILED_SUBSERVERS:$failedNames"
        throw "Subserver tasks failed for ${caseName}: $failedNames"
    }
    $rootState = Wait-QueueControlState -ControlUrl $ControlUrl `
        -RunId $RunId -CaseName $caseName -CaseIndex $index `
        -DesiredPhases @("complete") -TimeoutSec $ControlWaitTimeoutSec
    $caseTasks | ForEach-Object {
            Unregister-ScheduledTask -TaskName $_.TaskName -Confirm:$false `
                -ErrorAction SilentlyContinue
        }
    Write-State $caseName COMPLETE
    $completedInThisRun = $true
}

Write-State ALL COMPLETE
New-Item -ItemType File -Force -Path (Join-Path $StateDir "subserver_queue.complete") | Out-Null
