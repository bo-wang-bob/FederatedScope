param(
    [string]$RunId = "final_remaining_20260723_v1",
    [string]$ControlUrl = "http://10.112.81.135:60049",
    [int]$ControlWaitTimeoutSec = 172800
)

$ErrorActionPreference = "Stop"
$RepoDir = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$RunRoot = Join-Path $PSScriptRoot "runs\$RunId"
$StateDir = Join-Path $RunRoot "queue_state"
$StateLog = Join-Path $StateDir "client_queue.tsv"
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
        -Path (Join-Path $StateDir "client_queue.complete") | Out-Null
    return
}

$completedInThisRun = $false
foreach ($index in $startIndex..($Cases.Count - 1)) {
    $caseName = $Cases[$index]
    $caseDir = Join-Path $RunRoot $caseName
    $caseRelative = "scripts\distributed_scripts\ggeur_hierarchical_3machine\runs\$RunId\$caseName"
    $lastCaseState = Get-Content -LiteralPath $StateLog `
        -ErrorAction SilentlyContinue |
        Where-Object { $_ -like "*`t$caseName`t*" } |
        Select-Object -Last 1
    $rootState = Wait-QueueControlState -ControlUrl $ControlUrl `
        -RunId $RunId -CaseName $caseName -CaseIndex $index `
        -DesiredPhases @("starting", "running", "complete") `
        -TimeoutSec $ControlWaitTimeoutSec
    if ([int]$rootState.completed_count -gt $index -or
        ([int]$rootState.case_index -eq $index -and
         [string]$rootState.phase -eq "complete")) {
        if ($lastCaseState -notmatch "`tCOMPLETE$") {
            Write-State $caseName "ROOT_CONFIRMED_COMPLETE"
        }
        continue
    }
    if ($lastCaseState -match "`tCOMPLETE$") {
        Write-State $caseName "REPLAY_STALE_LOCAL_COMPLETE"
    }
    $caseManifest = Join-Path $caseDir 'manifest.json'
    if (-not (Test-Path -LiteralPath $caseManifest)) {
        throw "Missing case manifest: $caseManifest"
    }
    $manifestContent = Get-Content -LiteralPath $caseManifest -Raw |
        ConvertFrom-Json
    $group = [string]$manifestContent.group
    if ([string]::IsNullOrWhiteSpace($group)) {
        throw "Case manifest does not define group: $caseManifest"
    }
    $marker = Join-Path $RepoDir "exp\distributed_feature_cache\$group\.ggeur_feature_cache_ready.json"
    while (-not (Test-Path -LiteralPath $marker)) {
        Write-State $caseName "WAIT_CACHE:$group"
        Start-Sleep -Seconds 60
    }
    $status = Get-Content -LiteralPath $marker -Raw | ConvertFrom-Json
    while ($status.validated_on -ne $env:COMPUTERNAME -or
           -not $status.require_complete_feature_cache) {
        Write-State $caseName "WAIT_VALIDATED_CACHE:$group"
        Start-Sleep -Seconds 60
        $status = Get-Content -LiteralPath $marker -Raw | ConvertFrom-Json
    }

    $related = @(Get-CimInstance Win32_Process | Where-Object {
        $_.CommandLine -like "*federatedscope.main*" -and
        $_.CommandLine -like "*$caseName*"
    })
    if ($related.Count -gt 0) {
        Write-State $caseName "ADOPT_CLIENTS:count=$($related.Count)"
    } else {
        # Do not start terminal clients until every configured subserver is
        # listening.  Waiting for only port 61000 lets clients assigned to a
        # later subserver exit during slow Windows task scheduling.
        foreach ($subserverPort in 61000..61003) {
            Wait-Tcp "10.129.248.111" $subserverPort $true
        }
        Write-State $caseName START_CLIENTS
        $launchArgs = @{
            CaseDir = $caseRelative
            UseScheduledTasks = $true
            AllowConfigSubset = $true
            # Start clients in bounded parallel batches.  Importing PyTorch is
            # the dominant startup cost, so waiting for every client before
            # launching the next one unnecessarily adds several minutes.
            LaunchBatchSize = 12
            StartGapMilliseconds = 75
        }
        # A validated full feature cache may intentionally serve a smaller
        # balanced client view.  The manifest determines the active clients;
        # the cache marker can therefore record more source clients.
        $launcher = Join-Path $PSScriptRoot "launch_clients.ps1"
        try {
            & $launcher @launchArgs
        } catch {
            Write-State $caseName "CLIENT_LAUNCH_RETRY:$($_.Exception.Message)"
            $caseToken = [regex]::Replace($caseName, '[^A-Za-z0-9_.-]', '_')
            $runToken = [regex]::Replace($RunId, '[^A-Za-z0-9_.-]', '_')
            $retryTasks = @(Get-ScheduledTask `
                -TaskName "GGEUR-$runToken-$caseToken-client_*" `
                -ErrorAction SilentlyContinue)
            foreach ($task in $retryTasks) {
                Stop-ScheduledTask -TaskName $task.TaskName `
                    -ErrorAction SilentlyContinue
                Unregister-ScheduledTask -TaskName $task.TaskName `
                    -Confirm:$false -ErrorAction SilentlyContinue
            }
            Start-Sleep -Seconds 20
            & $launcher @launchArgs
        }
    }

    do {
        Start-Sleep -Seconds 10
        $related = @(Get-CimInstance Win32_Process | Where-Object {
            $_.CommandLine -like "*federatedscope.main*" -and
            $_.CommandLine -like "*$caseName*"
        })
    } while ($related.Count -gt 0)

    $rootState = Wait-QueueControlState -ControlUrl $ControlUrl `
        -RunId $RunId -CaseName $caseName -CaseIndex $index `
        -DesiredPhases @("complete") -TimeoutSec $ControlWaitTimeoutSec
    # The root completion contract already proves that every configured
    # client returned its update and the result files were produced.  Querying
    # and deleting one Task Scheduler entry at a time can add several minutes
    # between otherwise short cases, so keep that maintenance off the
    # training queue's critical path.
    Write-State $caseName COMPLETE
    $completedInThisRun = $true
}

Write-State ALL COMPLETE
New-Item -ItemType File -Force -Path (Join-Path $StateDir "client_queue.complete") | Out-Null
