param(
    [Parameter(Mandatory = $true)][ValidateSet("subservers", "clients")]
    [string]$Role,
    [Parameter(Mandatory = $true)][string]$RunId,
    [Parameter(Mandatory = $true)][string]$ConfigSet,
    [Parameter(Mandatory = $true)][string]$PythonBin,
    [string]$SitePackages = "",
    [string]$ControlUrl = "http://10.129.222.189:60049",
    [int]$ControlWaitTimeoutSec = 0,
    [switch]$AllowCudaClientRuntime
)

$ErrorActionPreference = "Stop"
$RepoDir = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$RunRoot = Join-Path $PSScriptRoot "runs\$RunId"
$StateDir = Join-Path $RunRoot "queue_state"
$ManifestPath = Join-Path $RunRoot "matrix_manifest.json"
if (-not (Test-Path -LiteralPath $ManifestPath)) {
    throw "Missing matrix manifest: $ManifestPath"
}
if (-not (Test-Path -LiteralPath $PythonBin)) {
    throw "Missing Python: $PythonBin"
}
New-Item -ItemType Directory -Force -Path $StateDir | Out-Null
$StateLog = Join-Path $StateDir "$($Role)_$($ConfigSet)_queue.tsv"
. (Join-Path $PSScriptRoot "queue_control_client.ps1")
$Cases = @(
    (Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json).cases |
        ForEach-Object { [string]$_.case }
)

function Write-State([string]$Case, [string]$State) {
    Add-Content -LiteralPath $StateLog `
        -Value "$(Get-Date -Format o)`t$Case`t$State"
}

function Test-Tcp([string]$HostName, [int]$Port) {
    $client = [Net.Sockets.TcpClient]::new()
    try {
        $task = $client.ConnectAsync($HostName, $Port)
        return $task.Wait(1500) -and $client.Connected
    } catch { return $false } finally { $client.Dispose() }
}

function Wait-Tcp([string]$HostName, [int]$Port) {
    while (-not (Test-Tcp $HostName $Port)) {
        Start-Sleep -Seconds 2
    }
}

foreach ($index in 0..($Cases.Count - 1)) {
    $caseName = $Cases[$index]
    $caseDir = Join-Path $RunRoot $caseName
    $configDir = Join-Path $caseDir "configs\$ConfigSet"
    $rootState = Wait-QueueControlState -ControlUrl $ControlUrl `
        -RunId $RunId -CaseName $caseName -CaseIndex $index `
        -DesiredPhases @("starting", "running", "complete") `
        -TimeoutSec $ControlWaitTimeoutSec
    if ([int]$rootState.completed_count -gt $index -or
        ([int]$rootState.case_index -eq $index -and
         [string]$rootState.phase -eq "complete")) {
        Write-State $caseName "ROOT_CONFIRMED_COMPLETE"
        continue
    }

    $pattern = if ($Role -eq "subservers") {
        "subserver_*.json"
    } else {
        "client_*.yaml"
    }
    $configs = if (Test-Path -LiteralPath $configDir) {
        @(Get-ChildItem -LiteralPath $configDir -Filter $pattern -File |
            Sort-Object Name)
    } else { @() }
    if ($configs.Count -eq 0) {
        Write-State $caseName "NO_LOCAL_ROLE"
    } else {
        if ($Role -eq "subservers") {
            Wait-Tcp "10.129.222.189" 60050
        } else {
            $endpoints = @{}
            foreach ($config in $configs) {
                $text = Get-Content -LiteralPath $config.FullName -Raw
                $hostMatch = [regex]::Match(
                    $text, '(?m)^\s*server_host:\s*[''\"]?([^''\"\r\n]+)')
                $portMatch = [regex]::Match(
                    $text, '(?m)^\s*server_port:\s*(\d+)')
                if (-not $hostMatch.Success -or -not $portMatch.Success) {
                    throw "Missing subserver endpoint in $($config.FullName)"
                }
                $hostName = $hostMatch.Groups[1].Value.Trim()
                $port = [int]$portMatch.Groups[1].Value
                $endpoints["${hostName}:$port"] = @($hostName, $port)
            }
            foreach ($endpoint in $endpoints.Values) {
                Wait-Tcp ([string]$endpoint[0]) ([int]$endpoint[1])
            }
        }

        $caseRelative = "scripts\distributed_scripts\ggeur_hierarchical_3machine\runs\$RunId\$caseName"
        Write-State $caseName "START:${Role}:${ConfigSet}:count=$($configs.Count)"
        try {
            if ($Role -eq "subservers") {
                $subserverArgs = @{
                    CaseDir = $caseRelative
                    PythonBin = $PythonBin
                    ConfigSet = $ConfigSet
                    UseScheduledTasks = $true
                    StartGapMilliseconds = 250
                }
                if ($SitePackages) {
                    $subserverArgs.SitePackages = $SitePackages
                }
                & (Join-Path $PSScriptRoot "launch_subservers.ps1") `
                    @subserverArgs |
                    Add-Content -LiteralPath (Join-Path $StateDir "$($Role)_launch.log")
            } else {
                $launchArgs = @{
                    CaseDir = $caseRelative
                    PythonBin = $PythonBin
                    ConfigSet = $ConfigSet
                    UseScheduledTasks = $true
                    # A validated 60-client cache is shared by two physical
                    # hosts in a 30/30 DomainNet placement.  Each launcher
                    # therefore consumes an intentional subset of the same
                    # complete cache contract.
                    AllowConfigSubset = $true
                    StartGapMilliseconds = 150
                    LaunchBatchSize = 8
                }
                if ($SitePackages) { $launchArgs.ClientSitePackages = $SitePackages }
                if ($AllowCudaClientRuntime) {
                    $launchArgs.AllowCudaClientRuntime = $true
                }
                & (Join-Path $PSScriptRoot "launch_clients.ps1") @launchArgs |
                    Add-Content -LiteralPath (Join-Path $StateDir "$($Role)_launch.log")
            }
        } catch {
            Write-State $caseName "FAILED_LAUNCH:$($_.Exception.Message)"
            throw
        }

        $needle = if ($Role -eq "subservers") {
            "hierarchical_subserver.py"
        } else { "federatedscope.main" }
        do {
            Start-Sleep -Seconds 10
            $related = @(Get-CimInstance Win32_Process | Where-Object {
                $_.CommandLine -like "*$needle*" -and
                $_.CommandLine -like "*$caseName*" -and
                $_.CommandLine -like "*$ConfigSet*"
            })
        } while ($related.Count -gt 0)

        $taskSuffix = if ($Role -eq "subservers") {
            "subserver_*"
        } else { "client_*" }
        $tasks = @(Get-ScheduledTask `
            -TaskName "GGEUR-$RunId-$caseName-$taskSuffix" `
            -ErrorAction SilentlyContinue)
        $failed = @($tasks | Where-Object {
            (Get-ScheduledTaskInfo -TaskName $_.TaskName).LastTaskResult -ne 0
        })
        if ($failed.Count -gt 0) {
            $summary = ($failed | ForEach-Object {
                "$($_.TaskName):$((Get-ScheduledTaskInfo -TaskName $_.TaskName).LastTaskResult)"
            }) -join ','
            Write-State $caseName "FAILED:$summary"
            throw "$caseName has failed $Role tasks: $summary"
        }
    }

    Wait-QueueControlState -ControlUrl $ControlUrl -RunId $RunId `
        -CaseName $caseName -CaseIndex $index -DesiredPhases @("complete") `
        -TimeoutSec $ControlWaitTimeoutSec | Out-Null
    if ($configs.Count -gt 0) {
        $taskSuffix = if ($Role -eq "subservers") {
            "subserver_*"
        } else { "client_*" }
        Get-ScheduledTask -TaskName "GGEUR-$RunId-$caseName-$taskSuffix" `
            -ErrorAction SilentlyContinue | ForEach-Object {
                Unregister-ScheduledTask -TaskName $_.TaskName `
                    -Confirm:$false -ErrorAction SilentlyContinue
            }
    }
    Write-State $caseName "COMPLETE"
}

$marker = Join-Path $StateDir "$($Role)_$($ConfigSet)_queue.complete"
New-Item -ItemType File -Force -Path $marker | Out-Null
Write-State "ALL" "COMPLETE"
