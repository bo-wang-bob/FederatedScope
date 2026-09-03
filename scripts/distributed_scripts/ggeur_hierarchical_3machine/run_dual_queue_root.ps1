param(
    [Parameter(Mandatory = $true)][string]$RunId,
    [string]$PythonBin = "",
    [string]$SitePackages = "",
    [int]$ControlPort = 60049
)

$ErrorActionPreference = "Stop"
$RepoDir = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$RunRoot = Join-Path $PSScriptRoot "runs\$RunId"
$StateDir = Join-Path $RunRoot "queue_state"
$ManifestPath = Join-Path $RunRoot "matrix_manifest.json"
if (-not (Test-Path -LiteralPath $ManifestPath)) {
    throw "Missing matrix manifest: $ManifestPath"
}
if (-not $SitePackages) {
    $SitePackages = Join-Path $RepoDir ".venv_client_cpu\Lib\site-packages"
}
if (-not $PythonBin) {
    $venvConfig = Join-Path $RepoDir ".venv_client_cpu\pyvenv.cfg"
    $homeLine = Get-Content -LiteralPath $venvConfig |
        Where-Object { $_ -match '^\s*home\s*=\s*(.+)\s*$' } |
        Select-Object -First 1
    $pythonHome = ([regex]::Match(
        $homeLine, '^\s*home\s*=\s*(.+)\s*$')).Groups[1].Value.Trim()
    $PythonBin = Join-Path $pythonHome "python.exe"
}
foreach ($path in @($PythonBin, $SitePackages)) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Missing: $path" }
}
$env:PYTHONPATH = $SitePackages
$env:FEDERATEDSCOPE_GGEUR_LIGHTWEIGHT = "1"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
$env:CUDA_VISIBLE_DEVICES = ""
New-Item -ItemType Directory -Force -Path $StateDir | Out-Null
$StateLog = Join-Path $StateDir "root_queue.tsv"
$ControlState = Join-Path $StateDir "control.json"
$ControlLog = Join-Path $StateDir "control_server.log"
$ControlErr = Join-Path $StateDir "control_server.stderr.log"
$ControlPidFile = Join-Path $StateDir "control_server.pid"
$QueueControl = Join-Path $PSScriptRoot "queue_control.py"
$Cases = @(
    (Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json).cases |
        ForEach-Object { [string]$_.case }
)
if ($Cases.Count -eq 0) { throw "No cases in $ManifestPath" }

function Write-State([string]$Case, [string]$State) {
    Add-Content -LiteralPath $StateLog `
        -Value "$(Get-Date -Format o)`t$Case`t$State"
}

function Consolidate-RootLog([string]$CaseDir) {
    $stdout = Join-Path $CaseDir "logs\root.stdout.log"
    $stderr = Join-Path $CaseDir "logs\root.stderr.log"
    if (-not (Test-Path -LiteralPath $stderr)) { return $stdout }
    if ((Get-Item -LiteralPath $stderr).Length -eq 0) { return $stdout }
    if ((-not (Test-Path -LiteralPath $stdout)) -or
            (Get-Item -LiteralPath $stdout).Length -eq 0) {
        Copy-Item -LiteralPath $stderr -Destination $stdout -Force
        return $stdout
    }
    $stderrHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $stderr).Hash
    $mergeMarker = Join-Path $CaseDir "logs\.stderr_merged_sha256"
    $previous = Get-Content -LiteralPath $mergeMarker -Raw `
        -ErrorAction SilentlyContinue
    if ($previous -and $previous.Trim() -eq $stderrHash) { return $stdout }
    Add-Content -LiteralPath $stdout -Value "`n===== STDERR ====="
    Get-Content -LiteralPath $stderr -ReadCount 4096 |
        Add-Content -LiteralPath $stdout
    Set-Content -LiteralPath $mergeMarker -Value $stderrHash
    return $stdout
}

function Test-LocalPort([int]$Port) {
    return $null -ne (Get-NetTCPConnection -State Listen -LocalPort $Port `
        -ErrorAction SilentlyContinue | Select-Object -First 1)
}

function Publish-State(
    [string]$Case, [int]$Index, [string]$Phase,
    [int]$Completed, [string]$Message, [int]$AccuracyRounds = 0
) {
    & $PythonBin $QueueControl write --state-file $ControlState `
        --run-id $RunId --case $Case --case-index $Index `
        --total-cases $Cases.Count --completed-count $Completed `
        --phase $Phase --accuracy-rounds $AccuracyRounds `
        --root-port 60050 --message $Message | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Failed to publish queue state" }
}

function Start-ControlServer {
    if (Test-Path -LiteralPath $ControlPidFile) {
        $oldPid = [int](Get-Content -LiteralPath $ControlPidFile -Raw)
        if (Get-Process -Id $oldPid -ErrorAction SilentlyContinue) { return }
    }
    if (Test-LocalPort $ControlPort) {
        throw "Control port $ControlPort is used by an unowned process"
    }
    $process = Start-Process -FilePath $PythonBin -ArgumentList @(
        $QueueControl, "serve", "--state-file", $ControlState,
        "--host", "0.0.0.0", "--port", $ControlPort
    ) -WorkingDirectory $RepoDir -WindowStyle Hidden `
      -RedirectStandardOutput $ControlLog `
      -RedirectStandardError $ControlErr -PassThru
    Set-Content -LiteralPath $ControlPidFile -Value $process.Id
    foreach ($attempt in 1..120) {
        if (-not (Get-Process -Id $process.Id -ErrorAction SilentlyContinue)) {
            throw "Queue control server exited; see $ControlErr"
        }
        if (Test-LocalPort $ControlPort) { return }
        Start-Sleep -Milliseconds 250
    }
    throw "Queue control server did not listen on $ControlPort"
}

function Stop-ControlServer {
    if (-not (Test-Path -LiteralPath $ControlPidFile)) { return }
    $controlPid = [int](Get-Content -LiteralPath $ControlPidFile -Raw)
    $process = Get-CimInstance Win32_Process -Filter "ProcessId=$controlPid" `
        -ErrorAction SilentlyContinue
    if ($process -and $process.CommandLine -and
            $process.CommandLine.Contains($ControlState)) {
        Stop-Process -Id $controlPid -Force -ErrorAction SilentlyContinue
    }
    Remove-Item -LiteralPath $ControlPidFile -Force `
        -ErrorAction SilentlyContinue
}

function Archive-Incomplete([string]$CaseDir, [string]$CaseName) {
    $log = Consolidate-RootLog $CaseDir
    if (-not (Test-Path -LiteralPath $log)) { return }
    if ((Get-Item -LiteralPath $log).Length -eq 0) { return }
    if (Test-Path -LiteralPath (Join-Path $CaseDir ".formal_complete")) {
        return
    }
    $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $log).Hash
    $lastPath = Join-Path $CaseDir "attempts\.last_archived_sha256"
    $previous = Get-Content -LiteralPath $lastPath -Raw `
        -ErrorAction SilentlyContinue
    if ($previous -and $previous.Trim() -eq $hash) { return }
    $attempt = Join-Path $CaseDir (
        "attempts\attempt_" + (Get-Date -Format "yyyyMMdd_HHmmss"))
    New-Item -ItemType Directory -Force -Path $attempt | Out-Null
    Copy-Item -LiteralPath $log -Destination (
        Join-Path $attempt "root.stdout.log")
    $stderr = Join-Path $CaseDir "logs\root.stderr.log"
    if (Test-Path -LiteralPath $stderr) {
        Copy-Item -LiteralPath $stderr -Destination (
            Join-Path $attempt "root.stderr.log")
    }
    Copy-Item -LiteralPath (Join-Path $CaseDir "configs\root_server.yaml") `
        -Destination (Join-Path $attempt "root_server.yaml")
    @(
        "status=INCOMPLETE",
        "archived_at=$(Get-Date -Format o)",
        "root_log_sha256=$hash"
    ) | Set-Content -LiteralPath (Join-Path $attempt "attempt_status.txt")
    Set-Content -LiteralPath $lastPath -Value $hash
    Write-State $CaseName "ARCHIVE_INCOMPLETE:${attempt}:$hash"
}

Start-ControlServer
$completed = @($Cases | Where-Object {
    Test-Path -LiteralPath (Join-Path $RunRoot "$_\.formal_complete")
}).Count

try {
    for ($index = 0; $index -lt $Cases.Count; $index++) {
        $caseName = $Cases[$index]
        $caseDir = Join-Path $RunRoot $caseName
        $completeMarker = Join-Path $caseDir ".formal_complete"
        if (Test-Path -LiteralPath $completeMarker) {
            Write-State $caseName "SKIP_COMPLETE"
            continue
        }
        Publish-State $caseName $index "starting" $completed `
            "dual-host roles may start"
        $rootPidFile = Join-Path $caseDir "pids\root.pid"
        $rootPid = 0
        if (Test-Path -LiteralPath $rootPidFile) {
            $candidatePid = [int](Get-Content -LiteralPath $rootPidFile -Raw)
            if (Get-Process -Id $candidatePid -ErrorAction SilentlyContinue) {
                $rootPid = $candidatePid
                Write-State $caseName "ADOPT_ROOT:pid=$rootPid"
            }
        }
        if ($rootPid -eq 0) {
            Archive-Incomplete $caseDir $caseName
            Write-State $caseName "START_ROOT"
            $caseRelative = "scripts\distributed_scripts\ggeur_hierarchical_3machine\runs\$RunId\$caseName"
            & (Join-Path $PSScriptRoot "launch_root_windows.ps1") `
                -CaseDir $caseRelative -PythonBin $PythonBin `
                -SitePackages $SitePackages |
                Add-Content -LiteralPath (Join-Path $StateDir "root_launch.log")
            $rootPid = [int](Get-Content -LiteralPath $rootPidFile -Raw)
        }
        Publish-State $caseName $index "running" $completed "training"
        $lastReport = Get-Date
        while (Get-Process -Id $rootPid -ErrorAction SilentlyContinue) {
            if (((Get-Date) - $lastReport).TotalSeconds -ge 300) {
                $rounds = 0
                foreach ($name in @("root.stdout.log", "root.stderr.log")) {
                    $log = Join-Path $caseDir "logs\$name"
                    $rounds += @(Select-String -LiteralPath $log `
                        -Pattern 'Round [0-9]+ MLP Test Accuracy' `
                        -ErrorAction SilentlyContinue).Count
                }
                Write-State $caseName "RUNNING:pid=${rootPid}:accuracy_rounds=$rounds"
                Publish-State $caseName $index "running" $completed `
                    "training" $rounds
                $lastReport = Get-Date
            }
            Start-Sleep -Seconds 10
        }
        $rootLog = Consolidate-RootLog $caseDir
        $expected = if ($caseName -like "mdsent_*") { 4 } else { 2 }
        $rootConfigText = Get-Content -LiteralPath (
            Join-Path $caseDir "configs\root_server.yaml") -Raw
        $roundMatch = [regex]::Match(
            $rootConfigText, '(?m)^\s*total_round_num:\s*(\d+)')
        if (-not $roundMatch.Success) {
            throw "total_round_num is missing for $caseName"
        }
        $expectedRounds = [int]$roundMatch.Groups[1].Value
        $expectedAccuracyRounds = [Math]::Max(0, $expectedRounds - 1)
        $evalMatch = [regex]::Match(
            $rootConfigText, '(?ms)^eval:\s*.*?^\s+freq:\s*(\d+)')
        $evalFrequency = if ($evalMatch.Success) {
            [int]$evalMatch.Groups[1].Value
        } else { 1 }
        $evalModeMatch = [regex]::Match(
            $rootConfigText, '(?m)^\s*headonly_eval_mode:\s*[''\"]?([^''\"\r\n]+)')
        $evalMode = if ($evalModeMatch.Success) {
            $evalModeMatch.Groups[1].Value.Trim().ToLowerInvariant()
        } else { 'server' }
        $terminalClientEvalOnly = [regex]::IsMatch(
            $rootConfigText,
            '(?m)^\s*terminal_client_eval_only:\s*true\s*$')
        $accuracyRecordsPerEval = if (
            $evalMode -eq 'both' -and -not $terminalClientEvalOnly
        ) { 2 } else { 1 }
        $validationArgs = @(
            $rootLog,
            '--output', (Join-Path $caseDir 'completion_validation.json'),
            '--expected-rounds', $expectedRounds,
            '--expected-accuracy-rounds', $expectedAccuracyRounds,
            '--eval-frequency', $evalFrequency,
            '--accuracy-records-per-eval', $accuracyRecordsPerEval,
            '--expected-updates', $expected
        )
        if ($terminalClientEvalOnly) {
            $validationArgs += '--terminal-client-eval-only'
        }
        if ($evalMode -in @('client', 'both')) {
            $clientNumMatch = [regex]::Match(
                $rootConfigText, '(?m)^\s*client_num:\s*(\d+)')
            if (-not $clientNumMatch.Success) {
                throw "client_num is missing for $caseName"
            }
            $terminalSearchRoot = Join-Path $RepoDir (
                "exp\hierarchical\$RunId\$caseName")
            $terminalSource = Get-ChildItem -LiteralPath $terminalSearchRoot `
                -Recurse -File `
                -Filter "client_model_accuracy_round_$expectedAccuracyRounds.json" `
                -ErrorAction SilentlyContinue |
                Sort-Object LastWriteTime -Descending |
                Select-Object -First 1 -ExpandProperty FullName
            $terminalEvidence = Join-Path $caseDir (
                "client_model_accuracy_round_$expectedAccuracyRounds.json")
            if ($terminalSource -and `
                    (Test-Path -LiteralPath $terminalSource)) {
                Copy-Item -LiteralPath $terminalSource `
                    -Destination $terminalEvidence -Force
            }
            $validationArgs += @(
                '--client-evidence', $terminalEvidence,
                '--expected-clients', [int]$clientNumMatch.Groups[1].Value
            )
        }
        & $PythonBin (Join-Path $PSScriptRoot "validate_case_completion.py") `
            @validationArgs |
            Add-Content -LiteralPath (Join-Path $StateDir "validation.log")
        if ($LASTEXITCODE -ne 0) {
            Write-State $caseName "FAILED_VALIDATION"
            Publish-State $caseName $index "failed" $completed `
                "completion validation failed"
            throw "$caseName failed completion validation"
        }
        & $PythonBin (Join-Path $PSScriptRoot "summarize_accuracy.py") `
            $rootLog --output (Join-Path $caseDir "accuracy_summary.json") |
            Add-Content -LiteralPath (Join-Path $StateDir "summarize.log")
        if ($LASTEXITCODE -ne 0) { throw "$caseName summary failed" }
        [ordered]@{
            run_id = $RunId
            case = $caseName
            completed_at = (Get-Date -Format o)
            validation = "completion_validation.json"
            summary = "accuracy_summary.json"
        } | ConvertTo-Json -Compress | Set-Content -LiteralPath $completeMarker
        $completed++
        Write-State $caseName "COMPLETE"
        Publish-State $caseName $index "complete" $completed `
            "completion contract passed" $expectedAccuracyRounds
        while (Test-LocalPort 60050) { Start-Sleep -Seconds 2 }
        Start-Sleep -Seconds 10
    }
    New-Item -ItemType File -Force -Path (
        Join-Path $StateDir "root_queue.complete") | Out-Null
    Write-State "ALL" "COMPLETE"
    Publish-State "ALL" $Cases.Count "all_complete" $Cases.Count `
        "all formal cases completed"
} catch {
    Write-State "QUEUE" "FAILED:$($_.Exception.Message)"
    throw
} finally {
    Stop-ControlServer
}
