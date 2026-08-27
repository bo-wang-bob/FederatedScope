param(
    [Parameter(Mandatory = $true)]
    [string]$CaseDir
)

$ErrorActionPreference = "Stop"
$RepoDir = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$ResolvedCase = (Resolve-Path (Join-Path $RepoDir $CaseDir)).Path
$PidDir = Join-Path $ResolvedCase "pids"
$running = 0
$stopped = 0

Get-ChildItem -LiteralPath $PidDir -Filter "client_*.pid" -ErrorAction SilentlyContinue |
    Sort-Object Name |
    ForEach-Object {
        $clientPid = [int](Get-Content $_.FullName -Raw)
        $process = Get-Process -Id $clientPid -ErrorAction SilentlyContinue
        $taskPath = Join-Path $PidDir "$($_.BaseName).task"
        $taskState = "none"
        if (Test-Path -LiteralPath $taskPath) {
            $taskName = (Get-Content -LiteralPath $taskPath -Raw).Trim()
            $task = Get-ScheduledTask -TaskName $taskName `
                -ErrorAction SilentlyContinue
            $taskState = if ($task) { [string]$task.State } else { "missing" }
        }
        $commandLine = if ($process) {
            (Get-CimInstance Win32_Process `
                -Filter "ProcessId = $clientPid").CommandLine
        } else { "" }
        if ($process -and $commandLine.Contains($ResolvedCase)) {
            $running += 1
            Write-Output "$($_.BaseName) pid=$clientPid state=running task_state=$taskState"
        } elseif ($process) {
            $stopped += 1
            Write-Output "$($_.BaseName) pid=$clientPid state=pid_reused task_state=$taskState"
        } else {
            $stopped += 1
            Write-Output "$($_.BaseName) pid=$clientPid state=stopped task_state=$taskState"
        }
    }
Write-Output "running=$running stopped=$stopped total=$($running + $stopped)"
