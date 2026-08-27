param(
    [Parameter(Mandatory = $true)]
    [string]$CaseDir
)

$ErrorActionPreference = "Stop"
$RepoDir = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$ResolvedCase = (Resolve-Path (Join-Path $RepoDir $CaseDir)).Path
$PidDir = Join-Path $ResolvedCase "pids"
$LogDir = Join-Path $ResolvedCase "logs"

$pidFiles = Get-ChildItem -LiteralPath $PidDir `
    -Filter "subserver_*.pid" -File -ErrorAction SilentlyContinue |
    Sort-Object Name
foreach ($pidFile in $pidFiles) {
    $name = [IO.Path]::GetFileNameWithoutExtension($pidFile.Name)
    $pidValue = [int](Get-Content -LiteralPath $pidFile.FullName -Raw)
    $process = Get-CimInstance Win32_Process `
        -Filter "ProcessId=$pidValue" -ErrorAction SilentlyContinue
    $taskPath = Join-Path $PidDir "$name.task"
    $taskState = "none"
    if (Test-Path -LiteralPath $taskPath) {
        $taskName = (Get-Content -LiteralPath $taskPath -Raw).Trim()
        $task = Get-ScheduledTask -TaskName $taskName `
            -ErrorAction SilentlyContinue
        $taskState = if ($task) { [string]$task.State } else { "missing" }
    }
    if ($process -and
            $process.CommandLine -like "*hierarchical_subserver.py*" -and
            $process.CommandLine -like "*$ResolvedCase*") {
        $state = "running"
    } elseif ($process) {
        $state = "pid_reused"
    } else {
        $state = "stopped"
    }
    Write-Output "$name pid=$pidValue state=$state task_state=$taskState"
    $stderr = Join-Path $LogDir "$name.stderr.log"
    if (Test-Path -LiteralPath $stderr) {
        Get-Content -LiteralPath $stderr -Tail 5
    }
}
