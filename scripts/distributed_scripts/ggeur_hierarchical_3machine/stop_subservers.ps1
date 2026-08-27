param(
    [Parameter(Mandatory = $true)]
    [string]$CaseDir,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$RepoDir = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$ResolvedCase = (Resolve-Path (Join-Path $RepoDir $CaseDir)).Path
$PidDir = Join-Path $ResolvedCase "pids"
$pidFiles = Get-ChildItem -LiteralPath $PidDir `
    -Filter "subserver_*.pid" -File -ErrorAction SilentlyContinue |
    Sort-Object Name

foreach ($pidFile in $pidFiles) {
    $name = [IO.Path]::GetFileNameWithoutExtension($pidFile.Name)
    $pidValue = [int](Get-Content -LiteralPath $pidFile.FullName -Raw)
    $taskPath = Join-Path $PidDir "$name.task"
    $taskName = $null
    if (Test-Path -LiteralPath $taskPath) {
        $taskName = (Get-Content -LiteralPath $taskPath -Raw).Trim()
        Stop-ScheduledTask -TaskName $taskName `
            -ErrorAction SilentlyContinue
    }
    $process = Get-CimInstance Win32_Process `
        -Filter "ProcessId=$pidValue" -ErrorAction SilentlyContinue
    if ($process) {
        if ($process.CommandLine -notlike "*hierarchical_subserver.py*" -or
                $process.CommandLine -notlike "*$ResolvedCase*") {
            Write-Warning "Refusing unrelated/reused PID $pidValue"
            continue
        }
        Stop-Process -Id $pidValue -Force:$Force
        foreach ($attempt in 1..60) {
            if (-not (Get-Process -Id $pidValue `
                    -ErrorAction SilentlyContinue)) {
                break
            }
            Start-Sleep -Milliseconds 250
        }
        if (Get-Process -Id $pidValue -ErrorAction SilentlyContinue) {
            throw "$name PID $pidValue did not stop; rerun with -Force"
        }
        Write-Output "stopped $name pid=$pidValue"
    }
    Remove-Item -LiteralPath $pidFile.FullName -Force
    if ($taskName) {
        Unregister-ScheduledTask -TaskName $taskName `
            -Confirm:$false -ErrorAction SilentlyContinue
        Remove-Item -LiteralPath $taskPath -Force `
            -ErrorAction SilentlyContinue
    }
}
