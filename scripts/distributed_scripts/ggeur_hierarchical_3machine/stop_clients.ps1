param(
    [Parameter(Mandatory = $true)]
    [string]$CaseDir,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$RepoDir = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$ResolvedCase = (Resolve-Path (Join-Path $RepoDir $CaseDir)).Path
$PidDir = Join-Path $ResolvedCase "pids"

Get-ChildItem -LiteralPath $PidDir -Filter "client_*.pid" -ErrorAction SilentlyContinue |
    ForEach-Object {
        $clientPid = [int](Get-Content $_.FullName -Raw)
        $taskPath = Join-Path $PidDir "$($_.BaseName).task"
        $taskName = $null
        if (Test-Path -LiteralPath $taskPath) {
            $taskName = (Get-Content -LiteralPath $taskPath -Raw).Trim()
            Stop-ScheduledTask -TaskName $taskName `
                -ErrorAction SilentlyContinue
        }
        $process = Get-Process -Id $clientPid -ErrorAction SilentlyContinue
        if ($process) {
            $commandLine = (Get-CimInstance Win32_Process `
                -Filter "ProcessId = $clientPid").CommandLine
            if (-not $commandLine -or `
                    -not $commandLine.Contains($ResolvedCase)) {
                Write-Warning "Refuse to stop reused/unrelated PID $clientPid"
                return
            }
            Stop-Process -Id $clientPid
            try {
                Wait-Process -Id $clientPid -Timeout 30 -ErrorAction Stop
            } catch {
                if ($Force) {
                    Stop-Process -Id $clientPid -Force
                    Wait-Process -Id $clientPid -Timeout 10 `
                        -ErrorAction SilentlyContinue
                } else {
                    Write-Warning "PID $clientPid did not stop; rerun with -Force"
                    return
                }
            }
            Write-Output "stopped $($_.BaseName) pid=$clientPid"
        }
        Remove-Item -LiteralPath $_.FullName -Force
        if ($taskName) {
            Unregister-ScheduledTask -TaskName $taskName `
                -Confirm:$false -ErrorAction SilentlyContinue
            Remove-Item -LiteralPath $taskPath -Force `
                -ErrorAction SilentlyContinue
        }
    }
