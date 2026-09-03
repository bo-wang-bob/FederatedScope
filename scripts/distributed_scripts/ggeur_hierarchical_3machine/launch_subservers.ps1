param(
    [Parameter(Mandatory = $true)]
    [string]$CaseDir,
    [string]$PythonBin = "python",
    [string]$ConfigSet = "",
    [string]$SitePackages = "",
    [int]$StartGapMilliseconds = 500,
    [switch]$UseScheduledTasks
)

$ErrorActionPreference = "Stop"
$RepoDir = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$ScriptPath = Join-Path $PSScriptRoot "hierarchical_subserver.py"
$ResolvedCase = (Resolve-Path (Join-Path $RepoDir $CaseDir)).Path
$ConfigDir = if ($ConfigSet) {
    Join-Path $ResolvedCase "configs\$ConfigSet"
} else {
    Join-Path $ResolvedCase "configs"
}
$LogDir = Join-Path $ResolvedCase "logs"
$PidDir = Join-Path $ResolvedCase "pids"

if (-not (Test-Path -LiteralPath $PythonBin)) {
    $resolvedPython = Get-Command $PythonBin -ErrorAction SilentlyContinue
    if (-not $resolvedPython) {
        throw "Subserver Python does not exist: $PythonBin"
    }
    $PythonBin = $resolvedPython.Source
}
if (-not (Test-Path -LiteralPath $ScriptPath)) {
    throw "Subserver script does not exist: $ScriptPath"
}
if ($SitePackages) {
    if (-not (Test-Path -LiteralPath $SitePackages)) {
        throw "Subserver site-packages does not exist: $SitePackages"
    }
    $env:PYTHONPATH = $SitePackages
}

New-Item -ItemType Directory -Force -Path $LogDir, $PidDir | Out-Null
$configs = Get-ChildItem -LiteralPath $ConfigDir `
    -Filter "subserver_*.json" -File | Sort-Object Name
if (-not $configs) {
    throw "No subserver configs found under $ConfigDir"
}

$env:FEDERATEDSCOPE_GGEUR_LIGHTWEIGHT = "1"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:CUDA_VISIBLE_DEVICES = ""

function Get-CompatibleSubserverProcess(
        [System.IO.FileInfo]$Config, [int]$Port) {
    $listeners = @(Get-NetTCPConnection -State Listen -LocalPort $Port `
        -ErrorAction SilentlyContinue)
    foreach ($listener in $listeners) {
        $candidate = Get-CimInstance Win32_Process `
            -Filter "ProcessId = $([int]$listener.OwningProcess)" `
            -ErrorAction SilentlyContinue
        if (-not $candidate -or -not $candidate.CommandLine) {
            continue
        }
        $commandLine = [string]$candidate.CommandLine
        if ($commandLine.IndexOf(
                $Config.FullName,
                [StringComparison]::OrdinalIgnoreCase) -ge 0 -and
            $commandLine.IndexOf(
                "hierarchical_subserver.py",
                [StringComparison]::OrdinalIgnoreCase) -ge 0) {
            return $candidate
        }
    }
    return $null
}

foreach ($config in $configs) {
    $name = [IO.Path]::GetFileNameWithoutExtension($config.Name)
    $pidPath = Join-Path $PidDir "$name.pid"
    if (Test-Path -LiteralPath $pidPath) {
        $oldPid = [int](Get-Content -LiteralPath $pidPath -Raw)
        $oldProcess = Get-CimInstance Win32_Process `
            -Filter "ProcessId = $oldPid" -ErrorAction SilentlyContinue
        if ($oldProcess -and $oldProcess.CommandLine -and
            ([string]$oldProcess.CommandLine).IndexOf(
                $config.FullName,
                [StringComparison]::OrdinalIgnoreCase) -lt 0) {
            throw "$name PID file points to an unrelated live process: $oldPid"
        }
    }
    $settings = Get-Content -LiteralPath $config.FullName -Raw |
        ConvertFrom-Json
    $port = [int]$settings.listen_port
    if ((Get-NetTCPConnection -State Listen -LocalPort $port `
            -ErrorAction SilentlyContinue) -and
        -not (Get-CompatibleSubserverProcess -Config $config -Port $port)) {
        throw "$name port $port is already in use"
    }
}

foreach ($config in $configs) {
    $name = [IO.Path]::GetFileNameWithoutExtension($config.Name)
    $settings = Get-Content -LiteralPath $config.FullName -Raw |
        ConvertFrom-Json
    $port = [int]$settings.listen_port
    $stdout = Join-Path $LogDir "$name.stdout.log"
    $stderr = Join-Path $LogDir "$name.stderr.log"
    $structuredLog = Join-Path $LogDir "$name.log"
    $process = Get-CompatibleSubserverProcess -Config $config -Port $port
    $taskName = $null
    if ($process) {
        $caseToken = [regex]::Replace(
            (Split-Path -Leaf $ResolvedCase), '[^A-Za-z0-9_.-]', '_')
        $runToken = [regex]::Replace(
            (Split-Path -Leaf (Split-Path -Parent $ResolvedCase)),
            '[^A-Za-z0-9_.-]', '_')
        $expectedTask = "GGEUR-$runToken-$caseToken-$name"
        if (Get-ScheduledTask -TaskName $expectedTask `
                -ErrorAction SilentlyContinue) {
            $taskName = $expectedTask
        }
        $processId = [int]$process.ProcessId
        Set-Content -LiteralPath (Join-Path $PidDir "$name.pid") `
            -Value $processId
        if ($taskName) {
            Set-Content -LiteralPath (Join-Path $PidDir "$name.task") `
                -Value $taskName
        }
        Write-Output "$name pid=$processId port=$port adopted=true task=$taskName"
        if ($StartGapMilliseconds -gt 0) {
            Start-Sleep -Milliseconds $StartGapMilliseconds
        }
        continue
    }
    if ($UseScheduledTasks) {
        $caseToken = [regex]::Replace(
            (Split-Path -Leaf $ResolvedCase), '[^A-Za-z0-9_.-]', '_')
        $runToken = [regex]::Replace(
            (Split-Path -Leaf (Split-Path -Parent $ResolvedCase)),
            '[^A-Za-z0-9_.-]', '_')
        $taskName = "GGEUR-$runToken-$caseToken-$name"
        $oldTask = Get-ScheduledTask -TaskName $taskName `
            -ErrorAction SilentlyContinue
        if ($oldTask -and $oldTask.State -eq "Running") {
            throw "$name scheduled task is already running: $taskName"
        }
        if ($oldTask) {
            Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
        }

        $cmdPath = Join-Path $LogDir "$name.run.cmd"
        $commandLine = ('"{0}" "{1}" --config "{2}" --log "{3}" ' +
            '1>"{4}" 2>"{5}"') -f $PythonBin, $ScriptPath,
            $config.FullName, $structuredLog, $stdout, $stderr
        @(
            '@echo off',
            $(if ($SitePackages) { "set PYTHONPATH=$SitePackages" }),
            'set FEDERATEDSCOPE_GGEUR_LIGHTWEIGHT=1',
            'set OMP_NUM_THREADS=1',
            'set MKL_NUM_THREADS=1',
            'set OPENBLAS_NUM_THREADS=1',
            'set CUDA_VISIBLE_DEVICES=',
            $commandLine
        ) | Set-Content -LiteralPath $cmdPath -Encoding ASCII

        $action = New-ScheduledTaskAction `
            -Execute "$env:SystemRoot\System32\cmd.exe" `
            -Argument "/d /c `"$cmdPath`"" `
            -WorkingDirectory $RepoDir
        $identity = [Security.Principal.WindowsIdentity]::GetCurrent().Name
        $principal = New-ScheduledTaskPrincipal `
            -UserId $identity -LogonType S4U -RunLevel Limited
        $settings = New-ScheduledTaskSettingsSet `
            -ExecutionTimeLimit (New-TimeSpan -Days 7) `
            -MultipleInstances IgnoreNew `
            -AllowStartIfOnBatteries `
            -DontStopIfGoingOnBatteries
        Register-ScheduledTask -TaskName $taskName -Action $action `
            -Principal $principal -Settings $settings -Force | Out-Null
        Start-ScheduledTask -TaskName $taskName
    } else {
        $process = Start-Process `
            -FilePath $PythonBin `
            -ArgumentList @(
                $ScriptPath, "--config", $config.FullName,
                "--log", $structuredLog
            ) `
            -WorkingDirectory $RepoDir `
            -WindowStyle Hidden `
            -RedirectStandardOutput $stdout `
            -RedirectStandardError $stderr `
            -PassThru
    }

    $ready = $false
    # Task Scheduler may need more than 30 seconds while many client tasks
    # are being registered on another node.  Keep a two-minute readiness
    # window so a delayed but healthy subserver is not removed prematurely.
    foreach ($attempt in 1..480) {
        if ($process -and -not (Get-Process -Id $process.Id `
                -ErrorAction SilentlyContinue)) {
            break
        }
        if (Get-NetTCPConnection -State Listen -LocalPort $port `
                -ErrorAction SilentlyContinue) {
            $ready = $true
            break
        }
        Start-Sleep -Milliseconds 250
    }
    if (-not $ready) {
        if ($process) {
            Stop-Process -Id $process.Id -Force `
                -ErrorAction SilentlyContinue
        }
        if ($taskName) {
            Stop-ScheduledTask -TaskName $taskName `
                -ErrorAction SilentlyContinue
            Unregister-ScheduledTask -TaskName $taskName `
                -Confirm:$false -ErrorAction SilentlyContinue
        }
        throw "$name failed to listen on port $port; see $stderr"
    }
    if (-not $process) {
        $process = Get-CimInstance Win32_Process |
            Where-Object {
                $_.ExecutablePath -ieq $PythonBin -and
                $_.CommandLine -like "*$($config.FullName)*"
            } | Select-Object -First 1
        if (-not $process) {
            throw "$name is listening but its Python PID was not found"
        }
    }
    $processId = if ($process -is [Microsoft.Management.Infrastructure.CimInstance]) {
        [int]$process.ProcessId
    } else {
        [int]$process.Id
    }
    Set-Content -LiteralPath (Join-Path $PidDir "$name.pid") `
        -Value $processId
    if ($taskName) {
        Set-Content -LiteralPath (Join-Path $PidDir "$name.task") `
            -Value $taskName
    }
    Write-Output "$name pid=$processId port=$port ready=true task=$taskName"
    if ($StartGapMilliseconds -gt 0) {
        Start-Sleep -Milliseconds $StartGapMilliseconds
    }
}

Write-Output "started_subservers=$($configs.Count)"
