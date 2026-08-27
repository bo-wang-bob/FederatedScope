param(
    [Parameter(Mandatory = $true)]
    [string]$CaseDir,
    [string]$PythonBin = "",
    [string]$ClientSitePackages = "",
    [string]$ConfigSet = "clients_8g",
    [int]$StartGapMilliseconds = 1000,
    [int]$LaunchBatchSize = 1,
    [int]$ReadyTimeoutSec = 180,
    [switch]$AllowCudaClientRuntime,
    [switch]$UseScheduledTasks,
    [switch]$AllowConfigSubset
)

$ErrorActionPreference = "Stop"
$RepoDir = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
if (-not $ClientSitePackages) {
    $ClientSitePackages = Join-Path $RepoDir ".venv_client_cpu\Lib\site-packages"
}
if (-not (Test-Path -LiteralPath $ClientSitePackages)) {
    throw "CPU client site-packages does not exist: $ClientSitePackages"
}
if (-not $PythonBin) {
    $venvConfig = Join-Path $RepoDir ".venv_client_cpu\pyvenv.cfg"
    if (-not (Test-Path -LiteralPath $venvConfig)) {
        throw "CPU client environment config does not exist: $venvConfig"
    }
    $homeLine = Get-Content -LiteralPath $venvConfig |
        Where-Object { $_ -match '^\s*home\s*=\s*(.+)\s*$' } |
        Select-Object -First 1
    if (-not $homeLine) {
        throw "Cannot find base Python home in $venvConfig"
    }
    $pythonHome = ([regex]::Match(
        $homeLine, '^\s*home\s*=\s*(.+)\s*$')).Groups[1].Value.Trim()
    $PythonBin = Join-Path $pythonHome "python.exe"
}
if (-not (Test-Path -LiteralPath $PythonBin)) {
    throw "Client Python does not exist: $PythonBin"
}
if ($LaunchBatchSize -lt 1 -or $LaunchBatchSize -gt 32) {
    throw "LaunchBatchSize must be between 1 and 32"
}

# Calling the venv launcher creates a second child process on this Windows
# installation, making PID-based lifecycle management unsafe. Run the base
# interpreter directly and prepend the CPU-only venv packages instead.
$oldPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = if ($oldPythonPath) {
    "$ClientSitePackages$([IO.Path]::PathSeparator)$oldPythonPath"
} else {
    $ClientSitePackages
}
$runtime = & $PythonBin -c `
    "import torch; print(torch.__version__ + '|' + str(torch.cuda.is_available()))"
if ($LASTEXITCODE -ne 0) {
    throw "Client Python cannot import torch: $PythonBin"
}
if (-not $AllowCudaClientRuntime -and $runtime.Trim().EndsWith("|True")) {
    throw "Refusing CUDA PyTorch for many independent clients ($runtime). Use the CPU client environment."
}

$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
$env:VECLIB_MAXIMUM_THREADS = "1"
$env:FEDERATEDSCOPE_GGEUR_LIGHTWEIGHT = "1"

$ResolvedCase = (Resolve-Path (Join-Path $RepoDir $CaseDir)).Path
$ConfigDir = Join-Path $ResolvedCase "configs\$ConfigSet"
if (-not (Test-Path -LiteralPath $ConfigDir) -and $ConfigSet -eq "clients_8g") {
    # Backward compatibility for runs generated before host-specific sets.
    $ConfigDir = Join-Path $ResolvedCase "configs\clients"
}
$LogDir = Join-Path $ResolvedCase "logs"
$PidDir = Join-Path $ResolvedCase "pids"
New-Item -ItemType Directory -Force -Path $LogDir, $PidDir | Out-Null

$configs = Get-ChildItem -LiteralPath $ConfigDir -Filter "client_*.yaml" |
    Sort-Object Name
if (-not $configs) {
    throw "No client configs found under $ConfigDir"
}

$listenerByPort = @{}
foreach ($listener in @(Get-NetTCPConnection -State Listen `
        -ErrorAction SilentlyContinue)) {
    $portKey = [int]$listener.LocalPort
    if (-not $listenerByPort.ContainsKey($portKey)) {
        $listenerByPort[$portKey] = @()
    }
    $listenerByPort[$portKey] = @($listenerByPort[$portKey]) + $listener
}
$processById = @{}
foreach ($candidate in @(Get-CimInstance Win32_Process `
        -ErrorAction SilentlyContinue)) {
    $processById[[int]$candidate.ProcessId] = $candidate
}

function Get-CompatibleClientProcess(
        [System.IO.FileInfo]$Config, [int]$Port, [switch]$Refresh) {
    $listeners = if ($Refresh) {
        @(Get-NetTCPConnection -State Listen -LocalPort $Port `
            -ErrorAction SilentlyContinue)
    } else {
        @($listenerByPort[$Port])
    }
    foreach ($listener in $listeners) {
        $candidatePid = [int]$listener.OwningProcess
        $candidate = if ($Refresh) {
            Get-CimInstance Win32_Process `
                -Filter "ProcessId = $candidatePid" `
                -ErrorAction SilentlyContinue
        } else {
            $processById[$candidatePid]
        }
        if (-not $candidate -or -not $candidate.CommandLine) {
            continue
        }
        $commandLine = [string]$candidate.CommandLine
        if ($commandLine.IndexOf(
                $Config.FullName,
                [StringComparison]::OrdinalIgnoreCase) -ge 0 -and
            $commandLine.IndexOf(
                "federatedscope.main",
                [StringComparison]::OrdinalIgnoreCase) -ge 0) {
            return $candidate
        }
    }
    return $null
}

function Confirm-StartedClient(
        [System.IO.FileInfo]$Config, [string]$Name, [int]$Port,
        [AllowNull()][string]$TaskName, [AllowNull()]$StartedProcess,
        [int]$Index) {
    $ready = $false
    $readyAttempts = [Math]::Max(1, $ReadyTimeoutSec * 4)
    foreach ($attempt in 1..$readyAttempts) {
        if ($StartedProcess -and -not (Get-Process -Id $StartedProcess.Id `
                -ErrorAction SilentlyContinue)) {
            break
        }
        if (Get-NetTCPConnection -State Listen -LocalPort $Port `
                -ErrorAction SilentlyContinue) {
            $ready = $true
            break
        }
        Start-Sleep -Milliseconds 250
    }
    if (-not $ready) {
        if ($StartedProcess) {
            Stop-Process -Id $StartedProcess.Id -Force `
                -ErrorAction SilentlyContinue
        }
        if ($TaskName) {
            Stop-ScheduledTask -TaskName $TaskName `
                -ErrorAction SilentlyContinue
            Unregister-ScheduledTask -TaskName $TaskName `
                -Confirm:$false -ErrorAction SilentlyContinue
        }
        $stderr = Join-Path $LogDir "$Name.stderr.log"
        throw "$Name failed to listen on port $Port; see $stderr"
    }
    $process = Get-CompatibleClientProcess -Config $Config -Port $Port `
        -Refresh
    if (-not $process) {
        throw "$Name is listening but its owning Python process did not match the config"
    }
    $processId = [int]$process.ProcessId
    Set-Content -LiteralPath (Join-Path $PidDir "$Name.pid") `
        -Value $processId
    if ($TaskName) {
        Set-Content -LiteralPath (Join-Path $PidDir "$Name.task") `
            -Value $TaskName
    }
    Write-Output "$Name pid=$processId port=$Port ready=true index=$Index task=$TaskName"
}

$firstConfig = Get-Content -LiteralPath $configs[0].FullName -Raw
$cacheMatch = [regex]::Match(
    $firstConfig,
    '(?m)^\s*feature_cache_dir:\s*[''\"]?([^''\"\r\n]+)'
)
if (-not $cacheMatch.Success) {
    throw "feature_cache_dir is missing from $($configs[0].FullName)"
}
$featureCacheDir = $cacheMatch.Groups[1].Value.Trim()
$cacheMarker = Join-Path $featureCacheDir ".ggeur_feature_cache_ready.json"
if (-not (Test-Path -LiteralPath $cacheMarker)) {
    throw "Feature cache is not validated. Run prepare_feature_cache.ps1 first; missing $cacheMarker"
}
$cacheStatus = Get-Content -LiteralPath $cacheMarker -Raw | ConvertFrom-Json
if (-not $cacheStatus.validated_on -or `
        $cacheStatus.validated_on -ne $env:COMPUTERNAME -or `
        -not $cacheStatus.require_complete_feature_cache) {
    throw "Feature cache marker was not validated on this host. Run prepare_feature_cache.ps1; marker=$cacheMarker"
}
if (-not $AllowConfigSubset -and `
        [int]$cacheStatus.client_num -ne $configs.Count) {
    throw "Feature cache marker client_num=$($cacheStatus.client_num) does not match configs=$($configs.Count)"
}

$clientIndex = 0
$pendingClients = @()
foreach ($config in $configs) {
    $name = [System.IO.Path]::GetFileNameWithoutExtension($config.Name)
    $pidPath = Join-Path $PidDir "$name.pid"
    if (Test-Path $pidPath) {
        $oldPid = [int](Get-Content $pidPath -Raw)
        $oldProcess = $processById[$oldPid]
        if ($oldProcess -and $oldProcess.CommandLine -and
            ([string]$oldProcess.CommandLine).IndexOf(
                $config.FullName,
                [StringComparison]::OrdinalIgnoreCase) -lt 0) {
            # Windows can reuse a PID after the previous case has exited.
            # The PID file is only a launcher hint; the command line is the
            # authority, so discard this stale hint instead of blocking the
            # whole 60-client launch.
            Remove-Item -LiteralPath $pidPath -Force
            Write-Warning (
                "$name removed stale PID file for reused process $oldPid")
        } elseif (-not $oldProcess) {
            Remove-Item -LiteralPath $pidPath -Force
        }
    }
    $configText = Get-Content -LiteralPath $config.FullName -Raw
    $portMatch = [regex]::Match($configText, '(?m)^\s*client_port:\s*(\d+)')
    if (-not $portMatch.Success) {
        throw "client_port is missing from $($config.FullName)"
    }
    $port = [int]$portMatch.Groups[1].Value
    $listeners = @(Get-NetTCPConnection -State Listen -LocalPort $port `
        -ErrorAction SilentlyContinue)
    if ($listeners.Count -gt 0 -and
        -not (Get-CompatibleClientProcess -Config $config -Port $port)) {
        throw "Client port $port is already in use by an unrelated process"
    }
}

foreach ($config in $configs) {
    $name = [System.IO.Path]::GetFileNameWithoutExtension($config.Name)
    $stdout = Join-Path $LogDir "$name.stdout.log"
    $stderr = Join-Path $LogDir "$name.stderr.log"
    $taskName = $null
    $configText = Get-Content -LiteralPath $config.FullName -Raw
    $port = [int]([regex]::Match(
        $configText, '(?m)^\s*client_port:\s*(\d+)').Groups[1].Value)
    $process = Get-CompatibleClientProcess -Config $config -Port $port
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
        Write-Output "$name pid=$processId port=$port adopted=true index=$clientIndex task=$taskName"
        $clientIndex += 1
        if ($StartGapMilliseconds -gt 0) {
            Start-Sleep -Milliseconds $StartGapMilliseconds
        }
        continue
    }
    $process = $null
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
        $commandLine = ('"{0}" -m federatedscope.main --cfg "{1}" ' +
            '1>"{2}" 2>"{3}"') -f $PythonBin, $config.FullName,
            $stdout, $stderr
        @(
            '@echo off',
            "set PYTHONPATH=$ClientSitePackages",
            'set FEDERATEDSCOPE_GGEUR_LIGHTWEIGHT=1',
            'set OMP_NUM_THREADS=1',
            'set MKL_NUM_THREADS=1',
            'set OPENBLAS_NUM_THREADS=1',
            'set NUMEXPR_NUM_THREADS=1',
            'set VECLIB_MAXIMUM_THREADS=1',
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
                "-m", "federatedscope.main", "--cfg", $config.FullName
            ) `
            -WorkingDirectory $RepoDir `
            -WindowStyle Hidden `
            -RedirectStandardOutput $stdout `
            -RedirectStandardError $stderr `
            -PassThru
    }
    $pendingClients += [pscustomobject]@{
        Config = $config
        Name = $name
        Port = $port
        TaskName = $taskName
        Process = $process
        Index = $clientIndex
    }
    $clientIndex += 1
    if ($StartGapMilliseconds -gt 0) {
        Start-Sleep -Milliseconds $StartGapMilliseconds
    }
    if ($pendingClients.Count -ge $LaunchBatchSize) {
        foreach ($pending in $pendingClients) {
            Confirm-StartedClient -Config $pending.Config `
                -Name $pending.Name -Port $pending.Port `
                -TaskName $pending.TaskName `
                -StartedProcess $pending.Process -Index $pending.Index
        }
        $pendingClients = @()
    }
}

foreach ($pending in $pendingClients) {
    Confirm-StartedClient -Config $pending.Config `
        -Name $pending.Name -Port $pending.Port `
        -TaskName $pending.TaskName `
        -StartedProcess $pending.Process -Index $pending.Index
}

Write-Output "started_clients=$($configs.Count)"
