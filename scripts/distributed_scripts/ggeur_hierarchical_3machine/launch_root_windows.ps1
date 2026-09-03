param(
    [Parameter(Mandatory = $true)][string]$CaseDir,
    [string]$PythonBin = "",
    [string]$SitePackages = ""
)

$ErrorActionPreference = "Stop"
$RepoDir = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$ResolvedCase = (Resolve-Path (Join-Path $RepoDir $CaseDir)).Path
$ConfigPath = Join-Path $ResolvedCase "configs\root_server.yaml"
$LogDir = Join-Path $ResolvedCase "logs"
$PidDir = Join-Path $ResolvedCase "pids"
if (-not $SitePackages) {
    $SitePackages = Join-Path $RepoDir ".venv_client_cpu\Lib\site-packages"
}
if (-not $PythonBin) {
    $venvConfig = Join-Path $RepoDir ".venv_client_cpu\pyvenv.cfg"
    $homeLine = Get-Content -LiteralPath $venvConfig |
        Where-Object { $_ -match '^\s*home\s*=\s*(.+)\s*$' } |
        Select-Object -First 1
    if (-not $homeLine) { throw "Cannot resolve base Python from $venvConfig" }
    $pythonHome = ([regex]::Match(
        $homeLine, '^\s*home\s*=\s*(.+)\s*$')).Groups[1].Value.Trim()
    $PythonBin = Join-Path $pythonHome "python.exe"
}
foreach ($path in @($ConfigPath, $PythonBin, $SitePackages)) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Missing: $path" }
}
if (Get-NetTCPConnection -State Listen -LocalPort 60050 `
        -ErrorAction SilentlyContinue) {
    throw "Root port 60050 is already in use"
}
New-Item -ItemType Directory -Force -Path $LogDir, $PidDir | Out-Null
$env:PYTHONPATH = $SitePackages
$env:FEDERATEDSCOPE_GGEUR_LIGHTWEIGHT = "1"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
$env:VECLIB_MAXIMUM_THREADS = "1"
$env:CUDA_VISIBLE_DEVICES = ""
$stdout = Join-Path $LogDir "root.stdout.log"
$stderr = Join-Path $LogDir "root.stderr.log"
$process = Start-Process -FilePath $PythonBin -ArgumentList @(
    "-m", "federatedscope.main", "--cfg", $ConfigPath
) -WorkingDirectory $RepoDir -WindowStyle Hidden `
    -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
Set-Content -LiteralPath (Join-Path $PidDir "root.pid") -Value $process.Id
foreach ($attempt in 1..240) {
    if (-not (Get-Process -Id $process.Id -ErrorAction SilentlyContinue)) {
        throw "Root exited before listening; see $stderr"
    }
    if (Get-NetTCPConnection -State Listen -LocalPort 60050 `
            -ErrorAction SilentlyContinue) {
        Write-Output "root pid=$($process.Id) port=60050 ready=true"
        return
    }
    Start-Sleep -Milliseconds 250
}
Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
throw "Root failed to listen on port 60050; see $stderr"
