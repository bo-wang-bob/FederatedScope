param(
    [Parameter(Mandatory = $true)][string]$CaseId,
    [Parameter(Mandatory = $true)][ValidateRange(1, 8)][int]$Step,
    [Parameter(Mandatory = $true)][ValidateSet("client", "subserver", "root")][string]$Role,
    [switch]$PlanOnly
)

$ErrorActionPreference = "Stop"
$Utf8NoBom = [System.Text.UTF8Encoding]::new($false)
[Console]::InputEncoding = $Utf8NoBom
[Console]::OutputEncoding = $Utf8NoBom
$OutputEncoding = $Utf8NoBom
$env:PYTHONIOENCODING = "utf-8"
$ProjectDir = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$PythonBin = if ($Role -eq "subserver" -and
    (Test-Path -LiteralPath "C:\Users\pc\miniconda3\envs\cerp\python.exe")) {
    "C:\Users\pc\miniconda3\envs\cerp\python.exe"
} elseif ($Role -eq "client" -and
    (Test-Path -LiteralPath "D:\Projects\FederatedScope\.venv_client_cpu\Scripts\python.exe")) {
    "D:\Projects\FederatedScope\.venv_client_cpu\Scripts\python.exe"
} else {
    (Get-Command python -ErrorAction Stop).Source
}

$Arguments = @(
    (Join-Path $PSScriptRoot "case_step_runner.py"),
    "--project-dir", $ProjectDir,
    "--case-id", $CaseId,
    "--step", [string]$Step,
    "--role", $Role
)
if ($PlanOnly) { $Arguments += "--plan-only" }
& $PythonBin @Arguments
exit $LASTEXITCODE
