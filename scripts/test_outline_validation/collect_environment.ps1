param(
    [Parameter(Mandatory = $true)][ValidateSet("client", "subserver")][string]$Role
)

$ErrorActionPreference = "Stop"
$ProjectDir = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$PythonBin = (Get-Command python -ErrorAction Stop).Source
$Os = Get-CimInstance Win32_OperatingSystem
$Gpu = @(Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name)
$Payload = [ordered]@{
    role = $Role
    os_caption = $Os.Caption
    os_version = $Os.Version
    os_build = $Os.BuildNumber
    python = (& $PythonBin -c "import platform; print(platform.python_version())").Trim()
    pytorch = (& $PythonBin -c "import torch; print(torch.__version__)" 2>$null).Trim()
    grpcio = (& $PythonBin -c "import grpc; print(grpc.__version__)" 2>$null).Trim()
    protobuf = (& $PythonBin -c "import google.protobuf; print(google.protobuf.__version__)" 2>$null).Trim()
    gpu = $Gpu
    project_dir = $ProjectDir
}
$OutputDir = Join-Path $ProjectDir "exp\test_outline_validation\environment"
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$OutputFile = Join-Path $OutputDir "$Role.json"
$Payload | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $OutputFile -Encoding UTF8
Write-Output "STATUS=PASS"
Write-Output "ROLE=$Role"
Write-Output "OS=$($Payload.os_caption) $($Payload.os_version) build $($Payload.os_build)"
Write-Output "PYTHON=$($Payload.python)"
Write-Output "PYTORCH=$($Payload.pytorch)"
Write-Output "EVIDENCE_FILE=$OutputFile"
