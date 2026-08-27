param(
    [string]$BaseUrl = "http://10.112.81.135:18080",
    [string]$PythonBin = "D:\ProgramData\anaconda3\envs\pi_fmd_gpu_py39\python.exe"
)

$ErrorActionPreference = "Stop"
$RepoDir = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$StateDir = Join-Path $RepoDir "exp\domainnet_resource_prepare"
$Log = Join-Path $StateDir "pipeline.log"
New-Item -ItemType Directory -Force -Path $StateDir | Out-Null

function Write-State([string]$Message) {
    Add-Content -LiteralPath $Log -Value "$(Get-Date -Format o) $Message"
}

$readyUrl = "$BaseUrl/domainnet_bundle.ready.json"
$readyPath = Join-Path $StateDir "domainnet_bundle.ready.json"
Write-State "WAIT_BUNDLE $readyUrl"
while ($true) {
    try {
        Invoke-WebRequest -UseBasicParsing -Uri $readyUrl `
            -OutFile $readyPath -TimeoutSec 20
        break
    } catch {
        Start-Sleep -Seconds 60
    }
}
$ready = Get-Content -LiteralPath $readyPath -Raw | ConvertFrom-Json
$bundle = Join-Path $StateDir $ready.bundle
$bundleUrl = "$BaseUrl/$($ready.bundle)"
Write-State "DOWNLOAD_START $bundleUrl"
$curl = (Get-Command curl.exe -ErrorAction Stop).Source
& $curl --fail --location --retry 10 --retry-delay 5 `
    --continue-at - --output $bundle $bundleUrl
$curlExit = $LASTEXITCODE
if ($curlExit -eq 33 -and (Test-Path -LiteralPath $bundle)) {
    # Python's simple HTTP publisher may not implement byte ranges. Preserve
    # the partial artifact for the failure audit, then retry from byte zero.
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $partial = "$bundle.resume_unsupported_$stamp.partial"
    Move-Item -LiteralPath $bundle -Destination $partial
    Write-State "RESUME_UNSUPPORTED partial=$partial; retry=full"
    & $curl --fail --location --retry 10 --retry-delay 5 `
        --output $bundle $bundleUrl
    $curlExit = $LASTEXITCODE
}
if ($curlExit -ne 0) {
    throw "DomainNet bundle curl download failed: exit=$curlExit"
}
$actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $bundle).Hash.ToLowerInvariant()
if ($actual -ne ([string]$ready.sha256).ToLowerInvariant()) {
    throw "DomainNet bundle hash mismatch: expected=$($ready.sha256), actual=$actual"
}
Write-State "DOWNLOAD_COMPLETE $actual"
tar -xzf $bundle -C $RepoDir
if ($LASTEXITCODE -ne 0) { throw "DomainNet bundle extraction failed" }

$prepare = Join-Path $PSScriptRoot "prepare_feature_cache.ps1"
foreach ($group in @("domainnet_vit", "domainnet_cnn", "domainnet_mixer")) {
    Write-State "VALIDATE_START $group"
    & $prepare -Group $group -PythonBin $PythonBin
    $markerPath = Join-Path $RepoDir "exp\distributed_feature_cache\$group\.ggeur_feature_cache_ready.json"
    $marker = Get-Content -LiteralPath $markerPath -Raw | ConvertFrom-Json
    if ($marker.validated_on -ne $env:COMPUTERNAME -or
        -not $marker.require_complete_feature_cache) {
        throw "DomainNet cache validation marker is invalid: $group"
    }
    Write-State "VALIDATE_COMPLETE $group"
}
New-Item -ItemType File -Force -Path (Join-Path $StateDir "pipeline.complete") | Out-Null
Write-State COMPLETE
