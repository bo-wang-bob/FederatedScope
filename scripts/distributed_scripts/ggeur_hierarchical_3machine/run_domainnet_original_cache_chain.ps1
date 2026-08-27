param(
    [string]$Repo = "C:/Users/pc/FederatedScope",
    [string]$Python = "C:/Users/pc/miniconda3/envs/cerp/python.exe",
    [string]$StageName = "domainnet_original_4domains_20260804",
    [string]$CacheNamespace = "distributed_feature_cache_domainnet_original_20260804"
)

$ErrorActionPreference = "Stop"
$scriptDir = Join-Path $Repo "scripts\distributed_scripts\ggeur_hierarchical_3machine"
$stage = Join-Path $Repo "data_staging\$StageName"
$dataRoot = Join-Path $stage "dataset"
$manifest = Join-Path $stage "domainnet_manifest.json"
$referenceDir = Join-Path $stage "reference_samples"
$cacheRoot = Join-Path $Repo "exp\$CacheNamespace"
$state = Join-Path $Repo "exp\domainnet_original_cache_20260804"
New-Item -ItemType Directory -Force -Path $state, $referenceDir, $cacheRoot | Out-Null

function Invoke-LoggedPython {
    param([string]$Step, [string[]]$Arguments)
    $log = Join-Path $state ($Step + ".stdout.log")
    $err = Join-Path $state ($Step + ".stderr.log")
    $started = Get-Date
    "step=$Step started=$($started.ToString('o'))" | Add-Content -LiteralPath $log
    $process = Start-Process -FilePath $Python -ArgumentList $Arguments `
        -WorkingDirectory $Repo -WindowStyle Hidden `
        -RedirectStandardOutput $log -RedirectStandardError $err `
        -Wait -PassThru
    if ($process.ExitCode -ne 0) {
        throw "$Step failed with exit code $($process.ExitCode); see $log and $err"
    }
    "step=$Step completed=$((Get-Date).ToString('o'))" | Add-Content -LiteralPath $log
}

if (-not (Test-Path -LiteralPath $manifest -PathType Leaf)) {
    throw "missing staged manifest: $manifest"
}
foreach ($domain in @("clipart", "painting", "real", "sketch")) {
    if (-not (Test-Path -LiteralPath (Join-Path $dataRoot $domain) -PathType Container)) {
        throw "missing staged domain: $domain"
    }
}

Invoke-LoggedPython -Step "00_reference" -Arguments @(
    "-u", (Join-Path $scriptDir "prepare_domainnet_reference_samples.py"),
    "--repo", $Repo,
    "--data-root", $dataRoot,
    "--manifest", $manifest,
    "--output-dir", $referenceDir,
    "--groups", "domainnet_vit", "domainnet_cnn", "domainnet_mixer",
    "--samples-per-domain", "4", "--seed", "42"
)

foreach ($group in @("domainnet_vit", "domainnet_cnn", "domainnet_mixer")) {
    Invoke-LoggedPython -Step ("10_train_" + $group) -Arguments @(
        "-u", (Join-Path $scriptDir "prepare_domainnet_train_cache.py"),
        "--repo", $Repo,
        "--data-root", $dataRoot,
        "--manifest", $manifest,
        "--reference-dir", $referenceDir,
        "--cache-root", $cacheRoot,
        "--group", $group,
        "--num-workers", "4", "--seed", "42", "--train-ratio", "0.7",
        "--min-reference-cosine", "0.999"
    )
}

Invoke-LoggedPython -Step "20_eval_all" -Arguments @(
    "-u", (Join-Path $scriptDir "prepare_domainnet_eval_cache.py"),
    "--repo", $Repo,
    "--data-root", $dataRoot,
    "--manifest", $manifest,
    "--reference-dir", $referenceDir,
    "--cache-root", $cacheRoot,
    "--groups", "domainnet_vit", "domainnet_cnn", "domainnet_mixer",
    "--num-workers", "4", "--seed", "42", "--train-ratio", "0.7",
    "--val-ratio", "0.0", "--min-reference-cosine", "0.999"
)

$completion = [ordered]@{
    completed_at = (Get-Date).ToString("o")
    stage = $stage
    manifest = $manifest
    cache_root = $cacheRoot
    reference_dir = $referenceDir
    records_sha256 = (
        (Get-Content -LiteralPath $manifest -Raw | ConvertFrom-Json).records_sha256
    )
}
$completion | ConvertTo-Json -Depth 6 | Set-Content `
    -LiteralPath (Join-Path $state "completed.json") -Encoding UTF8
