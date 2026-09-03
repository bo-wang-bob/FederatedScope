param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("domainnet_vit", "domainnet_cnn", "domainnet_mixer")]
    [string]$Group,
    [string]$Repo = "C:/Users/pc/FederatedScope",
    [string]$Python = "C:/Users/pc/miniconda3/envs/cerp/python.exe"
)

$ErrorActionPreference = "Stop"
$script = Join-Path $Repo (
    "scripts\distributed_scripts\ggeur_hierarchical_3machine\" +
    "prepare_domainnet_train_cache.py"
)
$state = Join-Path $Repo "exp\domainnet_train_cache_dual\$Group"
New-Item -ItemType Directory -Force -Path $state | Out-Null
$stdout = Join-Path $state "direct.stdout.log"
$stderr = Join-Path $state "direct.stderr.log"
$arguments = @(
    "-u", $script,
    "--repo", $Repo,
    "--data-root", "data/DomainNet",
    "--manifest", "exp/distributed_manifests/domainnet_4domains/domainnet_manifest.json",
    "--reference-dir", "exp/domainnet_reference_samples",
    "--group", $Group,
    "--num-workers", "4",
    "--seed", "42",
    "--train-ratio", "0.7",
    "--min-reference-cosine", "0.999"
)
$process = Start-Process -FilePath $Python -ArgumentList $arguments `
    -WorkingDirectory $Repo -WindowStyle Hidden `
    -RedirectStandardOutput $stdout -RedirectStandardError $stderr `
    -Wait -PassThru
if ($process.ExitCode -ne 0) {
    throw "Direct DomainNet training cache failed with exit code $($process.ExitCode)"
}
