param(
    [Parameter(Mandatory = $true)]
    [ValidateSet(
        "officehome_vit", "officehome_cnn", "officehome_mixer",
        "domainnet_vit", "domainnet_cnn", "domainnet_mixer",
        "mdsent_rnn", "mdsent_lstm"
    )]
    [string]$Group,
    [string]$PythonBin = "python",
    [string]$Method = "fedavg",
    [string]$FeatureCacheRoot = "D:/Projects/FederatedScope/exp/distributed_feature_cache",
    [string]$OfficeHomeRoot = "D:/Projects/FederatedScope/OfficeHomeDataset_10072016",
    [string]$OfficeHomeManifestRoot = "D:/Projects/FederatedScope/exp/distributed_manifests/officehome_60c_lds01_seed42",
    [string]$DomainNetRoot = "D:/Projects/FederatedScope/data/DomainNet",
    [string]$DomainNetManifestPath = "D:/Projects/FederatedScope/exp/distributed_manifests/domainnet_4domains/domainnet_manifest.json",
    [string]$MDSentRoot = "D:/Projects/FederatedScope/data/sentiment",
    [string]$ClipModelPath = "D:/Projects/FederatedScope/pretrained_models/ViT-B-16.pt",
    [string]$MixerCheckpointPath = "D:/Projects/FederatedScope/pretrained_models/mixer_b16_224_complete.pth",
    [string]$BertModelPath = "D:/Projects/FederatedScope/pretrained_models/nlptown_bert_base_multilingual_uncased_senti"
)

$ErrorActionPreference = "Stop"
$RepoDir = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$SourceConfig = Join-Path $RepoDir (
    "scripts\example_configs\ggeur_final_5models\{0}\{1}.yaml" -f `
        $Group, $Method
)
if (-not (Test-Path -LiteralPath $SourceConfig)) {
    throw "Source config does not exist: $SourceConfig"
}

$CacheDir = "$($FeatureCacheRoot.TrimEnd('/'))/$Group"
$OutDir = (Join-Path $RepoDir "exp\cache_warmup\$Group\out")
$LogDir = Join-Path $RepoDir "exp\cache_warmup\$Group"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$LogPath = Join-Path $LogDir "prepare_feature_cache.log"
$ReadyMarker = Join-Path $CacheDir ".ggeur_feature_cache_ready.json"
if (Test-Path -LiteralPath $ReadyMarker) {
    $importedMarker = "$ReadyMarker.imported_$((Get-Date).ToString('yyyyMMdd_HHmmss'))"
    Move-Item -LiteralPath $ReadyMarker -Destination $importedMarker
    Write-Output "preserved_previous_marker=$importedMarker"
}

Write-Output "Preparing the shared raw feature cache for $Group"
Write-Output "cache_dir=$CacheDir"
Write-Output "log=$LogPath"

$Arguments = @(
    "-m", "federatedscope.main",
    "--cfg", $SourceConfig,
    "federate.total_round_num", "1",
    "federate.mode", "standalone",
    "use_gpu", "True",
    "device", "0",
    "eval.freq", "0",
    "ggeur.feature_cache_dir", $CacheDir,
    "ggeur.use_feature_cache", "True",
    # Cache warm-up must be allowed to extract missing vectors. Formal runs
    # still set this flag to True and consume only the completed cache marker.
    "ggeur.require_complete_feature_cache", "False",
    "ggeur.reuse_augmented_feature_cache", "False",
    "ggeur.save_augmented_feature_cache", "False",
    "outdir", $OutDir,
    "expname", "cache_warmup"
)
$env:FEDERATEDSCOPE_GGEUR_LIGHTWEIGHT = "1"

if ($Group.StartsWith("officehome_")) {
    if (-not (Test-Path -LiteralPath $OfficeHomeManifestRoot)) {
        throw "OfficeHome manifest root does not exist: $OfficeHomeManifestRoot"
    }
    $Arguments += @(
        "data.root", $OfficeHomeRoot,
        "ggeur.officehome_manifest_base", $OfficeHomeManifestRoot,
        "ggeur.officehome_manifest_use_config_root", "True"
    )
} elseif ($Group.StartsWith("domainnet_")) {
    if (-not (Test-Path -LiteralPath $DomainNetManifestPath)) {
        throw "DomainNet manifest does not exist: $DomainNetManifestPath"
    }
    $Arguments += @(
        "data.root", $DomainNetRoot,
        "ggeur.domainnet_manifest_path", $DomainNetManifestPath
    )
} else {
    $Arguments += @("data.root", $MDSentRoot)
}
if ($Group.EndsWith("_vit")) {
    $Arguments += @("ggeur.clip_model_path", $ClipModelPath)
} elseif ($Group.EndsWith("_mixer")) {
    $Arguments += @("ggeur.timm_checkpoint_path", $MixerCheckpointPath)
} elseif ($Group.StartsWith("mdsent_")) {
    $Arguments += @("ggeur.bert_model_path", $BertModelPath)
}

$process = Start-Process `
    -FilePath $PythonBin `
    -ArgumentList $Arguments `
    -WorkingDirectory $RepoDir `
    -WindowStyle Hidden `
    -RedirectStandardOutput $LogPath `
    -RedirectStandardError "$LogPath.stderr" `
    -Wait `
    -PassThru
if ($process.ExitCode -ne 0) {
    throw "Cache preparation failed with exit code $($process.ExitCode)"
}

$expectedClientMatch = [regex]::Match(
    (Get-Content -LiteralPath $SourceConfig -Raw),
    '(?m)^\s*client_num:\s*(\d+)'
)
if (-not $expectedClientMatch.Success) {
    throw "Could not read client_num from $SourceConfig"
}
$expectedClients = [int]$expectedClientMatch.Groups[1].Value
$timingLines = Select-String -LiteralPath $LogPath, "$LogPath.stderr" `
    -Pattern 'GGEUR_TIMING_CLIENT .*stage=feature_extraction' `
    -ErrorAction SilentlyContinue
$seenClients = @{}
foreach ($match in $timingLines) {
    $idMatch = [regex]::Match($match.Line, 'client=(\d+)')
    if ($idMatch.Success) {
        $seenClients[$idMatch.Groups[1].Value] = $true
    }
}
if ($seenClients.Count -ne $expectedClients) {
    throw "Cache preparation covered $($seenClients.Count)/$expectedClients clients; see $LogPath"
}
$cacheFiles = Get-ChildItem -LiteralPath $CacheDir -Filter "*.npz" -File `
    -ErrorAction SilentlyContinue
if (-not $cacheFiles) {
    throw "No raw feature cache files were produced under $CacheDir"
}
$marker = [ordered]@{
    group = $Group
    method = $Method
    client_num = $expectedClients
    cache_files = $cacheFiles.Count
    validated_on = $env:COMPUTERNAME
    require_complete_feature_cache = $true
    completed_at = (Get-Date).ToString("o")
}
$marker | ConvertTo-Json | Set-Content `
    -LiteralPath $ReadyMarker `
    -Encoding UTF8

Write-Output "feature_cache_ready=$CacheDir"
