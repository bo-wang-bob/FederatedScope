param(
  [Parameter(Mandatory = $true)]
  [string]$RepoDir
)

$ErrorActionPreference = 'Stop'
$repo = [IO.Path]::GetFullPath($RepoDir)
$state = Join-Path $repo 'exp\domainnet_eval_cache_dual'
$pipelineLog = Join-Path $state 'pipeline.log'
$stdoutLog = Join-Path $state 'cache.stdout.log'
$stderrLog = Join-Path $state 'cache.stderr.log'
$python = 'C:\Users\pc\miniconda3\envs\cerp\python.exe'
$tool = Join-Path $repo `
  'scripts\distributed_scripts\ggeur_hierarchical_3machine\prepare_domainnet_eval_cache.py'
New-Item -ItemType Directory -Force -Path $state | Out-Null

function Write-Pipeline([string]$message) {
  $line = "$(Get-Date -Format o) $message"
  Add-Content -LiteralPath $pipelineLog -Value $line
  Write-Output $line
}

trap {
  $details = ($_ | Out-String).Trim()
  Add-Content -LiteralPath (Join-Path $state 'task.stderr.log') `
    -Value "$(Get-Date -Format o) $details"
  Write-Pipeline "TASK_FAILED details=$details"
  exit 1
}

$required = @(
  @{
    Name = 'clip'
    Path = (Join-Path $repo 'pretrained_models\ViT-B-16.pt')
    Bytes = 350837078
  },
  @{
    Name = 'convnext'
    Path = (Join-Path $repo 'pretrained_models\convnext_base-6075fbad.pth')
    Bytes = 354486097
  },
  @{
    Name = 'mixer'
    Path = (Join-Path $repo 'pretrained_models\mixer_b16_224_complete.pth')
    Bytes = 239544439
  }
)

while ($true) {
  $missing = @()
  foreach ($item in $required) {
    if (-not (Test-Path -LiteralPath $item.Path)) {
      $missing += "$($item.Name):missing"
    } elseif ((Get-Item -LiteralPath $item.Path).Length -ne $item.Bytes) {
      $actualBytes = (Get-Item -LiteralPath $item.Path).Length
      $missing += "$($item.Name):bytes=$actualBytes/$($item.Bytes)"
    }
  }
  foreach ($domain in @('clipart', 'painting', 'real', 'sketch')) {
    $domainPath = Join-Path $repo "data\DomainNet\$domain"
    $marker = Join-Path $repo `
      "exp\domainnet_resource_download_dual\$domain.extracted.json"
    if (-not (Test-Path -LiteralPath $domainPath)) {
      $missing += "$($domain):directory"
    }
    if (-not (Test-Path -LiteralPath $marker)) {
      $missing += "$($domain):marker"
    }
  }
  foreach ($item in @(
    $python,
    $tool,
    (Join-Path $repo `
      'exp\distributed_manifests\domainnet_4domains\domainnet_manifest.json'),
    (Join-Path $repo `
      'exp\domainnet_reference_samples\domainnet_vit_reference.npz'),
    (Join-Path $repo `
      'exp\domainnet_reference_samples\domainnet_cnn_reference.npz'),
    (Join-Path $repo `
      'exp\domainnet_reference_samples\domainnet_mixer_reference.npz')
  )) {
    if (-not (Test-Path -LiteralPath $item)) {
      $missing += "file:$item"
    }
  }
  if ($missing.Count -eq 0) {
    break
  }
  Write-Pipeline ("RESOURCE_WAIT missing=" + ($missing -join ','))
  Start-Sleep -Seconds 60
}

Write-Pipeline 'RESOURCES_READY exact_lengths=true domains=4 references=3'
$env:PYTHONUNBUFFERED = '1'
Write-Pipeline `
  'VERIFY_START groups=domainnet_vit,domainnet_cnn,domainnet_mixer'
$savedErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
try {
  & $python $tool --repo $repo `
    --groups domainnet_vit domainnet_cnn domainnet_mixer `
    --verify-only --num-workers 0 1>> $stdoutLog 2>> $stderrLog
  $verifyExitCode = $LASTEXITCODE
} finally {
  $ErrorActionPreference = $savedErrorActionPreference
}
if ($verifyExitCode -ne 0) {
  throw "extractor verification failed: $verifyExitCode"
}
Write-Pipeline 'VERIFY_COMPLETE groups=3'

Write-Pipeline `
  'CACHE_START groups=domainnet_vit,domainnet_cnn,domainnet_mixer'
$ErrorActionPreference = 'Continue'
try {
  & $python $tool --repo $repo `
    --groups domainnet_vit domainnet_cnn domainnet_mixer `
    --num-workers 4 1>> $stdoutLog 2>> $stderrLog
  $cacheExitCode = $LASTEXITCODE
} finally {
  $ErrorActionPreference = $savedErrorActionPreference
}
if ($cacheExitCode -ne 0) {
  throw "eval cache generation failed: $cacheExitCode"
}
Write-Pipeline 'CACHE_COMPLETE groups=3'

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$package = Join-Path $state "package_$stamp"
New-Item -ItemType Directory -Force -Path $package | Out-Null
$packaged = 0
foreach ($group in @('domainnet_vit', 'domainnet_cnn', 'domainnet_mixer')) {
  $source = Join-Path $repo "exp\distributed_feature_cache\$group"
  $target = Join-Path $package "exp\distributed_feature_cache\$group"
  New-Item -ItemType Directory -Force -Path $target | Out-Null
  foreach ($file in @(Get-ChildItem -LiteralPath $source -File `
      -Filter 'domainnet_*_test_*.npz')) {
    Copy-Item -LiteralPath $file.FullName `
      -Destination (Join-Path $target $file.Name) -Force
    $packaged += 1
  }
}
if ($packaged -ne 12) {
  throw "expected 12 test caches, packaged $packaged"
}

$completion = Join-Path $repo `
  'exp\distributed_feature_cache\domainnet_eval_cache_completion.json'
$completionTarget = Join-Path $package `
  'exp\distributed_feature_cache\domainnet_eval_cache_completion.json'
Copy-Item -LiteralPath $completion -Destination $completionTarget -Force
$archive = Join-Path $state 'domainnet_eval_cache.tar'
if (Test-Path -LiteralPath $archive) {
  $preserved = Join-Path $state `
    "domainnet_eval_cache.$stamp.previous.tar"
  Move-Item -LiteralPath $archive -Destination $preserved
}
& tar.exe -cf $archive -C $package .
if ($LASTEXITCODE -ne 0) {
  throw "eval cache package failed: $LASTEXITCODE"
}

$result = [ordered]@{
  completed_at = (Get-Date).ToString('o')
  package = $package
  archive = $archive
  archive_bytes = (Get-Item -LiteralPath $archive).Length
  cache_files = $packaged
}
$result | ConvertTo-Json -Depth 4 | Set-Content `
  -LiteralPath (Join-Path $state 'completed.json') -Encoding UTF8
Write-Pipeline `
  "PACKAGE_COMPLETE files=$packaged bytes=$($result.archive_bytes) archive=$archive"
