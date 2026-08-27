param(
    [string]$RunId = "digit3_vit_accuracy_cached_3machine_20260814_v1"
)

$ErrorActionPreference = "Stop"
$runRoot = Join-Path $PSScriptRoot "runs\$RunId"
$caseRoot = Join-Path $runRoot "digit3_vit_platform"
$oldClientCache = "D:/Projects/FederatedScope/exp/aug/$RunId/digit3_vit_platform"
$newClientCache = "D:/Projects/FederatedScope/exp/distributed_feature_cache/digit3_vit"
$oldRootCache = "/root/autodl-tmp/FederatedScope/exp/aug/$RunId/digit3_vit_platform"
$newRootCache = "/root/autodl-tmp/FederatedScope/exp/distributed_feature_cache/digit3_vit"

$files = @(Get-ChildItem -LiteralPath (Join-Path $caseRoot "configs") `
    -Recurse -File | Where-Object { $_.Extension -in @(".yaml", ".yml") })
foreach ($file in $files) {
    $text = Get-Content -Raw -LiteralPath $file.FullName
    $updated = $text.Replace($oldClientCache, $newClientCache).Replace(
        $oldRootCache, $newRootCache)
    if ($updated -ne $text) {
        [IO.File]::WriteAllText(
            $file.FullName,
            $updated,
            [Text.UTF8Encoding]::new($false)
        )
    }
}

$remaining = @(Get-ChildItem -LiteralPath (Join-Path $caseRoot "configs") `
    -Recurse -File | Select-String -SimpleMatch $oldClientCache)
if ($remaining.Count -ne 0) {
    throw "Old long cache path still exists in $($remaining.Count) config files"
}

"UPDATED_CONFIG_FILES=$($files.Count)"
"CLIENT_CACHE_PATH=$newClientCache"
