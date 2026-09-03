param(
    [string]$RepoDir = "C:\Users\pc\FederatedScope",
    [string]$PythonBin = "C:\Users\pc\miniconda3\envs\cerp\python.exe"
)

$ErrorActionPreference = "Stop"
$cacheDir = Join-Path $RepoDir "exp\distributed_feature_cache\digit3_vit"
$logDir = Join-Path $cacheDir "prepare_logs"
$stdout = Join-Path $logDir "clip_vitb16.stdout.log"
$stderr = Join-Path $logDir "clip_vitb16.stderr.log"
$complete = Join-Path $cacheDir ".clip_vitb16_feature_cache_complete"
$failed = Join-Path $cacheDir ".clip_vitb16_feature_cache_failed"

New-Item -ItemType Directory -Force -Path $cacheDir, $logDir | Out-Null
Remove-Item -LiteralPath $complete, $failed -Force -ErrorAction SilentlyContinue

$arguments = @(
    (Join-Path $RepoDir "scripts\prepare_digit_three_domain_vit_cache.py"),
    "--data-root", (Join-Path $RepoDir "data\digit_three_domain"),
    "--cache-dir", $cacheDir,
    "--backend", "clip",
    "--model", "ViT-B-16",
    "--clip-pretrained", "openai",
    "--checkpoint-path", (Join-Path $RepoDir "pretrained_models\ViT-B-16.pt"),
    "--batch-size", "128",
    "--num-workers", "4",
    "--device", "cuda",
    "--fp16"
)

$startedAt = Get-Date
try {
    $process = Start-Process -FilePath $PythonBin `
        -ArgumentList $arguments `
        -WorkingDirectory $RepoDir `
        -WindowStyle Hidden `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -Wait -PassThru
    if ($process.ExitCode -ne 0) {
        throw "feature cache command exited with code $($process.ExitCode)"
    }
    $required = @(
        "digits-3domain_emnist_digits_clip_ViT_B_16_openai_d512.npz",
        "digits-3domain_usps_clip_ViT_B_16_openai_d512.npz",
        "digits-3domain_svhn_clip_ViT_B_16_openai_d512.npz"
    )
    foreach ($name in $required) {
        $path = Join-Path $cacheDir $name
        if (-not (Test-Path -LiteralPath $path)) {
            throw "required cache was not generated: $path"
        }
    }
    $elapsed = [math]::Round(((Get-Date) - $startedAt).TotalSeconds, 3)
    "CACHE_PREPARATION_COMPLETE=true`nelapsed_seconds=$elapsed" |
        Set-Content -LiteralPath $complete -Encoding UTF8
    Write-Output "CLIP ViT-B/16 MDDigits feature cache preparation completed in $elapsed seconds."
} catch {
    $_.Exception.ToString() | Set-Content -LiteralPath $failed -Encoding UTF8
    throw
}
