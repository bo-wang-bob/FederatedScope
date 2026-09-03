param(
    [string]$RepoDir = "C:\Users\pc\FederatedScope",
    [string]$DataRoot = "C:\Users\pc\FederatedScope\data\DomainNet"
)

$ErrorActionPreference = "Stop"
$StateDir = Join-Path $RepoDir "exp\domainnet_resource_download_dual"
$DownloadDir = Join-Path $StateDir "downloads"
$ModelDir = Join-Path $RepoDir "pretrained_models"
$LogPath = Join-Path $StateDir "pipeline.log"
New-Item -ItemType Directory -Force -Path $StateDir, $DownloadDir, $ModelDir, $DataRoot | Out-Null

function Write-State([string]$Message) {
    $line = "{0} {1}" -f (Get-Date).ToString("o"), $Message
    Add-Content -LiteralPath $LogPath -Value $line -Encoding UTF8
    Write-Output $line
}

function Get-ValidFile([string]$Path, [long]$MinimumBytes) {
    $item = Get-Item -LiteralPath $Path -ErrorAction SilentlyContinue
    return ($null -ne $item -and $item.Length -ge $MinimumBytes)
}

function Download-Resource(
    [string]$Name,
    [string]$Url,
    [string]$Destination,
    [long]$MinimumBytes,
    [long]$ExpectedBytes = 0
) {
    $destinationItem = Get-Item -LiteralPath $Destination -ErrorAction SilentlyContinue
    $destinationValid = ($null -ne $destinationItem -and
        $destinationItem.Length -ge $MinimumBytes -and
        ($ExpectedBytes -le 0 -or $destinationItem.Length -eq $ExpectedBytes))
    if ($destinationValid) {
        Write-State "DOWNLOAD_REUSE name=$Name path=$Destination bytes=$((Get-Item -LiteralPath $Destination).Length)"
        return
    }

    if (Test-Path -LiteralPath $Destination) {
        $bad = "$Destination.bad_$((Get-Date).ToString('yyyyMMdd_HHmmss'))"
        Move-Item -LiteralPath $Destination -Destination $bad
        Write-State "DOWNLOAD_PRESERVE_INVALID name=$Name path=$bad"
    }

    $partial = "$Destination.part"
    if ($ExpectedBytes -gt 0) {
        $partialItem = Get-Item -LiteralPath $partial -ErrorAction SilentlyContinue
        if ($null -ne $partialItem -and $partialItem.Length -gt $ExpectedBytes) {
            $bad = "$partial.bad_$((Get-Date).ToString('yyyyMMdd_HHmmss'))"
            Move-Item -LiteralPath $partial -Destination $bad
            Write-State "DOWNLOAD_PRESERVE_OVERSIZE name=$Name path=$bad bytes=$($partialItem.Length)"
        }

        $chunkBytes = 64MB
        while ($true) {
            $partialItem = Get-Item -LiteralPath $partial -ErrorAction SilentlyContinue
            [long]$start = if ($null -eq $partialItem) { 0 } else { $partialItem.Length }
            if ($start -eq $ExpectedBytes) { break }
            [long]$end = [Math]::Min($start + $chunkBytes - 1, $ExpectedBytes - 1)
            [long]$requiredChunkBytes = $end - $start + 1
            $chunk = "$partial.chunk_${start}_${end}"
            if (Test-Path -LiteralPath $chunk) {
                $failedChunk = "$chunk.failed_$((Get-Date).ToString('yyyyMMdd_HHmmss'))"
                Move-Item -LiteralPath $chunk -Destination $failedChunk
                Write-State "DOWNLOAD_PRESERVE_CHUNK name=$Name path=$failedChunk"
            }
            $arguments = @(
                "--location", "--fail", "--silent", "--show-error",
                "--ssl-no-revoke", "--retry", "100", "--retry-all-errors",
                "--retry-delay", "30", "--connect-timeout", "30",
                "--speed-time", "300", "--speed-limit", "1024",
                "--range", "${start}-${end}", "--output", $chunk, $Url
            )
            Write-State "DOWNLOAD_CHUNK_START name=$Name start=$start end=$end expected=$ExpectedBytes"
            $previousErrorActionPreference = $ErrorActionPreference
            $ErrorActionPreference = "Continue"
            & curl.exe @arguments
            $curlExitCode = $LASTEXITCODE
            $ErrorActionPreference = $previousErrorActionPreference
            if ($curlExitCode -ne 0) {
                throw "curl failed for $Name range ${start}-${end} with exit code $curlExitCode"
            }
            $chunkItem = Get-Item -LiteralPath $chunk -ErrorAction SilentlyContinue
            if ($null -eq $chunkItem -or $chunkItem.Length -ne $requiredChunkBytes) {
                $actualChunkBytes = if ($null -eq $chunkItem) { 0 } else { $chunkItem.Length }
                throw "range size mismatch for ${Name}: expected $requiredChunkBytes, got $actualChunkBytes"
            }

            $destinationStream = $null
            $sourceStream = $null
            try {
                $destinationStream = [IO.File]::Open(
                    $partial, [IO.FileMode]::Append, [IO.FileAccess]::Write,
                    [IO.FileShare]::Read)
                $sourceStream = [IO.File]::OpenRead($chunk)
                $sourceStream.CopyTo($destinationStream)
                $destinationStream.Flush($true)
            } finally {
                if ($null -ne $sourceStream) { $sourceStream.Dispose() }
                if ($null -ne $destinationStream) { $destinationStream.Dispose() }
            }
            Remove-Item -LiteralPath $chunk -Force
            $currentBytes = (Get-Item -LiteralPath $partial).Length
            Write-State "DOWNLOAD_CHUNK_COMPLETE name=$Name bytes=$currentBytes expected=$ExpectedBytes"
        }
        if (-not (Get-ValidFile $partial $MinimumBytes) -or
            (Get-Item -LiteralPath $partial).Length -ne $ExpectedBytes) {
            throw "completed chunk download has an invalid size for $Name"
        }
        Move-Item -LiteralPath $partial -Destination $Destination -Force
        Write-State "DOWNLOAD_COMPLETE name=$Name path=$Destination bytes=$((Get-Item -LiteralPath $Destination).Length)"
        return
    }

    $arguments = @(
        "--location", "--fail", "--silent", "--show-error", "--ssl-no-revoke",
        "--retry", "100", "--retry-all-errors", "--retry-delay", "30",
        "--connect-timeout", "30", "--speed-time", "300", "--speed-limit", "1024"
    )
    if (Test-Path -LiteralPath $partial) {
        $arguments += @("--continue-at", "-")
    }
    $arguments += @("--output", $partial, $Url)

    Write-State "DOWNLOAD_START name=$Name url=$Url partial=$partial"
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & curl.exe @arguments
    $curlExitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousErrorActionPreference
    if ($curlExitCode -ne 0) {
        throw "curl failed for $Name with exit code $curlExitCode"
    }
    if (-not (Get-ValidFile $partial $MinimumBytes)) {
        $bytes = (Get-Item -LiteralPath $partial -ErrorAction SilentlyContinue).Length
        throw "download too small for ${Name}: $bytes bytes"
    }
    Move-Item -LiteralPath $partial -Destination $Destination -Force
    Write-State "DOWNLOAD_COMPLETE name=$Name path=$Destination bytes=$((Get-Item -LiteralPath $Destination).Length)"
}

$archives = @(
    @{ Name = "clipart"; Url = "https://csr.bu.edu/ftp/visda/2019/multi-source/clipart.zip"; MinimumBytes = 1000000000; ExpectedBytes = 1272750339 },
    @{ Name = "painting"; Url = "https://csr.bu.edu/ftp/visda/2019/multi-source/painting.zip"; MinimumBytes = 3000000000; ExpectedBytes = 3682180186 },
    @{ Name = "real"; Url = "https://csr.bu.edu/ftp/visda/2019/multi-source/real.zip"; MinimumBytes = 5000000000; ExpectedBytes = 6062039589 },
    @{ Name = "sketch"; Url = "https://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip"; MinimumBytes = 2000000000; ExpectedBytes = 2633992926 }
)

foreach ($archive in $archives) {
    $zipPath = Join-Path $DownloadDir "$($archive.Name).zip"
    Download-Resource $archive.Name $archive.Url $zipPath `
        $archive.MinimumBytes $archive.ExpectedBytes

    $marker = Join-Path $StateDir "$($archive.Name).extracted.json"
    $domainDir = Join-Path $DataRoot $archive.Name
    if ((Test-Path -LiteralPath $marker) -and (Test-Path -LiteralPath $domainDir)) {
        Write-State "EXTRACT_REUSE name=$($archive.Name) path=$domainDir"
        continue
    }

    Write-State "EXTRACT_START name=$($archive.Name) archive=$zipPath destination=$DataRoot"
    & tar.exe -xf $zipPath -C $DataRoot
    if ($LASTEXITCODE -ne 0) {
        throw "tar extraction failed for $($archive.Name) with exit code $LASTEXITCODE"
    }
    if (-not (Test-Path -LiteralPath $domainDir)) {
        throw "extracted domain directory missing: $domainDir"
    }
    $payload = [ordered]@{
        domain = $archive.Name
        archive = $zipPath
        archive_bytes = (Get-Item -LiteralPath $zipPath).Length
        extracted_dir = $domainDir
        completed_at = (Get-Date).ToString("o")
    }
    $payload | ConvertTo-Json | Set-Content -LiteralPath $marker -Encoding UTF8
    Write-State "EXTRACT_COMPLETE name=$($archive.Name) path=$domainDir"
}

Download-Resource `
    "open_clip_vitb16" `
    "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt" `
    (Join-Path $ModelDir "ViT-B-16.pt") `
    300000000 `
    350837078
Download-Resource `
    "convnext_base" `
    "https://download.pytorch.org/models/convnext_base-6075fbad.pth" `
    (Join-Path $ModelDir "convnext_base-6075fbad.pth") `
    300000000 `
    354486097
Download-Resource `
    "mixer_b16_224" `
    "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_mixer_b16_224-76587d61.pth" `
    (Join-Path $ModelDir "mixer_b16_224_complete.pth") `
    200000000 `
    239544439

$manifest = Join-Path $RepoDir "exp\distributed_manifests\domainnet_4domains\domainnet_manifest.json"
if (-not (Test-Path -LiteralPath $manifest)) {
    throw "DomainNet manifest is missing: $manifest"
}

$complete = [ordered]@{
    source = "https://csr.bu.edu/ftp/visda/2019/multi-source/"
    domains = @($archives | ForEach-Object { $_.Name })
    data_root = $DataRoot
    manifest = $manifest
    clip_checkpoint = (Join-Path $ModelDir "ViT-B-16.pt")
    convnext_checkpoint = (Join-Path $ModelDir "convnext_base-6075fbad.pth")
    mixer_checkpoint = (Join-Path $ModelDir "mixer_b16_224_complete.pth")
    completed_at = (Get-Date).ToString("o")
}
$complete | ConvertTo-Json -Depth 4 | Set-Content `
    -LiteralPath (Join-Path $StateDir "completed.json") -Encoding UTF8
Write-State "PIPELINE_COMPLETE"
