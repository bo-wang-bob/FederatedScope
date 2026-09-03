param(
  [Parameter(Mandatory = $true)]
  [string]$RepoDir
)

$ErrorActionPreference = 'Stop'
$repo = [IO.Path]::GetFullPath($RepoDir)
$state = Join-Path $repo 'exp\domainnet_resource_download_dual'
$pipelineLog = Join-Path $state 'handoff.pipeline.log'
$stdoutLog = Join-Path $state 'handoff.stdout.log'
$stderrLog = Join-Path $state 'handoff.stderr.log'
$resourceTask = 'GGEUR-domainnet-resources-download-third'
$resourceScript = Join-Path $repo `
  'scripts\distributed_scripts\ggeur_hierarchical_3machine\download_domainnet_resources.ps1'
New-Item -ItemType Directory -Force -Path $state | Out-Null

function Write-Pipeline([string]$message) {
  $line = "$(Get-Date -Format o) $message"
  Add-Content -LiteralPath $pipelineLog -Value $line
  Write-Output $line
}

trap {
  $details = ($_ | Out-String).Trim()
  Add-Content -LiteralPath $stderrLog `
    -Value "$(Get-Date -Format o) $details"
  Write-Pipeline "HANDOFF_FAILED details=$details"
  exit 1
}

while ($true) {
  $task = Get-ScheduledTask -TaskName $resourceTask `
    -ErrorAction SilentlyContinue
  if ($null -eq $task -or $task.State -ne 'Running') {
    break
  }
  Write-Pipeline "WAIT_RESOURCE_TASK state=$($task.State)"
  Start-Sleep -Seconds 60
}

if (-not (Test-Path -LiteralPath $resourceScript)) {
  throw "latest resource script is missing: $resourceScript"
}
Write-Pipeline "HANDOFF_START script=$resourceScript"
& $resourceScript -RepoDir $repo 1>> $stdoutLog 2>> $stderrLog
if ($LASTEXITCODE -ne 0) {
  throw "latest resource pipeline failed: $LASTEXITCODE"
}
$completed = [ordered]@{
  completed_at = (Get-Date).ToString('o')
  resource_script = $resourceScript
  resource_completion = (Join-Path $state 'completed.json')
}
$completed | ConvertTo-Json -Depth 3 | Set-Content `
  -LiteralPath (Join-Path $state 'handoff.completed.json') -Encoding UTF8
Write-Pipeline 'HANDOFF_COMPLETE'
