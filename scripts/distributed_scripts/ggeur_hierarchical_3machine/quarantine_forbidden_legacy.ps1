param(
  [Parameter(Mandatory = $true)]
  [string]$RepoDir
)

$ErrorActionPreference = 'Stop'
$repo = [IO.Path]::GetFullPath($RepoDir)
$forbidden = '10.112.81.135'
$legacyRuns = @(
  'mdsent_reference_distributed_20260730_v1',
  'mdsent_baselines_corrected_20260731_v1',
  'domainnet_remaining_20260731_v1',
  'remaining_20260731_v1'
)

function Test-Legacy([string]$command) {
  if ($command -like "*$forbidden*") {
    return $true
  }
  foreach ($run in $legacyRuns) {
    if ($command -like "*$run*") {
      return $true
    }
  }
  return $false
}

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$audit = Join-Path $repo "exp\dual_topology_quarantine\$stamp"
New-Item -ItemType Directory -Force -Path $audit | Out-Null
$tasks = @()
foreach ($task in @(Get-ScheduledTask -ErrorAction SilentlyContinue)) {
  $definition = (($task.Actions | ForEach-Object {
    "$($_.Execute) $($_.Arguments) $($_.WorkingDirectory)"
  }) -join ' ')
  if (Test-Legacy $definition) {
    $tasks += $task
  }
}
$taskEvidence = foreach ($task in $tasks | Sort-Object TaskName) {
  $info = Get-ScheduledTaskInfo -TaskName $task.TaskName
  $arguments = (($task.Actions | ForEach-Object { $_.Arguments }) -join ' ')
  "$($task.TaskName)`t$($task.State)`t$($info.LastTaskResult)`t$arguments"
}
$taskEvidence | Set-Content `
  -LiteralPath (Join-Path $audit 'tasks.tsv') -Encoding UTF8
$processes = @(Get-CimInstance Win32_Process | Where-Object {
  Test-Legacy ([string]$_.CommandLine)
})
$processEvidence = foreach ($process in $processes | Sort-Object ProcessId) {
  "$($process.ProcessId)`t$($process.ParentProcessId)`t$($process.Name)`t$($process.CommandLine)"
}
$processEvidence | Set-Content `
  -LiteralPath (Join-Path $audit 'processes.tsv') -Encoding UTF8
$connections = @(Get-NetTCPConnection -ErrorAction SilentlyContinue |
  Where-Object { $_.RemoteAddress -eq $forbidden })
$connectionEvidence = foreach ($connection in $connections |
    Sort-Object OwningProcess) {
  "$($connection.OwningProcess)`t$($connection.State)`t$($connection.LocalAddress):$($connection.LocalPort)`t$($connection.RemoteAddress):$($connection.RemotePort)"
}
$connectionEvidence | Set-Content `
  -LiteralPath (Join-Path $audit 'connections.tsv') -Encoding UTF8

foreach ($task in $tasks) {
  if ($task.State -eq 'Running') {
    Stop-ScheduledTask -TaskName $task.TaskName `
      -ErrorAction SilentlyContinue
  }
  Disable-ScheduledTask -TaskName $task.TaskName `
    -ErrorAction SilentlyContinue | Out-Null
}
Start-Sleep -Seconds 2
$stopped = 0
foreach ($process in @(Get-CimInstance Win32_Process | Where-Object {
  Test-Legacy ([string]$_.CommandLine)
}) | Sort-Object ProcessId -Descending) {
  Stop-Process -Id $process.ProcessId -Force -ErrorAction SilentlyContinue
  $stopped += 1
}
Start-Sleep -Seconds 3
$remainingProcesses = @(Get-CimInstance Win32_Process | Where-Object {
  Test-Legacy ([string]$_.CommandLine)
})
$remainingConnections = @(Get-NetTCPConnection -ErrorAction SilentlyContinue |
  Where-Object { $_.RemoteAddress -eq $forbidden })

"audit=$audit"
"quarantined_tasks=$($tasks.Count)"
"stopped_process_attempts=$stopped"
"remaining_legacy_processes=$($remainingProcesses.Count)"
"remaining_forbidden_connections=$($remainingConnections.Count)"
foreach ($taskName in @(
  'GGEUR-domainnet-resources-download-third',
  'GGEUR-domainnet-resource-handoff-third',
  'GGEUR-domainnet-eval-cache-third'
)) {
  $task = Get-ScheduledTask -TaskName $taskName `
    -ErrorAction SilentlyContinue
  if ($task) {
    $info = Get-ScheduledTaskInfo -TaskName $taskName
    "protected_task=$taskName state=$($task.State) result=$($info.LastTaskResult)"
  } else {
    "protected_task=$taskName state=MISSING"
  }
}
if ($remainingProcesses.Count -ne 0 -or
    $remainingConnections.Count -ne 0) {
  throw 'legacy topology quarantine incomplete'
}
