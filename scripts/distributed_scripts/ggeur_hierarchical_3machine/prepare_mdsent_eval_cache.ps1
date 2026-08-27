param(
  [Parameter(Mandatory = $true)][string]$Repo,
  [Parameter(Mandatory = $true)][string]$Python
)

$ErrorActionPreference = 'Stop'
$state = Join-Path $Repo 'exp\distributed_feature_cache\mdsent_cache_prepare'
$worker = Join-Path $Repo `
  'scripts\distributed_scripts\ggeur_hierarchical_3machine\prepare_mdsent_eval_cache.py'
New-Item -ItemType Directory -Force -Path $state | Out-Null
$stdout = Join-Path $state 'prepare.stdout.log'
$stderr = Join-Path $state 'prepare.stderr.log'
$audit = Join-Path $state 'launch_audit.tsv'
$exitCode = Join-Path $state 'exit_code.txt'

$timestamp = Get-Date -Format o
Add-Content -LiteralPath $audit -Value "$timestamp`tSTART`t$Python -u $worker"
try {
  # Windows PowerShell promotes a native program's first stderr line into a
  # terminating NativeCommandError when ErrorActionPreference is Stop.  Model
  # loaders legitimately emit progress/warnings on stderr, so preserve the
  # complete stream and rely on the native exit code for success or failure.
  $ErrorActionPreference = 'Continue'
  & $Python -u $worker --repo $Repo --groups mdsent_rnn mdsent_lstm `
    1>> $stdout 2>> $stderr
  $code = $LASTEXITCODE
} catch {
  $_ | Out-String | Add-Content -LiteralPath $stderr
  $code = 1
}
Set-Content -LiteralPath $exitCode -Value $code
$timestamp = Get-Date -Format o
Add-Content -LiteralPath $audit -Value "$timestamp`tEXIT`t$code"
exit $code
