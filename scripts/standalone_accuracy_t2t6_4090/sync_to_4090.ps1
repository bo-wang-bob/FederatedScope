param(
    [string]$Remote = "fs-server-4090",
    [string]$RemoteRepo = "/root/autodl-tmp/FederatedScope"
)

$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path

ssh $Remote "mkdir -p '$RemoteRepo/scripts/standalone_accuracy_t2t6_4090' '$RemoteRepo/scripts/example_configs' '$RemoteRepo/scripts/test_outline_validation'"
scp -r (Join-Path $Repo "federatedscope") "${Remote}:${RemoteRepo}/"
scp -r (Join-Path $Repo "scripts\standalone_accuracy_t2t6_4090") "${Remote}:${RemoteRepo}/scripts/"
scp -r (Join-Path $Repo "scripts\example_configs\ggeur_final_5models") "${Remote}:${RemoteRepo}/scripts/example_configs/"
scp -r (Join-Path $Repo "scripts\test_outline_validation\task_profiles") "${Remote}:${RemoteRepo}/scripts/test_outline_validation/"

Write-Output "T2-T6 files uploaded to ${Remote}:${RemoteRepo}"
