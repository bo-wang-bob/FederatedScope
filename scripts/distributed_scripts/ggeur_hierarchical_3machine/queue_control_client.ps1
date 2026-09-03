function Get-QueueControlState {
    param(
        [Parameter(Mandatory = $true)][string]$ControlUrl
    )
    try {
        return Invoke-RestMethod -UseBasicParsing `
            -Uri "$($ControlUrl.TrimEnd('/'))/state" `
            -TimeoutSec 5 -ErrorAction Stop
    } catch {
        return $null
    }
}

function Wait-QueueControlState {
    param(
        [Parameter(Mandatory = $true)][string]$ControlUrl,
        [Parameter(Mandatory = $true)][string]$RunId,
        [Parameter(Mandatory = $true)][string]$CaseName,
        [Parameter(Mandatory = $true)][int]$CaseIndex,
        [Parameter(Mandatory = $true)][string[]]$DesiredPhases,
        [int]$TimeoutSec = 172800,
        [int]$PollSec = 5
    )
    $deadline = if ($TimeoutSec -gt 0) {
        [DateTime]::UtcNow.AddSeconds($TimeoutSec)
    } else {
        [DateTime]::MaxValue
    }
    while ([DateTime]::UtcNow -lt $deadline) {
        $state = Get-QueueControlState -ControlUrl $ControlUrl
        if ($null -eq $state -or $state.run_id -ne $RunId) {
            Start-Sleep -Seconds $PollSec
            continue
        }
        $completedCount = [int]$state.completed_count
        if ($DesiredPhases -contains "complete" -and
            $completedCount -gt $CaseIndex) {
            return $state
        }
        if ([int]$state.case_index -eq $CaseIndex -and
            [string]$state.case -eq $CaseName) {
            if ([string]$state.phase -eq "failed") {
                throw "Root queue reported failure for ${CaseName}: $($state.message)"
            }
            if ($DesiredPhases -contains [string]$state.phase) {
                return $state
            }
        }
        if ([int]$state.case_index -gt $CaseIndex -and
            -not ($DesiredPhases -contains "complete")) {
            throw "Root queue advanced past $CaseName without this role queue observing its start"
        }
        Start-Sleep -Seconds $PollSec
    }
    throw "Timed out waiting for root control state: case=$CaseName phases=$($DesiredPhases -join ',')"
}
