param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$Controller = Join-Path $PSScriptRoot "remote_outline_control.py"

function Invoke-Control {
    param([string[]]$Arguments)
    $raw = (& $Python $Controller @Arguments | Out-String)
    if ($LASTEXITCODE -ne 0) {
        throw "controller failed: $($Arguments -join ' ')`n$raw"
    }
    return ($raw | ConvertFrom-Json)
}

function Get-State {
    param([string]$CaseId, [string]$Method)
    $result = Invoke-Control -Arguments @(
        "status", "--case", $CaseId, "--method", $Method
    )
    $state = if ($result.stdout -match "state=([A-Z]+)") {
        $Matches[1]
    } else {
        "UNKNOWN"
    }
    $rounds = if ($result.stdout -match "round_accuracy_lines=(\d+)") {
        [int]$Matches[1]
    } else {
        0
    }
    return [pscustomobject]@{
        State = $state
        Rounds = $rounds
        Text = [string]$result.stdout
    }
}

function Wait-Run {
    param([string]$CaseId, [string]$Method)
    $lastRounds = -1
    while ($true) {
        $status = Get-State -CaseId $CaseId -Method $Method
        if ($status.Rounds -ne $lastRounds) {
            Write-Output ("PROGRESS case={0} method={1} rounds={2} state={3}" -f `
                $CaseId, $Method, $status.Rounds, $status.State)
            $lastRounds = $status.Rounds
        }
        if ($status.State -eq "FINISHED") {
            $elapsed = if ($status.Text -match '"elapsed_seconds":\s*([0-9.]+)') {
                $Matches[1]
            } else {
                "unknown"
            }
            $accuracy = if ($status.Text -match 'average: final=([0-9.]+)') {
                $Matches[1]
            } else {
                "unknown"
            }
            Write-Output ("DONE case={0} method={1} accuracy={2} elapsed_seconds={3}" -f `
                $CaseId, $Method, $accuracy, $elapsed)
            return
        }
        if ($status.State -eq "FAILED") {
            throw "run failed: $CaseId $Method"
        }
        Start-Sleep -Seconds 10
    }
}

Write-Output "WAIT case=T-02 method=platform"
Wait-Run -CaseId "T-02" -Method "platform"

$runs = @(
    @("T-03", "fedavg"),
    @("T-03", "fedprox"),
    @("T-03", "platform"),
    @("T-04", "fedavg"),
    @("T-04", "fedprox"),
    @("T-04", "platform"),
    @("T-05", "fedavg"),
    @("T-05", "fedprox"),
    @("T-05", "platform"),
    @("T-06", "fedavg"),
    @("T-06", "fedprox"),
    @("T-06", "platform")
)

foreach ($run in $runs) {
    $caseId = $run[0]
    $method = $run[1]
    Write-Output ("START case={0} method={1}" -f $caseId, $method)
    $null = Invoke-Control -Arguments @(
        "start", "--case", $caseId, "--method", $method
    )
    Wait-Run -CaseId $caseId -Method $method
}

Write-Output "ALL_REMAINING_FORMAL_RUNS_COMPLETE"
