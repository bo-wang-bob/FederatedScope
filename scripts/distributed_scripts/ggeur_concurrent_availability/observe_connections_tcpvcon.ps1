param(
    [Parameter(Mandatory = $true)][string]$RunId,
    [int]$TargetConnections = 10000
)

$ErrorActionPreference = "Stop"
$ProjectDir = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$ToolDir = Join-Path $ProjectDir "tools\Sysinternals\TCPView"
$Tcpvcon = Join-Path $ToolDir "tcpvcon64.exe"
$OutputDir = Join-Path $ProjectDir "exp\concurrent_availability\$RunId\network_observation"
$RawCsv = Join-Path $OutputDir "tcpvcon_connections.csv"
$SummaryPath = Join-Path $OutputDir "tcpvcon_connection_summary.json"

New-Item -ItemType Directory -Force -Path $ToolDir, $OutputDir | Out-Null

if (-not (Test-Path -LiteralPath $Tcpvcon)) {
    $ZipPath = Join-Path $ToolDir "TCPView.zip"
    Invoke-WebRequest -UseBasicParsing `
        -Uri "https://download.sysinternals.com/files/TCPView.zip" `
        -OutFile $ZipPath
    Expand-Archive -LiteralPath $ZipPath -DestinationPath $ToolDir -Force
    $Candidate = Get-ChildItem -LiteralPath $ToolDir -Recurse -File |
        Where-Object { $_.Name -in @("tcpvcon64.exe", "tcpvcon.exe") } |
        Select-Object -First 1
    if ($null -eq $Candidate) {
        throw "Tcpvcon executable was not found in the Sysinternals archive"
    }
    $Tcpvcon = $Candidate.FullName
}

$OutputLines = @(& $Tcpvcon -acn -accepteula 2>&1 | ForEach-Object { "$_" })
$DataLines = @($OutputLines | Where-Object { $_ -match '^(TCP|TCPV6),' })
if ($DataLines.Count -eq 0) {
    throw "Tcpvcon did not return any TCP connection rows"
}
$Header = 'Protocol,Process,PID,State,Local Address,Remote Address'
$CsvLines = @($Header) + $DataLines
$CsvLines | Set-Content -LiteralPath $RawCsv -Encoding UTF8
$Connections = @($DataLines | ConvertFrom-Csv -Header @(
    'Protocol', 'Process', 'PID', 'State', 'Local Address', 'Remote Address'))
$Observed = @($Connections | Where-Object {
    $State = "$(if ($_.PSObject.Properties['State']) { $_.State })"
    $Process = "$(if ($_.PSObject.Properties['Process']) { $_.Process })"
    $RemoteAddress = "$(if ($_.PSObject.Properties['Remote Address']) { $_.'Remote Address' })"
    $State -eq "ESTABLISHED" -and $Process -match '^python(\.exe)?$' -and
        $RemoteAddress -eq '10.129.248.111'
})

$Summary = [ordered]@{
    run_id = $RunId
    observation_tool = "Microsoft Sysinternals Tcpvcon"
    observed_at = (Get-Date).ToString("o")
    target_connections = $TargetConnections
    observed_process = "python.exe"
    observed_remote_host = "10.129.248.111"
    expected_remote_ports = "62020-62029"
    observed_established_connections = $Observed.Count
    target_reached = ($Observed.Count -ge $TargetConnections)
    raw_csv = "exp/concurrent_availability/$RunId/network_observation/tcpvcon_connections.csv"
}
$Summary | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $SummaryPath -Encoding UTF8

"THIRD_PARTY_NETWORK_TOOL=Microsoft Sysinternals Tcpvcon"
"OBSERVED_ESTABLISHED_CONNECTIONS=$($Observed.Count)"
"TARGET_CONNECTIONS=$TargetConnections"
"TCPVCON_CONNECTION_CSV=exp/concurrent_availability/$RunId/network_observation/tcpvcon_connections.csv"
"TCPVCON_SUMMARY=exp/concurrent_availability/$RunId/network_observation/tcpvcon_connection_summary.json"
if (-not $Summary.target_reached) {
    throw "Tcpvcon observed $($Observed.Count) established connections; target is $TargetConnections"
}
"THIRD_PARTY_NETWORK_OBSERVATION=PASS"
