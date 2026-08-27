# Three-machine 10k concurrent availability validation

This validation does not train a backbone. It uses one deterministic,
serialized HeadOnly MLP state and proves four independent properties:

1. one root service and ten logical child servers are started on three
   physical machines;
2. the ten child servers retain 10,000 real TCP edge-client connections
   (1,000 per child server);
3. all edge clients perform a complete serialized MLP upload or download;
4. while those 10,000 connections remain present, one new edge client joins a
   child server and completes a full MLP download and upload round trip.

Formal business traffic is direct 10.x traffic only:

```text
third child registrations -> 4090 root port 62010 (10 connections)
8G edge load -> third child ports 62020-62029 (10,000 connections)
8G late edge probe -> third child port 62020
```

SSH may deploy, start, inspect, and collect artifacts, but no benchmark
`--connect-host` may be an SSH forward or loopback address.

## Metrics

The aggregate report keeps these metrics separate:

- `observed_load_connections_during_probe`: persistent concurrency;
- `aggregate_dispatch_start_qps`: synchronized transfer-start throughput;
- `aggregate_full_transfer_qps`: bandwidth-bound complete MLP transfer rate;
- `late_edge_probe_target_met`: full download + upload availability under load.

With the current 200 Mbps 8G and 400 Mbps third-node WLAN links, 10,000
complete 135 KiB round trips per second are physically impossible. This test
does not mislabel connection concurrency as full-payload QPS.

## Isolation

Do not run this test with the accuracy queue. The wrappers refuse to start
while the formal training roles/ports are active. Accuracy uses
60050/61000-61003; this test uses root port 62010 and child ports 62020-62029.

## Prepare the artifact

```powershell
python scripts\generate_headonly_mlp_state.py `
  --output exp\concurrent_availability\headonly_mlp_state.pt
```

Copy that exact artifact and the benchmark scripts to all three machines.
The SHA-256 must match in every command audit.

## Single-controller SSH run

The Windows controller can deploy and start every machine through management
SSH with one command. The business traffic still uses the real 10.x addresses;
SSH is only the control channel.

```powershell
python scripts\distributed_scripts\ggeur_concurrent_availability\run_t1_ssh_control.py start --run-id outline_t01
```

The command starts the root server, subservers, 10,000 load clients, and the
new-client probe. It then uses `Get-NetTCPConnection` on the load-client host
and `ss` on the Linux server to observe the real TCP connections. A successful
run prints `ESTABLISHED_CONNECTIONS=10000` (or a larger value) and writes:

```text
docs/test_logs/concurrent_availability/outline_t01/network_connection_monitor.tsv
docs/test_logs/concurrent_availability/outline_t01/network_connection_monitor.json
```

To observe an already running test again from the same controller:

```powershell
python scripts\distributed_scripts\ggeur_concurrent_availability\run_t1_ssh_control.py monitor --run-id outline_t01
```

## Synchronized formal run

Choose a start time at least two minutes in the future:

```text
START_AT_UNIX=<unix time>
OBSERVE_AT_UNIX=START_AT_UNIX+2
PROBE_AT_UNIX=START_AT_UNIX+2
```

On 4090, start `run_server_linux.sh` as one root service and accept ten child
registration requests. On the third Windows node, start
`run_server_windows.ps1` with ten child-server processes on ports 62020-62029.
On 8G, run `run_load_windows.ps1` with 1,000 edge clients per child server and
run `run_probe_windows.ps1` against one child server during the observation
window.

Collect the root registration summary, the child-server summary and the new
edge-client probe summary.  The 10,000-edge-client aggregate is calculated
from the child-server summary only:

```powershell
python scripts\summarize_concurrent_availability.py `
  --servers third_server_summary.json `
  --probe client_probe_summary.json `
  --target-connections 10000 `
  --target-qps 10000 `
  --output aggregate_summary.json
```

The availability requirement passes only when all ten child registrations
complete at the root service, all 10,000 persistent edge connections are
observed during the probe, all edge load transfers complete, the probe
completes both directions, all peer addresses are real 10.x addresses, and no
server error is recorded. The 10,000 QPS threshold remains a separate,
explicit result.
