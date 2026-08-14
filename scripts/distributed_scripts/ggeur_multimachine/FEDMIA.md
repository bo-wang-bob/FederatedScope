# Multi-machine GGEUR FedMIA-I/II

This path runs the normal real GGEUR distributed protocol and adds optional
FedMIA benchmark reports as a side channel. Each client keeps its raw images
and feature vectors local. At attack rounds the server sends the small local
MLP heads back to the clients; clients evaluate those heads on local
member/test/mix probes and upload only derived loss and gradient-cosine arrays.
This aligns the distributed path with the standalone GGEUR FedMIA target,
shadow, global-test, mix-nonmember, indexed calibration, and cross-round flow.

The instrumentation is disabled by default. Normal distributed experiments
are unchanged unless `attack.distributed_fedmia: true` is present.

## Generate a case

Run on the machine that prepares the case directory:

```bash
cd /root/autodl-tmp/zqq/FederatedScope-feature-GGEUR-git

PYTHON_BIN=/root/autodl-tmp/zqq/.conda/envs/fs_zqq/bin/python
CASE_ROOT=scripts/distributed_scripts/ggeur_multimachine/runs/fedmia_demo

$PYTHON_BIN scripts/distributed_scripts/ggeur_multimachine/generate_configs.py \
  --output-root "$CASE_ROOT" \
  --run-id fedmia_demo \
  --dataset officehome \
  --model cnn \
  --method fedavg \
  --head-only \
  --server-host 127.0.0.1 \
  --server-bind-host 127.0.0.1 \
  --server-port 55051 \
  --client-host 127.0.0.1 \
  --client-bind-host 127.0.0.1 \
  --client-port-base 56000 \
  --clients 4 \
  --sample-clients 4 \
  --rounds 100 \
  --data-root /root/autodl-tmp/zqq/OfficeHomeDataset_10072016 \
  --feature-cache-dir exp/distributed_fedmia/cache \
  --distributed-fedmia \
  --fedmia-target-client-id 3 \
  --fedmia-probe-size 200 \
  --fedmia-nonmember-size 0 \
  --fedmia-save-interval 10 \
  --fedmia-shadow-stat-mode indexed \
  --fedmia-mode mix \
  --fedmia-mix-length 1000 \
  --fedmia-ggeur-target-size 200 \
  --fedmia-normalize-fields train_losses
```

The case directory is:

```text
scripts/distributed_scripts/ggeur_multimachine/runs/fedmia_demo/officehome_cnn_fedavg_headonly
```

## Local multi-process validation

```bash
CASE_DIR=scripts/distributed_scripts/ggeur_multimachine/runs/fedmia_demo/officehome_cnn_fedavg_headonly
PYTHON_BIN=/root/autodl-tmp/zqq/.conda/envs/fs_zqq/bin/python \
  bash scripts/distributed_scripts/ggeur_multimachine/launch_server.sh "$CASE_DIR"

PYTHON_BIN=/root/autodl-tmp/zqq/.conda/envs/fs_zqq/bin/python \
CLIENT_START_GAP=2 \
  bash scripts/distributed_scripts/ggeur_multimachine/launch_clients.sh "$CASE_DIR"
```

## Real multi-machine use

Generate the case with the server's reachable address in `--server-host`, the
server bind address in `--server-bind-host`, and client callback addresses in
`--client-host`/`--client-hosts`. Copy the generated case directory and the
same code revision to both machines. Start `launch_server.sh` on the server
machine and `launch_clients.sh` on the client machine.

Both server and clients must use configs containing the same `attack` block.
The server does not need the clients' raw OfficeHome images for FedMIA score
collection; clients compute the derived probe scores locally.

## Outputs

Server output contains:

```text
distributed_fedmia_reports/client_<id>_round_<round>.pt
distributed_fedmia_results.json
```

The result JSON reports target-client FedMIA-I and FedMIA-II AUC and
TPR at FPR 0.001, 0.01, and 0.1. Set
`attack.fedmia_compute_all_clients: true` to evaluate every client as target.

For a lightweight loss-only run, pass `--no-fedmia-store-grad-cos`; FedMIA-I
will run and FedMIA-II will be reported as unavailable.
