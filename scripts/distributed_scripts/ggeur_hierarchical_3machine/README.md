# GGEUR three-machine hierarchical experiments

This directory turns every config under `ggeur_final_5models` into a real
three-machine distributed case:

```text
8G Windows clients (one logical client = one process + one listening port)
  -> Windows subserver host (all subservers, one process + port each)
  -> 4090 Linux root server (one root process)
```

The subserver is message-aware. Client join/statistics/augmentation/evaluation
messages remain per-client. Model updates are aggregated locally with
sample-count weights, so the root receives one model update per subserver per
round. The root still owns global FedAvg/FedOpt and evaluation.

## Experiment matrix

The generator reads the source configs rather than maintaining a second copy:

- vision models: ViT, ConvNeXt CNN, MLP-Mixer
- text models: RNN, LSTM
- datasets: OfficeHome, DomainNet, Multi-Domain Sentiment
- methods: FedAvg, FedProx, FedProto, FedOpt, GGEUR; vision also includes MOON

The full matrix currently contains 46 cases.

Client configs default to CPU execution because every logical client is an
independent process. Frozen ViT/CNN/Mixer/BERT features are prepared once and
shared through a read-only raw-feature cache; this prevents dozens of client
processes from loading the frozen backbone into the 8G GPU at the same time.
The lightweight worker path also avoids eager TensorFlow, SciPy and
scikit-learn imports. On the 8G host, a ten-client listen-only smoke test used
about 2.02 GB total working set (about 202.5 MB per client).

Raw feature-cache keys are dataset-relative and case-insensitive, so a cache
prepared on Linux can be copied to Windows. OfficeHome additionally uses the
same explicit `client_XXXXXX/client_manifest.json` set on both operating
systems; this avoids filesystem enumeration order changing LDS partitions.
For OfficeHome ViT, the validated
weight is `/root/.cache/clip/ViT-B-16.pt` on the 4090 host. Prepare the cache
there with the original client count and split:

```bash
bash scripts/distributed_scripts/ggeur_hierarchical_3machine/prepare_feature_cache.sh \
  officehome_vit
```

After copying the resulting `exp/distributed_feature_cache/officehome_vit`
directory to the same relative repository location on 8G, run the Windows
preparer once as a cache-completeness check. It must finish without trying to
load a backbone and writes the local ready marker:

```powershell
.\scripts\distributed_scripts\ggeur_hierarchical_3machine\prepare_feature_cache.ps1 `
  -Group officehome_vit `
  -PythonBin D:\ProgramData\anaconda3\envs\pi_fmd_gpu_py39\python.exe
```

## Generate all cases

```powershell
python scripts/distributed_scripts/ggeur_hierarchical_3machine/generate_matrix.py `
  --run-id final_three_machine_YYYYMMDD
```

Generate only the first requested group:

```powershell
python scripts/distributed_scripts/ggeur_hierarchical_3machine/generate_matrix.py `
  --run-id officehome_vit_three_machine_YYYYMMDD `
  --groups officehome_vit
```

The generator keeps the original client count. By default, client IDs 1-60
are assigned to the 8G Windows host. Any overflow IDs (for example IDs 61-120
in text cases) receive Linux paths and configs under `clients_4090`; use
`--windows-client-count` to change this split. Independently, 30 clients are
assigned to each subserver, giving two subservers for 60-client vision cases
and four subservers for 120-client text cases.

## Start one case in order

1. Root host:

```bash
PYTHON_BIN=/root/.local/share/mamba/envs/GGEUR/bin/python \
  bash scripts/distributed_scripts/ggeur_hierarchical_3machine/launch_root.sh \
  scripts/distributed_scripts/ggeur_hierarchical_3machine/runs/<run_id>/<case>
```

Use a Python interpreter that can import PyTorch explicitly. Non-interactive
SSH sessions may resolve `python3` outside the prepared GGEUR environment.

2. Windows subserver host (use scheduled tasks for SSH-launched processes):

```powershell
.\scripts\distributed_scripts\ggeur_hierarchical_3machine\launch_subservers.ps1 `
  -CaseDir scripts\distributed_scripts\ggeur_hierarchical_3machine\runs\<run_id>\<case> `
  -PythonBin C:\Users\pc\miniconda3\envs\cerp\python.exe `
  -UseScheduledTasks
```

3. 8G client host:

```powershell
.\scripts\distributed_scripts\ggeur_hierarchical_3machine\launch_clients.ps1 `
  -CaseDir scripts\distributed_scripts\ggeur_hierarchical_3machine\runs\<run_id>\<case> `
  -UseScheduledTasks
```

If a generated case has configs under `configs/clients_4090`, start those on
the root host after its cache is ready:

```bash
bash scripts/distributed_scripts/ggeur_hierarchical_3machine/launch_root_clients.sh \
  scripts/distributed_scripts/ggeur_hierarchical_3machine/runs/<run_id>/<case>
```

For the first final `officehome_vit_three_machine_20260721_v5` run there are 60
`clients_8g` configs and no `clients_4090` configs. Client ports are
20000-20059, subserver ports are 61000-61001, and the root port is 60050.

Use the matching PowerShell status/stop scripts on Windows. The stop scripts
validate recorded PIDs and unregister the case-scoped scheduled tasks:

```powershell
.\scripts\distributed_scripts\ggeur_hierarchical_3machine\status_clients.ps1 `
  -CaseDir scripts\distributed_scripts\ggeur_hierarchical_3machine\runs\<run_id>\<case>
.\scripts\distributed_scripts\ggeur_hierarchical_3machine\stop_clients.ps1 `
  -CaseDir scripts\distributed_scripts\ggeur_hierarchical_3machine\runs\<run_id>\<case> `
  -Force
.\scripts\distributed_scripts\ggeur_hierarchical_3machine\stop_subservers.ps1 `
  -CaseDir scripts\distributed_scripts\ggeur_hierarchical_3machine\runs\<run_id>\<case> `
  -Force
```

`federate.total_round_num: 100` means round 0 is statistics collection and
rounds 1-99 are classifier-training rounds in this pipeline.

Do not start the next case until root, subservers, and all clients from the
current case have exited or have been stopped explicitly. PID files are scoped
to each case directory.

## Results

Extract round-wise and best accuracy from the root log:

```bash
python scripts/distributed_scripts/ggeur_hierarchical_3machine/summarize_accuracy.py \
  <case_dir>/logs/root.stdout.log --output <case_dir>/accuracy_summary.json
```

## Accuracy acceptance gate

Final third-party results must satisfy both conditions:

1. GGEUR final average accuracy improves by at least 20% relative to the
   strongest non-GGEUR method in the same dataset/model group.
2. The absolute difference between three-machine and aligned standalone GGEUR
   final accuracy is at most 2.0 percentage points.

The checker always reports relative improvement and absolute percentage-point
improvement separately. A missing standalone result produces `PENDING`, never
`PASS`.

```bash
python scripts/distributed_scripts/ggeur_hierarchical_3machine/check_accuracy_acceptance.py \
  --distributed-root docs/test_logs/<distributed_group> \
  --standalone-root docs/test_logs/<standalone_group> \
  --min-relative-improvement-pct 20 \
  --max-parity-gap-pp 2 \
  --output docs/test_logs/<distributed_group>/accuracy_acceptance.json
```

Use `--parity-all-methods` when aligned standalone summaries exist for all six
methods. Use `--min-absolute-improvement-pp 20` only if the acceptance contract
explicitly requires 20 percentage points rather than 20 percent relative
improvement.

Passwords and credentials must never be stored in generated configs, scripts,
manifests, logs, or test records.
