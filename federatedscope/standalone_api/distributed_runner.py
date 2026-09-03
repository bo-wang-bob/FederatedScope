"""Three-machine execution backend for the unified experiment console.

The control service stays outside the training processes.  It generates one
of the already validated hierarchical cases, copies only that case to the
three isolated worktrees, starts each role in dependency order, and mirrors
the root-server accuracy log into the normal experiment event stream.
"""

from __future__ import annotations

import base64
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import yaml
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, Iterable, List, Sequence

from federatedscope.standalone_api.runner import (
    EventCallback, MetricCallback, RunnerPreflightError,
    StandaloneProcessRunner)


SCRIPT_RELATIVE = PurePosixPath(
    'scripts/distributed_scripts/ggeur_hierarchical_3machine')
SOURCE_RELATIVE = PurePosixPath('scripts/example_configs/ggeur_final_5models')
METHOD_TO_SOURCE = {'heterogeneous_solution': 'ggeur'}
QUICK_CACHE_GROUPS = frozenset({
    'digit3_cnn', 'digit3_vit',
    'officehome_cnn', 'officehome_mixer', 'officehome_vit',
})


def _cached_quick_validation(group_dir: Path,
                             client_count: int) -> Dict[str, Any]:
    """Build a one-round cache-only preset from the validated FedAvg case."""
    source = yaml.safe_load(
        (group_dir / 'fedavg.yaml').read_text(encoding='utf-8')) or {}
    federate = source.get('federate') or {}
    train = source.get('train') or {}
    optimizer = train.get('optimizer') or {}
    dataloader = source.get('dataloader') or {}
    sampled = int(federate.get('sample_client_num', 0) or 0)
    participation = (1.0 if sampled <= 0 else
                     min(1.0, sampled / float(client_count)))
    return {
        'method': 'fedavg',
        'rounds': 1,
        'localEpochs': int(train.get('local_update_steps', 1)),
        'participationRate': participation,
        'batchSize': int(dataloader.get('batch_size', 32)),
        'learningRate': float(optimizer.get('lr', 0.001)),
        'evaluationFrequency': 1,
        'clientsPerSubserver': 30,
        'windowsClientCount': client_count,
        'statisticsUploadStaggerSeconds': 0.0,
        'diagonalCovariance': False,
        'cacheOnly': True,
    }


@dataclass(frozen=True)
class DistributedNode:
    key: str
    label: str
    target: str
    operating_system: str
    repo: str
    python: str
    jump: str = ''
    site_packages: str = ''
    local: bool = False

    @property
    def is_windows(self) -> bool:
        return self.operating_system == 'windows'


@dataclass(frozen=True)
class LabTopology:
    topology_id: str
    client: DistributedNode
    subserver: DistributedNode
    root: DistributedNode
    client_resource_repo: str
    subserver_resource_repo: str
    root_resource_repo: str

    @property
    def nodes(self) -> Sequence[DistributedNode]:
        return self.client, self.subserver, self.root


def _env(name: str, default: str) -> str:
    return str(os.environ.get(name, default)).strip()


def load_lab_topology() -> LabTopology:
    """Load the lab topology without storing credentials in source/configs."""
    # The unified web/API service is deployed on the Linux root host.  The
    # root training role is therefore local; only the two Windows roles use
    # SSH.  Environment overrides keep development and recovery deployments
    # possible without changing source.
    jump = _env('FEDERATEDSCOPE_DISTRIBUTED_JUMP', '')
    client_repo = _env(
        'FEDERATEDSCOPE_CLIENT_REPO',
        'D:/Projects/FederatedScope-worktrees/unified-security-standalone')
    third_repo = _env(
        'FEDERATEDSCOPE_SUBSERVER_REPO',
        'C:/Users/pc/FederatedScope-worktrees/unified-security-standalone')
    root_repo = _env(
        'FEDERATEDSCOPE_ROOT_REPO',
        '/root/autodl-tmp/FederatedScope-worktrees/'
        'unified-security-standalone')
    return LabTopology(
        topology_id='lab-three-machine',
        client=DistributedNode(
            key='client', label='8G 客户端机',
            target=_env(
                'FEDERATEDSCOPE_CLIENT_SSH', 'fsuser@10.129.222.189'),
            operating_system='windows', repo=client_repo,
            python=_env(
                'FEDERATEDSCOPE_CLIENT_PYTHON',
                'D:/ProgramData/anaconda3/envs/pi_fmd_gpu_py39/python.exe'),
            site_packages=_env(
                'FEDERATEDSCOPE_CLIENT_SITE_PACKAGES',
                'D:/Projects/FederatedScope/.venv_client_cpu/Lib/site-packages')),
        subserver=DistributedNode(
            key='subserver', label='4090 子服务器',
            target=_env(
                'FEDERATEDSCOPE_SUBSERVER_SSH', 'pc@10.129.248.111'),
            operating_system='windows', repo=third_repo,
            python=_env(
                'FEDERATEDSCOPE_SUBSERVER_PYTHON',
                'C:/Users/pc/miniconda3/envs/cerp/python.exe'),
            site_packages=_env(
                'FEDERATEDSCOPE_SUBSERVER_SITE_PACKAGES',
                'C:/Users/pc/miniconda3/envs/cerp/Lib/site-packages'),
            jump=jump),
        root=DistributedNode(
            key='root', label='双 4090 根服务器',
            target=_env(
                'FEDERATEDSCOPE_ROOT_SSH', 'root@10.112.81.135'),
            operating_system='linux', repo=root_repo,
            python=_env(
                'FEDERATEDSCOPE_ROOT_PYTHON',
                '/root/.local/share/mamba/envs/GGEUR/bin/python'),
            local=_env('FEDERATEDSCOPE_ROOT_LOCAL', '1') == '1'),
        client_resource_repo=_env(
            'FEDERATEDSCOPE_CLIENT_RESOURCE_REPO',
            'D:/Projects/FederatedScope'),
        subserver_resource_repo=_env(
            'FEDERATEDSCOPE_SUBSERVER_RESOURCE_REPO',
            'C:/Users/pc/FederatedScope'),
        root_resource_repo=_env(
            'FEDERATEDSCOPE_ROOT_RESOURCE_REPO',
            '/root/autodl-tmp/FederatedScope'),
    )


def distributed_catalog(repo_root: Path) -> List[Dict[str, Any]]:
    source_root = repo_root / Path(str(SOURCE_RELATIVE))
    cases: List[Dict[str, Any]] = []
    if not source_root.is_dir():
        return cases
    for group_dir in sorted(path for path in source_root.iterdir()
                            if path.is_dir()):
        dataset, model = group_dir.name.split('_', 1)
        client_count = 120 if dataset == 'mdsent' else 60
        methods = []
        for config_path in sorted(group_dir.glob('*.yaml')):
            method = ('heterogeneous_solution'
                      if config_path.stem == 'ggeur' else config_path.stem)
            methods.append(method)
        cases.append({
            'group': group_dir.name,
            'dataset': dataset,
            'model': model,
            'methods': methods,
            'clientCount': client_count,
            'cachePolicy': 'complete-feature-cache-required',
            'quickStart': group_dir.name in QUICK_CACHE_GROUPS,
            'quickValidation': _cached_quick_validation(
                group_dir, client_count),
        })
    return cases


class DistributedProcessRunner:
    """Run a generated hierarchical case through SSH via the client jump."""

    def __init__(self, repo_root: Path, topology: LabTopology | None = None,
                 probe_remote: bool = True,
                 command_runner: Callable[..., subprocess.CompletedProcess] |
                 None = None):
        self.repo_root = repo_root
        self.topology = topology or load_lab_topology()
        self.probe_remote = probe_remote
        self.command_runner = command_runner or subprocess.run

    @staticmethod
    def _source_method(method: str) -> str:
        return METHOD_TO_SOURCE.get(method, method)

    @staticmethod
    def _powershell_encoded(script: str) -> str:
        prefix = (
            "$ErrorActionPreference='Stop';"
            "$ProgressPreference='SilentlyContinue';"
            "[Console]::OutputEncoding=[Text.UTF8Encoding]::new();")
        return base64.b64encode(
            (prefix + script).encode('utf-16-le')).decode('ascii')

    def _ssh_base(self, node: DistributedNode) -> List[str]:
        command = [
            'ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12',
            '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3',
        ]
        if node.jump:
            command.extend(['-J', node.jump])
        command.append(node.target)
        return command

    def _remote_argv(self, node: DistributedNode, script: str) -> List[str]:
        if node.is_windows:
            remote = (
                'powershell.exe -NoProfile -NonInteractive '
                '-ExecutionPolicy Bypass -EncodedCommand ' +
                self._powershell_encoded(script))
        else:
            remote = f"bash -lc {shlex.quote(script)}"
        return [*self._ssh_base(node), remote]

    def _run_command(self, command: Sequence[str], timeout: float = 120,
                     check: bool = True) -> subprocess.CompletedProcess:
        result = self.command_runner(
            list(command), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding='utf-8', errors='replace', timeout=timeout,
            check=False)
        if check and result.returncode != 0:
            output = str(result.stdout or '').strip()[-4000:]
            raise RuntimeError(
                f"命令执行失败（{result.returncode}）：{output}")
        return result

    def _remote(self, node: DistributedNode, script: str,
                timeout: float = 120, check: bool = True) -> str:
        if node.local:
            result = self._run_command(
                ['bash', '-lc', script], timeout=timeout, check=check)
            return str(result.stdout or '')
        result = self._run_command(
            self._remote_argv(node, script), timeout=timeout, check=check)
        return str(result.stdout or '')

    def _resource_paths(self, group: str,
                        config: Dict[str, Any] | None = None
                        ) -> Dict[str, List[str]]:
        family, model = group.split('_', 1)
        client_base = self.topology.client_resource_repo.rstrip('/')
        root_base = self.topology.root_resource_repo.rstrip('/')
        data_rel = {
            'officehome': 'OfficeHomeDataset_10072016',
            'domainnet': 'data/DomainNet',
            'digit3': 'data/digit_three_domain',
            'mdsent': 'data/sentiment',
        }[family]
        model_rel = {
            'vit': 'pretrained_models/ViT-B-16.pt',
            'cnn': '',
            'mixer': 'pretrained_models/mixer_b16_224_complete.pth',
            'rnn': ('pretrained_models/'
                    'nlptown_bert_base_multilingual_uncased_senti'),
            'lstm': ('pretrained_models/'
                     'nlptown_bert_base_multilingual_uncased_senti'),
        }[model]
        total_clients = 120 if family == 'mdsent' else 60
        execution = (config or {}).get('execution', {})
        windows_clients = min(total_clients, max(
            0, int(execution.get('windowsClientCount', total_clients))))
        root_clients = total_clients - windows_clients
        result = {
            'client': [],
            'subserver': [],
            'root': [
                f'{root_base}/exp/distributed_feature_cache/{group}/'
                '.ggeur_feature_cache_ready.json',
            ],
        }
        if windows_clients:
            # Formal Windows clients are head-only workers.  Their launcher
            # accepts only a cache that was validated on that host with
            # require_complete_feature_cache=true, so neither raw images nor
            # a frozen backbone are runtime dependencies.  The portable
            # manifests below are still required to reconstruct sample IDs.
            result['client'].append(
                f'{client_base}/exp/distributed_feature_cache/{group}/'
                '.ggeur_feature_cache_ready.json')
        if root_clients:
            result['root'].append(f'{root_base}/{data_rel}')
        if model_rel and root_clients:
            root_model = {
                'vit': '/root/.cache/clip/ViT-B-16.pt',
                'mixer': f'{root_base}/{model_rel}',
                'rnn': f'{root_base}/{model_rel}',
                'lstm': f'{root_base}/{model_rel}',
            }[model]
            result['root'].append(root_model)
        if family == 'officehome':
            manifest = ('exp/distributed_manifests/'
                        'officehome_60c_lds01_seed42')
            if windows_clients:
                result['client'].append(f'{client_base}/{manifest}')
            if root_clients:
                result['root'].append(f'{root_base}/{manifest}')
        if family == 'domainnet':
            manifest = ('exp/distributed_manifests/domainnet_4domains/'
                        'domainnet_manifest.json')
            if windows_clients:
                result['client'].append(f'{client_base}/{manifest}')
            # The root evaluator also uses the portable manifest when no raw
            # DomainNet tree is deployed beside the isolated worktree.
            result['root'].append(f'{root_base}/{manifest}')
        if family == 'digit3':
            manifest_root = 'data/digit_three_domain/manifests'
            global_manifest = 'data/digit_three_domain/dataset_manifest.json'
            if windows_clients:
                result['client'].append(
                    f'{client_base}/{manifest_root}')
            if root_clients:
                result['root'].append(f'{root_base}/{manifest_root}')
            result['root'].append(f'{root_base}/{global_manifest}')
        return result

    def _probe_node(self, node: DistributedNode,
                    resource_paths: Iterable[str]) -> Dict[str, Any]:
        if node.is_windows:
            checks = ','.join(
                "'" + value.replace("'", "''") + "'"
                for value in resource_paths)
            script = (
                f"$repo='{node.repo}';$python='{node.python}';"
                "$missing=@();"
                "if(-not(Test-Path -LiteralPath $repo)){$missing+='repo'};"
                "if(-not(Test-Path -LiteralPath $python)){$missing+='python'};"
                f"foreach($path in @({checks})){{"
                "if(-not(Test-Path -LiteralPath $path)){$missing+=$path}"
                "elseif($path.EndsWith('.ggeur_feature_cache_ready.json')){"
                "try{$state=Get-Content -LiteralPath $path -Raw|"
                "ConvertFrom-Json;if(-not $state.require_complete_feature_cache)"
                "{$missing+=($path+'[not-complete]')}}"
                "catch{$missing+=($path+'[invalid-marker]')}}};"
                "if($missing.Count -eq 0){"
                "& $python -c \"import torch;print(torch.__version__)\";"
                "if($LASTEXITCODE -ne 0){throw 'python cannot import torch'};"
                "'READY'}else{'MISSING='+($missing -join '|');exit 3}")
        else:
            quoted = ' '.join(shlex.quote(value) for value in resource_paths)
            script = (
                f"test -d {shlex.quote(node.repo)} && "
                f"test -x {shlex.quote(node.python)} && "
                f"for p in {quoted}; do test -e \"$p\" || "
                "{ echo MISSING=$p; exit 3; }; done; "
                f"{shlex.quote(node.python)} -c 'import torch;"
                "print(torch.__version__)'; echo READY")
        output = self._remote(node, script, timeout=45, check=False)
        return {
            'node': node.key,
            'label': node.label,
            'ready': 'READY' in output and 'MISSING=' not in output,
            'message': output.strip()[-1000:] or 'SSH 未返回状态',
        }

    def preflight(self, config: Dict[str, Any]) -> Dict[str, Any]:
        execution = config.get('execution', {})
        group = str(execution.get('group', ''))
        method = self._source_method(config['common']['method'])
        source = self.repo_root / Path(str(SOURCE_RELATIVE)) / group / \
            f'{method}.yaml'
        generator = self.repo_root / Path(str(SCRIPT_RELATIVE)) / \
            'generate_matrix.py'
        checks = [
            {'name': 'distributedGenerator', 'ready': generator.is_file(),
             'message': ('三机配置生成器已就绪' if generator.is_file()
                         else '缺少三机配置生成器'),
             'details': {'path': str(generator)}},
            {'name': 'distributedCase', 'ready': source.is_file(),
             'message': (f'{group}/{method} 配置已就绪' if source.is_file()
                         else f'缺少 {group}/{method} 配置'),
             'details': {'path': str(source)}},
            {'name': 'sshClient', 'ready': shutil.which('ssh') is not None,
             'message': ('SSH 客户端已就绪' if shutil.which('ssh')
                         else '未找到 SSH 客户端'), 'details': {}},
            {'name': 'scpClient', 'ready': shutil.which('scp') is not None,
             'message': ('SCP 客户端已就绪' if shutil.which('scp')
                         else '未找到 SCP 客户端'), 'details': {}},
        ]
        topology_nodes = []
        if self.probe_remote and all(item['ready'] for item in checks):
            resources = self._resource_paths(group, config)
            for node in self.topology.nodes:
                status = self._probe_node(node, resources[node.key])
                topology_nodes.append(status)
                checks.append({
                    'name': f"node.{node.key}",
                    'ready': status['ready'],
                    'message': (f"{node.label}已就绪" if status['ready']
                                else f"{node.label}未就绪："
                                f"{status['message']}"),
                    'details': {},
                })
        else:
            topology_nodes = [
                {'node': node.key, 'label': node.label, 'ready': None,
                 'message': '远端探测未执行'}
                for node in self.topology.nodes
            ]
        result = {
            'ready': all(item['ready'] for item in checks),
            'checks': checks,
            'template': str(source),
            'dataRoot': group.split('_', 1)[0],
            'modelPath': group.split('_', 1)[1],
            'partitionManifest': '',
            'executionMode': 'distributed',
            'topologyId': self.topology.topology_id,
            'topology': topology_nodes,
        }
        if not result['ready']:
            failures = [item['message'] for item in checks
                        if not item['ready']]
            raise RunnerPreflightError('；'.join(failures), result)
        return result

    def build_generate_command(self, config: Dict[str, Any],
                               control_root: Path) -> List[str]:
        execution = config['execution']
        common = config['common']
        group = execution['group']
        family, _ = group.split('_', 1)
        run_id = re.sub(r'[^A-Za-z0-9_.-]', '_', config['experimentId'])
        source_method = self._source_method(common['method'])
        command = [
            sys.executable,
            str(self.repo_root / Path(str(SCRIPT_RELATIVE)) /
                'generate_matrix.py'),
            '--run-id', run_id,
            '--output-root', str(control_root),
            '--case-specs', f'{group}:{source_method}',
            '--total-round-num', str(common['rounds']),
            '--sample-client-num', str(max(
                1, round((120 if family == 'mdsent' else 60) *
                         common['participationRate']))),
            '--train-learning-rate', str(common['learningRate']),
            '--train-local-update-steps', str(common['localEpochs']),
            '--eval-frequency', str(execution['evaluationFrequency']),
            '--clients-per-subserver', str(execution['clientsPerSubserver']),
            '--windows-client-count', str(execution['windowsClientCount']),
            '--root-device', str(execution['rootDevice']),
            '--statistics-upload-stagger-seconds',
            str(execution['statisticsUploadStaggerSeconds']),
            '--root-repo', self.topology.root.repo,
            '--subserver-repo', self.topology.subserver.repo,
            '--client-repo', self.topology.client.repo,
            '--root-client-repo', self.topology.root.repo,
            '--third-client-repo', self.topology.subserver.repo,
        ]
        client_resources = self.topology.client_resource_repo.rstrip('/')
        third_resources = self.topology.subserver_resource_repo.rstrip('/')
        root_resources = self.topology.root_resource_repo.rstrip('/')
        command.extend([
            '--client-feature-cache-root',
            f'{client_resources}/exp/distributed_feature_cache',
            '--root-client-feature-cache-root',
            f'{root_resources}/exp/distributed_feature_cache',
            '--third-client-feature-cache-root',
            f'{third_resources}/exp/distributed_feature_cache',
        ])
        resource_flags = {
            'officehome-root': 'OfficeHomeDataset_10072016',
            'domainnet-root': 'data/DomainNet',
            'digit3-root': 'data/digit_three_domain',
            'mdsent-root': 'data/sentiment',
        }
        for suffix, relative in resource_flags.items():
            command.extend([
                f'--client-{suffix}', f'{client_resources}/{relative}',
                f'--root-client-{suffix}', f'{root_resources}/{relative}',
                f'--third-client-{suffix}', f'{third_resources}/{relative}',
            ])
        command.extend([
            '--client-officehome-manifest-root',
            f'{client_resources}/exp/distributed_manifests/'
            'officehome_60c_lds01_seed42',
            '--root-client-officehome-manifest-root',
            f'{root_resources}/exp/distributed_manifests/'
            'officehome_60c_lds01_seed42',
            '--third-client-officehome-manifest-root',
            f'{third_resources}/exp/distributed_manifests/'
            'officehome_60c_lds01_seed42',
            '--client-domainnet-manifest-path',
            f'{client_resources}/exp/distributed_manifests/'
            'domainnet_4domains/domainnet_manifest.json',
            '--root-client-domainnet-manifest-path',
            f'{root_resources}/exp/distributed_manifests/'
            'domainnet_4domains/domainnet_manifest.json',
            '--third-client-domainnet-manifest-path',
            f'{third_resources}/exp/distributed_manifests/'
            'domainnet_4domains/domainnet_manifest.json',
            '--client-clip-model-path',
            f'{client_resources}/pretrained_models/ViT-B-16.pt',
            '--root-client-clip-model-path', '/root/.cache/clip/ViT-B-16.pt',
            '--third-client-clip-model-path',
            f'{third_resources}/pretrained_models/ViT-B-16.pt',
            '--client-mixer-checkpoint-path',
            f'{client_resources}/pretrained_models/'
            'mixer_b16_224_complete.pth',
            '--root-client-mixer-checkpoint-path',
            f'{root_resources}/pretrained_models/'
            'mixer_b16_224_complete.pth',
            '--third-client-mixer-checkpoint-path',
            f'{third_resources}/pretrained_models/'
            'mixer_b16_224_complete.pth',
            '--client-bert-model-path',
            f'{client_resources}/pretrained_models/'
            'nlptown_bert_base_multilingual_uncased_senti',
            '--root-client-bert-model-path',
            f'{root_resources}/pretrained_models/'
            'nlptown_bert_base_multilingual_uncased_senti',
            '--third-client-bert-model-path',
            f'{third_resources}/pretrained_models/'
            'nlptown_bert_base_multilingual_uncased_senti',
        ])
        if execution.get('diagonalCovariance'):
            command.append('--ggeur-diagonal-covariance')
        else:
            command.append('--no-ggeur-diagonal-covariance')
        if common['method'] == 'fedprox':
            # The source file owns the validated FedProx mu.  It remains in
            # the public config for reproducibility and is not silently
            # rewritten by the matrix generator.
            pass
        if common['method'] == 'heterogeneous_solution':
            block = config.get('heterogeneity') or {}
            target = int(block.get('expansionTarget', 50))
            command.extend([
                '--ggeur-target-size-per-class', str(target),
                '--ggeur-num-generated-per-sample', str(target),
            ])
        return command

    def _scp_base(self, node: DistributedNode) -> List[str]:
        command = [
            'scp', '-q', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12',
        ]
        if node.jump:
            command.extend(['-o', f'ProxyJump={node.jump}'])
        return command

    def _sync_run(self, run_root: Path, run_id: str,
                  emit: EventCallback) -> None:
        archive_path: Path | None = None
        try:
            for node in self.topology.nodes:
                remote_runs = (
                    f"{node.repo.rstrip('/')}/{SCRIPT_RELATIVE}/runs")
                if node.local:
                    destination = Path(remote_runs) / run_id
                    if destination.exists():
                        raise RuntimeError(
                            f'运行目录已存在，拒绝覆盖：{destination}')
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(run_root, destination)
                    emit('topology.status.changed', {
                        'node': node.key, 'label': node.label,
                        'status': '配置已同步', 'ready': True,
                    })
                    continue
                if node.is_windows:
                    self._remote(
                        node,
                        f"New-Item -ItemType Directory -Force -Path "
                        f"'{remote_runs}' | Out-Null;"
                        f"if(Test-Path -LiteralPath "
                        f"'{remote_runs}/{run_id}')"
                        "{throw 'run directory already exists'}")
                    if archive_path is None:
                        archive_base = run_root.parent / \
                            f'.{run_id}.sync'
                        archive_path = Path(shutil.make_archive(
                            str(archive_base), 'zip',
                            root_dir=run_root.parent,
                            base_dir=run_root.name))
                    # OpenSSH scp does not reliably accept a Windows drive
                    # path as its remote destination.  Upload to the SSH
                    # user's home and let PowerShell move the extracted run
                    # into the isolated worktree.
                    remote_archive = f'.federatedscope-{run_id}.zip'
                    command = [
                        *self._scp_base(node), str(archive_path),
                        f'{node.target}:{remote_archive}',
                    ]
                    self._run_command(command, timeout=600)
                    self._remote(
                        node,
                        f"$archive=Join-Path $HOME '{remote_archive}';"
                        f"Expand-Archive -LiteralPath $archive "
                        f"-DestinationPath '{remote_runs}';"
                        "Remove-Item -LiteralPath $archive -Force",
                        timeout=600)
                else:
                    self._remote(
                        node,
                        f"mkdir -p {shlex.quote(remote_runs)}; "
                        f"test ! -e "
                        f"{shlex.quote(remote_runs + '/' + run_id)}")
                    target = f'{node.target}:{remote_runs}/'
                    command = [
                        *self._scp_base(node), '-r', str(run_root), target]
                    self._run_command(command, timeout=600)
                emit('topology.status.changed', {
                    'node': node.key, 'label': node.label,
                    'status': '配置已同步', 'ready': True,
                })
        finally:
            if archive_path is not None:
                archive_path.unlink(missing_ok=True)

    @staticmethod
    def _windows_path(value: str) -> str:
        return value.replace('/', '\\')

    def _launch(self, case_relative: str, has_root_clients: bool,
                emit: EventCallback) -> None:
        root = self.topology.root
        root_case = f"{root.repo.rstrip('/')}/{case_relative}"
        root_script = f"{root.repo.rstrip('/')}/{SCRIPT_RELATIVE}"
        self._remote(
            root,
            f"REPO_DIR={shlex.quote(root.repo)} "
            f"PYTHON_BIN={shlex.quote(root.python)} bash "
            f"{shlex.quote(root_script + '/launch_root.sh')} "
            f"{shlex.quote(root_case)}", timeout=180)
        emit('topology.status.changed', {
            'node': 'root', 'label': root.label,
            'status': '根聚合服务运行中', 'ready': True})

        third = self.topology.subserver
        third_script = self._windows_path(
            f"{third.repo.rstrip('/')}/{SCRIPT_RELATIVE}/"
            'launch_subservers.ps1')
        third_case = self._windows_path(case_relative)
        self._remote(
            third,
            f"& '{third_script}' -CaseDir '{third_case}' "
            f"-PythonBin '{self._windows_path(third.python)}' "
            "-UseScheduledTasks", timeout=900)
        emit('topology.status.changed', {
            'node': 'subserver', 'label': third.label,
            'status': '域级子服务器运行中', 'ready': True})

        client = self.topology.client
        client_script = self._windows_path(
            f"{client.repo.rstrip('/')}/{SCRIPT_RELATIVE}/"
            'launch_clients.ps1')
        client_case = self._windows_path(case_relative)
        client_site = self._windows_path(client.site_packages)
        self._remote(
            client,
            f"& '{client_script}' -CaseDir '{client_case}' "
            f"-PythonBin '{self._windows_path(client.python)}' "
            f"-ClientSitePackages '{client_site}' "
            "-ConfigSet 'clients_8g' -UseScheduledTasks",
            timeout=3600)
        emit('topology.status.changed', {
            'node': 'client', 'label': client.label,
            'status': '逻辑客户端运行中', 'ready': True})

        if has_root_clients:
            self._remote(
                root,
                f"REPO_DIR={shlex.quote(root.repo)} "
                f"PYTHON_BIN={shlex.quote(root.python)} bash "
                f"{shlex.quote(root_script + '/launch_root_clients.sh')} "
                f"{shlex.quote(root_case)}", timeout=3600)

    def _stop_remote(self, case_relative: str,
                     emit: EventCallback | None = None,
                     status: str | None = '已停止') -> List[str]:
        """Best-effort cleanup for every role belonging to one case.

        Cleanup must never stop at the first unreachable node.  Otherwise a
        client-side failure can leave the root and subservers holding their
        well-known ports and every later experiment will fail preflight.
        """
        client = self.topology.client
        third = self.topology.subserver
        root = self.topology.root
        windows_case = self._windows_path(case_relative)
        commands = [
            (client,
             f"& '{self._windows_path(client.repo + '/' + str(SCRIPT_RELATIVE) + '/stop_clients.ps1')}' "
             f"-CaseDir '{windows_case}' -Force"),
            (third,
             f"& '{self._windows_path(third.repo + '/' + str(SCRIPT_RELATIVE) + '/stop_subservers.ps1')}' "
             f"-CaseDir '{windows_case}' -Force"),
        ]
        root_case = f"{root.repo.rstrip('/')}/{case_relative}"
        root_script = f"{root.repo.rstrip('/')}/{SCRIPT_RELATIVE}"
        commands.extend([
            (root, f"REPO_DIR={shlex.quote(root.repo)} bash "
                   f"{shlex.quote(root_script + '/stop_role.sh')} "
                   f"{shlex.quote(root_case)} client true"),
            (root, f"REPO_DIR={shlex.quote(root.repo)} bash "
                   f"{shlex.quote(root_script + '/stop_role.sh')} "
                   f"{shlex.quote(root_case)} root true"),
        ])
        errors: List[str] = []
        for node, command in commands:
            try:
                self._remote(node, command, timeout=180, check=False)
            except Exception as error:  # cleanup must continue on other nodes
                errors.append(f'{node.key}: {error}')
                if emit is not None:
                    emit('warning.raised', {
                        'level': 'error',
                        'message': f'{node.label}清理失败：{error}',
                    })
            finally:
                if emit is not None and status is not None:
                    emit('topology.status.changed', {
                        'node': node.key, 'label': node.label,
                        'status': status, 'ready': False})
        return errors

    def cleanup(self, config: Dict[str, Any], output_dir: Path,
                emit: EventCallback | None = None) -> List[str]:
        """Recover processes for an interrupted or failed experiment.

        The case path is deterministic, so cleanup also works after the API
        process has restarted and lost its in-memory worker/thread state.
        """
        run_id = re.sub(
            r'[^A-Za-z0-9_.-]', '_', str(config['experimentId']))
        group = str(config.get('execution', {}).get('group', '')).strip()
        if not group:
            return []
        method = self._source_method(config['common']['method'])
        case_name = f'{group}_{method}'
        matrix_path = (output_dir / 'distributed_cases' / run_id /
                       'matrix_manifest.json')
        if matrix_path.is_file():
            try:
                matrix = json.loads(
                    matrix_path.read_text(encoding='utf-8-sig'))
                case_name = str(matrix['cases'][0]['case'])
            except (KeyError, IndexError, TypeError, ValueError, OSError):
                # Fall back to the generator's deterministic case name.
                pass
        case_relative = str(
            SCRIPT_RELATIVE / 'runs' / run_id / case_name)
        return self._stop_remote(
            case_relative, emit=emit,
            status='失败后已清理' if emit is not None else None)

    def _root_status(self, remote_case: str) -> str:
        node = self.topology.root
        pid_path = remote_case.rstrip('/') + '/pids/root.pid'
        script = (
            f"if test ! -f {shlex.quote(pid_path)}; then echo missing; "
            f"else pid=$(cat {shlex.quote(pid_path)}); "
            "if kill -0 \"$pid\" 2>/dev/null; then echo running; "
            "else echo stopped; fi; fi")
        output = self._remote(node, script, timeout=30, check=False)
        for state in ('running', 'stopped', 'missing'):
            if state in output:
                return state
        return 'unknown'

    def _root_log(self, remote_case: str, first_line: int = 1) -> str:
        path = remote_case.rstrip('/') + '/logs/root.stdout.log'
        script = (
            f"test -f {shlex.quote(path)} && "
            f"tail -n +{max(1, int(first_line))} {shlex.quote(path)} || true")
        return self._remote(
            self.topology.root, script, timeout=45, check=False)

    @staticmethod
    def _display_client(client_index: int, round_index: int,
                        group: str, line: str) -> Dict[str, Any]:
        if group.startswith('officehome_') and 1 <= client_index <= 60:
            return StandaloneProcessRunner._client_payload(
                client_index, round_index, line)
        domain_keys = ['Art', 'Clipart', 'Product', 'Real_World']
        return {
            'clientId': f'CL-{client_index:03d}',
            'domainKey': domain_keys[(max(1, client_index) - 1) % 4],
            'status': ('上传中' if 'upload' in line.lower()
                       else '本地训练'),
            'progress': 80 if 'upload' in line.lower() else 55,
            'round': max(0, round_index),
        }

    def _consume_log(self, text: str, group: str, state: Dict[str, Any],
                     log_stream: Any, emit: EventCallback,
                     on_metric: MetricCallback) -> None:
        for line in text.splitlines():
            log_stream.write(line + '\n')
            log_stream.flush()
            parsed = StandaloneProcessRunner._parse_line(line)
            current_round = parsed.get('round')
            if current_round is not None and current_round != state['round']:
                state['round'] = int(current_round)
                emit('round.started', {
                    'phaseIndex': 2, 'round': state['round']})
            if 'clientIndex' in parsed:
                emit('client.status.changed', self._display_client(
                    int(parsed['clientIndex']), state['round'], group, line))
            if parsed.get('metrics') and 'Test Accuracy -' in line:
                metric = {'round': max(0, state['round']),
                          **parsed['metrics']}
                on_metric(metric)
                emit('metric.updated', metric)
            lowered = line.lower()
            if any(token in lowered for token in
                   ('traceback', 'exception', 'error')):
                emit('warning.raised', {
                    'level': 'error', 'message': line[-2000:]})

    def run(self, config: Dict[str, Any], output_dir: Path,
            stop_event: threading.Event, emit: EventCallback,
            on_metric: MetricCallback) -> int:
        self.preflight(config)
        output_dir.mkdir(parents=True, exist_ok=True)
        control_root = output_dir / 'distributed_cases'
        control_root.mkdir(parents=True, exist_ok=True)
        command = self.build_generate_command(config, control_root)
        generated = self._run_command(command, timeout=180)
        generator_output = str(generated.stdout or '')
        run_id = re.sub(r'[^A-Za-z0-9_.-]', '_', config['experimentId'])
        run_root = control_root / run_id
        matrix_path = run_root / 'matrix_manifest.json'
        if not matrix_path.is_file():
            raise RuntimeError(
                f'分布式配置生成失败：{generator_output[-2000:]}')
        matrix = json.loads(matrix_path.read_text(encoding='utf-8-sig'))
        case = matrix['cases'][0]
        case_name = case['case']
        case_relative = str(SCRIPT_RELATIVE / 'runs' / run_id /
                            case_name)
        has_root_clients = bool(case.get('client_hosts', {}).get('4090'))

        log_path = output_dir / 'runner.log'
        with log_path.open('a', encoding='utf-8') as log_stream:
            log_stream.write(generator_output)
            emit('stage.changed', {
                'phaseIndex': 0, 'round': 0, 'stage': '三机配置生成'})
            self._sync_run(run_root, run_id, emit)
            if stop_event.is_set():
                return -15
            emit('stage.changed', {
                'phaseIndex': 1, 'round': 0, 'stage': '分布式角色启动'})
            cleanup_status: str | None = None
            try:
                self._launch(case_relative, has_root_clients, emit)
                remote_case = (
                    f"{self.topology.root.repo.rstrip('/')}/{case_relative}")
                emit('stage.changed', {
                    'phaseIndex': 2, 'round': 0, 'stage': '分布式训练'})
                state: Dict[str, Any] = {'round': -1, 'line': 1}
                stopped_polls = 0
                while True:
                    if stop_event.is_set():
                        cleanup_status = '已停止'
                        return -15
                    new_text = self._root_log(remote_case, state['line'])
                    if new_text:
                        state['line'] += len(new_text.splitlines())
                        self._consume_log(
                            new_text, config['execution']['group'], state,
                            log_stream, emit, on_metric)
                    status = self._root_status(remote_case)
                    if status == 'running':
                        stopped_polls = 0
                    elif status in ('stopped', 'missing'):
                        stopped_polls += 1
                    if stopped_polls >= 2:
                        break
                    time.sleep(2.0)
                final_text = self._root_log(remote_case, 1)
                if final_text:
                    full_lines = final_text.splitlines()
                    already = max(0, state['line'] - 1)
                    if already < len(full_lines):
                        self._consume_log(
                            '\n'.join(full_lines[already:]),
                            config['execution']['group'], state, log_stream,
                            emit, on_metric)
                success = ('Best Average Accuracy:' in final_text and
                           'Traceback (most recent call last)' not in
                           final_text)
                if not success:
                    cleanup_status = '异常退出后已清理'
                for node in self.topology.nodes:
                    emit('topology.status.changed', {
                        'node': node.key, 'label': node.label,
                        'status': ('任务完成' if success else
                                   '任务异常退出'),
                        'ready': success})
                return 0 if success else 1
            except Exception:
                cleanup_status = '失败后已清理'
                raise
            finally:
                # Also reap normally exited roles to remove scheduled tasks,
                # pid files and any late/stuck client process.
                self._stop_remote(
                    case_relative, emit=emit, status=cleanup_status)


class DispatchingExperimentRunner:
    """Preserve standalone behavior while adding an isolated backend."""

    def __init__(self, standalone: StandaloneProcessRunner,
                 distributed: DistributedProcessRunner):
        self.standalone = standalone
        self.distributed = distributed

    def _select(self, config: Dict[str, Any]):
        mode = config.get('execution', {}).get('mode', 'standalone')
        return self.distributed if mode == 'distributed' else self.standalone

    def preflight(self, config: Dict[str, Any]) -> Dict[str, Any]:
        return self._select(config).preflight(config)

    def run(self, config: Dict[str, Any], output_dir: Path,
            stop_event: threading.Event, emit: EventCallback,
            on_metric: MetricCallback) -> int:
        return self._select(config).run(
            config, output_dir, stop_event, emit, on_metric)

    def cleanup(self, config: Dict[str, Any], output_dir: Path,
                emit: EventCallback | None = None) -> List[str]:
        selected = self._select(config)
        cleanup = getattr(selected, 'cleanup', None)
        if cleanup is None:
            return []
        return cleanup(config, output_dir, emit)
