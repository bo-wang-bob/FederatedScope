"""Validation and public schemas for the standalone control API."""

from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any, Dict, List


DOMAIN_KEYS = ['Art', 'Clipart', 'Product', 'Real_World']
EXPERIMENT_TYPES = {'heterogeneity', 'privacy', 'backdoor'}
STANDALONE_METHODS = {'fedavg', 'fedprox', 'heterogeneous_solution'}
DISTRIBUTED_METHODS = {
    'fedavg', 'fedprox', 'fedproto', 'fedopt', 'moon',
    'heterogeneous_solution',
}
METHODS = STANDALONE_METHODS | DISTRIBUTED_METHODS
DISTRIBUTED_GROUP_METHODS = {
    'digit3_cnn': {'fedavg', 'fedprox', 'heterogeneous_solution'},
    'digit3_vit': {'fedavg', 'fedprox', 'heterogeneous_solution'},
    'domainnet_cnn': DISTRIBUTED_METHODS,
    'domainnet_mixer': DISTRIBUTED_METHODS,
    'domainnet_vit': DISTRIBUTED_METHODS,
    'mdsent_lstm': DISTRIBUTED_METHODS - {'moon'},
    'mdsent_rnn': DISTRIBUTED_METHODS - {'moon'},
    'officehome_cnn': DISTRIBUTED_METHODS,
    'officehome_mixer': DISTRIBUTED_METHODS,
    'officehome_vit': DISTRIBUTED_METHODS,
}
PRIVACY_ATTACKS = {'membership', 'property', 'reconstruction'}
BACKDOOR_ATTACKS = {'trigger_injection', 'label_poisoning',
                    'model_update_poisoning'}


class ValidationError(ValueError):
    """A request validation error with optional field-level details."""

    def __init__(self, message: str, field_errors: Dict[str, str] | None = None):
        super().__init__(message)
        self.field_errors = field_errors or {}


def _number(value: Any, field: str, minimum: float,
            maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError('配置字段格式不正确', {field: '必须是数字'})
    if not minimum <= float(value) <= maximum:
        raise ValidationError(
            '配置字段超出范围',
            {field: f'必须在 {minimum:g}～{maximum:g} 之间'},
        )
    return float(value)


def _integer(value: Any, field: str, minimum: int, maximum: int) -> int:
    numeric = _number(value, field, minimum, maximum)
    if numeric != int(numeric):
        raise ValidationError('配置字段格式不正确', {field: '必须是整数'})
    return int(numeric)


def validate_scenario(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValidationError('请求体必须是 JSON 对象')
    dataset = payload.get('dataset', 'office-home')
    if dataset != 'office-home':
        raise ValidationError('当前版本仅支持 OfficeHome', {
            'dataset': '仅支持 office-home',
        })
    domains = payload.get('domains', DOMAIN_KEYS)
    if domains != DOMAIN_KEYS:
        raise ValidationError('数据域必须与后端四域一一对应', {
            'domains': '必须依次为 Art、Clipart、Product、Real_World',
        })
    clients = _integer(payload.get('clientsPerDomain', 15),
                       'clientsPerDomain', 1, 100)
    if clients != 15:
        raise ValidationError('当前版本固定每域 15 个逻辑客户端', {
            'clientsPerDomain': '必须为 15',
        })
    partition = payload.get('partition') or {}
    if partition.get('strategy', 'dirichlet') != 'dirichlet':
        raise ValidationError('当前版本仅支持狄利克雷划分', {
            'partition.strategy': '必须是 dirichlet',
        })
    alpha = _number(partition.get('alpha', 0.3),
                    'partition.alpha', 0.01, 100.0)
    seed = _integer(partition.get('seed', 20260816),
                    'partition.seed', 0, 2**31 - 1)
    return {
        'dataset': dataset,
        'domains': domains,
        'clientsPerDomain': clients,
        'partition': {
            'strategy': 'dirichlet',
            'alpha': alpha,
            'seed': seed,
        },
    }


def validate_experiment(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValidationError('请求体必须是 JSON 对象')
    normalized = copy.deepcopy(payload)
    errors: Dict[str, str] = {}

    name = str(normalized.get('name', '')).strip()
    if not 1 <= len(name) <= 80:
        errors['name'] = '实验名称长度必须为 1～80 个字符'
    experiment_type = normalized.get('type')
    if experiment_type not in EXPERIMENT_TYPES:
        errors['type'] = '必须选择异构、隐私或后门实验'
    scenario_id = str(normalized.get('scenarioId', '')).strip()
    if not re.fullmatch(r'SCN-[A-Z0-9-]{4,64}', scenario_id):
        errors['scenarioId'] = '必须先应用一个场景快照'

    execution = normalized.get('execution') or {'mode': 'standalone'}
    if not isinstance(execution, dict):
        errors['execution'] = '执行配置必须是对象'
        execution = {'mode': 'standalone'}
    execution_mode = execution.get('mode', 'standalone')
    if execution_mode not in {'standalone', 'distributed'}:
        errors['execution.mode'] = '必须选择单机或三机分布式执行'
        execution_mode = 'standalone'
    execution['mode'] = execution_mode
    if execution_mode == 'distributed':
        if experiment_type != 'heterogeneity':
            errors['type'] = '三机分布式阶段当前仅开放准确率实验'
        if execution.get('topologyId', 'lab-three-machine') != \
                'lab-three-machine':
            errors['execution.topologyId'] = '不支持该分布式拓扑'
        execution['topologyId'] = 'lab-three-machine'
        group = str(execution.get('group', '')).strip().lower()
        if group not in DISTRIBUTED_GROUP_METHODS:
            errors['execution.group'] = '必须选择已验证的数据集/模型组合'
        execution['group'] = group

    common = normalized.get('common')
    if not isinstance(common, dict):
        errors['common'] = '公共参数不能为空'
        common = {}
    method = common.get('method')
    allowed_methods = (DISTRIBUTED_METHODS if execution_mode == 'distributed'
                       else STANDALONE_METHODS)
    if method not in allowed_methods:
        errors['common.method'] = '不支持该运行方案'
    if execution_mode == 'distributed' and \
            execution.get('group') in DISTRIBUTED_GROUP_METHODS and \
            method not in DISTRIBUTED_GROUP_METHODS[execution['group']]:
        errors['common.method'] = '该数据集/模型组合不支持此方案'

    try:
        common['rounds'] = _integer(common.get('rounds', 30),
                                    'common.rounds', 1, 500)
        common['localEpochs'] = _integer(common.get('localEpochs', 1),
                                         'common.localEpochs', 1, 50)
        common['participationRate'] = _number(
            common.get('participationRate', 1.0),
            'common.participationRate', 0.01, 1.0)
        common['batchSize'] = _integer(common.get('batchSize', 8),
                                       'common.batchSize', 1, 1024)
        common['learningRate'] = _number(common.get('learningRate', 0.001),
                                         'common.learningRate', 1e-7, 1.0)
        common['seed'] = _integer(common.get('seed', 20260816),
                                  'common.seed', 0, 2**31 - 1)
    except ValidationError as error:
        errors.update(error.field_errors)
    if common.get('device') not in {'cpu', 'cuda'}:
        errors['common.device'] = '设备必须是 cpu 或 cuda'
    if method == 'fedprox':
        try:
            common['fedproxMu'] = _number(common.get('fedproxMu', 0.01),
                                          'common.fedproxMu', 0.0, 100.0)
        except ValidationError as error:
            errors.update(error.field_errors)
    else:
        common.pop('fedproxMu', None)

    if execution_mode == 'distributed':
        try:
            execution['evaluationFrequency'] = _integer(
                execution.get('evaluationFrequency', 1),
                'execution.evaluationFrequency', 1, 500)
            execution['clientsPerSubserver'] = _integer(
                execution.get('clientsPerSubserver', 30),
                'execution.clientsPerSubserver', 1, 120)
            execution['windowsClientCount'] = _integer(
                execution.get('windowsClientCount', 60),
                'execution.windowsClientCount', 0, 120)
            execution['rootDevice'] = _integer(
                execution.get('rootDevice', 1),
                'execution.rootDevice', 0, 15)
            execution['statisticsUploadStaggerSeconds'] = _number(
                execution.get('statisticsUploadStaggerSeconds', 3.0),
                'execution.statisticsUploadStaggerSeconds', 0.0, 120.0)
        except ValidationError as error:
            errors.update(error.field_errors)
        if not isinstance(execution.get('diagonalCovariance', False), bool):
            errors['execution.diagonalCovariance'] = '必须是布尔值'
        execution['diagonalCovariance'] = bool(
            execution.get('diagonalCovariance', False))
        if execution.get('cacheOnly', True) is not True:
            errors['execution.cacheOnly'] = \
                '三机训练必须使用已验证的完整特征缓存'
        execution['cacheOnly'] = True

    active_blocks: List[str] = []
    for key in EXPERIMENT_TYPES:
        block = normalized.get(key)
        if block is not None:
            if not isinstance(block, dict):
                errors[key] = '专属配置必须是对象或 null'
            else:
                active_blocks.append(key)
    if experiment_type and active_blocks != [experiment_type]:
        errors['type'] = '请求只能包含当前实验类型的专属配置'

    if experiment_type == 'heterogeneity':
        block = normalized.get('heterogeneity') or {}
        if method == 'heterogeneous_solution':
            try:
                block['expansionTarget'] = _integer(
                    block.get('expansionTarget', 50),
                    'heterogeneity.expansionTarget', 0, 10000)
                block['featureBatchSize'] = _integer(
                    block.get('featureBatchSize', 64),
                    'heterogeneity.featureBatchSize', 1, 1024)
            except ValidationError as error:
                errors.update(error.field_errors)
        normalized['heterogeneity'] = block
    elif experiment_type == 'privacy':
        block = normalized.get('privacy') or {}
        if block.get('attack') not in PRIVACY_ATTACKS:
            errors['privacy.attack'] = '不支持该隐私攻击类型'
        if not isinstance(block.get('defenseEnabled'), bool):
            errors['privacy.defenseEnabled'] = '必须明确是否启用保护'
        if block.get('defenseEnabled'):
            try:
                block['initialClip'] = _number(
                    block.get('initialClip', 1.0),
                    'privacy.initialClip', 0.001, 1000.0)
                block['targetQuantile'] = _number(
                    block.get('targetQuantile', 0.7),
                    'privacy.targetQuantile', 0.01, 0.99)
                block['noiseMultiplier'] = _number(
                    block.get('noiseMultiplier', 0.05),
                    'privacy.noiseMultiplier', 0.0, 100.0)
                block['epsilon'] = _number(block.get('epsilon', 6.0),
                                            'privacy.epsilon', 0.01, 10000.0)
            except ValidationError as error:
                errors.update(error.field_errors)
        normalized['privacy'] = block
    elif experiment_type == 'backdoor':
        block = normalized.get('backdoor') or {}
        if block.get('attack') not in BACKDOOR_ATTACKS:
            errors['backdoor.attack'] = '不支持该后门攻击类型'
        if not isinstance(block.get('defenseEnabled'), bool):
            errors['backdoor.defenseEnabled'] = '必须明确是否启用防御'
        try:
            block['maliciousRatio'] = _number(
                block.get('maliciousRatio', 0.05),
                'backdoor.maliciousRatio', 0.01, 0.5)
            block['startRound'] = _integer(block.get('startRound', 1),
                                           'backdoor.startRound', 0, 500)
            block['poisonRatio'] = _number(block.get('poisonRatio', 0.2),
                                           'backdoor.poisonRatio', 0.01, 1.0)
            block['targetLabel'] = _integer(block.get('targetLabel', 0),
                                            'backdoor.targetLabel', 0, 64)
        except ValidationError as error:
            errors.update(error.field_errors)
        if block.get('startRound', 0) > common.get('rounds', 0):
            errors['backdoor.startRound'] = '攻击起始轮次不能超过总轮次'
        malicious_clients = block.get('maliciousClients', [])
        if not isinstance(malicious_clients, list) or not all(
                isinstance(item, str) for item in malicious_clients):
            errors['backdoor.maliciousClients'] = '恶意客户端编号格式不正确'
        elif any(not re.fullmatch(
                r'OH-(?:DT|TS|ED|FR)-C(?:0[1-9]|1[0-5])', item)
                 for item in malicious_clients):
            errors['backdoor.maliciousClients'] = \
                '客户端编号必须属于当前四域的 60 个节点'
        elif not malicious_clients:
            prefixes = ['OH-DT', 'OH-TS', 'OH-ED', 'OH-FR']
            all_clients = [
                f'{prefix}-C{index:02d}'
                for prefix in prefixes for index in range(1, 16)
            ]
            count = max(1, round(
                60 * float(block.get('maliciousRatio', 0.05))))
            block['maliciousClients'] = all_clients[:count]
        if block.get('defenseEnabled'):
            for key in ('featureStageDefense', 'trainingStageDefense'):
                if not isinstance(block.get(key), bool):
                    errors[f'backdoor.{key}'] = '必须明确阶段防御开关'
        normalized['backdoor'] = block

    if errors:
        raise ValidationError('实验配置校验失败', errors)

    normalized['schemaVersion'] = str(normalized.get('schemaVersion', '1.0'))
    normalized['name'] = name
    normalized['scenarioId'] = scenario_id
    normalized['common'] = common
    normalized['execution'] = execution
    for key in EXPERIMENT_TYPES:
        if key != experiment_type:
            normalized[key] = None
    return normalized


def capabilities(repo_root: Path) -> Dict[str, Any]:
    from federatedscope.standalone_api.metric_registry import \
        public_metric_registry
    from federatedscope.standalone_api.distributed_runner import \
        distributed_catalog, load_lab_topology
    data_root = Path(os.environ.get(
        'FEDERATEDSCOPE_DATA_ROOT',
        '/root/autodl-tmp/datasets/OfficeHomeDataset_10072016'))
    model_path = Path(os.environ.get(
        'FEDERATEDSCOPE_MODEL_PATH',
        '/root/autodl-tmp/models/open_clip_vitb16.bin'))
    templates = repo_root / 'scripts' / 'standalone_configs'
    devices = ['cpu']
    try:
        import torch
        if torch.cuda.is_available():
            devices.append('cuda')
    except ImportError:
        pass
    distributed_cases = distributed_catalog(repo_root)
    topology = load_lab_topology()
    distributed_local_ready = bool(
        distributed_cases and
        (repo_root / 'scripts' / 'distributed_scripts' /
         'ggeur_hierarchical_3machine' / 'generate_matrix.py').is_file())
    return {
        'apiVersion': '1.0',
        'datasets': [{
            'key': 'office-home',
            'name': 'OfficeHome',
            'domains': DOMAIN_KEYS,
            'clientsPerDomain': 15,
            'available': data_root.exists(),
        }],
        'devices': devices,
        'methods': [
            'fedavg', 'fedprox', 'fedproto', 'fedopt', 'moon',
            'heterogeneous_solution'],
        'experimentTypes': sorted(EXPERIMENT_TYPES),
        'executionModes': ['standalone', 'distributed'],
        'runner': {
            'ready': data_root.exists() and model_path.exists() and templates.exists(),
            'dataReady': data_root.exists(),
            'modelReady': model_path.exists(),
            'templatesReady': templates.exists(),
        },
        'distributed': {
            'ready': distributed_local_ready,
            'remoteReadinessCheckedByPreflight': True,
            'topologyId': topology.topology_id,
            'nodes': [
                {'key': node.key, 'label': node.label,
                 'operatingSystem': node.operating_system}
                for node in topology.nodes
            ],
            'cases': distributed_cases,
        },
        'limits': {'maxConcurrentCpu': 1, 'maxConcurrentGpu': 1},
        'metrics': public_metric_registry(),
        'parameters': {
            'rounds': {'minimum': 1, 'maximum': 500, 'default': 30},
            'localEpochs': {'minimum': 1, 'maximum': 50, 'default': 1},
            'participationRate': {
                'minimum': 0.01, 'maximum': 1.0, 'default': 1.0},
            'batchSize': {'minimum': 1, 'maximum': 1024, 'default': 8},
            'learningRate': {
                'minimum': 1e-7, 'maximum': 1.0, 'default': 0.001},
            'fedproxMu': {'minimum': 0.0, 'maximum': 100.0,
                          'default': 0.01},
            'expansionTarget': {'minimum': 0, 'maximum': 10000,
                                'default': 50},
            'featureBatchSize': {'minimum': 1, 'maximum': 1024,
                                 'default': 64},
            'evaluationFrequency': {'minimum': 1, 'maximum': 500,
                                    'default': 1},
            'clientsPerSubserver': {'minimum': 1, 'maximum': 120,
                                    'default': 30},
            'statisticsUploadStaggerSeconds': {
                'minimum': 0, 'maximum': 120, 'default': 3},
        },
    }
