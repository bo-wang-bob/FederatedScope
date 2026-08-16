"""Validation and public schemas for the standalone control API."""

from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any, Dict, List


DOMAIN_KEYS = ['Art', 'Clipart', 'Product', 'Real_World']
EXPERIMENT_TYPES = {'heterogeneity', 'privacy', 'backdoor'}
METHODS = {'fedavg', 'fedprox', 'heterogeneous_solution'}
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

    common = normalized.get('common')
    if not isinstance(common, dict):
        errors['common'] = '公共参数不能为空'
        common = {}
    method = common.get('method')
    if method not in METHODS:
        errors['common.method'] = '不支持该运行方案'

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
    for key in EXPERIMENT_TYPES:
        if key != experiment_type:
            normalized[key] = None
    return normalized


def capabilities(repo_root: Path) -> Dict[str, Any]:
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
        'methods': ['fedavg', 'fedprox', 'heterogeneous_solution'],
        'experimentTypes': sorted(EXPERIMENT_TYPES),
        'runner': {
            'ready': data_root.exists() and model_path.exists() and templates.exists(),
            'dataReady': data_root.exists(),
            'modelReady': model_path.exists(),
            'templatesReady': templates.exists(),
        },
        'limits': {'maxConcurrentCpu': 1, 'maxConcurrentGpu': 1},
    }
