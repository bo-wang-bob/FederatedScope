"""Allowlisted, reproducible single-host accuracy configurations (no CUDA import)."""
from __future__ import annotations

import copy
import hashlib
import math
import os
from pathlib import Path

import yaml


METHODS = {'fedavg': 'FedAvg', 'fedprox': 'FedProx', 'fedproto': 'FedProto',
           'fedopt': 'FedOpt', 'moon': 'MOON', 'heterogeneous_solution': 'GGEUR'}
FAMILIES = {'officehome': ('Office-Home', 'OfficeHomeDataset_10072016', 4),
            'domainnet': ('DomainNet', 'DomainNet', 4),
            'digit3': ('Digits-3Domain', 'digit_three_domain', 3),
            'mdsent': ('MDSent', 'sentiment', 4)}


class PlatformError(ValueError):
    def __init__(self, message, status=422):
        super().__init__(message)
        self.status = status


def sha256(path):
    result = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


class ConfigFactory:
    def __init__(self, repo):
        self.repo = Path(repo).resolve()
        self.sources = self.repo / 'scripts/example_configs/ggeur_final_5models'
        self.resources = Path(os.environ.get('FS_PLATFORM_RESOURCES',
                                            '/root/autodl-tmp/FederatedScope'))
        self.datasets = Path(os.environ.get('FS_PLATFORM_DATASETS',
                                           '/root/autodl-tmp/datasets'))

    def source(self, group, method):
        if (not isinstance(group, str) or not isinstance(method, str)
                or group not in self.groups() or method not in METHODS):
            raise PlatformError('不支持的数据配置或算法')
        name = 'ggeur' if method == 'heterogeneous_solution' else method
        path = self.sources / group / (name + '.yaml')
        if not path.is_file():
            raise PlatformError(f'{group} 没有 {method} 的已验证配置')
        return path

    def groups(self):
        return sorted(p.name for p in self.sources.iterdir() if p.is_dir()
                      and p.name.split('_')[0] in FAMILIES)

    def cache_dir(self, group):
        return Path(os.environ.get('FS_PLATFORM_CACHE_' + group.upper(),
                    str(self.resources / 'exp/distributed_feature_cache' / group)))

    def defaults(self, group, method):
        raw = yaml.safe_load(self.source(group, method).read_text(encoding='utf-8'))
        g = raw['ggeur']
        return dict(group=group, method=method, name='', rounds=3,
                    clientCount=raw['federate']['client_num'],
                    sampleClients=raw['federate'].get('sample_client_num', 0),
                    batchSize=raw['dataloader']['batch_size'],
                    localEpochs=raw['train']['local_update_steps'],
                    learningRate=float(raw['train']['optimizer']['lr']),
                    seed=int(raw.get('seed', 42)), splitSeed=42,
                    alpha=float(g.get('lds_alpha', raw['data'].get('dirichlet_alpha', .1))),
                    gpu=1, evaluationFrequency=1, samplesPerClient=0)

    def catalog(self):
        entries = []
        for group in self.groups():
            family, backbone = group.split('_', 1)
            cache = self.cache_dir(group)
            files = list(cache.glob('*.npz'))
            methods = []
            for method, label in METHODS.items():
                try:
                    defaults = self.defaults(group, method)
                except PlatformError:
                    continue
                # Old augmented caches do not prove their sample membership.
                # Never silently fall back to generation or relabel a baseline.
                reason = ('增强缓存尚未登记可核验的样本划分；禁止临时生成'
                          if method == 'heterogeneous_solution' else None)
                methods.append(dict(id=method, label=label, defaults=defaults,
                                    enabled=reason is None, reason=reason))
            entries.append(dict(id=group, dataset=FAMILIES[family][0], backbone=backbone,
                                domains=FAMILIES[family][2], methods=methods,
                                cacheFound=bool(files), cacheFiles=len(files),
                                cacheBytes=sum(p.stat().st_size for p in files),
                                partitionLocked=family == 'digit3'))
        return dict(groups=entries, host='4090lziy', address='10.112.81.135',
                    mode='single-host', cacheOnly=True,
                    protocol='cache-only-v1 / 原始特征基线；不生成增强数据',
                    evaluationPolicy='final 模型默认；best 使用训练期测试集择优，不能视为无偏验证')

    def normalize(self, payload):
        if not isinstance(payload, dict):
            raise PlatformError('配置必须是 JSON 对象')
        group, method = payload.get('group', 'officehome_vit'), payload.get('method', 'fedavg')
        req = self.defaults(group, method)
        unknown = set(payload) - set(req) - {'idempotencyKey', 'preflightId'}
        if unknown:
            raise PlatformError('未知参数：' + ', '.join(sorted(unknown)))
        req.update({k: v for k, v in payload.items() if k in req})
        if method == 'heterogeneous_solution':
            raise PlatformError('GGEUR 需要可核验的完整增强缓存，当前未登记；不会临时生成')
        limits = {'rounds': (1, 1000), 'clientCount': (3, 240),
                  'sampleClients': (0, 240), 'batchSize': (1, 1024),
                  'localEpochs': (1, 100), 'seed': (0, 2147483647),
                  'splitSeed': (0, 2147483647), 'gpu': (-1, 7),
                  'evaluationFrequency': (1, 1000), 'samplesPerClient': (0, 100000)}
        for name, (lo, hi) in limits.items():
            value = req[name]
            if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
                raise PlatformError(f'{name} 必须是 {lo}–{hi} 之间的整数')
        for name, lo, hi in [('learningRate', 1e-8, 1.0), ('alpha', 1e-4, 100.0)]:
            value = req[name]
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not lo <= value <= hi:
                raise PlatformError(f'{name} 必须在 {lo}–{hi} 之间')
            req[name] = float(value)
        if not isinstance(req['name'], str) or len(req['name']) > 120:
            raise PlatformError('实验名称最多 120 字符')
        family = group.split('_')[0]
        if req['clientCount'] % FAMILIES[family][2]:
            raise PlatformError('客户端数量必须能被域数整除，禁止自动修改页面参数')
        if req['sampleClients'] > req['clientCount']:
            raise PlatformError('每轮参与数不能超过客户端总数')
        if req['evaluationFrequency'] > req['rounds']:
            raise PlatformError('评测频率不能大于轮数')
        if family == 'digit3' and any(req[k] != self.defaults(group, method)[k]
                                     for k in ('clientCount', 'alpha', 'splitSeed')):
            raise PlatformError('Digits 使用已有固定划分：60 客户端、alpha=0.1、splitSeed=42')
        return req

    def build(self, req, output):
        path = self.source(req['group'], req['method'])
        raw = copy.deepcopy(yaml.safe_load(path.read_text(encoding='utf-8')))
        output = Path(output).resolve()
        family = req['group'].split('_')[0]
        raw.update(use_gpu=req['gpu'] >= 0, device=max(0, req['gpu']), seed=req['seed'],
                   outdir=str(output / 'logs'), expname='training', log_file='')
        raw['federate'].update(mode='standalone', process_num=0,
                               client_num=req['clientCount'],
                               sample_client_num=req['sampleClients'],
                               total_round_num=req['rounds'] + 1)
        raw['dataloader'].update(batch_size=req['batchSize'], num_workers=0)
        raw['train']['local_update_steps'] = req['localEpochs']
        raw['train']['optimizer']['lr'] = req['learningRate']
        raw.setdefault('eval', {})['freq'] = req['evaluationFrequency']
        raw['data']['root'] = str(self.datasets / FAMILIES[family][1])
        g = raw['ggeur']
        g.update(hierarchical_training=False, use_feature_cache=True,
                 require_complete_feature_cache=True,
                 feature_cache_dir=str(self.cache_dir(req['group'])),
                 reuse_augmented_feature_cache=False, save_augmented_feature_cache=False,
                 num_generated_per_sample=0, num_generated_per_prototype=0,
                 target_size_per_class=0, baseline_target_samples_per_client=req['samplesPerClient'],
                 platform_target_samples_per_client=0, platform_auto_target_samples_per_client=0,
                 save_mlp_checkpoint=True, mlp_checkpoint_dir=str(output / 'checkpoints'),
                 data_split_seed=req['splitSeed'], officehome_data_seed=req['splitSeed'],
                 lds_seed=req['splitSeed'], lds_alpha=req['alpha'],
                 classifier_init_seed=req['seed'], training_data_seed=req['seed'],
                 runtime_seed=req['seed'], min_statistics_clients=req['clientCount'],
                 min_augmentation_clients=req['clientCount'], min_train_updates=0,
                 task_adaptation_file='', training_distribution_dir=str(output / 'distributions'))
        # Keep each original method's optimizer, prototype, proximal and MOON flags.
        # Normalize only the shared data protocol, explicitly recorded above.
        if family == 'digit3':
            g.update(digit3_manifest_base=str(self.datasets / 'digit_three_domain/manifests'),
                     digit3_manifest_path='', digit3_manifest_use_config_root=True,
                     digit3_global_manifest_path=str(self.datasets / 'digit_three_domain/dataset_manifest.json'))
        if family == 'domainnet':
            g['domainnet_manifest_path'] = str(self.resources / 'exp/distributed_manifests/domainnet_4domains/domainnet_manifest.json')
        if family == 'mdsent':
            raw['data']['dirichlet_alpha'] = req['alpha']
            raw['data']['args'][0]['seed'] = req['splitSeed']
            g['bert_model_path'] = str(self.resources / 'pretrained_models/nlptown_bert_base_multilingual_uncased_senti')
        return raw, {'source': str(path.relative_to(self.repo)), 'sourceSha256': sha256(path),
                     'testCacheDir': os.environ.get('FS_PLATFORM_TEST_CACHE_' + req['group'].upper()),
                     'protocol': 'cache-only-v1', 'parameterBindings': {
                         'rounds': 'federate.total_round_num = rounds + 1 (round 0 is initialization)', 'clientCount': 'federate.client_num',
                         'sampleClients': 'federate.sample_client_num (0=all)',
                         'batchSize': 'dataloader.batch_size', 'localEpochs': 'train.local_update_steps',
                         'learningRate': 'train.optimizer.lr', 'gpu': 'use_gpu/device',
                         'seed': 'seed/classifier_init_seed/training_data_seed/runtime_seed',
                         'splitSeed': 'data_split_seed/officehome_data_seed/lds_seed/data.args.seed',
                         'alpha': 'lds_alpha/data.dirichlet_alpha',
                         'samplesPerClient': 'baseline_target_samples_per_client (0=all)',
                         'evaluationFrequency': 'eval.freq'}}
