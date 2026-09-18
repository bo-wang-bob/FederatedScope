"""Adapt uploaded folders to the existing CNN training protocols."""
import copy
import os
from pathlib import Path

from .platform_config import PlatformError
from .uploaded_datasets import DatasetStore


class UploadedConfig:
    def __init__(self, base):
        self.base = base
        self.store = DatasetStore(base.repo)

    def __getattr__(self, name):
        return getattr(self.base, name)

    def defaults(self, group, method):
        if not group.startswith('uploaded_'):
            return self.base.defaults(group, method)
        self.store.training(group)
        req = self.base.defaults('officehome_cnn', method)
        req.update(group=group, clientCount=3,
                   gpu=-1 if os.environ.get('FS_PLATFORM_DEVICE') == 'cpu' else 0,
                   sampleClients=0, augmentationSourceId='',
                   augmentationMode='generate' if method == 'heterogeneous_solution' else 'none')
        return req

    def normalize(self, payload):
        group = payload.get('group', '') if isinstance(payload, dict) else ''
        if not isinstance(group, str) or not group.startswith('uploaded_'):
            return self.base.normalize(payload)
        self.store.training(group)
        req = {**self.defaults(group, payload.get('method', 'fedavg')), **payload}
        clients, sampled = req['clientCount'], req['sampleClients']
        if type(clients) is not int or not 1 <= clients <= 240 or type(sampled) is not int or not 0 <= sampled <= clients:
            raise PlatformError('客户端数应为 1–240，每轮参与数不能超过总数')
        if req.get('augmentationMode') == 'reuse' or req.get('augmentationSourceId'):
            raise PlatformError('上传数据集请重新生成增强特征，不可复用其他数据集缓存')
        validated = self.base.normalize({**req, 'group': 'officehome_cnn', 'clientCount': 4, 'sampleClients': 0})
        validated.update(group=group, clientCount=clients, sampleClients=sampled)
        return validated

    def build(self, req, output):
        if not req['group'].startswith('uploaded_'):
            return self.base.build(req, output)
        raw, provenance = self.base.build({**req, 'group': 'officehome_cnn'}, output)
        value = self.store.configure(raw, req['group'])
        raw['ggeur'].update(require_complete_feature_cache=True, use_feature_cache=True)
        provenance.update(uploadedDataset=value['id'], datasetFingerprint=value['fingerprint'],
                          datasetSplit='explicit-folders' if value.get('layout') == 'split' else 'seeded-70:30',
                          protocol='uploaded-folder-cnn / recorded split / original algorithms')
        return raw, provenance

    def catalog(self):
        catalog = self.base.catalog()
        for value in self.store.list():
            if value['kind'] != 'train':
                continue
            group = value['group']
            methods = [dict(id=m, label=label, enabled=True, reason=None, augmentedCacheFound=False,
                            defaults=self.defaults(group, m))
                       for m, label in [('fedavg', 'FedAvg'), ('fedprox', 'FedProx'), ('heterogeneous_solution', '本架构')]]
            catalog['groups'].append(dict(id=group, dataset=value['name'], backbone='cnn', domains=1,
                methods=methods, cacheFound=True, cacheFiles=0, cacheBytes=0, partitionLocked=False,
                uploaded=True, featurePreparation='首次预检提取特征', classes=value['classes']))
        return catalog
