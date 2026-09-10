"""Isolated cache-only adapter around the existing FederatedScope algorithms.

The control service never imports CUDA. This worker owns one process group,
checks every sample against frozen features, and never loads a backbone.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('FEDERATEDSCOPE_GGEUR_LIGHTWEIGHT', '1')
os.environ.setdefault('OMP_NUM_THREADS', '2')

from federatedscope.standalone_api.repository import JsonRepository


def save(path, value):
    JsonRepository._atomic_write(Path(path), value)


def digest(path):
    result = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def emit(kind, **payload):
    print('__PLATFORM__' + json.dumps(
        {'type': kind, **payload}, ensure_ascii=False, allow_nan=False), flush=True)


def sample_refs(dataset):
    """Resolve sample identities without decoding images or extracting data."""
    from torch.utils.data import Subset, ConcatDataset
    if isinstance(dataset, Subset):
        refs = sample_refs(dataset.dataset)
        return [refs[int(i)] for i in dataset.indices]
    if isinstance(dataset, ConcatDataset):
        return [item for child in dataset.datasets for item in sample_refs(child)]
    domain = getattr(dataset, 'domain', None)
    labels = getattr(dataset, 'targets', None)
    if labels is None or not domain:
        raise ValueError('数据集缺少可核验的样本标签或域信息')
    if hasattr(dataset, 'get_id'):
        paths = [dataset.get_id(i) for i in range(len(dataset))]
    else:
        paths = getattr(dataset, 'data', [])
    if len(paths) != len(labels) or not all(isinstance(p, str) for p in paths):
        raise ValueError('数据集缺少可追溯样本 ID，不允许重新提取特征')
    return [(str(domain), str(p), int(y)) for p, y in zip(paths, labels)]


def prepare(spec):
    import numpy as np
    import torch
    from federatedscope.core.configs.config import global_cfg
    from federatedscope.core.auxiliaries.data_builder import get_data
    from federatedscope.core.auxiliaries.utils import setup_seed
    from federatedscope.contrib.worker.ggeur_client import GGEURClient

    torch.set_num_threads(2)
    cfg = global_cfg.clone()
    cfg.merge_from_file(spec['configPath'])
    setup_seed(cfg.seed)
    data, modified = get_data(cfg.clone())
    cfg.merge_from_other_cfg(modified)
    # Only use the original loader for its deterministic sample membership.
    # Cache-only subclasses below never call its image __getitem__.
    probe = object.__new__(GGEURClient)
    probe._cfg = cfg
    probe.ggeur_cfg = cfg.ggeur
    probe.feature_extractor_type = cfg.ggeur.feature_extractor
    probe.embedding_dim = cfg.ggeur.embedding_dim
    probe.ID = 0
    cache_root = Path(cfg.ggeur.feature_cache_dir)
    if not cache_root.is_dir():
        raise ValueError(f'缺少特征缓存：{cache_root}')
    caches, files, clients, tests, train_ids = {}, {}, {}, {}, set()
    partition, class_names = {}, None
    for client_id in sorted(k for k in data if k > 0):
        entry = data[client_id]
        loader = entry.get('train')
        if loader is None:
            raise ValueError(f'客户端 {client_id} 没有训练集')
        train = []
        partition[str(client_id)] = []
        for split in ('train', 'test'):
            loader = entry.get(split)
            if loader is None:
                continue
            dataset = loader.dataset if hasattr(loader, 'dataset') else loader
            base = dataset
            while hasattr(base, 'dataset'):
                base = base.dataset
            names = getattr(base, 'classes', getattr(base, 'CLASSES', None))
            if class_names is None and names is not None:
                class_names = [str(x) for x in names]
            for domain, path, label in sample_refs(dataset):
                if label < 0 or label >= cfg.model.num_classes:
                    raise ValueError(f'{domain} 标签 {label} 超出模型类别范围')
                if domain not in caches:
                    cache_path = Path(probe._get_feature_cache_path(domain))
                    if not cache_path.is_file():
                        raise ValueError(f'{domain} 缺少缓存文件：{cache_path.name}')
                    with np.load(cache_path, allow_pickle=False) as bundle:
                        paths, features = bundle['paths'], bundle['features']
                        if (features.ndim != 2 or len(paths) != len(features)
                                or features.shape[1] != cfg.ggeur.embedding_dim
                                or not np.isfinite(features).all()):
                            raise ValueError(f'{domain} 缓存维度、数量或数值非法')
                        keys = [probe._feature_cache_key(str(p)) for p in paths]
                        if len(set(keys)) != len(keys):
                            raise ValueError(f'{domain} 缓存存在重复样本 ID')
                        caches[domain] = dict(zip(keys, features))
                    files[cache_path.name] = digest(cache_path)
                key = probe._feature_cache_key(path)
                if key not in caches[domain] and split == 'train':
                    raise ValueError(f'{domain} 的 {split} 特征缓存不完整：缺少 {key}')
                identity = domain + '/' + key
                if split == 'train':
                    train.append((domain, key, label))
                    train_ids.add(identity)
                    partition[str(client_id)].append([domain, key, label])
                else:
                    previous = tests.setdefault(domain, {}).get(key)
                    if previous is not None and previous != label:
                        raise ValueError(f'{domain}/{key} 标签不一致')
                    tests[domain][key] = label
        if not train:
            raise ValueError(f'客户端 {client_id} 训练集为空')
        clients[int(client_id)] = train
    overlap = train_ids & {d + '/' + k for d, rows in tests.items() for k in rows}
    if overlap:
        raise ValueError(f'训练/测试集交叉，拒绝运行：{len(overlap)} 个样本')
    if not tests or any(not rows for rows in tests.values()):
        raise ValueError('没有完整的独立测试集')
    raw_files = dict(files)
    test_provenance = {}
    from federatedscope.contrib.worker.ggeur_server import GGEURServer
    server_probe = object.__new__(GGEURServer)
    server_probe._cfg, server_probe.ggeur_cfg = cfg, cfg.ggeur
    server_probe.feature_extractor_type = cfg.ggeur.feature_extractor
    test_root = spec.get('provenance', {}).get('testCacheDir')
    if test_root:
        if not Path(test_root).is_dir():
            raise ValueError(f'缺少已登记的测试缓存目录：{test_root}')
        # A separate existing test cache must never replace the training cache.
        server_probe.ggeur_cfg = cfg.ggeur.clone()
        server_probe.ggeur_cfg.feature_cache_dir = test_root
    for domain, rows in tests.items():
        if all(key in caches[domain] for key in rows):
            test_provenance[domain] = 'embedded-sample-ids'
            continue
        # Legacy deployments saved train and held-out test caches separately.
        # Resolve the exact original cache key (backbone/split/seed), require
        # the complete deterministic label sequence, and record the weaker
        # provenance explicitly. Never generate or guess another split.
        test_path = Path(server_probe._get_test_cache_path(domain))
        if not test_path.is_file():
            raise ValueError(f'{domain} 缺少完整测试缓存：{test_path.name}')
        with np.load(test_path, allow_pickle=False) as bundle:
            features, labels = bundle['features'], bundle['labels']
            if (features.shape != (len(rows), cfg.ggeur.embedding_dim)
                    or not np.isfinite(features).all()
                    or not np.array_equal(labels, np.asarray(list(rows.values())))):
                raise ValueError(f'{domain} 测试缓存与划分样本数、标签顺序或特征维度不一致')
            caches[domain].update(dict(zip(rows, features)))
        files[test_path.name] = digest(test_path)
        test_provenance[domain] = 'legacy-split-seed-ordered-labels; no embedded sample IDs'
    if len(clients) != spec['request']['clientCount']:
        raise ValueError('数据加载器改变了客户端数量，拒绝运行')
    partition_fingerprint = hashlib.sha256(json.dumps(partition, sort_keys=True).encode()).hexdigest()
    feature_space = hashlib.sha256(json.dumps({'files': raw_files, 'classes': class_names,
        'dimension': cfg.ggeur.embedding_dim, 'classCount': cfg.model.num_classes,
        'group': spec['request']['group']}, sort_keys=True).encode()).hexdigest()
    fingerprint = hashlib.sha256(json.dumps({
        'files': files, 'partition': partition, 'test': tests,
        'classes': class_names, 'featureSpace': feature_space,
    }, sort_keys=True, default=str).encode()).hexdigest()
    # Evaluation compatibility excludes task-specific parameters and paths.
    test_fingerprint = hashlib.sha256(json.dumps({
        'files': files, 'test': tests, 'classes': class_names,
        'group': spec['request']['group'],
    }, sort_keys=True).encode()).hexdigest()
    info = {
        'ready': True, 'fingerprint': fingerprint,
        'partitionFingerprint': partition_fingerprint, 'featureSpace': feature_space,
        'testProvenance': test_provenance,
        'runtime': {'torch': str(torch.__version__), 'numpy': np.__version__,
                    'cuda': torch.version.cuda, 'inference': 'CPU float32 / batch 256 / threads 2'},
        'testFingerprint': test_fingerprint, 'cacheFiles': files,
        'cacheLocations': {'features': str(cache_root), 'tests': test_root or str(cache_root)},
        'featureDimension': int(cfg.ggeur.embedding_dim),
        'classCount': int(cfg.model.num_classes),
        'classes': class_names or [str(i) for i in range(cfg.model.num_classes)],
        'clientCount': len(clients), 'trainSamples': sum(map(len, clients.values())),
        'testSamples': sum(map(len, tests.values())),
        'clients': [{'id': i, 'domain': refs[0][0], 'samples': len(refs),
                     'histogram': np.bincount([r[2] for r in refs],
                                              minlength=cfg.model.num_classes).tolist()}
                    for i, refs in clients.items()],
        'domains': [{'name': d, 'testSamples': len(rows)} for d, rows in tests.items()],
    }
    mode = spec['request'].get('augmentationMode', 'none')
    if mode == 'reuse':
        from federatedscope.standalone_api.platform_augmentation import inspect_existing
        emit('stage', stage='逐客户端核验已有增强缓存')
        info['augmentation'] = inspect_existing(probe, clients, caches, info,
            spec['request'].get('allowLegacyAugmentation', False), spec.get('registeredAugmentation'))
    elif mode == 'generate':
        # Preflight is read-only: statistics / generation happen only after start.
        import shutil
        target = int(cfg.ggeur.target_size_per_class)
        prototype_limit = int(cfg.ggeur.max_cross_client_prototypes_per_class) or (
            len(clients) * int(cfg.ggeur.local_prototypes_per_class))
        estimates = [(target * cfg.model.num_classes if target else
            len(refs) * (1 + cfg.ggeur.num_generated_per_sample)
            + cfg.model.num_classes * prototype_limit * cfg.ggeur.num_generated_per_prototype)
            * cfg.ggeur.embedding_dim * 4 for refs in clients.values()]
        if max(estimates) > 240 * 1024 ** 2:
            raise ValueError('预计单客户端增强缓存超过 240 MiB，请降低生成数或每类目标数')
        if sum(estimates) * 1.2 > shutil.disk_usage(spec['output']).free:
            raise ValueError('本实验目录剩余空间不足以保存预计增强缓存')
        info['augmentation'] = dict(mode='generate', provenance='pending-generation',
            warning=None, clients={}, generatedClients=0, cachedSamples=0,
            estimatedCacheBytes=sum(estimates))
    else:
        info['augmentation'] = dict(mode='none', provenance='original-features', warning=None)
    cfg.freeze(inform=False, save=False)
    resolved = cfg.clone()
    resolved.defrost()
    resolved.clear_aux_info()
    (Path(spec['output']) / 'resolved_config.yaml').write_text(resolved.dump(), encoding='utf-8')
    return cfg, data, clients, caches, tests, info, partition


def train(spec):
    import numpy as np
    import torch
    from federatedscope.contrib.worker.ggeur_client import GGEURClient
    from federatedscope.contrib.worker.ggeur_server import GGEURServer
    from federatedscope.core.auxiliaries.runner_builder import get_runner
    from federatedscope.core.auxiliaries.logging import update_logger

    started = time.monotonic()
    emit('stage', stage='检查样本与特征缓存')
    cfg, data, clients, caches, tests, info, partition = prepare(spec)
    output = Path(spec['output'])
    if (spec.get('expectedData', info['testFingerprint']) != info['testFingerprint']
            or spec.get('expectedPartition', info['partitionFingerprint']) != info['partitionFingerprint']):
        raise ValueError('缓存或划分在预检后发生变化，请重新预检')
    if spec.get('expectedAugmentation') != info['augmentation'].get('fingerprint') and spec.get('expectedAugmentation'):
        raise ValueError('增强缓存在预检后变化，请重新预检')
    save(output / 'data_manifest.json', {**info, 'partition': partition, 'test': tests})
    emit('prepared', **info)
    if spec['action'] == 'inspect':
        save(output / 'result.json', info)
        return
    use_gpu = bool(cfg.use_gpu)
    if use_gpu and (not torch.cuda.is_available() or cfg.device >= torch.cuda.device_count()):
        raise ValueError(f'GPU {cfg.device} 不可用')
    training_stats, participants, training_exposure = {}, {}, {}
    # Frozen backbone features only. Augmentation is explicitly selected by the user.
    class CachedClient(GGEURClient):
        def _load_feature_extractor(self):
            raise RuntimeError('仅缓存模式禁止加载特征提取器')

        def _extract_features(self):
            refs = clients[self.ID]
            self.local_features = {}
            for domain, key, label in refs:
                self.local_features.setdefault(label, []).append(caches[domain][key])
            self.local_features = {k: np.asarray(v, dtype=np.float32)
                                   for k, v in self.local_features.items()}
            self.local_labels = {k: [k] * len(v) for k, v in self.local_features.items()}
            emit('client', clientId=self.ID, domain=refs[0][0], stage='缓存已加载',
                 samples=len(refs), round=int(self.state))

        def _extract_text_features(self):
            self._extract_features()

        def _try_load_augmented_feature_cache(self, restore_statistics=False):
            if info['augmentation']['mode'] != 'reuse':
                return False
            from federatedscope.standalone_api.platform_augmentation import safe_load, validate_arrays
            row = info['augmentation']['clients'][str(self.ID)]
            if digest(row['path']) != row['sha256']:
                raise ValueError(f'客户端 {self.ID} 增强缓存已变化')
            payload = safe_load(row['path'])
            self.augmented_features, self.augmented_labels = validate_arrays(payload,
                cfg.ggeur.embedding_dim, cfg.model.num_classes)
            self._capture_generated_distribution_snapshot()
            self.global_prototypes = self._copy_numpy_mapping(payload.get('global_prototypes', {}))
            if restore_statistics:
                self._restore_cached_local_statistics(payload.get('local_statistics', {}))
            self._apply_platform_training_sampling()
            from federatedscope.contrib.worker.ggeur_client import AugmentedFeatureDataset
            self.augmented_loader = torch.utils.data.DataLoader(
                AugmentedFeatureDataset(self.augmented_features, self.augmented_labels),
                batch_size=cfg.dataloader.batch_size, shuffle=True,
                generator=self._training_loader_generator())
            self._emit_training_distribution(cache_hit=True)
            emit('client', clientId=self.ID, stage='增强缓存已加载',
                 augmentedSamples=len(self.augmented_labels), cacheSha256=row['sha256'])
            return True

        def _compute_local_statistics(self):
            emit('client', clientId=self.ID, stage='计算本地统计', round=0)
            return super()._compute_local_statistics()

        def _save_augmented_feature_cache(self):
            if info['augmentation']['mode'] != 'generate':
                return
            from federatedscope.standalone_api.platform_augmentation import validate_arrays
            # Save before post-generation training sampling, just as the original
            # algorithm does. Persist only this experiment's artifacts atomically.
            payload = dict(features=torch.as_tensor(self.augmented_features).float(),
                labels=torch.as_tensor(self.augmented_labels).long(),
                metadata=self._augmented_cache_metadata(),
                global_prototypes={int(k): torch.as_tensor(v) for k, v in (self.global_prototypes or {}).items()},
                source_samples=[list(r) for r in clients[self.ID]],
                source_feature_space=info['featureSpace'], source_partition=info['partitionFingerprint'],
                source_files=info['cacheFiles'])
            validate_arrays(payload, cfg.ggeur.embedding_dim, cfg.model.num_classes)
            target = output / 'augmented_cache' / f'client_{self.ID:06d}.pt'
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix('.pt.tmp')
            torch.save(payload, temporary)
            os.replace(temporary, target)
            row = dict(path=str(target), sha256=digest(target), samples=len(self.augmented_labels),
                histogram=np.bincount(self.augmented_labels, minlength=cfg.model.num_classes).tolist())
            info['augmentation']['clients'][str(self.ID)] = row
            info['augmentation']['generatedClients'] = len(info['augmentation']['clients'])
            info['augmentation']['cachedSamples'] = sum(r['samples'] for r in info['augmentation']['clients'].values())
            emit('client', clientId=self.ID, stage='增强数据已生成',
                 augmentedSamples=row['samples'], cacheSha256=row['sha256'])

        def _perform_augmentation(self):
            if info['augmentation']['mode'] == 'reuse':
                if not self._try_load_augmented_feature_cache():
                    raise RuntimeError(f'客户端 {self.ID} 缺少匹配的增强缓存；禁止临时生成')
                self.augmentation_done = True
            else:
                if info['augmentation']['mode'] == 'generate':
                    emit('stage', stage=f'按配置生成增强数据：客户端 {self.ID}/{len(clients)}')
                super()._perform_augmentation()

        def _train_on_augmented_data(self):
            emit('client', clientId=self.ID, domain=clients[self.ID][0][0],
                 stage='本地训练', round=int(self.state))
            result = super()._train_on_augmented_data()
            training_exposure.setdefault(str(self.state), {})[str(self.ID)] = dict(
                samples=int(result[0]), optimizerSteps=len(self.augmented_loader) * cfg.train.local_update_steps)
            training_stats.setdefault(int(self.state), []).append((int(result[0]), float(result[2]['train_loss'])))
            emit('client', clientId=self.ID, domain=clients[self.ID][0][0],
                 stage='已上传', round=int(self.state), loss=float(result[2]['train_loss']),
                 accuracy=float(result[2]['train_acc']), trainedSamples=int(result[0]))
            return result

    class CachedServer(GGEURServer):
        def broadcast_model_para(self, msg_type='model_para', sample_client_num=-1, filter_unseen_clients=True):
            # Every client must prepare before per-round sampling. The legacy
            # distributed bootstrap sampled only once and could stall at quorum.
            if self.state == 0:
                sample_client_num = -1
            return super().broadcast_model_para(msg_type, sample_client_num, filter_unseen_clients)

        def _start_training_round(self):
            count = int(spec['request']['sampleClients']) or len(clients)
            all_ids = sorted(clients)
            if count < len(clients):
                rng = np.random.default_rng(int(cfg.seed) * 1000003 + int(self.state))
                selected = sorted(int(x) for x in rng.choice(all_ids, size=count, replace=False))
            else:
                selected = all_ids
            self.training_client_ids = set(selected)
            participants[str(self.state)] = selected
            emit('round_started', round=int(self.state), participants=selected)
            return super()._start_training_round()

        def _should_run_eval(self, round_idx):
            # Always capture the actual final model, even if frequency does not
            # divide the requested number of updates.
            return int(round_idx) == int(spec['request']['rounds']) or super()._should_run_eval(round_idx)

        def _load_feature_extractor(self):
            raise RuntimeError('仅缓存模式禁止加载特征提取器')

        def _load_test_data_and_features(self):
            if not self.test_data_loaded:
                self.test_features = {d: np.asarray([caches[d][k] for k in rows], dtype=np.float32)
                                      for d, rows in tests.items()}
                self.test_labels = {d: np.asarray(list(rows.values()), dtype=np.int64)
                                    for d, rows in tests.items()}
                self.test_data_loaded = True

        def _evaluate_on_test_sets(self):
            if self.global_mlp is None:
                return {}
            self._load_test_data_and_features()
            # One identical CPU/batch-size inference contract for monitoring
            # and reloaded heads. GPU/CPU LSTM kernels can flip near-tied logits.
            if self.domain_prototype_ensemble or getattr(self, 'domain_personalized_heads', {}):
                raise ValueError('当前检查点协议不支持额外的推理集成头')
            head = copy.deepcopy(self.global_mlp).cpu().eval()
            result = {}
            for domain, features in self.test_features.items():
                cm = predict_confusion(head, features, self.test_labels[domain], cfg.model.num_classes)
                result[domain] = float(cm.diagonal().sum() / cm.sum())
                self.test_accuracies_history.setdefault(domain, []).append(result[domain])
            result['average'] = sum(result.values()) / len(result)
            self.test_accuracies_history.setdefault('average', []).append(result['average'])
            if result['average'] > self.best_avg_accuracy:
                self.best_avg_accuracy = result['average']
                self.best_model_state = copy.deepcopy(self.global_mlp.state_dict())
            del head
            if result:
                domains = {k: v for k, v in result.items() if k != 'average'}
                total = sum(len(tests[d]) for d in domains)
                stats = training_stats.get(int(self.state), [])
                emit('metrics', round=int(self.state), domains=domains,
                     domainMean=result['average'],
                     trainLoss=(sum(n * loss for n, loss in stats) / sum(n for n, _ in stats)) if stats else None,
                     accuracy=sum(v * len(tests[d]) for d, v in domains.items()) / total,
                     worstDomain=min(domains.values()),
                     domainGap=max(domains.values()) - min(domains.values()))
            return result

    update_logger(cfg, clear_before_add=True)
    emit('stage', stage='单机联邦训练')
    runner = get_runner(server_class=CachedServer, client_class=CachedClient,
                        config=cfg, data=data)
    runner.run()
    checkpoint = output / 'checkpoints' / 'mlp_final.pt'
    if not checkpoint.is_file() or runner.server.state <= cfg.federate.total_round_num:
        raise RuntimeError('训练未完整结束或缺少模型产物')
    if set(training_stats) != set(range(1, spec['request']['rounds'] + 1)):
        raise RuntimeError('实际训练更新轮数与页面参数不一致')
    if info['augmentation']['mode'] == 'generate':
        if info['augmentation']['generatedClients'] != len(clients):
            raise ValueError('并非全部客户端完成增强生成，拒绝标记成功')
        info['augmentation']['provenance'] = 'generated-from-recorded-training-samples'
        info['augmentation']['fingerprint'] = hashlib.sha256(json.dumps(
            info['augmentation']['clients'], sort_keys=True).encode()).hexdigest()
    save(output / 'data_manifest.json', {**info, 'partition': partition, 'test': tests})
    emit('augmentation', **info['augmentation'])
    artifacts = {p.name: digest(p) for p in checkpoint.parent.iterdir() if p.is_file()}
    save(output / 'result.json', {**info, 'artifactHashes': artifacts,
         'checkpoint': 'checkpoints/mlp_final.pt', 'history': runner.server.test_accuracies_history,
         'completedRounds': len(training_stats), 'participants': participants,
         'trainingExposure': training_exposure,
         'elapsedSeconds': time.monotonic() - started})
    emit('stage', stage='模型与结果已保存')


def classification_metrics(matrix):
    import numpy as np
    cm = np.asarray(matrix, dtype=np.int64)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    tp = cm.diagonal()
    precision = np.divide(tp, predicted, out=np.zeros(len(tp)), where=predicted != 0)
    recall = np.divide(tp, support, out=np.zeros(len(tp)), where=support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(len(tp)), where=(precision + recall) != 0)
    active = (support + predicted) > 0
    return {'accuracy': float(tp.sum() / max(1, support.sum())),
            'macroF1': float(f1[active].mean()) if active.any() else 0.0,
            'precision': float(precision[active].mean()) if active.any() else 0.0,
            'recall': float(recall[active].mean()) if active.any() else 0.0,
            'samples': int(support.sum()),
            'perClass': [{'classIndex': i, 'support': int(support[i]),
                          'precision': float(precision[i]), 'recall': float(recall[i]),
                          'f1': float(f1[i])} for i in range(len(tp))],
            'confusionMatrix': cm.tolist()}


def predict_confusion(model, features, labels, classes):
    import numpy as np
    import torch
    features = torch.as_tensor(features, dtype=torch.float32, device='cpu')
    labels = torch.as_tensor(labels, dtype=torch.int64, device='cpu')
    cm = np.zeros((classes, classes), dtype=np.int64)
    with torch.inference_mode():
        for offset in range(0, len(labels), 256):
            y = labels[offset:offset + 256].numpy()
            logits = model(features[offset:offset + 256])
            if not torch.isfinite(logits).all():
                raise ValueError('分类器输出非有限数值，拒绝保存为成功实验')
            pred = logits.argmax(1).numpy()
            np.add.at(cm, (y, pred), 1)
    return cm


def evaluate(spec):
    import numpy as np
    import torch
    from scripts.test_outline_validation.evaluate_saved_mlp import build_model
    started = time.monotonic()
    torch.set_num_threads(2)
    if digest(spec['checkpointPath']) != spec['checkpointHash'] or digest(spec['bundlePath']) != spec['bundleHash']:
        raise ValueError('模型或测试包哈希变化，拒绝加载')
    checkpoint = torch.load(spec['checkpointPath'], map_location='cpu', weights_only=True)
    bundle = torch.load(spec['bundlePath'], map_location='cpu', weights_only=True)
    if checkpoint['backbone'] != bundle['backbone'] or checkpoint['dataset'] != bundle['dataset']:
        raise ValueError('模型和测试集的特征空间或数据集不匹配')
    model = build_model(checkpoint['architecture'])
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval()
    n = int(checkpoint['architecture']['num_classes'])
    selected = spec['request'].get('domains') or list(bundle['features'])
    classes = spec['request'].get('classes') or list(range(n))
    if not set(selected) <= set(bundle['features']) or not set(classes) <= set(range(n)):
        raise ValueError('选择的域或类别不属于此测试集')
    matrix = np.zeros((n, n), dtype=np.int64)
    domains = {}
    with torch.inference_mode():
        for domain in selected:
            features = bundle['features'][domain].float()
            labels = bundle['labels'][domain].long()
            mask = torch.tensor([int(y) in classes for y in labels], dtype=torch.bool)
            features, labels = features[mask], labels[mask]
            if not len(labels):
                continue
            cm = predict_confusion(model, features, labels, n)
            domains[domain] = classification_metrics(cm)
            matrix += cm
            emit('stage', stage=f'已评测 {domain}', completed=len(domains), total=len(selected))
    if not matrix.sum():
        raise ValueError('选中的测试子集没有样本')
    result = classification_metrics(matrix)
    accuracies = [m['accuracy'] for m in domains.values()]
    result.update(domains=domains, domainMean=float(np.mean(accuracies)),
                  worstDomain=min(accuracies), domainGap=max(accuracies)-min(accuracies),
                  elapsedSeconds=time.monotonic()-started,
                  metricDefinition='accuracy=sample weighted; domainMean=equal-domain; macro=present labels',
                  checkpointSha256=digest(spec['checkpointPath']),
                  testBundleSha256=digest(spec['bundlePath']))
    save(Path(spec['output']) / 'result.json', result)


def predict(spec):
    """Load the actual frozen classifier and infer, never infer from the label."""
    import torch
    from scripts.test_outline_validation.evaluate_saved_mlp import build_model
    started = time.monotonic()
    torch.set_num_threads(2)
    for path, expected in [('checkpointPath', 'checkpointHash'), ('bundlePath', 'bundleHash'),
                           ('manifestPath', 'manifestHash')]:
        if digest(spec[path]) != spec[expected]:
            raise ValueError('模型、测试包或样本清单已改变，拒绝错位预测')
    sample = spec['sample']
    if digest(sample['imagePath']) != spec['request']['imageSha256']:
        raise ValueError('原图已改变，请刷新样本')
    emit('stage', stage='加载已保存模型与冻结测试特征')
    checkpoint = torch.load(spec['checkpointPath'], map_location='cpu', weights_only=True)
    bundle = torch.load(spec['bundlePath'], map_location='cpu', weights_only=True)
    if checkpoint['backbone'] != bundle['backbone'] or checkpoint['dataset'] != bundle['dataset']:
        raise ValueError('模型与测试特征空间不匹配')
    model = build_model(checkpoint['architecture'])
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval()
    domain, index = sample['domain'], sample['index']
    features, labels = bundle['features'][domain], bundle['labels'][domain]
    manifest = json.loads(Path(spec['manifestPath']).read_text(encoding='utf-8'))
    refs = list(manifest['test'][domain].items())
    if (len(features) != len(refs) or len(labels) != len(refs)
            or refs[index] != (sample['key'], sample['label'])
            or int(labels[index]) != sample['label']):
        raise ValueError('样本、标签与冻结测试特征索引不一致')
    # Use the same original 256-row batch as full evaluation. Single-row GEMM
    # may round near-tied logits differently from the recorded evaluation.
    offset = index // 256 * 256
    block = features[offset:offset + 256].float()
    if not torch.isfinite(block).all():
        raise ValueError('测试特征包含非有限数值')
    emit('stage', stage='运行真实分类器推理')
    compute_started = time.monotonic()
    with torch.inference_mode():
        logits = model(block)[index - offset]
        if not torch.isfinite(logits).all():
            raise ValueError('分类器输出非有限数值')
        probabilities = torch.softmax(logits, dim=-1)
    inference_ms = (time.monotonic() - compute_started) * 1000
    names = spec['classNames']
    if len(logits) != len(names):
        raise ValueError('模型类别与测试集类别名称不一致')
    # Match torch.argmax's lowest-index tie break, without using ground truth.
    order = sorted(range(len(names)), key=lambda i: (-float(logits[i]), i))[:5]
    predicted = order[0]
    result = dict(sampleId=sample['id'], domain=domain, filename=sample['filename'],
                  label=sample['label'], labelName=names[sample['label']],
                  predictedClass=predicted, predictedName=names[predicted],
                  correct=predicted == sample['label'], confidence=float(probabilities[predicted]),
                  topK=[dict(classIndex=i, className=names[i], score=float(probabilities[i])) for i in order],
                  inferenceMs=inference_ms, elapsedSeconds=time.monotonic() - started,
                  checkpointSha256=spec['checkpointHash'], testBundleSha256=spec['bundleHash'],
                  manifestSha256=spec['manifestHash'], imageSha256=spec['request']['imageSha256'],
                  testProvenance=spec['testProvenance'], mode='frozen-feature-classifier',
                  architecture=checkpoint['architecture'],
                  inferenceContract='CPU float32 / original 256-row evaluation batch / threads 2',
                  scoreDefinition='Softmax scores; not calibrated correctness probabilities')
    save(Path(spec['output']) / 'result.json', result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('spec')
    args = parser.parse_args()
    spec = json.loads(Path(args.spec).read_text(encoding='utf-8'))
    import psutil
    save(Path(spec['output']) / 'process.json', {
        'pid': os.getpid(), 'created': psutil.Process().create_time(),
        'spec': str(Path(args.spec).resolve()),
    })
    try:
        if spec['action'] == 'predict':
            predict(spec)
        elif spec['action'] == 'evaluate':
            evaluate(spec)
        else:
            train(spec)
    except Exception as error:
        save(Path(spec['output']) / 'failure.json', {'message': str(error)})
        raise


if __name__ == '__main__':
    main()
