"""Read-only validation of existing augmentation artifacts; no generation fallback."""
from pathlib import Path
import hashlib
import json


LEGACY_WARNING = ('历史增强缓存未保存完整原始样本 ID/生成输入哈希；仅核验配置、客户端、'
                  '数值和文件版本，不能据此证明生成时无测试泄漏或严格方法提升。')


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def safe_load(path):
    import numpy as np
    import torch
    path = Path(path)
    if not path.is_file() or not 0 < path.stat().st_size <= 256 * 1024 ** 2:
        raise ValueError('增强缓存不存在或单文件超过 256 MiB')
    # Explicit numeric NumPy globals only. Never unpickle arbitrary Python code.
    with torch.serialization.safe_globals([
            np.core.multiarray._reconstruct, np.core.multiarray.scalar,
            np.ndarray, np.dtype, type(np.dtype('float32')),
            type(np.dtype('float64')), type(np.dtype('int64'))]):
        return torch.load(path, map_location='cpu', weights_only=True)


def validate_arrays(payload, dimension, classes):
    import numpy as np
    features, labels = np.asarray(payload['features']), np.asarray(payload['labels'])
    if (features.ndim != 2 or features.shape[1] != dimension or not len(features)
            or labels.shape != (len(features),) or labels.dtype.kind not in 'iu'
            or features.dtype.kind != 'f' or not np.isfinite(features).all()
            or labels.min() < 0 or labels.max() >= classes):
        raise ValueError('增强特征的数量、维度、数值或标签非法')
    for key, value in payload.get('global_prototypes', {}).items():
        value = np.asarray(value)
        if (int(key) < 0 or int(key) >= classes or value.ndim not in (1, 2)
                or value.shape[-1] != dimension or not np.isfinite(value).all()):
            raise ValueError('增强缓存的全局原型非法')
    return features.astype(np.float32, copy=False), labels.astype(np.int64, copy=False)


def metadata_matches(actual, expected):
    required = ('client_num', 'dataset', 'seed', 'splits', 'feature_extractor',
                'feature_extractor_model', 'embedding_dim', 'num_classes',
                'num_generated_per_sample', 'num_generated_per_prototype',
                'target_size_per_class', 'use_cross_client_prototypes',
                'max_cross_client_prototypes_per_class', 'use_fedproto', 'use_lds',
                'augmented_feature_cache_version')
    if any(actual.get(k) != expected.get(k) for k in required):
        return False
    if expected.get('use_lds') and any(actual.get(k) != expected.get(k) for k in ('lds_alpha', 'lds_seed')):
        return False
    defaults = {'local_prototypes_per_class': 1, 'generation_covariance_scale': 1.0,
                'cross_client_prototype_seed': 42, 'diagonal_covariance': False}
    if any(actual.get(k, v) != expected.get(k, v) for k, v in defaults.items()):
        return False
    # Post-generation sampling / classifier initialization do not change the
    # stored unsampled features. Task-specific generation, however, must match.
    from federatedscope.contrib.worker.ggeur_client import GGEURClient
    def task(meta):
        value = meta.get('task_adaptation', {})
        if GGEURClient._task_cache_metadata_is_generation_neutral(
                value, expected['num_classes'], expected['target_size_per_class']):
            return GGEURClient._disabled_task_cache_metadata(expected['target_size_per_class'])
        return value
    return task(actual) == task(expected)


def inspect_existing(probe, clients, caches, info, allow_legacy, registered=None):
    import numpy as np
    root = Path(probe.ggeur_cfg.augmented_feature_cache_dir).resolve()
    expected = probe._augmented_cache_metadata()
    version = root / 'augmented_features' / expected['augmented_feature_cache_version']
    if not registered and not version.is_dir():
        raise ValueError('缺少配置对应版本的增强缓存；请切换重新生成模式')
    if registered:
        paths = [Path(row['path']) for row in registered['clients'].values()]
        for row in registered['clients'].values():
            if file_hash(row['path']) != row['sha256']:
                raise ValueError('已登记的增强缓存被修改')
        namespaces = [sorted(paths)]
    else:
        namespaces = [sorted(p.glob('*.pt')) for p in version.iterdir() if p.is_dir()]
    candidates, errors = [], []
    # One coherent namespace only: never combine clients from different runs.
    for paths in namespaces:
        if len(paths) != len(clients):
            continue
        if any(root not in p.resolve().parents or p.is_symlink() for p in paths):
            raise ValueError('增强缓存路径越界')
        first = safe_load(paths[0])
        if not metadata_matches(first.get('metadata', {}), expected):
            continue
        try:
            rows, legacy = {}, False
            for path in paths:
                if root not in path.resolve().parents or path.is_symlink():
                    raise ValueError('增强缓存路径越界')
                payload = safe_load(path)
                metadata = payload.get('metadata', {})
                cid = metadata.get('client_id')
                if type(cid) is not int or cid not in clients or cid in rows:
                    raise ValueError('增强缓存客户端 ID 缺失、重复或不匹配')
                if not metadata_matches(metadata, expected):
                    raise ValueError(f'客户端 {cid} 增强缓存配置不一致')
                features, labels = validate_arrays(payload, expected['embedding_dim'], expected['num_classes'])
                original = [list(r) for r in clients[cid]]
                recorded = payload.get('source_samples')
                if recorded is not None:
                    if (recorded != original or payload.get('source_feature_space') != info['featureSpace']
                            or payload.get('source_partition') != info['partitionFingerprint']):
                        raise ValueError(f'客户端 {cid} 增强来源样本或特征版本不一致')
                else:
                    legacy = True
                stats = payload.get('local_statistics', {})
                counts = {int(k): int(v) for k, v in stats.get('counts', {}).items()}
                if counts:
                    observed = {label: sum(r[2] == label for r in original) for label in {r[2] for r in original}}
                    if counts != observed:
                        raise ValueError(f'客户端 {cid} 历史增强统计与当前划分不一致')
                    means = {int(k): v for k, v in stats.get('means', {}).items()}
                    for label in counts:
                        raw = np.asarray([caches[d][k] for d, k, y in original if y == label], dtype=np.float32)
                        if label not in means or not np.allclose(raw.mean(0), means[label], rtol=1e-5, atol=1e-6):
                            raise ValueError(f'客户端 {cid} 增强源特征均值不一致')
                rows[str(cid)] = dict(path=str(path), sha256=file_hash(path), samples=len(labels),
                    histogram=np.bincount(labels, minlength=expected['num_classes']).tolist())
            if set(map(int, rows)) != set(clients):
                raise ValueError('增强缓存客户端不完整')
            signature = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
            candidates.append(dict(mode='reuse', fingerprint=signature, clients=rows,
                provenance='legacy-metadata-only' if legacy else 'source-samples-and-feature-hash',
                warning=LEGACY_WARNING if legacy else None,
                cachedSamples=sum(r['samples'] for r in rows.values())))
        except (ValueError, KeyError) as e:
            errors.append(str(e))
    if not candidates:
        raise ValueError('没有完整匹配的增强缓存：' + (errors[0] if errors else '配置/版本/种子/客户端数不匹配'))
    if len(candidates) > 1:
        # Old neutral-task aliases may be duplicate copies. Accept identical
        # tensor contents, not merely matching labels, or an arbitrary first run.
        signatures = []
        for candidate in candidates:
            h = hashlib.sha256()
            for cid in sorted(candidate['clients'], key=int):
                payload = safe_load(candidate['clients'][cid]['path'])
                x, y = validate_arrays(payload, expected['embedding_dim'], expected['num_classes'])
                h.update(x.tobytes()); h.update(y.tobytes())
                for label, value in sorted(payload.get('global_prototypes', {}).items()):
                    h.update(str(label).encode()); h.update(np.asarray(value).tobytes())
            signatures.append(h.hexdigest())
        if len(set(signatures)) != 1:
            raise ValueError('有多个不同的匹配增强版本，拒绝自动混用；请使用重新生成模式')
    selected = candidates[0]
    if selected['warning'] and not allow_legacy:
        raise ValueError(LEGACY_WARNING + ' 请明确勾选历史缓存试跑确认，或选择重新生成。')
    return selected
