"""Local native GGEUR training from cloud presets (not the ViT cache adapter)."""
import copy
import hashlib
import json
from pathlib import Path
import time
import yaml

from .platform_worker import emit, save, sample_refs, digest
from .platform_paths import portable_config, resolve_path


def resolve_sample_image(path, root):
    """Accept loader paths relative to cwd or to the dataset, without double joining."""
    root = Path(root).resolve()
    image = Path(path)
    candidates = [image.resolve()]
    if not image.is_absolute():
        candidates.append((root / image).resolve())
    for candidate in candidates:
        if root in candidate.parents and candidate.is_file():
            return candidate
    raise ValueError(f'无法在数据集目录 {root} 内找到样本图片：{path}')


def run(spec):
    import torch
    # Keep the cloud backbone in the project's portable local cache.
    repo = Path(__file__).resolve().parents[2]
    local_hub = resolve_path(repo, 'resources/torch/hub')
    torch.hub.set_dir(str(local_hub))
    from federatedscope.core.configs.config import CN, global_cfg
    from federatedscope.core.auxiliaries.data_builder import get_data
    from federatedscope.core.auxiliaries.utils import setup_seed
    from federatedscope.core.auxiliaries.logging import update_logger
    from federatedscope.core.auxiliaries.runner_builder import get_runner
    from .platform_privacy_worker import privacy_bases, finalize
    started = time.monotonic()
    output = Path(spec['output'])
    raw = portable_config(yaml.safe_load(Path(spec['configPath']).read_text(encoding='utf-8')), repo)
    spec['cloudAttack'] = copy.deepcopy(raw['attack'])
    # These newer cloud fields are implemented by the private server adapter.
    raw['attack'].pop('fedmia_save_start_round', None)
    raw['attack'].pop('fedmia_save_only', None)
    cfg = global_cfg.clone()
    cfg.merge_from_other_cfg(CN(raw))
    setup_seed(cfg.seed)
    torch.set_num_threads(2)
    emit('stage', stage='核验云服务器预设、本地 GPU 与 ConvNeXt 权重')
    if cfg.use_gpu and (not torch.cuda.is_available() or cfg.device >= torch.cuda.device_count()):
        raise ValueError(f'云服务器预设使用 GPU {cfg.device}，当前后端 Python 环境不可用；请使用支持 CUDA 的环境')
    from torchvision.models import ConvNeXt_Base_Weights
    filename = ConvNeXt_Base_Weights.IMAGENET1K_V1.url.rsplit('/', 1)[-1]
    weight = Path(torch.hub.get_dir()) / 'checkpoints' / filename
    if not weight.is_file():
        raise ValueError(f'缺少云服务器同款 ConvNeXt-Base 预训练权重：{weight}。请将该权重放入本地缓存；不会自动下载或改用随机权重')
    emit('stage', stage='核验真实数据集和固定客户端划分')
    if spec['request']['group'].startswith('uploaded_'):
        from .uploaded_features import prepare_uploaded
        prepare_uploaded(cfg, spec['request']['group'])
    data, modified = get_data(cfg.clone())
    cfg.merge_from_other_cfg(modified)
    root = Path(cfg.data.root).resolve()
    partition, tests, clients, classes = {}, {}, [], None
    hashes = {}
    def refs(dataset):
        result = []
        for domain, path, label in sample_refs(dataset):
            image = resolve_sample_image(path, root)
            key = image.relative_to(root).as_posix()
            if key not in hashes:
                hashes[key] = digest(image)
            result.append((domain, key, label))
        return result
    for cid in sorted(k for k in data if k > 0):
        entry = data[cid]
        dataset = entry['train'].dataset
        base = dataset
        while hasattr(base, 'dataset'):
            base = base.dataset
        if classes is None:
            classes = [str(v) for v in getattr(base, 'classes', getattr(base, 'CLASSES', range(cfg.model.num_classes)))]
        rows = refs(dataset)
        if not rows:
            raise ValueError(f'客户端 {cid} 无训练样本，请减少客户端数量')
        partition[str(cid)] = rows
        histogram = [0] * cfg.model.num_classes
        for _, _, label in rows:
            histogram[label] += 1
        clients.append(dict(id=cid, domain=rows[0][0], samples=len(rows), histogram=histogram))
        for domain, key, label in refs(entry['test'].dataset):
            tests.setdefault(domain, {})[key] = label
    train_keys = {key for rows in partition.values() for _, key, _ in rows}
    if train_keys & {key for rows in tests.values() for key in rows}:
        raise ValueError('训练与测试原图存在交集，拒绝标记非成员')
    fingerprint = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    info = dict(clientCount=len(clients), clients=clients, classes=classes,
        trainSamples=sum(c['samples'] for c in clients), testSamples=sum(len(v) for v in tests.values()),
        domains=[dict(name=d, testSamples=len(rows)) for d, rows in tests.items()],
        partitionFingerprint=fingerprint(partition), testFingerprint=fingerprint(dict(test=tests, images=hashes)),
        featureSpace=digest(weight), augmentation=dict(mode='cloud-preset', provenance='cloud-yaml'),
        presetSha256=spec['provenance']['sourceSha256'])
    if spec.get('expectedPartition', info['partitionFingerprint']) != info['partitionFingerprint'] or spec.get('expectedData', info['testFingerprint']) != info['testFingerprint']:
        raise ValueError('数据、图片或划分在预检后发生变化，请重新预检')
    save(output / 'data_manifest.json', {**info, 'partition': partition, 'test': tests})
    # CN stores validation callbacks alongside values. Strip only the saved
    # copy so runtime checks (including adaptive DP validation) remain active.
    saved_cfg = cfg.clone()
    saved_cfg.de_arguments()
    saved_cfg.clear_aux_info()
    saved = portable_config(yaml.safe_load(saved_cfg.dump()), repo)
    (output / 'resolved_config.yaml').write_text(yaml.safe_dump(saved, allow_unicode=True), encoding='utf-8')
    emit('prepared', **info)
    if spec['action'] == 'inspect':
        save(output / 'result.json', info)
        return
    client_base, server_base = privacy_bases(spec, None)
    class MonitoredServer(server_base):
        def _start_training_round(self):
            emit('round_started', round=int(self.state), participants=sorted(partition and map(int, partition)))
            return super()._start_training_round()

        def _evaluate_on_test_sets(self):
            values = super()._evaluate_on_test_sets()
            domains = {k: float(v) for k, v in values.items() if k != 'average'}
            if domains:
                sizes = {d: len(self.test_labels[d]) for d in domains}
                emit('metrics', round=int(self.state), domains=domains,
                    domainMean=sum(domains.values()) / len(domains),
                    accuracy=sum(domains[d] * sizes[d] for d in domains) / sum(sizes.values()),
                    worstDomain=min(domains.values()), domainGap=max(domains.values())-min(domains.values()), trainLoss=None)
            return values
    update_logger(cfg, clear_before_add=True)
    emit('stage', stage='按云服务器原始配置进行本地训练与特征保存')
    cfg.freeze(inform=False, save=False)
    runner = get_runner(server_class=MonitoredServer, client_class=client_base, config=cfg, data=data)
    runner.run()
    if not runner.server.is_finish:
        raise ValueError('原生训练未完整结束')
    save(output / 'result.json', {**info, 'completedRounds': cfg.federate.total_round_num,
         'elapsedSeconds': time.monotonic() - started, 'history': runner.server.test_accuracies_history})
    finalize(runner.server, spec, info, partition, tests)
