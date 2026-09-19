"""Single-host platform HTTP entrypoint; legacy three-host API remains separate."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import inspect
from http.server import ThreadingHTTPServer
import io
import json
import math
import mimetypes
import os
from pathlib import Path
import re
import signal
import threading
from typing import Any
from urllib.parse import parse_qs, urlparse
import zipfile

from .app import ApiHandler
from .platform_backdoor import BackdoorService
from .platform_config import PlatformError, sha256
from .platform_service import PlatformService
from .platform_privacy import PrivacyService
from .paths import env_path, project_path
from .schemas import ValidationError
from .platform_paths import resolve_path
from .privacy_replay import align_indexed_replay, align_image_index_replay


class PlatformHandler(ApiHandler):
    server_version = 'FederatedScopeSingleHost/1.0'

    def _fedmia_root(self) -> Path:
        resources = env_path('FS_PLATFORM_RESOURCES', 'resources')
        root = env_path('FS_FEDMIA_LOCAL_ROOT', resources / 'fedmia_local')
        # Prefer the current deployment layout; recognize an explicitly
        # selected legacy sibling package only when that layout is absent.
        if not os.environ.get('FS_FEDMIA_LOCAL_ROOT') and not root.exists():
            legacy = env_path('FS_FEDMIA_LOCAL_ROOT', '../fedmia_local')
            if (legacy / 'active_package.json').is_file():
                root = legacy
        selection = root / 'active_package.json'
        if selection.is_file():
            relative = json.loads(selection.read_text(encoding='utf-8'))['path']
            selected = (root / relative).resolve()
            try:
                selected.relative_to(root.resolve())
            except ValueError as error:
                raise PlatformError('成员推理展示包路径非法', 500) from error
            return selected
        return root

    def _fedmia_image_index(self):
        path = self._fedmia_root() / 'image_index.json'
        return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None

    def _fedmia_clients(self) -> list[int]:
        return sorted(set(self._fedmia_available_clients(
            self._fedmia_run_dir('no_defense'))) & set(
                self._fedmia_available_clients(self._fedmia_run_dir('defense'))))

    def _fedmia_membership(self, query) -> dict[str, Any]:
        allowed = {'clientId', 'group', 'limit', 'seed', 'threshold'}
        if set(query) - allowed or any(len(values) != 1 for values in query.values()):
            raise PlatformError('成员推理查询参数非法')
        try:
            client = int(query['clientId'][0]) if 'clientId' in query else None
            group = query.get('group', ['member'])[0]
            limit = int(query.get('limit', ['20'])[0])
            seed = int(query.get('seed', ['2026'])[0])
            threshold = float(query['threshold'][0]) if 'threshold' in query else None
        except (ValueError, TypeError) as error:
            raise PlatformError('成员推理查询参数类型错误') from error
        if (group not in {'member', 'nonmember'} or not 1 <= limit <= 200
                or not 0 <= seed < 2 ** 32 or (client is not None and client < 1)
                or (threshold is not None and
                    (not math.isfinite(threshold) or not 0 <= threshold <= 1))):
            raise PlatformError('成员推理查询参数超出范围')
        root = self._fedmia_root()
        files = ['show_fedmia_examples.py', 'runs/no_defense/config.yaml',
                 'runs/defense/config.yaml']
        index = self._fedmia_image_index()
        dataset = index['datasetRoot'] if index else 'datasets/OfficeHomeDataset_10072016'
        try:
            (root / dataset).resolve().relative_to(root.resolve())
        except ValueError as error:
            raise PlatformError('数据集目录非法', 500) from error
        directories = ['runs/no_defense/ggeur_fedmia_features',
                       'runs/defense/ggeur_fedmia_features', dataset]
        missing = [p for p in files if not (root / p).is_file()]
        missing += [p for p in directories if not (root / p).is_dir()]
        clients = self._fedmia_clients() if not missing else []
        if missing or not clients:
            return {'configured': False, 'source': None, 'items': [],
                    'message': '隐私模块资源未就绪，训练与模型评测不受影响。',
                    'expectedPath': str(root), 'missing': missing,
                    'clients': clients}
        return self._fedmia_client_payload(
            client if client is not None else clients[0], group, limit, seed, threshold)

    def _fedmia_module(self):
        cached = getattr(self.context, 'fedmia_examples_module', None)
        if cached is not None:
            return cached
        script = self._fedmia_root() / 'show_fedmia_examples.py'
        if not script.is_file():
            raise PlatformError(f'找不到本地 FedMIA 脚本：{script}', 500)
        spec = importlib.util.spec_from_file_location(
            'local_show_fedmia_examples', script)
        if spec is None or spec.loader is None:
            raise PlatformError('无法加载本地 FedMIA 脚本', 500)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except ModuleNotFoundError as error:
            missing = getattr(error, 'name', str(error))
            raise PlatformError(
                f'本地 FedMIA 推断缺少 Python 依赖：{missing}。'
                '请在启动后端的 Python 环境安装 torch、torchvision、scipy、yacs、pyyaml、pillow。',
                500) from error
        # A display package can carry its original scoring implementation.
        # Keep these classes private rather than replacing registered attacks.
        for variant in ('i', 'ii'):
            saved = self._fedmia_root() / f'attack_fedmia_{variant}.py'
            if saved.is_file():
                name = f'federatedscope.contrib.attack.plugins.saved_membership_{variant}'
                saved_spec = importlib.util.spec_from_file_location(name, saved)
                saved_module = importlib.util.module_from_spec(saved_spec)
                saved_spec.loader.exec_module(saved_module)
                attribute = 'FedMIAIPlugin' if variant == 'i' else 'FedMIAIIPlugin'
                setattr(module, attribute, getattr(saved_module, attribute))
        self.context.fedmia_examples_module = module
        return module

    def _fedmia_run_dir(self, kind: str) -> Path:
        root = self._fedmia_root()
        run = root / 'runs' / kind
        if not (run / 'config.yaml').is_file():
            raise PlatformError(f'缺少 {kind} 的 config.yaml：{run}', 500)
        if not (run / 'ggeur_fedmia_features').is_dir():
            raise PlatformError(f'缺少 {kind} 的 ggeur_fedmia_features：{run}', 500)
        return run

    def _fedmia_dataset_root(self) -> Path:
        package = self._fedmia_root().resolve()
        index = self._fedmia_image_index()
        relative = index['datasetRoot'] if index else 'datasets/OfficeHomeDataset_10072016'
        root = (package / relative).resolve()
        try:
            root.relative_to(package)
        except ValueError as error:
            raise PlatformError('数据集目录非法', 500) from error
        if not root.is_dir():
            raise PlatformError(f'缺少成员推理数据集目录：{root}', 500)
        return root

    def _fedmia_cfg(self, run_dir: Path, module):
        cfg = module.load_cfg(run_dir, 'fedmia_ii',
                              None if hasattr(module, 'real_image_scores') else 'test')
        cfg.defrost()
        cfg.data.root = str(self._fedmia_dataset_root())
        # FederatedScope CN.freeze saves config.yaml by default; inference must
        # not rewrite the source run (which is mounted read-only in deployment).
        parameters = inspect.signature(cfg.freeze).parameters
        cfg.freeze(**{key: False for key in ('save', 'inform') if key in parameters})
        return cfg

    def _fedmia_available_clients(self, run_dir: Path) -> list[int]:
        feature_dir = run_dir / 'ggeur_fedmia_features'
        clients = set()
        for path in feature_dir.glob('client_*_features_round*.pt'):
            match = re.search(r'client_(\d+)_features_round', path.name)
            if match:
                clients.add(int(match.group(1)))
        return sorted(clients)

    def _fedmia_attack_metrics(self, member_scores, nonmember_scores,
                               module, point=None) -> dict[str, float]:
        pos = module.np.asarray(member_scores, dtype=module.np.float64)
        neg = module.np.asarray(nonmember_scores, dtype=module.np.float64)
        if len(pos) == 0 or len(neg) == 0:
            return {'auc': 0.0, 'tprAt1Fpr': 0.0, 'fprAtThreshold': 0.0}
        scores = module.np.concatenate([pos, neg])
        order = module.np.argsort(scores)
        ranks = module.np.empty(len(scores), dtype=module.np.float64)
        sorted_scores = scores[order]
        start = 0
        while start < len(sorted_scores):
            end = start + 1
            while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
                end += 1
            ranks[order[start:end]] = (start + end + 1) / 2.0
            start = end
        pos_rank_sum = ranks[:len(pos)].sum()
        auc = (pos_rank_sum - len(pos) * (len(pos) + 1) / 2.0) / \
            (len(pos) * len(neg))
        # Use an attainable empirical operating point with FPR <= 1%,
        # including ties. A 99th percentile exceeds 1% on small sample sets.
        if point is None:
            allowed_false_positives = int(math.floor(0.01 * len(neg)))
            cutoff = module.np.sort(neg)[::-1][allowed_false_positives]
            threshold = float(module.np.nextafter(cutoff, module.np.inf))
        else:
            threshold = float(point['threshold'])
        tpr = float(module.np.mean(pos >= threshold))
        fpr = float(module.np.mean(neg >= threshold))
        return {
            'auc': float(auc),
            'tprAt1Fpr': tpr,
            'fprAtThreshold': fpr,
        }

    def _fedmia_score_distribution(self, member_scores, nonmember_scores,
                                   module) -> dict[str, Any]:
        member = module.np.asarray(member_scores, dtype=module.np.float64)
        nonmember = module.np.asarray(nonmember_scores, dtype=module.np.float64)
        bins = module.np.linspace(0.0, 1.0, 41)
        member_counts, _ = module.np.histogram(member, bins=bins)
        nonmember_counts, _ = module.np.histogram(nonmember, bins=bins)
        member_total = max(float(member_counts.sum()), 1.0)
        nonmember_total = max(float(nonmember_counts.sum()), 1.0)
        member_norm = member_counts.astype(module.np.float64) / member_total
        nonmember_norm = nonmember_counts.astype(module.np.float64) / nonmember_total
        return {
            'bins': [float(value) for value in bins],
            'member': [float(value) for value in member_norm],
            'nonmember': [float(value) for value in nonmember_norm],
            'memberMean': float(member.mean()) if len(member) else 0.0,
            'nonmemberMean': float(nonmember.mean()) if len(nonmember) else 0.0,
            'meanGap': float(member.mean() - nonmember.mean())
            if len(member) and len(nonmember) else 0.0,
            'memberSamples': int(len(member)),
            'nonmemberSamples': int(len(nonmember)),
        }

    def _fedmia_bundle(self, client_id: int) -> dict[str, Any]:
        cache = getattr(self.context, 'fedmia_cache', None)
        if cache is None:
            cache = {}
            self.context.fedmia_cache = cache
        lock = getattr(self.context, 'fedmia_cache_lock', None)
        if lock is None:
            lock = threading.RLock()
            self.context.fedmia_cache_lock = lock
        with lock:
            root = self._fedmia_root().resolve()
            watched = [root / name for name in (
                'show_fedmia_examples.py', 'image_index.json',
                'attack_fedmia_i.py', 'attack_fedmia_ii.py',
                'runs/no_defense/config.yaml', 'runs/defense/config.yaml',
                'runs/no_defense/ggeur_fedmia_features',
                'runs/defense/ggeur_fedmia_features')]
            signature = (str(root), tuple(
                (str(path), path.stat().st_mtime_ns, path.stat().st_size)
                if path.exists() else (str(path), None, None)
                for path in watched))
            if cache.get('_signature') != signature:
                had_cached_state = bool(cache)
                cache.clear()
                cache['_signature'] = signature
                if had_cached_state:
                    self.context.fedmia_examples_module = None
            version = hashlib.sha256(repr(signature).encode('utf-8')).hexdigest()[:16]
            if client_id in cache:
                return cache[client_id]
            shared = cache.setdefault('_shared', {})
            module = shared.get('module')
            if module is None:
                module = self._fedmia_module()
                shared['module'] = module
            no_dir = shared.get('no_dir')
            if no_dir is None:
                no_dir = self._fedmia_run_dir('no_defense')
                shared['no_dir'] = no_dir
            defense_dir = shared.get('defense_dir')
            if defense_dir is None:
                defense_dir = self._fedmia_run_dir('defense')
                shared['defense_dir'] = defense_dir
            clients = shared.get('clients')
            if clients is None:
                clients = self._fedmia_clients()
                shared['clients'] = clients
            if client_id not in clients:
                raise PlatformError(f'客户端 {client_id} 没有可用攻击特征', 404)
            no_cfg = shared.get('no_cfg')
            if no_cfg is None:
                no_cfg = self._fedmia_cfg(no_dir, module)
                shared['no_cfg'] = no_cfg
            defense_cfg = shared.get('defense_cfg')
            if defense_cfg is None:
                defense_cfg = self._fedmia_cfg(defense_dir, module)
                shared['defense_cfg'] = defense_cfg
            no_features = shared.get('no_features')
            if no_features is None:
                no_features = module.load_feature_artifacts(
                    no_dir / 'ggeur_fedmia_features', no_cfg,
                    unsafe_load=True)
                shared['no_features'] = no_features
            defense_features = shared.get('defense_features')
            if defense_features is None:
                defense_features = module.load_feature_artifacts(
                    defense_dir / 'ggeur_fedmia_features', defense_cfg,
                    unsafe_load=True)
                shared['defense_features'] = defense_features
            image_items = shared.get('image_items')
            if image_items is None:
                image_items = {}
                shared['image_items'] = image_items
            index = self._fedmia_image_index()
            if client_id not in image_items:
                if index:
                    dataset_root = self._fedmia_dataset_root()
                    entry = index['clients'][str(client_id)]
                    image_items[client_id] = tuple([
                        (str(dataset_root / path), int(label), int(original))
                        for path, label, original in entry[group]]
                        for group in ('member', 'nonmember'))
                    shared['names'] = index['classNames']
                    shared['dataset'] = index['dataset']
                else:
                    image_items[client_id] = module.load_image_items(
                        no_dir, no_cfg, client_id)

            modern = hasattr(module, 'real_image_scores')
            score_fn = module.real_image_scores if modern else module.compute_scores
            no_result = score_fn(no_features, no_cfg, client_id, 'fedmia_ii')
            defense_result = score_fn(defense_features, defense_cfg, client_id, 'fedmia_ii')
            if modern:
                no_mix_cfg = no_cfg.clone()
                no_mix_cfg.defrost()
                no_mix_cfg.attack.mode = 'mix'
                no_mix_cfg.freeze(save=False)
                defense_mix_cfg = defense_cfg.clone()
                defense_mix_cfg.defrost()
                defense_mix_cfg.attack.mode = 'mix'
                defense_mix_cfg.freeze(save=False)
                no_mix = module.compute_scores(no_features, no_mix_cfg, client_id, 'fedmia_ii')
                defense_mix = module.compute_scores(defense_features, defense_mix_cfg, client_id, 'fedmia_ii')
                no_point = module.operating_point(no_mix.scores_member, no_mix.scores_nonmember)
                defense_point = module.operating_point(defense_mix.scores_member, defense_mix.scores_nonmember)
            else:
                no_mix, defense_mix = no_result, defense_result
                no_point = defense_point = None
            member_items, nonmember_items = image_items[client_id]
            no_member_scores = module.np.asarray(
                no_result.scores_member, dtype=module.np.float64)
            no_nonmember_scores = module.np.asarray(
                no_result.scores_nonmember, dtype=module.np.float64)
            defense_member_scores = module.np.asarray(
                defense_result.scores_member, dtype=module.np.float64)
            defense_nonmember_scores = module.np.asarray(
                defense_result.scores_nonmember, dtype=module.np.float64)
            alignment = None
            arrays = (no_member_scores, no_nonmember_scores,
                      defense_member_scores, defense_nonmember_scores)
            if any(scores.ndim != 1 or not len(scores)
                   or not module.np.isfinite(scores).all() for scores in arrays):
                raise PlatformError('成员推理分数与样本数量不一致或含无效值', 409)
            if (len(member_items) != len(no_member_scores)
                    or len(member_items) != len(defense_member_scores)
                    or len(nonmember_items) != len(no_nonmember_scores)
                    or len(nonmember_items) != len(defense_nonmember_scores)):
                # This is not a generic min-length fallback. Require the
                # historical algorithm's explicit metadata, matching replay
                # partitions and labels for every saved round first.
                if not all(getattr(result, 'metadata', None)
                           for result in (no_result, defense_result)):
                    raise PlatformError('成员推理分数与样本数量不一致或含无效值', 409)
                defense_items = module.load_image_items(
                    defense_dir, defense_cfg, client_id)
                if tuple(defense_items) != tuple(image_items[client_id]):
                    raise PlatformError('两组历史实验的数据划分不一致', 409)
                align = align_image_index_replay if modern and index else align_indexed_replay
                common, alignment = align(
                    module,
                    ((no_cfg, no_features[client_id], no_result),
                     (defense_cfg, defense_features[client_id], defense_result)),
                    (member_items, nonmember_items))
                member_items = member_items[:common[0]]
                nonmember_items = nonmember_items[:common[1]]
                no_member_scores = no_member_scores[:common[0]]
                defense_member_scores = defense_member_scores[:common[0]]
                no_nonmember_scores = no_nonmember_scores[:common[1]]
                defense_nonmember_scores = defense_nonmember_scores[:common[1]]
            for items, before, after in (
                    (member_items, no_member_scores, defense_member_scores),
                    (nonmember_items, no_nonmember_scores, defense_nonmember_scores)):
                if (before.ndim != 1 or after.ndim != 1
                        or len(items) != len(before) or len(items) != len(after)
                        or not module.np.isfinite(before).all()
                        or not module.np.isfinite(after).all()):
                    raise PlatformError('成员推理分数与样本数量不一致或含无效值', 409)
            bundle = {
                'module': module,
                'clients': clients,
                'names': shared.get('names', module.class_names_for(no_cfg)),
                'dataset': shared.get('dataset', 'Office-Home'),
                'version': version,
                'modern': modern,
                'thresholds': {
                    'noDefense': no_point['threshold'] if no_point else 0.5,
                    'defense': defense_point['threshold'] if defense_point else 0.5,
                },
                'member_items': member_items,
                'nonmember_items': nonmember_items,
                'no_member_scores': no_member_scores,
                'no_nonmember_scores': no_nonmember_scores,
                'defense_member_scores': defense_member_scores,
                'defense_nonmember_scores': defense_nonmember_scores,
                'metrics': {
                    'noDefense': self._fedmia_attack_metrics(
                        no_mix.scores_member if modern else no_member_scores,
                        no_mix.scores_nonmember if modern else no_nonmember_scores, module, no_point),
                    'defense': self._fedmia_attack_metrics(
                        defense_mix.scores_member if modern else defense_member_scores,
                        defense_mix.scores_nonmember if modern else defense_nonmember_scores, module, defense_point),
                },
                'distributions': {
                    'noDefense': self._fedmia_score_distribution(
                        no_mix.scores_member if modern else no_member_scores,
                        no_mix.scores_nonmember if modern else no_nonmember_scores, module),
                    'defense': self._fedmia_score_distribution(
                        defense_mix.scores_member if modern else defense_member_scores,
                        defense_mix.scores_nonmember if modern else defense_nonmember_scores, module),
                },
                'payloads': {},
                'alignment': alignment,
            }
            cache[client_id] = bundle
            return bundle

    def _fedmia_client_payload(self, client_id: int, group: str,
                               limit: int, seed: int,
                               threshold: float | None) -> dict[str, Any]:
        if group not in {'member', 'nonmember'}:
            raise PlatformError('成员状态只能是 member 或 nonmember')
        bundle = self._fedmia_bundle(client_id)
        payload_key = (group, int(limit), int(seed), threshold)
        cache_lock = getattr(self.context, 'fedmia_cache_lock', None)
        if cache_lock is not None:
            with cache_lock:
                cached = bundle['payloads'].get(payload_key)
                if cached is not None:
                    return cached
        module = bundle['module']
        names = bundle['names']
        no_threshold = bundle['thresholds']['noDefense'] if threshold is None else threshold
        defense_threshold = bundle['thresholds']['defense'] if threshold is None else threshold
        no_scores = {
            'member': bundle['no_member_scores'],
            'nonmember': bundle['no_nonmember_scores'],
        }[group]
        defense_scores = {
            'member': bundle['defense_member_scores'],
            'nonmember': bundle['defense_nonmember_scores'],
        }[group]
        source_items = {
            'member': bundle['member_items'],
            'nonmember': bundle['nonmember_items'],
        }[group]
        truth = 1 if group == 'member' else 0
        usable = min(len(source_items), len(no_scores), len(defense_scores))
        if usable <= 0:
            raise PlatformError('该客户端没有可展示的成员推理样本', 404)
        rng = module.np.random.default_rng(int(seed))
        priority_args = (module.np.arange(usable), truth, module.np.asarray(no_scores),
                         module.np.asarray(defense_scores), no_threshold)
        order = (module.comparison_priority(*priority_args, defense_threshold, rng)
                 if bundle['modern'] else module.comparison_priority(*priority_args, rng))
        items = []
        for rank, index in enumerate(order[:max(1, min(limit, usable))],
                                     start=1):
            path, label, original_index = source_items[int(index)]
            no_pred = 'member' if float(no_scores[index]) >= no_threshold \
                else 'nonmember'
            defense_pred = 'member' if float(defense_scores[index]) >= defense_threshold \
                else 'nonmember'
            items.append({
                'id': f'{client_id}-{group}-{int(index)}',
                'rank': rank,
                'clientId': client_id,
                'group': group,
                'truth': group,
                'domain': Path(path).parts[-3] if len(Path(path).parts) >= 3 else '',
                'className': module.label_name(int(label), names),
                'filename': Path(path).name,
                'sampleIndex': int(index),
                'originalDatasetIndex': int(original_index),
                'imageUrl': (
                    f'/api/platform/privacy/membership/images/'
                    f'{client_id}/{group}/{int(index)}?v={bundle["version"]}'
                ),
                'noDefenseScore': float(no_scores[index]),
                'defenseScore': float(defense_scores[index]),
                'noDefensePrediction': no_pred,
                'defensePrediction': defense_pred,
            })
        payload = {
            'configured': True,
            'clientId': client_id,
            'group': group,
            'clients': bundle['clients'],
            'threshold': threshold,
            'thresholds': {'noDefense': no_threshold, 'defense': defense_threshold},
            'dataset': bundle['dataset'],
            'calibrationMode': 'mix' if bundle['modern'] else 'test',
            'metrics': bundle['metrics'],
            'distributions': bundle['distributions'],
            'items': items,
            'source': str(self._fedmia_root()),
            'alignment': bundle.get('alignment'),
        }
        if cache_lock is not None:
            with cache_lock:
                bundle['payloads'][payload_key] = payload
        return payload

    def _fedmia_image(self, client_id: int, group: str, index: int) -> None:
        if group not in {'member', 'nonmember'}:
            raise PlatformError('成员状态只能是 member 或 nonmember')
        bundle = self._fedmia_bundle(client_id)
        source_items = (bundle['member_items'] if group == 'member'
                        else bundle['nonmember_items'])
        if index < 0 or index >= len(source_items):
            raise PlatformError('样本图片索引不存在', 404)
        image = Path(source_items[index][0]).resolve()
        dataset_root = self._fedmia_dataset_root()
        try:
            image.relative_to(dataset_root)
        except ValueError as error:
            raise PlatformError('样本图片路径非法', 403) from error
        if not image.is_file():
            raise PlatformError('样本图片不存在', 404)
        self._download(image, image.name,
                       mimetypes.guess_type(image.name)[0] or
                       'application/octet-stream', inline=True)

    def _body(self):
        try:
            length = int(self.headers.get('Content-Length', '0'))
        except ValueError as error:
            raise PlatformError('Content-Length 非法', 400) from error
        if not 0 <= length <= 64 * 1024:
            raise PlatformError('配置请求必须在 64 KiB 以内', 413)
        return super()._body()

    def _cors_headers(self, cache=False):
        # No wildcard cross-origin write access to a training control plane.
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header(
            'Cache-Control',
            'public, max-age=3600' if cache else 'no-store')

    def _dispatch(self, write=False):
        try:
            if write:
                origin = self.headers.get('Origin')
                if origin and urlparse(origin).netloc != self.headers.get('Host'):
                    raise PlatformError('不允许跨站启动或停止任务', 403)
                upload_path = re.fullmatch(r'/api/platform/datasets/[a-f0-9]{32}/files',
                                          urlparse(self.path).path.rstrip('/'))
                expected_type = 'application/octet-stream' if upload_path else 'application/json'
                if self.headers.get_content_type() != expected_type:
                    raise PlatformError(f'仅接受 {expected_type}', 415)
            self._route(write)
        except (PlatformError, ValidationError) as error:
            self._error(getattr(error, 'status', 422), 'PLATFORM_ERROR', str(error))
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as error:
            self._error(500, 'INTERNAL_ERROR', str(error))

    def do_GET(self):
        self._dispatch()

    def do_POST(self):
        self._dispatch(True)

    def _download(self, payload, name, content_type, inline=False):
        self.send_response(200)
        self._cors_headers(cache=inline)
        self.send_header('Content-Type', content_type)
        disposition = 'inline' if inline else 'attachment'
        self.send_header('Content-Disposition', f'{disposition}; filename="{name}"')
        size = payload.stat().st_size if isinstance(payload, Path) else len(payload)
        self.send_header('Content-Length', str(size))
        self.end_headers()
        try:
            if isinstance(payload, Path):
                with payload.open('rb') as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                        self.wfile.write(chunk)
            else:
                self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError, ValueError):
            return

    def _privacy_membership_root(self) -> Path:
        configured = os.environ.get('FS_PLATFORM_MEMBERSHIP_DIR')
        if configured:
            return project_path(configured)
        return (self.context.platform.state /
                'privacy_membership').resolve()

    def _privacy_membership_payload(self) -> dict[str, Any]:
        root = self._privacy_membership_root()
        manifest = root / 'membership_examples.json'
        if not manifest.is_file():
            return {
                'configured': False,
                'source': None,
                'items': [],
                'message': '尚未放置成员推理展示包',
                'expectedPath': str(manifest),
            }
        try:
            payload = json.loads(manifest.read_text(encoding='utf-8'))
        except json.JSONDecodeError as error:
            raise PlatformError(f'成员推理展示包 JSON 解析失败：{error}', 500)
        items = payload.get('items', [])
        if not isinstance(items, list):
            raise PlatformError('membership_examples.json 的 items 必须是数组', 500)
        public_items = []
        for index, raw in enumerate(items):
            if not isinstance(raw, dict):
                continue
            identifier = str(raw.get('id') or f'sample-{index + 1:03d}')
            truth = raw.get('truth')
            if truth not in {'member', 'nonmember'}:
                continue
            no_defense = raw.get('noDefensePrediction',
                                 raw.get('no_defense_prediction'))
            defense = raw.get('defensePrediction',
                              raw.get('defense_prediction'))
            if no_defense not in {'member', 'nonmember'}:
                no_defense = 'member' if float(raw.get(
                    'noDefenseScore', raw.get('no_defense_score', 0))) >= 0.5 \
                    else 'nonmember'
            if defense not in {'member', 'nonmember'}:
                defense = 'member' if float(raw.get(
                    'defenseScore', raw.get('defense_score', 0))) >= 0.5 \
                    else 'nonmember'
            public_items.append({
                'id': identifier,
                'truth': truth,
                'domain': raw.get('domain') or '',
                'className': raw.get('className') or raw.get('class_name') or '',
                'filename': raw.get('filename') or '',
                'imageUrl': f'/api/platform/privacy/membership/images/{identifier}',
                'noDefensePrediction': no_defense,
                'defensePrediction': defense,
            })
        return {
            'configured': True,
            'source': payload.get('source'),
            'items': public_items,
            'message': payload.get('message'),
            'expectedPath': str(manifest),
        }

    def _privacy_membership_image(self, identifier: str) -> None:
        root = self._privacy_membership_root()
        manifest = root / 'membership_examples.json'
        if not manifest.is_file():
            raise PlatformError('尚未放置成员推理展示包', 404)
        payload = json.loads(manifest.read_text(encoding='utf-8'))
        items = payload.get('items', [])
        target = None
        for index, raw in enumerate(items if isinstance(items, list) else []):
            if not isinstance(raw, dict):
                continue
            item_id = str(raw.get('id') or f'sample-{index + 1:03d}')
            if item_id == identifier:
                target = raw.get('image') or raw.get('imagePath') or \
                    raw.get('image_path')
                break
        if not target:
            raise PlatformError('成员推理样本图片不存在', 404)
        image = (root / str(target)).resolve()
        try:
            image.relative_to(root)
        except ValueError as error:
            raise PlatformError('成员推理图片路径非法', 403) from error
        if not image.is_file():
            raise PlatformError('成员推理样本图片不存在', 404)
        self._download(image, image.name,
                       mimetypes.guess_type(image.name)[0] or
                       'application/octet-stream', inline=True)

    def _backdoor(self, write, path, backdoor):
        """后门研究: 测试集浏览、随机挑图、clean/triggered/defense 三连图生成。"""
        if not write:
            if path == '/api/platform/backdoor/testset':
                self._data(backdoor.testset())
                return
            image = re.fullmatch(r'/api/platform/backdoor/testset/([A-Za-z][A-Za-z0-9_]*_\d{5})/image', path)
            if image:
                file = backdoor.image_path(image.group(1))
                self._download(file, file.name, mimetypes.guess_type(file.name)[0] or 'image/jpeg', inline=True)
                return
            if path == '/api/platform/backdoor/jobs':
                self._data(backdoor.list())
                return
            job = re.fullmatch(r'/api/platform/backdoor/jobs/([a-f0-9]{32})(?:/(image/clean|image/triggered|image/defense|image/defenseClean|logs))?', path)
            if job:
                job_id, action = job.groups()
                if action is None:
                    self._data(backdoor.get(job_id))
                elif action == 'logs':
                    self._data(backdoor.logs(job_id))
                else:
                    name = action.split('/')[1]
                    file = backdoor.artifact(job_id, name)
                    self._download(file, f'{job_id}-{name}{file.suffix.lower()}',
                                   mimetypes.guess_type(file.name)[0] or 'image/png', inline=True)
                return
        else:
            if path == '/api/platform/backdoor/pick':
                self._data(backdoor.pick(self._body()))
                return
            if path == '/api/platform/backdoor/jobs':
                self._data(backdoor.create(self._body()), 202)
                return
            stop = re.fullmatch(r'/api/platform/backdoor/jobs/([a-f0-9]{32})/stop', path)
            if stop:
                self._body()
                self._data(backdoor.stop(stop.group(1)))
                return
        raise PlatformError('接口不存在', 404)

    def _route(self, write):
        path = urlparse(self.path).path.rstrip('/') or '/'
        service = self.context.platform
        if path == '/api/platform/datasets/import' and write:
            from .dataset_import import DatasetImporter
            self._data(DatasetImporter(service.repo).create(self._body()))
            return
        if path == '/api/platform/datasets':
            from .uploaded_datasets import DatasetStore
            store = DatasetStore(service.repo)
            self._data(store.create(self._body()) if write else store.list())
            return
        uploaded = re.fullmatch(r'/api/platform/datasets/([a-f0-9]{32})/(files|finish|import-next|images/(\d+))', path)
        if uploaded:
            from .uploaded_datasets import DatasetStore, MAX_IMAGE
            store = DatasetStore(service.repo)
            identifier, action, image_index = uploaded.groups()
            if write and action == 'import-next':
                from .dataset_import import DatasetImporter
                self._body()
                self._data(DatasetImporter(service.repo).next(identifier))
                return
            if write and action == 'files':
                query = parse_qs(urlparse(self.path).query, keep_blank_values=True)
                if set(query) != {'path'} or len(query['path']) != 1:
                    raise PlatformError('需要图片相对路径')
                try:
                    length = int(self.headers.get('Content-Length', '0'))
                except ValueError as error:
                    raise PlatformError('Content-Length 非法', 400) from error
                if not 0 < length <= MAX_IMAGE:
                    raise PlatformError('每张图片不得超过 25 MiB', 413)
                content = self.rfile.read(length)
                if len(content) != length:
                    raise PlatformError('图片上传不完整')
                self._data(store.put(identifier, query['path'][0], content))
                return
            if write and action == 'finish':
                self._body()
                self._data(store.finish(identifier))
                return
            if not write and image_index is not None:
                file = store.image(identifier, int(image_index))
                self._download(file, file.name, mimetypes.guess_type(file.name)[0] or 'image/jpeg', inline=True)
                return
            raise PlatformError('上传接口不存在', 404)
        if not write:
            endpoints = {'/api/health': lambda: {'status': 'ok', 'mode': 'single-host', 'commit': service.commit},
                         '/api/platform/catalog': service.catalog,
                         '/api/platform/resources': service.resources,
                         '/api/platform/jobs': service.list,
                         '/api/platform/library': service.library}
            if path in endpoints:
                self._data(endpoints[path]())
                return
        samples = re.fullmatch(r'/api/platform/testsets/([a-f0-9]{32})/samples(?:/([a-f0-9]{24})/image)?', path)
        if not write and samples:
            testset_id, identifier = samples.groups()
            if identifier:
                job, _, sample = service.samples.resolve(testset_id, identifier)
                file = service.samples.image_path(job, sample)
                self._download(file, identifier + file.suffix.lower(), mimetypes.guess_type(file.name)[0] or 'application/octet-stream', inline=True)
            else:
                self._data(service.samples.page(testset_id, parse_qs(urlparse(self.path).query, keep_blank_values=True)))
            return
        membership_image = re.fullmatch(
            r'/api/platform/privacy/membership/images/(\d+)/(member|nonmember)/(\d+)',
            path)
        if not write and membership_image:
            client_id, group, index = membership_image.groups()
            self._fedmia_image(int(client_id), group, int(index))
            return
        membership = re.fullmatch(r'/api/platform/privacy/membership', path)
        if not write and membership:
            query = parse_qs(urlparse(self.path).query, keep_blank_values=True)
            self._data(self._fedmia_membership(query))
            return
        if path.startswith('/api/platform/privacy/experiments'):
            privacy = self.context.privacy
            suffix = path[len('/api/platform/privacy/experiments'):]
            if not write and suffix in {'/catalog', '/jobs'}:
                self._data(privacy.catalog() if suffix == '/catalog' else privacy.list())
                return
            if write and suffix in {'/preflight', '/train'}:
                self._data(privacy.create('inspect' if suffix == '/preflight' else 'train', self._body()), 202)
                return
            match = re.fullmatch(r'/jobs/([a-f0-9]{32})(?:/(stop|logs|results|features|images/(\d+)/(member|nonmember)/(\d+)))?', suffix)
            if match:
                job_id, action, client, group, index = match.groups()
                if write and action == 'stop':
                    self._body()
                    self._data(privacy.stop(job_id))
                    return
                if not write:
                    if action is None:
                        self._data(privacy.get(job_id))
                    elif action == 'logs':
                        self._data(privacy.logs(job_id))
                    elif action == 'results':
                        query = parse_qs(urlparse(self.path).query)
                        value = query.get('clientId', ['1'])
                        if set(query) - {'clientId'} or len(value) != 1 or not value[0].isdecimal():
                            raise PlatformError('客户端参数非法')
                        self._data(privacy.results(job_id, int(value[0])))
                    elif action == 'features':
                        self._download(privacy.feature_bundle(job_id), job_id + '-fedmia.zip', 'application/zip')
                    elif action.startswith('images/'):
                        file = privacy.image(job_id, int(client), group, int(index))
                        self._download(file, file.name, mimetypes.guess_type(file.name)[0] or 'image/jpeg', inline=True)
                    else:
                        raise PlatformError('接口不存在', 404)
                    return
            raise PlatformError('隐私实验接口不存在', 404)
        if path.startswith('/api/platform/backdoor'):
            self._backdoor(write, path, self.context.backdoor)
            return
        if write and path in {'/api/platform/preflight', '/api/platform/train', '/api/platform/evaluate', '/api/platform/predict', '/api/platform/test-upload'}:
            action = {'preflight': 'inspect', 'train': 'train', 'evaluate': 'evaluate', 'predict': 'predict', 'test-upload': 'test-upload'}[path.split('/')[-1]]
            self._data(service.create(action, self._body()), 202)
            return
        match = re.fullmatch(r'/api/platform/jobs/([a-f0-9]{32})(?:/(stop|logs|export|csv|bundle|model-final|model-best))?', path)
        if match:
            job_id, action = match.groups()
            job = service.get(job_id)
            if write and action == 'stop':
                self._body()
                self._data(service.stop(job_id))
                return
            if not write:
                if action is None:
                    self._data(job)
                elif action == 'logs':
                    self._data(service.logs(job_id))
                elif action == 'export':
                    self._download(json.dumps(job, ensure_ascii=False, indent=2).encode(), job_id + '.json', 'application/json')
                elif action == 'csv':
                    stream = io.StringIO()
                    writer = csv.writer(stream)
                    writer.writerow(['domain', 'class', 'support', 'precision', 'recall', 'f1'])
                    result = job.get('result', {})
                    for domain, metric in {'overall': result, **result.get('domains', {})}.items() if job['action'] == 'evaluate' else []:
                        for row in metric.get('perClass', []):
                            writer.writerow([domain, row['classIndex'], row['support'], row['precision'], row['recall'], row['f1']])
                    self._download(stream.getvalue().encode('utf-8-sig'), job_id + '.csv', 'text/csv')
                elif action in {'model-final', 'model-best'}:
                    file = service.directory(job_id) / ('checkpoints/mlp_' + action.split('-')[1] + '.pt')
                    if job['status'] != 'completed' or not file.is_file():
                        raise PlatformError('模型未完成或不存在', 404)
                    if sha256(file) != job['result']['artifactHashes'][file.name]:
                        raise PlatformError('模型文件已改变，拒绝以原版本导出', 409)
                    self._download(file, job_id + '-' + file.name, 'application/octet-stream')
                elif action == 'bundle':
                    if job['status'] not in {'completed', 'failed', 'stopped', 'interrupted'}:
                        raise PlatformError('结束后才能导出完整复现包', 409)
                    output = service.directory(job_id)
                    target = output / 'reproducibility.zip'
                    # Rebuild to include the current terminal state. Never package resource roots.
                    with service.lock:
                        with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
                            for file in output.rglob('*'):
                                if file.is_file() and not file.is_symlink() and file != target:
                                    archive.write(file, file.relative_to(output))
                    self._download(target, job_id + '.zip', 'application/zip')
                else:
                    raise PlatformError('接口不存在', 404)
                return
        if not write and not path.startswith('/api') and self._serve_frontend(path):
            return
        raise PlatformError('接口不存在', 404)


def create_server(host='127.0.0.1', port=8001, state=None):
    repo = Path(__file__).resolve().parents[2]
    root = resolve_path(repo, state or 'exp/platform')
    service = PlatformService(repo, root)
    backdoor = BackdoorService(repo, root)
    privacy = PrivacyService(repo, root)
    context = type('PlatformContext', (), {'platform': service, 'backdoor': backdoor, 'privacy': privacy,
        'fedmia_cache': {}, 'fedmia_cache_lock': threading.RLock(),
        'frontend_dist': resolve_path(repo, os.environ.get('FEDERATEDSCOPE_FRONTEND_DIST', '../frontend/dist'))})()
    handler = type('BoundPlatformHandler', (PlatformHandler,), {'context': context})
    try:
        server = ThreadingHTTPServer((host, port), handler)
    except Exception:
        service.close()
        backdoor.close()
        privacy.close()
        raise
    return server


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8001)
    parser.add_argument('--state-dir', type=Path)
    args = parser.parse_args()
    server = create_server(args.host, args.port, args.state_dir)
    def shutdown(*_):
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    print(f'Single-host platform http://{args.host}:{args.port}', flush=True)
    try:
        server.serve_forever()
    finally:
        server.RequestHandlerClass.context.backdoor.close()
        server.RequestHandlerClass.context.privacy.close()
        server.RequestHandlerClass.context.platform.close()
        server.server_close()


if __name__ == '__main__':
    main()
