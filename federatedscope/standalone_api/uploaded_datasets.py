"""Immutable, shared image uploads. No caller supplied absolute filesystem paths."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import threading
import uuid

from .paths import env_path
from .platform_config import PlatformError
from .repository import JsonRepository
from .uploaded_images import IMAGE_TYPES, MAX_IMAGE, normalize_image

UPLOAD_LOCK = threading.RLock()
MAX_BYTES = 20 * 1024 ** 3
SPLIT_NAMES = {'train', 'test', 'val', 'valid', 'validation'}


class DatasetStore:
    def __init__(self, repo):
        self.repo = Path(repo).resolve()
        self.root = env_path('FS_PLATFORM_UPLOADS', self.repo / 'resources' / 'uploaded_datasets', self.repo)

    def directory(self, identifier):
        if not isinstance(identifier, str) or not re.fullmatch(r'[a-f0-9]{32}', identifier):
            raise PlatformError('数据集编号非法')
        target = (self.root / identifier).resolve()
        if target.parent != self.root.resolve():
            raise PlatformError('数据集目录非法', 403)
        return target

    def get(self, identifier, ready=True):
        file = self.directory(identifier) / 'dataset.json'
        if not file.is_file():
            raise PlatformError('数据集不存在', 404)
        value = json.loads(file.read_text(encoding='utf-8'))
        if ready and value['status'] != 'ready':
            raise PlatformError('请完成数据集上传后再使用', 409)
        return value

    def list(self):
        if not self.root.is_dir():
            return []
        return [self.get(p.name) for p in sorted(self.root.iterdir())
                if p.is_dir() and re.fullmatch(r'[a-f0-9]{32}', p.name)
                and (p / 'dataset.json').is_file()
                and self.get(p.name, False)['status'] == 'ready']

    def create(self, body):
        if not isinstance(body, dict) or set(body) - {'name', 'kind', 'layout'}:
            raise PlatformError('上传参数非法')
        name, kind = body.get('name'), body.get('kind')
        if not isinstance(name, str) or not 1 <= len(name.strip()) <= 120 or kind not in ('train', 'test'):
            raise PlatformError('请输入数据集名称并选择训练集或测试集')
        layout = body.get('layout', 'classes')
        if layout not in ('classes', 'split') or (layout == 'split' and kind != 'train'):
            raise PlatformError('数据目录格式非法')
        identifier = uuid.uuid4().hex
        directory = self.directory(identifier)
        directory.mkdir(parents=True)
        value = dict(id=identifier, name=name.strip(), kind=kind, layout=layout, status='uploading',
                     classes=[], count=0, bytes=0, items=[])
        JsonRepository._atomic_write(directory / 'dataset.json', value)
        return value

    def put(self, identifier, relative, content):
        # Reject Windows drive names, ADS, reserved filenames and escaping paths.
        parts = str(relative).split('/')
        if (not parts or len(parts) > 3 or any(not p or p in ('.', '..') or
            re.search(r'[\\:\x00-\x1f<>"|?*]', p) or p.endswith((' ', '.')) or
            re.fullmatch(r'(?i)(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?', p) for p in parts)):
            raise PlatformError('图片路径必须为 类别/图片 或测试图片名')
        normalized, image_metadata = normalize_image(content, PurePosixPath(relative).suffix.lower())
        with UPLOAD_LOCK:
            value = self.get(identifier, False)
            if value['status'] != 'uploading':
                raise PlatformError('已登记数据集不可修改；请上传为新版本', 409)
            split_layout = value.get('layout') == 'split'
            if split_layout and (len(parts) != 3 or parts[0].lower() not in SPLIT_NAMES):
                raise PlatformError('完整数据集应为 train/类别/图片 和 test（或 val）/类别/图片')
            if not split_layout and (len(parts) > 2 or value['kind'] == 'train' and len(parts) != 2):
                raise PlatformError('训练集需 类别/图片；测试集支持图片文件夹或单张图片')
            previous = next((r for r in value['items'] if r.get('originalPath', r['path']).casefold() == relative.casefold()), None)
            original_checksum = hashlib.sha256(content).hexdigest()
            checksum = hashlib.sha256(normalized).hexdigest()
            if previous:
                if previous.get('originalSha256', previous['sha256']) == original_checksum and previous.get('originalPath', previous['path']) == relative:
                    return dict(count=value['count'], bytes=value['bytes'])
                raise PlatformError('存在重名图片（包括大小写冲突）', 409)
            stored_bytes = value.get('storedBytes', value['bytes']) + len(content) + len(normalized)
            if value['count'] >= 50000 or stored_bytes > MAX_BYTES:
                raise PlatformError('单次上传最多 50000 张、20 GiB')
            if any(r['sha256'] == checksum or r.get('originalSha256', r['sha256']) == original_checksum for r in value['items']):
                raise PlatformError('存在内容相同的重复图片，请去重后上传，避免训练/测试泄漏')
            root = self.directory(identifier) / 'images'
            # Appending rather than replacing the suffix keeps a.jpg/a.png distinct.
            normalized_relative = relative + '.png'
            file = root / ('' if split_layout else 'uploaded' if value['kind'] == 'train' else 'test') / normalized_relative
            file.resolve().relative_to(root.resolve())
            file.parent.mkdir(parents=True, exist_ok=True)
            original = self.directory(identifier) / 'originals' / relative
            original.resolve().relative_to((self.directory(identifier) / 'originals').resolve())
            original.parent.mkdir(parents=True, exist_ok=True)
            created = []
            try:
                for target, data in ((file, normalized), (original, content)):
                    with target.open('xb') as stream:
                        created.append(target)
                        stream.write(data)
                value['items'].append(dict(path=normalized_relative, sha256=checksum, bytes=len(content),
                    originalPath=relative, originalSha256=original_checksum, normalizedBytes=len(normalized), **image_metadata))
                value['count'] += 1
                value['bytes'] += len(content)
                value['storedBytes'] = stored_bytes
                JsonRepository._atomic_write(self.directory(identifier) / 'dataset.json', value)
            except OSError:
                # Remove only new files from this failed write, never old uploads.
                for target in reversed(created):
                    target.unlink(missing_ok=True)
                raise PlatformError('保存图片失败，请检查磁盘空间后重试')
            return dict(count=value['count'], bytes=value['bytes'])

    def finish(self, identifier):
        with UPLOAD_LOCK:
            value = self.get(identifier, False)
            if value['status'] == 'ready':
                return value
            if not value['count']:
                raise PlatformError('请先上传图片')
            value['items'].sort(key=lambda item: item['path'])
            split_layout = value.get('layout') == 'split'
            if split_layout:
                buckets = {r['path'].split('/')[0].lower() for r in value['items']}
                validation = buckets & {'val', 'valid', 'validation'}
                if 'train' not in buckets or not (buckets & {'test', 'val', 'valid', 'validation'}):
                    raise PlatformError('完整数据集必须含 train 和 test（或 val/valid/validation）')
                if len(validation) > 1:
                    raise PlatformError('存在多个验证目录，请仅保留 val、valid、validation 中的一个')
                fallback = next(iter(validation), None)
                for row in value['items']:
                    parts = row['path'].split('/')
                    split = parts[0].lower()
                    row['split'] = ('train' if split == 'train' else 'test' if split == 'test' or ('test' not in buckets and split == fallback) else 'val')
                    row['className'] = parts[1]
                value['testSource'] = 'test' if 'test' in buckets else fallback
                value['trainCount'] = sum(r['split'] == 'train' for r in value['items'])
                value['testCount'] = sum(r['split'] == 'test' for r in value['items'])
                value['validationCount'] = sum(r['split'] == 'val' for r in value['items'])
            if value['kind'] == 'test':
                labelled = ['/' in r['path'] for r in value['items']]
                if any(labelled) and not all(labelled):
                    raise PlatformError('不能混合带类别目录的图片和无标签图片')
                value['classes'] = sorted({r['path'].split('/')[0] for r in value['items']}) if all(labelled) else []
            if value['kind'] == 'train':
                training_rows = [r for r in value['items'] if not split_layout or r['split'] == 'train']
                category = lambda row: row['className'] if split_layout else row['path'].split('/')[0]
                classes = sorted({category(r) for r in training_rows})
                if len(classes) < 2 or len({c.casefold() for c in classes}) != len(classes):
                    raise PlatformError('训练集至少两个类别，类别名不能仅大小写不同')
                if any(sum(category(r) == c for r in training_rows) < 3 for c in classes):
                    raise PlatformError('每个类别至少 3 张图片，以固定留出非训练样本')
                if split_layout and any(r['className'] not in classes for r in value['items']):
                    raise PlatformError('测试/验证类别必须存在于训练集，文件夹名称应完全一致')
                value['classes'] = classes
                value['group'] = 'uploaded_' + identifier
            value['fingerprint'] = hashlib.sha256(json.dumps(value['items'], sort_keys=True).encode()).hexdigest()
            value['status'] = 'ready'
            if value['kind'] == 'train':
                records = [dict(path=r['path'], label=value['classes'].index(r['className']), split=r['split'])
                           if split_layout else dict(path='uploaded/' + r['path'], label=value['classes'].index(r['path'].split('/')[0]))
                           for r in value['items']]
                JsonRepository._atomic_write(self.directory(identifier) / 'manifest.json',
                    dict(classes=value['classes'], domains=['uploaded'], records={'uploaded': records}))
            JsonRepository._atomic_write(self.directory(identifier) / 'dataset.json', value)
            return value

    def image(self, identifier, index):
        value = self.get(identifier)
        if not 0 <= index < len(value['items']):
            raise PlatformError('图片不存在', 404)
        root = self.directory(identifier) / 'images'
        result = (root / ('' if value.get('layout') == 'split' else 'uploaded' if value['kind'] == 'train' else 'test') / value['items'][index]['path']).resolve()
        result.relative_to(root.resolve())
        return result

    def training(self, group):
        if not isinstance(group, str) or not re.fullmatch(r'uploaded_[a-f0-9]{32}', group):
            raise PlatformError('上传数据集编号非法')
        value = self.get(group[9:])
        if value['kind'] != 'train':
            raise PlatformError('请选择带类别的训练数据集')
        return value

    def configure(self, raw, group):
        value = self.training(group)
        raw['data'].update(type='domainnet', root=os.path.relpath(self.directory(value['id']) / 'images', self.repo), splits=[0.7, 0., 0.3])
        raw['model']['num_classes'] = len(value['classes'])
        raw['ggeur'].update(domainnet_domains=['uploaded'], domainnet_manifest_path=os.path.relpath(self.directory(value['id']) / 'manifest.json', self.repo), domainnet_shared_classes_only=False,
            use_feature_cache=True, require_complete_feature_cache=True,
            domain_stratified_sampling=False, domain_sampling_group_num=1,
            feature_cache_dir=os.path.relpath(self.directory(value['id']) / 'features', self.repo),
            cnn_checkpoint_path=os.path.relpath(self.repo / 'resources/torch/hub/checkpoints/convnext_base-6075fbad.pth', self.repo),
            cnn_backbone='convnext_base', cnn_pretrained=False)
        return value
