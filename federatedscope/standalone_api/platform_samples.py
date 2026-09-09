"""Read-only sample browsing bound to a completed training run's frozen test set."""
import hashlib
import json
from pathlib import Path, PurePosixPath

from .platform_config import FAMILIES, PlatformError, sha256


def sample_id(domain, key):
    return hashlib.sha256((domain + '\0' + key).encode()).hexdigest()[:24]


class SampleCatalog:
    def __init__(self, service):
        self.service = service

    def manifest(self, testset_id):
        job = self.service.get(testset_id)
        if job['action'] != 'train' or job['status'] != 'completed':
            raise PlatformError('测试集未完成或未登记', 404)
        if job['request']['group'].split('_')[0] not in {'officehome', 'digit3', 'domainnet'}:
            raise PlatformError('单图体验台当前仅支持图像测试集；文本模型可使用独立评测')
        path = self.service.directory(testset_id) / 'data_manifest.json'
        data = json.loads(path.read_text(encoding='utf-8'))
        if data['classes'] != job['result']['classes']:
            raise PlatformError('测试类别名称已改变，禁止错位预测', 409)
        name_sources = [data['classes']]
        # Earlier cache-only-v1 runs hash an absent loader class list as null,
        # then persist numeric display names. Keep that exact legacy contract;
        # never accept altered labels, sample ordering, or renamed classes.
        if data['classes'] == [str(i) for i in range(len(data['classes']))]:
            name_sources.append(None)
        fingerprints = [hashlib.sha256(json.dumps({
            'files': data['cacheFiles'], 'test': data['test'], 'classes': names,
            'group': job['request']['group'],
        }, sort_keys=True).encode()).hexdigest() for names in name_sources]
        if job['result']['testFingerprint'] not in fingerprints:
            raise PlatformError('测试样本清单已改变，请重新登记，禁止错位推理', 409)
        return job, data

    @staticmethod
    def rows(data):
        for domain, rows in data['test'].items():
            for index, (key, label) in enumerate(rows.items()):
                yield dict(id=sample_id(domain, key), domain=domain, index=index,
                           key=key, label=label, className=data['classes'][label],
                           filename=PurePosixPath(key).name)

    def resolve(self, testset_id, identifier):
        job, data = self.manifest(testset_id)
        sample = next((row for row in self.rows(data) if row['id'] == identifier), None)
        if sample is None:
            raise PlatformError('样本不属于此测试集', 404)
        return job, data, sample

    def image_path(self, job, sample):
        # No caller-supplied filesystem paths. Resolve case-folded legacy keys
        # segment by segment and reject ambiguity, traversal and escaping links.
        family = job['request']['group'].split('_')[0]
        root = (self.service.configs.datasets / FAMILIES[family][1]).resolve()
        key = PurePosixPath(sample['key'])
        if key.is_absolute() or not key.parts or any(p in {'.', '..'} or '\\' in p or ':' in p for p in key.parts):
            raise PlatformError('样本路径非法', 403)
        if key.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}:
            raise PlatformError('不支持此样本的预览类型', 415)
        current = root
        try:
            for part in key.parts:
                matches = [p for p in current.iterdir() if p.name.casefold() == part.casefold()]
                if len(matches) != 1:
                    raise PlatformError('原图不存在或路径大小写不唯一', 404)
                current = matches[0].resolve()
                current.relative_to(root)
            if not current.is_file() or current.stat().st_size > 25 * 1024 * 1024:
                raise PlatformError('原图不可读取或超过 25 MiB', 422)
        except ValueError as error:
            raise PlatformError('原图链接越出登记数据目录', 403) from error
        except OSError as error:
            raise PlatformError('原图不可读取；不会下载或生成图片', 404) from error
        return current

    def page(self, testset_id, params):
        if set(params) - {'domain', 'class', 'offset', 'limit'} or any(len(v) != 1 for v in params.values()):
            raise PlatformError('样本查询参数非法')
        def number(name, default, maximum):
            value = params.get(name, [str(default)])[0]
            if not value.isdecimal() or not 0 <= int(value) <= maximum:
                raise PlatformError(f'{name} 参数非法')
            return int(value)
        offset, limit = number('offset', 0, 10000000), number('limit', 12, 24)
        if not limit:
            raise PlatformError('limit 必须大于 0')
        job, data = self.manifest(testset_id)
        domain = params.get('domain', [None])[0]
        label = number('class', 0, len(data['classes']) - 1) if 'class' in params else None
        if domain is not None and domain not in data['test']:
            raise PlatformError('测试域不存在')
        rows = [r for r in self.rows(data) if (domain is None or r['domain'] == domain)
                and (label is None or r['label'] == label)]
        items = []
        for row in rows[offset:offset + limit]:
            item = {k: v for k, v in row.items() if k != 'key'}
            try:
                path = self.image_path(job, row)
                item.update(imageAvailable=True, imageSha256=sha256(path))
            except PlatformError as error:
                item.update(imageAvailable=False, imageError=str(error))
            item['imageUrl'] = f'/api/platform/testsets/{testset_id}/samples/{row["id"]}/image'
            items.append(item)
        return dict(items=items, total=len(rows), offset=offset, limit=limit,
                    testFingerprint=job['result']['testFingerprint'],
                    provenance=data.get('testProvenance', {}))
