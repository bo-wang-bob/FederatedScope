"""Read resources and run real preflights in disposable state, never train."""
import argparse
from html.parser import HTMLParser
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from urllib.parse import unquote, urlsplit
import uuid

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


class AssetParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.assets = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in {'script', 'img', 'source'} and attrs.get('src'):
            self.assets.append(attrs['src'])
        if tag == 'link' and attrs.get('rel') in {'stylesheet', 'modulepreload', 'icon'}:
            self.assets.append(attrs.get('href', ''))


def check_frontend(directory):
    directory = Path(directory).resolve()
    checked = {}

    def local_file(value):
        if not isinstance(value, str) or '\\' in value:
            raise ValueError('前端资源路径非法')
        url = urlsplit(value)
        if url.scheme or url.netloc or not url.path:
            raise ValueError('前端不得依赖外部脚本/样式/图片：' + value)
        path = (directory / unquote(url.path).lstrip('/')).resolve()
        if directory not in path.parents or not path.is_file() or not path.stat().st_size:
            raise ValueError('前端资源缺失、为空或路径越界：' + value)
        checked[path.relative_to(directory).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    parser = AssetParser()
    parser.feed(local_file('index.html').read_text(encoding='utf-8'))
    if not parser.assets:
        raise ValueError('前端 index.html 没有构建资源')
    index_files = {local_file(value).relative_to(directory).as_posix() for value in parser.assets}
    manifest = json.loads(local_file('.vite/manifest.json').read_text(encoding='utf-8'))
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError('缺少完整 Vite 构建清单，请重新构建前端')
    entries = set()
    for name, chunk in manifest.items():
        if not isinstance(chunk, dict) or not isinstance(chunk.get('file'), str):
            raise ValueError('Vite 构建清单记录非法：' + name)
        file = local_file(chunk['file']).relative_to(directory).as_posix()
        if chunk.get('isEntry'):
            entries.add(file)
        for key in ('css', 'assets', 'imports', 'dynamicImports'):
            values = chunk.get(key, [])
            if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
                raise ValueError('Vite 构建清单引用非法：' + name)
            for value in values:
                if key in {'imports', 'dynamicImports'}:
                    if value not in manifest:
                        raise ValueError('Vite 动态/共享模块未登记：' + value)
                else:
                    local_file(value)
    if not entries or not entries <= index_files:
        raise ValueError('前端首页和构建清单不属于同一版本')
    return {'indexAssets': len(parser.assets), 'runtimeFiles': len(checked),
            'files': checked,
            'fingerprint': hashlib.sha256(json.dumps(checked, sort_keys=True).encode()).hexdigest()}


def wait_preflight(service, job, timeout=180):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = service.get(job['id'])
        if job['status'] in {'completed', 'failed', 'stopped', 'interrupted'}:
            if job['status'] != 'completed' or not job.get('cleanup', {}).get('ok'):
                raise RuntimeError(job.get('error') or f"预检失败：{job['status']}")
            return job
        time.sleep(.2)
    service.stop(job['id'])
    raise TimeoutError('预检超时；已停止本次检查任务')


def check_resources(require_cache=False, gpu=False):
    from PIL import Image
    from federatedscope.standalone_api.platform_config import sha256
    from federatedscope.standalone_api.platform_service import PlatformService
    reports, image_hashes = [], {}
    with tempfile.TemporaryDirectory(prefix='fs-deployment-check-') as temp:
        service = PlatformService(REPO, temp)
        try:
            for method in ('fedavg', 'fedprox', 'heterogeneous_solution'):
                request = service.configs.defaults('military_vit', method)
                request['gpu'] = 0 if gpu else -1
                _, choice, _, _ = service._augmentation_execution(request)
                if require_cache and method == 'heterogeneous_solution' and choice['mode'] != 'reuse':
                    raise RuntimeError('缺少匹配默认 40/20/20 配置的独立增强缓存包；检查不会临时生成')
                job = wait_preflight(service, service.create('inspect',
                    dict(request, idempotencyKey=uuid.uuid4().hex)))
                data = job['result']
                if [data['clientCount'], data['trainSamples'], data['testSamples'],
                    data['featureDimension'], data['classCount']] != [15, 525, 225, 512, 5]:
                    raise RuntimeError('数据规模不符合军机默认配置：15客户端/525训练/225测试/512维/5类')
                reports.append(dict(method=method, parameters=request, selection=choice,
                    fingerprint=data['fingerprint'], testFingerprint=data['testFingerprint'],
                    trainSamples=data['trainSamples'], testSamples=data['testSamples'],
                    provenance=data['testProvenance']))
                if not image_hashes:
                    manifest = json.loads((service.directory(job['id']) / 'data_manifest.json').read_text())
                    keys = {row[1] for rows in manifest['partition'].values() for row in rows}
                    keys.update(key for rows in manifest['test'].values() for key in rows)
                    for key in sorted(keys):
                        path = service.samples.image_path(job, {'key': key})
                        with Image.open(path) as image:
                            image.verify()
                        image_hashes[key] = sha256(path)
                    if len(image_hashes) != 750:
                        raise RuntimeError('原图数量不符合 750 张的军机默认数据范围')
            if len({row['testFingerprint'] for row in reports}) != 1:
                raise RuntimeError('三种方法的测试集不一致')
        finally:
            service.close()
    import hashlib
    return dict(methods=reports, imageCount=len(image_hashes),
                imageFingerprint=hashlib.sha256(json.dumps(image_hashes, sort_keys=True).encode()).hexdigest())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--require-cache', action='store_true')
    parser.add_argument('--state-dir', default='exp/platform')
    parser.add_argument('--frontend', default='../frontend/dist')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    os.environ['FS_PLATFORM_OFFLINE'] = '1'
    os.chdir(REPO)
    from check_platform_environment import check_environment
    from federatedscope.standalone_api.platform_paths import resolve_path
    report = dict(ok=False, scope='read-only resources / disposable preflight / no training', errors=[])
    try:
        environment = check_environment(gpu=args.gpu, strict=True)
        report['environment'] = environment
        if not environment['ok']:
            raise RuntimeError('运行环境未通过：' + '; '.join(environment['errors']))
        report['frontend'] = check_frontend(resolve_path(REPO, args.frontend))
        state = resolve_path(REPO, args.state_dir)
        if not state.is_dir():
            raise RuntimeError('状态目录不存在；请先创建并赋予运行用户写权限：' + str(state))
        with tempfile.TemporaryFile(dir=state) as probe:
            probe.write(b'platform-write-check')
            probe.flush()
        report['resources'] = check_resources(args.require_cache, args.gpu)
        report['ok'] = True
    except Exception as error:
        report['errors'].append(str(error))
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        args.report.write_text(encoded, encoding='utf-8')
    print(encoded)
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
