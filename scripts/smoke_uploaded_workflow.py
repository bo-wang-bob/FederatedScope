"""Real CPU upload/train/test smoke in isolated state, using synthetic fixtures.

Run from backend with the deployment environment loaded. No benchmark claims.
"""
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
import uuid

from PIL import Image


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    root = Path(tempfile.mkdtemp(prefix='upload-smoke-', dir=repo.parent / '.runtime-temp'))
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    base = f'http://127.0.0.1:{port}/api/platform/'
    env = {**os.environ, 'FS_PLATFORM_UPLOADS': str(root / 'uploads'), 'FS_PLATFORM_DEVICE': 'cpu', 'CUDA_VISIBLE_DEVICES': '-1'}
    report = {'fixture': 'synthetic images; functional smoke only', 'root': str(root), 'jobs': [], 'ok': False}
    log = (root / 'service.log').open('wb')
    process = subprocess.Popen([sys.executable, str(repo / 'scripts/start_platform.py'), '--host', '127.0.0.1', '--port', str(port), '--state-dir', str(root / 'state')], cwd=repo, env=env, stdout=log, stderr=log,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    def api(route, payload=None, raw=False):
        data = payload if raw else json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(base + route, data=data, headers={'Content-Type': 'application/octet-stream' if raw else 'application/json'})
        with urllib.request.urlopen(req, timeout=90) as response:
            return json.load(response)['data']
    def image(number, suffix):
        from pillow_heif import register_heif_opener
        register_heif_opener(thumbnails=False)
        formats = {'.jpg': 'JPEG', '.png': 'PNG', '.webp': 'WEBP', '.bmp': 'BMP',
                   '.gif': 'GIF', '.tif': 'TIFF', '.avif': 'AVIF', '.heic': 'HEIF'}
        output = io.BytesIO()
        im = Image.new('RGB', (32, 32))
        im.putdata([((x * 13 + number * 5) % 256, (y * 17 + number * 7) % 256, (number * 19) % 256) for y in range(32) for x in range(32)])
        im.save(output, format=formats[suffix])
        return output.getvalue()
    def upload(name, kind, paths):
        value = api('datasets', dict(name=name, kind=kind))
        from urllib.parse import quote
        for path, index in paths:
            api(f'datasets/{value["id"]}/files?path={quote(path)}', image(index, Path(path).suffix), raw=True)
        return api(f'datasets/{value["id"]}/finish', {})
    def job(action, payload):
        value = api('preflight' if action == 'inspect' else action, {**payload, 'idempotencyKey': uuid.uuid4().hex})
        report['jobs'].append({'id': value['id'], 'action': action})
        deadline = time.monotonic() + 600
        stage = None
        while value['status'] in ('queued', 'running', 'stopping'):
            if value.get('stage') != stage:
                stage = value.get('stage')
                print(action, value['status'], stage, flush=True)
            if time.monotonic() > deadline:
                api(f'jobs/{value["id"]}/stop', {})
                raise TimeoutError(value['id'])
            time.sleep(1)
            value = api('jobs/' + value['id'])
        report['jobs'][-1]['status'] = value['status']
        if value['status'] != 'completed':
            raise RuntimeError(value.get('error') or value)
        return value
    try:
        for _ in range(60):
            try:
                api('catalog')
                break
            except Exception:
                if process.poll() is not None:
                    raise RuntimeError('Smoke server exited; see ' + str(root / 'service.log'))
                time.sleep(1)
        suffixes = ['.jpg', '.png', '.webp', '.bmp', '.gif', '.tif', '.avif', '.heic']
        training = upload('混合格式功能验收训练', 'train', [(f'{"Alpha" if i < 18 else "Beta"}/{i}{suffixes[i % len(suffixes)]}', i) for i in range(36)])
        report['formats'] = suffixes
        assert training['classes'] == ['Alpha', 'Beta']
        group = next(g for g in api('catalog')['groups'] if g['id'] == training['group'])
        assert group['backbone'] == 'vit'
        report['backbone'] = 'ViT-B-16'
        trained = None
        for method in group['methods']:
            request = {**method['defaults'], 'rounds': 1, 'localEpochs': 1, 'batchSize': 8, 'samplesPerClient': 0, 'gpu': -1, 'name': '上传链路验收 ' + method['id']}
            check = job('inspect', request)
            trained = job('train', {**request, 'preflightId': check['id']})
            import torch
            saved = torch.load(root / 'state/jobs' / trained['id'] / 'checkpoints/mlp_final.pt', map_location='cpu', weights_only=True)
            assert saved['backbone']['feature_extractor'] == 'clip' and saved['architecture']['input_dim'] == 512
            assert saved['uploadFeatureSource']['preprocessing'] == 'openclip-vit-b16-openai-eval-float32-v1'
        model = next(m for m in api('library')['models'] if m['jobId'] == trained['id'] and m['kind'] == 'final')
        samples = api(f'testsets/{trained["id"]}/samples?offset=0&limit=12')
        assert samples['items'] and all(item['imageAvailable'] for item in samples['items'])
        for name, paths, labelled in [
            ('平铺测试', [('test_a.tif', 60), ('test_b.heic', 61)], False),
            ('单张测试', [('single.avif', 62)], False),
            ('带标签测试', [('Alpha/a.gif', 63), ('Beta/b.webp', 64)], True),
        ]:
            dataset = upload(name, 'test', paths)
            prediction = job('test-upload', dict(modelId=model['id'], uploadId=dataset['id'], name=name))
            result = prediction['result']
            assert result['samples'] == len(paths) and result['labelled'] == labelled
            assert (result['metrics'] is not None) == labelled
            assert all(row['predictedName'] in training['classes'] for row in result['items'])
            assert all(0 <= row['confidence'] <= 1 for row in result['items'])
            assert {row['filename'] for row in result['items']} == {name for name, _ in paths}
            for row in result['items']:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}' + row['imageUrl'], timeout=90) as response:
                    with Image.open(io.BytesIO(response.read())) as preview:
                        assert preview.format == 'PNG' and preview.mode == 'RGB'
            with urllib.request.urlopen(base + f'jobs/{prediction["id"]}/export', timeout=90) as response:
                assert response.status == 200
        report['ok'] = True
    finally:
        # Stop only this script's test jobs before its own server process.
        try:
            for value in api('jobs'):
                if value['status'] in ('queued', 'running', 'stopping'):
                    api(f'jobs/{value["id"]}/stop', {})
        finally:
            process.terminate()
            process.wait(timeout=15)
            log.close()
            (args.report or repo.parent / 'upload-workflow-smoke.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
