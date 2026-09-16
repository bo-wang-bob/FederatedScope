"""Isolated HTTP smoke test with offline workers; no copying or packaging assets.

Three methods run two rounds only. This checks execution, not accuracy targets.
Creates its own state directory and never connects to an existing API service.
"""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import urllib.request
import uuid

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    os.environ.update(FS_PLATFORM_OFFLINE='1', HF_HUB_OFFLINE='1', WANDB_MODE='disabled')
    from federatedscope.standalone_api.platform_app import create_server
    state = Path(tempfile.mkdtemp(prefix='fs-offline-smoke-'))
    report = dict(ok=False, state=str(state), workerOfflineGuard=True, rounds=2,
                  scope='isolated HTTP / frozen-feature training / no accuracy threshold', methods=[])
    server, thread, base = None, None, ''
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def close():
        if server:
            server.shutdown()
            server.RequestHandlerClass.context.platform.close()
            server.server_close()
            thread.join(timeout=10)

    def start():
        nonlocal server, thread, base
        server = create_server('127.0.0.1', 0, state)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = 'http://127.0.0.1:' + str(server.server_port)

    def download(path):
        with opener.open(base + path, timeout=30) as response:
            return response.read(), response.headers.get_content_type()

    def call(path, payload=None):
        req = urllib.request.Request(base + '/api/platform/' + path,
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={'Content-Type': 'application/json'})
        with opener.open(req, timeout=30) as response:
            return json.load(response)['data']

    def wait(job):
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            job = call('jobs/' + job['id'])
            if job['status'] in {'completed', 'failed', 'stopped', 'interrupted'}:
                if job['status'] != 'completed' or not job.get('cleanup', {}).get('ok'):
                    raise RuntimeError(job.get('error') or job['status'])
                return job
            time.sleep(.25)
        call('jobs/' + job['id'] + '/stop', {})
        raise TimeoutError(job['id'])

    try:
        start()
        service = server.RequestHandlerClass.context.platform
        defaults = service.configs.defaults('military_vit', 'heterogeneous_solution')
        choice = service._augmentation_execution(defaults)[1]
        if choice != {'mode': 'reuse', 'bundle': True}:
            raise RuntimeError('短流程要求独立增强缓存包；不会临时生成或复用正式任务记录')
        for method in ('fedavg', 'fedprox', 'heterogeneous_solution'):
            request = dict(group='military_vit', method=method, rounds=2, gpu=args.gpu)
            preflight = wait(call('preflight', dict(request, idempotencyKey=uuid.uuid4().hex)))
            trained = wait(call('train', dict(request, preflightId=preflight['id'], idempotencyKey=uuid.uuid4().hex)))
            evaluated = wait(call('evaluate', dict(modelId=trained['id'] + ':final',
                testsetId=trained['id'], idempotencyKey=uuid.uuid4().hex)))
            result, expected = evaluated['result'], trained['metrics'][-1]
            if result['samples'] != 225 or abs(result['accuracy'] - expected['accuracy']) > 1e-12:
                raise RuntimeError('保存模型重新评测与训练结果不一致')
            for domain, accuracy in expected['domains'].items():
                if abs(result['domains'][domain]['accuracy'] - accuracy) > 1e-12:
                    raise RuntimeError('分域结果不一致：' + domain)
            sample = call('testsets/' + trained['id'] + '/samples?limit=1')['items'][0]
            picture, content_type = download(sample['imageUrl'])
            if not picture or not content_type.startswith('image/'):
                raise RuntimeError('测试原图 HTTP 返回无效')
            prediction = wait(call('predict', dict(modelId=trained['id'] + ':final', testsetId=trained['id'],
                sampleId=sample['id'], imageSha256=sample['imageSha256'], idempotencyKey=uuid.uuid4().hex)))
            if prediction['result']['mode'] != 'frozen-feature-classifier':
                raise RuntimeError('模型验证口径已改变')
            payload, _ = download('/api/platform/jobs/' + evaluated['id'] + '/csv')
            rows = list(csv.reader(io.StringIO(payload.decode('utf-8-sig'))))
            if len(rows) != 21:
                raise RuntimeError('评测 CSV 未覆盖总体及三域的五类指标')
            payload, _ = download('/api/platform/jobs/' + evaluated['id'] + '/export')
            if json.loads(payload)['id'] != evaluated['id']:
                raise RuntimeError('评测导出来源错误')
            payload, _ = download('/api/platform/jobs/' + trained['id'] + '/model-final')
            import hashlib
            if hashlib.sha256(payload).hexdigest() != trained['result']['artifactHashes']['mlp_final.pt']:
                raise RuntimeError('模型下载哈希不一致')
            report['methods'].append(dict(method=method, trainId=trained['id'], evaluationId=evaluated['id'],
                predictionId=prediction['id'], samples=result['samples'], evaluationAgreement=True,
                modelDownloadVerified=True, csvClassRows=len(rows) - 1,
                augmentation=trained['result']['augmentation']['mode'], cleanup=trained['cleanup']['ok']))
            print(json.dumps(report['methods'][-1], ensure_ascii=False), flush=True)
        before = call('library')
        close()
        server = None
        start()
        after = call('library')
        if before != after or len(after['models']) != 6 or len(after['testsets']) != 3:
            raise RuntimeError('服务重启后模型库或测试集发生变化')
        report.update(ok=True, restartLibraryPreserved=True)
    except Exception as error:
        report['error'] = str(error)
    finally:
        close()
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('REPORT', args.report, flush=True)
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
