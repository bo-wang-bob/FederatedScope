"""Real cache training and migration regression, isolated from production."""
import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
import uuid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--source-job', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    package = Path(tempfile.mkdtemp(prefix='fs-portable-flow-'))
    backend = package / 'backend'
    shutil.copytree(repo, backend, ignore=shutil.ignore_patterns('.git', '__pycache__', 'exp', 'resources'))
    shutil.copytree(args.dataset, backend / 'resources/datasets/MilitaryAircraft3D')
    shutil.copytree(args.cache, backend / 'resources/exp/distributed_feature_cache/military_aircraft_vit_fixedsplit_v2')
    imported = backend / 'exp/platform/jobs' / args.source_job.name
    shutil.copytree(args.source_job, imported)
    env = {k: v for k, v in os.environ.items() if not k.startswith('FS_PLATFORM_')}
    env.update(FS_PLATFORM_OFFLINE='1', HF_HUB_OFFLINE='1', WANDB_MODE='disabled', PYTHONDONTWRITEBYTECODE='1')
    report = {'package': str(package), 'training': []}
    proc = None
    log = None
    base = ''

    def start():
        nonlocal proc, log, base
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        base = f'http://127.0.0.1:{port}/api/platform/'
        log = (package / ('service-' + str(port) + '.log')).open('wb')
        proc = subprocess.Popen([sys.executable, '-B', str(backend / 'scripts/start_platform.py'),
            '--port', str(port)], cwd='/tmp', env=env, stdout=log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError('test service exited; see ' + str(package))
            try:
                return call('catalog')
            except OSError:
                time.sleep(.2)
        raise TimeoutError('service startup')

    def stop():
        if proc is not None and proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=30)
        if log:
            log.close()

    def call(path, body=None):
        request = urllib.request.Request(base + path,
            data=json.dumps(body).encode() if body is not None else None,
            headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)['data']

    def wait(job):
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            job = call('jobs/' + job['id'])
            if job['status'] in ('completed', 'failed', 'stopped', 'interrupted'):
                if job['status'] != 'completed':
                    raise RuntimeError(json.dumps(job, ensure_ascii=False))
                assert job['cleanup']['ok']
                return job
            time.sleep(.3)
        call('jobs/' + job['id'] + '/stop', {})
        raise TimeoutError(job['id'])

    def train(method, **extra):
        req = dict(group='military_vit', method=method, rounds=2, **extra)
        preflight = wait(call('preflight', dict(req, idempotencyKey=uuid.uuid4().hex)))
        job = wait(call('train', dict(req, preflightId=preflight['id'], idempotencyKey=uuid.uuid4().hex)))
        result = wait(call('evaluate', dict(modelId=job['id'] + ':final', testsetId=job['id'], idempotencyKey=uuid.uuid4().hex)))
        assert abs(result['result']['accuracy'] - job['metrics'][-1]['accuracy']) < 1e-12
        for domain, accuracy in job['metrics'][-1]['domains'].items():
            assert abs(result['result']['domains'][domain]['accuracy'] - accuracy) < 1e-12
        summary = dict(method=method, id=job['id'], accuracy=result['result']['accuracy'],
            request=job['request'], augmentation=job['result']['augmentation']['mode'],
            cleanup=job['cleanup']['ok'])
        report['training'].append(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        return job

    try:
        catalog = start()
        own = next(m['defaults'] for g in catalog['groups'] if g['id'] == 'military_vit'
                   for m in g['methods'] if m['id'] == 'heterogeneous_solution')
        assert [own[k] for k in ('rounds', 'localEpochs', 'targetPerClass', 'generatedPerSample', 'generatedPerPrototype')] == [100, 1, 40, 20, 20]
        train('fedavg')
        train('fedprox')
        cached = train('heterogeneous_solution')
        assert cached['result']['augmentation']['mode'] == 'reuse'
        generated = train('heterogeneous_solution', augmentationMode='generate')
        assert generated['result']['augmentation']['mode'] == 'generate'
        stop()
        old = package
        package = old.with_name(old.name + '-moved')
        old.rename(package)
        backend = package / 'backend'
        report['package'] = str(package)
        start()
        moved = train('heterogeneous_solution')
        assert moved['result']['augmentation']['mode'] == 'reuse'
        # New model + legacy test bundle: path-only backbone changes are allowed
        # only after the service verifies the identical feature-space hashes.
        result = wait(call('evaluate', dict(modelId=moved['id'] + ':final', testsetId=args.source_job.name,
            idempotencyKey=uuid.uuid4().hex)))
        assert result['result']['samples'] == 225
        sample = call('testsets/' + moved['id'] + '/samples?limit=1')['items'][0]
        assert sample['imageAvailable']
        prediction = wait(call('predict', dict(modelId=moved['id'] + ':final', testsetId=moved['id'],
            sampleId=sample['id'], imageSha256=sample['imageSha256'], idempotencyKey=uuid.uuid4().hex)))
        assert prediction['result']['mode'] == 'frozen-feature-classifier'
        report.update(migration=True, legacyEvaluation=True, prediction=True,
            generationReuseAccuracyMatch=abs(generated['metrics'][-1]['accuracy'] - moved['metrics'][-1]['accuracy']) < 1e-12)
        for j in (cached, generated, moved):
            directory = backend / 'exp/platform/jobs' / j['id']
            spec = json.loads((directory / 'spec.json').read_text())
            assert not Path(spec['output']).is_absolute()
            assert not Path(spec['configPath']).is_absolute()
            if j['result']['augmentation']['mode'] == 'generate':
                assert all(not Path(r['path']).is_absolute() for r in j['result']['augmentation']['clients'].values())
        report['relativeRecords'] = True
    finally:
        stop()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print('REPORT', args.output, flush=True)


if __name__ == '__main__':
    main()
