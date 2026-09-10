"""Paired real-data smoke run; report observed differences without claiming significance."""
import argparse
import json
import time
import urllib.request
import uuid


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--url', default='http://127.0.0.1:8001')
    p.add_argument('--source-id', default='')
    p.add_argument('--rounds', type=int, default=5)
    args = p.parse_args()

    def call(path, body=None):
        request = urllib.request.Request(args.url + '/api/platform/' + path,
            data=json.dumps(body).encode() if body is not None else None,
            headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)['data']

    def wait(job):
        deadline, previous = time.monotonic() + 900, None
        while time.monotonic() < deadline:
            job = call('jobs/' + job['id'])
            stage = (job['status'], job['stage'])
            if stage != previous:
                print(job['id'], *stage, flush=True)
                previous = stage
            if job['status'] in {'completed', 'failed', 'stopped', 'interrupted'}:
                if job['status'] != 'completed':
                    raise RuntimeError(str(job.get('error')) + '\n' + call('jobs/' + job['id'] + '/logs')[-1800:])
                return job
            time.sleep(2)
        call('jobs/' + job['id'] + '/stop', {})
        raise TimeoutError('Only this acceptance job was stopped after timeout')

    def submit(action, body):
        return wait(call(action, dict(body, idempotencyKey=uuid.uuid4().hex)))

    group = next(g for g in call('catalog')['groups'] if g['id'] == 'officehome_vit')
    settings = dict(rounds=args.rounds, localEpochs=1, learningRate=.0001, batchSize=32,
        clientCount=60, sampleClients=0, samplesPerClient=128, gpu=1, seed=42,
        splitSeed=42, alpha=.1, evaluationFrequency=1)
    runs = []
    for method, name in [('fedavg', '配对短训-FedAvg'), ('heterogeneous_solution', '配对短训-本架构')]:
        req = dict(next(m for m in group['methods'] if m['id'] == method)['defaults'], **settings)
        req['name'] = name
        if method == 'heterogeneous_solution' and args.source_id:
            req.update(augmentationMode='reuse', augmentationSourceId=args.source_id)
        preflight = submit('preflight', req)
        trained = submit('train', dict(req, preflightId=preflight['id']))
        testing = runs[0]['trainId'] if runs else trained['id']
        evaluation = submit('evaluate', dict(modelId=trained['id'] + ':final',
            testsetId=testing, name=name + '-独立评测'))
        result = evaluation['result']
        assert trained['metrics'][-1]['accuracy'] == result['accuracy']
        runs.append(dict(method=method, trainId=trained['id'], evaluationId=evaluation['id'],
            accuracy=result['accuracy'], macroF1=result['macroF1'], samples=result['samples'],
            testFingerprint=trained['result']['testFingerprint'],
            partitionFingerprint=trained['result']['partitionFingerprint'],
            augmentationProvenance=trained['result'].get('augmentation', {}).get('provenance')))
    assert runs[0]['testFingerprint'] == runs[1]['testFingerprint']
    assert runs[0]['partitionFingerprint'] == runs[1]['partitionFingerprint']
    print(json.dumps(dict(settings=settings, runs=runs,
        accuracyDifferencePoints=100 * (runs[1]['accuracy'] - runs[0]['accuracy']),
        scope='single-seed short-run functional comparison, not evidence of a stable improvement'),
        ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
