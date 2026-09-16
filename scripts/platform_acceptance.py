"""Exercise the real HTTP cache -> train -> checkpoint -> independent evaluation flow."""
import argparse
import json
import time
import urllib.request
import uuid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8001')
    parser.add_argument('--group', default='officehome_vit')
    parser.add_argument('--method', default='fedavg')
    parser.add_argument('--rounds', type=int, default=2)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--name', default='')
    parser.add_argument('--generated-per-sample', type=int)
    parser.add_argument('--generated-per-prototype', type=int)
    parser.add_argument('--target-per-class', type=int)
    parser.add_argument('--covariance-scale', type=float)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()

    def call(path, payload=None):
        body = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(args.url + path, data=body,
                                         headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)['data']

    def wait(job):
        deadline = time.monotonic() + 600
        last_stage = None
        while time.monotonic() < deadline:
            job = call('/api/platform/jobs/' + job['id'])
            if job['stage'] != last_stage:
                print(job['id'], job['status'], job['stage'], flush=True)
                last_stage = job['stage']
            if job['status'] in {'completed', 'failed', 'stopped', 'interrupted'}:
                if job['status'] != 'completed':
                    print(call('/api/platform/jobs/' + job['id'] + '/logs')[-14000:], flush=True)
                    raise RuntimeError(job.get('error') or job['status'])
                return job
            time.sleep(1)
        call('/api/platform/jobs/' + job['id'] + '/stop', {})
        raise TimeoutError('Acceptance task timed out and was stopped')

    request = dict(group=args.group, method=args.method, rounds=args.rounds,
                   name=args.name or '真实闭环验收-' + args.group, gpu=args.gpu)
    augmentation = {
        'generatedPerSample': args.generated_per_sample,
        'generatedPerPrototype': args.generated_per_prototype,
        'targetPerClass': args.target_per_class,
        'covarianceScale': args.covariance_scale,
    }
    if any(value is not None for value in augmentation.values()):
        if args.method != 'heterogeneous_solution':
            parser.error('augmentation parameters require --method heterogeneous_solution')
        request.update({key: value for key, value in augmentation.items()
                        if value is not None})
    preflight = wait(call('/api/platform/preflight', dict(request, idempotencyKey=uuid.uuid4().hex)))
    if args.preflight_only:
        print(json.dumps({'group': args.group, 'preflightId': preflight['id'],
            'clients': preflight['result']['clientCount'], 'trainSamples': preflight['result']['trainSamples'],
            'testSamples': preflight['result']['testSamples']}, ensure_ascii=False), flush=True)
        return
    trained = wait(call('/api/platform/train', dict(request, preflightId=preflight['id'], idempotencyKey=uuid.uuid4().hex)))
    evaluation = wait(call('/api/platform/evaluate', dict(modelId=trained['id'] + ':final',
        testsetId=trained['id'], name='独立评测验收', idempotencyKey=uuid.uuid4().hex)))
    expected = trained['metrics'][-1]
    result = evaluation['result']
    for domain, accuracy in expected['domains'].items():
        assert abs(accuracy - result['domains'][domain]['accuracy']) < 1e-7, domain
    assert abs(expected['accuracy'] - result['accuracy']) < 1e-7
    print(json.dumps({'trainId': trained['id'], 'evaluationId': evaluation['id'],
        'accuracy': result['accuracy'], 'macroF1': result['macroF1'],
        'domains': {d: r['accuracy'] for d, r in result['domains'].items()},
        'testSamples': result['samples'], 'agreement': True,
        'cleanup': trained['cleanup']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
