"""Preflight / short GPU workflow tests in a new, isolated platform state.

Never edits datasets, substitutes missing caches, or changes live experiments.
This is functional verification, not an accuracy acceptance benchmark.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
import uuid

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', required=True, type=Path)
    parser.add_argument('--groups', nargs='+')
    parser.add_argument('--methods', nargs='+', default=['fedavg'])
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()
    if args.state.exists():
        parser.error('state directory must be new; live state must not be used')
    os.environ.update(FS_PLATFORM_OFFLINE='1', FEDERATEDSCOPE_GGEUR_LIGHTWEIGHT='1',
                      WANDB_MODE='disabled', PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python')
    from federatedscope.standalone_api.platform_service import PlatformService, TERMINAL
    service = PlatformService(REPO, args.state)
    report = []

    def run(action, payload):
        job = service.create(action, {**payload, 'idempotencyKey': uuid.uuid4().hex})
        deadline = time.monotonic() + 600
        while job['status'] not in TERMINAL:
            if time.monotonic() > deadline:
                service.stop(job['id'])
                raise TimeoutError('smoke job exceeded 600s: ' + job['id'])
            time.sleep(.5)
            job = service.get(job['id'])
        if job['status'] != 'completed':
            raise RuntimeError(job.get('error') or job['status'])
        return job

    try:
        groups = args.groups or [g for g in service.configs.groups() if g != 'military_vit']
        for group in groups:
            for method in args.methods:
                row = {'group': group, 'method': method, 'ok': False}
                try:
                    request = service.configs.defaults(group, method)
                    request.update(rounds=1, localEpochs=1, gpu=0, name='三方接入功能检查 ' + group)
                    checked = run('inspect', request)
                    row.update(preflight=checked['id'], trainSamples=checked['result']['trainSamples'],
                               testSamples=checked['result']['testSamples'], classes=checked['result']['classCount'])
                    if args.train:
                        trained = run('train', {**request, 'preflightId': checked['id']})
                        evaluated = run('evaluate', {'modelId': trained['id'] + ':final', 'testsetId': trained['id'],
                                                     'name': '三方接入独立评测 ' + group})
                        row.update(train=trained['id'], evaluation=evaluated['id'], accuracy=evaluated['result']['accuracy'])
                        points = trained.get('metrics') or []
                        if not points or not math.isclose(points[-1]['accuracy'], row['accuracy'], abs_tol=1e-10):
                            raise RuntimeError('Saved-model accuracy differs from final training monitoring')
                        row['monitoringMatchesEvaluation'] = True
                        if not group.startswith('mdsent_'):
                            samples = service.samples.page(trained['id'], {'limit': ['1']})
                            sample = samples['items'][0]
                            if not sample.get('imageAvailable'):
                                raise RuntimeError('sample preview: ' + sample.get('imageError', 'unavailable'))
                            prediction = run('predict', {'modelId': trained['id'] + ':final', 'testsetId': trained['id'],
                                                         'sampleId': sample['id'], 'imageSha256': sample['imageSha256']})
                            row['prediction'] = prediction['id']
                    row['ok'] = True
                except Exception as exc:
                    row['error'] = str(exc)
                report.append(row)
                (args.state / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
                print(json.dumps(row), flush=True)
    finally:
        for job in service.list():
            if job['status'] not in TERMINAL:
                service.stop(job['id'])
        service.close()
    return 0 if all(row['ok'] for row in report) else 1


if __name__ == '__main__':
    sys.exit(main())
