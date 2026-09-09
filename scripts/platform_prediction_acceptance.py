"""Real saved model -> held-out image -> classifier inference HTTP acceptance."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request
import uuid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8001')
    parser.add_argument('--group', default='digit3_vit')
    parser.add_argument('--state', type=Path, default=Path('exp/platform'))
    args = parser.parse_args()
    def call(path, payload=None, raw=False):
        request = urllib.request.Request(args.url + '/api/platform/' + path,
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read()
            return (body, response.headers['Content-Type']) if raw else json.loads(body)['data']
    model_info = next(m for m in call('library')['models'] if m['group'] == args.group and m['kind'] == 'final')
    testset_id = model_info['jobId']
    samples = call(f'testsets/{testset_id}/samples?limit=2')['items']
    assert len(samples) == 2 and all(s['imageAvailable'] for s in samples)
    import torch
    from scripts.test_outline_validation.evaluate_saved_mlp import build_model
    torch.set_num_threads(2)
    directory = args.state / 'jobs' / testset_id / 'checkpoints'
    checkpoint = torch.load(directory / 'mlp_final.pt', map_location='cpu', weights_only=True)
    bundle = torch.load(directory / 'pretrained_test_features.pt', map_location='cpu', weights_only=True)
    model = build_model(checkpoint['architecture'])
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval()
    for sample in samples:
        image, mime = call(f'testsets/{testset_id}/samples/{sample["id"]}/image', raw=True)
        assert mime.startswith('image/') and hashlib.sha256(image).hexdigest() == sample['imageSha256']
        payload = dict(modelId=model_info['id'], testsetId=testset_id, sampleId=sample['id'],
                       imageSha256=sample['imageSha256'], name='单图接口验收', idempotencyKey=uuid.uuid4().hex)
        job = call('predict', payload)
        deadline = time.monotonic() + 50
        while time.monotonic() < deadline:
            job = call('jobs/' + job['id'])
            if job['status'] in {'completed', 'failed', 'stopped', 'interrupted'}:
                break
            time.sleep(.3)
        if job['status'] != 'completed':
            call('jobs/' + job['id'] + '/stop', {})
            raise RuntimeError(job.get('error') or job['status'])
        result = job['result']
        start = sample['index'] // 256 * 256
        with torch.inference_mode():
            logits = model(bundle['features'][sample['domain']][start:start + 256].float())[sample['index'] - start]
            expected = int(logits.argmax())
            score = float(torch.softmax(logits, -1)[expected])
        assert result['sampleId'] == sample['id']
        assert result['predictedClass'] == expected
        assert result['label'] == sample['label']
        assert abs(result['confidence'] - score) < 1e-7
        assert result['checkpointSha256'] == model_info['sha256']
        assert job['cleanup']['ok']
        assert call('predict', payload)['id'] == job['id']
        print(json.dumps({'jobId': job['id'], 'group': args.group, 'image': sample['filename'],
            'sampleId': sample['id'], 'predicted': result['predictedName'], 'label': result['labelName'],
            'correct': result['correct'], 'confidence': result['confidence'], 'directInferenceAgreement': True},
            ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
