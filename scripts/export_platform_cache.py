"""Copy one verified generation run into a portable default-cache directory."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from federatedscope.standalone_api.platform_config import sha256
from federatedscope.standalone_api.platform_paths import cache_file, resolve_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--state', default='exp/platform')
    parser.add_argument('--job', required=True)
    parser.add_argument('--output', default='resources/caches/military_vit_default')
    args = parser.parse_args()
    import re
    if not re.fullmatch(r'[a-f0-9]{32}', args.job):
        parser.error('invalid job ID')
    job_dir = resolve_path(REPO, args.state) / 'jobs' / args.job
    job = json.loads((job_dir / 'job.json').read_text(encoding='utf-8'))
    aug = copy.deepcopy(job.get('result', {}).get('augmentation', {}))
    if (job['action'] != 'train' or job['status'] != 'completed' or aug.get('mode') != 'generate'
            or aug.get('provenance') != 'generated-from-recorded-training-samples'
            or len(aug.get('clients', {})) != job['request']['clientCount']):
        parser.error('source must be a complete, provenance-checked generation run')
    output = resolve_path(REPO, args.output)
    if output.exists():
        parser.error('output already exists; choose a new directory')
    sources = []
    for cid, row in aug['clients'].items():
        source = cache_file(job_dir / 'augmented_cache', row['path'], job['id'])
        if not source.is_file() or sha256(source) != row['sha256']:
            raise ValueError('source cache changed: ' + cid)
        name = f'client_{int(cid):06d}.pt'
        row['path'] = name
        sources.append((source, name))
    output.mkdir(parents=True)
    for source, name in sources:
        shutil.copy2(source, output / name)
    (output / 'augmentation.json').write_text(json.dumps(dict(
        formatVersion=1, request=job['request'], augmentation=aug,
        sourceJobId=job['id']), ensure_ascii=False, indent=2), encoding='utf-8')
    print(output)


if __name__ == '__main__':
    main()
