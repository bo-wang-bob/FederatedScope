"""Check or export the current sibling frontend/backend deployment, without Docker.

No dependencies beyond Python's standard library. Resource contents are copied
unchanged; this is a layout/integrity check, not an algorithm acceptance test.
"""
import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.prepare_platform_directory import copy_checked, files, sha256, state_snapshot

BACKEND_CODE = ('federatedscope', 'scripts', 'deploy', 'docs', 'tests',
                'run.py', 'setup.py', 'README.md', 'BACKDOOR_RUNBOOK.md', 'LICENSE')
FRONTEND_CODE = ('src', 'resource', 'tests', 'public', 'package.json',
                 'package-lock.json', 'index.html', 'vite.config.ts',
                 'vitest.config.ts', 'playwright.config.ts', 'README.md', 'LICENSE', 'dist')
REQUIRED_RESOURCES = (
    'datasets/MilitaryAircraft3D',
    'exp/distributed_feature_cache/military_aircraft_vit_fixedsplit_v2',
    'caches/military_vit_default',
    'models/ViT-B-16.pt', 'torch/hub/checkpoints/convnext_base-6075fbad.pth',
    'fedmia_local', 'backdoor/exp',
)


def inventory(root):
    return {p.relative_to(root).as_posix(): (p.stat().st_size, p.stat().st_mtime_ns)
            for p in files(root)}


def inspect(backend, frontend):
    for relative in ('scripts/start_platform.py', 'docs/RELEASE_LAYOUT.md'):
        if not (backend / relative).is_file():
            raise ValueError('missing backend file: ' + relative)
    for relative in ('package-lock.json', 'dist/index.html'):
        if not (frontend / relative).is_file():
            raise ValueError('missing frontend file (run npm ci && npm run build): ' + relative)
    resources = backend / 'resources'
    for relative in REQUIRED_RESOURCES:
        path = resources / relative
        if not path.exists() or not list(files(path)):
            raise ValueError('missing/empty resource: ' + relative)
    resource_files = inventory(resources)
    retired = [name for name in resource_files
               if 'officehome' in name.lower().replace('-', '').replace('_', '')]
    if retired:
        raise ValueError('retired OfficeHome resources must be removed: ' + retired[0])
    uploads = resources / 'uploaded_datasets'
    if uploads.exists():
        for directory in uploads.iterdir():
            if directory.is_dir():
                metadata = directory / 'dataset.json'
                if not metadata.is_file() or json.loads(metadata.read_text(encoding='utf-8')).get('status') != 'ready':
                    raise ValueError('unfinished dataset upload: ' + directory.name)
    snapshot = state_snapshot(backend / 'exp/platform')
    return resource_files, snapshot


def export(backend, frontend, output):
    backend, frontend, output = backend.resolve(), frontend.resolve(), output.resolve()
    if output.exists():
        raise ValueError('destination already exists; use a new directory')
    for source in (backend, frontend):
        if output == source or source in output.parents or output in source.parents:
            raise ValueError('destination must not overlap source directories')
    resources_before, snapshot = inspect(backend, frontend)
    output.mkdir(parents=True)
    marker = output / 'PREPARATION_INCOMPLETE'
    marker.write_text('Do not deploy until export succeeds.\n', encoding='utf-8')
    for name in BACKEND_CODE:
        copy_checked(backend / name, output / 'backend' / name)
    for name in FRONTEND_CODE:
        if (frontend / name).exists():
            copy_checked(frontend / name, output / 'frontend' / name)
    for path in frontend.glob('tsconfig*.json'):
        copy_checked(path, output / 'frontend' / path.name)
    copy_checked(backend / 'resources', output / 'backend/resources')
    for name, digest in snapshot.items():
        target = output / 'backend/exp/platform' / name
        copy_checked(backend / 'exp/platform' / name, target)
        if sha256(target) != digest:
            raise ValueError('experiment changed during copy: ' + name)
    if inventory(backend / 'resources') != resources_before:
        raise ValueError('resources changed during export; retry with service stopped')
    if state_snapshot(backend / 'exp/platform') != snapshot:
        raise ValueError('experiments changed during export; retry with service stopped')
    copy_checked(backend / 'docs/RELEASE_LAYOUT.md', output / 'README.md')
    manifest = {p.relative_to(output).as_posix(): {'bytes': p.stat().st_size, 'sha256': sha256(p)}
                for p in files(output) if p != marker}
    (output / 'FILES.sha256.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    marker.unlink()
    return {'directory': str(output), 'files': len(manifest),
            'bytes': sum(v['bytes'] for v in manifest.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', type=Path, default=REPO)
    parser.add_argument('--frontend', type=Path)
    parser.add_argument('--output', type=Path, help='New release directory; omitted = read-only check')
    args = parser.parse_args()
    backend = args.backend.resolve()
    frontend = (args.frontend or backend.parent / 'frontend').resolve()
    try:
        if args.output:
            result = export(backend, frontend, args.output)
        else:
            resources, snapshot = inspect(backend, frontend)
            result = {'layoutOk': True, 'resourceFiles': len(resources),
                      'resourceBytes': sum(v[0] for v in resources.values()),
                      'stateFiles': len(snapshot), 'note': 'Layout only; run runtime preflight separately.'}
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, str(exc) + '\n')


if __name__ == '__main__':
    main()
