"""Verify copied files and exercise saved models/privacy using ONLY package paths.

Run from any CWD. No training or historical-result rewriting. Reports should be
outside the release. --files-only needs only Python's standard library.
"""
import argparse
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tempfile
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))
from prepare_platform_directory import files, sha256, state_snapshot


def inside(root, path):
    path = Path(path).resolve()
    path.relative_to(root.resolve())
    if not path.exists():
        raise ValueError('missing resource: ' + str(path))
    return path


def verify_files(root):
    if (root / 'PREPARATION_INCOMPLETE').exists():
        raise ValueError('directory preparation did not complete')
    inventory = json.loads((root / 'FILES.sha256.json').read_text(encoding='utf-8'))
    for _ in files(root):
        pass  # Reject links, including links not present in the inventory.
    for name, expected in inventory.items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or ':' in name or '\\' in name:
            raise ValueError('unsafe inventory path: ' + name)
        path = inside(root, root.joinpath(*relative.parts))
        if path.stat().st_size != expected['bytes'] or sha256(path) != expected['sha256']:
            raise ValueError('file changed: ' + name)
    return dict(files=len(inventory), bytes=sum(v['bytes'] for v in inventory.values()))


def check_runtime():
    from check_platform_deployment import check_frontend
    from federatedscope.standalone_api.platform_app import PlatformHandler
    from federatedscope.standalone_api.platform_paths import resolve_path
    from federatedscope.standalone_api.platform_service import PlatformService
    from federatedscope.standalone_api.platform_worker import evaluate, predict
    root = REPO.parent
    before = state_snapshot(REPO / 'exp/platform')
    service = PlatformService(REPO, 'exp/platform', recover=False)
    try:
        handler = object.__new__(PlatformHandler)
        handler.context = SimpleNamespace()
        handler._download = lambda path, *args, **kwargs: inside(root, path)
        paths = [service.configs.resources, service.configs.datasets,
                 service.configs.cache_dir('military_vit'),
                 service.configs.augmentation_dir('military_vit'), handler._fedmia_root(),
                 resolve_path(REPO, os.environ.get('FEDERATEDSCOPE_FRONTEND_DIST', '../frontend/dist'))]
        for path in paths:
            inside(root, path)
        frontend = check_frontend(paths[-1])
        library = service.library()
        if not library['models'] or not library['testsets']:
            raise ValueError('saved model library is empty')
        # Empty state proves the default does not depend on a historical job ID.
        with tempfile.TemporaryDirectory(prefix='fs-empty-state-') as empty:
            clean = PlatformService(REPO, empty)
            try:
                request = clean.configs.defaults('military_vit', 'heterogeneous_solution')
                selection = clean._augmentation_execution(request)[1]
                if selection != {'mode': 'reuse', 'bundle': True}:
                    raise ValueError('default independent 40/20/20 cache not available')
            finally:
                clean.close()
        image_count, evaluated = 0, []
        for testset in library['testsets']:
            job, manifest = service.samples.manifest(testset['id'])
            for sample in service.samples.rows(manifest):
                inside(root, service.samples.image_path(job, sample))
                image_count += 1
        for model in library['models']:
            job = service.get(model['jobId'])
            directory = service.directory(model['jobId'])
            manifest = directory / 'data_manifest.json'
            service._evaluation_request({'modelId': model['id'], 'testsetId': job['id']})
            with tempfile.TemporaryDirectory(prefix='fs-model-check-') as temporary:
                spec = dict(output=temporary, request={}, featureSpace=model['featureSpace'],
                    checkpointPath=str(directory / ('checkpoints/mlp_' + model['kind'] + '.pt')),
                    checkpointHash=model['sha256'],
                    bundlePath=str(directory / 'checkpoints/pretrained_test_features.pt'),
                    bundleHash=job['result']['artifactHashes']['pretrained_test_features.pt'])
                evaluate(spec)
                result = json.loads((Path(temporary) / 'result.json').read_text())
                if result['samples'] != job['result']['testSamples']:
                    raise ValueError('test sample count changed: ' + model['id'])
                if model['kind'] == 'final' and job.get('metrics'):
                    if abs(result['accuracy'] - job['metrics'][-1]['accuracy']) > 1e-12:
                        raise ValueError('re-evaluation disagrees with recorded final: ' + model['id'])
                _, data = service.samples.manifest(job['id'])
                sample = next(service.samples.rows(data))
                picture = service.samples.image_path(job, sample)
                spec.update(request=dict(imageSha256=sha256(picture)),
                    sample=dict(sample, imagePath=str(picture)), classNames=data['classes'],
                    manifestPath=str(manifest), manifestHash=sha256(manifest),
                    testProvenance=data.get('testProvenance', {}).get(sample['domain'], 'unknown'))
                predict(spec)
                prediction = json.loads((Path(temporary) / 'result.json').read_text())
                if prediction['mode'] != 'frozen-feature-classifier':
                    raise ValueError('unexpected inference mode')
                evaluated.append(dict(modelId=model['id'], samples=result['samples'],
                                      accuracy=result['accuracy'], prediction=True))
        clients = handler._fedmia_clients()
        if not clients:
            raise ValueError('no privacy clients')
        privacy_images = 0
        for client in clients:
            for group in ('member', 'nonmember'):
                payload = handler._fedmia_membership({'clientId': [str(client)], 'group': [group], 'limit': ['200']})
                if not payload['configured'] or not payload['items']:
                    raise ValueError('missing privacy results for client ' + str(client))
                for item in payload['items']:
                    # Resolve using the same endpoint handler, including its
                    # dataset containment and index validation.
                    handler._fedmia_image(client, group, item['sampleIndex'])
                    privacy_images += 1
        if state_snapshot(REPO / 'exp/platform') != before:
            raise ValueError('historical experiment files changed during checking')
        return dict(frontend=frontend, defaultCache=selection,
                    jobs=len(service.list()), models=len(library['models']),
                    testsets=len(library['testsets']), testImageReferences=image_count,
                    evaluated=evaluated, privacyClients=len(clients), privacyImageReferences=privacy_images)
    finally:
        service.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--files-only', action='store_true')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    report = dict(ok=False, directory=str(REPO.parent), cwd=str(Path.cwd()), errors=[])
    try:
        report['inventory'] = verify_files(REPO.parent)
        if not args.files_only:
            # This is a dedicated validation process, not the HTTP server.
            os.environ.update(FS_PLATFORM_OFFLINE='1', FEDERATEDSCOPE_GGEUR_LIGHTWEIGHT='1')
            from federatedscope.standalone_api.platform_offline import configure_offline_worker
            configure_offline_worker()
            report['runtime'] = check_runtime()
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
