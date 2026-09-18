"""Local, per-run FedMIA artifacts; never overwrite an imported experiment."""
import os
from pathlib import Path
import re
import shutil

import yaml

from .paths import env_path
from .platform_paths import relative_path
from .repository import JsonRepository


def run_directory(repo, request, job_id):
    repo = Path(repo).resolve()
    if not re.fullmatch(r'[a-f0-9]{32}', job_id):
        raise ValueError('Invalid privacy experiment id')
    group = request['group']
    if group not in {'military_cnn', 'officehome_cnn'} and not re.fullmatch(r'uploaded_[a-f0-9]{32}', group):
        raise ValueError('Unknown privacy dataset')
    default = repo.parent / 'fedmia_local'
    if not default.is_dir():
        default = repo / 'resources' / 'fedmia_local'
    root = env_path('FS_FEDMIA_LOCAL_ROOT', default, repo)
    mode = 'defense' if request['defense'] else 'no_defense'
    return root / 'training_runs' / group / mode / job_id


def feature_files(output, archive):
    files = {p.name: p for p in Path(output).rglob('client_*_features_round*.pt') if p.stat().st_size}
    files.update({p.name: p for p in (Path(archive) / 'ggeur_fedmia_features').glob('client_*_features_round*.pt') if p.stat().st_size})
    return sorted(files.values(), key=lambda p: p.name)


def feature_summary(repo, job, output):
    archive = run_directory(repo, job['request'], job['id'])
    files = feature_files(output, archive)
    rounds, clients = set(), set()
    for file in files:
        match = re.fullmatch(r'client_(\d+)_features_round(\d+)\.pt', file.name)
        if match:
            clients.add(int(match[1]))
            rounds.add(int(match[2]))
    return dict(files=len(files), rounds=sorted(rounds), clients=len(clients),
                directory=relative_path(repo, archive), autoSaved=True,
                resultsReady=(archive / 'privacy_results.json').is_file() or (Path(output) / 'privacy_results.json').is_file())


def prepare_run(repo, request, job_id, output):
    """Save portable configuration and exact image identities next to features."""
    output = Path(output)
    archive = run_directory(repo, request, job_id)
    archive.mkdir(parents=True, exist_ok=True)
    (archive / 'ggeur_fedmia_features').mkdir(exist_ok=True)
    for name in ('effective.yaml', 'resolved_config.yaml', 'data_manifest.json'):
        source = output / name
        if source.is_file():
            destination = archive / name
            if not destination.exists():
                shutil.copy2(source, destination)
    config = output / 'resolved_config.yaml'
    if not config.is_file():
        config = output / 'effective.yaml'
    if config.is_file() and not (archive / 'config.yaml').exists():
        raw = yaml.safe_load(config.read_text(encoding='utf-8'))
        root = Path(raw['data']['root'])
        if not root.is_absolute():
            root = Path(repo) / root
        raw['data']['root'] = os.path.relpath(root.resolve(), archive).replace('\\', '/')
        raw['outdir'] = '.'
        (archive / 'config.yaml').write_text(yaml.safe_dump(raw, allow_unicode=True), encoding='utf-8')
    return archive


def archive_existing_run(repo, job, output):
    archive = prepare_run(repo, job['request'], job['id'], output)
    for source in feature_files(output, archive):
        destination = archive / 'ggeur_fedmia_features' / source.name
        if not destination.exists():
            shutil.copy2(source, destination)
    for name in ('privacy_results.json', 'privacy_feature_manifest.json'):
        source = Path(output) / name
        if source.is_file() and not (archive / name).exists():
            shutil.copy2(source, archive / name)
    JsonRepository._atomic_write(archive / 'run.json', dict(jobId=job['id'], request=job['request'],
        status=job['status'], sourceJob=relative_path(archive, output),
        features=feature_summary(repo, job, output)))
    return archive
