"""Recover local display results from existing native features, without training."""
import json
import os
from pathlib import Path

import yaml


def recompute(repo, job_id):
    from .platform_service import read
    from .privacy_artifacts import archive_existing_run
    from .platform_privacy_worker import finalize
    from federatedscope.core.configs.config import CN, global_cfg
    from federatedscope.contrib.worker.ggeur_fedmia_server import GGEURFedMIAServer
    repo = Path(repo).resolve()
    output = repo / 'exp/single_host_platform/privacy_experiments/jobs' / job_id
    job = read(output / 'job.json')
    fallback = read(output / 'job.recovery.json')
    if fallback and fallback.get('updatedAt', '') > job.get('updatedAt', ''):
        job = fallback
    if job['status'] not in {'completed', 'failed', 'stopped', 'interrupted'}:
        raise ValueError('Wait for the training process to stop before recomputing')
    archive = archive_existing_run(repo, job, output)
    raw = yaml.safe_load((output / 'resolved_config.yaml').read_text(encoding='utf-8'))
    cfg = global_cfg.clone()
    cfg.merge_from_other_cfg(CN(raw))
    server = object.__new__(GGEURFedMIAServer)
    server._cfg = cfg
    server.attack_feature_dir = str(archive / 'ggeur_fedmia_features')
    info = read(output / 'data_manifest.json')
    spec = read(output / 'spec.json')
    spec['trainingComplete'] = job['status'] == 'completed'
    finalize(server, spec, info, info['partition'], info['test'])
    return dict(jobId=job_id, archive=str(archive), completeTraining=spec['trainingComplete'])


if __name__ == '__main__':
    import sys
    os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')
    repo = Path(__file__).resolve().parents[2]
    os.chdir(repo)
    print(json.dumps(recompute(repo, sys.argv[1]), ensure_ascii=True))
