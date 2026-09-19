import json
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from federatedscope.standalone_api.platform_backdoor_training import BackdoorTrainingService
from federatedscope.standalone_api.platform_config import PlatformError


@pytest.fixture
def service(tmp_path, monkeypatch):
    repo = Path(__file__).resolve().parents[1]
    for name in ('FS_BACKDOOR_DEVICE', 'FS_PLATFORM_DEVICE', 'FS_BACKDOOR_VIT_WEIGHTS'):
        monkeypatch.delenv(name, raising=False)
    resources = tmp_path / 'resources'
    weights = resources / 'models/ViT-B-16.pt'
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b'config-only test fixture')
    monkeypatch.setenv('FS_PLATFORM_RESOURCES', str(resources))
    dataset = tmp_path / 'dataset'
    (dataset / 'images').mkdir(parents=True)
    (dataset / 'manifest.json').write_text('{}')
    manager = BackdoorTrainingService(repo, tmp_path / 'state', tmp_path / 'groups', tmp_path / 'runs')
    monkeypatch.setattr(manager, '_resolve_dataset', lambda identifier: (
        dict(id='a' * 32, name='fixture', classes=['A', 'B']), dataset))
    return manager, weights


@pytest.mark.parametrize('backdoor,platform,explicit,expected', [
    ('cpu', None, None, 'cpu'), (None, 'cpu', None, 'cpu'),
    (None, None, None, 'cuda'), ('cpu', 'cpu', 'cuda', 'cuda'),
    ('cuda', 'cpu', None, 'cuda'), ('cuda', None, 'cpu', 'cpu'),
])
def test_device_propagates_to_all_three_configs(service, monkeypatch, backdoor, platform, explicit, expected):
    manager, weights = service
    if backdoor:
        monkeypatch.setenv('FS_BACKDOOR_DEVICE', backdoor)
    if platform:
        monkeypatch.setenv('FS_PLATFORM_DEVICE', platform)
    payload = {'rounds': 1}
    if explicit:
        payload['device'] = explicit
    # Do not launch actual training in configuration unit tests.
    with patch('federatedscope.standalone_api.platform_backdoor_training.threading.Thread') as thread:
        job = manager.start(payload)
        thread.return_value.start.assert_called_once()
    assert job['device'] == expected
    spec = json.loads((manager.directory(job['id']) / 'spec.json').read_text())
    assert spec['device'] == expected
    assert len(spec['runs']) == 3
    for run in spec['runs']:
        raw = yaml.safe_load(Path(run['config']).read_text(encoding='utf-8'))
        assert raw['use_gpu'] == (expected == 'cuda')
        assert raw['device'] == 0
        assert Path(raw['ggeur']['clip_model_path']) == weights


def test_missing_weights_or_invalid_device_does_not_queue_job(service, monkeypatch):
    manager, weights = service
    with pytest.raises(PlatformError, match='设备只能'):
        manager.start({'device': 'invalid'})
    monkeypatch.setenv('FS_BACKDOOR_VIT_WEIGHTS', str(weights.parent / 'missing.pt'))
    with pytest.raises(PlatformError, match='缺少后门训练'):
        manager.start({})
    assert manager.list() == []
