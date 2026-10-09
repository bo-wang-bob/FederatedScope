from pathlib import Path
import json
from unittest.mock import patch

import pytest

from federatedscope.standalone_api.platform_config import ConfigFactory, PlatformError
from federatedscope.standalone_api.platform_samples import SampleCatalog
from scripts.add_thirdparty_resources import add_resources

REPO = Path(__file__).resolve().parents[1]


def test_full_training_catalog_retains_military_and_adds_testing_models():
    factory = ConfigFactory(REPO)
    groups = {group['id']: group for group in factory.catalog()['groups']}
    expected = {'officehome_vit', 'officehome_cnn', 'officehome_mixer',
                'digit3_vit', 'digit3_cnn', 'mdsent_rnn', 'mdsent_lstm', 'military_vit'}
    assert set(groups) == expected
    for identifier in expected:
        assert {'fedavg', 'fedprox', 'heterogeneous_solution'} <= {m['id'] for m in groups[identifier]['methods']}
    military = factory.defaults('military_vit', 'heterogeneous_solution')
    assert (military['targetPerClass'], military['generatedPerSample'], military['generatedPerPrototype']) == (40, 20, 20)
    assert military['localEpochs'] == 1


def test_new_cnn_weight_path_is_resolved_under_resource_root(tmp_path):
    with patch.dict('os.environ', {'FS_PLATFORM_RESOURCES': str(REPO / 'resources')}):
        factory = ConfigFactory(REPO)
        cfg, _ = factory.build(factory.normalize(dict(group='officehome_cnn', method='fedavg')), REPO / 'exp/check')
    assert cfg['ggeur']['cnn_checkpoint_path'] == 'resources/pretrained_models/convnext_base-6075fbad.pth'


def test_testing_resource_registry_is_additive_and_portable(tmp_path):
    resources = tmp_path / 'resources'
    resources.mkdir()
    registry = {'officehome_vit': {'dataset': 'datasets/thirdparty/OfficeHome',
                                 'features': 'exp/thirdparty/officehome_vit',
                                 'manifest': 'exp/thirdparty/manifest.json'}}
    (resources / 'thirdparty_resources.json').write_text(json.dumps(registry))
    with patch.dict('os.environ', {'FS_PLATFORM_RESOURCES': str(resources), 'FS_PLATFORM_DATASETS': str(resources / 'datasets')}):
        factory = ConfigFactory(REPO)
        assert factory.dataset_dir('military_vit') == resources / 'datasets/MilitaryAircraft3D'
        assert factory.cache_dir('military_vit') == resources / 'exp/distributed_feature_cache/military_aircraft_vit_fixedsplit_v2'
        assert factory.dataset_dir('officehome_vit') == resources / 'datasets/thirdparty/OfficeHome'
        assert factory.cache_dir('officehome_vit') == resources / 'exp/thirdparty/officehome_vit'
        image = resources / 'datasets/thirdparty/OfficeHome/Art/example/image.jpg'
        image.parent.mkdir(parents=True)
        image.write_bytes(b'new dataset image')
        samples = SampleCatalog(type('Service', (), {'configs': factory})())
        assert samples.image_path({'request': {'group': 'officehome_vit'}},
                                  {'key': 'Art/example/image.jpg'}) == image
        for bad in ('../outside', '/outside', 'C:/outside'):
            factory.resource_registry['officehome_vit']['dataset'] = bad
            with pytest.raises(ValueError):
                factory.dataset_dir('officehome_vit')


@pytest.mark.parametrize('group', ['domainnet_vit', 'domainnet_cnn', 'domainnet_mixer'])
def test_retired_domainnet_cannot_be_started_from_saved_requests(group):
    factory = ConfigFactory(REPO)
    assert group not in factory.groups()
    with pytest.raises(PlatformError, match='不支持的数据配置'):
        factory.normalize({'group': group, 'method': 'fedavg'})


def test_resource_addition_never_overwrites_existing_datasets(tmp_path):
    resources, source = tmp_path / 'resources', tmp_path / 'source'
    resources.mkdir()
    source.mkdir()
    (source / 'picture.jpg').write_bytes(b'new input')
    existing = resources / 'datasets/Existing'
    existing.mkdir(parents=True)
    original = existing / 'picture.jpg'
    original.write_bytes(b'existing input')
    before = original.stat()
    report = add_resources(resources, {'datasets/Existing': str(source), 'datasets/Added': str(source)})
    assert report[0]['status'] == 'preserved-existing'
    assert original.read_bytes() == b'existing input'
    assert original.stat().st_mtime_ns == before.st_mtime_ns
    assert (resources / 'datasets/Added/picture.jpg').read_bytes() == b'new input'
    assert (resources / 'datasets/Added/picture.jpg').samefile(source / 'picture.jpg')
    assert (source / 'picture.jpg').read_bytes() == b'new input'


def test_resource_addition_rejects_escape_and_missing_input_before_writing(tmp_path):
    resources, source = tmp_path / 'resources', tmp_path / 'input'
    resources.mkdir()
    source.write_bytes(b'input')
    for plan in ({'../escape': str(source)}, {'datasets/First': str(source), 'datasets/Missing': str(tmp_path / 'missing')}):
        with pytest.raises((ValueError, FileNotFoundError)):
            add_resources(resources, plan)
        assert list(resources.iterdir()) == []
