import io
import shutil
from pathlib import Path

import pytest
from PIL import Image

from federatedscope.standalone_api.uploaded_datasets import DatasetStore
from federatedscope.standalone_api.uploaded_config import UploadedConfig
from federatedscope.standalone_api.platform_config import ConfigFactory, PlatformError


def picture(number=0):
    stream = io.BytesIO()
    Image.new('RGB', (12, 12), (number, 50, 90)).save(stream, format='PNG')
    return stream.getvalue()


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv('FS_PLATFORM_UPLOADS', str(tmp_path / 'uploads'))
    source = Path(__file__).resolve().parents[1]
    repo = tmp_path / 'backend'
    for relative in ('scripts/example_configs/ggeur_final_5models/officehome_cnn', 'scripts/example_configs/ggeur_final_5models/officehome_vit', 'scripts/privacy_presets'):
        shutil.copytree(source / relative, repo / relative)
    return DatasetStore(repo)


def train_dataset(store):
    value = store.create(dict(name='自定义飞机', kind='train'))
    for index in range(12):
        store.put(value['id'], f'{"运输机" if index < 6 else "战斗机"}/{index}.png', picture(index))
    return store.finish(value['id'])


def test_class_folders_are_immutable_and_shared(store):
    value = train_dataset(store)
    assert value['classes'] == sorted(['运输机', '战斗机'])
    assert value['count'] == 12 and len(store.list()) == 1
    assert store.image(value['id'], 0).is_file()
    with pytest.raises(PlatformError, match='不可修改'):
        store.put(value['id'], '战斗机/new.png', picture(99))


@pytest.mark.parametrize('path', ['../x.png', 'a/../x.png', '/x.png', 'C:/x.png', 'a\\x.png', 'NUL.png', 'a/x.png:ads', 'a/b/c.png'])
def test_upload_rejects_escaping_and_reserved_paths(store, path):
    identifier = store.create(dict(name='拒绝越界', kind='train'))['id']
    with pytest.raises(PlatformError):
        store.put(identifier, path, picture())


def test_corrupt_duplicate_and_insufficient_data(store):
    identifier = store.create(dict(name='检查', kind='train'))['id']
    with pytest.raises(PlatformError):
        store.put(identifier, 'A/a.png', b'not a picture')
    store.put(identifier, 'A/a.png', picture())
    with pytest.raises(PlatformError, match='重复'):
        store.put(identifier, 'B/b.png', picture())
    with pytest.raises(PlatformError):
        store.finish(identifier)
    assert store.list() == []


def test_labelled_test_folder_and_single_image(store):
    identifier = store.create(dict(name='分类测试', kind='test'))['id']
    store.put(identifier, '类别A/a.png', picture())
    store.put(identifier, '类别B/b.png', picture(1))
    assert store.finish(identifier)['classes'] == ['类别A', '类别B']
    single = store.create(dict(name='单图', kind='test'))['id']
    store.put(single, 'image.png', picture())
    assert store.finish(single)['classes'] == []


def test_flat_test_folder_accepts_multiple_unlabelled_images(store):
    identifier = store.create(dict(name='独立测试图片', kind='test'))['id']
    for index in range(3):
        store.put(identifier, f'picture{index}.png', picture(index))
    value = store.finish(identifier)
    assert value['count'] == 3 and value['classes'] == []
    assert all(store.image(identifier, i).is_file() for i in range(3))
    assert 'group' not in value


def test_test_folder_cannot_mix_labelled_and_unlabelled_images(store):
    identifier = store.create(dict(name='混合目录', kind='test'))['id']
    store.put(identifier, 'a.png', picture())
    store.put(identifier, 'A/b.png', picture(1))
    with pytest.raises(PlatformError, match='不能混合'):
        store.finish(identifier)


@pytest.mark.parametrize('method', ['fedavg', 'fedprox', 'heterogeneous_solution'])
def test_uploaded_training_uses_real_class_count_and_private_cache(store, tmp_path, method):
    value = train_dataset(store)
    base = ConfigFactory(store.repo)
    configs = UploadedConfig(base)
    before, _ = base.build(base.defaults('officehome_cnn', 'fedavg'), tmp_path / 'before')
    req = configs.normalize(configs.defaults(value['group'], method))
    raw, source = configs.build(req, tmp_path / 'training')
    assert raw['data']['type'] == 'domainnet'
    assert raw['model']['num_classes'] == 2
    assert raw['ggeur']['domainnet_domains'] == ['uploaded']
    assert str(value['id']) in raw['ggeur']['feature_cache_dir']
    assert source['datasetFingerprint'] == value['fingerprint']
    assert req['clientCount'] == 3
    assert raw['ggeur']['feature_extractor'] == 'clip'
    assert raw['ggeur']['embedding_dim'] == 512
    assert raw['ggeur']['clip_model'] == 'ViT-B-16'
    assert raw['ggeur']['clip_model_path'] == 'resources/models/ViT-B-16.pt'
    assert raw['ggeur']['feature_cache_dir'].replace('\\', '/').endswith('/features/vit-b16')
    assert source['protocol'].startswith('uploaded-folder-vit')
    after, _ = base.build(base.defaults('officehome_cnn', 'fedavg'), tmp_path / 'before')
    assert before == after
    assert next(g for g in configs.catalog()['groups'] if g['id'] == value['group'])['backbone'] == 'vit'


@pytest.mark.parametrize('defense', [False, True])
def test_privacy_accepts_same_uploaded_dataset(store, tmp_path, defense):
    from federatedscope.standalone_api.platform_privacy import PrivacyConfig
    value = train_dataset(store)
    configs = PrivacyConfig(UploadedConfig(ConfigFactory(store.repo)))
    req = configs.normalize(configs.defaults(value['group'], defense))
    raw, _ = configs.build(req, tmp_path / 'privacy')
    assert raw['model']['num_classes'] == 2
    assert raw['ggeur']['domainnet_domains'] == ['uploaded']
    assert str(value['id']) in raw['data']['root']
    assert raw['ggeur']['use_feature_cache'] is True
    assert raw['ggeur']['require_complete_feature_cache'] is True
    assert raw['ggeur']['feature_extractor'] == 'cnn'
    assert raw['ggeur']['embedding_dim'] == 1024


@pytest.mark.parametrize('evaluation_folder', ['test', 'val', 'valid', 'validation'])
def test_whole_dataset_preserves_user_split(store, evaluation_folder):
    identifier = store.create(dict(name='完整数据集', kind='train', layout='split'))['id']
    for i in range(6):
        store.put(identifier, f'train/{"ants" if i < 3 else "bees"}/{i}.png', picture(i))
    store.put(identifier, f'{evaluation_folder}/ants/test.png', picture(77))
    value = store.finish(identifier)
    assert (value['trainCount'], value['testCount']) == (6, 1)
    assert value['classes'] == ['ants', 'bees']
    import json
    records = json.loads((store.directory(identifier) / 'manifest.json').read_text(encoding='utf-8'))['records']['uploaded']
    assert sum(r['split'] == 'train' for r in records) == 6
    assert [r['path'] for r in records if r['split'] == 'test'] == [f'{evaluation_folder}/ants/test.png.png']
    assert all(store.image(identifier, i).is_file() for i in range(value['count']))


def test_separate_validation_is_not_merged_into_test(store):
    identifier = store.create(dict(name='三个划分', kind='train', layout='split'))['id']
    for i in range(6):
        store.put(identifier, f'train/{"ants" if i < 3 else "bees"}/{i}.png', picture(i))
    store.put(identifier, 'val/ants/a.png', picture(77))
    store.put(identifier, 'test/ants/b.png', picture(78))
    value = store.finish(identifier)
    assert (value['trainCount'], value['testCount'], value['validationCount']) == (6, 1, 1)


def test_whole_dataset_rejects_unknown_test_class(store):
    identifier = store.create(dict(name='错误标签', kind='train', layout='split'))['id']
    for i in range(6):
        store.put(identifier, f'train/{"ants" if i < 3 else "bees"}/{i}.png', picture(i))
    store.put(identifier, 'test/unknown/a.png', picture(77))
    with pytest.raises(PlatformError, match='类别'):
        store.finish(identifier)
