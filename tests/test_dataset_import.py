import json

import pytest
from PIL import Image

from federatedscope.standalone_api.dataset_import import DatasetImporter, PLAN
from federatedscope.standalone_api.platform_config import PlatformError


def picture(path, color=1):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', (8, 8), (color, 20, 30)).save(path)
    return path


@pytest.fixture
def setup(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    source.mkdir()
    monkeypatch.setenv('FS_PLATFORM_IMPORT_ROOTS', json.dumps([str(source)]))
    monkeypatch.setenv('FS_PLATFORM_UPLOADS', str(tmp_path / 'uploads'))
    return DatasetImporter(tmp_path / 'backend'), source


@pytest.mark.parametrize('kind,labelled,single', [('train', True, False), ('test', True, False), ('test', False, False), ('test', False, True)])
def test_import_copies_images_and_registers_portable_dataset(setup, kind, labelled, single):
    importer, source = setup
    files = [picture(source / ('A' if i < 3 else 'B') / f'{i}.png', i) if labelled
             else picture(source / f'{i}.png', i) for i in range(1 if single else 6)]
    before = {str(p): p.read_bytes() for p in files}
    plan = importer.create(dict(path=str(files[0] if single else source), kind=kind, single=single))
    assert plan['total'] == len(files)
    identifier = plan['id']
    for i in range(len(files)):
        # Re-creating the manager simulates resuming after a service restart.
        progress = DatasetImporter(importer.repo).next(identifier)
        assert progress['count'] == i + 1
    assert importer.next(identifier)['count'] == len(files)
    saved = importer.store.finish(identifier)
    assert saved['status'] == 'ready'
    assert saved['classes'] == (['A', 'B'] if labelled else [])
    assert not (importer.store.directory(identifier) / PLAN).exists()
    assert str(source) not in json.dumps(saved)
    assert all(p.read_bytes() == before[str(p)] for p in files)
    assert all(importer.store.image(identifier, i).is_file() for i in range(len(files)))
    assert importer.next(identifier)['count'] == len(files)


def test_scope_missing_shape_and_source_changes(setup):
    importer, source = setup
    with pytest.raises(PlatformError, match='允许的数据目录'):
        importer.create(dict(path=str(source.parent), kind='test'))
    with pytest.raises(PlatformError, match='不存在'):
        importer.create(dict(path=str(source / 'missing'), kind='test'))
    with pytest.raises(PlatformError, match='不允许包含'):
        importer.create(dict(path=str(source / '..' / 'source'), kind='test'))
    image = picture(source / 'a.png')
    with pytest.raises(PlatformError, match='训练集需'):
        importer.create(dict(path=str(source), kind='train'))
    plan = importer.create(dict(path=str(source), kind='test'))
    picture(image, 99)
    with pytest.raises(PlatformError, match='源图片已变化'):
        importer.next(plan['id'])
    assert importer.store.get(plan['id'], False)['count'] == 0


def test_incomplete_import_cannot_finish_or_mix_browser_upload(setup):
    importer, source = setup
    image = picture(source / 'a.png')
    picture(source / 'b.png', 2)
    identifier = importer.create(dict(path=str(source), kind='test'))['id']
    importer.next(identifier)
    with pytest.raises(PlatformError, match='尚未导入完成'):
        importer.store.finish(identifier)
    with pytest.raises(PlatformError, match='不能混入'):
        importer.store.put(identifier, 'c.png', image.read_bytes())


def test_invalid_image_is_not_registered(setup):
    importer, source = setup
    (source / 'bad.png').write_bytes(b'not a real image')
    identifier = importer.create(dict(path=str(source), kind='test'))['id']
    with pytest.raises(PlatformError):
        importer.next(identifier)
    assert importer.store.list() == []


def test_link_is_rejected(setup):
    importer, source = setup
    outside = picture(source.parent / 'secret.png')
    try:
        (source / 'link.png').symlink_to(outside)
    except OSError:
        pytest.skip('Creating symlinks requires OS permission')
    with pytest.raises(PlatformError, match='符号链接|重解析点'):
        importer.create(dict(path=str(source), kind='test'))


def test_unsupported_and_mixed_files_are_rejected(setup):
    importer, source = setup
    bad = source / 'bad.svg'
    bad.write_text('<svg/>')
    with pytest.raises(PlatformError, match='不支持的文件'):
        importer.create(dict(path=str(source), kind='test'))
    bad.unlink()
    picture(source / 'A/a.png')
    picture(source / 'b.png', 2)
    with pytest.raises(PlatformError, match='不能混合'):
        importer.create(dict(path=str(source), kind='test'))


def test_default_relative_dataset_root(tmp_path, monkeypatch):
    monkeypatch.delenv('FS_PLATFORM_IMPORT_ROOTS', raising=False)
    monkeypatch.delenv('FS_PLATFORM_DATASETS', raising=False)
    monkeypatch.delenv('FS_PLATFORM_RESOURCES', raising=False)
    picture(tmp_path / 'resources/datasets/Test/a.png')
    plan = DatasetImporter(tmp_path).create(dict(path='resources/datasets/Test', kind='test'))
    assert plan['total'] == 1
