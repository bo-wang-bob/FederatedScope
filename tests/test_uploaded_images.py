import hashlib
import io

import pytest
from PIL import Image
from pillow_heif import register_heif_opener

from federatedscope.standalone_api.uploaded_datasets import DatasetStore
from federatedscope.standalone_api.uploaded_images import IMAGE_TYPES, normalize_image
from federatedscope.standalone_api.platform_config import PlatformError

register_heif_opener(thumbnails=False)
FORMATS = [('.jpg', 'JPEG'), ('.jpeg', 'JPEG'), ('.jpe', 'JPEG'), ('.jfif', 'JPEG'),
    ('.png', 'PNG'), ('.apng', 'PNG'), ('.webp', 'WEBP'), ('.bmp', 'BMP'), ('.dib', 'DIB'),
    ('.tif', 'TIFF'), ('.tiff', 'TIFF'), ('.gif', 'GIF'), ('.avif', 'AVIF'),
    ('.heic', 'HEIF'), ('.heif', 'HEIF'), ('.jp2', 'JPEG2000'), ('.j2k', 'JPEG2000'),
    ('.ppm', 'PPM'), ('.pgm', 'PPM'), ('.pbm', 'PPM'), ('.pnm', 'PPM'), ('.tga', 'TGA')]


def encoded(fmt, color=(123, 55, 21), mode='RGB', **options):
    output = io.BytesIO()
    Image.new('RGB', (32, 24), color).convert(mode).save(output, format=fmt, **options)
    return output.getvalue()


@pytest.mark.parametrize('kind', ['train', 'test'])
@pytest.mark.parametrize('suffix,fmt', FORMATS)
def test_every_advertised_format_uploads_as_decoded_rgb_png(tmp_path, monkeypatch, suffix, fmt, kind):
    monkeypatch.setenv('FS_PLATFORM_UPLOADS', str(tmp_path / 'uploads'))
    store = DatasetStore(tmp_path)
    identifier = store.create(dict(name='格式验收', kind=kind))['id']
    mode = '1' if suffix == '.pbm' else 'L' if suffix == '.pgm' else 'RGB'
    raw = encoded(fmt, mode=mode)
    relative = ('类别/' if kind == 'train' else '') + 'photo' + suffix.upper()
    store.put(identifier, relative, raw)
    # Retry is idempotent even though source and normalized bytes differ.
    store.put(identifier, relative, raw)
    value = store.get(identifier, ready=False)
    assert value['count'] == 1
    row = value['items'][0]
    assert row['originalPath'] == relative and row['path'] == relative + '.png'
    assert row['originalSha256'] == hashlib.sha256(raw).hexdigest()
    assert (store.directory(identifier) / 'originals' / relative).read_bytes() == raw
    file = store.directory(identifier) / 'images' / ('uploaded' if kind == 'train' else 'test') / row['path']
    assert hashlib.sha256(file.read_bytes()).hexdigest() == row['sha256']
    with Image.open(file) as image:
        image.load()
        assert image.format == 'PNG' and image.mode == 'RGB' and image.size == (32, 24)
    if kind == 'test':
        assert store.finish(identifier)['classes'] == []
        assert store.image(identifier, 0) == file.resolve()


def test_format_coverage_is_complete():
    assert {suffix for suffix, _ in FORMATS} == IMAGE_TYPES


def test_exif_orientation_is_applied():
    exif = Image.Exif()
    exif[274] = 6
    raw = encoded('JPEG', exif=exif)
    content, metadata = normalize_image(raw, '.jpg')
    with Image.open(io.BytesIO(content)) as image:
        assert image.size == (24, 32) and image.getexif().get(274, 1) == 1
    assert metadata['width'] == 24 and metadata['height'] == 32


@pytest.mark.parametrize('fmt,suffix', [('GIF', '.gif'), ('TIFF', '.tiff'), ('WEBP', '.webp'), ('PNG', '.apng')])
def test_multiframe_upload_uses_only_first_frame(fmt, suffix):
    output = io.BytesIO()
    first = Image.new('RGB', (24, 24), 'red')
    first.save(output, format=fmt, save_all=True, append_images=[Image.new('RGB', (24, 24), 'blue')])
    content, metadata = normalize_image(output.getvalue(), suffix)
    with Image.open(io.BytesIO(content)) as image:
        assert getattr(image, 'n_frames', 1) == 1
        assert image.getpixel((0, 0))[0] > 200 and image.getpixel((0, 0))[2] < 30
    assert metadata['frame'] == 0


def test_transparency_uses_white_background():
    content, _ = normalize_image(transparent_png(), '.png')
    with Image.open(io.BytesIO(content)) as image:
        assert image.mode == 'RGB' and image.getpixel((0, 0)) == (255, 255, 255)


def transparent_png():
    stream = io.BytesIO()
    Image.new('RGBA', (8, 8), (255, 0, 0, 0)).save(stream, format='PNG')
    return stream.getvalue()


def test_truncated_and_nonimage_files_are_rejected():
    with pytest.raises(PlatformError):
        normalize_image(encoded('JPEG')[:-25], '.jpg')
    with pytest.raises(PlatformError):
        normalize_image(b'<svg></svg>', '.svg')
    with pytest.raises(PlatformError):
        normalize_image(b'not an image', '.jpg')


def test_legacy_uploads_remain_readable(tmp_path, monkeypatch):
    import json
    monkeypatch.setenv('FS_PLATFORM_UPLOADS', str(tmp_path / 'uploads'))
    store = DatasetStore(tmp_path)
    identifier = store.create(dict(name='旧数据', kind='test'))['id']
    directory = store.directory(identifier)
    raw = encoded('JPEG')
    file = directory / 'images/test/legacy.jpg'
    file.parent.mkdir(parents=True)
    file.write_bytes(raw)
    value = store.get(identifier, ready=False)
    value.update(status='ready', count=1, bytes=len(raw), items=[dict(path='legacy.jpg', sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))])
    (directory / 'dataset.json').write_text(json.dumps(value), encoding='utf-8')
    assert store.image(identifier, 0).read_bytes() == raw


def test_failed_metadata_write_rolls_back_only_new_files(tmp_path, monkeypatch):
    from unittest.mock import patch
    from federatedscope.standalone_api.repository import JsonRepository
    monkeypatch.setenv('FS_PLATFORM_UPLOADS', str(tmp_path / 'uploads'))
    store = DatasetStore(tmp_path)
    identifier = store.create(dict(name='磁盘失败恢复', kind='test'))['id']
    store.put(identifier, 'old.jpg', encoded('JPEG'))
    with patch.object(JsonRepository, '_atomic_write', side_effect=OSError('disk full')):
        with pytest.raises(PlatformError, match='保存图片失败'):
            store.put(identifier, 'new.tif', encoded('TIFF', color=(10, 20, 30)))
    assert store.get(identifier, ready=False)['count'] == 1
    assert not (store.directory(identifier) / 'images/test/new.tif.png').exists()
    assert not (store.directory(identifier) / 'originals/new.tif').exists()
    assert (store.directory(identifier) / 'originals/old.jpg').is_file()
    store.put(identifier, 'new.tif', encoded('TIFF', color=(10, 20, 30)))
    assert store.finish(identifier)['count'] == 2
