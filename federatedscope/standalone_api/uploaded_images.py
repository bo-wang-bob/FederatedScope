"""Decode new uploads once; training and browsers use the same RGB PNG pixels."""
import io
import warnings

from PIL import Image, ImageOps, UnidentifiedImageError

from .platform_config import PlatformError

IMAGE_TYPES = {'.jpg', '.jpeg', '.jpe', '.jfif', '.png', '.apng', '.webp',
               '.bmp', '.dib', '.tif', '.tiff', '.gif', '.avif', '.heic', '.heif',
               '.jp2', '.j2k', '.ppm', '.pgm', '.pbm', '.pnm', '.tga'}
FORMATS = {'JPEG', 'PNG', 'WEBP', 'BMP', 'DIB', 'TIFF', 'GIF', 'AVIF',
           'HEIF', 'JPEG2000', 'PPM', 'TGA'}
MAX_IMAGE = 25 * 1024 * 1024
MAX_PIXELS = 40_000_000


def normalize_image(content, suffix):
    if suffix not in IMAGE_TYPES:
        raise PlatformError('不支持此图片格式，请转换为 PNG 或 JPG')
    if not 0 < len(content) <= MAX_IMAGE:
        raise PlatformError('单张图片不能超过 25 MiB，且不能为空')
    if suffix in {'.heic', '.heif'}:
        try:
            from pillow_heif import register_heif_opener
        except ImportError as error:
            raise PlatformError('服务器缺少 HEIC/HEIF 解码组件 pillow-heif，请安装部署依赖') from error
        register_heif_opener(thumbnails=False)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(content)) as image:
                if image.format not in FORMATS:
                    raise PlatformError('文件内容不是支持的图片格式')
                source_format = image.format
                image.seek(0)
                if image.width * image.height > MAX_PIXELS:
                    raise PlatformError('图片像素过大，最多 4000 万像素')
                # load() also rejects truncated image data; verify() alone does not.
                image.load()
                oriented = ImageOps.exif_transpose(image)
                if oriented.mode in {'RGBA', 'LA'} or 'transparency' in oriented.info:
                    rgba = oriented.convert('RGBA')
                    rgb = Image.new('RGB', rgba.size, 'white')
                    rgb.paste(rgba, mask=rgba.getchannel('A'))
                else:
                    rgb = oriented.convert('RGB')
                # Remove metadata after orientation correction for stable pixels/hash.
                rgb.info.clear()
                output = io.BytesIO()
                rgb.save(output, format='PNG')
                normalized = output.getvalue()
                if len(normalized) > MAX_IMAGE:
                    raise PlatformError('图片解码后超过 25 MiB，请缩小图片尺寸')
                return normalized, dict(sourceFormat=source_format, frame=0,
                    width=rgb.width, height=rgb.height, mode='RGB',
                    normalization='exif-first-frame-white-alpha-rgb-png-v1')
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError, EOFError,
            Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise PlatformError('图片损坏或服务器无法解码，请重新保存为 PNG 或 JPG') from error
