"""Bounded server-side image import; source files are never modified."""
import json
import os
from pathlib import Path
import re
import stat

from .paths import env_path
from .platform_config import PlatformError
from .repository import JsonRepository
from .uploaded_datasets import DatasetStore, UPLOAD_LOCK, MAX_BYTES
from .uploaded_images import IMAGE_TYPES, MAX_IMAGE

PLAN = '.server-import.json'
SIDECAR = re.compile(r'(?i)^(thumbs\.db|desktop\.ini)$|\.(txt|csv|json|xml|yaml|yml|md)$')


class DatasetImporter:
    def __init__(self, repo):
        self.store = DatasetStore(repo)
        self.repo = self.store.repo

    def roots(self):
        resources = env_path('FS_PLATFORM_RESOURCES', self.repo / 'resources', self.repo)
        roots = [env_path('FS_PLATFORM_DATASETS', resources / 'datasets', self.repo)]
        extra = os.environ.get('FS_PLATFORM_IMPORT_ROOTS', '')
        if extra:
            try:
                values = json.loads(extra)
                if not isinstance(values, list) or any(not isinstance(p, str) or not p for p in values):
                    raise ValueError()
                for value in values:
                    path = Path(value).expanduser()
                    roots.append((path if path.is_absolute() else self.repo / path).resolve())
            except (ValueError, TypeError):
                raise PlatformError('服务器 FS_PLATFORM_IMPORT_ROOTS 必须为目录字符串的 JSON 数组', 500)
        return roots

    def checked(self, value):
        if not isinstance(value, str) or not value.strip() or len(value) > 4096 or '\x00' in value:
            raise PlatformError('请输入服务器上的文件夹或图片路径')
        path = Path(value.strip()).expanduser()
        if '..' in path.parts:
            raise PlatformError('服务器路径不允许包含 ..', 403)
        path = path if path.is_absolute() else self.repo / path
        # Reject links/junctions instead of following them out of an approved root.
        for part in (path, *path.parents):
            if part.is_symlink() or getattr(part, 'is_junction', lambda: False)():
                raise PlatformError('导入路径不能包含符号链接或目录联接', 403)
            if os.name == 'nt' and part.exists() and getattr(part.lstat(), 'st_file_attributes', 0) & 0x400:
                raise PlatformError('导入路径不能包含重解析点', 403)
        resolved = path.resolve()
        if not any(resolved == root or root in resolved.parents for root in self.roots()):
            raise PlatformError('路径不在允许的数据目录内；请由管理员配置 FS_PLATFORM_IMPORT_ROOTS', 403)
        if not resolved.exists():
            raise PlatformError('服务器路径不存在', 404)
        return resolved

    @staticmethod
    def signature(info):
        return [info.st_size, info.st_mtime_ns, info.st_dev, info.st_ino]

    def create(self, body):
        if not isinstance(body, dict) or set(body) - {'name', 'kind', 'path', 'single'}:
            raise PlatformError('路径导入参数非法')
        kind, single = body.get('kind'), body.get('single', False)
        if kind not in ('train', 'test') or not isinstance(single, bool) or (single and kind != 'test'):
            raise PlatformError('请选择训练集、测试集或单张测试图片')
        try:
            source = self.checked(body.get('path'))
            if (single and not source.is_file()) or (not single and not source.is_dir()):
                raise PlatformError('请选择单张图片路径' if single else '请输入数据集文件夹路径')
            base = source.parent if single else source
            files, total = [], 0

            def add(path):
                nonlocal total
                path = self.checked(str(path))
                relative = path.relative_to(base).as_posix()
                if path.suffix.lower() not in IMAGE_TYPES:
                    if SIDECAR.search(path.name):
                        return
                    raise PlatformError('不支持的文件：' + relative + '，请转换为 PNG 或 JPG')
                info = path.stat()
                if not stat.S_ISREG(info.st_mode):
                    raise PlatformError('只能导入普通图片文件')
                if not 0 < info.st_size <= MAX_IMAGE:
                    raise PlatformError('图片为空或超过 25 MiB：' + relative)
                total += info.st_size
                files.append(dict(path=relative, signature=self.signature(info)))
                if len(files) > 50000 or total > MAX_BYTES:
                    raise PlatformError('每个数据集最多 50000 张图片、20 GiB')

            if single:
                add(source)
            else:
                with os.scandir(source) as entries:
                    for entry in entries:
                        if entry.name.startswith('.'):
                            continue
                        path = self.checked(entry.path)
                        if path.is_dir():
                            with os.scandir(path) as children:
                                for child in children:
                                    if child.name.startswith('.'):
                                        continue
                                    if self.checked(child.path).is_dir():
                                        raise PlatformError('目录仅支持 类别/图片 或直接放置测试图片')
                                    add(Path(child.path))
                        else:
                            add(path)
            if not files:
                raise PlatformError('目录中没有支持的图片')
            depths = {len(row['path'].split('/')) for row in files}
            if (kind == 'train' and depths != {2}) or (kind == 'test' and len(depths) != 1):
                raise PlatformError('训练集需 类别/图片；测试集不能混合带类别和无标签图片')
            classes = sorted({r['path'].split('/')[0] for r in files}) if depths == {2} else []
            if kind == 'train' and (len(classes) < 2 or len({c.casefold() for c in classes}) != len(classes)
                    or any(sum(r['path'].split('/')[0] == c for r in files) < 3 for c in classes)):
                raise PlatformError('训练集至少两个类别，每类至少三张图片，类别名不能仅大小写不同')
            # Only create an upload after validating the complete inventory.
            value = self.store.create(dict(name=body.get('name') or source.name, kind=kind))
            JsonRepository._atomic_write(self.store.directory(value['id']) / PLAN,
                dict(base=str(base), files=sorted(files, key=lambda row: row['path'])))
            return dict(id=value['id'], total=len(files), count=0, classes=classes)
        except OSError as error:
            raise PlatformError('无法读取服务器数据目录，请检查权限及文件状态') from error

    def next(self, identifier):
        # A retry advances from the persisted count. One image per request keeps
        # large folders out of the browser's overall 90-second request timeout.
        with UPLOAD_LOCK:
            value = self.store.get(identifier, False)
            if value['status'] == 'ready':
                return dict(count=value['count'], total=value['count'])
            file = self.store.directory(identifier) / PLAN
            if not file.is_file():
                raise PlatformError('路径导入记录不存在', 404)
            plan = json.loads(file.read_text(encoding='utf-8'))
            count, total = value['count'], len(plan['files'])
            if count >= total:
                return dict(count=count, total=total)
            row = plan['files'][count]
            try:
                path = self.checked(str(Path(plan['base']) / row['path']))
                info = path.stat()
                if not stat.S_ISREG(info.st_mode) or self.signature(info) != row['signature']:
                    raise PlatformError('源图片已变化，请重新导入：' + row['path'], 409)
                flags = os.O_RDONLY | getattr(os, 'O_BINARY', 0) | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0)
                with os.fdopen(os.open(path, flags), 'rb') as stream:
                    if self.signature(os.fstat(stream.fileno())) != row['signature']:
                        raise PlatformError('源图片已变化，请重新导入', 409)
                    content = stream.read(MAX_IMAGE + 1)
                    if self.signature(os.fstat(stream.fileno())) != row['signature']:
                        raise PlatformError('源图片读取期间发生变化，请重新导入', 409)
                self.checked(str(path))
                self.store.put(identifier, row['path'], content, server_import=True)
                return dict(count=count + 1, total=total)
            except OSError as error:
                raise PlatformError('读取服务器图片失败，请检查权限后重试') from error
