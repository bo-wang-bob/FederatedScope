"""后门研究任务: 挑选测试图片, 渲染 clean / triggered / defense 三连对比图。

与训练任务完全隔离: 不使用 PlatformService 的 job 目录与单任务锁, 只借用
同样的"spec.json + 子进程 + job.json 轮询"模式。绘图只做前向推理, 不训练。

目录约定:
    base                   实验基目录 (默认 {repo}/exp/sabre)
      <attack_run>/        无防御实验, 含 *_final_mlp_head.pt 与 trigger
      <defense_run>/       有防御实验
      testset_images/       测试集导出图 + index.csv (供前端浏览挑图)
"""
from __future__ import annotations

import copy
import csv
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import threading
import uuid

from .platform_config import PlatformError, sha256
from .repository import JsonRepository
from .paths import env_path, project_path

TERMINAL = {'completed', 'failed', 'stopped', 'interrupted'}
MAX_IDS = 20
# 域名本身可含下划线 (Office-Home 的 Real_World), 编号固定 5 位
ID_PATTERN = re.compile(r'^[A-Za-z][A-Za-z0-9_]*_\d{5}$')
IMAGE_NAMES = {'clean', 'triggered', 'defense', 'defenseClean'}


def now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (FileNotFoundError, ValueError):
        return default


class BackdoorService:
    def __init__(self, repo, state):
        self.repo = Path(repo).resolve()
        self.root = project_path(state, self.repo) / 'backdoor'
        resources = env_path('FS_PLATFORM_RESOURCES', 'resources', self.repo)
        self.base = env_path('FS_BACKDOOR_BASE', resources / 'backdoor/exp/sabre', self.repo)
        privacy = env_path('FS_FEDMIA_LOCAL_ROOT', resources / 'fedmia_local', self.repo)
        self.data_root = str(env_path('FS_BACKDOOR_DATA_ROOT', privacy / 'datasets/OfficeHomeDataset_10072016', self.repo))
        self.device = os.environ.get('FS_BACKDOOR_DEVICE', 'cuda')
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.processes = {}
        self._index = None
        self._class_names = None
        self._class_names_loaded = False
        self.recover()

    # ------------------------------------------------------------------ #
    # 实验目录发现
    # ------------------------------------------------------------------ #
    def runs(self):
        """attack / defense 两个 run 目录名。可用 FS_BACKDOOR_RUNS 覆盖。"""
        override = os.environ.get('FS_BACKDOOR_RUNS', '')
        if override:
            pairs = dict(item.split('=', 1) for item in override.split(',') if '=' in item)
            if 'attack' in pairs:
                for name in pairs.values():
                    if not re.fullmatch(r'[A-Za-z0-9_-]+', name) or (self.base / name).resolve().parent != self.base:
                        raise PlatformError('后门实验目录名称非法')
                return {'attack': pairs['attack'], 'defense': pairs.get('defense')}
        candidates = []
        if self.base.is_dir():
            for entry in sorted(self.base.iterdir()):
                if entry.is_dir() and any(entry.glob('*_final_mlp_head.pt')):
                    candidates.append(entry.name)
        if not candidates:
            return {'attack': None, 'defense': None}
        seed = self._preferred_seed(candidates)
        group = [name for name in candidates if name.endswith('_' + seed)] if seed else candidates
        if not group:
            group = candidates
        attack = next((name for name in group if 'defense' not in name.lower()), None)
        defense = next((name for name in group if 'defense' in name.lower()), None)
        return {'attack': attack, 'defense': defense}

    @staticmethod
    def _preferred_seed(names):
        seeds = set()
        for name in names:
            match = re.search(r'_(\d+)$', name)
            if match:
                seeds.add(match.group(1))
        if not seeds:
            return ''
        return '42' if '42' in seeds else min(seeds, key=int)

    # ------------------------------------------------------------------ #
    # 测试集索引
    # ------------------------------------------------------------------ #
    def _load_index(self):
        """[(id, label), ...]; 测试集图片目录不存在时返回 None。"""
        if self._index is not None:
            return self._index
        index_file = self.base / 'testset_images' / 'index.csv'
        if not index_file.is_file():
            return None
        rows = []
        with index_file.open(encoding='utf-8-sig', newline='') as stream:
            for row in csv.DictReader(stream):
                image_id = (row.get('id') or '').strip()
                if ID_PATTERN.match(image_id):
                    rows.append((image_id, int(row['label'])))
        self._index = rows
        return rows

    def class_names(self):
        """类别名。只依赖 numpy/PIL, 不加载 torch。"""
        if self._class_names_loaded:
            return self._class_names or []
        self._class_names_loaded = True
        try:
            import yaml
            data_type = ''
            for config in sorted(self.base.glob('*/config.yaml')):
                with config.open(encoding='utf-8') as stream:
                    data_type = str((yaml.safe_load(stream) or {}).get('data', {}).get('type', '')).lower()
                if data_type:
                    break
            if 'office' in data_type and 'home' in data_type:
                from federatedscope.cv.dataset.office_home import OfficeHome
                self._class_names = list(OfficeHome.CLASSES)
            elif 'pacs' in data_type:
                from federatedscope.cv.dataset.pacs import PACS
                self._class_names = list(PACS.CLASSES)
        except Exception:
            self._class_names = None
        return self._class_names or []

    def testset(self):
        rows = self._load_index()
        names = self.class_names()
        runs = self.runs()
        if rows is None:
            return dict(exported=False, total=0, classNames=names, domains=[], labels=[], runs=runs,
                        base=str(self.base), message=f'未找到测试集导出: {self.base / "testset_images"}')
        domains, labels = {}, {}
        for image_id, label in rows:
            domain = image_id.rsplit('_', 1)[0]
            domains[domain] = domains.get(domain, 0) + 1
            labels[label] = labels.get(label, 0) + 1
        return dict(exported=True, total=len(rows), classNames=names,
                    domains=[dict(name=name, count=count) for name, count in sorted(domains.items())],
                    labels=[dict(index=index, name=names[index] if index < len(names) else str(index),
                                 count=count) for index, count in sorted(labels.items())],
                    runs=runs, base=str(self.base), maxIds=MAX_IDS)

    def image_path(self, image_id):
        if not ID_PATTERN.match(image_id):
            raise PlatformError('图片编号非法', 400)
        file = (self.base / 'testset_images' / (image_id + '.jpg')).resolve()
        root = (self.base / 'testset_images').resolve()
        if file.parent != root or not file.is_file():
            raise PlatformError('测试图片不存在', 404)
        return file

    def pick(self, payload):
        """随机挑选图片编号, 支持按域/类别过滤。"""
        if not isinstance(payload, dict):
            raise PlatformError('请求必须是 JSON 对象')
        rows = self._load_index()
        if rows is None:
            raise PlatformError('测试集尚未导出, 请先运行一次绘图脚本以生成测试集图片', 409)
        count = payload.get('count', MAX_IDS)
        if not isinstance(count, int) or isinstance(count, bool):
            raise PlatformError('图片数量非法')
        if not 1 <= count <= MAX_IDS:
            raise PlatformError(f'一次最多选择 {MAX_IDS} 张图片')
        domain = payload.get('domain')
        label = payload.get('label')
        pool = rows
        if domain:
            if not isinstance(domain, str):
                raise PlatformError('测试域非法')
            pool = [row for row in pool if row[0].rsplit('_', 1)[0] == domain]
        if label is not None:
            if not isinstance(label, int) or isinstance(label, bool):
                raise PlatformError('测试类别非法')
            pool = [row for row in pool if row[1] == label]
        if not pool:
            raise PlatformError('没有符合筛选条件的图片', 404)
        seed = payload.get('seed')
        if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
            raise PlatformError('随机种子非法')
        rng = random.Random(seed)
        picked = rng.sample(pool, min(count, len(pool)))
        return dict(ids=[row[0] for row in picked], labels=[row[1] for row in picked],
                    total=len(pool), seed=seed, count=len(picked))

    # ------------------------------------------------------------------ #
    # 任务
    # ------------------------------------------------------------------ #
    def directory(self, job_id):
        if not re.fullmatch(r'[a-f0-9]{32}', job_id):
            raise PlatformError('任务不存在', 404)
        return self.root / job_id

    def get(self, job_id):
        job = read_json(self.directory(job_id) / 'job.json')
        if job is None:
            raise PlatformError('任务不存在', 404)
        return job

    def list(self):
        jobs = [read_json(p) for p in sorted((self.root).glob('*/job.json'), reverse=True)]
        jobs = [j for j in jobs if j]
        jobs.sort(key=lambda j: j['createdAt'], reverse=True)
        return [{k: v for k, v in j.items() if k != 'result'} for j in jobs]

    def _save(self, job):
        job['updatedAt'] = now()
        JsonRepository._atomic_write(self.directory(job['id']) / 'job.json', job)

    def _validate_ids(self, ids):
        if not isinstance(ids, list) or not ids:
            raise PlatformError('请至少选择一张图片')
        if len(ids) > MAX_IDS:
            raise PlatformError(f'一次最多选择 {MAX_IDS} 张图片')
        for image_id in ids:
            if not isinstance(image_id, str) or not ID_PATTERN.match(image_id):
                raise PlatformError(f'图片编号非法: {image_id}')
        known = {row[0] for row in (self._load_index() or [])}
        if not known:
            raise PlatformError('测试集索引未就绪', 409)
        missing = [i for i in ids if i not in known]
        if missing:
            raise PlatformError(f'测试集中不存在: {", ".join(missing[:5])}', 404)
        if len(set(ids)) != len(ids):
            raise PlatformError('图片编号不能重复')
        for image_id in ids:
            self.image_path(image_id)
        return list(ids)

    def create(self, payload):
        if not isinstance(payload, dict):
            raise PlatformError('请求必须是 JSON 对象')
        with self.lock:
            key = payload.get('idempotencyKey')
            if not isinstance(key, str) or not re.fullmatch(r'[A-Za-z0-9_-]{8,100}', key):
                raise PlatformError('需要有效的 idempotencyKey')
            for job in self.list():
                if job['idempotencyKey'] == key:
                    if job['submitted'] != payload:
                        raise PlatformError('幂等键已用于另一请求', 409)
                    return self.get(job['id'])
            for job in self.list():
                if job['status'] not in TERMINAL:
                    raise PlatformError(f'请先等待后门对比任务 {job["id"][:8]} 结束', 409)
            runs = self.runs()
            if not runs.get('attack'):
                raise PlatformError(f'未找到后门攻击实验目录: {self.base}', 404)
            ids = self._validate_ids(payload.get('ids'))
            font_size = payload.get('fontSize', 28)
            if not isinstance(font_size, int) or isinstance(font_size, bool) or not 12 <= font_size <= 48:
                raise PlatformError('字号必须为 12–48 的整数')
            if not Path(self.data_root).is_dir():
                raise PlatformError('后门研究共享的 OfficeHome 数据集不存在', 409)
            name = str(payload.get('name', '') or '')[:120]
            job_id = uuid.uuid4().hex
            output = self.directory(job_id)
            output.mkdir(parents=True)
            script = self.repo / 'scripts' / 'backdoor' / 'run_group.py'
            if not script.is_file():
                raise PlatformError(f'缺少绘图脚本: {script}', 500)
            spec = dict(base=str(self.base), runs={k: v for k, v in runs.items() if v},
                        ids=ids, output=str(output), device=self.device,
                        dataRoot=self.data_root, fontSize=font_size)
            JsonRepository._atomic_write(output / 'spec.json', spec)
            job = dict(id=job_id, action='backdoor', ids=ids, name=name,
                       runs=spec['runs'], base=str(self.base), device=self.device,
                       idempotencyKey=key, submitted=payload, createdAt=now(), updatedAt=now(),
                       status='queued', stage='等待启动', error=None, progress=0)
            self._save(job)
            threading.Thread(target=self._run, args=(job_id,), daemon=True).start()
            return copy.deepcopy(job)

    def _run(self, job_id):
        output = self.directory(job_id)
        try:
            with self.lock:
                job = self.get(job_id)
                if job['status'] != 'queued':
                    return
                env = {**os.environ, 'PYTHONUNBUFFERED': '1', 'OMP_NUM_THREADS': '2',
                       'MPLCONFIGDIR': str(output / 'mpl')}
                with (output / 'runner.log').open('wb') as log:
                    proc = subprocess.Popen(
                        [sys.executable, str(self.repo / 'scripts' / 'backdoor' / 'run_group.py'),
                         '--spec', str(output / 'spec.json')],
                        cwd=str(self.repo), env=env, stdout=log, stderr=subprocess.STDOUT,
                        start_new_session=os.name == 'posix')
                self.processes[job_id] = proc
                job.update(status='running', startedAt=now(), stage='加载模型与测试集')
                self._save(job)
            returncode = proc.wait()
            with self.lock:
                job = self.get(job_id)
                result = read_json(output / 'result.json')
                job['exitCode'] = returncode
                if job['status'] in {'stopping', 'stopped'}:
                    job.update(status='stopped', stage='已停止')
                elif returncode == 0 and result:
                    paths = result.get('paths', {})
                    job.update(status='completed', stage='已完成', result=result,
                               images={name: f'/api/platform/backdoor/jobs/{job_id}/image/{name}'
                                       for name, file in paths.items()
                                       if (output / file).is_file()})
                else:
                    job.update(status='failed', stage='生成失败',
                               error=self._failure(output) or f'进程退出码 {returncode}，请查看日志')
                job['endedAt'] = now()
                self._save(job)
        except Exception as error:
            with self.lock:
                job = self.get(job_id)
                job.update(status='failed', stage='启动或监控失败', error=str(error), endedAt=now())
                self._save(job)
        finally:
            with self.lock:
                self.processes.pop(job_id, None)

    @staticmethod
    def _failure(output):
        log = output / 'runner.log'
        if not log.is_file():
            return None
        lines = [line for line in log.read_text(encoding='utf-8', errors='replace').splitlines() if line.strip()]
        for line in reversed(lines[-40:]):
            if line.startswith('[ERROR]') or 'Error' in line or 'Traceback' in line:
                return line.strip()[:400]
        return lines[-1].strip()[:400] if lines else None

    def stop(self, job_id):
        with self.lock:
            job = self.get(job_id)
            if job['status'] in TERMINAL:
                return job
            job.update(status='stopping', stage='正在停止')
            self._save(job)
            proc = self.processes.get(job_id)
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
        with self.lock:
            job = self.get(job_id)
            if job['status'] not in TERMINAL:
                job.update(status='stopped', stage='已停止', endedAt=now())
                self._save(job)
            return job

    def artifact(self, job_id, name):
        if name not in IMAGE_NAMES:
            raise PlatformError('结果图不存在', 404)
        job = self.get(job_id)
        paths = (job.get('result') or {}).get('paths', {})
        file = paths.get(name)
        if not file:
            raise PlatformError('结果图尚未生成', 404)
        path = (self.directory(job_id) / file).resolve()
        if path.parent != self.directory(job_id).resolve() or not path.is_file():
            raise PlatformError('结果图不存在', 404)
        return path

    def logs(self, job_id):
        self.get(job_id)
        path = self.directory(job_id) / 'runner.log'
        if not path.is_file():
            return ''
        with path.open('rb') as stream:
            stream.seek(max(0, path.stat().st_size - 128 * 1024))
            return re.sub(r'\x1b\[[0-9;]*m', '', stream.read().decode('utf-8', errors='replace'))

    def recover(self):
        for path in sorted(self.root.glob('*/job.json')):
            job = read_json(path)
            if job and job.get('status') not in TERMINAL:
                job.update(status='interrupted', stage='服务重启，任务已中断',
                           error='未自动续跑，请重新生成', endedAt=now())
                JsonRepository._atomic_write(path, job)

    def close(self):
        for job in self.list():
            if job['status'] not in TERMINAL:
                self.stop(job['id'])
