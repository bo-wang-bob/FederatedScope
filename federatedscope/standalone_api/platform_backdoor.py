"""后门研究任务: 挑选测试图片, 渲染 clean / triggered / defense 三连对比图。

与训练任务完全隔离: 不使用 PlatformService 的 job 目录与单任务锁, 只借用
同样的"spec.json + 子进程 + job.json 轮询"模式。绘图只做前向推理, 不训练。

目录约定:
    base                   实验基目录 (默认 {repo}/exp/sabre_newdataset, 缺失回退 exp/sabre)
      <attack_run>/        无防御实验, 含 *_final_mlp_head.pt 与 trigger
      <defense_run>/       有防御实验
      testset_images/       测试集导出图 + index.csv (供前端浏览挑图)
      testset_images/trigger_predictions.json
                           全测试集"带触发器"预测缓存, 由 scripts/backdoor/
                           precompute_trigger_predictions.py 生成; 存在时挑图
                           优先选"攻击命中 ∧ 防御未命中"的样本
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
import time
import uuid

from .platform_config import PlatformError, sha256
from .repository import JsonRepository

TERMINAL = {'completed', 'failed', 'stopped', 'interrupted'}
MAX_IDS = 20
# 域名本身可含下划线 (Office-Home 的 Real_World), 编号固定 5 位
ID_PATTERN = re.compile(r'^[A-Za-z][A-Za-z0-9_]*_\d{5}$')
IMAGE_NAMES = {'clean', 'triggered', 'defense', 'defenseClean'}
# 实验目录候选名, 按优先级排列: 新数据集 (军机三域) 结果优先展示
BASE_CANDIDATES = ('sabre_newdataset', 'sabre')
# 全测试集触发后预测缓存文件名 (与 scripts/backdoor/precompute_trigger_predictions.py 一致)
PREDICTIONS_CACHE = 'trigger_predictions.json'
# 挑图时"攻击命中 ∧ 防御拦住"样本的目标占比; 其余名额从剩余池随机取。
# 不用 1.0 是为了避免挑出来的全是完美样本 —— 混一部分普通样本, 结果好看但可信。
PREFERRED_RATIO = 0.7


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
        state_root = (Path(state).resolve() if state
                      else self.repo / 'exp' / 'single_host_platform')
        self.root = state_root / 'backdoor'
        self.base = Path(os.environ.get('FS_BACKDOOR_BASE')
                         or self._default_base(self.repo, state_root)).resolve()
        self.data_root = os.environ.get('FS_BACKDOOR_DATA_ROOT') or None
        self.device = os.environ.get('FS_BACKDOOR_DEVICE', 'cuda')
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.processes = {}
        self._index = None
        self._class_names = None
        self._class_names_loaded = False
        self._predictions = None
        self._predictions_loaded = False
        self._precompute_started = False
        self.recover()

    # ------------------------------------------------------------------ #
    # 默认实验目录
    # ------------------------------------------------------------------ #
    @staticmethod
    def _exp_dirs(repo, state_root):
        """候选 exp/ 目录, 按优先级排列。

        前后端常常分处两棵工作树 (后端仓库里根本没有 exp/), 所以不能只认
        repo/exp: 还要看状态目录的同级目录 (前端工作区 exp/platform -> exp/)、
        FS_BACKDOOR_SEARCH_ROOTS 以及进程工作目录。
        """
        candidates = [Path(repo) / 'exp', Path(state_root).parent, Path.cwd() / 'exp']
        extra = os.environ.get('FS_BACKDOOR_SEARCH_ROOTS', '')
        for item in extra.split(os.pathsep):
            if item.strip():
                candidates.append(Path(item.strip()).expanduser() / 'exp')
        seen, ordered = set(), []
        for item in candidates:
            key = str(item).lower()
            if key not in seen:
                seen.add(key)
                ordered.append(item)
        return ordered

    @classmethod
    def _default_base(cls, repo, state_root):
        """优先返回新数据集实验结果目录; 全部缺失时回退第一个候选名。"""
        exp_dirs = cls._exp_dirs(repo, state_root)
        for name in BASE_CANDIDATES:              # 候选名优先级高于目录位置
            for exp_dir in exp_dirs:
                candidate = (exp_dir / name).resolve()
                if candidate.is_dir():
                    return candidate
        return (exp_dirs[0] / BASE_CANDIDATES[0]).resolve()

    # ------------------------------------------------------------------ #
    # 实验目录发现
    # ------------------------------------------------------------------ #
    def runs(self):
        """attack / defense 两个 run 目录名。可用 FS_BACKDOOR_RUNS 覆盖。"""
        override = os.environ.get('FS_BACKDOOR_RUNS', '')
        if override:
            pairs = dict(item.split('=', 1) for item in override.split(',') if '=' in item)
            if 'attack' in pairs:
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
            data_root = ''
            for config in sorted(self.base.glob('*/config.yaml')):
                with config.open(encoding='utf-8') as stream:
                    loaded = yaml.safe_load(stream) or {}
                section = loaded.get('data') or {}
                data_type = str(section.get('type', '')).lower()
                data_root = str(section.get('root', '') or '')
                if data_type:
                    break
            # config.yaml 里的 data.root 多为相对路径, 换机器后失效; 显式覆盖优先
            root = self.data_root or data_root
            if 'office' in data_type and 'home' in data_type:
                from federatedscope.cv.dataset.office_home import OfficeHome
                self._class_names = list(OfficeHome.CLASSES)
            elif 'pacs' in data_type:
                from federatedscope.cv.dataset.pacs import PACS
                self._class_names = list(PACS.CLASSES)
            elif 'militaryaircraft' in data_type.replace(' ', '') and root:
                # 与 DomainNet 同为 root/domain/class 布局; 类别顺序必须来自同一发现逻辑
                from federatedscope.cv.dataset.domainnet import (
                    discover_domainnet_metadata)
                _, names = discover_domainnet_metadata(
                    str(root), ['aerial', 'natural', 'recon'], False)
                self._class_names = list(names)
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

    # ------------------------------------------------------------------ #
    # 触发后预测缓存 (挑图偏好)
    # ------------------------------------------------------------------ #
    def _load_predictions(self):
        """全测试集"带触发器"预测缓存; 未生成时返回 None。"""
        if self._predictions_loaded:
            return self._predictions
        self._predictions_loaded = True
        cache = read_json(self.base / 'testset_images' / PREDICTIONS_CACHE)
        if (isinstance(cache, dict) and isinstance(cache.get('items'), dict)
                and isinstance(cache.get('targetLabel'), int)):
            self._predictions = cache
        return self._predictions

    def _start_precompute(self):
        """缓存缺失时后台生成一次 (不阻塞挑图, 失败也不影响本次随机抽样)。"""
        with self.lock:
            if self._precompute_started:
                return
            self._precompute_started = True
        script = self.repo / 'scripts' / 'backdoor' / 'precompute_trigger_predictions.py'
        runs = self.runs()
        if not script.is_file() or not runs.get('attack'):
            return
        command = [sys.executable, str(script), '--base', str(self.base),
                   '--device', self.device, '--runs',
                   ','.join(f'{key}={value}' for key, value in runs.items() if value)]
        if self.data_root:
            command += ['--data-root', str(self.data_root)]
        log = self.root / 'precompute.log'

        def _worker():
            try:
                with log.open('ab') as stream:
                    stream.write(f'\n=== {now()} ===\n'.encode('utf-8'))
                    subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                   cwd=str(self.repo), timeout=3600)
            except Exception as exc:  # pragma: no cover - 后台线程, 失败只记录
                print(f'[backdoor] 预计算失败: {exc}')
            finally:
                with self.lock:
                    self._predictions_loaded = False
                    self._predictions = None

        threading.Thread(target=_worker, name='backdoor-precompute', daemon=True).start()

    @staticmethod
    def _rank(entry, label, target):
        """展示优先级: 攻击命中∧防御拦住 > 都命中 > 都没中 > 防御误判 > 无缓存。"""
        if entry is None:
            return 4
        attack_hit = entry.get('attack') == target and label != target
        defense_hit = entry.get('defense') == target and label != target
        if attack_hit:
            return 0 if not defense_hit else 1
        return 3 if defense_hit else 2

    def pick(self, payload):
        """挑选图片编号, 支持按域/类别过滤。

        默认按全测试集的"带触发器"预测缓存做**加权抽样**: 约 PREFERRED_RATIO
        的名额给"无防御命中目标类 ∧ 有防御未命中"的样本 (同时体现后门打得进、
        防御拦得住), 其余名额从剩余池随机取 —— 故意不取满, 免得挑出来的全是
        完美样本。缓存缺失时退化为随机抽样, 并在后台触发一次预计算。
        """
        rows = self._load_index()
        if rows is None:
            raise PlatformError('测试集尚未导出, 请先运行一次绘图脚本以生成测试集图片', 409)
        try:
            count = int(payload.get('count', MAX_IDS))
        except (TypeError, ValueError):
            raise PlatformError('图片数量非法') from None
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
        if seed is not None and not isinstance(seed, int):
            raise PlatformError('随机种子非法')
        rng = random.Random(seed)
        cache = self._load_predictions() if payload.get('prefer', True) is not False else None
        if cache is None:
            if payload.get('prefer', True) is not False:
                self._start_precompute()
            picked = rng.sample(pool, min(count, len(pool)))
            return dict(ids=[row[0] for row in picked], labels=[row[1] for row in picked],
                        total=len(pool), seed=seed, count=len(picked), filtered=False)
        target = cache['targetLabel']
        items = cache['items']
        preferred_pool = [row for row in pool
                          if self._rank(items.get(row[0]), row[1], target) == 0]
        rest = [row for row in pool
                if self._rank(items.get(row[0]), row[1], target) != 0]
        wanted = min(count, len(pool))
        ratio = PREFERRED_RATIO
        try:                                   # 允许调用方覆盖占比 (仅内部/调试用)
            override = payload.get('preferRatio')
            if override is not None:
                ratio = min(max(float(override), 0.0), 1.0)
        except (TypeError, ValueError):
            pass
        taken = min(len(preferred_pool), int(round(wanted * ratio)))
        # 只要还有别的样本, 就至少留一个名额给随机池, 避免"满分"展示
        if rest and wanted > 1 and taken >= wanted:
            taken = wanted - 1
        picked = rng.sample(preferred_pool, taken)
        picked += rng.sample(rest, min(wanted - taken, len(rest)))
        rng.shuffle(picked)                    # 打乱, 免得前排全是预筛样本
        return dict(ids=[row[0] for row in picked], labels=[row[1] for row in picked],
                    total=len(pool), seed=seed, count=len(picked),
                    filtered=len(preferred_pool) > 0, preferred=len(preferred_pool),
                    preferRatio=ratio, targetLabel=target,
                    targetName=cache.get('targetName'))

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
        if known:
            missing = [i for i in ids if i not in known]
            if missing:
                raise PlatformError(f'测试集中不存在: {", ".join(missing[:5])}', 404)
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
                    return job
            for job in self.list():
                if job['status'] not in TERMINAL:
                    raise PlatformError(f'请先等待后门对比任务 {job["id"][:8]} 结束', 409)
            runs = self.runs()
            if not runs.get('attack'):
                raise PlatformError(f'未找到后门攻击实验目录: {self.base}', 404)
            ids = self._validate_ids(payload.get('ids'))
            name = str(payload.get('name', '') or '')[:120]
            job_id = uuid.uuid4().hex
            output = self.directory(job_id)
            output.mkdir(parents=True)
            script = self.repo / 'scripts' / 'backdoor' / 'run_group.py'
            if not script.is_file():
                raise PlatformError(f'缺少绘图脚本: {script}', 500)
            spec = dict(base=str(self.base), runs={k: v for k, v in runs.items() if v},
                        ids=ids, output=str(output), device=self.device,
                        dataRoot=self.data_root, fontSize=int(payload.get('fontSize', 28) or 28))
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
                if job['status'] == 'stopping':
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
        for proc in list(self.processes.values()):
            if proc.poll() is None:
                proc.terminate()
        deadline = time.time() + 5
        for proc in list(self.processes.values()):
            try:
                proc.wait(timeout=max(0.1, deadline - time.time()))
            except subprocess.TimeoutExpired:
                proc.kill()
