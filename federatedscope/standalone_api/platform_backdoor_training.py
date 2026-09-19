"""后门研究: 用「用户上传的数据集」跑一组攻防实验。

与 BackdoorService (挑图 + 三连对比绘图) 职责分离: 这里只负责训练。三个配置
模板来自 scripts/backdoor/tests/, 数据通道与平台上传数据集完全一致
(domainnet + 便携 manifest), 因此用户传什么数据就跑什么数据。

跑完后把结果登记给 BackdoorService: 写 <backdoor_root>/groups.json, 后者据此
挑 attack / defense 两个 run, 并切到对应的 data_root 与类别名, 让后续
"挑图 -> 三连对比" 直接可用, 不必改环境变量重启服务。

目录约定:
    base (= BackdoorService.base)
      <stem>_<token>/          本组实验, token 唯一标识一次训练
      testset_images/          按当前组重新导出的测试集 (.group 记录归属)
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import uuid

from .platform_config import PlatformError
from .repository import JsonRepository

TERMINAL = {'completed', 'failed', 'stopped', 'interrupted'}
TOKEN_PATTERN = re.compile(r'^grp[0-9a-f]{8}$')
# 三个模板的执行顺序: 干净基线 -> 注入后门 -> 加防御后门
TEMPLATES = {
    'baseline': 'vit_newdataset.yaml',
    'attack': 'sabre_vit_newdataset.yaml',
    'defense': 'sabre_vit_newdataset_defense.yaml',
}
ORDER = ('baseline', 'attack', 'defense')
LABELS = {
    'baseline': '干净基线（无攻击）',
    'attack': 'SABRE 后门攻击',
    'defense': 'SABRE 攻击 + multi_metrics 防御',
}


def now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (FileNotFoundError, ValueError, OSError):
        return default


class BackdoorTrainingService:
    """一次启动 = 顺序跑完三个实验 + 导出测试集 + 登记结果组。"""

    def __init__(self, repo, state_root, upload_root, base):
        self.repo = Path(repo).resolve()
        self.state_root = Path(state_root).resolve()
        # BackdoorService.root: groups.json 与后者共用同一份状态目录
        self.upload_root = Path(upload_root)
        self.base = Path(base)
        self.lock = threading.RLock()
        self.process = None
        self.templates_dir = self.repo / 'scripts' / 'backdoor' / 'tests'
        self.recover()

    # ------------------------------------------------------------------ #
    # 任务目录
    # ------------------------------------------------------------------ #
    @property
    def root(self):
        return self.state_root / 'backdoor-training'

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
        jobs = []
        for path in sorted((self.root).glob('*/job.json')):
            job = read_json(path)
            if job:
                jobs.append(job)
        jobs.sort(key=lambda j: j.get('createdAt', ''), reverse=True)
        return jobs

    def _save(self, job):
        job['updatedAt'] = now()
        target = self.directory(job['id'])
        target.mkdir(parents=True, exist_ok=True)
        JsonRepository._atomic_write(target / 'job.json', job)

    # ------------------------------------------------------------------ #
    # 概览
    # ------------------------------------------------------------------ #
    def templates(self):
        return [dict(key=key, file=TEMPLATES[key], label=LABELS[key],
                     exists=(self.templates_dir / TEMPLATES[key]).is_file())
                for key in ORDER]

    def _train_datasets(self):
        from .uploaded_datasets import DatasetStore
        store = DatasetStore(self.repo)
        rows = []
        for value in store.list():
            if value.get('kind') != 'train':
                continue
            directory = store.directory(value['id'])
            if not (directory / 'manifest.json').is_file():
                continue
            rows.append((value, directory))
        return rows

    def status(self):
        datasets = [dict(id=value['id'], name=value['name'],
                         classes=len(value.get('classes') or []),
                         count=value.get('count', 0),
                         layout=value.get('layout', 'classes'))
                    for value, _ in self._train_datasets()]
        running = next((j for j in self.list()
                        if j.get('status') not in TERMINAL), None)
        return dict(runnable=bool(datasets) and
                    all(item['exists'] for item in self.templates()),
                    datasets=datasets, templates=self.templates(),
                    missing=[item['file'] for item in self.templates()
                             if not item['exists']],
                    base=str(self.base), job=running, group=self.read_group())

    # ------------------------------------------------------------------ #
    # 结果登记 (给 BackdoorService 消费)
    # ------------------------------------------------------------------ #
    @property
    def group_file(self):
        return self.upload_root / 'groups.json'

    def read_group(self):
        value = read_json(self.group_file)
        return value.get('active') if isinstance(value, dict) else None

    def _write_group(self, group):
        self.upload_root.mkdir(parents=True, exist_ok=True)
        JsonRepository._atomic_write(self.group_file, dict(active=group))

    # ------------------------------------------------------------------ #
    # 启动
    # ------------------------------------------------------------------ #
    def start(self, payload):
        if not isinstance(payload, dict):
            raise PlatformError('请求必须是 JSON 对象')
        with self.lock:
            running = next((j for j in self.list()
                            if j.get('status') not in TERMINAL), None)
            if running:
                raise PlatformError(f'训练任务 {running["id"][:8]} 仍在运行', 409)
            missing = [item['file'] for item in self.templates() if not item['exists']]
            if missing:
                raise PlatformError(f'缺少实验配置模板: {", ".join(missing)}', 500)
            value, directory = self._resolve_dataset(payload.get('datasetId'))
            rounds = payload.get('rounds')
            if rounds is not None and (not isinstance(rounds, int)
                                       or isinstance(rounds, bool)
                                       or not 1 <= rounds <= 500):
                raise PlatformError('训练轮数必须是 1–500 的整数')
            device = payload.get('device')
            if device is not None and device not in ('cuda', 'cpu'):
                raise PlatformError('设备只能是 cuda 或 cpu')

            job_id = uuid.uuid4().hex
            out = self.directory(job_id)
            if out.exists():
                shutil.rmtree(out)
            out.mkdir(parents=True)
            # token 保证目录唯一: FS 在 expname 冲突时会退到 sub_exp_<ts>,
            # 而任务只会启用一次的组=不会出现重名的 group。
            try:
                token = next(t for t in ('grp' + uuid.uuid4().hex[:8]
                                         for _ in range(20))
                             if not any((self.base / f'{stem}_{t}').exists()
                                        for stem in (Path(TEMPLATES[k]).stem
                                                     for k in ORDER)))
            except StopIteration:
                raise PlatformError('无法生成唯一的实验编号', 500)

            job = dict(id=job_id, action='backdoor-training', token=token,
                       datasetId=value['id'], datasetName=value['name'],
                       base=str(self.base), device=device or 'cuda',
                       total=len(ORDER), createdAt=now(), updatedAt=now(),
                       status='queued', stage='等待启动', stageIndex=0,
                       error=None, results=[])
            self._save(job)
            specs = [self._build_spec(key, value, directory, token, job_id,
                                      rounds, device) for key in ORDER]
            JsonRepository._atomic_write(out / 'spec.json', dict(
                base=str(self.base), token=token, datasetId=value['id'],
                device=device or 'cuda', runs=specs))
            threading.Thread(target=self._run, args=(job_id, specs, value,
                                                     directory, token),
                             daemon=True).start()
            return copy.deepcopy(job)

    def _resolve_dataset(self, dataset_id):
        rows = self._train_datasets()
        if dataset_id is not None:
            if not isinstance(dataset_id, str):
                raise PlatformError('数据集编号非法')
            for value, directory in rows:
                if value['id'] == dataset_id:
                    return value, directory
            raise PlatformError('数据集不存在或未完成上传', 404)
        if not rows:
            raise PlatformError('请先上传带类别的训练数据集', 409)
        return rows[-1]

    # ------------------------------------------------------------------ #
    # 配置生成
    # ------------------------------------------------------------------ #
    def _build_spec(self, key, dataset, directory, token, job_id,
                    rounds, device):
        import yaml
        template = self.templates_dir / TEMPLATES[key]
        with template.open(encoding='utf-8') as stream:
            cfg = yaml.safe_load(stream) or {}

        images = directory / 'images'
        if not images.is_dir():
            raise PlatformError('数据集图片目录不存在', 409)
        manifest = directory / 'manifest.json'
        if not manifest.is_file():
            raise PlatformError('数据集缺少 manifest.json，请重新上传', 409)
        classes = list(dataset.get('classes') or [])
        if not classes:
            raise PlatformError('数据集没有可用类别', 409)
        stem = template.stem

        cfg['outdir'] = str(self.base)
        cfg['expname'] = f'{stem}_{token}'
        cfg['model']['num_classes'] = len(classes)
        cfg['data']['type'] = 'domainnet'
        cfg['data']['root'] = str(images)
        ggeur = cfg.setdefault('ggeur', {})
        ggeur.update(domainnet_domains=['uploaded'],
                     domainnet_manifest_path=str(manifest),
                     domainnet_shared_classes_only=False,
                     feature_cache_dir=str(directory / 'features'),
                     use_feature_cache=True,
                     head_only_mode=True)
        if device == 'cpu':
            cfg['use_gpu'] = False
        if rounds:
            cfg['federate']['total_round_num'] = int(rounds)
            sabre = (cfg.get('attack') or {}).get('sabre')
            if isinstance(sabre, dict) and \
                    int(sabre.get('start_round', 0)) >= int(rounds):
                sabre['start_round'] = max(0, int(rounds) // 2)
                sabre['poison_epochs'] = max(1, int(rounds) - sabre['start_round'])

        config_path = self.directory(job_id) / f'{stem}.yaml'
        with config_path.open('w', encoding='utf-8') as stream:
            yaml.safe_dump(cfg, stream, allow_unicode=True, sort_keys=False)
        return dict(key=key, label=LABELS[key], expname=cfg['expname'],
                    config=str(config_path), template=TEMPLATES[key])

    # ------------------------------------------------------------------ #
    # 执行
    # ------------------------------------------------------------------ #
    def _run(self, job_id, specs, dataset, directory, token):
        out = self.directory(job_id)
        log_path = out / 'runner.log'
        try:
            with self.lock:
                job = self.get(job_id)
                job.update(status='running', startedAt=now(), stage='启动训练')
                self._save(job)
            for index, spec in enumerate(specs):
                with self.lock:
                    job = self.get(job_id)
                    if job.get('status') in {'stopping', 'stopped'}:
                        break
                    job.update(stage=f'训练 {index + 1}/{len(specs)}: {spec["label"]}',
                               stageIndex=index)
                    self._save(job)
                returncode = self._spawn(job_id, spec, log_path)
                if returncode != 0:
                    self._fail(job_id, f'{spec["label"]} 训练失败'
                                       f'（退出码 {returncode}），请查看日志')
                    return
            with self.lock:
                job = self.get(job_id)
                if job.get('status') in {'stopping', 'stopped'}:
                    job.update(status='stopped', stage='已停止', endedAt=now())
                    self._save(job)
                    return
                job.update(stage='导出测试集', stageIndex=len(specs))
                self._save(job)
            self._export_testset(token, specs, log_path)
            group = dict(token=token, base=str(self.base),
                         baseline=specs[0]['expname'],
                         attack=specs[1]['expname'],
                         defense=specs[2]['expname'],
                         dataRoot=str(directory / 'images'),
                         datasetId=dataset['id'],
                         datasetName=dataset['name'],
                         classes=list(dataset.get('classes') or []),
                         createdAt=now())
            self._write_group(group)
            with self.lock:
                job = self.get(job_id)
                job.update(status='completed', stage='已完成', endedAt=now(),
                           group=group,
                           results=[dict(key=s['key'], label=s['label'],
                                         expname=s['expname']) for s in specs])
                self._save(job)
        except Exception as error:  # pragma: no cover - 后台线程
            self._fail(job_id, str(error))
        finally:
            with self.lock:
                self.process = None

    def _spawn(self, job_id, spec, log_path):
        env = {**os.environ, 'PYTHONUNBUFFERED': '1',
               'PYTHONIOENCODING': 'utf-8', 'OMP_NUM_THREADS': '2',
               'MPLCONFIGDIR': str(log_path.parent / 'mpl')}
        with log_path.open('ab') as log:
            log.write(f'\n=== {now()} {spec["label"]} ===\n'.encode('utf-8'))
            proc = subprocess.Popen(
                [sys.executable, str(self.repo / 'run.py'),
                 '--cfg', spec['config']],
                cwd=str(self.repo), env=env, stdout=log,
                stderr=subprocess.STDOUT)
            with self.lock:
                self.process = proc
                job = self.get(job_id)
                job['pid'] = proc.pid
                self._save(job)
            return proc.wait()

    def _export_testset(self, token, specs, log_path):
        """导出测试集图片/索引, 供 BackdoorService 挑图。换组即重建。"""
        target = self.base / 'testset_images'
        if target.is_dir():
            marker = target / '.group'
            existing = marker.read_text(encoding='utf-8').strip() \
                if marker.is_file() else ''
            if existing and existing != token:
                shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)
        (target / '.group').write_text(token, encoding='utf-8')
        script = self.repo / 'scripts' / 'backdoor' / 'export_testset.py'
        if not script.is_file():
            raise PlatformError(f'缺少导出脚本: {script}', 500)
        env = {**os.environ, 'PYTHONUNBUFFERED': '1',
               'PYTHONIOENCODING': 'utf-8',
               'MPLCONFIGDIR': str(log_path.parent / 'mpl')}
        with log_path.open('ab') as log:
            log.write(f'\n=== {now()} export testset ===\n'.encode('utf-8'))
            code = subprocess.call(
                [sys.executable, str(script), '--run',
                 str(self.base / specs[1]['expname']), '--out', str(target),
                 '--device', os.environ.get('FS_BACKDOOR_DEVICE', 'cuda')],
                cwd=str(self.repo), env=env, stdout=log,
                stderr=subprocess.STDOUT)
        if code != 0:
            raise PlatformError('测试集导出失败，请查看日志')

    def _fail(self, job_id, message):
        with self.lock:
            job = self.get(job_id)
            job.update(status='failed', stage='训练失败',
                       error=str(message)[:500], endedAt=now())
            self._save(job)

    # ------------------------------------------------------------------ #
    # 停止 / 日志 / 恢复
    # ------------------------------------------------------------------ #
    def stop(self, job_id):
        with self.lock:
            job = self.get(job_id)
            if job.get('status') in TERMINAL:
                return job
            job.update(status='stopping', stage='正在停止')
            self._save(job)
            proc, pid = self.process, job.get('pid')
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=15)
        elif pid:
            try:
                os.kill(pid, 9)
            except OSError:
                pass
        with self.lock:
            job = self.get(job_id)
            if job.get('status') not in TERMINAL:
                job.update(status='stopped', stage='已停止', endedAt=now())
                self._save(job)
            return job

    def logs(self, job_id):
        self.get(job_id)
        path = self.directory(job_id) / 'runner.log'
        if not path.is_file():
            return ''
        with path.open('rb') as stream:
            stream.seek(max(0, path.stat().st_size - 256 * 1024))
            return re.sub(r'\x1b\[[0-9;]*m', '',
                          stream.read().decode('utf-8', errors='replace'))

    def recover(self):
        for path in sorted((self.root).glob('*/job.json')):
            job = read_json(path)
            if job and job.get('status') not in TERMINAL:
                job.update(status='interrupted', stage='服务重启，任务已中断',
                           error='未自动续跑，请重新启动训练', endedAt=now())
                JsonRepository._atomic_write(path, job)

    def close(self):
        for job in self.list():
            if job.get('status') not in TERMINAL:
                try:
                    self.stop(job['id'])
                except PlatformError:
                    pass
