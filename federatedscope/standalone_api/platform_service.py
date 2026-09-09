"""Durable single-host job control. Every subprocess is owned by one experiment."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import threading
import time
import uuid
import zipfile

import psutil
import yaml

from .platform_config import ConfigFactory, PlatformError, sha256
from .repository import JsonRepository

TERMINAL = {'completed', 'failed', 'stopped', 'interrupted'}


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except FileNotFoundError:
        return default


class PlatformService:
    def __init__(self, repo, state, recover=True):
        self.repo, self.state = Path(repo).resolve(), Path(state).resolve()
        self.state.mkdir(parents=True, exist_ok=True)
        self.configs = ConfigFactory(self.repo)
        self.lock = threading.RLock()
        self.processes = {}
        self.closed = False
        self._lease = (self.state / 'service.lock').open('a+b')
        if os.name == 'posix':
            import fcntl
            try:
                fcntl.flock(self._lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                self._lease.close()
                raise PlatformError('该状态目录已被另一服务占用', 409) from error
        try:
            self.commit = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], cwd=self.repo, text=True,
                stderr=subprocess.DEVNULL, timeout=5).strip()
        except (OSError, subprocess.SubprocessError):
            self.commit = 'unknown'
        if recover:
            self.recover()

    def directory(self, job_id):
        if not re.fullmatch(r'[a-f0-9]{32}', job_id):
            raise PlatformError('任务不存在', 404)
        return self.state / 'jobs' / job_id

    def _save(self, job):
        job['updatedAt'] = now()
        JsonRepository._atomic_write(self.directory(job['id']) / 'job.json', job)

    def get(self, job_id):
        with self.lock:
            job = read(self.directory(job_id) / 'job.json')
            if job is None:
                raise PlatformError('任务不存在', 404)
            return job

    def list(self, detail=False):
        with self.lock:
            jobs = [read(p) for p in (self.state / 'jobs').glob('*/job.json')]
            jobs.sort(key=lambda j: j['createdAt'], reverse=True)
            if detail:
                return jobs
            return [{k: v for k, v in j.items() if k not in {'clients', 'events', 'config', 'result', 'metrics', 'data'}}
                    for j in jobs]

    def _owned_processes(self, job_id):
        """Exact spec identity, never a port/name-wide process match."""
        spec = str((self.directory(job_id) / 'spec.json').resolve())
        metadata = read(self.directory(job_id) / 'process.json', {})
        owned = []
        for proc in psutil.process_iter(['pid', 'create_time', 'cmdline']):
            try:
                cmd = proc.info['cmdline'] or []
                if (spec in cmd and 'federatedscope.standalone_api.platform_worker' in cmd
                        and (not metadata or proc.pid != metadata['pid']
                             or abs(proc.create_time() - metadata['created']) < .01)):
                    owned.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return owned

    def cleanup(self, job_id):
        try:
            owned = self._owned_processes(job_id)
            descendants = []
            for proc in owned:
                descendants.extend(proc.children(recursive=True))
            targets = list({p.pid: p for p in owned + descendants}.values())
            for proc in reversed(targets):
                try:
                    proc.terminate()
                except psutil.NoSuchProcess:
                    pass
            _, alive = psutil.wait_procs(targets, timeout=5)
            for proc in alive:
                proc.kill()  # psutil verifies PID creation time before killing.
            _, alive = psutil.wait_procs(alive, timeout=3)
            if alive or self._owned_processes(job_id):
                return {'ok': False, 'message': '本任务进程仍存在，禁止启动新任务', 'at': now()}
            return {'ok': True, 'pids': [p.pid for p in targets], 'at': now(),
                    'message': '本任务进程已退出；单机训练不分配网络端口'}
        except (psutil.Error, OSError) as error:
            return {'ok': False, 'message': str(error), 'at': now()}

    def recover(self):
        with self.lock:
            for job in self.list(detail=True):
                if job['status'] not in TERMINAL or not job.get('cleanup', {}).get('ok', True):
                    job['cleanup'] = self.cleanup(job['id'])
                    job.update(status='interrupted', stage='服务重启，中断任务已回收' if job['cleanup']['ok'] else '服务重启，任务清理失败',
                               error='未自动续训；保留配置和日志，可重跑以避免重复聚合', endedAt=now())
                    self._save(job)

    def _assert_available(self):
        if self.closed:
            raise PlatformError('服务正在关闭', 409)
        for job in self.list(detail=True):
            if job['status'] not in TERMINAL or not job.get('cleanup', {}).get('ok', True):
                raise PlatformError(f'请先等待或清理任务 {job["id"][:8]}；平台一次运行一个任务', 409)

    def create(self, action, payload):
        if action not in {'inspect', 'train', 'evaluate'}:
            raise PlatformError('不支持的任务类型')
        with self.lock:
            key = payload.get('idempotencyKey')
            if not isinstance(key, str) or not re.fullmatch(r'[A-Za-z0-9_-]{8,100}', key):
                raise PlatformError('需要有效的 idempotencyKey')
            for job in self.list(detail=True):
                if job['idempotencyKey'] == key:
                    if job['action'] != action or job['submitted'] != payload:
                        raise PlatformError('幂等键已用于另一请求', 409)
                    return job
            self._assert_available()
            req = self._evaluation_request(payload) if action == 'evaluate' else self.configs.normalize(payload)
            preflight = None
            if action == 'train':
                preflight = self.get(str(payload.get('preflightId', '')))
                before = {k: v for k, v in preflight['request'].items() if k != 'name'}
                after = {k: v for k, v in req.items() if k != 'name'}
                if preflight['action'] != 'inspect' or preflight['status'] != 'completed' or before != after:
                    raise PlatformError('配置已改变或预检未通过，请重新预检', 409)
            job_id = uuid.uuid4().hex
            output = self.directory(job_id)
            output.mkdir(parents=True)
            spec = {'action': action, 'request': req, 'output': str(output)}
            config, provenance = {}, {}
            if action != 'evaluate':
                config, provenance = self.configs.build(req, output)
                config_path = output / 'effective.yaml'
                config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding='utf-8')
                spec['configPath'] = str(config_path)
                if preflight:
                    spec['expectedData'] = preflight['result']['testFingerprint']
                    spec['expectedPartition'] = preflight['result']['partitionFingerprint']
                provenance['configSha256'] = sha256(config_path)
            else:
                model_job, test_job = self.get(req['modelId'].split(':')[0]), self.get(req['testsetId'])
                kind = req['modelId'].split(':')[1]
                spec.update(checkpointPath=str(self.directory(model_job['id']) / f'checkpoints/mlp_{kind}.pt'),
                            bundlePath=str(self.directory(test_job['id']) / 'checkpoints/pretrained_test_features.pt'),
                            checkpointHash=model_job['result']['artifactHashes'][f'mlp_{kind}.pt'],
                            bundleHash=test_job['result']['artifactHashes']['pretrained_test_features.pt'])
            provenance.update(codeCommit=self.commit, python=sys.version, protocol='cache-only-v1')
            # Capture the actual Python sources too: a branch may legitimately
            # have local edits after deployment, so a Git SHA alone is not enough.
            source_zip = output / 'source.zip'
            with zipfile.ZipFile(source_zip, 'w', zipfile.ZIP_DEFLATED) as archive:
                sources = list((self.repo / 'federatedscope').rglob('*.py'))
                sources.extend([self.repo / 'scripts/test_outline_validation/evaluate_saved_mlp.py',
                                self.repo / 'setup.py'])
                for source in sources:
                    if source.is_file() and not source.is_symlink():
                        archive.write(source, source.relative_to(self.repo))
            provenance['sourceArchiveSha256'] = sha256(source_zip)
            spec['provenance'] = provenance
            job = dict(id=job_id, action=action, request=req, submitted=payload, config=config,
                       idempotencyKey=key, provenance=provenance, createdAt=now(),
                       status='queued', stage='等待启动', clients={}, metrics=[], events=[],
                       cleanup={'ok': True, 'message': '尚未启动'}, error=None)
            JsonRepository._atomic_write(output / 'spec.json', spec)
            self._save(job)
            threading.Thread(target=self._run, args=(job_id,), daemon=True).start()
            return copy.deepcopy(job)

    def _event(self, job_id, event):
        with self.lock:
            job = self.get(job_id)
            event['at'] = now()
            with (self.directory(job_id) / 'events.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + '\n')
            job['events'] = (job['events'] + [event])[-100:]
            if event['type'] == 'stage':
                job['stage'] = event['stage']
            elif event['type'] == 'prepared':
                job['data'] = {k: v for k, v in event.items() if k not in {'type', 'at', 'clients'}}
                resolved = self.directory(job_id) / 'resolved_config.yaml'
                if resolved.exists():
                    job['config'] = yaml.safe_load(resolved.read_text(encoding='utf-8'))
                job['clients'] = {str(c['id']): {**c, 'stage': '待训练'} for c in event['clients']}
            elif event['type'] == 'round_started':
                selected = set(event['participants'])
                for client in job['clients'].values():
                    client.update(stage='等待本地训练' if client['id'] in selected else '本轮未参与', round=event['round'])
            elif event['type'] == 'client':
                client = job['clients'].setdefault(str(event['clientId']), {})
                client.update(event)
            elif event['type'] == 'metrics':
                job['metrics'].append(event)
            self._save(job)

    def _run(self, job_id):
        output, proc = self.directory(job_id), None
        try:
            with self.lock:
                job = self.get(job_id)
                if job['status'] != 'queued':
                    return
                env = {**os.environ, 'PYTHONUNBUFFERED': '1', 'OMP_NUM_THREADS': '2',
                       'FEDERATEDSCOPE_GGEUR_LIGHTWEIGHT': '1', 'MPLCONFIGDIR': str(output / 'mpl')}
                with (output / 'runner.log').open('wb') as log:
                    proc = subprocess.Popen([sys.executable, '-m',
                        'federatedscope.standalone_api.platform_worker', str(output / 'spec.json')],
                        cwd=self.repo, env=env, stdout=log, stderr=subprocess.STDOUT,
                        start_new_session=os.name == 'posix')
                self.processes[job_id] = proc
                metadata = {'pid': proc.pid, 'created': psutil.Process(proc.pid).create_time(),
                            'spec': str(output / 'spec.json')}
                JsonRepository._atomic_write(output / 'process.json', metadata)
                job.update(status='running', startedAt=now(), stage='启动独立进程',
                           cleanup={'ok': False, 'message': '任务运行中'})
                self._save(job)
            with (output / 'runner.log').open(encoding='utf-8', errors='replace') as log:
                pending = ''
                while True:
                    chunk = log.read()
                    pending += chunk
                    lines = pending.split('\n')
                    pending = lines.pop()
                    for line in lines:
                        if line.startswith('__PLATFORM__'):
                            self._event(job_id, json.loads(line[len('__PLATFORM__'):]))
                    if proc.poll() is not None and not chunk:
                        break
                    time.sleep(.15)
            with self.lock:
                job = self.get(job_id)
                result = read(output / 'result.json')
                failure = read(output / 'failure.json', {})
                stopping = job['status'] in {'stopping', 'stopped'}
                job['cleanup'] = self.cleanup(job_id)
                job['exitCode'] = proc.returncode
                if stopping:
                    job.update(status='stopped', stage='已停止')
                elif proc.returncode == 0 and result and job['cleanup']['ok']:
                    job.update(status='completed', result=result, stage='预检通过' if job['action'] == 'inspect' else '已完成')
                else:
                    job.update(status='failed', stage='运行失败',
                               error=failure.get('message', f'进程退出 {proc.returncode}，请查看日志'))
                job['endedAt'] = now()
                self._save(job)
        except Exception as error:
            with self.lock:
                job = self.get(job_id)
                job.update(status='failed', stage='启动或监控失败', error=str(error), endedAt=now(),
                           cleanup=self.cleanup(job_id))
                self._save(job)
        finally:
            with self.lock:
                self.processes.pop(job_id, None)

    def stop(self, job_id):
        with self.lock:
            job = self.get(job_id)
            if job['status'] in TERMINAL and job.get('cleanup', {}).get('ok'):
                return job
            job.update(status='stopping', stage='停止并回收本任务')
            self._save(job)
        cleanup = self.cleanup(job_id)
        with self.lock:
            job = self.get(job_id)
            job.update(status='stopped' if cleanup['ok'] else 'failed',
                       cleanup=cleanup, endedAt=now(), stage='已停止' if cleanup['ok'] else '清理失败')
            self._save(job)
            return job

    def library(self):
        models, testsets = [], []
        for job in self.list(detail=True):
            if job['action'] != 'train' or job['status'] != 'completed':
                continue
            result = job['result']
            base = dict(jobId=job['id'], group=job['request']['group'],
                        name=job['request']['name'] or job['id'][:8],
                        method=job['request']['method'], createdAt=job['createdAt'],
                        classes=result['classes'], domains=result['domains'],
                        testFingerprint=result['testFingerprint'],
                        featureSpace=result['featureSpace'])
            for kind in ('final', 'best'):
                if (self.directory(job['id']) / f'checkpoints/mlp_{kind}.pt').is_file():
                    models.append(dict(base, id=job['id'] + ':' + kind, kind=kind,
                                       sha256=result['artifactHashes'][f'mlp_{kind}.pt']))
            if (self.directory(job['id']) / 'checkpoints/pretrained_test_features.pt').is_file():
                testsets.append(dict(base, id=job['id'], samples=result['testSamples'],
                                     sha256=result['artifactHashes']['pretrained_test_features.pt']))
        return dict(models=models, testsets=testsets)

    def catalog(self):
        catalog = self.configs.catalog()
        latest = {}
        for job in self.list():
            if job['action'] == 'inspect' and job['status'] in TERMINAL:
                latest.setdefault(job['request']['group'], {
                    'id': job['id'], 'status': job['status'], 'at': job['updatedAt'],
                    'error': job['error']})
        for group in catalog['groups']:
            group['lastPreflight'] = latest.get(group['id'])
        return catalog

    def _evaluation_request(self, payload):
        if set(payload) - {'modelId', 'testsetId', 'domains', 'classes', 'name', 'idempotencyKey'}:
            raise PlatformError('评测含未知参数')
        library = self.library()
        model = next((m for m in library['models'] if m['id'] == payload.get('modelId')), None)
        test = next((t for t in library['testsets'] if t['id'] == payload.get('testsetId')), None)
        if not model or not test:
            raise PlatformError('模型或测试集未登记，或训练尚未完成')
        if model['featureSpace'] != test['featureSpace']:
            raise PlatformError('模型与测试集的特征空间、标签或缓存版本不一致')
        domains, classes = payload.get('domains', []), payload.get('classes', [])
        if (not isinstance(domains, list) or not all(isinstance(x, str) for x in domains)
                or len(set(domains)) != len(domains)
                or not set(domains) <= {d['name'] for d in test['domains']}):
            raise PlatformError('测试域选择无效')
        if (not isinstance(classes, list) or not all(type(x) is int for x in classes)
                or len(set(classes)) != len(classes)
                or not set(classes) <= set(range(len(test['classes'])))):
            raise PlatformError('测试类别选择无效')
        name = payload.get('name', '')
        if not isinstance(name, str) or len(name) > 120:
            raise PlatformError('评测名称最多 120 字符')
        # A different test split is allowed only if none of its selected samples
        # occurred in this model's training membership.
        training = read(self.directory(model['jobId']) / 'data_manifest.json')
        testing = read(self.directory(test['jobId']) / 'data_manifest.json')
        train_ids = {(d, key) for rows in training['partition'].values() for d, key, _ in rows}
        test_ids = {(d, key) for d, rows in testing['test'].items() if not domains or d in domains
                    for key, label in rows.items() if not classes or label in classes}
        if train_ids & test_ids:
            raise PlatformError('选中测试集包含此模型的训练样本，拒绝泄漏评测')
        return dict(modelId=model['id'], testsetId=test['id'], domains=domains,
                    classes=classes, name=name, group=model['group'], method=model['method'])

    def logs(self, job_id):
        self.get(job_id)
        path = self.directory(job_id) / 'runner.log'
        if not path.exists():
            return ''
        with path.open('rb') as stream:
            stream.seek(max(0, path.stat().st_size - 128 * 1024))
            return re.sub(r'\x1b\[[0-9;]*m', '', stream.read().decode('utf-8', errors='replace'))

    def resources(self):
        gpus, error = [], None
        try:
            data = subprocess.check_output(['nvidia-smi',
                '--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu',
                '--format=csv,noheader,nounits'], text=True, timeout=4)
            for line in data.strip().splitlines():
                index, name, util, used, total, temp = [s.strip() for s in line.split(',')]
                gpus.append(dict(index=int(index), name=name, utilization=float(util),
                                 memoryUsedMiB=float(used), memoryTotalMiB=float(total), temperature=float(temp)))
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            error = str(exc)
        vm, disk = psutil.virtual_memory(), psutil.disk_usage(self.state)
        return dict(at=now(), hostname=socket.gethostname(), cpuPercent=psutil.cpu_percent(interval=.1),
                    memoryPercent=vm.percent, memoryUsed=vm.used, memoryTotal=vm.total,
                    diskFree=disk.free, diskTotal=disk.total, gpus=gpus, gpuError=error)

    def close(self):
        self.closed = True
        for job in self.list(detail=True):
            if job['status'] not in TERMINAL:
                self.stop(job['id'])
        self._lease.close()
