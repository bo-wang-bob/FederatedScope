"""Isolated privacy experiments backed by verbatim cloud YAML presets."""
import copy
import os
import re
from pathlib import Path
import socket
import zipfile
import yaml

from .platform_config import FAMILIES, PlatformError, sha256
from .platform_paths import relative_path, resolve_path
from .platform_service import PlatformService, TERMINAL, read, now
from .privacy_artifacts import run_directory, feature_files, feature_summary, archive_existing_run
from .repository import JsonRepository
from .privacy_text import decode_log, restore_status_text

PRESETS = {
    'military_cnn': ('MilitaryAircraft-3D', 3, 'ggeur_fedmia_militaryaircraft3d_60'),
    'officehome_cnn': ('Office-Home', 4, 'ggeur_fedmia_example_60'),
}
FORM_FIELDS = {'group', 'method', 'name', 'defense', 'rounds', 'clientCount'}

class PrivacyConfig:
    def __init__(self, base):
        self.base, self.repo = base, base.repo

    def __getattr__(self, name):
        return getattr(self.base, name)

    def preset(self, group, defense=False):
        if isinstance(group, str) and group.startswith('uploaded_'):
            self.base.store.training(group)
            group = 'military_cnn'
        if not isinstance(group, str) or group not in PRESETS:
            raise PlatformError('请选择已同步云服务器配置的数据集 / ConvNeXt-Base')
        name = PRESETS[group][2] + ('_defense' if defense else '') + '.yaml'
        path = self.repo / 'scripts/privacy_presets' / name
        return path, yaml.safe_load(path.read_text(encoding='utf-8'))

    def defaults(self, group, defense=False):
        _, raw = self.preset(group, defense)
        return dict(group=group, method='ggeur', name='', defense=defense,
                    rounds=raw['federate']['total_round_num'], clientCount=raw['federate']['client_num'])

    def normalize(self, payload):
        if not isinstance(payload, dict):
            raise PlatformError('隐私配置必须是 JSON 对象')
        unknown = set(payload) - FORM_FIELDS - {'idempotencyKey', 'preflightId'}
        if unknown:
            raise PlatformError('只开放六项配置；其他参数由云服务器预设固定：' + ', '.join(sorted(unknown)))
        defense = payload.get('defense', False)
        if not isinstance(defense, bool):
            raise PlatformError('防御开关必须为布尔值')
        req = self.defaults(payload.get('group', 'military_cnn'), defense)
        req.update({k: v for k, v in payload.items() if k in FORM_FIELDS})
        if req['method'] != 'ggeur':
            raise PlatformError('当前同步的原始训练方案为 GGEUR + FedMIA')
        if not isinstance(req['name'], str) or len(req['name']) > 120:
            raise PlatformError('实验名称最多 120 字符')
        for name, low, high in [('rounds', 1, 1000), ('clientCount', 3, 240)]:
            value = req[name]
            if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                raise PlatformError(f'{name} 必须为 {low}–{high} 的整数')
        domains = 1 if req['group'].startswith('uploaded_') else PRESETS[req['group']][1]
        if req['clientCount'] % domains:
            raise PlatformError(f'此数据集客户端数须为 {domains} 的倍数')
        return req

    def dataset_root(self, group):
        if group.startswith('uploaded_'):
            value = self.base.store.training(group)
            return self.base.store.directory(value['id']) / 'images'
        family = group.split('_')[0]
        configured = os.environ.get('FS_PLATFORM_PRIVACY_DATASET_' + family.upper())
        if configured:
            return resolve_path(self.repo, configured)
        standard = self.base.datasets / FAMILIES[family][1]
        legacy = self.repo.parent / 'fedmia_local'
        candidates = [standard, legacy / 'militaryaircraft3d/datasets' / FAMILIES[family][1],
                      legacy / 'datasets' / FAMILIES[family][1], self.repo.parent / FAMILIES[family][1]]
        return next((p for p in candidates if p.is_dir()), standard)

    def build(self, req, output):
        source, raw = self.preset(req['group'], req['defense'])
        raw = copy.deepcopy(raw)
        if os.environ.get('FS_PLATFORM_DEVICE') == 'cpu':
            raw.update(use_gpu=False, device=0)
        raw['data']['root'] = relative_path(self.repo, self.dataset_root(req['group']))
        raw['federate'].update(client_num=req['clientCount'], sample_client_num=req['clientCount'],
                               total_round_num=req['rounds'])
        raw['outdir'] = relative_path(self.repo, Path(output) / 'logs')
        raw['expname'] = 'privacy_training'
        raw['ggeur'].update(save_mlp_checkpoint=True,
            mlp_checkpoint_dir=relative_path(self.repo, Path(output) / 'checkpoints'),
            training_distribution_dir=relative_path(self.repo, Path(output) / 'distributions'))
        if req['group'].startswith('uploaded_'):
            self.base.store.configure(raw, req['group'])
        return raw, dict(source=source.relative_to(self.repo).as_posix(), sourceSha256=sha256(source),
            privacyExperiment=True, privacyProtocol='cloud-yaml-native-v2', cloudPreset=True,
            thresholdPolicy='mix ROC attainable FPR<=1%; real-image test scores for display',
            configOverrides=['data.root', 'outdir', 'expname', 'federate.client_num',
                'federate.sample_client_num', 'federate.total_round_num', 'ggeur.save_mlp_checkpoint',
                'ggeur.mlp_checkpoint_dir', 'ggeur.training_distribution_dir'] +
                (['use_gpu', 'device'] if os.environ.get('FS_PLATFORM_DEVICE') == 'cpu' else []))

class PrivacyService(PlatformService):
    def __init__(self, repo, state):
        super().__init__(repo, Path(state) / 'privacy_experiments')
        self.configs = PrivacyConfig(self.configs)

    def _save(self, job):
        job['updatedAt'] = now()
        directory = self.directory(job['id'])
        try:
            JsonRepository._atomic_write(directory / 'job.json', job)
        except OSError as error:
            if getattr(error, 'winerror', None) not in (5, 32, 33):
                raise
            # Persist a complete alternate snapshot if Windows keeps the main
            # file locked. get/list always choose the newest durable snapshot.
            JsonRepository._atomic_write(directory / 'job.recovery.json', job)

    def get(self, job_id):
        with self.lock:
            directory = self.directory(job_id)
            snapshots = [read(directory / name) for name in ('job.json', 'job.recovery.json')]
            snapshots = [job for job in snapshots if job]
            if not snapshots:
                raise PlatformError('任务不存在', 404)
            job = max(snapshots, key=lambda j: j.get('updatedAt', j.get('createdAt', '')))
            stages = [job.get('stage', ''), *(c.get('stage', '') for c in job.get('clients', {}).values())]
            log = directory / 'runner.log'
            if any('\ufffd' in stage for stage in stages) and log.is_file():
                with log.open('rb') as stream:
                    job = restore_status_text(job, stream.read())
            if job['action'] == 'train':
                job['featureStorage'] = feature_summary(self.repo, job, directory)
            return job

    def list(self, detail=False):
        with self.lock:
            jobs = [self.get(p.name) for p in (self.state / 'jobs').iterdir()
                    if p.is_dir() and ((p / 'job.json').is_file() or (p / 'job.recovery.json').is_file())] if (self.state / 'jobs').is_dir() else []
            jobs.sort(key=lambda j: j['createdAt'], reverse=True)
            if detail:
                return jobs
            return [{k: v for k, v in j.items() if k not in {'clients', 'events', 'config', 'result', 'metrics', 'data'}} for j in jobs]

    def _run(self, job_id):
        try:
            super()._run(job_id)
        finally:
            job = self.get(job_id)
            if job['action'] == 'train':
                try:
                    archive_existing_run(self.repo, job, self.directory(job_id))
                except OSError as error:
                    job['storageError'] = str(error)
                    self._save(job)

    def create(self, action, payload):
        if action not in {'inspect', 'train'}:
            raise PlatformError('隐私实验仅支持预检和训练')
        return super().create(action, payload)

    def logs(self, job_id):
        self.get(job_id)
        path = self.directory(job_id) / 'runner.log'
        if not path.is_file():
            return ''
        with path.open('rb') as stream:
            offset = max(0, path.stat().st_size - 128 * 1024)
            stream.seek(offset)
            if offset:
                stream.readline()
            text = '\n'.join(decode_log(line) for line in stream.read().splitlines())
        return re.sub(r'\x1b\[[0-9;]*m', '', text)

    def catalog(self):
        groups = dict(PRESETS)
        for value in self.configs.base.store.list():
            if value['kind'] == 'train':
                groups[value['group']] = (value['name'], 1, '')
        return dict(groups=[dict(id=group, dataset=label, backbone='ConvNeXt-Base', domains=domains,
            datasetFound=self.configs.dataset_root(group).is_dir(),
            methods=[dict(id='ggeur', label='GGEUR + FedMIA（云服务器原配置）', enabled=True,
                defaults=self.configs.defaults(group), defenseDefaults=self.configs.defaults(group, True))])
            for group, (label, domains, _) in groups.items()], host=socket.gethostname(),
            protocol='六项配置 / 云服务器原始 YAML / 本地原生训练')

    def results(self, job_id, client_id=1):
        job = self.get(job_id)
        if job['action'] != 'train':
            raise PlatformError('请选择训练实验', 409)
        archive = run_directory(self.repo, job['request'], job_id)
        report = read(archive / 'privacy_results.json') or read(self.directory(job_id) / 'privacy_results.json')
        if not report or str(client_id) not in report['clients']:
            raise PlatformError('此客户端没有完整攻击结果', 404)
        return {**{k: v for k, v in report.items() if k != 'clients'},
                'clientId': client_id, 'clientIds': list(map(int, report['clients'])), **report['clients'][str(client_id)]}

    def image(self, job_id, client_id, group, index):
        result = self.results(job_id, client_id)
        rows = result.get('samples', {}).get(group, [])
        if not 0 <= index < len(rows):
            raise PlatformError('样本不存在', 404)
        job = self.get(job_id)
        from .platform_samples import SampleCatalog
        from types import SimpleNamespace
        root = resolve_path(self.repo, job['config']['data']['root'])
        resolver = SampleCatalog(SimpleNamespace(configs=SimpleNamespace(datasets=root.parent)))
        return resolver.image_path(job, rows[index])

    def feature_bundle(self, job_id):
        job = self.get(job_id)
        if job['status'] not in TERMINAL or job['action'] != 'train':
            raise PlatformError('任务结束后可导出已保存特征', 409)
        output = self.directory(job_id)
        target = output / 'privacy_features.zip'
        with self.lock, zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
            for name in ('effective.yaml', 'resolved_config.yaml', 'data_manifest.json',
                         'privacy_results.json', 'privacy_feature_manifest.json'):
                file = output / name
                if file.is_file():
                    archive.write(file, name)
            source, _ = self.configs.preset(job['request']['group'], job['request']['defense'])
            archive.write(source, 'cloud_preset.yaml')
            local_run = run_directory(self.repo, job['request'], job_id)
            for file in feature_files(output, local_run):
                if file.is_file() and not file.is_symlink():
                    archive.write(file, 'ggeur_fedmia_features/' + file.name)
        return target
