"""Environment and scenario integrity checks before starting a task."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List


def _check(name: str, ready: bool, message: str,
           details: Dict[str, Any] | None = None) -> Dict[str, Any]:
    return {
        'name': name,
        'ready': bool(ready),
        'message': message,
        'details': details or {},
    }


def inspect_runtime(config: Dict[str, Any], template: Path, data_root: Path,
                    model_path: Path, workspace: Path) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []
    checks.append(_check(
        'template', template.is_file(),
        '运行模板可用' if template.is_file() else '运行模板不存在',
        {'path': str(template)}))

    required_domains = ['Art', 'Clipart', 'Product', 'Real_World']
    resolved_domains = []
    image_count = 0
    if data_root.is_dir():
        for domain in required_domains:
            candidates = [data_root / domain,
                          data_root / domain.replace('_', ' ')]
            found = next((path for path in candidates if path.is_dir()), None)
            if found is not None:
                resolved_domains.append(domain)
                image_count += sum(
                    1 for path in found.rglob('*')
                    if path.is_file() and path.suffix.lower() in
                    {'.jpg', '.jpeg', '.png'})
    data_ready = len(resolved_domains) == 4 and image_count > 0
    checks.append(_check(
        'dataset', data_ready,
        'OfficeHome 四域数据可用' if data_ready else
        'OfficeHome 数据目录缺失、四域不完整或没有图像',
        {'path': str(data_root), 'domains': resolved_domains,
         'imageCount': image_count}))

    scenario = config.get('_scenario') or {}
    manifest_value = scenario.get('artifacts', {}).get(
        'partitionManifest', '')
    manifest_path = Path(manifest_value) if manifest_value else None
    manifest = None
    manifest_error = ''
    if manifest_path and manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError) as error:
            manifest_error = str(error)
    preview_fingerprint = scenario.get('preview', {}).get(
        'datasetFingerprint')
    current_fingerprint = None
    if data_ready:
        from federatedscope.standalone_api.scenarios import \
            current_dataset_fingerprint
        current_fingerprint = current_dataset_fingerprint(int(
            scenario.get('request', {}).get('partition', {}).get('seed', 0)))
    manifest_ready = bool(
        manifest and manifest.get('schemaVersion') == '2.0' and
        len(manifest.get('clients', [])) == 60 and
        manifest.get('datasetFingerprint') == preview_fingerprint and
        current_fingerprint == preview_fingerprint and
        Path(manifest.get('root', '')).resolve() == data_root.resolve())
    checks.append(_check(
        'partitionManifest', manifest_ready,
        '客户端划分清单与场景及数据目录一致' if manifest_ready else
        '客户端划分清单缺失或与当前数据不一致，请重新应用场景',
        {'path': str(manifest_path) if manifest_path else '',
         'error': manifest_error,
         'scenarioFingerprint': preview_fingerprint,
         'currentDatasetFingerprint': current_fingerprint,
         'manifestFingerprint': manifest.get('datasetFingerprint')
         if manifest else None}))

    requires_local_model = config['type'] in {'heterogeneity', 'backdoor'}
    model_ready = model_path.is_file() if requires_local_model else True
    checks.append(_check(
        'featureModel', model_ready,
        ('特征模型可用' if model_path.is_file() else
         '当前实验不需要本地特征模型') if model_ready else
        '当前实验需要本地特征模型文件',
        {'path': str(model_path), 'required': requires_local_model}))

    device = config['common']['device']
    device_ready = device == 'cpu'
    cuda_count = 0
    if device == 'cuda':
        try:
            import torch
            cuda_count = torch.cuda.device_count()
            device_ready = torch.cuda.is_available() and cuda_count > 0
        except ImportError:
            device_ready = False
    checks.append(_check(
        'device', device_ready,
        f'{device.upper()} 设备可用' if device_ready else
        '所选 CUDA 设备不可用',
        {'device': device, 'cudaDeviceCount': cuda_count}))

    disk = shutil.disk_usage(workspace)
    minimum_free = 2 * 1024**3
    checks.append(_check(
        'disk', disk.free >= minimum_free,
        '实验输出空间充足' if disk.free >= minimum_free else
        '实验输出目录剩余空间不足 2 GiB',
        {'freeBytes': disk.free, 'requiredBytes': minimum_free,
         'path': str(workspace)}))
    return {
        'ready': all(item['ready'] for item in checks),
        'checks': checks,
        'template': str(template),
        'dataRoot': str(data_root),
        'modelPath': str(model_path),
        'partitionManifest': str(manifest_path) if manifest_path else '',
    }
