"""Validate the deployment dependency contract without starting a service."""
import argparse
import importlib
from importlib import metadata
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def check_environment(gpu=False, strict=False):
    os.environ['FEDERATEDSCOPE_GGEUR_LIGHTWEIGHT'] = '1'
    from federatedscope.standalone_api.platform_offline import configure_offline_worker
    configure_offline_worker()
    errors, versions = [], {}
    if strict and sys.version_info[:2] != (3, 9):
        errors.append('发布环境要求 CPython 3.9')
    for line in (REPO / 'deploy/requirements-runtime.lock').read_text().splitlines():
        if not line or line.startswith('#'):
            continue
        name, expected = line.split('==')
        try:
            actual = metadata.version(name)
            versions[name] = actual
            if strict and actual != expected:
                errors.append(f'{name}: {actual} != {expected}')
        except metadata.PackageNotFoundError:
            errors.append(f'缺少依赖 {name}=={expected}')
    try:
        from packaging.requirements import Requirement
        from packaging.utils import canonicalize_name
        pinned_names = {canonicalize_name(name) for name in versions}
        for name in versions:
            for entry in metadata.requires(name) or []:
                required = Requirement(entry)
                if required.marker and not required.marker.evaluate({'extra': ''}):
                    continue
                if canonicalize_name(required.name) not in pinned_names:
                    errors.append(f'{name} 的依赖未锁定或未安装：{required}')
                elif metadata.version(required.name) not in required.specifier:
                    errors.append(f'{name} 的依赖版本冲突：{required}')
    except Exception as error:
        errors.append('依赖闭包检查失败：' + str(error))
    modules = [
        'federatedscope.standalone_api.platform_app',
        'federatedscope.core.configs.config',
        'federatedscope.core.auxiliaries.data_builder',
        'federatedscope.core.auxiliaries.runner_builder',
        'federatedscope.contrib.worker.ggeur_client',
        'federatedscope.contrib.worker.ggeur_server',
        'scipy', 'sklearn', 'fvcore', 'pympler',
        'open_clip', 'timm', 'safetensors',
    ]
    for module in modules:
        try:
            importlib.import_module(module)
        except Exception as error:
            errors.append(f'{module}: {error}')
    device = None
    try:
        import torch
        if gpu and not torch.cuda.is_available():
            raise RuntimeError('CUDA 不可用；检查驱动和 NVIDIA Container Toolkit')
        device = 'cuda:0' if gpu else 'cpu'
        values = torch.arange(16, dtype=torch.float32, device=device).reshape(4, 4)
        actual = (values @ values.T).cpu()
        expected = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        if not torch.equal(actual, expected @ expected.T):
            raise RuntimeError('张量计算校验失败')
        if gpu:
            torch.cuda.synchronize()
            device = torch.cuda.get_device_name(0)
    except Exception as error:
        errors.append(str(error))
    return dict(ok=not errors, python=sys.version, device=device, versions=versions, errors=errors)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--strict', action='store_true')
    args = parser.parse_args()
    report = check_environment(args.gpu, args.strict)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
