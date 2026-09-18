"""Portable platform paths. Persist paths relative to the backend directory."""
import copy
import os
from pathlib import Path, PurePosixPath
from .paths import project_path


def resolve_path(base, value):
    return project_path(value, base)


def relative_path(base, value):
    return Path(os.path.relpath(resolve_path(base, value), Path(base).resolve())).as_posix()


def portable_config(config, repo, *, path_base=None):
    """Read paths from repo and persist them relative to path_base (repo by default)."""
    result = copy.deepcopy(config)
    path_base = resolve_path(repo, path_base) if path_base is not None else Path(repo).resolve()
    fields = [(result, 'outdir'), (result, 'log_file'), (result.get('data', {}), 'root')]
    fields += [(result.get('federate', {}), key) for key in ('save_to', 'restore_from')]
    g = result.get('ggeur', {})
    fields += [(g, key) for key in (
        'feature_cache_dir', 'augmented_feature_cache_dir', 'mlp_checkpoint_dir',
        'training_distribution_dir', 'clip_model_path', 'cnn_checkpoint_path',
        'timm_checkpoint_path', 'task_adaptation_file', 'domainnet_manifest_path',
        'digit3_manifest_base', 'digit3_manifest_path', 'digit3_global_manifest_path',
        'bert_model_path', 'bert_tokenizer_path')]
    for mapping, key in fields:
        if mapping.get(key):
            mapping[key] = relative_path(path_base, resolve_path(repo, mapping[key]))
    return result


def cache_file(root, stored, source_id=None):
    """Accept relative cache members and exact legacy job-cache locations."""
    root = Path(root).resolve()
    text = str(stored).replace('\\', '/')
    path = PurePosixPath(text)
    absolute = path.is_absolute() or (len(text) > 1 and text[1] == ':')
    if absolute:
        if source_id is None or path.parent.name != 'augmented_cache' or path.parent.parent.name != source_id:
            raise ValueError('缓存来源路径越界')
        path = PurePosixPath(path.name)
    if not path.parts or any(part in ('.', '..') for part in path.parts) or ':' in str(path):
        raise ValueError('缓存来源路径越界')
    candidate = root.joinpath(*path.parts)
    resolved = candidate.resolve()
    if root not in resolved.parents or candidate.is_symlink():
        raise ValueError('缓存来源路径越界')
    return resolved


def portable_backbone(backbone):
    """Frozen-feature identity must not depend on a machine's weight location."""
    result = dict(backbone)
    if result.get('feature_extractor') == 'clip' and result.get('model') == 'ViT-B-16':
        result['checkpoint'] = 'models/ViT-B-16.pt'
    return result


def compatible_backbones(checkpoint, bundle):
    a, b = checkpoint['backbone'], bundle['backbone']
    if a == b:
        return True
    # API also checks feature-space hashes. Old/new artifacts must explicitly
    # share that exact cache identity before ignoring a filesystem location.
    return (checkpoint.get('featureSpace') is not None
            and checkpoint['featureSpace'] == bundle.get('featureSpace')
            and portable_backbone(a) == portable_backbone(b))
