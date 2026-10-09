"""Dataset-specific retirement boundaries; never match shared loader filenames."""
from pathlib import Path
import json

# These are resource-relative dataset roots, not generic '*domainnet*' matches.
# Military, privacy and uploaded caches also use the common domainnet loader.
RETIRED_DOMAINNET_PATHS = (
    'datasets/DomainNet',
    'datasets/thirdparty/DomainNet',
    'exp/distributed_manifests/domainnet_4domains',
    'exp/distributed_feature_cache/domainnet_cnn',
    'exp/distributed_feature_cache/domainnet_mixer',
    'exp/distributed_feature_cache/domainnet_vit',
    'exp/thirdparty/manifests/domainnet_manifest.json',
    'exp/thirdparty/manifests/domainnet_label_provenance.json',
    'exp/thirdparty/features/domainnet_cnn',
    'exp/thirdparty/features/domainnet_cnn_labeled',
    'exp/thirdparty/features/domainnet_mixer',
    'exp/thirdparty/features/domainnet_mixer_labeled',
    'exp/thirdparty/features/domainnet_vit',
    'exp/thirdparty/features/domainnet_vit_complete',
)


def check_retired_resources(root):
    """Refuse a package that would silently reintroduce retired dataset assets."""
    root = Path(root)
    remaining = [name for name in RETIRED_DOMAINNET_PATHS
                 if (root / name).exists() or (root / name).is_symlink()]
    registry = root / 'thirdparty_resources.json'
    if registry.is_file():
        entries = json.loads(registry.read_text(encoding='utf-8'))
        if not isinstance(entries, dict):
            raise ValueError('invalid thirdparty_resources.json')
        remaining.extend('thirdparty_resources.json:' + name for name in entries
                         if name.lower().startswith('domainnet_'))
    if remaining:
        raise ValueError('retired DomainNet resources must be removed before packaging: ' + ', '.join(remaining))
