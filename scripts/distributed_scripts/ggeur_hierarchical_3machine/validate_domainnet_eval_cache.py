"""Validate the transferred held-out DomainNet feature caches."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    repo = Path(sys.argv[1]).resolve()
    root = repo / "exp" / "distributed_feature_cache"
    marker_path = root / "domainnet_eval_cache_completion.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8-sig"))
    expected_dims = {
        "domainnet_vit": 512,
        "domainnet_cnn": 1024,
        "domainnet_mixer": 768,
    }
    domains = ("clipart", "painting", "real", "sketch")
    results = []
    for group, dim in expected_dims.items():
        recorded = marker["groups"][group]["domains"]
        if set(recorded) != set(domains):
            raise RuntimeError(
                f"invalid completion domains for {group}: {recorded}")
        group_dir = root / group
        files = sorted(group_dir.glob("domainnet_*_test_*.npz"))
        if len(files) != 4:
            raise RuntimeError(
                f"expected four test caches for {group}, found {len(files)}")
        for domain in domains:
            matches = list(group_dir.glob(f"domainnet_{domain}_test_*.npz"))
            if len(matches) != 1:
                raise RuntimeError(
                    f"expected one {group}/{domain} cache, "
                    f"found {len(matches)}")
            path = matches[0]
            item = recorded[domain]
            digest = sha256_file(path)
            if digest != item["sha256"]:
                raise RuntimeError(f"hash mismatch for {path}")
            with np.load(path, allow_pickle=False) as data:
                features = data["features"]
                labels = data["labels"]
                expected = int(item["count"])
                if features.shape != (expected, dim):
                    raise RuntimeError(
                        f"shape mismatch for {path}: {features.shape}")
                if labels.shape != (expected,):
                    raise RuntimeError(
                        f"label mismatch for {path}: {labels.shape}")
                if int(data["class_count"]) != 345:
                    raise RuntimeError(f"class_count mismatch for {path}")
            results.append({
                "group": group,
                "domain": domain,
                "count": expected,
                "dim": dim,
                "bytes": path.stat().st_size,
                "sha256": digest,
            })
    print(json.dumps({
        "status": "pass",
        "files": len(results),
        "results": results,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
