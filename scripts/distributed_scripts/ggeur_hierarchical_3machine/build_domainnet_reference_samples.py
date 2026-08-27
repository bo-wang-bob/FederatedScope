#!/usr/bin/env python3
"""Create compact extractor-verification samples from formal train caches."""

import argparse
import json
from pathlib import Path

import numpy as np


SPECS = {
    "domainnet_vit": (
        "domainnet_clipart_clip_ViT_B_16_openai_d512.npz", 512),
    "domainnet_cnn": (
        "domainnet_clipart_cnn_convnext_base_d1024.npz", 1024),
    "domainnet_mixer": (
        "domainnet_clipart_timm_mixer_b16_224_d768.npz", 768),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument(
        "--output-dir", default="exp/domainnet_reference_samples")
    return parser.parse_args()


def normalize(value):
    value = str(value).replace("\\", "/").strip()
    while "//" in value:
        value = value.replace("//", "/")
    return value.lstrip("./").casefold()


def main():
    args = parse_args()
    repo = Path(args.repo).resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for group, (filename, expected_dim) in SPECS.items():
        source = (repo / "exp" / "distributed_feature_cache" /
                  group / filename)
        data = np.load(source, allow_pickle=True)
        paths = np.asarray([normalize(value) for value in data["paths"]])
        features = np.asarray(data["features"])
        if features.ndim != 2 or features.shape[1] != expected_dim:
            raise RuntimeError(
                f"invalid source feature shape for {group}: {features.shape}")
        count = min(max(1, args.samples), len(paths))
        indices = np.linspace(0, len(paths) - 1, num=count, dtype=np.int64)
        target = output_dir / f"{group}_reference.npz"
        with target.open("wb") as handle:
            np.savez(handle, paths=paths[indices], features=features[indices])
        summary[group] = {
            "source": str(source),
            "source_count": len(paths),
            "samples": count,
            "feature_dim": expected_dim,
            "feature_dtype": str(features.dtype),
            "target": str(target),
            "bytes": target.stat().st_size,
        }
        print(json.dumps({
            "event": "reference_complete", "group": group,
            **summary[group],
        }), flush=True)
    marker = output_dir / "reference_samples.json"
    marker.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "event": "complete", "marker": str(marker),
    }), flush=True)


if __name__ == "__main__":
    main()
