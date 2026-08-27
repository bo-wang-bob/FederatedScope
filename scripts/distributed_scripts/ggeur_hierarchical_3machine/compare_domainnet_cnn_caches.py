#!/usr/bin/env python3
"""Compare active and previously validated DomainNet CNN feature caches."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


DOMAINS = ("clipart", "painting", "real", "sketch")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe(path):
    result = {"path": str(path), "exists": path.is_file()}
    if path.is_file():
        result.update(bytes=path.stat().st_size, sha256=sha256(path))
    return result


def compare_npz(active_path, reference_path):
    result = {}
    with np.load(active_path, allow_pickle=True) as active, \
            np.load(reference_path, allow_pickle=True) as reference:
        result["keys_active"] = sorted(active.files)
        result["keys_reference"] = sorted(reference.files)
        for key in ("features", "labels", "paths"):
            if key not in active.files or key not in reference.files:
                continue
            left = np.asarray(active[key])
            right = np.asarray(reference[key])
            item = {
                "shape_active": list(left.shape),
                "shape_reference": list(right.shape),
                "dtype_active": str(left.dtype),
                "dtype_reference": str(right.dtype),
            }
            if left.shape == right.shape:
                item["array_equal"] = bool(np.array_equal(left, right))
                if np.issubdtype(left.dtype, np.number):
                    delta = np.abs(left.astype(np.float64) -
                                   right.astype(np.float64))
                    item["max_abs"] = (
                        float(delta.max()) if delta.size else 0.0)
                    item["mean_abs"] = (
                        float(delta.mean()) if delta.size else 0.0)
            result[key] = item
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    args = parser.parse_args()
    repo = Path(args.repo)
    active = repo / "exp" / "distributed_feature_cache" / "domainnet_cnn"
    reference = (repo / "exp" /
                 "distributed_feature_cache_domainnet_original_20260804" /
                 "domainnet_cnn")

    report = {
        "active_dir": str(active),
        "reference_dir": str(reference),
        "manifests": [],
        "domains": {},
    }
    for path in (
        repo / "exp" / "distributed_manifests" / "domainnet_4domains" /
            "domainnet_manifest.json",
        repo / "data_staging" / "domainnet_original_4domains_20260804" /
            "domainnet_manifest.json",
    ):
        report["manifests"].append(describe(path))

    for domain in DOMAINS:
        test_name = (
            f"domainnet_{domain}_test_cnn_convnext_base_"
            "split70_0_30_seed42_union_clipart-painting-real-sketch.npz")
        train_name = f"domainnet_{domain}_cnn_convnext_base_d1024.npz"
        active_test = active / test_name
        reference_test = reference / test_name
        active_train = active / train_name
        reference_train = reference / train_name
        entry = {
            "active_test": describe(active_test),
            "reference_test": describe(reference_test),
            "active_train": describe(active_train),
            "reference_train": describe(reference_train),
        }
        if active_test.is_file() and reference_test.is_file():
            entry["test_comparison"] = compare_npz(
                active_test, reference_test)
        if active_train.is_file() and reference_train.is_file():
            entry["train_comparison"] = compare_npz(
                active_train, reference_train)
        report["domains"][domain] = entry

    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
