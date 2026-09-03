#!/usr/bin/env python3
"""Evaluate deterministic multi-centroid heads on cached DomainNet features.

Only training features are used to construct the centroids.  Held-out test
features are read after construction and are used solely for final scoring.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


DOMAINS = ("clipart", "painting", "real", "sketch")
GROUPS = {
    "domainnet_vit": (512, "clip_ViT_B_16_openai_d512"),
    "domainnet_cnn": (1024, "cnn_convnext_base_d1024"),
    "domainnet_mixer": (768, "timm_mixer_b16_224_d768"),
}


def _one(cache_dir, pattern):
    matches = [
        path for path in cache_dir.glob(pattern)
        if ".part" not in path.name and "_terminal_client" not in path.name
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {pattern!r}, found {matches}")
    return matches[0]


def _normalize(values):
    return values / np.maximum(
        np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def _spherical_kmeans(features, cluster_count, seed, iterations=12):
    values = _normalize(np.asarray(features, dtype=np.float32))
    if values.shape[0] <= cluster_count:
        return values
    rng = np.random.default_rng(seed)
    centers = [values[int(rng.integers(values.shape[0]))]]
    best = 1.0 - values @ centers[0]
    for _ in range(1, cluster_count):
        index = int(np.argmax(best))
        centers.append(values[index])
        best = np.minimum(best, 1.0 - values @ centers[-1])
    centers = np.stack(centers)
    for _ in range(iterations):
        assignment = np.argmax(values @ centers.T, axis=1)
        updated = []
        for cluster in range(cluster_count):
            members = values[assignment == cluster]
            if members.size == 0:
                updated.append(centers[cluster])
            else:
                updated.append(members.mean(axis=0))
        updated = _normalize(np.stack(updated).astype(np.float32))
        if np.array_equal(np.argmax(values @ updated.T, axis=1), assignment):
            centers = updated
            break
        centers = updated
    return centers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--group", choices=tuple(GROUPS), required=True)
    parser.add_argument("--clusters", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir).resolve()
    feature_dim, suffix = GROUPS[args.group]
    class_count = 345
    centroids = []
    centroid_labels = []
    train_samples = 0
    test_files = {}
    for domain_index, domain in enumerate(DOMAINS):
        train_path = _one(cache_dir, f"domainnet_{domain}_{suffix}.npz")
        test_files[domain] = _one(cache_dir, f"domainnet_{domain}_test_*.npz")
        with np.load(train_path, allow_pickle=False) as data:
            features = np.asarray(data["features"], dtype=np.float32)
            labels = np.asarray(data["labels"], dtype=np.int64)
        train_samples += int(labels.size)
        for class_index in range(class_count):
            class_features = features[labels == class_index]
            if class_features.size == 0:
                raise RuntimeError(f"missing {domain}/{class_index}")
            centers = _spherical_kmeans(
                class_features, args.clusters,
                seed=42 + domain_index * class_count + class_index)
            centroids.append(centers)
            centroid_labels.extend([class_index] * centers.shape[0])
    centroid_array = np.concatenate(centroids, axis=0).astype(np.float32)
    centroid_labels = np.asarray(centroid_labels, dtype=np.int64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    centroid_tensor = torch.from_numpy(centroid_array).to(device)
    label_tensor = torch.from_numpy(centroid_labels).to(device)
    totals = {"correct": 0, "total": 0, "domains": {}}
    for domain, test_path in test_files.items():
        with np.load(test_path, allow_pickle=False) as data:
            features = np.asarray(data["features"], dtype=np.float32)
            labels = np.asarray(data["labels"], dtype=np.int64)
        correct = 0
        for offset in range(0, labels.size, args.batch_size):
            batch = torch.from_numpy(
                features[offset:offset + args.batch_size]).to(device)
            scores = batch @ centroid_tensor.T
            best_scores = torch.full(
                (batch.shape[0], class_count), -torch.inf, device=device)
            best_scores.scatter_reduce_(
                1, label_tensor.expand(batch.shape[0], -1), scores,
                reduce="amax", include_self=True)
            prediction = torch.argmax(best_scores, dim=1).cpu().numpy()
            correct += int(np.sum(
                prediction == labels[offset:offset + args.batch_size]))
        total = int(labels.size)
        totals["correct"] += correct
        totals["total"] += total
        totals["domains"][domain] = {
            "correct": correct,
            "total": total,
            "accuracy": correct / total,
        }
    totals["weighted_accuracy"] = totals["correct"] / totals["total"]
    totals["equal_domain_accuracy"] = float(np.mean([
        item["accuracy"] for item in totals["domains"].values()
    ]))
    result = {
        "status": "pass",
        "group": args.group,
        "cache_dir": str(cache_dir),
        "train_samples": train_samples,
        "test_samples": totals["total"],
        "clusters_per_domain_class": args.clusters,
        "centroid_count": int(centroid_array.shape[0]),
        "feature_dim": feature_dim,
        "device": str(device),
        "metrics": totals,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
