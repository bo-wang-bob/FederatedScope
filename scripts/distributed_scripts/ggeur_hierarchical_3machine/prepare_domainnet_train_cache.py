#!/usr/bin/env python3
"""Build complete manifest-backed DomainNet training feature caches."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from prepare_domainnet_eval_cache import (
    DOMAINS,
    SPECS,
    ImageRecords,
    build_model,
    encode_loader,
    normalize_key,
    resolve,
    sha256_file,
    verify_reference,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--data-root", default="data/DomainNet")
    parser.add_argument(
        "--manifest",
        default=("exp/distributed_manifests/domainnet_4domains/"
                 "domainnet_manifest.json"))
    parser.add_argument("--group", choices=tuple(SPECS), required=True)
    parser.add_argument("--reference-dir",
                        default="exp/domainnet_reference_samples")
    parser.add_argument(
        "--cache-root", default="exp/distributed_feature_cache",
        help="Cache namespace relative to --repo, or an absolute path.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--min-reference-cosine", type=float, default=0.999)
    parser.add_argument("--no-fp16", action="store_true")
    return parser.parse_args()


def train_indices(record_count, train_ratio, seed):
    rng = np.random.RandomState(seed)
    indices = rng.permutation(record_count)
    return indices[:int(record_count * train_ratio)]


def cache_filename(domain, spec):
    if spec["kind"] == "clip":
        prefix = "clip"
        model = f"{spec['model']}_{spec['pretrained']}"
    elif spec["kind"] == "cnn":
        prefix = "cnn"
        model = spec["model"]
    else:
        prefix = "timm"
        model = spec["model"]
    model = model.replace("/", "_").replace("-", "_")
    return f"domainnet_{domain}_{prefix}_{model}_d{spec['dim']}.npz"


def save_cache(path, features, labels, paths):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + f".pid{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez(handle,
                 features=np.asarray(features, dtype=np.float32),
                 labels=np.asarray(labels, dtype=np.int64),
                 paths=np.asarray(paths, dtype=object))
    os.replace(temporary, path)


def validate_cache(path, expected_count, expected_dim):
    data = np.load(path, allow_pickle=True)
    features = np.asarray(data["features"])
    labels = np.asarray(data["labels"])
    paths = [normalize_key(value) for value in data["paths"]]
    failures = []
    if features.shape != (expected_count, expected_dim):
        failures.append(f"features={features.shape}")
    if labels.shape != (expected_count,):
        failures.append(f"labels={labels.shape}")
    if len(paths) != expected_count:
        failures.append(f"paths={len(paths)}")
    if len(set(paths)) != expected_count:
        failures.append(f"unique_paths={len(set(paths))}")
    if not np.isfinite(features).all():
        failures.append("non_finite_features")
    if failures:
        raise RuntimeError(f"invalid cache {path}: {', '.join(failures)}")
    return {
        "path": str(path),
        "count": expected_count,
        "dim": expected_dim,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main():
    args = parse_args()
    repo = Path(args.repo).resolve()
    data_root = resolve(repo, args.data_root)
    manifest_path = resolve(repo, args.manifest)
    reference_dir = resolve(repo, args.reference_dir)
    cache_root = resolve(repo, args.cache_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    records_by_domain = manifest["records"]
    if tuple(manifest.get("domains") or records_by_domain) != DOMAINS:
        raise RuntimeError("DomainNet manifest domain order mismatch")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]),
    ])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fp16 = bool(device.type == "cuda" and not args.no_fp16)
    spec = SPECS[args.group]
    model, encode = build_model(args.group, spec, repo, device)
    verification = verify_reference(
        args.group, encode, transform, reference_dir, data_root,
        device, fp16, args.min_reference_cosine)
    outputs = {}
    for domain in DOMAINS:
        records = records_by_domain[domain]
        indices = train_indices(len(records), args.train_ratio, args.seed)
        target = cache_root / args.group / cache_filename(domain, spec)
        if target.is_file():
            try:
                outputs[domain] = validate_cache(
                    target, len(indices), spec["dim"])
                outputs[domain]["reused"] = True
                print(json.dumps({"event": "train_cache_reused",
                                  "group": args.group, "domain": domain,
                                  **outputs[domain]}), flush=True)
                continue
            except Exception as error:
                print(json.dumps({
                    "event": "invalid_existing_train_cache",
                    "group": args.group, "domain": domain,
                    "path": str(target), "error": repr(error),
                }), flush=True)
        dataset = ImageRecords(data_root, records, indices, transform)
        loader = DataLoader(
            dataset, batch_size=spec["batch_size"], shuffle=False,
            num_workers=max(0, args.num_workers),
            pin_memory=device.type == "cuda", persistent_workers=False)
        features, labels, paths = encode_loader(
            encode, loader, device, fp16)
        save_cache(target, features, labels, paths)
        outputs[domain] = validate_cache(
            target, len(indices), spec["dim"])
        outputs[domain]["reused"] = False
        print(json.dumps({"event": "train_cache_complete",
                          "group": args.group, "domain": domain,
                          **outputs[domain]}), flush=True)

    marker = cache_root / args.group / ".ggeur_feature_cache_ready.json"
    completion = {
        "group": args.group,
        "method": "direct_manifest_train_extraction",
        "client_num": 60,
        "require_complete_feature_cache": True,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "device": str(device),
        "fp16": fp16,
        "reference_verification": verification,
        "domains": outputs,
    }
    marker.write_text(json.dumps(completion, indent=2), encoding="utf-8")
    print(json.dumps({"event": "complete", "marker": str(marker),
                      "group": args.group}), flush=True)


if __name__ == "__main__":
    main()
