#!/usr/bin/env python3
"""Build verified DomainNet held-out feature caches on the permitted GPU host."""

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DOMAINS = ("clipart", "painting", "real", "sketch")
SPECS = {
    "domainnet_vit": {
        "kind": "clip",
        "model": "ViT-B-16",
        "pretrained": "openai",
        "dim": 512,
        "checkpoint": "pretrained_models/ViT-B-16.pt",
        "batch_size": 128,
    },
    "domainnet_cnn": {
        "kind": "cnn",
        "model": "convnext_base",
        "dim": 1024,
        "checkpoint": "pretrained_models/convnext_base-6075fbad.pth",
        "batch_size": 64,
    },
    "domainnet_mixer": {
        "kind": "timm",
        "model": "mixer_b16_224",
        "dim": 768,
        "checkpoint": "pretrained_models/mixer_b16_224_complete.pth",
        "batch_size": 128,
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=str(REPO_ROOT))
    parser.add_argument("--data-root", default="data/DomainNet")
    parser.add_argument(
        "--manifest",
        default="exp/distributed_manifests/domainnet_4domains/"
                "domainnet_manifest.json")
    parser.add_argument(
        "--groups", nargs="+", choices=tuple(SPECS),
        default=tuple(SPECS))
    parser.add_argument("--reference-dir", default="exp/domainnet_reference_samples")
    parser.add_argument(
        "--cache-root", default="exp/distributed_feature_cache",
        help="Cache namespace relative to --repo, or an absolute path.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.0)
    parser.add_argument("--min-reference-cosine", type=float, default=0.999)
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def resolve(repo, value):
    path = Path(value)
    return path if path.is_absolute() else repo / path


def normalize_key(value):
    value = str(value).replace("\\", "/").strip()
    while "//" in value:
        value = value.replace("//", "/")
    return value.lstrip("./").casefold()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_indices(record_count, train_ratio, val_ratio, seed):
    # Match DomainNet._split_records, which uses NumPy's legacy MT19937 API.
    rng = np.random.RandomState(seed)
    indices = rng.permutation(record_count)
    train_size = int(record_count * train_ratio)
    val_size = int(record_count * val_ratio)
    return indices[train_size + val_size:]


class ImageRecords(Dataset):
    def __init__(self, data_root, records, indices, transform):
        self.data_root = Path(data_root)
        self.records = records
        self.indices = [int(index) for index in indices]
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        record = self.records[self.indices[item]]
        relative = str(record["path"])
        path = Path(relative)
        if not path.is_absolute():
            path = self.data_root / path
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, int(record["label"]), normalize_key(relative)


def build_model(group, spec, repo, device):
    checkpoint = resolve(repo, spec["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if spec["kind"] == "clip":
        import open_clip
        model, _, _ = open_clip.create_model_and_transforms(
            spec["model"], pretrained=str(checkpoint))

        def encode(images):
            return model.encode_image(images)
    elif spec["kind"] == "cnn":
        from federatedscope.contrib.model.ggeur_cnn_extractor import (
            CNNFeatureExtractor,
        )
        model = CNNFeatureExtractor(
            model_name=spec["model"], pretrained=False, freeze=True,
            checkpoint_path=str(checkpoint))

        def encode(images):
            return model(images)
    else:
        from federatedscope.contrib.model.ggeur_timm_extractor import (
            TimmFeatureExtractor,
        )
        model = TimmFeatureExtractor(
            model_name=spec["model"], pretrained=False, freeze=True,
            checkpoint_path=str(checkpoint), in_chans=3, global_pool="avg")

        def encode(images):
            return model(images)
    model = model.to(device)
    model.eval()
    print(json.dumps({
        "event": "model_ready",
        "group": group,
        "checkpoint": str(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": sha256_file(checkpoint),
        "device": str(device),
    }), flush=True)
    return model, encode


def autocast_context(device, fp16):
    if fp16 and device.type == "cuda":
        return torch.cuda.amp.autocast(dtype=torch.float16)
    return nullcontext()


@torch.no_grad()
def encode_loader(encode, loader, device, fp16):
    feature_batches = []
    label_batches = []
    path_batches = []
    total = len(loader.dataset)
    done = 0
    for images, labels, paths in loader:
        images = images.to(device, non_blocking=True)
        with autocast_context(device, fp16):
            features = encode(images)
        features = features.detach().float().cpu().numpy()
        feature_batches.append(features)
        label_batches.append(labels.numpy().astype(np.int64, copy=False))
        path_batches.extend(list(paths))
        done += len(paths)
        print(json.dumps({
            "event": "encode_progress", "done": done, "total": total,
        }), flush=True)
    return (np.concatenate(feature_batches, axis=0),
            np.concatenate(label_batches, axis=0), path_batches)


def reference_path(reference_dir, group):
    return reference_dir / f"{group}_reference.npz"


def verify_reference(group, encode, transform, reference_dir, data_root,
                     device, fp16, min_cosine):
    path = reference_path(reference_dir, group)
    if not path.is_file():
        raise FileNotFoundError(
            f"reference verification bundle is missing: {path}")
    data = np.load(path, allow_pickle=True)
    paths = [str(value) for value in data["paths"]]
    expected = np.asarray(data["features"], dtype=np.float32)
    records = [{"path": value, "label": 0} for value in paths]
    dataset = ImageRecords(
        data_root, records, np.arange(len(records)), transform)
    loader = DataLoader(dataset, batch_size=min(32, len(dataset)),
                        shuffle=False, num_workers=0)
    actual, _, actual_paths = encode_loader(
        encode, loader, device, fp16)
    if [normalize_key(value) for value in paths] != list(actual_paths):
        raise RuntimeError(f"reference path order mismatch for {group}")
    if actual.shape != expected.shape:
        raise RuntimeError(
            f"reference shape mismatch for {group}: "
            f"actual={actual.shape}, expected={expected.shape}")
    denominator = (np.linalg.norm(actual, axis=1) *
                   np.linalg.norm(expected, axis=1))
    cosine = np.sum(actual * expected, axis=1) / np.maximum(
        denominator, 1e-12)
    max_abs = float(np.max(np.abs(actual - expected)))
    result = {
        "event": "reference_verified",
        "group": group,
        "samples": len(paths),
        "cosine_min": float(np.min(cosine)),
        "cosine_mean": float(np.mean(cosine)),
        "max_abs": max_abs,
        "threshold": min_cosine,
    }
    print(json.dumps(result), flush=True)
    if result["cosine_min"] < min_cosine:
        raise RuntimeError(
            f"reference verification failed for {group}: {result}")
    return result


def cache_filename(domain, spec, seed, train_ratio, val_ratio):
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
    train = int(train_ratio * 100)
    val = int(val_ratio * 100)
    test = int(100 - train_ratio * 100 - val_ratio * 100)
    suffix = "union_" + "-".join(DOMAINS)
    return (f"domainnet_{domain}_test_{prefix}_{model}_"
            f"split{train}_{val}_{test}_seed{seed}_{suffix}.npz")


def save_cache(path, features, labels, class_count):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + f".pid{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, features=np.asarray(features, dtype=np.float32),
                 labels=np.asarray(labels, dtype=np.int64),
                 class_count=np.int64(class_count),
                 shared_classes_only=np.int64(0),
                 selected_domains="|".join(DOMAINS))
    os.replace(temporary, path)


def main():
    args = parse_args()
    repo = Path(args.repo).resolve()
    data_root = resolve(repo, args.data_root)
    manifest_path = resolve(repo, args.manifest)
    reference_dir = resolve(repo, args.reference_dir)
    cache_root = resolve(repo, args.cache_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    classes = list(manifest["classes"])
    records_by_domain = manifest["records"]
    if tuple(manifest.get("domains") or records_by_domain) != DOMAINS:
        raise RuntimeError("DomainNet manifest domain order does not match formal config")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]),
    ])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fp16 = bool(device.type == "cuda" and not args.no_fp16)
    completion = {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "data_root": str(data_root),
        "domains": {},
        "groups": {},
        "seed": args.seed,
        "splits": [args.train_ratio, args.val_ratio,
                   1.0 - args.train_ratio - args.val_ratio],
        "device": str(device),
        "fp16": fp16,
    }

    for group in args.groups:
        spec = SPECS[group]
        model, encode = build_model(group, spec, repo, device)
        verification = verify_reference(
            group, encode, transform, reference_dir, data_root,
            device, fp16, args.min_reference_cosine)
        group_result = {"verification": verification, "domains": {}}
        if args.verify_only:
            completion["groups"][group] = group_result
            model.to("cpu")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        for domain in DOMAINS:
            records = records_by_domain[domain]
            indices = test_indices(
                len(records), args.train_ratio, args.val_ratio, args.seed)
            expected_count = len(records) - int(
                len(records) * args.train_ratio) - int(
                len(records) * args.val_ratio)
            if len(indices) != expected_count:
                raise RuntimeError(f"split count mismatch for {domain}")
            cache_dir = cache_root / group
            target = cache_dir / cache_filename(
                domain, spec, args.seed, args.train_ratio, args.val_ratio)
            if target.is_file():
                existing = np.load(target)
                shape = tuple(existing["features"].shape)
                if shape == (expected_count, spec["dim"]):
                    print(json.dumps({
                        "event": "cache_reused", "group": group,
                        "domain": domain, "path": str(target),
                        "shape": shape,
                    }), flush=True)
                    group_result["domains"][domain] = {
                        "count": expected_count, "path": str(target),
                        "bytes": target.stat().st_size,
                        "sha256": sha256_file(target), "reused": True,
                    }
                    continue
            dataset = ImageRecords(data_root, records, indices, transform)
            loader = DataLoader(
                dataset, batch_size=spec["batch_size"], shuffle=False,
                num_workers=max(0, args.num_workers),
                pin_memory=device.type == "cuda", persistent_workers=False)
            features, labels, _ = encode_loader(
                encode, loader, device, fp16)
            if features.shape != (expected_count, spec["dim"]):
                raise RuntimeError(
                    f"invalid {group}/{domain} output shape: {features.shape}")
            save_cache(target, features, labels, len(classes))
            group_result["domains"][domain] = {
                "count": expected_count, "path": str(target),
                "bytes": target.stat().st_size,
                "sha256": sha256_file(target), "reused": False,
            }
            print(json.dumps({
                "event": "cache_complete", "group": group,
                "domain": domain, **group_result["domains"][domain],
            }), flush=True)
        completion["groups"][group] = group_result
        model.to("cpu")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    marker_name = ("domainnet_eval_cache_verification.json"
                   if args.verify_only else
                   "domainnet_eval_cache_completion.json")
    marker = cache_root / marker_name
    marker.write_text(json.dumps(completion, indent=2), encoding="utf-8")
    print(json.dumps({
        "event": ("verification_complete" if args.verify_only else "complete"),
        "marker": str(marker),
        "groups": list(completion["groups"]),
    }), flush=True)


if __name__ == "__main__":
    main()
