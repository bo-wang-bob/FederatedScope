#!/usr/bin/env python3
"""Create deterministic extractor reference bundles for a DomainNet manifest."""

import argparse
import json
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
    resolve,
    sha256_file,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--groups", nargs="+", choices=tuple(SPECS),
                        default=tuple(SPECS))
    parser.add_argument("--samples-per-domain", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-fp16", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    repo = Path(args.repo).resolve()
    data_root = resolve(repo, args.data_root)
    manifest_path = resolve(repo, args.manifest)
    output_dir = resolve(repo, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    records_by_domain = manifest["records"]
    if tuple(manifest.get("domains") or records_by_domain) != DOMAINS:
        raise RuntimeError("DomainNet manifest domain order mismatch")

    chosen = []
    rng = np.random.RandomState(args.seed)
    for domain in DOMAINS:
        records = records_by_domain[domain]
        count = min(args.samples_per_domain, len(records))
        indices = sorted(int(value) for value in
                         rng.choice(len(records), size=count, replace=False))
        chosen.extend(records[index] for index in indices)
    paths = [str(record["path"]) for record in chosen]
    labels = [int(record["label"]) for record in chosen]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]),
    ])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fp16 = bool(device.type == "cuda" and not args.no_fp16)
    dataset = ImageRecords(
        data_root, chosen, np.arange(len(chosen)), transform)
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False,
                        num_workers=0, pin_memory=device.type == "cuda")
    completion = {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "records_sha256": manifest.get("records_sha256"),
        "data_root": str(data_root),
        "paths": paths,
        "labels": labels,
        "groups": {},
        "device": str(device),
        "fp16": fp16,
    }
    for group in args.groups:
        model, encode = build_model(group, SPECS[group], repo, device)
        features, actual_labels, actual_paths = encode_loader(
            encode, loader, device, fp16)
        if list(actual_labels) != labels or list(actual_paths) != [
                str(value).replace("\\", "/").strip().lstrip("./").casefold()
                for value in paths]:
            raise RuntimeError(f"reference order mismatch for {group}")
        target = output_dir / f"{group}_reference.npz"
        temporary = Path(str(target) + ".tmp")
        with temporary.open("wb") as handle:
            np.savez(handle, paths=np.asarray(paths, dtype=object),
                     labels=np.asarray(labels, dtype=np.int64),
                     features=np.asarray(features, dtype=np.float32))
        temporary.replace(target)
        completion["groups"][group] = {
            "path": str(target),
            "samples": len(paths),
            "shape": list(features.shape),
            "sha256": sha256_file(target),
        }
        print(json.dumps({"event": "reference_complete", "group": group,
                          **completion["groups"][group]}), flush=True)
        model.to("cpu")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    marker = output_dir / "reference_completion.json"
    marker.write_text(json.dumps(completion, indent=2), encoding="utf-8")
    print(json.dumps({"event": "complete", "marker": str(marker)}),
          flush=True)


if __name__ == "__main__":
    main()
