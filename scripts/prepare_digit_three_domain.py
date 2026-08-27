#!/usr/bin/env python3
"""Prepare EMNIST Digits, USPS and SVHN as a portable FL dataset.

The script exports a balanced, deterministic sample of each public dataset to
PNG files, then creates one disjoint Dirichlet training partition per logical
client.  The same manifests are consumed by FedAvg, FedProx and Ours so the
accuracy comparison cannot be affected by a new random split.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from torchvision import datasets


DOMAINS = ("emnist_digits", "usps", "svhn")
CLASSES = tuple(range(10))


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8")


def image_for_dataset(dataset, index, domain):
    if domain == "svhn":
        image = Image.fromarray(np.transpose(dataset.data[index], (1, 2, 0)))
    else:
        image = Image.fromarray(np.asarray(dataset.data[index], dtype=np.uint8))
    if domain == "emnist_digits":
        # EMNIST stores glyphs transposed relative to the human-readable form.
        image = ImageOps.mirror(image.rotate(-90, expand=True))
    return image.convert("RGB")


def targets_for_dataset(dataset, domain):
    raw = dataset.labels if domain == "svhn" else dataset.targets
    if hasattr(raw, "numpy"):
        raw = raw.numpy()
    return np.asarray(raw, dtype=np.int64) % 10


def select_balanced_indices(targets, per_class, seed):
    rng = np.random.RandomState(seed)
    selected = []
    for label in CLASSES:
        candidates = np.flatnonzero(targets == label)
        if len(candidates) < per_class:
            raise ValueError(
                f"Class {label} has only {len(candidates)} samples; "
                f"requested {per_class}")
        selected.extend(
            rng.choice(candidates, size=per_class, replace=False).tolist())
    rng.shuffle(selected)
    return selected


def allocate_dirichlet(records,
                       client_ids,
                       alpha,
                       seed,
                       min_client_samples,
                       max_attempts=10000):
    """Allocate every record exactly once with class-wise Dirichlet draws."""
    if alpha <= 0:
        raise ValueError("Dirichlet alpha must be positive")
    labels = np.asarray([record["label"] for record in records], dtype=int)
    num_clients = len(client_ids)
    for attempt in range(max_attempts):
        rng = np.random.RandomState(seed + attempt)
        buckets = {client_id: [] for client_id in client_ids}
        for label in CLASSES:
            class_indices = np.flatnonzero(labels == label)
            rng.shuffle(class_indices)
            proportions = rng.dirichlet([alpha] * num_clients)
            counts = rng.multinomial(len(class_indices), proportions)
            cursor = 0
            for client_id, count in zip(client_ids, counts):
                chosen = class_indices[cursor:cursor + count]
                buckets[client_id].extend(records[idx] for idx in chosen)
                cursor += int(count)
        sizes = [len(buckets[client_id]) for client_id in client_ids]
        if min(sizes) >= min_client_samples:
            for client_id in client_ids:
                rng.shuffle(buckets[client_id])
            return buckets, attempt
    raise RuntimeError(
        f"Could not produce a partition with at least {min_client_samples} "
        f"samples/client after {max_attempts} attempts; reduce the minimum")


def allocate_balanced_test(records, client_ids, seed):
    labels = np.asarray([record["label"] for record in records], dtype=int)
    rng = np.random.RandomState(seed)
    buckets = {client_id: [] for client_id in client_ids}
    for label in CLASSES:
        class_indices = np.flatnonzero(labels == label)
        rng.shuffle(class_indices)
        for offset, record_index in enumerate(class_indices):
            client_id = client_ids[offset % len(client_ids)]
            buckets[client_id].append(records[record_index])
    return buckets


def count_classes(records):
    counts = {str(label): 0 for label in CLASSES}
    for record in records:
        counts[str(int(record["label"]))] += 1
    return counts


def export_records(dataset, domain, split, indices, output_root):
    targets = targets_for_dataset(dataset, domain)
    records = []
    for source_index in indices:
        label = int(targets[source_index])
        relative = (Path("images") / domain / split / str(label) /
                    f"{int(source_index):08d}.png")
        destination = output_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            image_for_dataset(dataset, source_index, domain).save(
                destination, optimize=True)
        records.append({
            "path": relative.as_posix(),
            "label": label,
            "source_index": int(source_index),
        })
    return records


def load_sources(raw_root, download):
    return {
        "emnist_digits": {
            "train": datasets.EMNIST(raw_root,
                                     split="digits",
                                     train=True,
                                     download=download),
            "test": datasets.EMNIST(raw_root,
                                    split="digits",
                                    train=False,
                                    download=download),
        },
        "usps": {
            "train": datasets.USPS(raw_root,
                                   train=True,
                                   download=download),
            "test": datasets.USPS(raw_root,
                                  train=False,
                                  download=download),
        },
        "svhn": {
            "train": datasets.SVHN(raw_root,
                                   split="train",
                                   download=download),
            "test": datasets.SVHN(raw_root,
                                  split="test",
                                  download=download),
        },
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root",
                        default="data/digit_three_domain")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--client-num", type=int, default=60)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-per-class", type=int, default=200)
    parser.add_argument("--test-per-class", type=int, default=50)
    parser.add_argument("--min-client-samples", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.client_num < 3 or args.client_num % len(DOMAINS) != 0:
        raise ValueError("client-num must be a positive multiple of 3")

    output_root = Path(args.output_root).resolve()
    raw_root = output_root / "_raw"
    sources = load_sources(raw_root, args.download)
    clients_per_domain = args.client_num // len(DOMAINS)
    all_records = {}
    client_manifests = {}
    partition_attempts = {}

    for domain_index, domain in enumerate(DOMAINS):
        domain_records = {}
        for split_index, split in enumerate(("train", "test")):
            dataset = sources[domain][split]
            targets = targets_for_dataset(dataset, domain)
            per_class = (args.train_per_class if split == "train" else
                         args.test_per_class)
            indices = select_balanced_indices(
                targets, per_class,
                args.seed + domain_index * 100 + split_index)
            domain_records[split] = export_records(
                dataset, domain, split, indices, output_root)
        all_records[domain] = domain_records

        start = domain_index * clients_per_domain + 1
        client_ids = list(range(start, start + clients_per_domain))
        train_buckets, attempts = allocate_dirichlet(
            domain_records["train"], client_ids, args.alpha,
            args.seed + domain_index * 1000, args.min_client_samples)
        test_buckets = allocate_balanced_test(
            domain_records["test"], client_ids,
            args.seed + domain_index * 1000 + 500)
        partition_attempts[domain] = attempts

        for client_id in client_ids:
            client_manifests[client_id] = {
                "version": 1,
                "dataset": "digits-3domain",
                "root": output_root.as_posix(),
                "client_id": client_id,
                "domain": domain,
                "classes": [str(label) for label in CLASSES],
                "partition": {
                    "type": "class-wise-dirichlet",
                    "alpha": args.alpha,
                    "seed": args.seed,
                    "disjoint_train_records": True,
                },
                "splits": {
                    "train": train_buckets[client_id],
                    "val": [],
                    "test": test_buckets[client_id],
                },
            }

    global_manifest_path = output_root / "dataset_manifest.json"
    global_manifest = {
        "version": 1,
        "dataset": "digits-3domain",
        "domains": list(DOMAINS),
        "domain_descriptions": {
            "emnist_digits": "EMNIST Digits handwritten glyphs",
            "usps": "USPS scanned postal digits",
            "svhn": "SVHN color street-view house numbers",
        },
        "classes": [str(label) for label in CLASSES],
        "sampling": {
            "seed": args.seed,
            "train_per_class_per_domain": args.train_per_class,
            "test_per_class_per_domain": args.test_per_class,
        },
        "records": all_records,
    }
    write_json(global_manifest_path, global_manifest)

    for client_id, manifest in client_manifests.items():
        manifest_path = (output_root / "manifests" /
                         f"client_{client_id:06d}" / "client_manifest.json")
        write_json(manifest_path, manifest)

    manifest_sha256 = hashlib.sha256(
        global_manifest_path.read_bytes()).hexdigest()
    summary = {
        "dataset": "digits-3domain",
        "domains": list(DOMAINS),
        "classes": [str(label) for label in CLASSES],
        "client_num": args.client_num,
        "clients_per_domain": clients_per_domain,
        "dirichlet_alpha": args.alpha,
        "seed": args.seed,
        "global_manifest": global_manifest_path.as_posix(),
        "global_manifest_sha256": manifest_sha256,
        "partition_retry_count": partition_attempts,
        "clients": {},
    }
    for client_id, manifest in client_manifests.items():
        train_records = manifest["splits"]["train"]
        test_records = manifest["splits"]["test"]
        summary["clients"][str(client_id)] = {
            "domain": manifest["domain"],
            "train_samples": len(train_records),
            "test_samples": len(test_records),
            "train_class_counts": count_classes(train_records),
            "test_class_counts": count_classes(test_records),
        }
    write_json(output_root / "partition_summary.json", summary)

    csv_path = output_root / "client_class_distribution.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["client_id", "domain", "split"] +
                        [f"class_{label}" for label in CLASSES] + ["total"])
        for client_id, manifest in sorted(client_manifests.items()):
            for split in ("train", "test"):
                counts = count_classes(manifest["splits"][split])
                values = [counts[str(label)] for label in CLASSES]
                writer.writerow([client_id, manifest["domain"], split] +
                                values + [sum(values)])

    print(json.dumps({
        "output_root": output_root.as_posix(),
        "client_num": args.client_num,
        "domains": list(DOMAINS),
        "global_manifest_sha256": manifest_sha256,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
