#!/usr/bin/env python3
"""Split deterministic DomainNet held-out caches across logical clients."""

import argparse
import json
from pathlib import Path

import numpy as np


DOMAINS = ("clipart", "painting", "real", "sketch")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--client-num", type=int, default=60)
    return parser.parse_args()


def main():
    args = parse_args()
    cache_dir = Path(args.cache_dir).resolve()
    if args.client_num <= 0 or args.client_num % len(DOMAINS) != 0:
        raise SystemExit("client-num must be positive and divisible by 4")
    clients_per_domain = args.client_num // len(DOMAINS)
    records = []
    for domain_index, domain in enumerate(DOMAINS):
        sources = [
            path for path in cache_dir.glob(f"domainnet_{domain}_test_*.npz")
            if "_terminal_client" not in path.name
        ]
        if len(sources) != 1:
            raise RuntimeError(
                f"expected one source cache for {domain}, found {sources}")
        source = sources[0]
        with np.load(source, allow_pickle=False) as data:
            features = np.asarray(data["features"], dtype=np.float32)
            labels = np.asarray(data["labels"], dtype=np.int64)
        if labels.shape != (features.shape[0],):
            raise ValueError(f"invalid source shapes for {source}")

        partitions = np.array_split(np.arange(len(labels)), clients_per_domain)
        for local_index, indices in enumerate(partitions):
            client_id = domain_index * clients_per_domain + local_index + 1
            suffix = (
                f"_terminal_client{client_id:06d}of{args.client_num:06d}.npz")
            target = source.with_name(source.stem + suffix)
            np.savez(
                target,
                features=features[indices],
                labels=labels[indices])
            records.append({
                "client_id": client_id,
                "domain": domain,
                "count": int(len(indices)),
                "path": str(target),
            })

    marker = cache_dir / "domainnet_terminal_eval_shards.json"
    marker.write_text(json.dumps({
        "status": "pass",
        "client_num": args.client_num,
        "shard_count": len(records),
        "total_samples": sum(item["count"] for item in records),
        "domains": {
            domain: sum(
                item["count"] for item in records
                if item["domain"] == domain)
            for domain in DOMAINS
        },
        "shards": records,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(marker.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
