#!/usr/bin/env python3
"""Verify and atomically install relayed DomainNet training feature caches."""

import argparse
import hashlib
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--incoming-suffix", default=".incoming_20260813")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir).resolve()
    manifest_path = Path(args.manifest).resolve()
    state = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    domains = state.get("domains") or {}
    expected_domains = {"clipart", "painting", "real", "sketch"}
    if set(domains) != expected_domains:
        raise RuntimeError(
            f"domain set mismatch: {sorted(domains)} != "
            f"{sorted(expected_domains)}")

    verified = []
    for domain in sorted(domains):
        entry = domains[domain]
        name = Path(str(entry["name"])).name
        incoming = cache_dir / (name + args.incoming_suffix)
        if not incoming.is_file():
            raise FileNotFoundError(incoming)
        actual_bytes = incoming.stat().st_size
        actual_sha256 = sha256_file(incoming)
        if actual_bytes != int(entry["bytes"]):
            raise RuntimeError(
                f"byte mismatch for {domain}: {actual_bytes} != "
                f"{entry['bytes']}")
        if actual_sha256 != str(entry["sha256"]).lower():
            raise RuntimeError(
                f"sha256 mismatch for {domain}: {actual_sha256} != "
                f"{entry['sha256']}")
        verified.append((domain, entry, incoming, name))

    marker_domains = {}
    for domain, entry, incoming, name in verified:
        target = cache_dir / name
        os.replace(incoming, target)
        marker_domains[domain] = {
            "path": str(target),
            "count": int(entry["count"]),
            "dim": int(entry["dim"]),
            "bytes": int(entry["bytes"]),
            "sha256": str(entry["sha256"]).lower(),
        }

    marker = {
        "group": str(state["group"]),
        "method": "verified_cache_relay",
        "client_num": int(state["client_num"]),
        "cache_files": len(marker_domains),
        "require_complete_feature_cache": True,
        "validated_on": platform.node(),
        "validated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "domains": marker_domains,
    }
    marker_path = cache_dir / ".ggeur_feature_cache_ready.json"
    temporary = marker_path.with_suffix(marker_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(marker, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, marker_path)
    print(json.dumps({
        "status": "pass",
        "marker": str(marker_path),
        "validated_on": marker["validated_on"],
        "client_num": marker["client_num"],
        "domains": {
            domain: {
                "count": entry["count"],
                "bytes": entry["bytes"],
                "sha256": entry["sha256"],
            }
            for domain, entry in marker_domains.items()
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
