#!/usr/bin/env python3
"""Build balanced 12-client views from the validated 60-client manifests."""

import json
import shutil
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
DIGIT_SOURCE = REPO / "data" / "digit_three_domain"
DIGIT_TARGET = REPO / "data" / "digit_three_domain_fast12"
OFFICE_SOURCE = (
    REPO / "exp" / "distributed_manifests" /
    "officehome_60c_lds01_seed42")
OFFICE_TARGET = (
    REPO / "exp" / "distributed_manifests" /
    "officehome_fast12_balanced_from60")

DIGIT_SOURCE_IDS = (1, 2, 3, 4, 21, 22, 23, 24, 41, 42, 43, 44)
OFFICE_SOURCE_IDS = (1, 2, 3, 16, 17, 18, 31, 32, 33, 46, 47, 48)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")


def copy_client_views(source, target, source_ids):
    summary = []
    for new_id, source_id in enumerate(source_ids, 1):
        source_path = (
            source / f"client_{source_id:06d}" / "client_manifest.json")
        payload = json.loads(source_path.read_text(encoding="utf-8-sig"))
        payload["source_client_id"] = int(source_id)
        payload["client_id"] = int(new_id)
        target_path = (
            target / f"client_{new_id:06d}" / "client_manifest.json")
        write_json(target_path, payload)
        summary.append({
            "client_id": new_id,
            "source_client_id": source_id,
            "domain": payload.get("domain"),
            "train_samples": len(payload.get("splits", {}).get("train", [])),
            "test_samples": len(payload.get("splits", {}).get("test", [])),
        })
    return summary


def main():
    digit_summary = copy_client_views(
        DIGIT_SOURCE / "manifests", DIGIT_TARGET / "manifests",
        DIGIT_SOURCE_IDS)
    shutil.copyfile(
        DIGIT_SOURCE / "dataset_manifest.json",
        DIGIT_TARGET / "dataset_manifest.json")
    office_summary = copy_client_views(
        OFFICE_SOURCE, OFFICE_TARGET, OFFICE_SOURCE_IDS)
    result = {
        "status": "PASS",
        "client_num": 12,
        "digit": {
            "manifest_root": str(DIGIT_TARGET),
            "clients": digit_summary,
        },
        "officehome": {
            "manifest_root": str(OFFICE_TARGET),
            "clients": office_summary,
        },
    }
    write_json(
        REPO / "exp" / "distributed_manifests" /
        "fast12_balanced_summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
