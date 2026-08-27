#!/usr/bin/env python3
"""Validate per-client platform training-distribution reports."""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--expected-clients", type=int, required=True)
    return parser.parse_args()


def validate_report(path):
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("method") != "Platform":
        raise ValueError(f"{path}: method is not Platform")
    task = report.get("task_adaptation", {})
    if not task.get("enabled", False):
        raise ValueError(f"{path}: task adaptation is not enabled")

    actual = {
        str(key): int(value)
        for key, value in report.get("class_counts", {}).items()
    }
    targets = {
        str(key): int(value)
        for key, value in task.get("class_targets", {}).items()
    }
    mismatches = {
        key: {"expected": expected, "actual": actual.get(key, 0)}
        for key, expected in targets.items()
        if actual.get(key, 0) != expected
    }
    if mismatches:
        preview = dict(list(mismatches.items())[:10])
        raise ValueError(f"{path}: class target mismatches: {preview}")
    if int(report.get("total_samples", -1)) != sum(actual.values()):
        raise ValueError(f"{path}: total_samples does not match class_counts")
    return report


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    reports = sorted(input_dir.glob("client_*.json"))
    if len(reports) != args.expected_clients:
        raise ValueError(
            f"Expected {args.expected_clients} reports, found {len(reports)} "
            f"under {input_dir}")

    validated = [validate_report(path) for path in reports]
    signatures = {
        item["task_adaptation"].get("signature", "")
        for item in validated
    }
    if len(signatures) != 1 or "" in signatures:
        raise ValueError(
            f"Clients used inconsistent task signatures: {signatures}")
    print(json.dumps({
        "status": "PASS",
        "clients": len(validated),
        "task_signature": next(iter(signatures)),
        "total_training_samples": sum(
            int(item["total_samples"]) for item in validated),
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
