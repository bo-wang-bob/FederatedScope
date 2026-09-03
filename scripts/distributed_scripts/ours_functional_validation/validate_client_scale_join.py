#!/usr/bin/env python3
"""Audit small/large-client manifests and delayed-join scenario evidence."""

import argparse
import csv
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--small-manifest", required=True)
    parser.add_argument("--large-manifest", required=True)
    parser.add_argument("--scenario-summary", required=True)
    parser.add_argument("--small-max", type=int, default=4)
    parser.add_argument("--large-min", type=int, default=60)
    return parser.parse_args()


def load_manifest(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_scenarios(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        return {
            row["scenario"]: row
            for row in csv.DictReader(stream, delimiter="\t")
        }


def main():
    args = parse_args()
    small = load_manifest(args.small_manifest)
    large = load_manifest(args.large_manifest)
    scenarios = load_scenarios(args.scenario_summary)

    small_count = int(small.get("client_num", 0))
    large_count = int(large.get("client_num", 0))
    if not 1 <= small_count <= args.small_max:
        raise ValueError(
            f"Small-client case has invalid client_num={small_count}")
    if large_count < args.large_min:
        raise ValueError(
            f"Large-client case has client_num={large_count}, expected at "
            f"least {args.large_min}")

    delayed = scenarios.get("delayed_client_join")
    if delayed is None:
        raise ValueError("Scenario summary lacks delayed_client_join")
    if delayed.get("expected") != "success" or delayed.get("status") != "pass":
        raise ValueError(
            f"Delayed join did not pass: {delayed}")

    print(json.dumps({
        "status": "PASS",
        "small_clients": small_count,
        "large_clients": large_count,
        "delayed_client_join": "pass",
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
