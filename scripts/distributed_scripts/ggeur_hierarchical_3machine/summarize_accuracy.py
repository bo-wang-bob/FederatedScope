#!/usr/bin/env python3
"""Extract round-wise and best accuracy from a root-server log."""

import argparse
import json
import re
from pathlib import Path


ROUND_RE = re.compile(
    r"Round\s+(?P<round>\d+)\s+MLP Test Accuracy\s+-\s+(?P<body>.*)")
METRIC_RE = re.compile(
    r"(?P<name>[A-Za-z0-9_./-]+):\s*(?P<value>-?\d+(?:\.\d+)?)")
BEST_RE = re.compile(
    r"(?:Classifier )?Best Average Accuracy:\s*(?P<value>\d+(?:\.\d+)?)")


def parse_log(path):
    rounds = []
    reported_best = None
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        match = ROUND_RE.search(line)
        if match:
            metrics = {
                item.group("name"): float(item.group("value"))
                for item in METRIC_RE.finditer(match.group("body"))
            }
            rounds.append({"round": int(match.group("round")),
                           "metrics": metrics})
        best_match = BEST_RE.search(line)
        if best_match:
            reported_best = float(best_match.group("value"))

    average_values = [
        item["metrics"]["average"] for item in rounds
        if "average" in item["metrics"]
    ]
    calculated_best = max(average_values) if average_values else None
    return {
        "log": str(path),
        "round_count": len(rounds),
        "last": rounds[-1] if rounds else None,
        "best_average": reported_best
        if reported_best is not None else calculated_best,
        "rounds": rounds,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    summary = parse_log(args.log)
    payload = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
