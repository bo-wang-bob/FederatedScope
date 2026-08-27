#!/usr/bin/env python3
"""Print a compact final table from a T2-T6 summary JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "summary",
        nargs="?",
        type=Path,
        default=Path("exp/t2t6_4090_final/summary.json"),
    )
    args = parser.parse_args()
    payload = json.loads(args.summary.read_text(encoding="utf-8"))
    print("用例\t模型-数据集\tFedAvg\tFedProx\t平台\t平台-FedAvg/百分点\t平台-FedProx/百分点\t总用时/秒")
    for _, case in payload["cases"].items():
        runs = case["runs"]
        fedavg = runs["formal_fedavg"]
        fedprox = runs["formal_fedprox"]
        platform = runs["formal_platform"]
        comparison = case["comparison"]
        total = sum(
            float(run["elapsed_seconds"])
            for run in (fedavg, fedprox, platform)
        )
        print(
            "\t".join(
                [
                    case["test_case"],
                    f"{case['model_name']}-{case['dataset_name']}",
                    f"{fedavg['average_final']:.4f}",
                    f"{fedprox['average_final']:.4f}",
                    f"{platform['average_final']:.4f}",
                    f"{comparison['platform_minus_fedavg_points']:.2f}",
                    f"{comparison['platform_minus_fedprox_points']:.2f}",
                    f"{total:.1f}",
                ]
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
