#!/usr/bin/env python3
"""Summarize the formal three-domain ViT accuracy comparison."""

import argparse
import csv
import json
import re
from pathlib import Path


ROUND_PATTERN = re.compile(
    r"Round\s+(?P<round>\d+)\s+MLP Test Accuracy\s+-\s+"
    r"emnist_digits:\s*(?P<emnist>[0-9.]+),\s+"
    r"usps:\s*(?P<usps>[0-9.]+),\s+"
    r"svhn:\s*(?P<svhn>[0-9.]+),\s+"
    r"average:\s*(?P<average>[0-9.]+)")


def parse_log(path):
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    matches = list(ROUND_PATTERN.finditer(text))
    if not matches:
        raise ValueError(f"No three-domain accuracy record found in {path}")
    records = []
    for match in matches:
        records.append({
            "round": int(match.group("round")),
            "emnist_digits": float(match.group("emnist")),
            "usps": float(match.group("usps")),
            "svhn": float(match.group("svhn")),
            "average": float(match.group("average")),
        })
    final = records[-1]
    best = max(records, key=lambda record: record["average"])
    return {
        "log": Path(path).resolve().as_posix(),
        "evaluated_rounds": len(records),
        "final": final,
        "best_average": best,
    }


def improvement(platform, baseline):
    absolute_pp = (platform - baseline) * 100.0
    relative_pct = ((platform - baseline) / baseline * 100.0
                    if baseline > 0 else None)
    return {
        "absolute_improvement_pp": round(absolute_pp, 4),
        "relative_improvement_pct": (
            round(relative_pct, 4) if relative_pct is not None else None),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fedavg-log", required=True)
    parser.add_argument("--fedprox-log", required=True)
    parser.add_argument("--platform-log", required=True)
    parser.add_argument("--partition-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-relative-improvement-pct",
                        type=float,
                        default=20.0)
    parser.add_argument(
        "--execution",
        default="single-host 60-logical-client validation",
        help="Human-readable execution environment recorded in evidence")
    return parser.parse_args()


def main():
    args = parse_args()
    methods = {
        "FedAvg": parse_log(args.fedavg_log),
        "FedProx": parse_log(args.fedprox_log),
        "平台": parse_log(args.platform_log),
    }
    partition = json.loads(
        Path(args.partition_summary).read_text(encoding="utf-8"))
    platform_average = methods["平台"]["final"]["average"]
    comparisons = {
        baseline: improvement(platform_average,
                              methods[baseline]["final"]["average"])
        for baseline in ("FedAvg", "FedProx")
    }
    threshold = float(args.min_relative_improvement_pct)
    passed = all(
        item["relative_improvement_pct"] is not None and
        item["relative_improvement_pct"] >= threshold
        for item in comparisons.values())

    report = {
        "status": "PASS" if passed else "FAIL",
        "acceptance": {
            "metric": "final equal-domain average accuracy",
            "minimum_relative_improvement_pct": threshold,
            "requires_both_baselines": True,
        },
        "experiment": {
            "dataset": "TriDomainDigits",
            "domains": ["emnist_digits", "usps", "svhn"],
            "class_labels": [str(value) for value in range(10)],
            "model": "ViT-Tiny/16 (frozen pretrained feature extractor)",
            "client_num": int(partition["client_num"]),
            "clients_per_domain": int(partition["clients_per_domain"]),
            "dirichlet_alpha": float(partition["dirichlet_alpha"]),
            "partition_seed": int(partition["seed"]),
            "partition_manifest_sha256": partition[
                "global_manifest_sha256"],
            "training_rounds_configured": 100,
            "final_training_round": 99,
            "execution": str(args.execution),
        },
        "methods": methods,
        "comparisons": comparisons,
        "note": (
            "Three-machine cases were generated separately. This report is "
            "the completed local logical-client validation and must not be "
            "mislabelled as a completed three-machine run."
            if str(args.execution) ==
            "single-host 60-logical-client validation" else
            "This report is labelled with the operator-supplied formal "
            "execution environment; retain the three-machine logs and "
            "terminal-accuracy evidence files for review."),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "accuracy_acceptance.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                         encoding="utf-8")

    csv_path = output_dir / "accuracy_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "method", "final_round", "emnist_digits", "usps", "svhn",
            "equal_domain_average"
        ])
        for method, item in methods.items():
            final = item["final"]
            writer.writerow([
                method, final["round"], final["emnist_digits"],
                final["usps"], final["svhn"], final["average"]
            ])

    lines = [
        "# 三数字域 ViT 准确率验收记录",
        "",
        f"- 状态：{report['status']}",
        f"- 客户端：{partition['client_num']}（每域 "
        f"{partition['clients_per_domain']}）",
        f"- 狄利克雷参数：α={partition['dirichlet_alpha']}",
        f"- 清单校验值：`{partition['global_manifest_sha256']}`",
        "",
        "| 方法 | EMNIST Digits | USPS | SVHN | 三域平均 |",
        "|---|---:|---:|---:|---:|",
    ]
    for method, item in methods.items():
        final = item["final"]
        lines.append(
            f"| {method} | {final['emnist_digits'] * 100:.2f}% | "
            f"{final['usps'] * 100:.2f}% | {final['svhn'] * 100:.2f}% | "
            f"{final['average'] * 100:.2f}% |")
    lines.extend(["", "| 对比项 | 相对提升 | 绝对提升 |", "|---|---:|---:|"])
    for baseline, item in comparisons.items():
        lines.append(
            f"| 平台相对 {baseline} | "
            f"{item['relative_improvement_pct']:.2f}% | "
            f"{item['absolute_improvement_pp']:.2f} 个百分点 |")
    lines.extend([
        "",
        f"> 注：{report['note']}",
        "",
    ])
    (output_dir / "accuracy_report.md").write_text(
        "\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "json": json_path.resolve().as_posix(),
        "csv": csv_path.resolve().as_posix(),
        "comparisons": comparisons,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
