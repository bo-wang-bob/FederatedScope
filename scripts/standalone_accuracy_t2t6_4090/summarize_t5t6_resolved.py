"""Summarize downloaded final T5/T6 three-seed evidence."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
evidence = ROOT / "docs" / "test_logs" / "t5t6_resolved_final_20260822"
case_names = {
    "T5": "t5_mdsent_rnn_fullstats_s3_t50_lr3e5",
    "T6": "t6_mdsent_lstm_fullstats_s3_t100_lr1e4",
}
rows = []
for test_case, case_name in case_names.items():
    for seed in (42, 43, 44):
        summary = json.loads(
            (evidence / test_case / f"seed{seed}" / "summary.json").read_text(
                encoding="utf-8"))
        runs = summary["cases"][case_name]["runs"]
        values = {
            method: float(runs[f"formal_{method}"]["average_final"])
            for method in ("fedavg", "fedprox", "platform")
        }
        rows.append({
            "test_case": test_case,
            "seed": seed,
            **values,
            "platform_minus_fedavg": values["platform"] - values["fedavg"],
            "platform_minus_fedprox": values["platform"] - values["fedprox"],
            "passed": (
                values["platform"] - values["fedavg"] >= 0.2
                and values["platform"] - values["fedprox"] >= 0.2),
        })

averages = {}
for test_case in case_names:
    selected = [row for row in rows if row["test_case"] == test_case]
    averages[test_case] = {
        key: sum(row[key] for row in selected) / len(selected)
        for key in (
            "fedavg", "fedprox", "platform",
            "platform_minus_fedavg", "platform_minus_fedprox")
    }

payload = {"rows": rows, "three_seed_averages": averages}
(evidence / "results.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

lines = [
    "# T5/T6最终三随机种子结果", "",
    "| 测试项 | 种子 | FedAvg | FedProx | 平台 | 相比FedAvg | 相比FedProx | 结论 |",
    "|---|---:|---:|---:|---:|---:|---:|---|",
]
for row in rows:
    lines.append(
        f"| {row['test_case']} | {row['seed']} | "
        f"{row['fedavg']:.2%} | {row['fedprox']:.2%} | "
        f"{row['platform']:.2%} | {row['platform_minus_fedavg']:.2%} | "
        f"{row['platform_minus_fedprox']:.2%} | "
        f"{'通过' if row['passed'] else '未通过'} |")
lines.extend(["", "三随机种子均值："])
for test_case, values in averages.items():
    lines.append(
        f"- {test_case}：FedAvg {values['fedavg']:.2%}，"
        f"FedProx {values['fedprox']:.2%}，平台 {values['platform']:.2%}；"
        f"绝对提升 {values['platform_minus_fedavg']:.2%}/"
        f"{values['platform_minus_fedprox']:.2%}。")
(evidence / "results.md").write_text(
    "\n".join(lines) + "\n", encoding="utf-8")
print(evidence / "results.md")
