"""Audit the downloaded T02-T06 outline rerun evidence.

The audit intentionally checks the stable log names and evidence paths written in
the current test outline, so it also acts as a regression check for the outline's
manual execution steps.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ROUND_RE = re.compile(
    r"Server: Round\s+(\d+)\s+.*?Test Accuracy.*?average:\s*([0-9.]+)"
)
ERROR_RE = re.compile(r"Traceback \(most recent call last\)|CUDA out of memory|RuntimeError:")


CASES = {
    "T-02": {
        "slug": "t2_digit3_vit",
        "model": "ViT",
        "dataset": "MDDigits",
        "distribution": "exp/digit_three_domain/training_distributions",
        "clients": 60,
        "classes": 10,
        "target": 40,
    },
    "T-03": {
        "slug": "t3_officehome_cnn",
        "model": "CNN",
        "dataset": "OfficeHome",
        "distribution": "exp/test_outline_validation/T-03/training_distributions",
        "clients": 60,
        "classes": 65,
        "target": 20,
    },
    "T-04": {
        "slug": "t4_officehome_mlp",
        "model": "MLP",
        "dataset": "OfficeHome",
        "distribution": "exp/test_outline_validation/T-04/training_distributions",
        "clients": 60,
        "classes": 65,
        "target": 20,
    },
    "T-05": {
        "slug": "t5_mdsent_rnn",
        "model": "RNN",
        "dataset": "MDSent",
        "distribution": "exp/test_outline_validation/T-05/training_distributions",
        "clients": 3,
        "classes": 4,
        "target": 50,
    },
    "T-06": {
        "slug": "t6_mdsent_lstm",
        "model": "LSTM",
        "dataset": "MDSent",
        "distribution": "exp/test_outline_validation/T-06/training_distributions",
        "clients": 6,
        "classes": 4,
        "target": 50,
    },
}


def read_timing(root: Path, case_id: str, method: str) -> dict:
    number = case_id[-2:]
    path = (
        root
        / "exp/t2t6_outline_rerun_20260818/control"
        / f"t{number}_{method}_formal.timing.json"
    )
    result = json.loads(path.read_text(encoding="utf-8"))
    result["relative_path"] = path.relative_to(root).as_posix()
    return result


def audit_log(root: Path, case: dict, method: str) -> dict:
    path = root / f"{case['slug']}_formal_{method}.log"
    text = path.read_text(encoding="utf-8", errors="replace")
    matches = [(int(round_no), float(value)) for round_no, value in ROUND_RE.findall(text)]
    rounds = [round_no for round_no, _ in matches]
    augmented_cache_count = text.count("Loaded augmented feature cache")
    task_profile_count = text.count("Loaded task-adaptive generation profile")
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "round_accuracy_lines": len(matches),
        "rounds_are_0_to_99": rounds == list(range(100)),
        "final_accuracy": matches[-1][1] if matches else None,
        "error_signature_found": bool(ERROR_RE.search(text)),
        "augmented_cache_load_count": augmented_cache_count,
        "task_profile_load_count": task_profile_count,
        "task_profile_role_valid": (
            task_profile_count == 0
            if method in {"fedavg", "fedprox"}
            else task_profile_count > 0
        ),
        "cache_role_valid": (
            augmented_cache_count == 0
            if method in {"fedavg", "fedprox"}
            else augmented_cache_count > 0
        ),
    }


def audit_distribution(root: Path, case: dict) -> dict:
    directory = root / case["distribution"]
    files = sorted(directory.glob("client_*.json"))
    expected_counts = {str(index): case["target"] for index in range(case["classes"])}
    invalid = []
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        if (
            data.get("class_counts") != expected_counts
            or data.get("total_samples") != case["classes"] * case["target"]
            or not data.get("task_adaptation", {}).get("enabled")
        ):
            invalid.append(path.name)
    return {
        "relative_path": directory.relative_to(root).as_posix(),
        "file_count": len(files),
        "expected_file_count": case["clients"],
        "expected_class_counts": expected_counts,
        "invalid_files": invalid,
        "valid": len(files) == case["clients"] and not invalid,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence_root", type=Path)
    args = parser.parse_args()
    root = args.evidence_root.resolve()

    result = {"evidence_root": str(root), "cases": {}}
    for case_id, case in CASES.items():
        methods = {}
        for method in ("fedavg", "fedprox", "platform"):
            log = audit_log(root, case, method)
            timing = read_timing(root, case_id, method)
            methods[method] = {
                **log,
                "elapsed_seconds": timing["elapsed_seconds"],
                "exit_code": timing["exit_code"],
                "command": timing["command"],
                "timing_relative_path": timing["relative_path"],
            }

        platform = methods["platform"]["final_accuracy"]
        gain_avg = platform - methods["fedavg"]["final_accuracy"]
        gain_prox = platform - methods["fedprox"]["final_accuracy"]
        distribution = audit_distribution(root, case)
        case_pass = (
            gain_avg >= 0.20
            and gain_prox >= 0.20
            and distribution["valid"]
            and all(
                item["exit_code"] == 0
                and item["round_accuracy_lines"] == 100
                and item["rounds_are_0_to_99"]
                and not item["error_signature_found"]
                and item["cache_role_valid"]
                and item["task_profile_role_valid"]
                for item in methods.values()
            )
        )
        result["cases"][case_id] = {
            "model": case["model"],
            "dataset": case["dataset"],
            "methods": methods,
            "platform_gain_vs_fedavg_percentage_points": round(gain_avg * 100, 2),
            "platform_gain_vs_fedprox_percentage_points": round(gain_prox * 100, 2),
            "formal_elapsed_seconds": round(
                sum(item["elapsed_seconds"] for item in methods.values()), 3
            ),
            "distribution": distribution,
            "pass": case_pass,
        }

    result["t02_t06_formal_elapsed_seconds"] = round(
        sum(item["formal_elapsed_seconds"] for item in result["cases"].values()), 3
    )
    result["all_cases_pass"] = all(item["pass"] for item in result["cases"].values())
    json_path = root / "evidence_audit.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# T02-T06 大纲步骤证据核验",
        "",
        "| 用例 | 模型/数据集 | FedAvg | FedProx | 平台 | 平台绝对提升 | 正式耗时 | 结论 |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for case_id, item in result["cases"].items():
        methods = item["methods"]
        lines.append(
            f"| {case_id} | {item['model']} / {item['dataset']} | "
            f"{methods['fedavg']['final_accuracy']:.4f} | "
            f"{methods['fedprox']['final_accuracy']:.4f} | "
            f"{methods['platform']['final_accuracy']:.4f} | "
            f"+{item['platform_gain_vs_fedavg_percentage_points']:.2f}/"
            f"+{item['platform_gain_vs_fedprox_percentage_points']:.2f} 个百分点 | "
            f"{item['formal_elapsed_seconds']:.3f} 秒 | "
            f"{'通过' if item['pass'] else '不通过'} |"
        )
    lines.extend(
        [
            "",
            "核验规则：三种方法均退出码为 0；日志均包含第 0—99 轮共 100 条准确率；"
            "无 Python 回溯、CUDA 内存不足或运行时错误；FedAvg/FedProx 不加载平台增强缓存；"
            "平台加载增强缓存并读取任务文件；训练分布文件数量与各类别目标数符合大纲；"
            "平台准确率分别比 FedAvg、FedProx 绝对提升不少于 20 个百分点。",
            "",
            f"T02—T06 三种方法正式执行总耗时：{result['t02_t06_formal_elapsed_seconds']:.3f} 秒。",
        ]
    )
    (root / "大纲步骤核验.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "all_cases_pass": result["all_cases_pass"],
        "formal_elapsed_seconds": result["t02_t06_formal_elapsed_seconds"],
        "json": str(json_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
