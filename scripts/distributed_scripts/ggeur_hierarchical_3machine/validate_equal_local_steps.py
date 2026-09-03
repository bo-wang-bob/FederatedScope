#!/usr/bin/env python3
"""Validate the equal-local-step and cache-only formal-run contract."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import yaml


def load_yaml(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8-sig"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--output", default="")
    parser.add_argument("--expected-rounds", type=int, default=100)
    parser.add_argument("--expected-eval-frequency", type=int, default=1)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    matrix = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    groups = defaultdict(list)
    failures = []

    for case in matrix.get("cases", []):
        case_name = str(case["case"])
        group = str(case["group"])
        method = str(case["method"])
        root = load_yaml(case["root_config"])
        root_steps = int(root["train"]["local_update_steps"])
        rounds = int(root["federate"]["total_round_num"])
        eval_frequency = int(root.get("eval", {}).get("freq", 1))
        ggeur = root.get("ggeur", {})
        complete_cache = bool(ggeur.get("require_complete_feature_cache"))
        groups[group].append({
            "case": case_name,
            "method": method,
            "local_update_steps": root_steps,
        })
        if rounds != args.expected_rounds:
            failures.append(
                f"{case_name}: total_round_num={rounds}, expected "
                f"{args.expected_rounds}")
        if eval_frequency != args.expected_eval_frequency:
            failures.append(
                f"{case_name}: eval.freq={eval_frequency}, expected "
                f"{args.expected_eval_frequency}")
        if not complete_cache:
            failures.append(
                f"{case_name}: root require_complete_feature_cache is false")
        if (str(ggeur.get("headonly_eval_mode", "server")).lower() ==
                "both" and not bool(
                    ggeur.get("terminal_client_eval_only", False))):
            failures.append(
                f"{case_name}: terminal_client_eval_only must be true when "
                "headonly_eval_mode=both")

        for client_path in case.get("client_configs", []):
            client = load_yaml(client_path)
            client_steps = int(client["train"]["local_update_steps"])
            if client_steps != root_steps:
                failures.append(
                    f"{case_name}: {client_path} local_update_steps="
                    f"{client_steps}, root={root_steps}")
            if not bool(client.get("ggeur", {}).get(
                    "require_complete_feature_cache")):
                failures.append(
                    f"{case_name}: {client_path} does not require complete "
                    "feature cache")

    group_summary = {}
    for group, cases in sorted(groups.items()):
        values = sorted({item["local_update_steps"] for item in cases})
        group_summary[group] = {
            "local_update_steps": values[0] if len(values) == 1 else values,
            "methods": [item["method"] for item in cases],
            "case_count": len(cases),
        }
        if len(values) != 1:
            failures.append(
                f"{group}: methods use inconsistent local update steps "
                f"{values}")

    result = {
        "status": "pass" if not failures else "fail",
        "manifest": str(manifest_path),
        "case_count": sum(len(items) for items in groups.values()),
        "groups": group_summary,
        "failures": failures,
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
