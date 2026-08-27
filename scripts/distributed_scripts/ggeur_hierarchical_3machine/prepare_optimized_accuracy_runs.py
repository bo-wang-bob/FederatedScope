#!/usr/bin/env python3
"""Generate the optimized equal-local-step accuracy queue.

The script deliberately invokes the matrix generator once per dataset/model
group.  That keeps each family's validated feature-cache root while appending
all cases to one sequential dual-host queue.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
GENERATOR = THIS_DIR / "generate_matrix.py"
VALIDATOR = THIS_DIR / "validate_equal_local_steps.py"

METHODS = {
    "officehome_vit": [
        "fedavg", "fedprox", "fedproto", "fedopt", "moon", "platform"],
    "domainnet_vit": [
        "fedavg", "fedprox", "fedproto", "fedopt", "moon", "platform"],
    "digit3_vit": ["fedavg", "fedprox", "platform"],
    "officehome_cnn": [
        "fedavg", "fedprox", "fedproto", "fedopt", "moon", "platform"],
    "domainnet_cnn": [
        "fedavg", "fedprox", "fedproto", "fedopt", "moon", "platform"],
    "officehome_mixer": [
        "fedavg", "fedprox", "fedproto", "fedopt", "moon", "platform"],
    "domainnet_mixer": [
        "fedavg", "fedprox", "fedproto", "fedopt", "moon", "platform"],
    "mdsent_rnn": [
        "fedavg", "fedprox", "fedproto", "fedopt", "platform"],
    "mdsent_lstm": [
        "fedavg", "fedprox", "fedproto", "fedopt", "platform"],
}

DOMAINNET_ARGS = [
    "--client-feature-cache-root",
    "D:/Projects/FederatedScope/exp/"
    "distributed_feature_cache_domainnet_original_20260804",
    "--client-domainnet-root",
    "D:/Projects/FederatedScope/data_staging/"
    "domainnet_original_4domains_20260804/dataset",
    "--client-domainnet-manifest-path",
    "D:/Projects/FederatedScope/data_staging/"
    "domainnet_original_4domains_20260804/domainnet_manifest.json",
    "--windows-client-count", "60",
]


def run(command):
    completed = subprocess.run(command, check=True, text=True,
                               capture_output=True)
    if completed.stdout.strip():
        print(completed.stdout.strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--cache-run-id", required=True)
    parser.add_argument(
        "--platform-only", action="store_true",
        help="Generate only the nine platform cache/preflight cases")
    args = parser.parse_args()
    if args.rounds < 2:
        raise SystemExit("--rounds must be at least 2")

    first = True
    expected_cases = []
    for group, all_methods in METHODS.items():
        methods = ["platform"] if args.platform_only else all_methods
        case_specs = ",".join(
            f"{group}:{method}" for method in methods)
        command = [
            sys.executable, str(GENERATOR),
            "--run-id", args.run_id,
            "--case-specs", case_specs,
            "--dual-simulated-topology",
            "--total-round-num", str(args.rounds),
            "--eval-frequency", "1",
            "--reuse-augmented-feature-cache",
            "--save-augmented-feature-cache",
            "--augmented-feature-cache-run-id", args.cache_run_id,
            "--ggeur-headonly-eval-mode", "both",
            "--ggeur-terminal-client-eval-only",
            "--statistics-upload-stagger-seconds", "0.2",
            "--domainnet-statistics-upload-stagger-seconds", "0.2",
        ]
        if not first:
            command.append("--append-manifest")
        if group.startswith("domainnet_"):
            command.extend(DOMAINNET_ARGS)
        run(command)
        expected_cases.extend(
            f"{group}_{method}" for method in methods)
        first = False

    manifest = THIS_DIR / "runs" / args.run_id / "matrix_manifest.json"
    evidence = manifest.parent / "equal_local_steps_validation.json"
    run([
        sys.executable, str(VALIDATOR), str(manifest),
        "--output", str(evidence),
        "--expected-rounds", str(args.rounds),
        "--expected-eval-frequency", "1",
    ])
    matrix = json.loads(manifest.read_text(encoding="utf-8-sig"))
    actual = [item["case"] for item in matrix["cases"]]
    if actual != expected_cases:
        raise SystemExit(
            "matrix order mismatch:\nexpected=" + repr(expected_cases) +
            "\nactual=" + repr(actual))
    print(json.dumps({
        "status": "pass",
        "run_id": args.run_id,
        "rounds": args.rounds,
        "case_count": len(actual),
        "manifest": str(manifest),
        "validation": str(evidence),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
