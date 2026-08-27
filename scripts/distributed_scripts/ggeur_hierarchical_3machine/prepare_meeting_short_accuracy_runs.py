#!/usr/bin/env python3
"""Prepare the shortened T02-T06 accuracy queue used after the review meeting.

The queue contains a two-round cache-building preflight before each missing
Platform case, followed by the formal 100-round FedAvg/FedProx/Platform cases.
The already completed, equal-local-step MDSent-RNN results are intentionally
not regenerated; their run IDs are recorded in the output manifest.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml


THIS_DIR = Path(__file__).resolve().parent
GENERATOR = THIS_DIR / "generate_matrix.py"
ASSEMBLER = THIS_DIR / "assemble_optimized_accuracy_pipeline.py"
VALIDATOR = THIS_DIR / "validate_equal_local_steps.py"
RUNS_DIR = THIS_DIR / "runs"

GROUPS = (
    "digit3_vit",
    "officehome_cnn",
    "officehome_mixer",
    "mdsent_lstm",
)
METHODS = ("fedavg", "fedprox", "platform")
REUSED_RESULTS = {
    "mdsent_rnn_fedavg": "final_remaining_20260723_v1",
    "mdsent_rnn_fedprox": "final_remaining_20260723_v1",
    "mdsent_rnn_platform": "mdsent_reference_distributed_20260730_v1",
}


def run(command):
    completed = subprocess.run(
        [str(item) for item in command], check=True, text=True,
        capture_output=True)
    if completed.stdout.strip():
        print(completed.stdout.strip())


def generate(run_id, rounds, cache_run_id, methods):
    first = True
    for group in GROUPS:
        specs = ",".join(f"{group}:{method}" for method in methods)
        command = [
            sys.executable, GENERATOR,
            "--run-id", run_id,
            "--case-specs", specs,
            "--total-round-num", str(rounds),
            "--train-local-update-steps", "1",
            "--eval-frequency", "1",
            "--reuse-augmented-feature-cache",
            "--save-augmented-feature-cache",
            "--augmented-feature-cache-run-id", cache_run_id,
            "--ggeur-headonly-eval-mode", "both",
            "--ggeur-terminal-client-eval-only",
            "--statistics-upload-stagger-seconds", "0.2",
        ]
        if not first:
            command.append("--append-manifest")
        run(command)
        first = False


def configure_digit_task(run_id):
    case_name = "digit3_vit_platform"
    case_dir = RUNS_DIR / run_id / case_name
    profile_path = case_dir / "mddigits_same_distribution_task.json"
    profile_path.write_text(json.dumps({
        "default_target_size": 80,
        "class_counts": {str(class_id): 80 for class_id in range(10)},
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    repo_by_config = {
        "root_server.yaml": "/root/autodl-tmp/FederatedScope",
        "clients_8g": "D:/Projects/FederatedScope",
        "clients_4090": "/root/autodl-tmp/FederatedScope",
        "clients_third": "C:/Users/pc/FederatedScope",
    }
    changed = 0
    for path in sorted((case_dir / "configs").rglob("*.yaml")):
        if path.name == "root_server.yaml":
            repo = repo_by_config[path.name]
        else:
            repo = repo_by_config.get(path.parent.name)
        if not repo:
            continue
        remote_case = (
            f"{repo}/scripts/distributed_scripts/"
            f"ggeur_hierarchical_3machine/runs/{run_id}/{case_name}")
        content = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
        ggeur = content.setdefault("ggeur", {})
        ggeur["task_adaptation_file"] = (
            f"{remote_case}/mddigits_same_distribution_task.json")
        ggeur["task_class_counts"] = []
        ggeur["task_default_target_size"] = ""
        ggeur["training_distribution_dir"] = (
            f"{remote_case}/training_distributions")
        path.write_text(yaml.safe_dump(
            content, sort_keys=False, allow_unicode=True), encoding="utf-8")
        changed += 1
    if changed != 61:
        raise RuntimeError(
            f"Expected 60 clients plus root config, changed {changed}")


def validate(run_id, rounds, expected):
    manifest = RUNS_DIR / run_id / "matrix_manifest.json"
    evidence = RUNS_DIR / run_id / "equal_local_steps_validation.json"
    run([
        sys.executable, VALIDATOR, manifest,
        "--output", evidence,
        "--expected-rounds", str(rounds),
        "--expected-eval-frequency", "1",
    ])
    payload = json.loads(manifest.read_text(encoding="utf-8-sig"))
    actual = [item["case"] for item in payload["cases"]]
    if actual != expected:
        raise RuntimeError(f"case order mismatch: {actual!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline-run-id", required=True)
    parser.add_argument("--preflight-run-id", required=True)
    parser.add_argument("--formal-run-id", required=True)
    args = parser.parse_args()

    generate(args.preflight_run_id, 2, args.pipeline_run_id, ("platform",))
    configure_digit_task(args.preflight_run_id)
    validate(
        args.preflight_run_id, 2,
        [f"{group}_platform" for group in GROUPS])

    generate(args.formal_run_id, 100, args.pipeline_run_id, METHODS)
    configure_digit_task(args.formal_run_id)
    validate(
        args.formal_run_id, 100,
        [f"{group}_{method}" for group in GROUPS for method in METHODS])

    run([
        sys.executable, ASSEMBLER,
        "--pipeline-run-id", args.pipeline_run_id,
        "--smoke-run-id", args.preflight_run_id,
        "--formal-run-id", args.formal_run_id,
    ])
    pipeline_manifest = RUNS_DIR / args.pipeline_run_id / "matrix_manifest.json"
    payload = json.loads(pipeline_manifest.read_text(encoding="utf-8-sig"))
    payload["reused_completed_results"] = REUSED_RESULTS
    payload["acceptance_contract"] = {
        "rounds": 100,
        "eval_frequency": 1,
        "local_update_steps": 1,
        "methods": list(METHODS),
        "platform_cache_preflight_rounds": 2,
    }
    pipeline_manifest.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "status": "pass",
        "pipeline_run_id": args.pipeline_run_id,
        "queued_cases": payload["case_count"],
        "formal_cases": payload["formal_case_count"],
        "reused_cases": len(REUSED_RESULTS),
        "manifest": str(pipeline_manifest),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
