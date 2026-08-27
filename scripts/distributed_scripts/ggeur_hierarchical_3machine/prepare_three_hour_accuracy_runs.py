#!/usr/bin/env python3
"""Prepare the 12-client, cache-first, 100-round accuracy queue."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml


THIS_DIR = Path(__file__).resolve().parent
RUNS_DIR = THIS_DIR / "runs"
GENERATOR = THIS_DIR / "generate_matrix.py"
ASSEMBLER = THIS_DIR / "assemble_optimized_accuracy_pipeline.py"
VALIDATOR = THIS_DIR / "validate_equal_local_steps.py"
MANIFEST_PREPARER = THIS_DIR / "prepare_fast12_manifests.py"
GROUPS = (
    "digit3_vit",
    "officehome_cnn",
    "officehome_mixer",
    "mdsent_rnn",
    "mdsent_lstm",
)
METHODS = ("fedavg", "fedprox", "platform")


def run(command):
    completed = subprocess.run(
        [str(item) for item in command], check=True, text=True,
        capture_output=True)
    if completed.stdout.strip():
        print(completed.stdout.strip())


def group_paths(group):
    if group == "digit3_vit":
        return [
            "--client-digit3-root",
            "D:/Projects/FederatedScope/data/digit_three_domain_fast12",
            "--root-client-digit3-root",
            "/root/autodl-tmp/datasets/digit_three_domain_fast12",
        ]
    if group.startswith("officehome_"):
        return [
            "--client-officehome-manifest-root",
            ("D:/Projects/FederatedScope/exp/distributed_manifests/"
             "officehome_fast12_balanced_from60"),
            "--root-client-officehome-manifest-root",
            ("/root/autodl-tmp/FederatedScope/exp/distributed_manifests/"
             "officehome_fast12_balanced_from60"),
        ]
    return []


def generate(run_id, rounds, cache_run_id, methods):
    first = True
    for group in GROUPS:
        specs = ",".join(f"{group}:{method}" for method in methods)
        command = [
            sys.executable, GENERATOR,
            "--run-id", run_id,
            "--case-specs", specs,
            "--client-num", "12",
            "--sample-client-num", "12",
            "--windows-client-count", "12",
            "--subserver-num", "1",
            "--total-round-num", str(rounds),
            "--train-local-update-steps", "1",
            "--eval-frequency", "1",
            "--reuse-augmented-feature-cache",
            "--save-augmented-feature-cache",
            "--augmented-feature-cache-run-id", cache_run_id,
            "--ggeur-headonly-eval-mode", "both",
            "--ggeur-terminal-client-eval-only",
            "--ggeur-num-generated-per-sample", "0",
            "--ggeur-num-generated-per-prototype", "10",
            "--ggeur-target-size-per-class", "10",
            "--ggeur-max-cross-client-prototypes-per-class", "4",
            "--ggeur-diagonal-covariance",
            "--ggeur-prototype-classifier-init",
            "--statistics-upload-stagger-seconds", "0.05",
            "--join-timeout-seconds", "300",
            "--stage-timeout-seconds", "1800",
            "--subserver-round-timeout-seconds", "900",
        ] + group_paths(group)
        if group == "mdsent_lstm":
            command.extend(["--train-learning-rate", "0.001"])
        if not first:
            command.append("--append-manifest")
        run(command)
        first = False


def configure_digit_task(run_id):
    case_name = "digit3_vit_platform"
    case_dir = RUNS_DIR / run_id / case_name
    if not case_dir.is_dir():
        return
    profile_path = case_dir / "mddigits_task_counts.json"
    profile_path.write_text(json.dumps({
        "default_target_size": 10,
        "class_counts": {str(class_id): 10 for class_id in range(10)},
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    changed = 0
    for path in sorted((case_dir / "configs").rglob("*.yaml")):
        if path.name == "root_server.yaml":
            repo = "/root/autodl-tmp/FederatedScope"
        elif path.parent.name == "clients_8g":
            repo = "D:/Projects/FederatedScope"
        else:
            continue
        remote_case = (
            f"{repo}/scripts/distributed_scripts/"
            f"ggeur_hierarchical_3machine/runs/{run_id}/{case_name}")
        content = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
        ggeur = content.setdefault("ggeur", {})
        ggeur["task_adaptation_file"] = (
            f"{remote_case}/mddigits_task_counts.json")
        ggeur["task_class_counts"] = []
        ggeur["task_default_target_size"] = ""
        ggeur["training_distribution_dir"] = (
            f"{remote_case}/training_distributions")
        path.write_text(yaml.safe_dump(
            content, sort_keys=False, allow_unicode=True), encoding="utf-8")
        changed += 1
    if changed != 13:
        raise RuntimeError(f"Expected 12 clients plus root, changed {changed}")


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

    run([sys.executable, MANIFEST_PREPARER])
    generate(args.preflight_run_id, 2, args.pipeline_run_id, ("platform",))
    configure_digit_task(args.preflight_run_id)
    validate(args.preflight_run_id, 2, [
        f"{group}_platform" for group in GROUPS])

    generate(args.formal_run_id, 100, args.pipeline_run_id, METHODS)
    configure_digit_task(args.formal_run_id)
    validate(args.formal_run_id, 100, [
        f"{group}_{method}" for group in GROUPS for method in METHODS])

    run([
        sys.executable, ASSEMBLER,
        "--pipeline-run-id", args.pipeline_run_id,
        "--smoke-run-id", args.preflight_run_id,
        "--formal-run-id", args.formal_run_id,
    ])
    manifest = RUNS_DIR / args.pipeline_run_id / "matrix_manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8-sig"))
    payload["acceptance_contract"] = {
        "rounds": 100,
        "eval_frequency": 1,
        "local_update_steps": 1,
        "client_num": 12,
        "clients_per_round": 12,
        "methods": list(METHODS),
        "feature_cache_required": True,
        "platform_generated_per_prototype": 10,
        "platform_target_size_per_class": 10,
        "platform_cache_preflight_rounds": 2,
    }
    manifest.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "status": "PASS",
        "pipeline_run_id": args.pipeline_run_id,
        "queued_cases": payload["case_count"],
        "formal_cases": payload["formal_case_count"],
        "client_num": 12,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
