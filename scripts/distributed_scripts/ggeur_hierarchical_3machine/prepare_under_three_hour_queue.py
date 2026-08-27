#!/usr/bin/env python3
"""Select only missing tests and record the complete reusable evidence."""

import argparse
import json
import shutil
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
RUNS_DIR = THIS_DIR / "runs"
SOURCE_RUN_ID = "meeting_3h_20260813_v1"
SELECTED_CASES = (
    "digit3_vit_preflight_platform",
    "mdsent_lstm_preflight_platform",
    "mdsent_lstm_fedavg",
    "mdsent_lstm_fedprox",
    "mdsent_lstm_platform",
)
REUSED_RESULTS = {
    "digit3_vit_fedavg": "digit3_vit_dual_20260812_v1",
    "digit3_vit_fedprox": "digit3_vit_dual_20260812_v1",
    "digit3_vit_platform": "digit3_vit_dual_20260812_v1",
    "officehome_cnn_fedavg": "final_remaining_20260723_v1",
    "officehome_cnn_fedprox": "final_remaining_20260723_v1",
    "officehome_cnn_platform": "final_remaining_20260723_v1",
    "officehome_mixer_fedavg": "final_remaining_20260723_v1",
    "officehome_mixer_fedprox": "final_remaining_20260723_v1",
    "officehome_mixer_platform": "final_remaining_20260723_v1",
    "mdsent_rnn_fedavg": "final_remaining_20260723_v1",
    "mdsent_rnn_fedprox": "final_remaining_20260723_v1",
    "mdsent_rnn_platform": "mdsent_reference_distributed_20260730_v1",
}


def rewrite_tree(root, old_run_id, new_run_id):
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {
                ".json", ".yaml", ".yml", ".ps1", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8-sig")
        path.write_text(
            text.replace(old_run_id, new_run_id), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    source = RUNS_DIR / SOURCE_RUN_ID
    destination = RUNS_DIR / args.run_id
    if destination.exists():
        raise SystemExit(f"destination already exists: {destination}")
    destination.mkdir(parents=True)
    source_matrix = json.loads(
        (source / "matrix_manifest.json").read_text(encoding="utf-8-sig"))
    by_case = {item["case"]: item for item in source_matrix["cases"]}
    cases = []
    phases = []
    for case_name in SELECTED_CASES:
        shutil.copytree(source / case_name, destination / case_name)
        rewrite_tree(destination / case_name, SOURCE_RUN_ID, args.run_id)
        manifest = json.loads((
            destination / case_name / "manifest.json").read_text(
                encoding="utf-8-sig"))
        cases.append(manifest)
        phases.append({
            "case": case_name,
            "phase": manifest.get("phase", "formal"),
        })
    matrix = {
        "run_id": args.run_id,
        "case_count": len(cases),
        "preflight_case_count": 2,
        "formal_case_count": 3,
        "output_root": str(destination).replace("\\", "/"),
        "phases": phases,
        "cases": cases,
        "reused_completed_results": REUSED_RESULTS,
        "acceptance_contract": {
            "formal_rounds": 100,
            "eval_frequency": 1,
            "local_update_steps": 1,
            "client_num": 12,
            "lstm_methods": ["fedavg", "fedprox", "platform"],
            "feature_cache_required": True,
            "platform_augmented_cache_persisted": True,
        },
    }
    (destination / "matrix_manifest.json").write_text(
        json.dumps(matrix, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "status": "PASS",
        "run_id": args.run_id,
        "queued_cases": len(cases),
        "reused_cases": len(REUSED_RESULTS),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
