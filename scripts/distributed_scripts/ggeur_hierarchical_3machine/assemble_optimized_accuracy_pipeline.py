#!/usr/bin/env python3
"""Assemble cache-building preflights and formal cases into one queue."""

import argparse
import json
import shutil
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
RUNS_DIR = THIS_DIR / "runs"


def rewrite_tree(root, replacements, augmented_case=None):
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {
                ".json", ".yaml", ".yml", ".ps1", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8-sig")
        for old, new in replacements:
            text = text.replace(old, new)
        if augmented_case is not None:
            renamed_case, cache_case = augmented_case
            lines = []
            for line in text.splitlines(keepends=True):
                if "augmented_feature_cache_dir:" in line:
                    line = line.replace(renamed_case, cache_case)
                lines.append(line)
            text = "".join(lines)
        path.write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline-run-id", required=True)
    parser.add_argument("--smoke-run-id", required=True)
    parser.add_argument("--formal-run-id", required=True)
    args = parser.parse_args()

    destination = RUNS_DIR / args.pipeline_run_id
    if destination.exists():
        raise SystemExit(f"pipeline destination already exists: {destination}")
    destination.mkdir(parents=True)

    phases = []
    combined = []
    smoke_root = RUNS_DIR / args.smoke_run_id
    smoke_matrix = json.loads((smoke_root / "matrix_manifest.json").read_text(
        encoding="utf-8-sig"))
    for item in smoke_matrix["cases"]:
        old_case = item["case"]
        new_case = f"{item['group']}_preflight_{item['method']}"
        source = smoke_root / old_case
        target = destination / new_case
        shutil.copytree(source, target)
        rewrite_tree(target, [
            (args.smoke_run_id, args.pipeline_run_id),
            (old_case, new_case),
        ], augmented_case=(new_case, old_case))
        manifest = json.loads((target / "manifest.json").read_text(
            encoding="utf-8-sig"))
        manifest["phase"] = "preflight_and_cache"
        (target / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        combined.append(manifest)
        phases.append({"case": new_case, "phase": "preflight_and_cache"})

    formal_root = RUNS_DIR / args.formal_run_id
    formal_matrix = json.loads((formal_root / "matrix_manifest.json").read_text(
        encoding="utf-8-sig"))
    for item in formal_matrix["cases"]:
        case_name = item["case"]
        source = formal_root / case_name
        target = destination / case_name
        shutil.copytree(source, target)
        rewrite_tree(target, [
            (args.formal_run_id, args.pipeline_run_id),
        ])
        manifest = json.loads((target / "manifest.json").read_text(
            encoding="utf-8-sig"))
        manifest["phase"] = "formal"
        (target / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        combined.append(manifest)
        phases.append({"case": case_name, "phase": "formal"})

    matrix = {
        "run_id": args.pipeline_run_id,
        "case_count": len(combined),
        "preflight_case_count": len(smoke_matrix["cases"]),
        "formal_case_count": len(formal_matrix["cases"]),
        "output_root": str(destination).replace("\\", "/"),
        "phases": phases,
        "cases": combined,
    }
    (destination / "matrix_manifest.json").write_text(
        json.dumps(matrix, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "status": "pass",
        "pipeline_run_id": args.pipeline_run_id,
        "preflight_case_count": len(smoke_matrix["cases"]),
        "formal_case_count": len(formal_matrix["cases"]),
        "case_count": len(combined),
        "manifest": str(destination / "matrix_manifest.json"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
