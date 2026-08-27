#!/usr/bin/env python3
"""Create a one-case three-host run that reuses an existing cache namespace."""

import argparse
import json
import shutil
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
RUNS_DIR = THIS_DIR / "runs"


def rewrite_text_files(case_dir, source_run_id, run_id, case_name):
    source_cache = f"exp/aug/{source_run_id}/{case_name}"
    rewritten_cache = f"exp/aug/{run_id}/{case_name}"
    for path in case_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {
                ".json", ".yaml", ".yml", ".txt", ".ps1", ".sh"}:
            continue
        text = path.read_text(encoding="utf-8-sig")
        text = text.replace(source_run_id, run_id)
        # The timing run must read the already generated samples rather than
        # create a new namespace just because its evidence directory is new.
        text = text.replace(rewritten_cache, source_cache)
        path.write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--case", default="digit3_vit_platform")
    args = parser.parse_args()

    source_case = RUNS_DIR / args.source_run_id / args.case
    destination = RUNS_DIR / args.run_id
    destination_case = destination / args.case
    if not source_case.is_dir():
        raise FileNotFoundError(source_case)
    if destination.exists():
        raise FileExistsError(destination)

    shutil.copytree(source_case, destination_case)
    rewrite_text_files(destination_case, args.source_run_id, args.run_id,
                       args.case)
    manifest_path = destination_case / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    manifest["run_id"] = args.run_id
    manifest["cache_timing"] = {
        "source_run_id": args.source_run_id,
        "reused_augmented_cache": (
            f"exp/aug/{args.source_run_id}/{args.case}"),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    matrix = {
        "run_id": args.run_id,
        "case_count": 1,
        "cases": [manifest],
        "cache_timing": True,
    }
    (destination / "matrix_manifest.json").write_text(
        json.dumps(matrix, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "PASS",
        "run_id": args.run_id,
        "case": args.case,
        "total_round_num": 100,
        "reused_augmented_cache": (
            f"exp/aug/{args.source_run_id}/{args.case}"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
