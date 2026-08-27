#!/usr/bin/env python3
"""Check all local resources required by the T2-T6 4090 profile."""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from federatedscope.core.configs.config import global_cfg


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    plan_path = Path(__file__).with_name("final_experiment.json")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    checks: list[dict] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})
        print(f"{'PASS' if ok else 'FAIL'} {name}: {detail}")

    check("python_version", sys.version_info[:2] == (3, 9), sys.version.split()[0])
    for module in (
        "torch",
        "numpy",
        "yaml",
        "torchvision",
        "timm",
        "transformers",
        "open_clip",
    ):
        check(
            f"python_module_{module}",
            importlib.util.find_spec(module) is not None,
            module,
        )

    try:
        import torch

        check("cuda_available", torch.cuda.is_available(), str(torch.cuda.is_available()))
        if torch.cuda.is_available():
            check("cuda_device", True, torch.cuda.get_device_name(0))
    except Exception as error:
        check("cuda_import", False, repr(error))

    nvidia_smi = shutil.which("nvidia-smi")
    check("nvidia_smi", bool(nvidia_smi), nvidia_smi or "not found")
    if nvidia_smi:
        probe = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        check("gpu_probe", probe.returncode == 0, probe.stdout.strip())

    required_paths = {
        "digit3_manifest": repo / "data/digit_three_domain/dataset_manifest.json",
        "officehome_dataset": Path(
            "/root/autodl-tmp/datasets/OfficeHomeDataset_10072016"
        ),
        "mdsent_dataset": repo / "data/sentiment",
        "clip_vit_b16": Path("/root/.cache/clip/ViT-B-16.pt"),
        "mixer_b16": Path("/root/autodl-tmp/models/mixer_b16_224_complete.pth"),
        "bert_mdsent": Path(
            "/root/autodl-tmp/models/nlptown_bert_base_multilingual_uncased_senti"
        ),
    }
    for name, path in required_paths.items():
        check(name, path.exists(), str(path))
    digit_client_manifests = list(
        (repo / "data/digit_three_domain/manifests").glob(
            "client_*/client_manifest.json"
        )
    )
    check(
        "digit3_client_manifests",
        len(digit_client_manifests) == 60,
        f"count={len(digit_client_manifests)} expected=60",
    )
    convnext_path = repo / "pretrained_models/convnext_base-6075fbad.pth"
    cnn_cache_marker = (
        repo / "exp/distributed_feature_cache/officehome_cnn/"
        ".ggeur_feature_cache_ready.json"
    )
    check(
        "convnext_base_or_complete_cache",
        convnext_path.is_file() or cnn_cache_marker.is_file(),
        f"checkpoint={convnext_path.is_file()} "
        f"complete_cache={cnn_cache_marker.is_file()}",
    )

    config_root = repo / "scripts/example_configs/ggeur_final_5models"
    for case_name, case in plan["cases"].items():
        for method in ("fedavg", "fedprox", "ggeur"):
            path = config_root / case["config_dir"] / f"{method}.yaml"
            check(f"{case_name}_{method}_config", path.is_file(), str(path))
            if path.is_file():
                try:
                    parsed_cfg = global_cfg.clone()
                    parsed_cfg.merge_from_file(str(path))
                    check(
                        f"{case_name}_{method}_config_load",
                        True,
                        "configuration keys and value types accepted",
                    )
                except Exception as error:
                    check(
                        f"{case_name}_{method}_config_load",
                        False,
                        repr(error),
                    )
        task_path = repo / case["task_profile"]
        check(f"{case_name}_task_profile", task_path.is_file(), str(task_path))
        if task_path.is_file():
            payload = json.loads(task_path.read_text(encoding="utf-8"))
            counts = payload.get("class_counts", {})
            correct = (
                len(counts) == int(case["task_class_count"])
                and set(int(value) for value in counts.values())
                == {int(case["task_target_per_class"])}
            )
            check(
                f"{case_name}_task_profile_values",
                correct,
                f"classes={len(counts)} targets={sorted(set(counts.values()))}",
            )
        cache_path = Path(case["feature_cache_dir"])
        if not cache_path.is_absolute():
            cache_path = repo / cache_path
        cache_files = list(cache_path.rglob("*.npz")) if cache_path.is_dir() else []
        ready_marker = cache_path / ".ggeur_feature_cache_ready.json"
        check(
            f"{case_name}_raw_feature_cache",
            cache_path.is_dir() and bool(cache_files) and ready_marker.is_file(),
            f"path={cache_path} npz_files={len(cache_files)} "
            f"ready_marker={ready_marker.is_file()}",
        )

    report = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "repo": str(repo),
        "python": sys.executable,
        "all_passed": all(item["ok"] for item in checks),
        "checks": checks,
    }
    report_path = repo / "exp/t2t6_4090_preflight.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"REPORT={report_path}")
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
