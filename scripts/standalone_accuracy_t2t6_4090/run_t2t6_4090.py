#!/usr/bin/env python3
"""Run the final T2-T6 profile on one 4090 server.

The platform cache warm-up is always completed before the three formal
methods.  FedAvg and FedProx may read raw extractor features, but their
generated/augmented cache is explicitly disabled.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


METHODS = ("fedavg", "fedprox", "platform")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
AVERAGE_RE = re.compile(r"average:\s*final=([0-9.]+)(?:,\s*best=([0-9.]+))?")
FINISHED_RE = re.compile(r"Training finished after\s+(\d+)\s+rounds")
ROUND_ACCURACY_RE = re.compile(
    r"Server:\s*Round\s+(\d+).*?\baverage:\s*([0-9.]+)"
)


def reported_rounds_are_valid(phase: str, requested: int, reported: object) -> bool:
    """Generation warm-up has an explicit round-0 statistics phase."""
    if not isinstance(reported, int):
        return False
    if phase == "cache_warmup":
        return reported in {requested, requested + 1}
    return reported == requested


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def cli_value(value: object) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (list, dict)):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def parse_log(path: Path) -> dict:
    text = ANSI_RE.sub("", path.read_text(encoding="utf-8", errors="replace"))
    averages = AVERAGE_RE.findall(text)
    rounds = FINISHED_RE.findall(text)
    round_values = {
        int(round_index): float(accuracy)
        for round_index, accuracy in ROUND_ACCURACY_RE.findall(text)
    }
    return {
        "average_final": float(averages[-1][0]) if averages else None,
        "average_best": (
            float(averages[-1][1]) if averages and averages[-1][1] else None
        ),
        "reported_rounds": int(rounds[-1]) if rounds else None,
        "round_accuracies": [
            {"round": round_index, "average_accuracy": round_values[round_index]}
            for round_index in sorted(round_values)
        ],
        "augmented_cache_loads": text.count("Loaded augmented feature cache"),
        "augmented_cache_saves": text.count("Saved augmented feature cache"),
        "task_profile_loads": text.count(
            "Loaded task-adaptive generation profile"
        ),
        "distribution_outputs": text.count("PLATFORM_TRAIN_DISTRIBUTION"),
        "generated_sample_events": len(
            re.findall(r"generated_from_(?:samples|prototypes)=[1-9][0-9]*", text)
        ),
    }


def source_config(config_root: Path, case: dict, method: str) -> Path:
    filename = "ggeur.yaml" if method == "platform" else f"{method}.yaml"
    return config_root / case["config_dir"] / filename


def resolve_feature_cache_dir(repo: Path, case: dict) -> str:
    candidates = case.get(
        "feature_cache_candidates", [case["feature_cache_dir"]]
    )
    resolved = []
    for candidate in candidates:
        path = Path(candidate)
        if not path.is_absolute():
            path = repo / path
        resolved.append(path)
        if path.exists():
            return str(path)
    return str(resolved[0])


def effective_overrides(
    plan: dict,
    case_name: str,
    case: dict,
    method: str,
    phase: str,
    rounds: int,
    run_root: Path,
    platform_cache_root: Path,
) -> dict:
    overrides = dict(plan["common"])
    minimum = overrides.pop("minimum_absolute_improvement", None)
    assert minimum is not None
    overrides.update(case.get("resource_overrides", {}))
    phase_client_num = int(case["sample_client_num"])
    if phase == "cache_warmup":
        phase_client_num = int(
            case.get("warmup_sample_client_num", phase_client_num))
    overrides.update(
        {
            "federate.client_num": case["client_num"],
            "federate.sample_client_num": phase_client_num,
            "federate.total_round_num": rounds,
            "train.local_update_steps": case["local_update_steps"],
            "ggeur.use_feature_cache": True,
            "ggeur.feature_cache_dir": case["resolved_feature_cache_dir"],
            "ggeur.require_complete_feature_cache": phase == "formal",
        }
    )
    if phase == "cache_warmup" and "warmup_sample_client_num" in case:
        overrides.update({
            "ggeur.min_statistics_clients": phase_client_num,
            "ggeur.min_augmentation_clients": phase_client_num,
            "ggeur.min_train_updates": phase_client_num,
        })
    overrides.update(case["method_overrides"][method])

    if method == "platform":
        overrides.update(
            {
                "ggeur.reuse_augmented_feature_cache": True,
                "ggeur.save_augmented_feature_cache": True,
                "ggeur.augmented_feature_cache_dir": str(
                    platform_cache_root / case_name
                ),
                "ggeur.task_adaptation_file": case["task_profile"],
                "ggeur.task_class_counts": [],
                "ggeur.task_default_target_size": "",
                "ggeur.training_distribution_dir": str(
                    run_root
                    / "task_adaptation"
                    / case["test_case"]
                    / "training_distributions"
                ),
            }
        )
    else:
        overrides.update(
            {
                "ggeur.reuse_augmented_feature_cache": False,
                "ggeur.save_augmented_feature_cache": False,
                "ggeur.task_adaptation_file": "",
                "ggeur.task_class_counts": [],
                "ggeur.task_default_target_size": "",
                "ggeur.training_distribution_dir": "",
            }
        )

    overrides["outdir"] = str(
        run_root / "outputs" / case_name / phase / method
    )
    overrides["expname"] = f"{case_name}_{method}_{phase}"
    overrides["log_file"] = str(
        run_root / "framework_logs" / f"{case_name}_{phase}_{method}.log"
    )
    return overrides


def build_command(
    python: str,
    config_root: Path,
    plan: dict,
    case_name: str,
    case: dict,
    method: str,
    phase: str,
    rounds: int,
    run_root: Path,
    platform_cache_root: Path,
) -> tuple[list[str], dict]:
    config = source_config(config_root, case, method)
    overrides = effective_overrides(
        plan, case_name, case, method, phase, rounds, run_root,
        platform_cache_root
    )
    command = [python, "federatedscope/main.py", "--cfg", str(config)]
    for key, value in overrides.items():
        command.extend((key, cli_value(value)))
    return command, overrides


def run_command(
    command: list[str], cwd: Path, log_path: Path, environment: dict[str, str]
) -> tuple[int, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8", buffering=1) as stream:
        stream.write("COMMAND=" + subprocess.list2cmdline(command) + "\n")
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            stream.write(line)
            if any(
                marker in line
                for marker in (
                    "Training finished after",
                    "average: final=",
                    "Loaded augmented feature cache",
                    "Loaded task-adaptive generation profile",
                )
            ):
                print(line.rstrip(), flush=True)
        return_code = process.wait()
    return return_code, time.monotonic() - started


def validate_task_reports(
    repo: Path, run_root: Path, case: dict, not_before: float
) -> dict:
    profile_path = repo / case["task_profile"]
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    distribution_dir = (
        run_root
        / "task_adaptation"
        / case["test_case"]
        / "training_distributions"
    )
    reports = sorted(
        path
        for path in distribution_dir.glob("client_*.json")
    )
    valid = 0
    for report_path in reports:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        task = report.get("task_adaptation", {})
        if not task.get("enabled"):
            continue
        expected = {str(key): int(value) for key, value in profile["class_counts"].items()}
        actual = {
            str(key): int(value)
            for key, value in task.get("class_targets", {}).items()
        }
        if actual == expected:
            valid += 1
    return {
        "directory": str(distribution_dir.relative_to(repo)),
        "report_files": len(reports),
        "reports_with_expected_task_targets": valid,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--run-id", default="t2t6_4090_final")
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--warmup-rounds", type=int, default=1)
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--methods", nargs="+", choices=METHODS)
    parser.add_argument("--skip-cache-warmup", action="store_true")
    parser.add_argument("--cuda-visible-device", choices=("0", "1"), default="0")
    parser.add_argument("--platform-cache-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]
    plan_path = args.plan or Path(__file__).with_name("final_experiment.json")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if args.seed is not None:
        plan["common"]["seed"] = args.seed
    cases = plan["cases"]
    selected_cases = args.only or list(cases)
    unknown = sorted(set(selected_cases) - set(cases))
    if unknown:
        parser.error(f"unknown cases: {', '.join(unknown)}")
    selected_methods = args.methods or list(METHODS)
    formal_rounds = args.rounds or int(
        plan["common"]["federate.total_round_num"]
    )
    if formal_rounds <= 0 or args.warmup_rounds <= 0:
        parser.error("round counts must be positive")

    config_root = repo / "scripts" / "example_configs" / "ggeur_final_5models"
    run_root = repo / "exp" / args.run_id
    (run_root / "framework_logs").mkdir(parents=True, exist_ok=True)
    platform_cache_root = args.platform_cache_root or (
        Path("exp") / "t2t6_4090_platform_cache" / args.run_id
    )
    summary_path = run_root / "summary.json"
    summary = {
        "run_id": args.run_id,
        "plan": str(plan_path.relative_to(repo)),
        "profile_name": plan["profile_name"],
        "created_at": now(),
        "updated_at": now(),
        "formal_rounds": formal_rounds,
        "training_seed": int(plan["common"]["seed"]),
        "data_seed": args.data_seed,
        "cuda_visible_device": args.cuda_visible_device,
        "platform_cache_root": str(platform_cache_root),
        "cases": {},
    }
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONUNBUFFERED": "1",
            "CUDA_VISIBLE_DEVICES": args.cuda_visible_device,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONPATH": str(repo)
            + (os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""),
        }
    )

    for case_name in selected_cases:
        case = dict(cases[case_name])
        if case_name in {"t3_officehome_cnn", "t4_officehome_mlp"}:
            case.setdefault("resource_overrides", {})[
                "ggeur.officehome_data_seed"
            ] = args.data_seed
            case["resource_overrides"]["ggeur.lds_seed"] = args.data_seed
        case["resolved_feature_cache_dir"] = resolve_feature_cache_dir(
            repo, case
        )
        expected_active_clients = int(case["sample_client_num"] or case["client_num"])
        case_summary = summary["cases"].setdefault(
            case_name,
            {
                "test_case": case["test_case"],
                "model_name": case["model_name"],
                "dataset_name": case["dataset_name"],
                "recorded_result_reference": case["recorded_result"],
                "resolved_feature_cache_dir": case[
                    "resolved_feature_cache_dir"
                ],
                "runs": {},
            },
        )

        phases: list[tuple[str, str, int]] = []
        if not args.skip_cache_warmup:
            phases.append(("cache_warmup", "platform", args.warmup_rounds))
        phases.extend(("formal", method, formal_rounds) for method in selected_methods)

        for phase, method, rounds in phases:
            run_key = f"{phase}_{method}"
            previous = case_summary["runs"].get(run_key, {})
            if (
                previous.get("exit_code") == 0
                and reported_rounds_are_valid(
                    phase, rounds, previous.get("reported_rounds")
                )
            ):
                print(f"SKIP completed {case_name} {run_key}", flush=True)
                continue

            command, overrides = build_command(
                args.python,
                config_root,
                plan,
                case_name,
                case,
                method,
                "formal" if phase == "formal" else "cache_warmup",
                rounds,
                run_root,
                platform_cache_root,
            )
            log_path = run_root / "logs" / f"{case_name}_{run_key}.log"
            print(f"START {case_name} {phase} {method} rounds={rounds}", flush=True)
            if args.dry_run:
                print(subprocess.list2cmdline(command))
                continue
            started_at = now()
            started_epoch = time.time()
            exit_code, elapsed = run_command(command, repo, log_path, environment)
            parsed = parse_log(log_path)
            round_accuracy_path = (
                run_root / "round_accuracy" /
                f"{case_name}_{run_key}.json"
            )
            write_json(round_accuracy_path, {
                "case": case_name,
                "test_case": case["test_case"],
                "method": method,
                "phase": phase,
                "log": str(log_path.relative_to(repo)),
                "rounds_requested": rounds,
                "round_accuracies": parsed["round_accuracies"],
            })
            result = {
                "phase": phase,
                "method": method,
                "rounds_requested": rounds,
                "started_at": started_at,
                "finished_at": now(),
                "elapsed_seconds": round(elapsed, 3),
                "exit_code": exit_code,
                "log": str(log_path.relative_to(repo)),
                "round_accuracy_file": str(
                    round_accuracy_path.relative_to(repo)),
                "command": command,
                "effective_overrides": overrides,
                **parsed,
            }
            case_summary["runs"][run_key] = result
            summary["updated_at"] = now()
            write_json(summary_path, summary)
            print(
                f"DONE {case_name} {phase} {method} exit={exit_code} "
                f"seconds={elapsed:.1f} accuracy={parsed['average_final']}",
                flush=True,
            )

            if exit_code != 0 or not reported_rounds_are_valid(
                phase, rounds, parsed["reported_rounds"]
            ):
                print(f"STOP failed run; inspect {log_path}", flush=True)
                return 1
            recorded_rounds = [
                item["round"] for item in parsed["round_accuracies"]
            ]
            expected_accuracy_rounds = list(range(parsed["reported_rounds"]))
            if recorded_rounds != expected_accuracy_rounds:
                print(
                    "STOP per-round accuracy record is incomplete; "
                    f"expected=0..{parsed['reported_rounds'] - 1} "
                    f"actual_count={len(recorded_rounds)} "
                    f"inspect {round_accuracy_path}",
                    flush=True,
                )
                return 5
            if method != "platform" and (
                parsed["augmented_cache_loads"] or parsed["augmented_cache_saves"]
            ):
                print(
                    "STOP baseline accessed a generated/augmented cache; "
                    f"inspect {log_path}",
                    flush=True,
                )
                return 2
            if phase == "formal" and method == "platform":
                if parsed["augmented_cache_loads"] < expected_active_clients:
                    print(
                        "STOP platform formal run was not cache-hot for every "
                        f"active client; inspect {log_path}",
                        flush=True,
                    )
                    return 3
                task_validation = validate_task_reports(
                    repo, run_root, case, started_epoch
                )
                result["task_validation"] = task_validation
                if task_validation["reports_with_expected_task_targets"] < expected_active_clients:
                    print(
                        "STOP task-adaptation reports are incomplete; "
                        f"inspect {task_validation['directory']}",
                        flush=True,
                    )
                    write_json(summary_path, summary)
                    return 4
                write_json(summary_path, summary)

        formal = {
            method: case_summary["runs"].get(f"formal_{method}", {})
            for method in METHODS
        }
        if all(formal[method].get("average_final") is not None for method in METHODS):
            platform = float(formal["platform"]["average_final"])
            fedavg = float(formal["fedavg"]["average_final"])
            fedprox = float(formal["fedprox"]["average_final"])
            threshold = float(plan["common"]["minimum_absolute_improvement"])
            case_summary["comparison"] = {
                "platform_minus_fedavg_points": round((platform - fedavg) * 100, 2),
                "platform_minus_fedprox_points": round((platform - fedprox) * 100, 2),
                "minimum_required_points": round(threshold * 100, 2),
                "passes_requirement": (
                    platform - fedavg >= threshold
                    and platform - fedprox >= threshold
                ),
            }
            write_json(summary_path, summary)

    if not args.dry_run:
        summary["completed_at"] = now()
        write_json(summary_path, summary)
        print(f"ALL_DONE summary={summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
