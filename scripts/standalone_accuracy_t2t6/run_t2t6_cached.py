#!/usr/bin/env python3
"""Run the T2-T6 cached standalone accuracy experiments reproducibly.

Each method is first run for one round to populate/validate its augmented
feature cache.  The formal run then executes 100 rounds with the common
training parameters required by the test outline.
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

CASES = {
    "t2_digit3_vit": {
        "config_dir": "digit3_vit",
        "clients": 60,
        "common": {
            "device": 0,
            "seed": 42,
            "federate.mode": "standalone",
            "federate.client_num": 60,
            "federate.sample_client_num": 0,
            "dataloader.batch_size": 32,
            "dataloader.num_workers": 0,
            "train.local_update_steps": 5,
            "train.optimizer.type": "Adam",
            "train.optimizer.lr": 0.0001,
            "ggeur.use_feature_cache": True,
            "ggeur.require_complete_feature_cache": True,
            "ggeur.feature_cache_dir": "exp/distributed_feature_cache/digit3_vit",
            "ggeur.reuse_augmented_feature_cache": True,
            "ggeur.save_augmented_feature_cache": True,
            "ggeur.mlp_hidden_dim": 0,
            "ggeur.mlp_dropout": 0.0,
        },
    },
    "t3_officehome_cnn": {
        "config_dir": "officehome_cnn",
        "clients": 60,
        "common": {
            "device": 0,
            "seed": 42,
            "federate.mode": "standalone",
            "federate.client_num": 60,
            "federate.sample_client_num": 0,
            "dataloader.batch_size": 32,
            "dataloader.num_workers": 0,
            "train.local_update_steps": 2,
            "train.optimizer.type": "Adam",
            "train.optimizer.lr": 0.0001,
            "ggeur.use_feature_cache": True,
            "ggeur.require_complete_feature_cache": True,
            "ggeur.feature_cache_dir": "/root/autodl-tmp/datasets/clip_feature_cache",
            "ggeur.reuse_augmented_feature_cache": True,
            "ggeur.save_augmented_feature_cache": True,
            "ggeur.mlp_hidden_dim": 0,
            "ggeur.mlp_dropout": 0.0,
        },
    },
    "t4_officehome_mlp": {
        "config_dir": "officehome_mixer",
        "clients": 60,
        "common": {
            "device": 0,
            "seed": 42,
            "federate.mode": "standalone",
            "federate.client_num": 60,
            "federate.sample_client_num": 0,
            "dataloader.batch_size": 32,
            "dataloader.num_workers": 0,
            "train.local_update_steps": 2,
            "train.optimizer.type": "Adam",
            "train.optimizer.lr": 0.0001,
            "ggeur.use_feature_cache": True,
            "ggeur.require_complete_feature_cache": True,
            "ggeur.feature_cache_dir": "/root/autodl-tmp/datasets/clip_feature_cache",
            "ggeur.reuse_augmented_feature_cache": True,
            "ggeur.save_augmented_feature_cache": True,
            "ggeur.mlp_hidden_dim": 0,
            "ggeur.mlp_dropout": 0.0,
        },
    },
    "t5_mdsent_rnn": {
        "config_dir": "mdsent_rnn",
        "clients": 120,
        "common": {
            "device": 0,
            "seed": 42,
            "federate.mode": "standalone",
            "federate.client_num": 120,
            "federate.sample_client_num": 3,
            "dataloader.batch_size": 32,
            "dataloader.num_workers": 0,
            "train.local_update_steps": 1,
            "train.optimizer.type": "Adam",
            "train.optimizer.lr": 0.0001,
            "ggeur.min_statistics_clients": 3,
            "ggeur.min_augmentation_clients": 3,
            "ggeur.min_train_updates": 3,
            "ggeur.use_feature_cache": True,
            "ggeur.require_complete_feature_cache": True,
            "ggeur.feature_cache_dir": "data/clip_feature_cache",
            "ggeur.reuse_augmented_feature_cache": True,
            "ggeur.save_augmented_feature_cache": True,
        },
        "method_overrides": {
            "fedavg": {"ggeur.baseline_target_samples_per_client": 2},
            "fedprox": {"ggeur.baseline_target_samples_per_client": 2},
            "platform": {"ggeur.platform_target_samples_per_client": 20},
        },
    },
    "t6_mdsent_lstm": {
        "config_dir": "mdsent_lstm",
        "clients": 120,
        "common": {
            "device": 0,
            "seed": 42,
            "federate.mode": "standalone",
            "federate.client_num": 120,
            "federate.sample_client_num": 6,
            "dataloader.batch_size": 32,
            "dataloader.num_workers": 0,
            "train.local_update_steps": 1,
            "train.optimizer.type": "Adam",
            "train.optimizer.lr": 0.0001,
            "model.hidden": 64,
            "ggeur.min_statistics_clients": 6,
            "ggeur.min_augmentation_clients": 6,
            "ggeur.min_train_updates": 6,
            "ggeur.use_feature_cache": True,
            "ggeur.require_complete_feature_cache": True,
            "ggeur.feature_cache_dir": "data/clip_feature_cache",
            "ggeur.reuse_augmented_feature_cache": True,
            "ggeur.save_augmented_feature_cache": True,
        },
        "method_overrides": {
            "fedavg": {"ggeur.baseline_target_samples_per_client": 1},
            "fedprox": {"ggeur.baseline_target_samples_per_client": 1},
            "platform": {"ggeur.platform_target_samples_per_client": 300},
        },
    },
}

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
AVERAGE_RE = re.compile(
    r"average:\s*final=([0-9.]+)(?:,\s*best=([0-9.]+))?"
)
FINISHED_RE = re.compile(r"Training finished after\s+(\d+)\s+rounds")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def cli_value(value: object) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temp.replace(path)


def parse_log(log_path: Path) -> dict:
    text = ANSI_RE.sub("", log_path.read_text(encoding="utf-8", errors="replace"))
    averages = AVERAGE_RE.findall(text)
    finished = FINISHED_RE.findall(text)
    return {
        "average_final": float(averages[-1][0]) if averages else None,
        "average_best": (
            float(averages[-1][1]) if averages and averages[-1][1] else None
        ),
        "reported_rounds": int(finished[-1]) if finished else None,
        "augmented_cache_loads": text.count("Loaded augmented feature cache"),
        "augmented_cache_saves": text.count("Saved augmented feature cache"),
        "cache_hot_messages": text.count("Augmented cache-hot mode active"),
        "raw_cache_mentions": len(
            re.findall(r"(?:loaded|using).{0,80}feature cache", text, re.I)
        ),
    }


def source_config(config_root: Path, case: dict, method: str) -> Path:
    file_name = "ggeur.yaml" if method == "platform" else f"{method}.yaml"
    return config_root / case["config_dir"] / file_name


def make_command(
    python: str,
    config_root: Path,
    run_root: Path,
    case_name: str,
    case: dict,
    method: str,
    phase: str,
    rounds: int,
) -> list[str]:
    config = source_config(config_root, case, method)
    overrides = dict(case["common"])
    overrides.update(case.get("method_overrides", {}).get(method, {}))
    overrides["federate.total_round_num"] = rounds
    overrides["outdir"] = str(run_root / "outputs" / case_name / phase / method)
    overrides["expname"] = f"{case_name}_{method}_{phase}"
    command = [python, "federatedscope/main.py", "--cfg", str(config)]
    for key, value in overrides.items():
        command.extend((key, cli_value(value)))
    return command


def run_one(
    command: list[str], log_path: Path, cwd: Path, env: dict[str, str]
) -> tuple[int, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        log_file.write("COMMAND=" + subprocess.list2cmdline(command) + "\n")
        log_file.flush()
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log_file.write(line)
            if (
                "Training finished after" in line
                or "average: final=" in line
                or "Loaded augmented feature cache" in line
            ):
                print(line.rstrip(), flush=True)
        return_code = process.wait()
    return return_code, time.monotonic() - started


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--preflight-rounds", type=int, default=1)
    parser.add_argument("--run-id", default="t2t6_single_cached_20260814_v2")
    parser.add_argument("--only", nargs="*", choices=tuple(CASES))
    parser.add_argument("--methods", nargs="+", choices=METHODS)
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--sample-client-num", type=int)
    parser.add_argument("--local-update-steps", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--generated-target-per-class", type=int)
    parser.add_argument("--generated-per-sample", type=int)
    parser.add_argument("--generated-per-prototype", type=int)
    parser.add_argument("--baseline-target-samples-per-client", type=int)
    parser.add_argument("--platform-target-samples-per-client", type=int)
    parser.add_argument("--diagonal-covariance", action="store_true")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]
    config_root = repo / "scripts" / "example_configs" / "ggeur_final_5models"
    run_root = repo / "exp" / args.run_id
    summary_path = run_root / "summary.json"
    selected = args.only or list(CASES)
    selected_methods = args.methods or METHODS
    for case_name in selected:
        common = CASES[case_name]["common"]
        if args.sample_client_num is not None:
            if not 1 <= args.sample_client_num <= CASES[case_name]["clients"]:
                parser.error(
                    "--sample-client-num must be between 1 and the case's "
                    "configured client count"
                )
            common["federate.sample_client_num"] = args.sample_client_num
            common["ggeur.min_statistics_clients"] = args.sample_client_num
            common["ggeur.min_augmentation_clients"] = args.sample_client_num
            common["ggeur.min_train_updates"] = args.sample_client_num
        if args.local_update_steps is not None:
            if args.local_update_steps <= 0:
                parser.error("--local-update-steps must be positive")
            common["train.local_update_steps"] = args.local_update_steps
        if args.learning_rate is not None:
            if args.learning_rate <= 0:
                parser.error("--learning-rate must be positive")
            common["train.optimizer.lr"] = args.learning_rate
        if args.hidden_dim is not None:
            if args.hidden_dim <= 0:
                parser.error("--hidden-dim must be positive")
            common["model.hidden"] = args.hidden_dim
        if args.generated_target_per_class is not None:
            if args.generated_target_per_class <= 0:
                parser.error("--generated-target-per-class must be positive")
            CASES[case_name].setdefault("method_overrides", {}).setdefault(
                "platform", {}
            )["ggeur.target_size_per_class"] = args.generated_target_per_class
        for argument, config_key in (
            (args.generated_per_sample, "ggeur.num_generated_per_sample"),
            (args.generated_per_prototype, "ggeur.num_generated_per_prototype"),
        ):
            if argument is not None:
                if argument <= 0:
                    parser.error(
                        "--generated-per-sample and --generated-per-prototype "
                        "must be positive"
                    )
                CASES[case_name].setdefault("method_overrides", {}).setdefault(
                    "platform", {}
                )[config_key] = argument
        if args.baseline_target_samples_per_client is not None:
            if args.baseline_target_samples_per_client <= 0:
                parser.error(
                    "--baseline-target-samples-per-client must be positive"
                )
            for baseline in ("fedavg", "fedprox"):
                CASES[case_name].setdefault("method_overrides", {}).setdefault(
                    baseline, {}
                )[
                    "ggeur.baseline_target_samples_per_client"
                ] = args.baseline_target_samples_per_client
        if args.platform_target_samples_per_client is not None:
            if args.platform_target_samples_per_client <= 0:
                parser.error(
                    "--platform-target-samples-per-client must be positive"
                )
            CASES[case_name].setdefault("method_overrides", {}).setdefault(
                "platform", {}
            )[
                "ggeur.platform_target_samples_per_client"
            ] = args.platform_target_samples_per_client
        if args.diagonal_covariance:
            common["ggeur.diagonal_covariance"] = True
    summary = {
        "run_id": args.run_id,
        "repo": str(repo),
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "formal_rounds": args.rounds,
        "preflight_rounds": args.preflight_rounds,
        "cases": {},
    }
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    # The server also contains an older editable FederatedScope installation.
    # Put this isolated experiment tree first so federatedscope/main.py imports
    # the matching local package rather than that stale environment copy.
    old_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(repo) + (
        os.pathsep + old_pythonpath if old_pythonpath else ""
    )

    for case_name in selected:
        case = CASES[case_name]
        configured_sample_clients = int(
            case["common"].get("federate.sample_client_num", 0)
        )
        expected_clients = (
            configured_sample_clients
            if configured_sample_clients > 0
            else case["clients"]
        )
        case_result = summary["cases"].setdefault(
            case_name,
            {
                "common_parameters": case["common"],
                "method_overrides": case.get("method_overrides", {}),
                "expected_clients": expected_clients,
                "runs": {},
            },
        )
        case_result["common_parameters"] = case["common"]
        case_result["method_overrides"] = case.get("method_overrides", {})
        case_result["expected_clients"] = expected_clients
        phases = [] if args.skip_preflight else [("preflight", args.preflight_rounds)]
        phases.append(("formal", args.rounds))
        for phase, rounds in phases:
            for method in selected_methods:
                run_key = f"{phase}_{method}"
                previous = case_result["runs"].get(run_key, {})
                previous_rounds = previous.get("reported_rounds")
                previous_rounds_ok = (
                    previous_rounds == rounds
                    if phase == "formal"
                    else previous_rounds is not None and previous_rounds >= rounds
                )
                if previous.get("exit_code") == 0 and previous_rounds_ok:
                    print(f"SKIP completed {case_name} {run_key}", flush=True)
                    continue
                command = make_command(
                    args.python,
                    config_root,
                    run_root,
                    case_name,
                    case,
                    method,
                    phase,
                    rounds,
                )
                log_path = run_root / "logs" / f"{case_name}_{run_key}.log"
                print(
                    f"START {case_name} {phase} {method} rounds={rounds}",
                    flush=True,
                )
                started_at = utc_now()
                exit_code, elapsed = run_one(command, log_path, repo, env)
                parsed = parse_log(log_path)
                result = {
                    "phase": phase,
                    "method": method,
                    "rounds_requested": rounds,
                    "started_at": started_at,
                    "finished_at": utc_now(),
                    "elapsed_seconds": round(elapsed, 3),
                    "exit_code": exit_code,
                    "log": str(log_path.relative_to(repo)),
                    "command": command,
                    **parsed,
                }
                case_result["runs"][run_key] = result
                summary["updated_at"] = utc_now()
                atomic_json(summary_path, summary)
                print(
                    "DONE "
                    f"{case_name} {phase} {method} exit={exit_code} "
                    f"seconds={elapsed:.1f} average={parsed['average_final']} "
                    f"cache_loads={parsed['augmented_cache_loads']}",
                    flush=True,
                )
                reported_rounds = parsed["reported_rounds"]
                rounds_ok = (
                    reported_rounds == rounds
                    if phase == "formal"
                    else reported_rounds is not None and reported_rounds >= rounds
                )
                if exit_code != 0 or not rounds_ok:
                    print(f"STOP failed run; inspect {log_path}", flush=True)
                    return 1
                cache_coverage = (
                    parsed["augmented_cache_loads"]
                    + parsed["augmented_cache_saves"]
                )
                if phase == "preflight" and cache_coverage < expected_clients:
                    print(
                        "STOP preflight did not load or save a cache for every client: "
                        f"expected at least {expected_clients}, got {cache_coverage}",
                        flush=True,
                    )
                    return 3
                if (phase == "formal" and
                        parsed["augmented_cache_loads"] < expected_clients):
                    print(
                        "STOP formal run was not fully augmented-cache-hot: "
                        f"expected at least {expected_clients} cache loads, got "
                        f"{parsed['augmented_cache_loads']}",
                        flush=True,
                    )
                    return 2

        formal = {
            method: case_result["runs"].get(f"formal_{method}", {})
            for method in METHODS
        }
        platform = formal["platform"].get("average_final")
        fedavg = formal["fedavg"].get("average_final")
        fedprox = formal["fedprox"].get("average_final")
        if platform is not None and fedavg is not None and fedprox is not None:
            case_result["comparison"] = {
                "platform_minus_fedavg_points": round((platform - fedavg) * 100, 2),
                "platform_minus_fedprox_points": round((platform - fedprox) * 100, 2),
                "passes_20_point_requirement": (
                    platform - fedavg >= 0.2 and platform - fedprox >= 0.2
                ),
            }
            atomic_json(summary_path, summary)

    summary["completed_at"] = utc_now()
    atomic_json(summary_path, summary)
    print(f"ALL_DONE summary={summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
