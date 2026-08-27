#!/usr/bin/env python3
"""Run one T2-T6 method for three seeds and save reloadable heads."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
CASE_CONFIGS = {
    "T-02": {
        "fedavg": "scripts/military_aircraft_3domain/configs/fedavg_pilot.yaml",
        "fedprox": "scripts/military_aircraft_3domain/configs/fedprox_pilot.yaml",
        "platform": "scripts/military_aircraft_3domain/configs/platform_pilot.yaml",
        "task_file": "scripts/test_outline_validation/task_profiles/military_aircraft_class_targets.json",
        "fixed_lds_seed": 42,
    },
    "T-03": {
        "fedavg": "scripts/example_configs/ggeur_final_5models/digit3_cnn/fedavg.yaml",
        "fedprox": "scripts/example_configs/ggeur_final_5models/digit3_cnn/fedprox.yaml",
        "platform": "scripts/example_configs/ggeur_final_5models/digit3_cnn/ggeur.yaml",
        "task_file": "scripts/test_outline_validation/task_profiles/mddigits_class_targets.json",
    },
    "T-04": {
        "fedavg": "scripts/example_configs/ggeur_final_5models/officehome_mixer/fedavg.yaml",
        "fedprox": "scripts/example_configs/ggeur_final_5models/officehome_mixer/fedprox.yaml",
        "platform": "scripts/example_configs/ggeur_final_5models/officehome_mixer/ggeur.yaml",
        "task_file": "scripts/test_outline_validation/task_profiles/officehome_accuracy_class_targets.json",
    },
    "T-05": {
        "fedavg": "scripts/example_configs/ggeur_final_5models/mdsent_rnn/fedavg.yaml",
        "fedprox": "scripts/example_configs/ggeur_final_5models/mdsent_rnn/fedprox.yaml",
        "platform": "scripts/example_configs/ggeur_final_5models/mdsent_rnn/ggeur.yaml",
        "task_file": "scripts/test_outline_validation/task_profiles/mdsent_class_targets.json",
        "fixed_lds_seed": 42,
        "fixed_classifier_init_seed": 2025,
        "full_statistics_clients": 120,
    },
    "T-06": {
        "fedavg": "scripts/example_configs/ggeur_final_5models/mdsent_lstm/fedavg.yaml",
        "fedprox": "scripts/example_configs/ggeur_final_5models/mdsent_lstm/fedprox.yaml",
        "platform": "scripts/example_configs/ggeur_final_5models/mdsent_lstm/ggeur.yaml",
        "task_file": "scripts/test_outline_validation/task_profiles/mdsent_class_targets.json",
        "fixed_lds_seed": 42,
        "fixed_classifier_init_seed": 2026,
        "full_statistics_clients": 120,
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, choices=sorted(CASE_CONFIGS))
    parser.add_argument(
        "--method", required=True,
        choices=("fedavg", "fedprox", "platform"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def stream_command(command, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", newline="") as log_stream:
        process = subprocess.Popen(
            [str(item) for item in command], cwd=str(REPO),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1)
        for line in process.stdout:
            print(line, end="", flush=True)
            log_stream.write(line)
        return_code = process.wait()
    if return_code:
        raise RuntimeError(
            "training failed with exit code {}: {}".format(
                return_code, log_path))


def main():
    args = parse_args()
    spec = CASE_CONFIGS[args.case]
    config = REPO / spec[args.method]
    if not config.is_file():
        raise FileNotFoundError(config)
    summaries = []
    for seed in args.seeds:
        run_dir = (
            REPO / "exp" / "test_outline_validation" / args.case /
            "seed_{}".format(seed) / args.method)
        checkpoint_dir = run_dir / "checkpoints"
        log_path = run_dir / "training.log"
        if args.method == "platform" and spec.get(
                "full_statistics_clients"):
            full_clients = int(spec["full_statistics_clients"])
            warmup_dir = run_dir / "cache_warmup"
            warmup_log = run_dir / "cache_warmup.log"
            warmup_overrides = [
                "seed", str(seed),
                "ggeur.lds_seed", str(spec.get("fixed_lds_seed", seed)),
                "federate.total_round_num", "1",
                "federate.sample_client_num", str(full_clients),
                "ggeur.min_statistics_clients", str(full_clients),
                "ggeur.min_augmentation_clients", str(full_clients),
                "ggeur.min_train_updates", str(full_clients),
                "outdir", warmup_dir.as_posix(),
                "expname", "{}_platform_seed_{}_cache_warmup".format(
                    args.case.lower().replace("-", ""), seed),
                "log_file", warmup_log.as_posix(),
                "ggeur.task_adaptation_file", spec["task_file"],
                "ggeur.training_distribution_dir",
                (warmup_dir / "training_distributions").as_posix(),
                "ggeur.save_mlp_checkpoint", "False",
            ]
            warmup_command = [
                sys.executable, "-m", "federatedscope.main",
                "--cfg", config.relative_to(REPO).as_posix(),
                *warmup_overrides,
            ]
            print(
                "PLATFORM_CACHE_WARMUP_START case={} seed={} clients={}".
                format(args.case, seed, full_clients), flush=True)
            if args.plan_only:
                print(subprocess.list2cmdline(warmup_command))
            else:
                stream_command(warmup_command, warmup_log)
                print(
                    "PLATFORM_CACHE_WARMUP_COMPLETE case={} seed={} "
                    "clients={}".format(args.case, seed, full_clients),
                    flush=True)
        overrides = [
            "seed", str(seed),
            "ggeur.lds_seed", str(spec.get("fixed_lds_seed", seed)),
            "ggeur.runtime_seed", "-1",
            "ggeur.classifier_init_seed",
            str(spec.get("fixed_classifier_init_seed", seed)),
            "ggeur.training_data_seed", str(seed),
            "federate.total_round_num", str(args.rounds),
            "outdir", run_dir.as_posix(),
            "expname", "{}_{}_seed_{}".format(
                args.case.lower().replace("-", ""), args.method, seed),
            "log_file", log_path.as_posix(),
            "ggeur.save_mlp_checkpoint", "True",
            "ggeur.mlp_checkpoint_dir", checkpoint_dir.as_posix(),
        ]
        if args.method == "platform":
            overrides.extend([
                "ggeur.task_adaptation_file", spec["task_file"],
                "ggeur.training_distribution_dir",
                (run_dir / "training_distributions").as_posix(),
            ])
        command = [
            sys.executable, "-m", "federatedscope.main",
            "--cfg", config.relative_to(REPO).as_posix(), *overrides,
        ]
        print(
            "ACCURACY_RUN_START case={} method={} seed={} rounds={}".format(
                args.case, args.method, seed, args.rounds), flush=True)
        if args.plan_only:
            print(subprocess.list2cmdline(command))
            continue
        started = time.time()
        stream_command(command, log_path)
        required = [
            checkpoint_dir / "mlp_best.pt",
            checkpoint_dir / "mlp_final.pt",
            checkpoint_dir / "pretrained_test_features.pt",
            checkpoint_dir / "checkpoint_manifest.json",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise RuntimeError("missing checkpoint artifacts: {}".format(missing))
        summary = {
            "case": args.case,
            "method": args.method,
            "seed": seed,
            "rounds": args.rounds,
            "elapsed_seconds": round(time.time() - started, 3),
            "training_log": log_path.relative_to(REPO).as_posix(),
            "best_mlp": required[0].relative_to(REPO).as_posix(),
            "final_mlp": required[1].relative_to(REPO).as_posix(),
            "pretrained_test_features": required[2].relative_to(REPO).as_posix(),
        }
        summary_path = run_dir / "run_summary.json"
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        summaries.append(summary)
        print(
            "ACCURACY_RUN_COMPLETE case={} method={} seed={} checkpoint={}".
            format(args.case, args.method, seed, summary["best_mlp"]),
            flush=True)
    if not args.plan_only:
        print("THREE_SEED_RUNS_COMPLETE={}".format(len(summaries) == 3))


if __name__ == "__main__":
    main()
