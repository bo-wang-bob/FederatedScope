#!/usr/bin/env python3
"""Apply one task-adaptation profile to generated client YAML configs."""

import argparse
from pathlib import Path

import yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", required=True)
    parser.add_argument("--task-file", required=True)
    parser.add_argument("--distribution-dir", required=True)
    parser.add_argument(
        "--config-set",
        action="append",
        default=[],
        help="Client config directory name, e.g. clients_8g; repeatable",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    case_dir = Path(args.case_dir)
    config_root = case_dir / "configs"
    config_sets = args.config_set or [
        path.name
        for path in sorted(config_root.glob("clients_*"))
        if path.is_dir()
    ]
    if not config_sets:
        raise FileNotFoundError(
            f"No client config sets found under {config_root}")

    changed = []
    for config_set in config_sets:
        directory = config_root / config_set
        configs = sorted(directory.glob("client_*.yaml"))
        if not configs:
            raise FileNotFoundError(
                f"No client YAML files found under {directory}")
        for path in configs:
            content = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
            method_config = content.setdefault("ggeur", {})
            method_config["task_adaptation_file"] = args.task_file
            method_config["task_class_counts"] = []
            method_config["task_default_target_size"] = ""
            method_config["training_distribution_dir"] = args.distribution_dir
            method_config["reuse_augmented_feature_cache"] = False
            path.write_text(
                yaml.safe_dump(
                    content, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
            changed.append(path)

    print(f"configured_clients={len(changed)}")
    print(f"task_file={args.task_file}")
    print(f"distribution_dir={args.distribution_dir}")


if __name__ == "__main__":
    main()
