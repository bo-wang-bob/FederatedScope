#!/usr/bin/env python3
"""Preflight and summarize the privacy validation cases used by the test outline.

The privacy algorithms and their YAML files are maintained in a separately
deployed validation package.  This helper keeps the manual procedure honest:
it fails before training when that package is incomplete, and it turns the
training logs into deterministic PASS/FAIL JSON evidence afterwards.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable


FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
MARKED_ACCURACY_RE = re.compile(
    rf"(?:PRIVACY_ACCURACY|average(?:\s+local)?\s+test\s+accuracy)\s*[:=]\s*({FLOAT})",
    re.IGNORECASE,
)
TEST_ACC_RE = re.compile(rf"[\"']?test_acc[\"']?\s*[:=]\s*({FLOAT})", re.IGNORECASE)
NOISE_VARIANCE_RE = re.compile(
    rf"(?:PRIVACY_NOISE_VARIANCE|noise[_\s-]*variance)\s*[:=]\s*({FLOAT})",
    re.IGNORECASE,
)
EPSILON_RE = re.compile(
    rf"(?:PRIVACY_EPSILON|privacy[_\s-]*epsilon)\s*[:=]\s*({FLOAT})",
    re.IGNORECASE,
)


def _finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite: {value!r}")
    return result


def _normalize_accuracy(value: float) -> float:
    """Return accuracy on a 0..1 scale, accepting log values on 0..100."""
    value = _finite_float(value, "accuracy")
    if 1.0 < value <= 100.0:
        value /= 100.0
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"accuracy is outside 0..1 or 0..100: {value!r}")
    return value


def _walk_test_acc(value: Any) -> Iterable[float]:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() == "test_acc":
                yield _normalize_accuracy(_finite_float(item, "test_acc"))
            else:
                yield from _walk_test_acc(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk_test_acc(item)


def _line_accuracies(line: str) -> list[float]:
    marked = [
        _normalize_accuracy(float(match.group(1)))
        for match in MARKED_ACCURACY_RE.finditer(line)
    ]
    if marked:
        return marked

    parsed_values: list[float] = []
    if "test_acc" in line:
        for start in (line.find("{"), line.find("[")):
            if start < 0:
                continue
            try:
                payload = ast.literal_eval(line[start:])
            except (SyntaxError, ValueError):
                continue
            parsed_values.extend(_walk_test_acc(payload))
            if parsed_values:
                return parsed_values

    return [
        _normalize_accuracy(float(match.group(1)))
        for match in TEST_ACC_RE.finditer(line)
    ]


def parse_accuracy_rounds(log_path: Path) -> list[float]:
    rounds: list[float] = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        values = _line_accuracies(line)
        if values:
            rounds.append(sum(values) / len(values))
    if not rounds:
        raise ValueError(
            f"no accuracy was found in {log_path}; log PRIVACY_ACCURACY=<value> "
            "or a test_acc metric for each evaluation round"
        )
    return rounds


def parse_last_scalar(log_path: Path, pattern: re.Pattern[str], label: str) -> float:
    matches: list[float] = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        matches.extend(float(match.group(1)) for match in pattern.finditer(line))
    if not matches:
        raise ValueError(
            f"no {label} was found in {log_path}; log the explicit validation marker"
        )
    return _finite_float(matches[-1], label)


def optional_last_scalar(log_path: Path, pattern: re.Pattern[str]) -> float | None:
    matches: list[float] = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        matches.extend(float(match.group(1)) for match in pattern.finditer(line))
    return _finite_float(matches[-1], "privacy epsilon") if matches else None


def write_result(payload: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def command_preflight(args: argparse.Namespace) -> int:
    project_dir = Path(args.project_dir).expanduser().resolve()
    config_dir = Path(args.config_dir).expanduser().resolve()
    required = [project_dir / "federatedscope" / "main.py", project_dir / "scripts" / "privacy_validation.py"]
    required.extend(config_dir / name for name in args.configs)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required file(s):\n" + "\n".join(missing))
    print(json.dumps({
        "status": "PASS",
        "project_dir": str(project_dir),
        "config_dir": str(config_dir),
        "configs": [str(config_dir / name) for name in args.configs],
        "python": sys.executable,
    }, ensure_ascii=False, indent=2))
    return 0


def command_accuracy_loss(args: argparse.Namespace) -> int:
    baseline_log = Path(args.baseline_log).resolve()
    private_log = Path(args.private_log).resolve()
    baseline_rounds = parse_accuracy_rounds(baseline_log)
    private_rounds = parse_accuracy_rounds(private_log)
    window = args.last_rounds
    baseline_average = sum(baseline_rounds[-window:]) / min(window, len(baseline_rounds))
    private_average = sum(private_rounds[-window:]) / min(window, len(private_rounds))
    loss_points = max(0.0, (baseline_average - private_average) * 100.0)
    passed = loss_points <= args.max_loss_percentage_points
    payload = {
        "status": "PASS" if passed else "FAIL",
        "baseline_log": str(baseline_log),
        "private_log": str(private_log),
        "requested_last_rounds": window,
        "baseline_rounds_found": len(baseline_rounds),
        "private_rounds_found": len(private_rounds),
        "baseline_average_accuracy": baseline_average,
        "private_average_accuracy": private_average,
        "accuracy_loss_percentage_points": loss_points,
        "max_loss_percentage_points": args.max_loss_percentage_points,
    }
    write_result(payload, Path(args.output))
    return 0 if passed else 2


def command_noise_variance(args: argparse.Namespace) -> int:
    logs = {
        "platform": Path(args.platform_log).resolve(),
        "dpfl": Path(args.dpfl_log).resolve(),
        "ldp_fed": Path(args.ldp_fed_log).resolve(),
    }
    variances = {
        name: parse_last_scalar(path, NOISE_VARIANCE_RE, "noise variance")
        for name, path in logs.items()
    }
    if any(value < 0.0 for value in variances.values()):
        raise ValueError(f"noise variance must be non-negative: {variances}")
    comparison_average = (variances["dpfl"] + variances["ldp_fed"]) / 2.0
    if comparison_average <= 0.0:
        raise ValueError("the average DPFL/LDP-Fed noise variance must be positive")
    ratio = variances["platform"] / comparison_average

    epsilons = {
        name: optional_last_scalar(path, EPSILON_RE)
        for name, path in logs.items()
    }
    present_epsilons = [value for value in epsilons.values() if value is not None]
    epsilon_equal = len(present_epsilons) in (0, 3) and (
        not present_epsilons
        or max(present_epsilons) - min(present_epsilons) <= args.epsilon_tolerance
    )
    passed = ratio <= args.max_average_ratio and epsilon_equal
    payload = {
        "status": "PASS" if passed else "FAIL",
        "logs": {name: str(path) for name, path in logs.items()},
        "noise_variance": variances,
        "privacy_epsilon": epsilons,
        "privacy_epsilon_equal": epsilon_equal,
        "comparison_average_variance": comparison_average,
        "platform_to_comparison_average_ratio": ratio,
        "max_average_ratio": args.max_average_ratio,
    }
    write_result(payload, Path(args.output))
    return 0 if passed else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight", help="check project and privacy YAML files")
    preflight.add_argument("--project-dir", required=True)
    preflight.add_argument("--config-dir", required=True)
    preflight.add_argument("--configs", nargs="+", required=True)
    preflight.set_defaults(func=command_preflight)

    accuracy = subparsers.add_parser("accuracy-loss", help="check private versus baseline accuracy")
    accuracy.add_argument("--baseline-log", required=True)
    accuracy.add_argument("--private-log", required=True)
    accuracy.add_argument("--last-rounds", type=int, default=20)
    accuracy.add_argument("--max-loss-percentage-points", type=float, default=5.0)
    accuracy.add_argument("--output", required=True)
    accuracy.set_defaults(func=command_accuracy_loss)

    variance = subparsers.add_parser("noise-variance", help="compare platform, DPFL and LDP-Fed variance")
    variance.add_argument("--platform-log", required=True)
    variance.add_argument("--dpfl-log", required=True)
    variance.add_argument("--ldp-fed-log", required=True)
    variance.add_argument("--max-average-ratio", type=float, default=0.5)
    variance.add_argument("--epsilon-tolerance", type=float, default=1e-9)
    variance.add_argument("--output", required=True)
    variance.set_defaults(func=command_noise_variance)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if getattr(args, "last_rounds", 1) <= 0:
        parser.error("--last-rounds must be positive")
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
