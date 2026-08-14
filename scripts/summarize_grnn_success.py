#!/usr/bin/env python3
"""Summarize GRNN reconstruction success from tensor artifacts or logs."""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Iterable, List

import torch
import torch.nn.functional as F


ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
INDIVIDUAL_PSNR_RE = re.compile(
    r"Individual\s+SSIM\s*:\s*\[[^\]]*\]\s*,\s*"
    r"PSNR\s*:\s*(\[[^\]]*\])",
    re.IGNORECASE,
)
AVERAGE_PSNR_RE = re.compile(
    r"reconstruction quality.*?PSNR\s*:\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*dB",
    re.IGNORECASE,
)
FLOAT_RE = re.compile(
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def _parse_psnr_list(raw_value: str) -> List[float]:
    try:
        parsed = ast.literal_eval(raw_value)
    except (SyntaxError, ValueError):
        parsed = FLOAT_RE.findall(raw_value)

    if not isinstance(parsed, (list, tuple)):
        parsed = [parsed]

    values = []
    for item in parsed:
        try:
            value = float(item)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def extract_psnr_values(lines: Iterable[str]) -> List[float]:
    """Extract per-image PSNR values without double-counting averages."""
    values = []
    for raw_line in lines:
        line = ANSI_ESCAPE_RE.sub("", raw_line)
        if "[METRICS]" not in line or "PSNR" not in line:
            continue

        individual_match = INDIVIDUAL_PSNR_RE.search(line)
        if individual_match:
            values.extend(_parse_psnr_list(individual_match.group(1)))
            continue

        average_match = AVERAGE_PSNR_RE.search(line)
        if average_match:
            value = float(average_match.group(1))
            if math.isfinite(value):
                values.append(value)
    return values


def resolve_log_path(input_path: Path) -> Path:
    if input_path.is_file():
        return input_path
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    direct_log = input_path / "exp_print.log"
    if direct_log.is_file():
        return direct_log

    candidates = list(input_path.rglob("exp_print.log"))
    if not candidates:
        raise FileNotFoundError(
            f"No exp_print.log found below: {input_path}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _load_tensor_payload(path: Path) -> dict:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or not torch.is_tensor(payload.get("images")):
        raise ValueError(f"Invalid GRNN tensor artifact: {path}")
    return payload


def _artifact_psnr_values(log_path: Path):
    candidates = []
    for reconstruction_dir in log_path.parent.rglob("grnn_reconstructions"):
        reference_dir = reconstruction_dir.parent / "grnn_eval_references"
        if not reference_dir.is_dir():
            continue
        recon_files = list(reconstruction_dir.glob("state_*_client_*.pt"))
        if recon_files:
            newest = max(path.stat().st_mtime for path in recon_files)
            candidates.append((newest, reference_dir, reconstruction_dir))

    if not candidates:
        return [], None

    _, reference_dir, reconstruction_dir = max(
        candidates, key=lambda item: item[0])
    values = []
    matched_files = []
    missing_references = []

    for reconstruction_path in sorted(
            reconstruction_dir.glob("state_*_client_*.pt")):
        reference_path = reference_dir / reconstruction_path.name
        if not reference_path.is_file():
            missing_references.append(reconstruction_path.name)
            continue

        reference = _load_tensor_payload(reference_path)["images"].float()
        reconstruction = _load_tensor_payload(
            reconstruction_path)["images"].float()
        if reference.ndim == 3:
            reference = reference.unsqueeze(0)
        if reconstruction.ndim == 3:
            reconstruction = reconstruction.unsqueeze(0)
        if reference.ndim != 4 or reconstruction.ndim != 4:
            raise ValueError(
                f"Expected BCHW tensors in {reconstruction_path}")
        if reference.shape[1] != reconstruction.shape[1]:
            raise ValueError(
                f"Channel mismatch for {reconstruction_path.name}: "
                f"{reference.shape[1]} != {reconstruction.shape[1]}")
        if reference.shape[-2:] != reconstruction.shape[-2:]:
            reference = F.interpolate(
                reference, size=reconstruction.shape[-2:],
                mode="bilinear", align_corners=False)

        batch_size = min(reference.shape[0], reconstruction.shape[0])
        if batch_size == 0:
            continue
        reference = reference[:batch_size].clamp(0.0, 1.0)
        reconstruction = reconstruction[:batch_size].clamp(0.0, 1.0)
        mse = (reference - reconstruction).pow(2).flatten(1).mean(1)
        psnr = 10.0 * torch.log10(1.0 / mse.clamp_min(1.0e-12))
        values.extend(float(value) for value in psnr.tolist())
        matched_files.append(reconstruction_path.name)

    details = {
        "reference_dir": str(reference_dir.resolve()),
        "reconstruction_dir": str(reconstruction_dir.resolve()),
        "matched_artifact_files": matched_files,
        "missing_reference_files": missing_references,
    }
    return values, details


def build_summary(log_path: Path, threshold: float) -> dict:
    psnr_values, artifact_details = _artifact_psnr_values(log_path)
    source = "paired_tensor_artifacts"
    if not psnr_values:
        with log_path.open("r", encoding="utf-8", errors="replace") as stream:
            psnr_values = extract_psnr_values(stream)
        source = "log_metrics"

    if not psnr_values:
        raise ValueError(
            "No paired GRNN tensors or PSNR log metrics were found. Run GRNN "
            "once with the updated client/server code, then summarize that "
            "experiment directory.")

    successful = sum(value > threshold for value in psnr_values)
    total = len(psnr_values)
    summary = {
        "metric": "psnr_db",
        "success_rule": f"PSNR > {threshold}",
        "threshold": float(threshold),
        "metric_source": source,
        "source_log": str(log_path.resolve()),
        "total_reconstructions": total,
        "successful_reconstructions": successful,
        "failed_reconstructions": total - successful,
        "attack_success_rate": successful / total,
        "attack_success_rate_percent": successful / total * 100.0,
        "mean_psnr_db": statistics.fmean(psnr_values),
        "median_psnr_db": statistics.median(psnr_values),
        "min_psnr_db": min(psnr_values),
        "max_psnr_db": max(psnr_values),
        "psnr_values_db": psnr_values,
    }
    if artifact_details is not None:
        summary["artifacts"] = artifact_details
    return summary

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate GRNN attack success rate from paired tensors or PSNR entries "
            "in a FederatedScope exp_print.log."))
    parser.add_argument(
        "input",
        type=Path,
        help=(
            "Path to exp_print.log or an experiment directory. If multiple "
            "logs exist below a directory, the newest one is used."),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="A reconstruction succeeds when PSNR is greater than this value.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output JSON path. Defaults to grnn_attack_success_summary.json "
            "beside the selected log."),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        log_path = resolve_log_path(args.input)
        summary = build_summary(log_path, args.threshold)
    except (FileNotFoundError, ValueError) as error:
        print(f"[error] {error}", file=sys.stderr)
        return 2

    output_path = args.output or (
        log_path.parent / "grnn_attack_success_summary.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=True)
        stream.write("\n")

    print("GRNN attack success summary")
    print(f"  source log: {log_path}")
    print(f"  metric source: {summary['metric_source']}")
    print(f"  success rule: {summary['success_rule']}")
    print(f"  successful/total: {summary['successful_reconstructions']}/"
          f"{summary['total_reconstructions']}")
    print(f"  attack success rate: "
          f"{summary['attack_success_rate_percent']:.2f}%")
    print(f"  mean PSNR: {summary['mean_psnr_db']:.4f} dB")
    print(f"  result JSON: {output_path}")
    print(f"[RESULT] GRNN attack success rate: "
          f"{summary['attack_success_rate_percent']:.2f}% "
          f"({summary['successful_reconstructions']}/"
          f"{summary['total_reconstructions']}, "
          f"PSNR > {summary['threshold']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
