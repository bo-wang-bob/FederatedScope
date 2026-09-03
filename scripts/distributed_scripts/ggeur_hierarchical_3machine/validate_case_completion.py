#!/usr/bin/env python3
"""Validate the formal completion contract for one root training log."""

import argparse
import json
import os
import re
import tempfile
from pathlib import Path


FINISHED_RE = re.compile(r"Training finished after\s+(\d+)\s+rounds")
ACCURACY_RE = re.compile(r"Round\s+(\d+)\s+MLP Test Accuracy")
UPDATE_RE = re.compile(
    r"Round\s+(\d+)\s+aggregation complete.*valid_updates=(\d+)/(\d+)"
)


def atomic_json(path, value):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=output.name + ".", suffix=".tmp", dir=str(output.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log")
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-rounds", type=int, default=100)
    parser.add_argument("--expected-accuracy-rounds", type=int, default=99)
    parser.add_argument("--eval-frequency", type=int, default=1)
    parser.add_argument("--accuracy-records-per-eval", type=int, default=1)
    parser.add_argument("--terminal-client-eval-only", action="store_true")
    parser.add_argument("--client-evidence", default="")
    parser.add_argument("--expected-clients", type=int, default=0)
    parser.add_argument("--expected-updates", type=int, required=True)
    args = parser.parse_args()

    log_path = Path(args.log)
    text = log_path.read_text(encoding="utf-8", errors="replace")
    finished = [int(value) for value in FINISHED_RE.findall(text)]
    accuracy_rounds = [int(value) for value in ACCURACY_RE.findall(text)]
    updates = [
        (int(round_idx), int(valid), int(expected))
        for round_idx, valid, expected in UPDATE_RE.findall(text)
    ]
    if args.eval_frequency <= 0:
        expected_eval_rounds = []
    else:
        expected_eval_rounds = list(range(
            args.eval_frequency,
            args.expected_accuracy_rounds + 1,
            args.eval_frequency,
        ))
    expected_accuracy_records = [
        round_idx
        for round_idx in expected_eval_rounds
        for _ in range(args.accuracy_records_per_eval)
    ]
    if (args.terminal_client_eval_only and
            args.expected_accuracy_rounds in expected_eval_rounds):
        expected_accuracy_records.append(args.expected_accuracy_rounds)
    expected_update_rounds = list(
        range(1, args.expected_accuracy_rounds + 1))
    invalid_updates = [
        {
            "round": round_idx,
            "valid": valid,
            "expected": expected,
        }
        for round_idx, valid, expected in updates
        if valid != args.expected_updates or expected != args.expected_updates
    ]
    error_lines = [
        {"line": index, "text": line[:500]}
        for index, line in enumerate(text.splitlines(), start=1)
        if "ERROR" in line
    ]
    failures = []
    if finished != [args.expected_rounds]:
        failures.append(
            f"finished markers must be exactly [{args.expected_rounds}], got {finished}"
        )
    if accuracy_rounds != expected_accuracy_records:
        failures.append(
            "accuracy records must match configured evaluation rounds "
            f"{expected_accuracy_records}, got count={len(accuracy_rounds)} "
            f"first={accuracy_rounds[:3]} last={accuracy_rounds[-3:]}"
        )
    update_rounds = [round_idx for round_idx, _, _ in updates]
    if update_rounds != expected_update_rounds:
        failures.append(
            "aggregation update rounds must be exactly "
            f"1..{args.expected_accuracy_rounds}, got count={len(update_rounds)} "
            f"first={update_rounds[:3]} last={update_rounds[-3:]}"
        )
    if invalid_updates:
        failures.append(f"found {len(invalid_updates)} incomplete update rounds")
    if error_lines:
        failures.append(f"found {len(error_lines)} ERROR lines")
    client_evidence = None
    if args.client_evidence:
        evidence_path = Path(args.client_evidence)
        if not evidence_path.is_file():
            failures.append(
                f"client accuracy evidence is missing: {evidence_path}")
        else:
            try:
                client_evidence = json.loads(
                    evidence_path.read_text(encoding="utf-8-sig"))
                evidence_clients = client_evidence.get("clients", [])
                if (args.expected_clients > 0 and
                        (client_evidence.get("client_count") !=
                         args.expected_clients or
                         len(evidence_clients) != args.expected_clients)):
                    failures.append(
                        "client accuracy evidence must contain exactly "
                        f"{args.expected_clients} clients")
                if client_evidence.get("round") != args.expected_accuracy_rounds:
                    failures.append(
                        "client accuracy evidence round must be "
                        f"{args.expected_accuracy_rounds}")
            except Exception as error:
                failures.append(
                    f"client accuracy evidence is invalid: {error}")

    result = {
        "status": "pass" if not failures else "fail",
        "log": str(log_path),
        "expected_rounds": args.expected_rounds,
        "finished_markers": finished,
        "expected_accuracy_rounds": args.expected_accuracy_rounds,
        "eval_frequency": args.eval_frequency,
        "accuracy_records_per_eval": args.accuracy_records_per_eval,
        "terminal_client_eval_only": args.terminal_client_eval_only,
        "expected_accuracy_records": expected_accuracy_records,
        "accuracy_round_count": len(accuracy_rounds),
        "accuracy_rounds": accuracy_rounds,
        "expected_updates_per_round": args.expected_updates,
        "update_round_count": len(updates),
        "invalid_updates": invalid_updates,
        "error_count": len(error_lines),
        "error_lines": error_lines[:20],
        "client_evidence": args.client_evidence or None,
        "failures": failures,
    }
    atomic_json(args.output, result)
    if failures:
        raise SystemExit("; ".join(failures))


if __name__ == "__main__":
    main()
