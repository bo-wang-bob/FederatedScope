#!/usr/bin/env python3
"""Parse per-round military-aircraft accuracy lines into a JSON summary."""

import argparse
import json
import re
from pathlib import Path


ANSI = re.compile(r"\x1b\[[0-9;]*m")
ROUND = re.compile(
    r"Round\s+(\d+)\s+MLP Test Accuracy\s+-\s+"
    r"aerial:\s*([0-9.]+),\s*natural:\s*([0-9.]+),\s*"
    r"recon:\s*([0-9.]+),\s*average:\s*([0-9.]+)",
    re.IGNORECASE,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    log = Path(args.log)
    rounds = []
    for raw_line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        match = ROUND.search(ANSI.sub("", raw_line))
        if not match:
            continue
        rounds.append({
            "round": int(match.group(1)),
            "aerial": float(match.group(2)),
            "natural": float(match.group(3)),
            "recon": float(match.group(4)),
            "average": float(match.group(5)),
        })
    if not rounds:
        raise RuntimeError(f"No per-round accuracy found in {log}")
    result = {
        "method": args.method,
        "log": str(log),
        "round_count": len(rounds),
        "final": rounds[-1],
        "best_average": max(item["average"] for item in rounds),
        "rounds": rounds,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"ACCURACY_SUMMARY method={args.method} rounds={len(rounds)} "
        f"final={rounds[-1]['average']:.4f} "
        f"best={result['best_average']:.4f} file={output}"
    )


if __name__ == "__main__":
    main()
