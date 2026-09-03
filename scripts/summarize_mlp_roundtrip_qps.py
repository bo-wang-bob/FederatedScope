#!/usr/bin/env python3
"""Aggregate synchronized MLP round-trip QPS results across subservers."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+")
    parser.add_argument("--output", default="")
    parser.add_argument("--target-qps", type=float, default=10000.0)
    args = parser.parse_args()

    rows = []
    for item in args.results:
        path = Path(item)
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        value["source"] = str(path)
        rows.append(value)
    expected = sum(int(row["expected_clients"]) for row in rows)
    received = sum(int(row["received_clients"]) for row in rows)
    global_start = min(float(row["go_time_unix"]) for row in rows)
    global_end = max(
        float(row["last_upload_complete_time_unix"]) for row in rows
    )
    # Never sum local QPS values. The aggregate denominator spans from the
    # earliest subserver GO signal to the latest completed upload.
    global_window = global_end - global_start
    aggregate_qps = received / global_window if global_window > 0 else 0.0
    payload_bytes = sum(
        int(row.get("total_wire_payload_bytes", 0)) for row in rows
    )
    result = {
        "metric": "aggregate_mlp_download_upload_roundtrip_qps",
        "subservers": len(rows),
        "expected_transactions": expected,
        "completed_transactions": received,
        "success_ratio": received / expected if expected else 0.0,
        "strict_global_window_sec": global_window,
        "aggregate_qps": aggregate_qps,
        "target_qps": args.target_qps,
        "target_met": aggregate_qps >= args.target_qps and received == expected,
        "total_wire_payload_bytes": payload_bytes,
        "wire_payload_bytes_per_sec": (
            payload_bytes / global_window if global_window > 0 else 0.0
        ),
        "subserver_results": rows,
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(text + "\n", encoding="utf-8", newline="\n")
        temporary.replace(output)
    print(text)


if __name__ == "__main__":
    main()
