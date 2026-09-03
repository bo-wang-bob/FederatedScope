#!/usr/bin/env python3
"""Create concise evidence for the fixed-window sustained-request test."""

import argparse
import json
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def atomic_json(path, item):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(item, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True)
    parser.add_argument("--load", required=True)
    parser.add_argument("--target-connections", type=int, default=10000)
    parser.add_argument("--target-duration-sec", type=float, default=600.0)
    parser.add_argument("--target-total-requests", type=int, default=6000000)
    parser.add_argument("--target-qps", type=float, default=10000.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    server = read_json(args.server)
    load = read_json(args.load)
    legacy_pre_auth = [
        item for item in server.get("errors", [])
        if item.get("role") == "unknown"
        and "IncompleteReadError" in str(item.get("error", ""))
    ]
    active_errors = [
        item for item in server.get("errors", [])
        if item not in legacy_pre_auth
    ]
    duration = float(server.get("request_test_duration_sec", 0.0))
    successful = int(
        server.get("successful_requests_within_window", 0))
    total = int(server.get("total_requests", 0))
    failed = int(server.get("failed_requests", max(total - successful, 0)))
    average_qps = successful / duration if duration > 0 else 0.0
    success_rate = successful / total if total > 0 else 0.0
    checks = {
        "persistent_connections_reached": int(
            server.get("established_load_connections", 0))
            >= args.target_connections,
        "fixed_window_duration_met": duration >= args.target_duration_sec,
        "successful_request_total_met":
            successful >= args.target_total_requests,
        "average_qps_met": average_qps >= args.target_qps,
        "request_success_rate_at_least_99_percent": success_rate >= 0.99,
        "all_load_clients_completed": int(
            load.get("completed_clients", 0))
            >= args.target_connections,
        "load_generator_reported_no_errors":
            not load.get("errors") and bool(load.get("success")),
        "active_protocol_errors_zero": len(active_errors) == 0,
    }
    result = {
        "benchmark": "fixed_window_sustained_model_parameter_requests",
        "scope": (
            "formal_real_10x_network"
            if int(server.get("non_10x_peer_count", 0)) == 0
            else "local_loopback_protocol_and_capacity_validation"
        ),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "persistent_connections": int(
            server.get("established_load_connections", 0)),
        "statistics_duration_sec": duration,
        "successful_requests_within_window": successful,
        "total_requests": total,
        "failed_requests": failed,
        "poll_interval_sec": float(server.get("poll_interval_sec", 0.0)),
        "average_qps": average_qps,
        "request_success_rate": success_rate,
        "request_clients_completed": int(
            server.get("request_clients_completed", 0)),
        "load_elapsed_sec": float(load.get("elapsed_sec", 0.0)),
        "active_protocol_error_count": len(active_errors),
        "ignored_pre_auth_transport_retries": (
            int(server.get("ignored_pre_auth_disconnects", 0))
            + len(legacy_pre_auth)
        ),
        "checks": checks,
        "source_files": {
            "server": str(Path(args.server).as_posix()),
            "load": str(Path(args.load).as_posix()),
        },
    }
    atomic_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
