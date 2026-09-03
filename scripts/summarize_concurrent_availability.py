#!/usr/bin/env python3
"""Aggregate three-machine persistent-client availability evidence."""

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
    parser.add_argument("--servers", nargs="+", required=True)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-connections", type=int, default=10000)
    parser.add_argument("--target-qps", type=float, default=10000.0)
    parser.add_argument("--target-total-requests", type=int, default=0)
    args = parser.parse_args()

    servers = [read_json(item) for item in args.servers]
    probe = read_json(args.probe)
    expected = sum(
        int(item["expected_load_connections"]) for item in servers)
    established = sum(
        int(item["established_load_connections"]) for item in servers)
    observed = sum(
        int(item.get("observation", {}).get("load_connections", 0))
        for item in servers
    )
    active_at_observation = sum(
        int(item.get("observation", {}).get(
            "active_load_transfers", 0))
        for item in servers
    )
    completed = sum(
        int(item["completed_load_transfers"]) for item in servers)
    training_expected = sum(
        int(item.get("expected_training_clients", 0))
        for item in servers)
    training_completed = sum(
        int(item.get("completed_training_clients", 0))
        for item in servers)
    training_update_bytes = sum(
        int(item.get("training_update_bytes_total", 0))
        for item in servers)
    errors = sum(
        int(item.get("error_count", len(item.get("errors", []))))
        for item in servers)
    non_10x = sum(
        int(item.get("non_10x_peer_count", 0)) for item in servers)
    go_time = min(float(item["go_time_unix"]) for item in servers)
    last_start = max(
        float(item["last_transfer_start_unix"]) for item in servers)
    last_end = max(
        float(item["last_transfer_end_unix"]) for item in servers)
    dispatch_window = max(last_start - go_time, 1e-9)
    transfer_window = max(last_end - go_time, 1e-9)
    aggregate_dispatch_qps = established / dispatch_window
    request_duration = max(
        (float(item.get("request_test_duration_sec", 0.0))
         for item in servers),
        default=0.0,
    )
    expected_successful_requests = sum(
        int(item.get("expected_successful_requests", 0))
        for item in servers)
    successful_requests = sum(
        int(item.get("successful_requests", 0)) for item in servers)
    successful_requests_within_window = sum(
        int(item.get("successful_requests_within_window", 0))
        for item in servers)
    total_requests = sum(
        int(item.get("total_requests", 0)) for item in servers)
    failed_requests = sum(
        int(item.get("failed_requests", 0)) for item in servers)
    model_not_ready_responses = sum(
        int(item.get("model_not_ready_responses", 0))
        for item in servers)
    model_ready_responses = sum(
        int(item.get("model_ready_responses", 0))
        for item in servers)
    model_downloads_completed = sum(
        int(item.get("model_downloads_completed", 0))
        for item in servers)
    poll_interval = max(
        (float(item.get("poll_interval_sec", 0.0)) for item in servers),
        default=0.0,
    )
    target_total_requests = (
        args.target_total_requests
        if args.target_total_requests > 0
        else int(round(args.target_qps * request_duration))
    )
    average_request_qps = (
        successful_requests_within_window / request_duration
        if request_duration > 0 else 0.0
    )
    request_success_rate = (
        successful_requests_within_window / total_requests
        if total_requests > 0 else 0.0
    )
    sustained_request_target_met = (
        request_duration > 0
        and successful_requests_within_window >= target_total_requests
        and average_request_qps >= args.target_qps
        and request_success_rate >= 0.99
    )
    probe_bytes_ok = (
        int(probe.get("download_bytes", 0)) >= 133_380
        and int(probe.get("upload_bytes", 0)) >= 133_380
    )
    probe_ok = (
        bool(probe.get("success"))
        and bool(probe.get("connect_host_is_real_10x"))
        and probe_bytes_ok
    )
    result = {
        "benchmark": "headonly_10k_concurrent_availability",
        "server_count": len(servers),
        "subserver_count": sum(
            int(item["subservers"]) for item in servers),
        "target_connections": args.target_connections,
        "expected_load_connections": expected,
        "established_load_connections": established,
        "observed_load_connections_during_probe": observed,
        "active_load_transfers_at_observation":
            active_at_observation,
        "completed_load_transfers": completed,
        "training_clients_expected": training_expected,
        "training_clients_completed": training_completed,
        "training_update_bytes_total": training_update_bytes,
        "mixed_training_target_met":
            training_completed == training_expected,
        "strict_dispatch_start_window_sec": dispatch_window,
        "strict_full_transfer_window_sec": transfer_window,
        "target_qps": args.target_qps,
        "legacy_dispatch_start_qps": aggregate_dispatch_qps,
        "request_test_duration_sec": request_duration,
        "target_total_requests": target_total_requests,
        "expected_successful_requests": expected_successful_requests,
        "successful_requests": successful_requests,
        "successful_requests_within_window":
            successful_requests_within_window,
        "total_requests": total_requests,
        "failed_requests": failed_requests,
        "poll_interval_sec": poll_interval,
        "model_not_ready_responses": model_not_ready_responses,
        "model_ready_responses": model_ready_responses,
        "model_downloads_completed": model_downloads_completed,
        "average_request_qps": average_request_qps,
        "request_success_rate": request_success_rate,
        "sustained_request_target_met": sustained_request_target_met,
        "concurrent_connections_target_met": (
            established >= args.target_connections
            and observed >= args.target_connections
        ),
        "active_model_transactions_target_met":
            active_at_observation >= args.target_connections,
        "late_edge_probe_target_met": probe_ok,
        "probe": probe,
        "server_error_count": errors,
        "non_10x_peer_count": non_10x,
        "business_traffic_real_10x_only": non_10x == 0,
        "all_load_transfers_completed": completed == expected,
        "availability_target_met": (
            established >= args.target_connections
            and observed >= args.target_connections
            and active_at_observation >= args.target_connections
            and completed == expected
            and training_completed == training_expected
            and errors == 0
            and non_10x == 0
            and probe_ok
        ),
        "interpretation": {
            "concurrency": (
                "Number of persistent real TCP client connections present "
                "while the late edge probe is admitted."
            ),
            "active_model_transactions": (
                "Number of full serialized MLP upload/download transactions "
                "still active at the synchronized observation instant while "
                "the late edge probe is admitted."
            ),
            "qps": (
                "Successful model-parameter requests completed inside the "
                "fixed observation window divided by that window duration."
            ),
        },
        "server_results": [
            {
                key: value
                for key, value in item.items()
                if not key.endswith("_qps")
            }
            for item in servers
        ],
    }
    result["status"] = (
        "PASS"
        if (result["availability_target_met"]
            and result["sustained_request_target_met"])
        else "FAIL"
    )
    atomic_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
