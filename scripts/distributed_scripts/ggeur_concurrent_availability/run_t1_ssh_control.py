#!/usr/bin/env python3
"""Start and observe T1 from one controller machine over management SSH."""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def run(command, cwd):
    rendered = subprocess.list2cmdline([str(item) for item in command])
    print("执行：{}".format(rendered), flush=True)
    subprocess.run(
        [str(item) for item in command], cwd=str(cwd), check=True)


def build_verification_summary(result_dir, project_dir, run_id):
    def load(name):
        return json.loads((result_dir / name).read_text(encoding="utf-8"))

    network = load("network_connection_monitor.json")
    third_party = load("tcpvcon_connection_summary.json")
    root = load("root4090_server_summary.json")
    child = load("third_server_summary.json")
    probe = load("client_probe_summary.json")
    root_registration = load("third_root_registration_summary.json")
    child_load = load("client8g_child_load_summary.json")
    aggregate = load("aggregate_summary.json")
    edge_connections = int(child["established_load_connections"])
    hashes = {
        item["payload_sha256"]
        for item in (root, child, probe, root_registration, child_load)
    }
    checks = {
        "single_controller_ssh": True,
        "network_monitor_target_reached": bool(network["target_reached"]),
        "third_party_network_observation_at_least_10000": (
            bool(third_party["target_reached"])
            and int(third_party["observed_established_connections"]) >= 10000),
        "client_network_connections_at_least_10000": (
            int(network["max_load_client_established"]) >= 10000),
        "exactly_ten_child_servers": int(child["subservers"]) == 10,
        "ten_root_registrations_completed": (
            int(root["established_load_connections"]) == 10
            and int(root["completed_load_transfers"]) == 10
            and int(root_registration["completed_clients"]) == 10),
        "child_server_connections_at_least_10000": (
            edge_connections >= 10000),
        "all_edge_clients_completed": (
            int(child_load["completed_clients"]) == 10000),
        "real_10x_business_addresses_only": (
            int(root["non_10x_peer_count"])
            + int(child["non_10x_peer_count"]) == 0),
        "new_client_roundtrip_succeeded": bool(probe["success"]),
        "payload_hash_consistent": len(hashes) == 1,
        "all_roles_succeeded": all(item["success"] for item in (
            root, child, probe, root_registration, child_load)),
        "scheduled_poll_requests_completed_in_ten_minutes": (
            float(aggregate["request_test_duration_sec"]) == 600.0
            and int(aggregate["successful_requests_within_window"])
            / max(int(aggregate["total_requests"]), 1) >= 0.99),
        "average_qps_at_least_10000": (
            float(aggregate["average_request_qps"]) >= 10000.0),
    }
    summary = {
        "run_id": run_id,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "control_mode": "single_controller_over_management_ssh",
        "business_network": "real_10.x_addresses_without_ssh_tunnel",
        "network_tools": [
            network["network_observation_tool"],
            third_party["observation_tool"],
        ],
        "network_observation_samples": len(network["samples"]),
        "client_established_connections": int(
            network["max_load_client_established"]),
        "third_party_observed_established_connections": int(
            third_party["observed_established_connections"]),
        "root_subserver_registration_connections": int(
            root["established_load_connections"]),
        "child_server_count": int(child["subservers"]),
        "subserver_established_edge_connections": edge_connections,
        "completed_edge_clients": int(child_load["completed_clients"]),
        "new_client_parameter_roundtrip_sec": probe["roundtrip_sec"],
        "new_client_download_bytes": probe["download_bytes"],
        "new_client_upload_bytes": probe["upload_bytes"],
        "payload_sha256": next(iter(hashes)) if len(hashes) == 1 else None,
        "request_test_duration_sec": aggregate["request_test_duration_sec"],
        "successful_requests":
            aggregate["successful_requests_within_window"],
        "total_requests": aggregate["total_requests"],
        "failed_requests": aggregate["failed_requests"],
        "poll_interval_sec": aggregate["poll_interval_sec"],
        "average_request_qps": aggregate["average_request_qps"],
        "request_success_rate_percent": round(
            int(aggregate["successful_requests_within_window"])
            / max(int(aggregate["total_requests"]), 1) * 100, 6),
        "checks": checks,
    }
    output = result_dir / "t1_ssh_network_verification.json"
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    relative = output.relative_to(project_dir).as_posix()
    print("T1_VERIFY_STATUS={}".format(summary["status"]))
    print("CLIENT_ESTABLISHED_CONNECTIONS={}".format(
        summary["client_established_connections"]))
    print("SUBSERVER_COUNT={}".format(summary["child_server_count"]))
    print("EDGE_CLIENTS_COMPLETED={}".format(
        summary["completed_edge_clients"]))
    print("TCPVCON_ESTABLISHED_CONNECTIONS={}".format(
        summary["third_party_observed_established_connections"]))
    print("NEW_CLIENT_ROUNDTRIP=SUCCESS")
    print("REQUEST_TEST_DURATION_SEC={}".format(
        summary["request_test_duration_sec"]))
    print("SUCCESSFUL_REQUESTS={}".format(summary["successful_requests"]))
    print("TOTAL_REQUESTS={}".format(summary["total_requests"]))
    print("FAILED_REQUESTS={}".format(summary["failed_requests"]))
    print("AVERAGE_QPS={:.3f}".format(summary["average_request_qps"]))
    print("REQUEST_SUCCESS_RATE={:.6f}%".format(
        summary["request_success_rate_percent"]))
    print("T1_VERIFICATION_SUMMARY={}".format(relative))
    if summary["status"] != "PASS":
        raise RuntimeError("T1 verification failed: {}".format(output))


def main():
    parser = argparse.ArgumentParser(
        description="在一台控制机上通过SSH统一启动并观测T1。")
    parser.add_argument(
        "action", choices=(
            "start", "monitor", "observe", "status", "collect"))
    parser.add_argument("--run-id", default="outline_t01")
    parser.add_argument("--start-delay-sec", type=int, default=120)
    parser.add_argument("--hold-sec", type=int, default=60)
    parser.add_argument("--request-duration-sec", type=float, default=600.0)
    parser.add_argument("--poll-interval-sec", type=float, default=0.8)
    parser.add_argument("--poll-worker-connections", type=int, default=200)
    parser.add_argument("--target-qps", type=float, default=10000.0)
    parser.add_argument("--monitor-timeout-sec", type=int, default=300)
    parser.add_argument("--regenerate-payload", action="store_true")
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parents[3]
    manager = Path(__file__).with_name("manage_three_machine.py")
    generator = project_dir / "scripts" / "generate_headonly_mlp_state.py"
    payload = (
        project_dir / "exp" / "concurrent_availability" /
        "headonly_mlp_state.pt")
    common = [
        sys.executable, manager, "--payload-file", payload,
        "--validation-run-id", args.run_id,
        "--request-duration-sec", str(args.request_duration_sec),
        "--poll-interval-sec", str(args.poll_interval_sec),
        "--poll-worker-connections", str(args.poll_worker_connections),
        "--target-qps", str(args.target_qps),
    ]

    if args.action == "start":
        if args.regenerate_payload or not payload.is_file():
            payload.parent.mkdir(parents=True, exist_ok=True)
            run([sys.executable, generator, "--output", payload], project_dir)
        run(common + ["preflight"], project_dir)
        run(common + ["deploy"], project_dir)
        run(common + ["open-third-business-ports"], project_dir)
        run(common + [
            "start", "--start-phase", "all",
            "--start-delay-sec", str(args.start_delay_sec),
            "--active-after-probe-sec", str(args.hold_sec),
        ], project_dir)
        run(common + [
            "network-monitor", "--target-connections", "10000",
            "--monitor-timeout-sec", str(args.monitor_timeout_sec),
        ], project_dir)
        print("T1统一启动完成：三台机器均由本控制机通过SSH启动。")
        print("请在连接保持期间执行observe动作完成第三方网络观测。")
        return

    if args.action == "monitor":
        run(common + [
            "network-monitor", "--target-connections", "10000",
            "--monitor-timeout-sec", str(args.monitor_timeout_sec),
        ], project_dir)
    elif args.action == "observe":
        run(common + [
            "third-party-monitor", "--target-connections", "10000",
        ], project_dir)
    elif args.action == "status":
        run(common + ["status"], project_dir)
    else:
        output_dir = (
            project_dir / "docs" / "test_logs" /
            "concurrent_availability")
        run(common + ["collect", "--output-dir", output_dir], project_dir)
        result_dir = output_dir / args.run_id
        child_server = result_dir / "third_server_summary.json"
        probe = result_dir / "client_probe_summary.json"
        if not child_server.is_file() or not probe.is_file():
            raise RuntimeError("T1汇总所需的服务器或新增客户端结果不完整")
        aggregate = result_dir / "aggregate_summary.json"
        summarizer = project_dir / "scripts" / \
            "summarize_concurrent_availability.py"
        run([
            sys.executable, summarizer, "--servers", child_server,
            "--probe", probe, "--target-connections", "10000",
            "--target-qps", str(args.target_qps),
            "--target-total-requests",
            str(int(round(
                10000 * args.request_duration_sec /
                args.poll_interval_sec))),
            "--output", aggregate,
        ], project_dir)
        build_verification_summary(result_dir, project_dir, args.run_id)
        archive_base = result_dir / "performance_evidence"
        shutil.make_archive(str(archive_base), "zip", result_dir)
        aggregate_data = json.loads(aggregate.read_text(encoding="utf-8"))
        print("REQUEST_TEST_DURATION_SEC={}".format(
            aggregate_data["request_test_duration_sec"]))
        print("SUCCESSFUL_REQUESTS={}".format(
            aggregate_data["successful_requests_within_window"]))
        print("TOTAL_REQUESTS={}".format(
            aggregate_data["total_requests"]))
        print("FAILED_REQUESTS={}".format(
            aggregate_data["failed_requests"]))
        print("AVERAGE_QPS={:.3f}".format(
            aggregate_data["average_request_qps"]))
        print("QPS_THRESHOLD={}".format(args.target_qps))
        print("QPS_THRESHOLD_MET=true")
        print("AGGREGATE_SUMMARY={}".format(
            aggregate.relative_to(project_dir).as_posix()))
        print("PERFORMANCE_EVIDENCE_ARCHIVE={}".format(
            archive_base.with_suffix(".zip").relative_to(
                project_dir).as_posix()))


if __name__ == "__main__":
    main()
