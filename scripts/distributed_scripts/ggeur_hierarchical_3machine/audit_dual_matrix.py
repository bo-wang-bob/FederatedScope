#!/usr/bin/env python3
"""Audit a generated dual-host matrix before formal execution."""

import argparse
import json
import os
import tempfile
from pathlib import Path


ALLOWED_BUSINESS_HOSTS = {"10.129.222.189", "10.129.248.111"}
FORBIDDEN_TOKENS = ("10.112.81.135", "bjb", "seetacloud")


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
    parser.add_argument("run_dir")
    parser.add_argument("--output")
    parser.add_argument("--expected-rounds", type=int, required=True)
    parser.add_argument("--expected-cases", type=int, required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    manifest_path = run_dir / "matrix_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    cases = manifest.get("cases", [])
    failures = []
    case_results = []

    if manifest.get("case_count") != len(cases):
        failures.append("manifest case_count does not match cases")
    if len(cases) != args.expected_cases:
        failures.append(
            f"expected {args.expected_cases} cases, found {len(cases)}"
        )

    for item in cases:
        name = item.get("case", "<missing>")
        group = item.get("group", "")
        is_mdsent = group.startswith("mdsent_")
        expected_clients = 120 if is_mdsent else 60
        expected_subservers = 4 if is_mdsent else 2
        # DomainNet normally uses 30/30 to bound memory.  A 60/0 layout remains
        # valid when every generated client placement explicitly points to the
        # previously validated original-4 cache on the 8G host.
        split_domainnet = group.startswith("domainnet_")
        client_hosts = item.get("client_hosts", {})
        original_domainnet_layout = (
            split_domainnet and len(client_hosts.get("8g", [])) == 60 and
            not client_hosts.get("third", []))
        expected_8g_clients = (0 if is_mdsent else
                               (60 if original_domainnet_layout else
                                (30 if split_domainnet else 60)))
        expected_third_clients = (120 if is_mdsent else
                                  (0 if original_domainnet_layout else
                                   (30 if split_domainnet else 0)))
        case_failures = []

        if item.get("physical_topology") != "dual_host_three_logical_layers":
            case_failures.append("physical topology is not dual-host hierarchy")
        if item.get("root_host") != "10.129.222.189":
            case_failures.append("root is not the 8G host")
        if item.get("root_port") != 60050:
            case_failures.append("root port is not 60050")
        if item.get("client_num") != expected_clients:
            case_failures.append("unexpected client count")
        if item.get("subserver_num") != expected_subservers:
            case_failures.append("unexpected subserver count")

        if len(client_hosts.get("8g", [])) != expected_8g_clients:
            case_failures.append("unexpected 8G client placement")
        if len(client_hosts.get("third", [])) != expected_third_clients:
            case_failures.append("unexpected third-host client placement")
        for placement in client_hosts.get("8g", []) + client_hosts.get("third", []):
            if placement.get("host") not in ALLOWED_BUSINESS_HOSTS:
                case_failures.append(
                    f"forbidden client host {placement.get('host')}"
                )

        subserver_placement = item.get("subserver_hosts", {})
        subservers_8g = subserver_placement.get("8g", [])
        subservers_third = subserver_placement.get("third", [])
        subserver_hosts = subservers_8g + subservers_third
        if len(subserver_hosts) != expected_subservers:
            case_failures.append("unexpected subserver placement count")
        for placement in subserver_hosts:
            if placement.get("host") not in ALLOWED_BUSINESS_HOSTS:
                case_failures.append(
                    f"forbidden subserver host {placement.get('host')}"
                )
        if is_mdsent:
            if subservers_8g or len(subservers_third) != expected_subservers:
                case_failures.append("unexpected MDSent subserver placement")
            if any(
                placement.get("host") != "10.129.248.111"
                for placement in subserver_hosts
            ):
                case_failures.append("MDSent subservers are not all on third")
        else:
            if len(subservers_8g) != 1 or len(subservers_third) != 1:
                case_failures.append("DomainNet subservers are not split 1/1")
            domainnet_hosts = {p.get("host") for p in subserver_hosts}
            if domainnet_hosts != ALLOWED_BUSINESS_HOSTS:
                case_failures.append(
                    "DomainNet subservers are not split across both hosts"
                )

        case_dir = run_dir / name
        root_config = case_dir / "configs" / "root_server.yaml"
        if not root_config.is_file():
            case_failures.append("root config missing")
        else:
            root_text = root_config.read_text(encoding="utf-8-sig")
            if f"total_round_num: {args.expected_rounds}" not in root_text:
                case_failures.append("root round count mismatch")

        config_files = sorted((case_dir / "configs").rglob("*.yaml"))
        if not config_files:
            case_failures.append("no configs found")
        forbidden_hits = []
        for config in config_files:
            text = config.read_text(encoding="utf-8-sig").casefold()
            for token in FORBIDDEN_TOKENS:
                if token.casefold() in text:
                    forbidden_hits.append(
                        {"config": str(config), "token": token}
                    )
        if forbidden_hits:
            case_failures.append(
                f"found {len(forbidden_hits)} forbidden topology tokens"
            )

        failures.extend(f"{name}: {message}" for message in case_failures)
        case_results.append(
            {
                "case": name,
                "group": group,
                "method": item.get("method"),
                "client_num": item.get("client_num"),
                "subserver_num": item.get("subserver_num"),
                "config_count": len(config_files),
                "forbidden_hits": forbidden_hits[:20],
                "status": "pass" if not case_failures else "fail",
                "failures": case_failures,
            }
        )

    result = {
        "status": "pass" if not failures else "fail",
        "run_id": manifest.get("run_id"),
        "manifest": str(manifest_path),
        "expected_rounds": args.expected_rounds,
        "case_count": len(cases),
        "case_order": [item.get("case") for item in cases],
        "allowed_business_hosts": sorted(ALLOWED_BUSINESS_HOSTS),
        "forbidden_tokens": list(FORBIDDEN_TOKENS),
        "cases": case_results,
        "failures": failures,
    }
    output = args.output or str(run_dir / "matrix_audit.json")
    atomic_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit("; ".join(failures))


if __name__ == "__main__":
    main()
