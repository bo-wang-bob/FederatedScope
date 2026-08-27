#!/usr/bin/env python3
"""Fail-fast bandwidth feasibility check for full MLP round-trip QPS."""

import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-qps", type=float, default=10000.0)
    parser.add_argument("--input-dim", type=int, default=512)
    parser.add_argument("--num-classes", type=int, default=65)
    parser.add_argument("--hidden-dim", type=int, default=0)
    parser.add_argument("--dtype-bytes", type=int, default=4)
    parser.add_argument("--protocol-overhead", type=float, default=1.05)
    parser.add_argument("--client-link-mbps", type=float, required=True)
    parser.add_argument("--subserver-link-mbps", type=float, required=True)
    parser.add_argument("--minimum-headroom", type=float, default=1.25)
    args = parser.parse_args()

    if args.hidden_dim > 0:
        params = (
            args.input_dim * args.hidden_dim
            + args.hidden_dim
            + args.hidden_dim * args.num_classes
            + args.num_classes
        )
    else:
        params = args.input_dim * args.num_classes + args.num_classes
    payload_bytes = params * args.dtype_bytes
    one_direction_mbps = (
        args.target_qps * payload_bytes * 8 * args.protocol_overhead / 1_000_000
    )
    required_with_headroom = one_direction_mbps * args.minimum_headroom
    bottleneck = min(args.client_link_mbps, args.subserver_link_mbps)
    maximum_qps = (
        bottleneck * 1_000_000
        / (payload_bytes * 8 * args.protocol_overhead)
    )
    result = {
        "metric": "full_mlp_download_upload_roundtrip",
        "mlp_parameters": params,
        "payload_bytes_each_direction": payload_bytes,
        "target_qps": args.target_qps,
        "required_mbps_each_direction": one_direction_mbps,
        "required_mbps_each_direction_with_headroom": required_with_headroom,
        "client_link_mbps": args.client_link_mbps,
        "subserver_link_mbps": args.subserver_link_mbps,
        "bottleneck_link_mbps": bottleneck,
        "physical_upper_bound_qps": maximum_qps,
        "feasible_with_headroom": bottleneck >= required_with_headroom,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["feasible_with_headroom"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
