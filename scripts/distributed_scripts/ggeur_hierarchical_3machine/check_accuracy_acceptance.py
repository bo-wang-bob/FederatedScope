#!/usr/bin/env python3
"""Gate three-machine accuracy against baselines and standalone simulation."""

import argparse
import json
import sys
from pathlib import Path


METHODS = ("fedavg", "fedprox", "fedproto", "fedopt", "moon", "platform")
PATH_METHOD_ALIASES = {"ggeur": "platform", "ours": "platform"}


def _method_from_path(path):
    lowered = [part.lower() for part in path.parts]
    for path_method in (*METHODS, *PATH_METHOD_ALIASES):
        if any(part == path_method or part.endswith("_" + path_method)
               for part in lowered):
            return PATH_METHOD_ALIASES.get(path_method, path_method)
    return None


def load_results(root, methods=METHODS, case_prefix=""):
    """Load final/best averages from accuracy_summary.json files below root."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"result root does not exist: {root}")

    results = {}
    sources = {}
    for summary_path in sorted(root.rglob("accuracy_summary.json")):
        relative = summary_path.relative_to(root)
        if case_prefix and not any(
                part.lower().startswith(case_prefix.lower())
                for part in relative.parts):
            continue
        method = _method_from_path(relative)
        if method is None:
            continue
        if method not in methods:
            continue
        if method in results:
            raise ValueError(
                f"multiple summaries found for {method}: "
                f"{sources[method]} and {summary_path}")
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        results[method] = {
            "final": float(payload["last"]["metrics"]["average"]),
            "best": float(payload["best_average"]),
            "round_count": int(payload["round_count"]),
        }
        sources[method] = str(summary_path)
    return results, sources


def improvement_report(results, minimum_absolute_pp=20.0, methods=None):
    """Compare platform accuracy separately with FedAvg and FedProx.

    The project indicator is an absolute accuracy increase measured in
    percentage points.  Both comparisons must be strictly greater than the
    threshold; a relative percentage increase is not part of this gate.
    """
    required_methods = ("fedavg", "fedprox", "platform")
    if methods is not None:
        missing_required = [method for method in required_methods
                            if method not in methods]
        if missing_required:
            return {
                "status": "PENDING",
                "reason": "required methods were not requested: " +
                ", ".join(missing_required),
            }
    missing = [method for method in required_methods if method not in results]
    if missing:
        return {
            "status": "PENDING",
            "reason": "missing methods: " + ", ".join(missing),
        }

    fedavg = results["fedavg"]["final"]
    fedprox = results["fedprox"]["final"]
    platform = results["platform"]["final"]
    platform_minus_fedavg_pp = (platform - fedavg) * 100.0
    platform_minus_fedprox_pp = (platform - fedprox) * 100.0
    fedavg_pass = platform_minus_fedavg_pp > minimum_absolute_pp
    fedprox_pass = platform_minus_fedprox_pp > minimum_absolute_pp
    return {
        "status": "PASS" if fedavg_pass and fedprox_pass else "FAIL",
        "fedavg_final": fedavg,
        "fedprox_final": fedprox,
        "platform_final": platform,
        "platform_minus_fedavg_pp": platform_minus_fedavg_pp,
        "platform_minus_fedprox_pp": platform_minus_fedprox_pp,
        "minimum_absolute_improvement_pp": minimum_absolute_pp,
        "comparison_operator": ">",
        "platform_vs_fedavg_pass": fedavg_pass,
        "platform_vs_fedprox_pass": fedprox_pass,
    }


def parity_report(distributed, standalone, maximum_gap_pp, methods,
                  all_methods=False):
    methods = methods if all_methods else ("platform", )
    missing = [method for method in methods
               if method not in distributed or method not in standalone]
    if missing:
        return {
            "status": "PENDING",
            "reason": "missing comparable methods: " + ", ".join(missing),
            "maximum_gap_pp": maximum_gap_pp,
        }

    gaps = {}
    for method in methods:
        distributed_final = distributed[method]["final"]
        standalone_final = standalone[method]["final"]
        gap_pp = abs(distributed_final - standalone_final) * 100.0
        gaps[method] = {
            "distributed_final": distributed_final,
            "standalone_final": standalone_final,
            "gap_pp": gap_pp,
            "pass": gap_pp <= maximum_gap_pp,
        }
    return {
        "status": "PASS" if all(item["pass"] for item in gaps.values())
        else "FAIL",
        "maximum_gap_pp": maximum_gap_pp,
        "methods": gaps,
    }


def build_report(args):
    methods = tuple(
        item.strip().lower() for item in args.methods.split(",")
        if item.strip())
    methods = tuple(PATH_METHOD_ALIASES.get(item, item) for item in methods)
    if "platform" not in methods:
        raise ValueError("--methods must include platform")
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise ValueError("unknown methods: " + ", ".join(unknown))
    distributed, distributed_sources = load_results(
        args.distributed_root, methods, args.case_prefix)
    distributed_improvement = improvement_report(
        distributed, args.min_absolute_improvement_pp, methods)

    standalone = {}
    standalone_sources = {}
    if args.standalone_root:
        standalone, standalone_sources = load_results(
            args.standalone_root, methods, args.case_prefix)
        standalone_improvement = improvement_report(
            standalone, args.min_absolute_improvement_pp, methods)
        parity = parity_report(distributed, standalone,
                               args.max_parity_gap_pp,
                               methods,
                               args.parity_all_methods)
    else:
        standalone_improvement = {
            "status": "PENDING",
            "reason": "standalone result root was not provided",
        }
        parity = {
            "status": "PENDING",
            "reason": "standalone result root was not provided",
            "maximum_gap_pp": args.max_parity_gap_pp,
        }

    # Distributed improvement is always mandatory.  Parity is mandatory only
    # when the caller explicitly supplies a comparable standalone result root.
    checks = [distributed_improvement["status"]]
    if args.standalone_root:
        checks.append(parity["status"])
    if "FAIL" in checks:
        overall = "FAIL"
    elif "PENDING" in checks:
        overall = "PENDING"
    else:
        overall = "PASS"

    return {
        "overall_status": overall,
        "acceptance_policy": {
            "methods": list(methods),
            "case_prefix": args.case_prefix,
            "comparison":
            "platform final accuracy minus FedAvg and FedProx separately",
            "minimum_absolute_improvement_pp":
            args.min_absolute_improvement_pp,
            "comparison_operator": ">",
            "maximum_distributed_standalone_gap_pp":
            args.max_parity_gap_pp,
            "parity_scope": "all methods" if args.parity_all_methods
            else "platform",
        },
        "distributed": {
            "results": distributed,
            "sources": distributed_sources,
            "improvement": distributed_improvement,
        },
        "standalone": {
            "results": standalone,
            "sources": standalone_sources,
            "improvement": standalone_improvement,
        },
        "parity": parity,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--distributed-root", required=True)
    parser.add_argument("--standalone-root")
    parser.add_argument(
        "--methods", default=",".join(METHODS),
        help="Comma-separated methods required for this model/dataset group")
    parser.add_argument(
        "--case-prefix", default="",
        help="Only read result paths whose case directory starts with this prefix")
    parser.add_argument("--min-absolute-improvement-pp", type=float,
                        default=20.0)
    parser.add_argument("--max-parity-gap-pp", type=float, default=2.0)
    parser.add_argument("--parity-all-methods", action="store_true")
    parser.add_argument("--output")
    parser.add_argument("--allow-pending", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    report = build_report(args)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if report["overall_status"] == "PASS":
        return 0
    if report["overall_status"] == "PENDING" and args.allow_pending:
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
