#!/usr/bin/env python3
"""Combine the three fixed-configuration military-aircraft seed runs."""

import json
import statistics
from pathlib import Path


ROOT = Path("exp/military_aircraft_3domain/final_results")
PATHS = {
    42: ROOT / "accuracy",
    43: ROOT / "seed_repeats/seed_43/accuracy",
    44: ROOT / "seed_repeats/seed_44/accuracy",
}
METHODS = ("fedavg", "fedprox", "platform")


def load(seed, method):
    path = PATHS[seed] / f"{method}_accuracy_100round.json"
    return json.loads(path.read_text(encoding="utf-8"))


records = {}
for seed in PATHS:
    records[seed] = {}
    for method in METHODS:
        summary = load(seed, method)
        records[seed][method] = {
            "final": summary["final"],
            "best_average": summary["best_average"],
        }
    records[seed]["improvements"] = {
        "platform_vs_fedavg": (
            records[seed]["platform"]["final"]["average"]
            - records[seed]["fedavg"]["final"]["average"]
        ),
        "platform_vs_fedprox": (
            records[seed]["platform"]["final"]["average"]
            - records[seed]["fedprox"]["final"]["average"]
        ),
    }

aggregate = {}
for method in METHODS:
    values = [records[s][method]["final"]["average"] for s in PATHS]
    aggregate[method] = {
        "values": values,
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values),
    }
for comparison in ("platform_vs_fedavg", "platform_vs_fedprox"):
    values = [records[s]["improvements"][comparison] for s in PATHS]
    aggregate[comparison] = {
        "values": values,
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values),
        "all_seeds_above_20_percent": all(v > 0.20 for v in values),
    }

result = {
    "dataset": "MilitaryAircraft3D",
    "seeds": list(PATHS),
    "records": records,
    "aggregate": aggregate,
}
output = ROOT / "seed_repeats/three_seed_summary.json"
output.write_text(
    json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(result, ensure_ascii=False, indent=2))
print(f"OUTPUT={output}")
