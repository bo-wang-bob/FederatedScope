"""Generate a compact robustness sweep for the T5/T6 text cases."""

import copy
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
source = json.loads((HERE / "final_experiment.json").read_text(encoding="utf-8"))

plan = {
    "profile_name": "T5-T6 robust text sweep",
    "common": copy.deepcopy(source["common"]),
    "cases": {},
}


def add_candidate(base_name, suffix, hidden, dropout, local_steps):
    base = copy.deepcopy(source["cases"][base_name])
    base["test_case"] = f"{base['test_case']}-{suffix.upper()}"
    base["local_update_steps"] = int(local_steps)
    base["resource_overrides"]["model.hidden"] = int(hidden)
    base["resource_overrides"]["model.dropout"] = float(dropout)
    base["method_overrides"]["platform"].update({
        "ggeur.platform_target_samples_per_client": 200,
        "ggeur.platform_auto_target_samples_per_client": 0,
    })
    plan["cases"][f"{base_name}_{suffix}"] = base


add_candidate("t5_mdsent_rnn", "p200_s2_h256_d01", 256, 0.1, 2)
add_candidate("t5_mdsent_rnn", "p200_s5_h64_d01", 64, 0.1, 5)
add_candidate("t5_mdsent_rnn", "p200_s10_h64_d00", 64, 0.0, 10)
add_candidate("t5_mdsent_rnn", "p200_s5_h32_d00", 32, 0.0, 5)

add_candidate("t6_mdsent_lstm", "p200_s2_h64_d01", 64, 0.1, 2)
add_candidate("t6_mdsent_lstm", "p200_s5_h64_d01", 64, 0.1, 5)
add_candidate("t6_mdsent_lstm", "p200_s10_h64_d00", 64, 0.0, 10)
add_candidate("t6_mdsent_lstm", "p200_s5_h32_d00", 32, 0.0, 5)

(HERE / "t5t6_robust_sweep.json").write_text(
    json.dumps(plan, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(HERE / "t5t6_robust_sweep.json")
