"""Generate the prototype-initialized robustness sweep for T5/T6."""

import copy
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
source = json.loads((HERE / "final_experiment.json").read_text(encoding="utf-8"))
plan = {
    "profile_name": "T5-T6 prototype initialization sweep",
    "common": copy.deepcopy(source["common"]),
    "cases": {},
}


def add_candidate(base_name, suffix, lr, covariance_scale, dropout=None):
    base = copy.deepcopy(source["cases"][base_name])
    base["test_case"] = f"{base['test_case']}-{suffix.upper()}"
    base["resource_overrides"]["train.optimizer.lr"] = float(lr)
    base["resource_overrides"][
        "ggeur.generation_covariance_scale"
    ] = float(covariance_scale)
    if dropout is not None:
        base["resource_overrides"]["model.dropout"] = float(dropout)
    base["method_overrides"]["platform"][
        "ggeur.prototype_classifier_init"
    ] = True
    plan["cases"][f"{base_name}_{suffix}"] = base


for case_name in ("t5_mdsent_rnn", "t6_mdsent_lstm"):
    add_candidate(case_name, "proto_lr1e4_cov1", 1e-4, 1.0)
    add_candidate(case_name, "proto_lr5e5_cov1", 5e-5, 1.0)
    add_candidate(case_name, "proto_lr2e5_cov1", 2e-5, 1.0)
    add_candidate(case_name, "proto_lr5e5_cov01", 5e-5, 0.1)
    add_candidate(case_name, "proto_lr2e5_cov01", 2e-5, 0.1)
    add_candidate(case_name, "proto_lr2e5_cov01_d01", 2e-5, 0.1, 0.1)

(HERE / "t5t6_proto_sweep.json").write_text(
    json.dumps(plan, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(HERE / "t5t6_proto_sweep.json")
