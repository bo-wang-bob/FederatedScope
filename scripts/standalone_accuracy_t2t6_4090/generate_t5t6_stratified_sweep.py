"""Generate the four-domain stratified-client sweep for T5/T6."""

import copy
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
source = json.loads((HERE / "final_experiment.json").read_text(encoding="utf-8"))
plan = {
    "profile_name": "T5-T6 four-domain stratified sweep",
    "common": copy.deepcopy(source["common"]),
    "cases": {},
}


def add_candidate(base_name, suffix, lr, covariance_scale,
                  prototype_init, dropout=None):
    base = copy.deepcopy(source["cases"][base_name])
    base["test_case"] = f"{base['test_case']}-{suffix.upper()}"
    base["sample_client_num"] = 4
    base["resource_overrides"].update({
        "train.optimizer.lr": float(lr),
        "ggeur.generation_covariance_scale": float(covariance_scale),
        "ggeur.domain_stratified_sampling": True,
        "ggeur.domain_sampling_group_num": 4,
        "ggeur.min_statistics_clients": 4,
        "ggeur.min_augmentation_clients": 4,
        "ggeur.min_train_updates": 4,
    })
    if dropout is not None:
        base["resource_overrides"]["model.dropout"] = float(dropout)
    base["method_overrides"]["platform"][
        "ggeur.prototype_classifier_init"
    ] = bool(prototype_init)
    plan["cases"][f"{base_name}_{suffix}"] = base


for case_name in ("t5_mdsent_rnn", "t6_mdsent_lstm"):
    add_candidate(case_name, "strat_lr1e4", 1e-4, 1.0, False)
    add_candidate(case_name, "strat_proto_lr1e4", 1e-4, 1.0, True)
    add_candidate(case_name, "strat_proto_lr5e5_cov01", 5e-5, 0.1, True)
    add_candidate(case_name, "strat_proto_lr2e5_cov01", 2e-5, 0.1, True)
    add_candidate(case_name, "strat_proto_lr5e5_cov01_d01", 5e-5, 0.1,
                  True, 0.1)

(HERE / "t5t6_stratified_sweep.json").write_text(
    json.dumps(plan, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(HERE / "t5t6_stratified_sweep.json")
