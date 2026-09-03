"""Generate class-balanced platform sampling candidates for T5/T6."""

import copy
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
source = json.loads((HERE / "final_experiment.json").read_text(encoding="utf-8"))
plan = {
    "profile_name": "T5-T6 balanced platform sampling sweep",
    "common": copy.deepcopy(source["common"]),
    "cases": {},
}


def add_candidate(base_name, suffix, clients, platform_samples, lr,
                  covariance_scale=1.0, prototype_init=False):
    base = copy.deepcopy(source["cases"][base_name])
    base["test_case"] = f"{base['test_case']}-{suffix.upper()}"
    base["sample_client_num"] = int(clients)
    base["resource_overrides"].update({
        "train.optimizer.lr": float(lr),
        "ggeur.generation_covariance_scale": float(covariance_scale),
        "ggeur.domain_stratified_sampling": True,
        "ggeur.domain_sampling_group_num": 4,
        "ggeur.min_statistics_clients": int(clients),
        "ggeur.min_augmentation_clients": int(clients),
        "ggeur.min_train_updates": int(clients),
    })
    base["method_overrides"]["platform"].update({
        "ggeur.platform_target_samples_per_client": int(platform_samples),
        "ggeur.platform_auto_target_samples_per_client": 0,
        "ggeur.platform_class_balanced_sampling": True,
        "ggeur.prototype_classifier_init": bool(prototype_init),
    })
    plan["cases"][f"{base_name}_{suffix}"] = base


add_candidate("t5_mdsent_rnn", "s20_t20_bal_lr1e4", 20, 20, 1e-4)
add_candidate("t5_mdsent_rnn", "s20_t20_bal_proto_lr5e5_cov01", 20, 20,
              5e-5, 0.1, True)
add_candidate("t5_mdsent_rnn", "s20_t20_bal_proto_lr2e5_cov01", 20, 20,
              2e-5, 0.1, True)
add_candidate("t5_mdsent_rnn", "s40_t20_bal_proto_lr5e5_cov01", 40, 20,
              5e-5, 0.1, True)

add_candidate("t6_mdsent_lstm", "s20_t100_bal_lr1e4", 20, 100, 1e-4)
add_candidate("t6_mdsent_lstm", "s20_t100_bal_proto_lr7e5_cov01", 20, 100,
              7e-5, 0.1, True)
add_candidate("t6_mdsent_lstm", "s20_t100_bal_proto_lr5e5_cov01", 20, 100,
              5e-5, 0.1, True)
add_candidate("t6_mdsent_lstm", "s40_t100_bal_lr1e4", 40, 100, 1e-4)

(HERE / "t5t6_balanced_sweep.json").write_text(
    json.dumps(plan, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(HERE / "t5t6_balanced_sweep.json")
