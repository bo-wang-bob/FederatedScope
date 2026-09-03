"""Generate focused dropout candidates after the balanced-data sweep."""

import copy
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
source = json.loads((HERE / "final_experiment.json").read_text(encoding="utf-8"))
plan = {
    "profile_name": "T5-T6 robust dropout sweep",
    "common": copy.deepcopy(source["common"]),
    "cases": {},
}


def add(base_name, suffix, platform_samples, lr, dropout,
        covariance_scale=1.0, prototype_init=False):
    base = copy.deepcopy(source["cases"][base_name])
    base["test_case"] = f"{base['test_case']}-{suffix.upper()}"
    clients = 20
    base["sample_client_num"] = clients
    base["resource_overrides"].update({
        "model.dropout": float(dropout),
        "train.optimizer.lr": float(lr),
        "ggeur.generation_covariance_scale": float(covariance_scale),
        "ggeur.domain_stratified_sampling": True,
        "ggeur.domain_sampling_group_num": 4,
        "ggeur.min_statistics_clients": clients,
        "ggeur.min_augmentation_clients": clients,
        "ggeur.min_train_updates": clients,
    })
    base["method_overrides"]["platform"].update({
        "ggeur.platform_target_samples_per_client": int(platform_samples),
        "ggeur.platform_auto_target_samples_per_client": 0,
        "ggeur.platform_class_balanced_sampling": True,
        "ggeur.prototype_classifier_init": bool(prototype_init),
    })
    plan["cases"][f"{base_name}_{suffix}"] = base


add("t5_mdsent_rnn", "drop05_lr1e4", 20, 1e-4, 0.5)
add("t5_mdsent_rnn", "drop08_lr1e4", 20, 1e-4, 0.8)
add("t5_mdsent_rnn", "drop08_proto_lr7e5", 20, 7e-5, 0.8, 0.1, True)
add("t5_mdsent_rnn", "drop09_proto_lr7e5", 20, 7e-5, 0.9, 0.1, True)

add("t6_mdsent_lstm", "drop09_lr1e4", 100, 1e-4, 0.9)
add("t6_mdsent_lstm", "drop09_proto_lr7e5", 100, 7e-5, 0.9, 0.1, True)
add("t6_mdsent_lstm", "drop07_proto_lr5e5", 100, 5e-5, 0.7, 0.1, True)

(HERE / "t5t6_dropout_sweep.json").write_text(
    json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(HERE / "t5t6_dropout_sweep.json")
