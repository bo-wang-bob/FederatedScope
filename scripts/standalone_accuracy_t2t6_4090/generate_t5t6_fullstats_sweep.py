"""Generate full-statistics warm-up with sparse formal participation."""

import copy
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
source = json.loads((HERE / "final_experiment.json").read_text(encoding="utf-8"))
plan = {
    "profile_name": "T5-T6 full-statistics sparse-training sweep",
    "common": copy.deepcopy(source["common"]),
    "cases": {},
}


def add(base_name, suffix, formal_clients, platform_samples, lr,
        covariance_scale=0.1):
    base = copy.deepcopy(source["cases"][base_name])
    base["test_case"] = f"{base['test_case']}-{suffix.upper()}"
    base["sample_client_num"] = int(formal_clients)
    base["warmup_sample_client_num"] = int(base["client_num"])
    base["resource_overrides"].update({
        "train.optimizer.lr": float(lr),
        "ggeur.generation_covariance_scale": float(covariance_scale),
        "ggeur.domain_stratified_sampling": formal_clients >= 4,
        "ggeur.domain_sampling_group_num": 4,
        "ggeur.min_statistics_clients": int(formal_clients),
        "ggeur.min_augmentation_clients": int(formal_clients),
        "ggeur.min_train_updates": int(formal_clients),
    })
    base["method_overrides"]["platform"].update({
        "ggeur.platform_target_samples_per_client": int(platform_samples),
        "ggeur.platform_auto_target_samples_per_client": 0,
        "ggeur.platform_class_balanced_sampling": True,
        "ggeur.prototype_classifier_init": True,
    })
    plan["cases"][f"{base_name}_{suffix}"] = base


add("t6_mdsent_lstm", "fullstats_s3_t100_lr1e4", 3, 100, 1e-4, 0.1)

(HERE / "t5t6_fullstats_sweep.json").write_text(
    json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(HERE / "t5t6_fullstats_sweep.json")
