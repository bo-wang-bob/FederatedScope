"""Generate the validated final T5/T6 configuration."""

import copy
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
source = json.loads((HERE / "final_experiment.json").read_text(encoding="utf-8"))
plan = {
    "profile_name": "Validated T5-T6 final profile",
    "common": copy.deepcopy(source["common"]),
    "cases": {},
}


def resolved_case(base_name, platform_samples, learning_rate):
    base = copy.deepcopy(source["cases"][base_name])
    base["sample_client_num"] = 3
    base["warmup_sample_client_num"] = int(base["client_num"])
    base["resource_overrides"].update({
        "train.optimizer.lr": float(learning_rate),
        "ggeur.generation_covariance_scale": 0.1,
        "ggeur.domain_stratified_sampling": False,
        "ggeur.min_statistics_clients": 3,
        "ggeur.min_augmentation_clients": 3,
        "ggeur.min_train_updates": 3,
    })
    base["method_overrides"]["platform"].update({
        "ggeur.platform_target_samples_per_client": int(platform_samples),
        "ggeur.platform_auto_target_samples_per_client": 0,
        "ggeur.platform_class_balanced_sampling": True,
        "ggeur.prototype_classifier_init": True,
    })
    return base


plan["cases"]["t5_mdsent_rnn"] = resolved_case(
    "t5_mdsent_rnn", platform_samples=50, learning_rate=3e-5)
plan["cases"]["t6_mdsent_lstm"] = resolved_case(
    "t6_mdsent_lstm", platform_samples=100, learning_rate=1e-4)

plan["cases"]["t5_mdsent_rnn"]["recorded_result"] = {
    "source": "docs/test_logs/t5t6_resolved_final_20260822/results.json",
    "representative_seed": 43,
    "fedavg_accuracy": 0.4575,
    "fedprox_accuracy": 0.4575,
    "platform_accuracy": 0.6611,
    "platform_minus_fedavg_points": 20.36,
    "platform_minus_fedprox_points": 20.36,
    "fedavg_seconds": 5.1,
    "fedprox_seconds": 9.2,
    "platform_seconds": 5.5,
    "three_seed_average": {
        "fedavg_accuracy": 0.3806,
        "fedprox_accuracy": 0.3801,
        "platform_accuracy": 0.6603,
    },
}
plan["cases"]["t6_mdsent_lstm"]["recorded_result"] = {
    "source": "docs/test_logs/t5t6_resolved_final_20260822/results.json",
    "representative_seed": 43,
    "fedavg_accuracy": 0.4490,
    "fedprox_accuracy": 0.4509,
    "platform_accuracy": 0.6690,
    "platform_minus_fedavg_points": 22.00,
    "platform_minus_fedprox_points": 21.81,
    "fedavg_seconds": 5.4,
    "fedprox_seconds": 8.2,
    "platform_seconds": 6.8,
    "three_seed_average": {
        "fedavg_accuracy": 0.3599,
        "fedprox_accuracy": 0.3617,
        "platform_accuracy": 0.6669,
    },
}

(HERE / "t5t6_resolved_final.json").write_text(
    json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(HERE / "t5t6_resolved_final.json")
