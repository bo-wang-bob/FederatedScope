from types import SimpleNamespace

from scripts.distributed_scripts.ggeur_hierarchical_3machine.generate_matrix import (
    apply_experiment_overrides,
)


def _args(**overrides):
    values = {
        "train_learning_rate": None,
        "train_local_update_steps": None,
        "eval_frequency": None,
        "fedopt_server_learning_rate": None,
        "reuse_augmented_feature_cache": None,
        "save_augmented_feature_cache": None,
        "ggeur_headonly_eval_mode": None,
        "ggeur_num_generated_per_sample": None,
        "ggeur_num_generated_per_prototype": None,
        "ggeur_target_size_per_class": None,
        "ggeur_mlp_hidden_dim": None,
        "ggeur_mlp_dropout": None,
        "ggeur_max_cross_client_prototypes_per_class": None,
        "ggeur_diagonal_covariance": None,
        "ggeur_prototype_classifier_init": None,
        "ggeur_lda_classifier_init": None,
        "ggeur_domain_prototype_ensemble_per_class": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_client_lr_does_not_overwrite_fedopt_server_lr():
    cfg = {
        "train": {"optimizer": {"type": "Adam", "lr": 1e-4}},
        "fedopt": {
            "use": True,
            "optimizer": {"type": "Adam", "lr": 1e-2},
        },
    }

    apply_experiment_overrides(
        cfg, _args(train_learning_rate=2e-4), "fedopt")

    assert cfg["train"]["optimizer"]["lr"] == 2e-4
    assert cfg["fedopt"]["optimizer"]["lr"] == 1e-2


def test_fedopt_server_lr_and_generated_cache_are_explicit():
    cfg = {
        "train": {"optimizer": {"lr": 1e-4}},
        "fedopt": {"use": True, "optimizer": {"lr": 1e-2}},
    }

    apply_experiment_overrides(
        cfg,
        _args(
            fedopt_server_learning_rate=5e-3,
            reuse_augmented_feature_cache=False,
            save_augmented_feature_cache=False,
        ),
        "fedopt",
    )

    assert cfg["fedopt"]["optimizer"]["lr"] == 5e-3
    assert cfg["ggeur"]["reuse_augmented_feature_cache"] is False
    assert cfg["ggeur"]["save_augmented_feature_cache"] is False


def test_ggeur_generation_overrides_are_scoped_to_ggeur():
    args = _args(
        ggeur_num_generated_per_sample=50,
        ggeur_num_generated_per_prototype=50,
        ggeur_target_size_per_class=50,
    )
    ggeur_cfg = {}
    baseline_cfg = {}

    apply_experiment_overrides(ggeur_cfg, args, "ggeur")
    apply_experiment_overrides(baseline_cfg, args, "fedavg")

    assert ggeur_cfg["ggeur"]["num_generated_per_sample"] == 50
    assert ggeur_cfg["ggeur"]["num_generated_per_prototype"] == 50
    assert ggeur_cfg["ggeur"]["target_size_per_class"] == 50
    assert baseline_cfg["ggeur"]["num_generated_per_sample"] == 0
    assert baseline_cfg["ggeur"]["num_generated_per_prototype"] == 0
    assert baseline_cfg["ggeur"]["target_size_per_class"] == 0


def test_classifier_tuning_is_scoped_to_ggeur():
    args = _args(
        train_local_update_steps=5,
        ggeur_mlp_hidden_dim=256,
        ggeur_mlp_dropout=0.1,
        ggeur_max_cross_client_prototypes_per_class=12,
        ggeur_diagonal_covariance=True,
        ggeur_prototype_classifier_init=True,
        ggeur_lda_classifier_init=True,
        ggeur_domain_prototype_ensemble_per_class=4,
        ggeur_headonly_eval_mode="both",
    )
    ggeur_cfg = {}
    baseline_cfg = {}

    apply_experiment_overrides(ggeur_cfg, args, "ggeur")
    apply_experiment_overrides(baseline_cfg, args, "fedavg")

    assert ggeur_cfg["train"]["local_update_steps"] == 5
    assert ggeur_cfg["ggeur"]["mlp_hidden_dim"] == 256
    assert ggeur_cfg["ggeur"]["mlp_dropout"] == 0.1
    assert ggeur_cfg["ggeur"][
        "max_cross_client_prototypes_per_class"] == 12
    assert ggeur_cfg["ggeur"]["diagonal_covariance"] is True
    assert ggeur_cfg["ggeur"]["prototype_classifier_init"] is True
    assert ggeur_cfg["ggeur"]["lda_classifier_init"] is True
    assert ggeur_cfg["ggeur"][
        "domain_prototype_ensemble_per_class"] == 4
    assert ggeur_cfg["ggeur"]["headonly_eval_mode"] == "both"
    assert baseline_cfg["train"]["local_update_steps"] == 5
    assert "mlp_hidden_dim" not in baseline_cfg["ggeur"]
    assert "diagonal_covariance" not in baseline_cfg["ggeur"]
    assert "prototype_classifier_init" not in baseline_cfg["ggeur"]
    assert "lda_classifier_init" not in baseline_cfg["ggeur"]
    assert "domain_prototype_ensemble_per_class" not in baseline_cfg["ggeur"]


def test_eval_frequency_is_an_explicit_matrix_wide_override():
    args = _args(eval_frequency=99)
    ggeur_cfg = {"eval": {"freq": 1}}
    baseline_cfg = {"eval": {"freq": 1}}

    apply_experiment_overrides(ggeur_cfg, args, "ggeur")
    apply_experiment_overrides(baseline_cfg, args, "fedavg")

    assert ggeur_cfg["eval"]["freq"] == 99
    assert baseline_cfg["eval"]["freq"] == 99
