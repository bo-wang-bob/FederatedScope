import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from federatedscope.contrib.worker.ggeur_client import GGEURClient  # noqa: E402


def _task_client(task_file="", inline=None, default=""):
    client = object.__new__(GGEURClient)
    client.ID = 1
    client._task_adaptation_cache = None
    client.ggeur_cfg = SimpleNamespace(
        task_adaptation_file=str(task_file),
        task_class_counts=list(inline or []),
        task_default_target_size=default,
        target_size_per_class=50,
        prompt_class_names=[],
    )
    client._cfg = SimpleNamespace(model=SimpleNamespace(num_classes=4))
    return client


def test_task_file_and_inline_config_are_merged(tmp_path):
    task_file = tmp_path / "task.yaml"
    task_file.write_text(
        "default_target_size: 9\n"
        "class_counts:\n"
        "  '0': 3\n"
        "  '1': 4\n",
        encoding="utf-8",
    )
    client = _task_client(task_file, inline=["1:7", "2=5"])

    spec = client._load_task_adaptation_spec()

    assert spec["enabled"]
    assert spec["default_target_size"] == 9
    assert spec["class_counts"] == {"0": 3, "1": 7, "2": 5}
    assert client._task_target_size(0, 50, spec) == 3
    assert client._task_target_size(1, 50, spec) == 7
    assert client._task_target_size(3, 50, spec) == 9


def test_empty_task_configuration_keeps_legacy_path_disabled():
    client = _task_client()

    spec = client._load_task_adaptation_spec()
    report, output_path = client._emit_training_distribution()

    assert not spec["enabled"]
    assert spec["class_counts"] == {}
    assert client._task_target_size(0, 50, spec) == 50
    assert report is None
    assert output_path is None


def test_default_task_profile_is_cache_and_generation_neutral(tmp_path):
    task_file = tmp_path / "task.json"
    task_file.write_text(json.dumps({
        "default_target_size": 50,
        "class_counts": {str(i): 50 for i in range(4)},
    }), encoding="utf-8")
    task_client = _task_client(task_file)
    plain_client = _task_client()

    task_spec = task_client._load_task_adaptation_spec()
    assert task_client._task_adaptation_is_generation_neutral(task_spec)

    base_metadata = {
        "num_classes": 4,
        "target_size_per_class": 50,
        "task_adaptation": plain_client._disabled_task_cache_metadata(50),
    }
    task_metadata = dict(base_metadata)
    task_metadata["task_adaptation"] = {
        "enabled": True,
        "default_target_size": 50,
        "class_counts": {str(i): 50 for i in range(4)},
        "signature": task_spec["signature"],
    }
    assert task_client._augmented_cache_fingerprint(task_metadata) == \
        plain_client._augmented_cache_fingerprint(base_metadata)


def test_task_adaptation_generates_exact_training_distribution(tmp_path):
    client = object.__new__(GGEURClient)
    client.ID = 3
    client.embedding_dim = 2
    client.feature_extractor_type = "clip"
    client._cov_factor_cache = {}
    client._task_adaptation_cache = None
    client.ggeur_cfg = SimpleNamespace(
        task_adaptation_file="",
        task_class_counts=["0:3", "1:5"],
        task_default_target_size="",
        training_distribution_dir=str(tmp_path / "distributions"),
        target_size_per_class=2,
        num_generated_per_sample=0,
        num_generated_per_prototype=0,
        use_cross_client_prototypes=False,
        reuse_augmented_feature_cache=False,
        save_augmented_feature_cache=False,
        prompt_class_names=[],
    )
    client._cfg = SimpleNamespace(
        data=SimpleNamespace(type="toy", root=str(tmp_path)),
        dataloader=SimpleNamespace(batch_size=2),
        model=SimpleNamespace(num_classes=2),
        outdir=str(tmp_path / "run"),
    )
    client.local_features = {
        0: np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
        1: np.asarray([[2.0, 2.0]], dtype=np.float32),
    }
    client.local_labels = {}
    client.global_cov_matrices = {
        0: np.eye(2, dtype=np.float32) * 0.01,
        1: np.eye(2, dtype=np.float32) * 0.01,
    }
    client.other_prototypes = {}
    client.augmented_features = None
    client.augmented_labels = None
    client.augmented_loader = None

    np.random.seed(7)
    client._perform_augmentation()

    unique, counts = np.unique(client.augmented_labels, return_counts=True)
    assert dict(zip(unique.tolist(), counts.tolist())) == {0: 3, 1: 5}
    report_path = tmp_path / "distributions" / "client_000003.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["method"] == "Platform"
    assert report["total_samples"] == 8
    assert report["class_counts"] == {"0": 3, "1": 5}
    assert report["task_adaptation"]["class_targets"] == {
        "0": 3,
        "1": 5,
    }
    assert report["training_class_counts"] == {"0": 3, "1": 5}


def test_distribution_keeps_generated_counts_after_training_sampling(tmp_path):
    client = _task_client()
    client.ID = 5
    client.ggeur_cfg.task_class_counts = ["0:50", "1:50", "2:50", "3:50"]
    client.ggeur_cfg.training_distribution_dir = str(tmp_path)
    client._task_adaptation_cache = None
    client._cfg.data = SimpleNamespace(type="toy")
    client.augmented_labels = np.asarray([0, 1, 2, 3], dtype=np.int64)
    client._generated_distribution_snapshot = {
        "total_samples": 200,
        "class_counts": {str(i): 50 for i in range(4)},
    }

    report, _ = client._emit_training_distribution(cache_hit=True)

    assert report["class_counts"] == {str(i): 50 for i in range(4)}
    assert report["total_samples"] == 200
    assert report["training_class_counts"] == {
        "0": 1, "1": 1, "2": 1, "3": 1,
    }
    assert report["training_total_samples"] == 4
