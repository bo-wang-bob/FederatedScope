import json
import os
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

from federatedscope.cv.dataset.digit_three_domain import (
    DigitThreeDomain, load_digit_three_domain_manifest)
from federatedscope.contrib.worker.ggeur_client import GGEURClient
from federatedscope.contrib.worker.ggeur_server import GGEURServer
from scripts.prepare_digit_three_domain import allocate_dirichlet


def _records(domain, split, per_class=3):
    return [
        {
            "path": f"images/{domain}/{split}/{label}/{index}.png",
            "label": label,
        }
        for label in range(10)
        for index in range(per_class)
    ]


def test_global_manifest_has_three_shared_digit_domains(tmp_path):
    domains = ["emnist_digits", "usps", "svhn"]
    path = tmp_path / "dataset_manifest.json"
    path.write_text(json.dumps({
        "domains": domains,
        "classes": [str(value) for value in range(10)],
        "records": {
            domain: {
                "train": _records(domain, "train"),
                "test": _records(domain, "test"),
            }
            for domain in domains
        },
    }), encoding="utf-8")

    manifest, selected, classes = load_digit_three_domain_manifest(path)
    dataset = DigitThreeDomain(tmp_path,
                               domain="svhn",
                               split="test",
                               manifest_path=path)

    assert selected == domains
    assert classes == [str(value) for value in range(10)]
    assert manifest["domains"] == domains
    assert len(dataset) == 30
    assert set(dataset.targets) == set(range(10))


def test_dirichlet_partition_is_disjoint_and_complete():
    records = [
        {"path": f"{label}/{index}.png", "label": label}
        for label in range(10)
        for index in range(40)
    ]
    buckets, _ = allocate_dirichlet(records,
                                    client_ids=list(range(1, 7)),
                                    alpha=0.1,
                                    seed=42,
                                    min_client_samples=2)
    paths = [record["path"]
             for client_records in buckets.values()
             for record in client_records]

    assert len(paths) == len(records)
    assert len(set(paths)) == len(records)
    assert min(len(value) for value in buckets.values()) >= 2


def test_terminal_accuracy_evidence_contains_every_client_and_domain(tmp_path):
    server = GGEURServer.__new__(GGEURServer)
    server.client_eval_buffer = {
        99: {
            1: {
                "accuracy": 0.5,
                "loss": 1.0,
                "correct": 5,
                "total": 10,
                "domain": "emnist_digits",
            },
            2: {
                "accuracy": 0.8,
                "loss": 0.4,
                "correct": 8,
                "total": 10,
                "domain": "usps",
            },
            3: {
                "accuracy": 0.2,
                "loss": 1.8,
                "correct": 2,
                "total": 10,
                "domain": "svhn",
            },
        },
    }
    server._cfg = SimpleNamespace(outdir=str(tmp_path), expname="formal")
    server.test_accuracies_history = {}
    server.best_avg_accuracy = 0.0
    server.best_model_state = None
    server.global_mlp = None

    server._aggregate_client_eval_metrics(99)

    evidence_path = tmp_path / "client_model_accuracy_round_99.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["client_count"] == 3
    assert evidence["client_average_accuracy"] == 0.5
    assert set(evidence["domains"]) == {"emnist_digits", "usps", "svhn"}
    assert [item["client_id"] for item in evidence["clients"]] == [1, 2, 3]


def test_terminal_eval_reuses_portable_digit_feature_cache(tmp_path):
    dataset = DigitThreeDomain(
        tmp_path,
        domain="emnist_digits",
        split="test",
        records=[{
            "path": "images/emnist_digits/test/7/sample.png",
            "label": 7,
        }],
    )
    client = GGEURClient.__new__(GGEURClient)
    client._cfg = SimpleNamespace(data=SimpleNamespace(root=str(tmp_path)))
    client.ggeur_cfg = SimpleNamespace(
        extract_batch_size=64,
        use_fp16_extraction=False,
    )
    client.embedding_dim = 2
    client.device = "cpu"
    client.feature_extractor_type = "timm"
    client._get_feature_cache_path = lambda domain: "unused.npz"
    client._load_feature_cache = lambda path: {
        "images/emnist_digits/test/7/sample.png": np.asarray(
            [1.0, 2.0], dtype=np.float32),
    }
    client._load_feature_extractor = lambda: (_ for _ in ()).throw(
        AssertionError("a complete local test cache must avoid the extractor"))

    features, labels = client._extract_eval_features(
        SimpleNamespace(dataset=dataset))

    assert features.tolist() == [[1.0, 2.0]]
    assert labels.tolist() == [7]


def test_terminal_eval_rebuilds_linear_head_from_server_state():
    client = GGEURClient.__new__(GGEURClient)
    client.ID = 1
    client.device = "cpu"
    client.ggeur_cfg = SimpleNamespace(mlp_dropout=0.2)
    client.mlp_classifier = nn.Sequential(
        nn.Linear(4, 8), nn.ReLU(), nn.Dropout(0.2), nn.Linear(8, 3))
    state = nn.Linear(4, 3).state_dict()

    client._ensure_eval_classifier_matches(state)

    assert isinstance(client.mlp_classifier, nn.Linear)
    assert client.mlp_classifier.in_features == 4
    assert client.mlp_classifier.out_features == 3


def test_terminal_eval_loads_domainnet_heldout_cache(tmp_path):
    cache_dir = tmp_path / "domainnet_cnn"
    cache_dir.mkdir()
    np.savez(
        cache_dir / (
            "domainnet_clipart_test_cnn_convnext_base_"
            "terminal_client000001of000060.npz"),
        features=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        labels=np.asarray([1, 2], dtype=np.int64))
    client = GGEURClient.__new__(GGEURClient)
    client.ID = 1
    client.embedding_dim = 2
    client._cfg = SimpleNamespace(
        data=SimpleNamespace(type="domainnet"),
        federate=SimpleNamespace(client_num=60))
    client.ggeur_cfg = SimpleNamespace(
        feature_cache_dir=str(cache_dir),
        require_complete_feature_cache=True)

    features, labels = client._load_domainnet_eval_cache("clipart")

    assert torch.equal(features, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    assert torch.equal(labels, torch.tensor([1, 2]))
