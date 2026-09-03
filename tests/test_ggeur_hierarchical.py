import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from federatedscope.core.communication import gRPCCommManager  # noqa: E402
from federatedscope.core.message import Message, b64serializer  # noqa: E402
from federatedscope.contrib.worker.ggeur_client import GGEURClient  # noqa: E402
from federatedscope.contrib.worker.ggeur_server import GGEURServer  # noqa: E402
from federatedscope.contrib.model.ggeur_text_rnn import (  # noqa: E402
    GGEURTextRNNClassifier,
)
from federatedscope.contrib.data.ggeur_data import (  # noqa: E402
    _resolve_manifest_paths,
    _resolve_officehome_manifest_root,
)


MODULE_PATH = (
    REPO_ROOT / "scripts" / "distributed_scripts" /
    "ggeur_hierarchical_3machine" / "hierarchical_subserver.py"
)


def test_lightweight_trainers_export_fedprox_wrapper():
    env = os.environ.copy()
    env["FEDERATEDSCOPE_GGEUR_LIGHTWEIGHT"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from federatedscope.core.trainers import "
            "wrap_fedprox_trainer; assert callable(wrap_fedprox_trainer)",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def load_subserver_module():
    spec = importlib.util.spec_from_file_location(
        "ggeur_hierarchical_subserver", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_weighted_nested_parameter_aggregation():
    module = load_subserver_module()
    first = {
        "weight": torch.tensor([1.0, 3.0]),
        "nested": {"bias": torch.tensor([2.0])},
    }
    second = {
        "weight": torch.tensor([5.0, 7.0]),
        "nested": {"bias": torch.tensor([10.0])},
    }

    result, total = module.aggregate_parameter_trees([
        (1, first),
        (3, second),
    ])

    assert total == 4
    assert torch.allclose(result["weight"], torch.tensor([4.0, 6.0]))
    assert torch.allclose(result["nested"]["bias"], torch.tensor([8.0]))


def test_empty_updates_still_have_defined_result():
    module = load_subserver_module()
    result, total = module.aggregate_parameter_trees([(0, {"weight": [1.0]})])
    assert result is None
    assert total == 0


def test_server_linear_head_can_initialize_from_global_prototypes():
    server = object.__new__(GGEURServer)
    server.ggeur_cfg = SimpleNamespace(prototype_classifier_init=True)
    server.device = torch.device("cpu")
    server.global_mlp = torch.nn.Linear(2, 3)
    server.global_prototypes = {
        0: np.asarray([1.0, 2.0], dtype=np.float32),
        2: np.asarray([-1.0, 0.5], dtype=np.float32),
    }

    server._initialize_classifier_from_global_prototypes(3)

    assert torch.allclose(
        server.global_mlp.weight,
        torch.tensor([
            [1.0 / np.sqrt(5.0), 2.0 / np.sqrt(5.0)],
            [0.0, 0.0],
            [-1.0 / np.sqrt(1.25), 0.5 / np.sqrt(1.25)],
        ], dtype=torch.float32))
    assert torch.allclose(
        server.global_mlp.bias, torch.tensor([0.0, 0.0, 0.0]))


def test_server_restores_cached_global_prototypes_for_classifier_init():
    server = object.__new__(GGEURServer)
    server.ggeur_cfg = SimpleNamespace(prototype_classifier_init=True)
    server.global_prototypes = {}
    server._get_embedding_dim = lambda: 2
    serialized = GGEURClient._serialize_array_payload({
        0: np.asarray([1.0, 2.0], dtype=np.float32),
        1: np.asarray([-3.0, 4.0], dtype=np.float32),
    })

    server._restore_cached_global_prototypes(
        {'status': 'ready', 'global_prototypes': serialized}, client_id=7)

    assert set(server.global_prototypes) == {0, 1}
    assert np.allclose(server.global_prototypes[0], [1.0, 2.0])
    assert np.allclose(server.global_prototypes[1], [-3.0, 4.0])


def test_server_linear_head_can_initialize_from_diagonal_lda():
    server = object.__new__(GGEURServer)
    server.ggeur_cfg = SimpleNamespace(
        lda_classifier_init=True, diagonal_covariance=True)
    server.device = torch.device("cpu")
    server.global_mlp = torch.nn.Linear(2, 2)
    server.global_prototypes = {
        0: np.asarray([1.0, 2.0], dtype=np.float32),
        1: np.asarray([-1.0, 4.0], dtype=np.float32),
    }
    server.global_cov_matrices = {
        0: np.asarray([2.0, 4.0], dtype=np.float32),
        1: np.asarray([4.0, 8.0], dtype=np.float32),
    }
    server.local_statistics_buffer = {
        1: {"counts": {0: 2, 1: 1}},
        2: {"counts": {0: 1, 1: 2}},
    }

    server._initialize_classifier_from_diagonal_lda(2)

    # The pooled variance is [3, 6].
    expected_weight = torch.tensor([
        [1.0 / 3.0, 2.0 / 6.0],
        [-1.0 / 3.0, 4.0 / 6.0],
    ])
    expected_bias = torch.tensor([
        -0.5 * (1.0 / 3.0 + 4.0 / 6.0),
        -0.5 * (1.0 / 3.0 + 16.0 / 6.0),
    ])
    assert torch.allclose(server.global_mlp.weight, expected_weight)
    assert torch.allclose(server.global_mlp.bias, expected_bias)


def test_server_domain_prototype_ensemble_uses_max_cosine_score():
    server = object.__new__(GGEURServer)
    server.ggeur_cfg = SimpleNamespace(
        domain_prototype_ensemble_per_class=2)
    server._cfg = SimpleNamespace(model=SimpleNamespace(num_classes=2))
    server.global_mlp = torch.nn.Linear(2, 2)
    server.domain_prototype_ensemble = {
        0: np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        1: np.asarray([[-1.0, 0.0], [0.0, -1.0]], dtype=np.float32),
    }

    outputs = server._predict_head_outputs(torch.tensor([
        [0.8, 0.2], [-0.1, -0.9],
    ]))

    assert torch.allclose(
        outputs, torch.tensor([[0.8, -0.2], [-0.1, 0.9]]))


def test_server_domain_personalized_head_selects_requested_domain():
    server = object.__new__(GGEURServer)
    server._cfg = SimpleNamespace(model=SimpleNamespace(num_classes=2))
    server.global_mlp = torch.nn.Linear(2, 2)
    server.domain_prototype_ensemble = {}
    server.domain_personalized_heads = {
        'clipart': {
            'weight': torch.tensor([[1.0, 0.0], [-1.0, 0.0]]),
            'bias': torch.zeros(2),
        },
        'sketch': {
            'weight': torch.tensor([[-1.0, 0.0], [1.0, 0.0]]),
            'bias': torch.zeros(2),
        },
    }

    features = torch.tensor([[2.0, 0.0]])
    clipart = server._predict_head_outputs(features, domain='clipart')
    sketch = server._predict_head_outputs(features, domain='sketch')

    assert int(torch.argmax(clipart, dim=1)) == 0
    assert int(torch.argmax(sketch, dim=1)) == 1


def test_hierarchical_fedproto_reduction_preserves_class_counts():
    module = load_subserver_module()
    first = {
        "mlp": {"weight": torch.tensor([1.0])},
        "fedproto_local_prototypes": {
            0: torch.tensor([1.0, 3.0]),
        },
        "fedproto_local_counts": {0: 2},
    }
    second = {
        "mlp": {"weight": torch.tensor([5.0])},
        "fedproto_local_prototypes": {
            0: torch.tensor([5.0, 7.0]),
            1: torch.tensor([2.0, 4.0]),
        },
        "fedproto_local_counts": {0: 6, 1: 3},
    }

    result, total = module.aggregate_client_updates([
        (10, first),
        (30, second),
    ])

    assert total == 40
    assert torch.allclose(result["mlp"]["weight"], torch.tensor([4.0]))
    assert result["fedproto_local_counts"] == {0: 8, 1: 3}
    assert torch.allclose(
        result["fedproto_local_prototypes"][0],
        torch.tensor([4.0, 6.0]))
    assert torch.allclose(
        result["fedproto_local_prototypes"][1],
        torch.tensor([2.0, 4.0]))


def test_root_fedproto_reduction_matches_flat_client_reduction():
    server = object.__new__(GGEURServer)
    server.ggeur_cfg = SimpleNamespace(
        fedproto_keep_last_global_prototypes=True)
    server.fedproto_global_prototypes = {}
    subserver_updates = [
        (40, {
            "fedproto_local_prototypes": {
                0: torch.tensor([4.0, 6.0]),
                1: torch.tensor([2.0, 4.0]),
            },
            "fedproto_local_counts": {0: 8, 1: 3},
        }),
        (20, {
            "fedproto_local_prototypes": {
                0: torch.tensor([8.0, 10.0]),
            },
            "fedproto_local_counts": {0: 2},
        }),
    ]

    server._aggregate_fedproto_prototypes(subserver_updates)

    assert torch.allclose(
        server.fedproto_global_prototypes[0],
        torch.tensor([4.8, 6.8]))
    assert torch.allclose(
        server.fedproto_global_prototypes[1],
        torch.tensor([2.0, 4.0]))


def test_client_fedproto_uses_head_hidden_space_not_bert_input_space():
    client = object.__new__(GGEURClient)
    client.use_fedproto = True
    client.device = torch.device("cpu")
    client.mlp_classifier = GGEURTextRNNClassifier(
        input_dim=6,
        hidden_dim=3,
        num_classes=2,
        rnn_type="lstm",
    )
    features = torch.tensor([
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    ])
    labels = torch.tensor([0, 0, 1])
    client.augmented_loader = DataLoader(
        TensorDataset(features, labels),
        batch_size=2,
        shuffle=False,
    )

    prototypes, counts = client._compute_fedproto_local_prototypes()

    assert counts == {0: 2, 1: 1}
    assert set(prototypes) == {0, 1}
    assert prototypes[0].shape == (3,)
    assert prototypes[1].shape == (3,)


def test_client_fedproto_linear_head_uses_input_prototypes_and_weight_loss():
    client = object.__new__(GGEURClient)
    client.use_fedproto = True
    client.device = torch.device("cpu")
    client.fedproto_normalize = False
    client.fedproto_distance_metric = "mse"
    client.mlp_classifier = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        client.mlp_classifier.weight.zero_()

    features = torch.tensor([
        [1.0, 0.0],
        [3.0, 0.0],
        [0.0, 2.0],
    ])
    labels = torch.tensor([0, 0, 1])
    client.augmented_loader = DataLoader(
        TensorDataset(features, labels),
        batch_size=2,
        shuffle=False,
    )

    prototypes, counts = client._compute_fedproto_local_prototypes()

    assert counts == {0: 2, 1: 1}
    assert torch.allclose(prototypes[0], torch.tensor([2.0, 0.0]))
    assert torch.allclose(prototypes[1], torch.tensor([0.0, 2.0]))

    client.fedproto_global_prototypes = prototypes
    _, representations = client._forward_head_with_representation(features)
    loss = client._compute_fedproto_loss(labels, representations)
    assert loss.requires_grad
    loss.backward()
    assert client.mlp_classifier.weight.grad is not None
    assert torch.count_nonzero(client.mlp_classifier.weight.grad) > 0


def test_mdsent_matrix_keeps_reference_branch_learning_rates():
    config_root = (
        REPO_ROOT / "scripts" / "example_configs" /
        "ggeur_final_5models")
    for method in ("fedavg", "fedprox", "fedproto", "fedopt", "ggeur"):
        rnn = yaml.safe_load(
            (config_root / "mdsent_rnn" / f"{method}.yaml").read_text(
                encoding="utf-8"))
        lstm = yaml.safe_load(
            (config_root / "mdsent_lstm" / f"{method}.yaml").read_text(
                encoding="utf-8"))
        assert float(rnn["train"]["optimizer"]["lr"]) == 1e-3
        assert float(lstm["train"]["optimizer"]["lr"]) == 0.1


def test_grpc_send_deduplicates_shared_proxy_endpoint():
    manager = object.__new__(gRPCCommManager)
    manager.neighbors = {
        1: "10.0.0.2:61000",
        2: "10.0.0.2:61000",
        3: "10.0.0.3:61001",
    }
    sent = []
    manager._send_request = lambda address, request: sent.append(address)

    manager.send(Message(msg_type="model_para", receiver=[1, 2, 3]))

    assert sorted(sent) == ["10.0.0.2:61000", "10.0.0.3:61001"]


def test_grpc_fanout_builds_large_request_only_once():
    class CountingMessage(Message):
        def __init__(self):
            super().__init__(msg_type="model_para", receiver=[1, 2, 3])
            self.transform_calls = 0

        def transform(self, to_list=False):
            self.transform_calls += 1
            return object()

    manager = object.__new__(gRPCCommManager)
    manager.neighbors = {
        1: "10.0.0.1:20001",
        2: "10.0.0.1:20002",
        3: "10.0.0.1:20003",
    }
    sent = []
    manager._send_request = lambda address, request: sent.append(
        (address, request))
    message = CountingMessage()

    manager.send(message)

    assert message.transform_calls == 1
    assert sorted(address for address, _ in sent) == [
        "10.0.0.1:20001", "10.0.0.1:20002", "10.0.0.1:20003"]
    assert len({id(request) for _, request in sent}) == 1


def test_server_groups_identical_hierarchical_client_broadcasts():
    class RecordingCommManager:
        def __init__(self):
            self.messages = []

        def send(self, message):
            self.messages.append(message)

    server = object.__new__(GGEURServer)
    server.ID = 0
    server.hierarchical_training = True
    server.comm_manager = RecordingCommManager()

    server._broadcast_identical_to_clients(
        "model_para", [1, 2, 3], 7, {"weight": "shared"})

    assert len(server.comm_manager.messages) == 1
    assert server.comm_manager.messages[0].receiver == [1, 2, 3]

    server.hierarchical_training = False
    server.comm_manager.messages.clear()
    server._broadcast_identical_to_clients(
        "model_para", [1, 2, 3], 7, {"weight": "shared"})
    assert [message.receiver for message in server.comm_manager.messages] == [
        [1], [2], [3]
    ]


def test_client_decodes_nested_distributed_model_payload():
    tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    payload = {
        "phase": "classifier",
        "classifier": {
            "weight": b64serializer(tensor),
        },
    }

    decoded = GGEURClient._decode_parameter_tree(payload)

    assert decoded["phase"] == "classifier"
    assert torch.equal(decoded["classifier"]["weight"], tensor)


def test_client_decodes_numeric_list_model_payload_as_tensor():
    payload = {
        "weight": [[1.0, 2.0], [3.0, 4.0]],
        "bias": [5.0, 6.0],
    }

    decoded = GGEURClient._decode_parameter_tree(payload)

    assert torch.equal(
        decoded["weight"],
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    )
    assert torch.equal(decoded["bias"], torch.tensor([5.0, 6.0]))


def test_client_preserves_heterogeneous_control_lists():
    payload = ["classifier", {"weight": [[1.0, 2.0]]}]

    decoded = GGEURClient._decode_parameter_tree(payload)

    assert decoded[0] == "classifier"
    assert torch.equal(decoded[1]["weight"], torch.tensor([[1.0, 2.0]]))


def test_client_decodes_compact_ndarray_inference_payload():
    array = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    payload = GGEURServer._serialize_ndarray_payload(array)

    decoded = GGEURClient._decode_parameter_tree({"prototype": payload})

    assert isinstance(decoded["prototype"], np.ndarray)
    assert decoded["prototype"].dtype == np.float16
    np.testing.assert_allclose(decoded["prototype"], array, rtol=0, atol=0)


def test_model_state_survives_message_tensor_serialization():
    source = GGEURTextRNNClassifier(
        input_dim=4,
        hidden_dim=3,
        num_classes=2,
        rnn_type="rnn",
    )
    with torch.no_grad():
        for index, parameter in enumerate(source.parameters(), start=1):
            parameter.fill_(float(index))
    outbound = Message(
        msg_type="model_para",
        sender=0,
        receiver=[1],
        state=1,
        content={"mlp": source.state_dict()},
    )
    wire_content = outbound.transform_to_list(outbound.content)
    decoded = GGEURClient._decode_parameter_tree(wire_content)

    target = GGEURTextRNNClassifier(
        input_dim=4,
        hidden_dim=3,
        num_classes=2,
        rnn_type="rnn",
    )
    target.load_state_dict(decoded["mlp"])

    for key, expected in source.state_dict().items():
        assert torch.equal(target.state_dict()[key], expected)


def test_two_level_model_aggregation_decodes_wire_tensors():
    module = load_subserver_module()

    def serialize_parameters(parameters):
        message = Message(
            msg_type="model_para",
            content=parameters,
        )
        return message.transform_to_list(message.content)

    first_subserver, first_total = module.aggregate_client_updates([
        (1, serialize_parameters({
            "mlp": {"weight": torch.tensor([1.0])},
        })),
        (3, serialize_parameters({
            "mlp": {"weight": torch.tensor([5.0])},
        })),
    ])
    second_subserver, second_total = module.aggregate_client_updates([
        (2, serialize_parameters({
            "mlp": {"weight": torch.tensor([10.0])},
        })),
    ])
    first_wire = serialize_parameters(first_subserver)
    second_wire = serialize_parameters(second_subserver)

    server = object.__new__(GGEURServer)
    result = server._aggregate_model_params([
        (first_total, first_wire["mlp"]),
        (second_total, second_wire["mlp"]),
    ], first_total + second_total)

    assert first_total == 4
    assert second_total == 2
    assert torch.allclose(result["weight"], torch.tensor([6.0]))


def test_fedproto_root_decodes_subserver_grpc_tensor_payload():
    server = object.__new__(GGEURServer)
    server.ggeur_cfg = SimpleNamespace(
        fedproto_keep_last_global_prototypes=True)
    server.fedproto_global_prototypes = {}
    serialized = b64serializer(torch.tensor([3.0, 5.0]))

    server._aggregate_fedproto_prototypes([
        (2, {
            "fedproto_local_prototypes": {0: serialized},
            "fedproto_local_counts": {0: 2},
        }),
    ])

    assert torch.allclose(
        server.fedproto_global_prototypes[0],
        torch.tensor([3.0, 5.0]))


def test_feature_cache_keys_are_portable_across_hosts():
    client = object.__new__(GGEURClient)
    client._cfg = SimpleNamespace(data=SimpleNamespace(
        root="D:/Projects/FederatedScope/OfficeHomeDataset_10072016"))

    windows_path = (
        r"D:\Projects\FederatedScope\OfficeHomeDataset_10072016"
        r"\Art\Alarm_Clock\00001.jpg")
    linux_path = (
        "/root/autodl-tmp/datasets/OfficeHomeDataset_10072016/"
        "Art/Alarm_Clock/00001.jpg")

    assert client._feature_cache_key(windows_path) == \
        client._feature_cache_key(linux_path)


def test_server_loads_complete_feature_cache_without_extractor(tmp_path):
    server = object.__new__(GGEURServer)
    server.feature_extractor_type = "bert"
    server.inferred_embedding_dim = 3
    server.local_statistics_buffer = {}
    server._cfg = SimpleNamespace(
        seed=42,
        data=SimpleNamespace(
            type="mdsent",
            root="D:/Projects/FederatedScope/data/sentiment"),
    )
    server.ggeur_cfg = SimpleNamespace(
        feature_cache_dir=str(tmp_path),
        embedding_dim=3,
        bert_model_path=(
            "D:/Projects/FederatedScope/pretrained_models/"
            "nlptown_bert_base_multilingual_uncased_senti"),
        bert_use_pretrained_weights=True,
        bert_pooling="cls",
        bert_max_length=128,
    )
    cache_path = Path(server._get_full_feature_cache_path("books"))
    np.savez(
        cache_path,
        paths=np.asarray([
            "books:1",
            "D:/Projects/FederatedScope/data/sentiment/books/2",
        ]),
        features=np.asarray([[1., 2., 3.], [4., 5., 6.]],
                            dtype=np.float32),
    )

    cache, loaded_path = server._load_full_feature_cache("books")

    assert loaded_path == str(cache_path)
    assert set(cache) == {"books:1", "books/2"}
    assert np.array_equal(cache["books:1"], np.asarray([1., 2., 3.]))


def test_server_domainnet_metadata_uses_manifest_without_raw_images(tmp_path):
    manifest = tmp_path / "domainnet_manifest.json"
    manifest.write_text(json.dumps({
        "domains": ["clipart", "real"],
        "classes": ["bird", "car"],
        "records": {
            "clipart": [{"path": "clipart/bird/1.png", "label": 0}],
            "real": [{"path": "real/car/1.png", "label": 1}],
        },
    }), encoding="utf-8")
    server = object.__new__(GGEURServer)
    server._cfg = SimpleNamespace(
        data=SimpleNamespace(root="Z:/missing/DomainNet"))
    server.ggeur_cfg = SimpleNamespace(
        domainnet_domains=["clipart", "real"],
        domainnet_shared_classes_only=False,
        domainnet_manifest_path=str(manifest),
    )

    domains, classes = server._get_domainnet_eval_metadata()
    records = server._get_domainnet_eval_records()

    assert domains == ["clipart", "real"]
    assert classes == ["bird", "car"]
    assert records["clipart"][0]["path"] == "clipart/bird/1.png"


def test_statistics_quorum_waits_for_unaccounted_clients():
    server = object.__new__(GGEURServer)
    server._client_num = 4
    server.min_statistics_clients = 4
    server.local_statistics_buffer = {1: {}, 2: {}}
    server.augmentation_ready_clients = {3}

    quorum, mixed, statistics, cache_ready = \
        server._statistics_quorum_state()

    assert not quorum
    assert not mixed
    assert statistics == {1, 2}
    assert cache_ready == {3}


def test_statistics_quorum_accepts_fully_accounted_mixed_cache():
    server = object.__new__(GGEURServer)
    server._client_num = 4
    server.min_statistics_clients = 4
    server.local_statistics_buffer = {1: {}, 2: {}}
    server.augmentation_ready_clients = {3, 4}

    quorum, mixed, statistics, cache_ready = \
        server._statistics_quorum_state()

    assert quorum
    assert mixed
    assert statistics | cache_ready == {1, 2, 3, 4}
    assert server._statistics_broadcast_receivers() == [1, 2]


def test_target_sized_augmentation_does_not_materialize_candidate_pool():
    client = object.__new__(GGEURClient)
    client.ID = 7
    client.embedding_dim = 2
    client._cov_factor_cache = {}
    generated_shapes = []

    def fake_generate(means, _covariance):
        generated_shapes.append(means.shape)
        return means + 0.5

    client._generate_samples_from_means = fake_generate
    original = torch.arange(20, dtype=torch.float32).reshape(10, 2).numpy()
    prototypes = torch.arange(
        6, dtype=torch.float32).reshape(3, 2).numpy()

    result = client._sample_target_sized_augmentation(
        original,
        prototypes,
        torch.eye(2).numpy(),
        num_per_sample=200,
        num_per_prototype=200,
        target_size=8,
    )

    assert result["features"].shape == (8, 2)
    assert result["conceptual_candidates"] == 2610
    assert result["sample_generated"] + result["prototype_generated"] <= 8
    assert sum(shape[0] for shape in generated_shapes) <= 8


def test_diagonal_covariance_generation_uses_variance_vector():
    client = object.__new__(GGEURClient)
    client.ID = 3
    client.embedding_dim = 3
    client.ggeur_cfg = SimpleNamespace(diagonal_covariance=True)
    client._cov_factor_cache = {}
    means = np.zeros((2000, 3), dtype=np.float32)
    variances = np.asarray([0.0, 1.0, 4.0], dtype=np.float32)

    generated = client._generate_samples_from_means(means, variances)

    assert generated.shape == means.shape
    assert generated[:, 0].std() < 0.01
    assert 0.8 < generated[:, 1].std() < 1.2
    assert 1.6 < generated[:, 2].std() < 2.4


def test_server_aggregates_diagonal_covariance_with_between_client_variance():
    server = object.__new__(GGEURServer)
    server.ggeur_cfg = SimpleNamespace(diagonal_covariance=True)
    server.global_cov_matrices = {}
    server.local_statistics_buffer = {
        1: {
            'means': {0: np.asarray([0.0, 2.0], dtype=np.float32)},
            'covs': {0: np.asarray([1.0, 4.0], dtype=np.float32)},
            'counts': {0: 2},
        },
        2: {
            'means': {0: np.asarray([2.0, 4.0], dtype=np.float32)},
            'covs': {0: np.asarray([1.0, 4.0], dtype=np.float32)},
            'counts': {0: 2},
        },
    }
    server._get_embedding_dim = lambda: 2

    server._aggregate_covariances()

    assert np.allclose(server.global_cov_matrices[0], [2.0, 5.0])


def test_officehome_manifest_set_uses_current_host_root():
    with tempfile.TemporaryDirectory() as directory:
        base = Path(directory) / "manifests"
        paths = []
        for client_id in (1, 2):
            path = (base / f"client_{client_id:06d}" /
                    "client_manifest.json")
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({
                "client_id": client_id,
                "root": "D:/source-machine/OfficeHome",
                "splits": {},
            }), encoding="utf-8")
            paths.append(str(path))

        host_root = "/root/current-host/OfficeHome"
        config = SimpleNamespace(
            data=SimpleNamespace(root=host_root),
            federate=SimpleNamespace(client_num=2),
            ggeur=SimpleNamespace(
                officehome_manifest_path="",
                officehome_manifest_base=str(base),
                officehome_manifest_use_config_root=True,
            ),
        )

        assert _resolve_manifest_paths(config) == paths
        manifest = json.loads(Path(paths[0]).read_text(encoding="utf-8"))
        assert _resolve_officehome_manifest_root(
            config, manifest, paths[0]) == host_root


if __name__ == "__main__":
    test_weighted_nested_parameter_aggregation()
    test_empty_updates_still_have_defined_result()
    test_hierarchical_fedproto_reduction_preserves_class_counts()
    test_root_fedproto_reduction_matches_flat_client_reduction()
    test_client_fedproto_uses_head_hidden_space_not_bert_input_space()
    test_mdsent_matrix_keeps_reference_branch_learning_rates()
    test_grpc_send_deduplicates_shared_proxy_endpoint()
    test_server_groups_identical_hierarchical_client_broadcasts()
    test_client_decodes_nested_distributed_model_payload()
    test_model_state_survives_message_tensor_serialization()
    test_two_level_model_aggregation_decodes_wire_tensors()
    test_fedproto_root_decodes_subserver_grpc_tensor_payload()
    test_feature_cache_keys_are_portable_across_hosts()
    test_statistics_quorum_waits_for_unaccounted_clients()
    test_statistics_quorum_accepts_fully_accounted_mixed_cache()
    test_target_sized_augmentation_does_not_materialize_candidate_pool()
    test_officehome_manifest_set_uses_current_host_root()
    print("hierarchical tests: PASS")
