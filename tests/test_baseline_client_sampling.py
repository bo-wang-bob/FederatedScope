import numpy as np
from types import SimpleNamespace

from federatedscope.contrib.worker.ggeur_client import GGEURClient


def test_client_local_sampling_downsamples_without_replacement():
    features = np.arange(30, dtype=np.float32).reshape(10, 3)
    labels = np.arange(10, dtype=np.int64) % 2

    sampled_a = GGEURClient._sample_client_local_dataset(
        features, labels, target_size=6, seed=42)
    sampled_b = GGEURClient._sample_client_local_dataset(
        features, labels, target_size=6, seed=42)

    np.testing.assert_array_equal(sampled_a[0], sampled_b[0])
    np.testing.assert_array_equal(sampled_a[1], sampled_b[1])
    assert sampled_a[0].shape == (6, 3)
    assert sampled_a[1].shape == (6,)
    assert sampled_a[2] is False
    assert sampled_a[3] == 6


def test_client_local_sampling_upsamples_from_same_client():
    features = np.arange(9, dtype=np.float32).reshape(3, 3)
    labels = np.array([0, 1, 1], dtype=np.int64)

    sampled_features, sampled_labels, replacement, unique_samples = \
        GGEURClient._sample_client_local_dataset(
            features, labels, target_size=8, seed=7)

    source_rows = {
        (tuple(feature.tolist()), int(label))
        for feature, label in zip(features, labels)
    }
    sampled_rows = {
        (tuple(feature.tolist()), int(label))
        for feature, label in zip(sampled_features, sampled_labels)
    }
    assert sampled_features.shape == (8, 3)
    assert sampled_labels.shape == (8,)
    assert replacement is True
    assert unique_samples == 3
    assert sampled_rows <= source_rows


def test_platform_sampling_resizes_generated_rows():
    client = object.__new__(GGEURClient)
    client.ID = 3
    client._cfg = SimpleNamespace(seed=42)
    client.ggeur_cfg = SimpleNamespace(
        platform_target_samples_per_client=12,
        num_generated_per_sample=1,
        num_generated_per_prototype=0,
        target_size_per_class=2,
    )
    client.augmented_features = np.array(
        [[0.0], [2.0], [10.0], [14.0]], dtype=np.float32)
    client.augmented_labels = np.array([0, 0, 1, 1], dtype=np.int64)

    assert client._apply_platform_training_sampling() is True
    assert client.augmented_features.shape == (12, 1)
    source_rows = {tuple(row.tolist()) for row in np.array(
        [[0.0], [2.0], [10.0], [14.0]], dtype=np.float32)}
    sampled_rows = {
        tuple(row.tolist()) for row in client.augmented_features
    }
    assert sampled_rows <= source_rows
