"""Client heterogeneity must preserve every training row and its identity."""
import numpy as np
import pytest

from federatedscope.contrib.data.ggeur_data import _split_dataset_dirichlet_clients


class Labels:
    targets = np.repeat(np.arange(25), 40)

    def __len__(self):
        return len(self.targets)


def partition(alpha, seed=42):
    return [part.indices for part in _split_dataset_dirichlet_clients(Labels(), 3, alpha, seed)]


@pytest.mark.parametrize('alpha', [0.01, 0.1, 1.0, 100.0])
def test_full_disjoint_seeded_allocation(alpha):
    rows = partition(alpha)
    assert rows == partition(alpha)
    assert sorted(i for client in rows for i in client) == list(range(1000))
    assert all(rows)
    assert rows != partition(alpha, seed=123)


def test_alpha_changes_real_label_heterogeneity():
    def concentration(rows):
        hist = np.asarray([np.bincount(Labels.targets[client], minlength=25) for client in rows])
        return np.mean(np.sum((hist / hist.sum(axis=0)) ** 2, axis=0))
    low, high = partition(.01), partition(100.)
    assert low != high
    assert concentration(low) > concentration(high) + .4


def test_empty_clients_are_repaired_without_losing_or_reusing_samples():
    data = Labels()
    data.targets = np.zeros(5, dtype=int)
    rows = _split_dataset_dirichlet_clients(data, 5, .01, 42)
    assert all(len(row) == 1 for row in rows)
    assert sorted(i for row in rows for i in row.indices) == list(range(5))
