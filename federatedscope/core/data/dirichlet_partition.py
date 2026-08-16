"""Deterministic label-skew partitions for single-process simulations."""

from __future__ import annotations

from typing import Iterable, List

import numpy as np


def partition_indices_by_label(labels: Iterable[int], num_clients: int,
                               alpha: float, seed: int) -> List[List[int]]:
    """Assign every sample exactly once using class-wise Dirichlet draws.

    The input samples already belong to one feature domain.  Consequently this
    function only changes the label distribution among clients in that domain;
    it never moves samples across domains or discards them.
    """
    label_array = np.asarray(list(labels), dtype=np.int64)
    if num_clients <= 0:
        raise ValueError('num_clients must be positive')
    if alpha <= 0:
        raise ValueError('alpha must be positive')

    rng = np.random.default_rng(seed)
    partitions: List[List[int]] = [[] for _ in range(num_clients)]
    for class_id in np.unique(label_array):
        class_indices = np.flatnonzero(label_array == class_id)
        rng.shuffle(class_indices)
        weights = rng.dirichlet(np.full(num_clients, alpha, dtype=float))
        counts = rng.multinomial(len(class_indices), weights)
        cursor = 0
        for client_index, count in enumerate(counts):
            end = cursor + int(count)
            partitions[client_index].extend(
                class_indices[cursor:end].astype(int).tolist())
            cursor = end

    # Very small alpha values can leave a client empty.  When the domain has
    # enough samples, move one sample from the largest donor to make every
    # simulated node runnable without duplicating or dropping data.
    if len(label_array) >= num_clients:
        for client_index, indices in enumerate(partitions):
            if indices:
                continue
            donor_index = max(range(num_clients),
                              key=lambda index: len(partitions[index]))
            if len(partitions[donor_index]) <= 1:
                break
            indices.append(partitions[donor_index].pop())

    for indices in partitions:
        rng.shuffle(indices)
    return partitions


def class_histograms(labels: Iterable[int], partitions: List[List[int]],
                     class_count: int) -> np.ndarray:
    """Build client-by-class counts for a partition."""
    label_array = np.asarray(list(labels), dtype=np.int64)
    histograms = np.zeros((len(partitions), class_count), dtype=np.int64)
    for client_index, indices in enumerate(partitions):
        if indices:
            histograms[client_index] = np.bincount(
                label_array[np.asarray(indices, dtype=np.int64)],
                minlength=class_count)[:class_count]
    return histograms
