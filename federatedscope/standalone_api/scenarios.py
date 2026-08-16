"""Deterministic OfficeHome client partition previews."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np

from federatedscope.core.data.dirichlet_partition import (
    class_histograms, partition_indices_by_label)
from federatedscope.standalone_api.schemas import DOMAIN_KEYS


FALLBACK_DOMAIN_TOTALS = [1698, 3055, 3107, 3049]


def _dataset_labels(seed: int):
    """Read real OfficeHome class counts without decoding any image."""
    root = Path(os.environ.get(
        'FEDERATEDSCOPE_DATA_ROOT',
        '/root/autodl-tmp/datasets/OfficeHomeDataset_10072016'))
    if not root.exists():
        return None
    from federatedscope.cv.dataset.office_home import OfficeHome

    result = []
    for domain in DOMAIN_KEYS:
        domain_path = root / (domain.replace('_', ' ')
                              if 'Real' in domain else domain)
        if not domain_path.exists():
            domain_path = root / domain.replace(' ', '_')
        if not domain_path.exists():
            return None
        labels = []
        for class_index, class_name in enumerate(OfficeHome.CLASSES):
            class_path = domain_path / class_name
            if not class_path.exists():
                continue
            count = sum(
                1 for path in class_path.iterdir()
                if path.is_file() and
                path.suffix.lower() in {'.jpg', '.jpeg', '.png'})
            labels.extend([class_index] * count)
        if not labels:
            return None
        label_array = np.asarray(labels, dtype=np.int64)
        legacy_rng = np.random.RandomState(seed)
        train_indices = legacy_rng.permutation(len(label_array))[
            :int(len(label_array) * 0.7)]
        result.append(label_array[train_indices])
    return result


def build_partition(request: Dict[str, Any]) -> Dict[str, Any]:
    alpha = float(request['partition']['alpha'])
    seed = int(request['partition']['seed'])
    clients_per_domain = int(request['clientsPerDomain'])
    dataset_labels = _dataset_labels(seed)
    basis = 'actual_dataset' if dataset_labels is not None else \
        'built_in_simulation'
    domains = []
    fingerprints = []
    for domain_index, domain_key in enumerate(DOMAIN_KEYS):
        rng = np.random.default_rng(seed + domain_index * 1009)
        if dataset_labels is not None:
            labels = dataset_labels[domain_index]
            class_totals = np.bincount(labels, minlength=65)[:65]
        else:
            total = FALLBACK_DOMAIN_TOTALS[domain_index]
            class_weights = rng.dirichlet(np.full(65, 1.4))
            class_totals = rng.multinomial(total, class_weights)
            labels = np.repeat(np.arange(65, dtype=np.int64), class_totals)
        total = int(len(labels))
        fingerprints.append(class_totals.tolist())
        partitions = partition_indices_by_label(
            labels, clients_per_domain, alpha, seed + domain_index * 1009)
        histograms = class_histograms(labels, partitions, 65)
        clients = []
        prefix = ['OH-DT', 'OH-TS', 'OH-ED', 'OH-FR'][domain_index]
        for client_index, histogram in enumerate(histograms):
            sample_count = int(histogram.sum())
            proportions = (histogram / sample_count if sample_count else
                           np.zeros_like(histogram, dtype=float))
            nonzero = proportions[proportions > 0]
            entropy = float(-(nonzero * np.log2(nonzero)).sum())
            dominant = int(proportions.argmax()) if sample_count else 0
            covered = int((histogram > 0).sum())
            clients.append({
                'clientId': f'{prefix}-C{client_index + 1:02d}',
                'domainKey': domain_key,
                'sampleCount': sample_count,
                'domainSampleRatio': sample_count / total,
                'classHistogram': histogram.tolist(),
                'classProportions': proportions.tolist(),
                'coveredClassCount': covered,
                'missingClassCount': 65 - covered,
                'dominantClassIndex': dominant,
                'dominantClassRatio': float(proportions[dominant]),
                'labelEntropy': entropy,
            })
        domains.append({
            'domainKey': domain_key,
            'totalSamples': total,
            'classCount': 65,
            'clients': clients,
        })
    digest = hashlib.sha256(
        f'office-home:{alpha}:{seed}:{clients_per_domain}:'
        f'{fingerprints}'.encode()).hexdigest()
    return {
        'datasetKey': 'office-home',
        'alpha': alpha,
        'seed': seed,
        'partitionVersion': digest[:12],
        'source': 'backend',
        'basis': basis,
        'domains': domains,
    }
