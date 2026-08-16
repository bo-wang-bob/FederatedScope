"""Deterministic OfficeHome previews and replayable client manifests."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from federatedscope.core.data.dirichlet_partition import (
    class_histograms, partition_indices_by_label)
from federatedscope.standalone_api.schemas import DOMAIN_KEYS


FALLBACK_DOMAIN_TOTALS = [1698, 3055, 3107, 3049]


def _relative_record(path: Path, root: Path, label: int) -> Dict[str, Any]:
    return {'path': os.path.relpath(path, root), 'label': int(label)}


def _dataset_inventory(seed: int) -> Optional[Dict[str, Any]]:
    """Build the exact train/test inventory without decoding any image."""
    root = Path(os.environ.get(
        'FEDERATEDSCOPE_DATA_ROOT',
        '/root/autodl-tmp/datasets/OfficeHomeDataset_10072016'))
    if not root.exists():
        return None
    from federatedscope.cv.dataset.office_home import OfficeHome

    result = []
    fingerprint_rows: List[str] = []
    for domain in DOMAIN_KEYS:
        domain_path = root / (domain.replace('_', ' ')
                              if 'Real' in domain else domain)
        if not domain_path.exists():
            domain_path = root / domain.replace(' ', '_')
        if not domain_path.exists():
            return None
        records = []
        for class_index, class_name in enumerate(OfficeHome.CLASSES):
            class_path = domain_path / class_name
            if not class_path.exists():
                continue
            image_paths = [
                class_path / filename
                for filename in sorted(os.listdir(class_path))
                if (class_path / filename).is_file() and
                (class_path / filename).suffix.lower() in
                {'.jpg', '.jpeg', '.png'}
            ]
            for image_path in image_paths:
                record = _relative_record(image_path, root, class_index)
                records.append(record)
                fingerprint_rows.append(
                    f"{domain}:{record['path']}:{record['label']}")
        if not records:
            return None
        label_array = np.asarray(
            [record['label'] for record in records], dtype=np.int64)
        legacy_rng = np.random.RandomState(seed)
        permutation = legacy_rng.permutation(len(label_array))
        train_size = int(len(label_array) * 0.7)
        train_indices = permutation[:train_size]
        test_indices = permutation[train_size:]
        result.append({
            'domainKey': domain,
            'trainRecords': [records[int(index)] for index in train_indices],
            'trainLabels': label_array[train_indices],
            'valRecords': [],
            'testRecords': [records[int(index)] for index in test_indices],
        })
    fingerprint = hashlib.sha256(
        '\n'.join(sorted(fingerprint_rows)).encode()).hexdigest()
    return {
        'root': str(root.resolve()),
        'domains': result,
        'datasetFingerprint': fingerprint,
    }


def _build_partition_bundle(
        request: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    alpha = float(request['partition']['alpha'])
    seed = int(request['partition']['seed'])
    clients_per_domain = int(request['clientsPerDomain'])
    inventory = _dataset_inventory(seed)
    basis = 'actual_dataset' if inventory is not None else \
        'built_in_simulation'
    domains = []
    fingerprints = []
    manifest_clients = []
    manifest_domains = {}
    for domain_index, domain_key in enumerate(DOMAIN_KEYS):
        rng = np.random.default_rng(seed + domain_index * 1009)
        if inventory is not None:
            domain_inventory = inventory['domains'][domain_index]
            labels = domain_inventory['trainLabels']
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
            if inventory is not None:
                manifest_clients.append({
                    'clientId': domain_index * clients_per_domain +
                    client_index + 1,
                    'displayId': f'{prefix}-C{client_index + 1:02d}',
                    'domain': domain_key,
                    'train': [
                        domain_inventory['trainRecords'][index]
                        for index in partitions[client_index]
                    ],
                })
        if inventory is not None:
            manifest_domains[domain_key] = {
                'val': domain_inventory['valRecords'],
                'test': domain_inventory['testRecords'],
            }
        domains.append({
            'domainKey': domain_key,
            'totalSamples': total,
            'classCount': 65,
            'clients': clients,
        })
    dataset_fingerprint = (inventory['datasetFingerprint']
                           if inventory is not None else
                           'builtin-officehome-scale-v1')
    digest = hashlib.sha256(
        f'office-home:{alpha}:{seed}:{clients_per_domain}:'
        f'{dataset_fingerprint}:{fingerprints}'.encode()).hexdigest()
    preview = {
        'datasetKey': 'office-home',
        'alpha': alpha,
        'seed': seed,
        'partitionVersion': digest[:12],
        'source': 'backend',
        'basis': basis,
        'datasetFingerprint': dataset_fingerprint,
        'domains': domains,
    }
    manifest = None
    if inventory is not None:
        manifest = {
            'schemaVersion': '2.0',
            'dataset': 'office-home',
            'root': inventory['root'],
            'datasetFingerprint': dataset_fingerprint,
            'partitionVersion': preview['partitionVersion'],
            'alpha': alpha,
            'seed': seed,
            'clientsPerDomain': clients_per_domain,
            'domains': manifest_domains,
            'clients': manifest_clients,
        }
    return preview, manifest


def build_partition(request: Dict[str, Any]) -> Dict[str, Any]:
    return _build_partition_bundle(request)[0]


def build_partition_artifacts(
        request: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """Return the public preview and an exact training replay manifest."""
    return _build_partition_bundle(request)


def current_dataset_fingerprint(seed: int) -> Optional[str]:
    inventory = _dataset_inventory(seed)
    return inventory['datasetFingerprint'] if inventory is not None else None
