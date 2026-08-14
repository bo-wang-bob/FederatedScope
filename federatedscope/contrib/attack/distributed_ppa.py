"""Distributed Meta-PPA for the real GGEUR gRPC execution path.

Each client keeps its images and embedding probes local.  At selected rounds
it computes the same parameter-sensitivity feature used by the standalone
Meta-PPA implementation (global sensitivity minus local sensitivity), then
uploads only that derived vector and the local majority-class property label.
The server trains the configured meta-classifier after federated training.
"""

from __future__ import annotations

import copy
import json
import logging
import os
from collections import Counter
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, \
    confusion_matrix
from sklearn.model_selection import train_test_split

from federatedscope.core.message import Message

logger = logging.getLogger(__name__)


def _cpu_state(state):
    if state is None:
        return None
    return {
        key: value.detach().cpu().clone()
        if torch.is_tensor(value) else copy.deepcopy(value)
        for key, value in state.items()
    }


def _get_classifier(classifier: str):
    classifier = str(classifier).lower()
    if classifier == 'lr':
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(random_state=0, max_iter=1000)
    if classifier == 'randomforest':
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(random_state=0)
    if classifier == 'svm':
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        return make_pipeline(StandardScaler(), SVC(gamma='auto'))
    raise ValueError(
        f'Unsupported property inference classifier: {classifier}')


def _can_stratify(labels: np.ndarray) -> bool:
    _, counts = np.unique(labels, return_counts=True)
    return len(counts) > 1 and bool(np.all(counts >= 2))


class DistributedPPAClientReporter:
    """Build standalone-compatible Meta-PPA features on a real client."""

    def __init__(self, client):
        self.client = client
        self.cfg = client._cfg
        self.attack_cfg = client._cfg.attack
        self._probe_features_by_class = None
        self._probe_counts = {}
        self._property_label = None

    def should_report(self, round_idx: int) -> bool:
        interval = max(1, int(self.attack_cfg.meta_ppa_save_interval))
        final_round = max(1, int(self.cfg.federate.total_round_num) - 1)
        return int(round_idx) % interval == 0 or \
            int(round_idx) >= final_round

    def _get_train_loader(self):
        data = None
        try:
            data = self.client.trainer.ctx.data.get('train', None)
        except Exception:
            data = None
        if data is None and isinstance(self.client.data, dict):
            data = self.client.data.get('train', None)
        return data

    def _ensure_probe_features(self):
        if self._probe_features_by_class is not None:
            return self._probe_features_by_class

        loader = self._get_train_loader()
        if loader is None:
            raise ValueError('distributed PPA requires a local train split')
        features, labels = self.client._extract_eval_features(loader)
        if features is None or labels is None or len(labels) == 0:
            raise ValueError('distributed PPA could not extract train probes')

        features = torch.as_tensor(features, dtype=torch.float32).cpu()
        labels = torch.as_tensor(labels, dtype=torch.long).cpu()
        label_counts = Counter(int(value) for value in labels.tolist())
        self._property_label = int(label_counts.most_common(1)[0][0])

        num_classes = int(self.cfg.model.num_classes)
        limit = max(1, int(
            self.attack_cfg.meta_ppa_probe_samples_per_class))
        probes = {}
        counts = {}
        for class_idx in range(num_classes):
            indices = torch.nonzero(labels == class_idx, as_tuple=False).view(-1)
            if len(indices) > limit:
                generator = torch.Generator().manual_seed(
                    int(self.cfg.seed) + 1009 * int(self.client.ID) +
                    97 * int(class_idx))
                order = torch.randperm(len(indices), generator=generator)
                indices = indices[order[:limit]]
            if len(indices) > 0:
                probes[class_idx] = features[indices].contiguous()
            counts[class_idx] = int(len(indices))

        self._probe_features_by_class = probes
        self._probe_counts = counts
        logger.info(
            'Client %s: prepared distributed PPA probes; property=%s, '
            'nonempty_classes=%s/%s', self.client.ID, self._property_label,
            sum(value > 0 for value in counts.values()), num_classes)
        return probes

    def _target_param_names(self, model) -> List[str]:
        configured = list(self.attack_cfg.meta_ppa_target_layers)
        names = [name for name, _ in model.named_parameters()]
        if not configured:
            return names
        selected = [
            name for name in names
            if any(name == layer or name.startswith(layer + '.')
                   for layer in configured)
        ]
        return selected or names

    def _snapshot_params(self, model) -> Dict[str, np.ndarray]:
        selected = set(self._target_param_names(model))
        return {
            name: param.detach().cpu().numpy().copy()
            for name, param in model.named_parameters()
            if name in selected
        }

    @staticmethod
    def _layer_sensitivity(old_params: Dict[str, np.ndarray],
                           new_params: Dict[str, np.ndarray]) -> np.ndarray:
        values = []
        for name in sorted(old_params.keys()):
            if name not in new_params:
                continue
            diff = old_params[name] - new_params[name]
            values.extend([
                float(np.abs(diff).sum()),
                float(np.sqrt(np.square(diff).sum())),
                float(np.abs(diff).max()) if diff.size else 0.0,
            ])
        return np.asarray(values, dtype=np.float64)

    def _compute_multi_sensitivity(self, state_dict) -> np.ndarray:
        probes = self._ensure_probe_features()
        template = self.client.mlp_classifier
        if template is None:
            raise ValueError('distributed PPA requires the GGEUR MLP head')

        device = self.client.device
        epochs = max(1, int(self.attack_cfg.meta_ppa_probe_epochs))
        lr = float(self.attack_cfg.meta_ppa_probe_lr)
        batch_size = max(1, int(
            self.attack_cfg.meta_ppa_probe_batch_size))
        num_classes = int(self.cfg.model.num_classes)
        stat_width = 3 * len(self._target_param_names(template))
        sensitivities = []

        for class_idx in range(num_classes):
            class_features = probes.get(class_idx)
            if class_features is None or len(class_features) == 0:
                sensitivities.append(np.zeros(stat_width, dtype=np.float64))
                continue

            model = copy.deepcopy(template).to(device)
            model.load_state_dict(state_dict, strict=False)
            model.train()
            old_params = self._snapshot_params(model)
            x = class_features.to(device)
            y = torch.full((len(x),), class_idx, dtype=torch.long,
                           device=device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()

            for _ in range(epochs):
                for start in range(0, len(x), batch_size):
                    end = min(start + batch_size, len(x))
                    optimizer.zero_grad()
                    loss = criterion(model(x[start:end]), y[start:end])
                    loss.backward()
                    optimizer.step()

            new_params = self._snapshot_params(model)
            sensitivities.append(
                self._layer_sensitivity(old_params, new_params))
            del model

        return np.stack(sensitivities)

    def build_report(self, round_idx: int, global_state, local_state):
        if global_state is None or local_state is None:
            return None
        global_state = _cpu_state(global_state)
        local_state = _cpu_state(local_state)
        global_sensitivity = self._compute_multi_sensitivity(global_state)
        local_sensitivity = self._compute_multi_sensitivity(local_state)
        feature = (global_sensitivity - local_sensitivity).reshape(-1)
        feature = np.nan_to_num(feature, nan=0.0, posinf=0.0, neginf=0.0)
        return {
            'schema_version': 1,
            'client_id': int(self.client.ID),
            'round': int(round_idx),
            'property_label': int(self._property_label),
            'feature': feature.astype(np.float32).tolist(),
            'metadata': {
                'feature_shape': list(global_sensitivity.shape),
                'feature_dim': int(feature.size),
                'target_layers': self._target_param_names(
                    self.client.mlp_classifier),
                'probe_counts': {
                    str(key): int(value)
                    for key, value in self._probe_counts.items()
                },
                'feature_extractor': str(
                    self.client.feature_extractor_type),
                'embedding_dim': int(self.client.embedding_dim),
                'derived_features_only': 1,
            },
        }

    def report(self, round_idx: int, server_id: int, timestamp,
               global_state, local_state):
        if not self.should_report(round_idx):
            return
        try:
            payload = self.build_report(round_idx, global_state, local_state)
            if payload is None:
                return
            self.client.comm_manager.send(Message(
                msg_type='ppa_features',
                sender=self.client.ID,
                receiver=[server_id],
                state=int(round_idx),
                timestamp=timestamp,
                content=payload,
            ))
            logger.info(
                'Client %s: uploaded distributed PPA feature for round %s '
                '(property=%s, dim=%s)', self.client.ID, round_idx,
                payload['property_label'],
                payload['metadata']['feature_dim'])
        except Exception as error:
            logger.exception(
                'Client %s: distributed PPA report failed at round %s: %s',
                self.client.ID, round_idx, error)


class DistributedPPACollector:
    """Collect client-derived PPA features and train the meta-classifier."""

    def __init__(self, server):
        self.server = server
        self.cfg = server._cfg
        self.attack_cfg = server._cfg.attack
        self.reports: Dict[int, Dict[int, Dict]] = {}
        self.report_dir = os.path.join(
            self.cfg.outdir, 'distributed_ppa_reports')
        self._finalized = False

    def handle(self, message):
        payload = message.content
        if not isinstance(payload, dict):
            logger.warning('Server: invalid distributed PPA payload from %s',
                           message.sender)
            return
        client_id = int(payload.get('client_id', message.sender))
        round_idx = int(payload.get('round', message.state))
        if client_id != int(message.sender):
            logger.warning('Server: PPA payload client mismatch: sender=%s '
                           'payload=%s', message.sender, client_id)
            return
        feature = np.asarray(payload.get('feature', []), dtype=np.float64)
        if feature.size == 0 or not bool(np.all(np.isfinite(feature))):
            logger.warning('Server: invalid distributed PPA feature from '
                           'client %s round %s', client_id, round_idx)
            return
        self.reports.setdefault(round_idx, {})[client_id] = payload
        self._save_report(round_idx, client_id, payload)
        logger.info('Server: received distributed PPA feature from client %s '
                    'for round %s', client_id, round_idx)

    def _save_report(self, round_idx: int, client_id: int, payload):
        os.makedirs(self.report_dir, exist_ok=True)
        path = os.path.join(
            self.report_dir, f'client_{client_id}_round_{round_idx}.json')
        with open(path, 'w', encoding='utf-8') as file:
            json.dump(payload, file, indent=2)

    def _collect_samples(self):
        per_client = {}
        for round_idx, clients in sorted(self.reports.items()):
            for client_id, report in sorted(clients.items()):
                per_client.setdefault(client_id, []).append(
                    (round_idx, report))

        max_clients = int(self.attack_cfg.meta_ppa_max_clients)
        max_rounds = int(self.attack_cfg.meta_ppa_max_attack_rounds)
        features, labels, metadata = [], [], []
        feature_dim = None
        selected_clients = sorted(per_client.keys())
        if max_clients > 0:
            selected_clients = selected_clients[:max_clients]

        for client_id in selected_clients:
            entries = per_client[client_id]
            if max_rounds > 0:
                entries = entries[-max_rounds:]
            for round_idx, report in entries:
                feature = np.asarray(
                    report.get('feature', []), dtype=np.float32).reshape(-1)
                label = report.get('property_label')
                if feature.size == 0 or label is None:
                    continue
                if feature_dim is None:
                    feature_dim = int(feature.size)
                if int(feature.size) != feature_dim:
                    logger.warning(
                        'Server: skip PPA feature with mismatched dimension '
                        'client=%s round=%s got=%s expected=%s', client_id,
                        round_idx, feature.size, feature_dim)
                    continue
                features.append(feature)
                labels.append(int(label))
                metadata.append({
                    'round': int(round_idx),
                    'client_id': int(client_id),
                    'label': int(label),
                })
        return features, labels, metadata

    def _run_attack(self):
        features, labels, sample_meta = self._collect_samples()
        if len(features) < 4:
            raise ValueError(
                f'distributed meta_ppa only collected {len(features)} samples')
        x = np.stack(features).astype(np.float32)
        y = np.asarray(labels, dtype=np.int64)
        if len(np.unique(y)) < 2:
            raise ValueError('distributed meta_ppa requires at least two '
                             'observed property classes')

        stratify = y if _can_stratify(y) else None
        x_train, x_test, y_train, y_test, meta_train, meta_test = \
            train_test_split(
                x, y, sample_meta, test_size=0.33,
                random_state=int(self.cfg.seed), stratify=stratify)
        classifier_name = str(self.attack_cfg.classifier_PIA)
        classifier = _get_classifier(classifier_name)
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)

        accuracy = float(accuracy_score(y_test, y_pred))
        report = classification_report(
            y_test, y_pred, output_dict=True, zero_division=0)
        first_report = None
        for clients in self.reports.values():
            if clients:
                first_report = next(iter(clients.values()))
                break
        first_metadata = first_report.get('metadata', {}) \
            if first_report else {}
        num_classes = int(self.cfg.model.num_classes)
        return {
            'attack_success_rate': float(
                report.get('macro avg', {}).get('recall', 0.0)),
            'attack_type': 'property_inference',
            'accuracy': accuracy,
            'random_baseline': float(1.0 / max(num_classes, 1)),
            'num_total_samples': int(len(x)),
            'num_train_samples': int(len(x_train)),
            'num_test_samples': int(len(x_test)),
            'num_classes': num_classes,
            'observed_num_classes': int(len(np.unique(y))),
            'classifier': classifier_name,
            'target_layers': first_metadata.get('target_layers', []),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': report,
            'test_samples': meta_test,
            'metadata': {
                'feature_shape': list(x.shape),
                'train_samples': meta_train,
                'probe_counts': first_metadata.get('probe_counts', {}),
                'derived_features_only': 1,
            },
        }

    def finalize(self):
        if self._finalized:
            return
        self._finalized = True
        save_path = os.path.join(
            self.cfg.outdir, 'distributed_ppa_results.json')
        os.makedirs(self.cfg.outdir, exist_ok=True)
        try:
            result = {'meta_ppa': self._run_attack()}
            metrics = result['meta_ppa']
            logger.info(
                'Server: distributed Meta-PPA accuracy=%s baseline=%s '
                'samples=%s', metrics.get('accuracy'),
                metrics.get('random_baseline'),
                metrics.get('num_total_samples'))
        except Exception as error:
            logger.exception('Server: distributed Meta-PPA failed: %s', error)
            result = {'meta_ppa': {'error': str(error)}}
        with open(save_path, 'w', encoding='utf-8') as file:
            json.dump(result, file, indent=2)
        logger.info('Server: distributed PPA result saved to %s', save_path)
