"""
Meta-classifier property inference plugin for GGEUR MLP heads.

The feature follows the standalone PPA pipeline:
for each saved round/client MLP state, compute per-class sensitivity on a
small probe set, then use ``global_sensitivity - local_sensitivity`` as the
attack feature.  The inferred property is the dominant class of the client's
local training data.
"""

from __future__ import annotations

import copy
import logging
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, \
    confusion_matrix
from sklearn.model_selection import train_test_split

from ..base import AttackPlugin
from ..registry import AttackRegistry

logger = logging.getLogger(__name__)


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
    raise ValueError(f"Unsupported property inference classifier: {classifier}")


@AttackRegistry.register
class MetaPPAPlugin(AttackPlugin):
    """Multi-class property inference attack for server-visible GGEUR MLPs."""

    @property
    def name(self) -> str:
        return "meta_ppa"

    @property
    def is_cross_round(self) -> bool:
        return True

    def compute_scores(
        self,
        target_data: Dict,
        shadow_data: Optional[List[Dict]] = None,
        global_model=None,
        config=None,
    ) -> Dict:
        if global_model is None:
            raise ValueError("meta_ppa requires global_model")

        all_client_data = [target_data] + list(shadow_data or [])
        probe_features = self._get_probe_features(all_client_data)
        if not probe_features:
            raise ValueError("meta_ppa requires ppa_probe_features_by_class")

        num_classes = int(getattr(config.model, 'num_classes', 0)) \
            if config is not None else 0
        if num_classes <= 0:
            num_classes = max(int(k) for k in probe_features.keys()) + 1

        max_rounds = int(getattr(config.attack, 'meta_ppa_max_attack_rounds',
                                getattr(config.attack, 'ppa_max_attack_rounds',
                                        5))) if config is not None else 5
        max_clients = int(getattr(config.attack, 'meta_ppa_max_clients',
                                 getattr(config.attack, 'ppa_max_clients',
                                         0))) if config is not None else 0

        features, labels, sample_meta = self._collect_attack_samples(
            all_client_data, global_model, probe_features, config, max_rounds,
            max_clients)
        if len(features) < 4:
            raise ValueError(f"meta_ppa only collected {len(features)} samples")

        x = np.stack(features).astype(np.float32)
        y = np.asarray(labels, dtype=np.int64)

        stratify = y if self._can_stratify(y) else None
        x_train, x_test, y_train, y_test, meta_train, meta_test = \
            train_test_split(x,
                             y,
                             sample_meta,
                             test_size=0.33,
                             random_state=int(getattr(config, 'seed', 0))
                             if config is not None else 0,
                             stratify=stratify)

        classifier_name = getattr(config.attack, 'classifier_PIA', 'svm') \
            if config is not None else 'svm'
        classifier = _get_classifier(classifier_name)
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)

        accuracy = float(accuracy_score(y_test, y_pred))
        report = classification_report(y_test,
                                       y_pred,
                                       output_dict=True,
                                       zero_division=0)
        attack_success_rate = float(report.get('macro avg',
                                               {}).get('recall', 0.0))
        return {
            'attack_success_rate': attack_success_rate,
            'attack_type': 'property_inference',
            'accuracy': accuracy,
            'random_baseline': float(1.0 / max(num_classes, 1)),
            'num_total_samples': int(len(x)),
            'num_train_samples': int(len(x_train)),
            'num_test_samples': int(len(x_test)),
            'num_classes': int(num_classes),
            'classifier': classifier_name,
            'target_layers': self._target_param_names(global_model, config),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': report,
            'test_samples': meta_test,
            'metadata': {
                'feature_shape': list(x.shape),
                'train_samples': meta_train,
                'probe_counts': {
                    str(k): int(len(v))
                    for k, v in probe_features.items()
                },
            },
        }

    def _collect_attack_samples(self, all_client_data, global_model,
                                probe_features, config, max_rounds,
                                max_clients):
        features, labels, sample_meta = [], [], []
        global_cache = {}
        client_count = 0

        for client_data in all_client_data:
            label = client_data.get('property_label')
            if label is None:
                continue
            client_count += 1
            if max_clients > 0 and client_count > max_clients:
                break

            local_states = client_data.get('ppa_local_state', {})
            global_states = client_data.get('ppa_global_state', {})
            rounds = sorted(set(local_states.keys()) & set(global_states.keys()))
            if max_rounds > 0:
                rounds = rounds[-max_rounds:]

            client_id = client_data.get('client_id')
            for round_num in rounds:
                global_sensitivity = global_cache.get(round_num)
                if global_sensitivity is None:
                    global_sensitivity = self._compute_multi_sensitivity(
                        global_model, global_states[round_num], probe_features,
                        config)
                    global_cache[round_num] = global_sensitivity

                local_sensitivity = self._compute_multi_sensitivity(
                    global_model, local_states[round_num], probe_features,
                    config)
                features.append((global_sensitivity -
                                 local_sensitivity).reshape(-1))
                labels.append(int(label))
                sample_meta.append({
                    'round': int(round_num),
                    'client_id': int(client_id)
                    if client_id is not None else None,
                    'label': int(label),
                })

        return features, labels, sample_meta

    def _compute_multi_sensitivity(self, template_model, state_dict,
                                   probe_features, config):
        device = next(template_model.parameters()).device
        epochs = int(getattr(config.attack, 'meta_ppa_probe_epochs',
                            getattr(config.attack, 'ppa_probe_epochs', 1))) \
            if config is not None else 1
        lr = float(getattr(config.attack, 'meta_ppa_probe_lr',
                          getattr(config.attack, 'ppa_probe_lr', 1e-3))) \
            if config is not None else 1e-3
        batch_size = int(getattr(config.attack, 'meta_ppa_probe_batch_size',
                                getattr(config.attack, 'ppa_probe_batch_size',
                                        16))) if config is not None else 16

        sensitivities = []
        for class_idx in sorted(probe_features.keys()):
            features = np.asarray(probe_features[class_idx], dtype=np.float32)
            if features.size == 0:
                continue

            model = copy.deepcopy(template_model).to(device)
            model.load_state_dict(state_dict, strict=False)
            model.train()

            old_params = self._snapshot_params(model, config)
            x = torch.from_numpy(features).float().to(device)
            y = torch.full((len(features),),
                           int(class_idx),
                           dtype=torch.long,
                           device=device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()

            for _ in range(max(1, epochs)):
                for start in range(0, len(features), batch_size):
                    end = min(start + batch_size, len(features))
                    optimizer.zero_grad()
                    loss = criterion(model(x[start:end]), y[start:end])
                    loss.backward()
                    optimizer.step()

            new_params = self._snapshot_params(model, config)
            sensitivities.append(
                self._layer_sensitivity(old_params, new_params))

        if not sensitivities:
            raise ValueError("meta_ppa could not compute any sensitivities")
        return np.stack(sensitivities)

    def _target_param_names(self, model, config) -> List[str]:
        configured = []
        if config is not None:
            configured = list(getattr(config.attack, 'meta_ppa_target_layers',
                                      getattr(config.attack,
                                              'ppa_target_layers', [])))
        names = [name for name, _ in model.named_parameters()]
        if not configured:
            return names
        selected = [
            name for name in names
            if any(name == layer or name.startswith(layer + '.')
                   for layer in configured)
        ]
        return selected or names

    def _snapshot_params(self, model, config) -> Dict[str, np.ndarray]:
        target_names = set(self._target_param_names(model, config))
        return {
            name: param.detach().cpu().numpy().copy()
            for name, param in model.named_parameters()
            if name in target_names
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

    @staticmethod
    def _get_probe_features(all_client_data):
        for client_data in all_client_data:
            probe = client_data.get('ppa_probe_features_by_class')
            if probe:
                return {
                    int(k): np.asarray(v, dtype=np.float32)
                    for k, v in probe.items()
                    if np.asarray(v).size > 0
                }
        return {}

    @staticmethod
    def _can_stratify(y: np.ndarray) -> bool:
        _, counts = np.unique(y, return_counts=True)
        return len(counts) > 1 and np.all(counts >= 2)
