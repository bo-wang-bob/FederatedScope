"""
GGEUR-native FedMIA server.

This version runs on the real `federate.method: ggeur` training path by
inheriting from `GGEURServer`, then writes compact feature-space FedMIA
artifacts to disk and runs the final attack from those artifacts.
"""

from __future__ import annotations

import copy
import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from federatedscope.contrib.worker.ggeur_server import GGEURServer
from federatedscope.core.monitoring.events import emit_training_event

logger = logging.getLogger(__name__)


class GGEURFedMIAServer(GGEURServer):
    """FedMIA server that follows the real GGEUR training protocol.

    Current implementation saves compact loss/logit and per-sample gradient
    features for the GGEUR MLP head, enabling the standard modular FedMIA
    attack family without keeping all round messages in memory.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.target_client_id = int(getattr(self._cfg.attack,
                                            'fedmia_target_client_id', 1))
        self.full_train_dataset = None
        self.client_train_datasets: Dict[int, object] = {}
        self.client_data_indices: Dict[int, List[int]] = {}
        self.client_refs: Dict[int, object] = {}
        self._attack_executed = False
        self.attack_feature_dir = os.path.join(self._cfg.outdir,
                                               'ggeur_fedmia_features')
        self.saved_feature_index: Dict[int, set] = {}
        self.round_global_classifier_states: Dict[int, Dict[str, torch.Tensor]] = {}
        self.fedmia_save_interval = int(getattr(
            self._cfg.attack, 'fedmia_save_interval',
            getattr(self._cfg.attack, 'save_interval', 5)))
        self.store_grad_cos = bool(getattr(
            self._cfg.attack, 'fedmia_store_grad_cos', True))

        self.het_handler = None
        if getattr(self._cfg.attack, 'use_ggeur', False):
            from federatedscope.contrib.attack.heterogeneity import GGEURHandler
            self.het_handler = GGEURHandler(self._cfg)

        self.orchestrator = None
        if getattr(self._cfg.attack, 'modular_attacks', False):
            import federatedscope.contrib.attack.plugins.blackbox_loss  # noqa: F401
            import federatedscope.contrib.attack.plugins.grad_cosine  # noqa: F401
            import federatedscope.contrib.attack.plugins.grad_diff  # noqa: F401
            import federatedscope.contrib.attack.plugins.grad_norm  # noqa: F401
            import federatedscope.contrib.attack.plugins.loss_series  # noqa: F401
            import federatedscope.contrib.attack.plugins.avg_cosine  # noqa: F401
            import federatedscope.contrib.attack.plugins.fedmia_i  # noqa: F401
            import federatedscope.contrib.attack.plugins.fedmia_ii  # noqa: F401
            from federatedscope.contrib.attack.orchestrator import AttackOrchestrator

            attack_names = list(getattr(self._cfg.attack, 'attack_plugins', []))
            self.orchestrator = AttackOrchestrator(
                het_handler=self.het_handler,
                attack_names=attack_names if attack_names else None,
                config=self._cfg,
            )
            logger.info('GGEURFedMIAServer: modular loss-based FedMIA ready '
                        f'with plugins {attack_names}')

    def link_clients(self, clients):
        self.clients = clients
        try:
            if hasattr(super(), 'link_clients'):
                super().link_clients(clients)
        except Exception as e:
            logger.debug(f'GGEURFedMIAServer.link_clients super skipped: {e}')

        if not clients:
            return

        try:
            first_client = clients.get(1) if 1 in clients else list(clients.values())[0]
            if first_client is not None:
                client_data = getattr(first_client, 'data', None)
                if client_data and 'train' in client_data and hasattr(client_data['train'], 'dataset'):
                    dataset = client_data['train'].dataset
                    self.full_train_dataset = dataset.dataset if hasattr(dataset, 'dataset') else dataset
                elif hasattr(first_client, 'trainer'):
                    trainer = first_client.trainer
                    if trainer and hasattr(trainer, 'ctx') and hasattr(trainer.ctx, 'data'):
                        client_ctx_data = trainer.ctx.data
                        if client_ctx_data and hasattr(client_ctx_data, 'train_data') and client_ctx_data.train_data is not None:
                            train_dataset = client_ctx_data.train_data
                            self.full_train_dataset = train_dataset.dataset if hasattr(train_dataset, 'dataset') else train_dataset
                        elif client_ctx_data and 'train' in client_ctx_data and hasattr(client_ctx_data['train'], 'dataset'):
                            dataset = client_ctx_data['train'].dataset
                            self.full_train_dataset = dataset.dataset if hasattr(dataset, 'dataset') else dataset
        except Exception as e:
            logger.warning(f'GGEURFedMIAServer: failed to get full_train_dataset: {e}')

        self._get_client_indices_from_data(clients)


    def _get_client_indices_from_data(self, clients):
        self.client_data_indices = {}
        self.client_train_datasets = {}
        self.client_refs = {}
        for client_id, client in clients.items():
            self.client_refs[int(client_id)] = client
            try:
                found_indices = None
                found_dataset = None

                client_data = getattr(client, 'data', None)
                if client_data and 'train' in client_data and hasattr(client_data['train'], 'dataset'):
                    train_dataset = client_data['train'].dataset
                    found_dataset = train_dataset
                    if hasattr(train_dataset, 'indices'):
                        found_indices = list(train_dataset.indices)
                    elif hasattr(train_dataset, 'dataset') and hasattr(train_dataset.dataset, 'indices'):
                        found_indices = list(train_dataset.dataset.indices)

                if found_dataset is None:
                    trainer = getattr(client, 'trainer', None)
                    if trainer and hasattr(trainer, 'ctx') and hasattr(trainer.ctx, 'data'):
                        client_ctx_data = trainer.ctx.data
                        if client_ctx_data and hasattr(client_ctx_data, 'train_data') and client_ctx_data.train_data is not None:
                            train_dataset = client_ctx_data.train_data
                            found_dataset = train_dataset
                            if hasattr(train_dataset, 'indices'):
                                found_indices = list(train_dataset.indices)
                        elif client_ctx_data and 'train' in client_ctx_data and hasattr(client_ctx_data['train'], 'dataset'):
                            train_dataset = client_ctx_data['train'].dataset
                            found_dataset = train_dataset
                            if hasattr(train_dataset, 'indices'):
                                found_indices = list(train_dataset.indices)

                if found_dataset is not None:
                    self.client_train_datasets[int(client_id)] = found_dataset
                if found_indices is not None:
                    self.client_data_indices[int(client_id)] = found_indices
            except Exception as e:
                logger.warning(f'GGEURFedMIAServer: failed to get indices for client {client_id}: {e}')
        logger.info(f'GGEURFedMIAServer: collected train datasets for '
                    f'{len(self.client_train_datasets)} clients and indices for '
                    f'{len(self.client_data_indices)} clients')

    def callback_funcs_model_para(self, message):
        round_idx, sender, content = message.state, message.sender, message.content
        self._save_attack_features_if_needed(round_idx, sender, content)
        return super().callback_funcs_model_para(message)

    def _should_save_attack_round(self, round_idx: int) -> bool:
        if self.fedmia_save_interval <= 1:
            return True
        is_interval_round = round_idx % self.fedmia_save_interval == 0
        # FederatedScope variants may count rounds as 0..T-1 or 1..T.
        is_finalish_round = round_idx >= max(0, self.total_round_num - 1)
        return is_interval_round or is_finalish_round

    def _feature_file_path(self, round_idx: int, client_id: int) -> str:
        return os.path.join(
            self.attack_feature_dir,
            f'client_{client_id}_features_round{round_idx}.pt')

    def _to_cpu_state(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            k: v.detach().cpu().clone() if torch.is_tensor(v) else copy.deepcopy(v)
            for k, v in state.items()
        }

    def _extract_protected_delta(self, content):
        if isinstance(content, tuple) and len(content) >= 3:
            return content[2]
        return None

    def _reconstruct_server_visible_state(self, round_num: int, content):
        payload = content[1] if isinstance(content, tuple) and len(content) >= 2 else content
        raw_state = self._extract_classifier_state(payload)
        # The unified client uploads the already protected full state. The
        # optional third tuple item contains public mechanism statistics, not
        # a second model delta, so attacks observe what aggregation observes.
        if raw_state is not None:
            return raw_state

        protected_delta = self._extract_protected_delta(content)
        if protected_delta is None:
            return raw_state

        base_state = self.round_global_classifier_states.get(round_num)
        if base_state is None:
            logger.warning('GGEURFedMIAServer: missing round-%s global classifier '
                           'state, cannot reconstruct protected local model',
                           round_num)
            return raw_state

        delta_state = None
        if isinstance(protected_delta, dict):
            if protected_delta.get('mlp') is not None:
                delta_state = protected_delta['mlp']
            elif protected_delta.get('classifier') is not None:
                delta_state = protected_delta['classifier']
            elif all(torch.is_tensor(v) for v in protected_delta.values()):
                delta_state = protected_delta
        if delta_state is None:
            logger.warning('GGEURFedMIAServer: unsupported protected delta '
                           'structure for round %s', round_num)
            return raw_state

        reconstructed = copy.deepcopy(base_state)
        for key, delta_value in delta_state.items():
            if key not in reconstructed or not torch.is_tensor(delta_value):
                continue
            reconstructed[key] = reconstructed[key].detach().cpu().float() + \
                delta_value.detach().cpu().float()
        return reconstructed

    def _prune_round_global_classifier_states(self, keep_round: int):
        for round_num in list(self.round_global_classifier_states.keys()):
            if round_num < keep_round:
                del self.round_global_classifier_states[round_num]

    def _save_attack_features_if_needed(self, round_idx: int, sender: int,
                                        content):
        if not self._should_save_attack_round(round_idx):
            return
        if self.global_mlp is None:
            logger.debug('GGEURFedMIAServer: global_mlp is None, skip feature save')
            return
        if not self.client_train_datasets:
            logger.debug('GGEURFedMIAServer: no client train datasets, skip feature save')
            return

        if round_idx not in self.round_global_classifier_states:
            self.round_global_classifier_states[round_idx] = self._to_cpu_state(
                self.global_mlp.state_dict())

        classifier_state = self._reconstruct_server_visible_state(round_idx,
                                                                  content)
        if classifier_state is None:
            logger.warning('GGEURFedMIAServer: client %s classifier state missing '
                           'at round %s, skip feature save', sender, round_idx)
            return

        try:
            os.makedirs(self.attack_feature_dir, exist_ok=True)
            local_state = self._to_cpu_state(classifier_state)
            global_state = self.round_global_classifier_states.get(round_idx)
            if global_state is None:
                global_state = self._to_cpu_state(self.global_mlp.state_dict())
            test_features, test_labels = self._get_global_test_feature_arrays()
            feature_data = self._build_client_attack_dict(
                round_idx,
                local_state,
                self.client_train_datasets.get(int(sender)),
                None,
                test_features,
                test_labels,
                int(sender),
                global_state if self.store_grad_cos else None,
            )
            file_path = self._feature_file_path(round_idx, int(sender))
            torch.save({
                'round': int(round_idx),
                'client_id': int(sender),
                'data': feature_data,
            }, file_path)
            self.saved_feature_index.setdefault(int(round_idx), set()).add(int(sender))
            logger.info('GGEURFedMIAServer: saved feature-space FedMIA '
                        'artifact %s', file_path)
        except Exception as e:
            logger.exception('GGEURFedMIAServer: failed to save attack features '
                             'for client %s round %s: %s', sender, round_idx, e)
        finally:
            self._prune_round_global_classifier_states(round_idx)
            del classifier_state
            if 'local_state' in locals():
                del local_state
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _extract_classifier_state(self, payload):
        if payload is None:
            return None
        if isinstance(payload, dict):
            if 'mlp' in payload and payload.get('mlp') is not None:
                return payload['mlp']
            if 'classifier' in payload and payload.get('classifier') is not None:
                return payload['classifier']
            if all(torch.is_tensor(v) for v in payload.values()):
                return payload
        return None

    def _prepare_images(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(self.device)
        if images.ndim == 3:
            images = images.unsqueeze(0)
        if images.ndim == 4 and images.shape[1] not in [1, 3] and images.shape[-1] in [1, 3]:
            images = images.permute(0, 3, 1, 2)
        images = images.float()
        if images.max() > 2.0:
            images = images / 255.0
        if images.shape[-1] != 224 or images.shape[-2] != 224:
            images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
        if float(images.min()) >= 0.0 and float(images.max()) <= 1.0:
            mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
            images = (images - mean) / std
        return images

    def _extract_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        self._load_feature_extractor()
        images = self._prepare_images(images)
        with torch.no_grad():
            if self.feature_extractor_type == 'cnn':
                return self.cnn_extractor(images)
            if self.feature_extractor_type == 'timm':
                return self.timm_extractor(images)
            return self.clip_model.encode_image(images)

    def _empty_eval_result(self) -> Dict:
        return {
            'loss': np.array([]),
            'logit': torch.tensor([]),
            'labels': torch.tensor([]),
            'cos': np.array([]),
            'grad_diff': np.array([]),
            'grad_norm': np.array([]),
        }

    def _build_classifier(self, model_state: Dict[str, torch.Tensor]):
        classifier = copy.deepcopy(self.global_mlp).to(self.device)
        classifier.load_state_dict(model_state, strict=False)
        return classifier

    def _sample_indices(self, dataset, limit: int, seed: int):
        if dataset is None or limit is None or limit <= 0:
            return None
        dataset_len = len(dataset)
        if dataset_len <= limit:
            return None
        rng = np.random.default_rng(seed)
        return sorted(rng.choice(dataset_len, size=limit, replace=False).tolist())

    def _grad_context(self,
                      local_state: Dict[str, torch.Tensor],
                      global_state: Optional[Dict[str, torch.Tensor]]):
        if global_state is None:
            return None, [], None

        classifier = self._build_classifier(global_state)
        classifier.eval()
        # cuDNN only records the reserve space required by RNN backward when
        # the RNN module itself is in training mode. FedMIA-II needs backward
        # to obtain per-sample gradients even though this private model copy is
        # otherwise used for inference. Keep the parent model (including
        # dropout and batch norm), the real training model, and FedMIA-I in
        # eval mode; only RNN-family submodules need training mode here.
        for module in classifier.modules():
            if isinstance(module, (nn.RNN, nn.GRU, nn.LSTM)):
                module.train()
        params, delta_parts = [], []
        for name, param in classifier.named_parameters():
            if name not in global_state or name not in local_state:
                continue
            params.append(param)
            g = global_state[name].detach().to(self.device)
            l = local_state[name].detach().to(self.device)
            delta_parts.append((g - l).flatten())

        if not params or not delta_parts:
            return None, [], None

        update_direction = torch.cat(delta_parts).detach()
        if torch.norm(update_direction, p=2).item() == 0:
            return None, [], None
        return classifier, params, update_direction

    def _grad_features_for_embeddings(self,
                                      embeddings: torch.Tensor,
                                      labels: torch.Tensor,
                                      grad_classifier,
                                      grad_params,
                                      update_direction) -> Dict[str, List[float]]:
        empty = {'cos': [], 'grad_diff': [], 'grad_norm': []}
        if grad_classifier is None or update_direction is None:
            return empty

        model_diff_norm = torch.norm(update_direction, p=2) ** 2
        features = {'cos': [], 'grad_diff': [], 'grad_norm': []}
        for idx in range(embeddings.shape[0]):
            x_i = embeddings[idx:idx + 1].detach()
            y_i = labels[idx:idx + 1]
            grad_classifier.zero_grad(set_to_none=True)
            loss = F.cross_entropy(grad_classifier(x_i), y_i)
            grads = torch.autograd.grad(loss,
                                        grad_params,
                                        retain_graph=False,
                                        allow_unused=True)
            flat_grads = []
            for grad, param in zip(grads, grad_params):
                if grad is None:
                    flat_grads.append(torch.zeros_like(param).flatten())
                else:
                    flat_grads.append(grad.detach().flatten())
            if not flat_grads:
                features['cos'].append(0.0)
                features['grad_diff'].append(0.0)
                features['grad_norm'].append(0.0)
                continue
            sample_grad = torch.cat(flat_grads)
            cos = F.cosine_similarity(sample_grad,
                                      update_direction,
                                      dim=0,
                                      eps=1e-12)
            grad_diff = model_diff_norm - torch.norm(
                update_direction - sample_grad, p=2) ** 2
            grad_norm = torch.norm(sample_grad, p=2) ** 2
            features['cos'].append(float(cos.detach().cpu()))
            features['grad_diff'].append(float(grad_diff.detach().cpu()))
            features['grad_norm'].append(float(grad_norm.detach().cpu()))
        return features

    def _evaluate_feature_arrays(self,
                                 features: np.ndarray,
                                 labels: np.ndarray,
                                 model_state: Dict[str, torch.Tensor],
                                 global_state: Optional[Dict[str, torch.Tensor]] = None,
                                 batch_size: int = 256):
        if features is None or labels is None or len(labels) == 0:
            return self._empty_eval_result()

        classifier = self._build_classifier(model_state)
        classifier.eval()
        criterion = nn.CrossEntropyLoss(reduction='none')
        grad_classifier, grad_params, update_direction = self._grad_context(
            model_state, global_state)

        all_losses, all_logits, all_labels = [], [], []
        all_cos, all_grad_diff, all_grad_norm = [], [], []
        num_samples = len(labels)
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            feat_tensor = torch.from_numpy(features[start:end]).float().to(self.device)
            label_tensor = torch.from_numpy(labels[start:end]).long().to(self.device)
            with torch.no_grad():
                logits = classifier(feat_tensor)
                losses = criterion(logits, label_tensor)
            all_losses.append(losses.cpu().numpy())
            all_logits.append(logits.cpu())
            all_labels.append(label_tensor.cpu())
            if grad_classifier is not None:
                grad_features = self._grad_features_for_embeddings(
                    feat_tensor, label_tensor, grad_classifier,
                    grad_params, update_direction)
                all_cos.extend(grad_features['cos'])
                all_grad_diff.extend(grad_features['grad_diff'])
                all_grad_norm.extend(grad_features['grad_norm'])

        return {
            'loss': np.concatenate(all_losses),
            'logit': torch.cat(all_logits),
            'labels': torch.cat(all_labels),
            'cos': np.asarray(all_cos, dtype=np.float64),
            'grad_diff': np.asarray(all_grad_diff, dtype=np.float64),
            'grad_norm': np.asarray(all_grad_norm, dtype=np.float64),
        }

    def _evaluate_dataset_subset(self,
                                 dataset,
                                 indices: Optional[List[int]],
                                 model_state: Dict[str, torch.Tensor],
                                 batch_size: int = 32,
                                 global_state: Optional[Dict[str, torch.Tensor]] = None):
        if dataset is None:
            return self._empty_eval_result()

        classifier = self._build_classifier(model_state)
        classifier.eval()
        criterion = nn.CrossEntropyLoss(reduction='none')
        grad_classifier, grad_params, update_direction = self._grad_context(
            model_state, global_state)

        eval_dataset = dataset
        if indices:
            eval_dataset = Subset(dataset, indices)

        loader = DataLoader(eval_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=0)

        all_losses, all_logits, all_labels = [], [], []
        all_cos, all_grad_diff, all_grad_norm = [], [], []
        for images, labels in loader:
            embeddings = self._extract_embeddings(images)
            labels = labels.to(self.device)
            with torch.no_grad():
                logits = classifier(embeddings)
                losses = criterion(logits, labels)
            all_losses.append(losses.cpu().numpy())
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            if grad_classifier is not None:
                grad_features = self._grad_features_for_embeddings(
                    embeddings, labels, grad_classifier, grad_params,
                    update_direction)
                all_cos.extend(grad_features['cos'])
                all_grad_diff.extend(grad_features['grad_diff'])
                all_grad_norm.extend(grad_features['grad_norm'])

        if not all_losses:
            return self._empty_eval_result()

        return {
            'loss': np.concatenate(all_losses),
            'logit': torch.cat(all_logits),
            'labels': torch.cat(all_labels),
            'cos': np.asarray(all_cos, dtype=np.float64),
            'grad_diff': np.asarray(all_grad_diff, dtype=np.float64),
            'grad_norm': np.asarray(all_grad_norm, dtype=np.float64),
        }

    def _get_client_feature_arrays(
            self, client_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return one client's already extracted GGEUR feature arrays."""
        client_refs = self.__dict__.get('client_refs', {})
        client = client_refs.get(int(client_id)) if client_refs else None
        local_features = getattr(client, 'local_features', None)
        if not isinstance(local_features, dict) or not local_features:
            embedding_dim = int(getattr(
                self.ggeur_cfg, 'embedding_dim', 0))
            return (
                np.empty((0, embedding_dim), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
            )

        feature_parts = []
        label_parts = []
        for class_id in sorted(local_features.keys(), key=int):
            features = np.asarray(
                local_features[class_id], dtype=np.float32)
            if features.ndim != 2 or len(features) == 0:
                continue
            feature_parts.append(features)
            label_parts.append(np.full(
                len(features), int(class_id), dtype=np.int64))
        if not feature_parts:
            embedding_dim = int(getattr(
                self.ggeur_cfg, 'embedding_dim', 0))
            return (
                np.empty((0, embedding_dim), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
            )
        return np.vstack(feature_parts), np.concatenate(label_parts)

    def _sample_feature_arrays(self, features, labels, limit, seed):
        if features is None or labels is None or len(labels) == 0:
            return features, labels
        limit = int(limit or 0)
        if limit <= 0 or len(labels) <= limit:
            return features, labels
        rng = np.random.default_rng(int(seed))
        selected = rng.choice(len(labels), size=limit, replace=False)
        return features[selected], labels[selected]

    def _merge_eval_results(self, results: List[Dict]) -> Dict:
        valid = [res for res in results if res and len(res.get('loss', [])) > 0]
        if not valid:
            return self._empty_eval_result()
        logits = [res['logit'] for res in valid if res.get('logit') is not None and len(res['logit']) > 0]
        labels = [res['labels'] for res in valid if res.get('labels') is not None and len(res['labels']) > 0]
        cos = [np.asarray(res.get('cos', []), dtype=np.float64) for res in valid]
        grad_diff = [np.asarray(res.get('grad_diff', []), dtype=np.float64) for res in valid]
        grad_norm = [np.asarray(res.get('grad_norm', []), dtype=np.float64) for res in valid]
        cos = [arr for arr in cos if arr.size > 0]
        grad_diff = [arr for arr in grad_diff if arr.size > 0]
        grad_norm = [arr for arr in grad_norm if arr.size > 0]
        return {
            'loss': np.concatenate([np.asarray(res['loss']) for res in valid]),
            'logit': torch.cat(logits) if logits else torch.tensor([]),
            'labels': torch.cat(labels) if labels else torch.tensor([]),
            'cos': np.concatenate(cos) if cos else np.array([]),
            'grad_diff': np.concatenate(grad_diff) if grad_diff else np.array([]),
            'grad_norm': np.concatenate(grad_norm) if grad_norm else np.array([]),
        }

    def _evaluate_mix_datasets(self,
                               excluded_client_id: int,
                               model_state: Dict[str, torch.Tensor],
                               global_state: Optional[Dict[str, torch.Tensor]]):
        mix_length = int(getattr(self._cfg.attack, 'mix_length', 1000))
        if mix_length <= 0:
            return self._empty_eval_result()

        if str(getattr(
                self.ggeur_cfg, 'feature_extractor', '')).lower() == 'bert':
            candidate_ids = [
                client_id
                for client_id in sorted(self.client_refs.keys())
                if int(client_id) != int(excluded_client_id)
            ]
            if not candidate_ids:
                return self._empty_eval_result()
            per_client = max(1, mix_length // len(candidate_ids))
            remaining = mix_length
            results = []
            base_seed = int(getattr(self._cfg, 'seed', 0))
            for client_id in candidate_ids:
                if remaining <= 0:
                    break
                features, labels = self._get_client_feature_arrays(
                    client_id)
                sample_num = min(per_client, remaining, len(labels))
                if sample_num <= 0:
                    continue
                features, labels = self._sample_feature_arrays(
                    features,
                    labels,
                    sample_num,
                    base_seed + int(client_id),
                )
                results.append(self._evaluate_feature_arrays(
                    features,
                    labels,
                    model_state,
                    global_state=global_state,
                ))
                remaining -= len(labels)
            return self._merge_eval_results(results)

        candidate_ids = [
            cid for cid in sorted(self.client_train_datasets.keys())
            if cid != excluded_client_id
        ]
        if not candidate_ids:
            return self._empty_eval_result()

        per_client = max(1, mix_length // len(candidate_ids))
        remaining = mix_length
        results = []
        for cid in candidate_ids:
            if remaining <= 0:
                break
            dataset = self.client_train_datasets.get(cid)
            if dataset is None or len(dataset) == 0:
                continue
            sample_num = min(per_client, remaining, len(dataset))
            indices = self._sample_indices(
                dataset, sample_num, int(getattr(self._cfg, 'seed', 0)) + cid)
            if indices is None and len(dataset) > sample_num:
                indices = list(range(sample_num))
            results.append(self._evaluate_dataset_subset(
                dataset, indices, model_state, global_state=global_state))
            remaining -= sample_num

        return self._merge_eval_results(results)

    def _get_global_test_feature_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        self._load_test_data_and_features()
        if not self.test_features:
            return np.empty((0, getattr(self.ggeur_cfg, 'embedding_dim', 1024))), np.empty((0,), dtype=np.int64)
        features = np.vstack([self.test_features[d] for d in sorted(self.test_features.keys())])
        labels = np.concatenate([self.test_labels[d] for d in sorted(self.test_labels.keys())])
        return features, labels

    def _evaluate_image_aug_member_samples(self,
                                           client_id: int,
                                           model_state: Dict[str, torch.Tensor],
                                           global_state: Optional[Dict[str, torch.Tensor]]):
        if not bool(getattr(self._cfg.attack, 'fedmia_use_image_aug_member', False)):
            return self._empty_eval_result()

        client_refs = self.__dict__.get('client_refs', {})
        client = client_refs.get(int(client_id)) if client_refs else None
        if client is None:
            return self._empty_eval_result()
        features = getattr(client, 'image_aug_member_features', None)
        labels = getattr(client, 'image_aug_member_labels', None)
        if features is None or labels is None:
            return self._empty_eval_result()

        features = np.asarray(features)
        labels = np.asarray(labels)
        if features.size == 0 or labels.size == 0:
            return self._empty_eval_result()

        max_aug = int(getattr(self._cfg.attack, 'fedmia_image_aug_member_size', 0))
        selected = np.arange(len(labels))
        if max_aug > 0 and selected.size > max_aug:
            rng = np.random.default_rng(
                int(getattr(self._cfg, 'seed', 0)) + 7919 + int(client_id))
            selected = rng.choice(selected, size=max_aug, replace=False)

        return self._evaluate_feature_arrays(features[selected], labels[selected],
                                             model_state,
                                             global_state=global_state)

    def _evaluate_generated_augmented_samples(self,
                                             client_id: int,
                                             model_state: Dict[str, torch.Tensor],
                                             global_state: Optional[Dict[str, torch.Tensor]]):
        if not bool(getattr(self._cfg.attack, 'fedmia_use_augmented_nonmember', False)):
            return self._empty_eval_result()

        client_refs = self.__dict__.get('client_refs', {})
        client = client_refs.get(int(client_id)) if client_refs else None
        if client is None:
            return self._empty_eval_result()
        features = getattr(client, 'augmented_features', None)
        labels = getattr(client, 'augmented_labels', None)
        generated_mask = getattr(client, 'augmented_generated_mask', None)
        if features is None or labels is None or generated_mask is None:
            return self._empty_eval_result()

        generated_mask = np.asarray(generated_mask, dtype=bool)
        if generated_mask.size == 0 or not np.any(generated_mask):
            return self._empty_eval_result()

        features = np.asarray(features)
        labels = np.asarray(labels)
        max_aug = int(getattr(self._cfg.attack, 'fedmia_augmented_nonmember_size', 0))
        selected = np.where(generated_mask)[0]
        if max_aug > 0 and selected.size > max_aug:
            rng = np.random.default_rng(int(getattr(self._cfg, 'seed', 0)) + int(client_id))
            selected = rng.choice(selected, size=max_aug, replace=False)

        return self._evaluate_feature_arrays(features[selected], labels[selected],
                                             model_state,
                                             global_state=global_state)

    def _build_client_attack_dict(self,
                                  round_num: int,
                                  model_state: Dict[str, torch.Tensor],
                                  train_dataset,
                                  train_indices: Optional[List[int]],
                                  test_features: np.ndarray,
                                  test_labels: np.ndarray,
                                  client_id: int,
                                  global_state: Optional[Dict[str, torch.Tensor]] = None) -> Dict:
        target_size = int(getattr(
            self._cfg.attack, 'ggeur_target_size', 0))
        if str(getattr(
                self.ggeur_cfg, 'feature_extractor', '')).lower() == 'bert':
            train_features, train_labels = (
                self._get_client_feature_arrays(client_id)
            )
            train_features, train_labels = self._sample_feature_arrays(
                train_features,
                train_labels,
                target_size,
                int(getattr(self._cfg, 'seed', 0)) + int(client_id),
            )
            train_res = self._evaluate_feature_arrays(
                train_features,
                train_labels,
                model_state,
                global_state=global_state,
            )
        else:
            if train_indices is None:
                train_indices = self._sample_indices(
                    train_dataset,
                    target_size,
                    int(getattr(self._cfg, 'seed', 0)) + int(client_id),
                )
            train_res = self._evaluate_dataset_subset(
                train_dataset,
                train_indices,
                model_state,
                global_state=global_state,
            )
        test_res = self._evaluate_feature_arrays(test_features,
                                                 test_labels,
                                                 model_state,
                                                 global_state=global_state)
        mix_res = self._evaluate_mix_datasets(client_id,
                                              model_state,
                                              global_state)
        augmented_res = self._evaluate_generated_augmented_samples(
            client_id, model_state, global_state)
        image_aug_res = self._evaluate_image_aug_member_samples(
            client_id, model_state, global_state)
        return {
            'train_losses': {round_num: np.asarray(train_res['loss'], dtype=np.float64)},
            'train_logit': {round_num: train_res['logit']},
            'train_labels': {round_num: train_res['labels']},
            'train_cos': {round_num: np.asarray(train_res['cos'], dtype=np.float64)},
            'train_grad_diff': {round_num: np.asarray(train_res['grad_diff'], dtype=np.float64)},
            'train_grad_norm': {round_num: np.asarray(train_res['grad_norm'], dtype=np.float64)},
            'image_aug_losses': {round_num: np.asarray(image_aug_res['loss'], dtype=np.float64)},
            'image_aug_logit': {round_num: image_aug_res['logit']},
            'image_aug_labels': {round_num: image_aug_res['labels']},
            'image_aug_cos': {round_num: np.asarray(image_aug_res['cos'], dtype=np.float64)},
            'image_aug_grad_diff': {round_num: np.asarray(image_aug_res['grad_diff'], dtype=np.float64)},
            'image_aug_grad_norm': {round_num: np.asarray(image_aug_res['grad_norm'], dtype=np.float64)},
            'test_losses': {round_num: np.asarray(test_res['loss'], dtype=np.float64)},
            'test_logit': {round_num: test_res['logit']},
            'test_labels': {round_num: test_res['labels']},
            'test_cos': {round_num: np.asarray(test_res['cos'], dtype=np.float64)},
            'test_grad_diff': {round_num: np.asarray(test_res['grad_diff'], dtype=np.float64)},
            'test_grad_norm': {round_num: np.asarray(test_res['grad_norm'], dtype=np.float64)},
            'mix_losses': {round_num: np.asarray(mix_res['loss'], dtype=np.float64)},
            'mix_logit': {round_num: mix_res['logit']},
            'mix_labels': {round_num: mix_res['labels']},
            'mix_cos': {round_num: np.asarray(mix_res['cos'], dtype=np.float64)},
            'mix_grad_diff': {round_num: np.asarray(mix_res['grad_diff'], dtype=np.float64)},
            'mix_grad_norm': {round_num: np.asarray(mix_res['grad_norm'], dtype=np.float64)},
            'augmented_losses': {round_num: np.asarray(augmented_res['loss'], dtype=np.float64)},
            'augmented_logit': {round_num: augmented_res['logit']},
            'augmented_labels': {round_num: augmented_res['labels']},
            'augmented_cos': {round_num: np.asarray(augmented_res['cos'], dtype=np.float64)},
            'augmented_grad_diff': {round_num: np.asarray(augmented_res['grad_diff'], dtype=np.float64)},
            'augmented_grad_norm': {round_num: np.asarray(augmented_res['grad_norm'], dtype=np.float64)},
        }

    def _empty_attack_data(self) -> Dict:
        return {
            'train_losses': {},
            'train_logit': {},
            'train_labels': {},
            'train_cos': {},
            'train_grad_diff': {},
            'train_grad_norm': {},
            'image_aug_losses': {},
            'image_aug_logit': {},
            'image_aug_labels': {},
            'image_aug_cos': {},
            'image_aug_grad_diff': {},
            'image_aug_grad_norm': {},
            'test_losses': {},
            'test_logit': {},
            'test_labels': {},
            'test_cos': {},
            'test_grad_diff': {},
            'test_grad_norm': {},
            'mix_losses': {},
            'mix_logit': {},
            'mix_labels': {},
            'mix_cos': {},
            'mix_grad_diff': {},
            'mix_grad_norm': {},
            'augmented_losses': {},
            'augmented_logit': {},
            'augmented_labels': {},
            'augmented_cos': {},
            'augmented_grad_diff': {},
            'augmented_grad_norm': {},
        }

    def _merge_attack_data(self, dst: Dict, src: Dict):
        for key, value in src.items():
            if isinstance(value, dict):
                dst.setdefault(key, {}).update(value)
            else:
                dst[key] = value

    def _truncate_feature_value(self, value, max_len: int):
        if value is None:
            return value
        if torch.is_tensor(value):
            return value[:max_len]
        arr = np.asarray(value)
        return arr[:max_len]

    def _align_train_feature_lengths(self, per_client: Dict[int, Dict]):
        shadow_stat_mode = str(getattr(
            self._cfg.attack,
            'fedmia_shadow_stat_mode',
            'indexed',
        )).lower()
        if shadow_stat_mode == 'global':
            # Global shadow calibration pools all shadow scores into one
            # distribution and therefore does not require index alignment.
            # Keeping native lengths is essential for highly heterogeneous
            # text clients, where the smallest client may contain only a few
            # samples while the target contains hundreds.
            return

        rounds = set()
        for data in per_client.values():
            rounds.update(data.get('train_losses', {}).keys())

        for round_num in sorted(rounds):
            lengths = []
            for data in per_client.values():
                values = data.get('train_losses', {})
                if round_num not in values:
                    continue
                arr = np.asarray(values[round_num])
                if arr.size > 0:
                    lengths.append(len(arr))
            if len(lengths) < 2:
                continue

            min_len = min(lengths)
            for data in per_client.values():
                for key in ['train_losses', 'train_logit', 'train_labels',
                            'train_cos', 'train_grad_diff', 'train_grad_norm']:
                    values = data.get(key, {})
                    if round_num in values and values[round_num] is not None:
                        values[round_num] = self._truncate_feature_value(
                            values[round_num], min_len)

    def _load_saved_attack_features(self) -> Dict[int, Dict]:
        per_client: Dict[int, Dict] = {}
        if not os.path.isdir(self.attack_feature_dir):
            return per_client

        for filename in sorted(os.listdir(self.attack_feature_dir)):
            if not filename.endswith('.pt'):
                continue
            file_path = os.path.join(self.attack_feature_dir, filename)
            try:
                artifact = torch.load(file_path, map_location='cpu', weights_only=False)
                client_id = int(artifact['client_id'])
                data = artifact.get('data', {})
                if client_id not in per_client:
                    per_client[client_id] = self._empty_attack_data()
                self._merge_attack_data(per_client[client_id], data)
                per_client[client_id]['_fedmia_client_id'] = client_id
                per_client[client_id]['_fedmia_base_seed'] = int(getattr(self._cfg, 'seed', 0))
            except Exception as e:
                logger.warning('GGEURFedMIAServer: failed to load %s: %s',
                               file_path, e)
        self._align_train_feature_lengths(per_client)
        return per_client

    def _run_all_client_fedmia_attacks(self, per_client_data: Dict[int, Dict]):
        if not bool(getattr(self._cfg.attack, 'fedmia_compute_all_clients', False)):
            return
        if not per_client_data:
            return

        from federatedscope.contrib.attack.orchestrator import AttackOrchestrator

        all_client_orchestrator = AttackOrchestrator(
            het_handler=self.het_handler,
            attack_names=['fedmia_i', 'fedmia_ii'],
            config=self._cfg,
        )

        all_results = {}
        csv_rows = []
        for client_id, target_data in sorted(per_client_data.items()):
            shadow_data = [
                data for cid, data in sorted(per_client_data.items())
                if cid != client_id
            ]
            if not shadow_data:
                logger.warning('GGEURFedMIAServer: skip all-client FedMIA for '
                               'client %s because no shadow data exists',
                               client_id)
                continue

            logger.info('GGEURFedMIAServer: running all-client FedMIA for '
                        'client %s with %s shadow clients',
                        client_id, len(shadow_data))
            results = all_client_orchestrator.run_all_attacks(
                target_data=target_data,
                shadow_data=shadow_data,
                global_model=self.global_mlp,
            )
            all_results[str(client_id)] = results

            fedmia_i = results.get('fedmia_i', {})
            fedmia_ii = results.get('fedmia_ii', {})
            csv_rows.append({
                'client_id': int(client_id),
                'fedmia_i_auc': fedmia_i.get('auc'),
                'fedmia_i_tpr_at_0.01': fedmia_i.get('tpr_at_fpr', {}).get(0.01),
                'fedmia_i_members': fedmia_i.get('num_members'),
                'fedmia_i_nonmembers': fedmia_i.get('num_nonmembers'),
                'fedmia_ii_auc': fedmia_ii.get('auc'),
                'fedmia_ii_tpr_at_0.01': fedmia_ii.get('tpr_at_fpr', {}).get(0.01),
                'fedmia_ii_members': fedmia_ii.get('num_members'),
                'fedmia_ii_nonmembers': fedmia_ii.get('num_nonmembers'),
            })

        payload = {
            'metadata': {
                'source_feature_dir': self.attack_feature_dir,
                'clients': sorted(int(cid) for cid in per_client_data.keys()),
                'attack_plugins': ['fedmia_i', 'fedmia_ii'],
                'mode': getattr(self._cfg.attack, 'mode', 'mix'),
                'mix_length': int(getattr(self._cfg.attack, 'mix_length', 1000)),
                'fedmia_i_round_agg': getattr(
                    self._cfg.attack, 'fedmia_i_round_agg', 'mean'),
                'fedmia_shadow_stat_mode': getattr(
                    self._cfg.attack, 'fedmia_shadow_stat_mode', 'indexed'),
                'fedmia_var_floor': float(getattr(
                    self._cfg.attack, 'fedmia_var_floor', 1e-8)),
                'fedmia_use_augmented_nonmember': bool(getattr(
                    self._cfg.attack, 'fedmia_use_augmented_nonmember', False)),
            },
            'clients': all_results,
        }

        save_path = os.path.join(
            self._cfg.outdir, 'all_clients_fedmia_i_ii_results.json')
        os.makedirs(self._cfg.outdir, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(payload, f, indent=2, default=str)

        csv_path = os.path.join(
            self._cfg.outdir, 'all_clients_fedmia_i_ii_summary.csv')
        csv_fields = [
            'client_id',
            'fedmia_i_auc', 'fedmia_i_tpr_at_0.01',
            'fedmia_i_members', 'fedmia_i_nonmembers',
            'fedmia_ii_auc', 'fedmia_ii_tpr_at_0.01',
            'fedmia_ii_members', 'fedmia_ii_nonmembers',
        ]
        with open(csv_path, 'w') as f:
            f.write(','.join(csv_fields) + '\n')
            for row in csv_rows:
                f.write(','.join(str(row.get(field, '')) for field in csv_fields) + '\n')

        logger.info('Saved all-client FedMIA-I/II results to %s', save_path)
        logger.info('Saved all-client FedMIA-I/II summary to %s', csv_path)

    def _run_modular_attacks(self):
        if self.orchestrator is None:
            logger.warning('GGEURFedMIAServer: no orchestrator configured')
            return
        if self.global_mlp is None:
            logger.warning('GGEURFedMIAServer: global_mlp is None, skip attack')
            return

        per_client_data = self._load_saved_attack_features()
        if not per_client_data:
            logger.warning('GGEURFedMIAServer: no saved feature-space FedMIA '
                           'artifacts found in %s', self.attack_feature_dir)
            return

        target_data = per_client_data.get(self.target_client_id)
        if target_data is None:
            logger.warning('GGEURFedMIAServer: target client %s has no saved '
                           'attack features', self.target_client_id)
            return

        shadow_data = [
            data for cid, data in sorted(per_client_data.items())
            if cid != self.target_client_id
        ]
        if not shadow_data:
            logger.warning('GGEURFedMIAServer: no shadow client attack features found')

        results = self.orchestrator.run_all_attacks(
            target_data=target_data,
            shadow_data=shadow_data,
            global_model=self.global_mlp,
        )

        save_path = os.path.join(self._cfg.outdir, 'ggeur_fedmia_results.json')
        os.makedirs(self._cfg.outdir, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info('=' * 20 + ' GGEUR FedMIA Results ' + '=' * 20)
        for name, metrics in results.items():
            tpr_values = metrics.get('tpr_at_fpr', {}) or {}
            tpr = tpr_values.get(0.01, tpr_values.get('0.01'))
            logger.info('%s: AUC=%s, TPR@0.01=%s',
                        name,
                        metrics.get('auc'),
                        tpr)
            metric_payload = {
                'round': int(getattr(self, 'state', 0)),
                'attackName': str(name),
            }
            if metrics.get('auc') is not None:
                metric_payload['privacyRisk'] = float(metrics['auc'])
            if tpr is not None:
                metric_payload['truePositiveRate'] = float(tpr)
            emit_training_event('metric.updated', **metric_payload)
        logger.info('Saved GGEUR FedMIA results to %s', save_path)

        self._run_all_client_fedmia_attacks(per_client_data)

    def _finish(self):
        if not self._attack_executed:
            self._attack_executed = True
            try:
                self._run_modular_attacks()
            except Exception as e:
                logger.exception('GGEURFedMIAServer: final attack failed: %s', e)
        super()._finish()


class GGEURFedMIAHook(GGEURFedMIAServer):
    """Side-effect-only FedMIA hook attached to the standard GGEURServer.

    The hook reuses the existing feature extraction and final attack helpers,
    but it is not selected as the FL server class. GGEUR aggregation, round
    progression, statistics collection, and finish handling stay owned by
    GGEURServer.
    """

    def __init__(self, server):
        self.server = server
        self._cfg = server._cfg
        self.ggeur_cfg = server.ggeur_cfg
        self.device = server.device
        self.clients = {}

        self.target_client_id = int(getattr(self._cfg.attack,
                                            'fedmia_target_client_id', 1))
        self.full_train_dataset = None
        self.client_train_datasets = {}
        self.client_data_indices = {}
        self.client_refs = {}
        self._attack_executed = False
        self.attack_feature_dir = os.path.join(self._cfg.outdir,
                                               'ggeur_fedmia_features')
        self.saved_feature_index = {}
        self.round_global_classifier_states = {}
        self.fedmia_save_interval = int(getattr(
            self._cfg.attack, 'fedmia_save_interval',
            getattr(self._cfg.attack, 'save_interval', 5)))
        self.store_grad_cos = bool(getattr(
            self._cfg.attack, 'fedmia_store_grad_cos', True))

        self.het_handler = None
        if getattr(self._cfg.attack, 'use_ggeur', False):
            from federatedscope.contrib.attack.heterogeneity import GGEURHandler
            self.het_handler = GGEURHandler(self._cfg)

        self.orchestrator = None
        if getattr(self._cfg.attack, 'modular_attacks', False):
            import federatedscope.contrib.attack.plugins.blackbox_loss  # noqa: F401
            import federatedscope.contrib.attack.plugins.grad_cosine  # noqa: F401
            import federatedscope.contrib.attack.plugins.grad_diff  # noqa: F401
            import federatedscope.contrib.attack.plugins.grad_norm  # noqa: F401
            import federatedscope.contrib.attack.plugins.loss_series  # noqa: F401
            import federatedscope.contrib.attack.plugins.avg_cosine  # noqa: F401
            import federatedscope.contrib.attack.plugins.fedmia_i  # noqa: F401
            import federatedscope.contrib.attack.plugins.fedmia_ii  # noqa: F401
            from federatedscope.contrib.attack.orchestrator import AttackOrchestrator

            attack_names = list(getattr(self._cfg.attack, 'attack_plugins', []))
            self.orchestrator = AttackOrchestrator(
                het_handler=self.het_handler,
                attack_names=attack_names if attack_names else None,
                config=self._cfg,
            )
            logger.info('GGEURFedMIAHook: modular loss-based FedMIA ready '
                        f'with plugins {attack_names}')

    def __getattr__(self, name):
        return getattr(self.server, name)

    def _load_test_data_and_features(self):
        return self.server._load_test_data_and_features()

    def _load_feature_extractor(self):
        return self.server._load_feature_extractor()

    def _finish(self):  # pragma: no cover - kept to avoid accidental super flow
        self.finish()

    def link_clients(self, clients):
        self.clients = clients
        self.client_refs = {int(client_id): client
                            for client_id, client in clients.items()} if clients else {}
        if not clients:
            return

        try:
            first_client = clients.get(1) if 1 in clients else list(clients.values())[0]
            if first_client is not None:
                client_data = getattr(first_client, 'data', None)
                if client_data and 'train' in client_data and hasattr(client_data['train'], 'dataset'):
                    dataset = client_data['train'].dataset
                    self.full_train_dataset = dataset.dataset if hasattr(dataset, 'dataset') else dataset
                elif hasattr(first_client, 'trainer'):
                    trainer = first_client.trainer
                    if trainer and hasattr(trainer, 'ctx') and hasattr(trainer.ctx, 'data'):
                        client_ctx_data = trainer.ctx.data
                        if client_ctx_data and hasattr(client_ctx_data, 'train_data') and client_ctx_data.train_data is not None:
                            train_dataset = client_ctx_data.train_data
                            self.full_train_dataset = train_dataset.dataset if hasattr(train_dataset, 'dataset') else train_dataset
                        elif client_ctx_data and 'train' in client_ctx_data and hasattr(client_ctx_data['train'], 'dataset'):
                            dataset = client_ctx_data['train'].dataset
                            self.full_train_dataset = dataset.dataset if hasattr(dataset, 'dataset') else dataset
        except Exception as e:
            logger.warning(f'GGEURFedMIAHook: failed to get full_train_dataset: {e}')

        for client_id, client in clients.items():
            try:
                found_dataset = None
                found_indices = None
                client_data = getattr(client, 'data', None)
                if client_data and 'train' in client_data and hasattr(client_data['train'], 'dataset'):
                    train_dataset = client_data['train'].dataset
                    found_dataset = train_dataset
                    if hasattr(train_dataset, 'indices'):
                        found_indices = list(train_dataset.indices)
                elif hasattr(client, 'trainer'):
                    trainer = client.trainer
                    if trainer and hasattr(trainer, 'ctx') and hasattr(trainer.ctx, 'data'):
                        client_ctx_data = trainer.ctx.data
                        if client_ctx_data and hasattr(client_ctx_data, 'train_data') and client_ctx_data.train_data is not None:
                            train_dataset = client_ctx_data.train_data
                            found_dataset = train_dataset
                            if hasattr(train_dataset, 'indices'):
                                found_indices = list(train_dataset.indices)
                        elif client_ctx_data and 'train' in client_ctx_data and hasattr(client_ctx_data['train'], 'dataset'):
                            train_dataset = client_ctx_data['train'].dataset
                            found_dataset = train_dataset
                            if hasattr(train_dataset, 'indices'):
                                found_indices = list(train_dataset.indices)

                if found_dataset is not None:
                    self.client_train_datasets[int(client_id)] = found_dataset
                if found_indices is not None:
                    self.client_data_indices[int(client_id)] = found_indices
            except Exception as e:
                logger.warning(f'GGEURFedMIAHook: failed to get indices for client {client_id}: {e}')
        logger.info(f'GGEURFedMIAHook: collected train datasets for '
                    f'{len(self.client_train_datasets)} clients and indices for '
                    f'{len(self.client_data_indices)} clients')

    def after_model_para(self, message):
        statistics_round = int(getattr(self.ggeur_cfg, 'statistics_round', 0))
        if int(message.state) == statistics_round:
            return
        self._save_attack_features_if_needed(
            message.state, message.sender, message.content)

    def finish(self):
        if self._attack_executed:
            return
        self._attack_executed = True
        try:
            self._run_modular_attacks()
        except Exception as e:
            logger.exception('GGEURFedMIAHook: final attack failed: %s', e)
