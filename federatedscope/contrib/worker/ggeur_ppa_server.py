"""
GGEUR-native property inference hook.

This module is intentionally independent from the GGEUR FedMIA hook.  It
collects only the GGEUR MLP states and client property labels needed by the
Meta-PPA plugin, then runs the modular attack orchestrator with ``meta_ppa``.
"""

from __future__ import annotations

import copy
import json
import logging
import os
from collections import Counter
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class GGEURPPAHook:
    """Side-effect-only property inference hook for the standard GGEURServer."""

    def __init__(self, server):
        self.server = server
        self._cfg = server._cfg
        self.ggeur_cfg = server.ggeur_cfg
        self.device = server.device
        self.clients = {}
        self.client_train_datasets = {}
        self.client_majority_class = {}
        self.ppa_probe_features_by_class = {}
        self.round_global_classifier_states = {}
        self._attack_executed = False
        self.attack_feature_dir = os.path.join(self._cfg.outdir,
                                               'ggeur_ppa_features')
        self.save_interval = int(getattr(
            self._cfg.attack, 'meta_ppa_save_interval',
            getattr(self._cfg.attack, 'fedmia_save_interval',
                    getattr(self._cfg.attack, 'save_interval', 5))))

        import federatedscope.contrib.attack.plugins.property_inference  # noqa: F401
        from federatedscope.contrib.attack.orchestrator import AttackOrchestrator
        self.orchestrator = AttackOrchestrator(
            het_handler=None,
            attack_names=['meta_ppa'],
            config=self._cfg,
        )
        logger.info('GGEURPPAHook: modular property inference ready')

    def __getattr__(self, name):
        return getattr(self.server, name)

    def _load_feature_extractor(self):
        return self.server._load_feature_extractor()

    def link_clients(self, clients):
        self.clients = clients
        self.client_train_datasets = {}
        self.client_majority_class = {}
        self.ppa_probe_features_by_class = {}
        if not clients:
            return

        for client_id, client in clients.items():
            dataset = self._find_client_train_dataset(client)
            if dataset is None:
                continue
            self.client_train_datasets[int(client_id)] = dataset
            majority = self._get_dataset_majority_class(dataset)
            if majority is not None:
                self.client_majority_class[int(client_id)] = majority

        logger.info('GGEURPPAHook: collected train datasets for %d clients '
                    'and labels for %d clients',
                    len(self.client_train_datasets),
                    len(self.client_majority_class))

    def after_model_para(self, message):
        statistics_round = int(getattr(self.ggeur_cfg, 'statistics_round', 0))
        if int(message.state) == statistics_round:
            return
        self._save_attack_state_if_needed(message.state, message.sender,
                                          message.content)

    def finish(self):
        if self._attack_executed:
            return
        self._attack_executed = True
        try:
            self._run_property_attack()
        except Exception as e:
            logger.exception('GGEURPPAHook: final property attack failed: %s', e)

    def _find_client_train_dataset(self, client):
        client_data = getattr(client, 'data', None)
        if client_data and 'train' in client_data and \
                hasattr(client_data['train'], 'dataset'):
            return client_data['train'].dataset

        trainer = getattr(client, 'trainer', None)
        if trainer and hasattr(trainer, 'ctx') and hasattr(trainer.ctx, 'data'):
            ctx_data = trainer.ctx.data
            if ctx_data and hasattr(ctx_data, 'train_data') and \
                    ctx_data.train_data is not None:
                return ctx_data.train_data
            if ctx_data and 'train' in ctx_data and \
                    hasattr(ctx_data['train'], 'dataset'):
                return ctx_data['train'].dataset
        return None

    def _get_label_from_item(self, item):
        if not isinstance(item, (tuple, list)) or len(item) < 2:
            return None
        label = item[1]
        if torch.is_tensor(label):
            return int(label.item())
        return int(label)

    def _get_dataset_majority_class(self, dataset):
        counter = Counter()
        try:
            for idx in range(len(dataset)):
                label = self._get_label_from_item(dataset[idx])
                if label is not None:
                    counter[label] += 1
        except Exception as e:
            logger.warning('GGEURPPAHook: failed to infer majority class: %s', e)
            return None
        if not counter:
            return None
        return int(counter.most_common(1)[0][0])

    def _should_save_round(self, round_idx: int) -> bool:
        if self.save_interval <= 1:
            return True
        is_interval_round = round_idx % self.save_interval == 0
        is_finalish_round = round_idx >= max(0, self.total_round_num - 1)
        return is_interval_round or is_finalish_round

    def _feature_file_path(self, round_idx: int, client_id: int) -> str:
        return os.path.join(
            self.attack_feature_dir,
            f'client_{client_id}_ppa_round{round_idx}.pt')

    def _to_cpu_state(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            k: v.detach().cpu().clone() if torch.is_tensor(v)
            else copy.deepcopy(v)
            for k, v in state.items()
        }

    def _extract_classifier_state(self, payload):
        if payload is None:
            return None
        if isinstance(payload, dict):
            if payload.get('mlp') is not None:
                return payload['mlp']
            if payload.get('classifier') is not None:
                return payload['classifier']
            if all(torch.is_tensor(v) for v in payload.values()):
                return payload
        return None

    def _extract_protected_delta(self, content):
        if isinstance(content, tuple) and len(content) >= 3:
            return content[2]
        return None

    def _reconstruct_server_visible_state(self, round_num: int, content):
        payload = content[1] if isinstance(content, tuple) and len(content) >= 2 else content
        raw_state = self._extract_classifier_state(payload)
        # Position 1 is already protected locally; position 2 is public DP
        # metadata in the unified message contract.
        if raw_state is not None:
            return raw_state

        protected_delta = self._extract_protected_delta(content)
        if protected_delta is None:
            return raw_state

        base_state = self.round_global_classifier_states.get(round_num)
        if base_state is None:
            logger.warning('GGEURPPAHook: missing round-%s global MLP state',
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
            logger.warning('GGEURPPAHook: unsupported protected delta for round %s',
                           round_num)
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

    def _save_attack_state_if_needed(self, round_idx: int, sender: int, content):
        if not self._should_save_round(round_idx):
            return
        if self.global_mlp is None:
            logger.debug('GGEURPPAHook: global_mlp is None, skip save')
            return
        if not self.client_train_datasets:
            logger.debug('GGEURPPAHook: no client train datasets, skip save')
            return

        if round_idx not in self.round_global_classifier_states:
            self.round_global_classifier_states[round_idx] = self._to_cpu_state(
                self.global_mlp.state_dict())

        local_state = self._reconstruct_server_visible_state(round_idx, content)
        if local_state is None:
            logger.warning('GGEURPPAHook: missing client %s MLP state at round %s',
                           sender, round_idx)
            return

        try:
            os.makedirs(self.attack_feature_dir, exist_ok=True)
            global_state = self.round_global_classifier_states[round_idx]
            data = {
                'client_id': int(sender),
                'property_label': self.client_majority_class.get(int(sender)),
                'ppa_local_state': {int(round_idx): self._to_cpu_state(local_state)},
                'ppa_global_state': {int(round_idx): self._to_cpu_state(global_state)},
                'ppa_probe_features_by_class': self._ensure_probe_features(),
            }
            torch.save({
                'round': int(round_idx),
                'client_id': int(sender),
                'data': data,
            }, self._feature_file_path(round_idx, int(sender)))
            logger.info('GGEURPPAHook: saved property artifact for client %s '
                        'round %s', sender, round_idx)
        except Exception as e:
            logger.exception('GGEURPPAHook: failed to save PPA state for client '
                             '%s round %s: %s', sender, round_idx, e)
        finally:
            self._prune_round_global_classifier_states(round_idx)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _prepare_images(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(self.device)
        if images.ndim == 3:
            images = images.unsqueeze(0)
        if images.ndim == 4 and images.shape[1] not in [1, 3] and \
                images.shape[-1] in [1, 3]:
            images = images.permute(0, 3, 1, 2)
        images = images.float()
        if images.max() > 2.0:
            images = images / 255.0
        if images.shape[-1] != 224 or images.shape[-2] != 224:
            images = F.interpolate(images, size=(224, 224), mode='bilinear',
                                   align_corners=False)
        if float(images.min()) >= 0.0 and float(images.max()) <= 1.0:
            mean = torch.tensor([0.485, 0.456, 0.406],
                                device=self.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225],
                               device=self.device).view(1, 3, 1, 1)
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

    def _ensure_probe_features(self):
        if self.ppa_probe_features_by_class:
            return self.ppa_probe_features_by_class

        num_classes = int(getattr(self._cfg.model, 'num_classes', 0))
        target_per_class = int(getattr(
            self._cfg.attack, 'meta_ppa_probe_samples_per_class', 8))
        target_per_class = max(target_per_class, 1)
        # Text clients already hold frozen BERT embeddings grouped by class.
        # Reuse them directly; the existing image probe path below is unchanged.
        if str(getattr(
                self.ggeur_cfg, 'feature_extractor', '')).lower() == 'bert':
            samples = {
                class_idx: []
                for class_idx in range(max(num_classes, 1))
            }
            for client in self.clients.values():
                local_features = getattr(client, 'local_features', None)
                if not isinstance(local_features, dict):
                    continue
                for class_idx, features in local_features.items():
                    class_idx = int(class_idx)
                    samples.setdefault(class_idx, [])
                    remaining = target_per_class - len(samples[class_idx])
                    if remaining <= 0:
                        continue
                    array = np.asarray(features, dtype=np.float32)
                    if array.ndim != 2 or len(array) == 0:
                        continue
                    samples[class_idx].extend(array[:remaining])
                if all(len(values) >= target_per_class
                       for values in samples.values()):
                    break

            self.ppa_probe_features_by_class = {
                int(class_idx): np.asarray(features, dtype=np.float32)
                for class_idx, features in samples.items()
                if len(features) > 0
            }
            logger.info(
                'GGEURPPAHook: built BERT probe features with counts %s',
                {key: len(value) for key, value in
                 self.ppa_probe_features_by_class.items()})
            return self.ppa_probe_features_by_class
        samples = {class_idx: [] for class_idx in range(max(num_classes, 1))}

        for dataset in self.client_train_datasets.values():
            for idx in range(len(dataset)):
                item = dataset[idx]
                label = self._get_label_from_item(item)
                if label is None:
                    continue
                samples.setdefault(label, [])
                if len(samples[label]) >= target_per_class:
                    continue
                samples[label].append(item[0])
                if all(len(v) >= target_per_class for v in samples.values()):
                    break
            if all(len(v) >= target_per_class for v in samples.values()):
                break

        probe_features = {}
        batch_size = int(getattr(self._cfg.attack,
                                 'meta_ppa_probe_batch_size', 16))
        for class_idx, images in samples.items():
            if not images:
                continue
            embeddings = []
            for start in range(0, len(images), batch_size):
                batch_images = images[start:start + batch_size]
                try:
                    tensor_images = torch.stack([
                        image if torch.is_tensor(image)
                        else torch.from_numpy(np.asarray(image))
                        for image in batch_images
                    ])
                except Exception as e:
                    logger.warning('GGEURPPAHook: failed to stack probe images '
                                   'for class %s: %s', class_idx, e)
                    continue
                emb = self._extract_embeddings(tensor_images).detach().cpu().numpy()
                embeddings.append(emb)
            if embeddings:
                probe_features[int(class_idx)] = np.vstack(embeddings).astype(np.float32)

        self.ppa_probe_features_by_class = probe_features
        logger.info('GGEURPPAHook: built probe features with counts %s',
                    {k: len(v) for k, v in probe_features.items()})
        return self.ppa_probe_features_by_class

    def _empty_attack_data(self):
        return {
            'client_id': None,
            'property_label': None,
            'ppa_local_state': {},
            'ppa_global_state': {},
            'ppa_probe_features_by_class': {},
        }

    def _merge_attack_data(self, dst, src):
        for key, value in src.items():
            if key == 'ppa_probe_features_by_class' and value:
                if not dst.get(key):
                    dst[key] = value
            elif isinstance(value, dict):
                dst.setdefault(key, {}).update(value)
            else:
                dst[key] = value

    def _load_saved_attack_features(self):
        per_client = {}
        if not os.path.isdir(self.attack_feature_dir):
            return per_client
        for filename in sorted(os.listdir(self.attack_feature_dir)):
            if not filename.endswith('.pt'):
                continue
            file_path = os.path.join(self.attack_feature_dir, filename)
            try:
                artifact = torch.load(file_path, map_location='cpu', weights_only=False)
                client_id = int(artifact['client_id'])
                if client_id not in per_client:
                    per_client[client_id] = self._empty_attack_data()
                self._merge_attack_data(per_client[client_id], artifact.get('data', {}))
            except Exception as e:
                logger.warning('GGEURPPAHook: failed to load %s: %s', file_path, e)
        return per_client

    def _run_property_attack(self):
        if self.global_mlp is None:
            logger.warning('GGEURPPAHook: global_mlp is None, skip attack')
            return
        per_client_data = self._load_saved_attack_features()
        if not per_client_data:
            logger.warning('GGEURPPAHook: no saved property artifacts in %s',
                           self.attack_feature_dir)
            return

        target_client_id = int(getattr(self._cfg.attack,
                                       'meta_ppa_target_client_id',
                                       getattr(self._cfg.attack,
                                               'fedmia_target_client_id', 1)))
        target_data = per_client_data.get(target_client_id)
        if target_data is None:
            target_client_id, target_data = sorted(per_client_data.items())[0]
            logger.warning('GGEURPPAHook: configured target client missing; '
                           'using client %s for orchestrator target',
                           target_client_id)

        shadow_data = [data for cid, data in sorted(per_client_data.items())
                       if cid != target_client_id]
        results = self.orchestrator.run_all_attacks(
            target_data=target_data,
            shadow_data=shadow_data,
            global_model=self.global_mlp,
        )

        save_path = os.path.join(self._cfg.outdir, 'ggeur_ppa_results.json')
        os.makedirs(self._cfg.outdir, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        metrics = results.get('meta_ppa', {})
        logger.info('GGEURPPAHook: Meta-PPA accuracy=%s baseline=%s saved=%s',
                    metrics.get('accuracy'), metrics.get('random_baseline'),
                    save_path)
