"""Distributed GGEUR FedMIA-I/II for the real gRPC path.

The client processes keep images and feature vectors local.  At configured
attack rounds the server sends the small MLP heads back to the clients, and
each client evaluates those heads on its own member/test/mix probes.  Only
per-sample derived loss/cosine arrays and labels for member-score
normalization are returned.  This reproduces the standalone GGEUR FedMIA data
flow (target train, global test, cross-client mix, indexed shadow calibration,
class-conditional loss normalization, and multi-round aggregation) without
moving raw samples between machines.
"""

from __future__ import annotations

import base64
import copy
import json
import logging
import math
import os
import pickle
from typing import Dict, Iterable, List

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, roc_curve

from federatedscope.core.auxiliaries.utils import recursive_param2tensor
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


def _encode_for_message(value):
    """Encode nested tensors for a non-``model_para`` gRPC message."""
    if isinstance(value, dict):
        return {str(key): _encode_for_message(item)
                for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode_for_message(item) for item in value]
    if torch.is_tensor(value):
        raw = pickle.dumps(value.detach().cpu())
        return base64.b64encode(raw).decode('ascii')
    return value


def _as_float_array(value) -> np.ndarray:
    if value is None:
        return np.asarray([], dtype=np.float64)
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64).reshape(-1)


def _as_int_array(value) -> np.ndarray:
    if value is None:
        return np.asarray([], dtype=np.int64)
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.int64).reshape(-1)


def _concat(parts, dtype=np.float64) -> np.ndarray:
    arrays = [np.asarray(part, dtype=dtype).reshape(-1)
              for part in parts if part is not None and len(part) > 0]
    return np.concatenate(arrays) if arrays else np.asarray([], dtype=dtype)


def _json_value(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    raise TypeError(f'Unsupported JSON value: {type(value)}')


class DistributedFedMIAClientReporter:
    """Compute local and cross-model FedMIA score artifacts."""

    def __init__(self, client):
        self.client = client
        self.cfg = client._cfg
        self.attack_cfg = client._cfg.attack
        self._member_probe = None
        self._test_probe = None
        self._mix_probe = None

    def should_report(self, round_idx: int) -> bool:
        interval = max(1, int(self.attack_cfg.fedmia_save_interval))
        final_round = max(1, int(self.cfg.federate.total_round_num) - 1)
        return int(round_idx) % interval == 0 or int(round_idx) >= final_round

    def _get_loader(self, split: str):
        data = None
        try:
            data = self.client.trainer.ctx.data.get(split, None)
        except Exception:
            data = None
        if data is None and isinstance(self.client.data, dict):
            data = self.client.data.get(split, None)
        return data

    def _sample_probe(self, features, labels, limit: int, salt: int):
        if features is None or labels is None:
            return None
        features = torch.as_tensor(features, dtype=torch.float32).cpu()
        labels = torch.as_tensor(labels, dtype=torch.long).cpu()
        if len(labels) == 0:
            return None
        limit = int(limit)
        if limit <= 0 or limit >= len(labels):
            return features.contiguous(), labels.contiguous()
        generator = torch.Generator().manual_seed(
            int(self.cfg.seed) + 1009 * int(self.client.ID) + int(salt))
        indices = torch.randperm(len(labels), generator=generator)[:limit]
        return features[indices].contiguous(), labels[indices].contiguous()

    def _load_probe(self, split: str, limit: int, salt: int):
        loader = self._get_loader(split)
        if loader is None:
            logger.warning('Client %s: no %s split for distributed FedMIA',
                           self.client.ID, split)
            return None
        features, labels = self.client._extract_eval_features(loader)
        return self._sample_probe(features, labels, limit, salt)

    def _ensure_probes(self):
        if self._member_probe is None:
            self._member_probe = self._load_probe(
                'train', int(self.attack_cfg.fedmia_probe_size), 17)
        if self._test_probe is None:
            self._test_probe = self._load_probe(
                'test', int(self.attack_cfg.fedmia_nonmember_size), 29)
        if self._mix_probe is None:
            client_num = max(2, int(self.cfg.federate.client_num))
            mix_length = int(getattr(self.attack_cfg, 'mix_length', 1000))
            per_client = int(math.ceil(mix_length / (client_num - 1)))
            self._mix_probe = self._load_probe('train', per_client, 43)

    def _build_classifier(self, state):
        classifier = copy.deepcopy(self.client.mlp_classifier)
        classifier.load_state_dict(state, strict=False)
        classifier = classifier.to(self.client.device)
        classifier.eval()
        return classifier

    def _loss_scores(self, classifier, probe) -> np.ndarray:
        if probe is None:
            return np.asarray([], dtype=np.float64)
        features, labels = probe
        batch_size = max(1, int(self.cfg.dataloader.batch_size))
        losses = []
        with torch.no_grad():
            for start in range(0, len(labels), batch_size):
                end = min(start + batch_size, len(labels))
                x = features[start:end].to(self.client.device)
                y = labels[start:end].to(self.client.device)
                losses.append(F.cross_entropy(
                    classifier(x), y, reduction='none').detach().cpu())
        if not losses:
            return np.asarray([], dtype=np.float64)
        return torch.cat(losses).numpy().astype(np.float64, copy=False)

    def _gradient_context(self, global_state, local_state):
        classifier = self._build_classifier(global_state)
        params = []
        update_parts = []
        for name, param in classifier.named_parameters():
            if name not in global_state or name not in local_state:
                continue
            params.append(param)
            global_value = global_state[name].to(self.client.device).float()
            local_value = local_state[name].to(self.client.device).float()
            update_parts.append((global_value - local_value).reshape(-1))
        if not params or not update_parts:
            return None, [], None
        update = torch.cat(update_parts).detach()
        if float(torch.norm(update, p=2)) <= 0.0:
            return None, [], None
        return classifier, params, update

    def _cosine_scores(self, probe, global_state, local_state) -> np.ndarray:
        if probe is None or not bool(self.attack_cfg.fedmia_store_grad_cos):
            return np.asarray([], dtype=np.float64)
        classifier, params, update = self._gradient_context(
            global_state, local_state)
        if classifier is None:
            return np.asarray([], dtype=np.float64)
        features, labels = probe
        values = []
        for idx in range(len(labels)):
            classifier.zero_grad(set_to_none=True)
            x = features[idx:idx + 1].to(self.client.device)
            y = labels[idx:idx + 1].to(self.client.device)
            loss = F.cross_entropy(classifier(x), y)
            grads = torch.autograd.grad(
                loss, params, retain_graph=False, allow_unused=True)
            parts = []
            for grad, param in zip(grads, params):
                value = torch.zeros_like(param) if grad is None else grad
                parts.append(value.detach().reshape(-1))
            sample_grad = torch.cat(parts)
            cosine = F.cosine_similarity(
                sample_grad, update, dim=0, eps=1e-12)
            values.append(float(cosine.detach().cpu()))
        return np.asarray(values, dtype=np.float64)

    def _score_probe(self, probe, global_state, local_state):
        if probe is None:
            return np.asarray([]), np.asarray([])
        classifier = self._build_classifier(local_state)
        return (
            self._loss_scores(classifier, probe),
            self._cosine_scores(probe, global_state, local_state),
        )

    @staticmethod
    def _probe_labels(probe):
        if probe is None:
            return []
        return probe[1].detach().cpu().tolist()

    def build_report(self, round_idx: int, global_state, local_state):
        if global_state is None or local_state is None:
            return None
        self._ensure_probes()
        if self._member_probe is None or self._test_probe is None:
            logger.warning('Client %s: incomplete FedMIA probes; skip round %s',
                           self.client.ID, round_idx)
            return None

        global_state = _cpu_state(global_state)
        local_state = _cpu_state(local_state)
        member_loss, member_cos = self._score_probe(
            self._member_probe, global_state, local_state)

        # With cross evaluation enabled, global test and mix scores are built
        # after the server has received every local head for this round.
        if bool(getattr(self.attack_cfg, 'fedmia_cross_eval', False)):
            nonmember_loss = np.asarray([], dtype=np.float64)
            nonmember_cos = np.asarray([], dtype=np.float64)
        else:
            nonmember_loss, nonmember_cos = self._score_probe(
                self._test_probe, global_state, local_state)

        return {
            'schema_version': 2,
            'client_id': int(self.client.ID),
            'round': int(round_idx),
            'member_loss': member_loss.tolist(),
            'member_cos': member_cos.tolist(),
            'member_labels': self._probe_labels(self._member_probe),
            'nonmember_loss': nonmember_loss.tolist(),
            'nonmember_cos': nonmember_cos.tolist(),
            'member_count': int(len(member_loss)),
            'nonmember_count': int(len(nonmember_loss)),
            'metadata': {
                'feature_extractor': str(self.client.feature_extractor_type),
                'embedding_dim': int(self.client.embedding_dim),
                'derived_scores_only': 1,
                'cross_eval': int(bool(getattr(
                    self.attack_cfg, 'fedmia_cross_eval', False))),
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
                msg_type='fedmia_features',
                sender=self.client.ID,
                receiver=[server_id],
                state=int(round_idx),
                timestamp=timestamp,
                content=payload,
            ))
            logger.info(
                'Client %s: uploaded distributed FedMIA member report for '
                'round %s (members=%s, grad_cos=%s)',
                self.client.ID, round_idx, payload['member_count'],
                bool(payload['member_cos']))
        except Exception as error:
            logger.exception(
                'Client %s: distributed FedMIA report failed at round %s: %s',
                self.client.ID, round_idx, error)

    def handle_cross_eval_request(self, message: Message):
        """Evaluate all round-local heads on this client's local probes."""
        round_idx = int(message.state)
        try:
            payload = message.content
            models = recursive_param2tensor(payload.get('models', {}))
            global_state = recursive_param2tensor(payload.get('global_state'))
            if not isinstance(models, dict) or global_state is None:
                raise ValueError('missing models or global state')

            self._ensure_probes()
            entries = {}
            for owner_text, local_state in sorted(
                    models.items(), key=lambda item: int(item[0])):
                owner_id = int(owner_text)
                if local_state is None:
                    continue
                test_loss, test_cos = self._score_probe(
                    self._test_probe, global_state, local_state)
                entry = {
                    'test_loss': test_loss.tolist(),
                    'test_cos': test_cos.tolist(),
                }
                if owner_id == int(self.client.ID):
                    member_loss, member_cos = self._score_probe(
                        self._member_probe, global_state, local_state)
                    entry.update({
                        'member_loss': member_loss.tolist(),
                        'member_cos': member_cos.tolist(),
                        'member_labels': self._probe_labels(
                            self._member_probe),
                    })
                else:
                    mix_loss, mix_cos = self._score_probe(
                        self._mix_probe, global_state, local_state)
                    entry.update({
                        'mix_loss': mix_loss.tolist(),
                        'mix_cos': mix_cos.tolist(),
                    })
                entries[str(owner_id)] = entry

            response = {
                'schema_version': 2,
                'round': round_idx,
                'evaluator_client_id': int(self.client.ID),
                'entries': entries,
                'derived_scores_only': 1,
            }
            self.client.comm_manager.send(Message(
                msg_type='fedmia_cross_eval_response',
                sender=self.client.ID,
                receiver=[message.sender],
                state=round_idx,
                timestamp=message.timestamp,
                content=response,
            ))
            logger.info(
                'Client %s: returned distributed FedMIA cross evaluation for '
                'round %s (%s model heads)', self.client.ID, round_idx,
                len(entries))
        except Exception as error:
            logger.exception(
                'Client %s: distributed FedMIA cross evaluation failed at '
                'round %s: %s', self.client.ID, round_idx, error)
            self.client.comm_manager.send(Message(
                msg_type='fedmia_cross_eval_response',
                sender=self.client.ID,
                receiver=[message.sender],
                state=round_idx,
                timestamp=message.timestamp,
                content={
                    'schema_version': 2,
                    'round': round_idx,
                    'evaluator_client_id': int(self.client.ID),
                    'entries': {},
                    'error': str(error),
                    'derived_scores_only': 1,
                },
            ))


class DistributedFedMIACollector:
    """Collect score artifacts and run aligned FedMIA-I/II."""

    def __init__(self, server):
        self.server = server
        self.cfg = server._cfg
        self.attack_cfg = server._cfg.attack
        self.reports: Dict[int, Dict[int, Dict]] = {}
        self.cross_eval_responses: Dict[int, Dict[int, Dict]] = {}
        self.cross_eval_expected: Dict[int, set] = {}
        self.cross_eval_owners: Dict[int, List[int]] = {}
        self.pending_cross_eval = set()
        self.report_dir = os.path.join(
            self.cfg.outdir, 'distributed_fedmia_reports')
        self._finalized = False

    def should_report(self, round_idx: int) -> bool:
        interval = max(1, int(self.attack_cfg.fedmia_save_interval))
        final_round = max(1, int(self.cfg.federate.total_round_num) - 1)
        return int(round_idx) % interval == 0 or int(round_idx) >= final_round

    def handle(self, message):
        payload = message.content
        if not isinstance(payload, dict):
            logger.warning('Server: invalid distributed FedMIA payload from %s',
                           message.sender)
            return
        client_id = int(payload.get('client_id', message.sender))
        round_idx = int(payload.get('round', message.state))
        if client_id != int(message.sender):
            logger.warning('Server: FedMIA payload client mismatch: sender=%s '
                           'payload=%s', message.sender, client_id)
            return
        self.reports.setdefault(round_idx, {})[client_id] = payload
        self._save_report(round_idx, client_id, payload)
        logger.info('Server: received distributed FedMIA member report from '
                    'client %s for round %s', client_id, round_idx)

    def _save_report(self, round_idx, client_id, payload):
        os.makedirs(self.report_dir, exist_ok=True)
        torch.save(payload, os.path.join(
            self.report_dir,
            f'client_{client_id}_round_{round_idx}.pt'))

    @staticmethod
    def _extract_mlp_state(model_para):
        if isinstance(model_para, dict) and 'mlp' in model_para:
            return model_para.get('mlp')
        return model_para

    def start_cross_eval(self, round_idx: int, buffered_updates) -> bool:
        """Start aligned test/mix scoring and delay FedAvg until it returns."""
        if not bool(getattr(self.attack_cfg, 'fedmia_cross_eval', False)):
            return False
        if not self.should_report(round_idx):
            return False
        if round_idx in self.pending_cross_eval:
            return True

        models = {}
        for _, model_para, sender in buffered_updates:
            state = self._extract_mlp_state(model_para)
            if state is not None:
                models[int(sender)] = _cpu_state(state)
        if not models or self.server.global_mlp is None:
            logger.warning('Server: cannot start FedMIA cross evaluation for '
                           'round %s: missing model states', round_idx)
            return False

        receivers = sorted(set(self.server._active_clients_for_broadcast()))
        encoded_payload = {
            'schema_version': 2,
            'round': int(round_idx),
            'models': _encode_for_message(models),
            'global_state': _encode_for_message(
                _cpu_state(self.server.global_mlp.state_dict())),
            'mode': str(getattr(self.attack_cfg, 'mode', 'mix')),
            'mix_length': int(getattr(self.attack_cfg, 'mix_length', 1000)),
            'derived_scores_only': 1,
        }
        self.cross_eval_responses[round_idx] = {}
        self.cross_eval_expected[round_idx] = set(receivers)
        self.cross_eval_owners[round_idx] = sorted(models.keys())
        self.pending_cross_eval.add(round_idx)

        send_bytes = self.server._sizeof_content(encoded_payload)
        self.server._bytes_sent += send_bytes * len(receivers)
        logger.info('Server: starting distributed FedMIA cross evaluation for '
                    'round %s with owners=%s evaluators=%s', round_idx,
                    sorted(models.keys()), receivers)
        for client_id in receivers:
            self.server.comm_manager.send(Message(
                msg_type='fedmia_cross_eval_request',
                sender=self.server.ID,
                receiver=[client_id],
                state=int(round_idx),
                content=encoded_payload,
            ))
        return True

    def handle_cross_eval_response(self, message):
        round_idx = int(message.state)
        if round_idx not in self.pending_cross_eval:
            logger.warning('Server: ignore unexpected FedMIA cross response '
                           'from client %s for round %s', message.sender,
                           round_idx)
            return
        sender = int(message.sender)
        expected = self.cross_eval_expected.get(round_idx, set())
        if sender not in expected:
            logger.warning('Server: ignore FedMIA cross response from inactive '
                           'client %s for round %s', sender, round_idx)
            return
        responses = self.cross_eval_responses.setdefault(round_idx, {})
        if sender in responses:
            logger.warning('Server: ignore duplicate FedMIA cross response '
                           'from client %s for round %s', sender, round_idx)
            return
        responses[sender] = message.content
        logger.info('Server: received FedMIA cross evaluation from client %s '
                    'for round %s (%s/%s)', sender, round_idx,
                    len(responses), len(expected))
        if len(responses) < len(expected):
            return

        self._merge_cross_eval_round(round_idx)
        self.pending_cross_eval.discard(round_idx)
        logger.info('Server: distributed FedMIA cross evaluation complete for '
                    'round %s; resume FedAvg', round_idx)
        self.server._perform_fedavg(round_idx)

    def _merge_cross_eval_round(self, round_idx: int):
        responses = self.cross_eval_responses.get(round_idx, {})
        owners = self.cross_eval_owners.get(round_idx, [])
        existing = self.reports.get(round_idx, {})
        mode = str(getattr(self.attack_cfg, 'mode', 'mix')).lower()
        mix_length = int(getattr(self.attack_cfg, 'mix_length', 1000))
        merged = {}

        for owner_id in owners:
            owner_entry = None
            test_loss_parts = []
            test_cos_parts = []
            mix_loss_parts = []
            mix_cos_parts = []
            for evaluator_id, payload in sorted(responses.items()):
                entries = payload.get('entries', {}) \
                    if isinstance(payload, dict) else {}
                entry = entries.get(str(owner_id), {})
                if evaluator_id == owner_id:
                    owner_entry = entry
                test_loss_parts.append(entry.get('test_loss', []))
                test_cos_parts.append(entry.get('test_cos', []))
                if evaluator_id != owner_id:
                    mix_loss_parts.append(entry.get('mix_loss', []))
                    mix_cos_parts.append(entry.get('mix_cos', []))

            fallback = existing.get(owner_id, {})
            owner_entry = owner_entry or {}
            member_loss = _as_float_array(
                owner_entry.get('member_loss', fallback.get('member_loss')))
            member_cos = _as_float_array(
                owner_entry.get('member_cos', fallback.get('member_cos')))
            member_labels = _as_int_array(
                owner_entry.get('member_labels',
                                fallback.get('member_labels')))
            test_loss = _concat(test_loss_parts)
            test_cos = _concat(test_cos_parts)
            mix_loss = _concat(mix_loss_parts)
            mix_cos = _concat(mix_cos_parts)
            if mode == 'mix' and mix_length > 0 and \
                    len(test_loss) > mix_length:
                rng = np.random.default_rng(
                    int(self.cfg.seed) + 10007 * int(round_idx) + owner_id)
                selected = rng.permutation(len(test_loss))[:mix_length]
                test_loss = test_loss[selected]
                if len(test_cos) >= int(np.max(selected)) + 1:
                    test_cos = test_cos[selected]
            if mix_length > 0:
                mix_loss = mix_loss[:mix_length]
                mix_cos = mix_cos[:mix_length]

            nonmember_loss_parts = [test_loss]
            nonmember_cos_parts = [test_cos]
            if mode == 'mix':
                nonmember_loss_parts.append(mix_loss)
                nonmember_cos_parts.append(mix_cos)
            nonmember_loss = _concat(nonmember_loss_parts)
            nonmember_cos = _concat(nonmember_cos_parts)

            report = {
                'schema_version': 2,
                'client_id': int(owner_id),
                'round': int(round_idx),
                'member_loss': member_loss.tolist(),
                'member_cos': member_cos.tolist(),
                'member_labels': member_labels.tolist(),
                'nonmember_loss': nonmember_loss.tolist(),
                'nonmember_cos': nonmember_cos.tolist(),
                'member_count': int(len(member_loss)),
                'nonmember_count': int(len(nonmember_loss)),
                'metadata': {
                    'derived_scores_only': 1,
                    'cross_eval': 1,
                    'test_count': int(len(test_loss)),
                    'mix_count': int(len(mix_loss)),
                    'mode': mode,
                    'mix_length': mix_length,
                },
            }
            merged[owner_id] = report
            self._save_report(round_idx, owner_id, report)

        self.reports[round_idx] = merged

    def _processed_reports(self):
        reports = copy.deepcopy(self.reports)
        mode = str(self.attack_cfg.fedmia_shadow_stat_mode).lower()

        # The standalone GGEUR hook truncates every client's train artifacts to
        # the minimum length before indexed Q_out estimation.
        if mode == 'indexed':
            for _, clients in sorted(reports.items()):
                lengths = [len(_as_float_array(report.get('member_loss')))
                           for report in clients.values()
                           if len(_as_float_array(
                               report.get('member_loss'))) > 0]
                if len(lengths) < 2:
                    continue
                min_len = min(lengths)
                for report in clients.values():
                    for key in ['member_loss', 'member_cos', 'member_labels']:
                        report[key] = report.get(key, [])[:min_len]

        # The standalone plugin keeps train logits and prefers CE recomputed
        # from those logits over normalized ``train_losses``.  Preserve the raw
        # loss so the distributed path matches that actual runtime behavior.
        for clients in reports.values():
            for report in clients.values():
                report['member_loss_raw'] = copy.deepcopy(
                    report.get('member_loss', []))

        fields = getattr(
            self.attack_cfg, 'ggeur_normalize_fields', ['train_losses'])
        if isinstance(fields, str):
            fields = [item.strip() for item in fields.split(',')
                      if item.strip()]
        field_map = {
            'train_losses': 'member_loss',
            'train_cos': 'member_cos',
        }
        for _, score_key in [(name, field_map[name])
                             for name in fields if name in field_map]:
            client_ids = sorted({
                client_id for clients in reports.values()
                for client_id in clients
            })
            for client_id in client_ids:
                score_parts = []
                label_parts = []
                for round_idx in sorted(reports):
                    report = reports[round_idx].get(client_id)
                    if not report:
                        continue
                    scores = _as_float_array(report.get(score_key))
                    labels = _as_int_array(report.get('member_labels'))
                    if len(scores) == len(labels) and len(scores) > 0:
                        score_parts.append(scores)
                        label_parts.append(labels)
                if not score_parts:
                    continue
                all_scores = np.concatenate(score_parts)
                all_labels = np.concatenate(label_parts)
                class_means = {
                    int(class_id): float(np.mean(
                        all_scores[all_labels == class_id]))
                    for class_id in np.unique(all_labels)
                }
                for round_idx in sorted(reports):
                    report = reports[round_idx].get(client_id)
                    if not report:
                        continue
                    scores = _as_float_array(report.get(score_key))
                    labels = _as_int_array(report.get('member_labels'))
                    if len(scores) != len(labels):
                        continue
                    adjusted = scores.copy()
                    for class_id, mean in class_means.items():
                        adjusted[labels == class_id] -= mean
                    report[score_key] = adjusted.tolist()
        return reports

    @staticmethod
    def _client_reports(reports, client_id: int) -> Dict[int, Dict]:
        return {
            round_idx: clients[client_id]
            for round_idx, clients in sorted(reports.items())
            if client_id in clients
        }

    def _calibrate_round(self, target_report: Dict,
                         shadow_reports: Iterable[Dict],
                         member_key: str, nonmember_key: str):
        member = _as_float_array(target_report.get(member_key))
        nonmember = _as_float_array(target_report.get(nonmember_key))
        shadow = [_as_float_array(report.get(member_key))
                  for report in shadow_reports]
        shadow = [values for values in shadow if values.size > 0]
        if member.size == 0 or nonmember.size == 0 or not shadow:
            return None, None

        var_floor = float(self.attack_cfg.fedmia_var_floor)
        mode = str(self.attack_cfg.fedmia_shadow_stat_mode).lower()
        if mode == 'indexed' and len(shadow) > 1:
            min_shadow = min(len(values) for values in shadow)
            stack = np.vstack([values[:min_shadow] for values in shadow])
            mu = np.mean(stack, axis=0)
            var = np.maximum(np.var(stack, axis=0), var_floor)
            member_len = min(len(member), len(mu))
            nonmember_len = min(len(nonmember), len(mu))
            return (
                norm.cdf(member[:member_len], mu[:member_len],
                         np.sqrt(var[:member_len])),
                norm.cdf(nonmember[:nonmember_len], mu[:nonmember_len],
                         np.sqrt(var[:nonmember_len])),
            )

        shadow_values = np.concatenate(shadow)
        mu = float(np.mean(shadow_values))
        var = max(float(np.var(shadow_values)), var_floor)
        scale = np.sqrt(var)
        return norm.cdf(member, mu, scale), norm.cdf(nonmember, mu, scale)

    def _aggregate_rounds(self, values: List[np.ndarray],
                          mode: str = 'mean') -> np.ndarray:
        if not values:
            return np.asarray([], dtype=np.float64)
        min_len = min(len(value) for value in values)
        if min_len <= 0:
            return np.asarray([], dtype=np.float64)
        stack = np.vstack([value[:min_len] for value in values])
        mode = str(mode).lower()
        if mode == 'min':
            return np.min(stack, axis=0)
        if mode == 'max':
            return np.max(stack, axis=0)
        if mode == 'last':
            return stack[-1]
        return np.mean(stack, axis=0)

    @staticmethod
    def _metrics(members: np.ndarray, nonmembers: np.ndarray) -> Dict:
        if members.size == 0 or nonmembers.size == 0:
            return {
                'error': 'empty member or nonmember scores',
                'auc': None,
                'tpr_at_fpr': {},
                'num_members': int(len(members)),
                'num_nonmembers': int(len(nonmembers)),
            }
        labels = np.concatenate([
            np.ones(len(members), dtype=np.int64),
            np.zeros(len(nonmembers), dtype=np.int64),
        ])
        scores = np.concatenate([members, nonmembers])
        auc = float(roc_auc_score(labels, scores))
        fpr, tpr, _ = roc_curve(labels, scores)
        tpr_at_fpr = {}
        for threshold in [0.001, 0.01, 0.1]:
            valid = np.where(fpr <= threshold)[0]
            value = float(np.max(tpr[valid])) if valid.size else 0.0
            tpr_at_fpr[str(threshold)] = value
        return {
            'auc': auc,
            'tpr_at_fpr': tpr_at_fpr,
            'num_members': int(len(members)),
            'num_nonmembers': int(len(nonmembers)),
        }

    def _attack_client(self, target_client: int, reports) -> Dict:
        target_reports = self._client_reports(reports, target_client)
        member_loss_rounds = []
        nonmember_loss_rounds = []
        member_cos_rounds = []
        nonmember_cos_rounds = []
        loss_rounds = []
        cos_rounds = []

        for round_idx, target_report in sorted(target_reports.items()):
            shadow_reports = [
                report for client_id, report
                in reports.get(round_idx, {}).items()
                if int(client_id) != int(target_client)
            ]
            loss_target = {
                'member_loss': -_as_float_array(target_report.get(
                    'member_loss_raw', target_report.get('member_loss'))),
                'nonmember_loss': -_as_float_array(
                    target_report.get('nonmember_loss')),
            }
            loss_shadows = [{
                'member_loss': -_as_float_array(report.get(
                    'member_loss_raw', report.get('member_loss')))
            } for report in shadow_reports]
            loss_member, loss_nonmember = self._calibrate_round(
                loss_target, loss_shadows, 'member_loss', 'nonmember_loss')
            if loss_member is not None:
                member_loss_rounds.append(loss_member)
                nonmember_loss_rounds.append(loss_nonmember)
                loss_rounds.append(int(round_idx))

            cos_member, cos_nonmember = self._calibrate_round(
                target_report, shadow_reports,
                'member_cos', 'nonmember_cos')
            if cos_member is not None:
                member_cos_rounds.append(cos_member)
                nonmember_cos_rounds.append(cos_nonmember)
                cos_rounds.append(int(round_idx))

        fedmia_i_mode = str(getattr(
            self.attack_cfg, 'fedmia_i_round_agg',
            self.attack_cfg.fedmia_round_agg))
        fedmia_i = self._metrics(
            self._aggregate_rounds(member_loss_rounds, fedmia_i_mode),
            self._aggregate_rounds(nonmember_loss_rounds, fedmia_i_mode))
        fedmia_i['rounds_used'] = loss_rounds
        fedmia_i['round_agg'] = fedmia_i_mode
        fedmia_ii = self._metrics(
            self._aggregate_rounds(member_cos_rounds, 'mean'),
            self._aggregate_rounds(nonmember_cos_rounds, 'mean'))
        fedmia_ii['rounds_used'] = cos_rounds
        fedmia_ii['round_agg'] = 'mean'
        return {'fedmia_i': fedmia_i, 'fedmia_ii': fedmia_ii}

    def finalize(self):
        if self._finalized:
            return
        self._finalized = True
        if not self.reports:
            logger.warning('Server: no distributed FedMIA reports received')
            return
        if self.pending_cross_eval:
            logger.warning('Server: finalize with pending FedMIA cross rounds: %s',
                           sorted(self.pending_cross_eval))

        reports = self._processed_reports()
        client_ids = sorted({
            client_id for clients in reports.values()
            for client_id in clients
        })
        target_client = int(self.attack_cfg.fedmia_target_client_id)
        results = {
            'metadata': {
                'target_client_id': target_client,
                'clients': client_ids,
                'rounds': sorted(reports.keys()),
                'mode': str(getattr(self.attack_cfg, 'mode', 'mix')),
                'mix_length': int(getattr(
                    self.attack_cfg, 'mix_length', 1000)),
                'shadow_stat_mode': str(
                    self.attack_cfg.fedmia_shadow_stat_mode),
                'var_floor': float(self.attack_cfg.fedmia_var_floor),
                'normalize_fields': list(getattr(
                    self.attack_cfg, 'ggeur_normalize_fields',
                    ['train_losses'])),
                'cross_eval': int(bool(getattr(
                    self.attack_cfg, 'fedmia_cross_eval', False))),
                'derived_scores_only': 1,
            },
            'target': self._attack_client(target_client, reports),
        }
        if bool(self.attack_cfg.fedmia_compute_all_clients):
            results['all_clients'] = {
                str(client_id): self._attack_client(client_id, reports)
                for client_id in client_ids
            }

        os.makedirs(self.cfg.outdir, exist_ok=True)
        path = os.path.join(
            self.cfg.outdir, 'distributed_fedmia_results.json')
        with open(path, 'w', encoding='utf-8') as file_obj:
            json.dump(results, file_obj, indent=2, default=_json_value)

        target = results['target']
        logger.info('=' * 20 + ' Distributed FedMIA Results ' + '=' * 20)
        for name in ['fedmia_i', 'fedmia_ii']:
            metrics = target.get(name, {})
            logger.info('%s target_client=%s AUC=%s TPR@0.01=%s',
                        name, target_client, metrics.get('auc'),
                        metrics.get('tpr_at_fpr', {}).get('0.01'))
        logger.info('Server: saved distributed FedMIA results to %s', path)
