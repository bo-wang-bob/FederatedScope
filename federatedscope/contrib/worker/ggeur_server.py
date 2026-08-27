"""
GGEUR_Clip Server Implementation

Handles:
1. Collecting local statistics from all clients
2. Aggregating covariance matrices using parallel axis theorem
3. Broadcasting global covariance matrices to clients
4. Standard FedAvg aggregation for MLP training
5. Evaluating on test sets from all domains
6. (Optional) CNN model aggregation and evaluation for knowledge distillation
7. (Optional) CNN model aggregation and evaluation for feature alignment
"""

import os
import time
import logging
import copy
import re
import json
import queue
import base64
import io
import zlib
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from federatedscope.core.message import Message, b64serializer
from federatedscope.core.sampler import Sampler
from federatedscope.core.workers.server import Server
from federatedscope.core.auxiliaries.optimizer_builder import get_optimizer
from federatedscope.core.auxiliaries.utils import param2tensor

logger = logging.getLogger(__name__)

# Shared BERT extractors avoid repeated loads during standalone evaluation.
_SHARED_BERT_EXTRACTORS = {}


class DomainStratifiedSampler(Sampler):
    """Sample equally from contiguous client ranges representing domains.

    Multi-domain loaders in this project assign the same number of consecutive
    client ids to every domain.  A dedicated RNG prevents client-side NumPy
    seeding from accidentally repeating the same uniform draw.
    """

    def __init__(self, client_num, group_num, seed):
        super().__init__(int(client_num))
        self.group_num = int(group_num)
        if self.group_num <= 0 or int(client_num) % self.group_num != 0:
            raise ValueError(
                "domain-stratified sampling requires client_num to be "
                "divisible by a positive group_num")
        self.clients_per_group = int(client_num) // self.group_num
        self.rng = np.random.RandomState(int(seed))

    def sample(self, size):
        size = int(size)
        if size < self.group_num or size > len(self.client_state) - 1:
            raise ValueError(
                "domain-stratified sample size must cover every group and "
                "not exceed client_num")
        base, remainder = divmod(size, self.group_num)
        sampled_clients = []
        for group_idx in range(self.group_num):
            count = base + (1 if group_idx < remainder else 0)
            first = group_idx * self.clients_per_group + 1
            last = first + self.clients_per_group
            candidates = np.asarray([
                client_id for client_id in range(first, last)
                if self.client_state[client_id] == 1
            ], dtype=np.int64)
            if len(candidates) < count:
                raise RuntimeError(
                    f"not enough idle clients in domain group {group_idx}: "
                    f"requested={count}, idle={len(candidates)}")
            sampled_clients.extend(
                self.rng.choice(candidates, size=count,
                                replace=False).astype(int).tolist())
        self.change_state(sampled_clients, 'working')
        return sampled_clients


class GGEURServer(Server):
    """
    GGEUR_Clip Server that:
    1. Collects local statistics (means, covariances, counts) from clients
    2. Aggregates covariance matrices using parallel axis theorem
    3. Broadcasts global covariances to all clients
    4. Performs standard FedAvg aggregation on MLP parameters
    5. Evaluates on test sets from all domains after each round
    6. (Optional) Aggregates and evaluates CNN model for knowledge distillation
    7. (Optional) Aggregates and evaluates CNN model for feature alignment
    """

    def __init__(self, ID=-1, state=0, config=None, data=None, model=None,
                 client_num=5, total_round_num=10, device='cpu',
                 strategy=None, unseen_clients_id=None, **kwargs):
        super(GGEURServer, self).__init__(ID, state, config, data, model,
                                          client_num, total_round_num, device,
                                          strategy, unseen_clients_id, **kwargs)

        if config is None:
            return

        self.ggeur_cfg = config.ggeur
        if bool(getattr(
                self.ggeur_cfg, 'domain_stratified_sampling', False)):
            group_num = int(getattr(
                self.ggeur_cfg, 'domain_sampling_group_num', 4))
            self.sampler = DomainStratifiedSampler(
                client_num=client_num,
                group_num=group_num,
                seed=int(getattr(config, 'seed', 42)))
            logger.info(
                "Server: Domain-stratified client sampling enabled "
                f"(groups={group_num}, sample_client_num="
                f"{int(config.federate.sample_client_num)}, "
                f"seed={int(getattr(config, 'seed', 42))})")
        self.head_only_mode = getattr(self.ggeur_cfg, 'head_only_mode', False)
        if self.head_only_mode:
            logger.info(
                "Server: GGEUR HeadOnly system mode active. Round 0 keeps "
                "statistics/covariance/prototype/augmentation; later rounds "
                "aggregate MLP-head parameters only.")

        # Statistics collection buffers
        self.local_statistics_buffer = {}  # {client_id: statistics}
        self.statistics_collected = False

        # Aggregated covariance matrices
        self.global_cov_matrices = {}  # {class_idx: cov_matrix}

        # All client prototypes for cross-client augmentation
        self.all_prototypes = {}  # {client_id: {class_idx: prototype}}

        # Global prototypes (aggregated means) for feature alignment
        self.global_prototypes = {}  # {class_idx: mean_vector}
        self.domain_prototype_ensemble = {}
        self.domain_personalized_heads = {}

        # FedProto prototypes are distinct from the round-0 frozen-feature
        # prototypes above.  They are recomputed every round in the trainable
        # RNN/LSTM hidden space and aggregated by per-class counts.
        self.use_fedproto = bool(
            getattr(self.ggeur_cfg, 'use_fedproto', False))
        self.fedproto_global_prototypes = {}

        # MLP model for aggregation
        self.global_mlp = None

        # ===== FedOpt (server-side) =====
        self.use_fedopt = getattr(config.fedopt, 'use', False) if hasattr(config, 'fedopt') else False
        self.fedopt_optimizer = None
        self.fedopt_scheduler = None
        self.fedopt_annealing = getattr(config.fedopt, 'annealing', False) if hasattr(config, 'fedopt') else False

        # Track which clients have completed augmentation
        self.augmentation_ready_clients = set()

        # Test data for evaluation
        self.test_features = {}  # {domain_name: features array}
        self.test_labels = {}    # {domain_name: labels array}
        self.test_data_loaded = False

        # CLIP model for test feature extraction
        self.clip_model = None
        self.clip_preprocess = None

        # BERT model/tokenizer for text feature extraction
        self.bert_model = None
        self.bert_tokenizer = None

        # Tracking best model
        self.best_avg_accuracy = 0.0
        self.best_model_state = None
        self.test_accuracies_history = {}  # {domain: [acc_per_round]}
        self.client_eval_buffer = {}
        self.pending_client_eval_round = None

        # ===== CNN Mode =====
        self.use_cnn_distillation = getattr(self.ggeur_cfg, 'use_cnn_distillation', False)
        self.use_feature_alignment = getattr(self.ggeur_cfg, 'use_feature_alignment', False)
        self.global_cnn = None  # Global CNN model
        self.cnn_test_accuracies_history = {}  # {domain: [acc_per_round]}
        self.best_cnn_avg_accuracy = 0.0
        self.best_cnn_model_state = None
        # Test images for CNN evaluation (original images, not CLIP features)
        self.test_images_loaded = False
        self.test_image_loaders = {}  # {domain_name: DataLoader}

        # ===== Feature Extractor Mode =====
        # 'clip': Use CLIP (ViT-based, original method)
        # 'cnn': Use pretrained CNN (ConvNeXt, ResNet, etc.)
        # 'timm': Use any timm vision backbone (e.g., GFNet / Mixer / etc.)
        self.feature_extractor_type = getattr(self.ggeur_cfg, 'feature_extractor', 'clip')
        self.cnn_extractor = None  # CNN feature extractor for evaluation
        self.timm_extractor = None  # timm feature extractor for evaluation

        # Embedding dim inferred from clients/statistics (preferred over cfg.ggeur.embedding_dim)
        self.inferred_embedding_dim = None

        # ===== Separated Training Mode =====
        self.use_separated_training = getattr(self.ggeur_cfg, 'use_separated_training', False)
        self.classifier_pretrain_rounds = getattr(self.ggeur_cfg, 'classifier_pretrain_rounds', 20)
        self.freeze_classifier = getattr(self.ggeur_cfg, 'freeze_classifier', True)
        # Phase tracking: 'classifier' or 'cnn_backbone'
        self.training_phase = 'classifier'
        # Store pretrained classifier for Phase 2
        self.pretrained_classifier = None

        # ===== PromptFL Mode =====
        self.use_promptfl = getattr(self.ggeur_cfg, 'use_promptfl', False)
        self.global_prompt_ctx = None          # Aggregated ctx tensor [n_ctx, ctx_dim]
        self.prompt_test_accuracies_history = {}
        self.best_prompt_avg_accuracy = 0.0

        # Training wall-clock timer
        self._train_start_time = time.time()

        # Per-round timing
        self._round_start_time = None
        self._round_durations = []   # seconds per round
        self._round_system_metrics = []

        # Communication volume (bytes)
        self._bytes_sent = 0      # server → clients
        self._bytes_recv = 0      # clients → server
        # CLIP model and prompt components for server-side evaluation
        self.prompt_learner_eval = None
        self.text_encoder_eval = None

        self.distributed_stage_timeout = int(
            getattr(self.ggeur_cfg, 'distributed_stage_timeout', 1800))
        self._stage_name = None
        self._stage_start_time = None
        self.min_statistics_clients = self._resolve_min_clients(
            'min_statistics_clients')
        self.min_augmentation_clients = self._resolve_min_clients(
            'min_augmentation_clients')
        self.min_train_updates = self._resolve_min_clients(
            'min_train_updates')
        self.active_client_ids = set()
        self.training_client_ids = set()
        self.hierarchical_training = bool(
            getattr(self.ggeur_cfg, 'hierarchical_training', False))
        self.hierarchical_subserver_num = max(
            1,
            int(getattr(self.ggeur_cfg,
                        'hierarchical_subserver_num', 1)))
        self.hierarchical_subserver_id_base = int(
            getattr(self.ggeur_cfg,
                    'hierarchical_subserver_id_base', 100000))
        if self.hierarchical_training:
            logger.info(
                "Server: Hierarchical training enabled with %s "
                "subservers (sender IDs %s..%s)",
                self.hierarchical_subserver_num,
                self.hierarchical_subserver_id_base + 1,
                self.hierarchical_subserver_id_base +
                self.hierarchical_subserver_num)

    def _resolve_min_clients(self, name):
        value = int(getattr(self.ggeur_cfg, name, 0))
        if value <= 0:
            return int(self._client_num)
        return max(1, min(value, int(self._client_num)))

    def _active_clients_for_broadcast(self):
        if self.training_client_ids:
            return sorted(self.training_client_ids)
        if self.active_client_ids:
            return sorted(self.active_client_ids)
        return list(range(1, self._client_num + 1))

    def _broadcast_identical_to_clients(self, msg_type, receivers, state,
                                        content):
        """Send one logical message when subservers can fan it out.

        The gRPC manager deduplicates receivers that share a proxy endpoint,
        but only inside a single ``Message``.  Sending one message per logical
        client therefore serialized the same multi-megabyte model 120 times
        across the root-to-subserver link.  A hierarchical subserver already
        accepts a receiver list and performs the final local fan-out, so keep
        all logical receivers in one message in hierarchical mode.
        """
        receivers = [int(client_id) for client_id in receivers]
        if not receivers:
            return
        receiver_batches = ([receivers] if self.hierarchical_training else
                            [[client_id] for client_id in receivers])
        for receiver_batch in receiver_batches:
            self.comm_manager.send(
                Message(msg_type=msg_type,
                        sender=self.ID,
                        receiver=receiver_batch,
                        state=state,
                        content=content))

    def _expected_augmentation_clients(self):
        base = len(self.active_client_ids) if self.active_client_ids else self._client_num
        return max(1, min(int(self.min_augmentation_clients), int(base)))

    def _statistics_quorum_state(self):
        """Return whether round-0 statistics may be finalized.

        A cache-hot client without legacy cached statistics reports
        ``augmentation_ready`` directly.  When only part of a deployment has
        such a cache, waiting for ``client_num`` statistics deadlocks: the
        cache-hot clients have already completed augmentation and will never
        upload statistics.  Treat that mixed state as complete only when every
        configured client is accounted for by either a statistics upload or an
        augmentation-ready message.
        """
        statistics_clients = set(self.local_statistics_buffer.keys())
        cache_ready_clients = set(self.augmentation_ready_clients)
        configured_quorum = (
            len(statistics_clients) >= self.min_statistics_clients)
        accounted_clients = statistics_clients | cache_ready_clients
        mixed_cache_quorum = (
            bool(statistics_clients) and
            len(accounted_clients) >= int(self._client_num))
        return (
            configured_quorum or mixed_cache_quorum,
            mixed_cache_quorum and not configured_quorum,
            statistics_clients,
            cache_ready_clients,
        )

    def _statistics_broadcast_receivers(self):
        """Clients waiting for global covariance after uploading statistics."""
        return sorted(self.local_statistics_buffer.keys())

    def _expected_train_updates(self):
        if self.hierarchical_training:
            return self.hierarchical_subserver_num
        base = len(self.training_client_ids) if self.training_client_ids else len(
            self._active_clients_for_broadcast())
        return max(1, min(int(self.min_train_updates), int(base)))

    def _expected_model_senders(self):
        if self.hierarchical_training:
            return {
                self.hierarchical_subserver_id_base + idx
                for idx in range(1, self.hierarchical_subserver_num + 1)
            }
        return set(self._active_clients_for_broadcast())

    def _mark_stage(self, stage_name):
        self._stage_name = stage_name
        self._stage_start_time = time.time()
        logger.info(
            f"Server: Entered distributed stage '{stage_name}' "
            f"(timeout={self.distributed_stage_timeout}s)")

    def _clear_stage(self):
        self._stage_name = None
        self._stage_start_time = None

    def _check_distributed_stage_timeout(self):
        if self.is_finish:
            return
        if self.distributed_stage_timeout <= 0:
            return
        if self._stage_name is None or self._stage_start_time is None:
            return

        elapsed = time.time() - self._stage_start_time
        if elapsed <= self.distributed_stage_timeout:
            return

        logger.error(
            f"Server: Timeout in distributed stage '{self._stage_name}' "
            f"after {elapsed:.1f}s. joined_clients="
            f"{list(getattr(self.comm_manager, 'neighbors', {}).keys())}, "
            f"augmentation_ready={sorted(self.augmentation_ready_clients)}, "
            f"train_buffer_rounds={list(self.msg_buffer.get('train', {}).keys())}")
        self._notify_joined_clients_to_finish()
        if hasattr(self.comm_manager, 'shutdown'):
            self.comm_manager.shutdown()
        self.is_finish = True
        raise TimeoutError(
            f"GGEUR distributed stage timeout: {self._stage_name}")

    def run(self):
        """Run GGEUR distributed server with GGEUR-specific stage timeouts."""
        while self.join_in_client_num < self.client_num:
            self._check_join_timeout()
            try:
                msg = self.comm_manager.receive(timeout=1.0)
            except queue.Empty:
                continue
            self.msg_handlers[msg.msg_type](msg)

        if self._stage_name is None:
            self._mark_stage('statistics')

        while self.state <= self.total_round_num and not self.is_finish:
            try:
                msg = self.comm_manager.receive(timeout=1.0)
            except queue.Empty:
                self._check_distributed_stage_timeout()
                continue
            self.msg_handlers[msg.msg_type](msg)

        if not self.is_finish:
            self.terminate(msg_type='finish')

    def _get_domainnet_eval_metadata(self):
        """Resolve DomainNet evaluation domains and classes."""
        from federatedscope.cv.dataset.domainnet import (
            discover_domainnet_metadata, load_domainnet_manifest)

        selected_domains = list(getattr(self.ggeur_cfg, 'domainnet_domains', []))
        manifest_path = str(getattr(
            self.ggeur_cfg, 'domainnet_manifest_path', '') or '')
        if manifest_path:
            domains, classes, _ = load_domainnet_manifest(
                manifest_path, selected_domains)
            return domains, classes
        shared_classes_only = getattr(self.ggeur_cfg,
                                      'domainnet_shared_classes_only', False)
        return discover_domainnet_metadata(self._cfg.data.root, selected_domains,
                                           shared_classes_only)

    def _get_domainnet_eval_records(self):
        """Return portable DomainNet records when a manifest is configured."""
        manifest_path = str(getattr(
            self.ggeur_cfg, 'domainnet_manifest_path', '') or '')
        if not manifest_path:
            return None
        from federatedscope.cv.dataset.domainnet import load_domainnet_manifest
        selected_domains = list(getattr(
            self.ggeur_cfg, 'domainnet_domains', []))
        _, _, records = load_domainnet_manifest(
            manifest_path, selected_domains)
        return records

    def _get_embedding_dim(self):
        """Get embedding dim, preferring inferred value from client statistics."""
        if isinstance(self.inferred_embedding_dim, int) and self.inferred_embedding_dim > 0:
            return int(self.inferred_embedding_dim)

        # Infer from any received mean vector
        for client_stats in self.local_statistics_buffer.values():
            means = client_stats.get('means', {})
            for _, mean in means.items():
                if hasattr(mean, 'shape') and len(mean.shape) == 1:
                    self.inferred_embedding_dim = int(mean.shape[0])
                    return int(self.inferred_embedding_dim)
                try:
                    # Fallback for list-like
                    self.inferred_embedding_dim = int(len(mean))
                    return int(self.inferred_embedding_dim)
                except Exception:
                    continue

        return int(getattr(self.ggeur_cfg, 'embedding_dim', 512))

    def _register_default_handlers(self):
        """Register message handlers"""
        super()._register_default_handlers()

        # Register handler for local statistics
        self.register_handlers('local_statistics',
                               self.callback_for_local_statistics)

        # Register handler for augmentation ready signal
        self.register_handlers('augmentation_ready',
                               self.callback_for_augmentation_ready)
        self.register_handlers('client_eval_metrics',
                               self.callback_for_client_eval_metrics)

    def _build_global_mlp(self, num_classes):
        """Build global trainable classifier."""
        input_dim = self._get_embedding_dim()
        hidden_dim = self.ggeur_cfg.mlp_hidden_dim
        model_type = str(getattr(self._cfg.model, 'type', '')).lower()

        init_seed = int(getattr(
            self.ggeur_cfg, 'classifier_init_seed', -1))
        cpu_rng_state = None
        if init_seed >= 0:
            cpu_rng_state = torch.get_rng_state()
            torch.manual_seed(init_seed)

        try:
            if model_type in ['ggeur_rnn', 'ggeur_lstm']:
                from federatedscope.contrib.model.ggeur_text_rnn import \
                    GGEURTextRNNClassifier

                rnn_type = 'rnn' if model_type == 'ggeur_rnn' else 'lstm'
                input_dim = int(getattr(self._cfg.model, 'in_channels',
                                        input_dim) or input_dim)
                self.global_mlp = GGEURTextRNNClassifier(
                    input_dim=input_dim,
                    hidden_dim=int(getattr(self._cfg.model, 'hidden', 256)),
                    num_classes=num_classes,
                    num_layers=int(getattr(self._cfg.model, 'layer', 1)),
                    dropout=float(getattr(self._cfg.model, 'dropout', 0.0)),
                    rnn_type=rnn_type)
            elif hidden_dim > 0:
                self.global_mlp = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(self.ggeur_cfg.mlp_dropout),
                    nn.Linear(hidden_dim, num_classes)
                )
            else:
                self.global_mlp = nn.Linear(input_dim, num_classes)
        finally:
            if cpu_rng_state is not None:
                torch.set_rng_state(cpu_rng_state)

        self.global_mlp = self.global_mlp.to(self.device)
        self._initialize_classifier_from_round0_statistics(num_classes)
        logger.info(
            f"Server: Built global {model_type or 'mlp'} classifier "
            f"with {num_classes} classes "
            f"(classifier_init_seed={init_seed})")

        # Initialize FedOpt optimizer if enabled (needs a built global model)
        self._init_fedopt_if_needed()

    def _initialize_classifier_from_round0_statistics(self, num_classes):
        """Select the configured classifier initialization from round-0 stats."""
        if bool(getattr(self.ggeur_cfg, 'lda_classifier_init', False)):
            self._initialize_classifier_from_diagonal_lda(num_classes)
            return
        self._initialize_classifier_from_global_prototypes(num_classes)

    def _initialize_classifier_from_diagonal_lda(self, num_classes):
        """Initialize a linear head from pooled diagonal training statistics."""
        if not isinstance(self.global_mlp, nn.Linear):
            logger.warning(
                "Server: Diagonal-LDA initialization requires "
                "mlp_hidden_dim=0; keep the configured random initialization")
            return
        if not bool(getattr(self.ggeur_cfg, 'diagonal_covariance', False)):
            logger.warning(
                "Server: Diagonal-LDA initialization requires "
                "diagonal_covariance=true; keep random initialization")
            return

        input_dim = int(self.global_mlp.in_features)
        pooled_sum = np.zeros(input_dim, dtype=np.float64)
        pooled_count = 0
        for class_idx, covariance in self.global_cov_matrices.items():
            prototype = self.global_prototypes.get(int(class_idx))
            if prototype is None:
                continue
            count = 0
            for statistics in self.local_statistics_buffer.values():
                count += int(self._get_class_value(
                    statistics.get('counts', {}), class_idx) or 0)
            covariance = np.asarray(covariance, dtype=np.float64)
            if count > 0 and covariance.shape == (input_dim,):
                pooled_sum += count * covariance
                pooled_count += count
        if pooled_count <= 0:
            logger.warning(
                "Server: No usable diagonal covariance statistics for "
                "LDA initialization; keep random initialization")
            return

        variance = np.maximum(
            pooled_sum / float(pooled_count), 1e-8)
        initialized = 0
        with torch.no_grad():
            self.global_mlp.weight.zero_()
            if self.global_mlp.bias is not None:
                self.global_mlp.bias.zero_()
            for class_idx, prototype in self.global_prototypes.items():
                class_idx = int(class_idx)
                mean = np.asarray(prototype, dtype=np.float64)
                if (class_idx < 0 or class_idx >= int(num_classes) or
                        mean.shape != (input_dim,)):
                    continue
                weight = mean / variance
                self.global_mlp.weight[class_idx].copy_(torch.as_tensor(
                    weight, device=self.device,
                    dtype=self.global_mlp.weight.dtype))
                if self.global_mlp.bias is not None:
                    self.global_mlp.bias[class_idx] = -0.5 * float(
                        np.dot(mean, weight))
                initialized += 1
        logger.info(
            "Server: Initialized linear classifier from pooled diagonal "
            f"LDA statistics ({initialized}/{int(num_classes)} classes)")

    def _initialize_classifier_from_global_prototypes(self, num_classes):
        """Initialize a linear head as a cosine-prototype classifier.

        Unit-normalized class prototypes avoid the class-dependent norm bias
        that an Euclidean-distance head introduces for heterogeneous visual
        features. For the text RNN/LSTM heads, each input-space prototype is
        first mapped through the recurrent feature block and the final linear
        classifier is initialized from the resulting hidden representation.
        Only round-0 client training statistics are used and the option remains
        disabled by default.
        """
        if not bool(getattr(
                self.ggeur_cfg, 'prototype_classifier_init', False)):
            return

        direct_linear = isinstance(self.global_mlp, nn.Linear)
        recurrent_classifier = getattr(
            self.global_mlp, 'classifier', None)
        recurrent_head = (
            not direct_linear
            and isinstance(recurrent_classifier, nn.Linear)
            and hasattr(self.global_mlp, 'rnn')
        )
        if not direct_linear and not recurrent_head:
            logger.warning(
                "Server: Prototype classifier initialization requires a "
                "linear or RNN/LSTM classification head; keep the configured "
                "random initialization")
            return

        classifier = self.global_mlp if direct_linear else recurrent_classifier
        input_dim = int(
            classifier.in_features if direct_linear
            else self.global_mlp.rnn.input_size)
        initialized = 0
        was_training = bool(self.global_mlp.training)
        self.global_mlp.eval()
        with torch.no_grad():
            classifier.weight.zero_()
            if classifier.bias is not None:
                classifier.bias.zero_()
            for class_idx, prototype in self.global_prototypes.items():
                class_idx = int(class_idx)
                array = np.asarray(prototype, dtype=np.float32)
                if (class_idx < 0 or class_idx >= int(num_classes) or
                        array.shape != (input_dim,)):
                    continue
                value = torch.as_tensor(
                    array, device=self.device,
                    dtype=classifier.weight.dtype)
                if recurrent_head:
                    _, recurrent_features = self.global_mlp(
                        value.unsqueeze(0), return_features=True)
                    value = recurrent_features[0]
                value_norm = torch.linalg.vector_norm(value)
                if not torch.isfinite(value_norm) or value_norm <= 0:
                    continue
                classifier.weight[class_idx].copy_(value / value_norm)
                initialized += 1
        self.global_mlp.train(was_training)
        logger.info(
            "Server: Initialized classifier from global class "
            f"prototypes ({initialized}/{int(num_classes)} classes)")

    def _init_fedopt_if_needed(self):
        """Initialize FedOpt optimizer/scheduler for global MLP if enabled."""
        if not self.use_fedopt:
            return
        if self.global_mlp is None:
            return
        if self.fedopt_optimizer is not None:
            return

        try:
            self.fedopt_optimizer = get_optimizer(model=self.global_mlp,
                                                  **self._cfg.fedopt.optimizer)
        except Exception as e:
            logger.error(f"Server: Failed to build FedOpt optimizer: {e}")
            self.fedopt_optimizer = None
            return

        if self.fedopt_annealing:
            try:
                self.fedopt_scheduler = torch.optim.lr_scheduler.StepLR(
                    self.fedopt_optimizer,
                    step_size=self._cfg.fedopt.annealing_step_size,
                    gamma=self._cfg.fedopt.annealing_gamma
                )
            except Exception as e:
                logger.warning(f"Server: Failed to build FedOpt scheduler, disabling annealing: {e}")
                self.fedopt_scheduler = None
                self.fedopt_annealing = False

        opt_type = getattr(self._cfg.fedopt.optimizer, 'type', 'Unknown')
        opt_lr = getattr(self._cfg.fedopt.optimizer, 'lr', 'Unknown')
        logger.info(f"Server: FedOpt enabled - optimizer={opt_type}, lr={opt_lr}, annealing={self.fedopt_annealing}")

    def _build_global_cnn(self, num_classes):
        """Build global CNN model for knowledge distillation or feature alignment"""
        from federatedscope.contrib.model.ggeur_cnn import GGEUR_CNN_FeatureAlign

        cnn_model_name = getattr(self.ggeur_cfg, 'cnn_model', 'resnet18')
        clip_dim = getattr(self.ggeur_cfg, 'embedding_dim', 512)

        # For feature alignment: no pretrained weights (from scratch)
        # For distillation: use pretrained weights
        if self.use_feature_alignment:
            cnn_pretrained = False
            mode_str = "feature alignment (from scratch)"
        else:
            cnn_pretrained = getattr(self.ggeur_cfg, 'cnn_pretrained', True)
            mode_str = f"distillation (pretrained={cnn_pretrained})"

        self.global_cnn = GGEUR_CNN_FeatureAlign(
            model_name=cnn_model_name,
            num_classes=num_classes,
            clip_dim=clip_dim,
            pretrained=cnn_pretrained
        )
        self.global_cnn = self.global_cnn.to(self.device)
        logger.info(f"Server: Built global CNN ({cnn_model_name}) for {mode_str} with {num_classes} classes")

    def _load_clip_model(self):
        """Load CLIP model for test feature extraction"""
        if self.clip_model is not None:
            return

        try:
            import open_clip

            model_name = self.ggeur_cfg.clip_model
            pretrained = self.ggeur_cfg.clip_pretrained
            local_path = self.ggeur_cfg.clip_model_path

            if local_path and os.path.exists(local_path):
                logger.info(f"Server: Loading CLIP from local path: {local_path}")
                self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                    model_name, pretrained=local_path
                )
            else:
                logger.info(f"Server: Loading CLIP from pretrained: {pretrained}")
                self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                    model_name, pretrained=pretrained
                )

            self.clip_model = self.clip_model.to(self.device)
            self.clip_model.eval()
            logger.info(f"Server: Loaded CLIP model {model_name}")

        except ImportError:
            logger.error("open_clip not installed. Please install: pip install open_clip_torch")
            raise

    def _load_cnn_extractor(self):
        """Load CNN feature extractor for test feature extraction"""
        if self.cnn_extractor is not None:
            return

        try:
            from federatedscope.contrib.model.ggeur_cnn_extractor import CNNFeatureExtractor

            model_name = getattr(self.ggeur_cfg, 'cnn_backbone', 'convnext_base')
            pretrained = getattr(self.ggeur_cfg, 'cnn_pretrained', True)
            checkpoint_path = getattr(self.ggeur_cfg, 'cnn_checkpoint_path', '')
            freeze = True  # Always freeze for feature extraction

            self.cnn_extractor = CNNFeatureExtractor(
                model_name=model_name,
                pretrained=pretrained,
                freeze=freeze,
                checkpoint_path=checkpoint_path,
            )
            self.cnn_extractor = self.cnn_extractor.to(self.device)

            logger.info(f"Server: Loaded CNN extractor {model_name}, "
                       f"feature_dim={self.cnn_extractor.get_feature_dim()}")

        except Exception as e:
            logger.error(f"Server: Failed to load CNN extractor: {e}")
            raise

    def _load_timm_extractor(self):
        """Load timm feature extractor for test feature extraction"""
        if self.timm_extractor is not None:
            return

        try:
            from federatedscope.contrib.model.ggeur_timm_extractor import TimmFeatureExtractor

            model_name = getattr(self.ggeur_cfg, 'timm_model', 'gfnet_tiny')
            pretrained = getattr(self.ggeur_cfg, 'timm_pretrained', True)
            checkpoint_path = getattr(self.ggeur_cfg, 'timm_checkpoint_path', '')
            freeze = True  # Always freeze for feature extraction
            in_chans = getattr(self.ggeur_cfg, 'timm_in_chans', 3)
            global_pool = getattr(self.ggeur_cfg, 'timm_global_pool', 'avg')

            self.timm_extractor = TimmFeatureExtractor(
                model_name=model_name,
                pretrained=pretrained,
                freeze=freeze,
                checkpoint_path=checkpoint_path,
                in_chans=in_chans,
                global_pool=global_pool,
            )
            self.timm_extractor = self.timm_extractor.to(self.device)

            logger.info(f"Server: Loaded timm extractor {model_name}, "
                        f"feature_dim={self.timm_extractor.get_feature_dim()}")

        except Exception as e:
            logger.error(f"Server: Failed to load timm extractor: {e}")
            raise

    def _load_bert_extractor(self):
        """Load a frozen BERT encoder for text test feature extraction."""
        if self.bert_model is not None and self.bert_tokenizer is not None:
            return

        try:
            from transformers import AutoConfig, AutoModel, AutoTokenizer
        except Exception as error:
            raise ImportError(
                "transformers is required for ggeur.feature_extractor='bert'"
            ) from error

        model_path = str(getattr(self.ggeur_cfg, 'bert_model_path', '') or '')
        if not model_path:
            raise ValueError(
                "Please set ggeur.bert_model_path for BERT text extraction")
        tokenizer_path = str(
            getattr(self.ggeur_cfg, 'bert_tokenizer_path', '') or ''
        ) or model_path
        local_only = bool(getattr(self.ggeur_cfg, 'bert_local_files_only',
                                  True))
        use_pretrained = bool(
            getattr(self.ggeur_cfg, 'bert_use_pretrained_weights', True))
        key = (model_path, tokenizer_path, local_only, use_pretrained,
               int(getattr(self.ggeur_cfg, 'bert_max_length', 128)),
               str(getattr(self.ggeur_cfg, 'bert_pooling', 'cls')))

        if key not in _SHARED_BERT_EXTRACTORS:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path, local_files_only=local_only)
            if use_pretrained:
                model = AutoModel.from_pretrained(
                    model_path, local_files_only=local_only)
            else:
                bert_cfg = AutoConfig.from_pretrained(
                    model_path, local_files_only=local_only)
                model = AutoModel.from_config(bert_cfg)
            for param in model.parameters():
                param.requires_grad_(False)
            model.eval()
            _SHARED_BERT_EXTRACTORS[key] = (tokenizer, model.cpu())
            logger.info(f"Server: Loaded shared BERT extractor from {model_path}")

        self.bert_tokenizer, self.bert_model = _SHARED_BERT_EXTRACTORS[key]
        self.bert_model = self.bert_model.to(self.device)
        self.bert_model.eval()
        hidden_size = getattr(getattr(self.bert_model, 'config', None),
                              'hidden_size', None)
        if hidden_size is not None:
            self.inferred_embedding_dim = int(hidden_size)
        logger.info(
            f"Server: BERT extractor ready, "
            f"feature_dim={self._get_embedding_dim()}")

    def _load_feature_extractor(self):
        """Load the appropriate feature extractor."""
        if self.feature_extractor_type == 'cnn':
            self._load_cnn_extractor()
        elif self.feature_extractor_type == 'timm':
            self._load_timm_extractor()
        elif self.feature_extractor_type == 'bert':
            self._load_bert_extractor()
        else:
            self._load_clip_model()

    def _get_test_cache_path(self, domain):
        """Get cache path for test features"""
        cache_dir = getattr(self.ggeur_cfg, 'feature_cache_dir', '')
        if not cache_dir:
            cache_dir = os.path.join(os.path.dirname(self._cfg.data.root), 'clip_feature_cache')

        os.makedirs(cache_dir, exist_ok=True)

        # Get split parameters to include in cache filename
        data_type = self._cfg.data.type.lower()
        if hasattr(self._cfg.data, 'splits'):
            splits = tuple(self._cfg.data.splits)
        else:
            if 'office' in data_type and 'home' in data_type:
                splits = (0.7, 0.0, 0.3)
            else:
                splits = (0.8, 0.1, 0.1)
        configured_split_seed = int(getattr(
            self.ggeur_cfg, 'data_split_seed', -1))
        seed = (int(self._cfg.seed) if configured_split_seed < 0
                else configured_split_seed)

        dataset_name = data_type
        cache_suffix = ''
        if 'domainnet' in data_type or 'domain-net' in data_type or 'domain_net' in data_type:
            selected_domains = list(getattr(self.ggeur_cfg, 'domainnet_domains', []))
            shared_classes_only = getattr(self.ggeur_cfg,
                                          'domainnet_shared_classes_only', False)
            domain_token = '-'.join(selected_domains) if selected_domains else 'auto'
            shared_token = 'shared' if shared_classes_only else 'union'
            cache_suffix = f"_{shared_token}_{domain_token}"

        # Build model string based on feature extractor type
        if self.feature_extractor_type == 'cnn':
            model_name = getattr(self.ggeur_cfg, 'cnn_backbone', 'convnext_base')
            model_str = model_name.replace('/', '_').replace('-', '_')
            prefix = 'cnn'
        elif self.feature_extractor_type == 'timm':
            model_name = getattr(self.ggeur_cfg, 'timm_model', 'gfnet_tiny')
            model_str = str(model_name).replace('/', '_').replace('-', '_')
            prefix = 'timm'
        elif self.feature_extractor_type == 'bert':
            model_name = os.path.basename(
                str(getattr(self.ggeur_cfg, 'bert_model_path',
                            'bert')).rstrip('/\\')) or 'bert'
            mode = 'pre' if getattr(
                self.ggeur_cfg, 'bert_use_pretrained_weights',
                True) else f"rand_seed{int(getattr(self._cfg, 'seed', 0))}"
            pooling = str(getattr(self.ggeur_cfg, 'bert_pooling', 'cls'))
            max_len = int(getattr(self.ggeur_cfg, 'bert_max_length', 128))
            model_str = (
                f"{model_name}_maxlen{max_len}_{pooling}_{mode}"
            ).replace('/', '_').replace('-', '_')
            prefix = 'bert'
        else:
            clip_model = self.ggeur_cfg.clip_model.replace('/', '_').replace('-', '_')
            pretrained = self.ggeur_cfg.clip_pretrained.replace('/', '_').replace('-', '_')
            model_str = f"{clip_model}_{pretrained}"
            prefix = 'clip'

        # Include split params in filename to ensure cache invalidation when params change
        split_str = f"split{int(splits[0]*100)}_{int(splits[1]*100)}_{int(100-splits[0]*100-splits[1]*100)}_seed{seed}"
        cache_filename = f"{dataset_name}_{domain}_test_{prefix}_{model_str}_{split_str}{cache_suffix}.npz"

        return os.path.join(cache_dir, cache_filename)

    def _feature_cache_key(self, sample_id):
        """Normalize a sample id exactly as distributed clients do."""
        value = str(sample_id).replace('\\', '/').strip()
        while '//' in value:
            value = value.replace('//', '/')
        root = str(getattr(self._cfg.data, 'root', '') or '')
        root = root.replace('\\', '/').rstrip('/')
        while '//' in root:
            root = root.replace('//', '/')
        if root:
            folded = value.casefold()
            root_folded = root.casefold()
            if folded == root_folded:
                value = ''
            elif folded.startswith(root_folded + '/'):
                value = value[len(root) + 1:]
            else:
                root_name = root.rsplit('/', 1)[-1]
                marker = f'/{root_name.casefold()}/'
                marker_index = folded.find(marker)
                if marker_index >= 0:
                    value = value[marker_index + len(marker):]
        return value.lstrip('./').casefold()

    def _get_full_feature_cache_path(self, domain=None):
        """Return the client-produced complete feature-cache path."""
        cache_dir = str(getattr(
            self.ggeur_cfg, 'feature_cache_dir', '') or '')
        if not cache_dir:
            return None
        dataset_name = self._cfg.data.type.lower()
        if self.feature_extractor_type == 'cnn':
            model_name = getattr(
                self.ggeur_cfg, 'cnn_backbone', 'convnext_base')
            model_str = str(model_name).replace('/', '_').replace('-', '_')
            prefix = 'cnn'
        elif self.feature_extractor_type == 'timm':
            model_name = getattr(self.ggeur_cfg, 'timm_model', 'gfnet_tiny')
            model_str = str(model_name).replace('/', '_').replace('-', '_')
            prefix = 'timm'
        elif self.feature_extractor_type == 'bert':
            model_name = os.path.basename(str(getattr(
                self.ggeur_cfg, 'bert_model_path', 'bert')).rstrip('/\\')) \
                or 'bert'
            mode = 'pre' if getattr(
                self.ggeur_cfg, 'bert_use_pretrained_weights', True) else \
                f"rand_seed{int(getattr(self._cfg, 'seed', 0))}"
            pooling = str(getattr(
                self.ggeur_cfg, 'bert_pooling', 'cls'))
            max_len = int(getattr(
                self.ggeur_cfg, 'bert_max_length', 128))
            model_str = (
                f"{model_name}_maxlen{max_len}_{pooling}_{mode}"
            ).replace('/', '_').replace('-', '_')
            prefix = 'bert'
        else:
            clip_model = self.ggeur_cfg.clip_model.replace(
                '/', '_').replace('-', '_')
            pretrained = self.ggeur_cfg.clip_pretrained.replace(
                '/', '_').replace('-', '_')
            model_str = f"{clip_model}_{pretrained}"
            prefix = 'clip'
        dim_str = f"d{int(self._get_embedding_dim())}"
        name = (f"{dataset_name}_{domain}_{prefix}_{model_str}_{dim_str}.npz"
                if domain else
                f"{dataset_name}_{prefix}_{model_str}_{dim_str}.npz")
        return os.path.join(cache_dir, name)

    def _load_full_feature_cache(self, domain=None):
        """Load a portable complete cache without loading an extractor."""
        cache_path = self._get_full_feature_cache_path(domain)
        if not cache_path or not os.path.exists(cache_path):
            return {}, cache_path
        data = np.load(cache_path, allow_pickle=True)
        if 'paths' not in data.files or 'features' not in data.files:
            raise ValueError(f'Invalid feature cache format: {cache_path}')
        expected_dim = int(self._get_embedding_dim())
        result = {}
        for path, feature in zip(data['paths'], data['features']):
            array = np.asarray(feature)
            if array.ndim == 1 and array.shape[0] == expected_dim:
                result[self._feature_cache_key(path)] = array
        logger.info(
            f"Server: Loaded {len(result)} complete cached features from "
            f"{cache_path}")
        return result, cache_path

    @torch.no_grad()
    def _encode_text_batch(self, texts):
        encoded = self.bert_tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=int(getattr(self.ggeur_cfg, 'bert_max_length', 128)),
            return_tensors='pt')
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        outputs = self.bert_model(**encoded)
        hidden = outputs.last_hidden_state
        pooling = str(getattr(self.ggeur_cfg, 'bert_pooling',
                              'cls')).lower()
        if pooling == 'mean':
            mask = encoded.get('attention_mask')
            if mask is None:
                features = hidden.mean(dim=1)
            else:
                mask = mask.unsqueeze(-1).float()
                features = (hidden * mask).sum(dim=1) / mask.sum(
                    dim=1).clamp(min=1e-6)
        else:
            features = hidden[:, 0, :]
        return features.detach().cpu().numpy().astype(np.float32)

    def _load_mdsent_test_data_and_features(self):
        """Load MDSent held-out test splits and extract BERT features."""
        from federatedscope.contrib.data.mdsent_data import \
            build_mdsent_test_sets

        test_sets = build_mdsent_test_sets(self._cfg)
        batch_size = int(getattr(self.ggeur_cfg, 'bert_batch_size', 32))

        for domain, dataset in test_sets.items():
            cache_path = self._get_test_cache_path(domain)
            if os.path.exists(cache_path):
                try:
                    cached = np.load(cache_path, allow_pickle=True)
                    self.test_features[domain] = cached['features']
                    self.test_labels[domain] = cached['labels']
                    logger.info(
                        f"Server: Loaded {len(self.test_labels[domain])} "
                        f"cached BERT test features for {domain}")
                    continue
                except Exception as error:
                    logger.warning(
                        f"Server: Failed to load BERT test cache for "
                        f"{domain}: {error}")

            full_cache, full_cache_path = self._load_full_feature_cache(domain)
            samples = [
                (dataset.texts[idx], int(dataset.targets[idx]),
                 self._feature_cache_key(dataset.get_id(idx)))
                for idx in range(len(dataset))
            ]
            missing = [item for item in samples if item[2] not in full_cache]
            if missing and bool(getattr(
                    self.ggeur_cfg, 'require_complete_feature_cache', False)):
                preview = ', '.join(item[2] for item in missing[:3])
                raise RuntimeError(
                    f"Server: complete BERT feature cache is missing "
                    f"{len(missing)}/{len(samples)} {domain} test samples "
                    f"from {full_cache_path} (first keys: {preview})")
            if missing:
                # Extractor loading is deliberately lazy: a complete cache
                # makes formal CPU-only roots independent of BERT weights.
                self._load_bert_extractor()
                for start in range(0, len(missing), max(batch_size, 1)):
                    batch = missing[start:start + max(batch_size, 1)]
                    encoded = self._encode_text_batch(
                        [item[0] for item in batch])
                    for feature, (_, _, key) in zip(encoded, batch):
                        full_cache[key] = feature

            features_list = [full_cache[item[2]] for item in samples]
            labels = [item[1] for item in samples]

            if not features_list:
                logger.warning(f"Server: No MDSent test data for {domain}")
                continue
            self.test_features[domain] = np.asarray(
                features_list, dtype=np.float32)
            self.test_labels[domain] = np.asarray(labels, dtype=np.int64)
            np.savez(cache_path,
                     features=self.test_features[domain],
                     labels=self.test_labels[domain])
            logger.info(
                f"Server: Extracted and cached "
                f"{len(self.test_labels[domain])} BERT test features for "
                f"{domain}")

        self.test_data_loaded = True
        logger.info(
            f"Server: Loaded MDSent test data for "
            f"{len(self.test_features)} domains")

    def _load_test_data_and_features(self):
        """Load test data from all domains and extract CLIP features"""
        if self.test_data_loaded:
            return

        logger.info("Server: Loading test data from all domains...")

        data_type = self._cfg.data.type.lower()
        data_root = self._cfg.data.root

        if data_type == 'mdsent':
            self._load_mdsent_test_data_and_features()
            return

        # Get the same split ratios and seed as client data loading
        if hasattr(self._cfg.data, 'splits'):
            splits = tuple(self._cfg.data.splits)
        else:
            # Default splits based on dataset type
            if 'office' in data_type and 'home' in data_type:
                splits = (0.7, 0.0, 0.3)  # Same as ggeur_data.py
            else:
                splits = (0.8, 0.1, 0.1)

        train_ratio, val_ratio = splits[0], splits[1]
        configured_split_seed = int(getattr(
            self.ggeur_cfg, 'data_split_seed', -1))
        seed = (int(self._cfg.seed) if configured_split_seed < 0
                else configured_split_seed)

        logger.info(f"Server: Using splits={splits}, seed={seed} (same as client data)")

        # Determine domains based on dataset type
        records_by_domain = None
        if 'pacs' in data_type:
            domains = ['photo', 'art_painting', 'cartoon', 'sketch']
            from federatedscope.cv.dataset.pacs import PACS
            dataset_class = PACS
            dataset_kwargs = {}
        elif 'office' in data_type and 'home' in data_type:
            domains = ['Art', 'Clipart', 'Product', 'Real_World']
            from federatedscope.cv.dataset.office_home import OfficeHome
            dataset_class = OfficeHome
            dataset_kwargs = {}
        elif 'domainnet' in data_type or 'domain-net' in data_type or \
                'domain_net' in data_type:
            from federatedscope.cv.dataset.domainnet import DomainNet
            domains, classes = self._get_domainnet_eval_metadata()
            dataset_class = DomainNet
            dataset_kwargs = {'classes': classes}
            records_by_domain = self._get_domainnet_eval_records()
            cache_meta = {
                'class_count': len(classes),
                'shared_classes_only': int(getattr(
                    self.ggeur_cfg, 'domainnet_shared_classes_only', False)),
                'selected_domains': '|'.join(
                    list(getattr(self.ggeur_cfg, 'domainnet_domains', [])))
            }
        elif data_type in ('digits-3domain', 'digits_3domain', 'digit3',
                            'digit-three-domain'):
            from federatedscope.cv.dataset.digit_three_domain import (
                DigitThreeDomain, load_digit_three_domain_manifest)
            manifest_path = str(getattr(
                self.ggeur_cfg, 'digit3_global_manifest_path', '') or '')
            selected_domains = list(getattr(
                self.ggeur_cfg, 'digit3_domains', []))
            _, domains, classes = load_digit_three_domain_manifest(
                manifest_path, selected_domains)
            dataset_class = DigitThreeDomain
            dataset_kwargs = {'manifest_path': manifest_path}
            cache_meta = {
                'class_count': len(classes),
                'shared_classes_only': -1,
                'selected_domains': '|'.join(domains),
            }
        elif 'office' in data_type and 'caltech' in data_type:
            domains = ['amazon', 'caltech', 'dslr', 'webcam']
            from federatedscope.cv.dataset.office_caltech import OfficeCaltech10
            dataset_class = OfficeCaltech10
            dataset_kwargs = {}
            cache_meta = {}
        else:
            logger.warning(f"Server: Unknown dataset type {data_type}, skipping test evaluation")
            return
        if 'domainnet' not in data_type and 'domain-net' not in data_type and \
                'domain_net' not in data_type:
            cache_meta = {}

        # Load test data for each domain
        for domain in domains:
            cache_path = self._get_test_cache_path(domain)

            # Try to load from cache first
            if os.path.exists(cache_path):
                try:
                    data = np.load(cache_path)
                    if cache_meta:
                        cached_class_count = int(data['class_count']) if 'class_count' in data.files else -1
                        cached_shared = int(data['shared_classes_only']) if 'shared_classes_only' in data.files else -1
                        cached_domains = str(data['selected_domains']) if 'selected_domains' in data.files else ''
                        if cached_class_count != cache_meta['class_count'] or \
                                cached_shared != cache_meta['shared_classes_only'] or \
                                cached_domains != cache_meta['selected_domains']:
                            raise ValueError('DomainNet test cache metadata mismatch')
                    self.test_features[domain] = data['features']
                    self.test_labels[domain] = data['labels']
                    logger.info(f"Server: Loaded {len(self.test_labels[domain])} cached test features for {domain}")
                    continue
                except Exception as e:
                    logger.warning(f"Server: Failed to load cache for {domain}: {e}")

            # Load test dataset with CLIP normalization (CRITICAL for PromptFL)
            try:
                from torchvision import transforms
                if data_type in ('digits-3domain', 'digits_3domain', 'digit3',
                                 'digit-three-domain') and str(getattr(
                                     self.ggeur_cfg, 'feature_extractor',
                                     'clip')).lower() != 'clip':
                    mean = [0.485, 0.456, 0.406]
                    std = [0.229, 0.224, 0.225]
                else:
                    mean = [0.48145466, 0.4578275, 0.40821073]
                    std = [0.26862954, 0.26130258, 0.27577711]
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)
                ])

                per_domain_kwargs = dict(dataset_kwargs)
                if records_by_domain is not None:
                    per_domain_kwargs['records'] = records_by_domain[domain]
                test_dataset = dataset_class(
                    root=data_root,
                    domain=domain,
                    split='test',
                    transform=transform,
                    train_ratio=train_ratio,
                    val_ratio=val_ratio,
                    seed=seed,
                    **per_domain_kwargs
                )

                if len(test_dataset) == 0:
                    logger.warning(f"Server: No test data for domain {domain}")
                    continue

                # Log the split info for verification
                logger.info(f"Server: {domain} test set has {len(test_dataset)} samples")

                # Client feature caches cover the complete deterministic
                # dataset, not only a client shard.  Hydrate the held-out
                # split directly so a root with no raw images or extractor
                # checkpoint can still perform identical formal evaluation.
                if hasattr(test_dataset, 'data') and hasattr(
                        test_dataset, 'targets'):
                    full_cache, full_cache_path = \
                        self._load_full_feature_cache(domain)
                    keys = [self._feature_cache_key(path)
                            for path in test_dataset.data]
                    missing = [key for key in keys
                               if key not in full_cache]
                    if not missing and keys:
                        self.test_features[domain] = np.asarray(
                            [full_cache[key] for key in keys],
                            dtype=np.float32)
                        self.test_labels[domain] = np.asarray(
                            test_dataset.targets, dtype=np.int64)
                        np.savez(
                            cache_path,
                            features=self.test_features[domain],
                            labels=self.test_labels[domain],
                            **cache_meta)
                        logger.info(
                            f"Server: Hydrated {len(keys)} {domain} test "
                            f"features from complete cache {full_cache_path}")
                        continue
                    if missing and bool(getattr(
                            self.ggeur_cfg,
                            'require_complete_feature_cache', False)):
                        raise RuntimeError(
                            f"Server: complete feature cache is missing "
                            f"{len(missing)}/{len(keys)} {domain} test "
                            f"samples from {full_cache_path}")

                # Extract features using appropriate extractor (CLIP or CNN)
                self._load_feature_extractor()
                features_list = []
                labels_list = []

                dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

                if self.feature_extractor_type == 'cnn':
                    extractor_name = 'CNN'
                elif self.feature_extractor_type == 'timm':
                    extractor_name = 'timm'
                else:
                    extractor_name = 'CLIP'

                with torch.no_grad():
                    for images, labels in dataloader:
                        images = images.to(self.device)
                        if self.feature_extractor_type == 'cnn':
                            features = self.cnn_extractor(images)
                        elif self.feature_extractor_type == 'timm':
                            features = self.timm_extractor(images)
                        else:
                            features = self.clip_model.encode_image(images)
                        features_list.append(features.cpu().numpy())
                        labels_list.append(labels.numpy())

                self.test_features[domain] = np.vstack(features_list)
                self.test_labels[domain] = np.concatenate(labels_list)

                # Save to cache
                np.savez(cache_path,
                        features=self.test_features[domain],
                        labels=self.test_labels[domain],
                        **cache_meta)

                logger.info(f"Server: Extracted and cached {len(self.test_labels[domain])} test features for {domain}")

            except Exception as e:
                if bool(getattr(
                        self.ggeur_cfg,
                        'require_complete_feature_cache', False)):
                    raise
                logger.warning(f"Server: Failed to load test data for {domain}: {e}")
                import traceback
                traceback.print_exc()

        self.test_data_loaded = True
        logger.info(f"Server: Loaded test data for {len(self.test_features)} domains")

    def _evaluate_on_test_sets(self):
        """Evaluate global model on all test sets"""
        if self.global_mlp is None:
            return {}

        # Load test data if not already loaded
        if not self.test_data_loaded:
            self._load_test_data_and_features()

        if not self.test_features:
            return {}

        self.global_mlp.eval()
        results = {}

        with torch.no_grad():
            for domain, features in self.test_features.items():
                labels = self.test_labels[domain]

                features_tensor = torch.from_numpy(features).float().to(self.device)
                labels_tensor = torch.from_numpy(labels).long().to(self.device)

                outputs = self._predict_head_outputs(
                    features_tensor, domain=str(domain))
                _, predicted = torch.max(outputs, 1)

                correct = (predicted == labels_tensor).sum().item()
                total = labels_tensor.size(0)
                accuracy = correct / total if total > 0 else 0

                results[domain] = accuracy

                # Track history
                if domain not in self.test_accuracies_history:
                    self.test_accuracies_history[domain] = []
                self.test_accuracies_history[domain].append(accuracy)

        # Compute average
        if results:
            avg_accuracy = sum(results.values()) / len(results)
            results['average'] = avg_accuracy

            if 'average' not in self.test_accuracies_history:
                self.test_accuracies_history['average'] = []
            self.test_accuracies_history['average'].append(avg_accuracy)

            # Track best model
            if avg_accuracy > self.best_avg_accuracy:
                self.best_avg_accuracy = avg_accuracy
                self.best_model_state = copy.deepcopy(self.global_mlp.state_dict())

        return results

    def _build_domain_prototype_ensemble(self, num_classes):
        """Retain deterministic per-client class centroids for inference."""
        limit = int(getattr(
            self.ggeur_cfg, 'domain_prototype_ensemble_per_class', 0))
        self.domain_prototype_ensemble = {}
        if limit <= 0:
            return
        # The distributed loaders assign consecutive client ranges to
        # consecutive domains.  Use the domain list for the active dataset,
        # rather than limiting grouped centroids to DomainNet.  Without this,
        # OfficeHome and MDSent fell back to the first few individual-client
        # centroids, which is both order-dependent and substantially less
        # accurate under severe Dirichlet skew.
        domain_names = []
        for config_key in (
                'domainnet_domains', 'officehome_domains',
                'digit3_domains'):
            values = list(getattr(self.ggeur_cfg, config_key, []))
            if values:
                domain_names = values
                break
        data_type = str(getattr(self._cfg.data, 'type', '')).lower()
        if not domain_names and data_type in ('office-home', 'officehome'):
            domain_names = ['Art', 'Clipart', 'Product', 'Real_World']
        if not domain_names and data_type == 'mdsent':
            domain_names = ['books', 'dvd', 'electronics', 'kitchen']
        domain_count = len(domain_names)
        clients_per_domain = (
            int(self._client_num) // domain_count
            if domain_count > 0 and int(self._client_num) % domain_count == 0
            else 0)
        for class_idx in range(int(num_classes)):
            candidates = []
            groups = []
            if clients_per_domain > 0:
                groups = [
                    range(offset * clients_per_domain + 1,
                          (offset + 1) * clients_per_domain + 1)
                    for offset in range(domain_count)
                ]
            else:
                groups = [[client_id] for client_id in sorted(
                    self.local_statistics_buffer)]
            for client_ids in groups:
                weighted_sum = None
                total_count = 0
                for client_id in client_ids:
                    statistics = self.local_statistics_buffer.get(
                        int(client_id), {})
                    prototype = self._get_class_value(
                        statistics.get('prototypes', {}), class_idx)
                    count = int(self._get_class_value(
                        statistics.get('counts', {}), class_idx) or 0)
                    if prototype is None or count <= 0:
                        continue
                    value = np.asarray(prototype, dtype=np.float64)
                    if value.shape != (int(self._get_embedding_dim()),):
                        continue
                    weighted_sum = (value * count if weighted_sum is None
                                    else weighted_sum + value * count)
                    total_count += count
                if weighted_sum is None or total_count <= 0:
                    continue
                value = weighted_sum / float(total_count)
                norm = float(np.linalg.norm(value))
                if np.isfinite(norm) and norm > 0:
                    candidates.append((value / norm).astype(np.float32))
            if candidates:
                self.domain_prototype_ensemble[class_idx] = np.stack(
                    candidates[:limit], axis=0)
        logger.info(
            "Server: Built domain prototype ensemble for "
            f"{len(self.domain_prototype_ensemble)}/{int(num_classes)} "
            f"classes (limit={limit}, domains={domain_names})")

    def _build_domain_personalized_heads(self, num_classes):
        """Fit one task-adaptive linear head per configured data domain."""
        epochs = int(getattr(
            self.ggeur_cfg, 'domain_personalized_head_epochs', 0))
        self.domain_personalized_heads = {}
        domains = list(getattr(self.ggeur_cfg, 'domainnet_domains', []))
        if epochs <= 0 or not domains:
            return
        if int(self._client_num) % len(domains) != 0:
            raise ValueError(
                "domain-personalized heads require equal clients per domain")
        clients_per_domain = int(self._client_num) // len(domains)
        feature_dim = int(self._get_embedding_dim())
        batch_size = max(1, int(getattr(
            self.ggeur_cfg, 'domain_personalized_head_batch_size', 2048)))
        learning_rate = float(getattr(
            self.ggeur_cfg, 'domain_personalized_head_lr', 0.01))
        for domain_index, domain in enumerate(domains):
            feature_parts, label_parts = [], []
            use_full_cache = bool(getattr(
                self.ggeur_cfg,
                'domain_personalized_head_use_full_train_cache', False))
            if use_full_cache:
                cache_dir = Path(str(getattr(
                    self.ggeur_cfg, 'feature_cache_dir', '')))
                extractor = str(getattr(
                    self.ggeur_cfg, 'feature_extractor', '')).lower()
                if extractor == 'clip':
                    suffix = 'clip_ViT_B_16_openai_d512'
                elif extractor == 'timm':
                    suffix = 'timm_mixer_b16_224_d768'
                elif extractor == 'cnn':
                    suffix = 'cnn_convnext_base_d1024'
                else:
                    raise ValueError(
                        f"unsupported personalized-head extractor {extractor}")
                matches = list(cache_dir.glob(
                    f"domainnet_{domain}_{suffix}.npz"))
                if len(matches) != 1:
                    raise RuntimeError(
                        f"expected one full training cache for {domain}, "
                        f"found {matches}")
                with np.load(matches[0], allow_pickle=False) as data:
                    features = np.asarray(
                        data['features'], dtype=np.float32)
                    labels = np.asarray(data['labels'], dtype=np.int64)
                norms = np.linalg.norm(features, axis=1, keepdims=True)
                features /= np.maximum(norms, 1e-12)
                feature_parts.append(features)
                label_parts.append(labels)
            else:
                generator = np.random.default_rng(
                    int(getattr(self._cfg, 'seed', 42)) + domain_index)
                for client_id in range(
                        domain_index * clients_per_domain + 1,
                        (domain_index + 1) * clients_per_domain + 1):
                    statistics = self.local_statistics_buffer.get(
                        client_id, {})
                    for class_idx, mean in statistics.get('means', {}).items():
                        class_idx = int(class_idx)
                        count = int(self._get_class_value(
                            statistics.get('counts', {}), class_idx) or 0)
                        covariance = self._get_class_value(
                            statistics.get('covs', {}), class_idx)
                        mean = np.asarray(mean, dtype=np.float32)
                        covariance = np.asarray(covariance, dtype=np.float32)
                        if count <= 0 or mean.shape != (feature_dim,):
                            continue
                        if covariance.shape == (feature_dim, feature_dim):
                            covariance = np.diag(covariance).copy()
                        if covariance.shape != (feature_dim,):
                            continue
                        noise = generator.standard_normal(
                            (count, feature_dim), dtype=np.float32)
                        samples = mean[None, :] + noise * np.sqrt(
                            np.maximum(covariance, 1e-8))[None, :]
                        norms = np.linalg.norm(
                            samples, axis=1, keepdims=True)
                        samples /= np.maximum(norms, 1e-12)
                        feature_parts.append(samples)
                        label_parts.append(np.full(
                            count, class_idx, np.int64))
            if not feature_parts:
                raise RuntimeError(
                    f"no statistics available for domain {domain}")
            features = np.concatenate(feature_parts, axis=0)
            labels = np.concatenate(label_parts, axis=0)
            counts = np.bincount(labels, minlength=int(num_classes)).astype(
                np.float32)
            class_weights = np.sqrt(
                counts.sum() / np.maximum(counts, 1.0))
            class_weights /= max(1e-12, float(class_weights.mean()))
            torch.manual_seed(
                int(getattr(self._cfg, 'seed', 42)) + domain_index)
            model = nn.Linear(feature_dim, int(num_classes)).to(self.device)
            criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(
                class_weights).to(self.device))
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=learning_rate, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, epochs))
            rng = np.random.default_rng(
                int(getattr(self._cfg, 'seed', 42)) + domain_index)
            model.train()
            for _ in range(epochs):
                order = rng.permutation(labels.size)
                for offset in range(0, labels.size, batch_size):
                    index = order[offset:offset + batch_size]
                    batch_features = torch.from_numpy(features[index]).to(
                        self.device)
                    batch_labels = torch.from_numpy(labels[index]).to(
                        self.device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = criterion(model(batch_features), batch_labels)
                    loss.backward()
                    optimizer.step()
                scheduler.step()
            self.domain_personalized_heads[str(domain)] = {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            }
            logger.info(
                f"Server: Built domain-personalized head for {domain} from "
                f"{labels.size} training samples "
                f"(full_train_cache={use_full_cache})")

    def _predict_head_outputs(self, features, domain=''):
        """Return normal head logits or maximum cosine prototype scores."""
        head_state = getattr(
            self, 'domain_personalized_heads', {}).get(str(domain))
        if head_state:
            weight = torch.as_tensor(
                head_state['weight'], device=features.device,
                dtype=features.dtype)
            bias = torch.as_tensor(
                head_state['bias'], device=features.device,
                dtype=features.dtype)
            norms = torch.linalg.vector_norm(
                features, ord=2, dim=1, keepdim=True).clamp_min(1e-12)
            return F.linear(features / norms, weight, bias)
        if not self.domain_prototype_ensemble:
            return self.global_mlp(features)
        outputs = torch.full(
            (features.shape[0], int(self._cfg.model.num_classes)),
            -torch.inf, device=features.device, dtype=features.dtype)
        for class_idx, prototypes in self.domain_prototype_ensemble.items():
            values = torch.as_tensor(
                prototypes, device=features.device, dtype=features.dtype)
            outputs[:, int(class_idx)] = torch.max(
                features @ values.transpose(0, 1), dim=1).values
        return outputs

    def callback_for_local_statistics(self, message: Message):
        """Handle receiving local statistics from a client"""
        timing_total_start = time.time()
        client_id = message.sender
        content = message.content
        _timing_t0 = time.time()
        self._bytes_recv += self._sizeof_content(content)
        timing_sizeof_recv = time.time() - _timing_t0

        if self.statistics_collected:
            logger.warning(
                f"Server: Ignore late local_statistics from client "
                f"{client_id}; statistics phase already completed with "
                f"active_clients={sorted(self.active_client_ids)}")
            return

        logger.info(f"Server: Received local statistics from client {client_id}")

        # Infer embedding dimension from client-reported value (preferred) or from means later
        client_embedding_dim = content.get('embedding_dim', None) if isinstance(content, dict) else None
        if client_embedding_dim is not None:
            try:
                client_embedding_dim = int(client_embedding_dim)
            except Exception:
                client_embedding_dim = None

        if client_embedding_dim:
            if self.inferred_embedding_dim is None:
                self.inferred_embedding_dim = client_embedding_dim
                logger.info(f"Server: Inferred embedding_dim={self.inferred_embedding_dim} from client {client_id}")
            elif int(self.inferred_embedding_dim) != int(client_embedding_dim):
                raise ValueError(
                    f"Embedding dim mismatch across clients: "
                    f"server_inferred={self.inferred_embedding_dim}, client_{client_id}={client_embedding_dim}"
                )

        _timing_t0 = time.time()
        self._validate_received_statistics(client_id, content)
        timing_validate = time.time() - _timing_t0
        _timing_t0 = time.time()
        content = self._normalize_received_statistics(client_id, content)
        timing_normalize = time.time() - _timing_t0

        # Store statistics
        _timing_t0 = time.time()
        self.local_statistics_buffer[client_id] = {
            'means': content['means'],
            'covs': content['covs'],
            'counts': content['counts'],
            'prototypes': content['prototypes']
        }

        # Store prototypes for cross-client sharing
        self.all_prototypes[client_id] = content['prototypes']
        timing_store = time.time() - _timing_t0
        logger.info(
            "GGEUR_TIMING_SERVER "
            f"stage=receive_local_statistics client={int(client_id)} "
            f"total_sec={time.time() - timing_total_start:.6f} "
            f"sizeof_recv_sec={timing_sizeof_recv:.6f} "
            f"validate_sec={timing_validate:.6f} "
            f"normalize_sec={timing_normalize:.6f} "
            f"store_sec={timing_store:.6f} "
            f"buffer_clients={len(self.local_statistics_buffer)} "
            f"classes={len(content['means'])} "
            f"payload_bytes={self._sizeof_content(message.content)}")

        # Check whether the configured statistics quorum has uploaded
        # statistics.  Also support a fully-accounted mixed cache state, where
        # cache-hot clients already reported augmentation_ready and only
        # cache-cold clients need the covariance response.
        (statistics_quorum, mixed_cache_quorum, statistics_clients,
         cache_ready_clients) = self._statistics_quorum_state()
        if statistics_quorum:
            quorum_t0 = time.time()
            self.active_client_ids = statistics_clients | cache_ready_clients
            if mixed_cache_quorum:
                logger.info(
                    "Server: Mixed cache/statistics quorum reached; "
                    f"statistics_clients={sorted(statistics_clients)}, "
                    f"cache_ready_clients={sorted(cache_ready_clients)}, "
                    f"accounted={len(self.active_client_ids)}/"
                    f"{self._client_num}")
            logger.info(
                "Server: Received statistics quorum "
                f"{len(self.local_statistics_buffer)}/{self._client_num}; "
                f"required={self.min_statistics_clients}; "
                f"active_clients={sorted(self.active_client_ids)}")

            # Aggregate covariance matrices
            _timing_t0 = time.time()
            self._aggregate_covariances()
            timing_aggregate = time.time() - _timing_t0

            # Compute global prototypes (aggregated means for each class)
            _timing_t0 = time.time()
            self._compute_global_prototypes()
            timing_global_proto = time.time() - _timing_t0

            # For large standalone runs, avoid materializing O(N^2)
            # per-client prototype payloads unless explicit down-sampling is
            # requested. The broadcast path can share the full prototype pool
            # and let each client filter out its own entries.
            max_cross_prototypes = int(getattr(
                self.ggeur_cfg, 'max_cross_client_prototypes_per_class', 0))
            _timing_t0 = time.time()
            other_prototypes = (
                self._prepare_other_prototypes()
                if max_cross_prototypes > 0 else None
            )
            timing_prepare_other = time.time() - _timing_t0

            # Build global MLP
            # IMPORTANT: Use config's num_classes, not the number of classes in covariance matrices
            # In LDS mode, some classes may have no data across all clients
            num_classes = self._cfg.model.num_classes
            _timing_t0 = time.time()
            if num_classes > 0:
                self._build_global_mlp(num_classes)
                self._build_domain_prototype_ensemble(num_classes)
                self._build_domain_personalized_heads(num_classes)

                # Build global CNN if using CNN mode
                if self.use_cnn_distillation or self.use_feature_alignment:
                    self._build_global_cnn(num_classes)

                # Initialize global prompt ctx if PromptFL enabled
                if self.use_promptfl:
                    self._init_global_prompt(num_classes)
            timing_build_models = time.time() - _timing_t0

            # Broadcast global covariances to all clients
            _timing_t0 = time.time()
            self._broadcast_global_covariances(other_prototypes)
            timing_broadcast = time.time() - _timing_t0

            self.statistics_collected = True
            logger.info(
                "GGEUR_TIMING_SERVER "
                f"stage=statistics_quorum total_sec={time.time() - quorum_t0:.6f} "
                f"aggregate_covariances_sec={timing_aggregate:.6f} "
                f"global_prototypes_sec={timing_global_proto:.6f} "
                f"prepare_other_prototypes_sec={timing_prepare_other:.6f} "
                f"build_models_sec={timing_build_models:.6f} "
                f"broadcast_sec={timing_broadcast:.6f} "
                f"clients={len(self.local_statistics_buffer)} "
                f"cov_classes={len(self.global_cov_matrices)} "
                f"global_proto_classes={len(self.global_prototypes)}")

    def _validate_received_statistics(self, client_id, content):
        """Fail early when a client uploads statistics with wrong dimensions."""
        if not isinstance(content, dict):
            raise ValueError(
                f"Server: local_statistics from client {client_id} must be "
                f"a dict, got {type(content)}.")

        expected_dim = int(self._get_embedding_dim())
        diagonal_covariance = bool(getattr(
            self.ggeur_cfg, 'diagonal_covariance', False))
        expected_cov_shape = ((expected_dim, ) if diagonal_covariance else
                              (expected_dim, expected_dim))
        invalid = []
        means = content.get('means', {})
        covs = content.get('covs', {})
        counts = content.get('counts', {})

        for class_idx, mean in means.items():
            mean_arr = self._payload_to_ndarray(mean)
            cov = self._get_class_value(covs, class_idx)
            count = self._get_class_value(counts, class_idx)
            cov_arr = self._payload_to_ndarray(cov)
            if mean_arr is None:
                invalid.append(f"class {class_idx}: mean decode failed")
                continue
            if cov_arr is None:
                invalid.append(f"class {class_idx}: cov decode failed")
                continue
            mean_arr = np.asarray(mean_arr)
            cov_arr = np.asarray(cov_arr)
            if mean_arr.shape != (expected_dim, ):
                invalid.append(
                    f"class {class_idx}: mean shape {mean_arr.shape}")
                continue
            if cov_arr.shape != expected_cov_shape:
                invalid.append(f"class {class_idx}: cov shape {cov_arr.shape}")
            if count is None:
                invalid.append(f"class {class_idx}: missing count")

        if invalid:
            detail = '; '.join(invalid[:5])
            raise ValueError(
                f"Server: Invalid local_statistics from client {client_id}. "
                f"Expected embedding_dim={expected_dim}. {detail}. "
                f"Please clear stale feature cache or align all clients to "
                f"the same feature_extractor/embedding_dim.")

    def _normalize_received_statistics(self, client_id, content):
        """Decode compact local-statistics payloads from a client."""
        normalized = dict(content)
        expected_dim = int(self._get_embedding_dim())

        means = {}
        for class_idx, mean in content.get('means', {}).items():
            key = int(class_idx)
            means[key] = np.asarray(
                self._payload_to_ndarray(mean), dtype=np.float32)

        covs = {}
        for class_idx, cov in content.get('covs', {}).items():
            key = int(class_idx)
            covs[key] = np.asarray(
                self._payload_to_ndarray(cov), dtype=np.float32)

        counts = {}
        for class_idx, count in content.get('counts', {}).items():
            counts[int(class_idx)] = int(count)

        prototypes = {}
        for class_idx, proto in content.get('prototypes', {}).items():
            key = int(class_idx)
            values = proto if isinstance(proto, list) else [proto]
            decoded = []
            for value in values:
                proto_arr = np.asarray(
                    self._payload_to_ndarray(value), dtype=np.float32)
                if proto_arr.shape != (expected_dim, ):
                    raise ValueError(
                        f"Server: Invalid prototype from client {client_id}, "
                        f"class {class_idx}: shape {proto_arr.shape}, "
                        f"expected ({expected_dim},).")
                decoded.append(proto_arr)
            prototypes[key] = decoded if isinstance(proto, list) else decoded[0]

        normalized['means'] = means
        normalized['covs'] = covs
        normalized['counts'] = counts
        normalized['prototypes'] = prototypes
        return normalized

    @staticmethod
    def _get_class_value(mapping, class_idx):
        """Read class-keyed stats regardless of int/string key serialization."""
        if not isinstance(mapping, dict):
            return None
        if class_idx in mapping:
            return mapping[class_idx]
        str_key = str(class_idx)
        if str_key in mapping:
            return mapping[str_key]
        try:
            int_key = int(class_idx)
        except (TypeError, ValueError):
            return None
        return mapping.get(int_key)

    @staticmethod
    def _payload_to_ndarray(payload):
        """Decode compact gRPC payloads back to ndarray-compatible values."""
        if isinstance(payload, bytes):
            payload = payload.decode('utf-8')
        if isinstance(payload, str):
            if payload.startswith('znp:'):
                try:
                    raw = zlib.decompress(base64.b64decode(payload[4:]))
                    return np.load(io.BytesIO(raw), allow_pickle=False)
                except Exception:
                    return None
            try:
                payload = param2tensor(payload)
            except Exception:
                return None
        if isinstance(payload, torch.Tensor):
            return payload.detach().cpu().numpy()
        return payload

    def _ensure_stat_shapes(self, client_id, class_idx, mean, cov, count,
                            embedding_dim):
        mean_arr = np.asarray(mean)
        cov_arr = np.asarray(cov)
        if mean_arr.shape != (embedding_dim, ):
            raise ValueError(
                f"Server: Invalid statistic during aggregation from client "
                f"{client_id}, class {class_idx}: mean shape "
                f"{mean_arr.shape}, expected ({embedding_dim},). "
                f"Clear stale feature cache or align all clients to the same "
                f"feature_extractor/embedding_dim.")
        diagonal_covariance = bool(getattr(
            self.ggeur_cfg, 'diagonal_covariance', False))
        expected_cov_shape = ((embedding_dim, ) if diagonal_covariance else
                              (embedding_dim, embedding_dim))
        if cov_arr.shape != expected_cov_shape:
            raise ValueError(
                f"Server: Invalid statistic during aggregation from client "
                f"{client_id}, class {class_idx}: cov shape {cov_arr.shape}, "
                f"expected {expected_cov_shape}.")
        if count is None:
            raise ValueError(
                f"Server: Missing count during aggregation from client "
                f"{client_id}, class {class_idx}.")
        return mean_arr, cov_arr, int(count)

    def callback_for_augmentation_ready(self, message: Message):
        """Handle client signaling augmentation is complete"""
        client_id = message.sender
        if self.active_client_ids and client_id not in self.active_client_ids:
            logger.warning(
                f"Server: Ignore augmentation_ready from inactive client "
                f"{client_id}; active_clients={sorted(self.active_client_ids)}")
            return
        if self.training_client_ids:
            logger.warning(
                f"Server: Ignore late augmentation_ready from client "
                f"{client_id}; training already started with "
                f"clients={sorted(self.training_client_ids)}")
            return
        self._restore_cached_global_prototypes(message.content, client_id)
        self.augmentation_ready_clients.add(client_id)

        expected_ready = self._expected_augmentation_clients()
        logger.info(
            f"Server: Client {client_id} augmentation ready "
            f"({len(self.augmentation_ready_clients)}/{expected_ready}; "
            f"configured_clients={self._client_num})")

        # When the configured augmentation quorum is ready, start training.
        if len(self.augmentation_ready_clients) >= expected_ready:
            self.training_client_ids = set(self.augmentation_ready_clients)
            logger.info(
                "Server: Augmentation quorum reached; training_clients="
                f"{sorted(self.training_client_ids)}")
            if self.global_mlp is None:
                num_classes = self._cfg.model.num_classes
                if num_classes > 0:
                    self._build_global_mlp(num_classes)
            # Round 0 performs statistics/augmentation preparation rather than
            # a model update.  Evaluate the initialized classifier here so a
            # 100-round run always emits a complete Round 0..99 accuracy
            # series without changing any training parameters or updates.
            if (self._should_run_eval(0) and
                    self._eval_mode() in ('server', 'both')):
                initial_results = self._evaluate_on_test_sets()
                if initial_results:
                    acc_str = ', '.join(
                        f"{key}: {value:.4f}"
                        for key, value in initial_results.items())
                    logger.info(
                        f"Server: Round 0 MLP Test Accuracy - {acc_str}")
            if self.use_fedopt:
                logger.info("Server: All clients ready, starting FedOpt training...")
            else:
                logger.info("Server: All clients ready, starting FedAvg training...")
            self.state = 1  # Move to round 1
            self._start_training_round()

    def _restore_cached_global_prototypes(self, content, client_id):
        """Restore round-0 prototype context from a cache-hot client."""
        if (not bool(getattr(
                self.ggeur_cfg, 'prototype_classifier_init', False))
                or self.global_prototypes or not isinstance(content, dict)):
            return
        serialized = content.get('global_prototypes')
        if not isinstance(serialized, dict):
            return

        input_dim = int(self._get_embedding_dim())
        restored = {}
        for class_idx, payload in serialized.items():
            try:
                class_idx = int(class_idx)
            except (TypeError, ValueError):
                continue
            array = self._payload_to_ndarray(payload)
            if array is None:
                continue
            array = np.asarray(array, dtype=np.float32)
            if array.shape == (input_dim, ):
                restored[class_idx] = array
        if restored:
            self.global_prototypes = restored
            logger.info(
                "Server: Restored cached global prototypes from client "
                f"{client_id} ({len(restored)} classes)")

    def _start_training_round(self):
        """Start a new training round by broadcasting model"""
        logger.info(f"Server: Starting training round {self.state}")

        # Check for phase transition in separated training mode
        if self.use_separated_training:
            if self.state == self.classifier_pretrain_rounds + 1 and self.training_phase == 'classifier':
                # Transition from Phase 1 to Phase 2
                self._transition_to_cnn_phase()

        # Prepare model parameters based on current mode and phase
        if self.use_separated_training:
            model_para = self._prepare_separated_training_params()
        elif self.use_cnn_distillation or self.use_feature_alignment:
            # Send both MLP and CNN parameters (for distillation or feature alignment)
            model_para = {
                'mlp': copy.deepcopy(self.global_mlp.state_dict()) if self.global_mlp else None,
                'cnn': copy.deepcopy(self.global_cnn.state_dict()) if self.global_cnn else None
            }
        else:
            # Standard mode: only MLP
            if self.global_mlp is not None:
                model_para = copy.deepcopy(self.global_mlp.state_dict())
            else:
                model_para = None

        # Attach global prompt ctx if PromptFL enabled
        if self.use_promptfl and self.global_prompt_ctx is not None:
            if not isinstance(model_para, dict):
                model_para = {'mlp': model_para}
            model_para['prompt'] = {'ctx': self.global_prompt_ctx.cpu()}

        if self.use_fedproto:
            prototype_payload = copy.deepcopy(
                self.fedproto_global_prototypes)
            is_wrapped_payload = isinstance(model_para, dict) and any(
                key in model_para
                for key in ('mlp', 'cnn', 'classifier', 'phase'))
            if is_wrapped_payload:
                model_para = copy.deepcopy(model_para)
                model_para['fedproto_global_prototypes'] = prototype_payload
            else:
                model_para = {
                    'mlp': copy.deepcopy(model_para),
                    'fedproto_global_prototypes': prototype_payload,
                }

        # Broadcast to active training clients only. In strict mode this is all
        # configured clients; in fault-tolerance validation it is the quorum
        # that completed statistics and augmentation.
        receivers = self._active_clients_for_broadcast()
        send_bytes = self._sizeof_content(model_para)
        self._bytes_sent += send_bytes * len(receivers)
        self._round_start_time = time.time()
        logger.info(
            f"Server: Broadcasting round {self.state} model to "
            f"{len(receivers)} clients: {receivers}")
        self._broadcast_identical_to_clients('model_para', receivers,
                                             self.state, model_para)
        self._mark_stage(f'round_{self.state}_model_updates')

    def _prepare_separated_training_params(self):
        """Prepare model parameters for separated training mode"""
        if self.training_phase == 'classifier':
            # Phase 1: Only send classifier (MLP) parameters
            logger.info(f"Server: Phase 1 (Classifier Training) - Round {self.state}/{self.classifier_pretrain_rounds}")
            return {
                'phase': 'classifier',
                'classifier': copy.deepcopy(self.global_mlp.state_dict()) if self.global_mlp else None
            }
        else:
            # Phase 2: Send frozen classifier + CNN backbone
            logger.info(f"Server: Phase 2 (CNN Backbone Training) - Round {self.state}")
            return {
                'phase': 'cnn_backbone',
                'classifier': copy.deepcopy(self.pretrained_classifier) if self.pretrained_classifier else None,
                'cnn_backbone': copy.deepcopy(self.global_cnn.state_dict()) if self.global_cnn else None,
                'freeze_classifier': self.freeze_classifier
            }

    def _transition_to_cnn_phase(self):
        """Transition from classifier training to CNN backbone training"""
        logger.info("=" * 60)
        logger.info("Server: Transitioning to Phase 2 - CNN Backbone Training")
        logger.info("=" * 60)

        # Save the pretrained classifier
        if self.global_mlp is not None:
            self.pretrained_classifier = copy.deepcopy(self.global_mlp.state_dict())
            logger.info(f"Server: Saved pretrained classifier (Best MLP accuracy: {self.best_avg_accuracy:.4f})")
            logger.info(f"Server: Classifier state_dict keys: {list(self.pretrained_classifier.keys())}")
        else:
            logger.error("Server: global_mlp is None! Cannot save pretrained classifier!")
            return

        # Build CNN backbone (without classifier - will use pretrained one)
        self._build_cnn_backbone_only()

        # Update phase
        self.training_phase = 'cnn_backbone'
        logger.info(f"Server: Phase transition complete. Now in '{self.training_phase}' phase")

    def _build_cnn_backbone_only(self):
        """Build CNN backbone for separated training (Phase 2)"""
        from federatedscope.contrib.model.ggeur_cnn import GGEUR_CNN_Backbone

        num_classes = self._cfg.model.num_classes
        cnn_model_name = getattr(self.ggeur_cfg, 'cnn_model', 'resnet18')

        # Always train from scratch in separated training mode
        self.global_cnn = GGEUR_CNN_Backbone(
            model_name=cnn_model_name,
            num_classes=num_classes,
            pretrained=False  # From scratch
        )
        self.global_cnn = self.global_cnn.to(self.device)
        logger.info(f"Server: Built CNN backbone ({cnn_model_name}) for Phase 2 - FROM SCRATCH")

    def _build_classifier_from_state_dict(self, state_dict):
        """Build classifier model from saved state dict for evaluation"""
        num_classes = self._cfg.model.num_classes
        input_dim = None
        hidden_dim = self.ggeur_cfg.mlp_hidden_dim

        # Infer input dim from state dict when possible (more robust than cfg.ggeur.embedding_dim)
        try:
            # Linear classifier
            if 'weight' in state_dict and hasattr(state_dict['weight'], 'shape'):
                input_dim = int(state_dict['weight'].shape[1])
            else:
                # Sequential: pick the smallest numeric prefix layer's weight
                candidates = []
                for k, v in state_dict.items():
                    if not hasattr(v, 'shape'):
                        continue
                    if len(v.shape) != 2:
                        continue
                    m = re.match(r'^(\\d+)\\.weight$', str(k))
                    if m:
                        candidates.append((int(m.group(1)), int(v.shape[1])))
                if candidates:
                    candidates.sort(key=lambda x: x[0])
                    input_dim = candidates[0][1]
        except Exception:
            input_dim = None

        if input_dim is None:
            input_dim = self._get_embedding_dim()

        if hidden_dim > 0:
            classifier = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.ggeur_cfg.mlp_dropout),
                nn.Linear(hidden_dim, num_classes)
            )
        else:
            classifier = nn.Linear(input_dim, num_classes)

        classifier = classifier.to(self.device)

        try:
            classifier.load_state_dict(state_dict)
            logger.debug(f"Server: Classifier loaded successfully, keys: {list(state_dict.keys())}")
        except Exception as e:
            logger.error(f"Server: Failed to load classifier state dict: {e}")
            logger.error(f"Server: Expected keys: {list(classifier.state_dict().keys())}")
            logger.error(f"Server: Received keys: {list(state_dict.keys())}")

        return classifier

    def _aggregate_covariances(self):
        """Aggregate covariance matrices using parallel axis theorem"""
        logger.info("Server: Aggregating covariance matrices...")
        timing_total_start = time.time()
        timing_collect_classes = 0.0
        timing_collect_stats = 0.0
        timing_mean = 0.0
        timing_cov_weighted = 0.0
        timing_cov_between = 0.0

        # Collect all class indices
        _timing_t0 = time.time()
        all_classes = set()
        for client_stats in self.local_statistics_buffer.values():
            all_classes.update(client_stats['means'].keys())
        timing_collect_classes += time.time() - _timing_t0

        embedding_dim = self._get_embedding_dim()
        diagonal_covariance = bool(getattr(
            self.ggeur_cfg, 'diagonal_covariance', False))
        stat_entries = 0

        for class_idx in all_classes:
            class_idx = int(class_idx)
            means = []
            covs = []
            counts = []

            # Collect statistics for this class from all clients
            _timing_t0 = time.time()
            for client_id, client_stats in self.local_statistics_buffer.items():
                mean = self._get_class_value(client_stats['means'], class_idx)
                if mean is not None:
                    cov = self._get_class_value(client_stats['covs'],
                                                class_idx)
                    count = self._get_class_value(client_stats['counts'],
                                                  class_idx)
                    mean, cov, count = self._ensure_stat_shapes(
                        client_id, class_idx, mean, cov, count, embedding_dim)
                    means.append(mean)
                    covs.append(cov)
                    counts.append(count)
                    stat_entries += 1
            timing_collect_stats += time.time() - _timing_t0

            if len(counts) == 0:
                if diagonal_covariance:
                    self.global_cov_matrices[class_idx] = \
                        np.full(embedding_dim, 0.01, dtype=np.float32)
                else:
                    self.global_cov_matrices[class_idx] = \
                        np.eye(embedding_dim) * 0.01
                continue

            # Compute aggregated mean
            _timing_t0 = time.time()
            total_count = sum(counts)
            aggregated_mean = np.zeros(embedding_dim)
            for i, (mean, count) in enumerate(zip(means, counts)):
                aggregated_mean += count * mean
            aggregated_mean /= total_count
            timing_mean += time.time() - _timing_t0

            # Compute aggregated covariance using parallel axis theorem
            if diagonal_covariance:
                aggregated_cov = np.zeros(embedding_dim)
            else:
                aggregated_cov = np.zeros((embedding_dim, embedding_dim))

            # First term: weighted average of local covariances
            _timing_t0 = time.time()
            for i, (cov, count) in enumerate(zip(covs, counts)):
                aggregated_cov += count * cov
            timing_cov_weighted += time.time() - _timing_t0

            # Second term: between-client variance
            _timing_t0 = time.time()
            for i, (mean, count) in enumerate(zip(means, counts)):
                diff = mean - aggregated_mean
                if diagonal_covariance:
                    aggregated_cov += count * diff * diff
                else:
                    aggregated_cov += count * np.outer(diff, diff)
            timing_cov_between += time.time() - _timing_t0

            aggregated_cov /= total_count

            self.global_cov_matrices[class_idx] = aggregated_cov

        logger.info(f"Server: Aggregated covariances for {len(self.global_cov_matrices)} classes")
        logger.info(
            "GGEUR_TIMING_SERVER "
            f"stage=aggregate_covariances total_sec={time.time() - timing_total_start:.6f} "
            f"collect_classes_sec={timing_collect_classes:.6f} "
            f"collect_stats_sec={timing_collect_stats:.6f} "
            f"mean_sec={timing_mean:.6f} "
            f"cov_weighted_sec={timing_cov_weighted:.6f} "
            f"cov_between_sec={timing_cov_between:.6f} "
            f"classes={len(self.global_cov_matrices)} "
            f"clients={len(self.local_statistics_buffer)} "
            f"stat_entries={stat_entries} embedding_dim={embedding_dim}")

    def _compute_global_prototypes(self):
        """Compute global prototypes (weighted average of local means) for each class"""
        logger.info("Server: Computing global prototypes...")
        timing_total_start = time.time()
        timing_collect_classes = 0.0
        timing_collect_stats = 0.0
        timing_mean = 0.0

        embedding_dim = self._get_embedding_dim()

        # Collect all class indices
        _timing_t0 = time.time()
        all_classes = set()
        for client_stats in self.local_statistics_buffer.values():
            all_classes.update(client_stats['means'].keys())
        timing_collect_classes += time.time() - _timing_t0
        stat_entries = 0

        for class_idx in all_classes:
            class_idx = int(class_idx)
            means = []
            counts = []

            # Collect means for this class from all clients
            _timing_t0 = time.time()
            for client_id, client_stats in self.local_statistics_buffer.items():
                mean = self._get_class_value(client_stats['means'], class_idx)
                if mean is not None:
                    count = self._get_class_value(client_stats['counts'],
                                                  class_idx)
                    mean_arr = np.asarray(mean)
                    if mean_arr.shape != (embedding_dim, ):
                        raise ValueError(
                            f"Server: Invalid prototype statistic from client "
                            f"{client_id}, class {class_idx}: mean shape "
                            f"{mean_arr.shape}, expected ({embedding_dim},).")
                    if count is None:
                        raise ValueError(
                            f"Server: Missing prototype count from client "
                            f"{client_id}, class {class_idx}.")
                    means.append(mean_arr)
                    counts.append(int(count))
                    stat_entries += 1
            timing_collect_stats += time.time() - _timing_t0

            if len(counts) == 0:
                continue

            # Compute weighted average mean (global prototype)
            _timing_t0 = time.time()
            total_count = sum(counts)
            global_mean = np.zeros(embedding_dim)
            for mean, count in zip(means, counts):
                global_mean += count * mean
            global_mean /= total_count
            timing_mean += time.time() - _timing_t0

            self.global_prototypes[class_idx] = global_mean

        logger.info(f"Server: Computed global prototypes for {len(self.global_prototypes)} classes")
        logger.info(
            "GGEUR_TIMING_SERVER "
            f"stage=compute_global_prototypes total_sec={time.time() - timing_total_start:.6f} "
            f"collect_classes_sec={timing_collect_classes:.6f} "
            f"collect_stats_sec={timing_collect_stats:.6f} "
            f"mean_sec={timing_mean:.6f} "
            f"classes={len(self.global_prototypes)} "
            f"clients={len(self.local_statistics_buffer)} "
            f"stat_entries={stat_entries} embedding_dim={embedding_dim}")

    def _prepare_other_prototypes(self):
        """Prepare prototypes from other clients for each client"""
        max_per_class = int(getattr(
            self.ggeur_cfg, 'max_cross_client_prototypes_per_class', 0))
        if max_per_class > 0:
            seed = int(getattr(self.ggeur_cfg, 'cross_client_prototype_seed',
                               42))
            prototypes_by_class = {}
            for other_client_id, prototypes in self.all_prototypes.items():
                for class_idx, prototype in prototypes.items():
                    class_idx = int(class_idx)
                    prototypes_by_class.setdefault(class_idx, []).append(
                        (int(other_client_id), prototype))

            other_prototypes = {}
            total_selected = 0
            for client_id in self.all_prototypes.keys():
                client_id = int(client_id)
                other_prototypes[client_id] = {}
                rng = np.random.RandomState(seed + client_id * 1009)

                for class_idx, candidates in prototypes_by_class.items():
                    available = [
                        prototype for other_id, prototype in candidates
                        if int(other_id) != client_id
                    ]
                    if not available:
                        continue
                    if len(available) > max_per_class:
                        picked = rng.choice(len(available),
                                            size=max_per_class,
                                            replace=False)
                        selected = [available[int(idx)] for idx in picked]
                    else:
                        selected = available
                    other_prototypes[client_id][int(class_idx)] = selected
                    total_selected += len(selected)

            logger.info(
                "Server: Prepared limited other-client prototypes: "
                f"clients={len(other_prototypes)}, "
                f"classes={len(prototypes_by_class)}, "
                f"max_per_class={max_per_class}, "
                f"selected_prototypes={total_selected}")
            return other_prototypes

        other_prototypes = {}

        for client_id in self.all_prototypes.keys():
            other_prototypes[client_id] = {}

            for other_client_id, prototypes in self.all_prototypes.items():
                if other_client_id == client_id:
                    continue

                for class_idx, prototype in prototypes.items():
                    class_idx = int(class_idx)
                    if class_idx not in other_prototypes[client_id]:
                        other_prototypes[client_id][class_idx] = []
                    other_prototypes[client_id][class_idx].append(prototype)

        return other_prototypes

    def _broadcast_global_covariances(self, other_prototypes):
        """Broadcast global covariance matrices and prototypes to all clients"""
        logger.info("Server: Broadcasting global covariances to clients...")
        timing_total_start = time.time()
        timing_serialize_cov = 0.0
        timing_serialize_global_proto = 0.0
        timing_serialize_all_proto = 0.0
        timing_serialize_client_other = 0.0
        timing_sizeof = 0.0
        timing_send = 0.0

        # Only clients that uploaded statistics are waiting for this response.
        # Cache-hot clients may already be augmentation-ready and broadcasting
        # to them can trigger redundant generation and duplicate readiness.
        receivers = self._statistics_broadcast_receivers()
        _timing_t0 = time.time()
        serialized_cov_matrices = self._serialize_array_payload(
            self.global_cov_matrices)
        timing_serialize_cov += time.time() - _timing_t0
        _timing_t0 = time.time()
        serialized_global_prototypes = self._serialize_array_payload(
            self.global_prototypes)
        timing_serialize_global_proto += time.time() - _timing_t0
        serialized_all_prototypes = None
        if other_prototypes is None:
            _timing_t0 = time.time()
            serialized_all_prototypes = self._serialize_array_payload(
                self.all_prototypes)
            timing_serialize_all_proto += time.time() - _timing_t0

        shared_payload = {
            'cov_matrices': serialized_cov_matrices,
            'global_prototypes': serialized_global_prototypes,
        }
        if serialized_all_prototypes is not None:
            shared_payload['all_prototypes_by_client'] = serialized_all_prototypes
        _timing_t0 = time.time()
        shared_content_size = self._sizeof_content(shared_payload)
        timing_sizeof += time.time() - _timing_t0
        logger.info(
            "Server: Prepared shared covariance broadcast payload: "
            f"classes={len(self.global_cov_matrices)}, "
            f"global_prototypes={len(self.global_prototypes)}, "
            f"all_client_prototypes="
            f"{len(self.all_prototypes) if serialized_all_prototypes is not None else 0}, "
            f"shared_bytes={shared_content_size}")

        if self.hierarchical_training and serialized_all_prototypes is not None:
            content = {
                'cov_matrices': serialized_cov_matrices,
                'global_prototypes': serialized_global_prototypes,
                'all_prototypes_by_client': serialized_all_prototypes,
                'other_prototypes': {},
            }
            self._bytes_sent += shared_content_size * len(receivers)
            logger.info(
                "Server: Sending one hierarchical covariance broadcast to "
                f"{len(receivers)} logical clients")
            _timing_t0 = time.time()
            self._broadcast_identical_to_clients(
                'global_covariances', receivers, self.state, content)
            timing_send += time.time() - _timing_t0
        else:
            for client_id in receivers:
                content = {
                    'cov_matrices': serialized_cov_matrices,
                    'global_prototypes': serialized_global_prototypes,
                }
                if serialized_all_prototypes is not None:
                    content['all_prototypes_by_client'] = serialized_all_prototypes
                    content['other_prototypes'] = {}
                    self._bytes_sent += shared_content_size
                else:
                    _timing_t0 = time.time()
                    client_other_prototypes = {
                        client_id: self._serialize_array_payload(
                            other_prototypes.get(client_id, {}))
                    }
                    timing_serialize_client_other += time.time() - _timing_t0
                    content['other_prototypes'] = client_other_prototypes
                    _timing_t0 = time.time()
                    client_other_size = self._sizeof_content({
                        'other_prototypes': client_other_prototypes
                    })
                    timing_sizeof += time.time() - _timing_t0
                    self._bytes_sent += shared_content_size + client_other_size
                logger.info(
                    f"Server: Sending global covariances to client {client_id}")
                _timing_t0 = time.time()
                self._broadcast_identical_to_clients(
                    'global_covariances', [client_id], self.state, content)
                timing_send += time.time() - _timing_t0
                logger.info(
                    f"Server: Sent global covariances to client {client_id}")

        logger.info(
            f"Server: Broadcasted global covariances to {len(receivers)} "
            f"active clients: {receivers}")
        logger.info(
            "GGEUR_TIMING_SERVER "
            f"stage=broadcast_global_covariances total_sec={time.time() - timing_total_start:.6f} "
            f"serialize_cov_sec={timing_serialize_cov:.6f} "
            f"serialize_global_proto_sec={timing_serialize_global_proto:.6f} "
            f"serialize_all_proto_sec={timing_serialize_all_proto:.6f} "
            f"serialize_client_other_sec={timing_serialize_client_other:.6f} "
            f"sizeof_sec={timing_sizeof:.6f} send_sec={timing_send:.6f} "
            f"receivers={len(receivers)} cov_classes={len(self.global_cov_matrices)} "
            f"global_proto_classes={len(self.global_prototypes)} "
            f"all_client_prototypes={len(self.all_prototypes) if serialized_all_prototypes is not None else 0} "
            f"shared_bytes={shared_content_size}")
        self._mark_stage('augmentation_ready')

    @staticmethod
    def _serialize_array_payload(payload):
        """Encode array/tensor leaves to compact strings for gRPC transfer."""
        if isinstance(payload, dict):
            return {
                key: GGEURServer._serialize_array_payload(value)
                for key, value in payload.items()
            }
        if isinstance(payload, list):
            return [
                GGEURServer._serialize_array_payload(value)
                for value in payload
            ]
        if isinstance(payload, tuple):
            return [
                GGEURServer._serialize_array_payload(value)
                for value in payload
            ]
        if isinstance(payload, (np.ndarray, torch.Tensor)):
            return GGEURServer._serialize_ndarray_payload(payload)
        return payload

    @staticmethod
    def _serialize_ndarray_payload(payload):
        """Serialize ndarray/tensor as compressed base64 to keep gRPC payloads small."""
        if isinstance(payload, torch.Tensor):
            payload = payload.detach().cpu().numpy()
        array = np.asarray(payload, dtype=np.float16)
        buffer = io.BytesIO()
        np.save(buffer, array, allow_pickle=False)
        compressed = zlib.compress(buffer.getvalue(), level=3)
        return 'znp:' + base64.b64encode(compressed).decode('ascii')

    def callback_funcs_model_para(self, message: Message):
        """
        Handle model parameter messages from clients.
        Aggregate using FedAvg.
        """
        round_idx = message.state
        sender = message.sender
        content = message.content
        self._bytes_recv += self._sizeof_content(content)

        expected_senders = self._expected_model_senders()
        if sender not in expected_senders:
            logger.warning(
                f"Server: Ignore model update from unexpected sender "
                f"{sender} for round {round_idx}; expected_senders="
                f"{sorted(expected_senders)}")
            return
        if round_idx != self.state:
            logger.warning(
                f"Server: Ignore stale/future model update from client "
                f"{sender}: message_round={round_idx}, server_round="
                f"{self.state}")
            return

        if isinstance(content, (tuple, list)) and len(content) == 2:
            sample_size, model_para = content
        else:
            sample_size, model_para = 0, content

        # Store in message buffer
        if round_idx not in self.msg_buffer['train']:
            self.msg_buffer['train'][round_idx] = []

        received_senders = {
            item[2] for item in self.msg_buffer['train'][round_idx]
        }
        if sender in received_senders:
            logger.warning(
                f"Server: Ignore duplicate model update from client "
                f"{sender} for round {round_idx}")
            return

        self.msg_buffer['train'][round_idx].append((sample_size, model_para, sender))

        expected_updates = self._expected_train_updates()
        sender_role = 'subserver' if self.hierarchical_training else 'client'
        logger.info(f"Server: Received model from {sender_role} {sender} for round {round_idx} "
                    f"({len(self.msg_buffer['train'][round_idx])}/{expected_updates}; "
                    f"configured_clients={self._client_num})")

        # Check if the configured training-update quorum has responded.
        if len(self.msg_buffer['train'][round_idx]) >= expected_updates:
            self._perform_fedavg(round_idx)

    def _perform_fedavg(self, round_idx):
        """Perform aggregation for MLP (and optionally CNN).

        - Default: FedAvg weighted averaging
        - If cfg.fedopt.use: FedOpt server-side update for the global MLP
        """
        logger.info(
            f"Server: Performing {'FedOpt' if self.use_fedopt else 'FedAvg'} aggregation for round {round_idx}"
        )
        self._clear_stage()

        # Collect all model parameters
        all_params = self.msg_buffer['train'][round_idx]

        # Filter out empty updates
        valid_params = [(s, p) for s, p, _ in all_params if s > 0 and p is not None]

        if not valid_params:
            logger.warning("Server: No valid model parameters received")
            self.msg_buffer['train'].pop(round_idx, None)
            del all_params
            del valid_params
            self.state = round_idx + 1
            if self.state < self._total_round_num:
                self._start_training_round()
            else:
                self._finish()
            return

        # Compute sample weights
        sample_sizes = [s for s, p in valid_params]
        total_samples = sum(sample_sizes)

        # Handle separated training mode
        if self.use_separated_training:
            self._perform_separated_fedavg(valid_params, total_samples, round_idx)
        else:
            # Check if we're in CNN distillation mode
            first_params = valid_params[0][1]
            is_combined_params = isinstance(first_params, dict) and 'mlp' in first_params

            if is_combined_params:
                # Aggregate MLP and CNN separately
                mlp_aggregated = self._aggregate_model_params(
                    [(s, p['mlp']) for s, p in valid_params if p.get('mlp') is not None],
                    total_samples
                )
                cnn_aggregated = self._aggregate_model_params(
                    [(s, p['cnn']) for s, p in valid_params if p.get('cnn') is not None],
                    total_samples
                )

                # Update global MLP (FedAvg or FedOpt)
                if mlp_aggregated and self.global_mlp is not None:
                    try:
                        if self.use_fedopt:
                            self._apply_fedopt_update(mlp_aggregated)
                        else:
                            self.global_mlp.load_state_dict(mlp_aggregated)
                    except Exception as e:
                        logger.debug(f"Server: Could not update MLP params: {e}")

                # Update global CNN
                if cnn_aggregated and self.global_cnn is not None:
                    try:
                        self.global_cnn.load_state_dict(cnn_aggregated)
                    except Exception as e:
                        logger.debug(f"Server: Could not load CNN params: {e}")
            else:
                # Standard mode: only MLP
                mlp_aggregated = self._aggregate_model_params(valid_params, total_samples)

                if mlp_aggregated and self.global_mlp is not None:
                    try:
                        if self.use_fedopt:
                            self._apply_fedopt_update(mlp_aggregated)
                        else:
                            self.global_mlp.load_state_dict(mlp_aggregated)
                    except Exception as e:
                        logger.debug(f"Server: Could not load MLP params: {e}")

        if self.use_fedproto:
            self._aggregate_fedproto_prototypes(valid_params)

        should_eval = self._should_run_eval(round_idx)
        client_eval_requested = False

        # Evaluate MLP on test sets (using CLIP features)
        if should_eval and self._eval_mode() in ('server', 'both'):
            test_results = self._evaluate_on_test_sets()
            if test_results:
                acc_str = ', '.join([f"{k}: {v:.4f}" for k, v in test_results.items()])
                logger.info(f"Server: Round {round_idx} MLP Test Accuracy - {acc_str}")
        terminal_client_eval_only = bool(getattr(
            self.ggeur_cfg, 'terminal_client_eval_only', False))
        should_eval_clients = (
            should_eval and self._eval_mode() in ('client', 'both') and
            (not terminal_client_eval_only or
             int(round_idx) == int(self._total_round_num) - 1)
        )
        if should_eval_clients:
            self._request_client_eval(round_idx)
            client_eval_requested = True

        # Aggregate and evaluate PromptFL if enabled
        if self.use_promptfl:
            self._aggregate_prompt(valid_params, total_samples)
            if should_eval and self._eval_mode() in ('server', 'both'):
                prompt_results = self._evaluate_prompt_on_test_sets()
                if prompt_results:
                    acc_str = ', '.join([f"{k}: {v:.4f}" for k, v in prompt_results.items()])
                    logger.info(f"Server: Round {round_idx} Prompt Test Accuracy - {acc_str}")

        # Evaluate CNN on test sets (using original images) if enabled
        # Include separated training Phase 2
        should_eval_cnn = (
            (self.use_cnn_distillation or self.use_feature_alignment) or
            (self.use_separated_training and self.training_phase == 'cnn_backbone')
        )
        if should_eval and self._eval_mode() in ('server', 'both') and \
                should_eval_cnn and self.global_cnn is not None:
            cnn_test_results = self._evaluate_cnn_on_test_sets()
            if cnn_test_results:
                acc_str = ', '.join([f"{k}: {v:.4f}" for k, v in cnn_test_results.items()])
                logger.info(f"Server: Round {round_idx} CNN Test Accuracy - {acc_str}")

        # Log progress
        if self._round_start_time is not None:
            round_elapsed = time.time() - self._round_start_time
            self._round_durations.append(round_elapsed)
            train_qps = total_samples / round_elapsed if round_elapsed > 0 else 0.0
            self._round_system_metrics.append({
                'round': int(round_idx),
                'total_samples': int(total_samples),
                'round_time_sec': float(round_elapsed),
                'train_qps_samples_per_sec': float(train_qps),
                'valid_client_updates': len(valid_params),
                'received_client_updates': len(all_params),
                'expected_train_updates': int(self._expected_train_updates()),
                'bytes_sent_total': int(self._bytes_sent),
                'bytes_recv_total': int(self._bytes_recv),
            })
            logger.info(f"Server: Round {round_idx} aggregation complete, "
                        f"total samples: {total_samples}, "
                        f"round time: {round_elapsed:.1f}s, "
                        f"train_qps={train_qps:.2f} samples/s, "
                        f"valid_updates={len(valid_params)}/{self._expected_train_updates()}")

        # Updates are single-use: after aggregation, retaining every client's
        # state dict for every completed round grows memory linearly (notably
        # about 0.5 GiB/round for 120-client LSTM runs).  Drop the completed
        # round before scheduling the next one; the aggregated global model is
        # already stored in ``self.global_mlp``.
        self.msg_buffer['train'].pop(round_idx, None)
        del all_params
        del valid_params

        if client_eval_requested:
            return

        self._complete_training_round(round_idx)

    def _complete_training_round(self, round_idx):
        # Move to next round
        self.state = round_idx + 1

        if self.state < self._total_round_num:
            self._start_training_round()
        else:
            self._finish()

    def _eval_mode(self):
        return str(getattr(self.ggeur_cfg, 'headonly_eval_mode',
                           'server') or 'server').lower()

    def _should_run_eval(self, round_idx):
        """Whether this GGEUR server should run evaluation this round."""
        eval_freq = getattr(getattr(self._cfg, 'eval', None), 'freq', 1)
        try:
            eval_freq = int(eval_freq)
        except Exception:
            eval_freq = 1
        return eval_freq > 0 and int(round_idx) % eval_freq == 0

    def _request_client_eval(self, round_idx):
        if self.global_mlp is None:
            self._complete_training_round(round_idx)
            return

        receivers = self._active_clients_for_broadcast()
        model_para = copy.deepcopy(self.global_mlp.state_dict())
        if self.domain_prototype_ensemble:
            model_para = {
                'mlp': model_para,
                'domain_prototype_ensemble': self._serialize_array_payload(
                    self.domain_prototype_ensemble),
            }
        if self.domain_personalized_heads:
            if not isinstance(model_para, dict) or 'mlp' not in model_para:
                model_para = {'mlp': model_para}
            model_para['domain_personalized_heads'] = copy.deepcopy(
                self.domain_personalized_heads)
        send_bytes = self._sizeof_content(model_para)
        self._bytes_sent += send_bytes * len(receivers)
        self.client_eval_buffer[round_idx] = {}
        self.pending_client_eval_round = round_idx
        logger.info(
            f"Server: Requesting client-side MLP evaluation for round "
            f"{round_idx} from {len(receivers)} clients: {receivers}")
        self._mark_stage(f'round_{round_idx}_client_eval')
        self._broadcast_identical_to_clients('client_eval', receivers,
                                             round_idx, model_para)

    def callback_for_client_eval_metrics(self, message: Message):
        round_idx = int(message.state)
        sender = message.sender
        content = message.content
        self._bytes_recv += self._sizeof_content(content)

        if round_idx != self.pending_client_eval_round:
            logger.warning(
                f"Server: Ignore client_eval_metrics from client {sender}: "
                f"message_round={round_idx}, pending_round="
                f"{self.pending_client_eval_round}")
            return
        expected_senders = set(self._active_clients_for_broadcast())
        if sender not in expected_senders:
            logger.warning(
                f"Server: Ignore client_eval_metrics from inactive client "
                f"{sender}; expected_clients={sorted(expected_senders)}")
            return

        if round_idx not in self.client_eval_buffer:
            self.client_eval_buffer[round_idx] = {}
        self.client_eval_buffer[round_idx][sender] = content

        # Evaluation metrics are still reported by every logical client even
        # when training updates are aggregated by subservers.
        expected = len(expected_senders)
        logger.info(
            f"Server: Received client-side eval metrics from client "
            f"{sender} for round {round_idx} "
            f"({len(self.client_eval_buffer[round_idx])}/{expected})")

        if len(self.client_eval_buffer[round_idx]) >= expected:
            self._aggregate_client_eval_metrics(round_idx)
            self.pending_client_eval_round = None
            self._clear_stage()
            self._complete_training_round(round_idx)

    def _aggregate_client_eval_metrics(self, round_idx):
        metrics = self.client_eval_buffer.get(round_idx, {})
        total = 0
        correct = 0
        loss_sum = 0.0
        clients = []
        client_records = []
        domain_totals = {}
        for client_id, item in sorted(metrics.items()):
            client_total = int(item.get('total', 0))
            client_correct = int(item.get('correct', 0))
            client_loss = float(item.get('loss', 0.0))
            client_accuracy = float(item.get(
                'accuracy',
                client_correct / client_total if client_total > 0 else 0.0))
            domain = str(item.get('domain', '') or 'unknown')
            total += client_total
            correct += client_correct
            loss_sum += client_loss * client_total
            clients.append(client_id)
            domain_item = domain_totals.setdefault(domain, {
                'correct': 0,
                'total': 0,
                'client_count': 0,
            })
            domain_item['correct'] += client_correct
            domain_item['total'] += client_total
            domain_item['client_count'] += 1
            client_records.append({
                'client_id': int(client_id),
                'domain': domain,
                'accuracy': client_accuracy,
                'loss': client_loss,
                'correct': client_correct,
                'total': client_total,
            })

        weighted_accuracy = correct / total if total > 0 else 0.0
        client_average = (
            sum(item['accuracy'] for item in client_records) /
            len(client_records) if client_records else 0.0)
        avg_loss = loss_sum / total if total > 0 else 0.0
        domain_results = {}
        for domain, item in sorted(domain_totals.items()):
            domain_results[domain] = {
                **item,
                'accuracy': (
                    item['correct'] / item['total']
                    if item['total'] > 0 else 0.0),
            }
        results = {
            'client_weighted_average': weighted_accuracy,
            'average': client_average,
        }
        for key, value in results.items():
            self.test_accuracies_history.setdefault(key, []).append(value)
        if client_average > self.best_avg_accuracy:
            self.best_avg_accuracy = client_average
            if self.global_mlp is not None:
                self.best_model_state = copy.deepcopy(
                    self.global_mlp.state_dict())

        evidence = {
            'round': int(round_idx),
            'client_count': len(client_records),
            'client_average_accuracy': client_average,
            'client_weighted_average_accuracy': weighted_accuracy,
            'average_loss': avg_loss,
            'domains': domain_results,
            'clients': client_records,
        }
        try:
            evidence_dir = Path(str(self._cfg.outdir))
            evidence_dir.mkdir(parents=True, exist_ok=True)
            evidence_path = evidence_dir / (
                f'client_model_accuracy_round_{int(round_idx)}.json')
            evidence_path.write_text(
                json.dumps(evidence, ensure_ascii=False, indent=2),
                encoding='utf-8')
            logger.info(
                f"Server: Client model accuracy evidence written to "
                f"{evidence_path}")
        except Exception as error:
            logger.warning(
                f"Server: Could not write client accuracy evidence: "
                f"{error}")

        logger.info(
            f"Server: Round {round_idx} MLP Test Accuracy - "
            f"client_weighted_average: {weighted_accuracy:.4f}, "
            f"average: {client_average:.4f}, loss: {avg_loss:.4f}, "
            f"correct: {correct}, total: {total}, clients: {clients}")

    def _apply_fedopt_update(self, averaged_state_dict):
        """
        Apply FedOpt server-side update based on the averaged client model.

        FedOpt uses the pseudo-gradient:
            g = w_t - w_bar
        and performs an optimizer step on the server model parameters.
        """
        if self.global_mlp is None:
            return
        self._init_fedopt_if_needed()
        if self.fedopt_optimizer is None:
            # Fallback to FedAvg if optimizer is not available
            self.global_mlp.load_state_dict(averaged_state_dict)
            return

        with torch.no_grad():
            current_state = self.global_mlp.state_dict()
            grads = {}
            for key, cur in current_state.items():
                if key not in averaged_state_dict:
                    continue
                avg = averaged_state_dict[key]
                if not isinstance(avg, torch.Tensor):
                    avg = torch.tensor(avg)
                if avg.device != cur.device:
                    avg = avg.to(cur.device)
                grads[key] = (cur.detach() - avg.detach()).type_as(cur)

        self.fedopt_optimizer.zero_grad()
        for name, p in self.global_mlp.named_parameters():
            if name in grads:
                p.grad = grads[name]
        self.fedopt_optimizer.step()

        if self.fedopt_annealing and self.fedopt_scheduler is not None:
            self.fedopt_scheduler.step()

    def _perform_separated_fedavg(self, valid_params, total_samples, round_idx):
        """Perform FedAvg for separated training mode"""
        first_params = valid_params[0][1]

        if self.training_phase == 'classifier':
            # Phase 1: Aggregate classifier parameters
            if isinstance(first_params, dict) and 'classifier' in first_params:
                classifier_aggregated = self._aggregate_model_params(
                    [(s, p['classifier']) for s, p in valid_params if p.get('classifier') is not None],
                    total_samples
                )
            else:
                classifier_aggregated = self._aggregate_model_params(valid_params, total_samples)

            if classifier_aggregated and self.global_mlp is not None:
                try:
                    self.global_mlp.load_state_dict(classifier_aggregated)
                    logger.info(f"Server: Phase 1 - Aggregated classifier from {len(valid_params)} clients")
                except Exception as e:
                    logger.debug(f"Server: Could not load classifier params: {e}")

        else:
            # Phase 2: Aggregate only CNN backbone parameters (classifier is frozen)
            if isinstance(first_params, dict) and 'cnn_backbone' in first_params:
                cnn_aggregated = self._aggregate_model_params(
                    [(s, p['cnn_backbone']) for s, p in valid_params if p.get('cnn_backbone') is not None],
                    total_samples
                )

                if cnn_aggregated and self.global_cnn is not None:
                    try:
                        self.global_cnn.load_state_dict(cnn_aggregated)
                        logger.info(f"Server: Phase 2 - Aggregated CNN backbone from {len(valid_params)} clients")
                    except Exception as e:
                        logger.debug(f"Server: Could not load CNN backbone params: {e}")

    def _aggregate_model_params(self, params_list, total_samples):
        """Helper function to aggregate model parameters using weighted average"""
        if not params_list:
            return None

        # Filter valid params
        valid_params = [(s, p) for s, p in params_list if s > 0 and p is not None]
        if not valid_params:
            return None

        first_params = valid_params[0][1]
        if not isinstance(first_params, dict):
            logger.warning(f"Server: Expected dict for model params, got {type(first_params)}, skipping aggregation")
            return None

        def _to_tensor(param):
            if isinstance(param, torch.Tensor):
                return param
            try:
                restored = param2tensor(param)
                if isinstance(restored, torch.Tensor):
                    return restored
                return torch.tensor(restored)
            except Exception as e:
                logger.debug(f"Server: Cannot convert model parameter to tensor ({e})")
                return None

        aggregated_params = {}

        for key in first_params.keys():
            param_tensor = first_params[key]
            if isinstance(param_tensor, dict):
                # Nested dict (e.g. combined params accidentally passed in); skip
                logger.debug(f"Server: Skipping nested dict value for key '{key}' during aggregation setup")
                continue
            param_tensor = _to_tensor(param_tensor)
            if param_tensor is None:
                logger.debug(f"Server: Cannot convert key '{key}' to tensor, skipping")
                continue
            aggregated_params[key] = torch.zeros_like(param_tensor).float()

        if not aggregated_params:
            return None

        # Weighted average — only iterate over keys validated from first_params
        for sample_size, params in valid_params:
            weight = sample_size / total_samples
            for key in aggregated_params.keys():
                if key not in params:
                    continue
                param_tensor = params[key]
                if isinstance(param_tensor, dict):
                    continue
                param_tensor = _to_tensor(param_tensor)
                if param_tensor is None:
                    continue
                aggregated_params[key] += weight * param_tensor.float()

        return aggregated_params

    def _aggregate_fedproto_prototypes(self, valid_params):
        """Aggregate hidden-space prototypes by their per-class counts.

        Hierarchical subservers emit the same sufficient statistics as a flat
        client set: a class mean plus the number of representations behind
        that mean.  Consequently this reduction is associative and produces
        the same result for flat and three-machine deployments.
        """
        keep_last = bool(
            getattr(self.ggeur_cfg,
                    'fedproto_keep_last_global_prototypes', True))
        aggregated = copy.deepcopy(
            self.fedproto_global_prototypes) if keep_last else {}
        prototype_sums = {}
        prototype_counts = {}

        for _, parameters in valid_params:
            if not isinstance(parameters, dict):
                continue
            local_prototypes = parameters.get(
                'fedproto_local_prototypes')
            local_counts = parameters.get('fedproto_local_counts')
            if not isinstance(local_prototypes, dict) or \
                    not isinstance(local_counts, dict):
                continue

            for class_idx, prototype in local_prototypes.items():
                try:
                    class_idx = int(class_idx)
                    count = int(
                        local_counts.get(
                            class_idx,
                            local_counts.get(str(class_idx), 0)))
                except (TypeError, ValueError):
                    continue
                if count <= 0 or prototype is None:
                    continue
                try:
                    if isinstance(prototype, (str, bytes)):
                        prototype = param2tensor(prototype)
                    tensor = prototype.detach().cpu().float() \
                        if isinstance(prototype, torch.Tensor) else \
                        torch.as_tensor(prototype, dtype=torch.float32)
                except (TypeError, ValueError, RuntimeError):
                    continue

                weighted = tensor * float(count)
                if class_idx in prototype_sums:
                    prototype_sums[class_idx] += weighted
                    prototype_counts[class_idx] += count
                else:
                    prototype_sums[class_idx] = weighted
                    prototype_counts[class_idx] = count

        for class_idx, prototype_sum in prototype_sums.items():
            count = int(prototype_counts[class_idx])
            aggregated[class_idx] = (
                prototype_sum / float(count)).float()

        self.fedproto_global_prototypes = aggregated
        logger.info(
            "Server: FedProto global prototypes aggregated "
            f"({len(prototype_sums)} classes updated)")

    def _load_test_images(self):
        """Load test images for CNN evaluation"""
        if self.test_images_loaded:
            return

        logger.info("Server: Loading test images for CNN evaluation...")

        data_type = self._cfg.data.type.lower()
        data_root = self._cfg.data.root

        # Get split parameters
        if hasattr(self._cfg.data, 'splits'):
            splits = tuple(self._cfg.data.splits)
        else:
            if 'office' in data_type and 'home' in data_type:
                splits = (0.7, 0.0, 0.3)
            else:
                splits = (0.8, 0.1, 0.1)

        train_ratio, val_ratio = splits[0], splits[1]
        configured_split_seed = int(getattr(
            self.ggeur_cfg, 'data_split_seed', -1))
        seed = (int(self._cfg.seed) if configured_split_seed < 0
                else configured_split_seed)

        # Determine domains based on dataset type
        if 'pacs' in data_type:
            domains = ['photo', 'art_painting', 'cartoon', 'sketch']
            from federatedscope.cv.dataset.pacs import PACS
            dataset_class = PACS
            dataset_kwargs = {}
        elif 'office' in data_type and 'home' in data_type:
            domains = ['Art', 'Clipart', 'Product', 'Real_World']
            from federatedscope.cv.dataset.office_home import OfficeHome
            dataset_class = OfficeHome
            dataset_kwargs = {}
        elif 'domainnet' in data_type or 'domain-net' in data_type or \
                'domain_net' in data_type:
            from federatedscope.cv.dataset.domainnet import DomainNet
            domains, classes = self._get_domainnet_eval_metadata()
            dataset_class = DomainNet
            dataset_kwargs = {'classes': classes}
        else:
            logger.warning(f"Server: Unknown dataset type {data_type} for CNN evaluation")
            return

        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        for domain in domains:
            try:
                test_dataset = dataset_class(
                    root=data_root,
                    domain=domain,
                    split='test',
                    transform=transform,
                    train_ratio=train_ratio,
                    val_ratio=val_ratio,
                    seed=seed,
                    **dataset_kwargs
                )

                if len(test_dataset) > 0:
                    self.test_image_loaders[domain] = DataLoader(
                        test_dataset,
                        batch_size=32,
                        shuffle=False,
                        num_workers=0
                    )
                    logger.info(f"Server: Loaded {len(test_dataset)} test images for {domain}")

            except Exception as e:
                logger.warning(f"Server: Failed to load test images for {domain}: {e}")

        self.test_images_loaded = True

    def _evaluate_cnn_on_test_sets(self):
        """Evaluate global CNN on test sets using original images"""
        if self.global_cnn is None:
            logger.warning("Server: global_cnn is None, skipping CNN evaluation")
            return {}

        # Load test images if not already loaded
        if not self.test_images_loaded:
            self._load_test_images()

        if not self.test_image_loaders:
            logger.warning("Server: No test image loaders available")
            return {}

        self.global_cnn.eval()
        results = {}

        # For separated training, we need to use the pretrained classifier
        classifier = None
        if self.use_separated_training:
            if self.pretrained_classifier is not None:
                classifier = self._build_classifier_from_state_dict(self.pretrained_classifier)
                classifier.eval()
                logger.info(f"Server: Using pretrained classifier for CNN evaluation")
            else:
                logger.warning("Server: Separated training mode but pretrained_classifier is None!")
                return {}

        with torch.no_grad():
            for domain, dataloader in self.test_image_loaders.items():
                correct = 0
                total = 0

                for images, labels in dataloader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    # Get CNN output
                    if self.use_separated_training and classifier is not None:
                        # Separated training: backbone -> features -> classifier -> logits
                        features = self.global_cnn(images)  # 512-dim features
                        outputs = classifier(features)  # logits
                    else:
                        # Other modes: CNN outputs logits directly
                        outputs = self.global_cnn(images)

                    _, predicted = torch.max(outputs, 1)

                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)

                accuracy = correct / total if total > 0 else 0
                results[domain] = accuracy

                # Track history
                if domain not in self.cnn_test_accuracies_history:
                    self.cnn_test_accuracies_history[domain] = []
                self.cnn_test_accuracies_history[domain].append(accuracy)

        # Compute average
        if results:
            avg_accuracy = sum(results.values()) / len(results)
            results['average'] = avg_accuracy

            if 'average' not in self.cnn_test_accuracies_history:
                self.cnn_test_accuracies_history['average'] = []
            self.cnn_test_accuracies_history['average'].append(avg_accuracy)

            # Track best CNN model
            if avg_accuracy > self.best_cnn_avg_accuracy:
                self.best_cnn_avg_accuracy = avg_accuracy
                self.best_cnn_model_state = copy.deepcopy(self.global_cnn.state_dict())

        return results

    @staticmethod
    def _sizeof_content(content) -> int:
        """Estimate byte size of message content (tensors + numpy arrays)."""
        total = 0
        if content is None:
            return 0
        if isinstance(content, torch.Tensor):
            return content.nelement() * content.element_size()
        if isinstance(content, np.ndarray):
            return content.nbytes
        if isinstance(content, dict):
            for v in content.values():
                total += GGEURServer._sizeof_content(v)
            return total
        if isinstance(content, (list, tuple)):
            for v in content:
                total += GGEURServer._sizeof_content(v)
            return total
        return 0

    def _finish(self):
        """Finish FL training"""
        self.is_finish = True
        logger.info("="*60)
        logger.info(f"Server: Training finished after {self.state} rounds")

        # Print MLP/Classifier final results
        if self.test_accuracies_history:
            logger.info("="*60)
            if self.use_separated_training:
                logger.info(f"Classifier Final Test Results (Pretrained in {self.classifier_pretrain_rounds} rounds):")
            else:
                logger.info("MLP Final Test Results:")
            for domain, acc_list in self.test_accuracies_history.items():
                if acc_list:
                    logger.info(f"  {domain}: final={acc_list[-1]:.4f}, best={max(acc_list):.4f}")

            logger.info(f"Classifier Best Average Accuracy: {self.best_avg_accuracy:.4f}")

        # Print CNN final results
        if self.use_separated_training and self.cnn_test_accuracies_history:
            logger.info("-"*60)
            logger.info("CNN Final Test Results (Separated Training - From Scratch):")
            for domain, acc_list in self.cnn_test_accuracies_history.items():
                if acc_list:
                    logger.info(f"  {domain}: final={acc_list[-1]:.4f}, best={max(acc_list):.4f}")

            logger.info(f"CNN Best Average Accuracy: {self.best_cnn_avg_accuracy:.4f}")

        elif (self.use_cnn_distillation or self.use_feature_alignment) and self.cnn_test_accuracies_history:
            logger.info("-"*60)
            logger.info("CNN Final Test Results (Feature Alignment):" if self.use_feature_alignment else "CNN Final Test Results (Distillation):")
            for domain, acc_list in self.cnn_test_accuracies_history.items():
                if acc_list:
                    logger.info(f"  {domain}: final={acc_list[-1]:.4f}, best={max(acc_list):.4f}")

            logger.info(f"CNN Best Average Accuracy: {self.best_cnn_avg_accuracy:.4f}")

        # Print PromptFL final results
        if self.use_promptfl and self.prompt_test_accuracies_history:
            logger.info("-"*60)
            logger.info("PromptFL Final Test Results:")
            for domain, acc_list in self.prompt_test_accuracies_history.items():
                if acc_list:
                    logger.info(f"  {domain}: final={acc_list[-1]:.4f}, best={max(acc_list):.4f}")
            logger.info(f"Prompt Best Average Accuracy: {self.best_prompt_avg_accuracy:.4f}")

        logger.info("="*60)

        elapsed = time.time() - self._train_start_time
        h, rem = divmod(int(elapsed), 3600)
        m, s = divmod(rem, 60)
        logger.info(f"Server: Total training time: {h:02d}h {m:02d}m {s:02d}s ({elapsed:.1f}s)")

        if self._round_durations:
            avg_round = sum(self._round_durations) / len(self._round_durations)
            logger.info(f"Server: Avg round time: {avg_round:.1f}s  "
                        f"(min={min(self._round_durations):.1f}s, "
                        f"max={max(self._round_durations):.1f}s, "
                        f"rounds={len(self._round_durations)})")

        if self._round_system_metrics:
            qps_values = [
                item['train_qps_samples_per_sec']
                for item in self._round_system_metrics
            ]
            stable_qps_values = qps_values[1:] if len(qps_values) > 1 else qps_values
            total_samples = sum(
                item['total_samples'] for item in self._round_system_metrics
            )
            total_round_time = sum(
                item['round_time_sec'] for item in self._round_system_metrics
            )
            logger.info(
                "Server: HeadOnly/System QPS Summary - "
                f"rounds={len(qps_values)}, "
                f"total_samples={total_samples}, "
                f"overall_train_qps={total_samples / total_round_time if total_round_time > 0 else 0.0:.2f} samples/s, "
                f"avg_train_qps={sum(qps_values) / len(qps_values):.2f} samples/s, "
                f"stable_avg_train_qps={sum(stable_qps_values) / len(stable_qps_values):.2f} samples/s, "
                f"min_train_qps={min(qps_values):.2f} samples/s, "
                f"max_train_qps={max(qps_values):.2f} samples/s")

        def _fmt_bytes(n):
            for unit in ('B', 'KB', 'MB', 'GB'):
                if n < 1024:
                    return f"{n:.2f} {unit}"
                n /= 1024
            return f"{n:.2f} TB"

        total_comm = self._bytes_sent + self._bytes_recv
        logger.info(f"Server: Communication volume - "
                    f"sent={_fmt_bytes(self._bytes_sent)}, "
                    f"recv={_fmt_bytes(self._bytes_recv)}, "
                    f"total={_fmt_bytes(total_comm)}")
        logger.info("="*60)

        if bool(getattr(
                self.ggeur_cfg, 'save_mlp_checkpoint', False)):
            self._save_portable_mlp_checkpoints()

        receivers = self._active_clients_for_broadcast()
        self._broadcast_identical_to_clients('finish', receivers, self.state,
                                             {})

        self._monitor.finish_fl()
        if hasattr(self.comm_manager, 'shutdown'):
            self.comm_manager.shutdown()
        self.state = self.total_round_num + 1

    @staticmethod
    def _cpu_state_dict(state_dict):
        return {
            str(name): value.detach().cpu().clone()
            for name, value in state_dict.items()
        }

    @staticmethod
    def _atomic_torch_save(payload, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + '.tmp')
        torch.save(payload, temporary)
        os.replace(temporary, path)

    def _checkpoint_architecture(self):
        model_type = str(getattr(self._cfg.model, 'type', 'ggeur_mlp'))
        return {
            'model_type': model_type,
            'input_dim': int(getattr(
                self._cfg.model, 'in_channels', 0)
                or self._get_embedding_dim()),
            'hidden_dim': int(getattr(
                self._cfg.model, 'hidden', 0)
                if model_type.lower() in {'ggeur_rnn', 'ggeur_lstm'}
                else getattr(self.ggeur_cfg, 'mlp_hidden_dim', 0)),
            'num_classes': int(getattr(self._cfg.model, 'num_classes', 0)),
            'num_layers': int(getattr(self._cfg.model, 'layer', 1)),
            'dropout': float(getattr(
                self._cfg.model, 'dropout',
                getattr(self.ggeur_cfg, 'mlp_dropout', 0.0))),
        }

    def _checkpoint_backbone(self):
        extractor = str(getattr(
            self.ggeur_cfg, 'feature_extractor', 'clip')).lower()
        model_type = str(getattr(self._cfg.model, 'type', '')).lower()
        if model_type in {'ggeur_rnn', 'ggeur_lstm'}:
            return {
                'feature_extractor': 'bert',
                'model': str(getattr(
                    self.ggeur_cfg, 'bert_model_path', '')),
                'tokenizer': str(getattr(
                    self.ggeur_cfg, 'bert_tokenizer_path', '')),
                'pooling': str(getattr(
                    self.ggeur_cfg, 'bert_pooling', 'cls')),
            }
        if extractor == 'cnn':
            return {
                'feature_extractor': 'cnn',
                'model': str(getattr(
                    self.ggeur_cfg, 'cnn_backbone', 'convnext_base')),
                'checkpoint': str(getattr(
                    self.ggeur_cfg, 'cnn_checkpoint_path', '')),
            }
        if extractor == 'timm':
            return {
                'feature_extractor': 'timm',
                'model': str(getattr(
                    self.ggeur_cfg, 'timm_model', '')),
                'checkpoint': str(getattr(
                    self.ggeur_cfg, 'timm_checkpoint_path', '')),
            }
        return {
            'feature_extractor': 'clip',
            'model': str(getattr(
                self.ggeur_cfg, 'clip_model', 'ViT-B-16')),
            'pretrained': str(getattr(
                self.ggeur_cfg, 'clip_pretrained', 'openai')),
            'checkpoint': str(getattr(
                self.ggeur_cfg, 'clip_model_path', '')),
        }

    def _save_portable_mlp_checkpoints(self):
        """Save a reloadable head and its frozen-backbone test features."""
        if self.global_mlp is None:
            raise RuntimeError(
                'save_mlp_checkpoint=True but the global classifier is empty')
        configured = str(getattr(
            self.ggeur_cfg, 'mlp_checkpoint_dir', '') or '').strip()
        if not configured:
            raise ValueError(
                'ggeur.mlp_checkpoint_dir must be set when checkpoint saving '
                'is enabled')
        output_dir = Path(configured)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not self.test_data_loaded:
            self._load_test_data_and_features()
        if not self.test_features:
            raise RuntimeError(
                'No test features are available for independent evaluation')

        architecture = self._checkpoint_architecture()
        backbone = self._checkpoint_backbone()
        feature_bundle_path = output_dir / 'pretrained_test_features.pt'
        feature_bundle = {
            'format_version': 1,
            'dataset': str(getattr(self._cfg.data, 'type', '')),
            'backbone': backbone,
            'features': {
                str(domain): torch.as_tensor(value).detach().cpu().float()
                for domain, value in self.test_features.items()
            },
            'labels': {
                str(domain): torch.as_tensor(value).detach().cpu().long()
                for domain, value in self.test_labels.items()
            },
        }
        self._atomic_torch_save(feature_bundle, feature_bundle_path)

        common = {
            'format_version': 1,
            'architecture': architecture,
            'backbone': backbone,
            'dataset': str(getattr(self._cfg.data, 'type', '')),
            'seed': int(getattr(self._cfg, 'seed', 0)),
            'round': int(self.state),
            'test_feature_bundle': feature_bundle_path.name,
            'test_accuracy_history': {
                str(domain): [float(item) for item in values]
                for domain, values in self.test_accuracies_history.items()
            },
            'best_average_accuracy': float(self.best_avg_accuracy),
        }
        states = {
            'final': self.global_mlp.state_dict(),
            'best': self.best_model_state or self.global_mlp.state_dict(),
        }
        manifest = {
            'format_version': 1,
            'dataset': common['dataset'],
            'seed': common['seed'],
            'backbone': backbone,
            'feature_bundle': feature_bundle_path.name,
            'checkpoints': {},
        }
        for kind, state in states.items():
            checkpoint_path = output_dir / f'mlp_{kind}.pt'
            checkpoint = dict(common)
            checkpoint['kind'] = kind
            checkpoint['state_dict'] = self._cpu_state_dict(state)
            self._atomic_torch_save(checkpoint, checkpoint_path)
            manifest['checkpoints'][kind] = checkpoint_path.name
            logger.info(
                'MLP_CHECKPOINT_SAVED kind=%s path=%s',
                kind, checkpoint_path)

        manifest_path = output_dir / 'checkpoint_manifest.json'
        temporary = manifest_path.with_suffix('.json.tmp')
        temporary.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8')
        os.replace(temporary, manifest_path)
        logger.info(
            'PRETRAINED_TEST_FEATURES_SAVED path=%s domains=%s',
            feature_bundle_path, sorted(feature_bundle['features']))
        logger.info('MLP_CHECKPOINT_MANIFEST=%s', manifest_path)

    # ==================== PromptFL Methods ====================

    def _init_global_prompt(self, num_classes):
        """Initialize global prompt ctx with zeros (clients will randomize locally)."""
        n_ctx = getattr(self.ggeur_cfg, 'prompt_length', 16)
        ctx_dim = self._get_embedding_dim()
        self.global_prompt_ctx = torch.zeros(n_ctx, ctx_dim, device=self.device)
        logger.info(f"Server: Initialized global prompt ctx [{n_ctx}, {ctx_dim}]")

    def _aggregate_prompt(self, valid_params, total_samples):
        """Aggregate prompt ctx vectors from clients using weighted average."""
        prompt_params = []
        for _, p in valid_params:
            if isinstance(p, dict) and p.get('prompt') is not None:
                prompt_dict = p['prompt']
                ctx = prompt_dict.get('ctx')
                prompt_sample_size = int(prompt_dict.get('sample_size', 0))
                if ctx is not None and prompt_sample_size > 0:
                    prompt_params.append((prompt_sample_size, {'ctx': ctx}))

        if not prompt_params:
            return

        total_prompt_samples = sum(s for s, _ in prompt_params)
        aggregated = self._aggregate_model_params(prompt_params, total_prompt_samples)
        if aggregated and 'ctx' in aggregated:
            self.global_prompt_ctx = aggregated['ctx'].to(self.device)
            logger.info(f"Server: Aggregated prompt ctx from {len(prompt_params)} clients "
                        f"(prompt_samples={total_prompt_samples})")

    def _evaluate_prompt_on_test_sets(self):
        """Evaluate global prompt on test sets using CLIP image features."""
        if self.global_prompt_ctx is None:
            return {}

        # Load test features if not ready
        if not self.test_data_loaded:
            self._load_test_data_and_features()
        if not self.test_features:
            return {}

        # Build or update eval CustomCLIP (open_clip-based)
        if self.prompt_learner_eval is None:
            import open_clip
            from federatedscope.contrib.model.ggeur_prompt import CustomCLIP

            n_ctx = getattr(self.ggeur_cfg, 'prompt_length', 16)
            num_classes = self._cfg.model.num_classes
            clip_model_name = getattr(self.ggeur_cfg, 'clip_model', 'ViT-B-16')
            clip_pretrained = getattr(self.ggeur_cfg, 'clip_pretrained', 'openai')
            model_path = getattr(self.ggeur_cfg, 'clip_model_path', '')

            # Get class names
            cfg_names = getattr(self.ggeur_cfg, 'prompt_class_names', [])
            if cfg_names:
                class_names = list(cfg_names)
            else:
                data_type = self._cfg.data.type.lower()
                try:
                    if 'office' in data_type and 'home' in data_type:
                        from federatedscope.cv.dataset.office_home import OfficeHome
                        class_names = list(OfficeHome.CLASSES)
                    elif 'pacs' in data_type:
                        from federatedscope.cv.dataset.pacs import PACS
                        class_names = list(PACS.CLASSES)
                    elif 'office' in data_type and 'caltech' in data_type:
                        from federatedscope.cv.dataset.office_caltech import OfficeCaltech10
                        class_names = list(OfficeCaltech10.CLASSES)
                    else:
                        class_names = [f"class {i}" for i in range(num_classes)]
                except Exception:
                    class_names = [f"class {i}" for i in range(num_classes)]

            template = getattr(self.ggeur_cfg, 'prompt_template', 'a photo of a {}')

            try:
                if model_path and os.path.isfile(model_path):
                    clip_model, _, _ = open_clip.create_model_and_transforms(
                        clip_model_name, pretrained=model_path
                    )
                else:
                    clip_model, _, _ = open_clip.create_model_and_transforms(
                        clip_model_name, pretrained=clip_pretrained
                    )
                tokenizer = open_clip.get_tokenizer(clip_model_name)
            except Exception as e:
                logger.error(f"Server: Failed to load open_clip CLIP for prompt eval: {e}")
                return {}

            custom_clip = CustomCLIP(
                clip_model=clip_model,
                tokenizer=tokenizer,
                classnames=class_names,
                n_ctx=n_ctx,
                template=template,
                device=self.device,
            )
            self.prompt_learner_eval = custom_clip.prompt_learner
            self.text_encoder_eval = custom_clip.text_encoder
            self._eval_clip = custom_clip.clip_model

        # Load current global ctx
        self.prompt_learner_eval.ctx.data = self.global_prompt_ctx.to(self.device)

        results = {}
        self.prompt_learner_eval.eval()

        with torch.no_grad():
            prompts, eot_pos = self.prompt_learner_eval()
            text_feats = self.text_encoder_eval(prompts, eot_pos, self._eval_clip)
            text_feats = F.normalize(text_feats, p=2, dim=1)
            temperature = getattr(self.ggeur_cfg, 'prompt_temperature', 0.07)
            logit_scale = 1.0 / temperature

            for domain, features in self.test_features.items():
                labels = self.test_labels[domain]

                feat_tensor = torch.from_numpy(features).float().to(self.device)
                label_tensor = torch.from_numpy(labels).long().to(self.device)

                img_feats = F.normalize(feat_tensor, p=2, dim=1)
                logits = logit_scale * img_feats @ text_feats.T
                _, predicted = torch.max(logits, 1)

                accuracy = (predicted == label_tensor).float().mean().item()
                results[domain] = accuracy

                if domain not in self.prompt_test_accuracies_history:
                    self.prompt_test_accuracies_history[domain] = []
                self.prompt_test_accuracies_history[domain].append(accuracy)

        if results:
            avg_accuracy = sum(results.values()) / len(results)
            results['average'] = avg_accuracy

            if 'average' not in self.prompt_test_accuracies_history:
                self.prompt_test_accuracies_history['average'] = []
            self.prompt_test_accuracies_history['average'].append(avg_accuracy)

            if avg_accuracy > self.best_prompt_avg_accuracy:
                self.best_prompt_avg_accuracy = avg_accuracy

        return results
