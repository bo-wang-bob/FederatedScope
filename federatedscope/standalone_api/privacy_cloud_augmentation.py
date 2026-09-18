"""Verbatim cloud GGEUR decoy / augmentation methods, isolated from normal training.

Source: FederatedScope-feature-GGEUR-git/ggeur_client.py, 2026-09-18.
Only this private mixin uses these methods; no monkey-patching of shared workers.
"""
import logging
import time
import numpy as np
from torch.utils.data import DataLoader
from federatedscope.contrib.worker.ggeur_client import AugmentedFeatureDataset

logger = logging.getLogger(__name__)

class CloudAugmentation:
    def _get_local_decoy_cfg(self):
        return getattr(self.ggeur_cfg, 'local_decoy', None)

    def _is_local_decoy_enabled(self):
        decoy_cfg = self._get_local_decoy_cfg()
        return decoy_cfg is not None and \
            bool(getattr(decoy_cfg, 'use', False)) and \
            int(getattr(decoy_cfg, 'num_per_class', 0)) > 0

    def _generate_local_decoy_features(self):
        """Generate editable-compatible fake samples from local class stats."""
        if not self._is_local_decoy_enabled() or not self.local_features:
            return None, None

        decoy_cfg = self._get_local_decoy_cfg()
        num_per_class = int(getattr(decoy_cfg, 'num_per_class', 0))
        noise_scale = float(getattr(decoy_cfg, 'noise_scale', 0.15))
        min_std = float(getattr(decoy_cfg, 'min_std', 1e-4))
        max_total = int(getattr(decoy_cfg, 'max_total', 0))
        seed_offset = int(getattr(decoy_cfg, 'seed_offset', 1701))
        rng = np.random.default_rng(
            int(getattr(self._cfg, 'seed', 0)) + seed_offset + int(self.ID))

        all_local = [
            np.asarray(features, dtype=np.float32)
            for features in self.local_features.values()
            if len(features) > 0
        ]
        if not all_local:
            return None, None
        global_std = np.maximum(
            np.std(np.vstack(all_local), axis=0), min_std)

        decoy_features = []
        decoy_labels = []
        for class_idx in sorted(self.local_features.keys()):
            features = np.asarray(
                self.local_features[class_idx], dtype=np.float32)
            if features.size == 0:
                continue
            mean = np.mean(features, axis=0)
            std = np.std(features, axis=0) \
                if features.shape[0] > 1 else global_std
            std = np.maximum(std, min_std)
            samples = mean + rng.normal(
                loc=0.0, scale=1.0,
                size=(num_per_class, features.shape[1])) * std * noise_scale
            decoy_features.append(samples.astype(np.float32))
            decoy_labels.append(np.full(
                num_per_class, int(class_idx), dtype=np.int64))

        if not decoy_features:
            return None, None
        features = np.vstack(decoy_features)
        labels = np.concatenate(decoy_labels)
        if max_total > 0 and len(labels) > max_total:
            indices = rng.choice(len(labels), size=max_total, replace=False)
            features = features[indices]
            labels = labels[indices]

        logger.info(
            f"Client {self.ID}: Generated local decoy features - "
            f"samples={len(labels)}, classes={len(np.unique(labels))}, "
            f"noise_scale={noise_scale}, train_with_decoy="
            f"{bool(getattr(decoy_cfg, 'train_with_decoy', True))}")
        return features, labels

    def _perform_augmentation(self):
        """Perform GGEUR_Clip feature augmentation"""
        augmentation_start = time.time()
        timing_cache_load = 0.0
        timing_init = 0.0
        timing_original_collect = 0.0
        timing_cov_lookup = 0.0
        timing_sample_generation = 0.0
        timing_prototype_generation = 0.0
        timing_stack_select = 0.0
        timing_dataset_build = 0.0
        timing_cache_save = 0.0
        original_samples = 0
        generated_sample_count = 0
        generated_prototype_count = 0
        selected_samples = 0
        sample_generation_calls = 0
        prototype_generation_calls = 0
        _timing_t0 = time.time()
        if self._try_load_augmented_feature_cache():
            timing_cache_load = time.time() - _timing_t0
            self.augmentation_done = True
            self.local_features = {}
            self.local_labels = {}
            logger.info(
                "GGEUR_TIMING_CLIENT "
                f"client={int(self.ID)} stage=augmentation "
                f"dataset={self._cfg.data.type} extractor={self.feature_extractor_type} "
                f"cache_hit=1 total_sec={time.time() - augmentation_start:.6f} "
                f"cache_load_sec={timing_cache_load:.6f} "
                f"generated_per_sample={getattr(self.ggeur_cfg, 'num_generated_per_sample', 0)} "
                f"generated_per_prototype={getattr(self.ggeur_cfg, 'num_generated_per_prototype', 0)} "
                f"target_size_per_class={getattr(self.ggeur_cfg, 'target_size_per_class', 0)}")
            return
        timing_cache_load = time.time() - _timing_t0

        _timing_t0 = time.time()
        target_size = self.ggeur_cfg.target_size_per_class
        num_per_sample = self.ggeur_cfg.num_generated_per_sample
        num_per_prototype = self.ggeur_cfg.num_generated_per_prototype
        use_cross_client = self.ggeur_cfg.use_cross_client_prototypes

        # Check if augmentation is disabled (baseline mode)
        no_augmentation = (num_per_sample == 0 and num_per_prototype == 0) or not use_cross_client

        if no_augmentation:
            logger.info(f"Client {self.ID}: No augmentation mode - using original features only")
        else:
            logger.info(f"Client {self.ID}: Performing GGEUR_Clip augmentation...")

        # Generate local decoys before GGEUR when both mechanisms are enabled.
        decoy_before_ggeur = (
            self._is_local_decoy_enabled() and
            bool(getattr(self._get_local_decoy_cfg(), 'before_ggeur', True)))
        ggeur_local_features = {
            int(class_idx): np.asarray(features, dtype=np.float32)
            for class_idx, features in self.local_features.items()
        }
        ggeur_local_decoy_masks = {
            int(class_idx): np.zeros(len(features), dtype=bool)
            for class_idx, features in ggeur_local_features.items()
        }
        if decoy_before_ggeur:
            decoy_features, decoy_labels = self._generate_local_decoy_features()
            if decoy_features is not None and decoy_labels is not None and len(decoy_labels) > 0:
                for class_idx in np.unique(decoy_labels):
                    class_idx = int(class_idx)
                    selected = np.asarray(decoy_labels) == class_idx
                    values = np.asarray(decoy_features[selected], dtype=np.float32)
                    if class_idx in ggeur_local_features:
                        ggeur_local_features[class_idx] = np.vstack([
                            ggeur_local_features[class_idx], values])
                        ggeur_local_decoy_masks[class_idx] = np.concatenate([
                            ggeur_local_decoy_masks[class_idx],
                            np.ones(len(values), dtype=bool)])
                    else:
                        ggeur_local_features[class_idx] = values
                        ggeur_local_decoy_masks[class_idx] = np.ones(
                            len(values), dtype=bool)
                logger.info(
                    f"Client {self.ID}: Feeding local decoys into GGEUR - "
                    f"decoy_samples={len(decoy_labels)}")

        all_features = []
        all_labels = []
        all_generated_masks = []

        # Get all class indices
        all_classes = set(ggeur_local_features.keys())
        if not no_augmentation and self.global_cov_matrices:
            all_classes.update(self.global_cov_matrices.keys())
        if not no_augmentation and self.other_prototypes:
            all_classes.update(self.other_prototypes.keys())

        total_classes = len(all_classes)
        timing_init += time.time() - _timing_t0

        # 获取特征维度用于日志
        feature_dim = self.embedding_dim
        if ggeur_local_features:
            first_key = next(iter(ggeur_local_features.keys()))
            if len(ggeur_local_features[first_key]) > 0:
                feature_dim = ggeur_local_features[first_key].shape[1]

        logger.info(f"Client {self.ID}: Processing {total_classes} classes, feature_dim={feature_dim}, "
                   f"num_per_sample={num_per_sample}, num_per_prototype={num_per_prototype}")

        for idx, class_idx in enumerate(all_classes):
            class_idx = int(class_idx)
            class_features = []
            class_generated_masks = []

            # 显示进度
            if (idx + 1) % 10 == 0 or idx == 0:
                logger.info(f"Client {self.ID}: Augmenting class {idx+1}/{total_classes}")

            # 1. Original features from this client (always include)
            if class_idx in ggeur_local_features:
                _timing_t0 = time.time()
                original = ggeur_local_features[class_idx]
                class_features.append(original)
                class_generated_masks.append(ggeur_local_decoy_masks.get(
                    class_idx, np.zeros(len(original), dtype=bool)))
                original_samples += int(original.shape[0])
                timing_original_collect += time.time() - _timing_t0

            # Skip augmentation if disabled
            if no_augmentation:
                if class_features:
                    _timing_t0 = time.time()
                    combined = np.vstack(class_features)
                    all_features.append(combined)
                    all_labels.append(np.full(combined.shape[0], class_idx))
                    all_generated_masks.append(np.concatenate(class_generated_masks))
                    selected_samples += int(combined.shape[0])
                    timing_stack_select += time.time() - _timing_t0
                continue

            # 2. Get global covariance matrix
            _timing_t0 = time.time()
            if class_idx in self.global_cov_matrices:
                cov_matrix = self.global_cov_matrices[class_idx]
            else:
                cov_matrix = np.eye(self.embedding_dim) * 0.01
            timing_cov_lookup += time.time() - _timing_t0

            # 3. Expand original features using global covariance
            if (num_per_sample > 0 and class_idx in ggeur_local_features and
                    ggeur_local_features[class_idx].shape[0] > 0):
                source_masks = ggeur_local_decoy_masks.get(
                    class_idx,
                    np.zeros(len(ggeur_local_features[class_idx]), dtype=bool))
                for feat, source_is_decoy in zip(
                        ggeur_local_features[class_idx], source_masks):
                    _timing_t0 = time.time()
                    generated = self._generate_samples(feat, cov_matrix, num_per_sample)
                    timing_sample_generation += time.time() - _timing_t0
                    sample_generation_calls += 1
                    generated_sample_count += int(generated.shape[0])
                    class_features.append(generated)
                    class_generated_masks.append(np.full(
                        generated.shape[0], bool(source_is_decoy), dtype=bool))

            # 4. Generate from other clients' prototypes
            if use_cross_client and num_per_prototype > 0 and self.other_prototypes:
                if class_idx in self.other_prototypes:
                    for prototype in self.other_prototypes[class_idx]:
                        _timing_t0 = time.time()
                        generated = self._generate_samples(prototype, cov_matrix, num_per_prototype)
                        timing_prototype_generation += time.time() - _timing_t0
                        prototype_generation_calls += 1
                        generated_prototype_count += int(generated.shape[0])
                        class_features.append(generated)
                        class_generated_masks.append(np.zeros(
                            generated.shape[0], dtype=bool))

            # Combine and sample to target size
            if class_features:
                _timing_t0 = time.time()
                combined = np.vstack(class_features)
                combined_mask = np.concatenate(class_generated_masks)

                # target_size = 0 means use all samples
                if target_size > 0 and combined.shape[0] >= target_size:
                    indices = np.random.choice(combined.shape[0], target_size, replace=False)
                    selected = combined[indices]
                    selected_mask = combined_mask[indices]
                else:
                    selected = combined
                    selected_mask = combined_mask

                all_features.append(selected)
                all_labels.append(np.full(selected.shape[0], class_idx))
                all_generated_masks.append(selected_mask)
                selected_samples += int(selected.shape[0])
                timing_stack_select += time.time() - _timing_t0

        logger.info(f"Client {self.ID}: Augmentation complete, building dataset...")

        if all_features:
            _timing_t0 = time.time()
            base_features = np.vstack(all_features)
            base_labels = np.concatenate(all_labels)
            self.augmented_features = base_features
            self.augmented_labels = base_labels
            self.augmented_generated_mask = np.concatenate(
                all_generated_masks) if all_generated_masks else np.zeros(
                    len(base_labels), dtype=bool)
            train_features = base_features
            train_labels = base_labels

            decoy_features, decoy_labels = (None, None)
            if not decoy_before_ggeur:
                decoy_features, decoy_labels = \
                    self._generate_local_decoy_features()
            if decoy_features is not None and len(decoy_labels) > 0:
                self.augmented_features = np.vstack([
                    base_features, decoy_features])
                self.augmented_labels = np.concatenate([
                    base_labels, decoy_labels])
                self.augmented_generated_mask = np.concatenate([
                    np.zeros(len(base_labels), dtype=bool),
                    np.ones(len(decoy_labels), dtype=bool),
                ])

                if bool(getattr(
                        self._get_local_decoy_cfg(),
                        'train_with_decoy', True)):
                    train_features = self.augmented_features
                    train_labels = self.augmented_labels

            elif (decoy_before_ggeur and self._is_local_decoy_enabled() and
                  not bool(getattr(self._get_local_decoy_cfg(),
                                  'train_with_decoy', True))):
                # Keep decoy-derived GGEUR samples for attack evaluation, but
                # exclude them from the local classifier training set.
                keep = ~np.asarray(self.augmented_generated_mask, dtype=bool)
                train_features = self.augmented_features[keep]
                train_labels = self.augmented_labels[keep]
            elif (decoy_before_ggeur and self._is_local_decoy_enabled()):
                train_features = self.augmented_features
                train_labels = self.augmented_labels

            # Create data loader. Decoys enter training only when configured.
            dataset = AugmentedFeatureDataset(train_features, train_labels)
            self.augmented_loader = DataLoader(
                dataset,
                batch_size=self._cfg.dataloader.batch_size,
                shuffle=True
            )
            timing_dataset_build += time.time() - _timing_t0

            if no_augmentation and not self._is_local_decoy_enabled():
                logger.info(f"Client {self.ID}: Original data - {self.augmented_features.shape[0]} samples, "
                            f"{len(np.unique(self.augmented_labels))} classes")
            else:
                generated_count = int(np.sum(
                    self.augmented_generated_mask))
                logger.info(
                    f"Client {self.ID}: Augmented/decoy data - "
                    f"attack_samples={self.augmented_features.shape[0]}, "
                    f"train_samples={len(train_labels)}, "
                    f"generated_samples={generated_count}, "
                    f"classes={len(np.unique(self.augmented_labels))}")
            augmentation_elapsed = time.time() - augmentation_start
            aug_qps = (
                self.augmented_features.shape[0] / augmentation_elapsed
                if augmentation_elapsed > 0 else 0.0
            )
            logger.info(
                f"Client {self.ID}: Augmentation timing - "
                f"samples={self.augmented_features.shape[0]}, "
                f"time={augmentation_elapsed:.4f}s, "
                f"qps={aug_qps:.2f} samples/s")
            _timing_t0 = time.time()
            self._save_augmented_feature_cache()
            timing_cache_save += time.time() - _timing_t0

        self.augmentation_done = True
        output_samples = (
            int(self.augmented_features.shape[0])
            if self.augmented_features is not None else 0)
        output_classes = (
            int(len(np.unique(self.augmented_labels)))
            if self.augmented_labels is not None else 0)
        logger.info(
            "GGEUR_TIMING_CLIENT "
            f"client={int(self.ID)} stage=augmentation "
            f"dataset={self._cfg.data.type} extractor={self.feature_extractor_type} "
            f"cache_hit=0 no_augmentation={1 if no_augmentation else 0} "
            f"total_sec={time.time() - augmentation_start:.6f} "
            f"cache_load_sec={timing_cache_load:.6f} "
            f"init_sec={timing_init:.6f} "
            f"original_collect_sec={timing_original_collect:.6f} "
            f"cov_lookup_sec={timing_cov_lookup:.6f} "
            f"sample_generation_sec={timing_sample_generation:.6f} "
            f"prototype_generation_sec={timing_prototype_generation:.6f} "
            f"stack_select_sec={timing_stack_select:.6f} "
            f"dataset_build_sec={timing_dataset_build:.6f} "
            f"cache_save_sec={timing_cache_save:.6f} "
            f"classes={total_classes} feature_dim={feature_dim} "
            f"original_samples={original_samples} "
            f"generated_from_samples={generated_sample_count} "
            f"generated_from_prototypes={generated_prototype_count} "
            f"selected_samples={selected_samples} output_samples={output_samples} "
            f"output_classes={output_classes} "
            f"sample_generation_calls={sample_generation_calls} "
            f"prototype_generation_calls={prototype_generation_calls} "
            f"generated_per_sample={num_per_sample} "
            f"generated_per_prototype={num_per_prototype} "
            f"target_size_per_class={target_size}")

        # Free raw features from memory - augmented_features/loader are all we need now
        self.local_features = {}
        self.local_labels = {}
