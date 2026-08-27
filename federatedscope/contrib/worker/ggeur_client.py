"""
GGEUR_Clip Client Implementation

Handles:
1. CLIP feature extraction from local images
2. Local statistics computation (mean, covariance per class)
3. Feature augmentation using global covariance from server
4. Standard FedAvg training on augmented features
5. (Optional) CNN training with knowledge distillation from MLP teacher
6. (Optional) CNN training with feature alignment (from scratch)
"""

import os
import time
import logging
import copy
import base64
import hashlib
import io
import json
import zlib
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import Dataset, DataLoader

from federatedscope.core.message import Message, b64serializer
from federatedscope.core.auxiliaries.utils import param2tensor
from federatedscope.core.workers.client import Client
from federatedscope.register import register_worker

logger = logging.getLogger(__name__)

# Shared BERT extractors avoid loading one encoder per standalone client.
_SHARED_BERT_EXTRACTORS = {}

# In standalone evaluation every client receives the same global covariance
# matrices, but deserialization gives each client a different ndarray object.
# Cache factors by matrix content so the expensive high-dimensional
# eigendecomposition/Cholesky step is performed once per class, not once per
# class and client.  The cache is process-local and does not change the sampled
# Gaussian distribution.
_SHARED_COV_FACTOR_CACHE = {}
_SHARED_COV_FACTOR_CACHE_MAXSIZE = 128


class AugmentedFeatureDataset(Dataset):
    """Dataset for augmented CLIP features"""

    def __init__(self, features, labels):
        if isinstance(features, np.ndarray):
            self.features = torch.from_numpy(features).float()
        else:
            self.features = features.float()

        if isinstance(labels, np.ndarray):
            self.labels = torch.from_numpy(labels).long()
        else:
            self.labels = labels.long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class AugmentedImageDataset(Dataset):
    """
    Wrapper dataset that applies strong data augmentation for CNN training.

    This is CRITICAL for preventing overfitting when training CNN from scratch.
    The augmentation helps CNN learn more generalizable features.
    """

    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

        # 强数据增强变换
        from torchvision import transforms
        self.augment_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomGrayscale(p=0.1),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        ])

        # 标准化（与原始数据集一致）
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        self.to_pil = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # 获取原始数据
        item = self.base_dataset[idx]
        if len(item) >= 2:
            image, label = item[0], item[1]
        else:
            return item

        # 如果image已经是tensor，转换为PIL进行增强
        if isinstance(image, torch.Tensor):
            # 反归一化（假设已经用ImageNet均值/标准差归一化）
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image_denorm = image * std + mean
            image_denorm = torch.clamp(image_denorm, 0, 1)

            # 转为PIL
            pil_image = self.to_pil(image_denorm)

            # 应用增强
            augmented = self.augment_transform(pil_image)

            # 转回tensor并归一化
            image = self.normalize(self.to_tensor(augmented))

        return image, label


class GGEURClient(Client):
    """
    GGEUR_Clip Client that:
    1. Extracts features (CLIP or CNN) and computes local statistics
    2. Receives global covariance matrices from server
    3. Performs GGEUR_Clip feature augmentation
    4. Trains MLP classifier on augmented features
    5. (Optional) End-to-end fine-tuning with CNN backbone
    """

    # Shared CLIP model for PromptFL text encoding - loaded once, reused by all clients
    _shared_prompt_clip = None        # open_clip model (CPU when idle, GPU during forward)
    _shared_prompt_tokenizer = None
    _shared_prompt_clip_name = None   # track which model is loaded

    @staticmethod
    def _decode_parameter_tree(payload):
        """Restore tensor leaves serialized by ``Message.transform``.

        Distributed ``model_para`` messages pickle/base64-encode tensor
        leaves before gRPC transport. Control strings such as the separated
        training phase must stay unchanged, so failed tensor decoding falls
        back to the original value.
        """
        if isinstance(payload, dict):
            return {
                key: GGEURClient._decode_parameter_tree(value)
                for key, value in payload.items()
            }
        if isinstance(payload, list):
            # ``model_para`` tensor leaves can arrive in either of two wire
            # representations.  FederatedScope normally pickles tensor
            # leaves into base64 bytes, but protobuf also represents a plain
            # numeric tensor as nested Python lists.  Recursing first would
            # preserve the latter as a list and ``load_state_dict`` would then
            # reject it.  Convert a homogeneous numeric list (including a
            # nested matrix) as one tensor; heterogeneous control lists still
            # recurse element-by-element.
            try:
                restored = param2tensor(payload)
                if isinstance(restored, torch.Tensor):
                    return restored
            except (TypeError, ValueError, RuntimeError):
                pass
            return [
                GGEURClient._decode_parameter_tree(value)
                for value in payload
            ]
        if isinstance(payload, tuple):
            return tuple(
                GGEURClient._decode_parameter_tree(value)
                for value in payload
            )
        if isinstance(payload, (str, bytes)):
            # Platform inference heads can use the compact ``znp:`` ndarray
            # representation.  It is not a FederatedScope tensor payload, so
            # decode it before calling ``param2tensor`` (which otherwise may
            # interpret the string as a scalar and fail during evaluation).
            text_payload = payload.decode('utf-8') \
                if isinstance(payload, bytes) else payload
            if text_payload.startswith('znp:'):
                decoded_array = GGEURClient._payload_to_ndarray(text_payload)
                if isinstance(decoded_array, np.ndarray):
                    return decoded_array
            try:
                return param2tensor(payload)
            except Exception:
                return payload
        return payload

    def __init__(self, ID=-1, server_id=None, state=-1, config=None,
                 data=None, model=None, device='cpu', strategy=None,
                 is_unseen_client=False, *args, **kwargs):
        super(GGEURClient, self).__init__(ID, server_id, state, config,
                                          data, model, device, strategy,
                                          is_unseen_client, *args, **kwargs)

        if config is None:
            return

        self.ggeur_cfg = config.ggeur
        self.head_only_mode = getattr(self.ggeur_cfg, 'head_only_mode', False)
        if self.head_only_mode:
            logger.info(
                f"Client {self.ID}: GGEUR HeadOnly system mode active. "
                "Round 0 uses the real backbone for feature statistics and "
                "augmentation; later rounds train the MLP head on augmented "
                "feature cache.")

        # ===== Feature Extractor Mode =====
        # 'clip': Use CLIP (ViT-based, original method)
        # 'cnn': Use pretrained CNN (ConvNeXt, ResNet, etc.)
        # 'timm': Use any timm vision backbone (e.g., GFNet / Mixer / etc.)
        self.feature_extractor_type = getattr(self.ggeur_cfg, 'feature_extractor', 'clip')

        # CLIP model (for 'clip' mode)
        self.clip_model = None
        self.clip_preprocess = None

        # CNN feature extractor (for 'cnn' mode)
        self.cnn_extractor = None

        # timm feature extractor (for 'timm' mode)
        self.timm_extractor = None

        # BERT feature extractor (for 'bert' text mode)
        self.bert_model = None
        self.bert_tokenizer = None

        # The actual embedding dimension used by the feature extractor.
        # (May differ from cfg.ggeur.embedding_dim if user changes backbone.)
        self.embedding_dim = getattr(self.ggeur_cfg, 'embedding_dim', 512)

        # Local features and labels
        self.local_features = {}  # {class_idx: features array}
        self.local_labels = {}

        # Local statistics
        self.local_means = {}  # {class_idx: mean vector}
        self.local_covs = {}  # {class_idx: covariance matrix}
        self.local_counts = {}  # {class_idx: sample count}

        # Global covariance from server
        self.global_cov_matrices = None
        self.other_prototypes = None  # Prototypes from other clients
        self._cov_factor_cache = {}

        # Global prototypes for feature alignment
        self.global_prototypes = None  # {class_idx: mean vector}
        self.domain_prototype_ensemble = {}
        self.domain_personalized_head = None

        # Augmented data
        self.augmented_features = None
        self.augmented_labels = None
        self.augmented_loader = None
        self._task_adaptation_cache = None

        # Real local features saved before augmentation (for PromptFL training)
        self.real_local_features = {}
        self.prompt_loader = None

        # MLP classifier
        self.mlp_classifier = None

        # State tracking
        self.statistics_uploaded = False
        self.augmentation_done = False

        # ===== End-to-End Fine-tuning Mode (for CNN) =====
        self.use_end_to_end_finetune = getattr(self.ggeur_cfg, 'use_end_to_end_finetune', False)
        self.finetune_start_round = getattr(self.ggeur_cfg, 'finetune_start_round', 0)
        self.full_model = None  # Combined CNN + classifier for fine-tuning
        self.original_image_loader = None

        # ===== Legacy modes (for backward compatibility) =====
        self.use_cnn_distillation = getattr(self.ggeur_cfg, 'use_cnn_distillation', False)
        self.use_feature_alignment = getattr(self.ggeur_cfg, 'use_feature_alignment', False)
        self.cnn_model = None
        self.use_separated_training = getattr(self.ggeur_cfg, 'use_separated_training', False)
        self.training_phase = 'classifier'
        self.pretrained_classifier = None
        self.cnn_backbone = None

        # ===== MOON Mode =====
        self.use_moon = getattr(self.ggeur_cfg, 'use_moon', False)
        self.moon_prev_model = None    # Previous local model snapshot
        self.moon_global_model = None  # Global model snapshot (received this round)

        # ===== FedProto Mode =====
        # Preserve the representation-prototype exchange implemented by the
        # original MDSent RNN/LSTM branch.  These prototypes live in the
        # trainable head's hidden space (256 by default), not in the frozen
        # BERT input space (768 by default).
        self.use_fedproto = bool(
            getattr(self.ggeur_cfg, 'use_fedproto', False))
        self.fedproto_proto_weight = float(
            getattr(
                self.ggeur_cfg, 'fedproto_proto_weight',
                getattr(self.ggeur_cfg, 'proto_weight', 1.0)))
        self.fedproto_distance_metric = str(
            getattr(
                self.ggeur_cfg, 'fedproto_distance_metric',
                getattr(self.ggeur_cfg, 'proto_distance', 'mse'))).lower()
        self.fedproto_normalize = bool(
            getattr(self.ggeur_cfg, 'fedproto_normalize', False))
        self.fedproto_global_prototypes = {}
        self.fedproto_local_prototypes = {}
        self.fedproto_local_counts = {}

        # ===== PromptFL Mode =====
        self.use_promptfl = getattr(self.ggeur_cfg, 'use_promptfl', False)
        self.prompt_learner = None
        self.text_encoder = None
        self.custom_clip = None        # CustomCLIP (HuggingFace-based) for PromptFL
        self.hf_clip_model = None      # HuggingFace CLIPModel (separate from open_clip)
        self.global_prompt_ctx = None  # Latest global PromptFL context
        self.fail_after_stage = str(
            getattr(self.ggeur_cfg, 'fail_after_stage', '') or '')
        self.fail_on_round = int(getattr(self.ggeur_cfg, 'fail_on_round', -1))

    def _maybe_fail_for_distributed_validation(self, stage, round_idx=None):
        """Intentional process exit hook for distributed scenario tests."""
        if not self.fail_after_stage:
            return
        if self.fail_after_stage != stage:
            return
        if round_idx is not None and self.fail_on_round >= 0 and \
                int(round_idx) != int(self.fail_on_round):
            return
        logger.error(
            f"Client {self.ID}: Fault injection exit at stage={stage}, "
            f"round={round_idx}")
        raise SystemExit(
            f"GGEUR distributed validation fault injection: {stage}")

    def _register_default_handlers(self):
        """Register message handlers"""
        super()._register_default_handlers()

        # Register handler for receiving global covariance matrices
        self.register_handlers('global_covariances',
                               self.callback_for_global_covariances)
        self.register_handlers('client_eval',
                               self.callback_for_client_eval)

    def _load_feature_extractor(self):
        """Load feature extractor (CLIP / CNN / timm / BERT)."""
        if self.feature_extractor_type == 'cnn':
            self._load_cnn_extractor()
        elif self.feature_extractor_type == 'timm':
            self._load_timm_extractor()
        elif self.feature_extractor_type == 'bert':
            self._load_bert_extractor()
        else:
            self._load_clip_model()

    def _unload_feature_extractor(self):
        """Unload feature extractor from GPU to free VRAM after feature caching."""
        unloaded = False
        if self.clip_model is not None:
            self.clip_model = self.clip_model.cpu()
            del self.clip_model
            self.clip_model = None
            unloaded = True
        if self.cnn_extractor is not None:
            self.cnn_extractor = self.cnn_extractor.cpu()
            del self.cnn_extractor
            self.cnn_extractor = None
            unloaded = True
        if self.timm_extractor is not None:
            self.timm_extractor = self.timm_extractor.cpu()
            del self.timm_extractor
            self.timm_extractor = None
            unloaded = True
        if self.bert_model is not None:
            self.bert_model = self.bert_model.cpu()
            self.bert_model = None
            self.bert_tokenizer = None
            unloaded = True
        if unloaded:
            torch.cuda.empty_cache()
            logger.info(f"Client {self.ID}: Feature extractor unloaded from GPU")

    def _load_cnn_extractor(self):
        """Load CNN feature extractor"""
        if self.cnn_extractor is not None:
            return

        try:
            from federatedscope.contrib.model.ggeur_cnn_extractor import CNNFeatureExtractor

            model_name = getattr(self.ggeur_cfg, 'cnn_backbone', 'convnext_base')
            pretrained = getattr(self.ggeur_cfg, 'cnn_pretrained', True)
            checkpoint_path = getattr(self.ggeur_cfg, 'cnn_checkpoint_path', '')
            freeze = getattr(self.ggeur_cfg, 'freeze_backbone', True)

            self.cnn_extractor = CNNFeatureExtractor(
                model_name=model_name,
                pretrained=pretrained,
                freeze=freeze,
                checkpoint_path=checkpoint_path,
            )
            self.cnn_extractor = self.cnn_extractor.to(self.device)

            self.embedding_dim = int(self.cnn_extractor.get_feature_dim())
            logger.info(f"Client {self.ID}: Loaded CNN extractor {model_name}, "
                       f"feature_dim={self.cnn_extractor.get_feature_dim()}")

        except Exception as e:
            logger.error(f"Client {self.ID}: Failed to load CNN extractor: {e}")
            raise

    def _load_timm_extractor(self):
        """Load timm feature extractor"""
        if self.timm_extractor is not None:
            return

        try:
            from federatedscope.contrib.model.ggeur_timm_extractor import TimmFeatureExtractor

            model_name = getattr(self.ggeur_cfg, 'timm_model', 'gfnet_tiny')
            pretrained = getattr(self.ggeur_cfg, 'timm_pretrained', True)
            checkpoint_path = getattr(self.ggeur_cfg, 'timm_checkpoint_path', '')
            freeze = getattr(self.ggeur_cfg, 'freeze_backbone', True)
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

            self.embedding_dim = int(self.timm_extractor.get_feature_dim())
            logger.info(f"Client {self.ID}: Loaded timm extractor {model_name}, "
                        f"feature_dim={self.embedding_dim}")

        except Exception as e:
            logger.error(f"Client {self.ID}: Failed to load timm extractor: {e}")
            raise

    def _load_clip_model(self):
        """Load CLIP model for feature extraction"""
        if self.clip_model is not None:
            return

        try:
            import open_clip

            model_name = self.ggeur_cfg.clip_model
            pretrained = self.ggeur_cfg.clip_pretrained
            local_path = self.ggeur_cfg.clip_model_path

            # Check for local model path first
            if local_path and os.path.exists(local_path):
                logger.info(f"Client {self.ID}: Loading CLIP from local path: {local_path}")
                self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                    model_name, pretrained=local_path
                )
            else:
                # Try to load from pretrained source (may require internet)
                logger.info(f"Client {self.ID}: Loading CLIP from pretrained: {pretrained}")
                try:
                    self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                        model_name, pretrained=pretrained
                    )
                except Exception as e:
                    logger.error(f"Failed to download CLIP model. Please set ggeur.clip_model_path to local weights file.")
                    logger.error(f"Error: {e}")
                    raise

            self.clip_model = self.clip_model.to(self.device)
            self.clip_model.eval()
            visual_dim = getattr(getattr(self.clip_model, 'visual', None),
                                 'output_dim', None)
            if visual_dim is None and hasattr(self.clip_model,
                                              'text_projection'):
                visual_dim = int(self.clip_model.text_projection.shape[1])
            if visual_dim is not None:
                self.embedding_dim = int(visual_dim)
            logger.info(f"Client {self.ID}: Loaded CLIP model {model_name}")

        except ImportError:
            logger.error("open_clip not installed. Please install: pip install open_clip_torch")
            raise

    def _load_bert_extractor(self):
        """Load a frozen BERT encoder for text feature extraction."""
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
            logger.info(
                f"Client {self.ID}: Loaded shared BERT extractor from "
                f"{model_path}")

        self.bert_tokenizer, self.bert_model = _SHARED_BERT_EXTRACTORS[key]
        self.bert_model = self.bert_model.to(self.device)
        self.bert_model.eval()
        hidden_size = getattr(getattr(self.bert_model, 'config', None),
                              'hidden_size', None)
        if hidden_size is not None:
            self.embedding_dim = int(hidden_size)
        logger.info(
            f"Client {self.ID}: BERT extractor ready, "
            f"feature_dim={self.embedding_dim}")

    def _get_feature_cache_path(self, domain=None):
        """Get the path for cached CLIP features"""
        # Check if caching is enabled
        if not getattr(self.ggeur_cfg, 'use_feature_cache', True):
            return None

        # Get cache directory from config or use default
        cache_dir = getattr(self.ggeur_cfg, 'feature_cache_dir', '')
        if not cache_dir:
            cache_dir = os.path.join(os.path.dirname(self._cfg.data.root), 'clip_feature_cache')

        os.makedirs(cache_dir, exist_ok=True)

        # Build cache filename based on dataset, domain, and model
        dataset_name = self._cfg.data.type.lower()

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

        dim_str = f"d{int(self.embedding_dim)}"
        if domain:
            cache_filename = (
                f"{dataset_name}_{domain}_{prefix}_{model_str}_{dim_str}.npz")
        else:
            cache_filename = f"{dataset_name}_{prefix}_{model_str}_{dim_str}.npz"

        return os.path.join(cache_dir, cache_filename)

    def _load_feature_cache(self, cache_path):
        """Load cached features from file"""
        if cache_path is None or not os.path.exists(cache_path):
            return {}

        try:
            data = np.load(cache_path, allow_pickle=True)
            # Cache format: {'paths': array of paths, 'features': array of features}
            if 'paths' in data and 'features' in data:
                paths = data['paths']
                features = data['features']
                cache = {
                    self._feature_cache_key(p): f
                    for p, f in zip(paths, features)
                }
                logger.info(f"Client {self.ID}: Loaded {len(cache)} cached features from {cache_path}")
                return cache
        except Exception as e:
            logger.warning(f"Client {self.ID}: Failed to load cache: {e}")

        return {}

    def _save_feature_cache(self, cache_path, feature_cache):
        """Save features to cache file"""
        if cache_path is None:
            return

        try:
            normalized_cache = {
                self._feature_cache_key(path): feature
                for path, feature in feature_cache.items()
            }
            paths = list(normalized_cache.keys())
            features = np.array([normalized_cache[p] for p in paths])
            tmp_path = (
                f"{cache_path}.client{int(self.ID)}."
                f"pid{os.getpid()}.tmp.npz")
            np.savez(tmp_path, paths=np.array(paths), features=features)
            os.replace(tmp_path, cache_path)
            logger.info(
                f"Client {self.ID}: Saved {len(paths)} features to cache "
                f"{cache_path}")
        except Exception as e:
            logger.warning(f"Client {self.ID}: Failed to save cache: {e}")

    def _feature_cache_key(self, sample_id):
        """Return a portable cache key shared by Linux and Windows hosts.

        Vision datasets expose absolute image paths, which previously made a
        cache produced on Linux unusable on Windows.  Strip the configured
        dataset-root prefix (or its basename when crossing OS path styles),
        normalize separators, and compare case-insensitively.  Text sample
        ids are stable already and pass through the same harmless separator
        normalization.
        """
        value = str(sample_id).replace('\\', '/').strip()
        while '//' in value:
            value = value.replace('//', '/')

        data_root = str(getattr(self._cfg.data, 'root', '') or '')
        root = data_root.replace('\\', '/').rstrip('/')
        while '//' in root:
            root = root.replace('//', '/')
        if root:
            value_folded = value.casefold()
            root_folded = root.casefold()
            if value_folded == root_folded:
                value = ''
            elif value_folded.startswith(root_folded + '/'):
                value = value[len(root) + 1:]
            else:
                root_name = root.rsplit('/', 1)[-1]
                marker = f'/{root_name.casefold()}/'
                marker_index = value_folded.find(marker)
                if marker_index >= 0:
                    value = value[marker_index + len(marker):]
        return value.lstrip('./').casefold()

    def _is_valid_feature_vector(self, feat):
        """Check whether a cached/extracted feature matches current dim."""
        if feat is None:
            return False

        feat_arr = np.asarray(feat)
        if feat_arr.ndim != 1:
            return False

        expected_dim = int(getattr(self, 'embedding_dim',
                                   getattr(self.ggeur_cfg,
                                           'embedding_dim', 0)))
        if expected_dim > 0 and feat_arr.shape[0] != expected_dim:
            return False

        return True

    @staticmethod
    def _resolve_dataset_sample(dataset, idx):
        """Return (text, label, sample_id, domain) for text datasets."""
        from torch.utils.data import Subset

        if isinstance(dataset, Subset):
            return GGEURClient._resolve_dataset_sample(
                dataset.dataset, dataset.indices[idx])

        sample = dataset[idx]
        if not isinstance(sample, (tuple, list)) or len(sample) < 2:
            raise ValueError('BERT feature extraction expects text-label data')

        text = str(sample[0])
        label = int(sample[1])
        if hasattr(dataset, 'get_id'):
            sample_id = str(dataset.get_id(idx))
        elif hasattr(dataset, 'ids') and len(dataset.ids) > idx:
            sample_id = str(dataset.ids[idx])
        else:
            domain = getattr(dataset, 'domain', 'sample')
            sample_id = f'{domain}:{idx}'
        domain = getattr(dataset, 'domain', None)
        return text, label, sample_id, domain

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

    def _extract_text_features(self):
        """Extract frozen BERT embeddings from local text data."""
        timing_total_start = time.time()
        logger.info(f"Client {self.ID}: Extracting BERT text features...")

        train_data = self.trainer.ctx.data.get('train', None)
        if train_data is None:
            train_data = self.data.get('train', None)
        if train_data is None:
            logger.error(f"Client {self.ID}: No training data available")
            return

        dataset = train_data.dataset if hasattr(train_data, 'dataset') \
            else train_data
        domain = getattr(dataset, 'domain', None)
        if domain is None and hasattr(dataset, 'dataset'):
            domain = getattr(dataset.dataset, 'domain', None)

        timing_load_extractor = 0.0

        cache_path = self._get_feature_cache_path(domain)
        t0 = time.time()
        feature_cache = self._load_feature_cache(cache_path)
        timing_load_cache = time.time() - t0
        if feature_cache:
            feature_cache = {
                key: value
                for key, value in feature_cache.items()
                if self._is_valid_feature_vector(value)
            }

        self.local_features = {}
        self.local_labels = {}
        samples = []
        for idx in range(len(dataset)):
            text, label, sample_id, sample_domain = \
                self._resolve_dataset_sample(dataset, idx)
            if domain is None and sample_domain is not None:
                domain = sample_domain
            samples.append((text, label,
                            self._feature_cache_key(sample_id)))

        missing = [
            item for item in samples
            if item[2] not in feature_cache or not self._is_valid_feature_vector(
                feature_cache.get(item[2]))
        ]

        timing_forward = 0.0
        if missing:
            if bool(getattr(
                    self.ggeur_cfg,
                    'require_complete_feature_cache',
                    False)):
                preview = ', '.join(item[2] for item in missing[:3])
                raise RuntimeError(
                    f"Client {self.ID}: BERT feature cache is incomplete; "
                    f"missing {len(missing)}/{len(samples)} samples "
                    f"(first keys: {preview})")
            t0 = time.time()
            self._load_bert_extractor()
            timing_load_extractor = time.time() - t0
            batch_size = int(getattr(self.ggeur_cfg, 'bert_batch_size', 32))
            for start in range(0, len(missing), max(batch_size, 1)):
                batch = missing[start:start + max(batch_size, 1)]
                texts = [item[0] for item in batch]
                t0 = time.time()
                features = self._encode_text_batch(texts)
                timing_forward += time.time() - t0
                for feat, (_, _, sample_id) in zip(features, batch):
                    feature_cache[sample_id] = feat
            self._save_feature_cache(cache_path, feature_cache)

        for _, label, sample_id in samples:
            feat = feature_cache.get(sample_id)
            if not self._is_valid_feature_vector(feat):
                logger.warning(
                    f"Client {self.ID}: Skip invalid BERT feature "
                    f"for sample {sample_id}")
                continue
            self.local_features.setdefault(int(label), []).append(feat)
            self.local_labels.setdefault(int(label), []).append(int(label))

        for label in self.local_features:
            self.local_features[label] = np.asarray(
                self.local_features[label], dtype=np.float32)

        total_samples = sum(len(value) for value in self.local_features.values())
        logger.info(
            f"Client {self.ID}: Extracted {total_samples} BERT features "
            f"from {len(self.local_features)} classes")
        logger.info(
            "GGEUR_TIMING_CLIENT "
            f"client={int(self.ID)} stage=feature_extraction "
            f"dataset={self._cfg.data.type} extractor=bert "
            f"total_sec={time.time() - timing_total_start:.6f} "
            f"load_extractor_sec={timing_load_extractor:.6f} "
            f"load_cache_sec={timing_load_cache:.6f} "
            f"forward_sec={timing_forward:.6f} "
            f"samples_total={len(samples)} "
            f"samples_cached={len(samples) - len(missing)} "
            f"samples_need_extract={len(missing)} "
            f"output_samples={total_samples} "
            f"classes={len(self.local_features)} "
            f"feature_dim={self.embedding_dim}")

    def _extract_features(self):
        """
        Extract features from local data using either CLIP or CNN.
        Supports caching for both modes.
        """
        if self.feature_extractor_type == 'bert':
            self._extract_text_features()
            return

        timing_total_start = time.time()
        timing_load_extractor = 0.0
        timing_load_cache = 0.0
        timing_scan_cache = 0.0
        timing_image_load = 0.0
        timing_forward = 0.0
        timing_cache_save = 0.0
        samples_total = 0
        samples_cached = 0
        samples_need_extract = 0
        if self.feature_extractor_type == 'cnn':
            extractor_name = 'CNN'
        elif self.feature_extractor_type == 'timm':
            extractor_name = 'timm'
        else:
            extractor_name = 'CLIP'
        logger.info(f"Client {self.ID}: Extracting {extractor_name} features...")

        # Get train data
        train_data = self.trainer.ctx.data.get('train', None)
        if train_data is None:
            train_data = self.data.get('train', None)

        if train_data is None:
            logger.error(f"Client {self.ID}: No training data available")
            return

        # Get the underlying dataset (handle DataLoader -> Dataset -> possibly Subset)
        dataset = train_data.dataset if hasattr(train_data, 'dataset') else train_data

        # Handle Subset wrapper (when multiple clients share one domain)
        from torch.utils.data import Subset
        is_subset = isinstance(dataset, Subset)
        if is_subset:
            base_dataset = dataset.dataset
            subset_indices = dataset.indices
            domain = getattr(base_dataset, 'domain', None)
        else:
            base_dataset = dataset
            subset_indices = None
            domain = getattr(dataset, 'domain', None)

        cache_path = self._get_feature_cache_path(domain)

        # Load existing cache
        _timing_t0 = time.time()
        feature_cache = self._load_feature_cache(cache_path)
        timing_load_cache += time.time() - _timing_t0
        if feature_cache:
            valid_cache = {
                path: feat
                for path, feat in feature_cache.items()
                if self._is_valid_feature_vector(feat)
            }
            dropped = len(feature_cache) - len(valid_cache)
            if dropped:
                logger.warning(
                    f"Client {self.ID}: Dropped {dropped} cached features "
                    f"with wrong dim; expected {self.embedding_dim}")
            feature_cache = valid_cache
        cache_updated = False

        # Check if base dataset has image paths (PACS, Office-Home style)
        has_paths = hasattr(base_dataset, 'data') and len(base_dataset.data) > 0 and isinstance(base_dataset.data[0], str)

        self.local_features = {}
        self.local_labels = {}

        # QPS tracking
        _qps_samples = 0
        _qps_time = 0.0

        if has_paths:
            # Dataset with image paths - can use caching
            num_samples = len(subset_indices) if is_subset else len(base_dataset)
            samples_total = int(num_samples)
            logger.info(f"Client {self.ID}: Dataset has {num_samples} samples with paths")

            # Collect samples that need feature extraction
            paths_to_extract = []
            indices_to_extract = []  # Indices into the current dataset (Subset or base)
            base_indices_to_extract = []  # Indices into base_dataset for loading

            _timing_t0 = time.time()
            for local_idx in range(num_samples):
                # Get the index into the base dataset
                if is_subset:
                    base_idx = subset_indices[local_idx]
                else:
                    base_idx = local_idx

                img_path = base_dataset.data[base_idx]
                cache_key = self._feature_cache_key(img_path)
                label = base_dataset.targets[base_idx]

                if cache_key in feature_cache:
                    # Use cached feature
                    feat = feature_cache[cache_key]
                    if not self._is_valid_feature_vector(feat):
                        logger.warning(
                            f"Client {self.ID}: Skip invalid cached feature "
                            f"for {img_path} with shape "
                            f"{np.asarray(feat).shape}, expected dim "
                            f"{self.embedding_dim}")
                        paths_to_extract.append(cache_key)
                        indices_to_extract.append(local_idx)
                        base_indices_to_extract.append(base_idx)
                        continue
                    label = int(label)
                    if label not in self.local_features:
                        self.local_features[label] = []
                        self.local_labels[label] = []
                    self.local_features[label].append(feat)
                    self.local_labels[label].append(label)
                else:
                    # Need to extract
                    paths_to_extract.append(cache_key)
                    indices_to_extract.append(local_idx)
                    base_indices_to_extract.append(base_idx)
            timing_scan_cache += time.time() - _timing_t0

            cached_count = num_samples - len(paths_to_extract)
            samples_cached = int(cached_count)
            samples_need_extract = int(len(paths_to_extract))
            logger.info(f"Client {self.ID}: {cached_count} samples from cache, {len(paths_to_extract)} need extraction")

            # Extract features for non-cached samples
            if paths_to_extract:
                if bool(getattr(self.ggeur_cfg,
                                'require_complete_feature_cache', False)):
                    preview = ', '.join(paths_to_extract[:3])
                    raise RuntimeError(
                        f"Client {self.ID}: feature cache is incomplete; "
                        f"missing {len(paths_to_extract)}/{num_samples} "
                        f"samples (first keys: {preview})")
                _timing_t0 = time.time()
                self._load_feature_extractor()
                timing_load_extractor += time.time() - _timing_t0
                # Create a mini dataloader for samples to extract
                batch_size = getattr(self.ggeur_cfg, 'extract_batch_size', 64)
                use_fp16 = getattr(self.ggeur_cfg, 'use_fp16_extraction', True) and torch.cuda.is_available()
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_fp16):
                    for i in range(0, len(base_indices_to_extract), batch_size):
                        batch_base_indices = base_indices_to_extract[i:i + batch_size]
                        batch_paths = paths_to_extract[i:i + batch_size]

                        # Load images from base dataset
                        images = []
                        labels = []
                        _timing_t0 = time.time()
                        for base_idx in batch_base_indices:
                            img, lbl = base_dataset[base_idx]
                            images.append(img)
                            labels.append(lbl)
                        timing_image_load += time.time() - _timing_t0

                        images = torch.stack(images).to(self.device)

                        # Extract features using appropriate extractor
                        _t0 = time.time()
                        if self.feature_extractor_type == 'cnn':
                            features = self.cnn_extractor(images)
                        elif self.feature_extractor_type == 'timm':
                            features = self.timm_extractor(images)
                        else:
                            features = self.clip_model.encode_image(images)
                        _forward_elapsed = time.time() - _t0
                        _qps_samples += len(images)
                        _qps_time += _forward_elapsed
                        timing_forward += _forward_elapsed
                        features = features.cpu().numpy()

                        for feat, label, path in zip(features, labels, batch_paths):
                            if not self._is_valid_feature_vector(feat):
                                logger.warning(
                                    f"Client {self.ID}: Skip extracted "
                                    f"feature for {path} with invalid shape "
                                    f"{np.asarray(feat).shape}, expected dim "
                                    f"{self.embedding_dim}")
                                continue
                            # Update cache
                            feature_cache[path] = feat
                            cache_updated = True

                            # Add to local features
                            label = int(label)
                            if label not in self.local_features:
                                self.local_features[label] = []
                                self.local_labels[label] = []
                            self.local_features[label].append(feat)
                            self.local_labels[label].append(label)

                # Save updated cache
                if cache_updated:
                    _timing_t0 = time.time()
                    self._save_feature_cache(cache_path, feature_cache)
                    timing_cache_save += time.time() - _timing_t0

        else:
            # Fallback: Dataset without paths - cannot use caching
            logger.info(f"Client {self.ID}: Dataset does not have image paths, caching disabled")

            _timing_t0 = time.time()
            self._load_feature_extractor()
            timing_load_extractor += time.time() - _timing_t0

            use_fp16 = getattr(self.ggeur_cfg, 'use_fp16_extraction', True) and torch.cuda.is_available()
            extract_batch_size = getattr(self.ggeur_cfg, 'extract_batch_size', 64)
            dataloader = DataLoader(dataset, batch_size=extract_batch_size, shuffle=False)
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_fp16):
                for batch in dataloader:
                    if len(batch) >= 2:
                        images, labels = batch[0], batch[1]
                    else:
                        continue

                    images = images.to(self.device)
                    samples_total += int(images.shape[0])
                    samples_need_extract += int(images.shape[0])

                    # Skip invalid images
                    if images.shape[1] != 3:
                        continue

                    # Extract features using appropriate extractor
                    _t0 = time.time()
                    if self.feature_extractor_type == 'cnn':
                        features = self.cnn_extractor(images)
                    elif self.feature_extractor_type == 'timm':
                        features = self.timm_extractor(images)
                    else:
                        features = self.clip_model.encode_image(images)
                    _forward_elapsed = time.time() - _t0
                    _qps_samples += len(images)
                    _qps_time += _forward_elapsed
                    timing_forward += _forward_elapsed
                    features = features.cpu().numpy()
                    labels = labels.cpu().numpy()

                    for feat, label in zip(features, labels):
                        if not self._is_valid_feature_vector(feat):
                            logger.warning(
                                f"Client {self.ID}: Skip extracted feature "
                                f"with invalid shape {np.asarray(feat).shape}, "
                                f"expected dim {self.embedding_dim}")
                            continue
                        label = int(label)
                        if label not in self.local_features:
                            self.local_features[label] = []
                            self.local_labels[label] = []
                        self.local_features[label].append(feat)
                        self.local_labels[label].append(label)

        # Convert lists to numpy arrays
        for label in self.local_features:
            self.local_features[label] = np.array(self.local_features[label])

        total_samples = sum(len(v) for v in self.local_features.values())
        if _qps_time > 0:
            qps = _qps_samples / _qps_time
            logger.info(f"Client {self.ID}: Feature extraction QPS={qps:.1f} img/s "
                        f"({_qps_samples} samples in {_qps_time:.2f}s)")
        logger.info(f"Client {self.ID}: Extracted {total_samples} {extractor_name} features from {len(self.local_features)} classes")
        timing_total = time.time() - timing_total_start
        logger.info(
            "GGEUR_TIMING_CLIENT "
            f"client={int(self.ID)} stage=feature_extraction "
            f"dataset={self._cfg.data.type} extractor={self.feature_extractor_type} "
            f"total_sec={timing_total:.6f} "
            f"load_extractor_sec={timing_load_extractor:.6f} "
            f"load_cache_sec={timing_load_cache:.6f} "
            f"scan_cache_sec={timing_scan_cache:.6f} "
            f"image_load_sec={timing_image_load:.6f} "
            f"forward_sec={timing_forward:.6f} "
            f"cache_save_sec={timing_cache_save:.6f} "
            f"samples_total={samples_total} samples_cached={samples_cached} "
            f"samples_need_extract={samples_need_extract} "
            f"samples_forward={_qps_samples} output_samples={total_samples} "
            f"classes={len(self.local_features)} feature_dim={self.embedding_dim}")

    def _extract_clip_features(self):
        """Legacy method - now calls _extract_features()"""
        self._extract_features()

    def _compute_local_statistics(self):
        """Compute local mean and covariance for each class"""
        logger.info(f"Client {self.ID}: Computing local statistics...")
        timing_total_start = time.time()
        timing_mean = 0.0
        timing_center = 0.0
        timing_cov = 0.0

        self.local_means = {}
        self.local_covs = {}
        self.local_counts = {}
        sample_count = 0

        for class_idx, features in self.local_features.items():
            if features.shape[0] == 0:
                continue
            if len(features.shape) != 2:
                logger.warning(
                    f"Client {self.ID}: Skip class {class_idx} with invalid "
                    f"feature shape {features.shape}; expected 2D features.")
                continue
            if int(features.shape[1]) != int(self.embedding_dim):
                logger.warning(
                    f"Client {self.ID}: Skip class {class_idx} with feature "
                    f"dim {features.shape[1]}; expected {self.embedding_dim}.")
                continue

            n = features.shape[0]
            sample_count += int(n)
            _timing_t0 = time.time()
            mean = np.mean(features, axis=0)
            timing_mean += time.time() - _timing_t0

            # Compute covariance
            _timing_t0 = time.time()
            centered = features - mean
            timing_center += time.time() - _timing_t0
            _timing_t0 = time.time()
            if bool(getattr(self.ggeur_cfg, 'diagonal_covariance', False)):
                cov = np.mean(centered * centered, axis=0)
            else:
                cov = (1.0 / n) * np.dot(centered.T, centered)
            timing_cov += time.time() - _timing_t0

            self.local_means[class_idx] = mean
            self.local_covs[class_idx] = cov
            self.local_counts[class_idx] = n

        logger.info(f"Client {self.ID}: Computed statistics for {len(self.local_means)} classes")
        timing_total = time.time() - timing_total_start
        logger.info(
            "GGEUR_TIMING_CLIENT "
            f"client={int(self.ID)} stage=local_statistics "
            f"dataset={self._cfg.data.type} extractor={self.feature_extractor_type} "
            f"total_sec={timing_total:.6f} mean_sec={timing_mean:.6f} "
            f"center_sec={timing_center:.6f} cov_dot_sec={timing_cov:.6f} "
            f"classes={len(self.local_means)} samples={sample_count} "
            f"feature_dim={self.embedding_dim}")

    def _validate_local_statistics(self):
        """Validate statistics dimensions before sending them to the server."""
        expected_dim = int(self.embedding_dim)
        diagonal_covariance = bool(getattr(
            self.ggeur_cfg, 'diagonal_covariance', False))
        expected_cov_shape = ((expected_dim, ) if diagonal_covariance else
                              (expected_dim, expected_dim))
        invalid = []

        for class_idx, mean in self.local_means.items():
            mean_arr = np.asarray(mean)
            cov_arr = np.asarray(self.local_covs.get(class_idx))
            if mean_arr.shape != (expected_dim, ):
                invalid.append(
                    f"class {class_idx}: mean shape {mean_arr.shape}")
                continue
            if cov_arr.shape != expected_cov_shape:
                invalid.append(f"class {class_idx}: cov shape {cov_arr.shape}")

        if invalid:
            detail = '; '.join(invalid[:5])
            raise ValueError(
                f"Client {self.ID}: Invalid local statistics for "
                f"embedding_dim={expected_dim}. {detail}")

    def _upload_local_statistics(self):
        """Upload local statistics to server"""
        logger.info(f"Client {self.ID}: Uploading local statistics to server...")
        timing_total_start = time.time()

        stagger_seconds = max(
            0.0,
            float(getattr(self.ggeur_cfg,
                          'statistics_upload_stagger_seconds', 0.0)))
        if stagger_seconds > 0:
            client_num = max(1, int(self._cfg.federate.client_num))
            stagger_slot = (int(self.ID) - 1) % client_num
            stagger_delay = stagger_slot * stagger_seconds
            if stagger_delay > 0:
                logger.info(
                    f"Client {self.ID}: Stagger statistics upload by "
                    f"{stagger_delay:.1f}s (slot={stagger_slot}, "
                    f"interval={stagger_seconds:.1f}s)")
                time.sleep(stagger_delay)

        _timing_t0 = time.time()
        self._validate_local_statistics()
        timing_validate = time.time() - _timing_t0

        # Also send prototypes for cross-client augmentation
        _timing_t0 = time.time()
        prototypes = {}
        prototypes_per_class = max(
            1,
            int(getattr(
                self.ggeur_cfg, 'local_prototypes_per_class', 1)))
        for class_idx, mean in self.local_means.items():
            if prototypes_per_class == 1:
                prototypes[class_idx] = mean
                continue
            features = np.asarray(
                self.local_features.get(class_idx, []), dtype=np.float32)
            if features.ndim != 2 or features.shape[0] == 0:
                prototypes[class_idx] = mean
                continue
            keep = min(prototypes_per_class, int(features.shape[0]))
            if keep == int(features.shape[0]):
                indices = np.arange(keep, dtype=np.int64)
            else:
                indices = np.linspace(
                    0, int(features.shape[0]) - 1, keep,
                    dtype=np.int64)
            prototypes[class_idx] = [
                features[int(index)] for index in indices
            ]
        timing_prototype_prepare = time.time() - _timing_t0

        _timing_t0 = time.time()
        content = {
            'client_id': self.ID,
            'embedding_dim': int(self.embedding_dim),
            'means': self._serialize_array_payload(self.local_means),
            'covs': self._serialize_array_payload(self.local_covs),
            'counts': self.local_counts,
            'prototypes': self._serialize_array_payload(prototypes)
        }
        payload_bytes = self._sizeof_content(content)
        stats_class_count = len(self.local_means)
        timing_serialize = time.time() - _timing_t0
        logger.info(
            f"Client {self.ID}: Local statistics payload bytes={payload_bytes} "
            f"(classes={stats_class_count}, embedding_dim={self.embedding_dim})")

        _timing_t0 = time.time()
        max_attempts = max(
            1,
            int(getattr(self.ggeur_cfg,
                        'statistics_upload_max_attempts', 3)))
        retry_backoff = max(
            0.0,
            float(getattr(self.ggeur_cfg,
                          'statistics_upload_retry_backoff_seconds', 15.0)))
        for attempt in range(1, max_attempts + 1):
            try:
                self.comm_manager.send(
                    Message(
                        msg_type='local_statistics',
                        sender=self.ID,
                        receiver=[self.server_id],
                        state=self.state,
                        content=content
                    )
                )
                break
            except Exception as error:
                if attempt >= max_attempts:
                    raise
                delay = min(60.0, retry_backoff * attempt)
                logger.warning(
                    f"Client {self.ID}: Statistics upload attempt "
                    f"{attempt}/{max_attempts} failed with "
                    f"{type(error).__name__}: {error}; retrying in "
                    f"{delay:.1f}s")
                if delay > 0:
                    time.sleep(delay)
        timing_send = time.time() - _timing_t0

        self.statistics_uploaded = True
        self._release_round0_stat_buffers()
        logger.info(f"Client {self.ID}: Statistics uploaded")
        timing_total = time.time() - timing_total_start
        logger.info(
            "GGEUR_TIMING_CLIENT "
            f"client={int(self.ID)} stage=statistics_upload "
            f"dataset={self._cfg.data.type} extractor={self.feature_extractor_type} "
            f"total_sec={timing_total:.6f} validate_sec={timing_validate:.6f} "
            f"prototype_prepare_sec={timing_prototype_prepare:.6f} "
            f"prototypes_per_class={prototypes_per_class} "
            f"serialize_sec={timing_serialize:.6f} send_sec={timing_send:.6f} "
            f"payload_bytes={payload_bytes} classes={stats_class_count} "
            f"feature_dim={self.embedding_dim}")

    @staticmethod
    def _serialize_array_payload(payload):
        """Encode array/tensor leaves to compact strings for gRPC transfer."""
        if isinstance(payload, dict):
            return {
                key: GGEURClient._serialize_array_payload(value)
                for key, value in payload.items()
            }
        if isinstance(payload, list):
            return [
                GGEURClient._serialize_array_payload(value)
                for value in payload
            ]
        if isinstance(payload, tuple):
            return [
                GGEURClient._serialize_array_payload(value)
                for value in payload
            ]
        if isinstance(payload, (np.ndarray, torch.Tensor)):
            return GGEURClient._serialize_ndarray_payload(payload)
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

    @staticmethod
    def _sizeof_content(content) -> int:
        """Best-effort serialized payload size estimate for system logs."""
        if content is None:
            return 0
        if isinstance(content, str):
            return len(content.encode('utf-8'))
        if isinstance(content, bytes):
            return len(content)
        if isinstance(content, np.ndarray):
            return content.nbytes
        if isinstance(content, torch.Tensor):
            return content.numel() * content.element_size()
        if isinstance(content, dict):
            return sum(GGEURClient._sizeof_content(v)
                       for v in content.values())
        if isinstance(content, (list, tuple)):
            return sum(GGEURClient._sizeof_content(v) for v in content)
        return 0

    def callback_for_global_covariances(self, message: Message):
        """Handle receiving global covariance matrices from server"""
        logger.info(f"Client {self.ID}: Received global covariances from server")
        timing_total_start = time.time()

        content = self._decode_parameter_tree(message.content)
        message.content = content
        _timing_t0 = time.time()
        self.global_cov_matrices = self._normalize_covariance_mapping(
            content.get('cov_matrices', {}))
        timing_cov_decode = time.time() - _timing_t0
        all_prototypes_by_client = content.get('all_prototypes_by_client', {})
        _timing_t0 = time.time()
        if all_prototypes_by_client:
            self.other_prototypes = (
                self._normalize_other_prototypes_from_client_pool(
                    all_prototypes_by_client))
        else:
            other_prototypes = content.get('other_prototypes', {})
            self.other_prototypes = self._normalize_prototype_mapping(
                self._get_class_value(other_prototypes, self.ID) or {})
        timing_other_proto_decode = time.time() - _timing_t0
        _timing_t0 = time.time()
        self.global_prototypes = self._normalize_prototype_mapping(
            content.get('global_prototypes', {}))  # For feature alignment
        timing_global_proto_decode = time.time() - _timing_t0

        # Debug logging for received prototypes
        logger.info(f"Client {self.ID}: Received global_prototypes with {len(self.global_prototypes)} classes")
        if self.global_prototypes:
            sample_classes = list(self.global_prototypes.keys())[:3]
            for cls in sample_classes:
                proto = self.global_prototypes[cls]
                if hasattr(proto, 'shape'):
                    logger.info(f"  Class {cls}: shape={proto.shape}, mean={proto.mean():.4f}")

        # Save real local features before augmentation clears them (for PromptFL)
        if self.use_promptfl:
            self.real_local_features = {k: v.copy() for k, v in self.local_features.items()}

        # Perform augmentation
        self._perform_augmentation()
        self._release_round0_stat_buffers()

        # Build MLP classifier
        self._build_mlp_classifier()

        # Build PromptFL model if enabled
        if self.use_promptfl:
            self._build_prompt_model()
            self._build_prompt_loader()

        # Build CNN based on mode
        if self.use_feature_alignment:
            # Feature alignment mode: train CNN from scratch
            self._load_clip_model()
            self._build_cnn_feature_align()
            self._setup_original_image_loader()
            logger.info(f"Client {self.ID}: Feature alignment ready - CLIP: {self.clip_model is not None}, "
                       f"CNN: {self.cnn_model is not None}, Prototypes: {len(self.global_prototypes)}")
        elif self.use_cnn_distillation:
            # Knowledge distillation mode
            self._load_clip_model()
            self._build_cnn_model()
            self._setup_original_image_loader()
            logger.info(f"Client {self.ID}: CNN distillation ready - CLIP: {self.clip_model is not None}, "
                       f"MLP: {self.mlp_classifier is not None}, CNN: {self.cnn_model is not None}")

        cov_classes = len(self.global_cov_matrices)
        other_proto_classes = len(self.other_prototypes)
        global_proto_classes = len(self.global_prototypes)
        self._release_round0_broadcast_buffers_if_unused()

        # Notify server that augmentation is complete
        logger.info(f"Client {self.ID}: Notifying server that augmentation is ready")
        logger.info(
            "GGEUR_TIMING_CLIENT "
            f"client={int(self.ID)} stage=global_covariance_callback "
            f"dataset={self._cfg.data.type} extractor={self.feature_extractor_type} "
            f"total_sec={time.time() - timing_total_start:.6f} "
            f"cov_decode_sec={timing_cov_decode:.6f} "
            f"other_proto_decode_sec={timing_other_proto_decode:.6f} "
            f"global_proto_decode_sec={timing_global_proto_decode:.6f} "
            f"cov_classes={cov_classes} "
            f"other_proto_classes={other_proto_classes} "
            f"global_proto_classes={global_proto_classes}")
        self.comm_manager.send(
            Message(
                msg_type='augmentation_ready',
                sender=self.ID,
                receiver=[self.server_id],
                state=self.state,
                content='ready'
            )
        )
        self._maybe_fail_for_distributed_validation(
            'after_augmentation_ready', self.state)

    def _normalize_other_prototypes_from_client_pool(self,
                                                     all_prototypes_by_client):
        """Keep all prototypes except this client's own prototypes."""
        normalized = {}
        if not isinstance(all_prototypes_by_client, dict):
            return normalized

        for client_id, prototypes in all_prototypes_by_client.items():
            try:
                if int(client_id) == int(self.ID):
                    continue
            except (TypeError, ValueError):
                pass

            client_prototypes = self._normalize_prototype_mapping(prototypes)
            for class_idx, value in client_prototypes.items():
                if isinstance(value, list):
                    normalized.setdefault(class_idx, []).extend(value)
                else:
                    normalized.setdefault(class_idx, []).append(value)

        logger.info(
            f"Client {self.ID}: Prepared other_prototypes from shared pool "
            f"with {len(normalized)} classes")
        return normalized

    def _release_round0_stat_buffers(self):
        """Drop per-client round-0 statistics after augmented cache is built."""
        self.local_means = {}
        self.local_covs = {}
        self.local_counts = {}
        if hasattr(self, '_cov_factor_cache'):
            self._cov_factor_cache.clear()

    def _release_round0_broadcast_buffers_if_unused(self):
        """Drop broadcast payloads after HeadOnly augmentation when unused."""
        keep_global_prototypes = (
            getattr(self, 'use_fedproto', False)
            or getattr(self, 'use_feature_alignment', False)
            or getattr(self, 'use_cnn_distillation', False)
            or getattr(self, 'use_promptfl', False))
        if not keep_global_prototypes:
            self.global_prototypes = {}
        self.global_cov_matrices = {}
        self.other_prototypes = {}

    @staticmethod
    def _get_class_value(mapping, class_idx):
        """Read int-keyed payloads after serializers convert keys to strings."""
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

    def _normalize_covariance_mapping(self, cov_matrices):
        """Convert received covariance payloads to {int: np.ndarray}."""
        normalized = {}
        if not isinstance(cov_matrices, dict):
            return normalized

        expected_dim = int(self.embedding_dim)
        diagonal_covariance = bool(getattr(
            self.ggeur_cfg, 'diagonal_covariance', False))
        expected_shape = ((expected_dim, ) if diagonal_covariance else
                          (expected_dim, expected_dim))
        for class_idx, cov in cov_matrices.items():
            try:
                key = int(class_idx)
            except (TypeError, ValueError):
                logger.warning(
                    f"Client {self.ID}: Skip covariance with invalid class "
                    f"key {class_idx}")
                continue

            cov_arr = self._payload_to_ndarray(cov)
            if cov_arr is None:
                logger.warning(
                    f"Client {self.ID}: Skip covariance for class {key}; "
                    f"payload could not be decoded")
                continue
            cov_arr = np.asarray(cov_arr, dtype=np.float32)
            if cov_arr.shape != expected_shape:
                logger.warning(
                    f"Client {self.ID}: Skip covariance for class {key} "
                    f"with shape {cov_arr.shape}, expected {expected_shape}")
                continue
            normalized[key] = cov_arr

        return normalized

    def _normalize_prototype_mapping(self, prototypes):
        """Convert received prototype payloads to ndarray values keyed by int."""
        normalized = {}
        if not isinstance(prototypes, dict):
            return normalized

        expected_dim = int(self.embedding_dim)
        expected_shape = (expected_dim, )
        for class_idx, value in prototypes.items():
            try:
                key = int(class_idx)
            except (TypeError, ValueError):
                logger.warning(
                    f"Client {self.ID}: Skip prototype with invalid class "
                    f"key {class_idx}")
                continue

            if (isinstance(value, list) and value and isinstance(
                    value[0], (list, tuple, np.ndarray, str, bytes))):
                proto_list = []
                for proto in value:
                    proto_arr = self._payload_to_ndarray(proto)
                    if proto_arr is None:
                        logger.warning(
                            f"Client {self.ID}: Skip prototype for class "
                            f"{key}; payload could not be decoded")
                        continue
                    proto_arr = np.asarray(proto_arr, dtype=np.float32)
                    if proto_arr.shape == expected_shape:
                        proto_list.append(proto_arr)
                    else:
                        logger.warning(
                            f"Client {self.ID}: Skip prototype for class "
                            f"{key} with shape {proto_arr.shape}, expected "
                            f"{expected_shape}")
                normalized[key] = proto_list
            else:
                proto_arr = self._payload_to_ndarray(value)
                if proto_arr is None:
                    logger.warning(
                        f"Client {self.ID}: Skip prototype for class {key}; "
                        f"payload could not be decoded")
                    continue
                proto_arr = np.asarray(proto_arr, dtype=np.float32)
                if proto_arr.shape == expected_shape:
                    normalized[key] = proto_arr
                else:
                    logger.warning(
                        f"Client {self.ID}: Skip prototype for class {key} "
                        f"with shape {proto_arr.shape}, expected "
                        f"{expected_shape}")

        return normalized

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

    def _nearest_pos_def(self, cov_matrix):
        """Ensure covariance matrix is positive definite"""
        # 对于大维度矩阵，使用简化方法
        dim = cov_matrix.shape[0]
        if dim > 512:
            # 对于高维矩阵，直接添加正则化而不做特征值分解
            # 这样更快且通常足够
            min_eig = np.min(np.real(np.linalg.eigvalsh(cov_matrix)))
            if min_eig < 1e-6:
                cov_matrix = cov_matrix + (1e-6 - min_eig) * np.eye(dim)
            return cov_matrix

        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Scale small eigenvalues for better conditioning
        scale_factors = np.ones_like(eigenvalues)
        scaled_count = min(10, len(eigenvalues))
        scale_factors[:scaled_count] = np.linspace(5, 1, scaled_count)
        eigenvalues = eigenvalues * scale_factors

        # Clip negative eigenvalues
        eigenvalues[eigenvalues < 0] = 0

        return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    def _generate_samples(self, mean, cov_matrix, num_samples):
        """Generate samples from Gaussian distribution."""
        mean = np.asarray(mean, dtype=np.float32)
        expected_dim = int(self.embedding_dim)
        if mean.shape != (expected_dim, ):
            raise ValueError(
                f"Client {self.ID}: Cannot generate samples with mean shape "
                f"{mean.shape}, expected ({expected_dim},).")
        means = np.repeat(mean.reshape(1, -1), int(num_samples), axis=0)
        return self._generate_samples_from_means(means, cov_matrix)

    def _generate_samples_from_means(self, means, cov_matrix):
        """Generate one Gaussian sample for every row in ``means``."""
        means = np.asarray(means, dtype=np.float32)
        cov_matrix = np.asarray(cov_matrix, dtype=np.float32)
        covariance_scale = max(
            0.0,
            float(getattr(
                self.ggeur_cfg, 'generation_covariance_scale', 1.0)))
        cov_matrix = cov_matrix * covariance_scale

        expected_dim = int(self.embedding_dim)
        if means.ndim != 2 or means.shape[1] != expected_dim:
            raise ValueError(
                f"Client {self.ID}: Cannot generate samples with means shape "
                f"{means.shape}, expected (N, {expected_dim}).")
        diagonal_covariance = bool(getattr(
            self.ggeur_cfg, 'diagonal_covariance', False))
        expected_cov_shape = ((expected_dim, ) if diagonal_covariance else
                              (expected_dim, expected_dim))
        if cov_matrix.shape != expected_cov_shape:
            raise ValueError(
                f"Client {self.ID}: Cannot generate samples with covariance "
                f"shape {cov_matrix.shape}, expected "
                f"{expected_cov_shape}.")

        if diagonal_covariance:
            variance = np.maximum(cov_matrix, 1e-6)
            z = np.random.randn(*means.shape).astype(np.float32)
            return means + z * np.sqrt(variance).astype(np.float32)

        dim = cov_matrix.shape[0]
        cache_key = id(cov_matrix)
        cached_factor = self._cov_factor_cache.get(cache_key)

        if cached_factor is None:
            contiguous_cov = np.ascontiguousarray(cov_matrix)
            content_key = (
                contiguous_cov.shape,
                contiguous_cov.dtype.str,
                hashlib.blake2b(contiguous_cov.view(np.uint8),
                                digest_size=16).digest(),
            )
            cached_factor = _SHARED_COV_FACTOR_CACHE.get(content_key)

        if cached_factor is None:
            cov_matrix = self._nearest_pos_def(contiguous_cov)

            jitter = 1e-6
            factor = None
            factor_type = 'cholesky'
            while True:
                try:
                    factor = np.linalg.cholesky(
                        cov_matrix + jitter * np.eye(dim, dtype=np.float32))
                    break
                except np.linalg.LinAlgError:
                    jitter *= 10
                    if jitter > 1:
                        var = np.diag(cov_matrix)
                        var = np.maximum(var, 1e-6)
                        factor = np.sqrt(var)
                        factor_type = 'diag'
                        break

            cached_factor = (factor_type, factor.astype(np.float32))
            if len(_SHARED_COV_FACTOR_CACHE) >= \
                    _SHARED_COV_FACTOR_CACHE_MAXSIZE:
                _SHARED_COV_FACTOR_CACHE.pop(
                    next(iter(_SHARED_COV_FACTOR_CACHE)))
            _SHARED_COV_FACTOR_CACHE[content_key] = cached_factor

        self._cov_factor_cache[cache_key] = cached_factor

        factor_type, factor = cached_factor
        z = np.random.randn(means.shape[0], dim).astype(np.float32)
        if factor_type == 'diag':
            return means + z * factor
        return means + z @ factor.T

    @staticmethod
    def _normalize_task_class_counts(raw_counts):
        """Normalize task-file/config class targets to ``str -> int``."""
        if raw_counts in (None, ''):
            return {}

        items = []
        if isinstance(raw_counts, dict):
            items = list(raw_counts.items())
        elif isinstance(raw_counts, (list, tuple)):
            for item in raw_counts:
                if isinstance(item, dict):
                    key = item.get(
                        'class_id', item.get('class', item.get('name')))
                    value = item.get(
                        'target_size', item.get('count', item.get('samples')))
                    if key is None or value is None:
                        raise ValueError(
                            f"Invalid task class entry: {item!r}")
                    items.append((key, value))
                    continue
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    items.append((item[0], item[1]))
                    continue
                text = str(item).strip()
                separator = ':' if ':' in text else '=' if '=' in text else ''
                if not separator:
                    raise ValueError(
                        f"Invalid task class target {item!r}; expected "
                        "class_id:count")
                key, value = text.split(separator, 1)
                items.append((key, value))
        else:
            raise ValueError(
                "Task class targets must be a mapping or a sequence")

        normalized = {}
        for key, value in items:
            key = str(key).strip()
            if not key:
                raise ValueError("Task class target contains an empty key")
            try:
                count = int(value)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid target count for class {key!r}: {value!r}") \
                    from error
            if count < 0:
                raise ValueError(
                    f"Target count for class {key!r} must be non-negative")
            normalized[key] = count
        return normalized

    def _load_task_adaptation_spec(self):
        """Load the per-class generation task from YAML/JSON and config.

        File values are loaded first. Inline ``task_class_counts`` and
        ``task_default_target_size`` then override them, which makes the same
        task file reusable across deployments with small local adjustments.
        """
        cached = getattr(self, '_task_adaptation_cache', None)
        if cached is not None:
            return cached

        file_path = str(getattr(
            self.ggeur_cfg, 'task_adaptation_file', '') or '').strip()
        file_counts = {}
        file_default = -1
        sources = []
        resolved_path = ''
        if file_path:
            resolved_path = os.path.abspath(os.path.expanduser(
                os.path.expandvars(file_path)))
            if not os.path.isfile(resolved_path):
                raise FileNotFoundError(
                    f"Task adaptation file does not exist: {resolved_path}")
            with open(resolved_path, 'r', encoding='utf-8-sig') as stream:
                if resolved_path.lower().endswith('.json'):
                    payload = json.load(stream)
                else:
                    payload = yaml.safe_load(stream)
            if not isinstance(payload, dict):
                raise ValueError(
                    "Task adaptation file must contain a mapping")
            raw_counts = payload.get(
                'class_counts', payload.get(
                    'classes', payload.get('target_size_per_class', {})))
            file_counts = self._normalize_task_class_counts(raw_counts)
            raw_default = payload.get(
                'default_target_size', payload.get('default', -1))
            file_default = int(raw_default)
            if file_default < -1:
                raise ValueError(
                    "default_target_size must be -1 or non-negative")
            sources.append(resolved_path)

        inline_counts = self._normalize_task_class_counts(getattr(
            self.ggeur_cfg, 'task_class_counts', []))
        class_counts = dict(file_counts)
        class_counts.update(inline_counts)
        if inline_counts:
            sources.append('config:task_class_counts')

        raw_config_default = getattr(
            self.ggeur_cfg, 'task_default_target_size', '')
        config_default = (
            -1 if raw_config_default in (None, '')
            else int(raw_config_default))
        if config_default < -1:
            raise ValueError(
                "task_default_target_size must be -1 or non-negative")
        default_target = (
            config_default if config_default >= 0 else file_default)
        enabled = bool(file_path or inline_counts or config_default >= 0)
        if default_target < 0:
            default_target = int(getattr(
                self.ggeur_cfg, 'target_size_per_class', 0))

        canonical = {
            'enabled': enabled,
            'default_target_size': int(default_target),
            'class_counts': {
                key: int(value)
                for key, value in sorted(class_counts.items())
            },
        }
        signature_payload = json.dumps(
            canonical, sort_keys=True, ensure_ascii=True)
        canonical.update({
            'source': sources or ['config:target_size_per_class'],
            'file_path': resolved_path,
            'signature': hashlib.sha1(
                signature_payload.encode('utf-8')).hexdigest()[:16],
        })
        self._task_adaptation_cache = canonical
        if enabled:
            logger.info(
                f"Client {self.ID}: Loaded task-adaptive generation "
                f"profile signature={canonical['signature']} "
                f"default={canonical['default_target_size']} "
                f"explicit_classes={len(canonical['class_counts'])} "
                f"source={canonical['source']}")
        return canonical

    def _task_target_size(self, class_idx, fallback, spec=None):
        spec = spec or self._load_task_adaptation_spec()
        if not spec.get('enabled', False):
            return int(fallback)

        keys = [str(int(class_idx))]
        class_names = list(getattr(
            self.ggeur_cfg, 'prompt_class_names', []) or [])
        if 0 <= int(class_idx) < len(class_names):
            name = str(class_names[int(class_idx)]).strip()
            if name:
                keys.extend([name, name.lower()])

        counts = spec.get('class_counts', {})
        for key in keys:
            if key in counts:
                return int(counts[key])
        lower_counts = {str(key).lower(): int(value)
                        for key, value in counts.items()}
        for key in keys:
            if key.lower() in lower_counts:
                return lower_counts[key.lower()]
        return int(spec['default_target_size'])

    def _task_adaptation_is_generation_neutral(self, spec=None):
        """Return whether the task file requests the default class profile.

        The outline task profiles intentionally repeat the platform's normal
        per-class target so that reading the task file validates the adaptive
        interface without changing the accuracy experiment. Treating that
        profile as a different generation algorithm changed sampled features
        despite identical targets, so a neutral profile keeps the normal
        generation path while it is still loaded, logged and reported.
        """
        spec = spec or self._load_task_adaptation_spec()
        if not spec.get('enabled', False):
            return True
        fallback = int(getattr(
            self.ggeur_cfg, 'target_size_per_class', 0) or 0)
        if int(spec.get('default_target_size', fallback)) != fallback:
            return False
        num_classes = int(getattr(self._cfg.model, 'num_classes', 0) or 0)
        return all(
            self._task_target_size(class_idx, fallback, spec) == fallback
            for class_idx in range(num_classes)
        )

    def _sample_task_target_augmentation(self, original, prototypes,
                                         cov_matrix, target_size):
        """Return exactly the requested final class size when a mean exists."""
        original = np.asarray(original, dtype=np.float32)
        if original.size == 0:
            original = np.empty((0, int(self.embedding_dim)),
                                dtype=np.float32)
        elif original.ndim == 1:
            original = original.reshape(1, -1)
        prototypes = np.asarray(prototypes, dtype=np.float32)
        if prototypes.size == 0:
            prototypes = np.empty((0, int(self.embedding_dim)),
                                  dtype=np.float32)
        elif prototypes.ndim == 1:
            prototypes = prototypes.reshape(1, -1)

        target_size = int(target_size)
        if target_size < 0:
            raise ValueError("Task target size must be non-negative")
        if target_size == 0:
            return {
                'features': np.empty((0, int(self.embedding_dim)),
                                     dtype=np.float32),
                'sample_generated': 0,
                'prototype_generated': 0,
                'exact': True,
            }

        original_count = int(original.shape[0])
        if original_count >= target_size:
            indices = np.random.choice(
                original_count, target_size, replace=False)
            return {
                'features': original[indices],
                'sample_generated': 0,
                'prototype_generated': 0,
                'exact': True,
            }

        source_parts = []
        if original_count:
            source_parts.append(original)
        if int(prototypes.shape[0]):
            source_parts.append(prototypes)
        if not source_parts:
            return {
                'features': original,
                'sample_generated': 0,
                'prototype_generated': 0,
                'exact': False,
            }

        source_means = np.vstack(source_parts)
        required = target_size - original_count
        source_indices = np.random.choice(
            source_means.shape[0], required, replace=True)
        selected_means = source_means[source_indices]
        generated = self._generate_samples_from_means(
            selected_means, cov_matrix)
        selected_parts = [generated]
        if original_count:
            selected_parts.insert(0, original)
        selected = np.vstack(selected_parts)
        np.random.shuffle(selected)
        from_samples = int(np.sum(source_indices < original_count))
        return {
            'features': selected,
            'sample_generated': from_samples,
            'prototype_generated': int(required - from_samples),
            'exact': int(selected.shape[0]) == target_size,
        }

    def _emit_training_distribution(self, cache_hit=False):
        """Log and persist the exact data distribution used for training."""
        spec = self._load_task_adaptation_spec()
        configured_dir = str(getattr(
            self.ggeur_cfg, 'training_distribution_dir', '') or '').strip()
        if not spec.get('enabled', False) and not configured_dir:
            return None, None

        training_labels = (
            np.asarray(self.augmented_labels, dtype=np.int64)
            if self.augmented_labels is not None else
            np.asarray([], dtype=np.int64))
        training_unique, training_counts_values = np.unique(
            training_labels, return_counts=True)
        training_class_counts = {
            str(int(class_idx)): int(count)
            for class_idx, count in zip(
                training_unique, training_counts_values)
        }
        generated_snapshot = getattr(
            self, '_generated_distribution_snapshot', None) or {}
        class_counts = dict(generated_snapshot.get(
            'class_counts', training_class_counts))
        generated_total = int(generated_snapshot.get(
            'total_samples', training_labels.shape[0]))
        targets = {}
        if spec.get('enabled', False):
            num_classes = int(getattr(self._cfg.model, 'num_classes', 0))
            targets = {
                str(class_idx): self._task_target_size(
                    class_idx,
                    getattr(self.ggeur_cfg, 'target_size_per_class', 0),
                    spec)
                for class_idx in range(num_classes)
            }

        report = {
            'schema_version': 2,
            'method': 'Platform',
            'client_id': int(self.ID),
            'dataset': str(getattr(self._cfg.data, 'type', '')),
            'cache_hit': bool(cache_hit),
            'task_adaptation': {
                'enabled': bool(spec.get('enabled', False)),
                'source': list(spec.get('source', [])),
                'signature': str(spec.get('signature', '')),
                'default_target_size': int(spec.get(
                    'default_target_size', 0)),
                'class_targets': targets,
            },
            'total_samples': generated_total,
            'class_counts': class_counts,
            'training_total_samples': int(training_labels.shape[0]),
            'training_class_counts': training_class_counts,
        }

        output_dir = configured_dir or os.path.join(
            str(getattr(self._cfg, 'outdir', '') or os.getcwd()),
            'training_distributions')
        output_dir = os.path.abspath(os.path.expanduser(
            os.path.expandvars(output_dir)))
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(
            output_dir, f"client_{int(self.ID):06d}.json")
        temp_path = f"{output_path}.tmp.{os.getpid()}"
        with open(temp_path, 'w', encoding='utf-8') as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2,
                      sort_keys=True)
        os.replace(temp_path, output_path)
        logger.info(
            "PLATFORM_TRAIN_DISTRIBUTION " +
            json.dumps({
                'client_id': int(self.ID),
                'total_samples': report['total_samples'],
                'class_counts': class_counts,
                'task_signature': spec.get('signature', ''),
                'output': output_path,
            }, sort_keys=True, ensure_ascii=True))
        return report, output_path

    def _capture_generated_distribution_snapshot(self):
        """Remember generated rows before optional runtime subsampling."""
        labels = (
            np.asarray(self.augmented_labels, dtype=np.int64)
            if self.augmented_labels is not None else
            np.asarray([], dtype=np.int64))
        unique, counts = np.unique(labels, return_counts=True)
        self._generated_distribution_snapshot = {
            'total_samples': int(labels.shape[0]),
            'class_counts': {
                str(int(class_idx)): int(count)
                for class_idx, count in zip(unique, counts)
            },
        }
        return self._generated_distribution_snapshot

    def _sample_target_sized_augmentation(self,
                                          original,
                                          prototypes,
                                          cov_matrix,
                                          num_per_sample,
                                          num_per_prototype,
                                          target_size):
        """Sample a bounded result from the conceptual augmentation pool.

        The legacy path materialized every generated candidate before uniformly
        selecting ``target_size`` rows.  For large clients that can create
        millions of 768-dimensional temporary rows even though only a few
        hundred survive.  Selecting conceptual candidate indices first and
        generating only selected Gaussian rows has the same sampling
        distribution without the transient memory and compute explosion.
        """
        original = np.asarray(original, dtype=np.float32)
        if original.size == 0:
            original = np.empty((0, int(self.embedding_dim)),
                                dtype=np.float32)
        elif original.ndim == 1:
            original = original.reshape(1, -1)
        prototypes = np.asarray(prototypes, dtype=np.float32)
        if prototypes.size == 0:
            prototypes = np.empty((0, int(self.embedding_dim)),
                                  dtype=np.float32)
        elif prototypes.ndim == 1:
            prototypes = prototypes.reshape(1, -1)

        original_count = int(original.shape[0])
        prototype_count = int(prototypes.shape[0])
        sample_candidates = original_count * max(0, int(num_per_sample))
        prototype_candidates = (
            prototype_count * max(0, int(num_per_prototype)))
        total_candidates = (
            original_count + sample_candidates + prototype_candidates)
        if (int(target_size) <= 0 or
                total_candidates < int(target_size)):
            return None

        selected_indices = np.random.choice(
            total_candidates, int(target_size), replace=False)
        selected_original = selected_indices[
            selected_indices < original_count]
        generated_means = []

        sample_start = original_count
        sample_stop = sample_start + sample_candidates
        selected_sample_slots = selected_indices[
            (selected_indices >= sample_start) &
            (selected_indices < sample_stop)]
        if selected_sample_slots.size:
            source_indices = (
                (selected_sample_slots - sample_start) //
                int(num_per_sample))
            generated_means.append(original[source_indices])

        selected_prototype_slots = selected_indices[
            selected_indices >= sample_stop]
        if selected_prototype_slots.size:
            source_indices = (
                (selected_prototype_slots - sample_stop) //
                int(num_per_prototype))
            generated_means.append(prototypes[source_indices])

        selected_parts = []
        if selected_original.size:
            selected_parts.append(original[selected_original])
        if generated_means:
            selected_parts.append(
                self._generate_samples_from_means(
                    np.vstack(generated_means), cov_matrix))
        selected = np.vstack(selected_parts)
        np.random.shuffle(selected)
        return {
            'features': selected,
            'sample_generated': int(selected_sample_slots.size),
            'prototype_generated': int(selected_prototype_slots.size),
            'conceptual_candidates': int(total_candidates),
        }

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
            self._emit_training_distribution(cache_hit=True)
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
        task_spec = self._load_task_adaptation_spec()
        task_adaptation_enabled = bool(task_spec.get('enabled', False))
        task_adaptation_changes_generation = (
            task_adaptation_enabled and
            not self._task_adaptation_is_generation_neutral(task_spec))

        # Check if augmentation is disabled (baseline mode)
        no_augmentation = (
            ((num_per_sample == 0 and num_per_prototype == 0) or
             not use_cross_client) and
            not task_adaptation_changes_generation)

        if no_augmentation:
            logger.info(f"Client {self.ID}: No augmentation mode - using original features only")
        else:
            logger.info(f"Client {self.ID}: Performing GGEUR_Clip augmentation...")

        all_features = []
        all_labels = []

        # Get all class indices
        all_classes = set(self.local_features.keys())
        if not no_augmentation and self.global_cov_matrices:
            all_classes.update(self.global_cov_matrices.keys())
        if not no_augmentation and self.other_prototypes:
            all_classes.update(self.other_prototypes.keys())
        if task_adaptation_changes_generation:
            for task_key in task_spec.get('class_counts', {}):
                try:
                    all_classes.add(int(task_key))
                except (TypeError, ValueError):
                    continue

        total_classes = len(all_classes)
        timing_init += time.time() - _timing_t0

        # 获取特征维度用于日志
        feature_dim = self.embedding_dim
        if self.local_features:
            first_key = next(iter(self.local_features.keys()))
            if len(self.local_features[first_key]) > 0:
                feature_dim = self.local_features[first_key].shape[1]

        logger.info(f"Client {self.ID}: Processing {total_classes} classes, feature_dim={feature_dim}, "
                   f"num_per_sample={num_per_sample}, num_per_prototype={num_per_prototype}")

        for idx, class_idx in enumerate(all_classes):
            class_idx = int(class_idx)
            class_features = []

            # 显示进度
            if (idx + 1) % 10 == 0 or idx == 0:
                logger.info(f"Client {self.ID}: Augmenting class {idx+1}/{total_classes}")

            # 1. Original features from this client (always include)
            if class_idx in self.local_features:
                _timing_t0 = time.time()
                original = self.local_features[class_idx]
                class_features.append(original)
                original_samples += int(original.shape[0])
                timing_original_collect += time.time() - _timing_t0

            # Skip augmentation if disabled
            if no_augmentation:
                if class_features:
                    _timing_t0 = time.time()
                    combined = np.vstack(class_features)
                    all_features.append(combined)
                    all_labels.append(np.full(combined.shape[0], class_idx))
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
            original_for_generation = self.local_features.get(
                class_idx,
                np.empty((0, int(self.embedding_dim)), dtype=np.float32))
            prototypes_for_generation = []
            if (use_cross_client and self.other_prototypes and
                    class_idx in self.other_prototypes):
                prototypes_for_generation = self.other_prototypes[class_idx]

            if task_adaptation_changes_generation:
                class_target = self._task_target_size(
                    class_idx, target_size, task_spec)
                _timing_t0 = time.time()
                task_result = self._sample_task_target_augmentation(
                    original_for_generation,
                    prototypes_for_generation,
                    cov_matrix,
                    class_target,
                )
                timing_sample_generation += time.time() - _timing_t0
                selected = task_result['features']
                generated_sample_count += task_result['sample_generated']
                generated_prototype_count += \
                    task_result['prototype_generated']
                sample_generation_calls += int(
                    task_result['sample_generated'] > 0)
                prototype_generation_calls += int(
                    task_result['prototype_generated'] > 0)
                if selected.shape[0] > 0:
                    all_features.append(selected)
                    all_labels.append(
                        np.full(selected.shape[0], class_idx))
                    selected_samples += int(selected.shape[0])
                if not task_result['exact']:
                    logger.warning(
                        f"Client {self.ID}: Task target for class "
                        f"{class_idx} could not be met because no source "
                        f"mean was available (target={class_target}, "
                        f"actual={selected.shape[0]})")
                logger.info(
                    f"Client {self.ID}: Task-adaptive class {class_idx} "
                    f"target={class_target} selected={selected.shape[0]} "
                    f"generated_from_samples="
                    f"{task_result['sample_generated']} "
                    f"generated_from_prototypes="
                    f"{task_result['prototype_generated']}")
                continue

            _timing_t0 = time.time()
            bounded = self._sample_target_sized_augmentation(
                original_for_generation,
                prototypes_for_generation,
                cov_matrix,
                num_per_sample,
                num_per_prototype,
                target_size,
            )
            if bounded is not None:
                selected = bounded['features']
                generated_sample_count += bounded['sample_generated']
                generated_prototype_count += \
                    bounded['prototype_generated']
                sample_generation_calls += int(
                    bounded['sample_generated'] > 0)
                prototype_generation_calls += int(
                    bounded['prototype_generated'] > 0)
                timing_sample_generation += time.time() - _timing_t0
                all_features.append(selected)
                all_labels.append(
                    np.full(selected.shape[0], class_idx))
                selected_samples += int(selected.shape[0])
                logger.info(
                    f"Client {self.ID}: Bounded augmentation class "
                    f"{class_idx} selected={selected.shape[0]} from "
                    f"conceptual_candidates="
                    f"{bounded['conceptual_candidates']}")
                continue
            timing_sample_generation += time.time() - _timing_t0

            if num_per_sample > 0 and class_idx in self.local_features and self.local_features[class_idx].shape[0] > 0:
                for feat in self.local_features[class_idx]:
                    _timing_t0 = time.time()
                    generated = self._generate_samples(feat, cov_matrix, num_per_sample)
                    timing_sample_generation += time.time() - _timing_t0
                    sample_generation_calls += 1
                    generated_sample_count += int(generated.shape[0])
                    class_features.append(generated)

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

            # Combine and sample to target size
            if class_features:
                _timing_t0 = time.time()
                combined = np.vstack(class_features)

                # target_size = 0 means use all samples
                if target_size > 0 and combined.shape[0] >= target_size:
                    indices = np.random.choice(combined.shape[0], target_size, replace=False)
                    selected = combined[indices]
                else:
                    selected = combined

                all_features.append(selected)
                all_labels.append(np.full(selected.shape[0], class_idx))
                selected_samples += int(selected.shape[0])
                timing_stack_select += time.time() - _timing_t0

        logger.info(f"Client {self.ID}: Augmentation complete, building dataset...")

        if all_features:
            _timing_t0 = time.time()
            self.augmented_features = np.vstack(all_features)
            self.augmented_labels = np.concatenate(all_labels)
            self._capture_generated_distribution_snapshot()
            self._apply_baseline_client_sampling()

            # Create data loader
            dataset = AugmentedFeatureDataset(self.augmented_features, self.augmented_labels)
            self.augmented_loader = DataLoader(
                dataset,
                batch_size=self._cfg.dataloader.batch_size,
                shuffle=True,
                generator=self._training_loader_generator(),
            )
            timing_dataset_build += time.time() - _timing_t0

            if no_augmentation:
                logger.info(f"Client {self.ID}: Original data - {self.augmented_features.shape[0]} samples, "
                            f"{len(np.unique(self.augmented_labels))} classes")
            else:
                logger.info(f"Client {self.ID}: Augmented data - {self.augmented_features.shape[0]} samples, "
                            f"{len(np.unique(self.augmented_labels))} classes")
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
            if self._apply_platform_training_sampling():
                dataset = AugmentedFeatureDataset(
                    self.augmented_features, self.augmented_labels)
                self.augmented_loader = DataLoader(
                    dataset,
                    batch_size=self._cfg.dataloader.batch_size,
                    shuffle=True,
                    generator=self._training_loader_generator(),
                )

        self._emit_training_distribution(cache_hit=False)

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
            f"target_size_per_class={target_size} "
            f"task_adaptation={1 if task_adaptation_enabled else 0} "
            f"task_signature={task_spec.get('signature', '')}")

        # Free raw features from memory - augmented_features/loader are all we need now
        self.local_features = {}
        self.local_labels = {}

    def _build_mlp_classifier(self):
        """Build the trainable classifier for augmented features."""
        # IMPORTANT: Always use config's num_classes, not the unique labels in augmented data
        # In LDS mode, each client may only have a subset of classes, but the model
        # must support all classes for proper FedAvg aggregation
        num_classes = self._cfg.model.num_classes
        input_dim = self.embedding_dim
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
                self.mlp_classifier = GGEURTextRNNClassifier(
                    input_dim=input_dim,
                    hidden_dim=int(getattr(self._cfg.model, 'hidden', 256)),
                    num_classes=num_classes,
                    num_layers=int(getattr(self._cfg.model, 'layer', 1)),
                    dropout=float(getattr(self._cfg.model, 'dropout', 0.0)),
                    rnn_type=rnn_type)
            elif hidden_dim > 0:
                self.mlp_classifier = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(self.ggeur_cfg.mlp_dropout),
                    nn.Linear(hidden_dim, num_classes)
                )
            else:
                # Simple linear classifier
                self.mlp_classifier = nn.Linear(input_dim, num_classes)
        finally:
            if cpu_rng_state is not None:
                torch.set_rng_state(cpu_rng_state)

        self.mlp_classifier = self.mlp_classifier.to(self.device)
        logger.info(
            f"Client {self.ID}: Built {model_type or 'mlp'} classifier "
            f"with {num_classes} classes "
            f"(classifier_init_seed={init_seed})")

    def _training_loader_generator(self):
        """Return a repeatable per-client loader RNG when configured."""
        base_seed = int(getattr(
            self.ggeur_cfg, 'training_data_seed', -1))
        if base_seed < 0:
            return None
        generator = torch.Generator()
        generator.manual_seed(base_seed + int(self.ID))
        return generator

    def _augmented_cache_metadata(self):
        splits = getattr(self._cfg.data, 'splits', [])
        try:
            splits = list(splits)
        except Exception:
            splits = []
        version = str(getattr(
            self.ggeur_cfg, 'augmented_feature_cache_version',
            getattr(self.ggeur_cfg, 'headonly_cache_version',
                    'aug_fcache_v1')))
        task_spec = self._load_task_adaptation_spec()
        task_metadata = {
            'enabled': bool(task_spec.get('enabled', False)),
            'default_target_size': int(task_spec.get(
                'default_target_size', 0)),
            'class_counts': dict(task_spec.get('class_counts', {})),
            'signature': str(task_spec.get('signature', '')),
        }
        if self._task_adaptation_is_generation_neutral(task_spec):
            task_metadata = self._disabled_task_cache_metadata(
                int(self.ggeur_cfg.target_size_per_class))
        return {
            'source': 'real_dataset',
            'mode': 'ggeur_augmented_features',
            'client_id': int(self.ID),
            'client_num': int(self._cfg.federate.client_num),
            'dataset': str(self._cfg.data.type),
            'data_root': str(self._cfg.data.root),
            'splits': splits,
            'seed': int(getattr(self._cfg, 'seed', 0)),
            'feature_extractor': str(self.feature_extractor_type),
            'feature_extractor_model': self._feature_extractor_cache_name(),
            'embedding_dim': int(self.embedding_dim),
            'num_classes': int(self._cfg.model.num_classes),
            'num_generated_per_sample':
                int(self.ggeur_cfg.num_generated_per_sample),
            'num_generated_per_prototype':
                int(self.ggeur_cfg.num_generated_per_prototype),
            'target_size_per_class':
                int(self.ggeur_cfg.target_size_per_class),
            'baseline_target_samples_per_client': int(getattr(
                self.ggeur_cfg,
                'baseline_target_samples_per_client', 0)),
            'task_adaptation': task_metadata,
            'use_cross_client_prototypes':
                bool(getattr(self.ggeur_cfg, 'use_cross_client_prototypes',
                             True)),
            'max_cross_client_prototypes_per_class':
                int(getattr(
                    self.ggeur_cfg,
                    'max_cross_client_prototypes_per_class', 0)),
            'local_prototypes_per_class': int(getattr(
                self.ggeur_cfg, 'local_prototypes_per_class', 1)),
            'generation_covariance_scale': float(getattr(
                self.ggeur_cfg, 'generation_covariance_scale', 1.0)),
            'platform_target_samples_per_client': int(getattr(
                self.ggeur_cfg, 'platform_target_samples_per_client', 0)),
            'platform_auto_target_samples_per_client': int(getattr(
                self.ggeur_cfg,
                'platform_auto_target_samples_per_client', 0)),
            'platform_class_balanced_sampling': bool(getattr(
                self.ggeur_cfg, 'platform_class_balanced_sampling', False)),
            'prototype_classifier_init': bool(getattr(
                self.ggeur_cfg, 'prototype_classifier_init', False)),
            'cross_client_prototype_seed':
                int(getattr(self.ggeur_cfg,
                            'cross_client_prototype_seed', 42)),
            'diagonal_covariance':
                bool(getattr(self.ggeur_cfg, 'diagonal_covariance', False)),
            'use_fedproto':
                bool(getattr(self.ggeur_cfg, 'use_fedproto', False)),
            'use_lds':
                bool(getattr(self.ggeur_cfg, 'use_lds', False)),
            'lds_alpha':
                float(getattr(self.ggeur_cfg, 'lds_alpha', 0.1)),
            'lds_seed':
                int(getattr(self.ggeur_cfg, 'lds_seed', 42)),
            'domainnet_domains':
                self._jsonable_config_value(
                    getattr(self.ggeur_cfg, 'domainnet_domains', [])),
            'domainnet_shared_classes_only':
                bool(getattr(self.ggeur_cfg,
                             'domainnet_shared_classes_only', False)),
            'officehome_domains':
                self._jsonable_config_value(
                    getattr(self.ggeur_cfg, 'officehome_domains', [])),
            'officehome_split_strategy':
                str(getattr(self.ggeur_cfg, 'officehome_split_strategy',
                            'standard')),
            'officehome_random_clients_per_domain':
                int(getattr(
                    self.ggeur_cfg,
                    'officehome_random_clients_per_domain', 0)),
            'officehome_random_samples_per_client':
                int(getattr(
                    self.ggeur_cfg,
                    'officehome_random_samples_per_client', 0)),
            'officehome_random_sample_with_replacement':
                bool(getattr(
                    self.ggeur_cfg,
                    'officehome_random_sample_with_replacement', False)),
            'officehome_manifest_path':
                str(getattr(self.ggeur_cfg, 'officehome_manifest_path', '')),
            'augmented_feature_cache_version': version,
            'headonly_cache_version':
                str(getattr(self.ggeur_cfg, 'headonly_cache_version',
                            'fcache_v1')),
        }

    @staticmethod
    def _jsonable_config_value(value):
        if isinstance(value, (list, tuple)):
            return [GGEURClient._jsonable_config_value(item)
                    for item in value]
        if isinstance(value, dict):
            return {
                str(key): GGEURClient._jsonable_config_value(val)
                for key, val in value.items()
            }
        try:
            if isinstance(value, np.generic):
                return value.item()
        except Exception:
            pass
        return value

    @staticmethod
    def _safe_cache_token(value):
        token = str(value).replace('\\', '_').replace('/', '_')
        token = token.replace(':', '_').replace(' ', '_')
        return ''.join(ch if ch.isalnum() or ch in '._-' else '_'
                       for ch in token)

    def _feature_extractor_cache_name(self):
        if self.feature_extractor_type == 'cnn':
            return str(getattr(self.ggeur_cfg, 'cnn_backbone',
                               'convnext_base'))
        if self.feature_extractor_type == 'timm':
            return str(getattr(self.ggeur_cfg, 'timm_model',
                               'mixer_b16_224'))
        if self.feature_extractor_type == 'bert':
            model_name = os.path.basename(
                str(getattr(self.ggeur_cfg, 'bert_model_path',
                            'bert')).rstrip('/\\')) or 'bert'
            mode = 'pre' if getattr(
                self.ggeur_cfg, 'bert_use_pretrained_weights',
                True) else f"rand_seed{int(getattr(self._cfg, 'seed', 0))}"
            pooling = str(getattr(self.ggeur_cfg, 'bert_pooling', 'cls'))
            max_len = int(getattr(self.ggeur_cfg, 'bert_max_length', 128))
            return f'{model_name}_maxlen{max_len}_{pooling}_{mode}'
        clip_model = str(getattr(self.ggeur_cfg, 'clip_model', 'ViT-B-16'))
        pretrained = str(getattr(self.ggeur_cfg, 'clip_pretrained',
                                 'openai'))
        return f'{clip_model}_{pretrained}'

    def _augmented_cache_fingerprint(self, metadata):
        keys = [
            'client_num',
            'dataset',
            'splits',
            'seed',
            'feature_extractor',
            'feature_extractor_model',
            'embedding_dim',
            'num_classes',
            'num_generated_per_sample',
            'num_generated_per_prototype',
            'target_size_per_class',
            'baseline_target_samples_per_client',
            'task_adaptation',
            'use_cross_client_prototypes',
            'max_cross_client_prototypes_per_class',
            'local_prototypes_per_class',
            'generation_covariance_scale',
            'platform_target_samples_per_client',
            'platform_auto_target_samples_per_client',
            'platform_class_balanced_sampling',
            'prototype_classifier_init',
            'cross_client_prototype_seed',
            'diagonal_covariance',
            'use_fedproto',
            'use_lds',
            'lds_alpha',
            'lds_seed',
            'domainnet_domains',
            'domainnet_shared_classes_only',
            'officehome_domains',
            'officehome_split_strategy',
            'officehome_random_clients_per_domain',
            'officehome_random_samples_per_client',
            'officehome_random_sample_with_replacement',
            'officehome_manifest_path',
            'augmented_feature_cache_version',
        ]
        payload = {key: metadata.get(key) for key in keys}
        task_metadata = payload.get('task_adaptation')
        if self._task_cache_metadata_is_generation_neutral(
                task_metadata,
                int(metadata.get('num_classes', 0) or 0),
                int(metadata.get('target_size_per_class', 0) or 0)):
            payload['task_adaptation'] = \
                self._disabled_task_cache_metadata(
                    int(metadata.get('target_size_per_class', 0) or 0))
        text = json.dumps(payload, sort_keys=True, ensure_ascii=True,
                          default=str)
        return hashlib.sha1(text.encode('utf-8')).hexdigest()[:16]

    @staticmethod
    def _disabled_task_cache_metadata(target_size):
        canonical = {
            'enabled': False,
            'default_target_size': int(target_size),
            'class_counts': {},
        }
        signature_payload = json.dumps(
            canonical, sort_keys=True, ensure_ascii=True)
        return {
            **canonical,
            'signature': hashlib.sha1(
                signature_payload.encode('utf-8')).hexdigest()[:16],
        }

    @staticmethod
    def _task_cache_metadata_is_generation_neutral(
            task_metadata, num_classes, target_size):
        """Whether cache task metadata describes the default generation."""
        if not isinstance(task_metadata, dict):
            return True
        if not bool(task_metadata.get('enabled', False)):
            return True
        if int(task_metadata.get('default_target_size', -1)) != \
                int(target_size):
            return False
        counts = task_metadata.get('class_counts', {})
        if not isinstance(counts, dict):
            return False
        for class_idx in range(int(num_classes)):
            if int(counts.get(str(class_idx), target_size)) != \
                    int(target_size):
                return False
        return all(int(value) == int(target_size)
                   for value in counts.values())

    def _get_augmented_feature_cache_base_dir(self):
        cache_dir = getattr(self.ggeur_cfg,
                            'augmented_feature_cache_dir', '')
        if not cache_dir:
            cache_dir = getattr(self.ggeur_cfg, 'feature_cache_dir', '')
        if not cache_dir:
            cache_dir = os.path.join(os.path.dirname(self._cfg.data.root),
                                     'clip_feature_cache')
        return cache_dir

    def _get_augmented_feature_cache_path(self):
        if not (getattr(self.ggeur_cfg, 'reuse_augmented_feature_cache',
                        True) or
                getattr(self.ggeur_cfg, 'save_augmented_feature_cache',
                        True)):
            return None
        cache_dir = self._get_augmented_feature_cache_base_dir()
        metadata = self._augmented_cache_metadata()
        version = self._safe_cache_token(
            metadata.get('augmented_feature_cache_version',
                         metadata.get('headonly_cache_version',
                                      'aug_fcache_v1')))
        dataset = self._safe_cache_token(metadata['dataset'])
        extractor = self._safe_cache_token(
            f"{metadata['feature_extractor']}_"
            f"{metadata['feature_extractor_model']}")
        fingerprint = self._augmented_cache_fingerprint(metadata)
        namespace = (
            f"{dataset}_{extractor}_{metadata['client_num']}c_"
            f"gps{metadata['num_generated_per_sample']}_"
            f"gpp{metadata['num_generated_per_prototype']}_"
            f"target{metadata['target_size_per_class']}_{fingerprint}")
        subdir = (
            'headonly_augmented'
            if getattr(self.ggeur_cfg, 'head_only_mode', False)
            else 'augmented_features')
        path = os.path.join(
            cache_dir,
            subdir,
            version,
            namespace,
            f'{dataset}_client_{int(self.ID):06d}.pt',
        )
        # The fully descriptive namespace can exceed the legacy Win32
        # MAX_PATH limit when a distributed run also has a descriptive run
        # ID. Keep the metadata/fingerprint unchanged, but use a compact,
        # deterministic directory name on Windows when needed. A preflight
        # and its formal run therefore still resolve to the same cache.
        if os.name == 'nt' and len(os.path.abspath(path)) >= 240:
            compact_namespace = (
                f'{dataset[:20]}_{extractor[:20]}_'
                f'{metadata["client_num"]}c_'
                f'g{metadata["num_generated_per_sample"]}_'
                f'p{metadata["num_generated_per_prototype"]}_'
                f't{metadata["target_size_per_class"]}_{fingerprint}'
            )
            path = os.path.join(
                cache_dir,
                subdir,
                version,
                compact_namespace,
                f'c{int(self.ID):06d}.pt',
            )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def _get_augmented_feature_cache_candidates(self):
        path = self._get_augmented_feature_cache_path()
        if path is None:
            return []

        paths = [path]

        # Cache fingerprints evolve when new compatibility metadata is added.
        # Search sibling namespaces with the same descriptive configuration
        # prefix and rely on the metadata validator below before loading.  This
        # keeps previously generated, compatible caches reusable without
        # copying or renaming large feature files.
        namespace_dir = Path(path).parent
        namespace_parts = namespace_dir.name.rsplit('_', 1)
        if len(namespace_parts) == 2:
            namespace_prefix = f'{namespace_parts[0]}_'
            for candidate_dir in sorted(
                    namespace_dir.parent.glob(f'{namespace_prefix}*')):
                candidate_path = str(candidate_dir / Path(path).name)
                if candidate_path not in paths:
                    paths.append(candidate_path)

        cache_dir = self._get_augmented_feature_cache_base_dir()
        if cache_dir:
            legacy_ids = [int(self.ID) - 1, int(self.ID)]
            for legacy_id in legacy_ids:
                if legacy_id < 0:
                    continue
                legacy_path = os.path.join(
                    cache_dir, f'client_{legacy_id:06d}.pt')
                if legacy_path not in paths:
                    paths.append(legacy_path)
        return paths

    def _is_augmented_cache_metadata_compatible(self, metadata):
        """Check cache metadata while allowing moved legacy cache dirs."""
        if not isinstance(metadata, dict):
            return False

        expected = self._augmented_cache_metadata()
        if metadata == expected:
            return True

        expected_version = (
            expected.get('augmented_feature_cache_version') or
            expected.get('headonly_cache_version'))
        cached_version = (
            metadata.get('augmented_feature_cache_version') or
            metadata.get('headonly_cache_version') or
            metadata.get('feature_cache_version')
        )
        checks = [
            (str(metadata.get('dataset', '')) == expected['dataset']),
            (str(metadata.get('feature_extractor', '')) ==
             expected['feature_extractor']),
            (int(metadata.get('embedding_dim', -1)) ==
             expected['embedding_dim']),
            (int(metadata.get('num_classes', -1)) == expected['num_classes']),
            (int(metadata.get('num_generated_per_sample', -1)) ==
             expected['num_generated_per_sample']),
            (int(metadata.get('num_generated_per_prototype', -1)) ==
             expected['num_generated_per_prototype']),
            (int(metadata.get('target_size_per_class', -1)) ==
             expected['target_size_per_class']),
            (str(cached_version) == str(expected_version)),
        ]
        if metadata.get('augmented_feature_cache_version') is not None:
            checks[-1] = (
                str(metadata.get('augmented_feature_cache_version')) ==
                str(expected.get('augmented_feature_cache_version')))
        cached_baseline_target = int(metadata.get(
            'baseline_target_samples_per_client', 0) or 0)
        expected_baseline_target = int(expected.get(
            'baseline_target_samples_per_client', 0) or 0)
        checks.append(
            cached_baseline_target == expected_baseline_target or
            (expected_baseline_target > 0 and cached_baseline_target == 0))
        if metadata.get('feature_extractor_model') is not None:
            checks.append(
                str(metadata.get('feature_extractor_model')) ==
                str(expected['feature_extractor_model']))
        if metadata.get('use_fedproto') is not None:
            checks.append(
                bool(metadata.get('use_fedproto')) ==
                bool(expected['use_fedproto']))
        if metadata.get('use_lds') is not None:
            checks.append(
                bool(metadata.get('use_lds')) == bool(expected['use_lds']))
        if metadata.get('lds_alpha') is not None:
            checks.append(
                float(metadata.get('lds_alpha')) ==
                float(expected['lds_alpha']))
        if metadata.get('lds_seed') is not None:
            checks.append(
                int(metadata.get('lds_seed')) == int(expected['lds_seed']))
        if metadata.get('client_num') is not None:
            checks.append(
                int(metadata.get('client_num')) == expected['client_num'])
        if metadata.get('seed') is not None:
            checks.append(int(metadata.get('seed')) == expected['seed'])
        if metadata.get('splits') is not None:
            checks.append(list(metadata.get('splits')) == expected['splits'])
        if metadata.get('diagonal_covariance') is not None:
            checks.append(
                bool(metadata.get('diagonal_covariance')) ==
                bool(expected['diagonal_covariance']))
        for key in (
                'local_prototypes_per_class',
                'platform_target_samples_per_client',
                'platform_auto_target_samples_per_client'):
            if metadata.get(key) is not None:
                checks.append(int(metadata.get(key)) == int(expected[key]))
        if metadata.get('generation_covariance_scale') is not None:
            checks.append(float(metadata.get(
                'generation_covariance_scale')) == float(expected[
                    'generation_covariance_scale']))
        for key in (
                'use_cross_client_prototypes',
                'platform_class_balanced_sampling',
                'prototype_classifier_init'):
            if metadata.get(key) is not None:
                checks.append(bool(metadata.get(key)) == bool(expected[key]))
        for key in (
                'max_cross_client_prototypes_per_class',
                'cross_client_prototype_seed'):
            if metadata.get(key) is not None:
                checks.append(int(metadata.get(key)) == int(expected[key]))
        expected_task = expected.get('task_adaptation', {})
        cached_task = metadata.get('task_adaptation')
        expected_task_is_neutral = \
            self._task_cache_metadata_is_generation_neutral(
                expected_task, expected['num_classes'],
                expected['target_size_per_class'])
        cached_task_is_neutral = \
            self._task_cache_metadata_is_generation_neutral(
                cached_task, expected['num_classes'],
                expected['target_size_per_class'])
        if expected_task.get('enabled', False) and \
                not expected_task_is_neutral:
            checks.append(cached_task == expected_task)
        elif expected_task_is_neutral:
            checks.append(
                cached_task_is_neutral and
                not bool((cached_task or {}).get('enabled', False)))
        elif cached_task is not None:
            checks.append(not bool(cached_task.get('enabled', False)))

        cached_client_id = metadata.get('client_id')
        if cached_client_id is not None:
            valid_ids = {int(self.ID), int(self.ID) - 1}
            checks.append(int(cached_client_id) in valid_ids)

        return all(checks)

    @staticmethod
    def _copy_numpy_mapping(mapping, dtype=np.float32):
        if not isinstance(mapping, dict):
            return {}
        copied = {}
        for key, value in mapping.items():
            try:
                key = int(key)
            except (TypeError, ValueError):
                pass
            copied[key] = np.asarray(value, dtype=dtype)
        return copied

    @staticmethod
    def _sample_client_local_dataset(features, labels, target_size, seed):
        """Deterministically resize one client's dataset using local rows.

        Downsampling is without replacement. When the client owns fewer rows
        than requested, all original rows are retained and the remainder is
        sampled with replacement from the same client. This operation never
        creates a synthetic feature.
        """
        features = np.asarray(features)
        labels = np.asarray(labels)
        source_size = int(labels.shape[0])
        target_size = int(target_size)
        if source_size != int(features.shape[0]):
            raise ValueError(
                "Client-local sampling requires matching feature/label rows")
        if target_size <= 0 or source_size == target_size:
            return features, labels, False, source_size
        if source_size <= 0:
            raise ValueError(
                "Cannot sample a positive target from an empty client")

        rng = np.random.RandomState(int(seed) % (2 ** 32 - 1))
        if source_size > target_size:
            indices = rng.choice(source_size, target_size, replace=False)
            used_replacement = False
        else:
            extra = rng.choice(
                source_size, target_size - source_size, replace=True)
            indices = np.concatenate(
                [np.arange(source_size, dtype=np.int64), extra])
            rng.shuffle(indices)
            used_replacement = True

        unique_samples = int(np.unique(indices).shape[0])
        return (features[indices], labels[indices], used_replacement,
                unique_samples)

    @staticmethod
    def _sample_client_local_balanced_dataset(features, labels, target_size,
                                              seed):
        """Resize local rows while preserving an equal per-class target."""
        features = np.asarray(features)
        labels = np.asarray(labels)
        source_size = int(labels.shape[0])
        target_size = int(target_size)
        if source_size != int(features.shape[0]):
            raise ValueError(
                "Balanced client sampling requires matching feature/label rows")
        if target_size <= 0 or source_size <= 0:
            raise ValueError(
                "Balanced client sampling requires positive source and target")

        classes = np.unique(labels)
        if classes.size == 0:
            raise ValueError("Balanced client sampling found no classes")
        base, remainder = divmod(target_size, int(classes.size))
        rng = np.random.RandomState(int(seed) % (2 ** 32 - 1))
        selected = []
        used_replacement = False
        for class_offset, class_id in enumerate(classes.tolist()):
            class_target = base + (1 if class_offset < remainder else 0)
            if class_target <= 0:
                continue
            class_indices = np.flatnonzero(labels == class_id)
            replace = int(class_indices.size) < int(class_target)
            used_replacement = used_replacement or replace
            selected.append(rng.choice(
                class_indices, size=class_target, replace=replace))
        indices = np.concatenate(selected).astype(np.int64, copy=False)
        rng.shuffle(indices)
        unique_samples = int(np.unique(indices).shape[0])
        return (features[indices], labels[indices], used_replacement,
                unique_samples)

    def _apply_baseline_client_sampling(self):
        """Match baseline client sample counts without feature generation."""
        target_size = int(getattr(
            self.ggeur_cfg, 'baseline_target_samples_per_client', 0) or 0)
        if target_size <= 0:
            return False

        generation_enabled = any([
            int(getattr(self.ggeur_cfg,
                        'num_generated_per_sample', 0) or 0) > 0,
            int(getattr(self.ggeur_cfg,
                        'num_generated_per_prototype', 0) or 0) > 0,
            int(getattr(self.ggeur_cfg,
                        'target_size_per_class', 0) or 0) > 0,
            bool(str(getattr(
                self.ggeur_cfg, 'task_adaptation_file', '') or '').strip()),
            bool(list(getattr(
                self.ggeur_cfg, 'task_class_counts', []) or [])),
            bool(str(getattr(
                self.ggeur_cfg, 'task_default_target_size', '') or '').strip()),
        ])
        if generation_enabled:
            logger.warning(
                f"Client {self.ID}: Ignore baseline client-local sampling "
                "because augmentation or task adaptation is enabled")
            return False
        if self.augmented_features is None or self.augmented_labels is None:
            return False

        source_size = int(len(self.augmented_labels))
        base_seed = int(getattr(self._cfg, 'seed', 0))
        client_id = int(self.ID)
        sampling_seed = (
            base_seed * 1000003 + client_id * 9176 + target_size
        ) % (2 ** 32 - 1)
        sampled = self._sample_client_local_dataset(
            self.augmented_features,
            self.augmented_labels,
            target_size,
            sampling_seed,
        )
        (self.augmented_features, self.augmented_labels,
         used_replacement, unique_samples) = sampled
        logger.info(
            "BASELINE_CLIENT_SAMPLING "
            f"client={client_id} source_samples={source_size} "
            f"target_samples={target_size} "
            f"output_samples={len(self.augmented_labels)} "
            f"replacement={1 if used_replacement else 0} "
            f"unique_source_samples={unique_samples} seed={sampling_seed}")
        return True

    def _apply_platform_training_sampling(self):
        """Resize cached generated rows without regenerating noisy features."""
        target_size = int(getattr(
            self.ggeur_cfg, 'platform_target_samples_per_client', 0) or 0)
        target_source = 'explicit'
        if target_size <= 0:
            target_size = int(getattr(
                self.ggeur_cfg,
                'platform_auto_target_samples_per_client', 0) or 0)
            target_source = 'automatic'
        generation_enabled = any([
            int(getattr(self.ggeur_cfg,
                        'num_generated_per_sample', 0) or 0) > 0,
            int(getattr(self.ggeur_cfg,
                        'num_generated_per_prototype', 0) or 0) > 0,
            int(getattr(self.ggeur_cfg,
                        'target_size_per_class', 0) or 0) > 0,
        ])
        if (target_size <= 0 or not generation_enabled or
                self.augmented_features is None or
                self.augmented_labels is None):
            return False

        source_size = int(len(self.augmented_labels))
        base_seed = int(getattr(self._cfg, 'seed', 0))
        client_id = int(self.ID)
        sampling_seed = (
            base_seed * 1000003 + client_id * 9176 + target_size + 7919
        ) % (2 ** 32 - 1)
        balanced_sampling = bool(getattr(
            self.ggeur_cfg, 'platform_class_balanced_sampling', False))
        sampler = (
            self._sample_client_local_balanced_dataset
            if balanced_sampling else self._sample_client_local_dataset)
        sampled = sampler(self.augmented_features,
                          self.augmented_labels,
                          target_size,
                          sampling_seed)
        (self.augmented_features, self.augmented_labels,
         used_replacement, unique_samples) = sampled
        logger.info(
            "PLATFORM_TRAINING_SAMPLING "
            f"client={client_id} source_samples={source_size} "
            f"target_samples={target_size} "
            f"target_source={target_source} "
            f"output_samples={len(self.augmented_labels)} "
            f"class_balanced={1 if balanced_sampling else 0} "
            f"replacement={1 if used_replacement else 0} "
            f"unique_source_samples={unique_samples} seed={sampling_seed}")
        return True

    def _restore_cached_local_statistics(self, local_statistics):
        if not isinstance(local_statistics, dict):
            return False
        means = self._copy_numpy_mapping(local_statistics.get('means', {}))
        covs = self._copy_numpy_mapping(local_statistics.get('covs', {}))
        counts = {}
        for key, value in local_statistics.get('counts', {}).items():
            try:
                counts[int(key)] = int(value)
            except (TypeError, ValueError):
                continue
        if not means or not covs or not counts:
            return False
        prototypes = self._copy_numpy_mapping(
            local_statistics.get('prototypes', means))
        self.local_means = means
        self.local_covs = covs
        self.local_counts = counts
        self.cached_local_prototypes = prototypes
        return True

    def _try_load_augmented_feature_cache(self, restore_statistics=False):
        if not getattr(self.ggeur_cfg, 'reuse_augmented_feature_cache',
                       True):
            return False
        for path in self._get_augmented_feature_cache_candidates():
            if not os.path.exists(path):
                continue
            try:
                # These files are trusted, locally generated GGEUR cache
                # artifacts and contain NumPy statistics in addition to
                # tensors.  PyTorch 2.6 changed ``torch.load`` to default to
                # ``weights_only=True``, which rejects those statistics and
                # silently forces every client to regenerate its cache.
                # Request a full load explicitly while keeping compatibility
                # with older PyTorch releases that do not expose this option.
                try:
                    cached = torch.load(path,
                                        map_location='cpu',
                                        weights_only=False)
                except TypeError:
                    cached = torch.load(path, map_location='cpu')
                metadata = cached.get('metadata', {})
                if not self._is_augmented_cache_metadata_compatible(metadata):
                    logger.info(
                        f"Client {self.ID}: Ignore augmented cache metadata "
                        f"mismatch at {path}")
                    continue
                features = cached['features']
                labels = cached['labels']
                self.augmented_features = (
                    features.numpy() if isinstance(features, torch.Tensor)
                    else np.asarray(features)
                )
                self.augmented_labels = (
                    labels.numpy() if isinstance(labels, torch.Tensor)
                    else np.asarray(labels)
                )
                self._capture_generated_distribution_snapshot()
                self._apply_baseline_client_sampling()
                self._apply_platform_training_sampling()
                cached_global_prototypes = cached.get('global_prototypes', {})
                if cached_global_prototypes:
                    self.global_prototypes = self._copy_numpy_mapping(
                        cached_global_prototypes)
                restored_stats = False
                if restore_statistics:
                    restored_stats = self._restore_cached_local_statistics(
                        cached.get('local_statistics', {}))
                dataset = AugmentedFeatureDataset(
                    self.augmented_features, self.augmented_labels)
                self.augmented_loader = DataLoader(
                    dataset,
                    batch_size=self._cfg.dataloader.batch_size,
                    shuffle=True,
                    generator=self._training_loader_generator(),
                )
                # A cache hit is still a completed task-adaptation data build.
                # Persist the same directly observable client distribution as
                # the cache-miss path so validation evidence is never absent.
                self._emit_training_distribution(cache_hit=True)
                logger.info(
                    f"Client {self.ID}: Loaded augmented feature cache "
                    f"from {path} ({len(self.augmented_labels)} samples, "
                    f"restored_statistics={restored_stats})")
                return True
            except Exception as error:
                logger.warning(
                    f"Client {self.ID}: Failed to load augmented cache "
                    f"{path}: {error}")
        return False

    def _save_augmented_feature_cache(self):
        if not getattr(self.ggeur_cfg, 'save_augmented_feature_cache',
                       True):
            return
        path = self._get_augmented_feature_cache_path()
        if path is None or self.augmented_features is None:
            return
        try:
            local_statistics = {
                'means': copy.deepcopy(self.local_means),
                'covs': copy.deepcopy(self.local_covs),
                'counts': copy.deepcopy(self.local_counts),
                'prototypes': copy.deepcopy(self.local_means),
            }
            torch.save({
                'features': torch.as_tensor(self.augmented_features).float(),
                'labels': torch.as_tensor(self.augmented_labels).long(),
                'metadata': self._augmented_cache_metadata(),
                'local_statistics': local_statistics,
                'global_prototypes': copy.deepcopy(self.global_prototypes or {}),
            }, path)
            logger.info(
                f"Client {self.ID}: Saved augmented feature cache "
                f"to {path} ({len(self.augmented_labels)} samples)")
        except Exception as error:
            logger.warning(
                f"Client {self.ID}: Failed to save augmented cache {path}: "
                f"{error}")

    def _try_start_from_augmented_cache(self):
        """Load generated samples and avoid repeated feature generation.

        Returns:
            - 'statistics_uploaded' when cached local statistics were restored
              and sent to the server, so the normal covariance broadcast can
              continue.
            - 'augmentation_ready' when only generated features were restored
              and the client can immediately join training.
            - None on cache miss.
        """
        reuse_cache = getattr(self.ggeur_cfg,
                              'reuse_augmented_feature_cache', True)
        legacy_headonly_hot = getattr(
            self.ggeur_cfg,
            'headonly_skip_round0_if_augmented_cache_exists',
            False)
        if not (reuse_cache or legacy_headonly_hot):
            return None
        if self.use_cnn_distillation or self.use_feature_alignment:
            logger.info(
                f"Client {self.ID}: Augmented cache-hot mode disabled for "
                "CNN distillation/feature-alignment because those modes "
                "require original-image loaders.")
            return None
        if not self._try_load_augmented_feature_cache(
                restore_statistics=True):
            return None
        has_restored_stats = bool(
            self.local_means and self.local_covs and self.local_counts)
        needs_server_context = (
            bool(getattr(self.ggeur_cfg, 'use_fedproto', False)) or
            bool(getattr(self.ggeur_cfg, 'use_promptfl', False)))
        if needs_server_context and not has_restored_stats:
            logger.info(
                f"Client {self.ID}: Ignore augmented cache-hot direct start "
                "because this method needs server-side prototypes/context "
                "and the cache does not contain local statistics.")
            self.augmented_features = None
            self.augmented_labels = None
            self.augmented_loader = None
            return None

        self._build_mlp_classifier()
        self.augmentation_done = True
        self.local_features = {}
        self.local_labels = {}
        if has_restored_stats:
            logger.info(
                f"Client {self.ID}: Augmented cache-hot mode restored local "
                "statistics; upload cached statistics and skip feature "
                "extraction/generation.")
            self._upload_local_statistics()
            return 'statistics_uploaded'

        self.statistics_uploaded = True
        logger.info(
            f"Client {self.ID}: Augmented cache-hot mode active; skip "
            "round-0 feature extraction/statistics/generation and train "
            "on cached generated samples.")
        return 'augmentation_ready'

    def _should_skip_statistics_phase(self):
        """Whether baseline mode can skip server-side statistics exchange."""
        return (
            int(getattr(self.ggeur_cfg, 'num_generated_per_sample', 0)) == 0
            and int(getattr(self.ggeur_cfg, 'num_generated_per_prototype', 0)) == 0
            and int(getattr(self.ggeur_cfg, 'target_size_per_class', 0)) == 0
            and not getattr(self.ggeur_cfg, 'use_fedproto', False)
            and not getattr(self.ggeur_cfg, 'use_cnn_distillation', False)
            and not getattr(self.ggeur_cfg, 'use_feature_alignment', False)
            and not getattr(self.ggeur_cfg, 'use_promptfl', False)
        )

    def callback_funcs_for_model_para(self, message: Message):
        """
        Handle model parameters message.
        In Round 0: extract features and upload statistics
        In Round 1+: train on augmented data (and optionally CNN with distillation)

        For Separated Training Mode:
        - Phase 1 (classifier): Train classifier on GGEUR_Clip augmented features
        - Phase 2 (cnn_backbone): Train CNN backbone with frozen classifier
        """
        round_idx = message.state
        sender = message.sender
        timestamp = message.timestamp
        # gRPC serializes every tensor leaf in ``model_para`` messages to
        # bytes.  Hierarchical subservers intentionally forward that payload
        # unchanged, so restore the full tree before loading the global head
        # or reading FedProto metadata.
        content = self._decode_parameter_tree(message.content)
        message.content = content

        # Update state
        self.state = round_idx

        # Statistics collection round
        if round_idx == self.ggeur_cfg.statistics_round and not self.statistics_uploaded:
            logger.info(f"Client {self.ID}: Round {round_idx} - Statistics collection phase")

            cache_hot_status = self._try_start_from_augmented_cache()
            if cache_hot_status == 'statistics_uploaded':
                self._maybe_fail_for_distributed_validation(
                    'after_statistics_upload', round_idx)
                return
            if cache_hot_status == 'augmentation_ready':
                ready_content = 'ready'
                if (bool(getattr(
                        self.ggeur_cfg, 'prototype_classifier_init', False))
                        and self.global_prototypes):
                    ready_content = {
                        'status': 'ready',
                        'global_prototypes': self._serialize_array_payload(
                            self.global_prototypes),
                    }
                self.comm_manager.send(
                    Message(
                        msg_type='augmentation_ready',
                        sender=self.ID,
                        receiver=[self.server_id],
                        state=self.state,
                        content=ready_content
                    )
                )
                self._maybe_fail_for_distributed_validation(
                    'after_augmentation_ready', round_idx)
                return

            # Extract CLIP features
            self._extract_clip_features()

            if self._should_skip_statistics_phase():
                logger.info(f"Client {self.ID}: Baseline mode detected, skip "
                            f"statistics upload and prepare local features "
                            f"for FedAvg directly")
                self._perform_augmentation()
                self._build_mlp_classifier()
                self.statistics_uploaded = True

                if getattr(self.ggeur_cfg, 'unload_extractor_after_cache', True):
                    self._unload_feature_extractor()

                self.comm_manager.send(
                    Message(
                        msg_type='augmentation_ready',
                        sender=self.ID,
                        receiver=[self.server_id],
                        state=self.state,
                        content='ready'
                    )
                )
                self._maybe_fail_for_distributed_validation(
                    'after_augmentation_ready', round_idx)
                return

            # Compute local statistics
            self._compute_local_statistics()

            # Upload to server
            self._upload_local_statistics()
            self._maybe_fail_for_distributed_validation(
                'after_statistics_upload', round_idx)

            # Unload extractor to free VRAM (features are cached to disk)
            if getattr(self.ggeur_cfg, 'unload_extractor_after_cache', True):
                if not (self.use_cnn_distillation or self.use_feature_alignment):
                    self._unload_feature_extractor()

            return  # Don't do normal training in this round

        # Wait for augmentation to complete
        if not self.augmentation_done:
            logger.warning(f"Client {self.ID}: Augmentation not done, skipping training")
            # Send empty model update
            self.comm_manager.send(
                Message(
                    msg_type='model_para',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    timestamp=timestamp,
                    content=(0, self.trainer.get_model_para())
                )
            )
            return

        # Handle Separated Training Mode
        if self.use_separated_training:
            self._maybe_fail_for_distributed_validation(
                'before_train_round', round_idx)
            self._handle_separated_training(message)
            return

        # Normal training round on augmented data
        logger.info(f"Client {self.ID}: Round {round_idx} - Training on augmented data")
        self._maybe_fail_for_distributed_validation(
            'before_train_round', round_idx)

        # Parse content - may contain both MLP and CNN parameters
        mlp_para = None
        cnn_para = None

        if content is not None:
            if isinstance(content, dict):
                if self.use_fedproto and \
                        'fedproto_global_prototypes' in content:
                    self._set_fedproto_global_prototypes(
                        content.get('fedproto_global_prototypes'))
                if 'mlp' in content:
                    mlp_para = content.get('mlp')
                    cnn_para = content.get('cnn')
                else:
                    # Backward compatibility: content is just MLP parameters
                    mlp_para = {
                        key: value
                        for key, value in content.items()
                        if key != 'fedproto_global_prototypes'
                    }

        # Load global prompt ctx if PromptFL enabled
        if self.use_promptfl and content is not None and isinstance(content, dict):
            prompt_para = content.get('prompt')
            if prompt_para is not None and self.prompt_learner is not None:
                try:
                    ctx_tensor = prompt_para.get('ctx')
                    if ctx_tensor is not None:
                        self.prompt_learner.ctx.data = ctx_tensor.to(self.device)
                        # Save global ctx for FedProx proximal term
                        self.global_prompt_ctx = ctx_tensor.to(self.device).detach().clone()
                except Exception as e:
                    logger.debug(f"Client {self.ID}: Could not load prompt ctx: {e}")

        # Update MLP with global model parameters
        if mlp_para is not None and self.mlp_classifier is not None:
            try:
                self.mlp_classifier.load_state_dict(mlp_para)
            except Exception as e:
                logger.debug(f"Client {self.ID}: Could not load MLP state dict: {e}")

        # MOON: snapshot global model after loading global params, before local training
        if self.use_moon and self.mlp_classifier is not None:
            self.moon_global_model = copy.deepcopy(self.mlp_classifier)
            self.moon_global_model.eval()
            for p in self.moon_global_model.parameters():
                p.requires_grad_(False)

        # Update CNN with global model parameters (for both distillation and feature alignment)
        if (self.use_cnn_distillation or self.use_feature_alignment) and cnn_para is not None and self.cnn_model is not None:
            try:
                self.cnn_model.load_state_dict(cnn_para)
            except Exception as e:
                logger.debug(f"Client {self.ID}: Could not load CNN state dict: {e}")

        # Train MLP on augmented features
        mlp_sample_size, mlp_model_para, mlp_results = self._train_on_augmented_data()

        # MOON: save current local model as previous model for next round
        if self.use_moon and self.mlp_classifier is not None:
            self.moon_prev_model = copy.deepcopy(self.mlp_classifier)
            self.moon_prev_model.eval()
            for p in self.moon_prev_model.parameters():
                p.requires_grad_(False)

        # Train CNN based on mode
        cnn_warmup_rounds = getattr(self.ggeur_cfg, 'cnn_warmup_rounds', 0)

        if self.use_feature_alignment:
            # Feature alignment mode: train CNN from scratch
            should_train_cnn = round_idx > cnn_warmup_rounds

            if should_train_cnn:
                cnn_sample_size, cnn_model_para, cnn_results = self._train_cnn_with_feature_alignment()

                combined_para = {
                    'mlp': mlp_model_para,
                    'cnn': cnn_model_para
                }
                sample_size = mlp_sample_size
            else:
                logger.info(f"Client {self.ID}: CNN warmup - skipping training in round {round_idx}")
                combined_para = {
                    'mlp': mlp_model_para,
                    'cnn': copy.deepcopy(self.cnn_model.state_dict()) if self.cnn_model else None
                }
                sample_size = mlp_sample_size

        elif self.use_cnn_distillation:
            # Knowledge distillation mode
            should_train_cnn = round_idx > cnn_warmup_rounds

            if should_train_cnn:
                cnn_sample_size, cnn_model_para, cnn_results = self._train_cnn_with_distillation()

                combined_para = {
                    'mlp': mlp_model_para,
                    'cnn': cnn_model_para
                }
                sample_size = mlp_sample_size
            else:
                logger.info(f"Client {self.ID}: CNN warmup - skipping distillation in round {round_idx}")
                combined_para = {
                    'mlp': mlp_model_para,
                    'cnn': copy.deepcopy(self.cnn_model.state_dict()) if self.cnn_model else None
                }
                sample_size = mlp_sample_size
        else:
            # Standard mode: only MLP
            combined_para = mlp_model_para
            sample_size = mlp_sample_size

        # PromptFL: train soft prompts on augmented features and attach to combined_para
        if self.use_promptfl:
            _, prompt_para, _ = self._train_prompt_on_augmented_data()
            if isinstance(combined_para, dict):
                combined_para['prompt'] = prompt_para
            else:
                combined_para = {'mlp': combined_para, 'prompt': prompt_para}

        # FedProto metadata is aggregated independently from model weights.
        # Always wrap a plain state_dict so the root server and hierarchical
        # subservers can distinguish trainable parameters from prototypes.
        if self.use_fedproto:
            if not isinstance(combined_para, dict) or \
                    'mlp' not in combined_para:
                combined_para = {'mlp': combined_para}
            combined_para['fedproto_local_prototypes'] = copy.deepcopy(
                self.fedproto_local_prototypes)
            combined_para['fedproto_local_counts'] = copy.deepcopy(
                self.fedproto_local_counts)

        # Send model parameters
        self.comm_manager.send(
            Message(
                msg_type='model_para',
                sender=self.ID,
                receiver=[sender],
                state=self.state,
                timestamp=timestamp,
                content=(sample_size, combined_para)
            )
        )

    def callback_for_client_eval(self, message: Message):
        """Evaluate the aggregated MLP on this client's local test split."""
        round_idx = int(message.state)
        model_para = self._decode_parameter_tree(message.content)
        self.state = round_idx

        self.domain_prototype_ensemble = {}
        self.domain_personalized_head = None
        if isinstance(model_para, dict) and \
                'domain_prototype_ensemble' in model_para:
            raw_ensemble = model_para.get('domain_prototype_ensemble', {})
            for class_idx, prototypes in raw_ensemble.items():
                values = np.asarray(prototypes, dtype=np.float32)
                if values.ndim == 2 and values.shape[1] == self.embedding_dim:
                    self.domain_prototype_ensemble[int(class_idx)] = values
        if isinstance(model_para, dict) and \
                'domain_personalized_heads' in model_para:
            heads = model_para.get('domain_personalized_heads', {})
            dataloader = self._get_local_test_loader()
            dataset = getattr(dataloader, 'dataset', None) \
                if dataloader is not None else None
            domain = getattr(dataset, 'domain', None)
            if domain is None:
                domain = getattr(
                    getattr(dataset, 'dataset', None), 'domain', '')
            head = heads.get(str(domain))
            if head:
                self.domain_personalized_head = head
        if isinstance(model_para, dict) and 'mlp' in model_para:
            model_para = model_para.get('mlp')

        self._ensure_eval_classifier_matches(model_para)
        if model_para is not None and self.mlp_classifier is not None:
            try:
                self.mlp_classifier.load_state_dict(model_para)
            except Exception as error:
                # Continuing here evaluates the client's stale local head and
                # silently produces a formally shaped but invalid result.
                # Terminal evidence must always use the root-aggregated head.
                logger.error(
                    f"Client {self.ID}: Could not load MLP for local eval: "
                    f"{error}")
                raise RuntimeError(
                    f"Client {self.ID}: terminal evaluation rejected because "
                    "the root-aggregated classifier could not be loaded") \
                    from error

        metrics = self._evaluate_mlp_on_local_test(round_idx)
        self.comm_manager.send(
            Message(
                msg_type='client_eval_metrics',
                sender=self.ID,
                receiver=[message.sender],
                state=round_idx,
                timestamp=message.timestamp,
                content=metrics
            )
        )

    def _ensure_eval_classifier_matches(self, model_para):
        """Build the classifier architecture encoded by an eval state dict.

        Platform cases use a two-layer MLP while FedAvg/FedProx use a linear
        head.  Client evaluation is common to both, so it must infer the head
        shape from the server payload instead of reusing a classifier left by
        local training.
        """
        if not isinstance(model_para, dict):
            if self.mlp_classifier is None:
                self._build_mlp_classifier()
            return

        weight_keys = [
            key for key, value in model_para.items()
            if str(key).endswith('weight') and torch.is_tensor(value) and
            value.ndim == 2
        ]
        if not weight_keys:
            if self.mlp_classifier is None:
                self._build_mlp_classifier()
            return

        if 'weight' in model_para and torch.is_tensor(model_para['weight']):
            weight = model_para['weight']
            expected = nn.Linear(int(weight.shape[1]), int(weight.shape[0]))
        elif '0.weight' in model_para and '3.weight' in model_para:
            first = model_para['0.weight']
            last = model_para['3.weight']
            expected = nn.Sequential(
                nn.Linear(int(first.shape[1]), int(first.shape[0])),
                nn.ReLU(),
                nn.Dropout(float(self.ggeur_cfg.mlp_dropout)),
                nn.Linear(int(last.shape[1]), int(last.shape[0])))
        else:
            if self.mlp_classifier is None:
                self._build_mlp_classifier()
            return

        current = self.mlp_classifier
        current_state = current.state_dict() if current is not None else {}
        shape_mismatch = any(
            key not in current_state or
            tuple(current_state[key].shape) != tuple(value.shape)
            for key, value in model_para.items() if torch.is_tensor(value))
        if current is None or set(current_state) != set(model_para) or \
                shape_mismatch:
            self.mlp_classifier = expected.to(self.device)
            logger.info(
                f"Client {self.ID}: Rebuilt local eval classifier from "
                f"server state keys={sorted(model_para)}")

    def _get_local_test_loader(self):
        test_data = None
        try:
            test_data = self.trainer.ctx.data.get('test', None)
        except Exception:
            test_data = None
        if test_data is None and isinstance(self.data, dict):
            test_data = self.data.get('test', None)
        return test_data

    def _extract_eval_features(self, dataloader):
        dataset = dataloader.dataset if hasattr(dataloader, 'dataset') \
            else dataloader

        from torch.utils.data import Subset
        is_subset = isinstance(dataset, Subset)
        if is_subset:
            base_dataset = dataset.dataset
            subset_indices = dataset.indices
            domain = getattr(base_dataset, 'domain', None)
        else:
            base_dataset = dataset
            subset_indices = None
            domain = getattr(dataset, 'domain', None)

        shared_eval = self._load_shared_eval_cache(domain)
        if shared_eval is not None:
            return shared_eval

        cache_path = self._get_feature_cache_path(domain)
        feature_cache = self._load_feature_cache(cache_path)
        feature_cache = {
            path: feat
            for path, feat in feature_cache.items()
            if self._is_valid_feature_vector(feat)
        }

        # Text datasets expose stable sample IDs rather than image paths.
        # Their complete domain cache already contains the held-out IDs (the
        # cache-preparation step validates this).  Resolve those IDs directly
        # so terminal evaluation never imports or runs one BERT model per
        # client process.
        if self.feature_extractor_type == 'bert':
            text_samples = []
            missing = []
            for idx in range(len(dataset)):
                text, label, sample_id, _ = self._resolve_dataset_sample(
                    dataset, idx)
                cache_key = self._feature_cache_key(sample_id)
                text_samples.append((text, int(label), cache_key))
                if cache_key not in feature_cache or not \
                        self._is_valid_feature_vector(
                            feature_cache.get(cache_key)):
                    missing.append((text, int(label), cache_key))

            if missing:
                if bool(getattr(
                        self.ggeur_cfg,
                        'require_complete_feature_cache', False)):
                    preview = ', '.join(item[2] for item in missing[:3])
                    raise RuntimeError(
                        f"Client {self.ID}: BERT eval feature cache is "
                        f"incomplete; missing {len(missing)}/"
                        f"{len(text_samples)} samples (first keys: "
                        f"{preview})")
                self._load_bert_extractor()
                batch_size = int(getattr(
                    self.ggeur_cfg, 'bert_batch_size', 32))
                for start in range(0, len(missing), max(batch_size, 1)):
                    batch = missing[start:start + max(batch_size, 1)]
                    encoded = self._encode_text_batch(
                        [item[0] for item in batch])
                    for feat, (_, _, cache_key) in zip(encoded, batch):
                        feature_cache[cache_key] = feat
                self._save_feature_cache(cache_path, feature_cache)

            features = np.asarray(
                [feature_cache[item[2]] for item in text_samples],
                dtype=np.float32)
            labels = np.asarray(
                [item[1] for item in text_samples], dtype=np.int64)
            logger.info(
                f"Client {self.ID}: Loaded {len(labels)} held-out BERT "
                f"features from {cache_path}")
            return torch.from_numpy(features), torch.from_numpy(labels)

        has_paths = hasattr(base_dataset, 'data') and \
            len(base_dataset.data) > 0 and isinstance(base_dataset.data[0], str)
        features = []
        labels = []
        paths_to_extract = []
        base_indices_to_extract = []

        if has_paths:
            num_samples = len(subset_indices) if is_subset else len(base_dataset)
            for local_idx in range(num_samples):
                base_idx = subset_indices[local_idx] if is_subset else local_idx
                img_path = base_dataset.data[base_idx]
                label = int(base_dataset.targets[base_idx])
                cache_key = self._feature_cache_key(img_path)
                cached = feature_cache.get(cache_key)
                if cached is not None and self._is_valid_feature_vector(cached):
                    features.append(cached)
                    labels.append(label)
                else:
                    paths_to_extract.append(cache_key)
                    base_indices_to_extract.append(base_idx)

            if paths_to_extract:
                self._load_feature_extractor()
                batch_size = getattr(self.ggeur_cfg, 'extract_batch_size', 64)
                use_fp16 = getattr(self.ggeur_cfg, 'use_fp16_extraction',
                                   True) and torch.cuda.is_available()
                cache_updated = False
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_fp16):
                    for i in range(0, len(base_indices_to_extract), batch_size):
                        batch_indices = base_indices_to_extract[i:i + batch_size]
                        batch_paths = paths_to_extract[i:i + batch_size]
                        images = []
                        batch_labels = []
                        for base_idx in batch_indices:
                            image, label = base_dataset[base_idx]
                            images.append(image)
                            batch_labels.append(int(label))
                        images = torch.stack(images).to(self.device)
                        if self.feature_extractor_type == 'cnn':
                            batch_features = self.cnn_extractor(images)
                        elif self.feature_extractor_type == 'timm':
                            batch_features = self.timm_extractor(images)
                        else:
                            batch_features = self.clip_model.encode_image(images)
                        batch_features = batch_features.cpu().numpy()
                        for feat, label, path in zip(batch_features,
                                                     batch_labels,
                                                     batch_paths):
                            if not self._is_valid_feature_vector(feat):
                                continue
                            feature_cache[path] = feat
                            cache_updated = True
                            features.append(feat)
                            labels.append(label)
                if cache_updated:
                    self._save_feature_cache(cache_path, feature_cache)
        else:
            self._load_feature_extractor()
            use_fp16 = getattr(self.ggeur_cfg, 'use_fp16_extraction',
                               True) and torch.cuda.is_available()
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_fp16):
                for batch in dataloader:
                    if len(batch) < 2:
                        continue
                    images, batch_labels = batch[0].to(self.device), batch[1]
                    if self.feature_extractor_type == 'cnn':
                        batch_features = self.cnn_extractor(images)
                    elif self.feature_extractor_type == 'timm':
                        batch_features = self.timm_extractor(images)
                    else:
                        batch_features = self.clip_model.encode_image(images)
                    features.extend(batch_features.cpu().numpy())
                    labels.extend([int(x) for x in batch_labels.cpu().numpy()])

        if not features:
            return None, None
        return (
            torch.as_tensor(np.asarray(features), dtype=torch.float32),
            torch.as_tensor(np.asarray(labels), dtype=torch.long),
        )

    def _load_domainnet_eval_cache(self, domain):
        """Compatibility wrapper for the former DomainNet-only cache API."""
        return self._load_shared_eval_cache(domain)

    def _load_shared_eval_cache(self, domain):
        """Load a deterministic held-out vision cache when available.

        Every logical OfficeHome client in a domain uses the same manifest
        test split.  DomainNet uses an equivalent per-client sharded cache.
        Reusing those verified features avoids loading a large frozen vision
        extractor independently in every terminal process.
        """
        data_type = str(getattr(self._cfg.data, 'type', '')).lower()
        is_domainnet = any(token in data_type for token in (
            'domainnet', 'domain-net', 'domain_net'))
        is_officehome = any(token in data_type for token in (
            'officehome', 'office-home', 'office_home'))
        if not is_domainnet and not is_officehome:
            return None
        domain = str(domain or '')
        if not domain:
            return None
        cache_dir = Path(str(getattr(
            self.ggeur_cfg, 'feature_cache_dir', '') or ''))
        if is_domainnet:
            client_num = int(self._cfg.federate.client_num)
            suffix = (
                f'_terminal_client{int(self.ID):06d}of{client_num:06d}.npz')
            matches = sorted(
                path for path in cache_dir.glob(
                    f'domainnet_{domain}_test_*{suffix}')
                if path.name.endswith(suffix))
            cache_description = 'DomainNet terminal feature shard'
        else:
            # The server cache is named from the configured data type, which
            # may use any of the supported OfficeHome spellings.  Match by
            # domain and exclude any unrelated terminal shards.
            matches = sorted(
                path for path in cache_dir.glob(f'*_{domain}_test_*.npz')
                if 'terminal_client' not in path.name and any(
                    token in path.name.lower() for token in (
                        'officehome', 'office-home', 'office_home')))
            cache_description = 'OfficeHome shared test feature cache'
        if len(matches) != 1:
            if bool(getattr(
                    self.ggeur_cfg, 'require_complete_feature_cache', False)):
                raise RuntimeError(
                    f"Client {self.ID}: Expected one {cache_description} "
                    f"for domain={domain} in {cache_dir}, "
                    f"found {len(matches)}")
            return None
        with np.load(matches[0], allow_pickle=False) as data:
            features = np.asarray(data['features'], dtype=np.float32)
            labels = np.asarray(data['labels'], dtype=np.int64)
        if features.ndim != 2 or features.shape[1] != int(self.embedding_dim):
            raise ValueError(
                f"Client {self.ID}: Invalid held-out feature shape "
                f"{features.shape}; expected (*, {self.embedding_dim})")
        if labels.shape != (features.shape[0],):
            raise ValueError(
                f"Client {self.ID}: Invalid held-out label shape "
                f"{labels.shape}")
        logger.info(
            f"Client {self.ID}: Loaded {len(labels)} held-out features "
            f"from {matches[0]}")
        return torch.from_numpy(features), torch.from_numpy(labels)

    def _evaluate_mlp_on_local_test(self, round_idx):
        dataloader = self._get_local_test_loader()
        dataset = getattr(dataloader, 'dataset', None) \
            if dataloader is not None else None
        domain = getattr(dataset, 'domain', None)
        if domain is None:
            domain = getattr(getattr(dataset, 'dataset', None), 'domain', '')
        domain = str(domain or '')
        if dataloader is None or self.mlp_classifier is None:
            logger.warning(
                f"Client {self.ID}: No local test data or MLP for eval")
            return {
                'accuracy': 0.0,
                'loss': 0.0,
                'correct': 0,
                'total': 0,
                'domain': domain,
            }

        features, labels = self._extract_eval_features(dataloader)
        if features is None:
            return {
                'accuracy': 0.0,
                'loss': 0.0,
                'correct': 0,
                'total': 0,
                'domain': domain,
            }

        self.mlp_classifier.eval()
        criterion = nn.CrossEntropyLoss(reduction='sum')
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        eval_loader = DataLoader(
            AugmentedFeatureDataset(features, labels),
            batch_size=getattr(self._cfg.dataloader, 'batch_size', 32),
            shuffle=False,
            num_workers=0)

        with torch.no_grad():
            for batch_features, batch_labels in eval_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                if self.domain_personalized_head:
                    weight = torch.as_tensor(
                        self.domain_personalized_head['weight'],
                        device=self.device, dtype=batch_features.dtype)
                    bias = torch.as_tensor(
                        self.domain_personalized_head['bias'],
                        device=self.device, dtype=batch_features.dtype)
                    normalized = F.normalize(batch_features, p=2, dim=1)
                    outputs = F.linear(normalized, weight, bias)
                elif self.domain_prototype_ensemble:
                    outputs = torch.full(
                        (batch_features.shape[0],
                         int(self._cfg.model.num_classes)),
                        -torch.inf, device=self.device,
                        dtype=batch_features.dtype)
                    for class_idx, prototypes in \
                            self.domain_prototype_ensemble.items():
                        values = torch.as_tensor(
                            prototypes, device=self.device,
                            dtype=batch_features.dtype)
                        outputs[:, int(class_idx)] = torch.max(
                            batch_features @ values.transpose(0, 1),
                            dim=1).values
                else:
                    outputs = self.mlp_classifier(batch_features)
                total_loss += criterion(outputs, batch_labels).item()
                predicted = torch.argmax(outputs, dim=1)
                total_correct += (predicted == batch_labels).sum().item()
                total_samples += int(batch_labels.numel())

        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        logger.info(
            f"Client {self.ID}: Round {round_idx} local MLP eval - "
            f"acc={accuracy:.4f}, loss={avg_loss:.4f}, "
            f"correct={total_correct}, total={total_samples}")
        return {
            'accuracy': float(accuracy),
            'loss': float(avg_loss),
            'correct': int(total_correct),
            'total': int(total_samples),
            'domain': domain,
        }

    def _handle_separated_training(self, message: Message):
        """Handle training in separated training mode"""
        round_idx = message.state
        sender = message.sender
        timestamp = message.timestamp
        content = message.content

        # Parse phase from content
        if isinstance(content, dict) and 'phase' in content:
            phase = content['phase']
        else:
            phase = 'classifier'  # Default to classifier phase

        if phase == 'classifier':
            # Phase 1: Train classifier on GGEUR_Clip augmented features
            self.training_phase = 'classifier'
            logger.info(f"Client {self.ID}: Phase 1 (Classifier Training) - Round {round_idx}")

            # Update classifier with global parameters
            classifier_para = content.get('classifier') if isinstance(content, dict) else content
            if classifier_para is not None and self.mlp_classifier is not None:
                try:
                    self.mlp_classifier.load_state_dict(classifier_para)
                except Exception as e:
                    logger.debug(f"Client {self.ID}: Could not load classifier: {e}")

            # Train classifier on augmented features
            sample_size, model_para, results = self._train_on_augmented_data()

            # Send classifier parameters
            self.comm_manager.send(
                Message(
                    msg_type='model_para',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    timestamp=timestamp,
                    content=(sample_size, {'classifier': model_para})
                )
            )

        else:
            # Phase 2: Train CNN backbone with frozen classifier
            self.training_phase = 'cnn_backbone'
            logger.info(f"Client {self.ID}: Phase 2 (CNN Backbone Training) - Round {round_idx}")

            # Get pretrained classifier (frozen)
            classifier_para = content.get('classifier') if isinstance(content, dict) else None
            if classifier_para is not None:
                self._setup_pretrained_classifier(classifier_para)

            # Get CNN backbone parameters
            cnn_backbone_para = content.get('cnn_backbone') if isinstance(content, dict) else None

            # Build CNN backbone if not already built
            if self.cnn_backbone is None:
                self._build_cnn_backbone()

            # Load CNN backbone parameters
            if cnn_backbone_para is not None and self.cnn_backbone is not None:
                try:
                    self.cnn_backbone.load_state_dict(cnn_backbone_para)
                except Exception as e:
                    logger.debug(f"Client {self.ID}: Could not load CNN backbone: {e}")

            # Setup image loader if not ready
            if self.original_image_loader is None:
                self._setup_original_image_loader()

            # Train CNN backbone with frozen classifier
            sample_size, backbone_para, results = self._train_cnn_backbone_separated()

            # Send only CNN backbone parameters
            self.comm_manager.send(
                Message(
                    msg_type='model_para',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    timestamp=timestamp,
                    content=(sample_size, {'cnn_backbone': backbone_para})
                )
            )

    def _setup_pretrained_classifier(self, classifier_para):
        """Setup pretrained classifier for Phase 2"""
        if self.pretrained_classifier is not None:
            return  # Already setup

        # Build classifier with same architecture as MLP
        num_classes = self._cfg.model.num_classes
        input_dim = self.embedding_dim
        hidden_dim = self.ggeur_cfg.mlp_hidden_dim

        if hidden_dim > 0:
            self.pretrained_classifier = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.ggeur_cfg.mlp_dropout),
                nn.Linear(hidden_dim, num_classes)
            )
        else:
            self.pretrained_classifier = nn.Linear(input_dim, num_classes)

        self.pretrained_classifier = self.pretrained_classifier.to(self.device)

        # Load pretrained weights
        try:
            self.pretrained_classifier.load_state_dict(classifier_para)
            logger.info(f"Client {self.ID}: Loaded pretrained classifier")
        except Exception as e:
            logger.warning(f"Client {self.ID}: Could not load pretrained classifier: {e}")

        # Freeze classifier
        for param in self.pretrained_classifier.parameters():
            param.requires_grad = False
        self.pretrained_classifier.eval()

        logger.info(f"Client {self.ID}: Pretrained classifier frozen for Phase 2")

    def _build_cnn_backbone(self):
        """Build CNN backbone for separated training Phase 2"""
        from federatedscope.contrib.model.ggeur_cnn import GGEUR_CNN_Backbone

        num_classes = self._cfg.model.num_classes
        cnn_model_name = getattr(self.ggeur_cfg, 'cnn_model', 'resnet18')

        self.cnn_backbone = GGEUR_CNN_Backbone(
            model_name=cnn_model_name,
            num_classes=num_classes,
            pretrained=False  # From scratch
        )
        self.cnn_backbone = self.cnn_backbone.to(self.device)

        logger.info(f"Client {self.ID}: Built CNN backbone ({cnn_model_name}) for Phase 2 - FROM SCRATCH")

    def _train_cnn_backbone_separated(self):
        """
        Train CNN backbone with frozen pretrained classifier.

        Loss function: CE(classifier(backbone(x)), y) + λ * MSE(backbone_feat, CLIP_feat)

        The alignment loss is CRITICAL because:
        - The classifier was trained on CLIP features
        - CNN backbone must produce CLIP-like features for classifier to work
        - Without alignment, CNN features are in a completely different space
        """
        if self.cnn_backbone is None or self.pretrained_classifier is None:
            logger.warning(f"Client {self.ID}: CNN backbone or classifier not ready")
            return 0, {}, {}

        if self.original_image_loader is None:
            logger.warning(f"Client {self.ID}: Image loader not ready")
            return 0, {}, {}

        # Load CLIP model for feature alignment (CRITICAL!)
        if self.clip_model is None:
            logger.info(f"Client {self.ID}: Loading CLIP model for feature alignment...")
            self._load_clip_model()

        if self.clip_model is None:
            logger.warning(f"Client {self.ID}: CLIP model not available, training without alignment")

        # Training settings
        cnn_lr = getattr(self.ggeur_cfg, 'cnn_lr', 0.01)
        cnn_epochs = getattr(self.ggeur_cfg, 'cnn_local_epochs', 10)
        align_weight = getattr(self.ggeur_cfg, 'align_weight', 1.0)  # Feature alignment weight
        cnn_weight_decay = getattr(self.ggeur_cfg, 'cnn_weight_decay', 5e-4)  # 正则化
        cnn_dropout = getattr(self.ggeur_cfg, 'cnn_dropout', 0.5)  # Dropout比例

        # 诊断信息
        num_samples = len(self.original_image_loader.dataset) if hasattr(self.original_image_loader, 'dataset') else 0
        logger.info(f"Client {self.ID}: Phase 2 训练诊断 - CLIP加载:{self.clip_model is not None}, "
                   f"align_weight={align_weight}, lr={cnn_lr}, epochs={cnn_epochs}, samples={num_samples}")

        self.cnn_backbone.train()
        self.pretrained_classifier.eval()  # Classifier stays frozen
        if self.clip_model is not None:
            self.clip_model.eval()

        optimizer = torch.optim.SGD(
            self.cnn_backbone.parameters(),
            lr=cnn_lr,
            momentum=0.9,
            weight_decay=cnn_weight_decay  # 增强正则化
        )

        ce_criterion = nn.CrossEntropyLoss()
        mse_criterion = nn.MSELoss()

        total_loss = 0.0
        total_ce_loss = 0.0
        total_align_loss = 0.0
        total_correct = 0
        total_samples = 0

        logger.info(f"Client {self.ID}: CNN training with CLIP feature alignment, align_weight={align_weight}")

        for epoch in range(cnn_epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0

            for batch in self.original_image_loader:
                if len(batch) >= 2:
                    images, labels = batch[0], batch[1]
                else:
                    continue

                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()

                # Forward: backbone -> features
                cnn_features = self.cnn_backbone(images)  # 512-dim features

                # Get CLIP features as alignment target (原始特征，不归一化!)
                if self.clip_model is not None:
                    with torch.no_grad():
                        clip_features = self.clip_model.encode_image(images).float()
                        # 不再归一化! 保持原始尺度以匹配分类器期望的输入

                # Classification loss (through frozen classifier)
                logits = self.pretrained_classifier(cnn_features)
                ce_loss = ce_criterion(logits, labels)

                # Feature alignment loss - 使用原始特征对齐 (不归一化)
                # 这样CNN学到的特征尺度也会匹配CLIP
                if self.clip_model is not None:
                    # 方法1: 直接MSE (可能尺度太大)
                    # align_loss = mse_criterion(cnn_features, clip_features)

                    # 方法2: 余弦相似度 + 尺度匹配
                    # 余弦相似度损失 (方向对齐)
                    cnn_norm = F.normalize(cnn_features, p=2, dim=1)
                    clip_norm = F.normalize(clip_features, p=2, dim=1)
                    cosine_loss = 1 - (cnn_norm * clip_norm).sum(dim=1).mean()

                    # 尺度匹配损失 (让CNN特征的范数接近CLIP特征的范数)
                    cnn_magnitude = torch.norm(cnn_features, p=2, dim=1)
                    clip_magnitude = torch.norm(clip_features, p=2, dim=1)
                    magnitude_loss = mse_criterion(cnn_magnitude, clip_magnitude) / (clip_magnitude.mean() ** 2 + 1e-6)

                    # 组合: 方向 + 尺度
                    align_loss = cosine_loss + 0.1 * magnitude_loss

                    loss = ce_loss + align_weight * align_loss
                    total_align_loss += align_loss.item() * images.size(0)

                    # 第一个batch打印诊断信息
                    if epoch == 0 and epoch_samples == 0:
                        logger.info(f"Client {self.ID}: 诊断 - CNN范数={cnn_magnitude.mean().item():.2f}, "
                                   f"CLIP范数={clip_magnitude.mean().item():.2f}, "
                                   f"cosine_loss={cosine_loss.item():.4f}, "
                                   f"magnitude_loss={magnitude_loss.item():.4f}")
                else:
                    loss = ce_loss
                    align_loss = torch.tensor(0.0)

                # Backward: only updates backbone
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.cnn_backbone.parameters(), max_norm=1.0)
                optimizer.step()

                # Statistics
                batch_size = images.size(0)
                total_loss += loss.item() * batch_size
                total_ce_loss += ce_loss.item() * batch_size
                _, predicted = torch.max(logits, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += batch_size

                epoch_loss += loss.item() * batch_size
                epoch_correct += (predicted == labels).sum().item()
                epoch_samples += batch_size

            if epoch_samples > 0:
                epoch_acc = epoch_correct / epoch_samples
                # 计算平均特征范数用于诊断
                logger.info(f"Client {self.ID}: CNN Epoch {epoch+1}/{cnn_epochs}, "
                           f"Loss={epoch_loss/epoch_samples:.4f}, Acc={epoch_acc:.4f}")

        # Compute averages
        if total_samples > 0:
            avg_loss = total_loss / total_samples
            avg_ce_loss = total_ce_loss / total_samples
            avg_align_loss = total_align_loss / total_samples
            accuracy = total_correct / total_samples
        else:
            avg_loss = avg_ce_loss = avg_align_loss = accuracy = 0

        logger.info(f"Client {self.ID}: CNN Training - loss={avg_loss:.4f}, "
                   f"ce={avg_ce_loss:.4f}, align={avg_align_loss:.4f}, acc={accuracy:.4f}")

        # Get backbone parameters
        backbone_para = copy.deepcopy(self.cnn_backbone.state_dict())

        results = {
            'cnn_train_loss': avg_loss,
            'cnn_ce_loss': avg_ce_loss,
            'cnn_align_loss': avg_align_loss,
            'cnn_train_acc': accuracy,
            'cnn_train_total': total_samples
        }

        return total_samples, backbone_para, results

    def _set_fedproto_global_prototypes(self, prototypes):
        """Load global head-representation prototypes from the server."""
        if not self.use_fedproto:
            return

        normalized = {}
        if isinstance(prototypes, dict):
            for class_idx, prototype in prototypes.items():
                try:
                    class_idx = int(class_idx)
                except (TypeError, ValueError):
                    continue
                if prototype is None:
                    continue
                if isinstance(prototype, torch.Tensor):
                    tensor = prototype.detach().to(self.device).float()
                else:
                    try:
                        tensor = torch.as_tensor(
                            prototype,
                            device=self.device,
                            dtype=torch.float32)
                    except (TypeError, ValueError):
                        continue
                normalized[class_idx] = tensor

        self.fedproto_global_prototypes = normalized
        logger.info(
            f"Client {self.ID}: FedProto global prototypes updated "
            f"({len(normalized)} classes)")

    def _forward_head_with_representation(self, features):
        """Run the classifier and expose the representation used by FedProto.

        Text heads implement ``return_features`` directly.  DomainNet's
        reference heads are deliberately plain ``Linear``/``Sequential``
        modules, so asking those modules for ``return_features`` raises a
        ``TypeError``.  A linear head lives in the input-feature space, while
        a two-layer head uses the activated output of its first linear layer.
        """
        representation = None
        try:
            forward_res = self.mlp_classifier(
                features, return_features=True)
            if isinstance(forward_res, tuple) and len(forward_res) == 2:
                outputs, representation = forward_res
            else:
                outputs = forward_res
        except TypeError:
            outputs = self.mlp_classifier(features)
            if isinstance(self.mlp_classifier, nn.Linear):
                representation = features
            elif isinstance(self.mlp_classifier, nn.Sequential) and \
                    len(self.mlp_classifier) >= 1 and \
                    isinstance(self.mlp_classifier[0], nn.Linear):
                representation = self.mlp_classifier[0](features)
                if len(self.mlp_classifier) >= 2 and \
                        isinstance(self.mlp_classifier[1], nn.ReLU):
                    representation = self.mlp_classifier[1](representation)
        return outputs, representation

    def _compute_fedproto_loss(self, labels, representations):
        """Return a differentiable FedProto loss for every reference head.

        For a linear head the sample representation is the fixed cached
        feature, so aligning it directly would not affect training.  Preserve
        the original standalone behavior by aligning trainable class weights
        with the corresponding global prototypes.  Heads with a trainable
        hidden representation use the usual per-sample prototype objective.
        """
        loss = torch.tensor(0.0, device=self.device)
        if not self.use_fedproto or not self.fedproto_global_prototypes:
            return loss

        if isinstance(self.mlp_classifier, nn.Linear):
            class_indices = []
            prototypes = []
            weight = self.mlp_classifier.weight
            for class_idx, prototype in \
                    self.fedproto_global_prototypes.items():
                class_idx = int(class_idx)
                if class_idx < 0 or class_idx >= weight.shape[0]:
                    continue
                prototype = prototype.to(
                    device=weight.device, dtype=weight.dtype)
                if prototype.numel() != weight.shape[1]:
                    continue
                class_indices.append(class_idx)
                prototypes.append(prototype.reshape(-1))
            if not class_indices:
                return loss
            embeddings = weight[torch.as_tensor(
                class_indices, device=weight.device, dtype=torch.long)]
            targets = torch.stack(prototypes, dim=0)
        else:
            if representations is None:
                return loss
            targets = torch.zeros_like(representations)
            valid_mask = torch.zeros(
                labels.shape[0], device=self.device, dtype=torch.bool)
            for class_idx, prototype in \
                    self.fedproto_global_prototypes.items():
                class_mask = labels == int(class_idx)
                if not class_mask.any():
                    continue
                prototype = prototype.to(
                    device=representations.device,
                    dtype=representations.dtype)
                if prototype.numel() != representations.shape[1]:
                    continue
                targets[class_mask] = prototype.reshape(-1)
                valid_mask |= class_mask
            if not valid_mask.any():
                return loss
            embeddings = representations[valid_mask]
            targets = targets[valid_mask]

        if self.fedproto_normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
            targets = F.normalize(targets, p=2, dim=1)
        if self.fedproto_distance_metric in {'cos', 'cosine'}:
            return 1.0 - F.cosine_similarity(
                embeddings, targets, dim=1).mean()
        return (embeddings - targets).pow(2).sum(dim=1).mean()

    def _compute_fedproto_local_prototypes(self):
        """Compute per-class prototypes in the trainable head hidden space."""
        if not self.use_fedproto or self.augmented_loader is None or \
                self.mlp_classifier is None:
            return {}, {}

        was_training = self.mlp_classifier.training
        self.mlp_classifier.eval()
        prototype_sums = {}
        prototype_counts = {}

        with torch.no_grad():
            for features, labels in self.augmented_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                _, embeddings = self._forward_head_with_representation(
                    features)
                if embeddings is None:
                    if was_training:
                        self.mlp_classifier.train()
                    return {}, {}

                for class_idx in labels.unique():
                    class_idx = int(class_idx.item())
                    class_mask = labels == class_idx
                    count = int(class_mask.sum().item())
                    if count <= 0:
                        continue
                    embedding_sum = embeddings[class_mask].detach().sum(
                        dim=0).cpu()
                    if class_idx not in prototype_sums:
                        prototype_sums[class_idx] = embedding_sum
                        prototype_counts[class_idx] = count
                    else:
                        prototype_sums[class_idx] += embedding_sum
                        prototype_counts[class_idx] += count

        prototypes = {
            class_idx: (value /
                        float(prototype_counts[class_idx])).float()
            for class_idx, value in prototype_sums.items()
            if int(prototype_counts.get(class_idx, 0)) > 0
        }
        if was_training:
            self.mlp_classifier.train()
        return prototypes, prototype_counts

    def _train_on_augmented_data(self):
        """Train MLP classifier on augmented features with optional FedProto/FedProx regularization"""
        if self.augmented_loader is None or self.mlp_classifier is None:
            return 0, {}, {}

        self.mlp_classifier = self.mlp_classifier.to(self.device)
        self.mlp_classifier.train()
        optimizer = torch.optim.Adam(self.mlp_classifier.parameters(),
                                     lr=self._cfg.train.optimizer.lr)
        criterion = nn.CrossEntropyLoss()

        # FedProx settings (proximal term to the received global model)
        use_fedprox = getattr(self._cfg.fedprox, 'use', False) if hasattr(self._cfg, 'fedprox') else False
        fedprox_mu = getattr(self._cfg.fedprox, 'mu', 0.0) if hasattr(self._cfg, 'fedprox') else 0.0
        global_params_ref = None
        if use_fedprox and fedprox_mu > 0:
            # Snapshot current model params as the global reference before local updates
            global_params_ref = {
                k: v.detach().clone()
                for k, v in self.mlp_classifier.state_dict().items()
            }

        # MOON settings
        use_moon = getattr(self.ggeur_cfg, 'use_moon', False)
        moon_mu = getattr(self.ggeur_cfg, 'moon_mu', 5.0)
        moon_temperature = getattr(self.ggeur_cfg, 'moon_temperature', 0.5)
        if use_moon:
            if self.moon_global_model is not None:
                self.moon_global_model = self.moon_global_model.to(self.device)
            if self.moon_prev_model is not None:
                self.moon_prev_model = self.moon_prev_model.to(self.device)
            logger.info(f"Client {self.ID}: MOON enabled - mu={moon_mu}, temperature={moon_temperature}")

        use_fedproto = self.use_fedproto
        self.fedproto_local_prototypes = {}
        self.fedproto_local_counts = {}
        logger.info(
            f"Client {self.ID}: FedProto settings - "
            f"use_fedproto={use_fedproto}, "
            f"proto_weight={self.fedproto_proto_weight}, "
            f"proto_distance={self.fedproto_distance_metric}, "
            f"global_prototypes_count="
            f"{len(self.fedproto_global_prototypes)}")
        if use_fedprox and fedprox_mu > 0:
            logger.info(f"Client {self.ID}: FedProx enabled - mu={fedprox_mu}")

        total_loss = 0.0
        total_ce_loss = 0.0
        total_proto_loss = 0.0
        total_prox_loss = 0.0
        total_moon_loss = 0.0
        total_correct = 0
        total_samples = 0

        local_epochs = self._cfg.train.local_update_steps

        for epoch in range(local_epochs):
            for features, labels in self.augmented_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs, rep_features = \
                    self._forward_head_with_representation(features)
                ce_loss = criterion(outputs, labels)

                proto_loss = self._compute_fedproto_loss(
                    labels, rep_features)

                # Compute FedProx proximal loss on model parameters
                prox_loss = torch.tensor(0.0, device=self.device)
                if use_fedprox and fedprox_mu > 0 and global_params_ref is not None:
                    for name, param in self.mlp_classifier.named_parameters():
                        if not param.requires_grad:
                            continue
                        ref = global_params_ref.get(name, None)
                        if ref is None:
                            continue
                        if isinstance(ref, torch.Tensor) and ref.device != param.device:
                            ref = ref.to(param.device)
                        prox_loss = prox_loss + torch.sum((param - ref) ** 2)

                # Compute MOON contrastive loss
                # L_con = -log( exp(sim(z,z_global)/τ) / (exp(sim(z,z_global)/τ) + exp(sim(z,z_prev)/τ)) )
                moon_loss = torch.tensor(0.0, device=self.device)
                if use_moon and self.moon_global_model is not None:
                    with torch.no_grad():
                        z_global = self.moon_global_model(features)
                        if self.moon_prev_model is not None:
                            z_prev = self.moon_prev_model(features)
                        else:
                            # First round: no previous local model, use global as prev (loss = 0)
                            z_prev = z_global.clone()

                    z = outputs  # current model logits as representation
                    z_norm = F.normalize(z, dim=1)
                    z_global_norm = F.normalize(z_global, dim=1)
                    z_prev_norm = F.normalize(z_prev, dim=1)

                    sim_global = (z_norm * z_global_norm).sum(dim=1, keepdim=True) / moon_temperature
                    sim_prev = (z_norm * z_prev_norm).sum(dim=1, keepdim=True) / moon_temperature

                    logits_con = torch.cat([sim_global, sim_prev], dim=1)
                    labels_con = torch.zeros(features.size(0), dtype=torch.long, device=self.device)
                    moon_loss = criterion(logits_con, labels_con)

                # Total loss
                loss = ce_loss + self.fedproto_proto_weight * proto_loss
                if use_fedprox and fedprox_mu > 0:
                    loss = loss + (fedprox_mu / 2.0) * prox_loss
                if use_moon:
                    loss = loss + moon_mu * moon_loss
                loss.backward()
                optimizer.step()

                batch_size = features.size(0)
                total_loss += loss.item() * batch_size
                total_ce_loss += ce_loss.item() * batch_size
                total_proto_loss += proto_loss.item() * batch_size
                total_prox_loss += prox_loss.item() * batch_size
                total_moon_loss += moon_loss.item() * batch_size
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += batch_size

        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        avg_ce_loss = total_ce_loss / total_samples if total_samples > 0 else 0
        avg_proto_loss = total_proto_loss / total_samples if total_samples > 0 else 0
        avg_prox_loss = total_prox_loss / total_samples if total_samples > 0 else 0
        avg_moon_loss = total_moon_loss / total_samples if total_samples > 0 else 0
        accuracy = total_correct / total_samples if total_samples > 0 else 0

        if use_fedproto:
            logger.info(f"Client {self.ID}: Train loss={avg_loss:.4f} (CE={avg_ce_loss:.4f}, "
                       f"proto={avg_proto_loss:.4f}), accuracy={accuracy:.4f}")
        elif use_fedprox and fedprox_mu > 0:
            logger.info(f"Client {self.ID}: Train loss={avg_loss:.4f} (CE={avg_ce_loss:.4f}, "
                       f"prox={avg_prox_loss:.4f}), accuracy={accuracy:.4f}")
        elif use_moon:
            logger.info(f"Client {self.ID}: Train loss={avg_loss:.4f} (CE={avg_ce_loss:.4f}, "
                       f"moon={avg_moon_loss:.4f}), accuracy={accuracy:.4f}")
        else:
            logger.info(f"Client {self.ID}: Train loss={avg_loss:.4f}, accuracy={accuracy:.4f}")

        if use_fedproto:
            prototypes, counts = self._compute_fedproto_local_prototypes()
            self.fedproto_local_prototypes = prototypes
            self.fedproto_local_counts = counts

        # Get model parameters on CPU so sequential standalone clients do not
        # accumulate GPU-resident state dicts.
        model_para = {
            k: v.detach().cpu().clone()
            for k, v in self.mlp_classifier.state_dict().items()
        }

        self.mlp_classifier = self.mlp_classifier.cpu()
        if self.moon_global_model is not None:
            self.moon_global_model = self.moon_global_model.cpu()
        if self.moon_prev_model is not None:
            self.moon_prev_model = self.moon_prev_model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        results = {
            'train_loss': avg_loss,
            'train_ce_loss': avg_ce_loss,
            'train_proto_loss': avg_proto_loss,
            'train_prox_loss': avg_prox_loss,
            'train_moon_loss': avg_moon_loss,
            'train_acc': accuracy,
            'train_total': total_samples
        }

        return total_samples, model_para, results

    # ==================== CNN Knowledge Distillation Methods ====================

    def _build_cnn_model(self):
        """Build CNN model for knowledge distillation"""
        from federatedscope.contrib.model.ggeur_cnn import GGEUR_CNN_FeatureAlign

        num_classes = self._cfg.model.num_classes
        cnn_model_name = getattr(self.ggeur_cfg, 'cnn_model', 'resnet18')
        cnn_pretrained = getattr(self.ggeur_cfg, 'cnn_pretrained', True)
        clip_dim = getattr(self.ggeur_cfg, 'embedding_dim', 512)

        # Use the new FeatureAlign model for consistency
        self.cnn_model = GGEUR_CNN_FeatureAlign(
            model_name=cnn_model_name,
            num_classes=num_classes,
            clip_dim=clip_dim,
            pretrained=cnn_pretrained
        )
        self.cnn_model = self.cnn_model.to(self.device)

        logger.info(f"Client {self.ID}: Built CNN model ({cnn_model_name}) with {num_classes} classes, pretrained={cnn_pretrained}")

    def _build_cnn_feature_align(self):
        """Build CNN model for feature alignment (from scratch training)"""
        from federatedscope.contrib.model.ggeur_cnn import GGEUR_CNN_FeatureAlign

        num_classes = self._cfg.model.num_classes
        cnn_model_name = getattr(self.ggeur_cfg, 'cnn_model', 'resnet18')
        clip_dim = getattr(self.ggeur_cfg, 'embedding_dim', 512)

        # Feature alignment mode: always start from scratch (no pretrained weights)
        self.cnn_model = GGEUR_CNN_FeatureAlign(
            model_name=cnn_model_name,
            num_classes=num_classes,
            clip_dim=clip_dim,
            pretrained=False  # From scratch!
        )
        self.cnn_model = self.cnn_model.to(self.device)

        logger.info(f"Client {self.ID}: Built CNN ({cnn_model_name}) for feature alignment - FROM SCRATCH, {num_classes} classes")

    def _setup_original_image_loader(self):
        """Setup DataLoader for original images (for CNN training)"""
        # Get train data
        train_data = self.trainer.ctx.data.get('train', None)
        if train_data is None:
            train_data = self.data.get('train', None)

        if train_data is None:
            logger.error(f"Client {self.ID}: No training data available for CNN")
            return

        # Get the underlying dataset
        dataset = train_data.dataset if hasattr(train_data, 'dataset') else train_data

        # 为CNN训练创建带数据增强的包装数据集
        use_augmentation = getattr(self.ggeur_cfg, 'cnn_use_augmentation', True)
        if use_augmentation:
            dataset = AugmentedImageDataset(dataset)
            logger.info(f"Client {self.ID}: CNN training with data augmentation enabled")

        # Create DataLoader for original images
        batch_size = self._cfg.dataloader.batch_size
        self.original_image_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Use 0 to avoid multiprocessing issues
            drop_last=False
        )

        logger.info(f"Client {self.ID}: Setup original image loader with {len(dataset)} samples")

    def _train_cnn_with_distillation(self):
        """
        Train CNN with knowledge distillation from MLP teacher.

        The loss function is:
        L = alpha * CE(CNN(x), y) + (1-alpha) * T^2 * KL(softmax(CNN(x)/T), softmax(MLP(CLIP(x))/T))

        Returns:
            sample_size: Number of samples trained
            cnn_para: CNN model parameters
            results: Training results dict
        """
        if self.cnn_model is None or self.original_image_loader is None:
            logger.warning(f"Client {self.ID}: CNN or image loader not ready")
            return 0, {}, {}

        # Ensure CLIP model is loaded (may not be loaded if features were cached)
        if self.clip_model is None:
            logger.info(f"Client {self.ID}: Loading CLIP model for CNN distillation...")
            self._load_clip_model()

        if self.mlp_classifier is None or self.clip_model is None:
            logger.warning(f"Client {self.ID}: MLP teacher or CLIP not ready - "
                          f"MLP: {self.mlp_classifier is not None}, CLIP: {self.clip_model is not None}")
            return 0, {}, {}

        # Training settings
        temperature = getattr(self.ggeur_cfg, 'distill_temperature', 4.0)
        alpha = getattr(self.ggeur_cfg, 'distill_alpha', 0.5)
        cnn_lr = getattr(self.ggeur_cfg, 'cnn_lr', 0.01)
        cnn_epochs = getattr(self.ggeur_cfg, 'cnn_local_epochs', 5)

        self.cnn_model.train()
        self.mlp_classifier.eval()  # Teacher is frozen
        self.clip_model.eval()

        # Use differential learning rates: lower for backbone, higher for classifier
        backbone_params = []
        classifier_params = []
        for name, param in self.cnn_model.named_parameters():
            if 'fc' in name or 'classifier' in name:
                classifier_params.append(param)
            else:
                backbone_params.append(param)

        optimizer = torch.optim.SGD([
            {'params': backbone_params, 'lr': cnn_lr * 0.1},  # Lower LR for backbone
            {'params': classifier_params, 'lr': cnn_lr}       # Higher LR for classifier
        ], momentum=0.9, weight_decay=1e-4)

        ce_criterion = nn.CrossEntropyLoss()
        kl_criterion = nn.KLDivLoss(reduction='batchmean')

        total_loss = 0.0
        total_ce_loss = 0.0
        total_kl_loss = 0.0
        total_correct = 0
        total_samples = 0
        teacher_correct = 0  # Track teacher accuracy for debugging

        # Debug: check data loader
        num_batches = len(self.original_image_loader)
        logger.info(f"Client {self.ID}: CNN training with {num_batches} batches, "
                   f"backbone_lr={cnn_lr*0.1}, classifier_lr={cnn_lr}, epochs={cnn_epochs}")

        for epoch in range(cnn_epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0
            epoch_teacher_correct = 0

            for batch in self.original_image_loader:
                if len(batch) >= 2:
                    images, labels = batch[0], batch[1]
                else:
                    continue

                images = images.to(self.device)
                labels = labels.to(self.device)

                # Get teacher's soft labels
                with torch.no_grad():
                    # Extract CLIP features
                    clip_features = self.clip_model.encode_image(images)
                    # Get MLP teacher's logits
                    teacher_logits = self.mlp_classifier(clip_features.float())
                    # Track teacher accuracy for debugging
                    _, teacher_pred = torch.max(teacher_logits, 1)
                    epoch_teacher_correct += (teacher_pred == labels).sum().item()

                # Get student's (CNN) logits
                optimizer.zero_grad()
                student_logits = self.cnn_model(images)

                # Hard label loss (Cross-Entropy)
                ce_loss = ce_criterion(student_logits, labels)

                # Soft label loss (KL-Divergence with temperature)
                soft_student = F.log_softmax(student_logits / temperature, dim=1)
                soft_teacher = F.softmax(teacher_logits / temperature, dim=1)
                kl_loss = kl_criterion(soft_student, soft_teacher) * (temperature ** 2)

                # Combined loss
                loss = alpha * ce_loss + (1 - alpha) * kl_loss

                # Backward and optimize
                loss.backward()
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.cnn_model.parameters(), max_norm=1.0)
                optimizer.step()

                # Statistics
                batch_size = images.size(0)
                total_loss += loss.item() * batch_size
                total_ce_loss += ce_loss.item() * batch_size
                total_kl_loss += kl_loss.item() * batch_size

                _, predicted = torch.max(student_logits, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += batch_size
                teacher_correct += (teacher_pred == labels).sum().item()

                epoch_loss += loss.item() * batch_size
                epoch_correct += (predicted == labels).sum().item()
                epoch_samples += batch_size

            if epoch_samples > 0:
                epoch_acc = epoch_correct / epoch_samples
                epoch_teacher_acc = epoch_teacher_correct / epoch_samples
                logger.info(f"Client {self.ID}: CNN Epoch {epoch+1}/{cnn_epochs}, "
                            f"Loss={epoch_loss/epoch_samples:.4f}, "
                            f"CNN_Acc={epoch_acc:.4f}, Teacher_Acc={epoch_teacher_acc:.4f}")

        # Compute averages
        if total_samples > 0:
            avg_loss = total_loss / total_samples
            avg_ce_loss = total_ce_loss / total_samples
            avg_kl_loss = total_kl_loss / total_samples
            accuracy = total_correct / total_samples
            teacher_accuracy = teacher_correct / total_samples
        else:
            avg_loss = avg_ce_loss = avg_kl_loss = accuracy = teacher_accuracy = 0

        logger.info(f"Client {self.ID}: CNN Train - loss={avg_loss:.4f}, "
                   f"ce_loss={avg_ce_loss:.4f}, kl_loss={avg_kl_loss:.4f}, "
                   f"cnn_acc={accuracy:.4f}, teacher_acc={teacher_accuracy:.4f}")

        # Get CNN model parameters
        cnn_para = copy.deepcopy(self.cnn_model.state_dict())

        results = {
            'cnn_train_loss': avg_loss,
            'cnn_ce_loss': avg_ce_loss,
            'cnn_kl_loss': avg_kl_loss,
            'cnn_train_acc': accuracy,
            'cnn_teacher_acc': teacher_accuracy,
            'cnn_train_total': total_samples
        }

        return total_samples, cnn_para, results

    # ==================== CNN Feature Alignment Methods ====================

    def _train_cnn_with_feature_alignment(self):
        """
        Train CNN from scratch using feature alignment with CLIP.

        The loss function is:
        L = CE(CNN(x), y) + λ * MSE(proj(CNN_feat(x)), CLIP_feat(x))

        Optionally, also align with global prototypes:
        L += λ_proto * MSE(proj(CNN_feat(x)), prototype[y])

        Returns:
            sample_size: Number of samples trained
            cnn_para: CNN model parameters
            results: Training results dict
        """
        if self.cnn_model is None or self.original_image_loader is None:
            logger.warning(f"Client {self.ID}: CNN or image loader not ready")
            return 0, {}, {}

        # Ensure CLIP model is loaded
        if self.clip_model is None:
            logger.info(f"Client {self.ID}: Loading CLIP model for feature alignment...")
            self._load_clip_model()

        if self.clip_model is None:
            logger.warning(f"Client {self.ID}: CLIP model not available")
            return 0, {}, {}

        # Training settings
        cnn_lr = getattr(self.ggeur_cfg, 'cnn_lr', 0.01)
        cnn_epochs = getattr(self.ggeur_cfg, 'cnn_local_epochs', 10)
        align_weight = getattr(self.ggeur_cfg, 'align_weight', 1.0)
        use_prototype_align = getattr(self.ggeur_cfg, 'use_prototype_alignment', True)
        prototype_weight = getattr(self.ggeur_cfg, 'prototype_align_weight', 0.5)

        self.cnn_model.train()
        self.clip_model.eval()

        # Optimizer
        optimizer = torch.optim.SGD(
            self.cnn_model.parameters(),
            lr=cnn_lr,
            momentum=0.9,
            weight_decay=1e-4
        )

        # Loss functions
        ce_criterion = nn.CrossEntropyLoss()
        mse_criterion = nn.MSELoss()

        # Convert global prototypes to tensor
        prototype_tensor = None
        if use_prototype_align and self.global_prototypes:
            num_classes = self._cfg.model.num_classes
            clip_dim = getattr(self.ggeur_cfg, 'embedding_dim', 512)
            prototype_tensor = torch.zeros(num_classes, clip_dim).to(self.device)
            for class_idx, proto in self.global_prototypes.items():
                class_idx = int(class_idx)
                if class_idx < num_classes:
                    if isinstance(proto, np.ndarray):
                        prototype_tensor[class_idx] = torch.from_numpy(proto).float()
                    else:
                        prototype_tensor[class_idx] = proto.float()

        total_loss = 0.0
        total_ce_loss = 0.0
        total_align_loss = 0.0
        total_proto_loss = 0.0
        total_correct = 0
        total_samples = 0

        num_batches = len(self.original_image_loader)
        logger.info(f"Client {self.ID}: Feature alignment training with {num_batches} batches, "
                   f"lr={cnn_lr}, epochs={cnn_epochs}, align_weight={align_weight}")

        for epoch in range(cnn_epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0

            for batch in self.original_image_loader:
                if len(batch) >= 2:
                    images, labels = batch[0], batch[1]
                else:
                    continue

                images = images.to(self.device)
                labels = labels.to(self.device)

                # Get CLIP features (target for alignment)
                with torch.no_grad():
                    clip_features = self.clip_model.encode_image(images).float()
                    clip_features_norm = F.normalize(clip_features, p=2, dim=1)

                # Forward pass through CNN
                optimizer.zero_grad()
                logits, proj_features = self.cnn_model(images, return_features=True)
                proj_features_norm = F.normalize(proj_features, p=2, dim=1)

                # Classification loss
                ce_loss = ce_criterion(logits, labels)

                # Feature alignment loss (align CNN features with CLIP features)
                align_loss = mse_criterion(proj_features_norm, clip_features_norm)

                # Prototype alignment loss (optional)
                proto_loss = torch.tensor(0.0).to(self.device)
                if use_prototype_align and prototype_tensor is not None:
                    # Get prototypes for each sample's class
                    target_prototypes = prototype_tensor[labels]
                    target_prototypes_norm = F.normalize(target_prototypes, p=2, dim=1)
                    proto_loss = mse_criterion(proj_features_norm, target_prototypes_norm)

                # Combined loss
                loss = ce_loss + align_weight * align_loss
                if use_prototype_align:
                    loss = loss + prototype_weight * proto_loss

                # Backward and optimize
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.cnn_model.parameters(), max_norm=1.0)
                optimizer.step()

                # Statistics
                batch_size = images.size(0)
                total_loss += loss.item() * batch_size
                total_ce_loss += ce_loss.item() * batch_size
                total_align_loss += align_loss.item() * batch_size
                total_proto_loss += proto_loss.item() * batch_size

                _, predicted = torch.max(logits, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += batch_size

                epoch_loss += loss.item() * batch_size
                epoch_correct += (predicted == labels).sum().item()
                epoch_samples += batch_size

            if epoch_samples > 0:
                epoch_acc = epoch_correct / epoch_samples
                logger.info(f"Client {self.ID}: CNN Epoch {epoch+1}/{cnn_epochs}, "
                            f"Loss={epoch_loss/epoch_samples:.4f}, Acc={epoch_acc:.4f}")

        # Compute averages
        if total_samples > 0:
            avg_loss = total_loss / total_samples
            avg_ce_loss = total_ce_loss / total_samples
            avg_align_loss = total_align_loss / total_samples
            avg_proto_loss = total_proto_loss / total_samples
            accuracy = total_correct / total_samples
        else:
            avg_loss = avg_ce_loss = avg_align_loss = avg_proto_loss = accuracy = 0

        logger.info(f"Client {self.ID}: CNN Feature Align - loss={avg_loss:.4f}, "
                   f"ce={avg_ce_loss:.4f}, align={avg_align_loss:.4f}, "
                   f"proto={avg_proto_loss:.4f}, acc={accuracy:.4f}")

        # Get CNN model parameters
        cnn_para = copy.deepcopy(self.cnn_model.state_dict())

        results = {
            'cnn_train_loss': avg_loss,
            'cnn_ce_loss': avg_ce_loss,
            'cnn_align_loss': avg_align_loss,
            'cnn_proto_loss': avg_proto_loss,
            'cnn_train_acc': accuracy,
            'cnn_train_total': total_samples
        }

        return total_samples, cnn_para, results

    # ==================== PromptFL Methods ====================

    def _get_class_names(self):
        """Get class names from dataset or config, falling back to generic names."""
        # 1. Check config override
        cfg_names = getattr(self.ggeur_cfg, 'prompt_class_names', [])
        if cfg_names:
            return list(cfg_names)

        # 2. Try to infer from dataset type
        data_type = self._cfg.data.type.lower()
        try:
            if 'office' in data_type and 'home' in data_type:
                from federatedscope.cv.dataset.office_home import OfficeHome
                return list(OfficeHome.CLASSES)
            elif 'pacs' in data_type:
                from federatedscope.cv.dataset.pacs import PACS
                return list(PACS.CLASSES)
            elif 'office' in data_type and 'caltech' in data_type:
                from federatedscope.cv.dataset.office_caltech import OfficeCaltech10
                return list(OfficeCaltech10.CLASSES)
        except Exception as e:
            logger.warning(f"Client {self.ID}: Could not load class names from dataset: {e}")

        # 3. Fallback: generic names
        num_classes = self._cfg.model.num_classes
        logger.warning(f"Client {self.ID}: Using generic class names for {num_classes} classes")
        return [f"class {i}" for i in range(num_classes)]

    def _build_prompt_model(self):
        """Build PromptFL components. CLIP backbone is shared across all clients
        (class-level singleton) to avoid loading N copies into GPU memory."""
        import open_clip
        from federatedscope.contrib.model.ggeur_prompt import PromptLearner, TextEncoder

        n_ctx = getattr(self.ggeur_cfg, 'prompt_length', 16)
        class_names = self._get_class_names()
        clip_model_name = getattr(self.ggeur_cfg, 'clip_model', 'ViT-B-16')
        model_path = getattr(self.ggeur_cfg, 'clip_model_path', '')
        clip_pretrained = getattr(self.ggeur_cfg, 'clip_pretrained', 'openai')
        template = getattr(self.ggeur_cfg, 'prompt_template', 'a photo of a {}')

        # Load shared CLIP once (CPU), reused by all clients
        if GGEURClient._shared_prompt_clip is None or GGEURClient._shared_prompt_clip_name != clip_model_name:
            try:
                if model_path and os.path.isfile(model_path):
                    logger.info(f"Client {self.ID}: Loading shared prompt CLIP from {model_path}")
                    clip_model, _, _ = open_clip.create_model_and_transforms(
                        clip_model_name, pretrained=model_path
                    )
                else:
                    clip_model, _, _ = open_clip.create_model_and_transforms(
                        clip_model_name, pretrained=clip_pretrained
                    )
                for p in clip_model.parameters():
                    p.requires_grad = False
                clip_model.eval()
                GGEURClient._shared_prompt_clip = clip_model.cpu()
                GGEURClient._shared_prompt_tokenizer = open_clip.get_tokenizer(clip_model_name)
                GGEURClient._shared_prompt_clip_name = clip_model_name
                logger.info(f"Shared prompt CLIP loaded (CPU), will be moved to GPU only during forward pass")
            except Exception as e:
                logger.error(f"Client {self.ID}: Failed to load shared CLIP: {e}")
                return

        # Each client only stores its own ctx (16x512 = 32KB) and token buffers
        # PromptLearner is built on CPU using the shared clip_model
        try:
            self.prompt_learner = PromptLearner(
                clip_model=GGEURClient._shared_prompt_clip,
                tokenizer=GGEURClient._shared_prompt_tokenizer,
                classnames=class_names,
                n_ctx=n_ctx,
                template=template,
                device=torch.device('cpu'),  # buffers on CPU, moved to GPU during forward
            )
            self.text_encoder = TextEncoder(device=self.device)
            # Move only the small trainable ctx to GPU; buffers stay CPU until needed
            self.prompt_learner.ctx.data = self.prompt_learner.ctx.data.to(self.device)
        except Exception as e:
            logger.error(f"Client {self.ID}: Failed to build PromptLearner: {e}")
            return

        logger.info(f"Client {self.ID}: PromptLearner ready ({n_ctx} ctx tokens, "
                    f"{len(class_names)} classes, shared CLIP backbone)")

    def _build_prompt_loader(self):
        """
        Build a prompt-specific data loader using:
        1. Real local CLIP features (not Gaussian samples) for classes this client has
        2. Gaussian samples around OTHER clients' prototype means for missing classes
           (anchored to real feature statistics, provides cross-domain coverage)
        """
        all_features = []
        all_labels = []

        requested_n_per_proto = getattr(self.ggeur_cfg, 'prompt_samples_per_proto', 20)
        local_classes = set()
        local_class_sizes = []
        n_local_samples = 0

        for class_idx, feats in self.real_local_features.items():
            if feats.shape[0] <= 0:
                continue
            local_classes.add(int(class_idx))
            local_class_sizes.append(int(feats.shape[0]))

        avg_local_per_class = int(round(np.mean(local_class_sizes))) if local_class_sizes else 1
        effective_n_per_proto = max(1, min(int(requested_n_per_proto), avg_local_per_class))
        target_local_per_class = max(1, effective_n_per_proto)

        # 1. Real local features (upsample lightly for class balance)
        for class_idx, feats in self.real_local_features.items():
            if feats.shape[0] <= 0:
                continue

            balanced_feats = feats
            if feats.shape[0] < target_local_per_class:
                repeat_idx = np.random.choice(feats.shape[0],
                                              target_local_per_class - feats.shape[0],
                                              replace=True)
                balanced_feats = np.concatenate([feats, feats[repeat_idx]], axis=0)

            all_features.append(balanced_feats)
            all_labels.append(np.full(balanced_feats.shape[0], int(class_idx)))
            n_local_samples += int(balanced_feats.shape[0])

        # 2. Gaussian samples around other clients' prototype means for missing classes
        n_proto_samples = 0
        if self.other_prototypes:
            for class_idx, prototypes in self.other_prototypes.items():
                if int(class_idx) not in local_classes and len(prototypes) > 0:
                    proto_mean = np.mean(np.stack(prototypes), axis=0)
                    cov = self.global_cov_matrices.get(
                        class_idx,
                        self.global_cov_matrices.get(int(class_idx), None)
                    )
                    if cov is None:
                        cov = np.eye(self.embedding_dim) * 0.01
                    generated = self._generate_samples(proto_mean, cov, effective_n_per_proto)
                    all_features.append(generated)
                    all_labels.append(np.full(effective_n_per_proto, int(class_idx)))
                    n_proto_samples += int(effective_n_per_proto)

        if not all_features:
            logger.warning(f"Client {self.ID}: No data for prompt_loader, falling back to augmented_loader")
            self.prompt_loader = self.augmented_loader
            return

        features = np.vstack(all_features)
        labels = np.concatenate(all_labels)

        dataset = AugmentedFeatureDataset(features, labels)
        self.prompt_loader = DataLoader(
            dataset,
            batch_size=self._cfg.dataloader.batch_size,
            shuffle=True
        )

        logger.info(f"Client {self.ID}: prompt_loader built - "
                    f"{features.shape[0]} samples, {len(np.unique(labels))} classes "
                    f"({n_local_samples} balanced local, {n_proto_samples} proto-augmented, "
                    f"requested_proto={requested_n_per_proto}, effective_proto={effective_n_per_proto})")

    def _train_prompt_on_augmented_data(self):
        """
        Train soft prompt ctx vectors on augmented CLIP features.
        The shared CLIP backbone is moved to GPU only for this client's turn,
        then moved back to CPU to free VRAM for the next client.
        """
        if self.prompt_learner is None or self.prompt_loader is None:
            return 0, {}, {}

        lr = getattr(self.ggeur_cfg, 'prompt_lr', 0.002)
        local_epochs = getattr(self.ggeur_cfg, 'prompt_local_epochs', 1)
        max_train_batches = int(
            getattr(self.ggeur_cfg, 'prompt_max_train_batches', 0))

        # Move shared CLIP to GPU for this client's forward pass
        clip_model = GGEURClient._shared_prompt_clip.to(self.device)
        clip_model.eval()

        # Move prompt buffers to GPU
        self.prompt_learner.token_prefix = self.prompt_learner.token_prefix.to(self.device)
        self.prompt_learner.token_suffix = self.prompt_learner.token_suffix.to(self.device)
        self.prompt_learner.tokenized_prompts = self.prompt_learner.tokenized_prompts.to(self.device)
        self.prompt_learner.train()

        optimizer = torch.optim.Adam([self.prompt_learner.ctx], lr=lr)

        # Use configured temperature instead of CLIP's logit_scale (~100).
        # logit_scale ≈ 100 amplifies gradients 100x, causing client drift.
        temperature = getattr(self.ggeur_cfg, 'prompt_temperature', 0.07)
        logit_scale = 1.0 / temperature

        # FedProx proximal term: keeps local ctx close to global ctx
        prompt_mu = getattr(self.ggeur_cfg, 'prompt_proximal_mu', 0.0)
        global_ctx = self.global_prompt_ctx  # may be None in round 0

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for epoch in range(local_epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0

            for batch_idx, (features, labels) in enumerate(self.prompt_loader):
                if max_train_batches > 0 and batch_idx >= max_train_batches:
                    break

                features = features.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()

                prompt_embeds, eot_pos = self.prompt_learner()
                text_feats = self.text_encoder(prompt_embeds, eot_pos, clip_model)
                text_feats_norm = F.normalize(text_feats, p=2, dim=1)

                img_feats = F.normalize(features, p=2, dim=1)
                logits = logit_scale * (img_feats @ text_feats_norm.T)
                loss = F.cross_entropy(logits, labels)

                if prompt_mu > 0.0 and global_ctx is not None:
                    proximal_loss = (prompt_mu / 2.0) * ((self.prompt_learner.ctx - global_ctx) ** 2).sum()
                    loss = loss + proximal_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_([self.prompt_learner.ctx], max_norm=1.0)
                optimizer.step()

                with torch.no_grad():
                    _, predicted = torch.max(logits, 1)
                    epoch_correct += (predicted == labels).sum().item()
                    epoch_samples += features.size(0)
                    epoch_loss += loss.item() * features.size(0)

            total_loss += epoch_loss
            total_correct += epoch_correct
            total_samples += epoch_samples

        # Move shared CLIP back to CPU and free GPU memory
        GGEURClient._shared_prompt_clip = clip_model.cpu()
        # Move prompt buffers back to CPU
        self.prompt_learner.token_prefix = self.prompt_learner.token_prefix.cpu()
        self.prompt_learner.token_suffix = self.prompt_learner.token_suffix.cpu()
        self.prompt_learner.tokenized_prompts = self.prompt_learner.tokenized_prompts.cpu()
        torch.cuda.empty_cache()

        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        accuracy = total_correct / total_samples if total_samples > 0 else 0

        logger.info(f"Client {self.ID}: Prompt training - loss={avg_loss:.4f}, acc={accuracy:.4f}, "
                    f"samples={total_samples}")

        prompt_para = {
            'ctx': copy.deepcopy(self.prompt_learner.ctx.data.cpu()),
            'sample_size': int(total_samples)
        }
        results = {'prompt_loss': avg_loss, 'prompt_acc': accuracy, 'prompt_total': total_samples}

        return total_samples, prompt_para, results


def call_ggeur_worker(method):
    """Factory function for GGEUR_Clip worker"""
    if method.lower() == 'ggeur':
        from federatedscope.contrib.worker.ggeur_server import GGEURServer
        return {
            'client': GGEURClient,
            'server': GGEURServer
        }
    return None


register_worker('ggeur', call_ggeur_worker)
