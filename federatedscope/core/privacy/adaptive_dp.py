"""Client-local adaptive clipping for standalone GGEUR updates.

Each logical client owns its own ``LocalAdaptiveClipper`` instance. This keeps
client norm histories isolated even though standalone simulation creates all
clients in the same Python process.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import torch


def _cfg_get(node, name, default=None):
    if node is None:
        return default
    if isinstance(node, Mapping):
        return node.get(name, default)
    return getattr(node, name, default)


def _is_float_tensor(value: Any) -> bool:
    return torch.is_tensor(value) and torch.is_floating_point(value)


def _iter_float_tensors(payload):
    if _is_float_tensor(payload):
        yield payload
    elif isinstance(payload, Mapping):
        for value in payload.values():
            yield from _iter_float_tensors(value)
    elif isinstance(payload, (list, tuple)):
        for value in payload:
            yield from _iter_float_tensors(value)


def compute_update_norm(update) -> float:
    total_sq = 0.0
    has_tensor = False
    for tensor in _iter_float_tensors(update):
        value = tensor.detach().float().cpu()
        total_sq += float(torch.sum(value * value).item())
        has_tensor = True
    return float(total_sq ** 0.5) if has_tensor else 0.0


def scale_update(update, scale: float):
    if _is_float_tensor(update):
        return update.detach().cpu().float().clone() * float(scale)
    if torch.is_tensor(update):
        return update.detach().cpu().clone()
    if isinstance(update, dict):
        return {key: scale_update(value, scale)
                for key, value in update.items()}
    if isinstance(update, list):
        return [scale_update(value, scale) for value in update]
    if isinstance(update, tuple):
        return tuple(scale_update(value, scale) for value in update)
    return copy.deepcopy(update)


def add_gaussian_noise(update, clip_bound: float, noise_multiplier: float,
                       generator: Optional[torch.Generator] = None):
    noise_std = float(noise_multiplier) * float(clip_bound)
    if _is_float_tensor(update):
        value = update.detach().cpu().float().clone()
        if noise_std <= 0:
            return value
        noise = torch.randn(value.shape, generator=generator,
                            dtype=value.dtype, device='cpu')
        return value + noise * noise_std
    if torch.is_tensor(update):
        return update.detach().cpu().clone()
    if isinstance(update, dict):
        return {
            key: add_gaussian_noise(value, clip_bound, noise_multiplier,
                                    generator)
            for key, value in update.items()
        }
    if isinstance(update, list):
        return [add_gaussian_noise(value, clip_bound, noise_multiplier,
                                   generator) for value in update]
    if isinstance(update, tuple):
        return tuple(add_gaussian_noise(value, clip_bound, noise_multiplier,
                                        generator) for value in update)
    return copy.deepcopy(update)


def sanitize_update(update, clip_bound: float, noise_multiplier: float,
                    eps: float = 1e-12,
                    generator: Optional[torch.Generator] = None):
    raw_norm = compute_update_norm(update)
    factor = min(1.0, float(clip_bound) / (raw_norm + float(eps))) \
        if raw_norm > 0 else 1.0
    clipped = scale_update(update, factor)
    protected = add_gaussian_noise(clipped, clip_bound, noise_multiplier,
                                   generator)
    noise_std = float(noise_multiplier) * float(clip_bound)
    return protected, {
        'raw_norm': float(raw_norm),
        'clip_bound': float(clip_bound),
        'clip_factor': float(factor),
        'clipped': bool(factor < 1.0),
        'noise_multiplier': float(noise_multiplier),
        'noise_std': float(noise_std),
        'noise_variance': float(noise_std ** 2),
        'sanitized_norm': float(compute_update_norm(protected)),
    }


def _is_standard_ggeur_client_update_dp_enabled(cfg) -> bool:
    dp_cfg = _cfg_get(cfg, 'dp', None)
    clipping = _cfg_get(dp_cfg, 'clipping', None)
    return bool(_cfg_get(dp_cfg, 'enabled', False)) and \
        str(_cfg_get(dp_cfg, 'level', '')).lower() == 'client_update' and \
        bool(_cfg_get(dp_cfg, 'protect_ggeur_update', False)) and \
        str(_cfg_get(clipping, 'type', 'fixed')).lower() in ['fixed', 'adaptive']


def is_ggeur_client_update_dp_enabled(cfg) -> bool:
    return _is_standard_ggeur_client_update_dp_enabled(cfg) or bool(
        _cfg_get(_cfg_get(cfg, 'adaptive_dp', None), 'use', False))


def get_ggeur_client_update_dp_cfg(cfg):
    if _is_standard_ggeur_client_update_dp_enabled(cfg):
        return _cfg_get(cfg, 'dp', None)

    # Backward-compatible adapter for datapoison's ``adaptive_dp`` namespace.
    legacy = _cfg_get(cfg, 'adaptive_dp', None)
    if not bool(_cfg_get(legacy, 'use', False)):
        return None
    clipping = SimpleNamespace(
        type='adaptive',
        initial_clip=float(_cfg_get(legacy, 'initial_clip', 1.0)),
        target_quantile=float(_cfg_get(legacy, 'target_quantile', 0.7)),
        ema=float(_cfg_get(legacy, 'ema', 0.9)),
        min_clip=float(_cfg_get(legacy, 'min_clip', 0.05)),
        max_clip=float(_cfg_get(legacy, 'max_clip', 10.0)),
    )
    return SimpleNamespace(
        enabled=True,
        level='client_update',
        mechanism='gaussian',
        baseline='adaptive',
        accountant='empirical',
        epsilon=1.0,
        delta=1e-5,
        noise_multiplier=float(_cfg_get(legacy, 'noise_multiplier', 1.0)),
        accountant_sample_rate=-1.0,
        max_grad_norm=clipping.initial_clip,
        protect_ggeur_update=True,
        private_clip_update=True,
        upload_private_stats=False,
        log_private_stats=True,
        eps=float(_cfg_get(legacy, 'eps', 1e-12)),
        seed=int(_cfg_get(legacy, 'seed', 0)),
        clipping=clipping,
    )


@dataclass
class LocalAdaptiveClipper:
    """Adaptive controller whose raw norm history never leaves the client."""

    initial_clip: float
    target_quantile: float
    ema: float
    min_clip: float
    max_clip: float
    noise_multiplier: float
    eps: float = 1e-12
    seed: int = 0
    current_clip: float = field(init=False)
    norm_history: list = field(default_factory=list)

    def __post_init__(self):
        self.current_clip = float(self.initial_clip)

    @classmethod
    def from_cfg(cls, dp_cfg):
        clipping = _cfg_get(dp_cfg, 'clipping', None)
        initial = float(_cfg_get(
            clipping, 'initial_clip', _cfg_get(dp_cfg, 'max_grad_norm', 1.0)))
        return cls(
            initial_clip=initial,
            target_quantile=float(_cfg_get(
                clipping, 'target_quantile', 0.7)),
            ema=float(_cfg_get(clipping, 'ema', 0.9)),
            min_clip=float(_cfg_get(clipping, 'min_clip', 0.05)),
            max_clip=float(_cfg_get(clipping, 'max_clip', 10.0)),
            noise_multiplier=float(_cfg_get(dp_cfg, 'noise_multiplier', 0.0)),
            eps=float(_cfg_get(dp_cfg, 'eps', 1e-12)),
            seed=int(_cfg_get(dp_cfg, 'seed', 0)),
        )

    def sanitize(self, update, round_idx: int, client_id: int):
        clip_used = float(self.current_clip)
        generator = torch.Generator(device='cpu')
        generator.manual_seed(
            int(self.seed) + int(round_idx) * 100000 + int(client_id))
        protected, stats = sanitize_update(
            update, clip_bound=clip_used,
            noise_multiplier=self.noise_multiplier, eps=self.eps,
            generator=generator)

        self.norm_history.append(float(stats['raw_norm']))
        quantile = float(np.quantile(
            np.asarray(self.norm_history, dtype=np.float64),
            self.target_quantile))
        next_clip = self.ema * clip_used + (1.0 - self.ema) * quantile
        self.current_clip = float(np.clip(
            next_clip, self.min_clip, self.max_clip))
        stats.update({
            'enabled': True,
            'mechanism': 'standalone_local_adaptive_client_update_dp',
            'round': int(round_idx),
            'client_id': int(client_id),
            'target_quantile': float(self.target_quantile),
            'ema': float(self.ema),
            'next_clip_bound': float(self.current_clip),
            'history_size': int(len(self.norm_history)),
        })
        return protected, stats


def subtract_states(local_state: Dict, global_state: Dict) -> Dict:
    delta = {}
    for key, local_value in local_state.items():
        global_value = global_state.get(key)
        if torch.is_tensor(local_value) and torch.is_tensor(global_value):
            if torch.is_floating_point(local_value):
                delta[key] = local_value.detach().cpu().float() - \
                    global_value.detach().cpu().float()
            else:
                delta[key] = local_value.detach().cpu().clone()
    return delta


def add_delta(global_state: Dict, delta: Dict) -> Dict:
    defended = {}
    for key, global_value in global_state.items():
        if key in delta and torch.is_tensor(global_value) and \
                torch.is_tensor(delta[key]) and \
                torch.is_floating_point(global_value):
            defended[key] = global_value.detach().cpu().float() + \
                delta[key].detach().cpu().float()
        elif torch.is_tensor(global_value):
            defended[key] = global_value.detach().cpu().clone()
        else:
            defended[key] = copy.deepcopy(global_value)
    return defended
