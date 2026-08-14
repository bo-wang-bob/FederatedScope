"""Configuration and validation for the unified standalone security flow."""

from federatedscope.core.configs.config import CN
from federatedscope.register import register_config


BACKDOOR_METHODS = {
    'backdoor', 'a3fl', 'cerberus', 'sabre', 'label_flip',
    'label_flipping', 'data_poisoning', 'little_is_enough', 'lie', 'alie'
}
PRIVACY_METHODS = {
    'dlg', 'ig', 'grnn', 'passivepia', 'fedmia', 'ggeur_fedmia', 'ggeur_ppa'
}
PRIVACY_PLUGINS = {
    'blackbox_loss', 'grad_cosine', 'grad_diff', 'grad_norm',
    'loss_series', 'avg_cosine', 'fedmia_i', 'fedmia_ii',
    'meta_ppa', 'ppa', 'property_inference'
}


def _normalized_plugins(cfg):
    attack = getattr(cfg, 'attack', None)
    plugins = getattr(attack, 'attack_plugins', []) if attack is not None else []
    return {str(name).strip().lower() for name in plugins if str(name).strip()}


def resolve_security_mode(cfg):
    """Resolve ``auto`` while preserving legacy attack configurations."""
    configured = str(getattr(getattr(cfg, 'security', None), 'mode',
                             'auto')).strip().lower()
    if configured not in {'auto', 'none', 'backdoor', 'privacy'}:
        raise ValueError(
            "security.mode must be one of auto/none/backdoor/privacy, "
            f"got {configured!r}")
    if configured != 'auto':
        return configured

    attack_method = str(getattr(getattr(cfg, 'attack', None),
                                'attack_method', '')).strip().lower()
    plugins = _normalized_plugins(cfg)
    has_backdoor = attack_method in BACKDOOR_METHODS
    has_privacy = attack_method in PRIVACY_METHODS or bool(
        plugins & PRIVACY_PLUGINS)
    if has_backdoor and has_privacy:
        raise ValueError(
            "Backdoor and privacy attacks are mutually exclusive in one run")
    if has_backdoor:
        return 'backdoor'
    if has_privacy:
        return 'privacy'
    return 'none'


def extend_security_cfg(cfg):
    cfg.security = CN()
    # ``auto`` keeps old configs working. New configs should be explicit.
    cfg.security.mode = 'auto'
    cfg.security.standalone_only = True
    cfg.register_cfg_check_fun(assert_security_cfg)


def assert_security_cfg(cfg):
    mode = resolve_security_mode(cfg)
    # The mode contract orchestrates the merged GGEUR implementation only;
    # legacy non-GGEUR FederatedScope attacks keep their original behavior.
    if str(cfg.federate.method).lower() != 'ggeur':
        return
    attack = cfg.attack
    attack_method = str(attack.attack_method).strip().lower()
    plugins = _normalized_plugins(cfg)
    has_backdoor = attack_method in BACKDOOR_METHODS
    has_privacy = attack_method in PRIVACY_METHODS or bool(
        plugins & PRIVACY_PLUGINS)

    if bool(cfg.security.standalone_only) and mode != 'none' and \
            str(cfg.federate.mode).lower() != 'standalone':
        raise ValueError(
            "The unified GGEUR security flow only supports "
            "federate.mode=standalone")
    if mode == 'backdoor':
        if not has_backdoor:
            raise ValueError(
                "security.mode=backdoor requires a backdoor attack_method")
        if has_privacy or bool(getattr(attack, 'modular_attacks', False)) or \
                plugins:
            raise ValueError(
                "Privacy attacks/plugins must be disabled in backdoor mode")
    elif mode == 'privacy':
        if has_backdoor:
            raise ValueError(
                "Backdoor attacks must be disabled in privacy mode")
        defense_method = str(getattr(cfg.ggeur, 'defense_method', '')).lower()
        aggregator_method = str(getattr(
            getattr(cfg, 'aggregator', None), 'robust_rule', '')).lower()
        stats_defense = bool(getattr(
            cfg.ggeur, 'multi_metrics_stats_defense', False)) or bool(
                getattr(cfg.ggeur, 'multi_metrics_stats_enabled', False))
        multi_metrics_aliases = {
                'multi_metrics', 'multi-metrics', 'multimetrics',
                'multi_metric', 'multi_metrics_adaptive'}
        if defense_method in multi_metrics_aliases or \
                aggregator_method in multi_metrics_aliases or stats_defense:
            raise ValueError(
                "MultiMetric backdoor defenses must be disabled in privacy "
                "mode")
    elif mode == 'none' and (has_backdoor or has_privacy):
        raise ValueError(
            "security.mode=none cannot be combined with an active attack")


register_config('security', extend_security_cfg)
