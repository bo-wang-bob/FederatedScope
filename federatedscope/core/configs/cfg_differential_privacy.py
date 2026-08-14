from federatedscope.core.configs.config import CN
from federatedscope.register import register_config


def extend_dp_cfg(cfg):
    # ---------------------------------------------------------------------- #
    # nbafl(dp) related options
    # ---------------------------------------------------------------------- #
    cfg.nbafl = CN()

    # Params
    cfg.nbafl.use = False
    cfg.nbafl.mu = 0.
    cfg.nbafl.epsilon = 100.
    cfg.nbafl.w_clip = 1.
    cfg.nbafl.constant = 30.

    # ---------------------------------------------------------------------- #
    # VFL-SGDMF(dp) related options
    # ---------------------------------------------------------------------- #
    cfg.sgdmf = CN()

    cfg.sgdmf.use = False  # if use sgdmf algorithm
    cfg.sgdmf.R = 5.  # The upper bound of rating
    cfg.sgdmf.epsilon = 4.  # \epsilon in dp
    cfg.sgdmf.delta = 0.5  # \delta in dp
    cfg.sgdmf.constant = 1.  # constant

    # GGEUR client-update adaptive clipping and Gaussian perturbation.
    cfg.dp = CN()
    cfg.dp.enabled = False
    cfg.dp.level = ''
    cfg.dp.mechanism = 'gaussian'
    cfg.dp.baseline = ''
    cfg.dp.dpfl_noise_location = "update"
    cfg.dp.accountant = 'empirical'
    cfg.dp.epsilon = 1.0
    cfg.dp.delta = 1e-5
    cfg.dp.noise_multiplier = 0.0
    cfg.dp.accountant_sample_rate = -1.0
    cfg.dp.max_grad_norm = 1.0
    cfg.dp.clip_percentile = 75.0
    cfg.dp.param_level = True
    cfg.dp.protect_ggeur_update = False
    cfg.dp.private_clip_update = True
    cfg.dp.upload_private_stats = False
    cfg.dp.log_private_stats = True
    cfg.dp.eps = 1e-12
    cfg.dp.seed = 0
    cfg.dp.clipping = CN()
    cfg.dp.clipping.type = 'fixed'
    cfg.dp.clipping.initial_clip = 1.0
    cfg.dp.clipping.target_quantile = 0.7
    cfg.dp.clipping.ema = 0.9
    cfg.dp.clipping.min_clip = 0.05
    cfg.dp.clipping.max_clip = 10.0

    # --------------- register corresponding check function ----------
    cfg.register_cfg_check_fun(assert_dp_cfg)


def assert_dp_cfg(cfg):
    if not cfg.dp.enabled:
        return
    assert cfg.dp.level in ['client_update']
    assert cfg.dp.mechanism == 'gaussian'
    assert cfg.dp.noise_multiplier >= 0
    assert cfg.dp.clipping.type in ['fixed', 'adaptive']
    assert cfg.dp.clipping.initial_clip > 0
    assert 0.0 < cfg.dp.clipping.target_quantile <= 1.0
    assert 0.0 <= cfg.dp.clipping.ema < 1.0
    assert cfg.dp.clipping.min_clip > 0
    assert cfg.dp.clipping.max_clip >= cfg.dp.clipping.min_clip


register_config("dp", extend_dp_cfg)
