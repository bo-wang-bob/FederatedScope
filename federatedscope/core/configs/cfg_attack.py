from federatedscope.core.configs.config import CN
from federatedscope.register import register_config


def extend_attack_cfg(cfg):

    # ---------------------------------------------------------------------- #
    # attack
    # ---------------------------------------------------------------------- #
    cfg.attack = CN()
    cfg.attack.attack_method = ''
    # Compatibility fields used by the original editable GGEUR modular
    # attack configs. Distributed PPA uses distributed_ppa below.
    cfg.attack.use_ggeur = False
    cfg.attack.modular_attacks = False
    cfg.attack.attack_plugins = []
    # for gan_attack
    cfg.attack.target_label_ind = -1
    cfg.attack.attacker_id = -1

    # for backdoor attack

    cfg.attack.edge_path = 'edge_data/'
    cfg.attack.trigger_path = 'trigger/'
    cfg.attack.setting = 'fix'
    cfg.attack.freq = 10
    cfg.attack.insert_round = 100000
    cfg.attack.mean = [0.9637]
    cfg.attack.std = [0.1592]
    cfg.attack.trigger_type = 'edge'
    cfg.attack.label_type = 'dirty'
    # dirty, clean_label, dirty-label attack is all2one attack.
    cfg.attack.edge_num = 100
    cfg.attack.poison_ratio = 0.5
    cfg.attack.scale_poisoning = False
    cfg.attack.scale_para = 1.0
    cfg.attack.pgd_poisoning = False
    cfg.attack.pgd_lr = 0.1
    cfg.attack.pgd_eps = 2
    cfg.attack.self_opt = False
    cfg.attack.self_lr = 0.05
    cfg.attack.self_epoch = 6
    # Note: the mean and std should be the list type.

    # for reconstruct_opt
    cfg.attack.reconstruct_lr = 0.01
    cfg.attack.reconstruct_optim = 'Adam'
    cfg.attack.info_diff_type = 'l2'
    cfg.attack.max_ite = 400
    cfg.attack.alpha_TV = 0.001
    # for GRNN gradient reconstruction attack
    cfg.attack.grnn_g_in = 128
    cfg.attack.grnn_tv_weight = 1e-6
    cfg.attack.grnn_use_wd = False

    # for active PIA attack
    cfg.attack.alpha_prop_loss = 0

    # for passive PIA attack
    cfg.attack.classifier_PIA = 'randomforest'

    # for gradient Ascent --- MIA attack
    cfg.attack.inject_round = 0
    cfg.attack.mia_simulate_in_round = 20
    cfg.attack.mia_is_simulate_in = False

    # Distributed FedMIA benchmark instrumentation for GGEUR. When enabled,
    # each real client process evaluates a fixed member/non-member probe set
    # against its local MLP head and uploads only derived loss/cosine scores.
    # Raw images and feature vectors are not included in the report.
    cfg.attack.distributed_fedmia = False
    cfg.attack.fedmia_target_client_id = 1
    cfg.attack.fedmia_variant = "FedMIA-I"
    cfg.attack.fedmia_probe_size = 32
    cfg.attack.fedmia_nonmember_size = 32
    cfg.attack.fedmia_save_interval = 5
    cfg.attack.fedmia_store_grad_cos = True
    cfg.attack.fedmia_round_agg = 'mean'
    cfg.attack.fedmia_shadow_stat_mode = 'indexed'
    cfg.attack.fedmia_var_floor = 1e-8
    cfg.attack.fedmia_compute_all_clients = False
    # Fields aligned with the standalone GGEUR FedMIA experiment.
    cfg.attack.mode = 'mix'
    cfg.attack.mix_length = 1000
    cfg.attack.fedmia_i_round_agg = 'mean'
    cfg.attack.ggeur_target_size = 200
    cfg.attack.ggeur_normalize_fields = ['train_losses']
    cfg.attack.fedmia_use_augmented_nonmember = False
    cfg.attack.fedmia_augmented_nonmember_size = 0
    cfg.attack.fedmia_image_aug_member_size = 0
    cfg.attack.ggeur_num_per_sample = 50
    cfg.attack.ggeur_use_cross_client = True
    cfg.attack.fedmia_use_image_aug_member = False
    cfg.attack.fedmia_cross_eval = False

    # Distributed GGEUR Meta-PPA. Each client computes
    # global_sensitivity-local_sensitivity on local class probes and uploads
    # only the derived feature vector plus its majority-class label.
    cfg.attack.distributed_ppa = False
    cfg.attack.meta_ppa_save_interval = 10
    cfg.attack.meta_ppa_probe_samples_per_class = 16
    cfg.attack.meta_ppa_probe_epochs = 3
    cfg.attack.meta_ppa_probe_lr = 0.001
    cfg.attack.meta_ppa_probe_batch_size = 16
    cfg.attack.meta_ppa_max_attack_rounds = 10
    cfg.attack.meta_ppa_max_clients = 0
    cfg.attack.meta_ppa_target_layers = []

    # --------------- register corresponding check function ----------
    cfg.register_cfg_check_fun(assert_attack_cfg)


def assert_attack_cfg(cfg):
    if cfg.attack.distributed_fedmia:
        assert cfg.attack.fedmia_target_client_id > 0
        assert cfg.attack.fedmia_probe_size > 0
        assert cfg.attack.fedmia_nonmember_size >= 0
        assert cfg.attack.fedmia_save_interval > 0
        assert cfg.attack.mix_length > 0
        assert cfg.attack.ggeur_target_size > 0
        assert cfg.attack.fedmia_round_agg in ['mean', 'min', 'max', 'last']
        assert cfg.attack.fedmia_shadow_stat_mode in ['global', 'indexed']
        assert cfg.attack.fedmia_var_floor > 0
    if cfg.attack.distributed_ppa:
        assert cfg.attack.classifier_PIA in ['svm', 'randomforest', 'lr']
        assert cfg.attack.meta_ppa_save_interval > 0
        assert cfg.attack.meta_ppa_probe_samples_per_class > 0
        assert cfg.attack.meta_ppa_probe_epochs > 0
        assert cfg.attack.meta_ppa_probe_lr > 0
        assert cfg.attack.meta_ppa_probe_batch_size > 0
        assert cfg.attack.meta_ppa_max_attack_rounds >= 0
        assert cfg.attack.meta_ppa_max_clients >= 0


register_config("attack", extend_attack_cfg)
