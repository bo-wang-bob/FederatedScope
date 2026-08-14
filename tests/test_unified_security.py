import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from federatedscope.contrib.worker.ggeur_client import GGEURClient
from federatedscope.contrib.worker.ggeur_fedmia_server import \
    GGEURFedMIAHook
from federatedscope.contrib.worker.ggeur_ppa_server import GGEURPPAHook
from federatedscope.contrib.worker.ggeur_server import GGEURServer
from federatedscope.core.auxiliaries.worker_builder import (
    get_client_cls, get_server_cls)
from federatedscope.core.configs.config import global_cfg
from federatedscope.core.configs.cfg_security import resolve_security_mode
from federatedscope.core.privacy.adaptive_dp import LocalAdaptiveClipper


CONFIG_DIR = Path(__file__).resolve().parents[1] / 'scripts' / \
    'standalone_configs'


def _base_cfg():
    cfg = global_cfg.clone()
    cfg.federate.method = 'ggeur'
    cfg.federate.mode = 'standalone'
    cfg.federate.client_num = 4
    cfg.federate.sample_client_num = 4
    cfg.federate.total_round_num = 2
    return cfg


class GGEURUnifiedSecurityTest(unittest.TestCase):

    def test_all_canonical_standalone_configs_are_valid(self):
        config_paths = sorted(CONFIG_DIR.glob('*.yaml'))
        self.assertEqual(len(config_paths), 10)
        for config_path in config_paths:
            with self.subTest(config=config_path.name):
                cfg = global_cfg.clone()
                cfg.merge_from_file(str(config_path))
                cfg.freeze(inform=False, save=False)
                self.assertEqual(cfg.federate.mode, 'standalone')

    def test_backdoor_profiles_disable_privacy_flow(self):
        for name in ['backdoor_attack.yaml',
                     'backdoor_attack_defended.yaml']:
            with self.subTest(config=name):
                cfg = global_cfg.clone()
                cfg.merge_from_file(str(CONFIG_DIR / name))
                cfg.freeze(inform=False, save=False)
                self.assertEqual(resolve_security_mode(cfg), 'backdoor')
                self.assertFalse(cfg.dp.enabled)
                self.assertFalse(cfg.attack.modular_attacks)
                self.assertEqual(list(cfg.attack.attack_plugins), [])

        defended = global_cfg.clone()
        defended.merge_from_file(str(
            CONFIG_DIR / 'backdoor_attack_defended.yaml'))
        defended.freeze(inform=False, save=False)
        self.assertTrue(defended.ggeur.multi_metrics_stats_defense)
        self.assertEqual(defended.ggeur.defense_method, 'multi_metrics')

    def test_legacy_methods_resolve_to_separate_modes(self):
        for method in ['a3fl', 'cerberus', 'sabre', 'label_flip', 'lie']:
            cfg = _base_cfg()
            cfg.attack.attack_method = method
            self.assertEqual(resolve_security_mode(cfg), 'backdoor')

        for method in ['fedmia', 'ggeur_fedmia', 'ggeur_ppa', 'grnn']:
            cfg = _base_cfg()
            cfg.attack.attack_method = method
            self.assertEqual(resolve_security_mode(cfg), 'privacy')

    def test_worker_builder_keeps_ggeur_as_the_base(self):
        cfg = _base_cfg()
        cfg.attack.attack_method = 'a3fl'
        self.assertIs(get_server_cls(cfg), GGEURServer)
        self.assertIs(get_client_cls(cfg), GGEURClient)

        cfg.attack.attack_method = 'grnn'
        from federatedscope.contrib.worker.ggeur_attack_server import \
            GGEURPassiveServer
        self.assertIs(get_server_cls(cfg), GGEURPassiveServer)

    def test_backdoor_and_privacy_configuration_is_rejected(self):
        cfg = _base_cfg()
        cfg.security.mode = 'backdoor'
        cfg.attack.attack_method = 'sabre'
        cfg.attack.modular_attacks = True
        cfg.attack.attack_plugins = ['fedmia_i']
        with self.assertRaisesRegex(ValueError, 'Privacy attacks/plugins'):
            cfg.freeze()

    def test_privacy_mode_rejects_both_multimetric_phases(self):
        cfg = _base_cfg()
        cfg.security.mode = 'privacy'
        cfg.attack.attack_method = 'ggeur_fedmia'
        cfg.ggeur.defense_method = 'multi_metrics'
        cfg.ggeur.multi_metrics_stats_defense = True
        with self.assertRaisesRegex(ValueError, 'MultiMetric'):
            cfg.freeze()

    def test_unified_security_mode_is_standalone_only(self):
        cfg = _base_cfg()
        cfg.security.mode = 'privacy'
        cfg.attack.attack_method = 'ggeur_ppa'
        cfg.federate.mode = 'distributed'
        with self.assertRaisesRegex(ValueError, 'standalone'):
            cfg.freeze()

    def test_adaptive_clipping_state_is_client_local(self):
        cfg = _base_cfg()
        cfg.dp.enabled = True
        cfg.dp.level = 'client_update'
        cfg.dp.protect_ggeur_update = True
        cfg.dp.noise_multiplier = 0.1
        cfg.dp.clipping.type = 'adaptive'
        cfg.dp.clipping.initial_clip = 1.0
        cfg.dp.clipping.ema = 0.0

        client_one = GGEURClient.__new__(GGEURClient)
        client_one.ID = 1
        client_one._cfg = cfg
        client_one._adaptive_dp_global_mlp_state = {
            'weight': torch.zeros(2),
        }
        client_one._local_adaptive_clipper = None

        client_two = GGEURClient.__new__(GGEURClient)
        client_two.ID = 2
        client_two._cfg = cfg
        client_two._adaptive_dp_global_mlp_state = {
            'weight': torch.zeros(2),
        }
        client_two._local_adaptive_clipper = None

        protected, public_stats = client_one._apply_adaptive_dp_to_upload(
            {'weight': torch.tensor([3.0, 4.0])}, round_idx=1)

        self.assertIsInstance(client_one._local_adaptive_clipper,
                              LocalAdaptiveClipper)
        self.assertIsNone(client_two._local_adaptive_clipper)
        self.assertEqual(len(client_one._local_adaptive_clipper.norm_history),
                         1)
        self.assertNotIn('raw_norm', public_stats)
        self.assertIn('noise_variance', public_stats)
        self.assertFalse(torch.equal(
            protected['weight'], torch.tensor([3.0, 4.0])))

    def test_privacy_hooks_observe_the_uploaded_protected_state(self):
        protected_state = {'weight': torch.tensor([0.25, -0.5])}
        content = (
            8,
            protected_state,
            {'mechanism': 'standalone_local_adaptive_client_update_dp'},
        )
        fedmia_hook = GGEURFedMIAHook.__new__(GGEURFedMIAHook)
        ppa_hook = GGEURPPAHook.__new__(GGEURPPAHook)

        fedmia_visible = fedmia_hook._reconstruct_server_visible_state(
            1, content)
        ppa_visible = ppa_hook._reconstruct_server_visible_state(1, content)

        self.assertIs(fedmia_visible, protected_state)
        self.assertIs(ppa_visible, protected_state)

    def test_grnn_branch_is_privacy_only_and_can_be_protected(self):
        cfg = _base_cfg()
        cfg.security.mode = 'privacy'
        cfg.attack.attack_method = 'grnn'
        cfg.dp.enabled = True
        cfg.dp.level = 'client_update'
        cfg.dp.protect_ggeur_update = True
        cfg.dp.noise_multiplier = 0.1
        cfg.dp.clipping.type = 'adaptive'

        client = GGEURClient.__new__(GGEURClient)
        client.ID = 3
        client._cfg = cfg
        client.security_mode = 'privacy'
        client._grnn_adaptive_clipper = None
        client._attack_last_batch_data = None
        client._attack_cnn_gradients = None
        client._attack_backbone_gradients = None

        model = torch.nn.Linear(2, 1)
        model(torch.ones(1, 2)).sum().backward()
        client._record_grnn_batch_and_gradients(
            torch.ones(1, 2), torch.zeros(1, dtype=torch.long),
            model, 'cnn')
        self.assertTrue(client._is_grnn_attack_enabled())
        self.assertIsNotNone(client._attack_cnn_gradients)

        global_state = {'weight': torch.zeros(2)}
        local_state = {'weight': torch.tensor([1.0, -1.0])}
        protected, defended = client._protect_image_branch_for_upload(
            global_state, local_state, round_idx=1)
        self.assertTrue(defended)
        self.assertFalse(torch.equal(protected['weight'],
                                     local_state['weight']))

        client.security_mode = 'backdoor'
        self.assertFalse(client._is_grnn_attack_enabled())

    def test_multimetric_filters_feature_statistics(self):
        server = GGEURServer.__new__(GGEURServer)
        server.device = 'cpu'
        server._current_aggregation_round = None
        server.ggeur_cfg = SimpleNamespace(
            multi_metrics_stats_min_clients=4,
            multi_metrics_stats_feat_gap_thresh=0.5,
            multi_metrics_stats_cov_eps=1e-6,
            multi_metrics_stats_adaptive_threshold=False,
            multi_metrics_stats_keep_ratio=0.75,
            multi_metrics_stats_z_threshold=2.0,
            multi_metrics_debug=False,
        )
        statistics = {}
        for client_id, scale in enumerate([1.0, 1.1, 0.9, 100.0], 1):
            statistics[client_id] = {
                'covs': {
                    0: np.eye(2) * scale,
                    1: np.eye(2) * scale,
                },
                'prototypes': {
                    0: np.asarray([1.0, 0.0]),
                    1: np.asarray([0.0, 1.0]),
                },
            }

        filtered = server._multi_metrics_filter_statistics(statistics)

        self.assertEqual(set(filtered), {1, 2, 3})

    def test_multimetric_filters_training_updates(self):
        server = GGEURServer.__new__(GGEURServer)
        server.device = 'cpu'
        server._current_aggregation_round = None
        server.multi_metrics_score_history = {}
        server.ggeur_cfg = SimpleNamespace(
            multi_metrics_min_clients=4,
            multi_metrics_cov_eps=1e-6,
            multi_metrics_history_smoothing=False,
            multi_metrics_adaptive_threshold=False,
            multi_metrics_keep_ratio=0.75,
            multi_metrics_debug=False,
        )
        selected = {}

        def capture_fedavg(params, total_samples):
            selected['ids'] = [sender for _, _, sender in params]
            return {'weight': torch.zeros(2)}

        server._fedavg_model_params_no_defense = capture_fedavg
        updates = [
            (1, {'weight': torch.tensor([value, value * 2])}, client_id)
            for client_id, value in enumerate([0.1, 0.11, 0.12, 10.0], 1)
        ]

        result = server._multi_metrics_aggregate_model_params(
            updates, {'weight': torch.zeros(2)}, aggregation_name='mlp')

        self.assertEqual(len(selected['ids']), 3)
        self.assertEqual(set(result), {'weight'})


if __name__ == '__main__':
    unittest.main()
