"""Privacy isolation, request binding and actual score-threshold regression."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')
os.environ.setdefault('FEDERATEDSCOPE_GGEUR_LIGHTWEIGHT', '1')

from federatedscope.standalone_api.platform_config import ConfigFactory, PlatformError
from federatedscope.standalone_api.platform_privacy import PrivacyConfig, PrivacyService
from federatedscope.standalone_api.platform_privacy_worker import operating_point, privacy_bases
from federatedscope.standalone_api.platform_service import PlatformService

REPO = Path(__file__).resolve().parents[1]


class PrivacyTests(unittest.TestCase):
    def setUp(self):
        # Same drive is required for portable paths on Windows.
        self.temp = tempfile.TemporaryDirectory(dir=REPO / 'exp')
        self.privacy = PrivacyService(REPO, self.temp.name)
        self.normal = PlatformService(REPO, Path(self.temp.name) / 'normal')
        self.request = dict(group='military_cnn', method='ggeur', rounds=2,
                            clientCount=3, defense=True)

    def tearDown(self):
        self.privacy.close()
        self.normal.close()
        self.temp.cleanup()

    def test_private_config_does_not_mutate_normal_factory(self):
        base = ConfigFactory(REPO)
        normal_req = base.normalize(dict(group='military_vit', method='fedavg', rounds=2, clientCount=3))
        before = base.build(normal_req, self.temp.name)[0]
        request = self.privacy.configs.normalize(self.request)
        config, source = self.privacy.configs.build(request, self.temp.name)
        self.assertTrue(source['privacyExperiment'])
        self.assertTrue(config['dp']['enabled'])
        self.assertEqual(config['attack']['ggeur_target_size'], 0)
        self.assertEqual(config['attack']['fedmia_shadow_stat_mode'], 'indexed')
        self.assertTrue(config['ggeur']['local_decoy']['use'])
        self.assertEqual(before, base.build(normal_req, self.temp.name)[0])
        self.assertEqual(config['train']['optimizer']['lr'], .001)
        self.assertEqual(config['attack']['fedmia_save_interval'], 40)
        self.assertEqual(config['attack']['fedmia_var_floor'], .001)
        self.assertEqual(config['ggeur']['embedding_dim'], 1024)
        self.assertEqual(config['dp']['clipping']['min_clip'], .05)

    def test_isolated_jobs_and_preflight_exact_match(self):
        with patch('federatedscope.standalone_api.platform_service.threading.Thread'):
            pre = self.privacy.create('inspect', dict(self.request, idempotencyKey='privacy-preflight-key'))
            self.assertEqual(self.normal.list(), [])
            pre.update(status='completed', result=dict(testFingerprint='a', partitionFingerprint='b'))
            self.privacy._save(pre)
            with self.assertRaises(PlatformError):
                self.privacy.create('train', dict(self.request, defense=False, preflightId=pre['id'], idempotencyKey='privacy-mismatch-key'))
            task = self.privacy.create('train', dict(self.request, preflightId=pre['id'], idempotencyKey='privacy-training-key'))
            spec = json.loads((self.privacy.directory(task['id']) / 'spec.json').read_text(encoding='utf-8'))
            self.assertTrue(spec['provenance']['privacyExperiment'])
            self.assertEqual(spec['expectedPartition'], 'b')
            self.assertEqual(self.normal.list(), [])

    def test_reject_invalid_privacy_parameters(self):
        for change in ({'defense': 'yes'}, {'saveInterval': 0}, {'mixLength': 1.5},
                       {'noiseMultiplier': float('nan')}, {'shadowMode': 'anything'},
                       {'sampleClients': 1}, {'unexpected': 1}):
            with self.subTest(change=change), self.assertRaises(PlatformError):
                self.privacy.configs.normalize(dict(self.request, **change))

    def test_threshold_ties_and_perfect_separation(self):
        result = operating_point([.9, .95], [.1, .2, .3])
        self.assertEqual(result['auc'], 1)
        self.assertEqual(result['tprAt1Fpr'], 1)
        self.assertEqual(result['actualFpr'], 0)
        self.assertGreater(result['threshold'], .3)
        tied = operating_point([.5, .5], [.5, .5])
        self.assertEqual(tied['auc'], .5)
        self.assertEqual(tied['tprAt1Fpr'], 0)
        self.assertGreater(tied['threshold'], .5)
        with self.assertRaises(ValueError):
            operating_point([], [.5])

    def test_cached_evaluation_does_not_decode_images(self):
        import numpy as np
        import torch
        from types import SimpleNamespace
        client, server = privacy_bases({'request': {'defense': True}}, {'a': {'a/0.png': np.ones(4)}})
        instance = object.__new__(server)
        instance._evaluate_feature_arrays = lambda values, labels, state, **kw: (values, labels)
        dataset = SimpleNamespace(domain='a', targets=[1], data=['a/0.png'])
        values, labels = instance._evaluate_dataset_subset(dataset, None, {})
        self.assertEqual(values.shape, (1, 4))
        self.assertEqual(labels.tolist(), [1])

    def test_adaptive_protection_is_applied_to_actual_upload(self):
        import numpy as np
        import torch
        from types import SimpleNamespace
        from federatedscope.core.configs.config import global_cfg
        from federatedscope.contrib.worker.ggeur_client import GGEURClient
        client, _ = privacy_bases({'request': {'defense': True}}, {})
        instance = object.__new__(client)
        instance._cfg = global_cfg.clone()
        instance._cfg.adaptive_dp.use = True
        instance._cfg.adaptive_dp.initial_clip = .1
        instance._cfg.adaptive_dp.noise_multiplier = 0.
        instance._cfg.ggeur.local_decoy.use = False
        instance.ID, instance.state = 1, 1
        instance._local_adaptive_clipper = None
        instance._privacy_noise_multiplier = lambda config: 0.
        instance.mlp_classifier = torch.nn.Linear(2, 2)
        before = {k: v.clone() for k, v in instance.mlp_classifier.state_dict().items()}
        raw = {k: v + 10 for k, v in before.items()}
        with patch.object(GGEURClient, '_train_on_augmented_data', return_value=(2, raw, {'train_loss': .1, 'train_acc': .5})):
            _, upload, _ = instance._train_on_augmented_data()
        norm = sum(float(((upload[k] - before[k]) ** 2).sum()) for k in before) ** .5
        self.assertLessEqual(norm, .10001)
        self.assertGreater(norm, .09)


if __name__ == '__main__':
    unittest.main()
