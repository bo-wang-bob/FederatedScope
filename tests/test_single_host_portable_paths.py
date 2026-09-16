"""Path portability and cached membership display regression checks."""
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from federatedscope.standalone_api.paths import REPO_ROOT, env_path
from federatedscope.standalone_api.platform_app import PlatformHandler
from federatedscope.standalone_api.platform_config import ConfigFactory


class PortablePathsTests(unittest.TestCase):
    def test_relative_paths_do_not_depend_on_startup_directory(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            try:
                os.chdir(directory)
                configs = ConfigFactory(REPO_ROOT)
                self.assertEqual(configs.resources, REPO_ROOT / 'resources')
                self.assertEqual(configs.datasets, REPO_ROOT / 'resources/datasets')
                handler = object.__new__(PlatformHandler)
                self.assertEqual(handler._fedmia_root(), REPO_ROOT.parent / 'fedmia_local')
                with patch.dict(os.environ, {'FS_FEDMIA_LOCAL_ROOT': 'private/features'}):
                    self.assertEqual(handler._fedmia_root(), REPO_ROOT / 'private/features')
                with patch.dict(os.environ, {'FS_PLATFORM_DATASETS': directory}):
                    self.assertEqual(env_path('FS_PLATFORM_DATASETS', 'data'), Path(directory).resolve())
            finally:
                os.chdir(previous)

    def test_relative_resource_overrides_and_military_weights(self):
        with patch.dict(os.environ, {
            'FS_PLATFORM_RESOURCES': '../resources',
            'FS_PLATFORM_DATASETS': '../datasets',
            'FS_PLATFORM_CACHE_MILITARY_VIT': 'cache/military',
            'FS_PLATFORM_MILITARY_VIT_WEIGHTS': 'weights/ViT-B-16.pt',
        }):
            configs = ConfigFactory(REPO_ROOT)
            self.assertEqual(configs.resources, REPO_ROOT.parent / 'resources')
            self.assertEqual(configs.datasets, REPO_ROOT.parent / 'datasets')
            self.assertEqual(configs.cache_dir('military_vit'), REPO_ROOT / 'cache/military')
            req = configs.normalize({'group': 'military_vit', 'method': 'fedprox'})
            raw, _ = configs.build(req, REPO_ROOT / 'exp/path-test')
            self.assertEqual(raw['ggeur']['clip_model_path'], 'weights/ViT-B-16.pt')
            self.assertTrue(raw['fedprox']['use'])

    def test_membership_metrics_and_distribution_remain_available(self):
        handler = object.__new__(PlatformHandler)
        module = SimpleNamespace(np=np)
        metrics = handler._fedmia_attack_metrics([.8, .9], [.1, .2], module)
        self.assertEqual(metrics['auc'], 1.0)
        distribution = handler._fedmia_score_distribution([.8, .9], [.1, .2], module)
        self.assertEqual(len(distribution['member']), 40)
        self.assertAlmostEqual(sum(distribution['member']), 1)
        self.assertAlmostEqual(distribution['meanGap'], .7)


if __name__ == '__main__':
    unittest.main()
