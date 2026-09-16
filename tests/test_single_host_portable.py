"""Regression checks for moving a package without changing experiment identity."""
import copy
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from federatedscope.standalone_api.platform_config import ConfigFactory, PlatformError, sha256
from federatedscope.standalone_api.platform_paths import cache_file, resolve_path
from federatedscope.standalone_api.platform_service import PlatformService
from federatedscope.standalone_api.platform_worker import check_feature_contract

REPO = Path(__file__).resolve().parents[1]


class PortableTests(unittest.TestCase):
    def test_relative_settings_ignore_callers_current_directory(self):
        with patch.dict(os.environ, {'FS_PLATFORM_RESOURCES': 'resources',
                'FS_PLATFORM_DATASETS': 'resources/datasets',
                'FS_PLATFORM_CACHE_MILITARY_VIT': 'resources/cache'}):
            factory = ConfigFactory(REPO)
            self.assertEqual(factory.resources, REPO / 'resources')
            self.assertEqual(factory.cache_dir('military_vit'), REPO / 'resources/cache')
            cfg, _ = factory.build(factory.defaults('military_vit', 'fedavg'), REPO / 'exp/platform/jobs/example')
            self.assertEqual(cfg['data']['root'], 'resources/datasets/MilitaryAircraft3D')
            self.assertEqual(cfg['ggeur']['mlp_checkpoint_dir'], 'exp/platform/jobs/example/checkpoints')

    def test_cache_paths_allow_only_members_and_exact_legacy_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = '/old/server/jobs/' + 'a' * 32 + '/augmented_cache/client_000001.pt'
            self.assertEqual(cache_file(root, old, 'a' * 32), root / 'client_000001.pt')
            for path in ('../secret.pt', '/etc/secret.pt', 'C:/secret.pt'):
                with self.assertRaises(ValueError):
                    cache_file(root, path, 'a' * 32)

    def test_auto_reuse_survives_moved_state_and_is_pinned(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / 'before'
            service = PlatformService(REPO, first)
            req = service.configs.defaults('military_vit', 'heterogeneous_solution')
            req['clientCount'] = 3
            source_id = 'a' * 32
            root = service.directory(source_id) / 'augmented_cache'
            root.mkdir(parents=True)
            clients = {}
            for cid in range(1, 4):
                path = root / f'client_{cid:06d}.pt'
                path.write_bytes(b'unit-test-cache')
                clients[str(cid)] = dict(path=str(path), sha256=sha256(path))
            source = dict(id=source_id, action='train', status='completed', createdAt='2026-01-01',
                request=dict(req, augmentationMode='generate', localEpochs=5), result=dict(augmentation=dict(
                    mode='generate', provenance='generated-from-recorded-training-samples', clients=clients)))
            service._save(source)
            service.close()
            moved = Path(tmp) / 'after'
            shutil.move(str(first), str(moved))
            service = PlatformService(REPO, moved)
            try:
                execution, selected, registered, new_root = service._augmentation_execution(req)
                self.assertEqual(execution['augmentationMode'], 'reuse')
                self.assertEqual(selected['sourceId'], source_id)
                self.assertEqual(registered['clients']['1']['path'], 'client_000001.pt')
                # Preflight selected generation: a later cache must not switch it.
                self.assertEqual(service._augmentation_execution(req, {'mode': 'generate'})[0]['augmentationMode'], 'generate')
                (new_root / 'client_000001.pt').write_bytes(b'changed')
                with self.assertRaises(PlatformError):
                    service._augmentation_execution(req, selected)
            finally:
                service.close()

    def test_old_new_models_require_same_feature_identity(self):
        old = {'dataset': 'domainnet', 'backbone': {'feature_extractor': 'clip', 'model': 'ViT-B-16',
               'pretrained': 'openai', 'checkpoint': '/old/ViT-B-16.pt'}}
        new = copy.deepcopy(old)
        new['backbone']['checkpoint'] = 'models/ViT-B-16.pt'
        with self.assertRaises(ValueError):
            check_feature_contract(copy.deepcopy(old), copy.deepcopy(new), {})
        check_feature_contract(copy.deepcopy(old), copy.deepcopy(new), {'featureSpace': 'verified-cache-hash'})
        new['featureSpace'] = 'another-cache'
        with self.assertRaises(ValueError):
            check_feature_contract(copy.deepcopy(old), new, {'featureSpace': 'verified-cache-hash'})


if __name__ == '__main__':
    unittest.main()
