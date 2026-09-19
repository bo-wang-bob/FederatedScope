import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from federatedscope.standalone_api.platform_config import ConfigFactory
from federatedscope.standalone_api.platform_privacy import PrivacyConfig
from federatedscope.standalone_api.uploaded_config import UploadedConfig

class DeviceDefaultsTests(unittest.TestCase):
    def test_obsolete_local_cpu_environment_does_not_override_server_defaults(self):
        repo=Path(__file__).resolve().parents[1]
        with patch.dict(os.environ, {'FS_PLATFORM_DEVICE':'cpu'}), tempfile.TemporaryDirectory(dir=repo/'exp') as state:
            factory=ConfigFactory(repo)
            for method in ('fedavg','fedprox','heterogeneous_solution'):
                self.assertEqual(factory.defaults('military_vit',method)['gpu'],0)
                req=factory.normalize(dict(group='military_vit',method=method))
                self.assertEqual(req['gpu'],0)
                raw,_=factory.build(req,Path(state))
                self.assertTrue(raw['use_gpu'])
                explicit=factory.normalize(dict(group='military_vit',method=method,gpu=0))
                self.assertEqual(explicit['gpu'],0)
                explicit_raw,_=factory.build(explicit,Path(state))
                self.assertTrue(explicit_raw['use_gpu'])
            privacy=PrivacyConfig(factory)
            req=privacy.normalize(dict(group='military_cnn',method='ggeur',rounds=1,clientCount=3))
            raw,provenance=privacy.build(req,Path(state))
            _,preset=privacy.preset('military_cnn',False)
            self.assertEqual(raw['use_gpu'],preset['use_gpu'])
            self.assertNotIn('use_gpu',provenance['configOverrides'])
            uploaded=UploadedConfig(factory)
            with patch.object(uploaded.store,'training',return_value={}):
                self.assertEqual(uploaded.defaults('uploaded_'+'a'*32,'fedavg')['gpu'],0)

    def test_server_preserves_gpu_presets_and_explicit_device(self):
        repo=Path(__file__).resolve().parents[1]
        environment=dict(os.environ)
        environment.pop('FS_PLATFORM_DEVICE',None)
        with patch.dict(os.environ,environment,clear=True), tempfile.TemporaryDirectory(dir=repo/'exp') as state:
            factory=ConfigFactory(repo)
            for group in ('military_vit','officehome_vit'):
                for method in ('fedavg','fedprox','heterogeneous_solution'):
                    self.assertGreaterEqual(factory.defaults(group,method)['gpu'],0)
                    for device in (0,1):
                        req=factory.normalize(dict(group=group,method=method,gpu=device))
                        raw,_=factory.build(req,Path(state))
                        self.assertTrue(raw['use_gpu'])
                        self.assertEqual(raw['device'],device)
            uploaded=UploadedConfig(factory)
            with patch.object(uploaded.store,'training',return_value={}):
                for method in ('fedavg','fedprox','heterogeneous_solution'):
                    req=uploaded.defaults('uploaded_'+'a'*32,method)
                    self.assertEqual(req['gpu'],0)
                    self.assertEqual(uploaded.normalize(req)['gpu'],0)
            privacy=PrivacyConfig(factory)
            for group in ('military_cnn','officehome_cnn'):
                for defense in (False,True):
                    req=privacy.defaults(group,defense)
                    _,preset=privacy.preset(group,defense)
                    raw,provenance=privacy.build(req,Path(state))
                    self.assertEqual(raw['use_gpu'],preset['use_gpu'])
                    self.assertEqual(raw['device'],preset['device'])
                    self.assertNotIn('use_gpu',provenance['configOverrides'])
