"""Military demo registration; no production assets or training are modified."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from federatedscope.standalone_api.platform_config import ConfigFactory, PlatformError
from federatedscope.standalone_api.platform_samples import SampleCatalog
from federatedscope.standalone_api.platform_paths import resolve_path


class AircraftRegistrationTests(unittest.TestCase):
    def setUp(self):
        self.configs = ConfigFactory(Path(__file__).resolve().parents[1])

    def test_catalog_registers_real_three_method_configs(self):
        entry = next(g for g in self.configs.catalog()['groups'] if g['id'] == 'military_vit')
        self.assertEqual((entry['dataset'], entry['backbone'], entry['domains']),
                         ('MilitaryAircraft-3D', 'vit', 3))
        self.assertEqual([m['id'] for m in entry['methods']],
                         ['fedavg', 'fedprox', 'heterogeneous_solution'])
        self.assertTrue(str(self.configs.cache_dir('military_vit')).endswith('military_aircraft_vit_fixedsplit_v2'))
        with self.assertRaises(PlatformError):
            self.configs.normalize({'group': 'military_vit', 'method': 'fedopt'})

    def test_all_three_methods_keep_military_data_and_parameter_bindings(self):
        with tempfile.TemporaryDirectory() as root:
            self.configs.datasets = Path(root) / 'datasets'
            for method in ('fedavg', 'fedprox', 'heterogeneous_solution'):
                req = self.configs.normalize({'group': 'military_vit', 'method': method,
                    'rounds': 2, 'learningRate': .003, 'localEpochs': 2, 'batchSize': 9})
                raw, provenance = self.configs.build(req, Path(root) / method)
                self.assertFalse(Path(raw['data']['root']).is_absolute())
                self.assertEqual(resolve_path(self.configs.repo, raw['data']['root']), self.configs.datasets / 'MilitaryAircraft3D')
                self.assertEqual(raw['ggeur']['domainnet_domains'], ['aerial', 'natural', 'recon'])
                self.assertEqual(raw['ggeur']['domainnet_manifest_path'], '')
                self.assertEqual(raw['model']['num_classes'], 5)
                self.assertEqual(req['gpu'], 0)
                self.assertEqual(raw['device'], 0)
                self.assertEqual(raw['federate']['client_num'], 15)
                self.assertEqual(raw['federate']['total_round_num'], 3)
                self.assertEqual(raw['train']['optimizer']['lr'], .003)
                self.assertEqual(raw['train']['local_update_steps'], 2)
                self.assertEqual(raw['dataloader']['batch_size'], 9)
                self.assertIn('military_aircraft_3domain/configs/', provenance['source'].replace('\\', '/'))
                self.assertTrue(raw['ggeur']['require_complete_feature_cache'])
                if method == 'fedprox':
                    self.assertTrue(raw['fedprox']['use'])
                    self.assertEqual(raw['fedprox']['mu'], 5)
                self.assertEqual(raw['ggeur']['num_generated_per_prototype'], 20 if method == 'heterogeneous_solution' else 0)

    def test_defaults_are_the_aircraft_release_contract(self):
        req = self.configs.defaults('military_vit', 'heterogeneous_solution')
        self.assertEqual([req[k] for k in ('rounds', 'localEpochs', 'targetPerClass',
            'generatedPerSample', 'generatedPerPrototype', 'sampleClients')], [100, 1, 40, 20, 20, 0])
        self.assertEqual(req['augmentationMode'], 'auto')
        for method in ('fedavg', 'fedprox'):
            baseline = self.configs.defaults('military_vit', method)
            self.assertEqual(baseline['rounds'], 100)
            self.assertEqual(baseline['augmentationMode'], 'none')

    def test_cache_override_and_readonly_sample_path(self):
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, {'FS_PLATFORM_CACHE_MILITARY_VIT': root}):
            self.assertEqual(self.configs.cache_dir('military_vit'), Path(root))
            self.configs.datasets = Path(root)
            image = Path(root) / 'MilitaryAircraft3D/aerial/F-16/example.jpg'
            image.parent.mkdir(parents=True)
            image.write_bytes(b'fixture')
            catalog = SampleCatalog(type('Service', (), {'configs': self.configs})())
            job = {'request': {'group': 'military_vit'}}
            self.assertEqual(catalog.image_path(job, {'key': 'aerial/f-16/example.jpg'}), image)
            with self.assertRaises(PlatformError):
                catalog.image_path(job, {'key': '../example.jpg'})


if __name__ == '__main__':
    unittest.main()
