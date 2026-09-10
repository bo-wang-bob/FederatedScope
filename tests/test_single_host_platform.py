"""Control-plane safety and parameter regression tests; no production data writes."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import psutil

from federatedscope.standalone_api.platform_config import ConfigFactory, PlatformError
from federatedscope.standalone_api.platform_service import PlatformService, now
from federatedscope.standalone_api.platform_worker import classification_metrics, evaluate, digest
from federatedscope.standalone_api.repository import JsonRepository

REPO = Path(__file__).resolve().parents[1]


class PlatformTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.service = PlatformService(REPO, Path(self.temp.name) / 'state')
        self.configs = ConfigFactory(REPO)
        self.payload = dict(group='officehome_vit', method='fedavg', idempotencyKey='test-key-0001')

    def tearDown(self):
        self.service.close()
        self.temp.cleanup()

    def create_without_start(self, action='inspect', payload=None):
        with patch('federatedscope.standalone_api.platform_service.threading.Thread'):
            return self.service.create(action, payload or self.payload)

    def complete_preflight(self):
        job = self.create_without_start()
        job.update(status='completed', result={'testFingerprint': 'test', 'partitionFingerprint': 'partition'})
        self.service._save(job)
        return job

    def test_all_baselines_bind_parameters_and_disable_generation(self):
        count = 0
        for group in self.configs.catalog()['groups']:
            for method in group['methods']:
                if not method['enabled'] or method['id'] == 'heterogeneous_solution':
                    continue
                req = self.configs.normalize(dict(group=group['id'], method=method['id'],
                    rounds=7, learningRate=.002, localEpochs=2, batchSize=17, gpu=-1,
                    evaluationFrequency=2, seed=99, samplesPerClient=13))
                cfg, metadata = self.configs.build(req, Path(self.temp.name) / 'output')
                self.assertEqual(cfg['federate']['mode'], 'standalone')
                self.assertEqual(cfg['federate']['process_num'], 0)
                self.assertEqual(cfg['federate']['total_round_num'], 8)
                self.assertEqual(cfg['train']['optimizer']['lr'], .002)
                self.assertEqual(cfg['train']['local_update_steps'], 2)
                self.assertEqual(cfg['dataloader']['batch_size'], 17)
                self.assertEqual(cfg['eval']['freq'], 2)
                self.assertFalse(cfg['use_gpu'])
                self.assertTrue(cfg['ggeur']['require_complete_feature_cache'])
                self.assertFalse(cfg['ggeur']['save_augmented_feature_cache'])
                for key in ('num_generated_per_sample', 'num_generated_per_prototype', 'target_size_per_class'):
                    self.assertEqual(cfg['ggeur'][key], 0)
                self.assertEqual(cfg['ggeur']['baseline_target_samples_per_client'], 13)
                self.assertIn('sourceSha256', metadata)
                count += 1
        self.assertGreater(count, 30)

    def test_reject_invalid_and_ignored_parameters(self):
        for extra in ({'rounds': True}, {'learningRate': float('nan')}, {'gpu': '1'},
                      {'clientCount': 59}, {'sampleClients': 61}, {'unexpected': 2},
                      {'group': '../officehome_vit'}, {'method': []}, {'group': {}},
                      {'method': 'heterogeneous_solution', 'augmentationMode': 'none'},
                      {'method': 'heterogeneous_solution', 'augmentationMode': 'reuse'},
                      {'augmentationMode': 'generate'}, {'allowLegacyAugmentation': 'yes'}):
            with self.subTest(extra=extra), self.assertRaises(PlatformError):
                self.configs.normalize(dict(self.payload, **extra))

    def test_generation_does_not_require_augmented_cache_and_binds_parameters(self):
        req = self.configs.normalize(dict(group='officehome_vit', method='heterogeneous_solution',
            generatedPerSample=2, generatedPerPrototype=3, targetPerClass=20,
            covarianceScale=.25, samplesPerClient=40))
        cfg, _ = self.configs.build(req, self.temp.name)
        g = cfg['ggeur']
        self.assertEqual(req['augmentationMode'], 'generate')
        self.assertEqual(g['num_generated_per_sample'], 2)
        self.assertEqual(g['num_generated_per_prototype'], 3)
        self.assertEqual(g['target_size_per_class'], 20)
        self.assertEqual(g['generation_covariance_scale'], .25)
        self.assertEqual(g['platform_target_samples_per_client'], 40)
        self.assertFalse(g['reuse_augmented_feature_cache'])
        self.assertEqual(Path(g['augmented_feature_cache_dir']), Path(self.temp.name) / 'augmented_cache')
        self.assertEqual(next(m for m in self.configs.catalog()['groups'][0]['methods']
            if m['id'] == 'heterogeneous_solution')['label'], '本架构')

    def test_augmentation_tensor_validation_and_safe_loading(self):
        import torch
        from federatedscope.standalone_api.platform_augmentation import safe_load, validate_arrays
        path = Path(self.temp.name) / 'cache.pt'
        good = dict(features=torch.ones(3, 2), labels=torch.tensor([0, 1, 0]))
        torch.save(good, path)
        self.assertEqual(validate_arrays(safe_load(path), 2, 2)[0].shape, (3, 2))
        for bad in (dict(good, labels=torch.tensor([0., 1., 0.])),
                    dict(good, features=torch.full((3, 2), float('nan'))),
                    dict(good, labels=torch.tensor([0, 2, 0]))):
            with self.assertRaises(ValueError):
                validate_arrays(bad, 2, 2)

    def test_registered_augmentation_requires_matching_source_and_file_hash(self):
        import torch
        from types import SimpleNamespace
        from federatedscope.standalone_api.platform_augmentation import inspect_existing, file_hash
        root = Path(self.temp.name)
        metadata = dict(client_id=1, client_num=1, dataset='test', seed=42,
            splits=[1., 0., 0.], feature_extractor='clip', feature_extractor_model='test',
            embedding_dim=2, num_classes=2, num_generated_per_sample=1,
            num_generated_per_prototype=1, target_size_per_class=2,
            use_cross_client_prototypes=True, max_cross_client_prototypes_per_class=0,
            use_fedproto=False, use_lds=False, lds_alpha=.1, lds_seed=42,
            augmented_feature_cache_version='test')
        probe = SimpleNamespace(ggeur_cfg=SimpleNamespace(augmented_feature_cache_dir=str(root)),
                                _augmented_cache_metadata=lambda: metadata)
        clients = {1: [('A', 'x', 0)]}
        info = dict(featureSpace='space', partitionFingerprint='partition')
        path = root / 'client.pt'
        payload = dict(features=torch.ones(2, 2), labels=torch.tensor([0, 1]), metadata=metadata,
            source_samples=[['A', 'x', 0]], source_feature_space='space', source_partition='partition')
        torch.save(payload, path)
        def registered():
            return {'clients': {'1': {'path': str(path), 'sha256': file_hash(path)}}}
        result = inspect_existing(probe, clients, {}, info, False, registered())
        self.assertIsNone(result['warning'])
        payload['source_partition'] = 'wrong'
        torch.save(payload, path)
        with self.assertRaises(ValueError):
            inspect_existing(probe, clients, {}, info, True, registered())
        del payload['source_samples']
        torch.save(payload, path)
        with self.assertRaises(ValueError):
            inspect_existing(probe, clients, {}, info, False, registered())
        self.assertIn('历史', inspect_existing(probe, clients, {}, info, True, registered())['warning'])
        stale = registered()
        torch.save(dict(payload, labels=torch.tensor([1, 0])), path)
        with self.assertRaisesRegex(ValueError, '修改'):
            inspect_existing(probe, clients, {}, info, True, stale)

    def test_fixed_digit_manifest_cannot_accept_ineffective_partition_params(self):
        for extra in ({'alpha': .5}, {'splitSeed': 5}, {'clientCount': 30}):
            with self.assertRaises(PlatformError):
                self.configs.normalize(dict(group='digit3_vit', method='fedavg', **extra))

    def test_registered_test_cache_does_not_replace_training_cache(self):
        with patch.dict(os.environ, {'FS_PLATFORM_CACHE_OFFICEHOME_VIT': '/existing/train',
                                    'FS_PLATFORM_TEST_CACHE_OFFICEHOME_VIT': '/existing/test'}):
            config, provenance = self.configs.build(self.configs.normalize(self.payload), self.temp.name)
        self.assertEqual(config['ggeur']['feature_cache_dir'], str(Path('/existing/train')))
        self.assertEqual(provenance['testCacheDir'], '/existing/test')

    def test_catalog_reports_latest_terminal_preflight(self):
        job = self.complete_preflight()
        group = next(g for g in self.service.catalog()['groups'] if g['id'] == self.payload['group'])
        self.assertEqual(group['lastPreflight']['id'], job['id'])
        self.assertEqual(group['lastPreflight']['status'], 'completed')

    def test_idempotency_and_exclusive_reservation(self):
        first = self.create_without_start()
        self.assertEqual(first['id'], self.service.create('inspect', self.payload)['id'])
        with self.assertRaises(PlatformError):
            self.service.create('inspect', dict(self.payload, rounds=8))
        with self.assertRaises(PlatformError):
            self.service.create('inspect', dict(self.payload, idempotencyKey='other-key-001'))

    def test_train_requires_matching_successful_preflight(self):
        job = self.complete_preflight()
        with self.assertRaises(PlatformError):
            self.create_without_start('train', dict(self.payload, preflightId=job['id'], rounds=8, idempotencyKey='training-key'))
        trained = self.create_without_start('train', dict(self.payload, preflightId=job['id'], idempotencyKey='training-key'))
        self.assertEqual(trained['status'], 'queued')
        spec = json.loads((self.service.directory(trained['id']) / 'spec.json').read_text())
        self.assertEqual(spec['expectedPartition'], 'partition')

    def test_stop_queued_job_never_starts_late(self):
        job = self.create_without_start()
        self.service.stop(job['id'])
        with patch('federatedscope.standalone_api.platform_service.subprocess.Popen') as spawn:
            self.service._run(job['id'])
            spawn.assert_not_called()
        self.assertEqual(self.service.get(job['id'])['status'], 'stopped')

    def test_unsafe_paths_rejected(self):
        for value in ('../', '/tmp', 'x' * 32, ''):
            with self.assertRaises(PlatformError):
                self.service.get(value)

    def test_cleanup_failure_blocks_new_work(self):
        job = self.create_without_start()
        with patch.object(self.service, '_owned_processes', side_effect=psutil.AccessDenied(123)):
            stopped = self.service.stop(job['id'])
        self.assertFalse(stopped['cleanup']['ok'])
        with self.assertRaises(PlatformError):
            self.create_without_start(payload=dict(self.payload, idempotencyKey='another-0001'))

    def test_stop_and_recovery_do_not_touch_unrelated_process(self):
        job = self.create_without_start()
        spec = str(self.service.directory(job['id']) / 'spec.json')
        unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
        owned = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)',
            'federatedscope.standalone_api.platform_worker', spec])
        try:
            time.sleep(.1)
            JsonRepository._atomic_write(self.service.directory(job['id']) / 'process.json',
                {'pid': owned.pid, 'created': psutil.Process(owned.pid).create_time(), 'spec': spec})
            job.update(status='running', startedAt=now(), cleanup={'ok': False})
            self.service._save(job)
            self.service.recover()
            recovered = self.service.get(job['id'])
            self.assertEqual(recovered['status'], 'interrupted')
            self.assertTrue(recovered['cleanup']['ok'])
            self.assertIsNone(unrelated.poll())
            self.assertEqual(recovered['request'], job['request'])
        finally:
            for proc in (owned, unrelated):
                if proc.poll() is None:
                    proc.terminate()
                proc.wait(timeout=5)

    def test_macro_metric_definition(self):
        result = classification_metrics([[2, 1, 0], [1, 0, 0], [0, 0, 0]])
        self.assertEqual(result['accuracy'], .5)
        self.assertAlmostEqual(result['macroF1'], 1 / 3)
        self.assertEqual(result['samples'], 4)
        self.assertEqual(result['perClass'][2]['support'], 0)

    def test_independent_evaluation_and_hash_guard(self):
        import torch
        output = Path(self.temp.name)
        model = torch.nn.Linear(2, 2)
        with torch.no_grad():
            model.weight.copy_(torch.eye(2)); model.bias.zero_()
        checkpoint, bundle = output / 'model.pt', output / 'test.pt'
        torch.save({'backbone': {'test': 'frozen'}, 'dataset': 'test',
            'architecture': {'model_type': 'ggeur_mlp', 'input_dim': 2, 'hidden_dim': 0, 'num_classes': 2},
            'state_dict': model.state_dict()}, checkpoint)
        torch.save({'backbone': {'test': 'frozen'}, 'dataset': 'test',
            'features': {'A': torch.tensor([[2., 0.], [0., 2.]])},
            'labels': {'A': torch.tensor([0, 1])}}, bundle)
        spec = dict(output=str(output), checkpointPath=str(checkpoint), bundlePath=str(bundle),
                    checkpointHash=digest(checkpoint), bundleHash=digest(bundle), request={})
        evaluate(spec)
        result = json.loads((output / 'result.json').read_text())
        self.assertEqual(result['accuracy'], 1.)
        self.assertEqual(result['macroF1'], 1.)
        with self.assertRaises(ValueError):
            evaluate(dict(spec, checkpointHash='tampered'))


if __name__ == '__main__':
    unittest.main()
