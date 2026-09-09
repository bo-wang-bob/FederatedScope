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
                if not method['enabled']:
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
                      {'group': '../officehome_vit'}, {'method': 'heterogeneous_solution'}):
            with self.subTest(extra=extra), self.assertRaises(PlatformError):
                self.configs.normalize(dict(self.payload, **extra))

    def test_fixed_digit_manifest_cannot_accept_ineffective_partition_params(self):
        for extra in ({'alpha': .5}, {'splitSeed': 5}, {'clientCount': 30}):
            with self.assertRaises(PlatformError):
                self.configs.normalize(dict(group='digit3_vit', method='fedavg', **extra))

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
