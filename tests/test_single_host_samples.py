import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from federatedscope.standalone_api.platform_config import PlatformError, sha256
from federatedscope.standalone_api.platform_service import PlatformService, now
from federatedscope.standalone_api.platform_worker import predict
from federatedscope.standalone_api.repository import JsonRepository


class SampleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.service = PlatformService(Path(__file__).resolve().parents[1], self.root / 'state')
        self.service.configs.datasets = self.root / 'datasets'
        self.job_id = 'a' * 32
        self.output = self.service.directory(self.job_id)
        checkpoint_dir = self.output / 'checkpoints'
        checkpoint_dir.mkdir(parents=True)
        for name in ['Class_B/One.jpg', 'Class_A/Two.jpg']:
            path = self.service.configs.datasets / 'OfficeHomeDataset_10072016/Art' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'fixture-image-version')
        self.manifest = {'classes': ['Class_A', 'Class_B'], 'cacheFiles': {'raw': 'fixture'},
            'partition': {'1': [['Art', 'art/class_a/train.jpg', 0]]},
            'test': {'Art': {'art/class_b/one.jpg': 1, 'art/class_a/two.jpg': 0}},
            'testProvenance': {'Art': 'fixture-embedded-sample-ids'}}
        JsonRepository._atomic_write(self.output / 'data_manifest.json', self.manifest)
        model = torch.nn.Linear(2, 2)
        with torch.no_grad():
            model.weight.copy_(torch.eye(2)); model.bias.zero_()
        torch.save({'backbone': {'test': 'frozen'}, 'dataset': 'fixture',
            'architecture': {'model_type': 'ggeur_mlp', 'input_dim': 2, 'hidden_dim': 0, 'num_classes': 2},
            'state_dict': model.state_dict()}, checkpoint_dir / 'mlp_final.pt')
        torch.save({'backbone': {'test': 'frozen'}, 'dataset': 'fixture',
            'features': {'Art': torch.tensor([[2., 0.], [0., 2.]])},
            'labels': {'Art': torch.tensor([1, 0])}}, checkpoint_dir / 'pretrained_test_features.pt')
        fingerprint = hashlib.sha256(json.dumps({'files': self.manifest['cacheFiles'],
            'test': self.manifest['test'], 'classes': self.manifest['classes'],
            'group': 'officehome_vit'}, sort_keys=True).encode()).hexdigest()
        self.job = {'id': self.job_id, 'action': 'train', 'status': 'completed',
            'createdAt': now(), 'cleanup': {'ok': True}, 'idempotencyKey': 'fixture-training',
            'request': {'group': 'officehome_vit', 'method': 'fedavg', 'name': 'fixture', 'rounds': 2},
            'result': {'classes': self.manifest['classes'], 'domains': [{'name': 'Art', 'testSamples': 2}],
                'testFingerprint': fingerprint, 'featureSpace': 'fixture', 'testSamples': 2,
                'artifactHashes': {p.name: sha256(p) for p in checkpoint_dir.iterdir()}}}
        self.service._save(self.job)

    def tearDown(self):
        self.service.close()
        self.temp.cleanup()

    def page(self, **params):
        return self.service.samples.page(self.job_id, {k: [str(v)] for k, v in params.items()})

    def payload(self):
        sample = self.page()['items'][0]
        return dict(modelId=self.job_id + ':final', testsetId=self.job_id, sampleId=sample['id'],
                    imageSha256=sample['imageSha256'], idempotencyKey='prediction-fixture')

    def test_browse_real_paths_and_filter_preserves_bundle_index(self):
        page = self.page(**{'domain': 'Art', 'class': 0, 'limit': 1})
        self.assertEqual(page['total'], 1)
        self.assertEqual(page['items'][0]['index'], 1)
        self.assertTrue(page['items'][0]['imageAvailable'])
        self.assertNotIn('key', page['items'][0])
        self.assertNotIn(str(self.root), json.dumps(page))

    def test_invalid_queries_and_unregistered_samples_rejected(self):
        for query in [{'limit': 0}, {'offset': -1}, {'class': 3}, {'domain': 'missing'}, {'path': '/etc/passwd'}]:
            with self.subTest(query=query), self.assertRaises(PlatformError):
                self.page(**query)
        with self.assertRaises(PlatformError):
            self.service.samples.resolve(self.job_id, 'f' * 24)

    def test_legacy_numeric_names_still_reject_modified_names(self):
        self.manifest['classes'] = ['0', '1']
        self.job['result']['classes'] = ['0', '1']
        self.job['result']['testFingerprint'] = hashlib.sha256(json.dumps({
            'files': self.manifest['cacheFiles'], 'test': self.manifest['test'],
            'classes': None, 'group': self.job['request']['group'],
        }, sort_keys=True).encode()).hexdigest()
        self.service._save(self.job)
        JsonRepository._atomic_write(self.output / 'data_manifest.json', self.manifest)
        self.assertEqual(self.page()['items'][0]['className'], '1')
        self.manifest['classes'][1] = 'renamed'
        JsonRepository._atomic_write(self.output / 'data_manifest.json', self.manifest)
        with self.assertRaises(PlatformError):
            self.page()

    def test_traversal_and_escaping_symlink_rejected(self):
        with self.assertRaises(PlatformError):
            self.service.samples.image_path(self.job, {'key': '../secret.jpg'})
        outside = self.root / 'outside.jpg'
        outside.write_bytes(b'outside')
        link = self.service.configs.datasets / 'OfficeHomeDataset_10072016/Art/escape.jpg'
        try:
            link.symlink_to(outside)
        except OSError:
            return  # Windows without symlink privileges still tests traversal.
        with self.assertRaises(PlatformError):
            self.service.samples.image_path(self.job, {'key': 'art/escape.jpg'})

    def test_changed_manifest_and_image_are_rejected(self):
        payload = self.payload()
        path = self.service.samples.image_path(self.job, next(self.service.samples.rows(self.manifest)))
        path.write_bytes(b'changed-image')
        with self.assertRaises(PlatformError):
            self.service.create('predict', payload)
        self.manifest['test']['Art']['art/class_b/one.jpg'] = 0
        JsonRepository._atomic_write(self.output / 'data_manifest.json', self.manifest)
        with self.assertRaises(PlatformError):
            self.page()

    def test_predict_uses_model_not_label_and_records_versions(self):
        with patch('federatedscope.standalone_api.platform_service.threading.Thread'):
            job = self.service.create('predict', self.payload())
        spec = json.loads((self.service.directory(job['id']) / 'spec.json').read_text())
        predict(spec)
        result = json.loads((self.service.directory(job['id']) / 'result.json').read_text())
        self.assertEqual(result['predictedClass'], 0)
        self.assertEqual(result['label'], 1)
        self.assertFalse(result['correct'])
        self.assertAlmostEqual(result['confidence'], .880797, places=5)
        self.assertEqual(result['topK'][0]['classIndex'], 0)
        self.assertEqual(result['checkpointSha256'], spec['checkpointHash'])
        with self.assertRaises(ValueError):
            predict(dict(spec, checkpointHash='changed'))


if __name__ == '__main__':
    unittest.main()
