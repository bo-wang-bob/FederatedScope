import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from federatedscope.standalone_api.platform_backdoor import BackdoorService
from federatedscope.standalone_api.platform_config import PlatformError


class BackdoorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.repo = Path(self.temp.name)
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.service = BackdoorService(self.repo, 'state')
        self.base = self.repo / 'resources/backdoor/exp/sabre'
        images = self.base / 'testset_images'
        images.mkdir(parents=True)
        (images / 'index.csv').write_text('id,label\nArt_00001,0\nReal_World_00001,1\n')
        for name in ('Art_00001', 'Real_World_00001'):
            (images / (name + '.jpg')).write_bytes(b'fixture')

    def tearDown(self):
        self.service.close()
        self.env.stop()
        self.temp.cleanup()

    def test_defaults_are_portable_and_share_privacy_images(self):
        self.assertEqual(self.service.base, self.base)
        self.assertEqual(self.service.root, self.repo / 'state/backdoor')
        self.assertEqual(Path(self.service.data_root), self.repo / 'resources/fedmia_local/datasets/OfficeHomeDataset_10072016')

    def test_pick_filters_and_rejects_invalid_payload(self):
        result = self.service.pick(dict(domain='Real_World', label=1, seed=1, count=2))
        self.assertEqual(result['ids'], ['Real_World_00001'])
        for payload in ([], {'count':True}, {'count':1.2}, {'count':21}, {'seed':True}, {'label':True}):
            with self.subTest(payload=payload), self.assertRaises(PlatformError):
                self.service.pick(payload)

    def test_image_ids_cannot_escape_or_be_missing_or_duplicated(self):
        self.assertTrue(self.service.image_path('Real_World_00001').is_file())
        for ids in (['../config'], ['Art_99999'], ['Art_00001','Art_00001']):
            with self.subTest(ids=ids), self.assertRaises(PlatformError):
                self.service._validate_ids(ids)

    def test_military_results_use_portable_dataset_root(self):
        run = self.repo / 'resources/backdoor/exp/sabre_newdataset/attack'
        run.mkdir(parents=True)
        (run / 'config.yaml').write_text('data:\n  type: MilitaryAircraft-3D\n  root: C:/old/data\n')
        service = BackdoorService(self.repo, 'military-state')
        self.addCleanup(service.close)
        self.assertEqual(service.base, run.parent)
        self.assertEqual(Path(service.data_root), self.repo / 'resources/datasets/MilitaryAircraft3D')

    def test_missing_cache_does_not_launch_background_gpu_work(self):
        with patch('subprocess.run', side_effect=AssertionError('Unexpected GPU process')):
            result = self.service.pick(dict(count=2, seed=1))
        self.assertFalse(result['filtered'])
        self.assertFalse(self.service._precompute_started)

    def test_weighted_selection_fills_count_without_duplicates(self):
        self.service._index = [(f'Art_{index:05d}', 1) for index in range(10)]
        self.service._predictions_loaded = True
        self.service._predictions = dict(targetLabel=0, items={
            row[0]: dict(attack=0, defense=1) for row in self.service._index})
        result = self.service.pick(dict(count=10, seed=42))
        self.assertEqual(result['count'], 10)
        self.assertEqual(len(set(result['ids'])), 10)
        self.assertTrue(result['filtered'])
        self.assertEqual(result, self.service.pick(dict(count=10, seed=42)))

    def test_missing_defense_prediction_is_not_a_defense_success(self):
        self.assertEqual(self.service._rank(dict(attack=0), 1, 0), 4)

    def test_run_override_cannot_escape_resource_root(self):
        with patch.dict(os.environ, {'FS_BACKDOOR_RUNS':'attack=../../outside'}):
            with self.assertRaises(PlatformError):
                self.service.runs()

    def test_recovery_marks_only_backdoor_jobs_interrupted(self):
        job_id = 'a' * 32
        path = self.service.directory(job_id) / 'job.json'
        path.parent.mkdir()
        path.write_text(json.dumps(dict(id=job_id,status='running',createdAt='now')))
        self.service.recover()
        self.assertEqual(self.service.get(job_id)['status'], 'interrupted')


if __name__ == '__main__':
    unittest.main()
