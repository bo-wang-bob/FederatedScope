import json
import tempfile
import threading
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from federatedscope.standalone_api.platform_privacy import PrivacyService
from federatedscope.standalone_api.privacy_artifacts import archive_existing_run, run_directory
from federatedscope.standalone_api.repository import JsonRepository


class PrivacyArtifactTests(TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name) / 'backend'
        self.repo.mkdir()
        (self.repo.parent / 'fedmia_local').mkdir()
        self.service = object.__new__(PrivacyService)
        self.service.repo = self.repo
        self.service.state = self.repo / 'privacy'
        self.service.lock = threading.RLock()
        self.job = dict(id='a' * 32, request=dict(group='military_cnn', defense=False),
                        action='train', status='failed', createdAt='2026-09-18', events=[])
        self.output = self.service.directory(self.job['id'])
        self.service._save(self.job)

    def test_disk_count_survives_empty_recent_events(self):
        features = self.output / 'logs/run/ggeur_fedmia_features'
        features.mkdir(parents=True)
        for client in (1, 2):
            for rnd in (40, 80):
                (features / f'client_{client}_features_round{rnd}.pt').write_bytes(b'test fixture')
        archive = archive_existing_run(self.repo, self.job, self.output)
        result = self.service.get(self.job['id'])
        self.assertEqual(result['events'], [])
        self.assertEqual(result['featureStorage']['files'], 4)
        self.assertEqual(result['featureStorage']['rounds'], [40, 80])
        self.assertEqual(len(list((archive / 'ggeur_fedmia_features').glob('*.pt'))), 4)
        self.assertEqual(len(list(features.glob('*.pt'))), 4)
        self.assertIn('fedmia_local', str(archive))
        self.assertNotEqual(archive, run_directory(self.repo, dict(self.job['request'], defense=True), self.job['id']))

    def test_locked_primary_uses_durable_recovery_snapshot(self):
        write = JsonRepository._atomic_write
        self.job['stage'] = 'features saved'

        def deny_primary(path, data):
            if path.name == 'job.json':
                error = PermissionError('locked')
                error.winerror = 5
                raise error
            write(path, data)

        with patch.object(JsonRepository, '_atomic_write', side_effect=deny_primary):
            self.service._save(self.job)
        self.assertEqual(self.service.get(self.job['id'])['stage'], 'features saved')
        self.assertEqual(self.service.list()[0]['stage'], 'features saved')
        self.job['stage'] = 'completed later'
        self.service._save(self.job)
        self.assertEqual(self.service.get(self.job['id'])['stage'], 'completed later')

    def test_partial_report_readable_without_marking_training_complete(self):
        archive = archive_existing_run(self.repo, self.job, self.output)
        JsonRepository._atomic_write(archive / 'privacy_results.json',
            dict(completeTraining=False, clients={'1': {'rounds': [40, 80]}}))
        self.assertTrue(self.service.get(self.job['id'])['featureStorage']['resultsReady'])
        result = self.service.results(self.job['id'], 1)
        self.assertFalse(result['completeTraining'])
        self.assertEqual(self.service.get(self.job['id'])['status'], 'failed')
