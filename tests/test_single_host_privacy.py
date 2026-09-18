"""HTTP integration contracts using explicit fixtures, not real FedMIA validation."""
import json
import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError
from urllib.request import ProxyHandler, build_opener

import numpy as np

from federatedscope.standalone_api.platform_app import create_server


class PrivacyIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / 'fedmia_local'
        self.env = patch.dict(os.environ, {'FS_FEDMIA_LOCAL_ROOT': str(self.root)})
        self.env.start()
        self.server = create_server('127.0.0.1', 0, Path(self.temp.name) / 'state')
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.opener = build_opener(ProxyHandler({}))
        self.base = 'http://127.0.0.1:' + str(self.server.server_port)
        self.context = self.server.RequestHandlerClass.context

    def tearDown(self):
        self.server.shutdown()
        self.context.platform.close()
        self.context.privacy.close()
        self.context.backdoor.close()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.env.stop()
        self.temp.cleanup()

    def get(self, path):
        try:
            response = self.opener.open(self.base + path, timeout=10)
        except HTTPError as error:
            response = error
        with response:
            return response.status, response.read()

    def data(self, query=''):
        status, raw = self.get('/api/platform/privacy/membership' + query)
        return status, json.loads(raw)

    def fixture(self):
        self.root.mkdir()
        (self.root / 'show_fedmia_examples.py').write_text('# Fixture only\n')
        for kind in ('no_defense', 'defense'):
            run = self.root / 'runs' / kind
            features = run / 'ggeur_fedmia_features'
            features.mkdir(parents=True)
            (run / 'config.yaml').write_text('fixture: true\n')
            for client in (1, 2):
                (features / f'client_{client}_features_round1.pt').touch()
        image = self.root / 'datasets/OfficeHomeDataset_10072016/Art/Radio/one.jpg'
        image.parent.mkdir(parents=True)
        image.write_bytes(b'fixture-image-bytes')
        self.image = image
        self.cfg = SimpleNamespace(data=SimpleNamespace(root='old'), defrost=Mock(),
                                   freeze=Mock(spec=lambda save=True, inform=True: None))
        # A concrete signature also verifies config save is disabled.
        self.freeze_calls = []

        def freeze(save=True, inform=True):
            self.freeze_calls.append((save, inform))

        self.cfg.freeze = freeze
        self.module = SimpleNamespace(np=np, load_cfg=Mock(return_value=self.cfg),
            load_feature_artifacts=Mock(side_effect=lambda path, cfg, **kw: path),
            load_image_items=Mock(return_value=([(str(image), 0, 0)], [(str(image), 0, 1)])),
            compute_scores=Mock(side_effect=lambda path, cfg, client, method:
                SimpleNamespace(scores_member=[.9] if path.parent.name == 'no_defense' else [.4],
                                scores_nonmember=[.1] if path.parent.name == 'no_defense' else [.6])),
            class_names_for=lambda cfg: ['Radio'], label_name=lambda index, names: names[index],
            comparison_priority=lambda indices, *args: indices)
        self.context.fedmia_examples_module = self.module

    def test_missing_resources_do_not_break_accuracy_endpoints(self):
        status, result = self.data()
        self.assertEqual(status, 200)
        self.assertFalse(result['data']['configured'])
        self.assertEqual(result['data']['items'], [])
        self.assertIn('show_fedmia_examples.py', result['data']['missing'])
        for path in ('/api/health', '/api/platform/catalog', '/api/platform/library', '/api/platform/jobs'):
            self.assertEqual(self.get(path)[0], 200, path)

    def test_invalid_queries_fail_before_loading_resources(self):
        for query in ('clientId=abc', 'clientId=0', 'limit=0', 'limit=201', 'seed=-1',
                      'threshold=nan', 'threshold=inf', 'threshold=1.1', 'group=other',
                      'limit=', 'limit=1&limit=2', 'unknown=1'):
            with self.subTest(query=query):
                self.assertEqual(self.data('?' + query)[0], 422)

    def test_payload_images_and_computations_are_reused(self):
        self.fixture()
        status, response = self.data()
        self.assertEqual(status, 200)
        payload = response['data']
        self.assertEqual(payload['clients'], [1, 2])
        self.assertEqual(payload['metrics']['noDefense']['auc'], 1)
        self.assertEqual(payload['metrics']['defense']['auc'], 0)
        self.assertEqual(payload['items'][0]['noDefensePrediction'], 'member')
        self.assertEqual(payload['items'][0]['defensePrediction'], 'nonmember')
        self.assertEqual(self.get(payload['items'][0]['imageUrl']), (200, b'fixture-image-bytes'))
        self.data('?clientId=1')
        self.data('?clientId=1&group=nonmember')
        self.assertEqual(self.module.compute_scores.call_count, 2)
        self.assertEqual(self.freeze_calls, [(False, False), (False, False)])
        self.assertEqual(self.data('?clientId=2')[0], 200)
        self.assertEqual(self.module.load_feature_artifacts.call_count, 2)
        self.assertEqual(self.module.compute_scores.call_count, 4)

    def test_only_clients_with_both_feature_sets_are_selectable(self):
        self.fixture()
        (self.root / 'runs/defense/ggeur_fedmia_features/client_2_features_round1.pt').unlink()
        self.assertEqual(self.data()[1]['data']['clients'], [1])
        self.assertEqual(self.data('?clientId=2')[0], 404)

    def test_image_endpoint_rejects_paths_outside_dataset(self):
        self.fixture()
        outside = Path(self.temp.name) / 'private.jpg'
        outside.write_bytes(b'not-a-dataset-image')
        self.module.load_image_items.return_value = ([(str(outside), 0, 0)], [(str(self.image), 0, 1)])
        self.assertEqual(self.get('/api/platform/privacy/membership/images/1/member/0')[0], 403)

    def test_misaligned_or_nonfinite_scores_are_not_presented_as_results(self):
        self.fixture()
        for scores in ([.1, .2], [float('nan')]):
            self.module.compute_scores.side_effect = None
            self.module.compute_scores.return_value = SimpleNamespace(scores_member=scores, scores_nonmember=[.1])
            self.assertEqual(self.data()[0], 409)


if __name__ == '__main__':
    unittest.main()
