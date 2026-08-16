import json
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

import numpy as np

from federatedscope.core.data.dirichlet_partition import (
    class_histograms, partition_indices_by_label)
from federatedscope.standalone_api.app import create_server
from federatedscope.standalone_api.repository import JsonRepository
from federatedscope.standalone_api.runner import StandaloneProcessRunner
from federatedscope.standalone_api.schemas import (
    ValidationError, validate_experiment, validate_scenario)
from federatedscope.standalone_api.task_manager import ExperimentTaskManager
from federatedscope.standalone_api.task_manager import TaskConflict


def experiment_payload(experiment_type='heterogeneity'):
    payload = {
        'schemaVersion': '1.0',
        'idempotencyKey': 'test-key',
        'name': '测试实验',
        'type': experiment_type,
        'scenarioId': 'SCN-TEST',
        'common': {
            'method': 'heterogeneous_solution',
            'rounds': 2,
            'localEpochs': 1,
            'participationRate': 1.0,
            'batchSize': 8,
            'learningRate': 0.001,
            'seed': 42,
            'device': 'cpu',
        },
        'heterogeneity': {
            'expansionTarget': 10,
        } if experiment_type == 'heterogeneity' else None,
        'privacy': None,
        'backdoor': None,
    }
    if experiment_type == 'privacy':
        payload['privacy'] = {
            'attack': 'membership',
            'defenseEnabled': True,
            'initialClip': 1.0,
            'targetQuantile': 0.7,
            'noiseMultiplier': 0.05,
            'epsilon': 6.0,
        }
    if experiment_type == 'backdoor':
        payload['backdoor'] = {
            'attack': 'trigger_injection',
            'defenseEnabled': True,
            'maliciousRatio': 0.05,
            'maliciousClients': ['OH-DT-C01'],
            'startRound': 1,
            'poisonRatio': 0.2,
            'targetLabel': 0,
            'featureStageDefense': True,
            'trainingStageDefense': True,
        }
    return payload


class FakeRunner:
    def preflight(self, config):
        return {'ready': True}

    def run(self, config, output_dir, stop_event, emit, on_metric):
        for round_index in range(1, 3):
            if stop_event.is_set():
                return -15
            emit('round.started', {'phaseIndex': 2, 'round': round_index})
            metric = {'round': round_index, 'accuracy': 60 + round_index}
            on_metric(metric)
            emit('metric.updated', metric)
            time.sleep(0.01)
        return 0


class SlowRunner(FakeRunner):
    def run(self, config, output_dir, stop_event, emit, on_metric):
        while not stop_event.wait(0.01):
            pass
        return -15


class StandaloneApiSchemaTest(unittest.TestCase):
    def test_within_domain_partition_is_complete_and_deterministic(self):
        labels = np.repeat(np.arange(12), 100)
        first = partition_indices_by_label(labels, 15, 0.2, 73)
        second = partition_indices_by_label(labels, 15, 0.2, 73)
        self.assertEqual(first, second)
        flattened = [index for partition in first for index in partition]
        self.assertEqual(sorted(flattened), list(range(len(labels))))
        self.assertTrue(all(partition for partition in first))

    def test_lower_alpha_creates_stronger_client_label_skew(self):
        labels = np.repeat(np.arange(20), 400)
        low = partition_indices_by_label(labels, 15, 0.05, 19)
        high = partition_indices_by_label(labels, 15, 20.0, 19)
        low_hist = class_histograms(labels, low, 20)
        high_hist = class_histograms(labels, high, 20)

        def average_dominant_ratio(histograms):
            totals = histograms.sum(axis=1)
            return float(np.mean(histograms.max(axis=1) / totals))

        self.assertGreater(average_dominant_ratio(low_hist),
                           average_dominant_ratio(high_hist))

    def test_scenario_schema(self):
        value = validate_scenario({
            'dataset': 'office-home',
            'domains': ['Art', 'Clipart', 'Product', 'Real_World'],
            'clientsPerDomain': 15,
            'partition': {'strategy': 'dirichlet', 'alpha': 0.3, 'seed': 42},
        })
        self.assertEqual(value['partition']['alpha'], 0.3)

    def test_scenario_preview_uses_real_directory_counts_when_available(self):
        from federatedscope.cv.dataset.office_home import OfficeHome
        from federatedscope.standalone_api.scenarios import build_partition
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for domain in OfficeHome.DOMAINS:
                directory = root / (domain.replace('_', ' ')
                                    if 'Real' in domain else domain)
                for class_name in OfficeHome.CLASSES[:2]:
                    class_directory = directory / class_name
                    class_directory.mkdir(parents=True)
                    for index in range(20):
                        (class_directory / f'{index}.jpg').touch()
            with mock.patch.dict('os.environ', {
                    'FEDERATEDSCOPE_DATA_ROOT': str(root)}):
                preview = build_partition(validate_scenario({}))
        self.assertEqual(preview['basis'], 'actual_dataset')
        self.assertEqual([item['totalSamples'] for item in preview['domains']],
                         [28, 28, 28, 28])
        self.assertTrue(all(
            sum(client['sampleCount'] for client in domain['clients']) == 28
            for domain in preview['domains']))

    def test_privacy_and_backdoor_fields_are_mutually_exclusive(self):
        payload = experiment_payload('privacy')
        payload['backdoor'] = experiment_payload('backdoor')['backdoor']
        with self.assertRaises(ValidationError):
            validate_experiment(payload)

    def test_backdoor_ratio_resolves_to_explicit_simulated_nodes(self):
        payload = experiment_payload('backdoor')
        payload['backdoor']['maliciousClients'] = []
        payload['backdoor']['maliciousRatio'] = 0.05
        validated = validate_experiment(payload)
        self.assertEqual(validated['backdoor']['maliciousClients'], [
            'OH-DT-C01', 'OH-DT-C02', 'OH-DT-C03'])

    def test_method_specific_fields_are_cleaned(self):
        payload = experiment_payload()
        payload['common']['method'] = 'fedavg'
        payload['common']['fedproxMu'] = 0.5
        validated = validate_experiment(payload)
        self.assertNotIn('fedproxMu', validated['common'])

    def test_runner_maps_public_backdoor_modes_and_client_ids(self):
        payload = validate_experiment(experiment_payload('backdoor'))
        payload['backdoor']['attack'] = 'model_update_poisoning'
        payload['experimentId'] = 'EXP-TEST'
        payload['_scenario'] = {
            'request': validate_scenario({}),
            'preview': {'partitionVersion': 'test'},
        }
        with tempfile.TemporaryDirectory() as temporary:
            data_root = Path(temporary) / 'data'
            model_path = Path(temporary) / 'model.bin'
            data_root.mkdir()
            model_path.touch()
            runner = StandaloneProcessRunner(
                Path(__file__).resolve().parents[1])
            with mock.patch.dict('os.environ', {
                'FEDERATEDSCOPE_DATA_ROOT': str(data_root),
                'FEDERATEDSCOPE_MODEL_PATH': str(model_path),
            }):
                command = runner.build_command(
                    payload, Path(temporary) / 'output')
        joined = ' '.join(command)
        self.assertIn('attack.attack_method label_flip', joined)
        self.assertIn('attack.label_flip.update_reversal True', joined)
        self.assertIn('attack.attacker_id [1]', joined)
        self.assertNotIn('dp.enabled True', joined)
        self.assertIn('dp.enabled False', joined)
        self.assertIn('dp.protect_ggeur_update False', joined)
        self.assertIn('adaptive_dp.use False', joined)
        self.assertIn('ggeur.multi_metrics_stats_defense True', joined)
        self.assertIn('ggeur.defense_method multi_metrics', joined)

    def test_runner_explicitly_switches_privacy_protection(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_root = Path(temporary) / 'data'
            data_root.mkdir()
            runner = StandaloneProcessRunner(Path(__file__).resolve().parents[1])
            commands = []
            for enabled in (False, True):
                payload = validate_experiment(experiment_payload('privacy'))
                payload['privacy']['defenseEnabled'] = enabled
                if not enabled:
                    for field in ('initialClip', 'targetQuantile',
                                  'noiseMultiplier', 'epsilon'):
                        payload['privacy'].pop(field, None)
                payload['experimentId'] = f'EXP-{enabled}'
                payload['_scenario'] = {
                    'request': validate_scenario({}),
                    'preview': {'partitionVersion': 'test'},
                }
                with mock.patch.dict('os.environ', {
                    'FEDERATEDSCOPE_DATA_ROOT': str(data_root),
                }):
                    commands.append(' '.join(runner.build_command(
                        payload, Path(temporary) / 'output')))
        self.assertIn('dp.enabled False', commands[0])
        self.assertIn('dp.protect_ggeur_update False', commands[0])
        self.assertIn('dp.enabled True', commands[1])
        self.assertIn('dp.protect_ggeur_update True', commands[1])
        self.assertIn('dp.clipping.type adaptive', commands[1])

    def test_log_parser_extracts_mode_specific_metrics(self):
        parsed = StandaloneProcessRunner._parse_line(
            'Round 3 test_acc=0.81 ASR=0.22 TPR=0.90 FPR=0.08 AUC=0.61')
        self.assertEqual(parsed['round'], 3)
        self.assertEqual(parsed['metrics']['accuracy'], 0.81)
        self.assertEqual(parsed['metrics']['attackSuccess'], 0.22)
        self.assertEqual(parsed['metrics']['privacyRisk'], 0.61)

    def test_log_parser_aggregates_real_domain_and_privacy_log_formats(self):
        parsed = StandaloneProcessRunner._parse_line(
            'Server: Round 4 MLP Test Accuracy - Art: 0.4, '
            'Clipart: 0.5, Product: 0.7, Real_World: 0.6, average: 0.55')
        self.assertEqual(parsed['metrics']['accuracy'], 0.55)
        self.assertEqual(parsed['metrics']['worstDomain'], 0.4)
        self.assertAlmostEqual(parsed['metrics']['domainGap'], 0.3)
        attack = StandaloneProcessRunner._parse_line(
            'fedmia_i: AUC=0.73, TPR@0.01=0.42')
        self.assertEqual(attack['metrics']['privacyRisk'], 0.73)
        self.assertEqual(attack['metrics']['truePositiveRate'], 0.42)
        noise = StandaloneProcessRunner._parse_line(
            '[AdaptiveDP] client=2 round=3 noise_std=0.125')
        self.assertEqual(noise['metrics']['noiseStd'], 0.125)


class StandaloneTaskManagerTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.repository = JsonRepository(root / 'state')
        self.scenario = {
            'scenarioId': 'SCN-TEST',
            'request': validate_scenario({}),
            'preview': {'partitionVersion': 'test'},
        }
        self.repository.save_scenario(self.scenario)

    def tearDown(self):
        self.temporary.cleanup()

    def test_task_completes_and_is_persisted(self):
        manager = ExperimentTaskManager(
            self.repository, FakeRunner(), Path(self.temporary.name) / 'runs')
        record = manager.create(validate_experiment(experiment_payload()),
                                self.scenario)
        for _ in range(100):
            current = manager.get(record['experimentId'])
            if current['status'] == 'completed':
                break
            time.sleep(0.01)
        self.assertEqual(current['status'], 'completed')
        self.assertEqual(current['finalMetrics']['accuracy'], 62)
        self.assertEqual(len(manager.list()), 1)
        snapshot = manager.snapshot(record['experimentId'])
        self.assertEqual(snapshot['round'], 2)
        event_rows = manager.events_after(record['experimentId'], 0)
        self.assertEqual(
            [event['sequence'] for event in event_rows],
            list(range(1, len(event_rows) + 1)))
        self.assertEqual(event_rows[-1]['type'], 'experiment.completed')

    def test_task_can_be_stopped(self):
        manager = ExperimentTaskManager(
            self.repository, SlowRunner(), Path(self.temporary.name) / 'runs')
        record = manager.create(validate_experiment(experiment_payload()),
                                self.scenario)
        duplicate = manager.create(
            validate_experiment(experiment_payload()), self.scenario)
        self.assertEqual(duplicate['experimentId'], record['experimentId'])
        competing_payload = experiment_payload()
        competing_payload['idempotencyKey'] = 'different-key'
        with self.assertRaises(TaskConflict):
            manager.create(validate_experiment(competing_payload),
                           self.scenario)
        manager.stop(record['experimentId'])
        for _ in range(100):
            current = manager.get(record['experimentId'])
            if current['status'] == 'stopped':
                break
            time.sleep(0.01)
        self.assertEqual(current['status'], 'stopped')


class StandaloneHttpApiTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        try:
            self.server = create_server('127.0.0.1', 0,
                                        Path(self.temporary.name) / 'api')
        except PermissionError:
            self.temporary.cleanup()
            self.skipTest('当前沙箱禁止创建本地监听套接字')
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        self.base = f'http://127.0.0.1:{self.server.server_port}'

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.temporary.cleanup()

    def request(self, path, method='GET', payload=None):
        body = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(
            self.base + path, data=body, method=method,
            headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.load(response)['data']

    def test_health_and_scenario_lifecycle(self):
        status, health = self.request('/api/health')
        self.assertEqual(status, 200)
        self.assertEqual(health['status'], 'ok')
        status, preview = self.request(
            '/api/scenarios/preview', 'POST', validate_scenario({}))
        self.assertEqual(status, 200)
        self.assertEqual(len(preview['domains']), 4)
        self.assertEqual(len(preview['domains'][0]['clients']), 15)
        status, scenario = self.request(
            '/api/scenarios', 'POST', validate_scenario({}))
        self.assertEqual(status, 201)
        self.assertTrue(scenario['scenarioId'].startswith('SCN-'))

    def test_invalid_experiment_returns_field_errors(self):
        request = urllib.request.Request(
            self.base + '/api/experiments', data=b'{}', method='POST',
            headers={'Content-Type': 'application/json'})
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(request, timeout=5)
        self.assertEqual(caught.exception.code, 422)
        payload = json.load(caught.exception)
        self.assertIn('name', payload['error']['fieldErrors'])


if __name__ == '__main__':
    unittest.main()
