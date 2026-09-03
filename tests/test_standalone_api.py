import json
import io
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from contextlib import redirect_stdout

import numpy as np

from federatedscope.core.data.dirichlet_partition import (
    class_histograms, partition_indices_by_label)
from federatedscope.core.monitoring.events import (
    emit_training_event, parse_training_event_line)
from federatedscope.standalone_api.app import create_server
from federatedscope.standalone_api.repository import JsonRepository
from federatedscope.standalone_api.runner import StandaloneProcessRunner
from federatedscope.standalone_api.scenarios import (
    build_partition, build_partition_artifacts)
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


def runtime_resources(root: Path):
    data_root = root / 'data'
    for domain in ('Art', 'Clipart', 'Product', 'Real_World'):
        class_dir = data_root / domain / 'Alarm_Clock'
        class_dir.mkdir(parents=True)
        for index in range(4):
            (class_dir / f'sample-{index}.jpg').touch()
    model_path = root / 'model.bin'
    model_path.touch()
    manifest_path = root / 'manifest.json'
    with mock.patch.dict('os.environ', {
            'FEDERATEDSCOPE_DATA_ROOT': str(data_root)}):
        preview, manifest = build_partition_artifacts(validate_scenario({}))
    manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
    scenario = {
        'request': validate_scenario({}),
        'preview': preview,
        'artifacts': {'partitionManifest': str(manifest_path)},
    }
    return data_root, model_path, scenario


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


class RecoveryRunner(FakeRunner):
    def __init__(self):
        self.cleanup_calls = []

    def cleanup(self, config, output_dir, emit=None):
        self.cleanup_calls.append((config, output_dir, emit))
        return []


class DefenseRunner(FakeRunner):
    def run(self, config, output_dir, stop_event, emit, on_metric):
        emit('defense.decision', {
            'stage': 'feature_statistics', 'round': 0,
            'keptClientIds': list(range(2, 61)),
            'droppedClientIds': [1, 4], 'threshold': 1.5,
        })
        return 0


class RegressiveRoundRunner(FakeRunner):
    def run(self, config, output_dir, stop_event, emit, on_metric):
        emit('round.started', {'phaseIndex': 2, 'round': 1})
        on_metric({'round': 1, 'accuracy': 0.5})
        emit('round.started', {'phaseIndex': 2, 'round': 0})
        return 0


class StandaloneApiSchemaTest(unittest.TestCase):
    def test_structured_training_event_round_trip(self):
        output = io.StringIO()
        with redirect_stdout(output):
            emit_training_event(
                'metric.updated', round=3, accuracy=np.float32(0.75))
        parsed = parse_training_event_line(output.getvalue())
        self.assertEqual(parsed['type'], 'metric.updated')
        self.assertAlmostEqual(parsed['payload']['accuracy'], 0.75)

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
        with self.assertRaises(ValidationError):
            validate_scenario({'clientsPerDomain': 12})

    def test_scenario_preview_uses_real_directory_counts_when_available(self):
        from federatedscope.cv.dataset.office_home import OfficeHome
        from federatedscope.standalone_api.scenarios import (
            build_partition_artifacts)
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
                preview, manifest = build_partition_artifacts(
                    validate_scenario({}))
        self.assertEqual(preview['basis'], 'actual_dataset')
        self.assertEqual([item['totalSamples'] for item in preview['domains']],
                         [28, 28, 28, 28])
        self.assertTrue(all(
            sum(client['sampleCount'] for client in domain['clients']) == 28
            for domain in preview['domains']))
        self.assertIsNotNone(manifest)
        self.assertEqual(len(manifest['clients']), 60)
        self.assertEqual(sum(len(client['train'])
                             for client in manifest['clients']), 112)
        all_paths = [record['path'] for client in manifest['clients']
                     for record in client['train']]
        self.assertEqual(len(all_paths), len(set(all_paths)))

    def test_replay_manifest_loads_all_standalone_clients(self):
        from federatedscope.contrib.data.ggeur_data import \
            _load_officehome_manifest_data
        manifest = {
            'schemaVersion': '2.0', 'root': '/dataset',
            'partitionVersion': 'p1', 'datasetFingerprint': 'f1',
            'domains': {
                key: {'val': [], 'test': [
                    {'path': f'{key}/test.jpg', 'label': 0}]}
                for key in ('Art', 'Clipart', 'Product', 'Real_World')
            },
            'clients': [],
        }
        prefixes = ('Art', 'Clipart', 'Product', 'Real_World')
        for client_id in range(1, 61):
            domain = prefixes[(client_id - 1) // 15]
            manifest['clients'].append({
                'clientId': client_id, 'domain': domain,
                'train': [{'path': f'{domain}/{client_id}.jpg', 'label': 1}],
            })
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'manifest.json'
            path.write_text(json.dumps(manifest), encoding='utf-8')
            config = SimpleNamespace(
                data=SimpleNamespace(root='/unused'),
                dataloader=SimpleNamespace(batch_size=2, num_workers=0),
                federate=SimpleNamespace(client_num=1),
                ggeur=SimpleNamespace(officehome_manifest_path=str(path)))
            loaded, updated = _load_officehome_manifest_data(config, None)
        self.assertEqual(len(loaded), 60)
        self.assertEqual(updated.federate.client_num, 60)
        self.assertEqual(len(loaded[60]['train'].dataset), 1)

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
        with tempfile.TemporaryDirectory() as temporary:
            data_root, model_path, scenario = runtime_resources(
                Path(temporary))
            payload['_scenario'] = scenario
            runner = StandaloneProcessRunner(
                Path(__file__).resolve().parents[1])
            with mock.patch.dict('os.environ', {
                'FEDERATEDSCOPE_DATA_ROOT': str(data_root),
                'FEDERATEDSCOPE_MODEL_PATH': str(model_path),
            }):
                command = runner.build_command(
                    payload, Path(temporary) / 'output')
            feature_cache_index = command.index('ggeur.feature_cache_dir')
            augmented_cache_index = command.index(
                'ggeur.augmented_feature_cache_dir')
            self.assertEqual(
                Path(command[feature_cache_index + 1]),
                Path(temporary) / 'feature_cache')
            self.assertEqual(
                Path(command[augmented_cache_index + 1]),
                Path(temporary) / 'output' / 'augmented_feature_cache')
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
            data_root, model_path, scenario = runtime_resources(
                Path(temporary))
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
                payload['_scenario'] = scenario
                with mock.patch.dict('os.environ', {
                    'FEDERATEDSCOPE_DATA_ROOT': str(data_root),
                    'FEDERATEDSCOPE_MODEL_PATH': str(model_path),
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

    def test_log_parser_ignores_round_and_client_config_keys(self):
        timeout = StandaloneProcessRunner._parse_line(
            'round_timeout_seconds: 1000')
        convergence = StandaloneProcessRunner._parse_line(
            "global_convergence_round: 0, client_num: 60")
        configured = StandaloneProcessRunner._parse_line(
            'configured_clients=60')
        explicit = StandaloneProcessRunner._parse_line(
            'Server: Starting training (Round #3); Client 12 ready')

        self.assertNotIn('round', timeout)
        self.assertNotIn('round', convergence)
        self.assertNotIn('clientIndex', convergence)
        self.assertNotIn('clientIndex', configured)
        self.assertEqual(explicit['round'], 3)
        self.assertEqual(explicit['clientIndex'], 12)

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
            'preview': build_partition(validate_scenario({})),
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
        self.assertEqual(len(snapshot['clients']), 60)
        event_rows = manager.events_after(record['experimentId'], 0)
        self.assertEqual(
            [event['sequence'] for event in event_rows],
            list(range(1, len(event_rows) + 1)))
        self.assertEqual(event_rows[-1]['type'], 'experiment.completed')
        log_dir = Path(self.temporary.name) / 'runs' / record['experimentId']
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / 'runner.log').write_text(
            'line-one\nline-two\n', encoding='utf-8')
        self.assertEqual(manager.logs(record['experimentId'], 1),
                         ['line-two'])

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

    def test_restart_cleans_interrupted_distributed_processes(self):
        config = experiment_payload()
        config['execution'] = {
            'mode': 'distributed', 'group': 'officehome_vit'}
        record = {
            'experimentId': 'EXP-INTERRUPTED',
            'executionMode': 'distributed',
            'config': config,
            'status': 'running',
            'topology': {
                'root': {'ready': True, 'status': '根聚合服务运行中'},
                'client': {'ready': True, 'status': '逻辑客户端运行中'},
            },
        }
        self.repository.save_experiment(record)
        runner = RecoveryRunner()

        ExperimentTaskManager(
            self.repository, runner, Path(self.temporary.name) / 'runs')

        recovered = self.repository.get_experiment('EXP-INTERRUPTED')
        self.assertEqual(recovered['status'], 'failed')
        self.assertEqual(recovered['error']['code'], 'API_RESTARTED')
        self.assertIn('已清理', recovered['error']['message'])
        self.assertEqual(len(runner.cleanup_calls), 1)
        self.assertEqual(
            runner.cleanup_calls[0][0]['experimentId'], 'EXP-INTERRUPTED')
        self.assertFalse(recovered['topology']['root']['ready'])
        self.assertEqual(recovered['topology']['root']['status'],
                         '服务重启后已清理')

    def test_terminal_log_round_cannot_regress_public_progress(self):
        manager = ExperimentTaskManager(
            self.repository, RegressiveRoundRunner(),
            Path(self.temporary.name) / 'runs')
        record = manager.create(validate_experiment(experiment_payload()),
                                self.scenario)
        for _ in range(100):
            current = manager.get(record['experimentId'])
            if current['status'] == 'completed':
                break
            time.sleep(0.01)
        self.assertEqual(current['round'], 1)
        self.assertEqual(manager.snapshot(record['experimentId'])['round'], 1)

    def test_defense_event_persists_truth_based_detection_metrics(self):
        manager = ExperimentTaskManager(
            self.repository, DefenseRunner(),
            Path(self.temporary.name) / 'runs')
        payload = experiment_payload('backdoor')
        record = manager.create(validate_experiment(payload), self.scenario)
        for _ in range(100):
            current = manager.get(record['experimentId'])
            if current['status'] == 'completed':
                break
            time.sleep(0.01)
        self.assertEqual(current['finalMetrics']['truePositiveRate'], 1.0)
        self.assertAlmostEqual(
            current['finalMetrics']['falsePositiveRate'], 1 / 59)
        snapshot = manager.snapshot(record['experimentId'])
        self.assertEqual(snapshot['clients']['OH-DT-C01']['assessment'],
                         '过滤')


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
        status, fetched = self.request(
            f"/api/scenarios/{scenario['scenarioId']}")
        self.assertEqual(status, 200)
        self.assertEqual(fetched['scenarioId'], scenario['scenarioId'])

    def test_real_scenario_persists_replay_manifest(self):
        from federatedscope.cv.dataset.office_home import OfficeHome
        data_root = Path(self.temporary.name) / 'officehome'
        for domain in OfficeHome.DOMAINS:
            directory = data_root / (domain.replace('_', ' ')
                                     if 'Real' in domain else domain)
            class_directory = directory / OfficeHome.CLASSES[0]
            class_directory.mkdir(parents=True)
            for index in range(2):
                (class_directory / f'{index}.jpg').touch()
        with mock.patch.dict('os.environ', {
                'FEDERATEDSCOPE_DATA_ROOT': str(data_root)}):
            status, scenario = self.request(
                '/api/scenarios', 'POST', validate_scenario({}))
        self.assertEqual(status, 201)
        self.assertEqual(scenario['artifacts']['partitionManifest'],
                         'client_partitions.json')
        manifest_path = Path(self.temporary.name) / 'api' / 'scenarios' / \
            scenario['scenarioId'] / 'client_partitions.json'
        self.assertTrue(manifest_path.is_file())
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        self.assertEqual(len(manifest['clients']), 60)

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
