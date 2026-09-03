import unittest
from pathlib import Path

from federatedscope.standalone_api.distributed_runner import (
    DistributedProcessRunner, distributed_catalog, load_lab_topology)
from federatedscope.standalone_api.schemas import (
    ValidationError, validate_experiment)


REPO_ROOT = Path(__file__).resolve().parents[1]


def distributed_payload(group='officehome_vit', method='heterogeneous_solution'):
    return {
        'schemaVersion': '1.0',
        'idempotencyKey': 'distributed-test',
        'experimentId': 'EXP-DISTRIBUTED-TEST',
        'name': '三机准确率测试',
        'type': 'heterogeneity',
        'scenarioId': 'SCN-TEST',
        'execution': {
            'mode': 'distributed',
            'topologyId': 'lab-three-machine',
            'group': group,
            'evaluationFrequency': 5,
            'clientsPerSubserver': 30,
            'windowsClientCount': 60,
            'rootDevice': 1,
            'statisticsUploadStaggerSeconds': 3,
            'diagonalCovariance': False,
        },
        'common': {
            'method': method,
            'rounds': 100,
            'localEpochs': 1,
            'participationRate': 1.0,
            'batchSize': 8,
            'learningRate': 0.001,
            'seed': 42,
            'device': 'cpu',
        },
        'heterogeneity': {
            'expansionTarget': 50,
            'featureBatchSize': 64,
        },
        'privacy': None,
        'backdoor': None,
    }


class DistributedSchemaTest(unittest.TestCase):
    def test_catalog_exposes_every_checked_in_accuracy_case(self):
        catalog = distributed_catalog(REPO_ROOT)
        self.assertEqual(len(catalog), 10)
        case_count = sum(len(item['methods']) for item in catalog)
        self.assertEqual(case_count, 52)
        self.assertEqual(
            next(item for item in catalog
                 if item['group'] == 'mdsent_lstm')['clientCount'], 120)

    def test_distributed_accuracy_payload_is_normalized(self):
        config = validate_experiment(distributed_payload())
        self.assertEqual(config['execution']['mode'], 'distributed')
        self.assertEqual(config['execution']['group'], 'officehome_vit')
        self.assertEqual(config['execution']['clientsPerSubserver'], 30)

    def test_distributed_rejects_unvalidated_group_method_pair(self):
        with self.assertRaises(ValidationError) as caught:
            validate_experiment(distributed_payload('digit3_vit', 'moon'))
        self.assertIn('common.method', caught.exception.field_errors)

    def test_distributed_rejects_security_mode_for_now(self):
        payload = distributed_payload()
        payload['type'] = 'privacy'
        payload['heterogeneity'] = None
        payload['privacy'] = {
            'attack': 'membership', 'defenseEnabled': False,
        }
        with self.assertRaises(ValidationError) as caught:
            validate_experiment(payload)
        self.assertIn('type', caught.exception.field_errors)


class DistributedRunnerCommandTest(unittest.TestCase):
    def test_generator_uses_existing_case_and_isolated_remote_repos(self):
        config = validate_experiment(distributed_payload())
        runner = DistributedProcessRunner(
            REPO_ROOT, topology=load_lab_topology(), probe_remote=False)
        command = runner.build_generate_command(
            config, REPO_ROOT / 'exp' / 'test-control')
        rendered = ' '.join(str(item) for item in command)
        self.assertIn('--case-specs officehome_vit:ggeur', rendered)
        self.assertIn('--total-round-num 100', rendered)
        self.assertIn('--eval-frequency 5', rendered)
        self.assertIn('FederatedScope-worktrees', rendered)
        self.assertIn('--root-device 1', rendered)

    def test_remote_probe_can_be_disabled_for_offline_validation(self):
        config = validate_experiment(distributed_payload())
        runner = DistributedProcessRunner(
            REPO_ROOT, topology=load_lab_topology(), probe_remote=False)
        result = runner.preflight(config)
        self.assertTrue(result['ready'])
        self.assertEqual(result['executionMode'], 'distributed')
        self.assertEqual(len(result['topology']), 3)


if __name__ == '__main__':
    unittest.main()
