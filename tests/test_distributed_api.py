import unittest
import subprocess
import tempfile
from pathlib import Path

from federatedscope.standalone_api.distributed_runner import (
    DistributedNode, DistributedProcessRunner, LabTopology,
    SCRIPT_RELATIVE, distributed_catalog, load_lab_topology)
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
        quick = next(item for item in catalog
                     if item['group'] == 'officehome_vit')
        self.assertTrue(quick['quickStart'])
        self.assertEqual(quick['cachePolicy'],
                         'complete-feature-cache-required')
        self.assertEqual(quick['quickValidation']['method'], 'fedavg')
        self.assertEqual(quick['quickValidation']['rounds'], 1)
        self.assertEqual(quick['quickValidation']['learningRate'], 0.0001)
        self.assertTrue(quick['quickValidation']['cacheOnly'])

    def test_distributed_accuracy_payload_is_normalized(self):
        config = validate_experiment(distributed_payload())
        self.assertEqual(config['execution']['mode'], 'distributed')
        self.assertEqual(config['execution']['group'], 'officehome_vit')
        self.assertEqual(config['execution']['clientsPerSubserver'], 30)
        self.assertTrue(config['execution']['cacheOnly'])

    def test_distributed_rejects_non_cached_execution(self):
        payload = distributed_payload()
        payload['execution']['cacheOnly'] = False
        with self.assertRaises(ValidationError) as caught:
            validate_experiment(payload)
        self.assertIn('execution.cacheOnly', caught.exception.field_errors)

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

    def test_windows_run_sync_stages_archive_in_ssh_home(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            run_id = 'EXP-SYNC-TEST'
            run_root = base / 'control' / run_id
            run_root.mkdir(parents=True)
            (run_root / 'matrix_manifest.json').write_text(
                '{}', encoding='utf-8')
            root_repo = base / 'root-repo'
            topology = LabTopology(
                topology_id='test',
                client=DistributedNode(
                    key='client', label='client', target='client@test',
                    operating_system='windows', repo='D:/isolated',
                    python='D:/python.exe'),
                subserver=DistributedNode(
                    key='subserver', label='subserver', target='third@test',
                    operating_system='windows', repo='C:/isolated',
                    python='C:/python.exe'),
                root=DistributedNode(
                    key='root', label='root', target='root@test',
                    operating_system='linux', repo=str(root_repo),
                    python='/python', local=True),
                client_resource_repo='D:/resources',
                subserver_resource_repo='C:/resources',
                root_resource_repo='/resources')
            runner = DistributedProcessRunner(
                REPO_ROOT, topology=topology, probe_remote=False)
            remote_scripts = []
            commands = []
            runner._remote = lambda node, script, **kwargs: (
                remote_scripts.append((node.key, script)) or '')
            runner._run_command = lambda command, **kwargs: (
                commands.append(command) or subprocess.CompletedProcess(
                    command, 0, ''))

            runner._sync_run(run_root, run_id, lambda *_: None)

            scp_targets = [command[-1] for command in commands]
            self.assertEqual(scp_targets, [
                'client@test:.federatedscope-EXP-SYNC-TEST.zip',
                'third@test:.federatedscope-EXP-SYNC-TEST.zip'])
            self.assertTrue(all('Expand-Archive' in script
                                for _, script in remote_scripts[1::2]))
            self.assertTrue((root_repo / str(SCRIPT_RELATIVE) / 'runs' /
                             run_id / 'matrix_manifest.json').is_file())
            self.assertFalse(
                (run_root.parent / f'.{run_id}.sync.zip').exists())

    def test_cached_windows_clients_do_not_require_raw_data_or_backbone(self):
        runner = DistributedProcessRunner(
            REPO_ROOT, topology=load_lab_topology(), probe_remote=False)
        resources = runner._resource_paths(
            'domainnet_mixer', validate_experiment(distributed_payload(
                'domainnet_mixer', 'fedavg')))

        client_paths = resources['client']
        self.assertTrue(any('domainnet_mixer' in path
                            for path in client_paths))
        self.assertTrue(any('domainnet_manifest.json' in path
                            for path in client_paths))
        self.assertFalse(any(path.endswith('/data/DomainNet')
                             for path in client_paths))
        self.assertFalse(any(path.endswith('mixer_b16_224_complete.pth')
                             for path in client_paths))

    def test_digit3_preflight_keeps_portable_manifest_contract(self):
        runner = DistributedProcessRunner(
            REPO_ROOT, topology=load_lab_topology(), probe_remote=False)
        resources = runner._resource_paths(
            'digit3_cnn', validate_experiment(distributed_payload(
                'digit3_cnn', 'fedavg')))

        self.assertTrue(any(path.endswith(
            'data/digit_three_domain/manifests')
            for path in resources['client']))
        self.assertTrue(any(path.endswith(
            'data/digit_three_domain/dataset_manifest.json')
            for path in resources['root']))

    def test_cleanup_uses_deterministic_case_and_continues_after_error(self):
        config = validate_experiment(distributed_payload(
            'officehome_vit', 'fedavg'))
        runner = DistributedProcessRunner(
            REPO_ROOT, topology=load_lab_topology(), probe_remote=False)
        calls = []

        def remote(node, script, **_kwargs):
            calls.append((node.key, script))
            if len(calls) == 1:
                raise RuntimeError('client unreachable')
            return ''

        runner._remote = remote
        errors = runner.cleanup(
            config, REPO_ROOT / 'missing-output-directory')

        self.assertEqual(len(calls), 4)
        self.assertEqual(len(errors), 1)
        self.assertIn('client unreachable', errors[0])
        self.assertTrue(all(
            'EXP-DISTRIBUTED-TEST' in script for _, script in calls))
        self.assertTrue(all(
            'officehome_vit_fedavg' in script for _, script in calls))


if __name__ == '__main__':
    unittest.main()
