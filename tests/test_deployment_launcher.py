"""Exercise launcher ownership and offline arguments against a fake Docker CLI."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]
FAKE_DOCKER = r'''#!/usr/bin/env python3
import json, os, sys
args = sys.argv[1:]
with open(os.environ['DOCKER_CALLS'], 'a') as stream:
    stream.write(json.dumps(args) + '\n')
if args[:2] == ['image', 'inspect']:
    print('sha256:local-verified-image')
elif args[:2] == ['container', 'inspect']:
    sys.exit(0 if os.environ.get('EXISTING') == '1' else 1)
elif args and args[0] == 'inspect':
    fmt = args[2]
    if 'platform.root' in fmt:
        print(os.environ.get('OWNER', 'someone-else'))
    elif 'platform.settings' in fmt:
        print('different-settings')
    elif '.State.Status' in fmt:
        print(os.environ.get('HEALTH_STATUS', 'running/healthy'))
elif args and args[0] == 'run' and '--rm' in args:
    sys.exit(int(os.environ.get('PREFLIGHT_EXIT', '0')))
'''


@unittest.skipUnless(os.name == 'posix', 'Bash 启动器在 Linux 验证')
class DeploymentLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.calls = root / 'calls.jsonl'
        self.calls.touch()
        fake = root / 'docker'
        fake.write_text(FAKE_DOCKER)
        fake.chmod(0o755)
        for name in ('resources', 'state'):
            (root / name).mkdir()
        self.env = dict(os.environ, PATH=str(root) + os.pathsep + os.environ['PATH'],
            DOCKER_CALLS=str(self.calls), FS_RESOURCES=str(root / 'resources'),
            FS_STATE=str(root / 'state'), FS_PORT='8012', FS_CONTAINER_NAME='fs-test-platform')

    def tearDown(self):
        self.temp.cleanup()

    def run_launcher(self, action, **extra):
        result = subprocess.run(['bash', str(REPO / 'deploy/platform.sh'), action],
            cwd=self.temp.name, env=dict(self.env, **extra), capture_output=True, text=True, timeout=10)
        calls = [json.loads(row) for row in self.calls.read_text().splitlines()]
        return result, calls

    def test_check_is_offline_and_does_not_start_server(self):
        result, calls = self.run_launcher('check')
        self.assertEqual(result.returncode, 0, result.stderr)
        runs = [call for call in calls if call[0] == 'run']
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0][runs[0].index('--network') + 1], 'none')
        self.assertIn('--require-cache', runs[0])
        self.assertIn('--gpu', runs[0])
        self.assertNotIn('-d', runs[0])
        self.assertFalse(any(call[0] in {'pull', 'build', 'load', 'kill', 'rm'} for call in calls))

    def test_start_uses_readonly_resources_and_local_browser_port(self):
        result, calls = self.run_launcher('start')
        self.assertEqual(result.returncode, 0, result.stderr)
        launch = next(call for call in calls if call[0] == 'run' and '-d' in call)
        self.assertIn('127.0.0.1:8012:8002', launch)
        self.assertIn('--read-only', launch)
        self.assertIn('--init', launch)
        self.assertIn('never', launch)
        self.assertTrue(any('resources,readonly' in value for value in launch))
        self.assertTrue(any(value.startswith('org.federatedscope.platform.root=') for value in launch))

    def test_preflight_failure_prevents_start(self):
        result, calls = self.run_launcher('start', PREFLIGHT_EXIT='1')
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(any(call[0] == 'start' or '-d' in call for call in calls))

    def test_relative_mounts_resolve_from_release_not_callers_directory(self):
        result, calls = self.run_launcher('check',
            FS_RESOURCES=os.path.relpath(self.env['FS_RESOURCES'], REPO.parent),
            FS_STATE=os.path.relpath(self.env['FS_STATE'], REPO.parent))
        self.assertEqual(result.returncode, 0, result.stderr)
        run = next(call for call in calls if call[0] == 'run')
        self.assertIn('type=bind,source=' + self.env['FS_RESOURCES']
                      + ',target=/opt/platform/backend/resources,readonly', run)

    def test_unhealthy_start_reports_failure_without_deleting_evidence(self):
        result, calls = self.run_launcher('start', HEALTH_STATUS='running/unhealthy')
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(any(call[0] in {'rm', 'kill', 'stop'} for call in calls))

    def test_cannot_stop_foreign_container(self):
        result, calls = self.run_launcher('stop')
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(any(call[0] == 'stop' for call in calls))

    def test_can_stop_owned_container_only(self):
        result, calls = self.run_launcher('stop', OWNER=str(REPO.parent))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls[-1], ['stop', '--time', '30', 'fs-test-platform'])

    def test_settings_change_does_not_restart_existing_container(self):
        result, calls = self.run_launcher('start', OWNER=str(REPO.parent), EXISTING='1')
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(any(call[0] == 'start' or '-d' in call for call in calls))


if __name__ == '__main__':
    unittest.main()
