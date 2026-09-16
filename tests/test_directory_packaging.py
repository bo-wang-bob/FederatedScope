"""Directory preparation safeguards; no torch, network or real assets needed."""
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from scripts import prepare_platform_directory as prepare
from scripts import check_platform_directory as check


class DirectoryPackagingTests(unittest.TestCase):
    def test_officehome_resources_are_paired_and_use_portable_training_paths(self):
        self.assertEqual(prepare.officehome_inputs(SimpleNamespace()), {})
        with self.assertRaises(ValueError):
            prepare.officehome_inputs(SimpleNamespace(officehome_dataset=Path('images')))
        with self.assertRaises(ValueError):
            prepare.officehome_inputs(SimpleNamespace(officehome_features=Path('features')))
        self.assertEqual(prepare.officehome_inputs(SimpleNamespace(
            officehome_dataset=Path('images'), officehome_features=Path('features'))), {
                'datasets/OfficeHomeDataset_10072016': Path('images'),
                'exp/distributed_feature_cache/officehome_vit': Path('features')})

    def test_copy_does_not_include_host_environment_or_dependencies(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'source'
            source.mkdir()
            (source / 'code.py').write_text('pass')
            (source / '.env.local').write_text('not for delivery')
            (source / 'node_modules').mkdir()
            (source / 'node_modules/file').write_text('dependency')
            prepare.copy_checked(source, root / 'copy')
            self.assertEqual([p.name for p in (root / 'copy').iterdir()], ['code.py'])

    def test_active_and_unclean_jobs_block_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job = root / 'jobs/example/job.json'
            job.parent.mkdir(parents=True)
            for status, cleanup in [('running', True), ('completed', False)]:
                job.write_text(json.dumps(dict(id='example', status=status, cleanup=dict(ok=cleanup))))
                with self.assertRaises(ValueError):
                    prepare.state_snapshot(root)
            job.write_text(json.dumps(dict(id='example', status='completed')))
            (job.parent / 'process.json').write_text('old PID')
            (root / 'service.lock').touch()
            self.assertEqual(set(prepare.state_snapshot(root)), {'jobs/example/job.json'})

    def test_relative_inventory_survives_move_and_detects_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'before'
            root.mkdir()
            (root / 'file').write_bytes(b'original')
            manifest = {'file': dict(bytes=8, sha256=prepare.sha256(root / 'file'))}
            (root / 'FILES.sha256.json').write_text(json.dumps(manifest))
            moved = root.with_name('after with spaces')
            root.rename(moved)
            self.assertEqual(check.verify_files(moved)['files'], 1)
            (moved / 'file').write_bytes(b'changed!')
            with self.assertRaises(ValueError):
                check.verify_files(moved)

    def test_backdoor_jobs_are_included_and_active_jobs_block_packaging(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'jobs').mkdir()
            path = root / 'backdoor/example/job.json'
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(dict(id='example',status='running')))
            with self.assertRaises(ValueError):
                prepare.state_snapshot(root)
            path.write_text(json.dumps(dict(id='example',status='completed')))
            self.assertEqual(set(prepare.state_snapshot(root)), {'backdoor/example/job.json'})

    def test_partial_or_escaping_inventory_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'PREPARATION_INCOMPLETE').touch()
            with self.assertRaises(ValueError):
                check.verify_files(root)
            (root / 'PREPARATION_INCOMPLETE').unlink()
            for name in ('../outside', '/etc/passwd', 'C:/secret', 'a\\b'):
                (root / 'FILES.sha256.json').write_text(json.dumps({name: {}}))
                with self.subTest(name=name), self.assertRaises(ValueError):
                    check.verify_files(root)

    def test_links_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'file').write_text('data')
            try:
                (root / 'link').symlink_to(root / 'file')
            except OSError:
                self.skipTest('symlink privilege unavailable')
            with self.assertRaises(ValueError):
                list(prepare.files(root))

    def test_unknown_privacy_script_is_not_modified(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'helper.py'
            path.write_text('unknown')
            with self.assertRaises(ValueError):
                prepare.portable_privacy_helper(path)
            self.assertEqual(path.read_text(), 'unknown')

    def test_reviewed_helper_only_changes_paths_and_freeze(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'helper.py'
            source = '''from pathlib import Path
REPO_ROOT = Path('/old')
NO_DEFENSE_RUN_DIR = '/old/no_defense'
DEFENSE_RUN_DIR = '/old/defense'
def load_cfg(cfg):
    cfg.freeze()
    return cfg
'''
            path.write_text(source)
            original = prepare.sha256(path)
            with patch.object(prepare, 'HELPER_HASHES', {original}):
                metadata = prepare.portable_privacy_helper(path)
            self.assertEqual(metadata['originalSha256'], original)
            self.assertNotIn('/old', path.read_text())
            self.assertIn('cfg.freeze(save=False, inform=False)', path.read_text())
            self.assertIn('datasets/OfficeHomeDataset_10072016', path.read_text())


if __name__ == '__main__':
    unittest.main()
