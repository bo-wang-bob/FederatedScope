"""Current release exports all resource generations and module state."""
import json
from pathlib import Path
import tempfile
import unittest
from scripts import prepare_current_release as release
from scripts.check_platform_directory import verify_files


class CurrentReleaseTests(unittest.TestCase):
    def fixture(self, root):
        backend, frontend = root / 'backend', root / 'frontend'
        for name in release.BACKEND_CODE:
            path = backend / name
            if '.' in name or name == 'LICENSE':
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('fixture')
            else:
                path.mkdir(parents=True)
                (path / 'fixture').write_text('fixture')
        for relative in ('scripts/start_platform.py', 'docs/RELEASE_LAYOUT.md'):
            (backend / relative).write_text('fixture')
        for relative in ('package-lock.json', 'dist/index.html', 'src/code.ts', 'tsconfig.json'):
            path = frontend / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('fixture')
        for name in release.REQUIRED_RESOURCES:
            path = backend / 'resources' / name
            if path.suffix != '.pt' and path.suffix != '.pth':
                path = path / 'fixture'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('resource')
        for module in ('jobs', 'privacy_experiments/jobs', 'backdoor/jobs'):
            path = backend / 'exp/platform' / module / 'example/job.json'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({'id': 'example', 'status': 'completed'}))
            (path.parent / 'model.pt').write_text('model')
            (path.parent / 'process.json').write_text('old PID')
        return backend, frontend

    def test_export_move_verify_and_exclusions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            backend, frontend = self.fixture(root)
            (frontend / 'src/.env.local').write_text('host settings')
            (backend / 'resources/fedmia_local/active_package.json').write_text('{}')
            output = root / 'release'
            release.export(backend, frontend, output)
            moved = root / 'moved release'
            output.rename(moved)
            self.assertGreater(verify_files(moved)['files'], 10)
            self.assertFalse((moved / 'frontend/src/.env.local').exists())
            self.assertFalse(list(moved.rglob('process.json')))
            self.assertTrue((moved / 'backend/exp/platform/privacy_experiments/jobs/example/model.pt').is_file())
            self.assertTrue((moved / 'backend/resources/fedmia_local/active_package.json').is_file())
            self.assertTrue((moved / 'frontend/tsconfig.json').is_file())

    def test_active_privacy_missing_weight_and_partial_upload_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            backend, frontend = self.fixture(Path(temp))
            job = backend / 'exp/platform/privacy_experiments/jobs/example/job.json'
            original = job.read_text()
            job.write_text(json.dumps({'id': 'example', 'status': 'running'}))
            with self.assertRaisesRegex(ValueError, 'active/unclean'):
                release.inspect(backend, frontend)
            job.write_text(original)
            metadata = backend / 'resources/uploaded_datasets/example/dataset.json'
            metadata.parent.mkdir(parents=True)
            metadata.write_text('{"status":"uploading"}')
            with self.assertRaisesRegex(ValueError, 'unfinished dataset'):
                release.inspect(backend, frontend)
            metadata.write_text('{"status":"ready"}')
            release.inspect(backend, frontend)
            (backend / 'resources/models/ViT-B-16.pt').unlink()
            with self.assertRaisesRegex(ValueError, 'missing/empty resource'):
                release.inspect(backend, frontend)

    def test_export_refuses_existing_or_overlapping_destination(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            backend, frontend = self.fixture(root)
            for destination in (backend, backend / 'nested', frontend / 'nested', root):
                with self.assertRaises(ValueError):
                    release.export(backend, frontend, destination)


if __name__ == '__main__':
    unittest.main()
