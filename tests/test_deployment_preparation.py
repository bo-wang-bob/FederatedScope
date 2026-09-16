"""Local preparation tests, no Docker daemon, data assets or training needed."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('deployment_check', REPO / 'scripts/check_platform_deployment.py')
deployment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deployment)


class DeploymentPreparationTests(unittest.TestCase):
    def test_frontend_local_assets_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'app.js').write_text('console.log(1)')
            (root / 'index.html').write_text('<script src="/app.js"></script>')
            (root / '.vite').mkdir()
            (root / '.vite/manifest.json').write_text(json.dumps({'index.html': {
                'file': 'app.js', 'isEntry': True}}))
            self.assertEqual(deployment.check_frontend(root)['runtimeFiles'], 3)
            for asset in ('https://cdn.example/app.js', '//cdn.example/app.js', '/missing.js', '../app.js', '/%2e%2e/app.js'):
                (root / 'index.html').write_text(f'<script src="{asset}"></script>')
                with self.subTest(asset=asset), self.assertRaises(ValueError):
                    deployment.check_frontend(root)

    def test_dynamic_chunks_images_and_manifest_links_are_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'index.html').write_text('<script src="/app.js"></script>')
            (root / '.vite').mkdir()
            manifest = {'index.html': {'file': 'app.js', 'isEntry': True, 'dynamicImports': ['page']},
                        'page': {'file': 'page.js', 'css': ['page.css'], 'assets': ['plane.jpg']}}
            for name in ('app.js', 'page.js', 'page.css', 'plane.jpg'):
                (root / name).write_text('fixture')
            (root / '.vite/manifest.json').write_text(json.dumps(manifest))
            self.assertEqual(deployment.check_frontend(root)['runtimeFiles'], 6)
            for name in ('page.js', 'page.css', 'plane.jpg'):
                (root / name).rename(root / (name + '.missing'))
                with self.subTest(name=name), self.assertRaises(ValueError):
                    deployment.check_frontend(root)
                (root / (name + '.missing')).rename(root / name)
            manifest['index.html']['dynamicImports'] = ['unregistered']
            (root / '.vite/manifest.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                deployment.check_frontend(root)

    def test_stale_index_or_missing_build_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'index.html').write_text('<script src="/old.js"></script>')
            for name in ('old.js', 'new.js'):
                (root / name).write_text('fixture')
            with self.assertRaises(ValueError):
                deployment.check_frontend(root)
            (root / '.vite').mkdir()
            (root / '.vite/manifest.json').write_text(json.dumps({'index.html': {
                'file': 'new.js', 'isEntry': True}}))
            with self.assertRaises(ValueError):
                deployment.check_frontend(root)

    def test_offline_guard_is_scoped_to_opted_in_process(self):
        code = '''
import json, socket
from federatedscope.standalone_api.platform_offline import configure_offline_worker
configure_offline_worker()
configure_offline_worker()
results = []
for check in [lambda: socket.getaddrinfo('localhost', 80),
              lambda: socket.socket().connect(('127.0.0.1', 9)),
              lambda: socket.socket(type=socket.SOCK_DGRAM).sendto(b'x', ('127.0.0.1', 9))]:
    try:
        check()
    except RuntimeError:
        results.append('blocked')
print(json.dumps(results))
'''
        result = subprocess.run([sys.executable, '-c', code], cwd=REPO, check=True,
            capture_output=True, text=True, env=dict(os.environ, FS_PLATFORM_OFFLINE='1'))
        self.assertEqual(json.loads(result.stdout), ['blocked'] * 3)
        # Parent process can still resolve local addresses for its HTTP server.
        import socket
        self.assertTrue(socket.getaddrinfo('localhost', 80))

    def test_lock_has_exact_unique_versions(self):
        rows = [row for row in (REPO / 'deploy/requirements-runtime.lock').read_text().splitlines()
                if row and not row.startswith('#')]
        self.assertTrue(all(row.count('==') == 1 for row in rows))
        pins = dict(row.split('==') for row in rows)
        self.assertEqual(len(pins), len(rows))
        self.assertEqual(pins['torch'], '2.5.1+cu121')
        self.assertEqual(pins['numpy'], '1.22.4')
        self.assertEqual(pins['open-clip-torch'], '3.2.0')
        self.assertEqual(pins['timm'], '1.0.24')
        for name in ('huggingface-hub', 'safetensors', 'ftfy', 'regex', 'requests'):
            self.assertIn(name, pins)

    def test_container_recipe_has_only_existing_source_inputs(self):
        frontend = REPO.parent / 'frontend'
        recipe = (REPO / 'deploy/Dockerfile').read_text()
        for row in recipe.splitlines():
            if not row.startswith('COPY ') or row.startswith('COPY --from='):
                continue
            for source in row.split()[1:-1]:
                base, relative = source.split('/', 1)
                root = REPO if base == 'backend' else frontend
                if not root.exists():
                    self.skipTest('需要同级 frontend 工作树检查构建上下文')
                self.assertTrue(list(root.glob(relative)), source)
        self.assertIn('FS_PLATFORM_OFFLINE=1', recipe)
        self.assertIn('STOPSIGNAL SIGTERM', recipe)
        self.assertIn('USER 1000:1000', recipe)


if __name__ == '__main__':
    unittest.main()
