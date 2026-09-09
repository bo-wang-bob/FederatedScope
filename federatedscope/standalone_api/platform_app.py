"""Single-host platform HTTP entrypoint; legacy three-host API remains separate."""
from __future__ import annotations

import argparse
import csv
from http.server import ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import re
import signal
import threading
from urllib.parse import urlparse
import zipfile

from .app import ApiHandler
from .platform_config import PlatformError, sha256
from .platform_service import PlatformService
from .schemas import ValidationError


class PlatformHandler(ApiHandler):
    server_version = 'FederatedScopeSingleHost/1.0'

    def _body(self):
        try:
            length = int(self.headers.get('Content-Length', '0'))
        except ValueError as error:
            raise PlatformError('Content-Length 非法', 400) from error
        if not 0 <= length <= 64 * 1024:
            raise PlatformError('配置请求必须在 64 KiB 以内', 413)
        return super()._body()

    def _cors_headers(self):
        # No wildcard cross-origin write access to a training control plane.
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Cache-Control', 'no-store')

    def _dispatch(self, write=False):
        try:
            if write:
                origin = self.headers.get('Origin')
                if origin and urlparse(origin).netloc != self.headers.get('Host'):
                    raise PlatformError('不允许跨站启动或停止任务', 403)
                if self.headers.get_content_type() != 'application/json':
                    raise PlatformError('仅接受 application/json', 415)
            self._route(write)
        except (PlatformError, ValidationError) as error:
            self._error(getattr(error, 'status', 422), 'PLATFORM_ERROR', str(error))
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as error:
            self._error(500, 'INTERNAL_ERROR', str(error))

    def do_GET(self):
        self._dispatch()

    def do_POST(self):
        self._dispatch(True)

    def _download(self, payload, name, content_type):
        self.send_response(200)
        self._cors_headers()
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Disposition', f'attachment; filename="{name}"')
        size = payload.stat().st_size if isinstance(payload, Path) else len(payload)
        self.send_header('Content-Length', str(size))
        self.end_headers()
        if isinstance(payload, Path):
            with payload.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    self.wfile.write(chunk)
        else:
            self.wfile.write(payload)

    def _route(self, write):
        path = urlparse(self.path).path.rstrip('/') or '/'
        service = self.context.platform
        if not write:
            endpoints = {'/api/health': lambda: {'status': 'ok', 'mode': 'single-host', 'commit': service.commit},
                         '/api/platform/catalog': service.configs.catalog,
                         '/api/platform/resources': service.resources,
                         '/api/platform/jobs': service.list,
                         '/api/platform/library': service.library}
            if path in endpoints:
                self._data(endpoints[path]())
                return
        if write and path in {'/api/platform/preflight', '/api/platform/train', '/api/platform/evaluate'}:
            action = {'preflight': 'inspect', 'train': 'train', 'evaluate': 'evaluate'}[path.split('/')[-1]]
            self._data(service.create(action, self._body()), 202)
            return
        match = re.fullmatch(r'/api/platform/jobs/([a-f0-9]{32})(?:/(stop|logs|export|csv|bundle|model-final|model-best))?', path)
        if match:
            job_id, action = match.groups()
            job = service.get(job_id)
            if write and action == 'stop':
                self._body()
                self._data(service.stop(job_id))
                return
            if not write:
                if action is None:
                    self._data(job)
                elif action == 'logs':
                    self._data(service.logs(job_id))
                elif action == 'export':
                    self._download(json.dumps(job, ensure_ascii=False, indent=2).encode(), job_id + '.json', 'application/json')
                elif action == 'csv':
                    stream = io.StringIO()
                    writer = csv.writer(stream)
                    writer.writerow(['domain', 'class', 'support', 'precision', 'recall', 'f1'])
                    result = job.get('result', {})
                    for domain, metric in {'overall': result, **result.get('domains', {})}.items() if job['action'] == 'evaluate' else []:
                        for row in metric.get('perClass', []):
                            writer.writerow([domain, row['classIndex'], row['support'], row['precision'], row['recall'], row['f1']])
                    self._download(stream.getvalue().encode('utf-8-sig'), job_id + '.csv', 'text/csv')
                elif action in {'model-final', 'model-best'}:
                    file = service.directory(job_id) / ('checkpoints/mlp_' + action.split('-')[1] + '.pt')
                    if job['status'] != 'completed' or not file.is_file():
                        raise PlatformError('模型未完成或不存在', 404)
                    if sha256(file) != job['result']['artifactHashes'][file.name]:
                        raise PlatformError('模型文件已改变，拒绝以原版本导出', 409)
                    self._download(file, job_id + '-' + file.name, 'application/octet-stream')
                elif action == 'bundle':
                    if job['status'] not in {'completed', 'failed', 'stopped', 'interrupted'}:
                        raise PlatformError('结束后才能导出完整复现包', 409)
                    output = service.directory(job_id)
                    target = output / 'reproducibility.zip'
                    # Rebuild to include the current terminal state. Never package resource roots.
                    with service.lock:
                        with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
                            for file in output.rglob('*'):
                                if file.is_file() and not file.is_symlink() and file != target:
                                    archive.write(file, file.relative_to(output))
                    self._download(target, job_id + '.zip', 'application/zip')
                else:
                    raise PlatformError('接口不存在', 404)
                return
        if not write and not path.startswith('/api') and self._serve_frontend(path):
            return
        raise PlatformError('接口不存在', 404)


def create_server(host='127.0.0.1', port=8001, state=None):
    repo = Path(__file__).resolve().parents[2]
    service = PlatformService(repo, state or repo / 'exp/single_host_platform')
    context = type('PlatformContext', (), {'platform': service,
        'frontend_dist': Path(os.environ.get('FEDERATEDSCOPE_FRONTEND_DIST', repo / 'frontend/dist')).resolve()})()
    handler = type('BoundPlatformHandler', (PlatformHandler,), {'context': context})
    try:
        server = ThreadingHTTPServer((host, port), handler)
    except Exception:
        service.close()
        raise
    return server


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8001)
    parser.add_argument('--state-dir', type=Path)
    args = parser.parse_args()
    server = create_server(args.host, args.port, args.state_dir)
    def shutdown(*_):
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    print(f'Single-host platform http://{args.host}:{args.port}', flush=True)
    try:
        server.serve_forever()
    finally:
        server.RequestHandlerClass.context.platform.close()
        server.server_close()


if __name__ == '__main__':
    main()
