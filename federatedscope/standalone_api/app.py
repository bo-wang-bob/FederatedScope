"""Dependency-free HTTP and SSE server for the standalone experiment API."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import uuid
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Tuple
from urllib.parse import parse_qs, urlparse

from federatedscope.standalone_api.repository import JsonRepository
from federatedscope.standalone_api.runner import (
    RunnerPreflightError, StandaloneProcessRunner)
from federatedscope.standalone_api.scenarios import (
    build_partition, build_partition_artifacts)
from federatedscope.standalone_api.schemas import (
    ValidationError, capabilities, validate_experiment, validate_scenario)
from federatedscope.standalone_api.task_manager import (
    ExperimentTaskManager, TaskConflict, TaskNotFound, TERMINAL_STATES)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def public_scenario(scenario: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(scenario)
    if result.get('artifacts'):
        result['artifacts'] = {
            'partitionManifest': 'client_partitions.json',
        }
    return result


class ApiContext:
    def __init__(self, repo_root: Path, state_root: Path):
        self.repo_root = repo_root
        self.repository = JsonRepository(state_root)
        self.manager = ExperimentTaskManager(
            self.repository,
            StandaloneProcessRunner(repo_root),
            state_root / 'runs',
        )


class ApiHandler(BaseHTTPRequestHandler):
    server_version = 'FederatedScopeStandaloneAPI/1.0'
    context: ApiContext

    def log_message(self, format_string: str, *args: Any) -> None:
        if os.environ.get('FEDERATEDSCOPE_API_QUIET') != '1':
            super().log_message(format_string, *args)

    def _cors_headers(self) -> None:
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Headers',
                         'Content-Type, Last-Event-ID')
        self.send_header('Access-Control-Allow-Methods',
                         'GET, POST, OPTIONS')

    def _json(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self._cors_headers()
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _data(self, data: Any, status: int = HTTPStatus.OK) -> None:
        self._json(status, {'data': data, 'requestId': uuid.uuid4().hex})

    def _error(self, status: int, code: str, message: str,
               field_errors: Dict[str, str] | None = None) -> None:
        self._json(status, {
            'error': {
                'code': code,
                'message': message,
                'fieldErrors': field_errors or {},
                'requestId': uuid.uuid4().hex,
            },
        })

    def _body(self) -> Dict[str, Any]:
        length = int(self.headers.get('Content-Length', '0'))
        if length > 2 * 1024 * 1024:
            raise ValidationError('请求体超过 2 MiB 限制')
        raw = self.rfile.read(length)
        if not raw:
            return {}
        try:
            payload = json.loads(raw.decode('utf-8'))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValidationError('请求体不是有效 JSON') from error
        if not isinstance(payload, dict):
            raise ValidationError('请求体必须是 JSON 对象')
        return payload

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        self._cors_headers()
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        try:
            self._handle_get()
        except TaskNotFound:
            self._error(HTTPStatus.NOT_FOUND, 'NOT_FOUND', '实验不存在')
        except Exception as error:
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR,
                        'INTERNAL_ERROR', str(error))

    def _handle_get(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip('/') or '/'
        if path == '/api/health':
            self._data({'status': 'ok', 'time': utc_now()})
            return
        if path == '/api/capabilities':
            self._data(capabilities(self.context.repo_root))
            return
        scenario_match = re.fullmatch(r'/api/scenarios/([^/]+)', path)
        if scenario_match:
            scenario = self.context.repository.get_scenario(
                scenario_match.group(1))
            if scenario is None:
                self._error(HTTPStatus.NOT_FOUND, 'NOT_FOUND', '场景不存在')
            else:
                self._data(public_scenario(scenario))
            return
        if path == '/api/experiments':
            self._data(self.context.manager.list())
            return
        match = re.fullmatch(r'/api/experiments/([^/]+)(?:/(snapshot|events|metrics|logs))?', path)
        if not match:
            self._error(HTTPStatus.NOT_FOUND, 'NOT_FOUND', '接口不存在')
            return
        experiment_id, action = match.groups()
        if action == 'snapshot':
            self._data(self.context.manager.snapshot(experiment_id))
        elif action == 'metrics':
            self._data(self.context.manager.metrics(experiment_id))
        elif action == 'logs':
            query = parse_qs(parsed.query)
            try:
                limit = int(query.get('limit', ['200'])[0])
            except ValueError:
                limit = 200
            self._data(self.context.manager.logs(experiment_id, limit))
        elif action == 'events':
            query = parse_qs(parsed.query)
            header_sequence = self.headers.get('Last-Event-ID', '0').split('-')[-1]
            requested = query.get('afterSequence', [header_sequence])[0]
            try:
                sequence = int(requested)
            except ValueError:
                sequence = 0
            self._events(experiment_id, sequence)
        else:
            self._data(self.context.manager.get(experiment_id))

    def _events(self, experiment_id: str, sequence: int) -> None:
        self.context.manager.get(experiment_id)
        self.send_response(HTTPStatus.OK)
        self._cors_headers()
        self.send_header('Content-Type', 'text/event-stream; charset=utf-8')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Connection', 'keep-alive')
        self.end_headers()
        try:
            while True:
                events = self.context.manager.wait_for_events(
                    experiment_id, sequence, timeout=12.0)
                if not events:
                    self.wfile.write(b': keep-alive\n\n')
                    self.wfile.flush()
                for event in events:
                    sequence = event['sequence']
                    body = json.dumps(event, ensure_ascii=False)
                    self.wfile.write(
                        f"id: {event['id']}\ndata: {body}\n\n".encode('utf-8'))
                    self.wfile.flush()
                record = self.context.manager.get(experiment_id)
                if record['status'] in TERMINAL_STATES and not events:
                    return
        except (BrokenPipeError, ConnectionResetError):
            return

    def do_POST(self) -> None:  # noqa: N802
        try:
            self._handle_post()
        except ValidationError as error:
            self._error(HTTPStatus.UNPROCESSABLE_ENTITY, 'VALIDATION_ERROR',
                        str(error), error.field_errors)
        except RunnerPreflightError as error:
            self._error(HTTPStatus.CONFLICT, 'RUNNER_NOT_READY', str(error), {
                item['name']: item['message']
                for item in error.result.get('checks', [])
                if not item['ready']
            })
        except TaskConflict as error:
            self._error(HTTPStatus.CONFLICT, 'TASK_CONFLICT', str(error))
        except TaskNotFound:
            self._error(HTTPStatus.NOT_FOUND, 'NOT_FOUND', '实验不存在')
        except Exception as error:
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR,
                        'INTERNAL_ERROR', str(error))

    def _handle_post(self) -> None:
        path = urlparse(self.path).path.rstrip('/')
        if path == '/api/scenarios/preview':
            request = validate_scenario(self._body())
            self._data(build_partition(request))
            return
        if path == '/api/scenarios':
            request = validate_scenario(self._body())
            preview, manifest = build_partition_artifacts(request)
            scenario_id = f"SCN-{preview['partitionVersion'].upper()}"
            scenario = {
                'scenarioId': scenario_id,
                'createdAt': utc_now(),
                'request': request,
                'preview': preview,
                'artifacts': {},
            }
            if manifest is not None:
                manifest_path = self.context.repository.save_scenario_artifact(
                    scenario_id, 'client_partitions.json', manifest)
                scenario['artifacts']['partitionManifest'] = str(
                    manifest_path)
            self.context.repository.save_scenario(scenario)
            self._data(public_scenario(scenario), HTTPStatus.CREATED)
            return
        if path == '/api/experiments/preflight':
            config = validate_experiment(self._body())
            scenario = self.context.repository.get_scenario(config['scenarioId'])
            if not scenario:
                raise ValidationError('场景快照不存在或已失效', {
                    'scenarioId': '请重新应用异构环境',
                })
            self._data(self.context.manager.preflight(config, scenario))
            return
        if path == '/api/experiments':
            config = validate_experiment(self._body())
            scenario = self.context.repository.get_scenario(config['scenarioId'])
            if not scenario:
                raise ValidationError('场景快照不存在或已失效', {
                    'scenarioId': '请重新应用异构环境',
                })
            self._data(self.context.manager.create(config, scenario),
                       HTTPStatus.CREATED)
            return
        match = re.fullmatch(r'/api/experiments/([^/]+)/stop', path)
        if match:
            self._body()
            self._data(self.context.manager.stop(match.group(1)))
            return
        self._error(HTTPStatus.NOT_FOUND, 'NOT_FOUND', '接口不存在')


def create_server(host: str = '127.0.0.1', port: int = 8000,
                  state_root: Path | None = None) -> ThreadingHTTPServer:
    repo_root = Path(__file__).resolve().parents[2]
    root = state_root or Path(os.environ.get(
        'FEDERATEDSCOPE_API_STATE_DIR',
        str(repo_root / 'exp' / 'standalone_api')))
    context = ApiContext(repo_root, root)
    handler = type('BoundApiHandler', (ApiHandler,), {'context': context})
    return ThreadingHTTPServer((host, port), handler)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='FederatedScope standalone experiment API')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--state-dir', type=Path)
    args = parser.parse_args()
    server = create_server(args.host, args.port, args.state_dir)
    print(f'Standalone API listening on http://{args.host}:{args.port}')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
