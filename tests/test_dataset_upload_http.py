"""Exercise real HTTP dispatch, not just the dataset storage functions."""
import io
import json
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from threading import Thread
from types import SimpleNamespace

import pytest
from PIL import Image

from federatedscope.standalone_api.platform_app import PlatformHandler


@pytest.fixture
def endpoint(tmp_path, monkeypatch):
    monkeypatch.setenv('FS_PLATFORM_UPLOADS', str(tmp_path / 'datasets'))
    handler = type('UploadTestHandler', (PlatformHandler,), {
        'context': SimpleNamespace(platform=SimpleNamespace(repo=tmp_path)),
        'log_message': lambda *args: None,
    })
    server = ThreadingHTTPServer(('127.0.0.1', 0), handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def request(path, body=b'', content_type='application/json', extra=None):
        connection = HTTPConnection(*server.server_address, timeout=5)
        try:
            connection.request('POST', path, body=body,
                               headers={'Content-Type': content_type, **(extra or {})})
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()
    yield request
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def png(color):
    stream = io.BytesIO()
    Image.new('RGB', (8, 8), color).save(stream, format='PNG')
    return stream.getvalue()


def test_binary_images_reach_upload_and_complete_registration(endpoint):
    status, response = endpoint('/api/platform/datasets', b'{"name":"ants-bees","kind":"train"}')
    assert status == 200
    identifier = response['data']['id']
    for i in range(6):
        category = 'ants' if i < 3 else 'bees'
        status, response = endpoint(
            f'/api/platform/datasets/{identifier}/files?path={category}%2F{i}.png',
            png((i * 30, 10, 20)), 'application/octet-stream')
        assert status == 200, response
    status, response = endpoint(f'/api/platform/datasets/{identifier}/finish', b'{}')
    assert status == 200
    assert response['data']['classes'] == ['ants', 'bees']
    assert response['data']['count'] == 6
    assert response['data']['status'] == 'ready'


def test_binary_exception_does_not_relax_json_or_origin_guard(endpoint):
    upload = '/api/platform/datasets/' + 'a' * 32 + '/files?path=bees%2Fa.png'
    assert endpoint('/api/platform/train', b'{}', 'application/octet-stream')[0] == 415
    assert endpoint('/api/platform/datasets', b'{}', 'application/octet-stream')[0] == 415
    assert endpoint(upload, b'{}', 'application/json')[0] == 415
    assert endpoint(upload, png((1, 2, 3)), 'application/octet-stream',
                    {'Origin': 'https://untrusted.example'})[0] == 403


def test_upload_length_is_validated_before_read(endpoint):
    upload = '/api/platform/datasets/' + 'a' * 32 + '/files?path=bees%2Fa.png'
    assert endpoint(upload, b'', 'application/octet-stream', {'Content-Length': 'wrong'})[0] == 400
    assert endpoint(upload, b'', 'application/octet-stream', {'Content-Length': str(26 * 1024 ** 2)})[0] == 413
