import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from federatedscope.standalone_api.repository import JsonRepository


class AtomicWriteTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'job.json'
        JsonRepository._atomic_write(self.path, {'round': 1})

    def locked(self):
        error = PermissionError('temporarily locked')
        error.winerror = 5
        return error

    def test_transient_denial_retries_complete_payload(self):
        replace = os.replace
        attempts = []

        def intermittent(source, destination):
            attempts.append(source)
            if len(attempts) < 3:
                self.assertEqual(json.loads(self.path.read_text()), {'round': 1})
                raise self.locked()
            replace(source, destination)

        with patch('federatedscope.standalone_api.repository.os.replace', side_effect=intermittent), \
                patch('federatedscope.standalone_api.repository.time.sleep'):
            JsonRepository._atomic_write(self.path, {'round': 2})
        self.assertEqual(len(attempts), 3)
        self.assertEqual(len(set(attempts)), 1)
        self.assertEqual(json.loads(self.path.read_text()), {'round': 2})
        self.assertEqual(list(self.path.parent.glob('*.tmp')), [])

    def test_persistent_denial_preserves_previous_json(self):
        with patch('federatedscope.standalone_api.repository.os.replace', side_effect=self.locked()) as replace, \
                patch('federatedscope.standalone_api.repository.time.sleep'):
            with self.assertRaises(PermissionError):
                JsonRepository._atomic_write(self.path, {'round': 2})
        self.assertEqual(replace.call_count, 10)
        self.assertEqual(json.loads(self.path.read_text()), {'round': 1})
        self.assertEqual(list(self.path.parent.glob('*.tmp')), [])

    def test_unrelated_error_is_not_retried(self):
        with patch('federatedscope.standalone_api.repository.os.replace', side_effect=OSError('disk error')) as replace:
            with self.assertRaises(OSError):
                JsonRepository._atomic_write(self.path, {'round': 2})
        self.assertEqual(replace.call_count, 1)
        self.assertEqual(json.loads(self.path.read_text()), {'round': 1})

    @unittest.skipUnless(os.name == 'nt', 'Windows sharing semantics')
    def test_real_open_reader_releases_during_retry(self):
        reader = self.path.open('rb')
        release = threading.Timer(.12, reader.close)
        release.start()
        try:
            JsonRepository._atomic_write(self.path, {'round': 3})
        finally:
            release.join()
            reader.close()
        self.assertEqual(json.loads(self.path.read_text()), {'round': 3})


if __name__ == '__main__':
    unittest.main()
