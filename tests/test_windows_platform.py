import os
from pathlib import Path
import tempfile
import unittest

from federatedscope.standalone_api.platform_paths import relative_path
from federatedscope.standalone_api.platform_service import PlatformService
from federatedscope.standalone_api.platform_config import PlatformError


@unittest.skipUnless(os.name == 'nt', 'Windows-specific contracts')
class WindowsPlatformTests(unittest.TestCase):
    def test_cross_drive_runtime_paths_remain_absolute(self):
        self.assertEqual(relative_path('C:/platform', 'D:/runtime/job'), 'D:/runtime/job')

    def test_two_services_cannot_open_same_state(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as state:
            service = PlatformService(repo, state)
            try:
                with self.assertRaises(PlatformError):
                    PlatformService(repo, state)
            finally:
                service.close()
            reopened = PlatformService(repo, state)
            reopened.close()
