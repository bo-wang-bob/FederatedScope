import socket
import unittest
from unittest.mock import patch

from federatedscope.core.auxiliaries.logging import machine_label


class OfflineLoggingTests(unittest.TestCase):
    def test_offline_does_not_resolve_dns(self):
        with patch.dict('os.environ', {'FS_PLATFORM_OFFLINE': '1'}), \
                patch('socket.gethostname', return_value='container'), \
                patch('socket.gethostbyname', side_effect=AssertionError('DNS used')):
            self.assertEqual(machine_label(), 'container')

    def test_failed_dns_is_not_a_training_failure(self):
        with patch.dict('os.environ', {'FS_PLATFORM_OFFLINE': '0'}), \
                patch('socket.gethostname', return_value='container'), \
                patch('socket.gethostbyname', side_effect=socket.gaierror('unavailable')):
            self.assertEqual(machine_label(), 'container')

    def test_online_address_is_preserved(self):
        with patch.dict('os.environ', {'FS_PLATFORM_OFFLINE': '0'}), \
                patch('socket.gethostbyname', return_value='127.0.0.1'):
            self.assertEqual(machine_label(), '127.0.0.1')
