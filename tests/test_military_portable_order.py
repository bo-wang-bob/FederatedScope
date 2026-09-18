"""Unpacking/readdir order must not change cache-backed split membership."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from federatedscope.cv.dataset.domainnet import DomainNet, load_domainnet_manifest
from federatedscope.standalone_api.platform_config import ConfigFactory, PlatformError

REPO=Path(__file__).resolve().parents[1]
MANIFEST=REPO/'scripts/military_aircraft_3domain/manifests/cache_sample_order.json'

class PortableMilitaryOrderTests(unittest.TestCase):
    def test_split_is_independent_of_directory_enumeration(self):
        domains,classes,records=load_domainnet_manifest(str(MANIFEST))
        info=json.loads(MANIFEST.read_text(encoding='utf-8'))
        for domain in domains:
            with tempfile.TemporaryDirectory() as moved, patch('os.listdir',side_effect=AssertionError('Do not enumerate directories')):
                train=DomainNet(moved,domain,classes,split='train',seed=42,records=records[domain])
                test=DomainNet(moved,domain,classes,split='test',seed=42,records=records[domain])
                train_ids={Path(p).relative_to(moved).as_posix() for p in train.data}
                test_ids=[Path(p).relative_to(moved).as_posix() for p in test.data]
                self.assertEqual((len(train_ids),len(test_ids)),(175,75))
                self.assertFalse(train_ids.intersection(test_ids))
                self.assertEqual(test_ids,info['verification'][domain]['testPathsInOrder'])
                if domain=='aerial':
                    self.assertNotIn('aerial/B-52/0029.jpg',train_ids)
                    self.assertIn('aerial/B-52/0029.jpg',test_ids)

    def test_missing_order_manifest_fails_closed(self):
        factory=ConfigFactory(REPO)
        request=factory.normalize({'group':'military_vit','method':'fedavg'})
        original=Path.is_file
        with tempfile.TemporaryDirectory(dir=REPO.parent) as root:
            with patch.object(Path,'is_file',lambda p: False if p==MANIFEST else original(p)):
                with self.assertRaisesRegex(PlatformError,'样本顺序清单'):
                    factory.build(request,Path(root))
