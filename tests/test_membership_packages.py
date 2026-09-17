"""Dataset selection and verified exported FedMIA image-prefix contracts."""
import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np

from federatedscope.standalone_api.platform_app import PlatformHandler
from federatedscope.standalone_api.platform_config import PlatformError
from federatedscope.standalone_api.privacy_replay import align_image_index_replay


class MembershipPackageTests(unittest.TestCase):
    def test_active_package_keeps_legacy_data_and_rejects_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / 'militaryaircraft3d'
            package.mkdir()
            selection = root / 'active_package.json'
            selection.write_text(json.dumps({'path': 'militaryaircraft3d'}))
            handler = object.__new__(PlatformHandler)
            with patch.dict(os.environ, {'FS_FEDMIA_LOCAL_ROOT': str(root)}):
                self.assertEqual(handler._fedmia_root(), package.resolve())
                selection.write_text(json.dumps({'path': '..'}))
                with self.assertRaises(PlatformError):
                    handler._fedmia_root()
                selection.unlink()
                self.assertEqual(handler._fedmia_root(), root.resolve())

    def fixture(self):
        items = ([('a', 0, 0), ('b', 1, 1), ('c', 2, 2)],
                 [('d', 3, 0), ('e', 4, 1), ('f', 5, 2), ('g', 6, 3)])
        runs = []
        for _ in range(2):
            cfg = NS(attack=NS(mode='mix', fedmia_shadow_stat_mode='indexed'))
            fields = {name: {10: value, 19: value} for name, value in (
                ('train_cos', [.1, .2]), ('train_labels', [0, 1]),
                ('test_cos', [.1] * 4), ('test_labels', [3, 4, 5, 6]),
                ('augmented_cos', [.5]))}
            result = NS(scores_member=[.8, .9], scores_nonmember=[.1, .2],
                        metadata={'shadow_stat_mode': 'indexed', 'rounds_used': [10, 19],
                                  'use_augmented_nonmember': False,
                                  'use_image_aug_member': False})
            runs.append((cfg, fields, result))
        return items, runs

    def test_stored_generated_features_can_calibrate_but_not_be_displayed(self):
        items, runs = self.fixture()
        common, info = align_image_index_replay(NS(np=np), runs, items)
        self.assertEqual(common, (2, 2))
        self.assertEqual(info['metricsMode'], 'mix')
        self.assertEqual(len(items[1]), 4)

    def test_bad_labels_rounds_counts_or_generated_results_are_rejected(self):
        items, original = self.fixture()
        for issue in ('labels', 'rounds', 'counts', 'generated', 'statistics'):
            with self.subTest(issue=issue):
                runs = copy.deepcopy(original)
                cfg, fields, result = runs[1]
                if issue == 'labels':
                    fields['train_labels'][19] = [1, 0]
                elif issue == 'rounds':
                    result.metadata['rounds_used'] = [10]
                elif issue == 'counts':
                    result.scores_member = [.9]
                elif issue == 'generated':
                    result.metadata['use_augmented_nonmember'] = True
                else:
                    result.metadata['shadow_stat_mode'] = 'global'
                with self.assertRaises(PlatformError):
                    align_image_index_replay(NS(np=np), runs, items)


if __name__ == '__main__':
    unittest.main()
