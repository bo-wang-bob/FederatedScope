import copy
from types import SimpleNamespace as NS
import unittest

import numpy as np

from federatedscope.standalone_api.platform_app import PlatformHandler
from federatedscope.standalone_api.platform_config import PlatformError
from federatedscope.standalone_api.privacy_replay import align_indexed_replay


class PrivacyReplayTests(unittest.TestCase):
    def setUp(self):
        self.module = NS(np=np)
        self.items = ([('a', 0, 0), ('b', 1, 1), ('c', 2, 2)],
                      [('d', 3, 0), ('e', 4, 1), ('f', 5, 2), ('g', 6, 3)])
        self.runs = []
        for mode, count in [('indexed', 2), ('global', 3)]:
            cfg = NS(attack=NS(mode='test', fedmia_shadow_stat_mode=mode))
            fields = {key: {} for key in ('train_cos', 'test_cos', 'train_labels', 'test_labels')}
            for rnd in (10, 19):
                fields['train_cos'][rnd] = [.1] * count
                fields['train_labels'][rnd] = [0, 1, 2][:count]
                fields['test_cos'][rnd] = [.1] * 4
                fields['test_labels'][rnd] = [3, 4, 5, 6]
            result = NS(scores_member=[.9] * count,
                        scores_nonmember=[.1] * (2 if mode == 'indexed' else 4),
                        metadata={'rounds_used': [10, 19], 'shadow_stat_mode': mode})
            self.runs.append((cfg, fields, result))

    def test_explicit_prefix_with_matching_labels_and_rounds(self):
        common, info = align_indexed_replay(self.module, self.runs, self.items)
        self.assertEqual(common, (2, 2))
        self.assertEqual(info['originalCounts']['defense'], [3, 4])
        self.assertIn('不作为严格防御增益结论', info['message'])
        self.assertEqual(len(self.runs[1][2].scores_nonmember), 4)

    def test_unsafe_alignment_is_rejected(self):
        for issue in ('labels', 'rounds', 'mixed', 'generated', 'length', 'metadata'):
            with self.subTest(issue=issue):
                runs = copy.deepcopy(self.runs)
                cfg, fields, result = runs[0]
                if issue == 'labels':
                    fields['train_labels'][19] = [1, 0]
                elif issue == 'rounds':
                    result.metadata['rounds_used'] = [10]
                elif issue == 'mixed':
                    cfg.attack.mode = 'mix'
                elif issue == 'generated':
                    fields['augmented_cos'] = {10: [.3]}
                elif issue == 'length':
                    result.scores_member = [.9]
                else:
                    result.metadata = {}
                with self.assertRaises(PlatformError):
                    align_indexed_replay(self.module, runs, self.items)

    def test_small_sample_and_tied_scores_never_exceed_one_percent_fpr(self):
        handler = object.__new__(PlatformHandler)
        for negative in ([.1] * 27, [.1] * 100, np.linspace(0, .8, 250)):
            metrics = handler._fedmia_attack_metrics([.9, .8, .1], negative, self.module)
            self.assertLessEqual(metrics['fprAtThreshold'], .01)


if __name__ == '__main__':
    unittest.main()
