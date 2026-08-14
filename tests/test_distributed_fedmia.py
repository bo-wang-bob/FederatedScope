import base64
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from federatedscope.contrib.attack.distributed_fedmia import \
    DistributedFedMIACollector
from federatedscope.core.auxiliaries.utils import recursive_param2tensor


def _fake_server(outdir):
    attack = SimpleNamespace(
        fedmia_target_client_id=1,
        fedmia_var_floor=1e-8,
        fedmia_shadow_stat_mode='global',
        fedmia_round_agg='mean',
        fedmia_i_round_agg='mean',
        fedmia_compute_all_clients=True,
        fedmia_cross_eval=False,
        mode='mix',
        mix_length=1000,
        ggeur_normalize_fields=['train_losses'],
    )
    cfg = SimpleNamespace(
        outdir=str(outdir),
        attack=attack,
        federate=SimpleNamespace(total_round_num=2),
    )
    return SimpleNamespace(_cfg=cfg)


def _report(client_id, member_loss, nonmember_loss,
            member_cos, nonmember_cos, labels=None):
    return {
        'client_id': client_id,
        'round': 1,
        'member_loss': member_loss,
        'nonmember_loss': nonmember_loss,
        'member_cos': member_cos,
        'nonmember_cos': nonmember_cos,
        'member_labels': labels or list(range(len(member_loss))),
    }


class DistributedFedMIATest(unittest.TestCase):

    def test_recursive_param2tensor_restores_nested_model_state(self):
        weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        encoded = base64.b64encode(pickle.dumps(weight)).decode('ascii')
        restored = recursive_param2tensor({
            'mlp': {'weight': encoded},
            'extra': [encoded],
        })

        self.assertTrue(torch.equal(restored['mlp']['weight'], weight))
        self.assertTrue(torch.equal(restored['extra'][0], weight))

    def test_collector_separates_scores(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            collector = DistributedFedMIACollector(_fake_server(outdir))
            collector.reports = {
                1: {
                    1: _report(1, [0.1, 0.2], [2.1, 2.2],
                               [0.9, 0.8], [-0.2, -0.1]),
                    2: _report(2, [2.0, 1.9], [2.2, 2.3],
                               [0.0, 0.1], [-0.1, -0.2]),
                }
            }

            processed = collector._processed_reports()
            result = collector._attack_client(1, processed)
            self.assertEqual(result['fedmia_i']['auc'], 1.0)
            self.assertEqual(result['fedmia_ii']['auc'], 1.0)
            self.assertEqual(result['fedmia_i']['rounds_used'], [1])

            collector.finalize()
            self.assertTrue(
                (outdir / 'distributed_fedmia_results.json').is_file())

    def test_cross_eval_builds_global_test_and_mix_nonmembers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = DistributedFedMIACollector(
                _fake_server(Path(tmpdir)))
            collector.cross_eval_owners[1] = [1, 2]
            collector.cross_eval_responses[1] = {
                1: {
                    'entries': {
                        '1': {
                            'member_loss': [0.1, 0.2],
                            'member_cos': [0.9, 0.8],
                            'member_labels': [0, 1],
                            'test_loss': [1.0],
                            'test_cos': [-0.1],
                        },
                        '2': {
                            'mix_loss': [1.5],
                            'mix_cos': [-0.2],
                            'test_loss': [1.1],
                            'test_cos': [-0.1],
                        },
                    }
                },
                2: {
                    'entries': {
                        '1': {
                            'mix_loss': [1.6],
                            'mix_cos': [-0.3],
                            'test_loss': [1.2],
                            'test_cos': [-0.2],
                        },
                        '2': {
                            'member_loss': [0.3, 0.4],
                            'member_cos': [0.7, 0.6],
                            'member_labels': [0, 1],
                            'test_loss': [1.3],
                            'test_cos': [-0.2],
                        },
                    }
                },
            }

            collector._merge_cross_eval_round(1)
            report = collector.reports[1][1]
            self.assertEqual(report['member_loss'], [0.1, 0.2])
            self.assertEqual(report['nonmember_loss'], [1.0, 1.2, 1.6])
            self.assertEqual(report['metadata']['test_count'], 2)
            self.assertEqual(report['metadata']['mix_count'], 1)


if __name__ == '__main__':
    unittest.main()
