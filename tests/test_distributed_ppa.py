import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from federatedscope.contrib.attack.distributed_ppa import \
    DistributedPPAClientReporter, DistributedPPACollector


def _attack_cfg(classifier='svm'):
    return SimpleNamespace(
        classifier_PIA=classifier,
        meta_ppa_save_interval=1,
        meta_ppa_probe_samples_per_class=2,
        meta_ppa_probe_epochs=1,
        meta_ppa_probe_lr=0.001,
        meta_ppa_probe_batch_size=2,
        meta_ppa_max_attack_rounds=10,
        meta_ppa_max_clients=0,
        meta_ppa_target_layers=[],
    )


class DistributedPPATest(unittest.TestCase):

    def test_client_builds_majority_label_and_fixed_feature_shape(self):
        cfg = SimpleNamespace(
            seed=7,
            attack=_attack_cfg(),
            model=SimpleNamespace(num_classes=2),
            federate=SimpleNamespace(total_round_num=3),
        )
        features = torch.tensor([
            [1.0, 0.0], [0.8, 0.2], [0.9, 0.1], [0.0, 1.0]
        ])
        labels = torch.tensor([0, 0, 0, 1])
        model = torch.nn.Linear(2, 2)
        client = SimpleNamespace(
            ID=1,
            _cfg=cfg,
            trainer=SimpleNamespace(
                ctx=SimpleNamespace(data={'train': object()})),
            data={},
            mlp_classifier=model,
            device='cpu',
            feature_extractor_type='cnn',
            embedding_dim=2,
            _extract_eval_features=lambda _: (features, labels),
        )
        reporter = DistributedPPAClientReporter(client)
        global_state = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }
        local_state = {
            key: value.detach().clone() + 0.05
            for key, value in model.state_dict().items()
        }

        report = reporter.build_report(1, global_state, local_state)

        self.assertEqual(report['property_label'], 0)
        # 2 classes x (weight,bias) x (L1,L2,max)
        self.assertEqual(report['metadata']['feature_shape'], [2, 6])
        self.assertEqual(len(report['feature']), 12)
        self.assertTrue(np.all(np.isfinite(report['feature'])))
        self.assertEqual(report['metadata']['derived_features_only'], 1)

    def test_collector_trains_meta_classifier_and_saves_result(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = SimpleNamespace(
                outdir=tmpdir,
                seed=13,
                attack=_attack_cfg(),
                model=SimpleNamespace(num_classes=2),
            )
            collector = DistributedPPACollector(SimpleNamespace(_cfg=cfg))
            for round_idx in range(1, 6):
                collector.reports[round_idx] = {}
                for client_id, label in [(1, 0), (2, 0), (3, 1), (4, 1)]:
                    base = float(label * 10)
                    collector.reports[round_idx][client_id] = {
                        'client_id': client_id,
                        'round': round_idx,
                        'property_label': label,
                        'feature': [
                            base + round_idx * 0.01,
                            base + client_id * 0.001,
                        ],
                        'metadata': {
                            'target_layers': ['weight', 'bias'],
                            'probe_counts': {'0': 2, '1': 2},
                        },
                    }

            collector.finalize()

            result_path = Path(tmpdir) / 'distributed_ppa_results.json'
            self.assertTrue(result_path.is_file())
            result = json.loads(result_path.read_text())['meta_ppa']
            self.assertEqual(result['accuracy'], 1.0)
            self.assertEqual(result['num_total_samples'], 20)
            self.assertEqual(result['observed_num_classes'], 2)


if __name__ == '__main__':
    unittest.main()
