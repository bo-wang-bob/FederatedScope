import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = (Path(__file__).resolve().parents[1] / "scripts" /
              "distributed_scripts" / "ggeur_hierarchical_3machine")
sys.path.insert(0, str(SCRIPT_DIR))
import check_accuracy_acceptance as acceptance  # noqa: E402


def _write_summary(root, method, final, best=None):
    path = Path(root) / f"officehome_vit_{method}" / "accuracy_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "round_count": 99,
        "last": {"round": 99, "metrics": {"average": final}},
        "best_average": final if best is None else best,
    }), encoding="utf-8")


class AccuracyAcceptanceTest(unittest.TestCase):

    def test_absolute_improvement_must_pass_both_baselines(self):
        results = {
            "fedavg": {"final": 0.39},
            "fedprox": {"final": 0.47},
            "platform": {"final": 0.767},
        }
        report = acceptance.improvement_report(results, 20.0)
        self.assertEqual(report["status"], "PASS")
        self.assertAlmostEqual(report["platform_minus_fedavg_pp"], 37.7)
        self.assertAlmostEqual(report["platform_minus_fedprox_pp"], 29.7)

    def test_absolute_improvement_fails_if_either_baseline_is_not_over_20_pp(self):
        results = {
            "fedavg": {"final": 0.39},
            "fedprox": {"final": 0.58},
            "platform": {"final": 0.767},
        }
        report = acceptance.improvement_report(results, 20.0)
        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(report["platform_vs_fedavg_pass"])
        self.assertFalse(report["platform_vs_fedprox_pass"])

    def test_parity_threshold_is_in_percentage_points(self):
        distributed = {"platform": {"final": 0.767}}
        standalone = {"platform": {"final": 0.781}}
        report = acceptance.parity_report(
            distributed, standalone, 2.0, ("platform",))
        self.assertEqual(report["status"], "PASS")
        self.assertAlmostEqual(report["methods"]["platform"]["gap_pp"], 1.4)

    def test_missing_standalone_does_not_block_distributed_acceptance(self):
        with tempfile.TemporaryDirectory() as directory:
            for method, final in {
                    "fedavg": 0.3933, "fedprox": 0.4751,
                    "fedproto": 0.6210, "fedopt": 0.3944,
                    "moon": 0.3933, "ggeur": 0.7670}.items():
                _write_summary(directory, method, final)
            args = acceptance.parse_args([
                "--distributed-root", directory,
                "--allow-pending",
            ])
            report = acceptance.build_report(args)
            self.assertEqual(report["overall_status"], "PASS")
            self.assertEqual(
                report["distributed"]["improvement"]["status"], "PASS")

    def test_ggeur_only_standalone_can_complete_parity_gate(self):
        with tempfile.TemporaryDirectory() as distributed_dir, \
                tempfile.TemporaryDirectory() as standalone_dir:
            for method, final in {
                    "fedavg": 0.3933, "fedprox": 0.4751,
                    "fedproto": 0.6210, "fedopt": 0.3944,
                    "moon": 0.3933, "ggeur": 0.7670}.items():
                _write_summary(distributed_dir, method, final)
            _write_summary(standalone_dir, "ggeur", 0.778)
            args = acceptance.parse_args([
                "--distributed-root", distributed_dir,
                "--standalone-root", standalone_dir,
            ])
            report = acceptance.build_report(args)
            self.assertEqual(report["overall_status"], "PASS")
            self.assertEqual(report["parity"]["status"], "PASS")
            self.assertEqual(
                report["standalone"]["improvement"]["status"], "PENDING")


if __name__ == "__main__":
    unittest.main()
