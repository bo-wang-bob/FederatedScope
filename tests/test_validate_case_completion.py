import json
import subprocess
import sys
from pathlib import Path


SCRIPT = (Path(__file__).parents[1] / "scripts" / "distributed_scripts" /
          "ggeur_hierarchical_3machine" / "validate_case_completion.py")


def test_validator_accepts_server_and_terminal_records_at_round_99(tmp_path):
    log = tmp_path / "root.log"
    lines = [
        f"Round {round_idx} aggregation complete, valid_updates=2/2"
        for round_idx in range(1, 100)
    ]
    lines.extend([
        "Round 99 MLP Test Accuracy - emnist_digits: 0.3, average: 0.4",
        "Round 99 MLP Test Accuracy - client_weighted_average: 0.5, average: 0.6",
        "Training finished after 100 rounds",
    ])
    log.write_text("\n".join(lines), encoding="utf-8")
    evidence = tmp_path / "client_model_accuracy_round_99.json"
    evidence.write_text(json.dumps({
        "round": 99,
        "client_count": 2,
        "clients": [{"client_id": 1}, {"client_id": 2}],
    }), encoding="utf-8")
    output = tmp_path / "validation.json"

    result = subprocess.run([
        sys.executable, str(SCRIPT), str(log),
        "--output", str(output),
        "--expected-rounds", "100",
        "--expected-accuracy-rounds", "99",
        "--eval-frequency", "99",
        "--accuracy-records-per-eval", "2",
        "--expected-updates", "2",
        "--client-evidence", str(evidence),
        "--expected-clients", "2",
    ], check=False)

    assert result.returncode == 0
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "pass"
