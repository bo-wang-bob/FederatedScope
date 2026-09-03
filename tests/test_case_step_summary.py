import importlib.util
import tempfile
import unittest
from pathlib import Path


RUNNER_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" /
    "test_outline_validation" / "case_step_runner.py"
)
SPEC = importlib.util.spec_from_file_location("case_step_runner_test", RUNNER_PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class HumanStepSummaryTest(unittest.TestCase):

    def test_summary_records_completion_and_result_file(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "step_08_client.txt"
            RUNNER.write_human_step_summary(
                output,
                "T-01",
                8,
                "client",
                "PASS",
                "步骤8已完成：性能测试服务已停止，测试证据压缩包已生成。",
                {
                    "ARCHIVE_RELATIVE_PATH":
                    "docs/test_logs/concurrent_availability/outline_t01/"
                    "performance_evidence.zip",
                    "ARCHIVE_COMPLETE": "true",
                    "PORTS_RELEASED": "true",
                },
            )
            content = output.read_text(encoding="utf-8")
            self.assertIn("执行状态：通过", content)
            self.assertIn("步骤8已完成", content)
            self.assertNotIn("本步骤生成或校验的文件：", content)
            self.assertIn("归档文件：docs/test_logs", content)
            self.assertIn("ARCHIVE_COMPLETE=true", content)

    def test_result_paths_have_clear_chinese_labels(self):
        paths = RUNNER.result_path_entries({
            "ACCURACY_SUMMARY_GLOB": "runs/*/accuracy_summary.json",
            "RESULT_ROOT_RELATIVE_PATH": "runs/outline_t02",
        })
        self.assertEqual(paths, [
            ("准确率汇总文件", "runs/*/accuracy_summary.json"),
            ("结果目录", "runs/outline_t02"),
        ])


if __name__ == "__main__":
    unittest.main()
