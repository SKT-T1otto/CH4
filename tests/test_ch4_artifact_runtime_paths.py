from __future__ import annotations

import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from registry.ch4_artifact_layout import create_training_run_directories
from tools import ch4_run_and_log


class _FakeProcess:
    def __init__(self) -> None:
        self.stdout = io.StringIO("runtime log\n")

    def wait(self) -> int:
        return 0


class Ch4ArtifactRuntimePathTests(unittest.TestCase):
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    def test_locked_training_with_shared_run_directory(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            shared = Path(temporary) / "perf_rbe/runs/run_seed2"
            created = create_training_run_directories(
                shared, shared, write_once=True
            )
            self.assertEqual(created, (shared.resolve(),))
            self.assertTrue(shared.is_dir())
            with self.assertRaises(RuntimeError):
                create_training_run_directories(shared, shared, write_once=True)

    def _assert_logger_isolated(self, mode: str) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            data_root = Path(temporary) / "data/chapter4_final"
            formal = data_root / "ablations/evaluations/experiment_v2"
            smoke = data_root / "smoke/ablations/experiment_v2"
            console_log = (
                data_root
                / "manifests/console_logs/experiment_v2"
                / f"{mode}.console.log"
            )
            argv = [
                "ch4_run_and_log.py",
                "--log",
                str(console_log),
                "--",
                "python",
                "placeholder.py",
            ]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                ch4_run_and_log.subprocess, "Popen", return_value=_FakeProcess()
            ):
                self.assertEqual(ch4_run_and_log.main(), 0)
            self.assertTrue(console_log.is_file())
            self.assertFalse(formal.exists())
            self.assertFalse(smoke.exists())

    def _assert_bat_console_path(self, filename: str) -> None:
        text = (self.PROJECT_ROOT / "scripts" / filename).read_text(
            encoding="utf-8"
        )
        self.assertIn(
            r"manifests\console_logs\%CH4_EXPERIMENT_ID%", text
        )
        self.assertIn(r"%MODE%.console.log", text)
        self.assertNotIn(r'--log "%LOG_DIR%', text)

    def test_logger_does_not_precreate_smoke_root(self) -> None:
        self._assert_logger_isolated("smoke")
        self._assert_bat_console_path(
            "select_ch4_rbe_perf_ablation_checkpoints_v2.bat"
        )

    def test_logger_does_not_precreate_formal_root(self) -> None:
        self._assert_logger_isolated("formal")
        self._assert_bat_console_path(
            "eval_ch4_rbe_perf_ablation_comparison_v2.bat"
        )


if __name__ == "__main__":
    unittest.main()
