from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import registry.ch4_artifact_layout as layout
from tools._internal.migrate_ch4_artifacts import (
    apply_plan,
    build_plan,
    status,
)


class Ch4ArtifactFinalRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(
            dir=Path(__file__).resolve().parent
        )
        self.project = Path(self.temporary.name)
        self.data_root = self.project / "data/chapter4_final"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write(self, relative: str, payload: bytes) -> Path:
        path = self.project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return path

    def test_missing_old_absolute_legacy_path_rebases_to_current_project(self) -> None:
        current = self._write(
            "models_ch4_rbe/ch4_uniform_dr_seed1_from_clean_ep8200_2000ep_v1/"
            "snapshot_ep1400.pt",
            b"uniform-checkpoint",
        )
        old_project = self.project.parent / "retired_RBE7.15"
        old_absolute = (
            old_project
            / "models_ch4_rbe/ch4_uniform_dr_seed1_from_clean_ep8200_2000ep_v1/"
            "snapshot_ep1400.pt"
        )
        with mock.patch.object(layout, "PROJECT_ROOT", self.project), mock.patch.dict(
            os.environ, {"CH4_DATA_ROOT": str(self.data_root)}
        ):
            resolved = layout.resolve_artifact_path(old_absolute)
        self.assertEqual(resolved, current.resolve())

    def test_apply_recovers_delete_after_unrecorded_source_removal(self) -> None:
        selected = "ch4_uniform_dr_selected_full9d_v2"
        source = self._write(
            f"models_ch4_rbe/{selected}/model.bin", b"selected-model"
        ).parent
        build_plan(self.project, self.data_root)

        def interrupt(phase, _context):
            if phase == "after_source_delete_before_state_save":
                raise RuntimeError("delete-window interruption")

        with self.assertRaisesRegex(RuntimeError, "delete-window interruption"):
            apply_plan(
                self.project,
                self.data_root,
                validate_pt=False,
                failure_injector=interrupt,
            )

        self.assertFalse(source.exists())
        mid_status = status(self.project, self.data_root)
        self.assertFalse(mid_status["overall_pass"])
        self.assertFalse(mid_status["transaction_complete"])

        result = apply_plan(self.project, self.data_root, validate_pt=False)
        self.assertTrue(result["overall_pass"])
        final_status = status(self.project, self.data_root)
        self.assertTrue(final_status["overall_pass"])
        self.assertTrue(final_status["transaction_complete"])
        self.assertTrue(final_status["source_deletion_completed"])

    def test_final_summary_precedes_complete_transaction_marker(self) -> None:
        selected = "ch4_uniform_dr_selected_full9d_v2"
        self._write(f"models_ch4_rbe/{selected}/model.bin", b"selected-model")
        build_plan(self.project, self.data_root)

        def interrupt(phase, _context):
            if phase == "after_final_summary_before_complete":
                raise RuntimeError("summary-window interruption")

        with self.assertRaisesRegex(RuntimeError, "summary-window interruption"):
            apply_plan(
                self.project,
                self.data_root,
                validate_pt=False,
                failure_injector=interrupt,
            )

        summary = json.loads(
            (
                self.data_root / "manifests/migration_apply_summary.json"
            ).read_text(encoding="utf-8")
        )
        state = json.loads(
            (
                self.data_root / "manifests/migration_transaction_state.json"
            ).read_text(encoding="utf-8")
        )
        self.assertTrue(summary["overall_pass"])
        self.assertTrue(summary["source_deletion_completed"])
        self.assertFalse(state["complete"])
        self.assertFalse(status(self.project, self.data_root)["overall_pass"])

        result = apply_plan(self.project, self.data_root, validate_pt=False)
        self.assertTrue(result["overall_pass"])
        self.assertTrue(status(self.project, self.data_root)["overall_pass"])


if __name__ == "__main__":
    unittest.main()
