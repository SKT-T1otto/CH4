from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools._internal.migrate_ch4_artifacts import (
    MigrationError,
    apply_plan,
    build_plan,
    verify_plan,
)


class Ch4ArtifactMigrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(
            dir=Path(__file__).resolve().parent
        )
        self.project = Path(self.temporary.name)
        self.data_root = self.project / "data" / "chapter4_final"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write(self, relative: str, payload: bytes) -> Path:
        path = self.project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return path

    def test_log_model_same_run_merge(self) -> None:
        run = "ch4_rbe_perf_ablation_no_boundary_seed2_2000ep_v1"
        self._write(f"log_ch4_rbe/{run}/training_progress.csv", b"episode\n1\n")
        self._write(f"models_ch4_rbe/{run}/snapshot_ep0100.pt", b"checkpoint")
        plan = build_plan(self.project, self.data_root)
        rows = [row for row in plan["items"] if row["artifact_name"] == run]
        self.assertEqual(len(rows), 2)
        self.assertEqual({row["planned_action"] for row in rows}, {"merge"})
        self.assertEqual(
            {row["canonical_relative_to_data_root"] for row in rows},
            {f"ablations/no_boundary/runs/{run}"},
        )

    def test_identical_file_reuse(self) -> None:
        selected = "ch4_uniform_dr_selected_full9d_v2"
        relative = "selected_by_uniform_dr_validation.pt"
        self._write(f"models_ch4_rbe/{selected}/{relative}", b"same")
        self._write(
            f"data/chapter4_final/baselines/uniform_dr/selected/{selected}/{relative}",
            b"same",
        )
        plan = build_plan(self.project, self.data_root)
        row = next(row for row in plan["items"] if row["artifact_name"] == selected)
        self.assertEqual(row["collision_status"], "reused_identical")
        self.assertEqual(row["planned_action"], "reused")

    def test_different_content_same_target_file_conflicts(self) -> None:
        run = "ch4_rbe_perf_ablation_all_actors_seed2_2000ep_v1"
        self._write(f"log_ch4_rbe/{run}/shared.json", b'{"source":"log"}')
        self._write(f"models_ch4_rbe/{run}/shared.json", b'{"source":"model"}')
        plan = build_plan(self.project, self.data_root)
        rows = [row for row in plan["items"] if row["artifact_name"] == run]
        self.assertEqual(
            {row["collision_status"] for row in rows}, {"conflict"}
        )
        self.assertEqual({row["planned_action"] for row in rows}, {"reject"})

    def test_unclassified_item_is_never_auto_moved(self) -> None:
        self._write("eval_ch4_rbe/mystery_payload/result.txt", b"unknown")
        plan = build_plan(self.project, self.data_root)
        row = plan["items"][0]
        self.assertEqual(row["stage"], "unclassified")
        self.assertEqual(row["planned_action"], "manual_review")
        self.assertEqual(plan["summary"]["unclassified_items"], 1)
        with self.assertRaises(MigrationError):
            apply_plan(self.project, self.data_root)

    def test_verify_plan_reads_json_and_csv_headers(self) -> None:
        run = "ch4_rbe_perf_ablation_no_nominal_seed2_2000ep_v1"
        self._write(f"log_ch4_rbe/{run}/training_config.json", b'{"seed":2}')
        self._write(f"log_ch4_rbe/{run}/training_progress.csv", b"episode,reward\n1,0\n")
        build_plan(self.project, self.data_root)
        audit = verify_plan(self.project, self.data_root, validate_pt=False)
        self.assertTrue(audit["overall_pass"])
        self.assertEqual(audit["checks"]["json_files_parsed"], 1)
        self.assertEqual(audit["checks"]["csv_headers_read"], 1)

    def test_apply_interruption_before_source_delete(self) -> None:
        selected = "ch4_uniform_dr_selected_full9d_v2"
        source = self._write(
            f"models_ch4_rbe/{selected}/model.bin", b"selected-model"
        ).parent
        build_plan(self.project, self.data_root)

        def interrupt(phase, _context):
            if phase == "after_copy_all":
                raise RuntimeError("simulated interruption")

        with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
            apply_plan(
                self.project,
                self.data_root,
                validate_pt=False,
                failure_injector=interrupt,
            )
        self.assertTrue(source.is_dir())
        state = json.loads(
            (
                self.data_root
                / "manifests/migration_transaction_state.json"
            ).read_text(encoding="utf-8")
        )
        self.assertFalse(state["complete"])
        relocation = json.loads(
            (
                self.data_root / "manifests/relocation_manifest.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(relocation["relocations"], [])

    def test_apply_resume_after_partial_commit(self) -> None:
        uniform_id = "ch4_uniform_dr_selected_full9d_v2"
        perf_id = "ch4_rbe_boundary_core_perf_selected_seed2_v1"
        uniform_source = self._write(
            f"models_ch4_rbe/{uniform_id}/uniform.bin", b"uniform"
        ).parent
        perf_source = self._write(
            f"models_ch4_rbe/{perf_id}/perf.bin", b"perf"
        ).parent
        build_plan(self.project, self.data_root)
        commit_count = 0

        def interrupt(phase, _context):
            nonlocal commit_count
            if phase == "after_target_commit":
                commit_count += 1
                if commit_count == 1:
                    raise RuntimeError("partial commit")

        with self.assertRaisesRegex(RuntimeError, "partial commit"):
            apply_plan(
                self.project,
                self.data_root,
                validate_pt=False,
                failure_injector=interrupt,
            )
        self.assertTrue(uniform_source.exists())
        self.assertTrue(perf_source.exists())

        result = apply_plan(self.project, self.data_root, validate_pt=False)
        self.assertTrue(result["overall_pass"])
        self.assertGreaterEqual(result["reused_items"], 1)
        self.assertFalse(uniform_source.exists())
        self.assertFalse(perf_source.exists())
        state = json.loads(
            (
                self.data_root
                / "manifests/migration_transaction_state.json"
            ).read_text(encoding="utf-8")
        )
        self.assertTrue(state["complete"])
        self.assertTrue(all(record["deleted"] for record in state["sources"].values()))

    def test_tampered_plan_target_rejected(self) -> None:
        selected = "ch4_uniform_dr_selected_full9d_v2"
        self._write(f"models_ch4_rbe/{selected}/model.bin", b"model")
        build_plan(self.project, self.data_root)
        plan_path = (
            self.data_root / "manifests/artifact_migration_plan.json"
        )
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
        payload["items"][0]["canonical_relative_to_data_root"] = "../outside"
        plan_path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(MigrationError):
            verify_plan(self.project, self.data_root, validate_pt=False)
        with self.assertRaises(MigrationError):
            apply_plan(self.project, self.data_root, validate_pt=False)

    def test_missing_manifest_reference_fails_verification(self) -> None:
        run = "ch4_rbe_perf_ablation_no_boundary_seed2_2000ep_v1"
        self._write(
            f"log_ch4_rbe/{run}/training_config.json",
            json.dumps(
                {
                    "selected_model_path": "missing_model.pt",
                    "selected_model_sha256": "a" * 64,
                }
            ).encode("utf-8"),
        )
        build_plan(self.project, self.data_root)
        with self.assertRaisesRegex(MigrationError, "referenced artifact missing"):
            verify_plan(self.project, self.data_root, validate_pt=False)


if __name__ == "__main__":
    unittest.main()
