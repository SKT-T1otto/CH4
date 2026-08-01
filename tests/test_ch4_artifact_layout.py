from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import registry.ch4_artifact_layout as layout


class Ch4ArtifactLayoutTests(unittest.TestCase):
    def test_name_classification(self) -> None:
        cases = {
            ("ch4_clean_baseline_seed1", "log_ch4_rbe"): ("baselines/clean", "runs"),
            ("ch4_uniform_dr_selected_full9d_v2", "models_ch4_rbe"): (
                "baselines/uniform_dr",
                "selected",
            ),
            ("ch4_found_aware_reb_dataset_v1", "eval_ch4_rbe"): ("reb", "datasets"),
            ("ch4_rbe_boundary_core_seed2_v1", "models_ch4_rbe"): (
                "rbe_legacy",
                "runs",
            ),
            ("ch4_rbe_boundary_core_perf_seed2_v1", "models_ch4_rbe"): (
                "perf_rbe",
                "runs",
            ),
        }
        for (name, source), expected in cases.items():
            with self.subTest(name=name):
                result = layout.classify_artifact_name(name, source)
                self.assertEqual((result["family"], result["stage"]), expected)

    def test_ablation_selected_and_eval_stages(self) -> None:
        selected = layout.classify_artifact_name(
            "ch4_rbe_perf_ablation_no_boundary_selected_seed2_v1",
            "models_ch4_rbe",
        )
        selection = layout.classify_artifact_name(
            "ch4_rbe_perf_ablation_checkpoint_selection_v2", "eval_ch4_rbe"
        )
        comparison = layout.classify_artifact_name(
            "ch4_rbe_perf_ablation_comparison_v2", "eval_ch4_rbe"
        )
        training_audit = layout.classify_artifact_name(
            "ch4_rbe_perf_ablation_training_audit_v1", "eval_ch4_rbe"
        )
        self.assertEqual(selected["stage"], "selected")
        self.assertEqual(
            selected["canonical_relative_path"],
            "ablations/no_boundary/selected/"
            "ch4_rbe_perf_ablation_no_boundary_selected_seed2_v1",
        )
        self.assertEqual(selection["stage"], "selections")
        self.assertEqual(
            selection["canonical_relative_path"],
            "ablations/selections/ch4_rbe_perf_ablation_checkpoint_selection_v2",
        )
        self.assertEqual(comparison["stage"], "evaluations")
        self.assertEqual(training_audit["stage"], "training_audits")
        self.assertEqual(
            training_audit["canonical_relative_path"],
            "ablations/training_audits/ch4_rbe_perf_ablation_training_audit_v1",
        )

    def test_interrupted_directory_classification(self) -> None:
        result = layout.classify_artifact_name(
            "ch4_uniform_dr_seed1.interrupted_20260730", "archive"
        )
        self.assertEqual(result["family"], "archive")
        self.assertEqual(result["variant"], "interrupted")
        self.assertTrue(
            result["canonical_relative_path"].startswith("archive/interrupted/")
        )

    def test_stress_tests_family_supported(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            data_root = Path(temporary) / "data/chapter4_final"
            with mock.patch.dict(os.environ, {"CH4_DATA_ROOT": str(data_root)}):
                self.assertEqual(
                    layout.get_family_root("stress_tests"),
                    data_root.resolve() / "stress_tests",
                )
                self.assertEqual(
                    layout.get_evaluation_dir("stress_tests", "severity_sweep_v1"),
                    data_root.resolve()
                    / "stress_tests/evaluations/severity_sweep_v1",
                )
            classified = layout.classify_artifact_name(
                "ch4_disturbance_intensity_sweep_v1", "eval_ch4_rbe"
            )
            self.assertEqual(classified["family"], "stress_tests")
            self.assertEqual(classified["stage"], "evaluations")

    def test_canonical_path_resolves_legacy_before_apply(self) -> None:
        tests_dir = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory(dir=tests_dir) as temporary:
            project = Path(temporary)
            data_root = project / "data/chapter4_final"
            legacy = (
                project
                / "models_ch4_rbe/ch4_uniform_dr_selected_full9d_v2/model.pt"
            )
            legacy.parent.mkdir(parents=True)
            legacy.write_bytes(b"legacy-model")
            manifest_dir = data_root / "manifests"
            manifest_dir.mkdir(parents=True)
            (manifest_dir / "artifact_migration_plan.json").write_text(
                json.dumps(
                    {
                        "items": [
                            {
                                "source_path": (
                                    "models_ch4_rbe/"
                                    "ch4_uniform_dr_selected_full9d_v2"
                                ),
                                "canonical_relative_to_data_root": (
                                    "baselines/uniform_dr/selected/"
                                    "ch4_uniform_dr_selected_full9d_v2"
                                ),
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            canonical = (
                data_root
                / "baselines/uniform_dr/selected/"
                "ch4_uniform_dr_selected_full9d_v2/model.pt"
            )
            with mock.patch.object(layout, "PROJECT_ROOT", project), mock.patch.dict(
                os.environ, {"CH4_DATA_ROOT": str(data_root)}
            ):
                resolved = layout.resolve_artifact_path(canonical)
            self.assertEqual(resolved, legacy.resolve())

    def test_relocation_manifest_resolves_relative_absolute_and_windows_paths(self) -> None:
        tests_dir = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory(dir=tests_dir) as temporary:
            root = Path(temporary)
            project = root / "project"
            data_root = project / "data" / "chapter4_final"
            canonical = (
                data_root
                / "perf_rbe/selected/ch4_rbe_boundary_core_perf_selected_seed2_v1"
                / "selected_rbe_model.pt"
            )
            canonical.parent.mkdir(parents=True)
            canonical.write_bytes(b"model")
            manifest_dir = data_root / "manifests"
            manifest_dir.mkdir(parents=True)
            manifest = {
                "relocations": [
                    {
                        "legacy_relative_path": (
                            "models_ch4_rbe/"
                            "ch4_rbe_boundary_core_perf_selected_seed2_v1"
                        ),
                        "legacy_absolute_path_at_migration": (
                            r"E:\retired\RBE7.29\models_ch4_rbe"
                            r"\ch4_rbe_boundary_core_perf_selected_seed2_v1"
                        ),
                        "canonical_relative_path": (
                            "perf_rbe/selected/"
                            "ch4_rbe_boundary_core_perf_selected_seed2_v1"
                        ),
                    }
                ]
            }
            (manifest_dir / "relocation_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            with mock.patch.object(layout, "PROJECT_ROOT", project), mock.patch.dict(
                os.environ, {"CH4_DATA_ROOT": str(data_root)}
            ):
                relative = layout.resolve_artifact_path(
                    "models_ch4_rbe/"
                    "ch4_rbe_boundary_core_perf_selected_seed2_v1/"
                    "selected_rbe_model.pt"
                )
                absolute = layout.resolve_artifact_path(
                    r"E:\retired\RBE7.29\models_ch4_rbe"
                    r"\ch4_rbe_boundary_core_perf_selected_seed2_v1"
                    r"\selected_rbe_model.pt"
                )
                self.assertEqual(relative, canonical.resolve())
                self.assertEqual(absolute, canonical.resolve())

    def test_artifact_relative_path_avoids_machine_only_absolute_path(self) -> None:
        tests_dir = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory(dir=tests_dir) as temporary:
            project = Path(temporary)
            data_root = project / "data" / "chapter4_final"
            artifact = data_root / "manifests" / "example.json"
            artifact.parent.mkdir(parents=True)
            artifact.write_text("{}", encoding="utf-8")
            with mock.patch.object(layout, "PROJECT_ROOT", project), mock.patch.dict(
                os.environ, {"CH4_DATA_ROOT": str(data_root)}
            ):
                serialized = layout.artifact_relative_path(artifact)
            self.assertEqual(
                serialized["artifact_path_relative_to_data_root"],
                "manifests/example.json",
            )
            self.assertEqual(
                serialized["artifact_path_relative_to_project_root"],
                "data/chapter4_final/manifests/example.json",
            )
            self.assertEqual(len(serialized["artifact_sha256"]), 64)

    def test_source_lock_accepts_only_exact_attested_layout_change(self) -> None:
        tests_dir = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory(dir=tests_dir) as temporary:
            data_root = Path(temporary) / "data/chapter4_final"
            manifest_dir = data_root / "manifests"
            manifest_dir.mkdir(parents=True)
            (manifest_dir / "layout_refactor_attestation.json").write_text(
                json.dumps(
                    {
                        "modified_source_files": ["train.py"],
                        "old_source_hashes": {"train.py": "old"},
                        "new_source_hashes": {"train.py": "new"},
                        "algorithm_semantics_changed": False,
                        "model_artifacts_changed": False,
                        "historical_results_rewritten": False,
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch.dict(os.environ, {"CH4_DATA_ROOT": str(data_root)}):
                self.assertTrue(
                    layout.source_hash_matches_layout_attestation(
                        "train.py", "old", "new"
                    )
                )
                self.assertFalse(
                    layout.source_hash_matches_layout_attestation(
                        "train.py", "wrong", "new"
                    )
                )


if __name__ == "__main__":
    unittest.main()
