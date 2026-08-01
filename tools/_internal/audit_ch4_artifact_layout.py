#!/usr/bin/env python3
"""Audit the canonical Chapter-4 artifact layout without running experiments."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from registry.ch4_artifact_layout import (  # noqa: E402
    LAYOUT_VERSION,
    classify_artifact_name,
    file_sha256,
    get_ch4_data_root,
    get_evaluation_dir,
    get_selected_dir,
    get_training_run_dir,
    resolve_artifact_path,
)
from tools._internal.migrate_ch4_artifacts import (  # noqa: E402
    MIGRATION_TOOL_VERSION,
    SOURCE_ROOT_NAMES,
    atomic_json,
    status as migration_status,
)


AUDIT_TOOL_VERSION = "20260730-v1"
KEY_MODELS = {
    "perf_rbe_seed2": {
        "family": "perf_rbe",
        "selected_id": "ch4_rbe_boundary_core_perf_selected_seed2_v1",
        "filename": "selected_rbe_model.pt",
        "legacy": "models_ch4_rbe/ch4_rbe_boundary_core_perf_selected_seed2_v1/selected_rbe_model.pt",
        "sha256": "5aa6cc539d86863b5eac0b50cdcf5bc4c4d1a5c97fa59670c0742cc23f66331b",
    },
    "uniform_dr": {
        "family": "uniform_dr",
        "selected_id": "ch4_uniform_dr_selected_full9d_v2",
        "filename": "selected_by_uniform_dr_validation.pt",
        "legacy": "models_ch4_rbe/ch4_uniform_dr_selected_full9d_v2/selected_by_uniform_dr_validation.pt",
        "sha256": "47ae3748017c4c9a43efc16ecb9973b535716c0e09d119f2de43615b8e5405e7",
    },
    "no_boundary": {
        "family": "no_boundary",
        "selected_id": "ch4_rbe_perf_ablation_no_boundary_selected_seed2_v1",
        "filename": "selected_ablation_model.pt",
        "legacy": "models_ch4_rbe/ch4_rbe_perf_ablation_no_boundary_selected_seed2_v1/selected_ablation_model.pt",
        "sha256": "e09f13ceac6fa12bd5dff7ec41118919a885f9ff80cc6da79dd6233636e08b5b",
    },
    "all_actors": {
        "family": "all_actors",
        "selected_id": "ch4_rbe_perf_ablation_all_actors_selected_seed2_v1",
        "filename": "selected_ablation_model.pt",
        "legacy": "models_ch4_rbe/ch4_rbe_perf_ablation_all_actors_selected_seed2_v1/selected_ablation_model.pt",
        "sha256": "bd0ac25eefec17ae575e0f0ef656a97dc182457a95ceaf95209c88fcc49a7a67",
    },
    "no_nominal": {
        "family": "no_nominal",
        "selected_id": "ch4_rbe_perf_ablation_no_nominal_selected_seed2_v1",
        "filename": "selected_ablation_model.pt",
        "legacy": "models_ch4_rbe/ch4_rbe_perf_ablation_no_nominal_selected_seed2_v1/selected_ablation_model.pt",
        "sha256": "a426c67ed65cc7e5ebe6b6b60aacaaf5a83320a241bf3d5e3ace327523e4d540",
    },
}
LATEST_ABLATION_ID = "ch4_rbe_perf_ablation_comparison_v2"
LATEST_ABLATION_LEGACY = (
    "eval_ch4_rbe/ch4_rbe_perf_ablation_comparison_v2/"
    "ablation_comparison_summary.json"
)


def _existing_artifact(canonical: Path, legacy: str) -> Path:
    if canonical.is_file():
        return canonical.resolve()
    resolved = resolve_artifact_path(legacy)
    return resolved.resolve() if resolved.is_file() else canonical


def self_test() -> dict[str, Any]:
    cases = {
        "selected": classify_artifact_name(
            "ch4_rbe_boundary_core_perf_selected_seed2_v1", "models_ch4_rbe"
        ),
        "run": classify_artifact_name(
            "ch4_rbe_perf_ablation_no_boundary_seed2_2000ep_v1", "log_ch4_rbe"
        ),
        "comparison": classify_artifact_name(
            "ch4_rbe_perf_ablation_comparison_v2", "eval_ch4_rbe"
        ),
        "interrupted": classify_artifact_name(
            "ch4_uniform_dr_seed1.interrupted_20260730", "archive"
        ),
    }
    checks = {
        "layout_version": LAYOUT_VERSION == 1,
        "selected_classified": cases["selected"]["stage"] == "selected",
        "run_classified": cases["run"]["stage"] == "runs",
        "comparison_classified": cases["comparison"]["stage"] == "evaluations",
        "interrupted_classified": cases["interrupted"]["variant"] == "interrupted",
        "run_path_shape": get_training_run_dir("no_boundary", "run_x").as_posix().endswith(
            "ablations/no_boundary/runs/run_x"
        ),
    }
    return {
        "audit_tool_version": AUDIT_TOOL_VERSION,
        "checks": checks,
        "cases": cases,
        "overall_pass": all(checks.values()),
    }


def _model_checks() -> dict[str, Any]:
    results: dict[str, Any] = {}
    for label, spec in KEY_MODELS.items():
        canonical = (
            get_selected_dir(spec["family"], spec["selected_id"]) / spec["filename"]
        )
        actual_path = _existing_artifact(canonical, spec["legacy"])
        exists = actual_path.is_file()
        actual_sha = file_sha256(actual_path) if exists else None
        results[label] = {
            "path": str(actual_path),
            "expected_sha256": spec["sha256"],
            "actual_sha256": actual_sha,
            "exists": exists,
            "sha256_match": actual_sha == spec["sha256"],
        }
    return results


def _latest_result_check() -> dict[str, Any]:
    canonical = (
        get_ch4_data_root()
        / "ablations/evaluations"
        / LATEST_ABLATION_ID
        / "ablation_comparison_summary.json"
    )
    path = _existing_artifact(canonical, LATEST_ABLATION_LEGACY)
    if not path.is_file():
        return {
            "path": str(path),
            "exists": False,
            "overall_pass": False,
            "stage_completed": False,
            "total_episodes": None,
            "content_pass": False,
        }
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    result = {
        "path": str(path),
        "exists": True,
        "overall_pass": payload.get("overall_pass"),
        "stage_completed": payload.get("stage_completed"),
        "total_episodes": payload.get("total_episodes"),
    }
    result["content_pass"] = (
        result["overall_pass"] is True
        and result["stage_completed"] is True
        and result["total_episodes"] == 3750
    )
    return result


def _legacy_resolution_checks() -> dict[str, bool]:
    checks = {}
    for label, spec in KEY_MODELS.items():
        resolved = resolve_artifact_path(spec["legacy"])
        checks[label] = resolved.is_file()
    resolved_summary = resolve_artifact_path(LATEST_ABLATION_LEGACY)
    checks["latest_ablation_summary"] = resolved_summary.is_file()
    return checks


def _forbidden_nonempty_roots(project_root: Path) -> list[str]:
    return [
        name
        for name in SOURCE_ROOT_NAMES
        if (project_root / name).is_dir() and any((project_root / name).iterdir())
    ]


def _operational_old_path_references(project_root: Path) -> list[str]:
    allowed_files = {
        "registry/ch4_artifact_layout.py",
        "tools/_internal/migrate_ch4_artifacts.py",
        "tools/_internal/audit_ch4_artifact_layout.py",
        "tests/test_ch4_artifact_layout.py",
        "tests/test_ch4_artifact_migration.py",
        "tests/test_ch4_artifact_final_recovery.py",
    }
    old_root_pattern = re.compile(
        r"""(?:"|')(?:
        eval_ch4_rbe|eval_ch4_smoke|log_ch4_rbe|models_ch4_rbe
        )(?:"|'|/|\\)""",
        re.VERBOSE,
    )
    matches = []
    for suffix in ("*.py", "*.bat"):
        for path in project_root.rglob(suffix):
            relative = path.relative_to(project_root).as_posix()
            if relative in allowed_files or relative.startswith("data/"):
                continue
            try:
                lines = path.read_text(encoding="utf-8-sig").splitlines()
            except UnicodeError:
                continue
            for line_number, line in enumerate(lines, 1):
                if old_root_pattern.search(line):
                    matches.append(f"{relative}:{line_number}:{line.strip()}")
    return matches


def run_audit(project_root: Path, *, strict_layout: bool) -> dict[str, Any]:
    project_root = project_root.resolve()
    model_results = _model_checks()
    result_check = _latest_result_check()
    legacy_checks = _legacy_resolution_checks()
    nonempty_roots = _forbidden_nonempty_roots(project_root)
    old_refs = _operational_old_path_references(project_root)
    checks = {
        "legacy_roots_empty_or_absent": not nonempty_roots,
        "all_key_model_sha_match": all(
            value["sha256_match"] for value in model_results.values()
        ),
        "latest_ablation_result_loads_and_matches": result_check["content_pass"],
        "legacy_paths_resolve": all(legacy_checks.values()),
        "operational_scripts_do_not_recreate_old_roots": not old_refs,
    }
    if not strict_layout:
        # Before apply, non-empty legacy roots are expected and reported rather
        # than treated as a preflight failure.
        checks["legacy_roots_empty_or_absent"] = True
    payload = {
        "layout_version": LAYOUT_VERSION,
        "audit_tool_version": AUDIT_TOOL_VERSION,
        "migration_tool_version": MIGRATION_TOOL_VERSION,
        "mode": "audit" if strict_layout else "preflight",
        "checks": checks,
        "key_models": model_results,
        "latest_ablation_result": result_check,
        "legacy_resolution": legacy_checks,
        "nonempty_legacy_roots": nonempty_roots,
        "operational_old_path_references": old_refs,
        "migration_status": migration_status(project_root, get_ch4_data_root()),
        "overall_pass": all(checks.values()),
    }
    output = get_ch4_data_root() / "manifests" / "artifact_layout_audit.json"
    atomic_json(output, payload)
    return payload


def status(project_root: Path) -> dict[str, Any]:
    migration = migration_status(project_root.resolve(), get_ch4_data_root())
    audit_path = get_ch4_data_root() / "manifests" / "artifact_layout_audit.json"
    last_audit = None
    if audit_path.is_file():
        last_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    return {
        "layout_version": LAYOUT_VERSION,
        "audit_tool_version": AUDIT_TOOL_VERSION,
        "migration": migration,
        "last_audit_mode": last_audit.get("mode") if last_audit else None,
        "last_audit_pass": last_audit.get("overall_pass") if last_audit else None,
        "overall_pass": bool(last_audit and last_audit.get("overall_pass")),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("self-test", "preflight", "audit", "status"))
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args(argv)
    try:
        if args.mode == "self-test":
            result = self_test()
        elif args.mode == "preflight":
            result = run_audit(args.project_root, strict_layout=False)
        elif args.mode == "audit":
            result = run_audit(args.project_root, strict_layout=True)
        else:
            result = status(args.project_root)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if result["overall_pass"] else 1
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[CH4ArtifactLayoutAudit] ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
