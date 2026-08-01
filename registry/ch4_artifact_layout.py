"""Canonical Chapter-4 artifact layout and legacy-path compatibility.

This module is the only place where the retired Chapter-4 top-level artifact
roots are interpreted.  Experiment orchestration code should construct new
paths with the helpers below and resolve every input artifact through
``resolve_artifact_path``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping


LAYOUT_VERSION = 1
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CH4_DATA_ROOT = PROJECT_ROOT / "data" / "chapter4_final"

_FAMILY_PATHS = {
    "clean": Path("baselines/clean"),
    "baselines/clean": Path("baselines/clean"),
    "uniform_dr": Path("baselines/uniform_dr"),
    "baselines/uniform_dr": Path("baselines/uniform_dr"),
    "reb": Path("reb"),
    "rbe_legacy": Path("rbe_legacy"),
    "perf_rbe": Path("perf_rbe"),
    "no_boundary": Path("ablations/no_boundary"),
    "ablations/no_boundary": Path("ablations/no_boundary"),
    "all_actors": Path("ablations/all_actors"),
    "ablations/all_actors": Path("ablations/all_actors"),
    "no_nominal": Path("ablations/no_nominal"),
    "ablations/no_nominal": Path("ablations/no_nominal"),
    "ablations": Path("ablations"),
    "stress_tests": Path("stress_tests"),
}
_LEGACY_ROOTS = (
    "log_ch4_rbe",
    "models_ch4_rbe",
    "eval_ch4_rbe",
    "eval_ch4_smoke",
    "archive",
)


def get_ch4_data_root() -> Path:
    override = os.environ.get("CH4_DATA_ROOT")
    return Path(override).expanduser().resolve() if override else DEFAULT_CH4_DATA_ROOT


def _family_relative(family: str) -> Path:
    key = str(family).strip().replace("\\", "/").strip("/").lower()
    try:
        return _FAMILY_PATHS[key]
    except KeyError as exc:
        raise ValueError(f"unknown Chapter-4 artifact family: {family!r}") from exc


def get_family_root(family: str) -> Path:
    return get_ch4_data_root() / _family_relative(family)


def get_run_root(family: str) -> Path:
    return get_family_root(family) / "runs"


def get_training_run_dir(family: str, run_name: str) -> Path:
    return get_run_root(family) / _safe_component(run_name, "run_name")


def get_selected_dir(family: str, selected_id: str) -> Path:
    return get_family_root(family) / "selected" / _safe_component(selected_id, "selected_id")


def get_selection_dir(family: str, experiment_id: str) -> Path:
    return get_family_root(family) / "selections" / _safe_component(experiment_id, "experiment_id")


def get_evaluation_dir(family: str, experiment_id: str) -> Path:
    return get_family_root(family) / "evaluations" / _safe_component(experiment_id, "experiment_id")


def get_smoke_dir(family: str, experiment_id: str) -> Path:
    relative = _family_relative(family)
    if relative.parts[0] == "baselines":
        smoke_family = Path("baselines")
    elif relative.parts[0] == "ablations":
        smoke_family = Path("ablations")
    else:
        smoke_family = Path(relative.parts[0])
    return get_ch4_data_root() / "smoke" / smoke_family / _safe_component(
        experiment_id, "experiment_id"
    )


def get_manifest_dir() -> Path:
    return get_ch4_data_root() / "manifests"


def get_archive_dir(category: str) -> Path:
    normalized = str(category).strip().lower()
    if normalized not in {"interrupted", "superseded", "manual"}:
        raise ValueError(f"unknown Chapter-4 archive category: {category!r}")
    return get_ch4_data_root() / "archive" / normalized


def _safe_component(value: str, label: str) -> str:
    text = str(value).strip()
    if not text or text in {".", ".."} or "/" in text or "\\" in text:
        raise ValueError(f"{label} must be one non-empty path component: {value!r}")
    return text


def _source_root_name(source_root: Any) -> str:
    if source_root is None:
        return ""
    text = str(source_root).replace("\\", "/").rstrip("/")
    return text.rsplit("/", 1)[-1].lower()


def classify_artifact_name(
    name: str,
    source_root: Any = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify one top-level legacy artifact without inspecting its contents."""

    del metadata  # Reserved for future schema-compatible classification hints.
    artifact_name = Path(str(name).replace("\\", "/")).name
    lowered = artifact_name.lower()
    root_name = _source_root_name(source_root)

    if "interrupted" in lowered:
        return _classification(
            artifact_name,
            family="archive",
            stage="archive",
            variant="interrupted",
            relative=Path("archive/interrupted") / artifact_name,
            confidence=1.0,
            reason="artifact name contains interrupted",
        )
    if root_name == "archive":
        category = "superseded" if "superseded" in lowered else "manual"
        return _classification(
            artifact_name,
            family="archive",
            stage="archive",
            variant=category,
            relative=Path("archive") / category / artifact_name,
            confidence=0.9,
            reason=f"legacy archive root classified as {category}",
        )

    family: str | None = None
    variant: str | None = None
    reason = ""
    if any(
        token in lowered
        for token in (
            "stress_test",
            "disturbance_intensity",
            "intensity_sweep",
            "severity_sweep",
            "pressure_test",
        )
    ):
        family, variant, reason = (
            "stress_tests",
            "stress_tests",
            "matched stress-test artifact token",
        )
    elif "ch4_rbe_perf_ablation_no_boundary" in lowered:
        family, variant, reason = (
            "ablations/no_boundary",
            "no_boundary",
            "matched no-boundary Perf-RBE ablation",
        )
    elif "ch4_rbe_perf_ablation_all_actors" in lowered:
        family, variant, reason = (
            "ablations/all_actors",
            "all_actors",
            "matched all-actors Perf-RBE ablation",
        )
    elif "ch4_rbe_perf_ablation_no_nominal" in lowered:
        family, variant, reason = (
            "ablations/no_nominal",
            "no_nominal",
            "matched no-nominal Perf-RBE ablation",
        )
    elif "ch4_rbe_perf_ablation_" in lowered:
        family, variant, reason = (
            "ablations",
            "cross_ablation",
            "matched cross-ablation artifact",
        )
    elif lowered.startswith("ch4_clean_"):
        family, variant, reason = "baselines/clean", "clean", "matched clean baseline prefix"
    elif lowered.startswith("ch4_uniform_dr_"):
        family, variant, reason = (
            "baselines/uniform_dr",
            "uniform_dr",
            "matched Uniform-DR prefix",
        )
    elif any(token in lowered for token in ("found_aware_reb", "reb_only", "frozen_reb", "reb_dataset")):
        family, variant, reason = "reb", "reb", "matched REB artifact token"
    elif "ch4_rbe_boundary_core_perf_" in lowered:
        family, variant, reason = "perf_rbe", "perf_rbe", "matched Perf-RBE prefix"
    elif lowered.startswith("ch4_rbe_boundary_core_"):
        family, variant, reason = "rbe_legacy", "rbe_legacy", "matched legacy RBE prefix"

    if family is None:
        return {
            "family": None,
            "stage": "unclassified",
            "variant": None,
            "canonical_relative_path": None,
            "confidence": 0.0,
            "reason": "no reliable Chapter-4 artifact family rule matched",
        }

    if family == "stress_tests":
        stage = "evaluations"
        relative = Path("stress_tests/evaluations") / artifact_name
        stage_reason = "stress tests are evaluation artifacts"
    elif root_name == "eval_ch4_smoke":
        stage = "smoke"
        relative = get_smoke_dir(family, artifact_name).relative_to(get_ch4_data_root())
        stage_reason = "legacy smoke evaluation root"
    elif "training_audit" in lowered and family.startswith("ablations"):
        stage = "training_audits"
        relative = Path("ablations/training_audits") / artifact_name
        stage_reason = "matched ablation training audit"
    elif "reb_dataset" in lowered:
        stage = "datasets"
        relative = Path("reb/datasets") / artifact_name
        stage_reason = "matched REB dataset"
    elif "_selected_" in lowered or lowered.endswith("_selected"):
        stage = "selected"
        relative = _family_relative(family) / "selected" / artifact_name
        stage_reason = "matched selected-model artifact"
    elif any(token in lowered for token in ("checkpoint_selection", "checkpoint_validation", "selection")):
        stage = "selections"
        relative = _family_relative(family) / "selections" / artifact_name
        stage_reason = "matched selection/validation artifact"
    elif any(token in lowered for token in ("independent_test", "robust_test", "comparison", "validation")):
        stage = "evaluations"
        relative = _family_relative(family) / "evaluations" / artifact_name
        stage_reason = "matched evaluation/comparison artifact"
    elif root_name in {"log_ch4_rbe", "models_ch4_rbe"}:
        stage = "runs"
        relative = _family_relative(family) / "runs" / artifact_name
        stage_reason = "legacy training log/model root"
    elif root_name == "eval_ch4_rbe":
        stage = "evaluations"
        relative = _family_relative(family) / "evaluations" / artifact_name
        stage_reason = "legacy formal evaluation root"
    else:
        return {
            "family": family,
            "stage": "unclassified",
            "variant": variant,
            "canonical_relative_path": None,
            "confidence": 0.4,
            "reason": f"{reason}; source stage could not be determined",
        }

    return _classification(
        artifact_name,
        family=family,
        stage=stage,
        variant=variant,
        relative=relative,
        confidence=1.0 if root_name in _LEGACY_ROOTS else 0.9,
        reason=f"{reason}; {stage_reason}",
    )


def _classification(
    name: str,
    *,
    family: str,
    stage: str,
    variant: str,
    relative: Path,
    confidence: float,
    reason: str,
) -> dict[str, Any]:
    del name
    return {
        "family": family,
        "stage": stage,
        "variant": variant,
        "canonical_relative_path": relative.as_posix(),
        "confidence": float(confidence),
        "reason": reason,
    }


def _normalized_key(value: Any) -> str:
    text = str(value).strip().replace("\\", "/")
    text = re.sub(r"/+", "/", text)
    return text.rstrip("/").lower()


def _relocation_entries() -> list[Mapping[str, Any]]:
    path = get_manifest_dir() / "relocation_manifest.json"
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(payload, list):
        return [entry for entry in payload if isinstance(entry, Mapping)]
    if isinstance(payload, Mapping):
        entries = payload.get("relocations", payload.get("entries", []))
        return [entry for entry in entries if isinstance(entry, Mapping)]
    return []


def _canonical_from_relocation(path_text: Any) -> Path | None:
    requested = _normalized_key(path_text)
    candidates = [requested]
    try:
        candidate_path = Path(path_text)
        if not candidate_path.is_absolute():
            candidates.append(_normalized_key(PROJECT_ROOT / candidate_path))
    except (OSError, TypeError, ValueError):
        pass
    for entry in _relocation_entries():
        canonical = entry.get("canonical_relative_path") or entry.get("canonical_path")
        if not canonical:
            continue
        legacy_values = (
            entry.get("legacy_relative_path"),
            entry.get("legacy_absolute_path_at_migration"),
            entry.get("legacy_path"),
        )
        for legacy in legacy_values:
            if not legacy:
                continue
            legacy_key = _normalized_key(legacy)
            for candidate in candidates:
                if candidate == legacy_key or candidate.startswith(legacy_key + "/"):
                    suffix = candidate[len(legacy_key) :].lstrip("/")
                    resolved = get_ch4_data_root() / Path(str(canonical).replace("\\", "/"))
                    if suffix:
                        resolved /= Path(suffix)
                    if resolved.exists():
                        return resolved.resolve()
    return None


def _canonical_relative_request(path_text: Any) -> str | None:
    """Return a normalized data-root-relative canonical request, if applicable."""

    text_path = Path(path_text)
    data_root = get_ch4_data_root().resolve()
    project_root = PROJECT_ROOT.resolve()
    candidates: list[Path] = []
    if text_path.is_absolute():
        candidates.append(text_path)
    else:
        candidates.extend((project_root / text_path, data_root / text_path))
    for candidate in candidates:
        try:
            return candidate.resolve(strict=False).relative_to(data_root).as_posix()
        except (OSError, ValueError):
            continue
    return None


def _legacy_from_migration_plan(path_text: Any) -> Path | None:
    """Resolve a canonical request to its still-present pre-apply source tree."""

    requested = _canonical_relative_request(path_text)
    if not requested:
        return None
    plan_path = get_manifest_dir() / "artifact_migration_plan.json"
    if not plan_path.is_file():
        return None
    try:
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    items = payload.get("items", []) if isinstance(payload, Mapping) else []
    requested_key = _normalized_key(requested)
    for entry in items:
        if not isinstance(entry, Mapping):
            continue
        canonical = entry.get("canonical_relative_to_data_root")
        source_path = entry.get("source_path")
        if not canonical or not source_path:
            continue
        canonical_text = str(canonical).replace("\\", "/").strip("/")
        canonical_key = _normalized_key(canonical_text)
        if requested_key != canonical_key and not requested_key.startswith(canonical_key + "/"):
            continue
        suffix = requested[len(canonical_text) :].lstrip("/")
        source = (PROJECT_ROOT / Path(str(source_path))).resolve(strict=False)
        try:
            source.relative_to(PROJECT_ROOT.resolve())
        except ValueError:
            continue
        candidate = source / Path(suffix) if suffix else source
        if candidate.exists():
            return candidate.resolve()
    return None


def _legacy_parts(path_text: Any) -> tuple[str, str, str] | None:
    text = str(path_text).replace("\\", "/")
    parts = [part for part in text.split("/") if part not in {"", "."}]
    lowered = [part.lower() for part in parts]
    for legacy_root in _LEGACY_ROOTS:
        if legacy_root in lowered:
            index = lowered.index(legacy_root)
            if index + 1 >= len(parts):
                return None
            artifact_name = parts[index + 1]
            suffix = "/".join(parts[index + 2 :])
            return legacy_root, artifact_name, suffix
    return None


def resolve_artifact_path(path: os.PathLike[str] | str) -> Path:
    """Resolve a canonical or historical artifact path to an existing path."""

    original = Path(path)
    if original.is_absolute() and original.exists():
        return original.resolve()

    legacy = _legacy_parts(path)
    if not original.is_absolute():
        project_candidate = PROJECT_ROOT / original
        if project_candidate.exists():
            return project_candidate.resolve()
        data_candidate = get_ch4_data_root() / original
        if data_candidate.exists():
            return data_candidate.resolve()

    relocated = _canonical_from_relocation(path)
    if relocated is not None:
        return relocated

    planned_legacy = _legacy_from_migration_plan(path)
    if planned_legacy is not None:
        return planned_legacy

    if legacy is not None:
        legacy_root, artifact_name, suffix = legacy
        classified = classify_artifact_name(artifact_name, legacy_root)
        relative = classified.get("canonical_relative_path")
        if relative:
            candidate = get_ch4_data_root() / Path(relative)
            if suffix:
                candidate /= Path(suffix)
            if candidate.exists():
                return candidate.resolve()
        # Rebase a missing historical absolute legacy path (for example an
        # old RBE7.15 workspace) onto the current project root.  The legacy
        # root, artifact name, and suffix have already been parsed strictly,
        # so this never performs a broad or ambiguous file search.
        current_legacy_candidate = PROJECT_ROOT / legacy_root / artifact_name
        if suffix:
            current_legacy_candidate /= Path(suffix)
        if current_legacy_candidate.exists():
            return current_legacy_candidate.resolve()

    # Preserve normal pathlib behavior for diagnostics while never silently
    # substituting an unrelated artifact.
    if original.is_absolute():
        return original
    if legacy is None:
        return PROJECT_ROOT / original
    return original


def create_training_run_directories(
    log_dir: os.PathLike[str] | str,
    model_dir: os.PathLike[str] | str,
    *,
    write_once: bool,
) -> tuple[Path, ...]:
    """Create the one or two physical run directories without double-creating one path."""

    run_directories = {
        Path(log_dir).resolve(strict=False),
        Path(model_dir).resolve(strict=False),
    }
    ordered = tuple(sorted(run_directories, key=lambda value: os.path.normcase(str(value))))
    if write_once:
        existing = [path for path in ordered if path.exists()]
        if existing:
            raise RuntimeError(
                "Locked RBE run directories already exist; refusing overwrite: "
                + ", ".join(str(path) for path in existing)
            )
    for path in ordered:
        path.mkdir(parents=True, exist_ok=not write_once)
    return ordered


def file_sha256(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_tree_sha256(path: os.PathLike[str] | str) -> str:
    root = Path(path)
    digest = hashlib.sha256()
    for file_path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = file_path.relative_to(root).as_posix().encode("utf-8")
        file_hash = file_sha256(file_path).encode("ascii")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(file_hash)
    return digest.hexdigest()


def artifact_relative_path(path: os.PathLike[str] | str) -> dict[str, Any]:
    artifact = Path(path).resolve()
    data_root = get_ch4_data_root().resolve()
    project_root = PROJECT_ROOT.resolve()
    try:
        relative_data = artifact.relative_to(data_root).as_posix()
    except ValueError:
        relative_data = None
    try:
        relative_project = artifact.relative_to(project_root).as_posix()
    except ValueError:
        relative_project = None
    if artifact.is_file():
        artifact_hash = file_sha256(artifact)
    elif artifact.is_dir():
        artifact_hash = directory_tree_sha256(artifact)
    else:
        artifact_hash = None
    return {
        "artifact_path_relative_to_data_root": relative_data,
        "artifact_path_relative_to_project_root": relative_project,
        "artifact_sha256": artifact_hash,
    }


def source_hash_matches_layout_attestation(
    relative_path: str, expected_sha256: str, actual_sha256: str
) -> bool:
    """Allow an old source lock only when exact old/new hashes are attested."""

    path = get_manifest_dir() / "layout_refactor_attestation.json"
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    relative = str(relative_path).replace("\\", "/")
    return bool(
        payload.get("algorithm_semantics_changed") is False
        and payload.get("model_artifacts_changed") is False
        and payload.get("historical_results_rewritten") is False
        and relative in payload.get("modified_source_files", [])
        and payload.get("old_source_hashes", {}).get(relative) == expected_sha256
        and payload.get("new_source_hashes", {}).get(relative) == actual_sha256
    )


__all__ = [
    "LAYOUT_VERSION",
    "PROJECT_ROOT",
    "DEFAULT_CH4_DATA_ROOT",
    "artifact_relative_path",
    "classify_artifact_name",
    "create_training_run_directories",
    "directory_tree_sha256",
    "file_sha256",
    "get_archive_dir",
    "get_ch4_data_root",
    "get_evaluation_dir",
    "get_family_root",
    "get_manifest_dir",
    "get_run_root",
    "get_selected_dir",
    "get_selection_dir",
    "get_smoke_dir",
    "get_training_run_dir",
    "resolve_artifact_path",
    "source_hash_matches_layout_attestation",
]
