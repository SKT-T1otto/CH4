#!/usr/bin/env python3
"""Plan, verify, and explicitly apply the Chapter-4 artifact relocation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from registry.ch4_artifact_layout import (  # noqa: E402
    LAYOUT_VERSION,
    classify_artifact_name,
    file_sha256,
    get_ch4_data_root,
    resolve_artifact_path,
)


MIGRATION_TOOL_VERSION = "20260731-v3"
SOURCE_ROOT_NAMES = (
    "eval_ch4_rbe",
    "eval_ch4_smoke",
    "log_ch4_rbe",
    "models_ch4_rbe",
    "archive",
)
PLAN_BASENAME = "artifact_migration_plan"
PLAN_FIELDS = (
    "artifact_id",
    "source_root",
    "source_path",
    "artifact_name",
    "family",
    "stage",
    "variant",
    "canonical_path",
    "canonical_relative_to_data_root",
    "source_file_count",
    "source_total_bytes",
    "source_tree_sha256",
    "classification_confidence",
    "classification_reason",
    "collision_status",
    "planned_action",
)
PLAN_HASH_FIELDS = (
    "source_path",
    "source_tree_sha256",
    "family",
    "stage",
    "variant",
    "canonical_relative_to_data_root",
    "planned_action",
    "collision_status",
)
_LAYOUT_DIRECTORIES = (
    "manifests",
    "baselines/clean/runs",
    "baselines/clean/selected",
    "baselines/clean/selections",
    "baselines/clean/evaluations",
    "baselines/uniform_dr/runs",
    "baselines/uniform_dr/selected",
    "baselines/uniform_dr/selections",
    "baselines/uniform_dr/evaluations",
    "reb/datasets",
    "reb/runs",
    "reb/selected",
    "reb/selections",
    "reb/evaluations",
    "rbe_legacy/runs",
    "rbe_legacy/selected",
    "rbe_legacy/selections",
    "rbe_legacy/evaluations",
    "perf_rbe/runs",
    "perf_rbe/selected",
    "perf_rbe/selections",
    "perf_rbe/evaluations",
    "ablations/no_boundary/runs",
    "ablations/no_boundary/selected",
    "ablations/all_actors/runs",
    "ablations/all_actors/selected",
    "ablations/no_nominal/runs",
    "ablations/no_nominal/selected",
    "ablations/training_audits",
    "ablations/selections",
    "ablations/evaluations",
    "stress_tests/evaluations",
    "stress_tests/manifests",
    "smoke/baselines",
    "smoke/reb",
    "smoke/rbe_legacy",
    "smoke/perf_rbe",
    "smoke/ablations",
    "archive/interrupted",
    "archive/superseded",
    "archive/manual",
)


class MigrationError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(item for item in path.rglob("*") if item.is_file())


def _relative_file(path: Path, root: Path) -> Path:
    return Path(path.name) if root.is_file() else path.relative_to(root)


def tree_stats(path: Path) -> tuple[int, int, str]:
    digest = hashlib.sha256()
    count = 0
    total = 0
    for item in _files(path):
        relative = _relative_file(item, path).as_posix().encode("utf-8")
        item_hash = file_sha256(item).encode("ascii")
        size = item.stat().st_size
        count += 1
        total += size
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(size.to_bytes(8, "big"))
        digest.update(item_hash)
    return count, total, digest.hexdigest()


def ensure_layout_tree(data_root: Path) -> None:
    for relative in _LAYOUT_DIRECTORIES:
        (data_root / relative).mkdir(parents=True, exist_ok=True)
    layout_path = data_root / "manifests" / "layout_version.json"
    if not layout_path.exists():
        atomic_json(
            layout_path,
            {
                "layout_version": LAYOUT_VERSION,
                "layout_name": "chapter4_final",
                "migration_tool_version": MIGRATION_TOOL_VERSION,
            },
        )
    relocation_path = data_root / "manifests" / "relocation_manifest.json"
    if not relocation_path.exists():
        atomic_json(
            relocation_path,
            {
                "layout_version": LAYOUT_VERSION,
                "migration_tool_version": MIGRATION_TOOL_VERSION,
                "relocations": [],
            },
        )


def _item_id(source_relative: str) -> str:
    return "ch4-" + hashlib.sha256(source_relative.encode("utf-8")).hexdigest()[:16]


def scan_items(project_root: Path, data_root: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for root_name in SOURCE_ROOT_NAMES:
        source_root = project_root / root_name
        if not source_root.is_dir():
            continue
        for source in sorted(source_root.iterdir(), key=lambda value: value.name.lower()):
            classification = classify_artifact_name(source.name, root_name)
            count, total, tree_hash = tree_stats(source)
            canonical_relative = classification["canonical_relative_path"]
            canonical = data_root / canonical_relative if canonical_relative else None
            source_relative = source.relative_to(project_root).as_posix()
            items.append(
                {
                    "artifact_id": _item_id(source_relative),
                    "source_root": root_name,
                    "source_path": source_relative,
                    "artifact_name": source.name,
                    "family": classification["family"],
                    "stage": classification["stage"],
                    "variant": classification["variant"],
                    "canonical_path": (
                        canonical.relative_to(project_root).as_posix()
                        if canonical is not None and _is_relative_to(canonical, project_root)
                        else str(canonical) if canonical is not None else None
                    ),
                    "canonical_relative_to_data_root": canonical_relative,
                    "source_file_count": count,
                    "source_total_bytes": total,
                    "source_tree_sha256": tree_hash,
                    "classification_confidence": classification["confidence"],
                    "classification_reason": classification["reason"],
                    "collision_status": "unchecked",
                    "planned_action": "manual_review" if canonical is None else "migrate",
                }
            )
    _annotate_collisions(items, project_root, data_root)
    return items


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _source_for(item: Mapping[str, Any], project_root: Path) -> Path:
    return project_root / str(item["source_path"])


def _target_for(item: Mapping[str, Any], data_root: Path) -> Path:
    relative = item.get("canonical_relative_to_data_root")
    if not isinstance(relative, str) or not relative.strip():
        raise MigrationError(f"unclassified item has no target: {item.get('source_path')}")
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise MigrationError(f"unsafe canonical target: {relative!r}")
    root = data_root.resolve()
    target = (root / relative_path).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise MigrationError(f"canonical target escapes CH4_DATA_ROOT: {relative!r}") from exc
    return target


def compute_plan_sha256(items: Iterable[Mapping[str, Any]]) -> str:
    rows = [
        {field: item.get(field) for field in PLAN_HASH_FIELDS}
        for item in sorted(items, key=lambda value: str(value.get("source_path", "")))
    ]
    encoded = json.dumps(
        rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _annotate_collisions(
    items: list[dict[str, Any]], project_root: Path, data_root: Path
) -> None:
    file_targets: dict[str, list[tuple[dict[str, Any], Path, str]]] = defaultdict(list)
    target_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        if item["planned_action"] == "manual_review":
            item["collision_status"] = "not_applicable"
            continue
        source = _source_for(item, project_root)
        target = _target_for(item, data_root)
        target_groups[str(target).lower()].append(item)
        for source_file in _files(source):
            relative = _relative_file(source_file, source)
            file_targets[str((target / relative).resolve()).lower()].append(
                (item, source_file, file_sha256(source_file))
            )

    conflicts: set[str] = set()
    reused: set[str] = set()
    for target_text, entries in file_targets.items():
        hashes = {entry[2] for entry in entries}
        target_path = Path(target_text)
        if target_path.is_file():
            target_hash = file_sha256(target_path)
            if hashes == {target_hash}:
                reused.update(entry[0]["artifact_id"] for entry in entries)
            else:
                conflicts.update(entry[0]["artifact_id"] for entry in entries)
        if len(hashes) > 1:
            conflicts.update(entry[0]["artifact_id"] for entry in entries)

    for group in target_groups.values():
        is_merge = len(group) > 1
        for item in group:
            artifact_id = item["artifact_id"]
            if artifact_id in conflicts:
                item["collision_status"] = "conflict"
                item["planned_action"] = "reject"
            elif artifact_id in reused:
                item["collision_status"] = "reused_identical"
                item["planned_action"] = "reused"
            elif is_merge:
                item["collision_status"] = "merge_compatible"
                item["planned_action"] = "merge"
            else:
                item["collision_status"] = "none"


def _summary(items: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(items)
    return {
        "planned_items": len(rows),
        "classified_items": sum(row.get("stage") != "unclassified" for row in rows),
        "unclassified_items": sum(row.get("stage") == "unclassified" for row in rows),
        "conflict_items": sum(row.get("collision_status") == "conflict" for row in rows),
        "reused_items": sum(row.get("planned_action") == "reused" for row in rows),
        "merge_items": sum(row.get("planned_action") == "merge" for row in rows),
    }


def build_plan(project_root: Path = PROJECT_ROOT, data_root: Path | None = None) -> dict[str, Any]:
    project_root = project_root.resolve()
    data_root = (data_root or get_ch4_data_root()).resolve()
    ensure_layout_tree(data_root)
    items = scan_items(project_root, data_root)
    summary = _summary(items)
    plan = {
        "layout_version": LAYOUT_VERSION,
        "migration_tool_version": MIGRATION_TOOL_VERSION,
        "generated_at_utc": utc_now(),
        "project_root_at_plan_time": str(project_root),
        "data_root_at_plan_time": str(data_root),
        "source_roots": list(SOURCE_ROOT_NAMES),
        "items": items,
        "plan_sha256": compute_plan_sha256(items),
        "unclassified_items": [
            row["source_path"] for row in items if row["stage"] == "unclassified"
        ],
        "summary": summary,
        "apply_executed": False,
    }
    manifest_dir = data_root / "manifests"
    atomic_json(manifest_dir / f"{PLAN_BASENAME}.json", plan)
    _write_plan_csv(manifest_dir / f"{PLAN_BASENAME}.csv", items)
    _write_plan_markdown(manifest_dir / f"{PLAN_BASENAME}.md", plan)
    _write_catalog(manifest_dir, items, status="planned")
    return plan


def _write_plan_csv(path: Path, items: Iterable[Mapping[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PLAN_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(items)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_plan_markdown(path: Path, plan: Mapping[str, Any]) -> None:
    summary = plan["summary"]
    lines = [
        "# Chapter-4 Artifact Migration Plan",
        "",
        f"- layout_version: `{plan['layout_version']}`",
        f"- migration_tool_version: `{plan['migration_tool_version']}`",
        f"- planned_items: `{summary['planned_items']}`",
        f"- classified_items: `{summary['classified_items']}`",
        f"- unclassified_items: `{summary['unclassified_items']}`",
        f"- conflict_items: `{summary['conflict_items']}`",
        "",
        "| source | family | stage | canonical | collision | action |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in plan["items"]:
        lines.append(
            f"| `{item['source_path']}` | `{item['family']}` | `{item['stage']}` | "
            f"`{item['canonical_path']}` | `{item['collision_status']}` | "
            f"`{item['planned_action']}` |"
        )
    _atomic_text(path, "\n".join(lines) + "\n")


def _write_catalog(
    manifest_dir: Path, items: Iterable[Mapping[str, Any]], *, status: str
) -> None:
    rows = []
    for item in items:
        row = dict(item)
        row["catalog_status"] = status
        rows.append(row)
    atomic_json(
        manifest_dir / "artifact_catalog.json",
        {
            "layout_version": LAYOUT_VERSION,
            "migration_tool_version": MIGRATION_TOOL_VERSION,
            "catalog_status": status,
            "items": rows,
        },
    )
    fields = list(PLAN_FIELDS) + ["catalog_status"]
    temporary = manifest_dir / "artifact_catalog.csv.tmp"
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, manifest_dir / "artifact_catalog.csv")


def load_plan(data_root: Path | None = None) -> dict[str, Any]:
    data_root = (data_root or get_ch4_data_root()).resolve()
    path = data_root / "manifests" / f"{PLAN_BASENAME}.json"
    if not path.is_file():
        raise MigrationError(f"migration plan is missing; run plan first: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise MigrationError(f"invalid migration plan schema: {path}")
    return payload


def _validate_csv_header(path: Path) -> None:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
    if not header or not any(str(value).strip() for value in header):
        raise MigrationError(f"CSV header is unreadable: {path}")


def _validate_pt(path: Path) -> None:
    try:
        import torch

        try:
            torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            torch.load(path, map_location="cpu")
    except Exception as exc:
        raise MigrationError(f"PyTorch artifact is unreadable: {path}: {exc}") from exc


def _iter_path_sha_pairs(value: Any) -> Iterable[tuple[str, str]]:
    if isinstance(value, Mapping):
        for key, path_value in value.items():
            if not isinstance(path_value, str):
                continue
            if not (key.endswith("_path") or key.endswith("_model")):
                continue
            prefixes = [key[: -len("_path")], key]
            for prefix in prefixes:
                for sha_key in (f"{prefix}_sha256", f"{prefix}_sha", "selected_sha256", "source_sha256"):
                    sha_value = value.get(sha_key)
                    if isinstance(sha_value, str) and len(sha_value) == 64:
                        yield path_value, sha_value.lower()
                        break
                else:
                    continue
                break
        for nested in value.values():
            yield from _iter_path_sha_pairs(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _iter_path_sha_pairs(nested)


def _resolve_manifest_reference(
    reference: str, manifest_path: Path, project_root: Path
) -> Path | None:
    resolved = resolve_artifact_path(reference)
    if resolved.is_file():
        return resolved
    candidate = Path(reference)
    candidates = []
    if candidate.is_absolute():
        candidates.append(candidate)
    else:
        candidates.extend((project_root / candidate, manifest_path.parent / candidate))
    for path in candidates:
        if path.is_file():
            return path
    reference_name = candidate.name
    sibling = manifest_path.parent / reference_name
    return sibling if sibling.is_file() else None


def verify_plan(
    project_root: Path = PROJECT_ROOT,
    data_root: Path | None = None,
    *,
    validate_pt: bool = True,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    data_root = (data_root or get_ch4_data_root()).resolve()
    plan = load_plan(data_root)
    planned_items = list(plan["items"])
    current_items = scan_items(project_root, data_root)
    errors: list[str] = []
    checks: dict[str, Any] = {}

    declared_plan_hash = plan.get("plan_sha256")
    computed_plan_hash = compute_plan_sha256(planned_items)
    checks["plan_sha256_matches"] = declared_plan_hash == computed_plan_hash
    if not checks["plan_sha256_matches"]:
        errors.append(
            f"plan_sha256 mismatch: declared={declared_plan_hash!r} "
            f"computed={computed_plan_hash}"
        )

    target_errors: list[str] = []
    for row in planned_items:
        try:
            _target_for(row, data_root)
        except MigrationError as exc:
            target_errors.append(f"{row.get('source_path')}: {exc}")
    checks["canonical_targets_within_data_root"] = not target_errors
    errors.extend(target_errors)

    planned_sources = {str(row["source_path"]) for row in planned_items}
    current_sources = {str(row["source_path"]) for row in current_items}
    missing_from_plan = sorted(current_sources - planned_sources)
    vanished_since_plan = sorted(planned_sources - current_sources)
    checks["no_source_omissions"] = not missing_from_plan
    checks["all_planned_sources_still_exist"] = not vanished_since_plan
    if missing_from_plan:
        errors.append(f"source items missing from plan: {missing_from_plan}")
    if vanished_since_plan:
        errors.append(f"planned source items disappeared: {vanished_since_plan}")

    current_by_source = {row["source_path"]: row for row in current_items}
    changed: list[str] = []
    for row in planned_items:
        current = current_by_source.get(row["source_path"])
        if current:
            for field in PLAN_HASH_FIELDS:
                if current.get(field) != row.get(field):
                    changed.append(
                        f"{row['source_path']}: {field}: "
                        f"planned={row.get(field)!r} live={current.get(field)!r}"
                    )
    checks["plan_fields_match_live_scan"] = not changed
    checks["source_trees_unchanged"] = not any(
        ": source_tree_sha256:" in message for message in changed
    )
    if changed:
        errors.append("plan fields differ from live scan: " + "; ".join(changed))

    collisions = [
        row["source_path"]
        for row in current_items
        if row["collision_status"] == "conflict"
    ]
    checks["no_target_file_conflicts"] = not collisions
    if collisions:
        errors.append(f"conflicting target files: {collisions}")

    unclassified = [
        row["source_path"] for row in current_items if row["stage"] == "unclassified"
    ]
    checks["unclassified_count_explicit"] = (
        sorted(unclassified) == sorted(plan.get("unclassified_items", []))
    )
    checks["unclassified_items"] = len(unclassified)
    if not checks["unclassified_count_explicit"]:
        errors.append("unclassified item list differs from the plan")

    format_errors: list[str] = []
    manifest_sha_errors: list[str] = []
    pt_count = json_count = csv_count = 0
    for row in current_items:
        source = _source_for(row, project_root)
        for path in _files(source):
            try:
                suffix = path.suffix.lower()
                if suffix == ".json":
                    json_count += 1
                    payload = json.loads(path.read_text(encoding="utf-8-sig"))
                    for reference, expected_hash in _iter_path_sha_pairs(payload):
                        referenced = _resolve_manifest_reference(reference, path, project_root)
                        if referenced is None:
                            manifest_sha_errors.append(
                                f"{path}: referenced artifact missing: {reference}"
                            )
                        elif file_sha256(referenced) != expected_hash:
                            manifest_sha_errors.append(
                                f"{path}: SHA mismatch: {reference}"
                            )
                elif suffix == ".csv":
                    csv_count += 1
                    _validate_csv_header(path)
                elif suffix == ".pt" and validate_pt:
                    pt_count += 1
                    _validate_pt(path)
            except (OSError, UnicodeError, json.JSONDecodeError, MigrationError) as exc:
                format_errors.append(str(exc))
    checks["json_files_parsed"] = json_count
    checks["csv_headers_read"] = csv_count
    checks["pt_files_loaded"] = pt_count
    checks["artifact_formats_readable"] = not format_errors
    checks["manifest_model_sha_matches"] = not manifest_sha_errors
    errors.extend(format_errors)
    errors.extend(manifest_sha_errors)

    audit = {
        "layout_version": LAYOUT_VERSION,
        "migration_tool_version": MIGRATION_TOOL_VERSION,
        "verified_at_utc": utc_now(),
        "verification_scope": "plan_only_no_migration",
        "checks": checks,
        "errors": errors,
        "unclassified_items": unclassified,
        "conflict_items": collisions,
        "overall_pass": not errors,
        "apply_executed": False,
    }
    atomic_json(data_root / "manifests" / "migration_audit.json", audit)
    if errors:
        raise MigrationError("migration plan verification failed:\n- " + "\n- ".join(errors))
    return audit


def _expected_group_files(
    group: Iterable[Mapping[str, Any]], project_root: Path
) -> dict[str, dict[str, Any]]:
    expected: dict[str, dict[str, Any]] = {}
    for item in group:
        source = _source_for(item, project_root)
        if not source.exists():
            raise MigrationError(f"planned source is missing: {source}")
        for source_file in _files(source):
            relative = _relative_file(source_file, source).as_posix()
            metadata = {
                "size": source_file.stat().st_size,
                "sha256": file_sha256(source_file),
            }
            previous = expected.get(relative)
            if previous is not None and previous != metadata:
                raise MigrationError(
                    f"different source files map to one target file: {relative}"
                )
            expected[relative] = metadata
    return expected


def _expected_tree_sha256(expected: Mapping[str, Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for relative, metadata in sorted(expected.items()):
        encoded_relative = relative.encode("utf-8")
        digest.update(len(encoded_relative).to_bytes(4, "big"))
        digest.update(encoded_relative)
        digest.update(int(metadata["size"]).to_bytes(8, "big"))
        digest.update(str(metadata["sha256"]).encode("ascii"))
    return digest.hexdigest()


def _copy_group_to_staging(
    group: list[Mapping[str, Any]],
    project_root: Path,
    staging: Path,
    expected: Mapping[str, Mapping[str, Any]],
) -> None:
    staging.mkdir(parents=True, exist_ok=True)
    for item in group:
        source = _source_for(item, project_root)
        for source_file in _files(source):
            relative = _relative_file(source_file, source)
            destination = staging / relative
            metadata = expected[relative.as_posix()]
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if (
                    destination.stat().st_size != metadata["size"]
                    or file_sha256(destination) != metadata["sha256"]
                ):
                    raise MigrationError(f"staging collision: {destination}")
            else:
                shutil.copy2(source_file, destination)
    _verify_target_files(staging, expected)


def _verify_target_files(
    root: Path, expected: Mapping[str, Mapping[str, Any]]
) -> None:
    if not root.exists():
        raise MigrationError(f"target does not exist: {root}")
    actual_files = {
        path.relative_to(root).as_posix(): path
        for path in _files(root)
    }
    if set(actual_files) != set(expected):
        raise MigrationError(
            f"target file set mismatch: expected={len(expected)} actual={len(actual_files)}"
        )
    for relative, metadata in expected.items():
        if actual_files[relative].stat().st_size != int(metadata["size"]):
            raise MigrationError(f"target size mismatch: {actual_files[relative]}")
        actual_hash = file_sha256(actual_files[relative])
        if actual_hash != metadata["sha256"]:
            raise MigrationError(f"target SHA mismatch: {actual_files[relative]}")


def _delete_verified_source(source: Path) -> None:
    if source.is_file():
        source.unlink()
        return
    for path in sorted(source.rglob("*"), key=lambda value: len(value.parts), reverse=True):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            path.rmdir()
    source.rmdir()


def _load_transaction_state(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise MigrationError(f"invalid migration transaction state: {path}")
    return payload


def _call_failure_injector(
    failure_injector: Callable[[str, Mapping[str, Any]], None] | None,
    phase: str,
    context: Mapping[str, Any],
) -> None:
    if failure_injector is not None:
        failure_injector(phase, context)


def _initialize_transaction_state(
    plan: Mapping[str, Any],
    items: list[Mapping[str, Any]],
    grouped: Mapping[str, list[Mapping[str, Any]]],
    project_root: Path,
) -> dict[str, Any]:
    plan_hash = str(plan["plan_sha256"])
    transaction_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + plan_hash[:12]
    )
    targets: dict[str, Any] = {}
    for relative, group in grouped.items():
        expected = _expected_group_files(group, project_root)
        targets[relative] = {
            "source_paths": [str(item["source_path"]) for item in group],
            "expected_files": expected,
            "expected_tree_sha256": _expected_tree_sha256(expected),
            "staging_verified": False,
            "status": "pending",
            "migration_action": None,
        }
    return {
        "schema_version": 1,
        "transaction_id": transaction_id,
        "plan_sha256": plan_hash,
        "started_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "phase": "prepared",
        "targets": targets,
        "sources": {
            str(item["source_path"]): {
                "source_tree_sha256": item["source_tree_sha256"],
                "status": "pending",
                "deletion_path": None,
                "deleted": False,
            }
            for item in items
        },
        "manifests": {
            "relocation_manifest": False,
            "reverse_migration_plan": False,
            "artifact_catalog": False,
            "migration_apply_summary": False,
        },
        "complete": False,
    }


def _save_transaction_state(path: Path, state: dict[str, Any]) -> None:
    state["updated_at_utc"] = utc_now()
    atomic_json(path, state)


def _required_migration_manifests_written(state: Mapping[str, Any]) -> bool:
    manifests = state.get("manifests")
    return bool(
        isinstance(manifests, Mapping)
        and manifests
        and all(bool(value) for value in manifests.values())
    )


def _validate_resumed_transaction(
    state: dict[str, Any],
    plan: Mapping[str, Any],
    items: list[Mapping[str, Any]],
    project_root: Path,
    data_root: Path,
) -> bool:
    # Validate a resumable transaction and recover one narrow delete window.
    # Recovery is allowed only when all canonical targets are committed and
    # verified, all required migration manifests are written, and both the
    # legacy source and detached deletion tree are absent.

    computed = compute_plan_sha256(items)
    if plan.get("plan_sha256") != computed or state.get("plan_sha256") != computed:
        raise MigrationError("migration plan changed after transaction preparation")
    targets = state.get("targets")
    sources = state.get("sources")
    if not isinstance(targets, Mapping) or not isinstance(sources, Mapping):
        raise MigrationError("migration transaction state is incomplete")

    recovered_deleted_state = False
    manifests_written = _required_migration_manifests_written(state)

    for item in items:
        target = _target_for(item, data_root)
        source_relative = str(item["source_path"])
        target_relative = str(item["canonical_relative_to_data_root"]).replace("\\", "/")
        target_record = targets.get(target_relative)
        source_record = sources.get(source_relative)
        if not isinstance(target_record, Mapping):
            raise MigrationError(f"target missing from transaction state: {target_relative}")
        if not isinstance(source_record, Mapping):
            raise MigrationError(f"source missing from transaction state: {source_relative}")

        if target_record.get("status") in {"committed", "reused"}:
            _verify_target_files(target, target_record["expected_files"])

        if source_record.get("deleted"):
            continue

        source = _source_for(item, project_root)
        deletion_path = source_record.get("deletion_path")
        detached = Path(str(deletion_path)) if deletion_path else None

        if source.exists():
            if tree_stats(source)[2] != item["source_tree_sha256"]:
                raise MigrationError(f"source changed during migration: {source_relative}")
            continue

        if detached is not None and detached.exists():
            if tree_stats(detached)[2] != item["source_tree_sha256"]:
                raise MigrationError(f"detached source changed during migration: {source_relative}")
            continue

        recoverable_delete_window = (
            source_record.get("status") in {"deletion_intent", "source_detached"}
            and manifests_written
            and target_record.get("status") in {"committed", "reused"}
        )
        if not recoverable_delete_window:
            raise MigrationError(f"source changed during migration: {source_relative}")

        _verify_target_files(target, target_record["expected_files"])
        source_record["deleted"] = True
        source_record["status"] = "deleted_recovered_after_interruption"
        recovered_deleted_state = True

    return recovered_deleted_state

def apply_plan(
    project_root: Path = PROJECT_ROOT,
    data_root: Path | None = None,
    *,
    validate_pt: bool = True,
    failure_injector: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    data_root = (data_root or get_ch4_data_root()).resolve()
    plan = load_plan(data_root)
    items = list(plan["items"])
    manifest_dir = data_root / "manifests"
    state_path = manifest_dir / "migration_transaction_state.json"
    state = _load_transaction_state(state_path)
    if state is not None and state.get("complete"):
        _validate_resumed_transaction(state, plan, items, project_root, data_root)
        summary_path = manifest_dir / "migration_apply_summary.json"
        if not summary_path.is_file():
            raise MigrationError("completed transaction is missing migration_apply_summary.json")
        result = json.loads(summary_path.read_text(encoding="utf-8"))
        if not isinstance(result, dict) or not result.get("overall_pass"):
            raise MigrationError("completed transaction summary is invalid")
        return result
    if state is None:
        verify_plan(project_root, data_root, validate_pt=validate_pt)
    else:
        _validate_resumed_transaction(state, plan, items, project_root, data_root)
        for record in state["targets"].values():
            if record.get("status") == "committed":
                record["status"] = "reused"
                record["migration_action"] = "reused"
        _save_transaction_state(state_path, state)

    unclassified = [row for row in items if row["stage"] == "unclassified"]
    conflicts = [row for row in items if row["collision_status"] == "conflict"]
    if unclassified:
        raise MigrationError(
            "apply refused because unclassified items require manual handling: "
            + ", ".join(row["source_path"] for row in unclassified)
        )
    if conflicts:
        raise MigrationError(
            "apply refused because target conflicts exist: "
            + ", ".join(row["source_path"] for row in conflicts)
        )

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in items:
        relative = str(item["canonical_relative_to_data_root"]).replace("\\", "/")
        _target_for(item, data_root)
        grouped[relative].append(item)

    if state is None:
        state = _initialize_transaction_state(plan, items, grouped, project_root)
        _save_transaction_state(state_path, state)
    staging_root = data_root / ".migration_staging" / str(state["transaction_id"])

    # Stage every target and verify the complete merged file tree before any
    # canonical target is committed or any legacy source is removed.
    for relative, group in grouped.items():
        record = state["targets"][relative]
        target = _target_for(group[0], data_root)
        if record["status"] in {"committed", "reused"}:
            _verify_target_files(target, record["expected_files"])
            continue
        staging = staging_root / Path(relative)
        _copy_group_to_staging(
            group, project_root, staging, record["expected_files"]
        )
        record["staging_verified"] = True
        state["phase"] = "staging"
        _save_transaction_state(state_path, state)

    if not all(record["staging_verified"] for record in state["targets"].values()):
        raise MigrationError("not all staging targets were verified")
    state["phase"] = "staged"
    _save_transaction_state(state_path, state)
    _call_failure_injector(failure_injector, "after_copy_all", state)

    for relative, group in grouped.items():
        record = state["targets"][relative]
        target = _target_for(group[0], data_root)
        if record["status"] in {"committed", "reused"}:
            _verify_target_files(target, record["expected_files"])
            continue
        if target.exists():
            _verify_target_files(target, record["expected_files"])
            record["status"] = "reused"
            record["migration_action"] = "reused"
        else:
            staging = staging_root / Path(relative)
            _verify_target_files(staging, record["expected_files"])
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging, target)
            _verify_target_files(target, record["expected_files"])
            record["status"] = "committed"
            record["migration_action"] = "migrated"
        state["phase"] = "committing"
        _save_transaction_state(state_path, state)
        _call_failure_injector(
            failure_injector,
            "after_target_commit",
            {"canonical_relative_to_data_root": relative, "state": state},
        )

    if not all(
        record["status"] in {"committed", "reused"}
        for record in state["targets"].values()
    ):
        raise MigrationError("not all canonical targets were committed")

    relocation_entries: list[dict[str, Any]] = []
    reverse_actions: list[dict[str, Any]] = []
    for item in items:
        relative = str(item["canonical_relative_to_data_root"]).replace("\\", "/")
        source = _source_for(item, project_root)
        action = state["targets"][relative]["migration_action"]
        relocation_entries.append(
            {
                "legacy_relative_path": item["source_path"],
                "legacy_absolute_path_at_migration": str(source),
                "canonical_relative_path": relative,
                "directory_tree_sha256": item["source_tree_sha256"],
                "migration_time_utc": state["started_at_utc"],
                "migration_tool_version": MIGRATION_TOOL_VERSION,
                "migration_action": action,
            }
        )
        reverse_actions.append(
            {
                "manual_action": "copy_verified_canonical_tree_back_to_legacy_path",
                "canonical_relative_path": relative,
                "legacy_relative_path": item["source_path"],
                "expected_directory_tree_sha256": item["source_tree_sha256"],
            }
        )

    relocation_path = manifest_dir / "relocation_manifest.json"
    existing = (
        json.loads(relocation_path.read_text(encoding="utf-8"))
        if relocation_path.is_file()
        else {}
    )
    prior = existing.get("relocations", []) if isinstance(existing, dict) else []
    keyed = {
        (entry.get("legacy_relative_path"), entry.get("canonical_relative_path")): entry
        for entry in prior
        if isinstance(entry, Mapping)
    }
    for entry in relocation_entries:
        keyed[(entry["legacy_relative_path"], entry["canonical_relative_path"])] = entry
    atomic_json(
        relocation_path,
        {
            "layout_version": LAYOUT_VERSION,
            "migration_tool_version": MIGRATION_TOOL_VERSION,
            "relocations": list(keyed.values()),
        },
    )
    state["manifests"]["relocation_manifest"] = True
    _save_transaction_state(state_path, state)
    atomic_json(
        manifest_dir / "reverse_migration_plan.json",
        {
            "automatic_rollback": False,
            "warning": "Review and execute these actions manually; never overwrite existing paths.",
            "actions": reverse_actions,
        },
    )
    state["manifests"]["reverse_migration_plan"] = True
    _save_transaction_state(state_path, state)
    _write_catalog(manifest_dir, items, status="migrated")
    state["manifests"]["artifact_catalog"] = True
    _save_transaction_state(state_path, state)

    migrated = sum(
        len(grouped[relative])
        for relative, record in state["targets"].items()
        if record["migration_action"] == "migrated"
    )
    reused = sum(
        len(grouped[relative])
        for relative, record in state["targets"].items()
        if record["migration_action"] == "reused"
    )
    result = {
        "transaction_id": state["transaction_id"],
        "planned_items": len(items),
        "migrated_items": migrated,
        "reused_items": reused,
        "conflict_items": 0,
        "unclassified_items": 0,
        "source_roots_removed_if_empty": [],
        "source_deletion_completed": False,
        "overall_pass": False,
    }
    atomic_json(manifest_dir / "migration_apply_summary.json", result)
    state["manifests"]["migration_apply_summary"] = True
    state["phase"] = "manifests_written"
    _save_transaction_state(state_path, state)

    if not all(state["manifests"].values()):
        raise MigrationError("required migration manifests were not written")

    for item in items:
        source_relative = str(item["source_path"])
        source_record = state["sources"][source_relative]
        if source_record["deleted"]:
            continue
        relative = str(item["canonical_relative_to_data_root"]).replace("\\", "/")
        target = _target_for(item, data_root)
        _verify_target_files(target, state["targets"][relative]["expected_files"])
        source = _source_for(item, project_root)
        deletion_path = (
            staging_root / ".source_delete" / Path(source_relative)
        ).resolve()
        source_record["deletion_path"] = str(deletion_path)
        source_record["status"] = "deletion_intent"
        _save_transaction_state(state_path, state)
        if source.exists():
            if tree_stats(source)[2] != item["source_tree_sha256"]:
                raise MigrationError(f"source changed before deletion: {source_relative}")
            if deletion_path.exists():
                raise MigrationError(
                    f"source and detached deletion tree both exist: {source_relative}"
                )
            deletion_path.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, deletion_path)
        if not deletion_path.exists():
            raise MigrationError(f"source disappeared before verified deletion: {source_relative}")
        if tree_stats(deletion_path)[2] != item["source_tree_sha256"]:
            raise MigrationError(f"detached source changed before deletion: {source_relative}")
        source_record["status"] = "source_detached"
        _save_transaction_state(state_path, state)
        _delete_verified_source(deletion_path)
        _call_failure_injector(
            failure_injector,
            "after_source_delete_before_state_save",
            {"source_path": source_relative, "state": state},
        )
        source_record["deleted"] = True
        source_record["status"] = "deleted"
        state["phase"] = "deleting_sources"
        _save_transaction_state(state_path, state)

    roots_removed: list[str] = []
    for root_name in SOURCE_ROOT_NAMES:
        root = project_root / root_name
        if root.is_dir() and not any(root.iterdir()):
            root.rmdir()
            roots_removed.append(root_name)

    result["source_roots_removed_if_empty"] = roots_removed
    result["source_deletion_completed"] = True
    result["overall_pass"] = migrated + reused == len(items)
    atomic_json(manifest_dir / "migration_apply_summary.json", result)

    # The final successful summary is committed before the transaction is
    # marked complete.  Therefore a completed transaction can never point at
    # the earlier provisional overall_pass=false summary.
    state["phase"] = "summary_finalized"
    _save_transaction_state(state_path, state)
    _call_failure_injector(
        failure_injector,
        "after_final_summary_before_complete",
        {"result": result, "state": state},
    )
    state["phase"] = "complete"
    state["complete"] = True
    _save_transaction_state(state_path, state)
    return result


def status(project_root: Path = PROJECT_ROOT, data_root: Path | None = None) -> dict[str, Any]:
    project_root = project_root.resolve()
    data_root = (data_root or get_ch4_data_root()).resolve()
    try:
        plan = load_plan(data_root)
        items = list(plan["items"])
    except (MigrationError, OSError, json.JSONDecodeError):
        items = []

    manifest_dir = data_root / "manifests"
    relocation_path = manifest_dir / "relocation_manifest.json"
    relocations = []
    if relocation_path.is_file():
        payload = json.loads(relocation_path.read_text(encoding="utf-8"))
        relocations = payload.get("relocations", []) if isinstance(payload, dict) else []
    actions = [entry.get("migration_action") for entry in relocations]

    state_path = manifest_dir / "migration_transaction_state.json"
    transaction_state = _load_transaction_state(state_path)
    transaction_complete = bool(
        isinstance(transaction_state, Mapping)
        and transaction_state.get("complete") is True
    )
    transaction_phase = (
        transaction_state.get("phase")
        if isinstance(transaction_state, Mapping)
        else None
    )
    all_source_records_deleted = bool(
        transaction_complete
        and all(
            bool(record.get("deleted"))
            for record in (transaction_state.get("sources") or {}).values()
            if isinstance(record, Mapping)
        )
    )

    summary_path = manifest_dir / "migration_apply_summary.json"
    apply_summary = None
    if summary_path.is_file():
        value = json.loads(summary_path.read_text(encoding="utf-8"))
        apply_summary = value if isinstance(value, dict) else None
    apply_summary_overall_pass = bool(
        apply_summary and apply_summary.get("overall_pass") is True
    )
    source_deletion_completed = bool(
        apply_summary and apply_summary.get("source_deletion_completed") is True
    )

    remaining = [
        name
        for name in SOURCE_ROOT_NAMES
        if (project_root / name).is_dir() and any((project_root / name).iterdir())
    ]
    summary = _summary(items)
    item_counts_complete = (
        summary["planned_items"] == 0
        or actions.count("migrated") + actions.count("reused")
        == summary["planned_items"]
    )

    result = {
        "planned_items": summary["planned_items"],
        "migrated_items": actions.count("migrated"),
        "reused_items": actions.count("reused"),
        "conflict_items": summary["conflict_items"],
        "unclassified_items": summary["unclassified_items"],
        "source_roots_remaining": remaining,
        "transaction_phase": transaction_phase,
        "transaction_complete": transaction_complete,
        "all_source_records_deleted": all_source_records_deleted,
        "apply_summary_overall_pass": apply_summary_overall_pass,
        "source_deletion_completed": source_deletion_completed,
    }

    no_migration_needed = summary["planned_items"] == 0 and not remaining
    completed_migration = (
        item_counts_complete
        and transaction_complete
        and all_source_records_deleted
        and apply_summary_overall_pass
        and source_deletion_completed
        and not remaining
    )
    result["overall_pass"] = (
        result["conflict_items"] == 0
        and result["unclassified_items"] == 0
        and (no_migration_needed or completed_migration)
    )
    return result

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("version", "plan", "verify-plan", "apply", "status"))
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--data-root", type=Path, default=None)
    args = parser.parse_args(argv)
    project_root = args.project_root.resolve()
    data_root = args.data_root.resolve() if args.data_root else get_ch4_data_root()
    try:
        if args.mode == "version":
            result = {
                "migration_tool_version": MIGRATION_TOOL_VERSION,
                "layout_version": LAYOUT_VERSION,
            }
        elif args.mode == "plan":
            result = build_plan(project_root, data_root)["summary"]
        elif args.mode == "verify-plan":
            result = verify_plan(project_root, data_root)
        elif args.mode == "apply":
            result = apply_plan(project_root, data_root)
        else:
            result = status(project_root, data_root)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if result.get("overall_pass", True) else 1
    except (MigrationError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[CH4ArtifactMigration] ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
