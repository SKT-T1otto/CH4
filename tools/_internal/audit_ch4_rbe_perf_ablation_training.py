#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit formal training artifacts for the three Perf-RBE mechanism ablations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from registry.ch4_artifact_layout import (
    get_archive_dir,
    get_evaluation_dir,
    get_family_root,
    get_training_run_dir,
    resolve_artifact_path,
)
from tools._internal.prepare_ch4_rbe_perf_ablation_protocols import (
    SCRIPT_VERSION as PROTOCOL_SCRIPT_VERSION,
    SOURCE_NAME_MAP,
    VARIANTS,
    canonical_sha,
)

SCRIPT_VERSION = "20260727-v1-perf-ablation-training-audit"
EXPECTED_EPISODES = 2000
EXPECTED_MAX_STEPS = 400
EXPECTED_SNAPSHOT_EPISODES = list(range(100, 2001, 100))
EXPECTED_SEED = 2


class TrainingAuditError(RuntimeError):
    pass


def require(condition: Any, message: str) -> None:
    if not bool(condition):
        raise TrainingAuditError(message)


def load_json(path: Path) -> Dict[str, Any]:
    require(path.is_file(), f"missing JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root must be object: {path}")
    return value


def read_csv(path: Path):
    require(path.is_file(), f"missing CSV: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def artifact_paths(root: Path, variant: str) -> Dict[str, Path]:
    del root
    run_name = VARIANTS[variant]["run_name"]
    run_dir = resolve_artifact_path(get_training_run_dir(variant, run_name))
    return {
        "log": run_dir,
        "model": run_dir,
        "protocol": resolve_artifact_path(
            get_evaluation_dir(variant, VARIANTS[variant]["protocol_name"])
        ),
        "audit": get_family_root("ablations")
        / "training_audits/ch4_rbe_perf_ablation_training_audit_v1"
        / variant,
    }


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def audit_variant(root: Path, variant: str, write_output: bool = True) -> Dict[str, Any]:
    spec = VARIANTS[variant]
    paths = artifact_paths(root, variant)
    config_path = paths["log"] / "training_config.json"
    completion_path = paths["log"] / "training_completion.json"
    progress_path = paths["log"] / "training_progress.csv"
    episodes_path = paths["log"] / "episode_metrics.csv"
    boundary_path = paths["log"] / "boundary_dataset.csv"
    final_model = paths["model"] / "maddpg_uavenv_final.pt"
    manifest_path = paths["protocol"] / "rbe_training_protocol_manifest.json"
    required = [config_path, completion_path, progress_path, episodes_path, boundary_path, final_model, manifest_path]
    missing = [str(path) for path in required if not path.is_file()]
    require(not missing, f"missing formal artifacts: {missing}")

    config = load_json(config_path)
    completion = load_json(completion_path)
    manifest = load_json(manifest_path)
    progress = read_csv(progress_path)
    episodes = read_csv(episodes_path)
    boundary_rows = read_csv(boundary_path)

    require(config.get("run_name") == spec["run_name"], "training_config run name mismatch")
    require(config.get("mode") == "ch4_rbe_full", "training mode mismatch")
    require(config.get("seed") == EXPECTED_SEED, "training seed mismatch")
    require(config.get("max_episodes") == EXPECTED_EPISODES, "max episodes mismatch")
    require(config.get("max_steps") == EXPECTED_MAX_STEPS, "max steps mismatch")
    require(config.get("snapshot_interval") == 100, "snapshot interval mismatch")
    require(config.get("rbe_protocol_name") == spec["protocol_name"], "protocol name mismatch")
    require(config.get("rbe_ablation_id") == spec["ablation_id"], "ablation id mismatch")
    require(config.get("rbe_raw_ratios") == spec["sampling_ratios"], "raw ratio mismatch")
    require(config.get("rbe_effective_ratios") == spec["sampling_ratios"], "effective ratio mismatch")
    require(config.get("rbe_ratios_were_normalized") is False, "ratios were normalized")
    require(config.get("freeze_search_actors") is spec["search_actors_frozen"], "freeze flag mismatch")
    require(config.get("rbe_executor_only_actor") is spec["executor_only_actor_training"], "executor-only flag mismatch")
    require(config.get("optimizer_state_reset_applied") is True, "optimizer reset not applied")
    require(math.isclose(float(config.get("effective_actor_lr")), 1e-4, abs_tol=1e-12), "actor LR mismatch")
    require(math.isclose(float(config.get("effective_critic_lr")), 3e-4, abs_tol=1e-12), "critic LR mismatch")
    require(config.get("reb_enable") is False and config.get("rscu_enable") is False, "REB/RSCU unexpectedly enabled")
    require(config.get("rbe_sampler_enable") is True, "RBE sampler disabled")
    require(config.get("collect_boundary_dataset") is True, "boundary dataset collection disabled")

    require(manifest.get("protocol_name") == spec["protocol_name"], "manifest protocol mismatch")
    require(manifest.get("ablation_id") == spec["ablation_id"], "manifest ablation mismatch")
    require(manifest.get("sampling_ratios") == spec["sampling_ratios"], "manifest ratios mismatch")
    require(manifest.get("protocol_object_sha256") == canonical_sha(manifest), "manifest canonical SHA mismatch")
    require(sha256(manifest_path) == config.get("rbe_training_protocol_manifest_sha256"), "manifest file changed after training start")

    require(len(progress) == EXPECTED_EPISODES, "training_progress row count mismatch")
    require(len(episodes) == EXPECTED_EPISODES, "episode_metrics row count mismatch")
    require(len(boundary_rows) == EXPECTED_EPISODES, "boundary_dataset row count mismatch")
    expected_sequence = list(range(1, EXPECTED_EPISODES + 1))
    for label, rows in (("progress", progress), ("episodes", episodes), ("boundary", boundary_rows)):
        sequence = [int(float(row.get("episode", -1))) for row in rows]
        require(sequence == expected_sequence, f"{label} episode sequence mismatch")

    for row in progress:
        for field in ("avg_reward", "success", "found", "recovery_time", "safety_cost", "final_distance", "final_nav_distance", "action_smoothness"):
            require(_finite(row.get(field)), f"non-finite progress field: {field}")
    require(all(not (float(row["success"]) >= 0.5 and float(row["found"]) < 0.5) for row in progress), "success exceeds found")

    def snapshot_episode(path: Path) -> int:
        try:
            return int(path.stem.split("snapshot_ep", 1)[1])
        except Exception as exc:
            raise TrainingAuditError(f"invalid snapshot name: {path}") from exc

    snapshot_paths = sorted(paths["model"].glob("snapshot_ep*.pt"), key=snapshot_episode)
    snapshot_episodes = [snapshot_episode(path) for path in snapshot_paths]
    require(snapshot_episodes == EXPECTED_SNAPSHOT_EPISODES, f"snapshot episode set mismatch: {snapshot_episodes}")

    require(completion.get("run_name") == spec["run_name"], "completion run name mismatch")
    require(completion.get("episodes_completed") == EXPECTED_EPISODES, "completion episode count mismatch")
    require(completion.get("rbe_protocol_name") == spec["protocol_name"], "completion protocol mismatch")
    require(completion.get("rbe_sampling_ratios") == spec["sampling_ratios"], "completion ratio mismatch")
    require(completion.get("optimizer_state_reset_applied") is True, "completion optimizer reset false")
    require(int(completion.get("optimizer_update_count", 0)) > 0, "no optimizer updates")
    actor_counts = [int(value) for value in completion.get("per_agent_actor_update_count", [])]
    critic_counts = [int(value) for value in completion.get("per_agent_critic_update_count", [])]
    require(len(actor_counts) == 4 and len(critic_counts) == 4, "update count vector length mismatch")
    require(all(value > 0 for value in critic_counts), "a critic received no update")
    if spec["search_actors_frozen"]:
        require(actor_counts[:3] == [0, 0, 0] and actor_counts[3] > 0, "executor-only update count mismatch")
        require(completion.get("search_actor_policy_changed") is False, "frozen search policy changed")
        require(completion.get("executor_actor_policy_changed") is True, "executor policy did not change")
    else:
        require(all(value > 0 for value in actor_counts), "all-actors ablation did not update every actor")
        require(completion.get("search_actor_policy_changed") is True, "search policies did not change")
        require(completion.get("executor_actor_policy_changed") is True, "executor policy did not change")
    require(completion.get("final_model_sha256") == sha256(final_model), "final model SHA mismatch")

    source_counts = {"boundary": 0, "uniform": 0, "high_risk": 0, "nominal": 0}
    for row in progress:
        source = str(row.get("rbe_sample_source", ""))
        require(source in source_counts, f"unknown sample source: {source}")
        source_counts[source] += 1
    target_source_ratios = {SOURCE_NAME_MAP[key]: value for key, value in spec["sampling_ratios"].items()}
    observed_source_ratios = {key: value / float(EXPECTED_EPISODES) for key, value in source_counts.items()}
    for source, target in target_source_ratios.items():
        if target == 0.0:
            require(source_counts[source] == 0, f"inactive source appeared: {source}")
        else:
            require(source_counts[source] > 0, f"active source missing: {source}")
            require(abs(observed_source_ratios[source] - target) <= 0.035, f"source ratio drift too large: {source}")

    result = {
        "overall_pass": True,
        "script_version": SCRIPT_VERSION,
        "protocol_script_version": PROTOCOL_SCRIPT_VERSION,
        "variant": variant,
        "ablation_id": spec["ablation_id"],
        "run_name": spec["run_name"],
        "training_seed": EXPECTED_SEED,
        "episodes_completed": EXPECTED_EPISODES,
        "snapshot_count": len(snapshot_paths),
        "snapshot_episodes": snapshot_episodes,
        "optimizer_update_count": int(completion.get("optimizer_update_count")),
        "per_agent_actor_update_count": actor_counts,
        "per_agent_critic_update_count": critic_counts,
        "search_actor_policy_changed": completion.get("search_actor_policy_changed"),
        "executor_actor_policy_changed": completion.get("executor_actor_policy_changed"),
        "sampling_ratios": spec["sampling_ratios"],
        "source_counts": source_counts,
        "source_ratios": observed_source_ratios,
        "protocol_object_sha256": manifest.get("protocol_object_sha256"),
        "final_model_sha256": completion.get("final_model_sha256"),
        "source_artifacts_unchanged": True,
        "ready_for_checkpoint_selection": True,
        "errors": [],
    }
    if write_output:
        paths["audit"].mkdir(parents=True, exist_ok=True)
        atomic_json(paths["audit"] / "training_audit_summary.json", result)
    return result


def seed_status(root: Path, variant: str) -> int:
    paths = artifact_paths(root, variant)
    if not paths["log"].exists() and not paths["model"].exists():
        return 2
    try:
        audit_variant(root, variant, write_output=True)
        return 0
    except (TrainingAuditError, OSError, ValueError, KeyError, json.JSONDecodeError):
        return 3


def archive_incomplete(root: Path, variant: str) -> Dict[str, Any]:
    paths = artifact_paths(root, variant)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archived = []
    for key in ("log", "model"):
        path = paths[key]
        if path.exists():
            target = get_archive_dir("interrupted") / (
                path.name + f".interrupted_{timestamp}_{key}"
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            require(not target.exists(), f"archive target exists: {target}")
            os.replace(path, target)
            archived.append(str(target))
    require(archived, f"no incomplete artifacts found for {variant}")
    return {"overall_pass": True, "variant": variant, "archived": archived}


def batch_finalize(root: Path) -> Dict[str, Any]:
    results = [audit_variant(root, variant, write_output=True) for variant in VARIANTS]
    summary = {
        "overall_pass": all(result["overall_pass"] for result in results),
        "experiment_type": "ch4_rbe_perf_mechanism_ablations_training_v1",
        "script_version": SCRIPT_VERSION,
        "training_seed": EXPECTED_SEED,
        "variants": results,
        "variant_count": len(results),
        "total_training_episodes": EXPECTED_EPISODES * len(results),
        "ready_for_checkpoint_selection": all(result["ready_for_checkpoint_selection"] for result in results),
        "errors": [],
    }
    out_dir = (
        get_family_root("ablations")
        / "training_audits/ch4_rbe_perf_ablation_training_audit_v1"
    )
    atomic_json(out_dir / "formal_training_batch_summary.json", summary)
    lines = [
        "# Perf-RBE Mechanism Ablation Training v1",
        "",
        f"- overall_pass: `{summary['overall_pass']}`",
        f"- ready_for_checkpoint_selection: `{summary['ready_for_checkpoint_selection']}`",
        "",
        "| variant | episodes | snapshots | actor updates | critic updates | ready |",
        "| --- | ---: | ---: | --- | --- | --- |",
    ]
    for row in results:
        lines.append(
            f"| {row['variant']} | {row['episodes_completed']} | {row['snapshot_count']} | "
            f"{row['per_agent_actor_update_count']} | {row['per_agent_critic_update_count']} | {row['ready_for_checkpoint_selection']} |"
        )
    (out_dir / "formal_training_batch_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("version", "audit", "seed-status", "archive-incomplete", "batch-finalize", "status"))
    parser.add_argument("--artifact-root", default=str(PROJECT_ROOT))
    parser.add_argument("--variant", choices=tuple(VARIANTS), default=None)
    args = parser.parse_args()
    root = Path(args.artifact_root).resolve()
    if args.mode == "version":
        print(json.dumps({"script_version": SCRIPT_VERSION}, indent=2))
        return 0
    try:
        if args.mode == "batch-finalize":
            result = batch_finalize(root)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0 if result["overall_pass"] else 1
        require(args.variant is not None, f"--variant is required for {args.mode}")
        if args.mode == "audit":
            result = audit_variant(root, args.variant, write_output=True)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0
        if args.mode == "seed-status":
            code = seed_status(root, args.variant)
            print(json.dumps({"variant": args.variant, "status_code": code}, indent=2))
            return code
        if args.mode == "archive-incomplete":
            result = archive_incomplete(root, args.variant)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0
        if args.mode == "status":
            code = seed_status(root, args.variant)
            status = {0: "complete", 2: "not_started", 3: "incomplete_or_invalid"}.get(code, "unknown")
            print(json.dumps({"variant": args.variant, "status": status, "status_code": code}, indent=2))
            return 0
    except (TrainingAuditError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"[RBEPerfAblationAudit] ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
