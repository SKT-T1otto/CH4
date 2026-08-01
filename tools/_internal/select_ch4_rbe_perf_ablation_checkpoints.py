# -*- coding: utf-8 -*-
"""Select one checkpoint for each of three Perf-RBE mechanism ablations."""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = _THIS_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from registry.ch4_artifact_layout import (
    get_family_root,
    get_selected_dir,
    get_selection_dir,
    get_smoke_dir,
    get_training_run_dir,
    resolve_artifact_path,
)
from tools._internal.ch4_rbe_ablation_eval_common import (
    ExperimentError, audit_evaluation_unit, build_source_lock, ensure_evaluation,
    json_safe, model_load_audit, paired_bootstrap, pairing_audit, pool_units,
    read_json, require, sha256, verify_source_lock, write_csv, write_json,
)

SCRIPT_VERSION = "20260728-v2-ablation-checkpoint-selection"
COARSE_SEEDS = (266, 267)
FINAL_SEEDS = (268, 269, 270)
RESERVED_COMPARISON_SEEDS = (271, 272, 273, 274, 275)
VARIANTS = {
    "no_boundary": {
        "training_dir": "ch4_rbe_perf_ablation_no_boundary_seed2_2000ep_v1",
        "selected_dir": "ch4_rbe_perf_ablation_no_boundary_selected_seed2_v1",
    },
    "all_actors": {
        "training_dir": "ch4_rbe_perf_ablation_all_actors_seed2_2000ep_v1",
        "selected_dir": "ch4_rbe_perf_ablation_all_actors_selected_seed2_v1",
    },
    "no_nominal": {
        "training_dir": "ch4_rbe_perf_ablation_no_nominal_seed2_2000ep_v1",
        "selected_dir": "ch4_rbe_perf_ablation_no_nominal_selected_seed2_v1",
    },
}
PERF_MODEL = resolve_artifact_path(
    get_selected_dir("perf_rbe", "ch4_rbe_boundary_core_perf_selected_seed2_v1")
    / "selected_rbe_model.pt"
)
UNIFORM_MODEL = resolve_artifact_path(
    get_selected_dir("uniform_dr", "ch4_uniform_dr_selected_full9d_v2")
    / "selected_by_uniform_dr_validation.pt"
)
TRAINING_SUMMARY = resolve_artifact_path(
    get_family_root("ablations")
    / "training_audits/ch4_rbe_perf_ablation_training_audit_v1"
    / "formal_training_batch_summary.json"
)
PREFLIGHT_ROOT = get_selection_dir(
    "ablations", "ch4_rbe_perf_ablation_checkpoint_selection_preflight_v2"
)
SMOKE_ROOT = get_smoke_dir(
    "ablations", "ch4_rbe_perf_ablation_checkpoint_selection_v2"
)
FORMAL_ROOT = get_selection_dir(
    "ablations", "ch4_rbe_perf_ablation_checkpoint_selection_v2"
)
STAGING_ROOT = FORMAL_ROOT.with_name(FORMAL_ROOT.name + ".incomplete")


def checkpoint_episode(path: Path) -> int:
    match = re.search(r"snapshot_ep(\d+)", path.name)
    require(match is not None, f"invalid checkpoint name: {path.name}")
    return int(match.group(1))


def snapshots(variant: str) -> List[Path]:
    directory = resolve_artifact_path(
        get_training_run_dir(variant, VARIANTS[variant]["training_dir"])
    )
    return sorted(directory.glob("snapshot_ep*.pt"), key=checkpoint_episode)


def checkpoint_update_count(variant: str, episode: int) -> int | None:
    progress_path = resolve_artifact_path(
        get_training_run_dir(variant, VARIANTS[variant]["training_dir"])
        / "training_progress.csv"
    )
    if not progress_path.is_file():
        return None
    import csv
    with progress_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(float(row.get("episode", 0))) == episode:
                value = row.get("optimizer_update_count_cumulative")
                return None if value in (None, "") else int(float(value))
    return None


def evaluate_model(
    model_id: str,
    model_path: Path,
    seeds: Tuple[int, ...],
    episodes: int,
    protocol_id: str,
    root: Path,
    max_steps: int = 400,
):
    protocol = "uniform_9d_registry_v1" if protocol_id == "full9d" else "nominal_v1"
    directories = []
    for seed in seeds:
        unit_dir = root / protocol_id / model_id / f"seed{seed}"
        ensure_evaluation(
            model_path=model_path,
            seed=seed,
            episodes=episodes,
            max_steps=max_steps,
            protocol=protocol,
            result_dir=unit_dir,
        )
        directories.append(unit_dir)
    metrics, rows = pool_units(directories)
    metrics.update({
        "model_id": model_id,
        "model_path": str(model_path),
        "model_sha256": sha256(model_path),
        "checkpoint_episode": checkpoint_episode(model_path) if model_path.name.startswith("snapshot_ep") else None,
        "protocol_id": protocol_id,
    })
    return metrics, rows


def rank_key(metrics: Dict[str, Any]):
    return (
        float(metrics.get("success_rate") or -1.0),
        float(metrics.get("succ_if_found") or -1.0),
        -float(metrics.get("found_but_failed_rate") or 0.0),
        -float(metrics.get("avg_safety_cost") or 1e12),
        -float(metrics.get("avg_final_distance") or 1e12),
        -float(metrics.get("avg_action_smoothness") or 1e12),
        -int(metrics.get("checkpoint_episode") or 10**9),
    )


def run_self_test() -> int:
    checks = {
        "three_variants": len(VARIANTS) == 3,
        "coarse_seeds": COARSE_SEEDS == (266, 267),
        "final_seeds": FINAL_SEEDS == (268, 269, 270),
        "reserved_seeds": RESERVED_COMPARISON_SEEDS == (271, 272, 273, 274, 275),
        "disjoint_seed_sets": not (set(COARSE_SEEDS) | set(FINAL_SEEDS)) & set(RESERVED_COMPARISON_SEEDS),
    }
    result = {
        "overall_pass": all(checks.values()),
        "case_count": len(checks),
        "passed_case_count": sum(checks.values()),
        "checks": checks,
    }
    print(json.dumps(result, indent=2))
    return 0 if result["overall_pass"] else 2


def run_preflight() -> int:
    training = read_json(TRAINING_SUMMARY)
    require(training.get("overall_pass") is True, "ablation training audit did not pass")
    require(training.get("ready_for_checkpoint_selection") is True, "ablation training is not ready for checkpoint selection")
    require(training.get("variant_count") == 3, "ablation training variant count changed")
    require(training.get("total_training_episodes") == 6000, "ablation training episode count changed")

    model_audits = []
    for variant in VARIANTS:
        variant_snapshots = snapshots(variant)
        require(len(variant_snapshots) == 20, f"{variant}: expected 20 snapshots, got {len(variant_snapshots)}")
        require([checkpoint_episode(path) for path in variant_snapshots] == list(range(100, 2001, 100)), f"{variant}: snapshot episode sequence changed")
        for path in variant_snapshots:
            audit = model_load_audit(path)
            require(audit["overall_pass"], f"model load audit failed: {path}")
            audit.update({"variant": variant, "checkpoint_episode": checkpoint_episode(path)})
            model_audits.append(audit)
    for model_id, path in (("perf_rbe", PERF_MODEL), ("uniform_dr", UNIFORM_MODEL)):
        audit = model_load_audit(path)
        require(audit["overall_pass"], f"reference model load audit failed: {path}")
        audit["model_id"] = model_id
        model_audits.append(audit)

    source_lock = build_source_lock((_THIS_FILE, PROJECT_ROOT / "tools/_internal/ch4_rbe_ablation_eval_common.py"))
    result = {
        "overall_pass": True,
        "script_version": SCRIPT_VERSION,
        "snapshot_count": 60,
        "reference_model_count": 2,
        "model_load_audit_count": len(model_audits),
        "coarse_seeds": list(COARSE_SEEDS),
        "final_seeds": list(FINAL_SEEDS),
        "reserved_comparison_seeds": list(RESERVED_COMPARISON_SEEDS),
        "training_performed": False,
        "optimizer_update_count": 0,
        "source_lock": source_lock,
        "model_audits": model_audits,
    }
    write_json(PREFLIGHT_ROOT / "preflight_summary.json", result)
    print(json.dumps({key: result[key] for key in ("overall_pass", "script_version", "snapshot_count", "model_load_audit_count")}, indent=2))
    return 0


def require_preflight() -> Dict[str, Any]:
    preflight = read_json(PREFLIGHT_ROOT / "preflight_summary.json")
    require(preflight.get("overall_pass") is True, "preflight has not passed")
    verify_source_lock(preflight["source_lock"])
    return preflight


def run_smoke() -> int:
    preflight = require_preflight()
    if SMOKE_ROOT.exists():
        shutil.rmtree(SMOKE_ROOT)
    model_paths = {variant: snapshots(variant)[0] for variant in VARIANTS}
    model_paths.update({"perf_rbe": PERF_MODEL, "uniform_dr": UNIFORM_MODEL})
    protocol_audits = {}
    for protocol_id in ("full9d", "nominal"):
        rows_by_model = {}
        for model_id, model_path in model_paths.items():
            _, rows = evaluate_model(model_id, model_path, (929,), 2, protocol_id, SMOKE_ROOT, max_steps=20)
            rows_by_model[model_id] = rows
        audit = pairing_audit(rows_by_model)
        require(audit["overall_pass"], f"smoke pairing audit failed for {protocol_id}")
        protocol_audits[protocol_id] = audit
    result = {
        "overall_pass": True,
        "stage": "smoke",
        "script_version": SCRIPT_VERSION,
        "seed": 929,
        "episodes_per_unit": 2,
        "max_steps": 20,
        "model_count": len(model_paths),
        "protocol_audits": protocol_audits,
        "source_artifacts_unchanged": True,
    }
    write_json(SMOKE_ROOT / "smoke_summary.json", result)
    print(json.dumps(result, indent=2))
    return 0


def run_formal() -> int:
    preflight = require_preflight()
    smoke = read_json(SMOKE_ROOT / "smoke_summary.json")
    require(smoke.get("overall_pass") is True, "formal refused: run and pass smoke first")
    require(not FORMAL_ROOT.exists(), f"formal output already exists: {FORMAL_ROOT}")
    STAGING_ROOT.mkdir(parents=True, exist_ok=True)

    coarse_metrics = []
    finalists: Dict[str, List[Dict[str, Any]]] = {}
    for variant in VARIANTS:
        variant_metrics = []
        for model_path in snapshots(variant):
            episode = checkpoint_episode(model_path)
            model_id = f"{variant}_ep{episode:04d}"
            metrics, _ = evaluate_model(model_id, model_path, COARSE_SEEDS, 25, "full9d", STAGING_ROOT / "coarse")
            metrics.update({
                "variant": variant,
                "optimizer_update_count_cumulative": checkpoint_update_count(variant, episode),
            })
            coarse_metrics.append(metrics)
            variant_metrics.append(metrics)
        variant_metrics.sort(key=rank_key, reverse=True)
        finalists[variant] = variant_metrics[:3]
    write_csv(STAGING_ROOT / "coarse_checkpoint_metrics.csv", coarse_metrics)
    write_json(STAGING_ROOT / "finalists.json", finalists)

    final_metrics = []
    final_rows: Dict[Tuple[str, str], List[Dict[str, str]]] = {}
    finalist_paths: Dict[str, Path] = {}
    for variant, items in finalists.items():
        for item in items:
            model_id = item["model_id"]
            model_path = Path(item["model_path"])
            finalist_paths[model_id] = model_path
            for protocol_id, episodes in (("full9d", 100), ("nominal", 50)):
                metrics, rows = evaluate_model(model_id, model_path, FINAL_SEEDS, episodes, protocol_id, STAGING_ROOT / "final")
                metrics.update({"variant": variant, "optimizer_update_count_cumulative": item.get("optimizer_update_count_cumulative")})
                final_metrics.append(metrics)
                final_rows[(model_id, protocol_id)] = rows
    for model_id, model_path in (("perf_rbe", PERF_MODEL), ("uniform_dr", UNIFORM_MODEL)):
        finalist_paths[model_id] = model_path
        for protocol_id, episodes in (("full9d", 100), ("nominal", 50)):
            metrics, rows = evaluate_model(model_id, model_path, FINAL_SEEDS, episodes, protocol_id, STAGING_ROOT / "final")
            metrics["variant"] = "reference"
            final_metrics.append(metrics)
            final_rows[(model_id, protocol_id)] = rows
    write_csv(STAGING_ROOT / "finalist_metrics.csv", final_metrics)

    pairing = {}
    for protocol_id in ("full9d", "nominal"):
        rows_by_model = {
            model_id: rows
            for (model_id, protocol), rows in final_rows.items()
            if protocol == protocol_id
        }
        audit = pairing_audit(rows_by_model)
        require(audit["overall_pass"], f"formal pairing audit failed for {protocol_id}")
        pairing[protocol_id] = audit
    write_json(STAGING_ROOT / "paired_disturbance_audit.json", pairing)

    def get_metrics(model_id: str, protocol_id: str) -> Dict[str, Any]:
        return next(row for row in final_metrics if row["model_id"] == model_id and row["protocol_id"] == protocol_id)

    uniform_nominal = get_metrics("uniform_dr", "nominal")
    selections: Dict[str, Any] = {}
    paired_selected_comparisons = []
    for variant, items in finalists.items():
        candidates = []
        for item in items:
            model_id = item["model_id"]
            full9d = get_metrics(model_id, "full9d")
            nominal = get_metrics(model_id, "nominal")
            nominal_guard = all(
                nominal.get(field) is not None
                and uniform_nominal.get(field) is not None
                and nominal[field] >= uniform_nominal[field] - 0.03
                for field in ("success_rate", "found_rate", "succ_if_found")
            )
            candidates.append({
                "model_id": model_id,
                "model_path": item["model_path"],
                "nominal_guard_pass": nominal_guard,
                "full9d": full9d,
                "nominal": nominal,
                "rank_key": rank_key(full9d),
            })
        guarded = [candidate for candidate in candidates if candidate["nominal_guard_pass"]]
        selected = max(guarded or candidates, key=lambda candidate: candidate["rank_key"])
        diagnostic_only = not bool(guarded)
        source_path = Path(selected["model_path"])
        selected_dir = get_selected_dir(variant, VARIANTS[variant]["selected_dir"])
        selected_dir.mkdir(parents=True, exist_ok=True)
        destination = selected_dir / "selected_ablation_model.pt"
        shutil.copy2(source_path, destination)
        require(sha256(source_path) == sha256(destination), f"selected model copy hash mismatch: {variant}")

        manifest = {
            "variant": variant,
            "selected_checkpoint_episode": checkpoint_episode(source_path),
            "optimizer_update_count_cumulative": checkpoint_update_count(variant, checkpoint_episode(source_path)),
            "source_path": str(source_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "source_sha256": sha256(source_path),
            "selected_path": str(destination.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "selected_sha256": sha256(destination),
            "selection_coarse_seeds": list(COARSE_SEEDS),
            "selection_final_seeds": list(FINAL_SEEDS),
            "reserved_comparison_seeds": list(RESERVED_COMPARISON_SEEDS),
            "nominal_guard_margin": 0.03,
            "nominal_guard_pass": not diagnostic_only,
            "diagnostic_only": diagnostic_only,
            "training_performed_during_selection": False,
        }
        write_json(selected_dir / "selected_model_manifest.json", manifest)

        for baseline_id in ("perf_rbe", "uniform_dr"):
            for protocol_id in ("full9d", "nominal"):
                for metric in ("success", "found", "succ_if_found", "found_but_failed"):
                    result = paired_bootstrap(
                        final_rows[(selected["model_id"], protocol_id)],
                        final_rows[(baseline_id, protocol_id)],
                        metric,
                        repetitions=20000,
                        seed=20260728 + len(paired_selected_comparisons),
                    )
                    result.update({
                        "variant": variant,
                        "candidate_model_id": selected["model_id"],
                        "baseline_model_id": baseline_id,
                        "protocol_id": protocol_id,
                    })
                    paired_selected_comparisons.append(result)
        selections[variant] = {
            "manifest": manifest,
            "full9d": selected["full9d"],
            "nominal": selected["nominal"],
            "candidate_count": len(candidates),
            "guarded_candidate_count": len(guarded),
        }

    write_csv(STAGING_ROOT / "paired_selected_comparisons.csv", paired_selected_comparisons)
    summary = {
        "overall_pass": True,
        "ready_for_final_ablation_comparison": True,
        "stage_completed": True,
        "experiment_type": "ch4_rbe_perf_ablation_checkpoint_selection_v2",
        "script_version": SCRIPT_VERSION,
        "coarse_seeds": list(COARSE_SEEDS),
        "final_seeds": list(FINAL_SEEDS),
        "reserved_comparison_seeds": list(RESERVED_COMPARISON_SEEDS),
        "coarse_episode_count": 3000,
        "final_episode_count": 4950,
        "total_episode_count": 7950,
        "selections": selections,
        "pairing_audit": pairing,
        "training_performed": False,
        "optimizer_update_count": 0,
        "source_artifacts_unchanged": True,
    }
    write_json(STAGING_ROOT / "ablation_checkpoint_selection_summary.json", summary)
    verify_source_lock(preflight["source_lock"])
    os.replace(STAGING_ROOT, FORMAL_ROOT)
    print(json.dumps({
        "overall_pass": True,
        "ready_for_final_ablation_comparison": True,
        "output": str(FORMAL_ROOT),
        "selected_checkpoint_episodes": {
            variant: data["manifest"]["selected_checkpoint_episode"]
            for variant, data in selections.items()
        },
    }, indent=2))
    return 0


def run_status() -> int:
    result = {
        "formal_exists": FORMAL_ROOT.exists(),
        "incomplete_exists": STAGING_ROOT.exists(),
        "smoke_passed": (SMOKE_ROOT / "smoke_summary.json").is_file()
        and read_json(SMOKE_ROOT / "smoke_summary.json").get("overall_pass") is True,
    }
    if STAGING_ROOT.exists():
        result["completed_evaluation_unit_count"] = len(list(STAGING_ROOT.rglob("evaluation_summary.json")))
    print(json.dumps(result, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("version", "self-test", "preflight", "smoke", "formal", "status"))
    args = parser.parse_args()
    if args.mode == "version":
        print(f"script_version={SCRIPT_VERSION}")
        return 0
    if args.mode == "self-test":
        return run_self_test()
    if args.mode == "preflight":
        return run_preflight()
    if args.mode == "smoke":
        return run_smoke()
    if args.mode == "formal":
        return run_formal()
    return run_status()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ExperimentError as exc:
        print(f"[RBEAblationSelection] ERROR {type(exc).__name__}: {exc}")
        raise SystemExit(2)
