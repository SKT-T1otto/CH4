# -*- coding: utf-8 -*-
"""Final paired comparison of Perf-RBE, Uniform DR, and three mechanism ablations."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = _THIS_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from registry.ch4_artifact_layout import (
    get_evaluation_dir,
    get_selected_dir,
    get_selection_dir,
    get_smoke_dir,
    resolve_artifact_path,
)
from tools._internal.ch4_rbe_ablation_eval_common import (
    ExperimentError, build_source_lock, ensure_evaluation, model_load_audit,
    paired_bootstrap, pairing_audit, pool_units, read_json, require, sha256,
    verify_source_lock, write_csv, write_json,
)

SCRIPT_VERSION = "20260728-v2-ablation-final-comparison"
TEST_SEEDS = (271, 272, 273, 274, 275)
SELECTION_SUMMARY = resolve_artifact_path(
    get_selection_dir("ablations", "ch4_rbe_perf_ablation_checkpoint_selection_v2")
    / "ablation_checkpoint_selection_summary.json"
)
MODELS = {
    "uniform_dr": resolve_artifact_path(get_selected_dir("uniform_dr", "ch4_uniform_dr_selected_full9d_v2") / "selected_by_uniform_dr_validation.pt"),
    "perf_rbe": resolve_artifact_path(get_selected_dir("perf_rbe", "ch4_rbe_boundary_core_perf_selected_seed2_v1") / "selected_rbe_model.pt"),
    "no_boundary": resolve_artifact_path(get_selected_dir("no_boundary", "ch4_rbe_perf_ablation_no_boundary_selected_seed2_v1") / "selected_ablation_model.pt"),
    "all_actors": resolve_artifact_path(get_selected_dir("all_actors", "ch4_rbe_perf_ablation_all_actors_selected_seed2_v1") / "selected_ablation_model.pt"),
    "no_nominal": resolve_artifact_path(get_selected_dir("no_nominal", "ch4_rbe_perf_ablation_no_nominal_selected_seed2_v1") / "selected_ablation_model.pt"),
}
SELECTED_MANIFESTS = {
    variant: resolve_artifact_path(get_selected_dir(variant, selected_id) / "selected_model_manifest.json")
    for variant, selected_id in {
        "no_boundary": "ch4_rbe_perf_ablation_no_boundary_selected_seed2_v1",
        "all_actors": "ch4_rbe_perf_ablation_all_actors_selected_seed2_v1",
        "no_nominal": "ch4_rbe_perf_ablation_no_nominal_selected_seed2_v1",
    }.items()
}
PREFLIGHT_ROOT = get_evaluation_dir(
    "ablations", "ch4_rbe_perf_ablation_comparison_preflight_v2"
)
SMOKE_ROOT = get_smoke_dir("ablations", "ch4_rbe_perf_ablation_comparison_v2")
FORMAL_ROOT = get_evaluation_dir("ablations", "ch4_rbe_perf_ablation_comparison_v2")
STAGING_ROOT = FORMAL_ROOT.with_name(FORMAL_ROOT.name + ".incomplete")


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
        "protocol_id": protocol_id,
        "model_path": str(model_path),
        "model_sha256": sha256(model_path),
    })
    return metrics, rows


def run_self_test() -> int:
    checks = {
        "five_models": len(MODELS) == 5,
        "test_seed_contract": TEST_SEEDS == (271, 272, 273, 274, 275),
        "three_selected_manifests": len(SELECTED_MANIFESTS) == 3,
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
    selection = read_json(SELECTION_SUMMARY)
    require(selection.get("overall_pass") is True, "checkpoint selection did not pass")
    require(selection.get("ready_for_final_ablation_comparison") is True, "checkpoint selection is not ready for comparison")
    require(selection.get("reserved_comparison_seeds") == list(TEST_SEEDS), "reserved comparison seed contract changed")

    model_audits = {}
    for model_id, path in MODELS.items():
        audit = model_load_audit(path)
        require(audit["overall_pass"], f"model load audit failed: {model_id}")
        model_audits[model_id] = audit
    selected_manifests = {}
    for model_id, path in SELECTED_MANIFESTS.items():
        manifest = read_json(path)
        require(manifest.get("selected_sha256") == sha256(MODELS[model_id]), f"selected manifest hash mismatch: {model_id}")
        require(manifest.get("reserved_comparison_seeds") == list(TEST_SEEDS), f"reserved seed contract mismatch: {model_id}")
        selected_manifests[model_id] = manifest

    source_lock = build_source_lock((_THIS_FILE, PROJECT_ROOT / "tools/_internal/ch4_rbe_ablation_eval_common.py"))
    result = {
        "overall_pass": True,
        "script_version": SCRIPT_VERSION,
        "test_seeds": list(TEST_SEEDS),
        "model_count": len(MODELS),
        "model_audits": model_audits,
        "selected_manifests": selected_manifests,
        "source_lock": source_lock,
        "training_performed": False,
        "optimizer_update_count": 0,
    }
    write_json(PREFLIGHT_ROOT / "preflight_summary.json", result)
    print(json.dumps({"overall_pass": True, "script_version": SCRIPT_VERSION, "model_count": len(MODELS)}, indent=2))
    return 0


def require_preflight() -> Dict[str, Any]:
    preflight = read_json(PREFLIGHT_ROOT / "preflight_summary.json")
    require(preflight.get("overall_pass") is True, "preflight has not passed")
    verify_source_lock(preflight["source_lock"])
    return preflight


def run_smoke() -> int:
    require_preflight()
    if SMOKE_ROOT.exists():
        shutil.rmtree(SMOKE_ROOT)
    audits = {}
    for protocol_id in ("full9d", "nominal"):
        rows_by_model = {}
        for model_id, model_path in MODELS.items():
            _, rows = evaluate_model(model_id, model_path, (930,), 2, protocol_id, SMOKE_ROOT, max_steps=20)
            rows_by_model[model_id] = rows
        audit = pairing_audit(rows_by_model)
        require(audit["overall_pass"], f"smoke pairing audit failed: {protocol_id}")
        audits[protocol_id] = audit
    result = {
        "overall_pass": True,
        "stage": "smoke",
        "script_version": SCRIPT_VERSION,
        "seed": 930,
        "episodes_per_unit": 2,
        "max_steps": 20,
        "protocol_audits": audits,
    }
    write_json(SMOKE_ROOT / "smoke_summary.json", result)
    print(json.dumps(result, indent=2))
    return 0


def mechanism_decision(comparison_lookup: Dict[Tuple[str, str, str], Dict[str, Any]], preflight: Dict[str, Any]) -> Dict[str, Any]:
    def item(baseline: str, protocol: str, metric: str):
        return comparison_lookup[(baseline, protocol, metric)]

    no_boundary_sif = item("no_boundary", "full9d", "succ_if_found")
    no_boundary_success = item("no_boundary", "full9d", "success")
    boundary_supported = (
        no_boundary_sif["ci95_low"] > 0.0
        or no_boundary_success["ci95_low"] > 0.0
    )

    all_actors_sif = item("all_actors", "full9d", "succ_if_found")
    all_actors_found = item("all_actors", "full9d", "found")
    freeze_supported = (
        all_actors_sif["ci95_low"] > 0.0
        or all_actors_found["ci95_low"] > 0.0
    )

    no_nominal_manifest = preflight["selected_manifests"]["no_nominal"]
    no_nominal_success = item("no_nominal", "nominal", "success")
    no_nominal_found = item("no_nominal", "nominal", "found")
    no_nominal_sif = item("no_nominal", "nominal", "succ_if_found")
    anchor_supported = bool(no_nominal_manifest.get("diagnostic_only")) or any(
        result["candidate_minus_baseline"] > 0.03 and result["ci95_low"] > 0.0
        for result in (no_nominal_success, no_nominal_found, no_nominal_sif)
    )
    return {
        "boundary_core_evidence_supported": boundary_supported,
        "search_actor_freeze_evidence_supported": freeze_supported,
        "nominal_anchor_evidence_supported": anchor_supported,
        "interpretation": {
            "boundary_core": "Perf-RBE minus No-Boundary full-9D Success/SIF",
            "search_actor_freeze": "Perf-RBE minus All-Actors full-9D Found/SIF",
            "nominal_anchor": "No-Nominal diagnostic status or nominal degradation beyond 3 percentage points",
        },
    }


def run_formal() -> int:
    preflight = require_preflight()
    smoke = read_json(SMOKE_ROOT / "smoke_summary.json")
    require(smoke.get("overall_pass") is True, "formal refused: run and pass smoke first")
    require(not FORMAL_ROOT.exists(), f"formal output already exists: {FORMAL_ROOT}")
    STAGING_ROOT.mkdir(parents=True, exist_ok=True)

    metrics = []
    rows_by_key: Dict[Tuple[str, str], List[Dict[str, str]]] = {}
    for model_id, model_path in MODELS.items():
        for protocol_id, episodes in (("full9d", 100), ("nominal", 50)):
            model_metrics, rows = evaluate_model(model_id, model_path, TEST_SEEDS, episodes, protocol_id, STAGING_ROOT)
            metrics.append(model_metrics)
            rows_by_key[(model_id, protocol_id)] = rows
    write_csv(STAGING_ROOT / "ablation_model_metrics.csv", metrics)

    pairing = {}
    for protocol_id in ("full9d", "nominal"):
        audit = pairing_audit({
            model_id: rows_by_key[(model_id, protocol_id)]
            for model_id in MODELS
        })
        require(audit["overall_pass"], f"formal pairing audit failed: {protocol_id}")
        pairing[protocol_id] = audit
    write_json(STAGING_ROOT / "paired_disturbance_audit.json", pairing)

    comparisons = []
    comparison_lookup = {}
    metrics_to_compare = (
        "success", "found", "succ_if_found", "not_found", "found_but_failed",
        "recovery_time", "safety_cost", "final_distance", "final_nav_distance",
        "action_smoothness", "completion_steps",
    )
    for baseline_id in ("no_boundary", "all_actors", "no_nominal", "uniform_dr"):
        for protocol_id in ("full9d", "nominal"):
            for metric in metrics_to_compare:
                result = paired_bootstrap(
                    rows_by_key[("perf_rbe", protocol_id)],
                    rows_by_key[(baseline_id, protocol_id)],
                    metric,
                    repetitions=20000,
                    seed=20260728 + len(comparisons),
                )
                result.update({
                    "comparison_id": f"perf_rbe_vs_{baseline_id}",
                    "candidate_model_id": "perf_rbe",
                    "baseline_model_id": baseline_id,
                    "protocol_id": protocol_id,
                })
                comparisons.append(result)
                comparison_lookup[(baseline_id, protocol_id, metric)] = result
    write_csv(STAGING_ROOT / "paired_ablation_comparisons.csv", comparisons)

    decision = mechanism_decision(comparison_lookup, preflight)
    summary = {
        "overall_pass": True,
        "stage_completed": True,
        "experiment_type": "ch4_rbe_perf_ablation_comparison_v2",
        "script_version": SCRIPT_VERSION,
        "test_seeds": list(TEST_SEEDS),
        "full9d_episodes_per_seed": 100,
        "nominal_episodes_per_seed": 50,
        "total_evaluation_units": 50,
        "total_episodes": 3750,
        "model_metrics": metrics,
        "paired_comparisons": comparisons,
        "paired_disturbance_audit": pairing,
        "mechanism_decision": decision,
        "training_performed": False,
        "optimizer_update_count": 0,
        "test_results_used_for_reselection": False,
        "source_artifacts_unchanged": True,
    }
    write_json(STAGING_ROOT / "ablation_comparison_summary.json", summary)
    verify_source_lock(preflight["source_lock"])
    os.replace(STAGING_ROOT, FORMAL_ROOT)
    print(json.dumps({
        "overall_pass": True,
        "output": str(FORMAL_ROOT),
        "total_episodes": 3750,
        "mechanism_decision": decision,
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
        print(f"[RBEAblationComparison] ERROR {type(exc).__name__}: {exc}")
        raise SystemExit(2)
