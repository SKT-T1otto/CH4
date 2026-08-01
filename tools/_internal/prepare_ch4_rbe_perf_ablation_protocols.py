#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare and audit the three Perf-RBE mechanism-ablation protocols.

This tool never trains a policy.  It derives immutable ablation protocol
manifests from the already-finalized Perf-RBE protocol and reuses the exact
frozen candidate CSVs, Uniform-DR warm start, Found-aware REB model, controller
configuration, optimizer reset, and learning rates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from registry.rbe_disturbance import DEFAULT_DISTURBANCE_BOUNDS, DISTURBANCE_KEYS
from registry.ch4_artifact_layout import (
    get_ch4_data_root,
    get_evaluation_dir,
    get_smoke_dir,
    resolve_artifact_path,
)
from utils.rbe_sampler import RBEDisturbanceSampler

SCRIPT_VERSION = "20260727-v1-perf-ablation-protocols"
BASE_PROTOCOL_NAME = "ch4_rbe_boundary_core_training_protocol_perf_v1"
BASE_PROTOCOL_DIR = get_evaluation_dir("perf_rbe", BASE_PROTOCOL_NAME)
BASE_MANIFEST = BASE_PROTOCOL_DIR / "rbe_training_protocol_manifest.json"
BASE_BOUNDARY = BASE_PROTOCOL_DIR / "boundary_core_candidates.csv"
BASE_HIGH_RISK = BASE_PROTOCOL_DIR / "composite_high_risk_candidates.csv"
EXPECTED_UNIFORM_SHA = "47ae3748017c4c9a43efc16ecb9973b535716c0e09d119f2de43615b8e5405e7"
EXPECTED_REB_SHA = "675688640fbfc0aadb6982a16baaeb6ab103e45f2bf630fc439d6d3b1ec412cd"
EXPECTED_SELECTED_CANDIDATES_SHA = "64f6be43307f903d218e54f136b823911da9e2bec58999a9eb710ff865c88f4e"
EXPECTED_SCORE_RULE_SHA = "928eb37e02ad69f582a4f8f7016004acef51fd24b0140c4f06cab19c6cc0a1c9"
DRY_RUN_SAMPLES = 50000
DRY_RUN_SEED = 20260727

VARIANTS: Dict[str, Dict[str, Any]] = {
    "no_boundary": {
        "ablation_id": "no_boundary_core",
        "protocol_name": "ch4_rbe_perf_ablation_no_boundary_protocol_v1",
        "run_name": "ch4_rbe_perf_ablation_no_boundary_seed2_2000ep_v1",
        "smoke_name": "ch4_rbe_perf_ablation_no_boundary_smoke_v1",
        "method_name_zh": "Perf-RBE消融：移除边界核心采样",
        "method_name_en": "Perf-RBE Ablation: No Boundary-Core Sampling",
        "method_statement_zh": "该消融移除Boundary Core采样，并将其概率质量转移至Uniform覆盖；其余Perf-RBE训练设置保持不变，用于检验边界定向采样的独立贡献。High-risk仍仅作为低比例综合风险增强，不解释为纯粹的发现后执行失败样本。",
        "method_statement_en": "This ablation removes Boundary-Core sampling and reallocates its probability mass to Uniform coverage while retaining all other Perf-RBE settings. It isolates the contribution of boundary-directed sampling. High-risk remains low-ratio composite-risk augmentation and is not interpreted as pure post-detection execution failure.",
        "sampling_ratios": {"boundary_core": 0.00, "uniform_coverage": 0.75, "composite_high_risk_aux": 0.05, "nominal_anchor": 0.20},
        "search_actors_frozen": True,
        "executor_only_actor_training": True,
        "inactive_sources": ["boundary"],
    },
    "all_actors": {
        "ablation_id": "all_actors_trainable",
        "protocol_name": "ch4_rbe_perf_ablation_all_actors_protocol_v1",
        "run_name": "ch4_rbe_perf_ablation_all_actors_seed2_2000ep_v1",
        "smoke_name": "ch4_rbe_perf_ablation_all_actors_smoke_v1",
        "method_name_zh": "Perf-RBE消融：全部Actor可训练",
        "method_name_en": "Perf-RBE Ablation: All Actors Trainable",
        "method_statement_zh": "该消融保持Perf-RBE的扰动采样分布、优化器重置和学习率不变，但取消搜索Actor冻结，使四个Actor同时更新，用于检验角色受控微调的独立贡献。",
        "method_statement_en": "This ablation retains the Perf-RBE disturbance mixture, optimizer reset, and learning rates, but removes search-actor freezing so that all four actors update. It isolates the contribution of role-controlled fine-tuning.",
        "sampling_ratios": {"boundary_core": 0.40, "uniform_coverage": 0.35, "composite_high_risk_aux": 0.05, "nominal_anchor": 0.20},
        "search_actors_frozen": False,
        "executor_only_actor_training": False,
        "inactive_sources": [],
    },
    "no_nominal": {
        "ablation_id": "no_nominal_anchor",
        "protocol_name": "ch4_rbe_perf_ablation_no_nominal_protocol_v1",
        "run_name": "ch4_rbe_perf_ablation_no_nominal_seed2_2000ep_v1",
        "smoke_name": "ch4_rbe_perf_ablation_no_nominal_smoke_v1",
        "method_name_zh": "Perf-RBE消融：移除名义锚点",
        "method_name_en": "Perf-RBE Ablation: No Nominal Anchor",
        "method_statement_zh": "该消融移除Nominal Anchor，并将其概率质量转移至Uniform覆盖；边界采样、搜索Actor冻结、执行体定向微调、优化器重置和学习率保持不变，用于检验名义锚点对性能保持的独立贡献。",
        "method_statement_en": "This ablation removes the Nominal Anchor and reallocates its probability mass to Uniform coverage while retaining boundary sampling, search-actor freezing, executor-directed fine-tuning, optimizer reset, and learning rates. It isolates the nominal anchor's contribution to performance retention.",
        "sampling_ratios": {"boundary_core": 0.40, "uniform_coverage": 0.55, "composite_high_risk_aux": 0.05, "nominal_anchor": 0.00},
        "search_actors_frozen": True,
        "executor_only_actor_training": True,
        "inactive_sources": ["nominal"],
    },
}

SOURCE_NAME_MAP = {
    "boundary_core": "boundary",
    "uniform_coverage": "uniform",
    "composite_high_risk_aux": "high_risk",
    "nominal_anchor": "nominal",
}


class AblationProtocolError(RuntimeError):
    pass


def require(condition: Any, message: str) -> None:
    if not bool(condition):
        raise AblationProtocolError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    require(path.is_file(), f"missing JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root must be object: {path}")
    return value


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
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


def canonical_sha(value: Mapping[str, Any]) -> str:
    payload = {
        key: item
        for key, item in dict(value).items()
        if key not in {"protocol_object_sha256", "protocol_object_sha256_verified"}
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")


def protocol_dir(variant: str) -> Path:
    return get_evaluation_dir(variant, VARIANTS[variant]["protocol_name"])


def smoke_dirs(variant: str) -> Dict[str, Path]:
    name = VARIANTS[variant]["smoke_name"]
    root = get_smoke_dir(variant, name)
    return {"log": root, "model": root, "eval": root}


def verify_base_protocol() -> Dict[str, Any]:
    for path in (BASE_MANIFEST, BASE_BOUNDARY, BASE_HIGH_RISK):
        require(path.is_file(), f"base Perf-RBE protocol artifact missing: {path}")
    manifest = load_json(BASE_MANIFEST)
    require(manifest.get("protocol_name") == BASE_PROTOCOL_NAME, "base protocol name mismatch")
    require(manifest.get("protocol_object_sha256") == canonical_sha(manifest), "base protocol canonical SHA mismatch")
    require(manifest.get("protocol_object_sha256_verified") is True, "base protocol verification flag false")
    require(manifest.get("uniform_warm_start_model_sha256") == EXPECTED_UNIFORM_SHA, "base Uniform SHA mismatch")
    require(manifest.get("selected_reb_model_sha256") == EXPECTED_REB_SHA, "base REB SHA mismatch")
    require(manifest.get("source_selected_candidates_sha256") == EXPECTED_SELECTED_CANDIDATES_SHA, "base selected-candidates SHA mismatch")
    require(manifest.get("source_score_rule_sha256") == EXPECTED_SCORE_RULE_SHA, "base score-rule SHA mismatch")
    require(sha256(BASE_BOUNDARY) == manifest.get("boundary_candidate_sha256"), "base Boundary CSV SHA mismatch")
    require(sha256(BASE_HIGH_RISK) == manifest.get("high_risk_candidate_sha256"), "base High-risk CSV SHA mismatch")
    for label, source_path_text in (manifest.get("source_paths") or {}).items():
        source_path = resolve_artifact_path(str(source_path_text))
        require(source_path.is_file(), f"base source missing: {label}: {source_path}")
        require(sha256(source_path) == (manifest.get("source_hashes") or {}).get(label), f"base source SHA mismatch: {label}")
    return manifest


def build_manifest(variant: str, base: Mapping[str, Any], out_dir: Path) -> Dict[str, Any]:
    spec = VARIANTS[variant]
    manifest = dict(base)
    manifest.update(
        {
            "protocol_name": spec["protocol_name"],
            "protocol_version": 1,
            "method_name_zh": spec["method_name_zh"],
            "method_name_en": spec["method_name_en"],
            "method_statement_zh": spec["method_statement_zh"],
            "method_statement_en": spec["method_statement_en"],
            "sampling_ratios": dict(spec["sampling_ratios"]),
            "ratio_sum": float(sum(spec["sampling_ratios"].values())),
            "search_actors_frozen": bool(spec["search_actors_frozen"]),
            "executor_only_actor_training": bool(spec["executor_only_actor_training"]),
            "all_policy_actors_trainable": not bool(spec["search_actors_frozen"]),
            "optimization_scope": "mechanism_ablation_only",
            "ablation_id": spec["ablation_id"],
            "ablation_variant": variant,
            "ablation_contract": {
                "single_factor_change": True,
                "inactive_sources": list(spec["inactive_sources"]),
                "reference_protocol_name": BASE_PROTOCOL_NAME,
                "reference_protocol_object_sha256": base.get("protocol_object_sha256"),
                "reference_sampling_ratios": base.get("sampling_ratios"),
                "reference_search_actors_frozen": base.get("search_actors_frozen"),
                "reference_executor_only_actor_training": base.get("executor_only_actor_training"),
            },
            "boundary_candidate_csv": rel(out_dir / "boundary_core_candidates.csv"),
            "high_risk_candidate_csv": rel(out_dir / "composite_high_risk_candidates.csv"),
            "boundary_candidate_sha256": sha256(BASE_BOUNDARY),
            "high_risk_candidate_sha256": sha256(BASE_HIGH_RISK),
            "planned_training_seeds": [2],
            "planned_episodes_per_seed": 2000,
            "planned_max_steps": 400,
            "planned_snapshot_interval": 100,
            "ready_for_training_smoke": True,
            "ready_for_formal_training": False,
            "source_artifacts_unchanged": True,
            "errors": [],
            "warnings": [
                "This is a mechanism ablation and must not be interpreted as a new algorithmic module.",
                "The candidate CSV content is identical to the frozen Perf-RBE source; a zero ratio disables sampling without deleting the source artifact.",
                "Formal training is blocked until the variant-specific smoke passes.",
            ],
        }
    )
    manifest["protocol_object_sha256"] = canonical_sha(manifest)
    manifest["protocol_object_sha256_verified"] = True
    require(manifest["protocol_object_sha256"] == canonical_sha(manifest), "generated canonical SHA mismatch")
    return manifest


def dry_run(variant: str, out_dir: Path, manifest: Mapping[str, Any]) -> Dict[str, Any]:
    ratios = VARIANTS[variant]["sampling_ratios"]
    sampler = RBEDisturbanceSampler(
        boundary_csv=str(out_dir / "boundary_core_candidates.csv"),
        high_risk_csv=str(out_dir / "composite_high_risk_candidates.csv"),
        rng=np.random.default_rng(DRY_RUN_SEED),
        boundary_ratio=ratios["boundary_core"],
        uniform_ratio=ratios["uniform_coverage"],
        high_risk_ratio=ratios["composite_high_risk_aux"],
        nominal_ratio=ratios["nominal_anchor"],
        jitter_std=0.02,
        jitter_prob=0.50,
        include_flow_phase=True,
        strict_protocol=True,
        expected_boundary_count=64,
        expected_high_risk_count=64,
        expected_boundary_group="reb_boundary",
        expected_high_risk_group="reb_high_risk",
        require_ratio_sum_one=True,
        high_risk_required=True,
        method_role_labels={"boundary": "boundary_core", "high_risk": "composite_high_risk_aux"},
        score_rule_sha256=EXPECTED_SCORE_RULE_SHA,
        source_selected_candidates_sha256=EXPECTED_SELECTED_CANDIDATES_SHA,
    )
    counts = {"boundary": 0, "uniform": 0, "high_risk": 0, "nominal": 0}
    violations = []
    for _ in range(DRY_RUN_SAMPLES):
        xi = sampler.sample()
        counts[sampler.last_source] += 1
        for key in DISTURBANCE_KEYS:
            low, high = DEFAULT_DISTURBANCE_BOUNDS[key]
            if not (float(low) <= float(xi[key]) <= float(high)):
                violations.append(f"{key}_out_of_bounds")
        if not isinstance(xi["action_delay_steps"], (int, np.integer)):
            violations.append("action_delay_not_integer")
        if "flow_phase_x" not in xi or "flow_phase_y" not in xi:
            violations.append("flow_phase_missing")
    empirical = {key: value / float(DRY_RUN_SAMPLES) for key, value in counts.items()}
    targets = {SOURCE_NAME_MAP[key]: value for key, value in ratios.items()}
    for source, target in targets.items():
        tolerance = 0.0 if target == 0.0 else 0.012
        require(abs(empirical[source] - target) <= tolerance, f"dry-run ratio mismatch: {source}")
        if target == 0.0:
            require(counts[source] == 0, f"inactive source appeared: {source}")
        else:
            require(counts[source] > 0, f"active source absent: {source}")
    require(not violations, f"dry-run violations: {violations[:10]}")
    return {
        "overall_pass": True,
        "variant": variant,
        "samples": DRY_RUN_SAMPLES,
        "seed": DRY_RUN_SEED,
        "target_ratios": targets,
        "source_counts": counts,
        "source_ratios": empirical,
        "inactive_sources": list(VARIANTS[variant]["inactive_sources"]),
        "ratios_were_normalized": bool(sampler.ratios_were_normalized),
        "bounds_violation_count": 0,
        "noninteger_action_delay_count": 0,
        "flow_phase_missing_count": 0,
        "protocol_object_sha256": manifest.get("protocol_object_sha256"),
    }


def finalize_variant(variant: str) -> Dict[str, Any]:
    base = verify_base_protocol()
    out_dir = protocol_dir(variant)
    if out_dir.exists():
        return check_variant(variant)
    stage = out_dir.with_name(out_dir.name + ".incomplete")
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    shutil.copy2(BASE_BOUNDARY, stage / "boundary_core_candidates.csv")
    shutil.copy2(BASE_HIGH_RISK, stage / "composite_high_risk_candidates.csv")
    manifest = build_manifest(variant, base, out_dir)
    atomic_json(stage / "rbe_training_protocol_manifest.json", manifest)
    audit = dry_run(variant, stage, manifest)
    atomic_json(stage / "sampler_dry_run_audit.json", audit)
    summary = {
        "overall_pass": True,
        "variant": variant,
        "protocol_name": manifest["protocol_name"],
        "protocol_object_sha256": manifest["protocol_object_sha256"],
        "base_protocol_object_sha256": base["protocol_object_sha256"],
        "sampling_ratios": manifest["sampling_ratios"],
        "search_actors_frozen": manifest["search_actors_frozen"],
        "executor_only_actor_training": manifest["executor_only_actor_training"],
        "dry_run_pass": True,
        "smoke_completed": False,
        "ready_for_formal_training": False,
        "source_artifacts_unchanged": True,
        "errors": [],
    }
    atomic_json(stage / "rbe_training_protocol_summary.json", summary)
    os.replace(stage, out_dir)
    return summary


def check_variant(variant: str) -> Dict[str, Any]:
    out_dir = protocol_dir(variant)
    manifest = load_json(out_dir / "rbe_training_protocol_manifest.json")
    summary = load_json(out_dir / "rbe_training_protocol_summary.json")
    spec = VARIANTS[variant]
    require(manifest.get("protocol_name") == spec["protocol_name"], "protocol name mismatch")
    require(manifest.get("ablation_id") == spec["ablation_id"], "ablation id mismatch")
    require(manifest.get("sampling_ratios") == spec["sampling_ratios"], "sampling ratios mismatch")
    require(manifest.get("search_actors_frozen") is spec["search_actors_frozen"], "freeze flag mismatch")
    require(manifest.get("executor_only_actor_training") is spec["executor_only_actor_training"], "executor-only flag mismatch")
    require(manifest.get("protocol_object_sha256") == canonical_sha(manifest), "protocol canonical SHA mismatch")
    require(manifest.get("protocol_object_sha256_verified") is True, "protocol verification flag false")
    require(sha256(out_dir / "boundary_core_candidates.csv") == sha256(BASE_BOUNDARY), "Boundary CSV changed")
    require(sha256(out_dir / "composite_high_risk_candidates.csv") == sha256(BASE_HIGH_RISK), "High-risk CSV changed")
    require(load_json(out_dir / "sampler_dry_run_audit.json").get("overall_pass") is True, "dry-run audit failed")
    require(summary.get("overall_pass") is True, "protocol summary failed")
    return summary


def _read_csv(path: Path):
    require(path.is_file(), f"missing CSV: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def smoke_audit(variant: str) -> Dict[str, Any]:
    check_variant(variant)
    spec = VARIANTS[variant]
    out_dir = protocol_dir(variant)
    manifest_path = out_dir / "rbe_training_protocol_manifest.json"
    manifest = load_json(manifest_path)
    dirs = smoke_dirs(variant)
    config = load_json(dirs["log"] / "training_config.json")
    completion = load_json(dirs["log"] / "training_completion.json")
    progress = _read_csv(dirs["log"] / "training_progress.csv")
    episodes = _read_csv(dirs["log"] / "episode_metrics.csv")
    require(len(progress) == 200 and len(episodes) == 200, "smoke row count mismatch")
    require(config.get("run_name") == spec["smoke_name"], "smoke run name mismatch")
    require(config.get("seed") == 930 + list(VARIANTS).index(variant), "smoke seed mismatch")
    require(config.get("max_episodes") == 200 and config.get("max_steps") == 20, "smoke scale mismatch")
    require(config.get("rbe_protocol_name") == spec["protocol_name"], "smoke protocol mismatch")
    require(config.get("rbe_raw_ratios") == spec["sampling_ratios"], "smoke ratio mismatch")
    require(config.get("freeze_search_actors") is spec["search_actors_frozen"], "smoke freeze mismatch")
    require(config.get("rbe_executor_only_actor") is spec["executor_only_actor_training"], "smoke executor-only mismatch")
    require(config.get("optimizer_state_reset_applied") is True, "smoke optimizer reset missing")
    require(math.isclose(float(config.get("effective_actor_lr")), 1e-4, abs_tol=1e-12), "smoke actor LR mismatch")
    require(math.isclose(float(config.get("effective_critic_lr")), 3e-4, abs_tol=1e-12), "smoke critic LR mismatch")
    require(completion.get("episodes_completed") == 200, "smoke completion count mismatch")
    require(int(completion.get("optimizer_update_count", 0)) > 0, "smoke performed no updates")
    actor_counts = [int(value) for value in completion.get("per_agent_actor_update_count", [])]
    critic_counts = [int(value) for value in completion.get("per_agent_critic_update_count", [])]
    require(len(actor_counts) == 4 and len(critic_counts) == 4, "per-agent update count length mismatch")
    require(all(value > 0 for value in critic_counts), "a critic received no updates")
    if spec["search_actors_frozen"]:
        require(actor_counts[:3] == [0, 0, 0] and actor_counts[3] > 0, "executor-only actor counts invalid")
        require(completion.get("search_actor_policy_changed") is False, "frozen search actor changed")
        require(completion.get("executor_actor_policy_changed") is True, "executor actor did not change")
    else:
        require(all(value > 0 for value in actor_counts), "all-actors ablation did not update every actor")
        require(completion.get("search_actor_policy_changed") is True, "all-actors ablation did not change search policy")
        require(completion.get("executor_actor_policy_changed") is True, "all-actors ablation did not change executor policy")
    counts = {"boundary": 0, "uniform": 0, "high_risk": 0, "nominal": 0}
    for row in progress:
        source = str(row.get("rbe_sample_source", ""))
        require(source in counts, f"unknown smoke source: {source}")
        counts[source] += 1
    targets = {SOURCE_NAME_MAP[key]: value for key, value in spec["sampling_ratios"].items()}
    for source, target in targets.items():
        if target == 0.0:
            require(counts[source] == 0, f"inactive smoke source appeared: {source}")
        else:
            require(counts[source] > 0, f"active smoke source absent: {source}")
    require(sha256(manifest_path) == config.get("rbe_training_protocol_manifest_sha256"), "manifest changed after smoke start")
    require(sha256(BASE_BOUNDARY) == manifest.get("boundary_candidate_sha256"), "source Boundary CSV changed")
    require(sha256(BASE_HIGH_RISK) == manifest.get("high_risk_candidate_sha256"), "source High-risk CSV changed")
    audit = {
        "overall_pass": True,
        "variant": variant,
        "smoke_completed": True,
        "episodes": 200,
        "max_steps": 20,
        "source_counts": counts,
        "per_agent_actor_update_count": actor_counts,
        "per_agent_critic_update_count": critic_counts,
        "search_actor_policy_changed": completion.get("search_actor_policy_changed"),
        "executor_actor_policy_changed": completion.get("executor_actor_policy_changed"),
        "optimizer_update_count": completion.get("optimizer_update_count"),
        "source_artifacts_unchanged": True,
        "ready_for_formal_training": True,
        "errors": [],
    }
    dirs["eval"].mkdir(parents=True, exist_ok=True)
    audit_path = dirs["eval"] / "smoke_audit_summary.json"
    atomic_json(audit_path, audit)
    summary_path = out_dir / "rbe_training_protocol_summary.json"
    summary = load_json(summary_path)
    summary.update(
        {
            "smoke_completed": True,
            "smoke_audit_overall_pass": True,
            "smoke_audit_summary_path": rel(audit_path),
            "smoke_audit_summary_sha256": sha256(audit_path),
            "ready_for_formal_training": True,
        }
    )
    atomic_json(summary_path, summary)
    return audit


def formal_check(variant: str) -> Dict[str, Any]:
    summary = check_variant(variant)
    require(summary.get("smoke_completed") is True, f"{variant}: smoke not completed")
    require(summary.get("smoke_audit_overall_pass") is True, f"{variant}: smoke audit failed")
    require(summary.get("ready_for_formal_training") is True, f"{variant}: not ready for formal training")
    audit_path = resolve_artifact_path(str(summary.get("smoke_audit_summary_path", "")))
    require(audit_path.is_file(), f"{variant}: smoke audit summary missing")
    require(sha256(audit_path) == summary.get("smoke_audit_summary_sha256"), f"{variant}: smoke audit SHA mismatch")
    require(load_json(audit_path).get("overall_pass") is True, f"{variant}: smoke overall_pass false")
    return {"overall_pass": True, "variant": variant, "ready_for_formal_training": True}


def self_test() -> Dict[str, Any]:
    checks = []
    for variant, spec in VARIANTS.items():
        ratios = spec["sampling_ratios"]
        checks.append({"name": f"{variant}_ratio_sum", "pass": math.isclose(sum(ratios.values()), 1.0, abs_tol=1e-12)})
        checks.append({"name": f"{variant}_nonnegative", "pass": all(value >= 0 for value in ratios.values())})
        checks.append({"name": f"{variant}_identity", "pass": spec["protocol_name"].endswith("_v1") and bool(spec["ablation_id"])})
    checks.extend(
        [
            {"name": "no_boundary_zero_boundary", "pass": VARIANTS["no_boundary"]["sampling_ratios"]["boundary_core"] == 0.0},
            {"name": "all_actors_unfrozen", "pass": VARIANTS["all_actors"]["search_actors_frozen"] is False},
            {"name": "no_nominal_zero_nominal", "pass": VARIANTS["no_nominal"]["sampling_ratios"]["nominal_anchor"] == 0.0},
        ]
    )
    return {
        "overall_pass": all(item["pass"] for item in checks),
        "case_count": len(checks),
        "passed_case_count": sum(item["pass"] for item in checks),
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("version", "self-test", "finalize", "check", "smoke-audit", "formal-check", "status"))
    parser.add_argument("--variant", choices=tuple(VARIANTS) + ("all",), default="all")
    args = parser.parse_args()
    if args.mode == "version":
        print(json.dumps({"script_version": SCRIPT_VERSION, "artifact_root": str(get_ch4_data_root())}, indent=2))
        return 0
    if args.mode == "self-test":
        result = self_test()
        print(json.dumps(result, indent=2))
        return 0 if result["overall_pass"] else 1
    variants = list(VARIANTS) if args.variant == "all" else [args.variant]
    results = []
    try:
        for variant in variants:
            if args.mode == "finalize":
                results.append(finalize_variant(variant))
            elif args.mode == "check":
                results.append(check_variant(variant))
            elif args.mode == "smoke-audit":
                results.append(smoke_audit(variant))
            elif args.mode == "formal-check":
                results.append(formal_check(variant))
            elif args.mode == "status":
                out_dir = protocol_dir(variant)
                summary = load_json(out_dir / "rbe_training_protocol_summary.json") if out_dir.exists() else {}
                results.append({"variant": variant, "protocol_exists": out_dir.exists(), "summary": summary})
        overall = all(result.get("overall_pass") is True for result in results)
        print(json.dumps({"overall_pass": overall, "results": results}, indent=2, ensure_ascii=False))
        return 0 if overall else 1
    except (AblationProtocolError, OSError, ValueError, KeyError) as exc:
        print(f"[RBEPerfAblationProtocol] ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
