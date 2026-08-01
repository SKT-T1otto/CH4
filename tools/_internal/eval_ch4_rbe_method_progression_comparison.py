# -*- coding: utf-8 -*-
"""Unified method-progression comparison for Chapter-4 RBE.

This tool evaluates four frozen policies under identical common-random-number
rollouts: Clean/PSE, Uniform DR, original Boundary-Centric RBE v1 (training
seed 2), and performance-stabilized RBE (training seed 2).  It never trains,
updates, reselects, or modifies a model.

Modes
-----
version
self-test
preflight
smoke
formal
status
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


SCRIPT_VERSION = "20260727-v1-method-progression"
EXPERIMENT_TYPE = "ch4_rbe_method_progression_comparison_v1"
AB_MODE = "ch4_rbe_full"
PROFILE = "normal_comm"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from registry.ch4_artifact_layout import (
    get_ch4_data_root,
    get_evaluation_dir,
    get_smoke_dir,
    resolve_artifact_path,
    source_hash_matches_layout_attestation,
)

UPSTREAM_ROOT = resolve_artifact_path(
    get_evaluation_dir("perf_rbe", "ch4_rbe_boundary_core_perf_independent_test_v1")
)
FORMAL_FINAL_ROOT = get_evaluation_dir(
    "perf_rbe", "ch4_rbe_method_progression_comparison_v1"
)
FORMAL_STAGING_ROOT = Path(str(FORMAL_FINAL_ROOT) + ".incomplete")
SMOKE_ROOT = get_smoke_dir("perf_rbe", "ch4_rbe_method_progression_comparison_v1")
PREFLIGHT_ROOT = get_evaluation_dir(
    "perf_rbe", "ch4_rbe_method_progression_comparison_preflight_v1"
)

COMPARISON_SEEDS = [261, 262, 263, 264, 265]
PREVIOUSLY_USED_SEEDS = list(range(241, 256))
SMOKE_SEED = 928

FULL9D_PROTOCOL = "uniform_9d_registry_v1"
NOMINAL_PROTOCOL = "nominal_v1"
FORMAL_FULL9D_EPISODES_PER_SEED = 100
FORMAL_NOMINAL_EPISODES_PER_SEED = 50
FORMAL_MAX_STEPS = 400
SMOKE_EPISODES_PER_UNIT = 2
SMOKE_MAX_STEPS = 20

BOOTSTRAP_REPETITIONS = 20_000
BOOTSTRAP_SEED = 20260727
BOOTSTRAP_CI = 0.95
NOMINAL_GUARD_MARGIN = 0.03
SUCCESS_NONINFERIORITY_MARGIN = 0.03

EPISODE_SEED_MODE = "indexed_common_random_numbers"
EPISODE_SEED_FORMULA = "base_seed * 1000003 + episode_index"

DISTURBANCE_KEYS = (
    "flow_gain",
    "flow_z_gain",
    "drag_scale",
    "buoyancy_bias_delta",
    "a_max_scale",
    "v_max_scale",
    "actuator_lag",
    "action_delay_steps",
    "action_noise_std",
)
FLOW_PHASE_KEYS = ("flow_phase_x", "flow_phase_y")
PAIRED_FIELDS = (*DISTURBANCE_KEYS, *FLOW_PHASE_KEYS)
CONTINUOUS_FIELDS = (
    "reward",
    "recovery_time",
    "safety_cost",
    "final_distance",
    "final_nav_distance",
    "action_smoothness",
    "completion_steps",
)
COMPARISON_METRICS = (
    "success",
    "found",
    "succ_if_found",
    "not_found",
    "found_but_failed",
    "reward",
    "recovery_time",
    "safety_cost",
    "final_distance",
    "final_nav_distance",
    "action_smoothness",
    "completion_steps",
)

UPSTREAM_LOCKED_SOURCE_FILES = (
    "evaluate_pse.py",
    "env.py",
    "train.py",
    "algorithms/maddpg.py",
    "registry/rbe_disturbance.py",
    "utils/rbe_metrics.py",
    "utils/agents.py",
    "utils/networks.py",
    "utils/noise.py",
    "map/map_module.py",
)

CURRENT_SOURCE_PATHS = (
    *UPSTREAM_LOCKED_SOURCE_FILES,
    "tools/_internal/eval_ch4_rbe_method_progression_comparison.py",
    "scripts/eval_ch4_rbe_method_progression_comparison_v1.bat",
)

class IndependentTestError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    role: str
    relative_path: str
    expected_sha256: str
    training_seed: Optional[int]
    selected_episode: int

    @property
    def path(self) -> Path:
        candidate = resolve_artifact_path(self.relative_path)
        if candidate.is_file():
            actual = sha256_file(candidate)
            if actual == self.expected_sha256:
                return candidate.resolve()
            detail = f"; SHA mismatch: {candidate}={actual}"
        else:
            detail = ""
        raise IndependentTestError(
            f"model artifact not found with expected SHA for {self.model_id}: "
            f"{self.relative_path}{detail}"
        )

    @property
    def selected_manifest(self) -> Path:
        path = self.path.parent / "selected_model_manifest.json"
        if not path.is_file():
            raise IndependentTestError(f"selected manifest is missing for {self.model_id}: {path}")
        return path


MODELS: Tuple[ModelSpec, ...] = (
    ModelSpec(
        model_id="clean_pse_baseline",
        role="clean_baseline",
        relative_path="baselines/clean/selected/ch4_clean_baseline_selected_v1/selected_by_nominal_validation.pt",
        expected_sha256="c9807c948b50ca102d59de92801190e110eb2184c3976d6ef7d9c21a6e2f75d1",
        training_seed=1,
        selected_episode=8200,
    ),
    ModelSpec(
        model_id="uniform_dr_baseline",
        role="uniform_dr",
        relative_path="baselines/uniform_dr/selected/ch4_uniform_dr_selected_full9d_v2/selected_by_uniform_dr_validation.pt",
        expected_sha256="47ae3748017c4c9a43efc16ecb9973b535716c0e09d119f2de43615b8e5405e7",
        training_seed=None,
        selected_episode=1400,
    ),
    ModelSpec(
        model_id="old_rbe_v1_seed2",
        role="original_rbe",
        relative_path="rbe_legacy/selected/ch4_rbe_boundary_core_selected_seed2_v1/selected_rbe_model.pt",
        expected_sha256="15578ce5fbff00f393e8e2eb503131d2f4c6560d14dd55b4d6c42e8d913e8290",
        training_seed=2,
        selected_episode=800,
    ),
    ModelSpec(
        model_id="perf_rbe_seed2",
        role="performance_stabilized_rbe",
        relative_path="perf_rbe/selected/ch4_rbe_boundary_core_perf_selected_seed2_v1/selected_rbe_model.pt",
        expected_sha256="5aa6cc539d86863b5eac0b50cdcf5bc4c4d1a5c97fa59670c0742cc23f66331b",
        training_seed=2,
        selected_episode=2000,
    ),
)
MODEL_BY_ID = {model.model_id: model for model in MODELS}
BASELINE_ID = "uniform_dr_baseline"

PAIR_SPECS = (
    ("uniform_vs_clean", "uniform_dr_baseline", "clean_pse_baseline"),
    ("old_rbe_vs_uniform", "old_rbe_v1_seed2", "uniform_dr_baseline"),
    ("perf_rbe_vs_old_rbe", "perf_rbe_seed2", "old_rbe_v1_seed2"),
    ("perf_rbe_vs_uniform", "perf_rbe_seed2", "uniform_dr_baseline"),
)


def log(message: str) -> None:
    print(f"[RBEMethodProgression] {message}", flush=True)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise IndependentTestError(message)


def rel(path: Path | str) -> str:
    path_obj = Path(path).resolve()
    try:
        return path_obj.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path_obj)


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path | str) -> Dict[str, Any]:
    path_obj = Path(path)
    require(path_obj.is_file(), f"missing JSON file: {path_obj}")
    try:
        data = json.loads(path_obj.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise IndependentTestError(f"could not read JSON {path_obj}: {exc}") from exc
    require(isinstance(data, dict), f"JSON root is not an object: {path_obj}")
    return data


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_write_json(path: Path | str, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(
                json_safe(value),
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_text(path: Path | str, text: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_csv(path: Path | str, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in fields})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_csv(path: Path | str) -> List[Dict[str, str]]:
    path_obj = Path(path)
    require(path_obj.is_file(), f"missing CSV file: {path_obj}")
    try:
        with path_obj.open("r", newline="", encoding="utf-8-sig") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except (OSError, csv.Error) as exc:
        raise IndependentTestError(f"could not read CSV {path_obj}: {exc}") from exc


def parse_float(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise IndependentTestError(f"{label} must be numeric, got {value!r}") from exc
    require(math.isfinite(number), f"{label} must be finite, got {value!r}")
    return number


def parse_int(value: Any, label: str) -> int:
    number = parse_float(value, label)
    require(number.is_integer(), f"{label} must be an integer, got {value!r}")
    return int(number)


def parse_flag(value: Any, label: str) -> int:
    if isinstance(value, bool):
        return 1 if value else 0
    text = str(value).strip().lower()
    if text in {"1", "1.0", "true", "yes", "y"}:
        return 1
    if text in {"0", "0.0", "false", "no", "n"}:
        return 0
    raise IndependentTestError(f"{label} must be a boolean flag, got {value!r}")


def close(left: Any, right: Any, tol: float = 1e-12) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=float(tol))
    except (TypeError, ValueError):
        return False


def expected_episode_seed(base_seed: int, episode_index: int) -> int:
    return int(base_seed) * 1_000_003 + int(episode_index)


def rate(numerator: int, denominator: int) -> Optional[float]:
    return None if int(denominator) == 0 else float(numerator) / float(denominator)


def mean(values: Iterable[float]) -> Optional[float]:
    valid = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return None if not valid else float(sum(valid) / len(valid))


def hash_tree(paths: Sequence[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for relative_path in paths:
        path = PROJECT_ROOT / relative_path
        require(path.is_file(), f"required source file is missing: {path}")
        out[relative_path] = sha256_file(path)
    return out


def manifest_contains_sha(manifest: Mapping[str, Any], expected_sha: str) -> bool:
    def walk(value: Any) -> bool:
        if isinstance(value, dict):
            return any(walk(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return any(walk(item) for item in value)
        return isinstance(value, str) and value.lower() == expected_sha.lower()

    return walk(manifest)



def resolve_upstream_file(filename: str) -> Path:
    candidate = resolve_artifact_path(UPSTREAM_ROOT / filename)
    if candidate.is_file():
        return candidate.resolve()
    raise IndependentTestError(f"upstream independent-test file is missing: {candidate}")


def scan_comparison_seed_reuse() -> Dict[str, Any]:
    excluded_roots = {
        FORMAL_FINAL_ROOT.resolve(),
        FORMAL_STAGING_ROOT.resolve(),
        SMOKE_ROOT.resolve(),
        PREFLIGHT_ROOT.resolve(),
    }
    hits: List[Dict[str, Any]] = []

    def excluded(path: Path) -> bool:
        resolved = path.resolve()
        return any(root == resolved or root in resolved.parents for root in excluded_roots)

    def walk(value: Any, key_path: str, path: Path) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                child = f"{key_path}.{key}" if key_path else str(key)
                walk(item, child, path)
            return
        if isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{key_path}[{index}]", path)
            return
        leaf_key = key_path.lower()
        if "seed" not in leaf_key:
            return
        if isinstance(value, bool):
            return
        try:
            number = int(value)
        except (TypeError, ValueError):
            return
        if number in COMPARISON_SEEDS:
            hits.append({"path": rel(path), "key": key_path, "seed": number})

    for top in (get_ch4_data_root(),):
        if not top.exists():
            continue
        for path in top.rglob("*.json"):
            if excluded(path):
                continue
            try:
                data = load_json(path)
            except Exception:
                continue
            walk(data, "", path)
    require(not hits, f"comparison seeds were already found in prior experiment metadata: {hits[:10]}")
    return {
        "overall_pass": True,
        "comparison_seeds": list(COMPARISON_SEEDS),
        "prior_seed_hit_count": 0,
        "prior_seed_hits": [],
        "scan_roots": [rel(get_ch4_data_root())],
    }


def selected_entry(selection: Mapping[str, Any], training_seed: int) -> Mapping[str, Any]:
    rows = selection.get("per_training_seed_selection")
    require(isinstance(rows, list), "selection per_training_seed_selection is missing")
    matches = [row for row in rows if isinstance(row, dict) and row.get("training_seed") == training_seed]
    require(len(matches) == 1, f"selection must contain exactly one row for training seed {training_seed}")
    return matches[0]



def audit_selection_contract() -> Dict[str, Any]:
    summary_path = resolve_upstream_file("independent_test_summary.json")
    summary = load_json(summary_path)
    require(summary.get("overall_pass") is True, "upstream independent test overall_pass is not true")
    require(summary.get("stage_completed") is True, "upstream independent test is incomplete")
    require(summary.get("test_stage") == "formal", "upstream independent test stage is not formal")
    require(
        summary.get("experiment_type") == "ch4_rbe_boundary_core_perf_independent_test_v1",
        "upstream independent test experiment_type mismatch",
    )
    require(summary.get("training_performed") is False, "upstream independent test performed training")
    require(summary.get("optimizer_update_count") == 0, "upstream optimizer count is not zero")
    require(summary.get("test_results_used_for_reselection") is False, "upstream test was used for reselection")
    require(summary.get("source_artifacts_unchanged") is True, "upstream source artifacts changed")
    require(summary.get("selected_models_unchanged") is True, "upstream selected models changed")
    require(summary.get("test_seeds") == [246, 247, 248, 249, 250], "upstream final-test seeds changed")
    require(summary.get("reserved_final_test_seeds_used") is True, "upstream final-test seeds were not recorded as used")
    return {
        "overall_pass": True,
        "upstream_summary_path": rel(summary_path),
        "upstream_summary_sha256": sha256_file(summary_path),
        "upstream_test_seeds": list(summary.get("test_seeds") or []),
        "upstream_scientific_conclusion": (summary.get("scientific_decision") or {}).get("conclusion"),
        "test_results_used_for_reselection": False,
        "training_performed": False,
        "optimizer_update_count": 0,
    }


def audit_selection_source_lock() -> Dict[str, Any]:
    lock_path = resolve_upstream_file("source_lock_manifest.json")
    lock = load_json(lock_path)
    require(lock.get("overall_pass") is True, "upstream source lock overall_pass is not true")
    source_files = lock.get("source_files")
    require(isinstance(source_files, dict), "upstream source lock source_files is missing")
    rows: List[Dict[str, Any]] = []
    for relative_path in UPSTREAM_LOCKED_SOURCE_FILES:
        require(relative_path in source_files, f"upstream source lock lacks {relative_path}")
        path = PROJECT_ROOT / relative_path
        require(path.is_file(), f"locked source file is missing: {path}")
        actual = sha256_file(path)
        expected = str(source_files[relative_path])
        layout_attested = source_hash_matches_layout_attestation(
            relative_path, expected, actual
        )
        require(
            actual == expected or layout_attested,
            f"source file changed after final independent test without a valid "
            f"layout attestation: {relative_path}",
        )
        rows.append({
            "path": relative_path,
            "expected_sha256": expected,
            "actual_sha256": actual,
            "match": actual == expected,
            "layout_refactor_attested": layout_attested,
        })
    return {
        "overall_pass": True,
        "upstream_source_lock_path": rel(lock_path),
        "upstream_source_lock_sha256": sha256_file(lock_path),
        "locked_file_count": len(rows),
        "locked_files": rows,
    }

def audit_model_load(model: ModelSpec) -> Dict[str, Any]:
    try:
        import torch
        from algorithms.maddpg import MADDPG
    except Exception as exc:
        raise IndependentTestError(f"could not import model stack: {exc}") from exc

    try:
        maddpg = MADDPG.init_from_save(str(model.path), device=torch.device("cpu"))
        maddpg.prep_rollouts(device=torch.device("cpu"))
    except Exception as exc:
        raise IndependentTestError(f"could not load model {model.model_id}: {exc}") from exc

    require(int(getattr(maddpg, "nagents", 0)) == 4, f"{model.model_id} agent count is not 4")
    action_dims: List[int] = []
    obs_dims: List[int] = []
    all_finite = True
    for agent_index, agent in enumerate(maddpg.agents):
        policy = agent.policy
        input_dim = int(policy.fc1.in_features)
        output_dim = int(policy.fc3.out_features)
        obs_dims.append(input_dim)
        action_dims.append(output_dim)
        with torch.no_grad():
            output = policy(torch.zeros((1, input_dim), dtype=torch.float32))
        require(tuple(output.shape) == (1, output_dim), f"{model.model_id} policy output shape mismatch")
        all_finite = all_finite and bool(torch.isfinite(output).all().item())
        for parameter in policy.parameters():
            all_finite = all_finite and bool(torch.isfinite(parameter).all().item())
    require(obs_dims == [139, 139, 139, 139], f"{model.model_id} observation dimensions changed: {obs_dims}")
    require(action_dims == [3, 3, 3, 3], f"{model.model_id} action dimensions changed: {action_dims}")
    require(all_finite, f"{model.model_id} contains non-finite policy parameters or output")
    return {
        "model_load_pass": True,
        "agent_count": 4,
        "obs_dims": obs_dims,
        "action_dims": action_dims,
        "all_policy_parameters_finite": True,
        "fixed_forward_pass": True,
    }



def audit_models(load_models: bool) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for model in MODELS:
        path = model.path
        manifest_path = model.selected_manifest
        require(path.is_file() and path.stat().st_size > 0, f"model is missing: {path}")
        actual_sha = sha256_file(path)
        require(actual_sha == model.expected_sha256, f"{model.model_id} SHA256 mismatch")
        manifest = load_json(manifest_path)
        require(
            manifest_contains_sha(manifest, model.expected_sha256),
            f"{model.model_id} selected manifest does not contain the expected SHA",
        )
        if "selected_checkpoint_episode" in manifest:
            require(
                int(manifest.get("selected_checkpoint_episode")) == model.selected_episode,
                f"{model.model_id} selected checkpoint episode mismatch",
            )
        if model.training_seed is not None and "training_seed" in manifest:
            require(
                int(manifest.get("training_seed")) == model.training_seed,
                f"{model.model_id} manifest training_seed mismatch",
            )
        model_load_audit = audit_model_load(model) if load_models else None
        rows.append({
            "model_id": model.model_id,
            "role": model.role,
            "training_seed": model.training_seed,
            "selected_episode": model.selected_episode,
            "path": rel(path),
            "artifact_root": "data/chapter4_final",
            "sha256": actual_sha,
            "size": int(path.stat().st_size),
            "selected_manifest": rel(manifest_path),
            "selected_manifest_sha256": sha256_file(manifest_path),
            "model_load_audit": model_load_audit,
        })
    return {
        "overall_pass": True,
        "model_count": len(rows),
        "models": rows,
        "model_load_audit_performed": bool(load_models),
    }


def preflight_audit(load_models: bool = True) -> Dict[str, Any]:
    upstream = audit_selection_contract()
    upstream_lock = audit_selection_source_lock()
    seed_audit = scan_comparison_seed_reuse()
    model_audit = audit_models(load_models=load_models)
    current_sources = hash_tree(CURRENT_SOURCE_PATHS)
    audit = {
        "overall_pass": True,
        "experiment_type": EXPERIMENT_TYPE,
        "script_version": SCRIPT_VERSION,
        "project_root": str(PROJECT_ROOT),
        "upstream_independent_test_contract": upstream,
        "upstream_source_lock": upstream_lock,
        "comparison_seed_audit": seed_audit,
        "selected_model_identity_audit": model_audit,
        "current_source_sha256": current_sources,
        "comparison_seeds": list(COMPARISON_SEEDS),
        "previously_used_seed_range": list(PREVIOUSLY_USED_SEEDS),
        "smoke_seed": SMOKE_SEED,
        "seed_sets_disjoint": set(COMPARISON_SEEDS).isdisjoint({SMOKE_SEED, *PREVIOUSLY_USED_SEEDS}),
        "training_performed": False,
        "optimizer_update_count": 0,
    }
    require(audit["seed_sets_disjoint"], "smoke/comparison seeds overlap previous experiment seeds")
    return audit


def build_protocol_lock(stage: str, root: Path) -> Dict[str, Any]:
    require(stage in {"smoke", "formal"}, f"unknown stage {stage}")
    source_hashes = hash_tree(CURRENT_SOURCE_PATHS)
    model_hashes = {model.model_id: sha256_file(model.path) for model in MODELS}
    upstream_summary = resolve_upstream_file("independent_test_summary.json")
    upstream_source_lock = resolve_upstream_file("source_lock_manifest.json")
    return {
        "lock_version": 1,
        "experiment_type": EXPERIMENT_TYPE,
        "script_version": SCRIPT_VERSION,
        "stage": stage,
        "project_root": str(PROJECT_ROOT),
        "output_root": rel(FORMAL_FINAL_ROOT if stage == "formal" else root),
        "staging_root": rel(root),
        "ablation_mode": AB_MODE,
        "profile": PROFILE,
        "full9d_protocol": FULL9D_PROTOCOL,
        "nominal_protocol": NOMINAL_PROTOCOL,
        "comparison_seeds": list(COMPARISON_SEEDS) if stage == "formal" else [SMOKE_SEED],
        "full9d_episodes_per_seed": FORMAL_FULL9D_EPISODES_PER_SEED if stage == "formal" else SMOKE_EPISODES_PER_UNIT,
        "nominal_episodes_per_seed": FORMAL_NOMINAL_EPISODES_PER_SEED if stage == "formal" else SMOKE_EPISODES_PER_UNIT,
        "max_steps": FORMAL_MAX_STEPS if stage == "formal" else SMOKE_MAX_STEPS,
        "paired_episode_seeding": True,
        "episode_seed_mode": EPISODE_SEED_MODE,
        "episode_seed_formula": EPISODE_SEED_FORMULA,
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_ci": BOOTSTRAP_CI,
        "nominal_guard_margin": NOMINAL_GUARD_MARGIN,
        "success_noninferiority_margin": SUCCESS_NONINFERIORITY_MARGIN,
        "comparison_pairs": [list(pair) for pair in PAIR_SPECS],
        "model_paths": {model.model_id: rel(model.path) for model in MODELS},
        "model_sha256": model_hashes,
        "source_files": source_hashes,
        "upstream_summary_sha256": sha256_file(upstream_summary),
        "upstream_source_lock_sha256": sha256_file(upstream_source_lock),
        "training_performed": False,
        "optimizer_update_count": 0,
    }

def write_or_validate_lock(root: Path, lock: Mapping[str, Any]) -> None:
    path = root / "protocol_lock_manifest.json"
    if path.exists():
        existing = load_json(path)
        require(existing == dict(lock), f"protocol lock changed on resume: {path}")
    else:
        atomic_write_json(path, lock)


def protocol_spec(protocol_id: str, stage: str) -> Tuple[str, int, int]:
    require(protocol_id in {"full9d", "nominal"}, f"unknown protocol id {protocol_id}")
    if stage == "formal":
        episodes = (
            FORMAL_FULL9D_EPISODES_PER_SEED
            if protocol_id == "full9d"
            else FORMAL_NOMINAL_EPISODES_PER_SEED
        )
        max_steps = FORMAL_MAX_STEPS
    else:
        episodes = SMOKE_EPISODES_PER_UNIT
        max_steps = SMOKE_MAX_STEPS
    protocol = FULL9D_PROTOCOL if protocol_id == "full9d" else NOMINAL_PROTOCOL
    return protocol, episodes, max_steps


def unit_dir(root: Path, protocol_id: str, model_id: str, seed: int) -> Path:
    return root / protocol_id / model_id / f"seed{int(seed)}"


def resolve_record_flag(row: Mapping[str, Any], primary: str, fallback: str) -> int:
    if primary in row:
        return parse_flag(row.get(primary), primary)
    return parse_flag(row.get(fallback), fallback)


def audit_evaluation_unit(
    out_dir: Path,
    model: ModelSpec,
    protocol_id: str,
    seed: int,
    episodes: int,
    max_steps: int,
) -> Dict[str, Any]:
    from registry.rbe_disturbance import DEFAULT_DISTURBANCE_BOUNDS, nominal_disturbance

    summary_path = out_dir / "evaluation_summary.json"
    csv_path = out_dir / "episode_metrics.csv"
    summary = load_json(summary_path)
    rows = read_csv(csv_path)
    expected_protocol, _, _ = protocol_spec(protocol_id, "formal" if max_steps == FORMAL_MAX_STEPS else "smoke")

    require(summary.get("method") == AB_MODE, f"{out_dir} method mismatch")
    require(summary.get("profile") == PROFILE, f"{out_dir} profile mismatch")
    require(summary.get("episodes") == episodes, f"{out_dir} episode count mismatch")
    require(summary.get("max_steps") == max_steps, f"{out_dir} max_steps mismatch")
    require(summary.get("seed") == seed, f"{out_dir} seed mismatch")
    require(summary.get("paired_episode_seeding") is True, f"{out_dir} paired seeding disabled")
    require(summary.get("episode_seed_mode") == EPISODE_SEED_MODE, f"{out_dir} seed mode mismatch")
    require(summary.get("episode_seed_formula") == EPISODE_SEED_FORMULA, f"{out_dir} seed formula mismatch")
    require(summary.get("disturbance_protocol") == expected_protocol, f"{out_dir} protocol mismatch")
    require(summary.get("all_episode_disturbance_apply_match") is True, f"{out_dir} disturbance apply audit failed")
    require(summary.get("bounds_violation_count") == 0, f"{out_dir} disturbance bounds violation")
    require(Path(str(summary.get("model_path", ""))).resolve() == model.path.resolve(), f"{out_dir} model path mismatch")
    require(len(rows) == episodes, f"{out_dir} CSV row count mismatch")

    normalized_records: List[Dict[str, Any]] = []
    nominal = nominal_disturbance()
    for expected_index, row in enumerate(rows):
        base_seed = parse_int(row.get("base_seed"), f"{out_dir} base_seed")
        episode_index = parse_int(row.get("episode_index"), f"{out_dir} episode_index")
        episode_seed = parse_int(row.get("episode_seed"), f"{out_dir} episode_seed")
        disturbance_seed = parse_int(row.get("disturbance_seed"), f"{out_dir} disturbance_seed")
        require(base_seed == seed, f"{out_dir} base_seed mismatch")
        require(episode_index == expected_index, f"{out_dir} episode index sequence mismatch")
        expected_seed_value = expected_episode_seed(seed, expected_index)
        require(episode_seed == expected_seed_value, f"{out_dir} episode seed mismatch")
        require(disturbance_seed == expected_seed_value, f"{out_dir} disturbance seed mismatch")
        require(row.get("episode_seed_mode") == EPISODE_SEED_MODE, f"{out_dir} row seed mode mismatch")
        require(row.get("disturbance_protocol") == expected_protocol, f"{out_dir} row protocol mismatch")

        success = resolve_record_flag(row, "success_flag", "success")
        found = resolve_record_flag(row, "found_flag", "found")
        require(success <= found, f"{out_dir} success exceeds found at episode {expected_index}")

        record: Dict[str, Any] = {
            "model_id": model.model_id,
            "role": model.role,
            "protocol_id": protocol_id,
            "base_seed": base_seed,
            "episode_index": episode_index,
            "episode_seed": episode_seed,
            "disturbance_seed": disturbance_seed,
            "success": success,
            "found": found,
            "not_found": 1 - found,
            "found_but_failed": found - success,
        }

        for key in DISTURBANCE_KEYS:
            value = parse_float(row.get(key), f"{out_dir} {key}")
            low, high = DEFAULT_DISTURBANCE_BOUNDS[key]
            require(float(low) - 1e-12 <= value <= float(high) + 1e-12, f"{out_dir} {key} out of bounds")
            if key == "action_delay_steps":
                require(value.is_integer(), f"{out_dir} action_delay_steps is not integer")
                record[key] = int(value)
            else:
                record[key] = value

        for key in FLOW_PHASE_KEYS:
            record[key] = parse_float(row.get(key), f"{out_dir} {key}")

        for key in CONTINUOUS_FIELDS:
            record[key] = parse_float(row.get(key), f"{out_dir} {key}")

        if protocol_id == "nominal":
            for key in DISTURBANCE_KEYS:
                require(close(record[key], nominal[key]), f"{out_dir} nominal {key} mismatch")
            require(close(record["flow_phase_x"], 0.0), f"{out_dir} nominal flow_phase_x mismatch")
            require(close(record["flow_phase_y"], 0.0), f"{out_dir} nominal flow_phase_y mismatch")

        normalized_records.append(record)

    n_success = sum(row["success"] for row in normalized_records)
    n_found = sum(row["found"] for row in normalized_records)
    if "n_episodes" in summary:
        require(summary.get("n_episodes") == episodes, f"{out_dir} summary n_episodes mismatch")
    if "n_success" in summary:
        require(summary.get("n_success") == n_success, f"{out_dir} summary n_success mismatch")
    if "n_found" in summary:
        require(summary.get("n_found") == n_found, f"{out_dir} summary n_found mismatch")

    return {
        "overall_pass": True,
        "out_dir": rel(out_dir),
        "model_id": model.model_id,
        "protocol_id": protocol_id,
        "seed": seed,
        "episodes": episodes,
        "max_steps": max_steps,
        "n_success": n_success,
        "n_found": n_found,
        "records": normalized_records,
    }


def reusable_unit(
    out_dir: Path,
    model: ModelSpec,
    protocol_id: str,
    seed: int,
    episodes: int,
    max_steps: int,
) -> Optional[Dict[str, Any]]:
    if not out_dir.exists():
        return None
    try:
        audit = audit_evaluation_unit(out_dir, model, protocol_id, seed, episodes, max_steps)
    except Exception as exc:
        log(f"discard incomplete/invalid unit {rel(out_dir)}: {exc}")
        shutil.rmtree(out_dir, ignore_errors=True)
        return None
    log(f"reuse {rel(out_dir)}")
    return audit


def stream_evaluator(command: Sequence[str], log_path: Path, episodes: int) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            list(command),
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        require(process.stdout is not None, "subprocess stdout pipe was not created")
        for line in process.stdout:
            log_handle.write(line)
            if line.startswith("ep "):
                try:
                    current = int(line.split("/", 1)[0].split()[1])
                except Exception:
                    current = -1
                interval = max(1, int(episodes) // 10)
                if current in {1, episodes} or (current > 0 and current % interval == 0):
                    print(f"[evaluate_pse] {line.rstrip()}", flush=True)
            elif line.startswith("[EvalSummary]") or "Traceback" in line or "RuntimeError" in line:
                print(f"[evaluate_pse] {line.rstrip()}", flush=True)
        return int(process.wait())


def run_evaluation_unit(
    root: Path,
    model: ModelSpec,
    protocol_id: str,
    seed: int,
    stage: str,
) -> Dict[str, Any]:
    protocol, episodes, max_steps = protocol_spec(protocol_id, stage)
    out_dir = unit_dir(root, protocol_id, model.model_id, seed)
    reused = reusable_unit(out_dir, model, protocol_id, seed, episodes, max_steps)
    if reused is not None:
        return reused

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(PROJECT_ROOT / "evaluate_pse.py"),
        "--ablation_mode",
        AB_MODE,
        "--model_path",
        str(model.path),
        "--episodes",
        str(episodes),
        "--seed",
        str(seed),
        "--paired-episode-seeding",
        "--disturbance-protocol",
        protocol,
        "--max_steps",
        str(max_steps),
        "--profile",
        PROFILE,
        "--result_dir",
        str(out_dir),
    ]
    log(
        f"start protocol={protocol_id} model={model.model_id} seed={seed} "
        f"episodes={episodes} max_steps={max_steps}"
    )
    return_code = stream_evaluator(command, out_dir / "evaluate_stdout.log", episodes)
    require(return_code == 0, f"evaluate_pse failed with exit code {return_code}: {rel(out_dir)}")
    audit = audit_evaluation_unit(out_dir, model, protocol_id, seed, episodes, max_steps)
    log(
        f"done protocol={protocol_id} model={model.model_id} seed={seed} "
        f"success={audit['n_success']}/{episodes} found={audit['n_found']}/{episodes}"
    )
    return audit


def state_path(root: Path) -> Path:
    return root / "collection_state.json"


def update_collection_state(
    root: Path,
    stage: str,
    completed_units: Sequence[str],
    current_phase: str,
    errors: Optional[Sequence[str]] = None,
) -> None:
    expected_seeds = COMPARISON_SEEDS if stage == "formal" else [SMOKE_SEED]
    expected_unit_count = len(MODELS) * len(expected_seeds) * 2
    atomic_write_json(
        state_path(root),
        {
            "experiment_type": EXPERIMENT_TYPE,
            "script_version": SCRIPT_VERSION,
            "stage": stage,
            "current_phase": current_phase,
            "completed_units": sorted(set(completed_units)),
            "completed_unit_count": len(set(completed_units)),
            "expected_unit_count": expected_unit_count,
            "errors": list(errors or []),
            "training_performed": False,
            "optimizer_update_count": 0,
            "comparison_seeds": list(COMPARISON_SEEDS),
            "comparison_seeds_used": stage == "formal" and current_phase == "completed",
        },
    )


def collect_all_units(root: Path, stage: str) -> Tuple[Dict[Tuple[str, str, int], Dict[str, Any]], List[str]]:
    seeds = COMPARISON_SEEDS if stage == "formal" else [SMOKE_SEED]
    audits: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    completed: List[str] = []
    update_collection_state(root, stage, completed, "collecting")
    for protocol_id in ("full9d", "nominal"):
        for model in MODELS:
            for seed in seeds:
                audit = run_evaluation_unit(root, model, protocol_id, seed, stage)
                key = (protocol_id, model.model_id, seed)
                audits[key] = audit
                completed.append("/".join(map(str, key)))
                update_collection_state(root, stage, completed, "collecting")
    return audits, completed


def records_from_audits(
    audits: Mapping[Tuple[str, str, int], Mapping[str, Any]],
    protocol_id: str,
    model_id: str,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for (protocol, model, base_seed), audit in audits.items():
        if protocol != protocol_id or model != model_id:
            continue
        if seed is not None and base_seed != seed:
            continue
        rows.extend(dict(row) for row in audit["records"])
    rows.sort(key=lambda row: (int(row["base_seed"]), int(row["episode_index"])))
    return rows


def paired_disturbance_audit(
    audits: Mapping[Tuple[str, str, int], Mapping[str, Any]],
    stage: str,
) -> Dict[str, Any]:
    seeds = COMPARISON_SEEDS if stage == "formal" else [SMOKE_SEED]
    sections: Dict[str, Any] = {}
    total_mismatches = 0
    for protocol_id in ("full9d", "nominal"):
        mismatches: List[Dict[str, Any]] = []
        comparisons = 0
        for seed in seeds:
            baseline_rows = records_from_audits(audits, protocol_id, BASELINE_ID, seed)
            baseline_by_key = {
                (int(row["base_seed"]), int(row["episode_index"])): row
                for row in baseline_rows
            }
            for model in MODELS:
                if model.model_id == BASELINE_ID:
                    continue
                candidate_rows = records_from_audits(audits, protocol_id, model.model_id, seed)
                candidate_by_key = {
                    (int(row["base_seed"]), int(row["episode_index"])): row
                    for row in candidate_rows
                }
                require(
                    set(candidate_by_key) == set(baseline_by_key),
                    f"paired episode keys differ for {protocol_id}/{model.model_id}/seed{seed}",
                )
                for key in sorted(baseline_by_key):
                    candidate = candidate_by_key[key]
                    baseline = baseline_by_key[key]
                    comparisons += 1
                    for field in ("episode_seed", "disturbance_seed", *PAIRED_FIELDS):
                        left = candidate[field]
                        right = baseline[field]
                        equal = left == right if field in {"episode_seed", "disturbance_seed", "action_delay_steps"} else close(left, right)
                        if not equal:
                            mismatches.append(
                                {
                                    "protocol_id": protocol_id,
                                    "model_id": model.model_id,
                                    "base_seed": key[0],
                                    "episode_index": key[1],
                                    "field": field,
                                    "candidate": left,
                                    "baseline": right,
                                }
                            )
        total_mismatches += len(mismatches)
        sections[protocol_id] = {
            "overall_pass": not mismatches,
            "protocol": FULL9D_PROTOCOL if protocol_id == "full9d" else NOMINAL_PROTOCOL,
            "candidate_model_count": len(MODELS) - 1,
            "seed_count": len(seeds),
            "paired_episode_comparison_count": comparisons,
            "paired_fields": ["episode_seed", "disturbance_seed", *PAIRED_FIELDS],
            "mismatch_count": len(mismatches),
            "mismatch_examples": mismatches[:20],
            "all_episode_keys_match": True,
            "all_disturbance_vectors_match": not any(
                row.get("field") in DISTURBANCE_KEYS for row in mismatches
            ),
            "all_flow_phases_match": not any(
                row.get("field") in FLOW_PHASE_KEYS for row in mismatches
            ),
        }
        require(not mismatches, f"paired disturbance audit failed for {protocol_id}")
    return {
        "overall_pass": total_mismatches == 0,
        "stage": stage,
        "sections": sections,
        "total_mismatch_count": total_mismatches,
    }


def aggregate_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    episodes = len(rows)
    n_success = sum(int(row["success"]) for row in rows)
    n_found = sum(int(row["found"]) for row in rows)
    out: Dict[str, Any] = {
        "episodes": episodes,
        "n_success": n_success,
        "n_found": n_found,
        "n_not_found": episodes - n_found,
        "n_found_but_failed": n_found - n_success,
        "success_rate": rate(n_success, episodes),
        "found_rate": rate(n_found, episodes),
        "succ_if_found": rate(n_success, n_found),
        "not_found_rate": rate(episodes - n_found, episodes),
        "found_but_failed_rate": rate(n_found - n_success, episodes),
    }
    for field in CONTINUOUS_FIELDS:
        out[f"avg_{field}"] = mean(float(row[field]) for row in rows)
    return out


def build_metric_tables(
    audits: Mapping[Tuple[str, str, int], Mapping[str, Any]],
    stage: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    seeds = COMPARISON_SEEDS if stage == "formal" else [SMOKE_SEED]
    pooled_rows: List[Dict[str, Any]] = []
    by_seed_rows: List[Dict[str, Any]] = []
    for protocol_id in ("full9d", "nominal"):
        for model in MODELS:
            all_rows = records_from_audits(audits, protocol_id, model.model_id)
            pooled = aggregate_metrics(all_rows)
            pooled_rows.append(
                {
                    "protocol_id": protocol_id,
                    "model_id": model.model_id,
                    "role": model.role,
                    "training_seed": model.training_seed,
                    "selected_episode": model.selected_episode,
                    "model_sha256": model.expected_sha256,
                    **pooled,
                }
            )
            for seed in seeds:
                seed_metrics = aggregate_metrics(
                    records_from_audits(audits, protocol_id, model.model_id, seed)
                )
                by_seed_rows.append(
                    {
                        "protocol_id": protocol_id,
                        "model_id": model.model_id,
                        "role": model.role,
                        "base_seed": seed,
                        **seed_metrics,
                    }
                )
    return pooled_rows, by_seed_rows



def metric_value(rows: Sequence[Mapping[str, Any]], metric: str) -> Optional[float]:
    if metric in {"success", "found"}:
        return mean(float(row[metric]) for row in rows)
    if metric == "not_found":
        return mean(1.0 - float(row["found"]) for row in rows)
    if metric == "found_but_failed":
        return mean(float(row["found"]) - float(row["success"]) for row in rows)
    if metric == "succ_if_found":
        success = sum(int(row["success"]) for row in rows)
        found = sum(int(row["found"]) for row in rows)
        return rate(success, found)
    return mean(float(row[metric]) for row in rows)


def paired_rows_by_key(
    candidate_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
) -> Tuple[List[Tuple[int, int]], np.ndarray, np.ndarray]:
    candidate = {(int(row["base_seed"]), int(row["episode_index"])): row for row in candidate_rows}
    baseline = {(int(row["base_seed"]), int(row["episode_index"])): row for row in baseline_rows}
    require(set(candidate) == set(baseline), "candidate and baseline episode keys differ")
    keys = sorted(candidate)
    fields = (
        "success", "found", "not_found", "found_but_failed", "reward",
        "recovery_time", "safety_cost", "final_distance", "final_nav_distance",
        "action_smoothness", "completion_steps",
    )
    def value(row: Mapping[str, Any], field: str) -> float:
        if field == "not_found":
            return 1.0 - float(row["found"])
        if field == "found_but_failed":
            return float(row["found"]) - float(row["success"])
        return float(row[field])

    candidate_array = np.asarray(
        [[value(candidate[key], field) for field in fields] for key in keys],
        dtype=np.float64,
    )
    baseline_array = np.asarray(
        [[value(baseline[key], field) for field in fields] for key in keys],
        dtype=np.float64,
    )
    return keys, candidate_array, baseline_array

def bootstrap_paired_metrics(
    candidate_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    repetitions: int,
    rng_seed: int,
) -> Dict[str, Dict[str, Any]]:
    keys, candidate, baseline = paired_rows_by_key(candidate_rows, baseline_rows)
    field_names = (
        "success",
        "found",
        "not_found",
        "found_but_failed",
        "reward",
        "recovery_time",
        "safety_cost",
        "final_distance",
        "final_nav_distance",
        "action_smoothness",
        "completion_steps",
    )
    field_index = {name: index for index, name in enumerate(field_names)}
    by_seed: Dict[int, np.ndarray] = {}
    for seed in sorted({key[0] for key in keys}):
        by_seed[seed] = np.asarray(
            [index for index, key in enumerate(keys) if key[0] == seed],
            dtype=np.int64,
        )

    rng = np.random.default_rng(int(rng_seed))
    delta_samples = {
        metric: np.empty((int(repetitions),), dtype=np.float64)
        for metric in COMPARISON_METRICS
    }

    chunk_size = 500
    for start in range(0, int(repetitions), chunk_size):
        chunk = min(chunk_size, int(repetitions) - start)
        candidate_sum = np.zeros((chunk, len(field_names)), dtype=np.float64)
        baseline_sum = np.zeros((chunk, len(field_names)), dtype=np.float64)
        total_n = 0
        for indices in by_seed.values():
            n = int(indices.size)
            require(n > 0, "bootstrap seed group is empty")
            sampled_local = rng.integers(0, n, size=(chunk, n), endpoint=False)
            sampled_global = indices[sampled_local]
            candidate_sum += candidate[sampled_global].sum(axis=1)
            baseline_sum += baseline[sampled_global].sum(axis=1)
            total_n += n

        for metric in COMPARISON_METRICS:
            target = delta_samples[metric][start : start + chunk]
            if metric == "succ_if_found":
                c_success = candidate_sum[:, field_index["success"]]
                c_found = candidate_sum[:, field_index["found"]]
                b_success = baseline_sum[:, field_index["success"]]
                b_found = baseline_sum[:, field_index["found"]]
                valid = (c_found > 0.0) & (b_found > 0.0)
                target.fill(np.nan)
                target[valid] = c_success[valid] / c_found[valid] - b_success[valid] / b_found[valid]
            else:
                index = field_index[metric]
                target[:] = (candidate_sum[:, index] - baseline_sum[:, index]) / float(total_n)

    alpha = (1.0 - float(BOOTSTRAP_CI)) / 2.0
    result: Dict[str, Dict[str, Any]] = {}
    for metric in COMPARISON_METRICS:
        samples = delta_samples[metric]
        valid = samples[np.isfinite(samples)]
        point_candidate = metric_value(candidate_rows, metric)
        point_baseline = metric_value(baseline_rows, metric)
        point_delta = (
            None
            if point_candidate is None or point_baseline is None
            else float(point_candidate - point_baseline)
        )
        if valid.size:
            ci_low = float(np.quantile(valid, alpha))
            ci_high = float(np.quantile(valid, 1.0 - alpha))
            probability_positive = float(np.mean(valid > 0.0))
        else:
            ci_low = None
            ci_high = None
            probability_positive = None
        result[metric] = {
            "candidate_value": point_candidate,
            "baseline_value": point_baseline,
            "delta": point_delta,
            "ci95_low": ci_low,
            "ci95_high": ci_high,
            "probability_delta_positive": probability_positive,
            "bootstrap_repetitions_requested": int(repetitions),
            "bootstrap_repetitions_valid": int(valid.size),
            "bootstrap_method": "seed-stratified paired percentile bootstrap",
            "bootstrap_seed": int(rng_seed),
        }
    return result



def build_paired_comparisons(
    audits: Mapping[Tuple[str, str, int], Mapping[str, Any]],
    stage: str,
) -> List[Dict[str, Any]]:
    repetitions = BOOTSTRAP_REPETITIONS if stage == "formal" else 200
    rows: List[Dict[str, Any]] = []
    for protocol_index, protocol_id in enumerate(("full9d", "nominal")):
        for pair_index, (comparison_id, candidate_id, baseline_id) in enumerate(PAIR_SPECS):
            candidate_rows = records_from_audits(audits, protocol_id, candidate_id)
            baseline_rows = records_from_audits(audits, protocol_id, baseline_id)
            bootstrap = bootstrap_paired_metrics(
                candidate_rows,
                baseline_rows,
                repetitions=repetitions,
                rng_seed=BOOTSTRAP_SEED + 1000 * protocol_index + 100 * pair_index,
            )
            for metric in COMPARISON_METRICS:
                rows.append({
                    "comparison_id": comparison_id,
                    "protocol_id": protocol_id,
                    "candidate_model_id": candidate_id,
                    "candidate_role": MODEL_BY_ID[candidate_id].role,
                    "baseline_model_id": baseline_id,
                    "baseline_role": MODEL_BY_ID[baseline_id].role,
                    "metric": metric,
                    **bootstrap[metric],
                })
    return rows

def lookup_metric_row(
    pooled_rows: Sequence[Mapping[str, Any]],
    protocol_id: str,
    model_id: str,
) -> Mapping[str, Any]:
    matches = [
        row
        for row in pooled_rows
        if row.get("protocol_id") == protocol_id and row.get("model_id") == model_id
    ]
    require(len(matches) == 1, f"pooled metric row is missing: {protocol_id}/{model_id}")
    return matches[0]



def lookup_comparison(
    comparison_rows: Sequence[Mapping[str, Any]],
    protocol_id: str,
    candidate_id: str,
    baseline_id: str,
    metric: str,
) -> Mapping[str, Any]:
    matches = [
        row for row in comparison_rows
        if row.get("protocol_id") == protocol_id
        and row.get("candidate_model_id") == candidate_id
        and row.get("baseline_model_id") == baseline_id
        and row.get("metric") == metric
    ]
    require(len(matches) == 1, f"comparison row is missing: {protocol_id}/{candidate_id}/{baseline_id}/{metric}")
    return matches[0]


def nominal_guard(
    pooled_rows: Sequence[Mapping[str, Any]],
    candidate_id: str,
    baseline_id: str = BASELINE_ID,
    *,
    strict: bool,
) -> Dict[str, Any]:
    candidate = lookup_metric_row(pooled_rows, "nominal", candidate_id)
    baseline = lookup_metric_row(pooled_rows, "nominal", baseline_id)
    checks: Dict[str, Any] = {}
    for metric, field in (("success", "success_rate"), ("found", "found_rate"), ("succ_if_found", "succ_if_found")):
        candidate_value = candidate.get(field)
        baseline_value = baseline.get(field)
        available = candidate_value is not None and baseline_value is not None
        if strict:
            require(available, f"nominal {metric} is unavailable")
        delta = None if not available else float(candidate_value) - float(baseline_value)
        checks[metric] = {
            "candidate": candidate_value,
            "baseline": baseline_value,
            "delta": delta,
            "available": available,
            "margin": NOMINAL_GUARD_MARGIN,
            "pass": None if not available else delta >= -NOMINAL_GUARD_MARGIN,
        }
    return {
        "candidate_model_id": candidate_id,
        "baseline_model_id": baseline_id,
        "margin": NOMINAL_GUARD_MARGIN,
        "metrics": checks,
        "all_metrics_available": all(bool(item["available"]) for item in checks.values()),
        "pass": all(item["pass"] is True for item in checks.values()),
    }


def scientific_decision(
    pooled_rows: Sequence[Mapping[str, Any]],
    comparison_rows: Sequence[Mapping[str, Any]],
    stage: str,
) -> Dict[str, Any]:
    perf_id = "perf_rbe_seed2"
    old_id = "old_rbe_v1_seed2"
    uniform_id = "uniform_dr_baseline"
    clean_id = "clean_pse_baseline"

    perf_old_success = lookup_comparison(comparison_rows, "full9d", perf_id, old_id, "success")
    perf_old_sif = lookup_comparison(comparison_rows, "full9d", perf_id, old_id, "succ_if_found")
    perf_old_found_fail = lookup_comparison(comparison_rows, "full9d", perf_id, old_id, "found_but_failed")
    perf_uniform_success = lookup_comparison(comparison_rows, "full9d", perf_id, uniform_id, "success")
    perf_uniform_sif = lookup_comparison(comparison_rows, "full9d", perf_id, uniform_id, "succ_if_found")
    old_uniform_success = lookup_comparison(comparison_rows, "full9d", old_id, uniform_id, "success")
    old_uniform_sif = lookup_comparison(comparison_rows, "full9d", old_id, uniform_id, "succ_if_found")
    uniform_clean_success = lookup_comparison(comparison_rows, "full9d", uniform_id, clean_id, "success")
    uniform_clean_sif = lookup_comparison(comparison_rows, "full9d", uniform_id, clean_id, "succ_if_found")
    perf_nominal = nominal_guard(pooled_rows, perf_id, uniform_id, strict=stage == "formal")

    if stage == "formal":
        sif_restoration_pass = (
            perf_old_sif.get("delta") is not None
            and float(perf_old_sif["delta"]) > 0.0
            and perf_old_sif.get("ci95_low") is not None
            and float(perf_old_sif["ci95_low"]) > 0.0
        )
        success_noninferior_pass = (
            perf_old_success.get("delta") is not None
            and perf_old_success.get("ci95_low") is not None
            and float(perf_old_success["ci95_low"]) >= -SUCCESS_NONINFERIORITY_MARGIN
        )
        found_failure_reduction_supported = (
            perf_old_found_fail.get("delta") is not None
            and float(perf_old_found_fail["delta"]) < 0.0
        )
    else:
        sif_restoration_pass = False
        success_noninferior_pass = False
        found_failure_reduction_supported = False

    progression_confirmed = bool(
        stage == "formal"
        and sif_restoration_pass
        and success_noninferior_pass
        and perf_nominal["pass"]
    )
    return {
        "stage": stage,
        "decision_is_scientific_only_in_formal": True,
        "main_hypothesis": "performance stabilization improves full-9D SIF versus original RBE while retaining overall success and nominal performance",
        "models": {
            "clean": clean_id,
            "uniform": uniform_id,
            "original_rbe": old_id,
            "performance_stabilized_rbe": perf_id,
        },
        "primary_progression_test": {
            "perf_vs_old_success": dict(perf_old_success),
            "perf_vs_old_succ_if_found": dict(perf_old_sif),
            "perf_vs_old_found_but_failed": dict(perf_old_found_fail),
            "success_noninferiority_margin": SUCCESS_NONINFERIORITY_MARGIN,
            "success_noninferiority_pass": success_noninferior_pass,
            "sif_strict_improvement_pass": sif_restoration_pass,
            "found_but_failed_point_reduction_supported": found_failure_reduction_supported,
            "nominal_guard_vs_uniform": perf_nominal,
            "confirmed": progression_confirmed,
        },
        "context_comparisons": {
            "uniform_vs_clean_success": dict(uniform_clean_success),
            "uniform_vs_clean_succ_if_found": dict(uniform_clean_sif),
            "old_rbe_vs_uniform_success": dict(old_uniform_success),
            "old_rbe_vs_uniform_succ_if_found": dict(old_uniform_sif),
            "perf_rbe_vs_uniform_success": dict(perf_uniform_success),
            "perf_rbe_vs_uniform_succ_if_found": dict(perf_uniform_sif),
        },
        "overall_scientific_pass": progression_confirmed,
        "conclusion": (
            "performance_stabilization_progression_confirmed"
            if progression_confirmed
            else (
                "formal_comparison_completed_but_progression_gate_not_met"
                if stage == "formal"
                else "smoke_only_no_scientific_conclusion"
            )
        ),
    }

POOLED_FIELDS = (
    "protocol_id",
    "model_id",
    "role",
    "training_seed",
    "selected_episode",
    "model_sha256",
    "episodes",
    "n_success",
    "n_found",
    "n_not_found",
    "n_found_but_failed",
    "success_rate",
    "found_rate",
    "succ_if_found",
    "not_found_rate",
    "found_but_failed_rate",
    "avg_reward",
    "avg_recovery_time",
    "avg_safety_cost",
    "avg_final_distance",
    "avg_final_nav_distance",
    "avg_action_smoothness",
    "avg_completion_steps",
)
BY_SEED_FIELDS = (
    "protocol_id",
    "model_id",
    "role",
    "base_seed",
    "episodes",
    "n_success",
    "n_found",
    "n_not_found",
    "n_found_but_failed",
    "success_rate",
    "found_rate",
    "succ_if_found",
    "not_found_rate",
    "found_but_failed_rate",
    "avg_reward",
    "avg_recovery_time",
    "avg_safety_cost",
    "avg_final_distance",
    "avg_final_nav_distance",
    "avg_action_smoothness",
    "avg_completion_steps",
)
COMPARISON_FIELDS = (
    "comparison_id",
    "protocol_id",
    "candidate_model_id",
    "candidate_role",
    "baseline_model_id",
    "baseline_role",
    "metric",
    "candidate_value",
    "baseline_value",
    "delta",
    "ci95_low",
    "ci95_high",
    "probability_delta_positive",
    "bootstrap_repetitions_requested",
    "bootstrap_repetitions_valid",
    "bootstrap_method",
    "bootstrap_seed",
)



def markdown_summary(summary: Mapping[str, Any]) -> str:
    decision = summary.get("scientific_decision") or {}
    primary = decision.get("primary_progression_test") or {}
    lines = [
        "# Chapter-4 RBE Method Progression Comparison v1",
        "",
        f"- overall_pass: `{summary.get('overall_pass')}`",
        f"- test_stage: `{summary.get('test_stage')}`",
        f"- scientific_conclusion: `{decision.get('conclusion')}`",
        f"- progression confirmed: `{primary.get('confirmed')}`",
        f"- comparison seeds: `{summary.get('comparison_seeds')}`",
        f"- results used for reselection: `{summary.get('comparison_results_used_for_reselection')}`",
        "",
        "## Pooled Metrics",
        "",
        "| protocol | model | role | episodes | success | found | SIF | not found | found but failed | safety | final distance | smoothness |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.get("pooled_model_metrics", []):
        lines.append(
            "| {protocol_id} | {model_id} | {role} | {episodes} | {success_rate:.6f} | {found_rate:.6f} | {sif} | {not_found_rate:.6f} | {found_but_failed_rate:.6f} | {avg_safety_cost:.6f} | {avg_final_distance:.6f} | {avg_action_smoothness:.6f} |".format(
                protocol_id=row.get("protocol_id"),
                model_id=row.get("model_id"),
                role=row.get("role"),
                episodes=row.get("episodes"),
                success_rate=float(row.get("success_rate") or 0.0),
                found_rate=float(row.get("found_rate") or 0.0),
                sif="null" if row.get("succ_if_found") is None else f"{float(row['succ_if_found']):.6f}",
                not_found_rate=float(row.get("not_found_rate") or 0.0),
                found_but_failed_rate=float(row.get("found_but_failed_rate") or 0.0),
                avg_safety_cost=float(row.get("avg_safety_cost") or 0.0),
                avg_final_distance=float(row.get("avg_final_distance") or 0.0),
                avg_action_smoothness=float(row.get("avg_action_smoothness") or 0.0),
            )
        )
    lines.extend([
        "",
        "## Full-9D Primary Progression Test",
        "",
        "| comparison | metric | delta | CI95 low | CI95 high |",
        "| --- | --- | ---: | ---: | ---: |",
    ])
    for label, item in (
        ("perf - old", primary.get("perf_vs_old_success") or {}),
        ("perf - old", primary.get("perf_vs_old_succ_if_found") or {}),
        ("perf - old", primary.get("perf_vs_old_found_but_failed") or {}),
    ):
        lines.append(
            f"| {label} | {item.get('metric')} | {item.get('delta')} | {item.get('ci95_low')} | {item.get('ci95_high')} |"
        )
    lines.extend([
        "",
        "## Interpretation Contract",
        "",
        "- Clean/PSE is the no-DR policy.",
        "- Uniform DR is the traditional robust-training baseline.",
        "- Original RBE is the first Boundary-Centric RBE seed2 selected model (ep800).",
        "- Perf-RBE is the performance-stabilized seed2 selected model (ep2000).",
        "- Seeds 261--265 are diagnostic comparison seeds and must not be used for later checkpoint reselection.",
        "- This experiment evaluates frozen models only; training and optimizer updates are forbidden.",
    ])
    return "\n".join(lines) + "\n"


def finalize_outputs(
    root: Path,
    stage: str,
    audits: Mapping[Tuple[str, str, int], Mapping[str, Any]],
    completed_units: Sequence[str],
    preflight: Mapping[str, Any],
) -> Dict[str, Any]:
    paired_audit = paired_disturbance_audit(audits, stage)
    pooled_rows, by_seed_rows = build_metric_tables(audits, stage)
    comparison_rows = build_paired_comparisons(audits, stage)
    decision = scientific_decision(pooled_rows, comparison_rows, stage)

    write_csv(root / "method_metrics_full9d.csv", [row for row in pooled_rows if row["protocol_id"] == "full9d"], POOLED_FIELDS)
    write_csv(root / "method_metrics_nominal.csv", [row for row in pooled_rows if row["protocol_id"] == "nominal"], POOLED_FIELDS)
    write_csv(root / "metrics_by_seed.csv", by_seed_rows, BY_SEED_FIELDS)
    write_csv(root / "paired_method_comparisons.csv", comparison_rows, COMPARISON_FIELDS)
    atomic_write_json(root / "paired_disturbance_audit.json", paired_audit)
    atomic_write_json(root / "model_identity_audit.json", preflight["selected_model_identity_audit"])
    upstream_summary = resolve_upstream_file("independent_test_summary.json")
    upstream_source_lock = resolve_upstream_file("source_lock_manifest.json")
    atomic_write_json(root / "source_lock_manifest.json", {
        "overall_pass": True,
        "source_files": hash_tree(CURRENT_SOURCE_PATHS),
        "upstream_summary_sha256": sha256_file(upstream_summary),
        "upstream_source_lock_sha256": sha256_file(upstream_source_lock),
        "models": {model.model_id: {"path": rel(model.path), "sha256": sha256_file(model.path)} for model in MODELS},
    })

    source_after = hash_tree(CURRENT_SOURCE_PATHS)
    model_after = {model.model_id: sha256_file(model.path) for model in MODELS}
    protocol_lock = load_json(root / "protocol_lock_manifest.json")
    require(source_after == protocol_lock.get("source_files"), "source files changed during comparison")
    require(model_after == protocol_lock.get("model_sha256"), "model files changed during comparison")

    seeds = COMPARISON_SEEDS if stage == "formal" else [SMOKE_SEED]
    summary = {
        "overall_pass": True,
        "experiment_type": EXPERIMENT_TYPE,
        "script_version": SCRIPT_VERSION,
        "test_stage": stage,
        "stage_completed": True,
        "ablation_mode": AB_MODE,
        "profile": PROFILE,
        "comparison_seeds": list(seeds),
        "full9d_protocol": FULL9D_PROTOCOL,
        "nominal_protocol": NOMINAL_PROTOCOL,
        "full9d_episodes_per_seed": FORMAL_FULL9D_EPISODES_PER_SEED if stage == "formal" else SMOKE_EPISODES_PER_UNIT,
        "nominal_episodes_per_seed": FORMAL_NOMINAL_EPISODES_PER_SEED if stage == "formal" else SMOKE_EPISODES_PER_UNIT,
        "max_steps": FORMAL_MAX_STEPS if stage == "formal" else SMOKE_MAX_STEPS,
        "total_evaluation_units": len(audits),
        "total_episodes": sum(int(audit["episodes"]) for audit in audits.values()),
        "training_performed": False,
        "optimizer_update_count": 0,
        "selected_checkpoint_changed": False,
        "comparison_results_used_for_reselection": False,
        "paired_disturbance_audit": paired_audit,
        "pooled_model_metrics": pooled_rows,
        "metrics_by_seed": by_seed_rows,
        "paired_method_comparisons": comparison_rows,
        "scientific_decision": decision,
        "source_artifacts_unchanged": True,
        "models_unchanged": True,
        "upstream_contract": preflight["upstream_independent_test_contract"],
        "output_files": [
            "collection_state.json", "protocol_lock_manifest.json", "source_lock_manifest.json",
            "model_identity_audit.json", "paired_disturbance_audit.json",
            "method_metrics_full9d.csv", "method_metrics_nominal.csv", "metrics_by_seed.csv",
            "paired_method_comparisons.csv", "method_progression_summary.json", "method_progression_summary.md",
        ],
        "warnings": (
            [] if stage == "formal" and decision["overall_scientific_pass"]
            else (["Smoke results are pipeline-only and have no scientific meaning."] if stage == "smoke"
                  else ["The preregistered progression gate was not fully met."])
        ),
        "errors": [],
    }
    atomic_write_json(root / "method_progression_summary.json", summary)
    atomic_write_text(root / "method_progression_summary.md", markdown_summary(summary))
    update_collection_state(root, stage, completed_units, "completed")
    return summary

def run_preflight() -> int:
    audit = preflight_audit(load_models=True)
    PREFLIGHT_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_write_json(PREFLIGHT_ROOT / "preflight_summary.json", audit)
    print(json.dumps(json_safe(audit), indent=2, ensure_ascii=False, allow_nan=False))
    return 0


def run_smoke() -> int:
    preflight = preflight_audit(load_models=True)
    if SMOKE_ROOT.exists():
        shutil.rmtree(SMOKE_ROOT)
    SMOKE_ROOT.mkdir(parents=True, exist_ok=False)
    lock = build_protocol_lock("smoke", SMOKE_ROOT)
    write_or_validate_lock(SMOKE_ROOT, lock)
    audits, completed = collect_all_units(SMOKE_ROOT, "smoke")
    summary = finalize_outputs(SMOKE_ROOT, "smoke", audits, completed, preflight)
    require(summary.get("overall_pass") is True, "smoke summary failed")
    log(f"PASS smoke=true output={rel(SMOKE_ROOT)}")
    return 0


def smoke_passed() -> bool:
    summary_path = SMOKE_ROOT / "method_progression_summary.json"
    lock_path = SMOKE_ROOT / "protocol_lock_manifest.json"
    if not summary_path.is_file() or not lock_path.is_file():
        return False
    try:
        summary = load_json(summary_path)
        existing_lock = load_json(lock_path)
        expected_lock = build_protocol_lock("smoke", SMOKE_ROOT)
    except Exception:
        return False
    return (
        summary.get("overall_pass") is True
        and summary.get("script_version") == SCRIPT_VERSION
        and summary.get("test_stage") == "smoke"
        and summary.get("stage_completed") is True
        and summary.get("source_artifacts_unchanged") is True
        and existing_lock == expected_lock
    )


def atomic_commit_directory(staging: Path, final: Path) -> None:
    require(staging.is_dir(), f"staging directory is missing: {staging}")
    require(not final.exists(), f"final output already exists: {final}")
    os.replace(staging, final)


def run_formal() -> int:
    require(smoke_passed(), "formal refused: run and pass smoke first")
    require(not FORMAL_FINAL_ROOT.exists(), f"formal output already exists: {FORMAL_FINAL_ROOT}")
    preflight = preflight_audit(load_models=True)

    if not FORMAL_STAGING_ROOT.exists():
        FORMAL_STAGING_ROOT.mkdir(parents=True, exist_ok=False)
    lock = build_protocol_lock("formal", FORMAL_STAGING_ROOT)
    write_or_validate_lock(FORMAL_STAGING_ROOT, lock)

    audits, completed = collect_all_units(FORMAL_STAGING_ROOT, "formal")
    summary = finalize_outputs(FORMAL_STAGING_ROOT, "formal", audits, completed, preflight)
    require(summary.get("overall_pass") is True, "formal pipeline audit failed")
    atomic_commit_directory(FORMAL_STAGING_ROOT, FORMAL_FINAL_ROOT)
    log(
        f"PASS formal=true progression_conclusion="
        f"{summary['scientific_decision']['conclusion']} output={rel(FORMAL_FINAL_ROOT)}"
    )
    return 0


def inspect_status_root(root: Path, stage: str) -> Dict[str, Any]:
    seeds = COMPARISON_SEEDS if stage == "formal" else [SMOKE_SEED]
    expected = len(MODELS) * len(seeds) * 2
    completed = 0
    invalid = 0
    for protocol_id in ("full9d", "nominal"):
        protocol, episodes, max_steps = protocol_spec(protocol_id, stage)
        del protocol
        for model in MODELS:
            for seed in seeds:
                out_dir = unit_dir(root, protocol_id, model.model_id, seed)
                if not out_dir.exists():
                    continue
                try:
                    audit_evaluation_unit(out_dir, model, protocol_id, seed, episodes, max_steps)
                    completed += 1
                except Exception:
                    invalid += 1
    summary_path = root / "method_progression_summary.json"
    summary = load_json(summary_path) if summary_path.is_file() else None
    return {
        "root": rel(root),
        "exists": root.exists(),
        "stage": stage,
        "expected_unit_count": expected,
        "completed_valid_unit_count": completed,
        "invalid_or_incomplete_unit_count": invalid,
        "summary_exists": summary is not None,
        "stage_completed": None if summary is None else summary.get("stage_completed"),
        "overall_pass": None if summary is None else summary.get("overall_pass"),
        "progression_conclusion": (
            None
            if summary is None
            else (summary.get("scientific_decision") or {}).get("conclusion")
        ),
    }


def run_status() -> int:
    if FORMAL_FINAL_ROOT.exists():
        result = inspect_status_root(FORMAL_FINAL_ROOT, "formal")
        result["location_type"] = "formal_final"
    elif FORMAL_STAGING_ROOT.exists():
        result = inspect_status_root(FORMAL_STAGING_ROOT, "formal")
        result["location_type"] = "formal_incomplete"
    else:
        result = {
            "location_type": "not_started",
            "formal_final_root": rel(FORMAL_FINAL_ROOT),
            "formal_staging_root": rel(FORMAL_STAGING_ROOT),
            "smoke_passed": smoke_passed(),
        }
    print(json.dumps(json_safe(result), indent=2, ensure_ascii=False, allow_nan=False))
    return 0


def run_self_test() -> int:
    checks: List[Dict[str, Any]] = []

    def add(name: str, passed: bool, detail: Any = "") -> None:
        checks.append({"name": name, "pass": bool(passed), "detail": str(detail)})

    add("script_version_nonempty", bool(SCRIPT_VERSION), SCRIPT_VERSION)
    add("comparison_seed_count", COMPARISON_SEEDS == [261, 262, 263, 264, 265], COMPARISON_SEEDS)
    add("smoke_seed_disjoint", SMOKE_SEED not in COMPARISON_SEEDS, SMOKE_SEED)
    add("model_count", len(MODELS) == 4, len(MODELS))
    add("baseline_present", BASELINE_ID in MODEL_BY_ID, sorted(MODEL_BY_ID))
    add("comparison_pair_count", len(PAIR_SPECS) == 4, PAIR_SPECS)
    add("seed_formula", expected_episode_seed(246, 99) == 246 * 1_000_003 + 99)
    add("protocols_distinct", FULL9D_PROTOCOL != NOMINAL_PROTOCOL)
    add(
        "episode_budget",
        4 * 5 * FORMAL_FULL9D_EPISODES_PER_SEED
        + 4 * 5 * FORMAL_NOMINAL_EPISODES_PER_SEED
        == 3000,
    )

    synthetic_baseline = []
    synthetic_candidate = []
    for seed in (1, 2):
        for episode_index in range(5):
            common = {
                "base_seed": seed,
                "episode_index": episode_index,
                "episode_seed": expected_episode_seed(seed, episode_index),
                "disturbance_seed": expected_episode_seed(seed, episode_index),
                "found": 1,
                "reward": 0.0,
                "recovery_time": 10.0,
                "safety_cost": 1.0,
                "final_distance": 2.0,
                "final_nav_distance": 1.0,
                "action_smoothness": 0.1,
                "completion_steps": 20.0,
            }
            baseline = dict(common, success=0 if episode_index == 0 else 1)
            candidate = dict(common, success=1)
            synthetic_baseline.append(baseline)
            synthetic_candidate.append(candidate)
    agg_base = aggregate_metrics(synthetic_baseline)
    agg_candidate = aggregate_metrics(synthetic_candidate)
    add("aggregate_success", close(agg_base["success_rate"], 0.8), agg_base)
    add("aggregate_sif", close(agg_candidate["succ_if_found"], 1.0), agg_candidate)
    bootstrap = bootstrap_paired_metrics(
        synthetic_candidate,
        synthetic_baseline,
        repetitions=200,
        rng_seed=123,
    )
    add(
        "bootstrap_positive_success_delta",
        bootstrap["success"]["delta"] > 0.0
        and bootstrap["success"]["bootstrap_repetitions_valid"] == 200,
        bootstrap["success"],
    )
    add(
        "bootstrap_sif_available",
        bootstrap["succ_if_found"]["bootstrap_repetitions_valid"] == 200,
        bootstrap["succ_if_found"],
    )

    zero_found_a = [dict(row, found=0, success=0) for row in synthetic_candidate]
    zero_found_b = [dict(row, found=0, success=0) for row in synthetic_baseline]
    zero_bootstrap = bootstrap_paired_metrics(
        zero_found_a,
        zero_found_b,
        repetitions=20,
        rng_seed=456,
    )
    add(
        "zero_found_sif_is_unavailable_not_error",
        zero_bootstrap["succ_if_found"]["bootstrap_repetitions_valid"] == 0
        and zero_bootstrap["succ_if_found"]["ci95_low"] is None,
        zero_bootstrap["succ_if_found"],
    )

    overall = all(item["pass"] for item in checks)
    output = {
        "overall_pass": overall,
        "case_count": len(checks),
        "passed_case_count": sum(1 for item in checks if item["pass"]),
        "checks": checks,
        "training_performed": False,
        "optimizer_update_count": 0,
    }
    print(json.dumps(json_safe(output), indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if overall else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("version", "self-test", "preflight", "smoke", "formal", "status"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == "version":
        print(f"[RBEMethodProgression] script_version={SCRIPT_VERSION}")
        print(f"[RBEMethodProgression] script_file={Path(__file__).resolve()}")
        return 0
    if args.mode == "self-test":
        return run_self_test()
    if args.mode == "preflight":
        return run_preflight()
    if args.mode == "smoke":
        return run_smoke()
    if args.mode == "formal":
        return run_formal()
    if args.mode == "status":
        return run_status()
    raise IndependentTestError(f"unknown mode: {args.mode}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except IndependentTestError as exc:
        log(f"ERROR {type(exc).__name__}: {exc}")
        raise SystemExit(2)
