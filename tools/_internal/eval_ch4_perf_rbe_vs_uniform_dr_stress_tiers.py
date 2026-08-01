# -*- coding: utf-8 -*-
"""Paired disturbance-intensity stress test for frozen Perf-RBE and Uniform DR."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch

_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = _THIS_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.maddpg import MADDPG
from evaluate_pse import _apply_profile
from registry.ch4_artifact_layout import (
    get_evaluation_dir,
    get_selected_dir,
    get_smoke_dir,
    resolve_artifact_path,
)
from registry.ch4_stress_tiers import (
    FLOW_PHASE_KEYS,
    PROTOCOL_ID,
    STRESS_LEVELS,
    build_stress_case,
    direction_signature,
    tier_tag,
)
from registry.rbe_disturbance import DISTURBANCE_KEYS, nominal_disturbance
from tools._internal.ch4_rbe_ablation_eval_common import (
    ExperimentError,
    build_source_lock,
    model_load_audit,
    paired_bootstrap,
    pairing_audit,
    require,
    sha256,
    verify_source_lock,
)
from train import _build_train_env, get_ablation_config
from utils.rbe_metrics import EpisodeMetricTracker


SCRIPT_VERSION = "20260801-v1-radial-full9d-stress-tiers"
EXPERIMENT_ID = "ch4_perf_rbe_vs_uniform_dr_stress_tiers_v1"
STRESS_SEEDS = (381, 382, 383)
FORMAL_EPISODES_PER_SEED = 100
FORMAL_MAX_STEPS = 400
SMOKE_SEED = 938
SMOKE_EPISODES = 2
SMOKE_MAX_STEPS = 40
SMOKE_LEVELS = (0.0, 1.0, 1.5)
BOOTSTRAP_REPETITIONS = 10_000

MODELS = {
    "uniform_dr": resolve_artifact_path(
        get_selected_dir("uniform_dr", "ch4_uniform_dr_selected_full9d_v2")
        / "selected_by_uniform_dr_validation.pt"
    ),
    "perf_rbe": resolve_artifact_path(
        get_selected_dir("perf_rbe", "ch4_rbe_boundary_core_perf_selected_seed2_v1")
        / "selected_rbe_model.pt"
    ),
}

PREFLIGHT_ROOT = get_evaluation_dir(
    "stress_tests", f"{EXPERIMENT_ID}_preflight"
)
SMOKE_ROOT = get_smoke_dir("stress_tests", EXPERIMENT_ID)
FORMAL_ROOT = get_evaluation_dir("stress_tests", EXPERIMENT_ID)
STAGING_ROOT = FORMAL_ROOT.with_name(FORMAL_ROOT.name + ".incomplete")

DIRECTION_COLUMNS = tuple(f"direction_{key}" for key in DISTURBANCE_KEYS)
REQUESTED_COLUMNS = tuple(f"requested_{key}" for key in DISTURBANCE_KEYS)
ACTUAL_COLUMNS = tuple(f"actual_{key}" for key in DISTURBANCE_KEYS)
STEP_METRICS = {
    "residual_norm": "last_residual_norm",
    "prior_term_norm": "last_prior_term_norm",
    "residual_term_norm": "last_residual_term_norm",
    "final_acc_cmd_norm": "last_final_acc_cmd_norm",
    "residual_contribution_ratio_search": "last_residual_contribution_ratio_search",
    "residual_contribution_ratio_executor": "last_residual_contribution_ratio_executor",
    "residual_prior_cosine": "last_residual_prior_cosine_v2",
    "handoff_delay": "last_handoff_delay",
    "success_step_minus_found_step": "last_success_step_minus_found_step",
}

EPISODE_FIELDS = [
    "experiment_id", "script_version", "protocol", "model_id", "model_path",
    "model_sha256", "rho", "tier", "base_seed", "episode_index", "episode_seed", "disturbance_seed",
    *DIRECTION_COLUMNS, *REQUESTED_COLUMNS, *ACTUAL_COLUMNS,
    *DISTURBANCE_KEYS, "flow_phase_x", "flow_phase_y", "disturbance_apply_match",
    "found_flag", "executor_assigned_flag", "success_flag", "timeout_flag",
    "found_step", "assigned_step", "success_step", "completion_steps",
    "success_step_minus_found_step", "reward", "reward_mean", "recovery_time",
    "safety_cost", "final_distance", "final_nav_distance", "action_smoothness",
    "collision_count", "collision_episode", "near_target_at_end", "failure_mode",
    *STEP_METRICS.keys(),
]


def _set_global_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _read_json(path: Path) -> Dict[str, Any]:
    require(path.is_file(), f"missing JSON: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    require(isinstance(payload, dict), f"JSON root must be object: {path}")
    return payload


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = sorted({key for row in rows for key in row})
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _flag(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    text = str(value).strip().lower()
    return 1 if text in {"1", "1.0", "true", "yes", "y"} else 0


def _mean(rows: Sequence[Mapping[str, Any]], field: str) -> float | None:
    values = [_safe_float(row.get(field)) for row in rows]
    values = [value for value in values if value is not None]
    return None if not values else float(np.mean(values))


def _cvar_lower(rows: Sequence[Mapping[str, Any]], field: str, alpha: float = 0.10) -> float | None:
    values = [_safe_float(row.get(field)) for row in rows]
    values = sorted(value for value in values if value is not None)
    if not values:
        return None
    count = max(1, int(math.ceil(len(values) * float(alpha))))
    return float(np.mean(values[:count]))


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> Tuple[float | None, float | None]:
    if total <= 0:
        return None, None
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _finite_attr(env: Any, name: str) -> float | None:
    value = _safe_float(getattr(env, name, None))
    return value


def _audit_applied(requested: Mapping[str, Any], env: Any) -> Tuple[Dict[str, Any], bool]:
    actual = dict(env.get_current_disturbance())
    mismatches = []
    for key in DISTURBANCE_KEYS:
        req = requested[key]
        act = actual.get(key)
        if key == "action_delay_steps":
            if int(req) != int(act):
                mismatches.append(key)
        else:
            req_f, act_f = float(req), float(act)
            if not math.isclose(req_f, act_f, rel_tol=1e-12, abs_tol=1e-12):
                mismatches.append(key)
    for key, attr in (("flow_phase_x", "_flow_phase_x"), ("flow_phase_y", "_flow_phase_y")):
        if not math.isclose(float(requested[key]), float(getattr(env, attr)), rel_tol=1e-12, abs_tol=1e-12):
            mismatches.append(key)
    return actual, not mismatches


def _failure_mode(*, success: bool, found: bool, assigned: bool, collision: bool, near_target: bool) -> str:
    if success:
        return "success"
    if collision:
        return "safety_control_failure"
    if not found:
        return "search_failure"
    if not assigned:
        return "handoff_failure"
    if near_target:
        return "timeout_near_target"
    return "recovery_navigation_failure"


def _append_partial(path: Path, row: Mapping[str, Any], write_header: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if write_header else "a"
    with path.open(mode, encoding="utf-8-sig" if write_header else "utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EPISODE_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in EPISODE_FIELDS})
        handle.flush()
        os.fsync(handle.fileno())


def _validate_partial(
    rows: Sequence[Mapping[str, Any]], *, model_id: str, model_sha: str,
    rho: float, seed: int, episodes: int,
) -> None:
    require(len(rows) <= episodes, "partial row count exceeds requested episodes")
    for expected_index, row in enumerate(rows):
        require(row.get("model_id") == model_id, "partial model_id mismatch")
        require(row.get("model_sha256") == model_sha, "partial model hash mismatch")
        require(math.isclose(float(row.get("rho")), rho, abs_tol=1e-12), "partial rho mismatch")
        require(int(float(row.get("base_seed"))) == seed, "partial seed mismatch")
        require(int(float(row.get("episode_index"))) == expected_index, "partial episode sequence mismatch")
        require(_flag(row.get("disturbance_apply_match")) == 1, "partial disturbance audit failed")


def _evaluate_episode(
    *, env: Any, maddpg: MADDPG, model_id: str, model_path: Path, model_sha: str,
    rho: float, base_seed: int, episode_index: int, max_steps: int,
) -> Dict[str, Any]:
    case = build_stress_case(base_seed, episode_index, rho)
    episode_seed = int(case["episode_seed"])
    direction = dict(case["direction"])
    requested = dict(case["requested"])

    _set_global_seed(episode_seed)
    _apply_profile(env, "normal_comm")
    env.use_robust_disturbance = True
    env.set_next_disturbance(requested)
    obs = env.reset()
    actual, apply_match = _audit_applied(requested, env)
    require(apply_match, f"disturbance application mismatch: seed={base_seed} ep={episode_index} rho={rho}")

    tracker = EpisodeMetricTracker()
    tracker.reset(env, actual)
    reward_sum = 0.0
    reward_mean_values: List[float] = []
    found_step = assigned_step = success_step = None
    collision_count = 0
    step_values: Dict[str, List[float]] = {key: [] for key in STEP_METRICS}

    for step_i in range(1, int(max_steps) + 1):
        actions = maddpg.step(obs, explore=False)
        env_actions = torch.stack([action.squeeze(0) for action in actions], dim=0).to(
            device=env.device, dtype=torch.float32
        )
        obs, rewards, dones = env.step(env_actions)
        rewards_t = rewards if torch.is_tensor(rewards) else torch.as_tensor(rewards, dtype=torch.float32)
        reward_sum += float(rewards_t.sum().detach().cpu().item())
        reward_mean_values.append(float(rewards_t.mean().detach().cpu().item()))
        tracker.step(env, env_actions, rewards_t, dones)

        if found_step is None and bool(getattr(env, "task_found", False)):
            found_step = step_i
        if assigned_step is None and bool(getattr(env, "executor_target_assigned", False)):
            assigned_step = step_i
        if success_step is None and bool(getattr(env, "mission_complete", False)):
            success_step = step_i
        flags = getattr(env, "_collision_flags", None)
        if flags is not None:
            collision_count += int(flags.detach().to(dtype=torch.int32).sum().item())
        for metric, attr in STEP_METRICS.items():
            value = _finite_attr(env, attr)
            if value is not None:
                step_values[metric].append(value)
        if all(bool(done) for done in dones):
            break

    final = tracker.finalize(env)
    success = bool(final.get("success_flag", False))
    found = bool(final.get("found_flag", False))
    assigned = bool(getattr(env, "executor_target_assigned", False))
    completion_steps = int(final.get("completion_steps", getattr(env, "step_count", max_steps)))
    final_distance = float(final.get("final_distance", float("nan")))
    near_threshold = max(2.0, 2.0 * float(getattr(env, "executor_arrive_eps", 1.0)))
    near_target = math.isfinite(final_distance) and final_distance <= near_threshold
    timeout = (not success) and completion_steps >= int(max_steps)
    collision_episode = collision_count > 0

    row: Dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "script_version": SCRIPT_VERSION,
        "protocol": PROTOCOL_ID,
        "model_id": model_id,
        "model_path": str(model_path),
        "model_sha256": model_sha,
        "rho": float(rho),
        "tier": tier_tag(rho),
        "base_seed": int(base_seed),
        "episode_index": int(episode_index),
        "episode_seed": episode_seed,
        "disturbance_seed": episode_seed,
        "flow_phase_x": float(requested["flow_phase_x"]),
        "flow_phase_y": float(requested["flow_phase_y"]),
        "disturbance_apply_match": True,
        "found_flag": found,
        "executor_assigned_flag": assigned,
        "success_flag": success,
        "timeout_flag": timeout,
        "found_step": found_step,
        "assigned_step": assigned_step,
        "success_step": success_step,
        "completion_steps": completion_steps,
        "success_step_minus_found_step": (
            None if success_step is None or found_step is None else int(success_step - found_step)
        ),
        "reward": float(reward_sum),
        "reward_mean": float(np.mean(reward_mean_values)) if reward_mean_values else 0.0,
        "recovery_time": int(final.get("recovery_time", max_steps)),
        "safety_cost": float(final.get("safety_cost", 0.0)),
        "final_distance": final_distance,
        "final_nav_distance": float(final.get("final_nav_distance", float("nan"))),
        "action_smoothness": float(final.get("action_smoothness", 0.0)),
        "collision_count": int(collision_count),
        "collision_episode": collision_episode,
        "near_target_at_end": near_target,
        "failure_mode": _failure_mode(
            success=success, found=found, assigned=assigned,
            collision=collision_episode, near_target=near_target,
        ),
    }
    for key in DISTURBANCE_KEYS:
        row[f"direction_{key}"] = float(direction[key])
        row[f"requested_{key}"] = requested[key]
        row[f"actual_{key}"] = actual[key]
        row[key] = actual[key]
    for metric in STEP_METRICS:
        values = step_values[metric]
        row[metric] = None if not values else float(np.mean(values))
    return row


def _unit_paths(root: Path, model_id: str, rho: float, seed: int) -> Tuple[Path, Path, Path]:
    unit = root / tier_tag(rho) / model_id / f"seed{seed}"
    return unit, unit / "episode_metrics.csv", unit / "episode_metrics.partial.csv"


def _summarize_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    rows = list(rows)
    n = len(rows)
    n_success = sum(_flag(row.get("success_flag")) for row in rows)
    n_found = sum(_flag(row.get("found_flag")) for row in rows)
    n_assigned = sum(_flag(row.get("executor_assigned_flag")) for row in rows)
    n_timeout = sum(_flag(row.get("timeout_flag")) for row in rows)
    n_collision = sum(_flag(row.get("collision_episode")) for row in rows)
    success_ci = _wilson(n_success, n)
    found_ci = _wilson(n_found, n)
    assigned_ci = _wilson(n_assigned, n)
    summary = {
        "episodes": n,
        "n_success": n_success,
        "n_found": n_found,
        "n_assigned": n_assigned,
        "n_timeout": n_timeout,
        "n_collision_episode": n_collision,
        "success_rate": None if n == 0 else n_success / n,
        "success_ci95_low": success_ci[0],
        "success_ci95_high": success_ci[1],
        "found_rate": None if n == 0 else n_found / n,
        "found_ci95_low": found_ci[0],
        "found_ci95_high": found_ci[1],
        "assigned_rate": None if n == 0 else n_assigned / n,
        "assigned_ci95_low": assigned_ci[0],
        "assigned_ci95_high": assigned_ci[1],
        "succ_if_found": None if n_found == 0 else n_success / n_found,
        "succ_if_assigned": None if n_assigned == 0 else n_success / n_assigned,
        "timeout_rate": None if n == 0 else n_timeout / n,
        "collision_episode_rate": None if n == 0 else n_collision / n,
        "avg_reward": _mean(rows, "reward"),
        "reward_cvar10": _cvar_lower(rows, "reward", 0.10),
        "avg_completion_steps": _mean(rows, "completion_steps"),
        "avg_recovery_time": _mean(rows, "recovery_time"),
        "avg_safety_cost": _mean(rows, "safety_cost"),
        "avg_final_distance": _mean(rows, "final_distance"),
        "avg_final_nav_distance": _mean(rows, "final_nav_distance"),
        "avg_action_smoothness": _mean(rows, "action_smoothness"),
        "avg_residual_norm": _mean(rows, "residual_norm"),
        "avg_residual_contribution_ratio_search": _mean(rows, "residual_contribution_ratio_search"),
        "avg_residual_contribution_ratio_executor": _mean(rows, "residual_contribution_ratio_executor"),
    }
    return summary


def _unit_valid(
    unit: Path, *, model_id: str, model_path: Path, model_sha: str,
    rho: float, seed: int, episodes: int, max_steps: int,
) -> bool:
    try:
        summary = _read_json(unit / "evaluation_summary.json")
        rows = _read_csv(unit / "episode_metrics.csv")
        require(summary.get("overall_pass") is True, "unit summary not passed")
        require(summary.get("model_id") == model_id, "unit model mismatch")
        require(summary.get("model_sha256") == model_sha, "unit hash mismatch")
        require(Path(summary.get("model_path")).resolve() == model_path.resolve(), "unit path mismatch")
        require(math.isclose(float(summary.get("rho")), rho, abs_tol=1e-12), "unit rho mismatch")
        require(int(summary.get("seed")) == seed, "unit seed mismatch")
        require(int(summary.get("episodes")) == episodes, "unit episodes mismatch")
        require(int(summary.get("max_steps")) == max_steps, "unit max_steps mismatch")
        require(summary.get("protocol") == PROTOCOL_ID, "unit protocol mismatch")
        _validate_partial(rows, model_id=model_id, model_sha=model_sha, rho=rho, seed=seed, episodes=episodes)
        require(len(rows) == episodes, "unit row count mismatch")
        return True
    except Exception:
        return False


def _ensure_unit(
    *, root: Path, model_id: str, model_path: Path, rho: float, seed: int,
    episodes: int, max_steps: int,
) -> None:
    model_sha = sha256(model_path)
    unit, final_csv, partial_csv = _unit_paths(root, model_id, rho, seed)
    if _unit_valid(
        unit, model_id=model_id, model_path=model_path, model_sha=model_sha,
        rho=rho, seed=seed, episodes=episodes, max_steps=max_steps,
    ):
        print(f"[StressTier] reuse {unit}")
        return
    unit.mkdir(parents=True, exist_ok=True)

    resume_source = partial_csv if partial_csv.is_file() else final_csv
    rows = _read_csv(resume_source)
    try:
        _validate_partial(rows, model_id=model_id, model_sha=model_sha, rho=rho, seed=seed, episodes=episodes)
    except Exception:
        rows = []
        partial_csv.unlink(missing_ok=True)
        final_csv.unlink(missing_ok=True)
    if final_csv.is_file() and not partial_csv.is_file() and len(rows) < episodes:
        os.replace(final_csv, partial_csv)
    elif rows and not partial_csv.is_file() and len(rows) == episodes:
        _write_csv(partial_csv, rows, EPISODE_FIELDS)

    env_cfg, _ = get_ablation_config("ch4_rbe_full")
    env, _ = _build_train_env(torch.device("cpu"), int(max_steps), ablation_config=env_cfg)
    _apply_profile(env, "normal_comm")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    maddpg = MADDPG.init_from_save(str(model_path), device=device)
    maddpg.prep_rollouts(device=device)

    start = len(rows)
    for episode_index in range(start, episodes):
        row = _evaluate_episode(
            env=env, maddpg=maddpg, model_id=model_id, model_path=model_path,
            model_sha=model_sha, rho=rho, base_seed=seed,
            episode_index=episode_index, max_steps=max_steps,
        )
        _append_partial(partial_csv, row, write_header=(episode_index == 0 and start == 0))
        rows.append({key: row.get(key, "") for key in EPISODE_FIELDS})
        print(
            f"[StressTier] model={model_id} rho={rho:.2f} seed={seed} "
            f"ep={episode_index + 1}/{episodes} found={int(bool(row['found_flag']))} "
            f"assigned={int(bool(row['executor_assigned_flag']))} success={int(bool(row['success_flag']))}"
        )

    _validate_partial(rows, model_id=model_id, model_sha=model_sha, rho=rho, seed=seed, episodes=episodes)
    if partial_csv.is_file():
        os.replace(partial_csv, final_csv)
    summary = {
        "overall_pass": True,
        "experiment_id": EXPERIMENT_ID,
        "script_version": SCRIPT_VERSION,
        "protocol": PROTOCOL_ID,
        "model_id": model_id,
        "model_path": str(model_path),
        "model_sha256": model_sha,
        "rho": float(rho),
        "tier": tier_tag(rho),
        "seed": int(seed),
        "episodes": int(episodes),
        "max_steps": int(max_steps),
        "paired_episode_seed_formula": "base_seed * 1000003 + episode_index",
        "metrics": _summarize_rows(rows),
        "training_performed": False,
        "optimizer_update_count": 0,
        "episode_csv_sha256": sha256(final_csv),
    }
    _atomic_json(unit / "evaluation_summary.json", summary)
    require(_unit_valid(
        unit, model_id=model_id, model_path=model_path, model_sha=model_sha,
        rho=rho, seed=seed, episodes=episodes, max_steps=max_steps,
    ), f"completed unit failed audit: {unit}")


def _collect_rows(root: Path, model_id: str, rho: float, seeds: Iterable[int]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for seed in seeds:
        unit, final_csv, _ = _unit_paths(root, model_id, rho, seed)
        require(final_csv.is_file(), f"missing unit CSV: {unit}")
        rows.extend(_read_csv(final_csv))
    return rows


def _ray_audit(rows_by_level: Mapping[float, Sequence[Mapping[str, Any]]]) -> Dict[str, Any]:
    levels = sorted(rows_by_level)
    require(levels, "ray audit has no levels")
    baseline = {(int(float(r["base_seed"])), int(float(r["episode_index"]))): r for r in rows_by_level[levels[0]]}
    mismatches = []
    nominal = nominal_disturbance()
    for rho in levels:
        current = {(int(float(r["base_seed"])), int(float(r["episode_index"]))): r for r in rows_by_level[rho]}
        if set(current) != set(baseline):
            mismatches.append({"rho": rho, "reason": "episode_key_set"})
            continue
        for key in sorted(baseline):
            left, right = baseline[key], current[key]
            for column in (*DIRECTION_COLUMNS, "flow_phase_x", "flow_phase_y", "episode_seed"):
                a, b = float(left[column]), float(right[column])
                if not math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10):
                    mismatches.append({"rho": rho, "episode_key": key, "field": column, "a": a, "b": b})
                    break
            if rho == 0.0:
                for disturbance_key in DISTURBANCE_KEYS:
                    actual = float(right[f"actual_{disturbance_key}"])
                    expected = float(nominal[disturbance_key])
                    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
                        mismatches.append({"rho": rho, "episode_key": key, "field": disturbance_key})
                        break
            if len(mismatches) >= 20:
                break
    return {
        "overall_pass": not mismatches,
        "level_count": len(levels),
        "paired_ray_count": len(baseline),
        "mismatch_count": len(mismatches),
        "mismatch_examples": mismatches[:20],
    }


def _failure_rows(rows: Sequence[Mapping[str, Any]], model_id: str, rho: float) -> List[Dict[str, Any]]:
    modes = (
        "success", "search_failure", "handoff_failure", "recovery_navigation_failure",
        "timeout_near_target", "safety_control_failure",
    )
    total = len(rows)
    output = []
    for mode in modes:
        count = sum(1 for row in rows if row.get("failure_mode") == mode)
        output.append({
            "model_id": model_id,
            "rho": rho,
            "tier": tier_tag(rho),
            "failure_mode": mode,
            "count": count,
            "rate": None if total == 0 else count / total,
        })
    return output


def _trapezoid(levels: Sequence[float], values: Sequence[float]) -> float:
    return float(sum(
        0.5 * (values[i] + values[i - 1]) * (levels[i] - levels[i - 1])
        for i in range(1, len(levels))
    ))


def _contiguous_threshold(level_metrics: Sequence[Mapping[str, Any]], target: float) -> float | None:
    reached = None
    for item in sorted(level_metrics, key=lambda x: float(x["rho"])):
        low = _safe_float(item.get("success_ci95_low"))
        if low is None or low < target:
            break
        reached = float(item["rho"])
    return reached


def _first_below(level_metrics: Sequence[Mapping[str, Any]], target: float) -> float | None:
    for item in sorted(level_metrics, key=lambda x: float(x["rho"])):
        rate = _safe_float(item.get("success_rate"))
        if rate is not None and rate < target:
            return float(item["rho"])
    return None


def _markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Perf-RBE vs Uniform DR disturbance-tier stress test",
        "",
        f"- Protocol: `{summary['protocol']}`",
        f"- Seeds: {summary['stress_seeds']}",
        f"- Episodes per model/tier: {summary['episodes_per_model_tier']}",
        f"- Total formal episodes: {summary['total_episodes']}",
        "",
        "## Success-rate degradation",
        "",
        "| Model | rho | Success | 95% CI | Found | SIF | Timeout |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary["level_metrics"]:
        ci = f"[{item['success_ci95_low']:.3f}, {item['success_ci95_high']:.3f}]"
        sif = "NA" if item["succ_if_found"] is None else f"{item['succ_if_found']:.3f}"
        lines.append(
            f"| {item['model_id']} | {item['rho']:.2f} | {item['success_rate']:.3f} | "
            f"{ci} | {item['found_rate']:.3f} | {sif} | {item['timeout_rate']:.3f} |"
        )
    lines.extend(["", "## Robustness summary", ""])
    for model_id, item in summary["robustness_summary"].items():
        lines.append(
            f"- **{model_id}**: AURC={item['aurc']:.4f}, "
            f"rho80={item['rho80_contiguous']}, rho50={item['rho50_contiguous']}, "
            f"first rho with SR<0.20={item['first_rho_below_20pct']}"
        )
    decision = summary["scientific_decision"]
    lines.extend([
        "", "## Pre-registered decision checks", "",
        f"- Nominal non-inferiority (margin -0.03): {decision['nominal_noninferiority_pass']}",
        f"- High-stress average success advantage: {decision['high_stress_avg_success_delta']}",
        f"- AURC delta (Perf-RBE - Uniform DR): {decision['aurc_delta']}",
        f"- rho80 expansion: {decision['rho80_expansion']}",
        "",
        "Detailed episode, failure-mode, and paired-bootstrap results are stored in the CSV files beside this summary.",
    ])
    return "\n".join(lines) + "\n"


def _aggregate(root: Path, seeds: Tuple[int, ...], levels: Tuple[float, ...], episodes: int) -> Dict[str, Any]:
    all_rows: List[Dict[str, str]] = []
    level_metrics: List[Dict[str, Any]] = []
    failure_rows: List[Dict[str, Any]] = []
    paired_rows: List[Dict[str, Any]] = []
    pairing_by_level: Dict[str, Any] = {}
    rows_by_model_level: Dict[Tuple[str, float], List[Dict[str, str]]] = {}

    for rho in levels:
        rows_by_model = {}
        for model_id in MODELS:
            rows = _collect_rows(root, model_id, rho, seeds)
            rows_by_model[model_id] = rows
            rows_by_model_level[(model_id, rho)] = rows
            all_rows.extend(rows)
            metric = _summarize_rows(rows)
            metric.update({"model_id": model_id, "rho": rho, "tier": tier_tag(rho)})
            level_metrics.append(metric)
            failure_rows.extend(_failure_rows(rows, model_id, rho))
        audit = pairing_audit(rows_by_model)
        require(audit["overall_pass"], f"paired model audit failed at rho={rho}")
        pairing_by_level[tier_tag(rho)] = audit

        for metric_name in (
            "success", "found", "succ_if_found", "recovery_time",
            "safety_cost", "completion_steps",
        ):
            result = paired_bootstrap(
                rows_by_model["perf_rbe"], rows_by_model["uniform_dr"], metric_name,
                repetitions=BOOTSTRAP_REPETITIONS,
                seed=20260801 + len(paired_rows),
            )
            result.update({
                "comparison_id": "perf_rbe_minus_uniform_dr",
                "rho": rho,
                "tier": tier_tag(rho),
                "metric": metric_name,
            })
            paired_rows.append(result)

    ray_audits = {}
    for model_id in MODELS:
        audit = _ray_audit({rho: rows_by_model_level[(model_id, rho)] for rho in levels})
        require(audit["overall_pass"], f"cross-tier ray audit failed: {model_id}")
        ray_audits[model_id] = audit

    robustness = {}
    for model_id in MODELS:
        items = sorted(
            [item for item in level_metrics if item["model_id"] == model_id],
            key=lambda item: item["rho"],
        )
        xs = [float(item["rho"]) for item in items]
        ys = [float(item["success_rate"]) for item in items]
        robustness[model_id] = {
            "aurc": _trapezoid(xs, ys),
            "rho80_contiguous": _contiguous_threshold(items, 0.80),
            "rho50_contiguous": _contiguous_threshold(items, 0.50),
            "first_rho_below_20pct": _first_below(items, 0.20),
        }

    lookup = {(item["model_id"], float(item["rho"])): item for item in level_metrics}
    nominal_delta = lookup[("perf_rbe", 0.0)]["success_rate"] - lookup[("uniform_dr", 0.0)]["success_rate"]
    high_levels = [rho for rho in levels if rho >= 0.75]
    high_delta = float(np.mean([
        lookup[("perf_rbe", rho)]["success_rate"] - lookup[("uniform_dr", rho)]["success_rate"]
        for rho in high_levels
    ]))
    perf_rho80 = robustness["perf_rbe"]["rho80_contiguous"]
    uniform_rho80 = robustness["uniform_dr"]["rho80_contiguous"]
    rho80_expansion = (
        None if perf_rho80 is None or uniform_rho80 is None else perf_rho80 - uniform_rho80
    )
    decision = {
        "nominal_success_delta": nominal_delta,
        "nominal_noninferiority_margin": -0.03,
        "nominal_noninferiority_pass": nominal_delta >= -0.03,
        "high_stress_definition": "rho >= 0.75",
        "high_stress_avg_success_delta": high_delta,
        "high_stress_advantage_at_least_5pp": high_delta >= 0.05,
        "aurc_delta": robustness["perf_rbe"]["aurc"] - robustness["uniform_dr"]["aurc"],
        "rho80_expansion": rho80_expansion,
        "rho80_expansion_positive": rho80_expansion is not None and rho80_expansion > 0.0,
    }

    _write_csv(root / "stress_tier_episode_metrics.csv", all_rows, EPISODE_FIELDS)
    _write_csv(root / "stress_tier_summary_by_level.csv", level_metrics)
    _write_csv(root / "stress_tier_failure_modes.csv", failure_rows)
    _write_csv(root / "stress_tier_paired_comparison.csv", paired_rows)
    _atomic_json(root / "paired_disturbance_audit.json", {
        "overall_pass": True,
        "pairing_by_level": pairing_by_level,
        "cross_tier_ray_audits": ray_audits,
    })

    summary = {
        "overall_pass": True,
        "stage_completed": True,
        "experiment_id": EXPERIMENT_ID,
        "script_version": SCRIPT_VERSION,
        "protocol": PROTOCOL_ID,
        "models": {model_id: {"path": str(path), "sha256": sha256(path)} for model_id, path in MODELS.items()},
        "stress_levels": list(levels),
        "stress_seeds": list(seeds),
        "episodes_per_seed": episodes,
        "episodes_per_model_tier": episodes * len(seeds),
        "max_steps": FORMAL_MAX_STEPS,
        "total_evaluation_units": len(MODELS) * len(levels) * len(seeds),
        "total_episodes": len(MODELS) * len(levels) * len(seeds) * episodes,
        "level_metrics": level_metrics,
        "robustness_summary": robustness,
        "scientific_decision": decision,
        "paired_disturbance_audit": {
            "overall_pass": True,
            "pairing_by_level": pairing_by_level,
            "cross_tier_ray_audits": ray_audits,
        },
        "training_performed": False,
        "optimizer_update_count": 0,
        "test_results_used_for_reselection": False,
        "source_artifacts_unchanged": True,
    }
    _atomic_json(root / "stress_tier_summary.json", summary)
    (root / "stress_tier_summary.md").write_text(_markdown(summary), encoding="utf-8")
    return summary


def run_self_test() -> int:
    from registry.ch4_stress_tiers import disturbance_from_direction, sample_stress_direction

    checks: Dict[str, bool] = {}
    d1 = sample_stress_direction(381, 0)
    d2 = sample_stress_direction(381, 0)
    checks["deterministic_direction"] = direction_signature(d1) == direction_signature(d2)
    checks["all_dimensions_present"] = set(d1) == set(DISTURBANCE_KEYS)
    checks["linf_normalized"] = math.isclose(max(abs(v) for v in d1.values()), 1.0, abs_tol=1e-12)
    x0 = disturbance_from_direction(d1, 0.0)
    nominal = nominal_disturbance()
    checks["rho0_nominal"] = all(math.isclose(float(x0[k]), float(nominal[k]), abs_tol=1e-12) for k in DISTURBANCE_KEYS)
    x1 = disturbance_from_direction(d1, 1.0)
    checks["delay_integer"] = isinstance(x1["action_delay_steps"], int)
    checks["adverse_one_sided"] = (
        x1["flow_gain"] >= nominal["flow_gain"]
        and x1["a_max_scale"] <= nominal["a_max_scale"]
        and x1["v_max_scale"] <= nominal["v_max_scale"]
        and x1["actuator_lag"] >= nominal["actuator_lag"]
        and x1["action_noise_std"] >= nominal["action_noise_std"]
    )
    result = {
        "overall_pass": all(checks.values()),
        "script_version": SCRIPT_VERSION,
        "checks": checks,
    }
    print(json.dumps(result, indent=2))
    return 0 if result["overall_pass"] else 2


def run_preflight() -> int:
    model_audits = {}
    for model_id, path in MODELS.items():
        require(path.is_file(), f"frozen model missing: {model_id}: {path}")
        audit = model_load_audit(path)
        require(audit["overall_pass"], f"model load audit failed: {model_id}")
        model_audits[model_id] = audit
    source_lock = build_source_lock((
        _THIS_FILE,
        PROJECT_ROOT / "registry/ch4_stress_tiers.py",
        PROJECT_ROOT / "env.py",
        PROJECT_ROOT / "train.py",
        PROJECT_ROOT / "algorithms/maddpg.py",
        PROJECT_ROOT / "registry/rbe_disturbance.py",
        PROJECT_ROOT / "utils/rbe_metrics.py",
    ))
    result = {
        "overall_pass": True,
        "experiment_id": EXPERIMENT_ID,
        "script_version": SCRIPT_VERSION,
        "protocol": PROTOCOL_ID,
        "models": {
            model_id: {"path": str(path), "sha256": sha256(path), "load_audit": model_audits[model_id]}
            for model_id, path in MODELS.items()
        },
        "stress_levels": list(STRESS_LEVELS),
        "stress_seeds": list(STRESS_SEEDS),
        "episodes_per_seed": FORMAL_EPISODES_PER_SEED,
        "max_steps": FORMAL_MAX_STEPS,
        "expected_total_episodes": len(MODELS) * len(STRESS_LEVELS) * len(STRESS_SEEDS) * FORMAL_EPISODES_PER_SEED,
        "source_lock": source_lock,
        "training_performed": False,
        "optimizer_update_count": 0,
    }
    _atomic_json(PREFLIGHT_ROOT / "preflight_summary.json", result)
    print(json.dumps({"overall_pass": True, "expected_total_episodes": result["expected_total_episodes"]}, indent=2))
    return 0


def _require_preflight() -> Dict[str, Any]:
    preflight = _read_json(PREFLIGHT_ROOT / "preflight_summary.json")
    require(preflight.get("overall_pass") is True, "preflight has not passed")
    verify_source_lock(preflight["source_lock"])
    for model_id, path in MODELS.items():
        require(preflight["models"][model_id]["sha256"] == sha256(path), f"model changed after preflight: {model_id}")
    return preflight


def run_smoke() -> int:
    _require_preflight()
    if SMOKE_ROOT.exists():
        shutil.rmtree(SMOKE_ROOT)
    for rho in SMOKE_LEVELS:
        for model_id, path in MODELS.items():
            _ensure_unit(
                root=SMOKE_ROOT, model_id=model_id, model_path=path, rho=rho,
                seed=SMOKE_SEED, episodes=SMOKE_EPISODES, max_steps=SMOKE_MAX_STEPS,
            )
        rows_by_model = {
            model_id: _collect_rows(SMOKE_ROOT, model_id, rho, (SMOKE_SEED,))
            for model_id in MODELS
        }
        audit = pairing_audit(rows_by_model)
        require(audit["overall_pass"], f"smoke pairing failed: rho={rho}")
    ray_audit = _ray_audit({
        rho: _collect_rows(SMOKE_ROOT, "perf_rbe", rho, (SMOKE_SEED,))
        for rho in SMOKE_LEVELS
    })
    require(ray_audit["overall_pass"], "smoke ray audit failed")
    result = {
        "overall_pass": True,
        "stage": "smoke",
        "script_version": SCRIPT_VERSION,
        "levels": list(SMOKE_LEVELS),
        "seed": SMOKE_SEED,
        "episodes_per_unit": SMOKE_EPISODES,
        "max_steps": SMOKE_MAX_STEPS,
        "ray_audit": ray_audit,
    }
    _atomic_json(SMOKE_ROOT / "smoke_summary.json", result)
    print(json.dumps(result, indent=2))
    return 0


def run_formal() -> int:
    preflight = _require_preflight()
    smoke = _read_json(SMOKE_ROOT / "smoke_summary.json")
    require(smoke.get("overall_pass") is True, "formal refused: smoke has not passed")
    if FORMAL_ROOT.exists():
        summary = _read_json(FORMAL_ROOT / "stress_tier_summary.json")
        require(summary.get("overall_pass") is True, "existing formal result is incomplete")
        print(json.dumps({"overall_pass": True, "reused": True, "output": str(FORMAL_ROOT)}, indent=2))
        return 0
    STAGING_ROOT.mkdir(parents=True, exist_ok=True)

    for rho in STRESS_LEVELS:
        for model_id, path in MODELS.items():
            for seed in STRESS_SEEDS:
                _ensure_unit(
                    root=STAGING_ROOT, model_id=model_id, model_path=path, rho=rho,
                    seed=seed, episodes=FORMAL_EPISODES_PER_SEED,
                    max_steps=FORMAL_MAX_STEPS,
                )

    summary = _aggregate(STAGING_ROOT, STRESS_SEEDS, STRESS_LEVELS, FORMAL_EPISODES_PER_SEED)
    verify_source_lock(preflight["source_lock"])
    require(summary["total_episodes"] == 4200, "formal episode-count contract changed")
    os.replace(STAGING_ROOT, FORMAL_ROOT)
    print(json.dumps({
        "overall_pass": True,
        "output": str(FORMAL_ROOT),
        "total_episodes": summary["total_episodes"],
        "scientific_decision": summary["scientific_decision"],
    }, indent=2))
    return 0


def run_status() -> int:
    completed_units = len(list(STAGING_ROOT.rglob("evaluation_summary.json"))) if STAGING_ROOT.exists() else 0
    partial_files = list(STAGING_ROOT.rglob("episode_metrics.partial.csv")) if STAGING_ROOT.exists() else []
    partial_rows = sum(max(0, len(_read_csv(path))) for path in partial_files)
    result = {
        "experiment_id": EXPERIMENT_ID,
        "formal_exists": FORMAL_ROOT.exists(),
        "incomplete_exists": STAGING_ROOT.exists(),
        "preflight_passed": (PREFLIGHT_ROOT / "preflight_summary.json").is_file(),
        "smoke_passed": (SMOKE_ROOT / "smoke_summary.json").is_file(),
        "completed_evaluation_units": completed_units,
        "expected_evaluation_units": len(MODELS) * len(STRESS_LEVELS) * len(STRESS_SEEDS),
        "partial_unit_count": len(partial_files),
        "partial_episode_rows": partial_rows,
        "formal_output": str(FORMAL_ROOT),
    }
    print(json.dumps(result, indent=2))
    return 0


def run_all() -> int:
    for function in (run_self_test, run_preflight, run_smoke, run_formal):
        code = int(function())
        if code != 0:
            return code
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("version", "self-test", "preflight", "smoke", "formal", "status", "run"),
        nargs="?",
        default="status",
    )
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
    if args.mode == "run":
        return run_all()
    return run_status()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ExperimentError as exc:
        print(f"[StressTier] ERROR {type(exc).__name__}: {exc}")
        raise SystemExit(2)
