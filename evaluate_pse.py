# -*- coding: utf-8 -*-
import argparse
import csv
import json
import math
import os
import random
from datetime import datetime

import numpy as np
import torch

from algorithms.maddpg import MADDPG
from registry import experiment_registry
from registry.rbe_disturbance import (
    DEFAULT_DISTURBANCE_BOUNDS,
    DISTURBANCE_KEYS,
    nominal_disturbance,
    sample_uniform_disturbance,
)
from train import CHAPTER_CONFIGS, _build_train_env, get_ablation_config
from utils.rbe_metrics import EpisodeMetricTracker


PROFILES = ("normal_comm", "weak_comm", "severe_comm", "robust_full", "basic")
NORMAL_COMM_PROFILE = {
    "comm_scenario_id": 0,
    "comm_loss_prob": 0.08,
    "upper_comm_base_loss": 0.05,
    "comm_attenuation": 0.18,
    "upper_comm_attenuation": 0.18,
    "comm_max_range": 15.0,
    "upper_comm_max_range": 18.0,
    "lower_extra_delay_steps": 1,
    "upper_extra_delay_steps": 0,
    "lower_max_delay_steps": 16,
    "lower_rate_bps": 1000.0,
    "upper_rate_bps": 1000.0,
    "payload_loss_scale": 0.025,
    "critical_message_loss_prob": 0.0,
    "reconnect_island_timeout_steps": 8,
}
PSE_STEP_METRICS = {
    "belief_entropy": "last_belief_entropy",
    "standby_to_target_dist": "last_standby_to_target_dist",
    "exec_response_cost": "last_exec_response_cost",
    "search_score_mean": "last_search_score_mean",
    "pse_claim_overlap": "last_pse_claim_overlap",
    "success_step_minus_found_step": "last_success_step_minus_found_step",
    "residual_contribution_ratio_search": "last_residual_contribution_ratio_search",
    "residual_contribution_ratio_executor": "last_residual_contribution_ratio_executor",
    "handoff_delay": "last_handoff_delay",
    "pse_exec_cost_weight_effective": "last_pse_exec_cost_weight_effective",
    "pse_exec_cost_schedule_factor": "last_pse_exec_cost_schedule_factor",
    "pse_lazy_standby_active": "last_pse_lazy_standby_active",
    "pse_standby_update_allowed": "last_pse_standby_update_allowed",
    "pse_standby_update_skipped_by_lazy_gate": "last_pse_standby_update_skipped_by_lazy_gate",
}

PAIRED_EPISODE_SEED_MODE = "indexed_common_random_numbers"
PAIRED_EPISODE_SEED_FORMULA = "base_seed * 1000003 + episode_index"
DISTURBANCE_RNG_MODE = "numpy.default_rng(episode_seed)"

DISTURBANCE_PROTOCOL_AUTO = "auto"
DISTURBANCE_PROTOCOL_NOMINAL = "nominal_v1"
DISTURBANCE_PROTOCOL_UNIFORM_9D = "uniform_9d_registry_v1"

FLOW_PHASE_KEYS = (
    "flow_phase_x",
    "flow_phase_y",
)

SUPPORTED_DISTURBANCE_PROTOCOLS = (
    DISTURBANCE_PROTOCOL_AUTO,
    DISTURBANCE_PROTOCOL_NOMINAL,
    DISTURBANCE_PROTOCOL_UNIFORM_9D,
)

_UNIFORM_9D_AUTO_MODES = {
    "ch4_uniform_dr",
    "ch4_reb_only",
    "ch4_rbe_full",
}


def _set_global_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_disturbance_protocol(ablation_mode, requested_protocol):
    requested = str(requested_protocol)
    if requested not in SUPPORTED_DISTURBANCE_PROTOCOLS:
        raise ValueError(f"Unsupported disturbance protocol: {requested}")
    if requested != DISTURBANCE_PROTOCOL_AUTO:
        return requested
    if str(ablation_mode) in _UNIFORM_9D_AUTO_MODES:
        return DISTURBANCE_PROTOCOL_UNIFORM_9D
    return DISTURBANCE_PROTOCOL_NOMINAL


def disturbance_seed_for_episode(base_seed, episode_index):
    if base_seed is None:
        raise ValueError("A base seed is required for episode-local disturbance sampling")
    return int(base_seed) * 1_000_003 + int(episode_index)


def sample_episode_disturbance(resolved_protocol, base_seed, episode_index):
    if resolved_protocol == DISTURBANCE_PROTOCOL_NOMINAL:
        requested = dict(nominal_disturbance())
        requested.update({key: 0.0 for key in FLOW_PHASE_KEYS})
        disturbance_seed = (
            disturbance_seed_for_episode(base_seed, episode_index)
            if base_seed is not None
            else None
        )
        return requested, disturbance_seed, DISTURBANCE_RNG_MODE
    if resolved_protocol != DISTURBANCE_PROTOCOL_UNIFORM_9D:
        raise ValueError(f"Unsupported resolved disturbance protocol: {resolved_protocol}")
    disturbance_seed = disturbance_seed_for_episode(base_seed, episode_index)
    disturbance_rng = np.random.default_rng(disturbance_seed)
    requested = sample_uniform_disturbance(
        rng=disturbance_rng,
        bounds=DEFAULT_DISTURBANCE_BOUNDS,
        include_flow_phase=True,
    )
    return requested, disturbance_seed, DISTURBANCE_RNG_MODE


def apply_disturbance_before_reset(env, resolved_protocol, requested_xi_full):
    if not hasattr(env, "set_next_disturbance"):
        raise RuntimeError("Environment does not implement set_next_disturbance()")
    env.use_robust_disturbance = resolved_protocol == DISTURBANCE_PROTOCOL_UNIFORM_9D
    env.set_next_disturbance(requested_xi_full)
    return env.reset()


def audit_applied_disturbance(
    requested_xi_full,
    actual_xi,
    actual_flow_phase_x,
    actual_flow_phase_y,
):
    requested = dict(requested_xi_full or {})
    actual = dict(actual_xi or {})
    mismatched_fields = []
    bounds_violation_fields = []
    missing_fields = [key for key in DISTURBANCE_KEYS if key not in actual]
    missing_fields.extend(key for key in (*DISTURBANCE_KEYS, *FLOW_PHASE_KEYS) if key not in requested)
    if missing_fields:
        mismatched_fields.extend(sorted(set(missing_fields)))

    for key in DISTURBANCE_KEYS:
        if key not in requested or key not in actual:
            continue
        requested_value = requested[key]
        actual_value = actual[key]
        if key == "action_delay_steps":
            requested_is_int = isinstance(requested_value, (int, np.integer)) and not isinstance(requested_value, (bool, np.bool_))
            actual_is_int = isinstance(actual_value, (int, np.integer)) and not isinstance(actual_value, (bool, np.bool_))
            if not requested_is_int or not actual_is_int or int(requested_value) != int(actual_value):
                mismatched_fields.append(key)
            actual_numeric = float(actual_value)
        else:
            try:
                requested_numeric = float(requested_value)
                actual_numeric = float(actual_value)
            except (TypeError, ValueError):
                mismatched_fields.append(key)
                continue
            if not math.isfinite(requested_numeric) or not math.isfinite(actual_numeric):
                mismatched_fields.append(key)
            elif not math.isclose(requested_numeric, actual_numeric, rel_tol=1e-12, abs_tol=1e-12):
                mismatched_fields.append(key)
        try:
            low, high = DEFAULT_DISTURBANCE_BOUNDS[key]
            actual_numeric = float(actual_value)
            if not math.isfinite(actual_numeric) or actual_numeric < float(low) or actual_numeric > float(high):
                bounds_violation_fields.append(key)
        except (TypeError, ValueError):
            bounds_violation_fields.append(key)

    actual_phases = {
        "flow_phase_x": actual_flow_phase_x,
        "flow_phase_y": actual_flow_phase_y,
    }
    for key in FLOW_PHASE_KEYS:
        if key not in requested:
            continue
        try:
            requested_value = float(requested[key])
            actual_value = float(actual_phases[key])
        except (TypeError, ValueError):
            mismatched_fields.append(key)
            continue
        if (
            not math.isfinite(requested_value)
            or not math.isfinite(actual_value)
            or not math.isclose(requested_value, actual_value, rel_tol=1e-12, abs_tol=1e-12)
        ):
            mismatched_fields.append(key)

    result = {
        "match": not mismatched_fields and not bounds_violation_fields,
        "mismatched_fields": sorted(set(mismatched_fields)),
        "bounds_violation_fields": sorted(set(bounds_violation_fields)),
    }
    if not result["match"]:
        raise RuntimeError(f"Applied disturbance audit failed: {result}")
    return result


def _float_attr(obj, name, default=float("nan")):
    try:
        value = float(getattr(obj, name, default))
    except Exception:
        return float(default)
    return value if np.isfinite(value) else float("nan")


def _nanmean(values):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def safe_bool(value, default=False):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None:
        return bool(default)
    if isinstance(value, (int, float, np.integer, np.floating)):
        try:
            value_f = float(value)
        except Exception:
            return bool(default)
        if not np.isfinite(value_f):
            return bool(default)
        return value_f != 0.0
    if isinstance(value, str):
        text = value.strip().lower()
        if text == "":
            return bool(default)
        if text in {"1", "true", "t", "yes", "y"}:
            return True
        if text in {"0", "false", "f", "no", "n"}:
            return False
        try:
            value_f = float(text)
        except Exception:
            return bool(default)
        return bool(default) if not np.isfinite(value_f) else value_f != 0.0
    return bool(default)


def safe_float(value, default=np.nan):
    if value is None:
        return float(default)
    if isinstance(value, str) and value.strip() == "":
        return float(default)
    try:
        value_f = float(value)
    except Exception:
        return float(default)
    return value_f if np.isfinite(value_f) else float(default)


def safe_mean(values):
    valid = []
    for value in values:
        value_f = safe_float(value)
        if np.isfinite(value_f):
            valid.append(value_f)
    return float(np.mean(valid)) if valid else None


def rate(num, den):
    den_i = int(den)
    return None if den_i == 0 else float(num) / float(den_i)


def _first_present(row, field_candidates, default=None):
    for field in field_candidates:
        if field in row:
            return row.get(field)
    return default


def _row_bool(row, field_candidates, default=False):
    return safe_bool(_first_present(row, field_candidates), default=default)


def _row_float(row, field_candidates, default=np.nan):
    return safe_float(_first_present(row, field_candidates), default=default)


def _row_found_step(row):
    found_step = _row_float(row, ("found_step", "first_found_step"))
    if np.isfinite(found_step):
        return found_step
    success_step = _row_float(row, ("success_step",))
    delta = _row_float(row, ("success_step_minus_found_step",))
    if np.isfinite(success_step) and np.isfinite(delta):
        return success_step - delta
    return float("nan")


def _row_success_step(row, success_flag):
    success_step = _row_float(row, ("success_step",))
    if np.isfinite(success_step):
        return success_step
    if success_flag:
        return _row_float(row, ("completion_steps",))
    return float("nan")


def _row_completion_steps(row, max_steps):
    return _row_float(row, ("completion_steps", "episode_steps"), default=float(max_steps))


def group_mean(rows, mask, field_candidates):
    values = []
    for row in rows:
        if mask(row):
            values.append(_row_float(row, field_candidates))
    return safe_mean(values)


def build_failure_breakdown_summary(rows, max_steps, late_found_threshold):
    row_states = []
    for row in rows:
        success_flag = _row_bool(row, ("success_flag", "success", "mission_complete"))
        found_flag = _row_bool(row, ("found_flag", "found", "task_found"))
        found_step = _row_found_step(row)
        success_step = _row_success_step(row, success_flag)
        completion_steps = _row_completion_steps(row, max_steps)
        found_but_failed = found_flag and not success_flag
        late_found_fail = bool(found_but_failed and np.isfinite(found_step) and found_step >= late_found_threshold)
        early_found_fail = bool(found_but_failed and np.isfinite(found_step) and found_step < late_found_threshold)
        row_states.append(
            {
                "row": row,
                "success": success_flag,
                "found": found_flag,
                "not_found": not found_flag,
                "found_but_failed": found_but_failed,
                "late_found_fail": late_found_fail,
                "early_found_fail": early_found_fail,
                "found_step": found_step,
                "success_step": success_step,
                "completion_steps": completion_steps,
            }
        )

    n_episodes = int(len(row_states))
    n_success = int(sum(s["success"] for s in row_states))
    n_found = int(sum(s["found"] for s in row_states))
    n_not_found = int(sum(s["not_found"] for s in row_states))
    n_found_but_failed = int(sum(s["found_but_failed"] for s in row_states))
    n_late_found_fail = int(sum(s["late_found_fail"] for s in row_states))
    n_early_found_fail = int(sum(s["early_found_fail"] for s in row_states))

    def mask_success(row):
        return _state(row)["success"]

    def mask_found_failed(row):
        return _state(row)["found_but_failed"]

    def mask_not_found(row):
        return _state(row)["not_found"]

    state_by_id = {id(s["row"]): s for s in row_states}

    def _state(row):
        return state_by_id[id(row)]

    success_states = [s for s in row_states if s["success"]]
    failed_states = [s for s in row_states if not s["success"]]
    found_failed_states = [s for s in row_states if s["found_but_failed"]]

    failure_summary = {
        "n_episodes": n_episodes,
        "n_success": n_success,
        "n_found": n_found,
        "n_not_found": n_not_found,
        "n_found_but_failed": n_found_but_failed,
        "n_late_found_fail": n_late_found_fail,
        "n_early_found_fail": n_early_found_fail,
        "succ_if_found": rate(n_success, n_found),
        "not_found_rate": rate(n_not_found, n_episodes),
        "found_but_failed_rate": rate(n_found_but_failed, n_episodes),
        "late_found_fail_rate": rate(n_late_found_fail, n_found_but_failed),
        "early_found_fail_rate": rate(n_early_found_fail, n_found_but_failed),
        "late_found_threshold": int(late_found_threshold),
        "avg_found_step_success": safe_mean(s["found_step"] for s in success_states),
        "avg_found_step_failed": safe_mean(s["found_step"] for s in failed_states),
        "avg_found_to_success_steps": safe_mean(
            s["success_step"] - s["found_step"]
            for s in success_states
            if np.isfinite(s["success_step"]) and np.isfinite(s["found_step"])
        ),
        "avg_found_to_fail_remaining_steps": safe_mean(
            float(max_steps) - s["found_step"] for s in found_failed_states if np.isfinite(s["found_step"])
        ),
        "avg_completion_steps_success": safe_mean(s["completion_steps"] for s in success_states),
        "avg_completion_steps_failed": safe_mean(s["completion_steps"] for s in failed_states),
        "avg_final_distance_success": group_mean(rows, mask_success, ("final_distance",)),
        "avg_final_distance_found_but_failed": group_mean(rows, mask_found_failed, ("final_distance",)),
        "avg_final_distance_not_found": group_mean(rows, mask_not_found, ("final_distance",)),
        "avg_final_nav_distance_success": group_mean(rows, mask_success, ("final_nav_distance",)),
        "avg_final_nav_distance_found_but_failed": group_mean(rows, mask_found_failed, ("final_nav_distance",)),
        "avg_final_nav_distance_not_found": group_mean(rows, mask_not_found, ("final_nav_distance",)),
        "avg_safety_cost_success": group_mean(rows, mask_success, ("safety_cost",)),
        "avg_safety_cost_found_but_failed": group_mean(rows, mask_found_failed, ("safety_cost",)),
        "avg_safety_cost_not_found": group_mean(rows, mask_not_found, ("safety_cost",)),
        "avg_recovery_time_success": group_mean(rows, mask_success, ("recovery_time",)),
        "avg_recovery_time_found_but_failed": group_mean(rows, mask_found_failed, ("recovery_time",)),
        "avg_recovery_time_not_found": group_mean(rows, mask_not_found, ("recovery_time",)),
    }

    diagnostic_fields = (
        "standby_to_target_dist",
        "belief_entropy",
        "exec_response_cost",
        "residual_contribution_ratio_executor",
        "residual_contribution_ratio_search",
    )
    diagnostic_groups = (
        ("success", mask_success),
        ("found_but_failed", mask_found_failed),
        ("not_found", mask_not_found),
    )
    for field in diagnostic_fields:
        for suffix, mask in diagnostic_groups:
            failure_summary[f"avg_{field}_{suffix}"] = group_mean(rows, mask, (field,))

    return failure_summary


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _apply_profile(env, profile):
    if profile not in PROFILES:
        raise ValueError(f"Unknown PSE profile: {profile}")
    applied = dict(NORMAL_COMM_PROFILE)
    for key, value in NORMAL_COMM_PROFILE.items():
        setattr(env, key, value)
    if profile == "basic":
        applied.update({
            "comm_scenario_id": 0,
            "comm_loss_prob": 0.0,
            "upper_comm_base_loss": 0.0,
            "payload_loss_scale": 0.0,
        })
        env.use_comm = True
        env.use_upper_comm = True
    elif profile == "weak_comm":
        applied.update({
            "comm_scenario_id": 1,
            "comm_loss_prob": max(float(env.comm_loss_prob), 0.22),
            "upper_comm_base_loss": max(float(env.upper_comm_base_loss), 0.22),
            "comm_attenuation": max(float(env.comm_attenuation), 0.28),
            "upper_comm_attenuation": max(float(env.upper_comm_attenuation), 0.28),
            "lower_extra_delay_steps": max(int(env.lower_extra_delay_steps), 2),
            "upper_extra_delay_steps": max(int(env.upper_extra_delay_steps), 2),
            "lower_rate_bps": min(float(env.lower_rate_bps), 500.0),
            "upper_rate_bps": min(float(env.upper_rate_bps), 500.0),
            "payload_loss_scale": max(float(env.payload_loss_scale), 0.040),
            "critical_message_loss_prob": max(float(getattr(env, "critical_message_loss_prob", 0.0) or 0.0), 0.15),
            "reconnect_island_timeout_steps": min(int(env.reconnect_island_timeout_steps), 5),
        })
    elif profile == "severe_comm":
        applied.update({
            "comm_scenario_id": 2,
            "comm_loss_prob": max(float(env.comm_loss_prob), 0.45),
            "upper_comm_base_loss": max(float(env.upper_comm_base_loss), 0.45),
            "comm_attenuation": max(float(env.comm_attenuation), 0.45),
            "upper_comm_attenuation": max(float(env.upper_comm_attenuation), 0.45),
            "comm_max_range": min(float(env.comm_max_range), 10.0),
            "upper_comm_max_range": min(float(env.upper_comm_max_range), 12.0),
            "lower_extra_delay_steps": max(int(env.lower_extra_delay_steps), 5),
            "upper_extra_delay_steps": max(int(env.upper_extra_delay_steps), 4),
            "lower_max_delay_steps": max(int(env.lower_max_delay_steps), 32),
            "lower_rate_bps": min(float(env.lower_rate_bps), 300.0),
            "upper_rate_bps": min(float(env.upper_rate_bps), 300.0),
            "payload_loss_scale": max(float(env.payload_loss_scale), 0.060),
            "critical_message_loss_prob": max(float(getattr(env, "critical_message_loss_prob", 0.0) or 0.0), 0.35),
            "reconnect_island_timeout_steps": min(int(env.reconnect_island_timeout_steps), 4),
        })
    elif profile == "robust_full":
        applied.update({
            "comm_scenario_id": 3,
            "comm_loss_prob": max(float(env.comm_loss_prob), 0.45),
            "upper_comm_base_loss": max(float(env.upper_comm_base_loss), 0.45),
            "comm_attenuation": max(float(env.comm_attenuation), 0.45),
            "upper_comm_attenuation": max(float(env.upper_comm_attenuation), 0.45),
            "comm_max_range": min(float(env.comm_max_range), 10.0),
            "upper_comm_max_range": min(float(env.upper_comm_max_range), 12.0),
            "lower_extra_delay_steps": max(int(env.lower_extra_delay_steps), 5),
            "upper_extra_delay_steps": max(int(env.upper_extra_delay_steps), 4),
            "lower_max_delay_steps": max(int(env.lower_max_delay_steps), 32),
            "lower_rate_bps": min(float(env.lower_rate_bps), 300.0),
            "upper_rate_bps": min(float(env.upper_rate_bps), 300.0),
            "payload_loss_scale": max(float(env.payload_loss_scale), 0.060),
            "critical_message_loss_prob": max(float(getattr(env, "critical_message_loss_prob", 0.0) or 0.0), 0.35),
            "reconnect_island_timeout_steps": min(int(env.reconnect_island_timeout_steps), 4),
        })
    for key, value in applied.items():
        setattr(env, key, value)
    if hasattr(env, "_sync_lower_comm_graph_config"):
        env._sync_lower_comm_graph_config()

def evaluate(args):
    if args.ablation_mode not in CHAPTER_CONFIGS:
        raise ValueError(f"Unknown ablation_mode={args.ablation_mode}")
    if args.paired_episode_seeding and args.seed is None:
        raise ValueError("--paired-episode-seeding requires an explicit --seed")
    requested_protocol = getattr(args, "disturbance_protocol", DISTURBANCE_PROTOCOL_AUTO)
    resolved_protocol = resolve_disturbance_protocol(args.ablation_mode, requested_protocol)
    if resolved_protocol == DISTURBANCE_PROTOCOL_UNIFORM_9D and args.seed is None:
        raise ValueError("uniform_9d_registry_v1 requires an explicit --seed")

    # Support both Chapter-3 PSE modes and Chapter-4 RBE modes.
    # ch4_* modes are registered under the ch4_rbe scope, while the older
    # pse_* modes remain under the ch3_pse scope.
    scope = "ch4_rbe" if str(args.ablation_mode).startswith("ch4_") else "ch3_pse"
    active = experiment_registry.get_active_modes(scope, all_modes=CHAPTER_CONFIGS.keys())
    if args.ablation_mode not in active:
        raise ValueError(f"{args.ablation_mode} is not active under {scope} scope")

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"model_path does not exist: {args.model_path}")

    if args.seed is not None:
        _set_global_seed(args.seed)

    os.makedirs(args.result_dir, exist_ok=True)
    env_cfg, _ = get_ablation_config(args.ablation_mode)
    env, _ = _build_train_env(torch.device("cpu"), int(args.max_steps), ablation_config=env_cfg)
    _apply_profile(env, args.profile)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    maddpg = MADDPG.init_from_save(args.model_path, device=device)
    maddpg.prep_rollouts(device=device)

    rows = []
    missing_fields = set()
    tracker = EpisodeMetricTracker()
    base_seed = int(args.seed) if args.seed is not None else None
    for ep in range(int(args.episodes)):
        episode_index = int(ep)
        if args.paired_episode_seeding:
            episode_seed = disturbance_seed_for_episode(base_seed, episode_index)
            _set_global_seed(episode_seed)
            episode_seed_mode = PAIRED_EPISODE_SEED_MODE
        else:
            episode_seed = None
            episode_seed_mode = "legacy_stream"
        _apply_profile(env, args.profile)
        requested_xi_full, disturbance_seed, disturbance_rng_mode = sample_episode_disturbance(
            resolved_protocol,
            base_seed,
            episode_index,
        )
        obs = apply_disturbance_before_reset(env, resolved_protocol, requested_xi_full)
        if not hasattr(env, "get_current_disturbance"):
            raise RuntimeError("Environment does not implement get_current_disturbance()")
        actual_xi = env.get_current_disturbance()
        actual_flow_phase_x = float(getattr(env, "_flow_phase_x"))
        actual_flow_phase_y = float(getattr(env, "_flow_phase_y"))
        disturbance_audit = audit_applied_disturbance(
            requested_xi_full,
            actual_xi,
            actual_flow_phase_x,
            actual_flow_phase_y,
        )
        tracker.reset(env, actual_xi)
        episode_reward = 0.0
        found_step = float("nan")
        success_step = float("nan")
        step_values = {key: [] for key in PSE_STEP_METRICS}

        for step_i in range(1, int(args.max_steps) + 1):
            actions = maddpg.step(obs, explore=False)
            env_actions = torch.stack([a.squeeze(0) for a in actions], dim=0).to(device=env.device, dtype=torch.float32)
            obs, rewards, dones = env.step(env_actions)
            rewards_t = rewards if torch.is_tensor(rewards) else torch.as_tensor(rewards, dtype=torch.float32)
            episode_reward += float(rewards_t.mean().detach().cpu().item())
            tracker.step(env, env_actions, rewards_t, dones)

            if np.isnan(found_step) and bool(getattr(env, "task_found", False)):
                found_step = float(step_i)
            if np.isnan(success_step) and bool(getattr(env, "mission_complete", False)):
                success_step = float(step_i)

            for key, attr in PSE_STEP_METRICS.items():
                if not hasattr(env, attr):
                    missing_fields.add(attr)
                step_values[key].append(_float_attr(env, attr))

            if all(bool(d) for d in dones):
                break

        row = {
            "episode": ep,
            "base_seed": base_seed,
            "episode_index": episode_index,
            "episode_seed": episode_seed,
            "episode_seed_mode": episode_seed_mode,
            "disturbance_seed": disturbance_seed,
            "disturbance_rng_mode": disturbance_rng_mode,
            "disturbance_protocol": resolved_protocol,
            "disturbance_explicitly_applied": True,
            "disturbance_apply_match": disturbance_audit["match"],
            **{key: actual_xi[key] for key in DISTURBANCE_KEYS},
            "flow_phase_x": actual_flow_phase_x,
            "flow_phase_y": actual_flow_phase_y,
            "method": args.ablation_mode,
            "found": 1.0 if bool(getattr(env, "task_found", False)) else 0.0,
            "success": 1.0 if bool(getattr(env, "mission_complete", False)) else 0.0,
            "reward": float(episode_reward),
            "found_step": found_step,
            "success_step": success_step,
            "success_step_minus_found_step": (
                success_step - found_step if np.isfinite(success_step) and np.isfinite(found_step) else float("nan")
            ),
        }
        final_metrics = tracker.finalize(env)
        row.update(
            {
                "success_flag": bool(final_metrics.get("success_flag", bool(getattr(env, "mission_complete", False)))),
                "found_flag": bool(final_metrics.get("found_flag", bool(getattr(env, "task_found", False)))),
                "completion_steps": int(final_metrics.get("completion_steps", getattr(env, "step_count", args.max_steps))),
                "recovery_time": int(final_metrics.get("recovery_time", args.max_steps)),
                "safety_cost": float(final_metrics.get("safety_cost", float("nan"))),
                "final_distance": float(final_metrics.get("final_distance", float("nan"))),
                "final_nav_distance": float(final_metrics.get("final_nav_distance", float("nan"))),
                "action_smoothness": float(final_metrics.get("action_smoothness", float("nan"))),
            }
        )
        for key, values in step_values.items():
            row[key] = _nanmean(values)
        rows.append(row)
        print(f"ep {ep + 1}/{args.episodes}: found={row['found']:.0f} success={row['success']:.0f} reward={row['reward']:.2f}")

    if len(rows) != int(args.episodes):
        raise RuntimeError(f"Expected {args.episodes} episode rows, found {len(rows)}")
    for expected_index, row in enumerate(rows):
        missing_disturbance_fields = [
            key for key in (*DISTURBANCE_KEYS, *FLOW_PHASE_KEYS) if key not in row
        ]
        if missing_disturbance_fields:
            raise RuntimeError(f"Episode {expected_index} lacks disturbance fields: {missing_disturbance_fields}")
        if row.get("disturbance_apply_match") is not True:
            raise RuntimeError(f"Episode {expected_index} failed disturbance application audit")
        if row.get("disturbance_protocol") != resolved_protocol:
            raise RuntimeError(f"Episode {expected_index} has inconsistent disturbance protocol")
        if int(row.get("episode_index", -1)) != expected_index:
            raise RuntimeError(f"Episode index sequence is invalid at row {expected_index}")
        if args.paired_episode_seeding:
            expected_seed = disturbance_seed_for_episode(base_seed, expected_index)
            if int(row.get("episode_seed")) != expected_seed:
                raise RuntimeError(f"Episode seed formula mismatch at row {expected_index}")

    disturbance_vectors = {
        tuple(row[key] for key in DISTURBANCE_KEYS)
        for row in rows
    }
    full_disturbances = {
        tuple(row[key] for key in (*DISTURBANCE_KEYS, *FLOW_PHASE_KEYS))
        for row in rows
    }
    per_dimension_distinct_counts = {
        key: len({row[key] for row in rows})
        for key in DISTURBANCE_KEYS
    }
    observed_delays = sorted({int(row["action_delay_steps"]) for row in rows})
    observed_lags = [float(row["actuator_lag"]) for row in rows]
    observed_noise = [float(row["action_noise_std"]) for row in rows]

    found = np.asarray([r["found"] for r in rows], dtype=np.float32)
    success = np.asarray([r["success"] for r in rows], dtype=np.float32)
    late_found_threshold = int(os.environ.get("EVAL_LATE_FOUND_THRESHOLD", "350"))
    summary = {
        "method": args.ablation_mode,
        "profile": args.profile,
        "episodes": int(args.episodes),
        "max_steps": int(args.max_steps),
        "seed": args.seed,
        "paired_episode_seeding": bool(args.paired_episode_seeding),
        "episode_seed_mode": (
            PAIRED_EPISODE_SEED_MODE
            if args.paired_episode_seeding
            else "legacy_stream"
        ),
        "episode_seed_formula": (
            PAIRED_EPISODE_SEED_FORMULA
            if args.paired_episode_seeding
            else None
        ),
        "episode_index_base": 0,
        "first_episode_seed": (
            int(base_seed) * 1_000_003
            if args.paired_episode_seeding and int(args.episodes) > 0
            else None
        ),
        "last_episode_seed": (
            int(base_seed) * 1_000_003 + int(args.episodes) - 1
            if args.paired_episode_seeding and int(args.episodes) > 0
            else None
        ),
        "model_path": args.model_path,
        "requested_disturbance_protocol": requested_protocol,
        "disturbance_protocol": resolved_protocol,
        "disturbance_protocol_version": 2,
        "disturbance_sampler": (
            "registry.rbe_disturbance.sample_uniform_disturbance"
            if resolved_protocol == DISTURBANCE_PROTOCOL_UNIFORM_9D
            else "registry.rbe_disturbance.nominal_disturbance"
        ),
        "disturbance_rng_mode": DISTURBANCE_RNG_MODE,
        "disturbance_seed_formula": PAIRED_EPISODE_SEED_FORMULA,
        "disturbance_keys": list(DISTURBANCE_KEYS),
        "disturbance_bounds": {
            key: [DEFAULT_DISTURBANCE_BOUNDS[key][0], DEFAULT_DISTURBANCE_BOUNDS[key][1]]
            for key in DISTURBANCE_KEYS
        },
        "flow_phase_keys": list(FLOW_PHASE_KEYS),
        "flow_phase_randomized": resolved_protocol == DISTURBANCE_PROTOCOL_UNIFORM_9D,
        "flow_phase_in_9d_vector": False,
        "explicit_disturbance_application": True,
        "all_episode_disturbance_apply_match": all(
            row["disturbance_apply_match"] is True for row in rows
        ),
        "bounds_violation_count": 0,
        "distinct_disturbance_vector_count": len(disturbance_vectors),
        "distinct_full_disturbance_count": len(full_disturbances),
        "per_dimension_distinct_counts": per_dimension_distinct_counts,
        "observed_action_delay_values": observed_delays,
        "observed_actuator_lag_min": min(observed_lags) if observed_lags else None,
        "observed_actuator_lag_max": max(observed_lags) if observed_lags else None,
        "observed_action_noise_std_min": min(observed_noise) if observed_noise else None,
        "observed_action_noise_std_max": max(observed_noise) if observed_noise else None,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "found_rate": float(found.mean()) if found.size else float("nan"),
        "success_rate": float(success.mean()) if success.size else float("nan"),
        "avg_reward": safe_mean(row.get("reward") for row in rows),
        "avg_recovery_time": safe_mean(row.get("recovery_time") for row in rows),
        "avg_safety_cost": safe_mean(row.get("safety_cost") for row in rows),
        "avg_final_distance": safe_mean(row.get("final_distance") for row in rows),
        "avg_final_nav_distance": safe_mean(row.get("final_nav_distance") for row in rows),
        "avg_action_smoothness": safe_mean(row.get("action_smoothness") for row in rows),
        "avg_completion_steps": safe_mean(row.get("completion_steps") for row in rows),
        "avg_found_step": _nanmean([r["found_step"] for r in rows if r["found"] > 0.5]),
        "avg_success_step": _nanmean([r["success_step"] for r in rows if r["success"] > 0.5]),
        "avg_success_step_minus_found_step": _nanmean([r["success_step_minus_found_step"] for r in rows]),
        "avg_handoff_delay": _nanmean([r["handoff_delay"] for r in rows]),
        "avg_belief_entropy": _nanmean([r["belief_entropy"] for r in rows]),
        "avg_standby_to_target_dist": _nanmean([r["standby_to_target_dist"] for r in rows]),
        "avg_exec_response_cost": _nanmean([r["exec_response_cost"] for r in rows]),
        "avg_search_score_mean": _nanmean([r["search_score_mean"] for r in rows]),
        "avg_residual_contribution_ratio_search": _nanmean([r["residual_contribution_ratio_search"] for r in rows]),
        "avg_residual_contribution_ratio_executor": _nanmean([r["residual_contribution_ratio_executor"] for r in rows]),
        "avg_pse_exec_cost_weight_effective": _nanmean([r["pse_exec_cost_weight_effective"] for r in rows]),
        "avg_pse_exec_cost_schedule_factor": _nanmean([r["pse_exec_cost_schedule_factor"] for r in rows]),
        "avg_pse_lazy_standby_active": _nanmean([r["pse_lazy_standby_active"] for r in rows]),
        "avg_pse_standby_update_allowed": _nanmean([r["pse_standby_update_allowed"] for r in rows]),
        "avg_pse_standby_update_skipped_by_lazy_gate": _nanmean([r["pse_standby_update_skipped_by_lazy_gate"] for r in rows]),
        "missing_fields": sorted(missing_fields),
    }
    summary.update(build_failure_breakdown_summary(rows, int(args.max_steps), late_found_threshold))

    safe_summary = _json_safe(summary)
    with open(os.path.join(args.result_dir, "evaluation_summary.json"), "w", encoding="utf-8") as f:
        json.dump(safe_summary, f, indent=2, ensure_ascii=False, allow_nan=False)
    with open(os.path.join(args.result_dir, "episode_metrics.csv"), "w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys()) if rows else ["episode"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(safe_summary, indent=2, ensure_ascii=False, allow_nan=False))
    print(
        "[EvalSummary] added failure breakdown metrics: "
        f"n_success={safe_summary.get('n_success')}, "
        f"n_found_but_failed={safe_summary.get('n_found_but_failed')}, "
        f"n_not_found={safe_summary.get('n_not_found')}, "
        f"succ_if_found={safe_summary.get('succ_if_found')}"
    )


def main():
    parser = argparse.ArgumentParser(description="PSE-RMADDPG nominal task-chain evaluation.")
    parser.add_argument("--ablation_mode", required=True, choices=sorted(CHAPTER_CONFIGS))
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--paired-episode-seeding",
        "--paired_episode_seeding",
        dest="paired_episode_seeding",
        action="store_true",
    )
    parser.add_argument(
        "--disturbance-protocol",
        choices=SUPPORTED_DISTURBANCE_PROTOCOLS,
        default=DISTURBANCE_PROTOCOL_AUTO,
    )
    parser.add_argument("--max_steps", type=int, default=400)
    parser.add_argument("--profile", default="normal_comm", choices=PROFILES)
    parser.add_argument("--result_dir", default="eval_pse_results")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
