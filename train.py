# -*- coding: utf-8 -*-
"""Unified Chapter-3 PSE and Chapter-4 RBE training entry.

The former Chapter-5 robust-communication training surface has been removed
from this workspace.  ``python train.py`` is the canonical Chapter-4 training
entry; CH4_MODE selects clean PSE baseline, uniform DR, REB-only, or full RBE.
"""

import csv
import hashlib
import json
import os
from datetime import datetime

import numpy as np
import torch

from algorithms.maddpg import MADDPG
from env import UAVEnv
from registry.rbe_disturbance import (
    DEFAULT_DISTURBANCE_BOUNDS,
    DISTURBANCE_KEYS,
    nominal_disturbance,
    sample_uniform_disturbance,
)
from registry.ch4_artifact_layout import (
    create_training_run_directories,
    get_run_root,
    get_selected_dir,
    get_training_run_dir,
    resolve_artifact_path,
)
from utils.buffer import ReplayBuffer
from utils.rbe_metrics import EpisodeMetricTracker
from utils.reb_dataset import REBOutcomeDataset
from utils.reb_model import REBModel, REBTrainer
from utils.rbe_sampler import RBEDisturbanceSampler


ABLATION_MODE = "ch4_pse_baseline"
AB_MODE = os.environ.get("CH4_MODE", os.environ.get("ABLATION_MODE", os.environ.get("AB_MODE", ABLATION_MODE)))
REWARD_PROFILE = os.environ.get("REWARD_PROFILE", "residual_point_v3")
RESIDUAL_ACTION_MODE = os.environ.get("RESIDUAL_ACTION_MODE", "hybrid_v4")


def _env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in ("1", "true", "yes", "y", "on")


def _env_float(name, default):
    value = os.environ.get(name)
    return float(default if value is None or str(value).strip() == "" else value)


def _env_int(name, default):
    value = os.environ.get(name)
    return int(default if value is None or str(value).strip() == "" else value)


BASIC_COMM_SWITCHES = {
    "use_dynamic_comm_graph": False,
    "use_depth_comm_loss": False,
    "use_flow_comm_loss": False,
    "use_noise_comm_loss": False,
    "use_snr_comm_model": False,
    "use_semantic_messages": False,
    "use_voi_selector": False,
    "use_critical_priority": False,
    "force_heartbeat_only": False,
    "lower_message_topk": 1,
    "critical_reserve_slots": 0,
    "use_adaptive_voi": False,
    "use_stage_aware_voi": False,
    "use_multi_packet_semantic_comm": False,
    "use_role_topology": False,
    "role_topology_mode": "full",
    "use_direct_handover_lane": False,
    "role_topology_allow_executor_status": False,
    "role_topology_allow_searcher_executor_pre_found": True,
    "role_topology_strict_message_filter": False,
    "direct_handover_bypass_message_filter": False,
    "use_adaptive_handover_bypass": False,
    "use_adaptive_critical_semantic_filter": False,
    "use_guarded_adaptive_critical_filter": False,
    "adaptive_reconnect_voi_boost": False,
    "use_reconnect_lane": False,
    "use_teammate_prediction": False,
    "use_prediction_guided_fallback": False,
    "use_target_belief_memory": False,
    "use_reliability_fusion": False,
    "use_quarantine_buffer": False,
    "use_channel_attention": False,
    "use_channel_edge_features": False,
    "use_channel_critical_attention": False,
    "use_channel_voi_attention": False,
    "use_channel_reliability_attention": False,
    "use_channel_uncertainty_penalty": False,
    "use_channel_role_attention": False,
    "use_learned_role_edge_scorer": False,
    "use_upper_comm": False,
    "use_upper_belief": False,
}


PSE_BASELINE_SWITCHES = {
    **BASIC_COMM_SWITCHES,
    "use_pse_planner": True,
    "pse_use_belief": True,
    "pse_use_exec_cost": True,
    "pse_use_standby": True,
    "pse_lazy_standby": True,
    "pse_standby_entropy_gate": 0.75,
    "pse_standby_min_step": 80,
    "pse_standby_update_interval_lazy": 4,
    "pse_standby_move_weight_lazy": 0.40,
    "pse_standby_hysteresis_weight_lazy": 0.25,
    "reward_profile": "residual_point_v3",
    "residual_action_mode": "hybrid_v4",
    "use_robust_disturbance": False,
}


CHAPTER_CONFIGS = {
    "pse_pred_lite_54": dict(PSE_BASELINE_SWITCHES, pse_lazy_standby=False),
    "pse_lazy_standby_54": dict(PSE_BASELINE_SWITCHES),
    "pse_no_belief": dict(PSE_BASELINE_SWITCHES, pse_use_belief=False),
    "pse_no_exec_cost": dict(PSE_BASELINE_SWITCHES, pse_use_exec_cost=False),
    "pse_no_standby": dict(PSE_BASELINE_SWITCHES, pse_use_standby=False, pse_lazy_standby=False),
    "pse_no_residual": dict(PSE_BASELINE_SWITCHES, residual_scale_search=0.0, residual_scale_executor=0.0),
    "ch4_pse_baseline": dict(PSE_BASELINE_SWITCHES),
    "ch4_uniform_dr": dict(PSE_BASELINE_SWITCHES, use_robust_disturbance=True),
    "ch4_reb_only": dict(PSE_BASELINE_SWITCHES, use_robust_disturbance=True),
    "ch4_rbe_full": dict(PSE_BASELINE_SWITCHES, use_robust_disturbance=True),
}


def get_ablation_config(ablation_mode=None):
    mode = AB_MODE if ablation_mode is None else str(ablation_mode)
    aliases = {
        "clean": "ch4_pse_baseline",
        "baseline": "ch4_pse_baseline",
        "pse_baseline": "ch4_pse_baseline",
        "uniform": "ch4_uniform_dr",
        "uniform_dr": "ch4_uniform_dr",
        "reb": "ch4_reb_only",
        "reb_only": "ch4_reb_only",
        "rbe": "ch4_rbe_full",
        "rbe_full": "ch4_rbe_full",
    }
    mode = aliases.get(mode, mode)
    if mode not in CHAPTER_CONFIGS:
        raise ValueError(f"Unknown mode={mode}. Available: {sorted(CHAPTER_CONFIGS)}")
    return dict(CHAPTER_CONFIGS[mode]), {
        "use_tail_weighted_training": False,
        "use_tail_replay_priority": False,
        "use_tail_critic_loss": False,
        "use_tail_actor_loss": False,
    }


def _build_train_env(env_device, max_steps, ablation_config=None):
    kwargs = {
        "use_obstacles": False,
        "device": env_device,
        "return_numpy": False,
        "max_steps": max_steps,
        "use_comm": True,
        "comm_range": 8.0,
        "comm_loss_prob": 0.08,
        "comm_delay_steps": 1,
        "comm_reliable_range": 6.0,
        "comm_max_range": 15.0,
        "comm_attenuation": 0.18,
        "lower_effective_sound_speed": 12.0,
        "lower_extra_delay_steps": 1,
        "lower_max_delay_steps": 16,
        "lower_payload_bits": 160,
        "lower_rate_bps": 1000.0,
        "lower_msg_ttl_steps": 24,
        "lower_initial_sync_mode": "none",
        "use_burst_comm": True,
        "burst_state_pdr": (1.0, 0.45, 0.03),
        "payload_loss_scale": 0.025,
        "use_comm_energy": True,
        "comm_idle_power": 0.02,
        "comm_rx_power": 0.26,
        "comm_tx_power": 5.0,
        "lambda_comm_energy": 0.004,
        "use_upper_comm": False,
        "use_upper_belief": False,
        "hold_steps": 5,
        "search_hold_steps": 0,
        "executor_hold_steps": 5,
        "hold_speed_thresh": 0.20,
        "random_z_range": (0.50, 7.50),
        "random_search_waypoint_count": (2, 4),
        "random_executor_waypoint_count": (1, 1),
        "diverse_fallback_prob": 0.25,
        "diverse_fallback_tries": 64,
        "search_spread_reward_gain": 0.75,
        "detect_proximity_reward_gain": 0.8,
        "detect_proximity_radius": 4.5,
        "planner_grid_size": (10, 10, 8),
        "planner_visit_radius": 1,
        "planner_pheromone_decay": 0.990,
        "planner_suppression": 0.54,
        "planner_min_waypoint_separation": 4.8,
        "planner_step_update_interval": 2,
        "planner_step_update_suppress_only": False,
        "planner_coverage_weight": 1.10,
        "planner_claim_weight": 0.80,
        "planner_stochastic_topk": 14,
        "planner_stochastic_eps": 0.26,
        "use_residual_prior": True,
        "prior_kv_xy": 1.10,
        "prior_kv_z": 1.00,
        "prior_slow_radius_xy": 2.40,
        "prior_slow_radius_z": 1.20,
        "prior_strength_search": 0.45,
        "prior_strength_executor": 0.45,
        "residual_scale_search": 0.45,
        "residual_scale_executor": 0.35,
        "residual_penalty": 0.01,
        "use_robust_disturbance": True,
        "flow_gain_range": (0.15, 0.55),
        "flow_z_gain_range": (-0.08, 0.08),
        "flow_phase_random": True,
        "a_max_scale_range": (0.75, 1.15),
        "v_max_scale_range": (0.80, 1.15),
        "drag_scale_range": (0.70, 1.50),
        "buoyancy_bias_delta_range": (-0.04, 0.06),
        "actuator_lag": 0.25,
        "robust_action_delay_steps": 1,
        "action_noise_std_range": (0.00, 0.05),
        "reward_profile": REWARD_PROFILE,
        "residual_action_mode": RESIDUAL_ACTION_MODE,
        **BASIC_COMM_SWITCHES,
    }
    if ablation_config is not None:
        valid_keys = set(UAVEnv.__init__.__code__.co_varnames)
        kwargs.update({k: v for k, v in dict(ablation_config).items() if k in valid_keys})
    env = UAVEnv(**kwargs)
    env.max_steps = int(max_steps)
    return env, kwargs


def _make_replay_buffer(buffer_size, n_agents, obs_dims, ac_dims, buffer_device=None):
    if buffer_device is None:
        buffer_device = torch.device("cpu")
    return ReplayBuffer(buffer_size, n_agents, obs_dims, ac_dims, storage_device=buffer_device)


def _sample_meta(sample):
    batch_indices = sample[6] if len(sample) > 6 else None
    success_flags = sample[7] if len(sample) > 7 else None
    tail_scores = sample[8] if len(sample) > 8 else None
    return batch_indices, success_flags, tail_scores


def _safe_update_all_targets(maddpg):
    return maddpg.update_all_targets()


def _auto_r_config(level: int, use_matched_residual: bool = True) -> dict:
    level = int(np.clip(level, 0, 100))
    frac = level / 100.0
    prior_s = 0.25 + 0.45 * frac
    prior_e = 0.25 + 0.40 * frac
    if use_matched_residual:
        residual_s = 0.25 + 0.25 * frac
        residual_e = 0.20 + 0.20 * frac
    else:
        residual_s = 0.45
        residual_e = 0.35
    return {
        "prior_strength_search": prior_s,
        "prior_strength_executor": prior_e,
        "residual_scale_search": residual_s,
        "residual_scale_executor": residual_e,
        "residual_penalty": 0.01,
    }


def _apply_auto_r_config(env, cfg, reason="", verbose=True, effective_from_episode=None):
    for key, value in dict(cfg).items():
        if hasattr(env, key):
            setattr(env, key, value)
    if verbose:
        suffix = "" if effective_from_episode is None else f" from episode {effective_from_episode}"
        print(f"[AutoR] {reason}{suffix}: {cfg}")
    return dict(cfg)


def _apply_fixed_r_config(env, level, use_matched_residual=True):
    cfg = _auto_r_config(level, use_matched_residual)
    cfg = _apply_auto_r_config(env, cfg, reason="fixed_R_initial", verbose=False)
    print(
        f"[FixedR] using R{int(level)} | "
        f"prior={float(getattr(env, 'prior_strength_search', np.nan)):.2f}/"
        f"{float(getattr(env, 'prior_strength_executor', np.nan)):.2f} | "
        f"residual={float(getattr(env, 'residual_scale_search', np.nan)):.2f}/"
        f"{float(getattr(env, 'residual_scale_executor', np.nan)):.2f} | "
        f"penalty={float(getattr(env, 'residual_penalty', np.nan)):.3f}"
    )
    return cfg


CH4_RBE_MODES = ("ch4_pse_baseline", "ch4_uniform_dr", "ch4_reb_only", "ch4_rbe_full")
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
NO_RESUME_TOKENS = {
    "",
    "none",
    "null",
    "scratch",
    "from_scratch",
    "from-scratch",
}

def _resolve_training_initialization(mode):
    train_from_scratch_requested = _env_bool("TRAIN_FROM_SCRATCH", False)
    resume_model_path_raw = os.environ.get("RESUME_MODEL_PATH")

    if train_from_scratch_requested:
        if resume_model_path_raw is not None and resume_model_path_raw.strip():
            print(
                "[Initialization][warning] TRAIN_FROM_SCRATCH=1 overrides "
                f"RESUME_MODEL_PATH={resume_model_path_raw!r}"
            )
        return {
            "train_from_scratch": True,
            "train_from_scratch_requested": True,
            "resume_model_path_raw": resume_model_path_raw,
            "resume_model_path": None,
            "initialization_source": "from_scratch_flag",
            "loaded_checkpoint": False,
        }

    if resume_model_path_raw is None:
        raise RuntimeError(
            "Training initialization is ambiguous.\n"
            "Set TRAIN_FROM_SCRATCH=1 for a new model, or set\n"
            "RESUME_MODEL_PATH=<checkpoint path> for warm-start/resume.\n"
            "On Windows, do not use `set RESUME_MODEL_PATH=` as a\n"
            "from-scratch command because it removes the variable."
        )

    resume_value = resume_model_path_raw.strip()
    if resume_value.lower() in NO_RESUME_TOKENS:
        return {
            "train_from_scratch": True,
            "train_from_scratch_requested": False,
            "resume_model_path_raw": resume_model_path_raw,
            "resume_model_path": None,
            "initialization_source": "from_scratch_sentinel",
            "loaded_checkpoint": False,
        }

    resume_model_path = str(resolve_artifact_path(resume_value))
    if not os.path.exists(resume_model_path):
        raise FileNotFoundError(f"RESUME_MODEL_PATH does not exist: {resume_model_path}")
    return {
        "train_from_scratch": False,
        "train_from_scratch_requested": False,
        "resume_model_path_raw": resume_model_path_raw,
        "resume_model_path": resume_model_path,
        "initialization_source": "explicit_checkpoint",
        "loaded_checkpoint": True,
    }


TRAINING_PROGRESS_FIELDS = (
    "episode",
    "episode_steps",
    "avg_reward",
    "success",
    "found",
    "coverage",
    "finished_ratio",
    "actor_loss",
    "critic_loss",
    "disturbance_mode",
    "disturbance_level",
    "rbe_sample_source",
    "rbe_sample_rank",
    "rbe_method_role",
    "rbe_candidate_pool_index",
    "rbe_jitter_applied",
    "rbe_jitter_l2",
    "rbe_action_delay_before_jitter",
    "rbe_action_delay_after_jitter",
    "optimizer_update_count_cumulative",
    "per_agent_actor_update_count",
    "per_agent_critic_update_count",
    "recovery_time",
    "safety_cost",
    "final_distance",
    "final_nav_distance",
    "action_smoothness",
    "reb_loss",
    "rscu_lambda_rec",
    "rscu_lambda_safe",
    "rscu_rec_critic_loss",
    "rscu_safe_critic_loss",
    "rscu_actor_rec_penalty",
    "rscu_actor_safe_penalty",
    "rscu_actor_smooth_penalty",
    "rscu_post_found_sample_frac",
    "rscu_near_target_sample_frac",
    "rscu_aux_mask_frac",
    "rscu_aux_mask_count",
    "rscu_actor_applied",
)


def _mean_tail(values, window):
    if not values:
        return float("nan")
    arr = np.asarray(values[-min(len(values), int(window)):], dtype=np.float64)
    return float(np.nanmean(arr)) if arr.size else float("nan")


def _to_env_actions(actions, env_device):
    return torch.stack([a.detach().to(device=env_device, dtype=torch.float32).view(-1) for a in actions], dim=0)


def _loss_to_float(x):
    return float(x.detach().item()) if torch.is_tensor(x) else float(x)


def _success_flags(env, dones):
    success = bool(getattr(env, "mission_complete", False))
    done = bool(any(bool(x) for x in dones))
    return [success and done] * int(getattr(env, "num_agents", 4))


def _coverage(env):
    if hasattr(env, "_current_coverage_ratio_internal"):
        return float(env._current_coverage_ratio_internal())
    if hasattr(env, "map_module") and hasattr(env.map_module, "coverage"):
        coverage = env.map_module.coverage
        if torch.is_tensor(coverage):
            return float((coverage > 1e-6).float().mean().item())
        return float(np.mean(np.asarray(coverage) > 1e-6))
    return float("nan")


def _finished_ratio(env):
    if hasattr(env, "agent_finished"):
        finished = env.agent_finished
        if torch.is_tensor(finished):
            return float(finished.float().mean().item())
        return float(np.mean(np.asarray(finished, dtype=np.float32)))
    return float("nan")


def _executor_nav_distance(env):
    try:
        pos = getattr(env, "_agent_pos", getattr(env, "agent_pos", None))
        nav = getattr(env, "_nav_targets", getattr(env, "nav_targets", None))
        if pos is None or nav is None:
            return float("nan")
        pos_t = pos.detach() if torch.is_tensor(pos) else torch.as_tensor(pos, dtype=torch.float32)
        nav_t = nav.detach() if torch.is_tensor(nav) else torch.as_tensor(nav, dtype=torch.float32)
        exec_i = int(getattr(env, "executor_idx", pos_t.shape[0] - 1))
        value = float(torch.norm(pos_t[exec_i] - nav_t[exec_i]).detach().cpu().item())
        return value if np.isfinite(value) else float("nan")
    except Exception:
        return float("nan")


def _append_csv(path, row, fieldnames):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_rows_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _atomic_write_json(path, value):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temporary = path + ".tmp"
    try:
        with open(temporary, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, ensure_ascii=False, default=str)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_project_path(path):
    if not path:
        return None
    return str(resolve_artifact_path(path))


def _default_artifact_family(mode):
    explicit = os.environ.get("CH4_ARTIFACT_FAMILY", "").strip()
    if explicit:
        return explicit
    if mode == "ch4_pse_baseline":
        return "clean"
    if mode == "ch4_uniform_dr":
        return "uniform_dr"
    if mode == "ch4_reb_only":
        return "reb"
    ablation_id = os.environ.get("RBE_ABLATION_ID", "").strip().lower()
    ablation_families = {
        "no_boundary_core": "no_boundary",
        "all_actors_trainable": "all_actors",
        "no_nominal_anchor": "no_nominal",
    }
    if ablation_id in ablation_families:
        return ablation_families[ablation_id]
    return "perf_rbe" if _env_bool("RBE_PROTOCOL_LOCK", False) else "rbe_legacy"


def _float_equal(left, right, tol=1e-12):
    return abs(float(left) - float(right)) <= float(tol)


def _canonical_rbe_protocol_object_sha(manifest):
    payload = {
        key: value
        for key, value in dict(manifest).items()
        if key not in {"protocol_object_sha256", "protocol_object_sha256_verified"}
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_method_claims(method_statement_zh, method_statement_en):
    joined = f"{method_statement_zh or ''}\n{method_statement_en or ''}"
    forbidden = (
        "High-risk门控已经通过",
        "完整双分支富集已经通过",
        "candidate_enrichment_supported_v2 == true",
        "RBE策略已经提高鲁棒性",
        "RBE训练已经完成",
        "可恢复边界已经扩张",
    )
    for phrase in forbidden:
        if phrase in joined:
            raise RuntimeError(f"RBE protocol method statement contains forbidden claim: {phrase}")
    lowered = joined.lower()
    if "high-risk" in lowered and "pure execution failure" in lowered and "not" not in lowered:
        raise RuntimeError("RBE protocol method statement mislabels High-risk as pure execution failure")


def _load_rbe_protocol_manifest(path):
    resolved = _resolve_project_path(path)
    if not resolved or not os.path.exists(resolved):
        raise FileNotFoundError(f"RBE_TRAINING_PROTOCOL_MANIFEST missing: {path}")
    with open(resolved, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    return resolved, manifest, _sha256(resolved)


def _rbe_protocol_contract(protocol_name):
    contracts = {
        "ch4_rbe_boundary_core_training_protocol_v1": {
            "family": "legacy",
            "ablation_id": None,
            "search_actors_frozen": False,
            "executor_only_actor_training": False,
            "sampling_ratios": {
                "boundary_core": 0.50,
                "uniform_coverage": 0.30,
                "composite_high_risk_aux": 0.10,
                "nominal_anchor": 0.10,
            },
            "optimization_scope": None,
        },
        "ch4_rbe_boundary_core_training_protocol_perf_v1": {
            "family": "perf",
            "ablation_id": None,
            "search_actors_frozen": True,
            "executor_only_actor_training": True,
            "sampling_ratios": {
                "boundary_core": 0.40,
                "uniform_coverage": 0.35,
                "composite_high_risk_aux": 0.05,
                "nominal_anchor": 0.20,
            },
            "optimization_scope": "current_method_performance_stabilization_only",
        },
        "ch4_rbe_perf_ablation_no_boundary_protocol_v1": {
            "family": "ablation",
            "ablation_id": "no_boundary_core",
            "search_actors_frozen": True,
            "executor_only_actor_training": True,
            "sampling_ratios": {
                "boundary_core": 0.00,
                "uniform_coverage": 0.75,
                "composite_high_risk_aux": 0.05,
                "nominal_anchor": 0.20,
            },
            "optimization_scope": "mechanism_ablation_only",
        },
        "ch4_rbe_perf_ablation_all_actors_protocol_v1": {
            "family": "ablation",
            "ablation_id": "all_actors_trainable",
            "search_actors_frozen": False,
            "executor_only_actor_training": False,
            "sampling_ratios": {
                "boundary_core": 0.40,
                "uniform_coverage": 0.35,
                "composite_high_risk_aux": 0.05,
                "nominal_anchor": 0.20,
            },
            "optimization_scope": "mechanism_ablation_only",
        },
        "ch4_rbe_perf_ablation_no_nominal_protocol_v1": {
            "family": "ablation",
            "ablation_id": "no_nominal_anchor",
            "search_actors_frozen": True,
            "executor_only_actor_training": True,
            "sampling_ratios": {
                "boundary_core": 0.40,
                "uniform_coverage": 0.55,
                "composite_high_risk_aux": 0.05,
                "nominal_anchor": 0.00,
            },
            "optimization_scope": "mechanism_ablation_only",
        },
    }
    contract = contracts.get(str(protocol_name))
    if contract is None:
        raise RuntimeError(
            f"unsupported RBE protocol manifest protocol_name: {protocol_name!r}"
        )
    return contract


def _audit_rbe_protocol_lock(
    *,
    mode,
    initialization,
    rbe_sampler_enable,
    collect_boundary_dataset,
    reb_enable,
    rscu_enable,
    freeze_search_actors,
    rbe_executor_only_actor,
    rbe_boundary_csv,
    rbe_high_risk_csv,
    rbe_boundary_ratio,
    rbe_uniform_ratio,
    rbe_high_risk_ratio,
    rbe_nominal_ratio,
    rbe_jitter_std,
    rbe_jitter_prob,
    rbe_reset_optimizer_state,
    rbe_actor_lr,
    rbe_critic_lr,
):
    manifest_path, manifest, manifest_sha = _load_rbe_protocol_manifest(os.environ.get("RBE_TRAINING_PROTOCOL_MANIFEST"))
    protocol_name = manifest.get("protocol_name")
    protocol_contract = _rbe_protocol_contract(protocol_name)
    legacy_protocol = protocol_contract["family"] == "legacy"
    perf_protocol = protocol_contract["family"] == "perf"
    perf_family_protocol = protocol_contract["family"] in {"perf", "ablation"}
    expected_freeze_search_actors = bool(protocol_contract["search_actors_frozen"])
    expected_executor_only_actor = bool(protocol_contract["executor_only_actor_training"])
    required_env = {
        "CH4_MODE": "ch4_rbe_full",
        "RBE_SAMPLER_ENABLE": "1",
        "REB_ENABLE": "0",
        "RSCU_ENABLE": "0",
        "FREEZE_SEARCH_ACTORS": "1" if expected_freeze_search_actors else "0",
        "RBE_EXECUTOR_ONLY_ACTOR": "1" if expected_executor_only_actor else "0",
        "COLLECT_BOUNDARY_DATASET": "1",
        "REWARD_PROFILE": "residual_point_v3",
        "RESIDUAL_ACTION_MODE": "hybrid_v4",
    }
    for name, expected in required_env.items():
        actual = os.environ.get(name)
        if str(actual) != expected:
            raise RuntimeError(f"RBE protocol lock requires {name}={expected}, got {actual!r}")
    if mode != "ch4_rbe_full":
        raise RuntimeError("RBE protocol lock requires CH4_MODE=ch4_rbe_full")
    if initialization["train_from_scratch"]:
        raise RuntimeError("RBE protocol lock forbids TRAIN_FROM_SCRATCH/from-scratch initialization")
    if not rbe_sampler_enable:
        raise RuntimeError("RBE protocol lock requires RBE_SAMPLER_ENABLE=1")
    if not collect_boundary_dataset:
        raise RuntimeError("RBE protocol lock requires COLLECT_BOUNDARY_DATASET=1")
    if reb_enable:
        raise RuntimeError("RBE protocol lock requires REB_ENABLE=0")
    if rscu_enable:
        raise RuntimeError("RBE protocol lock requires RSCU_ENABLE=0")
    if bool(freeze_search_actors) != expected_freeze_search_actors:
        expected = 1 if expected_freeze_search_actors else 0
        raise RuntimeError(f"RBE protocol lock requires FREEZE_SEARCH_ACTORS={expected}")
    if bool(rbe_executor_only_actor) != expected_executor_only_actor:
        expected = 1 if expected_executor_only_actor else 0
        raise RuntimeError(f"RBE protocol lock requires RBE_EXECUTOR_ONLY_ACTOR={expected}")

    expected_resume_path = (
        manifest.get("uniform_warm_start_model_path")
        if perf_family_protocol
        else str(
            get_selected_dir("uniform_dr", "ch4_uniform_dr_selected_full9d_v2")
            / "selected_by_uniform_dr_validation.pt"
        )
    )
    expected_resume = _resolve_project_path(expected_resume_path)
    if os.path.abspath(initialization["resume_model_path"]) != expected_resume:
        raise RuntimeError(f"RBE protocol lock requires warm-start checkpoint {expected_resume}")
    expected_uniform_sha = "47ae3748017c4c9a43efc16ecb9973b535716c0e09d119f2de43615b8e5405e7"
    if _sha256(expected_resume) != expected_uniform_sha:
        raise RuntimeError("RBE protocol lock Uniform warm-start SHA256 mismatch")

    if manifest.get("protocol_version") != 1:
        raise RuntimeError("RBE protocol manifest protocol_version mismatch")
    protocol_object_sha = manifest.get("protocol_object_sha256")
    computed_protocol_sha = _canonical_rbe_protocol_object_sha(manifest)
    if (
        not isinstance(protocol_object_sha, str)
        or protocol_object_sha != computed_protocol_sha
        or manifest.get("protocol_object_sha256_verified") is not True
    ):
        raise RuntimeError("RBE protocol manifest canonical object SHA verification failed")
    if manifest.get("uniform_warm_start_model_sha256") != expected_uniform_sha:
        raise RuntimeError("RBE protocol manifest Uniform warm-start SHA mismatch")
    if manifest.get("selected_reb_model_sha256") != "675688640fbfc0aadb6982a16baaeb6ab103e45f2bf630fc439d6d3b1ec412cd":
        raise RuntimeError("RBE protocol manifest selected REB model SHA mismatch")
    if manifest.get("source_selected_candidates_sha256") != "64f6be43307f903d218e54f136b823911da9e2bec58999a9eb710ff865c88f4e":
        raise RuntimeError("RBE protocol manifest selected-candidate source SHA mismatch")
    if manifest.get("source_score_rule_sha256") != "928eb37e02ad69f582a4f8f7016004acef51fd24b0140c4f06cab19c6cc0a1c9":
        raise RuntimeError("RBE protocol manifest score-rule SHA mismatch")
    expected_manifest_flags = {
        "boundary_enrichment_validated": True,
        "high_risk_primary_effects_validated": True,
        "high_risk_found_rate_guard_passed": False,
        "full_candidate_enrichment_supported": False,
        "high_risk_is_pure_execution_failure": False,
        "online_reb_training": False,
        "frozen_found_aware_reb_used_only_for_candidate_derivation": True,
        "rscu_enabled": False,
        "search_actors_frozen": expected_freeze_search_actors,
        "executor_only_actor_training": expected_executor_only_actor,
        "all_policy_actors_trainable": not expected_freeze_search_actors,
        "ratios_locked": True,
        "jitter_locked": True,
    }
    for field, expected in expected_manifest_flags.items():
        if manifest.get(field) is not expected:
            raise RuntimeError(f"RBE protocol manifest field mismatch: {field}")
    if manifest.get("boundary_training_role") != "validated_recoverable_boundary_core":
        raise RuntimeError("RBE protocol Boundary training role mismatch")
    if manifest.get("high_risk_training_role") != "low_ratio_composite_risk_augmentation":
        raise RuntimeError("RBE protocol High-risk training role mismatch")
    if manifest.get("warm_start_mode") != "ch4_uniform_dr" or manifest.get("warm_start_checkpoint") != "snapshot_ep1400":
        raise RuntimeError("RBE protocol warm-start identity mismatch")
    if manifest.get("policy_training_mode") != "ch4_rbe_full":
        raise RuntimeError("RBE protocol policy_training_mode mismatch")
    if not _float_equal(manifest.get("ratio_sum"), 1.0):
        raise RuntimeError("RBE protocol manifest ratio_sum mismatch")
    controller_contract = manifest.get("controller_contract") or {}
    if controller_contract.get("reward_profile") != "residual_point_v3":
        raise RuntimeError("RBE protocol controller reward_profile mismatch")
    if controller_contract.get("residual_action_mode") != "hybrid_v4":
        raise RuntimeError("RBE protocol controller residual_action_mode mismatch")
    if REWARD_PROFILE != "residual_point_v3" or RESIDUAL_ACTION_MODE != "hybrid_v4":
        raise RuntimeError("RBE protocol lock controller environment override mismatch")
    _validate_method_claims(manifest.get("method_statement_zh"), manifest.get("method_statement_en"))

    expected_ratios = dict(protocol_contract["sampling_ratios"])
    actual_ratios = {
        "boundary_core": rbe_boundary_ratio,
        "uniform_coverage": rbe_uniform_ratio,
        "composite_high_risk_aux": rbe_high_risk_ratio,
        "nominal_anchor": rbe_nominal_ratio,
    }
    manifest_ratios = manifest.get("sampling_ratios") or {}
    for key, expected in expected_ratios.items():
        if not _float_equal(actual_ratios[key], expected):
            raise RuntimeError(f"RBE protocol lock ratio mismatch: {key}={actual_ratios[key]} expected {expected}")
        if not _float_equal(manifest_ratios.get(key), expected):
            raise RuntimeError(f"RBE protocol manifest ratio mismatch: {key}")
    if not _float_equal(rbe_jitter_std, 0.02) or not _float_equal(rbe_jitter_prob, 0.50):
        raise RuntimeError("RBE protocol lock requires RBE_JITTER_STD=0.02 and RBE_JITTER_PROB=0.50")
    if not _float_equal(manifest.get("jitter_std"), 0.02) or not _float_equal(manifest.get("jitter_prob"), 0.50):
        raise RuntimeError("RBE protocol manifest jitter mismatch")
    if perf_family_protocol:
        expected_scope = protocol_contract["optimization_scope"]
        if manifest.get("optimization_scope") != expected_scope:
            raise RuntimeError("RBE perf-family protocol optimization_scope mismatch")
        expected_ablation_id = protocol_contract.get("ablation_id")
        actual_ablation_id = manifest.get("ablation_id")
        if expected_ablation_id is None:
            if actual_ablation_id not in (None, ""):
                raise RuntimeError("base Perf-RBE protocol must not declare an ablation_id")
        elif actual_ablation_id != expected_ablation_id:
            raise RuntimeError("RBE ablation protocol ablation_id mismatch")
        if manifest.get("new_algorithmic_module_added") is not False:
            raise RuntimeError("RBE perf-family protocol must not add an algorithmic module")
        if manifest.get("optimizer_state_reset") is not True or not rbe_reset_optimizer_state:
            raise RuntimeError("RBE perf-family protocol requires RBE_RESET_OPTIMIZER_STATE=1")
        if not _float_equal(manifest.get("actor_lr"), 1e-4) or not _float_equal(rbe_actor_lr, 1e-4):
            raise RuntimeError("RBE perf-family protocol requires RBE_ACTOR_LR=1e-4")
        if not _float_equal(manifest.get("critic_lr"), 3e-4) or not _float_equal(rbe_critic_lr, 3e-4):
            raise RuntimeError("RBE perf-family protocol requires RBE_CRITIC_LR=3e-4")
    elif rbe_reset_optimizer_state or rbe_actor_lr is not None or rbe_critic_lr is not None:
        raise RuntimeError("legacy RBE protocol v1 forbids perf optimizer overrides")

    boundary_csv = _resolve_project_path(rbe_boundary_csv)
    high_risk_csv = _resolve_project_path(rbe_high_risk_csv)
    manifest_boundary_csv = _resolve_project_path(manifest.get("boundary_candidate_csv"))
    manifest_high_risk_csv = _resolve_project_path(manifest.get("high_risk_candidate_csv"))
    if boundary_csv != manifest_boundary_csv or high_risk_csv != manifest_high_risk_csv:
        raise RuntimeError("RBE protocol candidate CSV paths do not match the manifest")
    if not os.path.exists(boundary_csv) or not os.path.exists(high_risk_csv):
        raise FileNotFoundError("RBE protocol lock candidate CSV missing")
    boundary_sha = _sha256(boundary_csv)
    high_risk_sha = _sha256(high_risk_csv)
    if boundary_sha != manifest.get("boundary_candidate_sha256"):
        raise RuntimeError("Boundary candidate CSV SHA mismatch")
    if high_risk_sha != manifest.get("high_risk_candidate_sha256"):
        raise RuntimeError("High-risk candidate CSV SHA mismatch")
    if manifest.get("candidate_counts", {}).get("boundary_core") != 64:
        raise RuntimeError("Boundary candidate count must be 64")
    if manifest.get("candidate_counts", {}).get("composite_high_risk_aux") != 64:
        raise RuntimeError("High-risk candidate count must be 64")

    source_hashes = manifest.get("source_hashes") or {}
    source_paths = manifest.get("source_paths") or {}
    expected_source_labels = {
        "v2_summary", "selected_candidates", "score_rule_manifest", "collection_state",
        "uniform_model", "uniform_manifest", "reb_model", "reb_manifest",
    }
    if set(source_hashes) != expected_source_labels or set(source_paths) != expected_source_labels:
        raise RuntimeError("RBE protocol source hash/path key set mismatch")
    for label in sorted(expected_source_labels):
        expected_sha = source_hashes[label]
        source_path = _resolve_project_path(source_paths[label])
        if not source_path or not os.path.isfile(source_path):
            raise FileNotFoundError(f"RBE protocol source missing: {label}")
        if _sha256(source_path) != expected_sha:
            raise RuntimeError(f"RBE protocol source hash changed: {label}")

    return {
        "rbe_protocol_lock": True,
        "rbe_training_protocol_manifest": manifest_path,
        "rbe_training_protocol_manifest_sha256": manifest_sha,
        "rbe_protocol_name": manifest.get("protocol_name"),
        "rbe_ablation_id": protocol_contract.get("ablation_id"),
        "rbe_protocol_object_sha256": protocol_object_sha,
        "rbe_method_name_zh": manifest.get("method_name_zh"),
        "rbe_method_name_en": manifest.get("method_name_en"),
        "rbe_method_statement_zh": manifest.get("method_statement_zh"),
        "rbe_method_statement_en": manifest.get("method_statement_en"),
        "rbe_boundary_training_role": manifest.get("boundary_training_role"),
        "rbe_high_risk_training_role": manifest.get("high_risk_training_role"),
        "rbe_high_risk_is_pure_execution_failure": manifest.get("high_risk_is_pure_execution_failure"),
        "rbe_boundary_csv_sha256": boundary_sha,
        "rbe_high_risk_csv_sha256": high_risk_sha,
        "rbe_boundary_candidate_count": 64,
        "rbe_high_risk_candidate_count": 64,
        "rbe_sampling_ratios": dict(actual_ratios),
        "rbe_expected_sampling_ratios": dict(expected_ratios),
        "rbe_raw_ratios": dict(expected_ratios),
        "rbe_effective_ratios": dict(expected_ratios),
        "rbe_ratios_were_normalized": False,
        "rbe_jitter_std": 0.02,
        "rbe_jitter_prob": 0.50,
        "resume_model_sha256": expected_uniform_sha,
        "rbe_uniform_warm_start_model_sha256": expected_uniform_sha,
        "optimizer_state_reset": bool(rbe_reset_optimizer_state),
        "actor_lr": rbe_actor_lr,
        "critic_lr": rbe_critic_lr,
        "online_reb_training": False,
        "frozen_reb_candidate_derivation": True,
        "full_candidate_enrichment_supported": manifest.get("full_candidate_enrichment_supported"),
        "boundary_enrichment_validated": manifest.get("boundary_enrichment_validated"),
        "high_risk_found_rate_guard_passed": manifest.get("high_risk_found_rate_guard_passed"),
        "manifest": manifest,
    }


def _sample_disturbance(mode, rng, rbe_sampler=None):
    if mode == "ch4_pse_baseline":
        return nominal_disturbance()
    if mode == "ch4_rbe_full" and rbe_sampler is not None:
        return rbe_sampler.sample()
    return sample_uniform_disturbance(rng=rng)


def _configure_ch4_env(env, env_kwargs, mode):
    use_dr = mode != "ch4_pse_baseline"
    env.use_robust_disturbance = bool(use_dr)
    env_kwargs["use_robust_disturbance"] = bool(use_dr)
    if env.num_agents != 4 or env.n_search != 3 or env.executor_idx != 3:
        raise RuntimeError("Chapter-4 RBE expects 4 agents: 3 searchers and 1 executor.")
    forbidden_true = [
        "use_semantic_messages",
        "use_voi_selector",
        "use_role_topology",
        "use_teammate_prediction",
        "use_reliability_fusion",
        "use_channel_attention",
        "use_upper_comm",
    ]
    active_forbidden = [name for name in forbidden_true if bool(getattr(env, name, False))]
    if active_forbidden:
        raise RuntimeError(f"Chapter-5 communication modules are still enabled: {active_forbidden}")
    if not bool(getattr(env, "use_pse_planner", False)):
        raise RuntimeError("Chapter-4 baseline must keep use_pse_planner=True.")
    return env, env_kwargs


def _load_or_init_maddpg(env, train_device, resume_model_path):
    if resume_model_path:
        if not os.path.exists(resume_model_path):
            raise FileNotFoundError(f"RESUME_MODEL_PATH does not exist: {resume_model_path}")
        return MADDPG.init_from_save(resume_model_path, device=train_device)
    return MADDPG.init_from_env(env, gamma=0.95, tau=5e-4, lr_actor=3e-4, lr_critic=3e-4, hidden_dim=128)


RBE_OPTIMIZER_NAMES = (
    "policy_optimizer",
    "critic1_optimizer",
    "critic2_optimizer",
    "rec_critic_optimizer",
    "safe_critic_optimizer",
)


def _positive_finite_lr(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite positive number, got {value!r}")
    return value


def _apply_rbe_finetune_optimizer_policy(
    maddpg,
    *,
    reset_optimizer_state=False,
    actor_lr=None,
    critic_lr=None,
):
    """Apply the opt-in RBE fine-tuning optimizer policy without touching weights."""
    effective_actor_lr = (
        _positive_finite_lr(actor_lr, "RBE_ACTOR_LR")
        if actor_lr is not None
        else _positive_finite_lr(maddpg.lr_actor, "maddpg.lr_actor")
    )
    effective_critic_lr = (
        _positive_finite_lr(critic_lr, "RBE_CRITIC_LR")
        if critic_lr is not None
        else _positive_finite_lr(maddpg.lr_critic, "maddpg.lr_critic")
    )
    per_agent = []
    for agent_i, agent in enumerate(maddpg.agents):
        optimizer_audit = {}
        for optimizer_name in RBE_OPTIMIZER_NAMES:
            optimizer = getattr(agent, optimizer_name)
            state_entries_before = len(optimizer.state)
            if reset_optimizer_state:
                optimizer.state.clear()
            lr = effective_actor_lr if optimizer_name == "policy_optimizer" else effective_critic_lr
            should_override_lr = (
                optimizer_name == "policy_optimizer" and actor_lr is not None
            ) or (
                optimizer_name != "policy_optimizer" and critic_lr is not None
            )
            if should_override_lr:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
            optimizer_audit[optimizer_name] = {
                "state_entries_before": int(state_entries_before),
                "state_entries_after": int(len(optimizer.state)),
                "param_group_lrs": [float(group["lr"]) for group in optimizer.param_groups],
            }
        per_agent.append({"agent_index": agent_i, "optimizers": optimizer_audit})

    if actor_lr is not None:
        maddpg.lr_actor = effective_actor_lr
        if isinstance(maddpg.init_dict, dict):
            maddpg.init_dict["lr_actor"] = effective_actor_lr
    if critic_lr is not None:
        maddpg.lr_critic = effective_critic_lr
        if isinstance(maddpg.init_dict, dict):
            maddpg.init_dict["lr_critic"] = effective_critic_lr

    return {
        "optimizer_state_reset_requested": bool(reset_optimizer_state),
        "optimizer_state_reset_applied": bool(reset_optimizer_state),
        "effective_actor_lr": effective_actor_lr,
        "effective_critic_lr": effective_critic_lr,
        "per_agent_optimizer_audit": per_agent,
    }


def _snapshot_actor_policies(maddpg):
    return [
        {name: tensor.detach().cpu().clone() for name, tensor in agent.policy.state_dict().items()}
        for agent in maddpg.agents
    ]


def _actor_policy_changed(before, policy):
    after = policy.state_dict()
    if set(before) != set(after):
        return True
    return any(not torch.equal(before[name], after[name].detach().cpu()) for name in before)


def _sample_replay(memory, batch_size, device):
    try:
        return memory.sample(batch_size, norm_rews=False, device=device)
    except TypeError:
        return memory.sample(batch_size, to_gpu=(device.type == "cuda"), norm_rews=False)


def train_ch4():
    mode = os.environ.get("CH4_MODE", os.environ.get("RBE_MODE", "ch4_pse_baseline")).strip()
    if mode in ("clean", "baseline", "pse_baseline"):
        mode = "ch4_pse_baseline"
    elif mode in ("uniform", "uniform_dr"):
        mode = "ch4_uniform_dr"
    elif mode in ("reb", "reb_only"):
        mode = "ch4_reb_only"
    elif mode in ("rbe", "rbe_full"):
        mode = "ch4_rbe_full"
    if mode not in CH4_RBE_MODES:
        raise ValueError(f"Unknown CH4_MODE={mode}. Expected one of {CH4_RBE_MODES}")

    seed = int(os.environ.get("TRAIN_SEED", os.environ.get("SEED", "2")))
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_device = torch.device("cpu")
    max_episodes = int(os.environ.get("MAX_EPISODES", "1000"))
    max_steps = int(os.environ.get("MAX_STEPS", "400"))
    batch_size = int(os.environ.get("BATCH_SIZE", "256"))
    buffer_size = int(float(os.environ.get("BUFFER_SIZE", "5e5")))
    learn_interval = int(os.environ.get("LEARN_INTERVAL", "50"))
    updates_per_train = int(os.environ.get("UPDATES_PER_TRAIN", "2"))
    print_interval = int(os.environ.get("PRINT_INTERVAL", "20"))
    snapshot_interval = int(os.environ.get("SNAPSHOT_INTERVAL", "200"))
    default_collect_boundary_dataset = mode in (
        "ch4_uniform_dr",
        "ch4_reb_only",
        "ch4_rbe_full",
    )
    collect_boundary_dataset = _env_bool(
        "COLLECT_BOUNDARY_DATASET",
        default_collect_boundary_dataset,
    )
    boundary_dataset_flush_interval = max(
        1,
        _env_int(
            "BOUNDARY_DATASET_FLUSH_INTERVAL",
            snapshot_interval if snapshot_interval > 0 else 200,
        ),
    )

    initialization = _resolve_training_initialization(mode)
    resume_model_path = initialization["resume_model_path"]

    reb_enable = _env_bool("REB_ENABLE", mode in ("ch4_reb_only", "ch4_rbe_full"))
    reb_min_samples = int(os.environ.get("REB_MIN_SAMPLES", "16"))
    reb_batch_size = int(os.environ.get("REB_BATCH_SIZE", "64"))
    reb_updates_per_episode = int(os.environ.get("REB_UPDATES_PER_EPISODE", "4"))
    if reb_enable and not collect_boundary_dataset:
        raise RuntimeError("REB_ENABLE=1 requires COLLECT_BOUNDARY_DATASET=1.")
    rscu_enable = _env_bool("RSCU_ENABLE", False)
    rscu_agent_scope = os.environ.get("RSCU_AGENT_SCOPE", "executor").strip().lower()
    rscu_lambda_rec_init = _env_float("RSCU_LAMBDA_REC_INIT", 0.05)
    rscu_lambda_safe_init = _env_float("RSCU_LAMBDA_SAFE_INIT", 0.02)
    rscu_lambda_smooth = _env_float("RSCU_LAMBDA_SMOOTH", 0.001)
    rscu_eta_rec = _env_float("RSCU_ETA_REC", 0.005)
    rscu_eta_safe = _env_float("RSCU_ETA_SAFE", 0.005)
    rscu_t_limit = _env_float("RSCU_T_LIMIT", 140.0)
    rscu_c_limit = _env_float("RSCU_C_LIMIT", 5.0)
    rscu_lambda_max = _env_float("RSCU_LAMBDA_MAX", 2.0)
    rscu_near_target_radius = _env_float("RSCU_NEAR_TARGET_RADIUS", 6.0)
    rscu_postfound_only = _env_bool("RSCU_POSTFOUND_ONLY", False)
    rscu_mask_mode = os.environ.get("RSCU_MASK_MODE", "all").strip().lower()
    rscu_min_mask_count = _env_int("RSCU_MIN_MASK_COUNT", 4)
    rscu_aux_critic_masked = _env_bool("RSCU_AUX_CRITIC_MASKED", True)
    rscu_lambda_update_on_found_only = _env_bool("RSCU_LAMBDA_UPDATE_ON_FOUND_ONLY", False)
    current_lambda_rec = float(rscu_lambda_rec_init)
    current_lambda_safe = float(rscu_lambda_safe_init)

    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = os.environ.get("RUN_NAME", f"{mode}_seed{seed}_{run_tag}")
    artifact_family = _default_artifact_family(mode)
    default_run_root = str(get_run_root(artifact_family))
    log_root = os.environ.get("LOG_ROOT", default_run_root)
    model_root = os.environ.get("MODEL_ROOT", default_run_root)
    if not os.path.isabs(log_root):
        log_root = os.path.join(PROJECT_ROOT, log_root)
    if not os.path.isabs(model_root):
        model_root = os.path.join(PROJECT_ROOT, model_root)
    log_dir = os.path.join(log_root, run_name)
    model_dir = os.path.join(model_root, run_name)
    training_stage = os.environ.get("TRAINING_STAGE", "unspecified").strip().lower() or "unspecified"

    rbe_sampler = None
    rbe_sampler_enable = mode == "ch4_rbe_full" and _env_bool("RBE_SAMPLER_ENABLE", True)
    rbe_boundary_csv = os.environ.get(
        "RBE_BOUNDARY_CSV",
        str(
            get_training_run_dir("reb", "ch4_reb_only_seed1_from_uniform2000_v2")
            / "reb_boundary_candidates.csv"
        ),
    )
    rbe_high_risk_csv = os.environ.get(
        "RBE_HIGH_RISK_CSV",
        str(
            get_training_run_dir("reb", "ch4_reb_only_seed1_from_uniform2000_v2")
            / "reb_high_risk_candidates.csv"
        ),
    )
    rbe_boundary_ratio = float(os.environ.get("RBE_BOUNDARY_RATIO", "0.20"))
    rbe_uniform_ratio = float(os.environ.get("RBE_UNIFORM_RATIO", "0.60"))
    rbe_high_risk_ratio = float(os.environ.get("RBE_HIGH_RISK_RATIO", "0.10"))
    rbe_nominal_ratio = float(os.environ.get("RBE_NOMINAL_RATIO", "0.10"))
    rbe_jitter_std = float(os.environ.get("RBE_JITTER_STD", "0.02"))
    rbe_jitter_prob = float(os.environ.get("RBE_JITTER_PROB", "0.50"))
    rbe_executor_only_actor = _env_bool("RBE_EXECUTOR_ONLY_ACTOR", False)
    freeze_search_actors = _env_bool("FREEZE_SEARCH_ACTORS", False) or rbe_executor_only_actor
    rbe_reset_optimizer_state = _env_bool("RBE_RESET_OPTIMIZER_STATE", False)
    rbe_actor_lr = _env_float("RBE_ACTOR_LR", 0.0) if "RBE_ACTOR_LR" in os.environ else None
    rbe_critic_lr = _env_float("RBE_CRITIC_LR", 0.0) if "RBE_CRITIC_LR" in os.environ else None
    rbe_optimizer_policy_requested = (
        rbe_reset_optimizer_state or rbe_actor_lr is not None or rbe_critic_lr is not None
    )
    if rbe_optimizer_policy_requested and mode != "ch4_rbe_full":
        raise RuntimeError("RBE optimizer reset/LR overrides are only valid with CH4_MODE=ch4_rbe_full")
    rbe_protocol_lock_requested = _env_bool("RBE_PROTOCOL_LOCK", False)
    if rbe_protocol_lock_requested and mode != "ch4_rbe_full":
        raise RuntimeError("RBE_PROTOCOL_LOCK=1 is only valid with CH4_MODE=ch4_rbe_full")
    rbe_protocol_lock = bool(rbe_protocol_lock_requested)
    rbe_protocol_audit = None
    if rbe_protocol_lock:
        rbe_protocol_audit = _audit_rbe_protocol_lock(
            mode=mode,
            initialization=initialization,
            rbe_sampler_enable=rbe_sampler_enable,
            collect_boundary_dataset=collect_boundary_dataset,
            reb_enable=reb_enable,
            rscu_enable=rscu_enable,
            freeze_search_actors=freeze_search_actors,
            rbe_executor_only_actor=rbe_executor_only_actor,
            rbe_boundary_csv=rbe_boundary_csv,
            rbe_high_risk_csv=rbe_high_risk_csv,
            rbe_boundary_ratio=rbe_boundary_ratio,
            rbe_uniform_ratio=rbe_uniform_ratio,
            rbe_high_risk_ratio=rbe_high_risk_ratio,
            rbe_nominal_ratio=rbe_nominal_ratio,
            rbe_jitter_std=rbe_jitter_std,
            rbe_jitter_prob=rbe_jitter_prob,
            rbe_reset_optimizer_state=rbe_reset_optimizer_state,
            rbe_actor_lr=rbe_actor_lr,
            rbe_critic_lr=rbe_critic_lr,
        )
    if rbe_sampler_enable:
        rbe_sampler = RBEDisturbanceSampler(
            boundary_csv=rbe_boundary_csv,
            high_risk_csv=rbe_high_risk_csv,
            rng=rng,
            boundary_ratio=rbe_boundary_ratio,
            uniform_ratio=rbe_uniform_ratio,
            high_risk_ratio=rbe_high_risk_ratio,
            nominal_ratio=rbe_nominal_ratio,
            jitter_std=rbe_jitter_std,
            jitter_prob=rbe_jitter_prob,
            include_flow_phase=True,
            strict_protocol=bool(rbe_protocol_lock),
            expected_boundary_count=64 if rbe_protocol_lock else None,
            expected_high_risk_count=64 if rbe_protocol_lock else None,
            expected_boundary_group="reb_boundary" if rbe_protocol_lock else None,
            expected_high_risk_group="reb_high_risk" if rbe_protocol_lock else None,
            require_ratio_sum_one=bool(rbe_protocol_lock),
            high_risk_required=bool(rbe_protocol_lock),
            method_role_labels={
                "boundary": "boundary_core",
                "high_risk": "composite_high_risk_aux",
                "uniform": "uniform_coverage",
                "nominal": "nominal_anchor",
            } if rbe_protocol_lock else None,
            score_rule_sha256=(rbe_protocol_audit["manifest"].get("source_score_rule_sha256") if rbe_protocol_audit else None),
            source_selected_candidates_sha256=(rbe_protocol_audit["manifest"].get("source_selected_candidates_sha256") if rbe_protocol_audit else None),
        )
        if rbe_protocol_lock and rbe_protocol_audit is not None:
            sampler_audit = rbe_sampler.audit()
            if sampler_audit["boundary_csv_sha256"] != rbe_protocol_audit["rbe_boundary_csv_sha256"]:
                raise RuntimeError("strict RBE sampler boundary CSV SHA mismatch")
            if sampler_audit["high_risk_csv_sha256"] != rbe_protocol_audit["rbe_high_risk_csv_sha256"]:
                raise RuntimeError("strict RBE sampler high-risk CSV SHA mismatch")
            rbe_protocol_audit["rbe_raw_ratios"] = sampler_audit["raw_ratios"]
            rbe_protocol_audit["rbe_effective_ratios"] = sampler_audit["effective_ratios"]
            rbe_protocol_audit["rbe_ratios_were_normalized"] = sampler_audit["ratios_were_normalized"]

    env_cfg, _ = get_ablation_config(mode)
    env, env_kwargs = _build_train_env(env_device, max_steps, env_cfg)
    env, env_kwargs = _configure_ch4_env(env, env_kwargs, mode)
    controller_config = {
        "prior_strength_search": float(env.prior_strength_search),
        "prior_strength_executor": float(env.prior_strength_executor),
        "residual_scale_search": float(env.residual_scale_search),
        "residual_scale_executor": float(env.residual_scale_executor),
        "residual_penalty": float(env.residual_penalty),
        "reward_profile": str(env.reward_profile),
        "residual_action_mode": str(env.residual_action_mode),
        "residual_hybrid_use_soft_gate": bool(env.residual_hybrid_use_soft_gate),
        "use_pse_planner": bool(env.use_pse_planner),
        "pse_use_belief": bool(env.pse_use_belief),
        "pse_use_exec_cost": bool(env.pse_use_exec_cost),
        "pse_use_standby": bool(env.pse_use_standby),
        "pse_lazy_standby": bool(env.pse_lazy_standby),
        "use_robust_disturbance": bool(env.use_robust_disturbance),
    }

    n_agents = env.num_agents
    obs_dims = [env.observation_space[f"agent_{i}"].shape[0] for i in range(n_agents)]
    ac_dims = [env.action_space[f"agent_{i}"].shape[0] for i in range(n_agents)]
    space_diag = float(torch.norm(env.space_size).item()) if torch.is_tensor(env.space_size) else float(np.linalg.norm(env.space_size))

    maddpg = _load_or_init_maddpg(env, train_device, resume_model_path)
    optimizer_policy_audit = _apply_rbe_finetune_optimizer_policy(
        maddpg,
        reset_optimizer_state=rbe_reset_optimizer_state if mode == "ch4_rbe_full" else False,
        actor_lr=rbe_actor_lr if mode == "ch4_rbe_full" else None,
        critic_lr=rbe_critic_lr if mode == "ch4_rbe_full" else None,
    )
    actor_policy_start = _snapshot_actor_policies(maddpg)

    # Locked formal/smoke runs are write-once.  Delay directory creation until
    # protocol, source files, sampler, environment, checkpoint loading and
    # optimizer policy have all passed preflight.  This prevents a configuration
    # error from leaving empty directories that block a clean rerun.
    create_training_run_directories(
        log_dir,
        model_dir,
        write_once=bool(rbe_protocol_lock),
    )

    maddpg.prep_rollouts(device=train_device)
    memory = ReplayBuffer(
        buffer_size,
        n_agents,
        obs_dims,
        ac_dims,
        storage_device=torch.device("cpu"),
    )
    tracker = EpisodeMetricTracker()
    boundary_dataset = None
    if collect_boundary_dataset:
        boundary_dataset = REBOutcomeDataset(max_steps=max_steps, space_diag=space_diag)
    reb_trainer = None
    if reb_enable:
        reb_trainer = REBTrainer(REBModel(xi_dim=len(DISTURBANCE_KEYS)), device=train_device)

    with open(os.path.join(log_dir, "training_config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_name": run_name,
                "training_stage": training_stage,
                "log_dir": os.path.abspath(log_dir),
                "model_dir": os.path.abspath(model_dir),
                "mode": mode,
                "baseline": "pse_lazy_standby_54",
                "train_from_scratch_requested": initialization["train_from_scratch_requested"],
                "resume_model_path_raw": initialization["resume_model_path_raw"],
                "resume_model_path": resume_model_path,
                "initialization_source": initialization["initialization_source"],
                "loaded_checkpoint": initialization["loaded_checkpoint"],
                "seed": seed,
                "train_device": str(train_device),
                "env_device": str(env_device),
                "max_episodes": max_episodes,
                "max_steps": max_steps,
                "batch_size": batch_size,
                "buffer_size": buffer_size,
                "learn_interval": learn_interval,
                "updates_per_train": updates_per_train,
                "print_interval": print_interval,
                "snapshot_interval": snapshot_interval,
                "collect_boundary_dataset": bool(collect_boundary_dataset),
                "boundary_dataset_flush_interval": boundary_dataset_flush_interval,
                "disturbance_keys": DISTURBANCE_KEYS,
                "disturbance_bounds": DEFAULT_DISTURBANCE_BOUNDS,
                "reb_enable": reb_enable,
                "rbe_sampler_enable": bool(rbe_sampler_enable),
                "rbe_boundary_csv": rbe_boundary_csv,
                "rbe_high_risk_csv": rbe_high_risk_csv,
                "rbe_boundary_ratio": rbe_boundary_ratio,
                "rbe_uniform_ratio": rbe_uniform_ratio,
                "rbe_high_risk_ratio": rbe_high_risk_ratio,
                "rbe_nominal_ratio": rbe_nominal_ratio,
                "rbe_jitter_std": rbe_jitter_std,
                "rbe_jitter_prob": rbe_jitter_prob,
                "rbe_protocol_lock": bool(rbe_protocol_lock),
                **({key: value for key, value in rbe_protocol_audit.items() if key != "manifest"} if rbe_protocol_audit else {
                    "rbe_training_protocol_manifest": "",
                    "rbe_training_protocol_manifest_sha256": "",
                    "rbe_protocol_name": "",
                    "rbe_protocol_object_sha256": "",
                    "rbe_method_name_zh": "",
                    "rbe_method_name_en": "",
                    "rbe_method_statement_zh": "",
                    "rbe_method_statement_en": "",
                    "rbe_boundary_training_role": "",
                    "rbe_high_risk_training_role": "",
                    "rbe_high_risk_is_pure_execution_failure": "",
                    "rbe_boundary_csv_sha256": "",
                    "rbe_high_risk_csv_sha256": "",
                    "rbe_boundary_candidate_count": "",
                    "rbe_high_risk_candidate_count": "",
                    "rbe_sampling_ratios": {},
                    "rbe_expected_sampling_ratios": {},
                    "rbe_raw_ratios": {},
                    "rbe_effective_ratios": {},
                    "rbe_ratios_were_normalized": "",
                    "resume_model_sha256": "",
                    "rbe_uniform_warm_start_model_sha256": "",
                    "online_reb_training": bool(reb_enable),
                    "frozen_reb_candidate_derivation": "",
                    "full_candidate_enrichment_supported": "",
                    "boundary_enrichment_validated": "",
                    "high_risk_found_rate_guard_passed": "",
                }),
                "rbe_executor_only_actor": bool(rbe_executor_only_actor),
                "freeze_search_actors": bool(freeze_search_actors),
                "optimizer_state_reset_requested": bool(rbe_reset_optimizer_state),
                "optimizer_state_reset_applied": optimizer_policy_audit["optimizer_state_reset_applied"],
                "rbe_optimizer_policy_requested": bool(rbe_optimizer_policy_requested),
                "effective_actor_lr": optimizer_policy_audit["effective_actor_lr"],
                "effective_critic_lr": optimizer_policy_audit["effective_critic_lr"],
                "per_agent_optimizer_audit": optimizer_policy_audit["per_agent_optimizer_audit"],
                "rscu_enable": bool(rscu_enable),
                "rscu_agent_scope": rscu_agent_scope,
                "rscu_lambda_rec_init": rscu_lambda_rec_init,
                "rscu_lambda_safe_init": rscu_lambda_safe_init,
                "rscu_lambda_smooth": rscu_lambda_smooth,
                "rscu_eta_rec": rscu_eta_rec,
                "rscu_eta_safe": rscu_eta_safe,
                "rscu_t_limit": rscu_t_limit,
                "rscu_c_limit": rscu_c_limit,
                "rscu_lambda_max": rscu_lambda_max,
                "rscu_near_target_radius": rscu_near_target_radius,
                "rscu_postfound_only": bool(rscu_postfound_only),
                "rscu_mask_mode": rscu_mask_mode,
                "rscu_min_mask_count": rscu_min_mask_count,
                "rscu_aux_critic_masked": bool(rscu_aux_critic_masked),
                "rscu_lambda_update_on_found_only": bool(rscu_lambda_update_on_found_only),
                "controller_config": controller_config,
                "env_kwargs": env_kwargs,
                "obs_dims": obs_dims,
                "ac_dims": ac_dims,
            },
            f,
            indent=2,
            default=str,
        )

    print(f"CH4_MODE={mode} | PSE baseline retained")
    print("[ControllerConfig] %s" % json.dumps(controller_config, sort_keys=True))
    print(f"[Initialization] source={initialization['initialization_source']}")
    if initialization["loaded_checkpoint"]:
        print(f"[Initialization] checkpoint={resume_model_path}")
    else:
        print("[Initialization] model parameters initialized from scratch")
    print(f"obs_dims={obs_dims} | ac_dims={ac_dims} | log_dir={log_dir} | model_dir={model_dir}")

    reward_log, success_log, found_log = [], [], []
    episode_rows, reb_loss_rows = [], []
    total_steps = 0
    optimizer_update_count = 0
    per_agent_actor_update_count = [0 for _ in range(n_agents)]
    per_agent_critic_update_count = [0 for _ in range(n_agents)]
    last_reb_loss = None
    boundary_dataset_last_saved_episode = 0

    for episode in range(1, max_episodes + 1):
        xi = _sample_disturbance(mode, rng, rbe_sampler=rbe_sampler)
        env.set_next_disturbance(xi)
        obs = env.reset()
        actual_xi = env.get_current_disturbance()
        tracker.reset(env, actual_xi)
        maddpg.reset_noise()
        actor_losses, critic_losses = [], []
        rscu_rec_critic_losses, rscu_safe_critic_losses = [], []
        rscu_actor_rec_penalties, rscu_actor_safe_penalties, rscu_actor_smooth_penalties = [], [], []
        rscu_post_found_fracs, rscu_near_target_fracs = [], []
        rscu_aux_mask_fracs, rscu_aux_mask_counts, rscu_actor_applied_flags = [], [], []

        for _step in range(1, env.max_steps + 1):
            maddpg.prep_rollouts(device=train_device)
            task_found_t = bool(getattr(env, "task_found", False))
            executor_assigned_t = bool(getattr(env, "executor_target_assigned", False))
            executor_nav_distance_t = _executor_nav_distance(env)
            post_found_t = bool(task_found_t and executor_assigned_t)
            near_target_t = bool(
                post_found_t
                and np.isfinite(float(executor_nav_distance_t))
                and float(executor_nav_distance_t) <= float(rscu_near_target_radius)
            )
            actions = maddpg.step(obs, explore=True)
            env_actions = _to_env_actions(actions, env_device)
            next_obs, rewards, dones = env.step(env_actions)
            rewards_t = rewards.detach().to(dtype=torch.float32) if torch.is_tensor(rewards) else torch.as_tensor(rewards, dtype=torch.float32)
            step_costs = tracker.step(env, env_actions, rewards_t, dones)
            memory.push(
                obs,
                env_actions,
                rewards_t,
                next_obs,
                dones,
                _success_flags(env, dones),
                recovery_cost=step_costs["recovery_cost_t"],
                safety_cost=step_costs["safety_cost_t"],
                smooth_cost=step_costs["smooth_cost_t"],
                task_found=task_found_t,
                executor_assigned=executor_assigned_t,
                post_found=post_found_t,
                near_target=near_target_t,
                executor_nav_distance=executor_nav_distance_t if np.isfinite(float(executor_nav_distance_t)) else 0.0,
            )
            obs = next_obs
            total_steps += 1

            if total_steps % learn_interval == 0 and len(memory) >= batch_size:
                maddpg.prep_training(device=train_device)
                for _ in range(updates_per_train):
                    sample = _sample_replay(memory, batch_size, train_device)
                    last_td_error = None
                    for agent_i in range(n_agents):
                        update_actor = not (freeze_search_actors and agent_i < int(getattr(env, "n_search", 3)))
                        if rscu_enable:
                            critic_loss, actor_loss, last_td_error, aux = maddpg.update_rscu(
                                sample,
                                agent_i,
                                lambda_rec=current_lambda_rec,
                                lambda_safe=current_lambda_safe,
                                lambda_smooth=rscu_lambda_smooth,
                                rscu_actor_scope=rscu_agent_scope,
                                executor_idx=int(getattr(env, "executor_idx", 3)),
                                update_actor=update_actor,
                                rscu_postfound_only=rscu_postfound_only,
                                rscu_mask_mode=rscu_mask_mode,
                                rscu_min_mask_count=rscu_min_mask_count,
                                rscu_aux_critic_masked=rscu_aux_critic_masked,
                            )
                            rscu_rec_critic_losses.append(aux.get("rec_critic_loss", np.nan))
                            rscu_safe_critic_losses.append(aux.get("safe_critic_loss", np.nan))
                            rscu_post_found_fracs.append(aux.get("post_found_frac", np.nan))
                            rscu_near_target_fracs.append(aux.get("near_target_frac", np.nan))
                            rscu_aux_mask_fracs.append(aux.get("aux_mask_frac", np.nan))
                            rscu_aux_mask_counts.append(aux.get("aux_mask_count", np.nan))
                            rscu_actor_applied_flags.append(1.0 if aux.get("rscu_actor_applied", False) else 0.0)
                            if actor_loss is not None:
                                rscu_actor_rec_penalties.append(aux.get("actor_rec_penalty", np.nan))
                                rscu_actor_safe_penalties.append(aux.get("actor_safe_penalty", np.nan))
                                rscu_actor_smooth_penalties.append(aux.get("actor_smooth_penalty", np.nan))
                        elif not update_actor:
                            critic_loss, last_td_error = maddpg.update_critic_only(sample, agent_i)
                            actor_loss = None
                        else:
                            critic_loss, actor_loss, last_td_error = maddpg.update(sample, agent_i)
                        critic_losses.append(_loss_to_float(critic_loss))
                        per_agent_critic_update_count[agent_i] += 1
                        if actor_loss is not None:
                            actor_losses.append(_loss_to_float(actor_loss))
                            per_agent_actor_update_count[agent_i] += 1
                    maddpg.update_all_targets()
                    optimizer_update_count += 1
                    if last_td_error is not None:
                        memory.update_priorities(sample[6], last_td_error, sample[7])
                maddpg.prep_rollouts(device=train_device)

            if all(bool(x) for x in dones):
                break

        ep_metrics = tracker.finalize(env)
        ep_metrics["episode"] = int(episode)
        episode_rows.append(ep_metrics)
        if collect_boundary_dataset:
            boundary_dataset.append(ep_metrics)
            should_flush_boundary = (
                episode % boundary_dataset_flush_interval == 0
                or episode == max_episodes
            )
            if should_flush_boundary:
                boundary_dataset.save_csv(os.path.join(log_dir, "boundary_dataset.csv"))
                boundary_dataset_last_saved_episode = episode

        if (
            reb_enable
            and reb_trainer is not None
            and boundary_dataset is not None
            and len(boundary_dataset) >= reb_min_samples
        ):
            last_reb_loss = reb_trainer.update(boundary_dataset, batch_size=reb_batch_size, updates=reb_updates_per_episode, rng=rng)
            if last_reb_loss is not None:
                reb_loss_rows.append({"episode": episode, **last_reb_loss})
                _write_rows_csv(
                    os.path.join(log_dir, "reb_loss_log.csv"),
                    reb_loss_rows,
                    ("episode", "reb_loss", "reb_bce", "reb_t_mse", "reb_safe_mse", "reb_dist_mse"),
                )

        avg_reward = float(ep_metrics["episode_reward_mean"])
        success_v = 1.0 if ep_metrics["success_flag"] else 0.0
        found_v = 1.0 if ep_metrics["found_flag"] else 0.0
        reward_log.append(avg_reward)
        success_log.append(success_v)
        found_log.append(found_v)
        if rscu_enable and ((not rscu_lambda_update_on_found_only) or bool(ep_metrics["found_flag"])):
            current_lambda_rec = float(
                np.clip(
                    current_lambda_rec + rscu_eta_rec * ((float(ep_metrics["recovery_time"]) - rscu_t_limit) / max(1, max_steps)),
                    0.0,
                    rscu_lambda_max,
                )
            )
            current_lambda_safe = float(
                np.clip(
                    current_lambda_safe + rscu_eta_safe * ((float(ep_metrics["safety_cost"]) - rscu_c_limit) / max(rscu_c_limit, 1e-6)),
                    0.0,
                    rscu_lambda_max,
                )
            )

        _append_csv(
            os.path.join(log_dir, "training_progress.csv"),
            {
                "episode": episode,
                "episode_steps": int(ep_metrics["completion_steps"]),
                "avg_reward": avg_reward,
                "success": success_v,
                "found": found_v,
                "coverage": _coverage(env),
                "finished_ratio": _finished_ratio(env),
                "actor_loss": float(np.mean(actor_losses)) if actor_losses else "",
                "critic_loss": float(np.mean(critic_losses)) if critic_losses else "",
                "disturbance_mode": mode,
                "disturbance_level": actual_xi.get("flow_gain", np.nan),
                "rbe_sample_source": getattr(rbe_sampler, "last_source", "") if rbe_sampler else "",
                "rbe_sample_rank": getattr(rbe_sampler, "last_rank", "") if rbe_sampler else "",
                "rbe_method_role": getattr(rbe_sampler, "last_method_role", "") if rbe_sampler else "",
                "rbe_candidate_pool_index": getattr(rbe_sampler, "last_candidate_pool_index", "") if rbe_sampler else "",
                "rbe_jitter_applied": getattr(rbe_sampler, "last_jitter_applied", "") if rbe_sampler else "",
                "rbe_jitter_l2": getattr(rbe_sampler, "last_jitter_l2", "") if rbe_sampler else "",
                "rbe_action_delay_before_jitter": getattr(rbe_sampler, "last_action_delay_before_jitter", "") if rbe_sampler else "",
                "rbe_action_delay_after_jitter": getattr(rbe_sampler, "last_action_delay_after_jitter", "") if rbe_sampler else "",
                "optimizer_update_count_cumulative": optimizer_update_count,
                "per_agent_actor_update_count": json.dumps(per_agent_actor_update_count),
                "per_agent_critic_update_count": json.dumps(per_agent_critic_update_count),
                "recovery_time": ep_metrics["recovery_time"],
                "safety_cost": ep_metrics["safety_cost"],
                "final_distance": ep_metrics["final_distance"],
                "final_nav_distance": ep_metrics["final_nav_distance"],
                "action_smoothness": ep_metrics["action_smoothness"],
                "reb_loss": "" if last_reb_loss is None else last_reb_loss.get("reb_loss", ""),
                "rscu_lambda_rec": current_lambda_rec if rscu_enable else "",
                "rscu_lambda_safe": current_lambda_safe if rscu_enable else "",
                "rscu_rec_critic_loss": float(np.mean(rscu_rec_critic_losses)) if rscu_rec_critic_losses else "",
                "rscu_safe_critic_loss": float(np.mean(rscu_safe_critic_losses)) if rscu_safe_critic_losses else "",
                "rscu_actor_rec_penalty": float(np.mean(rscu_actor_rec_penalties)) if rscu_actor_rec_penalties else "",
                "rscu_actor_safe_penalty": float(np.mean(rscu_actor_safe_penalties)) if rscu_actor_safe_penalties else "",
                "rscu_actor_smooth_penalty": float(np.mean(rscu_actor_smooth_penalties)) if rscu_actor_smooth_penalties else "",
                "rscu_post_found_sample_frac": float(np.nanmean(rscu_post_found_fracs)) if rscu_post_found_fracs else "",
                "rscu_near_target_sample_frac": float(np.nanmean(rscu_near_target_fracs)) if rscu_near_target_fracs else "",
                "rscu_aux_mask_frac": float(np.nanmean(rscu_aux_mask_fracs)) if rscu_aux_mask_fracs else "",
                "rscu_aux_mask_count": float(np.nanmean(rscu_aux_mask_counts)) if rscu_aux_mask_counts else "",
                "rscu_actor_applied": float(np.nanmax(rscu_actor_applied_flags)) if rscu_actor_applied_flags else "",
            },
            TRAINING_PROGRESS_FIELDS,
        )

        if episode == 1 or episode % print_interval == 0 or episode == max_episodes:
            window = min(print_interval, len(reward_log))
            print(
                f"[Ep {episode}/{max_episodes}] "
                f"avgR={_mean_tail(reward_log, window):.3f} "
                f"success={_mean_tail(success_log, window) * 100.0:.2f}% "
                f"found={_mean_tail(found_log, window) * 100.0:.2f}% "
                f"recovery={ep_metrics['recovery_time']} "
                f"safety={ep_metrics['safety_cost']:.3f} "
                f"final_dist={ep_metrics['final_distance']:.3f}"
                + (f" rscu={current_lambda_rec:.3f}/{current_lambda_safe:.3f}" if rscu_enable else "")
            )

        if snapshot_interval > 0 and episode % snapshot_interval == 0:
            maddpg.save(os.path.join(model_dir, f"snapshot_ep{episode:04d}.pt"))

    if (
        collect_boundary_dataset
        and boundary_dataset is not None
        and len(boundary_dataset) > 0
        and boundary_dataset_last_saved_episode != max_episodes
    ):
        boundary_dataset.save_csv(os.path.join(log_dir, "boundary_dataset.csv"))

    _write_rows_csv(
        os.path.join(log_dir, "episode_metrics.csv"),
        episode_rows,
        ("episode", *DISTURBANCE_KEYS, "success_flag", "found_flag", "episode_reward_mean", "episode_reward_sum",
         "completion_steps", "recovery_time", "safety_cost", "safety_cost_mean", "final_distance",
         "final_nav_distance", "action_smoothness"),
    )
    actor_policy_changed_by_agent = [
        _actor_policy_changed(actor_policy_start[agent_i], maddpg.agents[agent_i].policy)
        for agent_i in range(n_agents)
    ]
    search_actor_policy_changed = any(
        actor_policy_changed_by_agent[: int(getattr(env, "n_search", 3))]
    )
    executor_i = int(getattr(env, "executor_idx", 3))
    executor_actor_policy_changed = actor_policy_changed_by_agent[executor_i]
    perf_freeze_audit_required = bool(
        rbe_protocol_audit
        and rbe_protocol_audit.get("rbe_protocol_name")
        == "ch4_rbe_boundary_core_training_protocol_perf_v1"
    )
    if perf_freeze_audit_required:
        if search_actor_policy_changed:
            raise RuntimeError("RBE perf freeze audit failed: a search actor policy changed")
        if not executor_actor_policy_changed:
            raise RuntimeError("RBE perf freeze audit failed: executor actor policy did not change")
        if per_agent_actor_update_count[:executor_i] != [0] * executor_i:
            raise RuntimeError("RBE perf freeze audit failed: search actor update count is nonzero")
        if per_agent_actor_update_count[executor_i] <= 0:
            raise RuntimeError("RBE perf freeze audit failed: executor actor update count is zero")
        if any(count <= 0 for count in per_agent_critic_update_count):
            raise RuntimeError("RBE perf freeze audit failed: a critic update count is zero")
    final_model_path = os.path.join(model_dir, "maddpg_uavenv_final.pt")
    maddpg.save(final_model_path)
    if reb_enable and reb_trainer is not None and boundary_dataset is not None:
        reb_trainer.save(os.path.join(model_dir, "reb_model_final.pt"), extra={"mode": mode, "dataset_size": len(boundary_dataset)})
    completion = {
        "run_name": run_name,
        "training_stage": training_stage,
        "mode": mode,
        "seed": seed,
        "max_episodes": max_episodes,
        "max_steps": max_steps,
        "batch_size": batch_size,
        "buffer_size": buffer_size,
        "learn_interval": learn_interval,
        "updates_per_train": updates_per_train,
        "snapshot_interval": snapshot_interval,
        "episodes_completed": max_episodes,
        "total_environment_steps": total_steps,
        "optimizer_update_count": optimizer_update_count,
        "per_agent_actor_update_count": per_agent_actor_update_count,
        "per_agent_critic_update_count": per_agent_critic_update_count,
        "actor_policy_changed_by_agent": actor_policy_changed_by_agent,
        "search_actor_policy_changed": search_actor_policy_changed,
        "executor_actor_policy_changed": executor_actor_policy_changed,
        "optimizer_state_reset_requested": bool(rbe_reset_optimizer_state),
        "optimizer_state_reset_applied": optimizer_policy_audit["optimizer_state_reset_applied"],
        "effective_actor_lr": optimizer_policy_audit["effective_actor_lr"],
        "effective_critic_lr": optimizer_policy_audit["effective_critic_lr"],
        "rbe_protocol_lock": bool(rbe_protocol_lock),
        "rbe_protocol_name": (rbe_protocol_audit or {}).get("rbe_protocol_name", ""),
        "rbe_protocol_object_sha256": (rbe_protocol_audit or {}).get("rbe_protocol_object_sha256", ""),
        "rbe_training_protocol_manifest_sha256": (rbe_protocol_audit or {}).get("rbe_training_protocol_manifest_sha256", ""),
        "rbe_sampling_ratios": (rbe_protocol_audit or {}).get("rbe_sampling_ratios", {
            "boundary_core": rbe_boundary_ratio,
            "uniform_coverage": rbe_uniform_ratio,
            "composite_high_risk_aux": rbe_high_risk_ratio,
            "nominal_anchor": rbe_nominal_ratio,
        }),
        "rbe_boundary_csv_sha256": (rbe_protocol_audit or {}).get("rbe_boundary_csv_sha256", ""),
        "rbe_high_risk_csv_sha256": (rbe_protocol_audit or {}).get("rbe_high_risk_csv_sha256", ""),
        "resume_model_sha256": (rbe_protocol_audit or {}).get("resume_model_sha256", _sha256(resume_model_path) if resume_model_path else ""),
        "final_model_path": os.path.abspath(final_model_path),
        "final_model_sha256": _sha256(final_model_path),
    }
    _atomic_write_json(os.path.join(log_dir, "training_completion.json"), completion)
    print(f"Training finished. Final model saved to {final_model_path}")



def main():
    train_ch4()


if __name__ == "__main__":
    main()
