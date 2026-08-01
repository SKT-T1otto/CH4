#!/usr/bin/env python3
"""Audit and finalize the frozen Clean versus Uniform DR full-9D robust test."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from registry.rbe_disturbance import DEFAULT_DISTURBANCE_BOUNDS, DISTURBANCE_KEYS
from registry.ch4_artifact_layout import get_evaluation_dir, get_smoke_dir


class ComparisonError(RuntimeError):
    pass


CLEAN_SHA256 = "c9807c948b50ca102d59de92801190e110eb2184c3976d6ef7d9c21a6e2f75d1"
UNIFORM_SHA256 = "47ae3748017c4c9a43efc16ecb9973b535716c0e09d119f2de43615b8e5405e7"
CLEAN_CHECKPOINT = "snapshot_ep8200"
CLEAN_CHECKPOINT_EPISODE = 8200
UNIFORM_CHECKPOINT = "snapshot_ep1400"
UNIFORM_CHECKPOINT_EPISODE = 1400
CLEAN_MODE = "ch4_pse_baseline"
UNIFORM_MODE = "ch4_uniform_dr"
PROFILE = "normal_comm"
PROTOCOL = "uniform_9d_registry_v1"
PROTOCOL_VERSION = 2
PAIRED_SEED_MODE = "indexed_common_random_numbers"
PAIRED_SEED_FORMULA = "base_seed * 1000003 + episode_index"
DISTURBANCE_RNG_MODE = "numpy.default_rng(episode_seed)"
FLOW_PHASE_KEYS = ("flow_phase_x", "flow_phase_y")
FORMAL_SEEDS = [1, 2, 3]
SMOKE_SEEDS = [909]
FORMAL_EPISODES_PER_SEED = 200
SMOKE_EPISODES_PER_SEED = 2
FORMAL_MAX_STEPS = 400
SMOKE_MAX_STEPS = 20
FORMAL_BOOTSTRAP_REPS = 20000
SMOKE_BOOTSTRAP_REPS = 200
BOOTSTRAP_SEED = 20260718
REL_TOL = 1e-7
ABS_TOL = 1e-9

CONTINUOUS_FIELDS = (
    "reward",
    "recovery_time",
    "safety_cost",
    "final_distance",
    "final_nav_distance",
    "completion_steps",
    "action_smoothness",
)

AGGREGATE_FIELDS = {
    "success_rate": "success_rate_weighted",
    "found_rate": "found_rate_weighted",
    "succ_if_found": "succ_if_found_weighted",
    "avg_reward": "avg_reward_weighted",
    "avg_recovery_time": "avg_recovery_time_weighted",
    "avg_safety_cost": "avg_safety_cost_weighted",
    "avg_final_distance": "avg_final_distance_weighted",
    "avg_final_nav_distance": "avg_final_nav_distance_weighted",
    "avg_completion_steps": "avg_completion_steps_weighted",
    "avg_action_smoothness": "avg_action_smoothness_weighted",
}

COUNT_FIELDS = (
    "total_episodes",
    "total_success",
    "total_found",
    "total_not_found",
    "total_found_but_failed",
)

METRIC_SPECS = {
    "success_rate": ("success", "uniform_minus_clean"),
    "found_rate": ("found", "uniform_minus_clean"),
    "succ_if_found": ("sif", "uniform_minus_clean"),
    "avg_reward": ("reward", "uniform_minus_clean"),
    "avg_safety_cost": ("safety_cost", "clean_minus_uniform"),
    "avg_recovery_time": ("recovery_time", "clean_minus_uniform"),
    "avg_final_distance": ("final_distance", "clean_minus_uniform"),
    "avg_final_nav_distance": ("final_nav_distance", "clean_minus_uniform"),
    "avg_completion_steps": ("completion_steps", "clean_minus_uniform"),
    "avg_action_smoothness": ("action_smoothness", "clean_minus_uniform"),
}

TABLE_FIELDS = (
    "metric",
    "clean",
    "uniform_dr",
    "paired_delta",
    "ci95_low",
    "ci95_high",
    "ci_includes_zero",
    "probability_uniform_dr_better",
    "valid_bootstrap_reps",
    "direction",
)


def _require(condition, message):
    if not condition:
        raise ComparisonError(message)


def _finite(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _close(left, right, rel_tol=REL_TOL, abs_tol=ABS_TOL):
    return math.isclose(
        float(left), float(right), rel_tol=rel_tol, abs_tol=abs_tol
    )


def _load_json(path, label):
    path = Path(path)
    _require(path.is_file(), "%s is missing: %s" % (label, path))
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, ValueError) as exc:
        raise ComparisonError("%s is not readable JSON: %s" % (label, exc))
    _require(isinstance(value, dict), "%s must contain a JSON object" % label)
    return value


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_hash_value(actual, expected, label):
    _require(
        isinstance(actual, str)
        and len(actual) == 64
        and all(char in "0123456789abcdef" for char in actual),
        "%s SHA256 is not lowercase 64-hex" % label,
    )
    _require(actual == expected, "%s SHA256 mismatch" % label)


def _resolve_reported_path(value):
    path = Path(str(value))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _path_equal(left, right):
    try:
        return _resolve_reported_path(left) == Path(right).resolve()
    except (OSError, ValueError):
        return False


def _parse_int(value, label):
    try:
        number = int(str(value).strip())
    except (TypeError, ValueError):
        raise ComparisonError("%s is not an integer" % label)
    return number


def _parse_float(value, label):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ComparisonError("%s is not numeric" % label)
    _require(math.isfinite(number), "%s is not finite" % label)
    return number


def _parse_flag(value, label):
    text = str(value).strip().lower()
    if text in {"1", "1.0", "true"}:
        return 1
    if text in {"0", "0.0", "false"}:
        return 0
    raise ComparisonError("%s is not Boolean" % label)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _atomic_write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(
                _json_safe(value),
                handle,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_text(path, text):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def _audit_clean_nominal_test(
    nominal_test_path,
    model_path,
    manifest_path,
    selection_path,
):
    nominal_test_path = Path(nominal_test_path).resolve()
    model_path = Path(model_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    selection_path = Path(selection_path).resolve()
    nominal = _load_json(nominal_test_path, "clean nominal formal-test summary")
    _require(nominal.get("overall_pass") is True, "clean nominal overall_pass must be true")
    _require(
        nominal.get("experiment_type") == "clean_nominal_formal_test",
        "clean nominal experiment_type mismatch",
    )
    _require(
        nominal.get("test_stage") == "formal_test",
        "clean nominal test_stage mismatch",
    )
    _require(
        nominal.get("formal_test_completed") is True,
        "clean nominal formal test is incomplete",
    )
    _require(nominal.get("errors") == [], "clean nominal errors must be empty")
    _require(
        nominal.get("selected_checkpoint_name") == CLEAN_CHECKPOINT,
        "clean nominal checkpoint mismatch",
    )
    _require(
        nominal.get("selected_checkpoint_episode") == CLEAN_CHECKPOINT_EPISODE,
        "clean nominal checkpoint episode mismatch",
    )
    _require(
        nominal.get("model_sha256") == CLEAN_SHA256,
        "clean nominal model SHA256 mismatch",
    )
    _require(
        nominal.get("hash_match") is True,
        "clean nominal hash_match must be true",
    )
    _require(
        _path_equal(nominal.get("selected_model_path"), model_path),
        "clean nominal selected model path mismatch",
    )
    _require(
        _path_equal(nominal.get("selection_summary"), selection_path),
        "clean nominal selection summary path mismatch",
    )
    _require(
        _path_equal(nominal.get("selected_manifest"), manifest_path),
        "clean nominal manifest path mismatch",
    )
    _require(
        nominal.get("test_seeds") == FORMAL_SEEDS,
        "clean nominal test seeds must be [1, 2, 3]",
    )
    _require(
        nominal.get("episodes_per_seed") == FORMAL_EPISODES_PER_SEED,
        "clean nominal episodes per seed mismatch",
    )
    _require(
        nominal.get("total_episodes") == 600,
        "clean nominal total episodes mismatch",
    )
    _require(
        nominal.get("max_steps") == FORMAL_MAX_STEPS,
        "clean nominal max steps mismatch",
    )
    _require(nominal.get("mode") == CLEAN_MODE, "clean nominal mode mismatch")
    _require(nominal.get("profile") == PROFILE, "clean nominal profile mismatch")
    _require(
        nominal.get("checkpoint_selection_performed") is False,
        "clean nominal performed checkpoint selection",
    )
    _require(
        nominal.get("test_used_for_selection") is False,
        "clean nominal test was used for selection",
    )
    _require(
        nominal.get("selected_checkpoint_changed") is False,
        "clean nominal checkpoint changed",
    )
    _require(
        nominal.get("manifest_updated") is True,
        "clean nominal manifest update was not recorded",
    )
    randomization = nominal.get("episode_randomization_audit") or {}
    _require(randomization.get("enabled") is True, "clean nominal randomization audit is disabled")
    _require(
        randomization.get("episode_seed_mode") == PAIRED_SEED_MODE,
        "clean nominal episode seed mode mismatch",
    )
    _require(
        randomization.get("episode_seed_formula") == PAIRED_SEED_FORMULA,
        "clean nominal episode seed formula mismatch",
    )
    _require(randomization.get("episode_index_base") == 0, "clean nominal episode index base mismatch")
    _require(randomization.get("total_episode_keys") == 600, "clean nominal episode key count mismatch")
    _require(randomization.get("all_episode_keys_unique") is True, "clean nominal episode keys are not unique")
    _require(randomization.get("all_episode_seeds_valid") is True, "clean nominal episode seeds are invalid")
    crosscheck = nominal.get("csv_aggregate_crosscheck") or {}
    _require(crosscheck.get("passed") is True, "clean nominal CSV aggregate audit failed")
    _require(crosscheck.get("count_fields_match") is True, "clean nominal count fields mismatch")
    _require(crosscheck.get("continuous_fields_match") is True, "clean nominal continuous fields mismatch")
    _require(
        _finite(crosscheck.get("relative_tolerance"))
        and math.isclose(
            float(crosscheck["relative_tolerance"]),
            1e-7,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ),
        "clean nominal relative tolerance mismatch",
    )
    _require(
        _finite(crosscheck.get("absolute_tolerance"))
        and math.isclose(
            float(crosscheck["absolute_tolerance"]),
            1e-9,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ),
        "clean nominal absolute tolerance mismatch",
    )
    performance = nominal.get("performance_targets") or {}
    stability = nominal.get("seed_stability") or {}
    _require(
        performance.get("performance_target_pass") is True,
        "clean nominal performance target did not pass",
    )
    _require(
        stability.get("seed_stability_pass") is True,
        "clean nominal seed stability did not pass",
    )
    return {
        "path": str(nominal_test_path),
        "sha256": _sha256(nominal_test_path),
        "overall_pass": True,
        "experiment_type": "clean_nominal_formal_test",
        "test_stage": "formal_test",
        "formal_test_completed": True,
        "test_seeds": list(FORMAL_SEEDS),
        "episodes_per_seed": FORMAL_EPISODES_PER_SEED,
        "total_episodes": 600,
        "max_steps": FORMAL_MAX_STEPS,
        "checkpoint_selection_performed": False,
        "test_used_for_selection": False,
        "selected_checkpoint_changed": False,
        "performance_target_pass": True,
        "seed_stability_pass": True,
        "episode_randomization_pass": True,
        "csv_aggregate_crosscheck_pass": True,
    }


def _audit_clean_sources(
    model_path,
    manifest_path,
    selection_path,
    nominal_test_path,
):
    model_path = Path(model_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    selection_path = Path(selection_path).resolve()
    _require(model_path.is_file() and model_path.stat().st_size > 0, "clean model is missing or empty")
    model_hash = _sha256(model_path)
    _require_hash_value(model_hash, CLEAN_SHA256, "clean model")
    manifest = _load_json(manifest_path, "clean manifest")
    selection = _load_json(selection_path, "clean selection summary")
    _require(manifest.get("selection_smoke") is False, "clean manifest must be formal")
    _require(manifest.get("selected_checkpoint_name") == CLEAN_CHECKPOINT, "clean manifest checkpoint mismatch")
    _require(manifest.get("selected_checkpoint_episode") == CLEAN_CHECKPOINT_EPISODE, "clean manifest episode mismatch")
    _require(manifest.get("selected_sha256") == CLEAN_SHA256, "clean manifest selected hash mismatch")
    _require(manifest.get("source_sha256") == CLEAN_SHA256, "clean manifest source hash mismatch")
    _require(_path_equal(manifest.get("selected_model_path"), model_path), "clean manifest model path mismatch")
    _require(manifest.get("ready_for_formal_test") is True, "clean manifest is not ready for formal test")
    _require(manifest.get("formal_test_used_for_selection") is False, "clean nominal test was used for selection")
    _require(manifest.get("selected_checkpoint_changed_after_test") is False, "clean checkpoint changed after test")
    _require(selection.get("overall_pass") is True, "clean selection overall_pass must be true")
    _require(selection.get("selection_stage") == "final_validation", "clean selection stage mismatch")
    _require(selection.get("selected_checkpoint_name") == CLEAN_CHECKPOINT, "clean selection checkpoint mismatch")
    _require(selection.get("selected_checkpoint_episode") == CLEAN_CHECKPOINT_EPISODE, "clean selection episode mismatch")
    _require(selection.get("selected_sha256") == CLEAN_SHA256, "clean selection hash mismatch")
    _require(selection.get("source_sha256") == CLEAN_SHA256, "clean selection source hash mismatch")
    _require(selection.get("hash_match") is True, "clean selection hash_match must be true")
    _require(_path_equal(selection.get("selected_model_path"), model_path), "clean selection model path mismatch")
    _require(selection.get("forbidden_test_seeds") == FORMAL_SEEDS, "clean selection did not reserve test seeds")
    _require((selection.get("settings") or {}).get("smoke") is False, "clean selection summary is smoke")
    _require(selection.get("errors") == [], "clean selection errors must be empty")
    nominal_test_audit = _audit_clean_nominal_test(
        nominal_test_path,
        model_path,
        manifest_path,
        selection_path,
    )
    return {
        "role": "clean_baseline",
        "evaluation_mode": CLEAN_MODE,
        "model_path": str(model_path),
        "model_sha256": model_hash,
        "manifest_path": str(manifest_path),
        "selection_summary_path": str(selection_path),
        "selected_checkpoint_name": CLEAN_CHECKPOINT,
        "selected_checkpoint_episode": CLEAN_CHECKPOINT_EPISODE,
        "selection_smoke": False,
        "nominal_test_summary_path": nominal_test_audit["path"],
        "nominal_test_summary_sha256": nominal_test_audit["sha256"],
        "nominal_validation_passed": nominal_test_audit["overall_pass"],
        "nominal_formal_test_completed": nominal_test_audit[
            "formal_test_completed"
        ],
        "nominal_test_seeds": nominal_test_audit["test_seeds"],
        "nominal_test_total_episodes": nominal_test_audit["total_episodes"],
        "nominal_test_performance_target_pass": nominal_test_audit[
            "performance_target_pass"
        ],
        "nominal_test_seed_stability_pass": nominal_test_audit[
            "seed_stability_pass"
        ],
        "nominal_test_randomization_pass": nominal_test_audit[
            "episode_randomization_pass"
        ],
        "nominal_test_csv_crosscheck_pass": nominal_test_audit[
            "csv_aggregate_crosscheck_pass"
        ],
        "test_used_for_selection": False,
    }


def _audit_uniform_sources(model_path, manifest_path, selection_path):
    model_path = Path(model_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    selection_path = Path(selection_path).resolve()
    _require(model_path.is_file() and model_path.stat().st_size > 0, "Uniform DR model is missing or empty")
    model_hash = _sha256(model_path)
    _require_hash_value(model_hash, UNIFORM_SHA256, "Uniform DR model")
    manifest = _load_json(manifest_path, "Uniform DR manifest")
    selection = _load_json(selection_path, "Uniform DR selection summary")
    for value, label in ((manifest, "manifest"), (selection, "selection summary")):
        _require(value.get("selection_smoke") is False, "Uniform DR %s must be formal" % label)
        _require(value.get("selected_checkpoint_name") == UNIFORM_CHECKPOINT, "Uniform DR %s checkpoint mismatch" % label)
        _require(value.get("selected_checkpoint_episode") == UNIFORM_CHECKPOINT_EPISODE, "Uniform DR %s episode mismatch" % label)
        _require(value.get("hash_match") is True, "Uniform DR %s hash_match must be true" % label)
        _require(value.get("test_used_for_selection") is False, "Uniform DR %s used test data for selection" % label)
        _require(value.get("ready_for_formal_robust_test") is True, "Uniform DR %s is not ready" % label)
    _require(selection.get("overall_pass") is True, "Uniform DR selection overall_pass must be true")
    _require(selection.get("evaluation_protocol") == PROTOCOL, "Uniform DR selection protocol mismatch")
    _require(selection.get("evaluation_protocol_version") == PROTOCOL_VERSION, "Uniform DR selection protocol version mismatch")
    _require(selection.get("legacy_coarse_v1_used") is False, "legacy reduced-disturbance result was used")
    _require(selection.get("errors") == [], "Uniform DR selection errors must be empty")
    _require(manifest.get("selected_sha256") == UNIFORM_SHA256, "Uniform DR manifest selected hash mismatch")
    _require(manifest.get("source_sha256") == UNIFORM_SHA256, "Uniform DR manifest source hash mismatch")
    _require(manifest.get("legacy_reduced_disturbance_results_used_for_selection") is False, "Uniform DR manifest reports legacy selection")
    _require(_path_equal(manifest.get("selected_model_path"), model_path), "Uniform DR manifest model path mismatch")
    _require(_path_equal(selection.get("selected_model_path"), model_path), "Uniform DR selection model path mismatch")
    return {
        "role": "uniform_dr",
        "evaluation_mode": UNIFORM_MODE,
        "model_path": str(model_path),
        "model_sha256": model_hash,
        "manifest_path": str(manifest_path),
        "selection_summary_path": str(selection_path),
        "selected_checkpoint_name": UNIFORM_CHECKPOINT,
        "selected_checkpoint_episode": UNIFORM_CHECKPOINT_EPISODE,
        "selection_smoke": False,
        "evaluation_protocol": PROTOCOL,
        "evaluation_protocol_version": PROTOCOL_VERSION,
        "test_used_for_selection": False,
        "ready_for_formal_robust_test": True,
    }


def audit_sources(args, load_models=False):
    clean = _audit_clean_sources(
        args.clean_model,
        args.clean_manifest,
        args.clean_selection_summary,
        args.clean_nominal_test_summary,
    )
    uniform = _audit_uniform_sources(
        args.uniform_model, args.uniform_manifest, args.uniform_selection_summary
    )
    if load_models:
        try:
            import torch
            from algorithms.maddpg import MADDPG

            device = torch.device("cpu")
            load_results = []
            for label, path in (
                ("clean", args.clean_model),
                ("uniform_dr", args.uniform_model),
            ):
                model = MADDPG.init_from_save(path, device=device)
                model.prep_rollouts(device=device)
                required_modules = (
                    "policy",
                    "critic1",
                    "critic2",
                    "rec_critic",
                    "safe_critic",
                )
                modules_ok = all(
                    hasattr(agent, name)
                    and sum(parameter.numel() for parameter in getattr(agent, name).parameters()) > 0
                    for agent in model.agents
                    for name in required_modules
                )
                _require(model.nagents == len(model.agents) > 0, "%s MADDPG agent count is invalid" % label)
                _require(modules_ok, "%s MADDPG structure is incomplete" % label)
                _require(
                    str(next(model.agents[0].policy.parameters()).device) == "cpu",
                    "%s model did not load on CPU" % label,
                )
                load_results.append(
                    {
                        "model": label,
                        "agents": model.nagents,
                        "roles": list(model.agent_role_names),
                        "device": "cpu",
                    }
                )
            clean["load_audit"] = load_results[0]
            uniform["load_audit"] = load_results[1]
        except ComparisonError:
            raise
        except Exception as exc:
            raise ComparisonError("MADDPG CPU load failed: %s" % exc)
    return clean, uniform


def _validate_mode(smoke, seeds, episodes_per_seed, max_steps, bootstrap_reps, bootstrap_seed):
    expected_seeds = SMOKE_SEEDS if smoke else FORMAL_SEEDS
    expected_episodes = SMOKE_EPISODES_PER_SEED if smoke else FORMAL_EPISODES_PER_SEED
    expected_steps = SMOKE_MAX_STEPS if smoke else FORMAL_MAX_STEPS
    expected_reps = SMOKE_BOOTSTRAP_REPS if smoke else FORMAL_BOOTSTRAP_REPS
    _require(list(seeds) == expected_seeds, "%s seeds must be %s" % ("smoke" if smoke else "formal", expected_seeds))
    _require(episodes_per_seed == expected_episodes, "episodes per seed mismatch")
    _require(max_steps == expected_steps, "max steps mismatch")
    _require(bootstrap_reps == expected_reps, "bootstrap repetitions mismatch")
    _require(bootstrap_seed == BOOTSTRAP_SEED, "bootstrap seed mismatch")
    if smoke:
        _require(not set(seeds).intersection(FORMAL_SEEDS), "smoke may not use test seeds")
        _require(not set(seeds).intersection(range(101, 109)), "smoke may not use validation seeds")
    return True


def _audit_seed_summary(summary, seed, expected_mode, model_path, episodes, max_steps):
    _require(summary.get("method") == expected_mode, "seed%d method mismatch" % seed)
    _require(summary.get("profile") == PROFILE, "seed%d profile mismatch" % seed)
    _require(summary.get("seed") == seed, "seed%d summary seed mismatch" % seed)
    _require(summary.get("episodes") == episodes, "seed%d episode setting mismatch" % seed)
    _require(summary.get("max_steps") == max_steps, "seed%d max steps mismatch" % seed)
    _require(summary.get("paired_episode_seeding") is True, "seed%d paired seeding is disabled" % seed)
    _require(summary.get("episode_seed_mode") == PAIRED_SEED_MODE, "seed%d episode seed mode mismatch" % seed)
    _require(summary.get("episode_seed_formula") == PAIRED_SEED_FORMULA, "seed%d episode seed formula mismatch" % seed)
    _require(summary.get("disturbance_seed_formula") == PAIRED_SEED_FORMULA, "seed%d disturbance seed formula mismatch" % seed)
    _require(summary.get("disturbance_rng_mode") == DISTURBANCE_RNG_MODE, "seed%d disturbance RNG mismatch" % seed)
    _require(summary.get("disturbance_protocol") == PROTOCOL, "seed%d protocol mismatch" % seed)
    _require(summary.get("disturbance_protocol_version") == PROTOCOL_VERSION, "seed%d protocol version mismatch" % seed)
    _require(summary.get("explicit_disturbance_application") is True, "seed%d disturbance is not explicit" % seed)
    _require(summary.get("all_episode_disturbance_apply_match") is True, "seed%d apply audit failed" % seed)
    _require(summary.get("bounds_violation_count") == 0, "seed%d reports bounds violations" % seed)
    _require(list(summary.get("missing_fields") or []) == [], "seed%d reports missing fields" % seed)
    _require(summary.get("flow_phase_in_9d_vector") is False, "seed%d includes flow phase in 9D vector" % seed)
    _require(summary.get("disturbance_keys") == list(DISTURBANCE_KEYS), "seed%d disturbance keys mismatch" % seed)
    _require(summary.get("flow_phase_keys") == list(FLOW_PHASE_KEYS), "seed%d flow phase keys mismatch" % seed)
    _require(_path_equal(summary.get("model_path"), model_path), "seed%d model path mismatch" % seed)
    _require(summary.get("distinct_disturbance_vector_count") == episodes, "seed%d 9D vectors are not distinct" % seed)
    _require(summary.get("distinct_full_disturbance_count") == episodes, "seed%d full disturbances are not distinct" % seed)
    return PROTOCOL_VERSION


def _load_episode_csv(path, seed, expected_rows, protocol_version):
    path = Path(path)
    _require(path.is_file(), "episode CSV is missing: %s" % path)
    records = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        required = {
            "base_seed",
            "episode_index",
            "episode_seed",
            "episode_seed_mode",
            "disturbance_seed",
            "disturbance_rng_mode",
            "disturbance_protocol",
            "disturbance_explicitly_applied",
            "disturbance_apply_match",
            "success_flag",
            "found_flag",
            *DISTURBANCE_KEYS,
            *FLOW_PHASE_KEYS,
            *CONTINUOUS_FIELDS,
        }
        missing = sorted(required.difference(fields))
        _require(not missing, "%s is missing fields: %s" % (path, missing))
        for raw in reader:
            base_seed = _parse_int(raw.get("base_seed"), "%s base_seed" % path)
            index = _parse_int(raw.get("episode_index"), "%s episode_index" % path)
            episode_seed = _parse_int(raw.get("episode_seed"), "%s episode_seed" % path)
            disturbance_seed = _parse_int(raw.get("disturbance_seed"), "%s disturbance_seed" % path)
            _require(base_seed == seed, "%s base seed mismatch" % path)
            _require(0 <= index < expected_rows, "%s episode index out of range" % path)
            _require(episode_seed == seed * 1_000_003 + index, "%s episode seed formula mismatch" % path)
            _require(disturbance_seed == episode_seed, "%s disturbance seed mismatch" % path)
            _require(raw.get("episode_seed_mode") == PAIRED_SEED_MODE, "%s episode seed mode mismatch" % path)
            _require(raw.get("disturbance_rng_mode") == DISTURBANCE_RNG_MODE, "%s disturbance RNG mismatch" % path)
            _require(raw.get("disturbance_protocol") == PROTOCOL, "%s protocol mismatch" % path)
            record = {
                "base_seed": base_seed,
                "episode_index": index,
                "episode_seed": episode_seed,
                "episode_seed_mode": raw.get("episode_seed_mode"),
                "disturbance_seed": disturbance_seed,
                "disturbance_protocol": raw.get("disturbance_protocol"),
                "disturbance_protocol_version": protocol_version,
                "disturbance_explicitly_applied": _parse_flag(
                    raw.get("disturbance_explicitly_applied"),
                    "%s explicit apply" % path,
                ),
                "disturbance_apply_match": _parse_flag(
                    raw.get("disturbance_apply_match"),
                    "%s apply match" % path,
                ),
                "success": _parse_flag(raw.get("success_flag"), "%s success" % path),
                "found": _parse_flag(raw.get("found_flag"), "%s found" % path),
            }
            _require(record["disturbance_explicitly_applied"] == 1, "%s disturbance was not explicitly applied" % path)
            _require(record["disturbance_apply_match"] == 1, "%s disturbance apply mismatch" % path)
            _require(record["success"] <= record["found"], "%s success exceeds found" % path)
            for field in DISTURBANCE_KEYS:
                if field == "action_delay_steps":
                    record[field] = _parse_int(raw.get(field), "%s %s" % (path, field))
                else:
                    record[field] = _parse_float(raw.get(field), "%s %s" % (path, field))
            for field in FLOW_PHASE_KEYS:
                record[field] = _parse_float(raw.get(field), "%s %s" % (path, field))
            for field in CONTINUOUS_FIELDS:
                record[field] = _parse_float(raw.get(field), "%s %s" % (path, field))
            key = (seed, index)
            _require(key not in records, "%s contains duplicate episode key %r" % (path, key))
            records[key] = record
    _require(len(records) == expected_rows, "%s row count is %d, expected %d" % (path, len(records), expected_rows))
    _require({key[1] for key in records} == set(range(expected_rows)), "%s episode indexes are incomplete" % path)
    return records


def _audit_bounds(records):
    violations = []
    for key, record in records.items():
        for field in DISTURBANCE_KEYS:
            low, high = DEFAULT_DISTURBANCE_BOUNDS[field]
            value = record[field]
            if field == "action_delay_steps":
                valid = isinstance(value, int) and not isinstance(value, bool)
            else:
                valid = _finite(value)
            if not valid or float(value) < float(low) or float(value) > float(high):
                violations.append((key, field, value))
    _require(not violations, "disturbance bounds violation: %s" % (violations[:3],))
    return 0


def _metrics(records):
    values = list(records.values())
    total = len(values)
    _require(total > 0, "metrics require at least one episode")
    success = sum(row["success"] for row in values)
    found = sum(row["found"] for row in values)
    result = {
        "total_episodes": total,
        "total_success": success,
        "total_found": found,
        "total_not_found": total - found,
        "total_found_but_failed": found - success,
        "success_rate": success / float(total),
        "found_rate": found / float(total),
        "succ_if_found": success / float(found) if found else None,
    }
    for field in CONTINUOUS_FIELDS:
        result["avg_" + field] = sum(row[field] for row in values) / float(total)
    result["wilson_ci95"] = {
        "success_rate": wilson_interval(success, total),
        "found_rate": wilson_interval(found, total),
        "succ_if_found": wilson_interval(success, found) if found else None,
    }
    return result


def _assert_optional_close(actual, expected, label):
    if expected is None:
        _require(actual is None, "%s should be null" % label)
    else:
        _require(_finite(actual) and _close(actual, expected), "%s mismatch" % label)


def _audit_seed_metric_summary(summary, metrics, seed):
    field_map = {
        "n_episodes": "total_episodes",
        "n_success": "total_success",
        "n_found": "total_found",
        "n_not_found": "total_not_found",
        "n_found_but_failed": "total_found_but_failed",
    }
    for summary_field, metric_field in field_map.items():
        _require(summary.get(summary_field) == metrics[metric_field], "seed%d %s differs from CSV" % (seed, summary_field))
    for field in ("success_rate", "found_rate", "succ_if_found"):
        _assert_optional_close(summary.get(field), metrics[field], "seed%d %s" % (seed, field))
    for field in CONTINUOUS_FIELDS:
        name = "avg_" + field
        _assert_optional_close(summary.get(name), metrics[name], "seed%d %s" % (seed, name))
    return True


def _audit_aggregate_values(aggregate, metrics, expected_name, mode, seeds, episodes_per_seed):
    _require(aggregate.get("summary_name") == expected_name, "aggregate summary name mismatch")
    _require(aggregate.get("method") == mode, "aggregate method mismatch")
    _require(aggregate.get("profile") == PROFILE, "aggregate profile mismatch")
    _require(aggregate.get("seeds") == list(seeds), "aggregate seeds mismatch")
    _require(aggregate.get("episodes_per_seed") == [episodes_per_seed] * len(seeds), "aggregate episodes-per-seed mismatch")
    for field in COUNT_FIELDS:
        _require(aggregate.get(field) == metrics[field], "aggregate %s differs from episode CSV" % field)
    for metric_field, aggregate_field in AGGREGATE_FIELDS.items():
        _assert_optional_close(
            aggregate.get(aggregate_field),
            metrics[metric_field],
            "aggregate %s" % aggregate_field,
        )
    return {
        "passed": True,
        "count_fields_match": True,
        "continuous_fields_match": True,
        "relative_tolerance": REL_TOL,
        "absolute_tolerance": ABS_TOL,
    }


def _load_model_data(raw_root, model_path, expected_mode, seeds, episodes, max_steps):
    raw_root = Path(raw_root).resolve()
    _require(raw_root.is_dir(), "raw model root is missing: %s" % raw_root)
    expected_dirs = {"seed%d" % seed for seed in seeds}
    actual_dirs = {path.name for path in raw_root.iterdir() if path.is_dir()}
    _require(actual_dirs == expected_dirs, "raw seed directories mismatch: %s" % sorted(actual_dirs))
    all_records = {}
    per_seed = []
    for seed in seeds:
        seed_dir = raw_root / ("seed%d" % seed)
        summary = _load_json(seed_dir / "evaluation_summary.json", "seed%d evaluation summary" % seed)
        version = _audit_seed_summary(
            summary, seed, expected_mode, model_path, episodes, max_steps
        )
        records = _load_episode_csv(
            seed_dir / "episode_metrics.csv", seed, episodes, version
        )
        _audit_bounds(records)
        metrics = _metrics(records)
        _audit_seed_metric_summary(summary, metrics, seed)
        all_records.update(records)
        per_seed.append({"seed": seed, "metrics": metrics})
    _require(len(all_records) == len(seeds) * episodes, "pooled episode count mismatch")
    return all_records, per_seed


def _audit_pairing(clean_records, uniform_records):
    clean_keys = set(clean_records)
    uniform_keys = set(uniform_records)
    _require(clean_keys == uniform_keys, "cross-model episode keys differ")
    for key in sorted(clean_keys):
        clean = clean_records[key]
        uniform = uniform_records[key]
        for field in (
            "base_seed",
            "episode_index",
            "episode_seed",
            "episode_seed_mode",
            "disturbance_seed",
            "disturbance_protocol",
            "disturbance_protocol_version",
        ):
            _require(clean[field] == uniform[field], "paired %s differs at %r" % (field, key))
        _require(clean["disturbance_explicitly_applied"] == uniform["disturbance_explicitly_applied"] == 1, "explicit apply mismatch at %r" % (key,))
        _require(clean["disturbance_apply_match"] == uniform["disturbance_apply_match"] == 1, "actual apply mismatch at %r" % (key,))
        for field in DISTURBANCE_KEYS:
            if field == "action_delay_steps":
                matches = clean[field] == uniform[field]
            else:
                matches = math.isclose(
                    clean[field], uniform[field], rel_tol=1e-12, abs_tol=1e-12
                )
            _require(matches, "paired 9D field %s differs at %r" % (field, key))
        for field in FLOW_PHASE_KEYS:
            _require(
                math.isclose(
                    clean[field], uniform[field], rel_tol=1e-12, abs_tol=1e-12
                ),
                "paired flow phase %s differs at %r" % (field, key),
            )
    return {
        "paired_episode_count": len(clean_keys),
        "all_episode_keys_match": True,
        "all_episode_seeds_match": True,
        "all_disturbance_seeds_match": True,
        "all_9d_vectors_match": True,
        "all_flow_phases_match": True,
        "all_episode_apply_match": True,
        "episode_seed_mode": PAIRED_SEED_MODE,
        "episode_seed_formula": PAIRED_SEED_FORMULA,
        "disturbance_seed_formula": PAIRED_SEED_FORMULA,
    }


def _audit_full_9d(clean_records, uniform_records, smoke):
    _audit_bounds(clean_records)
    _audit_bounds(uniform_records)
    reference = list(clean_records.values())
    distinct_9d = {
        tuple(row[field] for field in DISTURBANCE_KEYS) for row in reference
    }
    distinct_full = {
        tuple(row[field] for field in (*DISTURBANCE_KEYS, *FLOW_PHASE_KEYS))
        for row in reference
    }
    per_dimension = {
        field: len({row[field] for row in reference}) for field in DISTURBANCE_KEYS
    }
    observed_delays = sorted({int(row["action_delay_steps"]) for row in reference})
    total = len(reference)
    _require(len(distinct_9d) == total, "9D disturbance vectors are not all distinct")
    _require(len(distinct_full) == total, "full disturbances are not all distinct")
    for field in DISTURBANCE_KEYS:
        if field != "action_delay_steps":
            _require(per_dimension[field] == total, "%s does not vary for every episode" % field)
    if not smoke:
        _require(total == 600, "formal full-9D audit requires 600 episodes")
        _require(observed_delays == [0, 1, 2, 3], "formal action delays do not cover all legal values")
    return {
        "overall_pass": True,
        "protocol": PROTOCOL,
        "protocol_version": PROTOCOL_VERSION,
        "disturbance_keys": list(DISTURBANCE_KEYS),
        "disturbance_bounds": {
            key: list(DEFAULT_DISTURBANCE_BOUNDS[key]) for key in DISTURBANCE_KEYS
        },
        "flow_phase_keys": list(FLOW_PHASE_KEYS),
        "flow_phase_in_9d_vector": False,
        "paired_models_observed_same_disturbances": True,
        "distinct_9d_vector_count": len(distinct_9d),
        "distinct_full_disturbance_count": len(distinct_full),
        "per_dimension_distinct_counts": per_dimension,
        "observed_action_delay_values": observed_delays,
        "bounds_violation_count": 0,
        "missing_field_count": 0,
    }


def wilson_interval(successes, trials, z=1.959963984540054):
    _require(isinstance(successes, int) and isinstance(trials, int), "Wilson counts must be integers")
    _require(0 <= successes <= trials, "Wilson counts are invalid")
    if trials == 0:
        return None
    proportion = successes / float(trials)
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / trials
            + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    return [max(0.0, center - radius), min(1.0, center + radius)]


def exact_mcnemar(uniform_only, clean_only):
    _require(uniform_only >= 0 and clean_only >= 0, "McNemar counts must be non-negative")
    discordant = int(uniform_only) + int(clean_only)
    if discordant == 0:
        return {"discordant_n": 0, "exact_two_sided_p": 1.0}
    tail = min(int(uniform_only), int(clean_only))
    numerator = sum(math.comb(discordant, index) for index in range(tail + 1))
    probability = min(1.0, 2.0 * float(numerator / (2 ** discordant)))
    return {
        "discordant_n": discordant,
        "exact_two_sided_p": probability,
    }


def _discordant(clean_records, uniform_records, field):
    counts = {
        "uniform_positive_clean_negative": 0,
        "uniform_negative_clean_positive": 0,
        "both_positive": 0,
        "both_negative": 0,
    }
    for key in sorted(clean_records):
        clean = clean_records[key][field]
        uniform = uniform_records[key][field]
        if uniform and not clean:
            counts["uniform_positive_clean_negative"] += 1
        elif clean and not uniform:
            counts["uniform_negative_clean_positive"] += 1
        elif clean and uniform:
            counts["both_positive"] += 1
        else:
            counts["both_negative"] += 1
    test = exact_mcnemar(
        counts["uniform_positive_clean_negative"],
        counts["uniform_negative_clean_positive"],
    )
    return {"counts": counts, "exact_mcnemar": test}


def _point_deltas(clean_metrics, uniform_metrics):
    result = {}
    for metric, (_, direction) in METRIC_SPECS.items():
        clean = clean_metrics[metric]
        uniform = uniform_metrics[metric]
        if clean is None or uniform is None:
            result[metric] = None
        elif direction == "uniform_minus_clean":
            result[metric] = uniform - clean
        else:
            result[metric] = clean - uniform
    return result


def paired_stratified_bootstrap(
    clean_records,
    uniform_records,
    seeds,
    reps,
    random_seed,
    allow_zero_valid=False,
):
    _require(reps > 0, "bootstrap repetitions must be positive")
    generator = np.random.default_rng(random_seed)
    sampled_sums = {
        field: {
            "clean": np.zeros(reps, dtype=np.float64),
            "uniform": np.zeros(reps, dtype=np.float64),
        }
        for field in (
            "success",
            "found",
            "reward",
            "safety_cost",
            "recovery_time",
            "final_distance",
            "final_nav_distance",
            "completion_steps",
            "action_smoothness",
        )
    }
    total_sampled = 0
    for seed in seeds:
        keys = sorted(key for key in clean_records if key[0] == seed)
        _require(keys, "bootstrap seed%d has no records" % seed)
        sample_indexes = generator.integers(
            0, len(keys), size=(reps, len(keys)), endpoint=False
        )
        total_sampled += len(keys)
        for field in sampled_sums:
            clean_values = np.asarray(
                [clean_records[key][field] for key in keys], dtype=np.float64
            )
            uniform_values = np.asarray(
                [uniform_records[key][field] for key in keys], dtype=np.float64
            )
            sampled_sums[field]["clean"] += clean_values[sample_indexes].sum(axis=1)
            sampled_sums[field]["uniform"] += uniform_values[sample_indexes].sum(axis=1)
        del sample_indexes
    clean_metrics = _metrics(clean_records)
    uniform_metrics = _metrics(uniform_records)
    points = _point_deltas(clean_metrics, uniform_metrics)
    comparisons = {}
    for metric, (source, direction) in METRIC_SPECS.items():
        if source == "sif":
            clean_found = sampled_sums["found"]["clean"]
            uniform_found = sampled_sums["found"]["uniform"]
            valid = (clean_found > 0) & (uniform_found > 0)
            values = (
                sampled_sums["success"]["uniform"][valid] / uniform_found[valid]
                - sampled_sums["success"]["clean"][valid] / clean_found[valid]
            )
        else:
            valid = np.ones(reps, dtype=bool)
            clean_values = sampled_sums[source]["clean"] / float(total_sampled)
            uniform_values = sampled_sums[source]["uniform"] / float(total_sampled)
            values = (
                uniform_values - clean_values
                if direction == "uniform_minus_clean"
                else clean_values - uniform_values
            )
        if values.size == 0:
            _require(
                allow_zero_valid and metric == "succ_if_found",
                "%s bootstrap has no valid repetitions" % metric,
            )
            comparisons[metric] = {
                "clean": clean_metrics[metric],
                "uniform_dr": uniform_metrics[metric],
                "direction": direction,
                "point_estimate": points[metric],
                "ci95_low": None,
                "ci95_high": None,
                "ci_includes_zero": None,
                "probability_uniform_dr_better": None,
                "valid_bootstrap_reps": 0,
                "requested_bootstrap_reps": reps,
            }
            continue
        low, high = np.quantile(values, [0.025, 0.975])
        comparisons[metric] = {
            "clean": clean_metrics[metric],
            "uniform_dr": uniform_metrics[metric],
            "direction": direction,
            "point_estimate": points[metric],
            "ci95_low": float(low),
            "ci95_high": float(high),
            "ci_includes_zero": bool(float(low) <= 0.0 <= float(high)),
            "probability_uniform_dr_better": float(np.mean(values > 0.0)),
            "valid_bootstrap_reps": int(values.size),
            "requested_bootstrap_reps": reps,
        }
    return comparisons


def performance_conclusion(delta, interval, tolerance=1e-12):
    _require(delta is not None and interval is not None, "performance conclusion requires finite inputs")
    low, high = interval
    if abs(delta) <= tolerance:
        return "no_material_difference"
    if delta > 0:
        return (
            "significant_improvement"
            if low > 0
            else "positive_trend_not_significant"
        )
    return (
        "significant_degradation"
        if high < 0
        else "negative_trend_not_significant"
    )


def _format_number(value):
    if value is None:
        return "null"
    return "%.6f" % float(value)


def _format_interval(value):
    if value is None:
        return "null"
    return "[%s, %s]" % (_format_number(value[0]), _format_number(value[1]))


def _comparison_table(summary):
    rows = []
    for metric, comparison in summary["paired_comparison"].items():
        rows.append(
            {
                "metric": metric,
                "clean": comparison["clean"],
                "uniform_dr": comparison["uniform_dr"],
                "paired_delta": comparison["point_estimate"],
                "ci95_low": comparison["ci95_low"],
                "ci95_high": comparison["ci95_high"],
                "ci_includes_zero": comparison["ci_includes_zero"],
                "probability_uniform_dr_better": comparison[
                    "probability_uniform_dr_better"
                ],
                "valid_bootstrap_reps": comparison["valid_bootstrap_reps"],
                "direction": comparison["direction"],
            }
        )
    return rows


def _write_table(path, rows):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=TABLE_FIELDS)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field) for field in TABLE_FIELDS})
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def _markdown(summary):
    lines = [
        "# Clean vs Uniform DR Full-9D Robust Test",
        "",
        "This is a frozen-model test, not checkpoint selection. Test seeds were not used for retraining or reselection.",
        "",
        "## Protocol",
        "",
        "- Stage: %s" % summary["test_stage"],
        "- Seeds: %s" % summary["test_seeds"],
        "- Episodes per seed: %s" % summary["episodes_per_seed"],
        "- Total rollout episodes: %s" % summary["total_rollout_episodes"],
        "- Max steps: %s" % summary["max_steps"],
        "- Protocol: %s version %s" % (PROTOCOL, PROTOCOL_VERSION),
        "- Paired seed formula: %s" % PAIRED_SEED_FORMULA,
        "",
        "## Frozen Models",
        "",
        "- Clean: %s, SHA256 %s" % (
            summary["clean_model"]["selected_checkpoint_name"],
            summary["clean_model"]["model_sha256"],
        ),
        "- Uniform DR: %s, SHA256 %s" % (
            summary["uniform_dr_model"]["selected_checkpoint_name"],
            summary["uniform_dr_model"]["model_sha256"],
        ),
        "- Models frozen: True",
        "- Checkpoint selection performed: False",
        "- Test used for selection: False",
        "",
        "## Pairing Audit",
        "",
    ]
    for key, value in summary["pairing_audit"].items():
        lines.append("- %s: %s" % (key, value))
    lines.extend(["", "## Full-9D Audit", ""])
    for key, value in summary["disturbance_protocol_audit"].items():
        if key not in {"disturbance_bounds"}:
            lines.append("- %s: %s" % (key, value))
    lines.extend(
        [
            "",
            "## Aggregate Metrics",
            "",
            "Positive paired delta always means Uniform DR is better.",
            "",
            "| metric | Clean | Uniform DR | paired delta | 95% CI |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for metric, comparison in summary["paired_comparison"].items():
        lines.append(
            "| %s | %s | %s | %s | [%s, %s] |"
            % (
                metric,
                _format_number(comparison["clean"]),
                _format_number(comparison["uniform_dr"]),
                _format_number(comparison["point_estimate"]),
                _format_number(comparison["ci95_low"]),
                _format_number(comparison["ci95_high"]),
            )
        )
    lines.extend(
        [
            "",
            "## Per-Seed Metrics",
            "",
            "| seed | model | success | found | SIF | safety | recovery | distance |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["per_seed_metrics"]:
        for model_key, label in (("clean", "Clean"), ("uniform_dr", "Uniform DR")):
            metrics = row[model_key]
            lines.append(
                "| %s | %s | %s | %s | %s | %s | %s | %s |"
                % (
                    row["seed"],
                    label,
                    _format_number(metrics["success_rate"]),
                    _format_number(metrics["found_rate"]),
                    _format_number(metrics["succ_if_found"]),
                    _format_number(metrics["avg_safety_cost"]),
                    _format_number(metrics["avg_recovery_time"]),
                    _format_number(metrics["avg_final_distance"]),
                )
            )
    success_discordant = summary["discordant_outcomes"]["success"]
    found_discordant = summary["discordant_outcomes"]["found"]
    lines.extend(
        [
            "",
            "## Discordant Outcomes",
            "",
            "- Success: %s" % success_discordant["counts"],
            "- Found: %s" % found_discordant["counts"],
            "",
            "## Exact McNemar Test",
            "",
            "- Success: %s" % success_discordant["exact_mcnemar"],
            "- Found: %s" % found_discordant["exact_mcnemar"],
            "",
            "## Statistical Interpretation",
            "",
            "- Primary success delta: %s" % _format_number(summary["primary_success_delta"]),
            "- Primary success 95%% CI: %s" % _format_interval(summary["primary_success_delta_ci95"]),
            "- CI includes zero: %s"
            % summary["paired_comparison"]["success_rate"]["ci_includes_zero"],
            "- Exact McNemar p: %s"
            % _format_number(summary["primary_success_mcnemar_p"]),
            "- A trend is not described as significant when its CI contains zero.",
            "",
            "## Final Performance Conclusion",
            "",
            summary["performance_conclusion"],
            "",
        ]
    )
    if summary["test_stage"] == "smoke":
        lines.append("Smoke performance has no scientific interpretation.")
        lines.append("")
    return "\n".join(lines)


def _validate_paths(args):
    out_dir = Path(args.out_dir).resolve()
    experiment_id = (
        "ch4_clean_vs_uniform_dr_formal_robust_test_full9d_v2_smoke"
        if args.smoke
        else "ch4_clean_vs_uniform_dr_formal_robust_test_full9d_v2"
    )
    expected_out = (
        get_smoke_dir("uniform_dr", experiment_id)
        if args.smoke
        else get_evaluation_dir("uniform_dr", experiment_id)
    ).resolve()
    _require(out_dir == expected_out, "output directory is not the fixed %s path" % ("smoke" if args.smoke else "formal"))
    clean_raw = Path(args.clean_raw_root).resolve()
    uniform_raw = Path(args.uniform_raw_root).resolve()
    _require(clean_raw == out_dir / "raw" / "clean", "clean raw root path mismatch")
    _require(uniform_raw == out_dir / "raw" / "uniform_dr", "Uniform DR raw root path mismatch")
    _require(Path(args.clean_aggregate).resolve() == clean_raw / "aggregate_summary.json", "clean aggregate path mismatch")
    _require(Path(args.uniform_aggregate).resolve() == uniform_raw / "aggregate_summary.json", "Uniform DR aggregate path mismatch")
    return out_dir, clean_raw, uniform_raw


def finalize(args):
    _validate_mode(
        args.smoke,
        args.expected_seeds,
        args.expected_episodes_per_seed,
        args.expected_max_steps,
        args.bootstrap_reps,
        args.bootstrap_seed,
    )
    out_dir, clean_raw, uniform_raw = _validate_paths(args)
    _require(out_dir.is_dir(), "output directory is missing")
    clean_source, uniform_source = audit_sources(args, load_models=False)
    clean_records, clean_per_seed = _load_model_data(
        clean_raw,
        args.clean_model,
        CLEAN_MODE,
        args.expected_seeds,
        args.expected_episodes_per_seed,
        args.expected_max_steps,
    )
    uniform_records, uniform_per_seed = _load_model_data(
        uniform_raw,
        args.uniform_model,
        UNIFORM_MODE,
        args.expected_seeds,
        args.expected_episodes_per_seed,
        args.expected_max_steps,
    )
    pairing = _audit_pairing(clean_records, uniform_records)
    full_9d = _audit_full_9d(clean_records, uniform_records, args.smoke)
    clean_metrics = _metrics(clean_records)
    uniform_metrics = _metrics(uniform_records)
    clean_aggregate = _load_json(args.clean_aggregate, "clean aggregate")
    uniform_aggregate = _load_json(args.uniform_aggregate, "Uniform DR aggregate")
    clean_crosscheck = _audit_aggregate_values(
        clean_aggregate,
        clean_metrics,
        "clean_baseline_full9d_formal_test",
        CLEAN_MODE,
        args.expected_seeds,
        args.expected_episodes_per_seed,
    )
    uniform_crosscheck = _audit_aggregate_values(
        uniform_aggregate,
        uniform_metrics,
        "uniform_dr_full9d_formal_test",
        UNIFORM_MODE,
        args.expected_seeds,
        args.expected_episodes_per_seed,
    )
    paired = paired_stratified_bootstrap(
        clean_records,
        uniform_records,
        args.expected_seeds,
        args.bootstrap_reps,
        args.bootstrap_seed,
        allow_zero_valid=args.smoke,
    )
    success_discordant = _discordant(clean_records, uniform_records, "success")
    found_discordant = _discordant(clean_records, uniform_records, "found")
    primary = paired["success_rate"]
    conclusion = performance_conclusion(
        primary["point_estimate"],
        [primary["ci95_low"], primary["ci95_high"]],
    )
    clean_by_seed = {row["seed"]: row["metrics"] for row in clean_per_seed}
    uniform_by_seed = {row["seed"]: row["metrics"] for row in uniform_per_seed}
    per_seed = []
    for seed in args.expected_seeds:
        per_seed.append(
            {
                "seed": seed,
                "clean": clean_by_seed[seed],
                "uniform_dr": uniform_by_seed[seed],
                "paired_delta": _point_deltas(
                    clean_by_seed[seed], uniform_by_seed[seed]
                ),
            }
        )
    summary = {
        "overall_pass": True,
        "experiment_type": "clean_vs_uniform_dr_formal_robust_test",
        "test_stage": "smoke" if args.smoke else "formal",
        "formal_test_completed": not args.smoke,
        "models_frozen": True,
        "checkpoint_selection_performed": False,
        "test_used_for_selection": False,
        "evaluation_protocol": PROTOCOL,
        "evaluation_protocol_version": PROTOCOL_VERSION,
        "profile": PROFILE,
        "paired_episode_seeding": True,
        "episode_seed_formula": PAIRED_SEED_FORMULA,
        "disturbance_seed_formula": PAIRED_SEED_FORMULA,
        "test_seeds": args.expected_seeds,
        "episodes_per_seed": args.expected_episodes_per_seed,
        "total_episodes_per_model": len(clean_records),
        "total_rollout_episodes": len(clean_records) + len(uniform_records),
        "max_steps": args.expected_max_steps,
        "bootstrap_reps": args.bootstrap_reps,
        "bootstrap_seed": args.bootstrap_seed,
        "clean_model": clean_source,
        "uniform_dr_model": uniform_source,
        "pairing_audit": pairing,
        "disturbance_protocol_audit": full_9d,
        "clean_metrics": clean_metrics,
        "uniform_dr_metrics": uniform_metrics,
        "per_seed_metrics": per_seed,
        "aggregate_crosscheck": {
            "clean": clean_crosscheck,
            "uniform_dr": uniform_crosscheck,
            "overall_pass": True,
        },
        "paired_comparison": paired,
        "discordant_outcomes": {
            "success": success_discordant,
            "found": found_discordant,
        },
        "primary_success_delta": primary["point_estimate"],
        "primary_success_delta_ci95": [
            primary["ci95_low"],
            primary["ci95_high"],
        ],
        "primary_success_mcnemar_p": success_discordant[
            "exact_mcnemar"
        ]["exact_two_sided_p"],
        "uniform_dr_success_point_improved": primary["point_estimate"] > 0,
        "uniform_dr_success_ci_excludes_zero": not primary[
            "ci_includes_zero"
        ],
        "performance_conclusion": conclusion,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "errors": [],
        "warnings": (
            ["Smoke performance has no scientific interpretation."]
            if args.smoke
            else []
        ),
    }
    output_paths = [
        out_dir / "formal_robust_comparison_summary.json",
        out_dir / "formal_robust_comparison_summary.md",
        out_dir / "formal_robust_comparison_table.csv",
    ]
    _require(not any(path.exists() for path in output_paths), "one or more unified comparison outputs already exist")
    try:
        _atomic_write_json(output_paths[0], summary)
        _atomic_write_text(output_paths[1], _markdown(summary))
        _write_table(output_paths[2], _comparison_table(summary))
    except Exception:
        for path in output_paths:
            if path.exists():
                path.unlink()
        raise
    return summary


def _synthetic_records(seed=909, episodes=4):
    records = {}
    for index in range(episodes):
        record = {
            "base_seed": seed,
            "episode_index": index,
            "episode_seed": seed * 1_000_003 + index,
            "episode_seed_mode": PAIRED_SEED_MODE,
            "disturbance_seed": seed * 1_000_003 + index,
            "disturbance_protocol": PROTOCOL,
            "disturbance_protocol_version": PROTOCOL_VERSION,
            "disturbance_explicitly_applied": 1,
            "disturbance_apply_match": 1,
            "success": 1 if index % 2 == 0 else 0,
            "found": 1,
            "reward": 1.0 + index,
            "recovery_time": 10.0 + index,
            "safety_cost": 2.0 + index,
            "final_distance": 3.0 + index,
            "final_nav_distance": 2.0 + index,
            "completion_steps": 15.0 + index,
            "action_smoothness": 0.1 + index * 0.01,
            "flow_phase_x": 0.2 + index,
            "flow_phase_y": 0.4 + index,
        }
        for field, (low, high) in DEFAULT_DISTURBANCE_BOUNDS.items():
            if field == "action_delay_steps":
                record[field] = index % 4
            else:
                fraction = (index + 1) / float(episodes + 1)
                record[field] = float(low) + (float(high) - float(low)) * fraction
        records[(seed, index)] = record
    return records


def _expect_failure(action, label):
    try:
        action()
    except ComparisonError:
        return {"case": label, "pass": True}
    raise ComparisonError("%s did not fail as required" % label)


def run_self_tests():
    results = []
    conclusion_cases = (
        ("A", 0.10, [0.02, 0.18], "significant_improvement"),
        ("B", 0.05, [-0.02, 0.12], "positive_trend_not_significant"),
        ("C", 0.0, [-0.05, 0.05], "no_material_difference"),
        ("D", -0.05, [-0.12, 0.02], "negative_trend_not_significant"),
        ("E", -0.10, [-0.18, -0.02], "significant_degradation"),
    )
    for label, delta, interval, expected in conclusion_cases:
        actual = performance_conclusion(delta, interval)
        _require(actual == expected, "Case %s conclusion mismatch" % label)
        results.append({"case": label, "pass": True, "result": actual})
    clean = _synthetic_records()
    uniform = copy.deepcopy(clean)
    changed = copy.deepcopy(uniform)
    changed[(909, 0)]["episode_seed"] += 1
    results.append(_expect_failure(lambda: _audit_pairing(clean, changed), "F"))
    changed = copy.deepcopy(uniform)
    changed[(909, 0)]["flow_gain"] += 0.01
    results.append(_expect_failure(lambda: _audit_pairing(clean, changed), "G"))
    changed = copy.deepcopy(uniform)
    changed[(909, 0)]["flow_phase_x"] += 0.01
    results.append(_expect_failure(lambda: _audit_pairing(clean, changed), "H"))
    changed = copy.deepcopy(uniform)
    changed[(909, 0)]["disturbance_apply_match"] = 0
    results.append(_expect_failure(lambda: _audit_pairing(clean, changed), "I"))
    changed = copy.deepcopy(uniform)
    changed[(909, 0)]["flow_gain"] = DEFAULT_DISTURBANCE_BOUNDS["flow_gain"][1] + 1.0
    results.append(_expect_failure(lambda: _audit_bounds(changed), "J"))
    results.append(
        _expect_failure(
            lambda: _require_hash_value("0" * 64, CLEAN_SHA256, "synthetic"),
            "K",
        )
    )
    results.append(
        _expect_failure(
            lambda: _validate_mode(
                False, [1, 2], 200, 400, 20000, BOOTSTRAP_SEED
            ),
            "L",
        )
    )
    results.append(
        _expect_failure(
            lambda: _validate_mode(
                True, [1], 2, 20, 200, BOOTSTRAP_SEED
            ),
            "M",
        )
    )
    metrics = _metrics(clean)
    aggregate = {
        "summary_name": "clean_baseline_full9d_formal_test",
        "method": CLEAN_MODE,
        "profile": PROFILE,
        "seeds": [909],
        "episodes_per_seed": [4],
        "total_episodes": metrics["total_episodes"],
        "total_success": metrics["total_success"] + 1,
        "total_found": metrics["total_found"],
        "total_not_found": metrics["total_not_found"],
        "total_found_but_failed": metrics["total_found_but_failed"],
    }
    for metric_field, aggregate_field in AGGREGATE_FIELDS.items():
        aggregate[aggregate_field] = metrics[metric_field]
    results.append(
        _expect_failure(
            lambda: _audit_aggregate_values(
                aggregate,
                metrics,
                "clean_baseline_full9d_formal_test",
                CLEAN_MODE,
                [909],
                4,
            ),
            "N",
        )
    )
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        model_path = root / "selected_by_nominal_validation.pt"
        manifest_path = root / "selected_model_manifest.json"
        selection_path = root / "final_checkpoint_selection_summary.json"
        nominal_path = root / "clean_nominal_formal_test_summary.json"
        valid_nominal = {
            "overall_pass": True,
            "experiment_type": "clean_nominal_formal_test",
            "test_stage": "formal_test",
            "formal_test_completed": True,
            "errors": [],
            "selected_checkpoint_name": CLEAN_CHECKPOINT,
            "selected_checkpoint_episode": CLEAN_CHECKPOINT_EPISODE,
            "model_sha256": CLEAN_SHA256,
            "hash_match": True,
            "selected_model_path": str(model_path),
            "selection_summary": str(selection_path),
            "selected_manifest": str(manifest_path),
            "test_seeds": list(FORMAL_SEEDS),
            "episodes_per_seed": FORMAL_EPISODES_PER_SEED,
            "total_episodes": 600,
            "max_steps": FORMAL_MAX_STEPS,
            "mode": CLEAN_MODE,
            "profile": PROFILE,
            "checkpoint_selection_performed": False,
            "test_used_for_selection": False,
            "selected_checkpoint_changed": False,
            "manifest_updated": True,
            "episode_randomization_audit": {
                "enabled": True,
                "episode_seed_mode": PAIRED_SEED_MODE,
                "episode_seed_formula": PAIRED_SEED_FORMULA,
                "episode_index_base": 0,
                "total_episode_keys": 600,
                "all_episode_keys_unique": True,
                "all_episode_seeds_valid": True,
            },
            "csv_aggregate_crosscheck": {
                "passed": True,
                "count_fields_match": True,
                "continuous_fields_match": True,
                "relative_tolerance": 1e-7,
                "absolute_tolerance": 1e-9,
            },
            "performance_targets": {"performance_target_pass": True},
            "seed_stability": {"seed_stability_pass": True},
        }

        def write_nominal(value):
            nominal_path.write_text(
                json.dumps(value, indent=2),
                encoding="utf-8",
            )

        write_nominal(valid_nominal)
        nominal_audit = _audit_clean_nominal_test(
            nominal_path,
            model_path,
            manifest_path,
            selection_path,
        )
        _require(nominal_audit["overall_pass"] is True, "Case O did not pass")
        results.append({"case": "O", "pass": True})
        results.append(
            _expect_failure(
                lambda: _audit_clean_nominal_test(
                    root / "missing.json",
                    model_path,
                    manifest_path,
                    selection_path,
                ),
                "P",
            )
        )
        mutations = (
            ("Q", ("overall_pass",), False),
            ("R", ("formal_test_completed",), False),
            ("S", ("model_sha256",), "0" * 64),
            ("T", ("selected_model_path",), str(root / "wrong.pt")),
            ("U", ("test_seeds",), [1, 2]),
            (
                "V",
                ("performance_targets", "performance_target_pass"),
                False,
            ),
            ("W", ("seed_stability", "seed_stability_pass"), False),
            ("X", ("csv_aggregate_crosscheck", "passed"), False),
            ("Y", ("test_used_for_selection",), True),
            ("Z", ("selected_checkpoint_changed",), True),
        )
        for label, key_path, value in mutations:
            invalid = copy.deepcopy(valid_nominal)
            target = invalid
            for key in key_path[:-1]:
                target = target[key]
            target[key_path[-1]] = value
            write_nominal(invalid)
            results.append(
                _expect_failure(
                    lambda path=nominal_path: _audit_clean_nominal_test(
                        path,
                        model_path,
                        manifest_path,
                        selection_path,
                    ),
                    label,
                )
            )
    expected_case_names = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    actual_case_names = [row["case"] for row in results]
    _require(
        actual_case_names == expected_case_names
        and all(row["pass"] for row in results),
        "self-test result set mismatch",
    )
    return results


def _add_source_arguments(parser):
    parser.add_argument("--clean-model", required=True)
    parser.add_argument("--clean-manifest", required=True)
    parser.add_argument("--clean-selection-summary", required=True)
    parser.add_argument("--clean-nominal-test-summary", required=True)
    parser.add_argument("--uniform-model", required=True)
    parser.add_argument("--uniform-manifest", required=True)
    parser.add_argument("--uniform-selection-summary", required=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Finalize frozen Clean versus Uniform DR full-9D robust testing."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser(
        "preflight", help="Audit frozen identities and CPU checkpoint loading."
    )
    _add_source_arguments(preflight)
    finalize_parser = subparsers.add_parser(
        "finalize", help="Audit raw results and write the unified comparison."
    )
    _add_source_arguments(finalize_parser)
    finalize_parser.add_argument("--clean-raw-root", required=True)
    finalize_parser.add_argument("--uniform-raw-root", required=True)
    finalize_parser.add_argument("--clean-aggregate", required=True)
    finalize_parser.add_argument("--uniform-aggregate", required=True)
    finalize_parser.add_argument("--out-dir", required=True)
    finalize_parser.add_argument(
        "--expected-seeds", type=int, nargs="+", required=True
    )
    finalize_parser.add_argument(
        "--expected-episodes-per-seed", type=int, required=True
    )
    finalize_parser.add_argument("--expected-max-steps", type=int, required=True)
    finalize_parser.add_argument("--bootstrap-reps", type=int, required=True)
    finalize_parser.add_argument(
        "--bootstrap-seed", type=int, default=BOOTSTRAP_SEED
    )
    finalize_parser.add_argument("--smoke", action="store_true")
    subparsers.add_parser(
        "self-test", help="Run temporary and in-memory Cases A through Z."
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        if args.command == "preflight":
            clean, uniform = audit_sources(args, load_models=True)
            print(
                json.dumps(
                    {
                        "overall_pass": True,
                        "clean_model": clean,
                        "uniform_dr_model": uniform,
                        "models_frozen": True,
                        "checkpoint_selection_performed": False,
                        "test_used_for_selection": False,
                        "errors": [],
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
        elif args.command == "self-test":
            results = run_self_tests()
            print(
                json.dumps(
                    {"overall_pass": True, "cases": results, "errors": []},
                    indent=2,
                    ensure_ascii=False,
                )
            )
        else:
            summary = finalize(args)
            print(
                "[PASS] Clean versus Uniform DR finalized: stage=%s total_rollout=%d"
                % (summary["test_stage"], summary["total_rollout_episodes"])
            )
        return 0
    except Exception as exc:
        print(
            "[ERROR] Clean versus Uniform DR finalization failed: %s: %s"
            % (type(exc).__name__, exc),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
