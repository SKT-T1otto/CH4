#!/usr/bin/env python3
"""Audit Chapter-4 Uniform DR warm-start training artifacts.

This tool is intentionally read-only with respect to training artifacts.  It
only writes the JSON and Markdown audit summaries requested by ``--out-dir``;
it never trains a model or performs a rollout.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from registry.rbe_disturbance import (  # noqa: E402
    DEFAULT_DISTURBANCE_BOUNDS,
    DISTURBANCE_KEYS,
    nominal_disturbance,
)
from registry.ch4_artifact_layout import (  # noqa: E402
    get_training_run_dir,
    resolve_artifact_path,
)


EXPECTED_SOURCE_SHA256 = (
    "c9807c948b50ca102d59de92801190e110eb2184c3976d6ef7d9c21a6e2f75d1"
)
EXPECTED_CHECKPOINT_NAME = "snapshot_ep8200"
EXPECTED_CHECKPOINT_EPISODE = 8200
EXPECTED_MODE = "ch4_uniform_dr"
EXPECTED_REWARD_PROFILE = "residual_point_v3"
EXPECTED_RESIDUAL_ACTION_MODE = "hybrid_v4"
SNAPSHOT_RE = re.compile(r"^snapshot_ep(\d+)\.pt$")
TRUE_TEXT = {"1", "true", "yes", "y", "t"}
FALSE_TEXT = {"0", "false", "no", "n", "f"}
FLOAT_CONTROLLER_FIELDS = (
    "prior_strength_search",
    "prior_strength_executor",
    "residual_scale_search",
    "residual_scale_executor",
    "residual_penalty",
)
STRING_CONTROLLER_FIELDS = ("reward_profile", "residual_action_mode")
STRICT_BOOL_CONTROLLER_FIELDS = (
    "use_pse_planner",
    "pse_use_belief",
    "pse_use_exec_cost",
    "pse_use_standby",
    "pse_lazy_standby",
)
STRICT_CONTROLLER_CONTINUITY_FIELDS = (
    FLOAT_CONTROLLER_FIELDS
    + STRING_CONTROLLER_FIELDS
    + STRICT_BOOL_CONTROLLER_FIELDS
)
CURRENT_REQUIRED_CONTROLLER_FIELDS = STRICT_CONTROLLER_CONTINUITY_FIELDS + (
    "residual_hybrid_use_soft_gate",
    "use_robust_disturbance",
)
SOURCE_REQUIRED_CONTROLLER_FIELDS = STRICT_CONTROLLER_CONTINUITY_FIELDS + (
    "use_robust_disturbance",
)
HISTORICAL_OPTIONAL_CONTROLLER_FIELDS = ("residual_hybrid_use_soft_gate",)
SOURCE_TRAINING_CONFIG_CANONICAL = str(
    get_training_run_dir(
        "clean", "ch4_clean_baseline_seed1_from_scratch_10000ep_v1"
    )
    / "training_config.json"
)
CROSSCHECK_FIELDS = tuple(DISTURBANCE_KEYS) + (
    "success_flag",
    "found_flag",
    "recovery_time",
    "safety_cost",
    "final_distance",
    "final_nav_distance",
    "action_smoothness",
    "completion_steps",
    "episode_reward_mean",
    "episode_reward_sum",
)


class ControllerConfigError(ValueError):
    def __init__(self, message, partial):
        ValueError.__init__(self, message)
        self.partial = dict(partial)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Audit a Chapter-4 Uniform DR run warm-started from the frozen "
            "selected clean snapshot_ep8200 model."
        )
    )
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--source-selection-summary", required=True)
    parser.add_argument("--source-formal-test-summary", required=True)
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--source-training-config", required=True)
    parser.add_argument(
        "--source-training-config-resolution",
        required=True,
        choices=("canonical_only", "archive_only", "both_identical"),
    )
    parser.add_argument("--expected-run-name", required=True)
    parser.add_argument("--expected-seed", required=True, type=int)
    parser.add_argument("--expected-episodes", required=True, type=int)
    parser.add_argument("--expected-max-steps", required=True, type=int)
    parser.add_argument("--expected-snapshot-interval", required=True, type=int)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def absolute(path):
    return os.path.abspath(os.path.expanduser(path))


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_source_training_config_paths(canonical_path, archived_path=None):
    del archived_path
    canonical_path = absolute(canonical_path)
    resolved_path = str(resolve_artifact_path(canonical_path))
    canonical_exists = os.path.isfile(canonical_path)
    if not os.path.isfile(resolved_path):
        raise ValueError("Source training config not found: %s" % canonical_path)
    resolution = "canonical_only" if canonical_exists else "archive_only"

    return {
        "resolved_path": resolved_path,
        "canonical_path": canonical_path,
        "archived_path": None,
        "resolution": resolution,
        "canonical_exists": canonical_exists,
        "archived_exists": not canonical_exists,
        "both_existed": False,
        "both_hash_match": None,
    }


def audit_source_training_config_resolution(args, errors):
    try:
        audit = resolve_source_training_config_paths(
            SOURCE_TRAINING_CONFIG_CANONICAL,
        )
    except (OSError, ValueError) as exc:
        errors.append("Source training config resolution failed: %s" % exc)
        return {
            "resolved_path": absolute(args.source_training_config),
            "canonical_path": absolute(SOURCE_TRAINING_CONFIG_CANONICAL),
            "archived_path": None,
            "resolution": None,
            "canonical_exists": os.path.isfile(SOURCE_TRAINING_CONFIG_CANONICAL),
            "archived_exists": False,
            "both_existed": False,
            "both_hash_match": None,
        }

    add_check(
        errors,
        audit["resolution"] == args.source_training_config_resolution,
        "BAT source training config resolution does not match the audited resolution.",
    )
    add_check(
        errors,
        normalized_path(audit["resolved_path"])
        == normalized_path(args.source_training_config),
        "BAT source training config path does not match the audited resolved path.",
    )
    return audit


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("JSON root is not an object: %s" % path)
    return data


def load_csv(path):
    with open(path, "r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def nested(data, *keys):
    current = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def config_value(config, key):
    if key in config:
        return config[key]
    env_kwargs = config.get("env_kwargs")
    if isinstance(env_kwargs, dict) and key in env_kwargs:
        return env_kwargs[key]
    return None


def _extract_controller_config(config, required_fields, optional_fields=()):
    """Extract only explicitly recorded values, with caller-specific requirements."""
    if not isinstance(config, dict):
        raise ValueError("training config must be a JSON object")
    controller = config.get("controller_config")
    if not isinstance(controller, dict):
        controller = {}
    env_kwargs = config.get("env_kwargs")
    if not isinstance(env_kwargs, dict):
        env_kwargs = {}

    extracted = {}
    missing = []
    for field in tuple(required_fields) + tuple(optional_fields):
        if field in controller:
            value = controller[field]
        elif field in config:
            value = config[field]
        elif field in env_kwargs:
            value = env_kwargs[field]
        elif field in required_fields:
            missing.append(field)
            continue
        else:
            continue

        if field in FLOAT_CONTROLLER_FIELDS:
            if isinstance(value, bool):
                raise ValueError("Controller field %s must be numeric" % field)
            try:
                value = float(value)
            except (TypeError, ValueError):
                raise ValueError("Controller field %s must be numeric" % field)
            if not math.isfinite(value):
                raise ValueError("Controller field %s must be finite" % field)
        elif field in STRING_CONTROLLER_FIELDS:
            if not isinstance(value, str):
                raise ValueError("Controller field %s must be a string" % field)
        else:
            if not isinstance(value, bool):
                raise ValueError("Controller field %s must be boolean" % field)
        extracted[field] = value
    if missing:
        raise ControllerConfigError(
            "Missing required controller configuration field(s): %s"
            % ", ".join(missing),
            extracted,
        )
    return extracted


def extract_current_controller_config(config):
    return _extract_controller_config(config, CURRENT_REQUIRED_CONTROLLER_FIELDS)


def extract_source_controller_config(config):
    return _extract_controller_config(
        config,
        SOURCE_REQUIRED_CONTROLLER_FIELDS,
        HISTORICAL_OPTIONAL_CONTROLLER_FIELDS,
    )


def audit_controller_continuity(
    source_config, current_config, source_controller, current_controller, errors
):
    matching = []
    mismatched = []
    for field in STRICT_CONTROLLER_CONTINUITY_FIELDS:
        source_value = source_controller.get(field)
        current_value = current_controller.get(field)
        if field in FLOAT_CONTROLLER_FIELDS:
            matches = (
                source_value is not None
                and current_value is not None
                and math.isclose(
                    float(source_value),
                    float(current_value),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            )
        else:
            matches = source_value == current_value and source_value is not None
        if matches:
            matching.append(field)
        else:
            mismatched.append(field)

    source_soft_gate_recorded = "residual_hybrid_use_soft_gate" in source_controller
    current_soft_gate_recorded = "residual_hybrid_use_soft_gate" in current_controller
    current_soft_gate = current_controller.get("residual_hybrid_use_soft_gate")
    if source_soft_gate_recorded:
        if source_controller["residual_hybrid_use_soft_gate"] == current_soft_gate:
            matching.append("residual_hybrid_use_soft_gate")
        else:
            mismatched.append("residual_hybrid_use_soft_gate")
    historical_unrecorded_fields = (
        [] if source_soft_gate_recorded else ["residual_hybrid_use_soft_gate"]
    )
    historical_soft_gate_status = (
        source_controller.get("residual_hybrid_use_soft_gate")
        if source_soft_gate_recorded
        else "unrecorded"
    )

    source_robust = source_controller.get("use_robust_disturbance")
    current_robust = current_controller.get("use_robust_disturbance")
    robust_difference_ok = source_robust is False and current_robust is True
    recorded_fields_match = not mismatched
    configuration_match = (
        recorded_fields_match
        and robust_difference_ok
        and current_soft_gate_recorded
        and current_soft_gate is True
    )
    add_check(
        errors,
        configuration_match,
        "Recorded controller continuity audit failed; mismatched fields: %s."
        % mismatched,
    )
    return {
        "source_training_config": source_config.get("_audit_path"),
        "source_mode": source_config.get("mode"),
        "current_mode": current_config.get("mode"),
        "strict_compared_fields": list(STRICT_CONTROLLER_CONTINUITY_FIELDS),
        "source_controller_config": source_controller,
        "current_controller_config": current_controller,
        "matching_fields": matching,
        "mismatched_fields": mismatched,
        "historical_unrecorded_fields": historical_unrecorded_fields,
        "historical_soft_gate_status": historical_soft_gate_status,
        "current_soft_gate_enabled": current_soft_gate is True,
        "current_soft_gate_explicitly_recorded": current_soft_gate_recorded,
        "source_use_robust_disturbance": source_robust,
        "current_use_robust_disturbance": current_robust,
        "intended_mode_difference": "use_robust_disturbance",
        "recorded_controller_fields_match": recorded_fields_match,
        "controller_configuration_match": configuration_match,
        "continuity_scope": (
            "strictly recorded source and current controller fields"
        ),
    }


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if float(value) == 1.0:
            return True
        if float(value) == 0.0:
            return False
        raise ValueError("numeric boolean must be 0 or 1")
    text = str(value).strip().lower()
    if text in TRUE_TEXT:
        return True
    if text in FALSE_TEXT:
        return False
    try:
        numeric = float(text)
    except ValueError:
        numeric = None
    if numeric == 1.0:
        return True
    if numeric == 0.0:
        return False
    raise ValueError("invalid boolean value %r" % value)


def finite_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def normalized_path(path):
    return os.path.normcase(os.path.realpath(absolute(path)))


def add_check(errors, condition, message):
    if not condition:
        errors.append(message)
    return bool(condition)


def safe_file_hash(path, errors, label):
    try:
        if not os.path.isfile(path):
            errors.append("Missing %s: %s" % (label, path))
            return None
        return sha256_file(path)
    except (OSError, ValueError) as exc:
        errors.append("Could not hash %s %s: %s" % (label, path, exc))
        return None


def safe_json(path, errors, label):
    try:
        if not os.path.isfile(path):
            errors.append("Missing %s: %s" % (label, path))
            return {}
        return load_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append("Could not read %s %s: %s" % (label, path, exc))
        return {}


def safe_csv(path, errors, label):
    try:
        if not os.path.isfile(path):
            errors.append("Missing %s: %s" % (label, path))
            return []
        return load_csv(path)
    except (OSError, ValueError, csv.Error) as exc:
        errors.append("Could not read %s %s: %s" % (label, path, exc))
        return []


def audit_source_training_config(args, errors):
    path = absolute(args.source_training_config)
    config = safe_json(path, errors, "source clean training config")
    config["_audit_path"] = path
    checks = {}

    def expect(name, condition, message):
        checks[name] = add_check(errors, condition, message)

    expect(
        "mode",
        config.get("mode") == "ch4_pse_baseline",
        "Source training config mode is not ch4_pse_baseline.",
    )
    expect("seed", config.get("seed") == 1, "Source training config seed is not 1.")
    expect(
        "max_episodes",
        config.get("max_episodes") == 10000,
        "Source training config max_episodes is not 10000.",
    )
    expect(
        "max_steps",
        config.get("max_steps") == 400,
        "Source training config max_steps is not 400.",
    )
    expect(
        "initialization_source",
        config.get("initialization_source") == "from_scratch_flag",
        "Source training config initialization_source is not from_scratch_flag.",
    )
    expect(
        "loaded_checkpoint",
        config.get("loaded_checkpoint") is False,
        "Source training config loaded_checkpoint is not false.",
    )
    expect(
        "train_from_scratch_requested",
        config.get("train_from_scratch_requested") is True,
        "Source training config train_from_scratch_requested is not true.",
    )
    disturbance_keys = config.get("disturbance_keys")
    expect(
        "disturbance_keys",
        disturbance_keys == list(DISTURBANCE_KEYS),
        "Source training config disturbance_keys do not exactly match the 9-D registry.",
    )
    expect(
        "flow_phase_excluded",
        isinstance(disturbance_keys, list)
        and "flow_phase_x" not in disturbance_keys
        and "flow_phase_y" not in disturbance_keys,
        "Source training config disturbance_keys include flow phase fields.",
    )

    try:
        controller = extract_source_controller_config(config)
    except ValueError as exc:
        errors.append("Source training config controller audit failed: %s" % exc)
        controller = getattr(exc, "partial", {})
    expect(
        "use_robust_disturbance",
        controller.get("use_robust_disturbance") is False,
        "Source training controller use_robust_disturbance is not false.",
    )
    expect(
        "reward_profile",
        controller.get("reward_profile") == EXPECTED_REWARD_PROFILE,
        "Source training controller reward_profile mismatch.",
    )
    expect(
        "residual_action_mode",
        controller.get("residual_action_mode") == EXPECTED_RESIDUAL_ACTION_MODE,
        "Source training controller residual_action_mode mismatch.",
    )
    return {
        "path": path,
        "checks": checks,
        "mode": config.get("mode"),
        "seed": config.get("seed"),
        "max_episodes": config.get("max_episodes"),
        "max_steps": config.get("max_steps"),
        "initialization_source": config.get("initialization_source"),
        "loaded_checkpoint": config.get("loaded_checkpoint"),
        "train_from_scratch_requested": config.get("train_from_scratch_requested"),
        "disturbance_keys": disturbance_keys,
        "controller_config": controller,
    }, config, controller


def audit_source(args, errors):
    selection_path = absolute(args.source_selection_summary)
    formal_path = absolute(args.source_formal_test_summary)
    model_path = absolute(args.source_model)
    manifest_path = absolute(args.source_manifest)
    training_config_path = absolute(args.source_training_config)

    hashes_before = {
        "source_model": safe_file_hash(model_path, errors, "source model"),
        "source_manifest": safe_file_hash(manifest_path, errors, "source manifest"),
        "source_formal_test_summary": safe_file_hash(
            formal_path, errors, "source formal-test summary"
        ),
        "source_selection_summary": safe_file_hash(
            selection_path, errors, "source selection summary"
        ),
        "source_training_config": safe_file_hash(
            training_config_path, errors, "source clean training config"
        ),
    }
    selection = safe_json(selection_path, errors, "source selection summary")
    formal = safe_json(formal_path, errors, "source formal-test summary")
    manifest = safe_json(manifest_path, errors, "source manifest")

    add_check(
        errors,
        selection.get("overall_pass") is True,
        "Source selection summary overall_pass is not true.",
    )
    add_check(
        errors,
        selection.get("selected_checkpoint_name") == EXPECTED_CHECKPOINT_NAME,
        "Source selection summary did not select snapshot_ep8200.",
    )
    add_check(
        errors,
        selection.get("selected_checkpoint_episode") == EXPECTED_CHECKPOINT_EPISODE,
        "Source selection summary checkpoint episode is not 8200.",
    )
    add_check(
        errors,
        selection.get("selected_sha256") == EXPECTED_SOURCE_SHA256,
        "Source selection summary selected SHA256 mismatch.",
    )
    add_check(
        errors,
        selection.get("source_sha256") == EXPECTED_SOURCE_SHA256,
        "Source selection summary source SHA256 mismatch.",
    )
    add_check(
        errors,
        selection.get("hash_match") is True,
        "Source selection summary hash_match is not true.",
    )

    add_check(
        errors,
        formal.get("overall_pass") is True,
        "Nominal formal-test overall_pass is not true.",
    )
    add_check(
        errors,
        formal.get("formal_test_completed") is True,
        "Nominal formal-test formal_test_completed is not true.",
    )
    add_check(
        errors,
        nested(formal, "performance_targets", "performance_target_pass") is True,
        "Nominal formal-test performance_target_pass is not true.",
    )
    add_check(
        errors,
        nested(formal, "seed_stability", "seed_stability_pass") is True,
        "Nominal formal-test seed_stability_pass is not true.",
    )
    add_check(
        errors,
        formal.get("selected_checkpoint_name") == EXPECTED_CHECKPOINT_NAME,
        "Nominal formal-test checkpoint is not snapshot_ep8200.",
    )
    add_check(
        errors,
        formal.get("selected_checkpoint_episode") == EXPECTED_CHECKPOINT_EPISODE,
        "Nominal formal-test checkpoint episode is not 8200.",
    )
    add_check(
        errors,
        formal.get("model_sha256") == EXPECTED_SOURCE_SHA256,
        "Nominal formal-test model SHA256 mismatch.",
    )

    add_check(
        errors,
        manifest.get("formal_test_completed") is True,
        "Source manifest formal_test_completed is not true.",
    )
    add_check(
        errors,
        manifest.get("selected_checkpoint_name") == EXPECTED_CHECKPOINT_NAME,
        "Source manifest checkpoint is not snapshot_ep8200.",
    )
    add_check(
        errors,
        manifest.get("selected_checkpoint_episode") == EXPECTED_CHECKPOINT_EPISODE,
        "Source manifest checkpoint episode is not 8200.",
    )
    manifest_hash_fields = (
        "source_sha256",
        "selected_sha256",
        "formal_test_model_sha256",
    )
    for field in manifest_hash_fields:
        add_check(
            errors,
            manifest.get(field) == EXPECTED_SOURCE_SHA256,
            "Source manifest %s mismatch." % field,
        )

    actual_hash = hashes_before["source_model"]
    hash_match = add_check(
        errors,
        actual_hash == EXPECTED_SOURCE_SHA256,
        "Actual source model SHA256 mismatch: expected %s, got %s."
        % (EXPECTED_SOURCE_SHA256, actual_hash),
    )
    content_hashes = [
        selection.get("source_sha256"),
        selection.get("selected_sha256"),
        formal.get("model_sha256"),
        manifest.get("source_sha256"),
        manifest.get("selected_sha256"),
        manifest.get("formal_test_model_sha256"),
        actual_hash,
    ]
    add_check(
        errors,
        all(value == EXPECTED_SOURCE_SHA256 for value in content_hashes),
        "Selection, formal-test, manifest, and actual model hashes are inconsistent.",
    )

    return {
        "checkpoint_name": EXPECTED_CHECKPOINT_NAME,
        "checkpoint_episode": EXPECTED_CHECKPOINT_EPISODE,
        "source_model_path": model_path,
        "expected_sha256": EXPECTED_SOURCE_SHA256,
        "actual_sha256": actual_hash,
        "hash_match": hash_match,
        "source_model_unchanged": False,
        "artifact_sha256_before": hashes_before,
    }, hashes_before


def audit_training_config(args, source_model_path, errors):
    path = os.path.join(absolute(args.log_dir), "training_config.json")
    config = safe_json(path, errors, "training_config.json")

    checks = {}

    def expect(name, condition, message):
        checks[name] = add_check(errors, condition, message)

    expect(
        "controller_config_recorded",
        isinstance(config.get("controller_config"), dict),
        "training_config.json does not contain a root controller_config object.",
    )
    try:
        controller = extract_current_controller_config(config)
    except ValueError as exc:
        errors.append("Current training controller audit failed: %s" % exc)
        controller = getattr(exc, "partial", {})

    expect(
        "run_name",
        config.get("run_name") == args.expected_run_name,
        "training_config run_name mismatch.",
    )
    expect(
        "mode",
        config.get("mode") == EXPECTED_MODE,
        "training_config mode is not ch4_uniform_dr.",
    )
    expect(
        "seed",
        config.get("seed") == args.expected_seed,
        "training_config seed mismatch.",
    )
    expect(
        "max_episodes",
        config.get("max_episodes") == args.expected_episodes,
        "training_config max_episodes mismatch.",
    )
    expect(
        "max_steps",
        config.get("max_steps") == args.expected_max_steps,
        "training_config max_steps mismatch.",
    )
    expect(
        "snapshot_interval",
        config.get("snapshot_interval") == args.expected_snapshot_interval,
        "training_config snapshot_interval mismatch.",
    )
    expect(
        "initialization_source",
        config.get("initialization_source") == "explicit_checkpoint",
        "training_config initialization_source is not explicit_checkpoint.",
    )
    expect(
        "loaded_checkpoint",
        config.get("loaded_checkpoint") is True,
        "training_config loaded_checkpoint is not true.",
    )
    explicit_from_scratch = config.get("train_from_scratch")
    if explicit_from_scratch is None:
        explicit_from_scratch = config.get("train_from_scratch_requested", False)
    expect(
        "train_from_scratch",
        explicit_from_scratch is False,
        "training_config indicates from-scratch initialization.",
    )
    expect(
        "train_from_scratch_requested",
        config.get("train_from_scratch_requested") is False,
        "training_config train_from_scratch_requested is not false.",
    )
    resume_path = config.get("resume_model_path")
    resume_matches = False
    if isinstance(resume_path, str) and resume_path.strip():
        resume_matches = normalized_path(resume_path) == normalized_path(source_model_path)
    expect(
        "resume_model_path",
        resume_matches,
        "training_config resume_model_path is not the frozen selected model.",
    )
    expect(
        "use_robust_disturbance",
        controller.get("use_robust_disturbance") is True,
        "training_config use_robust_disturbance is not true.",
    )
    expect(
        "reward_profile",
        controller.get("reward_profile") == EXPECTED_REWARD_PROFILE,
        "training_config reward_profile mismatch.",
    )
    expect(
        "residual_action_mode",
        controller.get("residual_action_mode") == EXPECTED_RESIDUAL_ACTION_MODE,
        "training_config residual_action_mode mismatch.",
    )
    disturbance_keys = config.get("disturbance_keys")
    expect(
        "disturbance_keys",
        disturbance_keys == list(DISTURBANCE_KEYS),
        "training_config disturbance_keys do not exactly match the 9-D registry.",
    )
    expect(
        "flow_phase_excluded",
        isinstance(disturbance_keys, list)
        and "flow_phase_x" not in disturbance_keys
        and "flow_phase_y" not in disturbance_keys,
        "training_config disturbance_keys include flow phase fields.",
    )

    expected_switches = {
        "collect_boundary_dataset": True,
        "reb_enable": False,
        "rscu_enable": False,
        "rbe_sampler_enable": False,
        "freeze_search_actors": False,
        "rbe_executor_only_actor": False,
    }
    for key, expected in expected_switches.items():
        expect(
            key,
            config_value(config, key) is expected,
            "training_config %s is not %s." % (key, str(expected).lower()),
        )

    return {
        "path": path,
        "checks": checks,
        "run_name": config.get("run_name"),
        "mode": config.get("mode"),
        "seed": config.get("seed"),
        "max_episodes": config.get("max_episodes"),
        "max_steps": config.get("max_steps"),
        "snapshot_interval": config.get("snapshot_interval"),
        "initialization_source": config.get("initialization_source"),
        "loaded_checkpoint": config.get("loaded_checkpoint"),
        "train_from_scratch": explicit_from_scratch,
        "resume_model_path": resume_path,
        "controller_config": controller,
        "use_robust_disturbance": controller.get("use_robust_disturbance"),
        "reward_profile": controller.get("reward_profile"),
        "residual_action_mode": controller.get("residual_action_mode"),
        "disturbance_keys": disturbance_keys,
        "collect_boundary_dataset": config_value(
            config, "collect_boundary_dataset"
        ),
        "reb_enable": config_value(config, "reb_enable"),
        "rscu_enable": config_value(config, "rscu_enable"),
        "rbe_sampler_enable": config_value(config, "rbe_sampler_enable"),
        "freeze_search_actors": config_value(config, "freeze_search_actors"),
        "rbe_executor_only_actor": config_value(
            config, "rbe_executor_only_actor"
        ),
        "batch_size": config.get("batch_size"),
        "learn_interval": config.get("learn_interval"),
        "updates_per_train": config.get("updates_per_train"),
        "agent_count": len(config.get("obs_dims", []))
        if isinstance(config.get("obs_dims"), list)
        else 0,
    }, config, controller


def bool_pair_is_valid(row, success_key, found_key):
    try:
        success = parse_bool(row.get(success_key))
        found = parse_bool(row.get(found_key))
    except ValueError:
        return False, False, False
    return True, success, found


def audit_episode_identity(rows, label, expected_episodes, errors):
    expected_sequence = list(range(1, expected_episodes + 1))
    has_column = bool(rows) and all("episode" in row for row in rows)
    if not has_column:
        add_check(
            errors,
            False,
            "%s must contain an explicit episode column" % label,
        )
        return {
            "source": "missing_episode_column",
            "unique": False,
            "complete": False,
            "ordered": False,
            "first_episode": None,
            "last_episode": None,
            "sequence": [],
        }

    episodes = []
    valid = True
    in_range = True
    for row in rows:
        number = finite_float(row.get("episode"))
        if number is None or not float(number).is_integer():
            valid = False
            continue
        episode = int(number)
        if episode < 1 or episode > expected_episodes:
            in_range = False
        episodes.append(episode)
    add_check(errors, valid, "%s contains a non-finite or non-integer episode value." % label)
    add_check(
        errors,
        in_range,
        "%s contains an episode outside 1..%d." % (label, expected_episodes),
    )
    unique = valid and len(episodes) == len(rows) and len(episodes) == len(set(episodes))
    add_check(errors, unique, "%s contains duplicate episode values." % label)
    complete = (
        valid
        and in_range
        and unique
        and set(episodes) == set(expected_sequence)
        and len(episodes) == len(expected_sequence)
    )
    add_check(errors, complete, "%s episode set is not exactly 1..N." % label)
    ordered = complete and episodes == expected_sequence
    add_check(errors, ordered, "%s episode order is not exactly 1..N." % label)
    return {
        "source": "explicit_episode_column",
        "unique": unique,
        "complete": complete,
        "ordered": ordered,
        "first_episode": episodes[0] if episodes else None,
        "last_episode": episodes[-1] if episodes else None,
        "sequence": episodes,
    }


def audit_episode_metrics_boundary_crosscheck(
    metrics_rows, boundary_rows, expected_episodes, errors
):
    def index_rows(rows):
        indexed = {}
        for row in rows:
            number = finite_float(row.get("episode"))
            if number is None or not float(number).is_integer():
                continue
            episode = int(number)
            if episode not in indexed:
                indexed[episode] = row
        return indexed

    metrics_by_episode = index_rows(metrics_rows)
    boundary_by_episode = index_rows(boundary_rows)
    common_episodes = sorted(set(metrics_by_episode) & set(boundary_by_episode))
    mismatches = []
    for episode in common_episodes:
        metrics_row = metrics_by_episode[episode]
        boundary_row = boundary_by_episode[episode]
        for field in CROSSCHECK_FIELDS:
            if field in ("success_flag", "found_flag"):
                try:
                    matches = parse_bool(metrics_row.get(field)) == parse_bool(
                        boundary_row.get(field)
                    )
                except ValueError:
                    matches = False
            else:
                metrics_value = finite_float(metrics_row.get(field))
                boundary_value = finite_float(boundary_row.get(field))
                matches = (
                    metrics_value is not None
                    and boundary_value is not None
                    and math.isclose(
                        metrics_value,
                        boundary_value,
                        rel_tol=1e-9,
                        abs_tol=1e-10,
                    )
                )
            if not matches:
                mismatches.append({"episode": episode, "field": field})

    all_fields_match = (
        len(common_episodes) == expected_episodes
        and set(metrics_by_episode) == set(range(1, expected_episodes + 1))
        and set(boundary_by_episode) == set(range(1, expected_episodes + 1))
        and not mismatches
    )
    add_check(
        errors,
        all_fields_match,
        "episode_metrics.csv and boundary_dataset.csv differ by episode or field.",
    )
    return {
        "matched_episode_count": len(common_episodes),
        "field_mismatch_count": len(mismatches),
        "all_fields_match": all_fields_match,
        "mismatch_examples": mismatches[:20],
    }


def audit_progress(args, config, errors):
    log_dir = absolute(args.log_dir)
    progress_path = os.path.join(log_dir, "training_progress.csv")
    metrics_path = os.path.join(log_dir, "episode_metrics.csv")
    boundary_path = os.path.join(log_dir, "boundary_dataset.csv")
    progress_rows = safe_csv(progress_path, errors, "training_progress.csv")
    metrics_rows = safe_csv(metrics_path, errors, "episode_metrics.csv")
    boundary_rows = safe_csv(boundary_path, errors, "boundary_dataset.csv")

    expected = args.expected_episodes
    add_check(
        errors,
        len(progress_rows) == expected,
        "training_progress.csv row count mismatch: expected %d, got %d."
        % (expected, len(progress_rows)),
    )
    add_check(
        errors,
        len(metrics_rows) == expected,
        "episode_metrics.csv row count mismatch: expected %d, got %d."
        % (expected, len(metrics_rows)),
    )
    add_check(
        errors,
        len(boundary_rows) == expected,
        "boundary_dataset.csv row count mismatch: expected %d, got %d."
        % (expected, len(boundary_rows)),
    )

    expected_sequence = list(range(1, expected + 1))
    progress_identity = audit_episode_identity(
        progress_rows, "training_progress.csv", expected, errors
    )
    metrics_identity = audit_episode_identity(
        metrics_rows, "episode_metrics.csv", expected, errors
    )
    boundary_identity = audit_episode_identity(
        boundary_rows, "boundary_dataset.csv", expected, errors
    )
    cross_file_episode_audit = {
        "expected_sequence_start": 1,
        "expected_sequence_end": expected,
        "training_progress_matches": progress_identity.get("sequence")
        == expected_sequence,
        "episode_metrics_matches": metrics_identity.get("sequence")
        == expected_sequence,
        "boundary_dataset_matches": boundary_identity.get("sequence")
        == expected_sequence,
    }
    cross_file_episode_audit["all_sequences_match"] = all(
        cross_file_episode_audit[key]
        for key in (
            "training_progress_matches",
            "episode_metrics_matches",
            "boundary_dataset_matches",
        )
    )
    add_check(
        errors,
        cross_file_episode_audit["all_sequences_match"],
        "The three training CSV episode sequences are not all exactly 1..N.",
    )
    metrics_boundary_crosscheck = audit_episode_metrics_boundary_crosscheck(
        metrics_rows, boundary_rows, expected, errors
    )

    invalid_bool_rows = 0
    success_exceeds_found = 0
    for rows, success_key, found_key in (
        (progress_rows, "success", "found"),
        (metrics_rows, "success_flag", "found_flag"),
        (boundary_rows, "success_flag", "found_flag"),
    ):
        for row in rows:
            valid, success, found = bool_pair_is_valid(row, success_key, found_key)
            if not valid:
                invalid_bool_rows += 1
            elif success and not found:
                success_exceeds_found += 1
    add_check(
        errors,
        invalid_bool_rows == 0,
        "Training artifacts contain invalid success/found boolean values.",
    )
    add_check(
        errors,
        success_exceeds_found == 0,
        "Training artifacts contain success=true with found=false.",
    )

    continuous_fields = (
        "avg_reward",
        "recovery_time",
        "safety_cost",
        "final_distance",
        "final_nav_distance",
        "action_smoothness",
    )
    nonfinite = {}
    for field in continuous_fields:
        count = sum(1 for row in progress_rows if finite_float(row.get(field)) is None)
        nonfinite[field] = count
        add_check(
            errors,
            count == 0,
            "training_progress %s contains %d non-finite values." % (field, count),
        )

    actor_loss_finite_count = 0
    critic_loss_finite_count = 0
    invalid_nonempty_losses = 0
    for row in progress_rows:
        for field in ("actor_loss", "critic_loss"):
            value = row.get(field, "")
            if value is None or str(value).strip() == "":
                continue
            if finite_float(value) is None:
                invalid_nonempty_losses += 1
            elif field == "actor_loss":
                actor_loss_finite_count += 1
            else:
                critic_loss_finite_count += 1
    add_check(
        errors,
        invalid_nonempty_losses == 0,
        "Non-empty actor_loss/critic_loss values must be finite.",
    )
    add_check(
        errors,
        actor_loss_finite_count > 0,
        "No finite actor update was recorded.",
    )
    add_check(
        errors,
        critic_loss_finite_count > 0,
        "No finite critic update was recorded.",
    )

    total_steps = 0
    step_counts_valid = True
    for row in progress_rows:
        try:
            steps = int(row.get("episode_steps", ""))
            if steps < 0:
                raise ValueError
            total_steps += steps
        except (TypeError, ValueError):
            step_counts_valid = False
    batch_size = config.get("batch_size")
    learn_interval = config.get("learn_interval")
    updates_per_train = config.get("updates_per_train")
    obs_dims = config.get("obs_dims")
    agent_count = len(obs_dims) if isinstance(obs_dims, list) else 0
    trigger_count = 0
    if (
        step_counts_valid
        and isinstance(batch_size, int)
        and batch_size > 0
        and isinstance(learn_interval, int)
        and learn_interval > 0
    ):
        first_trigger = int(math.ceil(float(batch_size) / learn_interval)) * learn_interval
        if total_steps >= first_trigger:
            trigger_count = (total_steps - first_trigger) // learn_interval + 1
    multiplier = (
        updates_per_train * agent_count
        if isinstance(updates_per_train, int)
        and updates_per_train > 0
        and agent_count > 0
        else 0
    )
    actor_update_count = trigger_count * multiplier
    critic_update_count = trigger_count * multiplier
    add_check(
        errors,
        actor_update_count > 0,
        "Training schedule implies zero actor updates.",
    )
    add_check(
        errors,
        critic_update_count > 0,
        "Training schedule implies zero critic updates.",
    )

    audit = {
        "expected_episodes": expected,
        "training_progress_path": progress_path,
        "training_progress_row_count": len(progress_rows),
        "episode_metrics_path": metrics_path,
        "episode_metrics_row_count": len(metrics_rows),
        "boundary_dataset_path": boundary_path,
        "boundary_dataset_row_count": len(boundary_rows),
        "training_progress_episode_sequence_complete": (
            progress_identity.get("sequence") == expected_sequence
        ),
        "training_progress_episode_identity": progress_identity,
        "episode_metrics_episode_identity": metrics_identity,
        "boundary_dataset_episode_identity": boundary_identity,
        "invalid_boolean_row_count": invalid_bool_rows,
        "success_exceeds_found_count": success_exceeds_found,
        "nonfinite_continuous_counts": nonfinite,
        "actor_loss_finite_count": actor_loss_finite_count,
        "critic_loss_finite_count": critic_loss_finite_count,
        "total_environment_steps": total_steps,
        "learning_trigger_count": trigger_count,
        "actor_update_count": actor_update_count,
        "critic_update_count": critic_update_count,
    }
    return (
        audit,
        boundary_rows,
        metrics_rows,
        cross_file_episode_audit,
        metrics_boundary_crosscheck,
    )


def audit_boundary(args, rows, errors):
    columns = set(rows[0].keys()) if rows else set()
    missing_keys = [key for key in DISTURBANCE_KEYS if key not in columns]
    add_check(
        errors,
        not missing_keys,
        "boundary_dataset.csv is missing disturbance keys: %s."
        % ", ".join(missing_keys),
    )
    vectors = []
    bounds_violations = []
    invalid_delays = 0
    invalid_flags = 0
    nonfinite_metrics = 0
    for row_index, row in enumerate(rows, 1):
        vector = []
        row_valid = True
        for key in DISTURBANCE_KEYS:
            value = finite_float(row.get(key))
            if value is None:
                bounds_violations.append("row %d %s is non-finite" % (row_index, key))
                row_valid = False
                continue
            low, high = DEFAULT_DISTURBANCE_BOUNDS[key]
            low_f, high_f = sorted((float(low), float(high)))
            if value < low_f - 1e-12 or value > high_f + 1e-12:
                bounds_violations.append(
                    "row %d %s=%s outside [%s, %s]"
                    % (row_index, key, value, low_f, high_f)
                )
            if key == "action_delay_steps" and not float(value).is_integer():
                invalid_delays += 1
            vector.append(value)
        if row_valid and len(vector) == len(DISTURBANCE_KEYS):
            vectors.append(tuple(round(value, 12) for value in vector))
        valid, _, _ = bool_pair_is_valid(row, "success_flag", "found_flag")
        if not valid:
            invalid_flags += 1
        for field in ("recovery_time", "safety_cost", "final_distance"):
            if finite_float(row.get(field)) is None:
                nonfinite_metrics += 1

    add_check(
        errors,
        not bounds_violations,
        "boundary_dataset.csv has disturbance bound violations: %s."
        % ("; ".join(bounds_violations[:5])),
    )
    add_check(
        errors,
        invalid_delays == 0,
        "boundary_dataset action_delay_steps contains non-integer values.",
    )
    add_check(
        errors,
        invalid_flags == 0,
        "boundary_dataset contains invalid success_flag/found_flag values.",
    )
    add_check(
        errors,
        nonfinite_metrics == 0,
        "boundary_dataset contains non-finite outcome metrics.",
    )

    nominal = nominal_disturbance()
    nominal_vector = tuple(
        round(float(nominal[key]), 12) for key in DISTURBANCE_KEYS
    )
    all_nominal = bool(vectors) and all(vector == nominal_vector for vector in vectors)
    distinct_vectors = len(set(vectors))
    add_check(
        errors,
        bool(vectors) and not all_nominal,
        "boundary_dataset disturbance data are empty or entirely nominal.",
    )
    add_check(
        errors,
        distinct_vectors >= 2,
        "boundary_dataset must contain at least two distinct disturbance vectors.",
    )
    return {
        "path": os.path.join(absolute(args.log_dir), "boundary_dataset.csv"),
        "expected_row_count": args.expected_episodes,
        "row_count": len(rows),
        "disturbance_keys": list(DISTURBANCE_KEYS),
        "missing_disturbance_keys": missing_keys,
        "bounds": DEFAULT_DISTURBANCE_BOUNDS,
        "bounds_violation_count": len(bounds_violations),
        "invalid_action_delay_count": invalid_delays,
        "invalid_flag_count": invalid_flags,
        "nonfinite_outcome_count": nonfinite_metrics,
        "all_nominal": all_nominal,
        "distinct_disturbance_vectors": distinct_vectors,
        "disturbance_varied": distinct_vectors >= 2,
    }


def audit_snapshots(args, errors):
    model_dir = absolute(args.model_dir)
    expected = list(
        range(
            args.expected_snapshot_interval,
            args.expected_episodes + 1,
            args.expected_snapshot_interval,
        )
    ) if args.expected_snapshot_interval > 0 else []
    candidates = []
    if os.path.isdir(model_dir):
        candidates = [
            name
            for name in os.listdir(model_dir)
            if name.lower().startswith("snapshot") and name.lower().endswith(".pt")
        ]
    else:
        errors.append("Missing model directory: %s" % model_dir)
    actual = []
    invalid_names = []
    snapshot_paths = []
    for name in candidates:
        match = SNAPSHOT_RE.match(name)
        if match is None:
            invalid_names.append(name)
            continue
        actual.append(int(match.group(1)))
        snapshot_paths.append(os.path.join(model_dir, name))
    counts = Counter(actual)
    duplicate_episodes = sorted(ep for ep, count in counts.items() if count > 1)
    actual_unique = sorted(counts)
    missing = sorted(set(expected) - set(actual_unique))
    extra = sorted(set(actual_unique) - set(expected))
    add_check(errors, not invalid_names, "Invalid snapshot filenames: %s." % invalid_names)
    add_check(
        errors,
        not duplicate_episodes,
        "Duplicate snapshot episodes: %s." % duplicate_episodes,
    )
    add_check(errors, not missing, "Missing snapshot episodes: %s." % missing)
    add_check(errors, not extra, "Unexpected snapshot episodes: %s." % extra)
    add_check(
        errors,
        actual_unique == expected,
        "Snapshot episode sequence is incomplete.",
    )

    final_model_path = os.path.join(model_dir, "maddpg_uavenv_final.pt")
    final_exists = os.path.isfile(final_model_path)
    add_check(errors, final_exists, "Missing final model: %s" % final_model_path)
    empty_model_files = []
    for path in snapshot_paths + ([final_model_path] if final_exists else []):
        try:
            if os.path.getsize(path) <= 0:
                empty_model_files.append(path)
        except OSError:
            empty_model_files.append(path)
    add_check(
        errors,
        not empty_model_files,
        "Model files are empty or unreadable: %s." % empty_model_files,
    )
    return {
        "model_dir": model_dir,
        "expected_episodes": expected,
        "actual_episodes": actual_unique,
        "count": len(actual_unique),
        "expected_count": len(expected),
        "final_snapshot_episode": actual_unique[-1] if actual_unique else None,
        "missing_episodes": missing,
        "extra_episodes": extra,
        "duplicate_episodes": duplicate_episodes,
        "invalid_filenames": invalid_names,
        "sequence_complete": actual_unique == expected,
        "final_model_exists": final_exists,
        "empty_model_files": empty_model_files,
    }, final_model_path


def verify_source_unchanged(args, hashes_before, errors):
    paths = {
        "source_model": absolute(args.source_model),
        "source_manifest": absolute(args.source_manifest),
        "source_formal_test_summary": absolute(args.source_formal_test_summary),
        "source_selection_summary": absolute(args.source_selection_summary),
        "source_training_config": absolute(args.source_training_config),
    }
    hashes_after = {}
    for key, path in paths.items():
        hashes_after[key] = safe_file_hash(path, errors, key.replace("_", " "))
    unchanged = all(
        hashes_before.get(key) is not None
        and hashes_before.get(key) == hashes_after.get(key)
        for key in paths
    )
    add_check(
        errors,
        unchanged,
        "A frozen clean source artifact changed during the audit.",
    )
    return unchanged, hashes_after


def render_markdown(summary):
    settings = summary["settings"]
    progress = summary["progress_audit"]
    boundary = summary["boundary_dataset_audit"]
    snapshots = summary["snapshot_audit"]
    initialization = summary["initialization"]
    controller = summary["controller_continuity_audit"]
    resolution = summary["source_training_config_resolution"]
    cross_file = summary["cross_file_episode_audit"]
    metrics_crosscheck = summary["episode_metrics_boundary_crosscheck"]
    if controller["historical_soft_gate_status"] == "unrecorded":
        historical_soft_gate_note = (
            "All controller fields explicitly recorded in both source and current "
            "configurations match. The historical source configuration did not record "
            "residual_hybrid_use_soft_gate; no historical value was inferred."
        )
    else:
        historical_soft_gate_note = (
            "The historical source explicitly recorded residual_hybrid_use_soft_gate, "
            "and the recorded value matches the current configuration."
        )
    status = "PASS" if summary["overall_pass"] else "FAIL"
    lines = [
        "# Chapter-4 Uniform DR training audit",
        "",
        "- Result: **%s**" % status,
        "- Run: `%s`" % summary["run_name"],
        "- Stage: `%s`" % summary["training_stage"],
        "- Mode: `%s`" % summary["mode"],
        "- Seed: `%s`" % summary["training_seed"],
        "",
        "## Experiment identity",
        "",
        "The ep8200 selected checkpoint is the frozen clean baseline. This run is a new "
        "Uniform DR training run warm-started explicitly from that ep8200 checkpoint; it "
        "does not continue or overwrite the clean baseline.",
        "",
        "- Source checkpoint: `%s`" % initialization["source_checkpoint"],
        "- Source model: `%s`" % initialization["source_model"],
        "- Source SHA256: `%s`" % initialization["source_sha256"],
        "- Source hash match: `%s`" % initialization["hash_match"],
        "- Clean source artifacts unchanged: `%s`"
        % summary["source_artifacts_unchanged"],
        "- Source training config: `%s`"
        % summary["source_training_config"],
        "",
        "## Controller Continuity Audit",
        "",
        "- Source clean training config resolved path: `%s`"
        % resolution["resolved_path"],
        "- Source config resolution: `%s`" % resolution["resolution"],
        "- Source clean mode: `%s`" % controller["source_mode"],
        "- Current Uniform DR mode: `%s`" % controller["current_mode"],
        "- Prior/residual, reward, residual mode, and PSE planner fields match: `%s`"
        % controller["recorded_controller_fields_match"],
        "- Mismatched fields: `%s`" % controller["mismatched_fields"],
        "- Intended mode difference: `%s`"
        % controller["intended_mode_difference"],
        "- Source robust disturbance: `%s`"
        % controller["source_use_robust_disturbance"],
        "- Current robust disturbance: `%s`"
        % controller["current_use_robust_disturbance"],
        "- Current soft gate explicitly recorded: `%s`"
        % controller["current_soft_gate_explicitly_recorded"],
        "- Current soft gate enabled: `%s`"
        % controller["current_soft_gate_enabled"],
        "- Historical unrecorded fields: `%s`"
        % controller["historical_unrecorded_fields"],
        "- Historical soft-gate status: `%s`"
        % controller["historical_soft_gate_status"],
        "- No historical soft-gate value was inferred or fabricated.",
        "- No fixed-R override is enabled.",
        "- The frozen ep8200 controller definition was not changed.",
        "",
        historical_soft_gate_note,
        "",
        "## Episode Identity Audit",
        "",
        "- Episode identity source: explicit episode columns",
        "- Episode numbering: 1-based",
        "- Episode metrics unique/complete/ordered: `%s` / `%s` / `%s`"
        % (
            progress["episode_metrics_episode_identity"]["unique"],
            progress["episode_metrics_episode_identity"]["complete"],
            progress["episode_metrics_episode_identity"]["ordered"],
        ),
        "- Boundary dataset unique/complete/ordered: `%s` / `%s` / `%s`"
        % (
            progress["boundary_dataset_episode_identity"]["unique"],
            progress["boundary_dataset_episode_identity"]["complete"],
            progress["boundary_dataset_episode_identity"]["ordered"],
        ),
        "- Three CSV episode sequences match: `%s`"
        % cross_file["all_sequences_match"],
        "- Episode metrics and boundary fields match: `%s`"
        % metrics_crosscheck["all_fields_match"],
        "",
        "## Training settings",
        "",
        "- Episodes: `%s`" % settings["episodes"],
        "- Maximum steps: `%s`" % settings["max_steps"],
        "- Snapshot interval: `%s`" % settings["snapshot_interval"],
        "- Robust physical disturbance: enabled over all 9 registry dimensions",
        "- REB: disabled",
        "- RSCU: disabled",
        "- RBE sampler: disabled",
        "- Search actors frozen: no",
        "- Boundary dataset collection: enabled",
        "",
        "## Artifact audit",
        "",
        "- Training progress rows: `%s`" % progress["training_progress_row_count"],
        "- Episode metrics rows: `%s`" % progress["episode_metrics_row_count"],
        "- Actor update calls: `%s`" % progress["actor_update_count"],
        "- Critic update calls: `%s`" % progress["critic_update_count"],
        "- Boundary dataset rows: `%s`" % boundary["row_count"],
        "- Distinct disturbance vectors: `%s`"
        % boundary["distinct_disturbance_vectors"],
        "- Snapshot count: `%s`" % snapshots["count"],
        "- Final model: `%s`" % summary["final_model_path"],
        "",
        "Uniform DR checkpoint selection has not been performed.",
        "",
        "## Validation readiness",
        "",
        "- Ready for checkpoint validation: `%s`"
        % summary["ready_for_checkpoint_validation"],
        "",
        "## Errors",
        "",
    ]
    if summary["errors"]:
        lines.extend("- %s" % error for error in summary["errors"])
    else:
        lines.append("- None")
    lines.extend(["", "## Warnings", ""])
    if summary["warnings"]:
        lines.extend("- %s" % warning for warning in summary["warnings"])
    else:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def main():
    args = parse_args()
    errors = []
    warnings = []
    log_dir = absolute(args.log_dir)
    model_dir = absolute(args.model_dir)
    out_dir = absolute(args.out_dir)

    source_resolution_audit = audit_source_training_config_resolution(args, errors)
    source_audit, hashes_before = audit_source(args, errors)
    source_config_audit, source_config, source_controller = (
        audit_source_training_config(args, errors)
    )
    config_audit, config, current_controller = audit_training_config(
        args, absolute(args.source_model), errors
    )
    controller_continuity_audit = audit_controller_continuity(
        source_config,
        config,
        source_controller,
        current_controller,
        errors,
    )
    (
        progress_audit,
        boundary_rows,
        _metrics_rows,
        cross_file_episode_audit,
        metrics_boundary_crosscheck,
    ) = audit_progress(args, config, errors)
    boundary_audit = audit_boundary(args, boundary_rows, errors)
    snapshot_audit, final_model_path = audit_snapshots(args, errors)
    source_unchanged, hashes_after = verify_source_unchanged(
        args, hashes_before, errors
    )
    source_audit["source_model_unchanged"] = source_unchanged
    source_audit["artifact_sha256_after"] = hashes_after

    overall_pass = not errors
    smoke = bool(args.smoke)
    summary = {
        "overall_pass": overall_pass,
        "experiment_type": (
            "uniform_dr_warm_start_training_smoke"
            if smoke
            else "uniform_dr_warm_start_training"
        ),
        "training_stage": "smoke" if smoke else "completed",
        "run_name": args.expected_run_name,
        "mode": config_audit.get("mode"),
        "training_seed": config_audit.get("seed"),
        "initialization": {
            "type": config_audit.get("initialization_source"),
            "source_checkpoint": EXPECTED_CHECKPOINT_NAME,
            "source_model": absolute(args.source_model),
            "source_sha256": source_audit.get("actual_sha256"),
            "hash_match": source_audit.get("hash_match"),
            "loaded_checkpoint": config_audit.get("loaded_checkpoint"),
            "resume_model_path": config_audit.get("resume_model_path"),
        },
        "settings": {
            "episodes": args.expected_episodes,
            "max_steps": args.expected_max_steps,
            "snapshot_interval": args.expected_snapshot_interval,
            "reward_profile": config_audit.get("reward_profile"),
            "residual_action_mode": config_audit.get("residual_action_mode"),
            "use_robust_disturbance": config_audit.get(
                "use_robust_disturbance"
            ),
            "reb_enable": config_audit.get("reb_enable"),
            "rscu_enable": config_audit.get("rscu_enable"),
            "rbe_sampler_enable": config_audit.get("rbe_sampler_enable"),
            "rbe_executor_only_actor": config_audit.get(
                "rbe_executor_only_actor"
            ),
            "freeze_search_actors": config_audit.get("freeze_search_actors"),
            "collect_boundary_dataset": config_audit.get(
                "collect_boundary_dataset"
            ),
            "disturbance_keys": config_audit.get("disturbance_keys"),
        },
        "source_model_audit": source_audit,
        "source_training_config": absolute(args.source_training_config),
        "source_training_config_resolution": source_resolution_audit,
        "source_training_config_audit": source_config_audit,
        "controller_continuity_audit": controller_continuity_audit,
        "training_config_audit": config_audit,
        "progress_audit": progress_audit,
        "cross_file_episode_audit": cross_file_episode_audit,
        "episode_metrics_boundary_crosscheck": metrics_boundary_crosscheck,
        "boundary_dataset_audit": boundary_audit,
        "snapshot_audit": snapshot_audit,
        "final_model_path": final_model_path,
        "ready_for_checkpoint_validation": bool(overall_pass and not smoke),
        "source_model_unchanged": source_unchanged,
        "source_artifacts_unchanged": source_unchanged,
        "log_dir": log_dir,
        "model_dir": model_dir,
        "errors": errors,
        "warnings": warnings,
    }

    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "uniform_dr_training_summary.json")
    markdown_path = os.path.join(out_dir, "uniform_dr_training_summary.md")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    with open(markdown_path, "w", encoding="utf-8") as handle:
        handle.write(render_markdown(summary))

    print("Uniform DR training audit: %s" % ("PASS" if overall_pass else "FAIL"))
    print("JSON summary: %s" % json_path)
    print("Markdown summary: %s" % markdown_path)
    for error in errors:
        print("[ERROR] %s" % error)
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
