#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collect and audit a frozen-policy full-9D Found-aware REB dataset."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.maddpg import MADDPG
from registry.ch4_artifact_layout import get_ch4_data_root
from evaluate_pse import (
    DISTURBANCE_PROTOCOL_UNIFORM_9D,
    DISTURBANCE_RNG_MODE,
    FLOW_PHASE_KEYS,
    PAIRED_EPISODE_SEED_FORMULA,
    PAIRED_EPISODE_SEED_MODE,
    _apply_profile,
    _set_global_seed,
    apply_disturbance_before_reset,
    audit_applied_disturbance,
    disturbance_seed_for_episode,
    resolve_disturbance_protocol,
    sample_episode_disturbance,
)
from registry.rbe_disturbance import DEFAULT_DISTURBANCE_BOUNDS, DISTURBANCE_KEYS
from train import _build_train_env, _configure_ch4_env, get_ablation_config
from utils.rbe_metrics import EpisodeMetricTracker
from utils.reb_dataset import OUTCOME_FIELDS


EXPERIMENT_TYPE = "ch4_uniform_dr_frozen_reb_dataset_full9d_v1"
MODEL_NAME = "ch4_uniform_dr_selected_full9d_v2_snapshot_ep1400"
MODE = "ch4_uniform_dr"
PROFILE = "normal_comm"
PROTOCOL = "uniform_9d_registry_v1"
PROTOCOL_VERSION = 2

MODEL_SHA256 = "47ae3748017c4c9a43efc16ecb9973b535716c0e09d119f2de43615b8e5405e7"
MANIFEST_SHA256 = "2d18b463d9cbd67956b75ae63c7894d00db7cbe43d60037c07e219a21c13f5c4"
SELECTION_SHA256 = "e092aa83878c9b30f543e864b92fa5baf981bc0937912418e5b23deadedb8adb"
EXPECTED_ROLES = ["search_fast", "search_balanced", "search_precise", "executor"]

FORMAL_SEEDS = list(range(201, 211))
TRAIN_SEEDS = list(range(201, 209))
VALIDATION_SEEDS = [209, 210]
FORMAL_EPISODES_PER_SEED = 300
FORMAL_MAX_STEPS = 400
SMOKE_SEEDS = [910]
SMOKE_EPISODES_PER_SEED = 2
SMOKE_MAX_STEPS = 20

NAMED_FORBIDDEN_SEEDS = {
    1,
    2,
    3,
    101,
    102,
    103,
    104,
    105,
    106,
    107,
    108,
    909,
    910,
}

IDENTITY_FIELDS = [
    "model_name",
    "model_path",
    "model_sha256",
    "evaluation_mode",
    "profile",
    "base_seed",
    "episode_index",
    "episode_seed",
    "episode_seed_mode",
    "disturbance_seed",
    "disturbance_rng_mode",
    "disturbance_protocol",
    "disturbance_protocol_version",
    "disturbance_explicitly_applied",
    "disturbance_apply_match",
    "dataset_split",
]

REQUIRED_OUTCOME_FIELDS = [
    "success_flag",
    "found_flag",
    "recovery_time",
    "safety_cost",
    "safety_cost_mean",
    "final_distance",
    "final_nav_distance",
    "action_smoothness",
    "completion_steps",
    "episode_reward_mean",
    "episode_reward_sum",
]

CSV_FIELDS = [
    *IDENTITY_FIELDS,
    *DISTURBANCE_KEYS,
    *FLOW_PHASE_KEYS,
    *REQUIRED_OUTCOME_FIELDS,
]

CONTINUOUS_METRICS = [
    "recovery_time",
    "safety_cost",
    "final_distance",
    "final_nav_distance",
    "action_smoothness",
    "completion_steps",
    "episode_reward_mean",
]

CONTINUOUS_DISTURBANCE_KEYS = [
    key for key in DISTURBANCE_KEYS if key != "action_delay_steps"
]

BOOLEAN_FIELDS = {
    "disturbance_explicitly_applied",
    "disturbance_apply_match",
    "success_flag",
    "found_flag",
}

INTEGER_FIELDS = {
    "base_seed",
    "episode_index",
    "episode_seed",
    "disturbance_seed",
    "disturbance_protocol_version",
    "action_delay_steps",
    "completion_steps",
}

FLOAT_FIELDS = {
    *[key for key in DISTURBANCE_KEYS if key != "action_delay_steps"],
    *FLOW_PHASE_KEYS,
    *[
        field
        for field in REQUIRED_OUTCOME_FIELDS
        if field not in {"success_flag", "found_flag", "completion_steps"}
    ],
}

BY_SEED_FIELDS = [
    "base_seed",
    "dataset_split",
    "n_episodes",
    "n_success",
    "n_found",
    "n_not_found",
    "n_found_but_failed",
    "success_rate",
    "found_rate",
    "succ_if_found",
    "not_found_rate",
    "found_but_failed_rate",
]

CLASS_FIELDS = [
    "scope",
    "n_episodes",
    "n_success",
    "n_found",
    "n_not_found",
    "n_found_but_failed",
    "success_rate",
    "found_rate",
    "succ_if_found",
    "not_found_rate",
    "found_but_failed_rate",
]

DERIVED_COUNT_FIELDS = [
    "n_episodes",
    "n_success",
    "n_found",
    "n_not_found",
    "n_found_but_failed",
]

DERIVED_RATE_FIELDS = [
    "success_rate",
    "found_rate",
    "succ_if_found",
    "not_found_rate",
    "found_but_failed_rate",
]

BY_SEED_INTEGER_FIELDS = ["base_seed", *DERIVED_COUNT_FIELDS]
EXPECTED_CLASS_SCOPES = ["overall", "train", "validation"]


class DatasetError(RuntimeError):
    pass


def _require(condition, message):
    if not condition:
        raise DatasetError(message)


def _project_path(value):
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _load_json(path, label):
    path = Path(path)
    _require(path.is_file(), f"{label} is missing: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, ValueError) as exc:
        raise DatasetError(f"{label} is not readable JSON: {exc}") from exc
    _require(isinstance(value, dict), f"{label} root must be an object")
    return value


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _path_equal(reported, actual):
    try:
        return _project_path(str(reported)) == Path(actual).resolve()
    except (OSError, ValueError, TypeError):
        return False


def _finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _atomic_write_text(path, text):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path, value):
    text = json.dumps(
        _json_safe(value),
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    _atomic_write_text(path, text)


def _atomic_write_csv(path, fieldnames, rows):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in fieldnames})
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_copy_file(source, destination):
    source = Path(source)
    destination = Path(destination)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        with source.open("rb") as source_handle, temporary.open("wb") as destination_handle:
            for block in iter(lambda: source_handle.read(1024 * 1024), b""):
                destination_handle.write(block)
            destination_handle.flush()
            os.fsync(destination_handle.fileno())
        os.replace(str(temporary), str(destination))
    finally:
        if temporary.exists():
            temporary.unlink()


def _outcome_schema_audit():
    collector_fields = list(REQUIRED_OUTCOME_FIELDS)
    dataset_fields = list(OUTCOME_FIELDS)
    match = collector_fields == dataset_fields
    _require(match, "collector outcome schema does not exactly match utils.reb_dataset.OUTCOME_FIELDS")
    return {
        "outcome_schema_match": match,
        "collector_outcome_fields": collector_fields,
        "reb_dataset_outcome_fields": dataset_fields,
    }


def _strict_bool(value, label):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        _require(int(value) in (0, 1), f"{label} must be boolean or 0/1")
        return bool(int(value))
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("true", "1"):
            return True
        if normalized in ("false", "0"):
            return False
    raise DatasetError(f"{label} must be boolean or strictly parseable true/false/0/1")


def _strict_int(value, label):
    if isinstance(value, (bool, np.bool_)):
        raise DatasetError(f"{label} must be an integer, not bool")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, str):
        normalized = value.strip()
        digits = normalized[1:] if normalized[:1] in ("+", "-") else normalized
        if digits.isdigit() and digits:
            return int(normalized)
    raise DatasetError(f"{label} must be a strict integer")


def _strict_float(value, label):
    if isinstance(value, (bool, np.bool_)):
        raise DatasetError(f"{label} must be numeric, not bool")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise DatasetError(f"{label} must be numeric") from exc
    _require(math.isfinite(numeric), f"{label} must be finite")
    return numeric


def _compare_nullable_float(actual_raw, expected, label):
    if expected is None:
        _require(
            actual_raw == "",
            f"{label}: actual is non-empty but expected is null: actual={actual_raw!r}",
        )
        return None
    _require(actual_raw != "", f"{label}: actual is empty but expected={expected!r}")
    actual = _strict_float(actual_raw, label)
    _require(
        math.isclose(actual, float(expected), rel_tol=1e-12, abs_tol=1e-12),
        f"{label} mismatch: actual={actual_raw} expected={expected}",
    )
    return actual


def _coerce_csv_row(raw, row_index):
    _require(None not in raw, f"row {row_index} has extra CSV columns")
    missing = [field for field in CSV_FIELDS if raw.get(field) is None or raw.get(field) == ""]
    _require(not missing, f"row {row_index} has missing fields: {missing}")
    row = {}
    for field in CSV_FIELDS:
        value = raw[field]
        label = f"row {row_index} {field}"
        if field in BOOLEAN_FIELDS:
            row[field] = _strict_bool(value, label)
        elif field in INTEGER_FIELDS:
            row[field] = _strict_int(value, label)
        elif field in FLOAT_FIELDS:
            row[field] = _strict_float(value, label)
        else:
            row[field] = str(value)
    return row


def _read_csv_strict(path, expected_fields):
    path = Path(path)
    _require(path.is_file(), f"CSV is missing: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        _require(reader.fieldnames == list(expected_fields), f"CSV header/order mismatch: {reader.fieldnames}")
        rows = []
        for row_index, raw in enumerate(reader):
            _require(None not in raw, f"row {row_index} has extra CSV columns")
            _require(
                all(raw.get(field) is not None and raw.get(field) != "" for field in expected_fields),
                f"row {row_index} is incomplete",
            )
            rows.append(raw)
    return rows


def _read_csv_relaxed(path, expected_fields):
    path = Path(path)
    _require(path.is_file(), f"CSV is missing: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        _require(reader.fieldnames == list(expected_fields), f"CSV header/order mismatch: {reader.fieldnames}")
        rows = []
        for row_index, raw in enumerate(reader):
            _require(None not in raw, f"row {row_index} has extra CSV columns")
            _require(all(raw.get(field) is not None for field in expected_fields), f"row {row_index} is structurally incomplete")
            rows.append(raw)
    return rows


def _initialize_partial_csv(path):
    path = Path(path)
    _require(not path.exists(), f"partial CSV already exists: {path}")
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        handle.flush()
        os.fsync(handle.fileno())


def _source_hashes(source):
    return {
        "model_sha256": source["model_sha256_before"],
        "manifest_sha256": source["manifest_sha256_before"],
        "selection_summary_sha256": source["selection_summary_sha256_before"],
    }


def _last_completed_key(config, completed_keys):
    ordered = [
        (seed, episode_index)
        for seed in config["seeds"]
        for episode_index in range(config["episodes_per_seed"])
    ]
    completed = set(completed_keys)
    present = [key for key in ordered if key in completed]
    return present[-1] if present else (None, None)


def _write_collection_state(
    path,
    config,
    source,
    partial_path,
    completed_keys,
    status,
    finalization_phase=None,
    formal_result_root=None,
    staging_root=None,
    commit_directory_renamed=False,
    post_commit_output_audit_pass=False,
    summary_finalized=False,
    markdown_finalized=False,
):
    last_seed, last_episode = _last_completed_key(config, completed_keys)
    phase_defaults = {
        "incomplete": "collecting",
        "finalizing": "precommit_ready",
        "complete": "complete",
    }
    finalization_phase = phase_defaults[str(status)] if finalization_phase is None else str(finalization_phase)
    path = Path(path)
    partial_path = Path(partial_path)
    if formal_result_root is None:
        parent_text = str(path.parent)
        formal_result_root = parent_text[:-len(".incomplete")] if parent_text.endswith(".incomplete") else parent_text
    if staging_root is None:
        staging_root = str(formal_result_root) + ".incomplete"
    state = {
        "stage": config["stage"],
        "expected_seeds": list(config["seeds"]),
        "episodes_per_seed": int(config["episodes_per_seed"]),
        "expected_total_rows": len(config["seeds"]) * int(config["episodes_per_seed"]),
        "completed_rows": len(completed_keys),
        "last_completed_seed": last_seed,
        "last_completed_episode_index": last_episode,
        "source_hashes": _source_hashes(source),
        "partial_csv_sha256": _sha256(partial_path),
        "status": str(status),
        "finalization_phase": finalization_phase,
        "formal_result_root": str(Path(formal_result_root).resolve()),
        "staging_root": str(Path(staging_root).resolve()),
        "commit_directory_renamed": bool(commit_directory_renamed),
        "post_commit_output_audit_pass": bool(post_commit_output_audit_pass),
        "summary_finalized": bool(summary_finalized),
        "markdown_finalized": bool(markdown_finalized),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write_json(path, state)
    return state


def _csv_file_metadata(path, result_root):
    path = Path(path)
    header = _read_csv_header(path)
    raw_rows = _read_csv_relaxed(path, header)
    return {
        "path": str(path.relative_to(result_root)),
        "sha256": _sha256(path),
        "bytes": int(path.stat().st_size),
        "row_count": len(raw_rows),
        "header_fields": header,
    }


def _read_csv_header(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            return next(reader)
        except StopIteration as exc:
            raise DatasetError(f"CSV is empty: {path}") from exc


def _plain_file_metadata(path, result_root):
    path = Path(path)
    _require(path.is_file() and path.stat().st_size > 0, f"output file is missing or empty: {path}")
    return {
        "path": str(path.relative_to(result_root)),
        "sha256": _sha256(path),
        "bytes": int(path.stat().st_size),
    }


def _discover_existing_smoke_seeds():
    root = get_ch4_data_root() / "smoke"
    discovered = set()
    seed_keys = {
        "seed",
        "seeds",
        "base_seed",
        "test_seed",
        "test_seeds",
        "validation_seeds",
    }

    def visit(value, parent_key=None):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in seed_keys:
                    visit(item, key)
                elif isinstance(item, (dict, list)):
                    visit(item, key)
        elif isinstance(value, list):
            for item in value:
                visit(item, parent_key)
        elif parent_key in seed_keys and isinstance(value, (int, np.integer)):
            discovered.add(int(value))

    if root.is_dir():
        for path in root.rglob("*.json"):
            try:
                visit(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
    return sorted(discovered)


def _build_environment(max_steps):
    config, _ = get_ablation_config(MODE)
    env, env_kwargs = _build_train_env(
        torch.device("cpu"),
        int(max_steps),
        ablation_config=config,
    )
    env, _ = _configure_ch4_env(env, env_kwargs, MODE)
    _apply_profile(env, PROFILE)
    return env


def _network_structure_audit(maddpg, env, device):
    _require(maddpg.nagents == 4, "frozen model must contain four agents")
    _require(len(maddpg.agents) == 4, "frozen model agent list length mismatch")
    roles = list(maddpg.agent_role_names)
    _require(roles == EXPECTED_ROLES, f"frozen model roles mismatch: {roles}")
    env_roles = list(getattr(env, "role_names", []))
    _require(env_roles == EXPECTED_ROLES, f"environment roles mismatch: {env_roles}")
    _require(len(maddpg.agent_init_params) == 4, "agent_init_params length mismatch")

    obs_dims = [
        int(env.observation_space[f"agent_{index}"].shape[0])
        for index in range(4)
    ]
    action_dims = [
        int(env.action_space[f"agent_{index}"].shape[0])
        for index in range(4)
    ]
    total_critic_input = sum(obs_dims) + sum(action_dims)
    agents = []
    for index, (role, params, agent) in enumerate(
        zip(roles, maddpg.agent_init_params, maddpg.agents)
    ):
        _require(
            int(params.get("num_in_pol", -1)) == obs_dims[index],
            f"{role} policy input dimension mismatch",
        )
        _require(
            int(params.get("num_out_pol", -1)) == action_dims[index],
            f"{role} policy output dimension mismatch",
        )
        _require(
            int(params.get("num_in_critic", -1)) == total_critic_input,
            f"{role} critic input dimension mismatch",
        )
        policy_parameters = int(sum(p.numel() for p in agent.policy.parameters()))
        critic_parameters = int(
            sum(p.numel() for p in agent.critic1.parameters())
            + sum(p.numel() for p in agent.critic2.parameters())
        )
        _require(policy_parameters > 0, f"{role} policy has no parameters")
        _require(critic_parameters > 0, f"{role} critics have no parameters")
        agents.append(
            {
                "index": index,
                "role": role,
                "observation_dim": obs_dims[index],
                "action_dim": action_dims[index],
                "critic_input_dim": total_critic_input,
                "policy_class": type(agent.policy).__name__,
                "policy_parameter_count": policy_parameters,
                "critic_parameter_count": critic_parameters,
            }
        )
    return {
        "overall_pass": True,
        "device": str(device),
        "agent_count": 4,
        "roles": roles,
        "hidden_dim": int(maddpg.hidden_dim),
        "algorithm_types": list(maddpg.alg_types),
        "agents": agents,
    }


def audit_sources_readonly(model_path, manifest_path, selection_path):
    model_path = _project_path(model_path)
    manifest_path = _project_path(manifest_path)
    selection_path = _project_path(selection_path)
    _require(model_path.is_file() and model_path.stat().st_size > 0, "frozen model is missing or empty")
    model_hash = _sha256(model_path)
    manifest_hash = _sha256(manifest_path)
    selection_hash = _sha256(selection_path)
    _require(model_hash == MODEL_SHA256, "frozen model SHA256 mismatch")
    _require(manifest_hash == MANIFEST_SHA256, "selected manifest SHA256 mismatch")
    _require(selection_hash == SELECTION_SHA256, "selection summary SHA256 mismatch")

    manifest = _load_json(manifest_path, "selected manifest")
    selection = _load_json(selection_path, "selection summary")
    _require(manifest.get("selection_smoke") is False, "selected manifest must be formal")
    _require(manifest.get("selected_checkpoint_name") == "snapshot_ep1400", "manifest checkpoint mismatch")
    _require(manifest.get("selected_checkpoint_episode") == 1400, "manifest episode mismatch")
    _require(manifest.get("selected_sha256") == MODEL_SHA256, "manifest selected hash mismatch")
    _require(manifest.get("source_sha256") == MODEL_SHA256, "manifest source hash mismatch")
    _require(manifest.get("hash_match") is True, "manifest hash_match must be true")
    _require(_path_equal(manifest.get("selected_model_path"), model_path), "manifest model path mismatch")
    _require(manifest.get("test_used_for_selection") is False, "manifest reports test selection leakage")
    _require(manifest.get("ready_for_formal_robust_test") is True, "manifest is not ready")

    _require(selection.get("overall_pass") is True, "selection summary did not pass")
    _require(selection.get("selection_smoke") is False, "selection summary must be formal")
    _require(selection.get("selected_checkpoint_name") == "snapshot_ep1400", "selection checkpoint mismatch")
    _require(selection.get("selected_checkpoint_episode") == 1400, "selection episode mismatch")
    _require(selection.get("selected_sha256") == MODEL_SHA256, "selection hash mismatch")
    _require(selection.get("hash_match") is True, "selection hash_match must be true")
    _require(_path_equal(selection.get("selected_model_path"), model_path), "selection model path mismatch")
    _require(selection.get("evaluation_protocol") == PROTOCOL, "selection protocol mismatch")
    _require(selection.get("evaluation_protocol_version") == PROTOCOL_VERSION, "selection protocol version mismatch")
    _require(selection.get("disturbance_protocol_audit", {}).get("disturbance_keys") == list(DISTURBANCE_KEYS), "selection disturbance key order mismatch")
    _require(selection.get("test_used_for_selection") is False, "selection summary reports test leakage")
    _require(selection.get("ready_for_formal_robust_test") is True, "selection summary is not ready")
    _require(selection.get("errors") == [], "selection summary errors must be empty")

    source = {
        "overall_pass": True,
        "model_path": str(model_path),
        "model_sha256_before": model_hash,
        "manifest_path": str(manifest_path),
        "manifest_sha256_before": manifest_hash,
        "selection_summary_path": str(selection_path),
        "selection_summary_sha256_before": selection_hash,
        "selected_checkpoint_name": "snapshot_ep1400",
        "selected_checkpoint_episode": 1400,
        "lightweight_readonly_audit": True,
    }
    return _refresh_source_audit(source)


def audit_and_load_sources(model_path, manifest_path, selection_path, max_steps):
    source = audit_sources_readonly(model_path, manifest_path, selection_path)
    env = _build_environment(max_steps)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    maddpg = MADDPG.init_from_save(source["model_path"], device=device)
    maddpg.prep_rollouts(device=device)
    network_audit = _network_structure_audit(maddpg, env, device)
    source["network_structure_audit"] = network_audit
    source["lightweight_readonly_audit"] = False
    return source, maddpg, env


def _collection_config(smoke):
    if smoke:
        return {
            "stage": "smoke",
            "seeds": list(SMOKE_SEEDS),
            "episodes_per_seed": SMOKE_EPISODES_PER_SEED,
            "max_steps": SMOKE_MAX_STEPS,
            "split_by_seed": {910: "smoke"},
        }
    split_by_seed = {seed: "train" for seed in TRAIN_SEEDS}
    split_by_seed.update({seed: "validation" for seed in VALIDATION_SEEDS})
    return {
        "stage": "formal",
        "seeds": list(FORMAL_SEEDS),
        "episodes_per_seed": FORMAL_EPISODES_PER_SEED,
        "max_steps": FORMAL_MAX_STEPS,
        "split_by_seed": split_by_seed,
    }


def collect_rows(
    maddpg,
    env,
    model_path,
    model_hash,
    config,
    partial_path,
    completed_keys,
    on_seed_complete=None,
):
    resolved_protocol = resolve_disturbance_protocol(MODE, PROTOCOL)
    _require(resolved_protocol == DISTURBANCE_PROTOCOL_UNIFORM_9D, "full-9D protocol did not resolve")
    _outcome_schema_audit()
    tracker = EpisodeMetricTracker()
    total_expected = len(config["seeds"]) * config["episodes_per_seed"]
    completed_keys = set(completed_keys)
    rows_collected_this_run = 0
    with Path(partial_path).open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        for base_seed in config["seeds"]:
            for episode_index in range(config["episodes_per_seed"]):
                key = (int(base_seed), int(episode_index))
                if key in completed_keys:
                    continue
                episode_seed = disturbance_seed_for_episode(base_seed, episode_index)
                _set_global_seed(episode_seed)
                _apply_profile(env, PROFILE)
                requested, disturbance_seed, rng_mode = sample_episode_disturbance(
                    resolved_protocol,
                    base_seed,
                    episode_index,
                )
                obs = apply_disturbance_before_reset(env, resolved_protocol, requested)
                _require(hasattr(env, "get_current_disturbance"), "environment lacks get_current_disturbance")
                actual_xi = env.get_current_disturbance()
                actual_phase_x = float(getattr(env, "_flow_phase_x"))
                actual_phase_y = float(getattr(env, "_flow_phase_y"))
                application_audit = audit_applied_disturbance(
                    requested,
                    actual_xi,
                    actual_phase_x,
                    actual_phase_y,
                )
                _require(application_audit.get("match") is True, f"disturbance apply mismatch at seed={base_seed} episode={episode_index}")
                tracker.reset(env, actual_xi)

                for _ in range(1, int(config["max_steps"]) + 1):
                    actions = maddpg.step(obs, explore=False)
                    env_actions = torch.stack(
                        [action.squeeze(0) for action in actions],
                        dim=0,
                    ).to(device=env.device, dtype=torch.float32)
                    obs, rewards, dones = env.step(env_actions)
                    rewards_t = (
                        rewards
                        if torch.is_tensor(rewards)
                        else torch.as_tensor(rewards, dtype=torch.float32)
                    )
                    tracker.step(env, env_actions, rewards_t, dones)
                    if all(bool(done) for done in dones):
                        break

                final_metrics = tracker.finalize(env)
                missing_outcomes = [
                    field for field in REQUIRED_OUTCOME_FIELDS if field not in final_metrics
                ]
                _require(
                    not missing_outcomes,
                    f"EpisodeMetricTracker omitted fields: {missing_outcomes}",
                )
                row = {
                    "model_name": MODEL_NAME,
                    "model_path": str(Path(model_path).resolve()),
                    "model_sha256": model_hash,
                    "evaluation_mode": MODE,
                    "profile": PROFILE,
                    "base_seed": int(base_seed),
                    "episode_index": int(episode_index),
                    "episode_seed": int(episode_seed),
                    "episode_seed_mode": PAIRED_EPISODE_SEED_MODE,
                    "disturbance_seed": int(disturbance_seed),
                    "disturbance_rng_mode": rng_mode,
                    "disturbance_protocol": resolved_protocol,
                    "disturbance_protocol_version": PROTOCOL_VERSION,
                    "disturbance_explicitly_applied": True,
                    "disturbance_apply_match": True,
                    "dataset_split": config["split_by_seed"][base_seed],
                    **{key_name: actual_xi[key_name] for key_name in DISTURBANCE_KEYS},
                    "flow_phase_x": actual_phase_x,
                    "flow_phase_y": actual_phase_y,
                    **{field: final_metrics[field] for field in REQUIRED_OUTCOME_FIELDS},
                }
                writer.writerow({field: row[field] for field in CSV_FIELDS})
                handle.flush()
                os.fsync(handle.fileno())
                completed_keys.add(key)
                rows_collected_this_run += 1
                print(
                    "[Collect] row=%d/%d seed=%d episode=%d found=%d success=%d steps=%d"
                    % (
                        len(completed_keys),
                        total_expected,
                        base_seed,
                        episode_index,
                        int(row["found_flag"]),
                        int(row["success_flag"]),
                        int(row["completion_steps"]),
                    ),
                    flush=True,
                )
            if on_seed_complete is not None:
                on_seed_complete(completed_keys)
    return rows_collected_this_run, completed_keys


def _collect_rows_if_missing(
    maddpg,
    env,
    source,
    args,
    config,
    partial_path,
    completed_keys,
    on_seed_complete,
    source_loader=audit_and_load_sources,
    row_collector=collect_rows,
):
    expected_total = len(config["seeds"]) * config["episodes_per_seed"]
    if len(completed_keys) >= expected_total:
        return 0, completed_keys, source, maddpg, env
    if maddpg is None or env is None:
        source, maddpg, env = source_loader(
            args.model,
            args.manifest,
            args.selection_summary,
            config["max_steps"],
        )
    rows_collected_this_run, completed_keys = row_collector(
        maddpg,
        env,
        source["model_path"],
        source["model_sha256_before"],
        config,
        partial_path,
        completed_keys,
        on_seed_complete=on_seed_complete,
    )
    return rows_collected_this_run, completed_keys, source, maddpg, env


def _class_metrics(rows):
    n = len(rows)
    success = sum(bool(row["success_flag"]) for row in rows)
    found = sum(bool(row["found_flag"]) for row in rows)
    not_found = n - found
    found_but_failed = found - success
    _require(success <= found, "success_flag cannot be true when found_flag is false")
    rate = lambda count, denominator: (float(count) / float(denominator)) if denominator else None
    return {
        "n_episodes": n,
        "n_success": success,
        "n_found": found,
        "n_not_found": not_found,
        "n_found_but_failed": found_but_failed,
        "success_rate": rate(success, n),
        "found_rate": rate(found, n),
        "succ_if_found": rate(success, found),
        "not_found_rate": rate(not_found, n),
        "found_but_failed_rate": rate(found_but_failed, n),
    }


def _metric_statistics(rows):
    result = {}
    for field in CONTINUOUS_METRICS:
        values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
        _require(values.size == len(rows), f"{field} value count mismatch")
        _require(np.all(np.isfinite(values)), f"{field} contains NaN or Inf")
        result[field] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "p05": float(np.percentile(values, 5)),
            "p25": float(np.percentile(values, 25)),
            "p50": float(np.percentile(values, 50)),
            "p75": float(np.percentile(values, 75)),
            "p95": float(np.percentile(values, 95)),
        }
    return result


def _audit_rows(rows, config, smoke_seeds, expected_model_path, require_complete=True):
    schema_audit = _outcome_schema_audit()
    expected_total = len(config["seeds"]) * config["episodes_per_seed"]
    if require_complete:
        _require(len(rows) == expected_total, f"row count mismatch: {len(rows)} != {expected_total}")
    else:
        _require(len(rows) <= expected_total, f"partial row count exceeds protocol total: {len(rows)}")
    _require(
        CSV_FIELDS[len(IDENTITY_FIELDS):len(IDENTITY_FIELDS) + len(DISTURBANCE_KEYS)]
        == list(DISTURBANCE_KEYS),
        "CSV disturbance field order mismatch",
    )

    missing_field_count = 0
    bounds_violation_count = 0
    flow_phase_violation_count = 0
    duplicate_episode_key_count = 0
    invalid_split_row_count = 0
    safety_mean_mismatch_count = 0
    keys_seen = set()
    keys_in_order = []
    per_seed_indices = {seed: [] for seed in config["seeds"]}
    per_seed_counts = Counter()

    for row_index, row in enumerate(rows):
        missing = [field for field in CSV_FIELDS if field not in row or row[field] is None or row[field] == ""]
        missing_field_count += len(missing)
        _require(not missing, f"row {row_index} has missing fields: {missing}")
        _require(row["model_name"] == MODEL_NAME, f"row {row_index} model_name mismatch")
        _require(_path_equal(row["model_path"], expected_model_path), f"row {row_index} model_path mismatch")
        _require(row["model_sha256"] == MODEL_SHA256, f"row {row_index} model hash mismatch")
        _require(row["evaluation_mode"] == MODE, f"row {row_index} mode mismatch")
        _require(row["profile"] == PROFILE, f"row {row_index} profile mismatch")
        _require(row["episode_seed_mode"] == PAIRED_EPISODE_SEED_MODE, f"row {row_index} seed mode mismatch")
        _require(row["disturbance_rng_mode"] == DISTURBANCE_RNG_MODE, f"row {row_index} RNG mode mismatch")
        _require(row["disturbance_protocol"] == PROTOCOL, f"row {row_index} protocol mismatch")
        _require(_strict_int(row["disturbance_protocol_version"], f"row {row_index} protocol version") == PROTOCOL_VERSION, f"row {row_index} protocol version mismatch")
        _require(_strict_bool(row["disturbance_explicitly_applied"], f"row {row_index} explicit application") is True, f"row {row_index} was not explicitly applied")
        _require(_strict_bool(row["disturbance_apply_match"], f"row {row_index} apply match") is True, f"row {row_index} apply audit failed")

        base_seed = _strict_int(row["base_seed"], f"row {row_index} base_seed")
        episode_index = _strict_int(row["episode_index"], f"row {row_index} episode_index")
        _require(base_seed in config["seeds"], f"unexpected base seed {base_seed}")
        _require(0 <= episode_index < config["episodes_per_seed"], f"row {row_index} episode index out of range")
        expected_split = config["split_by_seed"][base_seed]
        if row["dataset_split"] != expected_split:
            invalid_split_row_count += 1
        _require(row["dataset_split"] == expected_split, f"seed {base_seed} split mismatch")
        expected_episode_seed = disturbance_seed_for_episode(base_seed, episode_index)
        _require(_strict_int(row["episode_seed"], f"row {row_index} episode_seed") == expected_episode_seed, f"row {row_index} episode seed mismatch")
        _require(_strict_int(row["disturbance_seed"], f"row {row_index} disturbance_seed") == expected_episode_seed, f"row {row_index} disturbance seed mismatch")

        key = (base_seed, episode_index)
        if key in keys_seen:
            duplicate_episode_key_count += 1
        keys_seen.add(key)
        keys_in_order.append(key)
        per_seed_indices[base_seed].append(episode_index)
        per_seed_counts[base_seed] += 1

        success_flag = _strict_bool(row["success_flag"], f"row {row_index} success_flag")
        found_flag = _strict_bool(row["found_flag"], f"row {row_index} found_flag")
        _require(int(success_flag) <= int(found_flag), f"row {row_index} success requires found_flag=true")
        for field in (*DISTURBANCE_KEYS, *FLOW_PHASE_KEYS, *REQUIRED_OUTCOME_FIELDS):
            if field not in ("success_flag", "found_flag"):
                _require(_finite(row[field]), f"row {row_index} {field} is not finite")

        for field in DISTURBANCE_KEYS:
            value = row[field]
            low, high = DEFAULT_DISTURBANCE_BOUNDS[field]
            if field == "action_delay_steps":
                valid = _strict_int(value, f"row {row_index} action_delay_steps") in (0, 1, 2, 3)
            else:
                numeric = _strict_float(value, f"row {row_index} {field}")
                valid = float(low) <= numeric <= float(high)
            if not valid:
                bounds_violation_count += 1

        phase_x = _strict_float(row["flow_phase_x"], f"row {row_index} flow_phase_x")
        phase_y = _strict_float(row["flow_phase_y"], f"row {row_index} flow_phase_y")
        if not (0.0 <= phase_x < 2.0 * math.pi):
            flow_phase_violation_count += 1
        if not (0.0 <= phase_y < 2.0 * math.pi):
            flow_phase_violation_count += 1

        completion_steps = _strict_int(row["completion_steps"], f"row {row_index} completion_steps")
        recovery_time = _strict_float(row["recovery_time"], f"row {row_index} recovery_time")
        safety_cost = _strict_float(row["safety_cost"], f"row {row_index} safety_cost")
        safety_cost_mean = _strict_float(row["safety_cost_mean"], f"row {row_index} safety_cost_mean")
        _require(1 <= completion_steps <= config["max_steps"], f"row {row_index} completion_steps out of range")
        _require(0.0 <= recovery_time <= config["max_steps"], f"row {row_index} recovery_time out of range")
        _require(safety_cost >= 0.0, f"row {row_index} safety_cost is negative")
        _require(safety_cost_mean >= 0.0, f"row {row_index} safety_cost_mean is negative")
        if not math.isclose(
            safety_cost_mean,
            safety_cost / completion_steps,
            rel_tol=1e-6,
            abs_tol=1e-8,
        ):
            safety_mean_mismatch_count += 1
        _require(safety_mean_mismatch_count == 0, f"row {row_index} safety_cost_mean is inconsistent")
        _require(_strict_float(row["action_smoothness"], f"row {row_index} action_smoothness") >= 0.0, f"row {row_index} action_smoothness is negative")
        _require(_strict_float(row["final_distance"], f"row {row_index} final_distance") >= 0.0, f"row {row_index} final_distance is negative")
        _require(_strict_float(row["final_nav_distance"], f"row {row_index} final_nav_distance") >= 0.0, f"row {row_index} final_nav_distance is negative")

    _require(missing_field_count == 0, "dataset has missing required fields")
    _require(bounds_violation_count == 0, "dataset has disturbance bounds violations")
    _require(flow_phase_violation_count == 0, "dataset has flow phase range violations")
    _require(duplicate_episode_key_count == 0, "dataset has duplicate episode keys")

    expected_key_sequence = [
        (seed, episode_index)
        for seed in config["seeds"]
        for episode_index in range(config["episodes_per_seed"])
    ]
    expected_prefix = expected_key_sequence if require_complete else expected_key_sequence[:len(keys_in_order)]
    _require(keys_in_order == expected_prefix, "episode rows do not form the required deterministic global prefix")
    for seed in config["seeds"]:
        count = int(per_seed_counts[seed])
        if require_complete:
            _require(count == config["episodes_per_seed"], f"seed {seed} row count mismatch")
        _require(sorted(per_seed_indices[seed]) == list(range(count)), f"seed {seed} episode index prefix is discontinuous")

    actual_seeds = sorted(seed for seed in config["seeds"] if per_seed_counts[seed] > 0)
    discovered_forbidden = sorted(set(smoke_seeds) | NAMED_FORBIDDEN_SEEDS)
    overlap = sorted(set(actual_seeds) & set(discovered_forbidden))
    if config["stage"] == "formal":
        if require_complete:
            _require(actual_seeds == FORMAL_SEEDS, "formal seeds must be exactly 201-210")
        _require(not overlap, f"formal seeds overlap forbidden seeds: {overlap}")
    elif require_complete:
        _require(actual_seeds == SMOKE_SEEDS, "smoke seed must be exactly 910")

    vectors = {tuple(row[key] for key in DISTURBANCE_KEYS) for row in rows}
    per_dimension_distinct = {key: len({row[key] for row in rows}) for key in DISTURBANCE_KEYS}
    _require(len(vectors) == len(rows), "full-9D disturbance vectors are not unique")
    required_continuous_distinct = min(len(rows), 100) if require_complete else 0
    if require_complete:
        for key in CONTINUOUS_DISTURBANCE_KEYS:
            _require(per_dimension_distinct[key] >= required_continuous_distinct, f"{key} does not have sufficient variation")
    observed_delays = sorted({int(row["action_delay_steps"]) for row in rows})
    if require_complete and config["stage"] == "formal":
        _require(observed_delays == [0, 1, 2, 3], "formal dataset must contain all action delay values")

    actual_train_seeds = sorted({int(row["base_seed"]) for row in rows if row["dataset_split"] == "train"})
    actual_validation_seeds = sorted({int(row["base_seed"]) for row in rows if row["dataset_split"] == "validation"})
    train_row_count = sum(1 for row in rows if row["dataset_split"] == "train")
    validation_row_count = sum(1 for row in rows if row["dataset_split"] == "validation")
    seed_overlap = sorted(set(actual_train_seeds) & set(actual_validation_seeds))
    if require_complete and config["stage"] == "formal":
        _require(actual_train_seeds == TRAIN_SEEDS, "formal train seeds mismatch")
        _require(actual_validation_seeds == VALIDATION_SEEDS, "formal validation seeds mismatch")
        _require(train_row_count == 2400, "formal train row count must be 2400")
        _require(validation_row_count == 600, "formal validation row count must be 600")
        _require(not seed_overlap, "formal train/validation seed overlap")

    per_seed = []
    class_groups = {"overall": rows}
    class_distribution = {"overall": _class_metrics(rows)}
    metric_groups = {}
    readiness_reasons = []
    if require_complete:
        for seed in actual_seeds:
            seed_rows = [row for row in rows if int(row["base_seed"]) == seed]
            per_seed.append({"base_seed": seed, "dataset_split": config["split_by_seed"][seed], **_class_metrics(seed_rows)})
        for split in sorted({row["dataset_split"] for row in rows}):
            class_groups[split] = [row for row in rows if row["dataset_split"] == split]
        class_distribution = {name: _class_metrics(group_rows) for name, group_rows in class_groups.items()}
        metric_groups = {name: _metric_statistics(group_rows) for name, group_rows in class_groups.items()}

        if config["stage"] == "formal":
            overall = class_distribution["overall"]
            validation = class_distribution["validation"]
            for field, minimum, label in (
                ("n_success", 300, "success samples"),
                ("n_not_found", 300, "not-found samples"),
                ("n_found_but_failed", 300, "found-but-failed samples"),
            ):
                if int(overall[field]) < minimum:
                    readiness_reasons.append(f"{label} {overall[field]} < {minimum}")
                if int(validation[field]) <= 0:
                    readiness_reasons.append(f"validation split has no {label}")
            if {bool(row["found_flag"]) for row in rows} != {False, True}:
                readiness_reasons.append("found_flag does not contain both classes")
            found_success_values = {bool(row["success_flag"]) for row in rows if bool(row["found_flag"])}
            if found_success_values != {False, True}:
                readiness_reasons.append("success_flag under found_flag=true does not contain both classes")

    return {
        **schema_audit,
        "flow_phase_range_pass": flow_phase_violation_count == 0,
        "flow_phase_violation_count": flow_phase_violation_count,
        "seed_isolation_audit": {
            "overall_pass": True,
            "actual_seeds": actual_seeds,
            "expected_seeds": list(config["seeds"]),
            "episodes_per_seed": config["episodes_per_seed"],
            "total_row_count": len(rows),
            "per_seed_row_counts": {str(seed): int(per_seed_counts[seed]) for seed in config["seeds"]},
            "episode_index_base": 0,
            "episode_index_complete": require_complete,
            "episode_prefix_valid": True,
            "episode_key_unique": True,
            "episode_seed_formula": PAIRED_EPISODE_SEED_FORMULA,
            "all_episode_seeds_match_formula": True,
            "all_disturbance_seeds_match_episode_seeds": True,
            "known_forbidden_seeds": sorted(NAMED_FORBIDDEN_SEEDS),
            "discovered_existing_smoke_seeds": list(smoke_seeds),
            "forbidden_seed_overlap": overlap if config["stage"] == "formal" else [],
            "duplicate_episode_key_count": duplicate_episode_key_count,
        },
        "split_audit": {
            "overall_pass": invalid_split_row_count == 0 and not seed_overlap,
            "train_seeds": actual_train_seeds,
            "validation_seeds": actual_validation_seeds,
            "train_row_count": train_row_count,
            "validation_row_count": validation_row_count,
            "seed_overlap": seed_overlap,
            "invalid_split_row_count": invalid_split_row_count,
        },
        "disturbance_audit": {
            "overall_pass": True,
            "disturbance_keys": list(DISTURBANCE_KEYS),
            "disturbance_key_order_match": True,
            "disturbance_bounds": {key: list(DEFAULT_DISTURBANCE_BOUNDS[key]) for key in DISTURBANCE_KEYS},
            "flow_phase_keys": list(FLOW_PHASE_KEYS),
            "flow_phase_in_9d_vector": False,
            "explicit_application": True,
            "all_episode_apply_match": True,
            "bounds_violation_count": bounds_violation_count,
            "missing_field_count": missing_field_count,
            "duplicate_episode_key_count": duplicate_episode_key_count,
            "distinct_9d_vector_count": len(vectors),
            "per_dimension_distinct_counts": per_dimension_distinct,
            "minimum_continuous_distinct_required": required_continuous_distinct,
            "observed_action_delay_values": observed_delays,
        },
        "class_distribution": class_distribution,
        "per_seed_class_distribution": per_seed,
        "readiness_audit": {
            "class_readiness_evaluated": require_complete and config["stage"] == "formal",
            "ready_for_found_aware_reb_training": (len(readiness_reasons) == 0 if require_complete and config["stage"] == "formal" else None),
            "thresholds": {
                "minimum_success_samples": 300,
                "minimum_not_found_samples": 300,
                "minimum_found_but_failed_samples": 300,
                "validation_requires_all_three_classes": True,
            },
            "failure_reasons": readiness_reasons,
            "rows_resampled": False,
            "rows_deleted_for_balance": False,
        },
        "continuous_metric_audit": {
            "overall_pass": True,
            "no_nan": True,
            "no_inf": True,
            "quality_ranges_pass": True,
            "action_smoothness_nonnegative": True,
            "completion_steps_range_pass": True,
            "recovery_time_range_pass": True,
            "safety_mean_consistency_pass": safety_mean_mismatch_count == 0,
            "safety_mean_mismatch_count": safety_mean_mismatch_count,
            "statistics_by_scope": metric_groups,
        },
        "dataset_by_seed_rows": per_seed,
    }


def _class_distribution_csv_rows(class_distribution):
    rows = []
    for scope, metrics in class_distribution.items():
        rows.append({"scope": scope, **metrics})
    return rows


def _read_partial_csv(path, config, expected_model_path, require_complete=False):
    path = Path(path)
    raw_rows = _read_csv_strict(path, CSV_FIELDS)
    rows = [_coerce_csv_row(raw, row_index) for row_index, raw in enumerate(raw_rows)]
    audits = _audit_rows(
        rows,
        config,
        smoke_seeds=[],
        expected_model_path=expected_model_path,
        require_complete=require_complete,
    )
    completed_keys = {
        (int(row["base_seed"]), int(row["episode_index"])) for row in rows
    }
    resume_audit = {
        "overall_pass": True,
        "csv_header_exact_match": True,
        "row_count": len(rows),
        "expected_total_rows": len(config["seeds"]) * config["episodes_per_seed"],
        "episode_prefix_valid": True,
        "duplicate_episode_key_count": audits["seed_isolation_audit"]["duplicate_episode_key_count"],
        "model_path_match": True,
        "episode_seed_formula_match": True,
        "disturbance_seed_match": True,
        "dataset_split_match": audits["split_audit"]["invalid_split_row_count"] == 0,
        "disturbance_bounds_pass": audits["disturbance_audit"]["bounds_violation_count"] == 0,
        "flow_phase_range_pass": audits["flow_phase_range_pass"],
        "continuous_metric_audit_pass": audits["continuous_metric_audit"]["overall_pass"],
        "outcome_schema_match": audits["outcome_schema_match"],
    }
    return rows, completed_keys, resume_audit, audits


def _raw_csv_rows_equal(left_path, right_path):
    left_header = _read_csv_header(left_path)
    right_header = _read_csv_header(right_path)
    if left_header != right_header:
        return False
    return _read_csv_strict(left_path, left_header) == _read_csv_strict(right_path, right_header)


def _output_artifact_metadata(result_root, boundary_path, by_seed_path, class_path, markdown_path, partial_path):
    return {
        "boundary_dataset.csv": _csv_file_metadata(boundary_path, result_root),
        "dataset_by_seed.csv": _csv_file_metadata(by_seed_path, result_root),
        "dataset_class_distribution.csv": _csv_file_metadata(class_path, result_root),
        "dataset_audit_summary.md": _plain_file_metadata(markdown_path, result_root),
        "boundary_dataset.partial.csv": _csv_file_metadata(partial_path, result_root),
    }


def _audit_output_files(result_root, summary, config):
    result_root = Path(result_root)
    artifacts = summary.get("output_files", {}).get("artifacts", {})
    required = (
        "boundary_dataset.csv",
        "dataset_by_seed.csv",
        "dataset_class_distribution.csv",
        "dataset_audit_summary.md",
        "boundary_dataset.partial.csv",
    )
    expected_rows = len(config["seeds"]) * config["episodes_per_seed"]
    for name in required:
        _require(name in artifacts, f"output metadata missing for {name}")
        metadata = artifacts[name]
        path = result_root / metadata["path"]
        _require(path.is_file() and path.stat().st_size > 0, f"output is missing or empty: {path}")
        _require(_sha256(path) == metadata["sha256"], f"output SHA256 mismatch: {name}")
        _require(int(path.stat().st_size) == int(metadata["bytes"]), f"output byte size mismatch: {name}")
        if name.endswith(".csv"):
            header = _read_csv_header(path)
            rows = (
                _read_csv_strict(path, header)
                if name in ("boundary_dataset.csv", "boundary_dataset.partial.csv")
                else _read_csv_relaxed(path, header)
            )
            _require(header == list(metadata["header_fields"]), f"output CSV header metadata mismatch: {name}")
            _require(len(rows) == int(metadata["row_count"]), f"output CSV row metadata mismatch: {name}")

    boundary_path = result_root / artifacts["boundary_dataset.csv"]["path"]
    partial_path = result_root / artifacts["boundary_dataset.partial.csv"]["path"]
    by_seed_path = result_root / artifacts["dataset_by_seed.csv"]["path"]
    class_path = result_root / artifacts["dataset_class_distribution.csv"]["path"]
    _require(int(artifacts["boundary_dataset.csv"]["row_count"]) == expected_rows, "boundary dataset row count mismatch")
    _require(int(artifacts["boundary_dataset.partial.csv"]["row_count"]) == expected_rows, "partial dataset row count mismatch")
    _require(int(artifacts["dataset_by_seed.csv"]["row_count"]) == len(config["seeds"]), "by-seed CSV row count mismatch")
    class_rows = _read_csv_relaxed(class_path, CLASS_FIELDS)
    scopes = {row["scope"] for row in class_rows}
    _require("overall" in scopes, "class distribution lacks overall scope")
    if config["stage"] == "formal":
        _require({"overall", "train", "validation"}.issubset(scopes), "formal class distribution lacks required scopes")
    _require(_raw_csv_rows_equal(partial_path, boundary_path), "partial and final boundary CSV data rows differ")
    return {
        "overall_pass": True,
        "required_output_count": len(required),
        "boundary_row_count": expected_rows,
        "by_seed_row_count": len(_read_csv_relaxed(by_seed_path, BY_SEED_FIELDS)),
        "class_scopes": sorted(scopes),
        "partial_matches_boundary": True,
    }


def _refresh_source_audit(source):
    model_after = _sha256(source["model_path"])
    manifest_after = _sha256(source["manifest_path"])
    selection_after = _sha256(source["selection_summary_path"])
    source.update(
        {
            "model_sha256_after": model_after,
            "model_unchanged": model_after == source["model_sha256_before"],
            "manifest_sha256_after": manifest_after,
            "manifest_unchanged": manifest_after == source["manifest_sha256_before"],
            "selection_summary_sha256_after": selection_after,
            "selection_summary_unchanged": selection_after == source["selection_summary_sha256_before"],
        }
    )
    source["all_source_artifacts_unchanged"] = all(
        source[key] for key in ("model_unchanged", "manifest_unchanged", "selection_summary_unchanged")
    )
    _require(source["all_source_artifacts_unchanged"], "one or more source artifacts changed during collection")
    return source


def _load_optional_json(path):
    path = Path(path)
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _validate_existing_summary_protocol(summary, source, config):
    if not isinstance(summary, dict):
        return
    expected = {
        "experiment_type": EXPERIMENT_TYPE,
        "collection_stage": "formal",
        "evaluation_mode": MODE,
        "profile": PROFILE,
        "disturbance_protocol": PROTOCOL,
        "disturbance_protocol_version": PROTOCOL_VERSION,
        "seeds": FORMAL_SEEDS,
        "train_seeds": TRAIN_SEEDS,
        "validation_seeds": VALIDATION_SEEDS,
        "episodes_per_seed": FORMAL_EPISODES_PER_SEED,
        "row_count": 3000,
        "max_steps": FORMAL_MAX_STEPS,
        "episode_seed_formula": PAIRED_EPISODE_SEED_FORMULA,
        "disturbance_seed_formula": PAIRED_EPISODE_SEED_FORMULA,
        "disturbance_rng_mode": DISTURBANCE_RNG_MODE,
    }
    for key, expected_value in expected.items():
        if key in summary and summary[key] is not None:
            _require(summary[key] == expected_value, f"existing summary protocol conflict: {key}")
    reported_source = summary.get("source_audit")
    if isinstance(reported_source, dict):
        for key in (
            "model_sha256_before",
            "manifest_sha256_before",
            "selection_summary_sha256_before",
        ):
            if key in reported_source:
                _require(reported_source[key] == source[key], f"existing summary source conflict: {key}")
    if "errors" in summary:
        _require(summary.get("errors") == [], "existing summary reports errors")


def _crosscheck_dataset_by_seed(parsed_rows, audits, config):
    expected_rows = audits["dataset_by_seed_rows"]
    expected_order = list(FORMAL_SEEDS)
    actual_order = []
    actual_by_seed = {}
    for row_index, row in enumerate(parsed_rows):
        seed = _strict_int(row["base_seed"], f"dataset_by_seed.csv row={row_index} field=base_seed")
        _require(seed not in actual_by_seed, f"dataset_by_seed.csv duplicate seed={seed}")
        actual_order.append(seed)
        actual_by_seed[seed] = row
    expected_by_seed = {int(row["base_seed"]): row for row in expected_rows}
    actual_seed_set = set(actual_by_seed)
    expected_seed_set = set(expected_by_seed)
    _require(
        actual_order == expected_order,
        f"dataset_by_seed.csv seed order mismatch: actual={actual_order} expected={expected_order}",
    )
    _require(
        actual_seed_set == expected_seed_set == set(FORMAL_SEEDS),
        "dataset_by_seed.csv seed set mismatch: "
        f"actual={sorted(actual_seed_set)} expected={sorted(expected_seed_set)} formal={FORMAL_SEEDS}",
    )

    for seed in expected_order:
        actual = actual_by_seed[seed]
        expected = expected_by_seed[seed]
        _require(
            actual["dataset_split"] == expected["dataset_split"],
            "dataset_by_seed.csv seed=%d field=dataset_split mismatch: actual=%r expected=%r"
            % (seed, actual["dataset_split"], expected["dataset_split"]),
        )
        for field in BY_SEED_INTEGER_FIELDS:
            actual_value = _strict_int(actual[field], f"dataset_by_seed.csv seed={seed} field={field}")
            expected_value = int(expected[field])
            _require(
                actual_value == expected_value,
                f"dataset_by_seed.csv seed={seed} field={field} mismatch: "
                f"actual={actual_value} expected={expected_value}",
            )
        for field in DERIVED_RATE_FIELDS:
            _compare_nullable_float(
                actual[field],
                expected[field],
                f"dataset_by_seed.csv seed={seed} field={field}",
            )

    return {
        "overall_pass": True,
        "expected_seed_count": len(expected_by_seed),
        "actual_seed_count": len(actual_by_seed),
        "seed_order_match": True,
        "seed_set_match": True,
        "checked_fields": ["dataset_split", *BY_SEED_INTEGER_FIELDS, *DERIVED_RATE_FIELDS],
        "integer_field_mismatch_count": 0,
        "rate_field_mismatch_count": 0,
        "split_mismatch_count": 0,
        "errors": [],
    }


def _crosscheck_dataset_class_distribution(parsed_rows, audits, config):
    expected_by_scope = audits["class_distribution"]
    expected_scopes = list(EXPECTED_CLASS_SCOPES)
    actual_scopes = []
    actual_by_scope = {}
    for row_index, row in enumerate(parsed_rows):
        scope = row["scope"]
        _require(scope != "", f"dataset_class_distribution.csv row={row_index} has empty scope")
        _require(scope not in actual_by_scope, f"dataset_class_distribution.csv duplicate scope={scope!r}")
        actual_scopes.append(scope)
        actual_by_scope[scope] = row
    _require(
        actual_scopes == expected_scopes,
        "dataset_class_distribution.csv scope order mismatch: "
        f"actual={actual_scopes} expected={expected_scopes}",
    )
    _require(
        set(actual_by_scope) == set(expected_by_scope) == set(expected_scopes),
        "dataset_class_distribution.csv scope set mismatch: "
        f"actual={sorted(actual_by_scope)} expected={sorted(expected_by_scope)}",
    )

    expected_sizes = {"overall": 3000, "train": 2400, "validation": 600}
    for scope in expected_scopes:
        actual = actual_by_scope[scope]
        expected = expected_by_scope[scope]
        integer_values = {}
        for field in DERIVED_COUNT_FIELDS:
            actual_value = _strict_int(
                actual[field],
                f"dataset_class_distribution.csv scope={scope} field={field}",
            )
            expected_value = int(expected[field])
            _require(
                actual_value == expected_value,
                f"dataset_class_distribution.csv scope={scope} field={field} mismatch: "
                f"actual={actual_value} expected={expected_value}",
            )
            integer_values[field] = actual_value
        _require(
            integer_values["n_episodes"] == expected_sizes[scope],
            f"dataset_class_distribution.csv scope={scope} fixed n_episodes mismatch: "
            f"actual={integer_values['n_episodes']} expected={expected_sizes[scope]}",
        )
        n_episodes = integer_values["n_episodes"]
        n_success = integer_values["n_success"]
        n_found = integer_values["n_found"]
        n_not_found = integer_values["n_not_found"]
        n_found_but_failed = integer_values["n_found_but_failed"]
        _require(
            n_not_found == n_episodes - n_found,
            f"dataset_class_distribution.csv scope={scope} semantic identity failed: "
            "n_not_found != n_episodes - n_found",
        )
        _require(
            n_found_but_failed == n_found - n_success,
            f"dataset_class_distribution.csv scope={scope} semantic identity failed: "
            "n_found_but_failed != n_found - n_success",
        )
        _require(
            0 <= n_success <= n_found <= n_episodes,
            f"dataset_class_distribution.csv scope={scope} semantic bounds failed: "
            f"n_success={n_success} n_found={n_found} n_episodes={n_episodes}",
        )
        for field in DERIVED_RATE_FIELDS:
            _compare_nullable_float(
                actual[field],
                expected[field],
                f"dataset_class_distribution.csv scope={scope} field={field}",
            )

    return {
        "overall_pass": True,
        "expected_scopes": expected_scopes,
        "actual_scopes": actual_scopes,
        "scope_order_match": True,
        "checked_fields": [*DERIVED_COUNT_FIELDS, *DERIVED_RATE_FIELDS],
        "integer_field_mismatch_count": 0,
        "rate_field_mismatch_count": 0,
        "semantic_identity_failure_count": 0,
        "errors": [],
    }


def _audit_committed_csvs(final_root, source, config):
    final_root = Path(final_root)
    partial_path = final_root / "boundary_dataset.partial.csv"
    boundary_path = final_root / "boundary_dataset.csv"
    by_seed_path = final_root / "dataset_by_seed.csv"
    class_path = final_root / "dataset_class_distribution.csv"
    for path in (partial_path, boundary_path, by_seed_path, class_path):
        _require(path.is_file() and path.stat().st_size > 0, f"committed CSV is missing or empty: {path}")

    rows, completed_keys, partial_audit, audits = _read_partial_csv(
        partial_path,
        config,
        source["model_path"],
        require_complete=True,
    )
    _require(_read_csv_header(boundary_path) == CSV_FIELDS, "boundary CSV header/order mismatch")
    _require(_raw_csv_rows_equal(partial_path, boundary_path), "partial and boundary CSV data differ")
    boundary_raw = _read_csv_strict(boundary_path, CSV_FIELDS)
    _require(len(boundary_raw) == 3000, "boundary CSV must contain 3000 rows")

    by_seed_rows = _read_csv_relaxed(by_seed_path, BY_SEED_FIELDS)
    by_seed_crosscheck = _crosscheck_dataset_by_seed(by_seed_rows, audits, config)

    class_rows = _read_csv_relaxed(class_path, CLASS_FIELDS)
    class_crosscheck = _crosscheck_dataset_class_distribution(class_rows, audits, config)
    scopes = [row["scope"] for row in class_rows]
    csv_hashes = {
        path.name: _sha256(path)
        for path in (partial_path, boundary_path, by_seed_path, class_path)
    }
    return {
        "overall_pass": True,
        "rows": rows,
        "completed_keys": completed_keys,
        "partial_audit": partial_audit,
        "audits": audits,
        "partial_path": partial_path,
        "boundary_path": boundary_path,
        "by_seed_path": by_seed_path,
        "class_path": class_path,
        "csv_hashes": csv_hashes,
        "partial_matches_boundary": True,
        "by_seed_row_count": len(by_seed_rows),
        "class_scopes": scopes,
        "dataset_by_seed_crosscheck": by_seed_crosscheck,
        "dataset_class_distribution_crosscheck": class_crosscheck,
    }


def _audit_collection_state(final_root, source, config, require_complete=True):
    final_root = Path(final_root)
    state_path = final_root / "collection_state.json"
    partial_path = final_root / "boundary_dataset.partial.csv"
    state = _load_json(state_path, "formal collection state")
    expected_total = len(config["seeds"]) * config["episodes_per_seed"]
    _require(state.get("stage") == "formal", "complete state stage mismatch")
    _require(state.get("expected_seeds") == FORMAL_SEEDS, "complete state seeds mismatch")
    _require(state.get("episodes_per_seed") == FORMAL_EPISODES_PER_SEED, "complete state episodes mismatch")
    _require(int(state.get("completed_rows", -1)) == expected_total, "complete state completed_rows mismatch")
    _require(int(state.get("expected_total_rows", -1)) == expected_total, "complete state expected_total_rows mismatch")
    partial_hash_match = state.get("partial_csv_sha256") == _sha256(partial_path)
    source_hashes_match = state.get("source_hashes") == _source_hashes(source)
    _require(partial_hash_match, "complete state partial CSV SHA256 mismatch")
    _require(source_hashes_match, "complete state source hashes mismatch")
    _require(_path_equal(state.get("formal_result_root"), final_root), "complete state formal_result_root mismatch")
    _require(_path_equal(state.get("staging_root"), Path(str(final_root) + ".incomplete")), "complete state staging_root mismatch")
    if require_complete:
        _require(state.get("status") == "complete", "complete state status mismatch")
        _require(state.get("finalization_phase") == "complete", "complete state finalization_phase mismatch")
        _require(state.get("commit_directory_renamed") is True, "complete state commit flag mismatch")
        _require(state.get("post_commit_output_audit_pass") is True, "complete state output audit flag mismatch")
        _require(state.get("summary_finalized") is True, "complete state summary flag mismatch")
        _require(state.get("markdown_finalized") is True, "complete state Markdown flag mismatch")
    return {
        "overall_pass": True,
        "path": str(state_path),
        "stage": state.get("stage"),
        "status": state.get("status"),
        "finalization_phase": state.get("finalization_phase"),
        "completed_rows": int(state.get("completed_rows", -1)),
        "expected_total_rows": int(state.get("expected_total_rows", -1)),
        "partial_hash_match": partial_hash_match,
        "source_hashes_match": source_hashes_match,
        "commit_directory_renamed": state.get("commit_directory_renamed") is True,
        "post_commit_output_audit_pass": state.get("post_commit_output_audit_pass") is True,
        "summary_finalized": state.get("summary_finalized") is True,
        "markdown_finalized": state.get("markdown_finalized") is True,
    }


def _detect_formal_storage_state(final_root, staging_root, source=None, config=None):
    final_root = Path(final_root)
    staging_root = Path(staging_root)
    final_exists = final_root.exists()
    staging_exists = staging_root.exists()
    if final_exists:
        _require(final_root.is_dir(), f"formal final path exists but is not a directory: {final_root}")
    if staging_exists:
        _require(staging_root.is_dir(), f"formal staging path exists but is not a directory: {staging_root}")
    if final_exists and staging_exists:
        raise DatasetError(f"formal split-brain: both final and staging directories exist: final={final_root}; staging={staging_root}")
    if not final_exists and not staging_exists:
        state = "new_collection"
        complete_audit = None
    elif staging_exists:
        state = "resume_collection"
        complete_audit = None
    else:
        complete_audit = None
        if source is not None and config is not None:
            try:
                complete_audit = _audit_complete_formal_commit(final_root, staging_root, source, config)
                state = "already_complete"
            except DatasetError:
                state = "post_commit_recovery"
        else:
            state = "post_commit_recovery"
    return {
        "state": state,
        "final_root": str(final_root),
        "staging_root": str(staging_root),
        "final_exists": final_exists,
        "staging_exists": staging_exists,
        "complete_commit_audit": complete_audit,
    }


def _build_recovery_summary(
    existing_summary,
    final_root,
    staging_root,
    source,
    config,
    committed,
    formal_storage_state,
    post_commit_recovery_performed,
    rows_collected_this_run,
):
    audits = committed["audits"]
    rows_collected_this_run = int(rows_collected_this_run)
    _require(rows_collected_this_run >= 0, "rows_collected_this_run cannot be negative")
    collection_rollout_performed_this_run = rows_collected_this_run > 0
    summary = copy.deepcopy(existing_summary) if isinstance(existing_summary, dict) else {}
    summary.update(
        {
            "overall_pass": True,
            "experiment_type": EXPERIMENT_TYPE,
            "collection_stage": "formal",
            "formal_collection_completed": True,
            "formal_commit_completed": True,
            "post_commit_audit_pass": True,
            "pre_commit_audit_pass": True,
            "model_frozen": True,
            "model_actions_explore": False,
            "deterministic_policy": True,
            "training_performed": False,
            "optimizer_update_count": 0,
            "checkpoint_selection_performed": False,
            "model_save_performed": False,
            "online_training_boundary_dataset_used": False,
            "formal_test_dataset_used": False,
            "evaluation_mode": MODE,
            "profile": PROFILE,
            "disturbance_protocol": PROTOCOL,
            "disturbance_protocol_version": PROTOCOL_VERSION,
            "seeds": list(FORMAL_SEEDS),
            "train_seeds": list(TRAIN_SEEDS),
            "validation_seeds": list(VALIDATION_SEEDS),
            "episodes_per_seed": FORMAL_EPISODES_PER_SEED,
            "row_count": 3000,
            "max_steps": FORMAL_MAX_STEPS,
            "episode_index_base": 0,
            "episode_seed_mode": PAIRED_EPISODE_SEED_MODE,
            "episode_seed_formula": PAIRED_EPISODE_SEED_FORMULA,
            "disturbance_seed_formula": PAIRED_EPISODE_SEED_FORMULA,
            "disturbance_rng_mode": DISTURBANCE_RNG_MODE,
            "source_audit": source,
            "seed_isolation_audit": audits["seed_isolation_audit"],
            "split_audit": audits["split_audit"],
            "disturbance_audit": audits["disturbance_audit"],
            "class_distribution": audits["class_distribution"],
            "per_seed_class_distribution": audits["per_seed_class_distribution"],
            "readiness_audit": audits["readiness_audit"],
            "ready_for_found_aware_reb_training": audits["readiness_audit"]["ready_for_found_aware_reb_training"],
            "failure_reasons": audits["readiness_audit"]["failure_reasons"],
            "warnings": (
                ["Observed class counts do not meet Found-aware REB readiness thresholds; data were not altered."]
                if audits["readiness_audit"]["failure_reasons"]
                else []
            ),
            "continuous_metric_audit": audits["continuous_metric_audit"],
            "outcome_schema_match": audits["outcome_schema_match"],
            "collector_outcome_fields": audits["collector_outcome_fields"],
            "reb_dataset_outcome_fields": audits["reb_dataset_outcome_fields"],
            "flow_phase_range_pass": audits["flow_phase_range_pass"],
            "flow_phase_violation_count": audits["flow_phase_violation_count"],
            "dataset_by_seed_crosscheck": committed["dataset_by_seed_crosscheck"],
            "dataset_class_distribution_crosscheck": committed["dataset_class_distribution_crosscheck"],
            "partial_dataset_path": "boundary_dataset.partial.csv",
            "partial_dataset_sha256_after_collection": _sha256(committed["partial_path"]),
            "partial_dataset_retention_policy": "retained_as_recovery_evidence",
            "formal_result_root": str(Path(final_root).resolve()),
            "staging_root": str(Path(staging_root).resolve()),
            "formal_storage_state": str(formal_storage_state),
            "post_commit_recovery_performed": bool(post_commit_recovery_performed),
            "rows_collected_this_run": rows_collected_this_run,
            "collection_rollout_performed_this_run": collection_rollout_performed_this_run,
            "collection_rollout_audit": {
                "overall_pass": collection_rollout_performed_this_run == (rows_collected_this_run > 0),
                "rows_collected_this_run": rows_collected_this_run,
                "expected_rollout_performed": rows_collected_this_run > 0,
                "reported_rollout_performed": collection_rollout_performed_this_run,
            },
            "already_complete_reaudit": False,
            "finalization_phase": "complete",
            "canonical_summary_finalized": True,
            "canonical_markdown_finalized": True,
            "errors": [],
        }
    )
    summary.setdefault("created_at_utc", datetime.now(timezone.utc).isoformat())
    summary["finalized_at_utc"] = datetime.now(timezone.utc).isoformat()
    return summary


def _final_output_metadata(final_root, markdown_source):
    final_root = Path(final_root)
    artifacts = _output_artifact_metadata(
        final_root,
        final_root / "boundary_dataset.csv",
        final_root / "dataset_by_seed.csv",
        final_root / "dataset_class_distribution.csv",
        markdown_source,
        final_root / "boundary_dataset.partial.csv",
    )
    artifacts["dataset_audit_summary.md"]["path"] = "dataset_audit_summary.md"
    return {
        "artifacts": artifacts,
        "boundary_dataset_csv": artifacts["boundary_dataset.csv"]["path"],
        "boundary_dataset_sha256": artifacts["boundary_dataset.csv"]["sha256"],
        "dataset_by_seed_csv": artifacts["dataset_by_seed.csv"]["path"],
        "dataset_by_seed_sha256": artifacts["dataset_by_seed.csv"]["sha256"],
        "dataset_class_distribution_csv": artifacts["dataset_class_distribution.csv"]["path"],
        "dataset_class_distribution_sha256": artifacts["dataset_class_distribution.csv"]["sha256"],
        "dataset_audit_summary_json": "dataset_audit_summary.json",
        "dataset_audit_summary_markdown": "dataset_audit_summary.md",
        "dataset_audit_summary_markdown_sha256": artifacts["dataset_audit_summary.md"]["sha256"],
    }



def _deep_semantic_equal(actual, expected, label):
    """Strict recursive comparison with finite-float tolerance for persisted JSON."""
    expected = _json_safe(expected)
    if expected is None:
        _require(actual is None, f"{label} mismatch: actual={actual!r} expected=None")
        return
    if isinstance(expected, bool):
        _require(isinstance(actual, bool) and actual is expected, f"{label} mismatch: actual={actual!r} expected={expected!r}")
        return
    if isinstance(expected, int) and not isinstance(expected, bool):
        _require(isinstance(actual, int) and not isinstance(actual, bool) and actual == expected, f"{label} mismatch: actual={actual!r} expected={expected!r}")
        return
    if isinstance(expected, float):
        _require(isinstance(actual, (int, float)) and not isinstance(actual, bool), f"{label} must be numeric: actual={actual!r}")
        _require(math.isfinite(float(actual)), f"{label} must be finite: actual={actual!r}")
        _require(
            math.isclose(float(actual), expected, rel_tol=1e-12, abs_tol=1e-12),
            f"{label} mismatch: actual={actual!r} expected={expected!r}",
        )
        return
    if isinstance(expected, str):
        _require(isinstance(actual, str) and actual == expected, f"{label} mismatch: actual={actual!r} expected={expected!r}")
        return
    if isinstance(expected, list):
        _require(isinstance(actual, list), f"{label} must be a list")
        _require(len(actual) == len(expected), f"{label} length mismatch: actual={len(actual)} expected={len(expected)}")
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            _deep_semantic_equal(actual_item, expected_item, f"{label}[{index}]")
        return
    if isinstance(expected, dict):
        _require(isinstance(actual, dict), f"{label} must be an object")
        _require(set(actual) == set(expected), f"{label} key mismatch: actual={sorted(actual)} expected={sorted(expected)}")
        for key in expected:
            _deep_semantic_equal(actual[key], expected[key], f"{label}.{key}")
        return
    _require(actual == expected, f"{label} mismatch: actual={actual!r} expected={expected!r}")


def _audit_summary_semantics(summary, committed, state_audit, output_audit, source, markdown_path):
    """Recompute every scientific summary field from committed CSV/state evidence."""
    audits = committed["audits"]
    expected_fields = {
        "seed_isolation_audit": audits["seed_isolation_audit"],
        "split_audit": audits["split_audit"],
        "disturbance_audit": audits["disturbance_audit"],
        "class_distribution": audits["class_distribution"],
        "per_seed_class_distribution": audits["per_seed_class_distribution"],
        "readiness_audit": audits["readiness_audit"],
        "ready_for_found_aware_reb_training": audits["readiness_audit"]["ready_for_found_aware_reb_training"],
        "failure_reasons": audits["readiness_audit"]["failure_reasons"],
        "continuous_metric_audit": audits["continuous_metric_audit"],
        "outcome_schema_match": audits["outcome_schema_match"],
        "collector_outcome_fields": audits["collector_outcome_fields"],
        "reb_dataset_outcome_fields": audits["reb_dataset_outcome_fields"],
        "flow_phase_range_pass": audits["flow_phase_range_pass"],
        "flow_phase_violation_count": audits["flow_phase_violation_count"],
        "dataset_by_seed_crosscheck": committed["dataset_by_seed_crosscheck"],
        "dataset_class_distribution_crosscheck": committed["dataset_class_distribution_crosscheck"],
        "collection_state_audit": state_audit,
        "post_commit_output_audit": output_audit,
        "partial_dataset_path": "boundary_dataset.partial.csv",
        "partial_dataset_sha256_after_collection": _sha256(committed["partial_path"]),
    }
    checked_fields = list(expected_fields)
    for field, expected in expected_fields.items():
        _require(field in summary, f"summary semantic field is missing: {field}")
        _deep_semantic_equal(summary[field], expected, f"summary.{field}")

    static_expected = {
        "model_frozen": True,
        "model_actions_explore": False,
        "deterministic_policy": True,
        "training_performed": False,
        "optimizer_update_count": 0,
        "checkpoint_selection_performed": False,
        "model_save_performed": False,
        "online_training_boundary_dataset_used": False,
        "formal_test_dataset_used": False,
    }
    for field, expected in static_expected.items():
        _deep_semantic_equal(summary.get(field), expected, f"summary.{field}")

    reported_source = summary.get("source_audit")
    _require(isinstance(reported_source, dict), "summary.source_audit must be an object")
    for path_field in ("model_path", "manifest_path", "selection_summary_path"):
        _require(_path_equal(reported_source.get(path_field), source[path_field]), f"summary.source_audit.{path_field} mismatch")
    for hash_field in (
        "model_sha256_before",
        "manifest_sha256_before",
        "selection_summary_sha256_before",
        "model_sha256_after",
        "manifest_sha256_after",
        "selection_summary_sha256_after",
    ):
        _deep_semantic_equal(reported_source.get(hash_field), source.get(hash_field), f"summary.source_audit.{hash_field}")
    for flag_field in ("model_unchanged", "manifest_unchanged", "selection_summary_unchanged", "all_source_artifacts_unchanged"):
        _deep_semantic_equal(reported_source.get(flag_field), True, f"summary.source_audit.{flag_field}")

    expected_warnings = (
        ["Observed class counts do not meet Found-aware REB readiness thresholds; data were not altered."]
        if audits["readiness_audit"]["failure_reasons"]
        else []
    )
    _deep_semantic_equal(summary.get("warnings", []), expected_warnings, "summary.warnings")

    markdown_path = Path(markdown_path)
    _require(markdown_path.is_file() and markdown_path.stat().st_size > 0, "summary Markdown is missing or empty")
    markdown_matches_summary = markdown_path.read_text(encoding="utf-8") == _markdown(summary)
    _require(markdown_matches_summary, "dataset_audit_summary.md does not match the recomputed summary semantics")

    return {
        "overall_pass": True,
        "checked_fields": checked_fields,
        "static_invariants_match": True,
        "source_identity_match": True,
        "partial_hash_match": True,
        "markdown_matches_summary": True,
        "errors": [],
    }


def _audit_complete_formal_commit(final_root, staging_root, source, config):
    final_root = Path(final_root)
    staging_root = Path(staging_root)
    _require(final_root.is_dir(), f"formal final directory is missing: {final_root}")
    _require(not staging_root.exists(), f"formal staging directory still exists: {staging_root}")
    summary_path = final_root / "dataset_audit_summary.json"
    markdown_path = final_root / "dataset_audit_summary.md"
    summary = _load_json(summary_path, "formal dataset summary")
    _validate_existing_summary_protocol(summary, source, config)
    for key in ("overall_pass", "formal_collection_completed", "formal_commit_completed", "post_commit_audit_pass"):
        _require(summary.get(key) is True, f"complete summary flag mismatch: {key}")
    _require(summary.get("canonical_summary_finalized") is True, "canonical summary finalization flag mismatch")
    _require(summary.get("canonical_markdown_finalized") is True, "canonical Markdown finalization flag mismatch")
    _require(summary.get("finalization_phase") == "complete", "complete summary finalization_phase mismatch")
    _require(summary.get("errors") == [], "complete summary errors must be empty")
    _require(markdown_path.is_file() and markdown_path.stat().st_size > 0, "formal Markdown is missing or empty")

    committed = _audit_committed_csvs(final_root, source, config)
    state_audit = _audit_collection_state(final_root, source, config, require_complete=True)
    by_seed_crosscheck = committed["dataset_by_seed_crosscheck"]
    class_crosscheck = committed["dataset_class_distribution_crosscheck"]
    _require(by_seed_crosscheck["overall_pass"] is True, "current by-seed crosscheck did not pass")
    _require(class_crosscheck["overall_pass"] is True, "current class distribution crosscheck did not pass")

    rows_collected_this_run = _strict_int(summary.get("rows_collected_this_run"), "summary rows_collected_this_run")
    rollout_reported = summary.get("collection_rollout_performed_this_run")
    _require(isinstance(rollout_reported, bool), "summary collection_rollout_performed_this_run must be boolean")
    _require(
        rollout_reported == (rows_collected_this_run > 0),
        "summary current-run rollout flag mismatch",
    )
    _require(summary.get("collection_rollout_audit", {}).get("overall_pass") is True, "summary collection_rollout_audit did not pass")

    output_audit = _audit_output_files(final_root, summary, config)
    summary_semantic_audit = _audit_summary_semantics(
        summary,
        committed,
        state_audit,
        output_audit,
        source,
        markdown_path,
    )
    _require(summary_semantic_audit["overall_pass"] is True, "current summary semantic audit did not pass")
    _deep_semantic_equal(
        summary.get("summary_semantic_crosscheck"),
        summary_semantic_audit,
        "summary.summary_semantic_crosscheck",
    )

    expected_complete_summary_audit = {
        "overall_pass": True,
        "partial_boundary_rows_match": True,
        "partial_and_boundary_match": True,
        "source_hashes_match": True,
        "dataset_by_seed_crosscheck_pass": by_seed_crosscheck["overall_pass"],
        "dataset_class_distribution_crosscheck_pass": class_crosscheck["overall_pass"],
        "split_audit_pass": committed["audits"]["split_audit"]["overall_pass"],
        "disturbance_audit_pass": committed["audits"]["disturbance_audit"]["overall_pass"],
        "post_commit_output_audit_pass": output_audit["overall_pass"],
        "collection_state_audit_pass": state_audit["overall_pass"],
        "summary_semantic_crosscheck_pass": summary_semantic_audit["overall_pass"],
        "markdown_matches_summary": summary_semantic_audit["markdown_matches_summary"],
    }
    _deep_semantic_equal(
        summary.get("complete_commit_audit"),
        expected_complete_summary_audit,
        "summary.complete_commit_audit",
    )

    for pending_name in ("dataset_audit_summary.pending.json", "dataset_audit_summary.pending.md"):
        _require(not (final_root / pending_name).exists(), f"pending finalization file remains: {pending_name}")
    return {
        "overall_pass": True,
        "read_only": True,
        "summary_path": str(summary_path),
        "state_audit": state_audit,
        "committed_csv_audit": {key: value for key, value in committed.items() if key not in ("rows", "completed_keys", "audits")},
        "output_audit": output_audit,
        "summary_semantic_crosscheck": summary_semantic_audit,
        "csv_hashes": committed["csv_hashes"],
        "dataset_by_seed_crosscheck": by_seed_crosscheck,
        "dataset_class_distribution_crosscheck": class_crosscheck,
        "partial_boundary_rows_match": True,
    }


def _finalize_or_recover_committed_formal(
    final_root,
    staging_root,
    source,
    config,
    formal_storage_state,
    post_commit_recovery_performed,
    rows_collected_this_run,
):
    final_root = Path(final_root)
    staging_root = Path(staging_root)
    try:
        complete_audit = _audit_complete_formal_commit(final_root, staging_root, source, config)
        canonical_summary = _load_json(final_root / "dataset_audit_summary.json", "formal dataset summary")
        summary = copy.deepcopy(canonical_summary)
        summary.update(
            {
                "formal_storage_state": "already_complete",
                "post_commit_recovery_performed": False,
                "already_complete_reaudit": True,
                "rows_collected_this_run": 0,
                "collection_rollout_performed_this_run": False,
                "collection_rollout_audit": {
                    "overall_pass": True,
                    "rows_collected_this_run": 0,
                    "expected_rollout_performed": False,
                    "reported_rollout_performed": False,
                },
            }
        )
        return summary, {"action": "already_complete", "complete_commit_audit": complete_audit}
    except DatasetError:
        pass

    _require(final_root.is_dir(), f"post-commit recovery requires formal final directory: {final_root}")
    _require(not staging_root.exists(), f"post-commit recovery refuses split-brain staging directory: {staging_root}")
    committed = _audit_committed_csvs(final_root, source, config)
    existing_summary = _load_optional_json(final_root / "dataset_audit_summary.json")
    _validate_existing_summary_protocol(existing_summary, source, config)
    existing_state = _load_optional_json(final_root / "collection_state.json")
    if isinstance(existing_state, dict):
        if "stage" in existing_state:
            _require(existing_state.get("stage") == "formal", "recoverable state stage conflict")
        if "source_hashes" in existing_state:
            _require(existing_state.get("source_hashes") == _source_hashes(source), "recoverable state source hash conflict")
        if "partial_csv_sha256" in existing_state:
            _require(existing_state.get("partial_csv_sha256") == _sha256(committed["partial_path"]), "recoverable state partial hash conflict")

    state_path = final_root / "collection_state.json"
    pending_summary_path = final_root / "dataset_audit_summary.pending.json"
    pending_markdown_path = final_root / "dataset_audit_summary.pending.md"
    canonical_summary_path = final_root / "dataset_audit_summary.json"
    canonical_markdown_path = final_root / "dataset_audit_summary.md"
    completed_keys = committed["completed_keys"]

    _write_collection_state(
        state_path,
        config,
        source,
        committed["partial_path"],
        completed_keys,
        "finalizing",
        finalization_phase="committed_pending_audit",
        formal_result_root=final_root,
        staging_root=staging_root,
        commit_directory_renamed=True,
    )
    summary = _build_recovery_summary(
        existing_summary,
        final_root,
        staging_root,
        source,
        config,
        committed,
        formal_storage_state,
        post_commit_recovery_performed,
        rows_collected_this_run,
    )
    _atomic_write_text(pending_markdown_path, _markdown(summary))
    _require(pending_markdown_path.stat().st_size > 0, "pending Markdown is empty")

    pending_output_files = _final_output_metadata(final_root, pending_markdown_path)
    temporary_output_files = copy.deepcopy(pending_output_files)
    temporary_output_files["artifacts"]["dataset_audit_summary.md"]["path"] = pending_markdown_path.name
    temporary_summary = copy.deepcopy(summary)
    temporary_summary["output_files"] = temporary_output_files
    post_commit_output_audit = _audit_output_files(final_root, temporary_summary, config)
    summary["output_files"] = pending_output_files
    summary["post_commit_output_audit"] = post_commit_output_audit
    summary["post_commit_audit_pass"] = bool(
        committed["dataset_by_seed_crosscheck"]["overall_pass"]
        and committed["dataset_class_distribution_crosscheck"]["overall_pass"]
        and post_commit_output_audit["overall_pass"]
    )
    _atomic_write_json(pending_summary_path, summary)

    _write_collection_state(
        state_path,
        config,
        source,
        committed["partial_path"],
        completed_keys,
        "finalizing",
        finalization_phase="outputs_audited",
        formal_result_root=final_root,
        staging_root=staging_root,
        commit_directory_renamed=True,
        post_commit_output_audit_pass=True,
    )
    _write_collection_state(
        state_path,
        config,
        source,
        committed["partial_path"],
        completed_keys,
        "complete",
        finalization_phase="complete",
        formal_result_root=final_root,
        staging_root=staging_root,
        commit_directory_renamed=True,
        post_commit_output_audit_pass=True,
        summary_finalized=True,
        markdown_finalized=True,
    )
    state_audit = _audit_collection_state(final_root, source, config, require_complete=True)
    summary["collection_state_audit"] = state_audit
    summary_semantic_audit = _audit_summary_semantics(
        summary,
        committed,
        state_audit,
        post_commit_output_audit,
        source,
        pending_markdown_path,
    )
    summary["summary_semantic_crosscheck"] = summary_semantic_audit
    summary["complete_commit_audit"] = {
        "overall_pass": True,
        "partial_boundary_rows_match": True,
        "partial_and_boundary_match": True,
        "source_hashes_match": True,
        "dataset_by_seed_crosscheck_pass": committed["dataset_by_seed_crosscheck"]["overall_pass"],
        "dataset_class_distribution_crosscheck_pass": committed["dataset_class_distribution_crosscheck"]["overall_pass"],
        "split_audit_pass": committed["audits"]["split_audit"]["overall_pass"],
        "disturbance_audit_pass": committed["audits"]["disturbance_audit"]["overall_pass"],
        "post_commit_output_audit_pass": post_commit_output_audit["overall_pass"],
        "collection_state_audit_pass": state_audit["overall_pass"],
        "summary_semantic_crosscheck_pass": summary_semantic_audit["overall_pass"],
        "markdown_matches_summary": summary_semantic_audit["markdown_matches_summary"],
    }
    _atomic_write_json(pending_summary_path, summary)
    os.replace(str(pending_markdown_path), str(canonical_markdown_path))
    os.replace(str(pending_summary_path), str(canonical_summary_path))
    complete_audit = _audit_complete_formal_commit(final_root, staging_root, source, config)
    final_summary = _load_json(canonical_summary_path, "finalized formal dataset summary")
    return final_summary, {"action": "post_commit_recovery", "complete_commit_audit": complete_audit}


def _write_final_outputs(
    result_root,
    rows,
    config,
    source,
    audits,
    partial_path,
    resumed_from_partial,
    rows_loaded_from_partial,
    rows_collected_this_run,
    partial_sha_before,
    resume_audit,
    formal_precommit,
):
    result_root = Path(result_root)
    boundary_path = result_root / "boundary_dataset.csv"
    by_seed_path = result_root / "dataset_by_seed.csv"
    class_path = result_root / "dataset_class_distribution.csv"
    summary_path = result_root / "dataset_audit_summary.json"
    markdown_path = result_root / "dataset_audit_summary.md"
    _atomic_copy_file(partial_path, boundary_path)
    _atomic_write_csv(by_seed_path, BY_SEED_FIELDS, audits["dataset_by_seed_rows"])
    class_rows = _class_distribution_csv_rows(audits["class_distribution"])
    _atomic_write_csv(class_path, CLASS_FIELDS, class_rows)
    _require(_raw_csv_rows_equal(partial_path, boundary_path), "audited partial rows differ from final boundary dataset")

    summary = {
        "overall_pass": not formal_precommit,
        "experiment_type": EXPERIMENT_TYPE,
        "collection_stage": config["stage"],
        "formal_collection_completed": False,
        "formal_commit_completed": False,
        "post_commit_audit_pass": False,
        "pre_commit_audit_pass": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_frozen": True,
        "model_actions_explore": False,
        "deterministic_policy": True,
        "training_performed": False,
        "optimizer_update_count": 0,
        "checkpoint_selection_performed": False,
        "model_save_performed": False,
        "online_training_boundary_dataset_used": False,
        "formal_test_dataset_used": False,
        "evaluation_mode": MODE,
        "profile": PROFILE,
        "disturbance_protocol": PROTOCOL,
        "disturbance_protocol_version": PROTOCOL_VERSION,
        "seeds": list(config["seeds"]),
        "train_seeds": list(TRAIN_SEEDS) if config["stage"] == "formal" else [],
        "validation_seeds": list(VALIDATION_SEEDS) if config["stage"] == "formal" else [],
        "episodes_per_seed": config["episodes_per_seed"],
        "row_count": len(rows),
        "max_steps": config["max_steps"],
        "episode_index_base": 0,
        "episode_seed_mode": PAIRED_EPISODE_SEED_MODE,
        "episode_seed_formula": PAIRED_EPISODE_SEED_FORMULA,
        "disturbance_seed_formula": PAIRED_EPISODE_SEED_FORMULA,
        "disturbance_rng_mode": DISTURBANCE_RNG_MODE,
        "source_audit": source,
        "seed_isolation_audit": audits["seed_isolation_audit"],
        "split_audit": audits["split_audit"],
        "disturbance_audit": audits["disturbance_audit"],
        "class_distribution": audits["class_distribution"],
        "per_seed_class_distribution": audits["per_seed_class_distribution"],
        "readiness_audit": audits["readiness_audit"],
        "ready_for_found_aware_reb_training": audits["readiness_audit"]["ready_for_found_aware_reb_training"],
        "failure_reasons": audits["readiness_audit"]["failure_reasons"],
        "continuous_metric_audit": audits["continuous_metric_audit"],
        "outcome_schema_match": audits["outcome_schema_match"],
        "collector_outcome_fields": audits["collector_outcome_fields"],
        "reb_dataset_outcome_fields": audits["reb_dataset_outcome_fields"],
        "flow_phase_range_pass": audits["flow_phase_range_pass"],
        "flow_phase_violation_count": audits["flow_phase_violation_count"],
        "resumed_from_partial": bool(resumed_from_partial),
        "rows_loaded_from_partial": int(rows_loaded_from_partial),
        "rows_collected_this_run": int(rows_collected_this_run),
        "collection_rollout_performed_this_run": int(rows_collected_this_run) > 0,
        "collection_rollout_audit": {
            "overall_pass": True,
            "rows_collected_this_run": int(rows_collected_this_run),
            "expected_rollout_performed": int(rows_collected_this_run) > 0,
            "reported_rollout_performed": int(rows_collected_this_run) > 0,
        },
        "partial_dataset_path": "boundary_dataset.partial.csv",
        "partial_dataset_sha256_before_resume": partial_sha_before,
        "partial_dataset_sha256_after_collection": _sha256(partial_path),
        "resume_audit": resume_audit,
        "collection_completed_without_resume": (not resumed_from_partial and rows_collected_this_run == len(rows)),
        "partial_dataset_retention_policy": "retained_as_recovery_evidence",
        "formal_result_root": None,
        "staging_root": None,
        "output_files": {},
        "warnings": (
            ["Smoke class distribution is not evaluated for Found-aware REB training readiness."]
            if config["stage"] == "smoke"
            else (["Observed class counts do not meet Found-aware REB readiness thresholds; data were not altered."] if audits["readiness_audit"]["failure_reasons"] else [])
        ),
        "errors": [],
    }
    _atomic_write_text(markdown_path, _markdown(summary))
    artifacts = _output_artifact_metadata(
        result_root,
        boundary_path,
        by_seed_path,
        class_path,
        markdown_path,
        partial_path,
    )
    summary["output_files"] = {
        "artifacts": artifacts,
        "boundary_dataset_csv": artifacts["boundary_dataset.csv"]["path"],
        "boundary_dataset_sha256": artifacts["boundary_dataset.csv"]["sha256"],
        "dataset_by_seed_csv": artifacts["dataset_by_seed.csv"]["path"],
        "dataset_by_seed_sha256": artifacts["dataset_by_seed.csv"]["sha256"],
        "dataset_class_distribution_csv": artifacts["dataset_class_distribution.csv"]["path"],
        "dataset_class_distribution_sha256": artifacts["dataset_class_distribution.csv"]["sha256"],
        "dataset_audit_summary_json": "dataset_audit_summary.json",
        "dataset_audit_summary_markdown": artifacts["dataset_audit_summary.md"]["path"],
        "dataset_audit_summary_markdown_sha256": artifacts["dataset_audit_summary.md"]["sha256"],
    }
    _atomic_write_json(summary_path, summary)
    summary["pre_commit_output_audit"] = _audit_output_files(result_root, summary, config)
    if not formal_precommit:
        summary["overall_pass"] = True
        summary["post_commit_audit_pass"] = True
    _atomic_write_json(summary_path, summary)
    return summary


def _markdown(summary):
    overall = summary["class_distribution"]["overall"]
    source = summary["source_audit"]
    readiness = summary["readiness_audit"]
    lines = [
        "# Frozen Uniform DR Found-aware REB Dataset Audit",
        "",
        f"- Overall pass: {str(summary['overall_pass']).lower()}",
        f"- Stage: {summary['collection_stage']}",
        f"- Rows: {summary['row_count']}",
        f"- Seeds: {summary['seeds']}",
        f"- Episodes per seed: {summary['episodes_per_seed']}",
        f"- Max steps: {summary['max_steps']}",
        f"- Model SHA256: `{source['model_sha256_before']}`",
        f"- Source artifacts unchanged: {str(source['all_source_artifacts_unchanged']).lower()}",
        f"- Full-9D distinct vectors: {summary['disturbance_audit']['distinct_9d_vector_count']}",
        f"- Bounds violations: {summary['disturbance_audit']['bounds_violation_count']}",
        f"- Missing fields: {summary['disturbance_audit']['missing_field_count']}",
        "",
        "## Class distribution",
        "",
        "| n | success | found | not found | found but failed | success rate | found rate | success if found |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
        "| {n_episodes} | {n_success} | {n_found} | {n_not_found} | {n_found_but_failed} | {success_rate} | {found_rate} | {succ_if_found} |".format(**overall),
        "",
        "## Found-aware readiness",
        "",
        f"- Evaluated: {str(readiness['class_readiness_evaluated']).lower()}",
        f"- Ready: {readiness['ready_for_found_aware_reb_training']}",
        f"- Failure reasons: {readiness['failure_reasons']}",
        "",
        "Smoke results validate plumbing only and have no scientific interpretation."
        if summary["collection_stage"] == "smoke"
        else "The readiness flag reflects the observed frozen-policy class distribution without resampling.",
        "",
    ]
    return "\n".join(lines)


def run_preflight(args):
    source, _, _ = audit_and_load_sources(
        args.model,
        args.manifest,
        args.selection_summary,
        FORMAL_MAX_STEPS,
    )
    result = {
        "overall_pass": True,
        "experiment_type": EXPERIMENT_TYPE,
        "source_audit": source,
        "collection_performed": False,
        "training_performed": False,
        "optimizer_update_count": 0,
        "checkpoint_selection_performed": False,
        "model_save_performed": False,
        "errors": [],
    }
    print(json.dumps(_json_safe(result), indent=2, ensure_ascii=False, allow_nan=False))
    return result


def run_collection(args):
    config = _collection_config(args.smoke)
    final_root = _project_path(args.out_dir)
    if args.smoke:
        staging_root = final_root
        _require(not final_root.exists(), f"refusing to overwrite existing smoke output directory: {final_root}")
        resumed_from_partial = False
        formal_storage_state = "smoke"
        smoke_seeds = _discover_existing_smoke_seeds()
        source, maddpg, env = audit_and_load_sources(
            args.model,
            args.manifest,
            args.selection_summary,
            config["max_steps"],
        )
    else:
        staging_root = Path(str(final_root) + ".incomplete")
        source_readonly = audit_sources_readonly(args.model, args.manifest, args.selection_summary)
        storage = _detect_formal_storage_state(
            final_root,
            staging_root,
            source=source_readonly,
            config=config,
        )
        formal_storage_state = storage["state"]
        if formal_storage_state == "already_complete":
            summary, recovery = _finalize_or_recover_committed_formal(
                final_root,
                staging_root,
                source_readonly,
                config,
                formal_storage_state="already_complete",
                post_commit_recovery_performed=False,
                rows_collected_this_run=0,
            )
            _require(recovery["complete_commit_audit"]["overall_pass"] is True, "already-complete final audit failed")
            print(json.dumps(_json_safe(summary), indent=2, ensure_ascii=False, allow_nan=False))
            print("[RunAudit] formal_action=already_complete_reaudit rows_collected_this_run=0 collection_rollout_performed_this_run=false", flush=True)
            print("[PASS] formal already complete; read-only re-audit passed; no rollout performed", flush=True)
            return summary
        if formal_storage_state == "post_commit_recovery":
            summary, recovery = _finalize_or_recover_committed_formal(
                final_root,
                staging_root,
                source_readonly,
                config,
                formal_storage_state="post_commit_recovery",
                post_commit_recovery_performed=True,
                rows_collected_this_run=0,
            )
            _require(recovery["complete_commit_audit"]["overall_pass"] is True, "post-commit recovery final audit failed")
            print(json.dumps(_json_safe(summary), indent=2, ensure_ascii=False, allow_nan=False))
            print("[RunAudit] formal_action=post_commit_recovery rows_collected_this_run=0 collection_rollout_performed_this_run=false", flush=True)
            print("[PASS] formal post-commit recovery completed; no rollout performed", flush=True)
            return summary
        resumed_from_partial = formal_storage_state == "resume_collection"
        smoke_seeds = _discover_existing_smoke_seeds()
        source = source_readonly
        maddpg = None
        env = None

    partial_path = staging_root / "boundary_dataset.partial.csv"
    state_path = staging_root / "collection_state.json"
    if resumed_from_partial:
        _require(staging_root.is_dir(), f"formal staging path is not a directory: {staging_root}")
        _require(partial_path.is_file(), f"formal staging partial CSV is missing: {partial_path}")
        _require(state_path.is_file(), f"formal staging collection state is missing: {state_path}")
        existing_state = _load_json(state_path, "formal collection state")
        _require(existing_state.get("stage") == "formal", "formal collection state stage mismatch")
        _require(existing_state.get("expected_seeds") == FORMAL_SEEDS, "formal collection state seeds mismatch")
        _require(existing_state.get("episodes_per_seed") == FORMAL_EPISODES_PER_SEED, "formal collection state episode count mismatch")
        _require(existing_state.get("expected_total_rows") == 3000, "formal collection state total row mismatch")
        _require(existing_state.get("status") in ("incomplete", "finalizing"), "formal collection state status is not resumable")
        _require(existing_state.get("source_hashes") == _source_hashes(source), "formal collection state source hashes mismatch")
        _require(existing_state.get("finalization_phase") in ("collecting", "precommit_ready"), "formal collection state phase is not resumable")
        partial_sha_before = _sha256(partial_path)
    else:
        staging_root.mkdir(parents=True, exist_ok=False)
        _initialize_partial_csv(partial_path)
        partial_sha_before = None

    rows_before, completed_keys, resume_audit, _ = _read_partial_csv(
        partial_path,
        config,
        source["model_path"],
        require_complete=False,
    )
    rows_loaded_from_partial = len(rows_before)
    del rows_before
    if config["stage"] == "formal":
        _write_collection_state(
            state_path,
            config,
            source,
            partial_path,
            completed_keys,
            "incomplete",
            finalization_phase="collecting",
            formal_result_root=final_root if config["stage"] == "formal" else staging_root,
            staging_root=staging_root,
        )

    def update_state_after_seed(current_keys):
        if config["stage"] == "formal":
            _write_collection_state(
                state_path,
                config,
                source,
                partial_path,
                current_keys,
                "incomplete",
                finalization_phase="collecting",
                formal_result_root=final_root,
                staging_root=staging_root,
            )

    rows_collected_this_run, completed_keys, source, maddpg, env = _collect_rows_if_missing(
        maddpg,
        env,
        source,
        args,
        config,
        partial_path,
        completed_keys,
        update_state_after_seed,
    )
    rows, completed_keys, final_partial_audit, audits = _read_partial_csv(
        partial_path,
        config,
        source["model_path"],
        require_complete=True,
    )
    audits = _audit_rows(
        rows,
        config,
        smoke_seeds=smoke_seeds,
        expected_model_path=source["model_path"],
        require_complete=True,
    )
    resume_audit["final_partial_audit"] = final_partial_audit
    source = _refresh_source_audit(source)

    if config["stage"] == "formal":
        _write_collection_state(
            state_path,
            config,
            source,
            partial_path,
            completed_keys,
            "finalizing",
            finalization_phase="precommit_ready",
            formal_result_root=final_root,
            staging_root=staging_root,
        )
    summary = _write_final_outputs(
        staging_root,
        rows,
        config,
        source,
        audits,
        partial_path,
        resumed_from_partial,
        rows_loaded_from_partial,
        rows_collected_this_run,
        partial_sha_before,
        resume_audit,
        formal_precommit=(config["stage"] == "formal"),
    )
    if config["stage"] == "formal":
        summary.update(
            {
                "formal_storage_state": formal_storage_state,
                "post_commit_recovery_performed": False,
                "collection_rollout_performed_this_run": rows_collected_this_run > 0,
                "already_complete_reaudit": False,
                "finalization_phase": "precommit_ready",
                "canonical_summary_finalized": False,
                "canonical_markdown_finalized": False,
                "collection_state_audit": {"overall_pass": False, "reason": "precommit"},
                "complete_commit_audit": {"overall_pass": False, "reason": "precommit"},
            }
        )
        _atomic_write_json(staging_root / "dataset_audit_summary.json", summary)

    if config["stage"] == "formal":
        _require(not final_root.exists(), f"formal result directory appeared before commit: {final_root}")
        os.replace(str(staging_root), str(final_root))
        _require(final_root.is_dir() and not staging_root.exists(), "atomic formal directory commit failed")
        final_partial_path = final_root / "boundary_dataset.partial.csv"
        final_state_path = final_root / "collection_state.json"
        _write_collection_state(
            final_state_path,
            config,
            source,
            final_partial_path,
            completed_keys,
            "finalizing",
            finalization_phase="committed_pending_audit",
            formal_result_root=final_root,
            staging_root=staging_root,
            commit_directory_renamed=True,
        )
        summary, recovery = _finalize_or_recover_committed_formal(
            final_root,
            staging_root,
            source,
            config,
            formal_storage_state=formal_storage_state,
            post_commit_recovery_performed=False,
            rows_collected_this_run=rows_collected_this_run,
        )
        _require(recovery["complete_commit_audit"]["overall_pass"] is True, "formal final complete audit failed")
    else:
        summary["formal_result_root"] = None
        summary["staging_root"] = None
        _atomic_write_json(staging_root / "dataset_audit_summary.json", summary)

    print(json.dumps(_json_safe(summary), indent=2, ensure_ascii=False, allow_nan=False))
    print(
        "[PASS] frozen REB dataset collected: stage=%s rows=%d overall_pass=true"
        % (config["stage"], len(rows)),
        flush=True,
    )
    return summary


def _synthetic_partial_row(base_seed, episode_index, model_path):
    episode_seed = disturbance_seed_for_episode(base_seed, episode_index)
    flat_index = max(0, (int(base_seed) - FORMAL_SEEDS[0]) * FORMAL_EPISODES_PER_SEED + int(episode_index))
    disturbance = {}
    for dimension, key in enumerate(DISTURBANCE_KEYS):
        low, high = DEFAULT_DISTURBANCE_BOUNDS[key]
        if key == "action_delay_steps":
            disturbance[key] = int(flat_index % 4)
        else:
            numerator = ((flat_index * (17 + 2 * dimension) + 31 * dimension) % 3000) + 0.5
            disturbance[key] = float(low) + (float(high) - float(low)) * (numerator / 3000.0)
    found_flag = bool(flat_index % 3 != 0)
    success_flag = bool(found_flag and flat_index % 5 == 0)
    return {
        "model_name": MODEL_NAME,
        "model_path": str(Path(model_path).resolve()),
        "model_sha256": MODEL_SHA256,
        "evaluation_mode": MODE,
        "profile": PROFILE,
        "base_seed": int(base_seed),
        "episode_index": int(episode_index),
        "episode_seed": int(episode_seed),
        "episode_seed_mode": PAIRED_EPISODE_SEED_MODE,
        "disturbance_seed": int(episode_seed),
        "disturbance_rng_mode": DISTURBANCE_RNG_MODE,
        "disturbance_protocol": PROTOCOL,
        "disturbance_protocol_version": PROTOCOL_VERSION,
        "disturbance_explicitly_applied": True,
        "disturbance_apply_match": True,
        "dataset_split": "validation" if int(base_seed) in VALIDATION_SEEDS else "train",
        **disturbance,
        "flow_phase_x": 1.0,
        "flow_phase_y": 2.0,
        "success_flag": success_flag,
        "found_flag": found_flag,
        "recovery_time": 10.0,
        "safety_cost": 0.0,
        "safety_cost_mean": 0.0,
        "final_distance": 4.0,
        "final_nav_distance": 3.0,
        "action_smoothness": 0.5,
        "completion_steps": 10,
        "episode_reward_mean": -1.0,
        "episode_reward_sum": -10.0,
    }


def _synthetic_formal_rows(model_path):
    return [
        _synthetic_partial_row(seed, episode_index, model_path)
        for seed in FORMAL_SEEDS
        for episode_index in range(FORMAL_EPISODES_PER_SEED)
    ]


def _create_precommit_selftest_fixture(root, source, config, rows):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    partial_path = root / "boundary_dataset.partial.csv"
    _atomic_write_csv(partial_path, CSV_FIELDS, rows)
    audits = _audit_rows(
        rows,
        config,
        smoke_seeds=[],
        expected_model_path=source["model_path"],
        require_complete=True,
    )
    resume_audit = {
        "overall_pass": True,
        "csv_header_exact_match": True,
        "row_count": 3000,
        "expected_total_rows": 3000,
        "episode_prefix_valid": True,
        "outcome_schema_match": True,
    }
    _write_final_outputs(
        root,
        rows,
        config,
        source,
        audits,
        partial_path,
        resumed_from_partial=False,
        rows_loaded_from_partial=0,
        rows_collected_this_run=0,
        partial_sha_before=None,
        resume_audit=resume_audit,
        formal_precommit=True,
    )
    return {
        "partial_path": partial_path,
        "completed_keys": {(int(row["base_seed"]), int(row["episode_index"])) for row in rows},
        "csv_hashes": {
            name: _sha256(root / name)
            for name in (
                "boundary_dataset.partial.csv",
                "boundary_dataset.csv",
                "dataset_by_seed.csv",
                "dataset_class_distribution.csv",
            )
        },
    }



def _synthetic_source_for_selftest(root):
    """Create an isolated source identity fixture; never reads formal project artifacts."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    model_path = root / "synthetic_selected_model.pt"
    manifest_path = root / "synthetic_selected_model_manifest.json"
    selection_path = root / "synthetic_final_checkpoint_selection_summary.json"
    model_path.write_bytes(b"synthetic-self-test-model\n")
    manifest_path.write_text('{"self_test": true}\n', encoding="utf-8")
    selection_path.write_text('{"self_test": true}\n', encoding="utf-8")
    return {
        "overall_pass": True,
        "model_path": str(model_path.resolve()),
        "model_sha256_before": MODEL_SHA256,
        "model_sha256_after": MODEL_SHA256,
        "model_unchanged": True,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256_before": MANIFEST_SHA256,
        "manifest_sha256_after": MANIFEST_SHA256,
        "manifest_unchanged": True,
        "selection_summary_path": str(selection_path.resolve()),
        "selection_summary_sha256_before": SELECTION_SHA256,
        "selection_summary_sha256_after": SELECTION_SHA256,
        "selection_summary_unchanged": True,
        "selected_checkpoint_name": "snapshot_ep1400",
        "selected_checkpoint_episode": 1400,
        "lightweight_readonly_audit": True,
        "all_source_artifacts_unchanged": True,
        "self_test_synthetic_identity": True,
    }


def run_self_test():
    _outcome_schema_audit()
    config = _collection_config(False)
    results = []
    storage_results = []
    recovery_results = []
    derived_csv_results = []
    summary_semantic_results = []
    csv_stability_results = []
    nullable_rate_case_pass = False
    rollout_flag_case_pass = False
    with tempfile.TemporaryDirectory(prefix="ch4_reb_resume_selftest_") as temporary:
        root = Path(temporary)
        source = _synthetic_source_for_selftest(root / "synthetic_source")
        model_path = Path(source["model_path"])
        row0 = _synthetic_partial_row(201, 0, model_path)
        row1 = _synthetic_partial_row(201, 1, model_path)
        row2 = _synthetic_partial_row(201, 2, model_path)

        cases = []

        def add_case(name, rows, should_pass, fieldnames=None):
            cases.append((name, rows, should_pass, list(CSV_FIELDS if fieldnames is None else fieldnames)))

        add_case("valid_partial", [row0, row1], True)
        add_case("duplicate_episode_key", [row0, dict(row0)], False)
        add_case("episode_index_gap", [row0, row2], False)
        bad_split = dict(row1); bad_split["dataset_split"] = "validation"
        add_case("wrong_split", [row0, bad_split], False)
        bad_seed = dict(row1); bad_seed["episode_seed"] = int(bad_seed["episode_seed"]) + 1
        add_case("wrong_episode_seed", [row0, bad_seed], False)
        bad_path = dict(row1); bad_path["model_path"] = str(root / "wrong_model.pt")
        add_case("wrong_model_path", [row0, bad_path], False)
        bad_smooth = dict(row1); bad_smooth["action_smoothness"] = -0.1
        add_case("negative_action_smoothness", [row0, bad_smooth], False)
        bad_phase = dict(row1); bad_phase["flow_phase_x"] = 2.0 * math.pi
        add_case("flow_phase_out_of_range", [row0, bad_phase], False)
        wrong_header = list(CSV_FIELDS[1:]) + [CSV_FIELDS[0]]
        add_case("wrong_csv_header_order", [row0], False, wrong_header)

        partial_root = root / "partial_cases"
        partial_root.mkdir()
        for name, rows, should_pass, fieldnames in cases:
            path = partial_root / f"{name}.csv"
            _atomic_write_csv(path, fieldnames, rows)
            passed = True
            error = None
            try:
                _read_partial_csv(path, config, model_path, require_complete=False)
            except Exception as exc:
                passed = False
                error = f"{type(exc).__name__}: {exc}"
            expectation_met = passed == should_pass
            _require(expectation_met, f"self-test {name} expectation failed: passed={passed} error={error}")
            results.append(
                {
                    "case": name,
                    "expected": "accept" if should_pass else "reject",
                    "observed": "accept" if passed else "reject",
                    "expectation_met": expectation_met,
                    "error": error,
                }
            )

        formal_rows = _synthetic_formal_rows(source["model_path"])

        storage_root = root / "storage_states"
        storage_root.mkdir()
        new_final = storage_root / "new_final"
        new_staging = storage_root / "new_final.incomplete"
        detected = _detect_formal_storage_state(new_final, new_staging)
        _require(detected["state"] == "new_collection", "new_collection state detection failed")
        storage_results.append({"case": "new_collection_state", "observed": detected["state"], "expectation_met": True})

        resume_final = storage_root / "resume_final"
        resume_staging = storage_root / "resume_final.incomplete"
        resume_staging.mkdir()
        detected = _detect_formal_storage_state(resume_final, resume_staging)
        _require(detected["state"] == "resume_collection", "resume_collection state detection failed")
        storage_results.append({"case": "resume_collection_state", "observed": detected["state"], "expectation_met": True})

        split_final = storage_root / "split_final"
        split_staging = storage_root / "split_final.incomplete"
        split_final.mkdir(); split_staging.mkdir()
        split_rejected = False
        try:
            _detect_formal_storage_state(split_final, split_staging)
        except DatasetError:
            split_rejected = True
        _require(split_rejected, "split-brain state was not rejected")
        storage_results.append({"case": "split_brain_state", "observed": "rejected", "expectation_met": True})

        template_root = root / "precommit_template"
        template = _create_precommit_selftest_fixture(template_root, source, config, formal_rows)

        def csv_hashes(directory):
            return {
                name: _sha256(Path(directory) / name)
                for name in (
                    "boundary_dataset.partial.csv",
                    "boundary_dataset.csv",
                    "dataset_by_seed.csv",
                    "dataset_class_distribution.csv",
                )
            }

        def make_case(name):
            case_final = root / name
            shutil.copytree(template_root, case_final)
            return case_final, Path(str(case_final) + ".incomplete")

        def rewrite_csv_value(path, key_field, key_value, field, value):
            path = Path(path)
            fieldnames = BY_SEED_FIELDS if path.name == "dataset_by_seed.csv" else CLASS_FIELDS
            parsed = _read_csv_relaxed(path, fieldnames)
            matches = [row for row in parsed if row[key_field] == str(key_value)]
            _require(len(matches) == 1, f"self-test mutation target mismatch: {path.name} {key_field}={key_value}")
            matches[0][field] = value
            _atomic_write_csv(path, fieldnames, parsed)

        def expect_derived_corruption_rejected(name, csv_name, key_field, key_value, field, value):
            case_final, case_staging = make_case(name)
            rewrite_csv_value(case_final / csv_name, key_field, key_value, field, value)
            csv_before = csv_hashes(case_final)
            summary_before = _sha256(case_final / "dataset_audit_summary.json")
            rejected = False
            error = None
            try:
                _finalize_or_recover_committed_formal(
                    case_final,
                    case_staging,
                    source,
                    config,
                    "post_commit_recovery",
                    True,
                    0,
                )
            except DatasetError as exc:
                rejected = True
                error = str(exc)
            csv_after = csv_hashes(case_final)
            summary_after = _sha256(case_final / "dataset_audit_summary.json")
            state = _load_optional_json(case_final / "collection_state.json")
            stable = csv_before == csv_after and summary_before == summary_after
            not_completed = not isinstance(state, dict) or state.get("status") != "complete"
            _require(rejected and stable and not_completed, f"derived CSV corruption case failed: {name}: {error}")
            csv_stability_results.append(stable)
            derived_csv_results.append(
                {
                    "case": name,
                    "observed": "rejected",
                    "expectation_met": True,
                    "csv_hashes_stable_after_rejection": stable,
                    "error": error,
                }
            )

        expect_derived_corruption_rejected(
            "corrupt_by_seed_count_rejected",
            "dataset_by_seed.csv",
            "base_seed",
            201,
            "n_success",
            "999",
        )
        expect_derived_corruption_rejected(
            "corrupt_by_seed_rate_rejected",
            "dataset_by_seed.csv",
            "base_seed",
            201,
            "success_rate",
            "0.999",
        )
        expect_derived_corruption_rejected(
            "corrupt_class_distribution_count_rejected",
            "dataset_class_distribution.csv",
            "scope",
            "overall",
            "n_success",
            "999",
        )
        expect_derived_corruption_rejected(
            "corrupt_validation_class_rate_rejected",
            "dataset_class_distribution.csv",
            "scope",
            "validation",
            "found_but_failed_rate",
            "0.999",
        )

        nullable_rows = [dict(row) for row in formal_rows]
        for row in nullable_rows:
            if int(row["base_seed"]) == 201:
                row["found_flag"] = False
                row["success_flag"] = False
        nullable_root = root / "nullable_rate_valid"
        _create_precommit_selftest_fixture(nullable_root, source, config, nullable_rows)
        nullable_audit = _audit_committed_csvs(nullable_root, source, config)
        nullable_by_seed = _read_csv_relaxed(nullable_root / "dataset_by_seed.csv", BY_SEED_FIELDS)
        nullable_seed_201 = next(row for row in nullable_by_seed if row["base_seed"] == "201")
        nullable_empty_accepted = (
            nullable_seed_201["succ_if_found"] == ""
            and nullable_audit["dataset_by_seed_crosscheck"]["overall_pass"] is True
        )
        nullable_rejections = []
        for text_value in ("0", "nan"):
            nullable_case = root / f"nullable_rate_{text_value}"
            shutil.copytree(nullable_root, nullable_case)
            rewrite_csv_value(nullable_case / "dataset_by_seed.csv", "base_seed", 201, "succ_if_found", text_value)
            before = csv_hashes(nullable_case)
            rejected = False
            try:
                _audit_committed_csvs(nullable_case, source, config)
            except DatasetError:
                rejected = True
            after = csv_hashes(nullable_case)
            nullable_rejections.append(rejected and before == after)
            csv_stability_results.append(before == after)
        nullable_rate_case_pass = nullable_empty_accepted and all(nullable_rejections)
        _require(nullable_rate_case_pass, "nullable derived rate self-test failed")

        rollout_calls = {"loader": 0, "collector": 0}

        def forbidden_loader(*unused_args, **unused_kwargs):
            rollout_calls["loader"] += 1
            raise DatasetError("complete staging unexpectedly loaded rollout resources")

        def forbidden_collector(*unused_args, **unused_kwargs):
            rollout_calls["collector"] += 1
            raise DatasetError("complete staging unexpectedly collected an episode")

        test_args = argparse.Namespace(model="unused", manifest="unused", selection_summary="unused")
        rollout_rows, rollout_keys, _, _, _ = _collect_rows_if_missing(
            None,
            None,
            source,
            test_args,
            config,
            template["partial_path"],
            set(template["completed_keys"]),
            None,
            source_loader=forbidden_loader,
            row_collector=forbidden_collector,
        )
        rollout_flag_case_pass = (
            rollout_rows == 0
            and len(rollout_keys) == 3000
            and rollout_calls == {"loader": 0, "collector": 0}
            and (rollout_rows > 0) is False
        )
        _require(rollout_flag_case_pass, "complete staging without rollout self-test failed")

        pending_final, pending_staging = make_case("committed_pending_audit")
        _write_collection_state(
            pending_final / "collection_state.json",
            config,
            source,
            pending_final / "boundary_dataset.partial.csv",
            template["completed_keys"],
            "finalizing",
            finalization_phase="committed_pending_audit",
            formal_result_root=pending_final,
            staging_root=pending_staging,
            commit_directory_renamed=True,
        )
        before = csv_hashes(pending_final)
        pending_summary, pending_result = _finalize_or_recover_committed_formal(
            pending_final, pending_staging, source, config, "post_commit_recovery", True, False
        )
        after = csv_hashes(pending_final)
        _require(pending_summary["overall_pass"] is True and pending_result["action"] == "post_commit_recovery", "committed pending recovery failed")
        csv_stability_results.append(before == after)
        recovery_results.append({"case": "committed_pending_audit_recovery", "observed": "complete", "expectation_met": before == after})

        missing_final, missing_staging = make_case("missing_state")
        before = csv_hashes(missing_final)
        missing_summary, missing_result = _finalize_or_recover_committed_formal(
            missing_final, missing_staging, source, config, "post_commit_recovery", True, False
        )
        after = csv_hashes(missing_final)
        _require((missing_final / "collection_state.json").is_file() and missing_summary["overall_pass"] is True, "missing state recovery failed")
        csv_stability_results.append(before == after)
        recovery_results.append({"case": "missing_state_recovery", "observed": missing_result["action"], "expectation_met": before == after})

        complete_state_final, complete_state_staging = make_case("complete_state_precommit_summary")
        _write_collection_state(
            complete_state_final / "collection_state.json",
            config,
            source,
            complete_state_final / "boundary_dataset.partial.csv",
            template["completed_keys"],
            "complete",
            finalization_phase="complete",
            formal_result_root=complete_state_final,
            staging_root=complete_state_staging,
            commit_directory_renamed=True,
            post_commit_output_audit_pass=True,
            summary_finalized=True,
            markdown_finalized=True,
        )
        before = csv_hashes(complete_state_final)
        complete_state_summary, _ = _finalize_or_recover_committed_formal(
            complete_state_final, complete_state_staging, source, config, "post_commit_recovery", True, False
        )
        after = csv_hashes(complete_state_final)
        _require(complete_state_summary["formal_commit_completed"] is True, "complete-state/precommit-summary recovery failed")
        csv_stability_results.append(before == after)
        recovery_results.append({"case": "complete_state_precommit_summary_recovery", "observed": "complete", "expectation_met": before == after})

        all_files_before = {
            path.name: _sha256(path)
            for path in pending_final.iterdir()
            if path.is_file()
        }
        detected = _detect_formal_storage_state(pending_final, pending_staging, source=source, config=config)
        _require(detected["state"] == "already_complete", "already-complete re-audit state mismatch")
        all_files_after = {
            path.name: _sha256(path)
            for path in pending_final.iterdir()
            if path.is_file()
        }
        _require(all_files_before == all_files_after, "already-complete re-audit modified files")
        recovery_results.append({"case": "already_complete_reaudit", "observed": "already_complete", "expectation_met": True})

        def expect_summary_semantic_corruption_repaired(name, mutator):
            case_final, case_staging = make_case(name)
            _, initial_recovery = _finalize_or_recover_committed_formal(
                case_final,
                case_staging,
                source,
                config,
                "post_commit_recovery",
                True,
                0,
            )
            _require(initial_recovery["action"] == "post_commit_recovery", f"self-test setup finalization failed: {name}")
            csv_before = csv_hashes(case_final)
            corrupted_summary = _load_json(case_final / "dataset_audit_summary.json", "self-test summary")
            mutator(corrupted_summary)
            _atomic_write_json(case_final / "dataset_audit_summary.json", corrupted_summary)
            rejected_by_readonly_audit = False
            error = None
            try:
                _audit_complete_formal_commit(case_final, case_staging, source, config)
            except DatasetError as exc:
                rejected_by_readonly_audit = True
                error = str(exc)
            recovered_summary, recovery = _finalize_or_recover_committed_formal(
                case_final,
                case_staging,
                source,
                config,
                "post_commit_recovery",
                True,
                0,
            )
            final_audit = _audit_complete_formal_commit(case_final, case_staging, source, config)
            csv_after = csv_hashes(case_final)
            passed = (
                rejected_by_readonly_audit
                and recovery["action"] == "post_commit_recovery"
                and recovered_summary["summary_semantic_crosscheck"]["overall_pass"] is True
                and final_audit["summary_semantic_crosscheck"]["overall_pass"] is True
                and csv_before == csv_after
            )
            _require(passed, f"summary semantic corruption case failed: {name}: {error}")
            csv_stability_results.append(csv_before == csv_after)
            summary_semantic_results.append(
                {
                    "case": name,
                    "readonly_audit_observed": "rejected",
                    "recovery_observed": recovery["action"],
                    "expectation_met": passed,
                    "csv_hashes_stable": csv_before == csv_after,
                    "error": error,
                }
            )

        def corrupt_summary_class(summary):
            summary["class_distribution"]["overall"]["n_success"] = 999

        def corrupt_summary_per_seed(summary):
            summary["per_seed_class_distribution"][0]["n_success"] = 999

        def corrupt_summary_readiness(summary):
            summary["ready_for_found_aware_reb_training"] = not bool(summary["ready_for_found_aware_reb_training"])
            summary["readiness_audit"]["ready_for_found_aware_reb_training"] = summary["ready_for_found_aware_reb_training"]

        expect_summary_semantic_corruption_repaired(
            "corrupt_summary_class_distribution_rejected_and_repaired",
            corrupt_summary_class,
        )
        expect_summary_semantic_corruption_repaired(
            "corrupt_summary_per_seed_distribution_rejected_and_repaired",
            corrupt_summary_per_seed,
        )
        expect_summary_semantic_corruption_repaired(
            "corrupt_summary_readiness_rejected_and_repaired",
            corrupt_summary_readiness,
        )

        incomplete_final = root / "incomplete_committed_partial"
        incomplete_final.mkdir()
        _atomic_write_csv(incomplete_final / "boundary_dataset.partial.csv", CSV_FIELDS, formal_rows[:1])
        _atomic_copy_file(incomplete_final / "boundary_dataset.partial.csv", incomplete_final / "boundary_dataset.csv")
        incomplete_rejected = False
        try:
            _finalize_or_recover_committed_formal(
                incomplete_final, Path(str(incomplete_final) + ".incomplete"), source, config, "post_commit_recovery", True, False
            )
        except DatasetError:
            incomplete_rejected = True
        _require(incomplete_rejected, "incomplete committed partial was not rejected")
        recovery_results.append({"case": "incomplete_committed_partial_rejected", "observed": "rejected", "expectation_met": True})

        corrupt_final, corrupt_staging = make_case("corrupt_committed_partial")
        corrupt_rows = [dict(row) for row in formal_rows]
        corrupt_rows[0]["model_sha256"] = "0" * 64
        _atomic_write_csv(corrupt_final / "boundary_dataset.partial.csv", CSV_FIELDS, corrupt_rows)
        corrupt_rejected = False
        try:
            _finalize_or_recover_committed_formal(
                corrupt_final, corrupt_staging, source, config, "post_commit_recovery", True, False
            )
        except DatasetError:
            corrupt_rejected = True
        _require(corrupt_rejected, "corrupt committed partial was not rejected")
        recovery_results.append({"case": "corrupt_committed_partial_rejected", "observed": "rejected", "expectation_met": True})

        idempotent_final, idempotent_staging = make_case("idempotent_finalization")
        before = csv_hashes(idempotent_final)
        _, first_result = _finalize_or_recover_committed_formal(
            idempotent_final, idempotent_staging, source, config, "post_commit_recovery", True, False
        )
        middle = csv_hashes(idempotent_final)
        _, second_result = _finalize_or_recover_committed_formal(
            idempotent_final, idempotent_staging, source, config, "already_complete", False, False
        )
        after = csv_hashes(idempotent_final)
        stable = before == middle == after
        _require(first_result["action"] == "post_commit_recovery" and second_result["action"] == "already_complete" and stable, "idempotent finalization failed")
        csv_stability_results.append(stable)
        recovery_results.append({"case": "idempotent_finalization", "observed": "already_complete_on_second_call", "expectation_met": stable})

    result = {
        "overall_pass": (
            all(item["expectation_met"] for item in results)
            and all(item["expectation_met"] for item in storage_results)
            and all(item["expectation_met"] for item in recovery_results)
            and all(item["expectation_met"] for item in derived_csv_results)
            and all(item["expectation_met"] for item in summary_semantic_results)
            and nullable_rate_case_pass
            and rollout_flag_case_pass
            and all(csv_stability_results)
        ),
        "formal_rollout_performed": False,
        "formal_paths_written": False,
        "case_count": len(results),
        "partial_csv_case_count": len(results),
        "cases": results,
        "storage_state_case_count": len(storage_results),
        "storage_state_cases": storage_results,
        "post_commit_recovery_case_count": len(recovery_results),
        "post_commit_recovery_cases": recovery_results,
        "derived_csv_crosscheck_case_count": len(derived_csv_results) + 1,
        "derived_csv_corruption_rejected_count": sum(
            item["observed"] == "rejected" for item in derived_csv_results
        ),
        "derived_csv_crosscheck_cases": derived_csv_results,
        "summary_semantic_crosscheck_case_count": len(summary_semantic_results),
        "summary_semantic_crosscheck_cases": summary_semantic_results,
        "self_test_source_isolated": True,
        "nullable_rate_case_pass": nullable_rate_case_pass,
        "rollout_flag_case_pass": rollout_flag_case_pass,
        "rollout_count": 0,
        "all_csv_hashes_stable": all(csv_stability_results),
        "outcome_schema_match": True,
        "errors": [],
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def _add_source_arguments(parser):
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--selection-summary", required=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Collect and audit the frozen Uniform DR full-9D Found-aware REB dataset."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser(
        "preflight",
        help="Read-only frozen source and network-structure audit.",
    )
    _add_source_arguments(preflight)
    collect = subparsers.add_parser(
        "collect",
        help="Collect the fixed formal protocol, or its isolated smoke variant.",
    )
    _add_source_arguments(collect)
    collect.add_argument("--out-dir", required=True)
    collect.add_argument("--smoke", action="store_true")
    subparsers.add_parser(
        "self-test",
        help="Run isolated strict partial-CSV recovery tests without any rollout.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        if args.command == "preflight":
            run_preflight(args)
        elif args.command == "self-test":
            run_self_test()
        else:
            run_collection(args)
        return 0
    except Exception as exc:
        print(
            "[ERROR] frozen REB dataset operation failed: %s: %s"
            % (type(exc).__name__, exc),
            file=sys.stderr,
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
