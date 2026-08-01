#!/usr/bin/env python3
"""Audit an independent Uniform DR final sweep and freeze one checkpoint."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from registry.rbe_disturbance import DEFAULT_DISTURBANCE_BOUNDS, DISTURBANCE_KEYS
from registry.ch4_artifact_layout import (
    get_selected_dir,
    get_selection_dir,
    get_smoke_dir,
)


TRAINING_RUN = "ch4_uniform_dr_seed1_from_clean_ep8200_2000ep_v1"
EXPECTED_COARSE_EPISODES = list(range(100, 2001, 100))
COARSE_SEEDS = [101, 102]
SMOKE_COARSE_SEEDS = [907]
FORMAL_TEST_SEEDS = [1, 2, 3]
CLEAN_FINAL_VALIDATION_SEEDS = [103, 104, 105]
FORMAL_VALIDATION_SEEDS = [106, 107, 108]
SMOKE_VALIDATION_SEEDS = [906]
FORMAL_FORBIDDEN_SEEDS = [1, 2, 3, 101, 102, 103, 104, 105]
SMOKE_FORBIDDEN_SEEDS = FORMAL_FORBIDDEN_SEEDS + FORMAL_VALIDATION_SEEDS + SMOKE_COARSE_SEEDS
WARM_START_SOURCE_CHECKPOINT = "snapshot_ep8200"
WARM_START_SOURCE_SHA256 = (
    "c9807c948b50ca102d59de92801190e110eb2184c3976d6ef7d9c21a6e2f75d1"
)
PAIRED_SEED_MODE = "indexed_common_random_numbers"
PAIRED_SEED_FORMULA = "base_seed * 1000003 + episode_index"
DISTURBANCE_PROTOCOL = "uniform_9d_registry_v1"
DISTURBANCE_PROTOCOL_VERSION = 2
DISTURBANCE_RNG_MODE = "numpy.default_rng(episode_seed)"
FLOW_PHASE_KEYS = ("flow_phase_x", "flow_phase_y")
SUCCESS_MARGIN = 0.01
REL_TOL = 1e-7
ABS_TOL = 1e-9

CONTINUOUS_FIELDS = (
    "reward",
    "recovery_time",
    "safety_cost",
    "final_distance",
    "final_nav_distance",
    "action_smoothness",
    "completion_steps",
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
    "avg_action_smoothness": "avg_action_smoothness_weighted",
    "avg_completion_steps": "avg_completion_steps_weighted",
}
COUNT_FIELDS = (
    "total_episodes",
    "total_success",
    "total_found",
    "total_not_found",
    "total_found_but_failed",
)
CSV_FIELDS = (
    "coarse_rank",
    "checkpoint_name",
    "checkpoint_episode",
    "checkpoint_path",
    "success_rate",
    "found_rate",
    "succ_if_found",
    "avg_reward",
    "avg_recovery_time",
    "avg_safety_cost",
    "avg_final_distance",
    "avg_final_nav_distance",
    "avg_action_smoothness",
    "avg_completion_steps",
    "total_episodes",
    "total_success",
    "total_found",
    "passed_success_gate",
    "passed_found_gate",
    "passed_sif_gate",
    "passed_safety_gate",
    "passed_recovery_gate",
    "passed_distance_gate",
    "selected",
)


class SelectionError(RuntimeError):
    pass


def _require(condition, message):
    if not condition:
        raise SelectionError(message)


def _load_json(path, label):
    path = Path(path)
    _require(path.is_file(), "%s is missing: %s" % (label, path))
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError) as exc:
        raise SelectionError("%s is not readable JSON: %s" % (label, exc))
    _require(isinstance(data, dict), "%s must contain a JSON object" % label)
    return data


def _finite(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _as_int(value, label):
    _require(_finite(value), "%s must be a finite integer" % label)
    number = int(value)
    _require(float(value) == float(number), "%s must be an integer" % label)
    return number


def _close(left, right, rel_tol=REL_TOL, abs_tol=ABS_TOL):
    return math.isclose(
        float(left), float(right), rel_tol=rel_tol, abs_tol=abs_tol
    )


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _atomic_write_json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(
                _json_safe(data),
                handle,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
        os.replace(str(temporary), str(path))
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def _atomic_write_text(path, text, encoding="utf-8"):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(text, encoding=encoding)
        os.replace(str(temporary), str(path))
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def _checkpoint_episode(name):
    match = re.fullmatch(r"snapshot_ep(\d+)", str(name))
    _require(match is not None, "invalid checkpoint name: %s" % name)
    return int(match.group(1))


def _expected_episode_seed(base_seed, episode_index):
    return int(base_seed) * 1_000_003 + int(episode_index)


def _parse_csv_int(value, label):
    text = str(value).strip()
    _require(text, "%s must not be empty" % label)
    try:
        number = float(text)
    except ValueError:
        raise SelectionError("%s must be an integer, got %r" % (label, value))
    _require(
        math.isfinite(number) and number.is_integer(),
        "%s must be an integer, got %r" % (label, value),
    )
    return int(number)


def _parse_flag(value, label):
    if isinstance(value, bool):
        return int(value)
    text = str(value).strip().lower()
    if text in ("1", "1.0", "true", "yes"):
        return 1
    if text in ("0", "0.0", "false", "no"):
        return 0
    raise SelectionError("%s must be a boolean flag, got %r" % (label, value))


def _audit_disturbance_protocol_summary(audit, label, candidate_count, episode_count):
    _require(isinstance(audit, dict) and audit, "%s lacks the full-9D disturbance protocol audit." % label)
    checks = {
        "overall_pass": audit.get("overall_pass") is True,
        "protocol": audit.get("protocol") == DISTURBANCE_PROTOCOL,
        "protocol_version": audit.get("protocol_version") == DISTURBANCE_PROTOCOL_VERSION,
        "disturbance_keys": audit.get("disturbance_keys") == list(DISTURBANCE_KEYS),
        "disturbance_bounds": audit.get("disturbance_bounds") == {
            key: list(DEFAULT_DISTURBANCE_BOUNDS[key]) for key in DISTURBANCE_KEYS
        },
        "flow_phase_keys": audit.get("flow_phase_keys") == list(FLOW_PHASE_KEYS),
        "flow_phase_in_9d_vector": audit.get("flow_phase_in_9d_vector") is False,
        "explicit_application": audit.get("explicit_application") is True,
        "paired_episode_seeding": audit.get("paired_episode_seeding") is True,
        "candidate_count": audit.get("candidate_count") == candidate_count,
        "episode_count_per_checkpoint": audit.get("episode_count_per_checkpoint") == episode_count,
        "all_candidate_keys_match": audit.get("all_candidate_keys_match") is True,
        "all_candidate_episode_seeds_match": audit.get("all_candidate_episode_seeds_match") is True,
        "all_candidate_disturbance_seeds_match": audit.get("all_candidate_disturbance_seeds_match") is True,
        "all_candidate_disturbance_vectors_match": audit.get("all_candidate_disturbance_vectors_match") is True,
        "all_candidate_flow_phases_match": audit.get("all_candidate_flow_phases_match") is True,
        "all_episode_apply_match": audit.get("all_episode_apply_match") is True,
        "bounds_violation_count": audit.get("bounds_violation_count") == 0,
        "missing_field_count": audit.get("missing_field_count") == 0,
        "errors": audit.get("errors") == [],
    }
    failed = [name for name, passed in checks.items() if not passed]
    _require(not failed, "%s full-9D disturbance protocol audit failed: %s" % (label, failed))
    return True


def audit_coarse_candidate_derivation(summary, expected_candidates=None, smoke=False):
    _require(summary.get("overall_pass") is True, "coarse overall_pass must be true")
    _require(summary.get("selection_policy") == "legacy", "coarse selection_policy must be legacy")
    settings = summary.get("settings") or {}
    _require(settings.get("selection_policy") == "legacy", "coarse settings selection_policy must be legacy")
    _require(settings.get("ablation_mode") == "ch4_uniform_dr", "coarse mode must be ch4_uniform_dr")
    _require(settings.get("profile") == "normal_comm", "coarse profile must be normal_comm")
    coarse_seeds = SMOKE_COARSE_SEEDS if smoke else COARSE_SEEDS
    coarse_episodes = 2 if smoke else 50
    coarse_steps = 20 if smoke else 400
    _require(settings.get("validation_seeds") == coarse_seeds, "coarse validation seeds are invalid")
    _require(settings.get("episodes_per_seed") == coarse_episodes, "coarse episodes_per_seed is invalid")
    _require(settings.get("max_steps") == coarse_steps, "coarse max_steps is invalid")
    _require(settings.get("paired_episode_seeding") is True, "coarse paired seeding must be enabled")
    _require(settings.get("disturbance_protocol") == DISTURBANCE_PROTOCOL, "coarse disturbance protocol setting is invalid")
    _require(settings.get("require_paired_disturbances") is True, "coarse paired disturbance requirement is disabled")
    _require(summary.get("validation_seeds") == coarse_seeds, "coarse root validation seeds are invalid")
    forbidden = summary.get("forbidden_test_seeds") or []
    _require(set(FORMAL_TEST_SEEDS).issubset(set(forbidden)), "coarse forbidden seeds must include 1, 2, and 3")
    _require(summary.get("seed_leakage_detected") is False, "coarse summary reports seed leakage")
    _require(summary.get("candidate_count") == 20, "coarse candidate_count must be 20")
    _require(summary.get("errors") == [], "coarse errors must be empty")
    snapshot = summary.get("snapshot_audit") or {}
    _require(snapshot.get("complete_sequence") is True, "coarse snapshot sequence is incomplete")
    _require(snapshot.get("found_snapshot_episodes") == EXPECTED_COARSE_EPISODES, "coarse snapshot episodes are not 100..2000")
    _audit_disturbance_protocol_summary(
        summary.get("disturbance_protocol_audit"),
        "Legacy coarse summary" if not summary.get("disturbance_protocol_audit") else "coarse summary",
        20,
        len(coarse_seeds) * coarse_episodes,
    )
    rows = summary.get("ranked_checkpoints") or []
    _require(len(rows) == 20, "coarse ranked_checkpoints must contain 20 rows")
    row_by_name = {row.get("checkpoint_name"): row for row in rows}
    _require(len(row_by_name) == 20, "coarse ranked checkpoints contain duplicates")
    for row in rows:
        _require(_finite(row.get("success_rate_weighted")), "coarse success rate must be finite")
        name = row.get("checkpoint_name")
        _require(_checkpoint_episode(name) == _as_int(row.get("checkpoint_episode"), "%s checkpoint episode" % name), "%s coarse episode mismatch" % name)
    best_success = max(float(row["success_rate_weighted"]) for row in rows)
    threshold = best_success - SUCCESS_MARGIN
    raw_margin_candidates = sorted(
        [
            row["checkpoint_name"]
            for row in rows
            if float(row["success_rate_weighted"]) >= threshold - 1e-12
        ],
        key=_checkpoint_episode,
    )
    _require(raw_margin_candidates, "coarse success margin produced no candidates")
    best_names = {
        row["checkpoint_name"]
        for row in rows
        if math.isclose(
            float(row["success_rate_weighted"]),
            best_success,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    }
    _require(
        bool(best_names.intersection(raw_margin_candidates)),
        "raw margin candidates do not contain a best checkpoint",
    )
    minimum_candidate_count = 1 if smoke else 2
    _require(1 <= minimum_candidate_count <= len(rows) <= 20, "invalid mode-specific minimum candidate count")
    minimum_fill_candidates = []
    if len(raw_margin_candidates) < minimum_candidate_count:
        raw_set = set(raw_margin_candidates)
        fill_pool = sorted(
            [row for row in rows if row["checkpoint_name"] not in raw_set],
            key=lambda row: (
                -float(row["success_rate_weighted"]),
                _as_int(
                    row["checkpoint_episode"],
                    "%s checkpoint episode" % row["checkpoint_name"],
                ),
            ),
        )
        required_fill_count = minimum_candidate_count - len(raw_margin_candidates)
        _require(required_fill_count <= len(fill_pool), "minimum candidate fill exceeds the coarse pool")
        minimum_fill_candidates = [
            row["checkpoint_name"] for row in fill_pool[:required_fill_count]
        ]
    derived = sorted(
        raw_margin_candidates + minimum_fill_candidates,
        key=_checkpoint_episode,
    )
    _require(len(derived) == len(set(derived)), "derived coarse candidates contain duplicates")
    _require(set(derived).issubset(set(row_by_name)), "derived candidate is absent from coarse ranked_checkpoints")
    _require(
        len(derived) >= minimum_candidate_count,
        "minimum candidate fill did not reach the mode-specific minimum",
    )
    _require(
        minimum_candidate_count <= len(derived) <= 20,
        "coarse dynamic candidate_count must be between %d and 20" % minimum_candidate_count,
    )
    if expected_candidates is not None:
        _require(
            len(expected_candidates) == len(set(expected_candidates)),
            "--expected-candidates contains duplicates",
        )
        _require(
            set(expected_candidates).issubset(set(row_by_name)),
            "--expected-candidates contains a checkpoint absent from coarse ranked_checkpoints",
        )
        _require(derived == list(expected_candidates), "derived coarse candidates differ from --expected-candidates: %s" % derived)
    coarse_ranks = {
        row["checkpoint_name"]: _as_int(row.get("rank"), "%s coarse rank" % row["checkpoint_name"])
        for row in rows
    }
    return {
        "best_success": best_success,
        "success_margin": SUCCESS_MARGIN,
        "threshold": threshold,
        "raw_margin_candidates": raw_margin_candidates,
        "raw_margin_candidate_count": len(raw_margin_candidates),
        "minimum_candidate_count": minimum_candidate_count,
        "minimum_fill_applied": bool(minimum_fill_candidates),
        "minimum_fill_candidates": minimum_fill_candidates,
        "minimum_fill_candidate_count": len(minimum_fill_candidates),
        "minimum_fill_sort_rule": [
            "success_rate_weighted desc",
            "checkpoint_episode asc",
        ],
        "derived_candidates": derived,
        "expected_candidates": list(derived if expected_candidates is None else expected_candidates),
        "candidate_count": len(derived),
        "candidate_set_match": True,
        "legacy_top_k_ignored": True,
        "candidate_derivation_dynamic": True,
        "candidate_minimum_fill_policy": (
            "apply only when success-margin candidates are fewer than the mode-specific minimum"
        ),
    }, coarse_ranks


def derive_coarse_candidates_for_cli(coarse_summary, smoke=False):
    summary = _load_json(coarse_summary, "coarse summary")
    derivation, _coarse_ranks = audit_coarse_candidate_derivation(
        summary,
        expected_candidates=None,
        smoke=bool(smoke),
    )
    return " ".join(derivation["derived_candidates"])


def derive_coarse_candidates_json_for_cli(coarse_summary, smoke=False):
    summary = _load_json(coarse_summary, "coarse summary")
    derivation, _coarse_ranks = audit_coarse_candidate_derivation(
        summary,
        expected_candidates=None,
        smoke=bool(smoke),
    )
    return derivation


def _config_value(config, key):
    for container in (
        config,
        config.get("controller_config"),
        config.get("env_kwargs"),
    ):
        if isinstance(container, dict) and key in container:
            return container[key]
    return None


def audit_training_sources(config, audit):
    config_checks = {
        "run_name": config.get("run_name") == TRAINING_RUN,
        "mode": config.get("mode") == "ch4_uniform_dr",
        "seed": config.get("seed") == 1,
        "max_episodes": config.get("max_episodes") == 2000,
        "max_steps": config.get("max_steps") == 400,
        "snapshot_interval": config.get("snapshot_interval") == 100,
        "initialization_source": config.get("initialization_source") == "explicit_checkpoint",
        "loaded_checkpoint": config.get("loaded_checkpoint") is True,
        "train_from_scratch_requested": config.get("train_from_scratch_requested") is False,
        "collect_boundary_dataset": _config_value(config, "collect_boundary_dataset") is True,
        "reb_enable": _config_value(config, "reb_enable") is False,
        "rscu_enable": _config_value(config, "rscu_enable") is False,
        "rbe_sampler_enable": _config_value(config, "rbe_sampler_enable") is False,
        "freeze_search_actors": _config_value(config, "freeze_search_actors") is False,
        "rbe_executor_only_actor": _config_value(config, "rbe_executor_only_actor") is False,
        "use_robust_disturbance": _config_value(config, "use_robust_disturbance") is True,
    }
    failed_config = [key for key, passed in config_checks.items() if not passed]
    _require(not failed_config, "training config audit failed: %s" % failed_config)

    progress = audit.get("progress_audit") or {}
    boundary = audit.get("boundary_dataset_audit") or {}
    snapshot = audit.get("snapshot_audit") or {}
    controller = audit.get("controller_continuity_audit") or {}
    cross_file = audit.get("cross_file_episode_audit") or {}
    metric_crosscheck = audit.get("episode_metrics_boundary_crosscheck") or {}
    initialization = audit.get("initialization") or {}
    audit_checks = {
        "overall_pass": audit.get("overall_pass") is True,
        "experiment_type": audit.get("experiment_type") == "uniform_dr_warm_start_training",
        "training_stage": audit.get("training_stage") == "completed",
        "mode": audit.get("mode") == "ch4_uniform_dr",
        "training_seed": audit.get("training_seed") == 1,
        "ready_for_checkpoint_validation": audit.get("ready_for_checkpoint_validation") is True,
        "source_model_unchanged": audit.get("source_model_unchanged") is True,
        "source_artifacts_unchanged": audit.get("source_artifacts_unchanged") is True,
        "source_checkpoint": initialization.get("source_checkpoint") == WARM_START_SOURCE_CHECKPOINT,
        "source_hash_match": initialization.get("hash_match") is True,
        "controller_continuity": controller.get("controller_configuration_match") is True,
        "controller_mismatches": controller.get("mismatched_fields") == [],
        "progress_rows": progress.get("training_progress_row_count") == 2000,
        "progress_episode_sequence": progress.get("training_progress_episode_sequence_complete") is True,
        "actor_updates": int(progress.get("actor_update_count") or 0) > 0,
        "critic_updates": int(progress.get("critic_update_count") or 0) > 0,
        "cross_file_episode_sequence": cross_file.get("all_sequences_match") is True,
        "metrics_boundary_crosscheck": metric_crosscheck.get("all_fields_match") is True,
        "boundary_rows": boundary.get("row_count") == 2000,
        "boundary_bounds": boundary.get("bounds_violation_count") == 0,
        "boundary_varied": boundary.get("disturbance_varied") is True,
        "snapshot_sequence": snapshot.get("sequence_complete") is True,
        "snapshot_count": snapshot.get("count") == 20,
        "final_model": snapshot.get("final_model_exists") is True,
        "errors": audit.get("errors") == [],
    }
    failed_audit = [key for key, passed in audit_checks.items() if not passed]
    _require(not failed_audit, "training audit validation failed: %s" % failed_audit)
    return {
        "training_config_checks": config_checks,
        "training_audit_checks": audit_checks,
        "training_config_pass": True,
        "training_audit_pass": True,
    }


def validate_seed_policy(validation_seeds, forbidden_seeds):
    _require(len(validation_seeds) == len(set(validation_seeds)), "validation seeds contain duplicates")
    reserved = set(FORMAL_TEST_SEEDS + COARSE_SEEDS + CLEAN_FINAL_VALIDATION_SEEDS)
    overlap_reserved = sorted(set(validation_seeds).intersection(reserved))
    _require(not overlap_reserved, "validation seeds overlap reserved seeds: %s" % overlap_reserved)
    overlap_forbidden = sorted(set(validation_seeds).intersection(set(forbidden_seeds)))
    _require(not overlap_forbidden, "validation seeds overlap forbidden seeds: %s" % overlap_forbidden)
    return True


def _load_episode_csv(path, seed, expected_rows, checkpoint_name):
    path = Path(path)
    _require(path.is_file(), "episode_metrics.csv is missing: %s" % path)
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
            *DISTURBANCE_KEYS,
            *FLOW_PHASE_KEYS,
            *CONTINUOUS_FIELDS,
        }
        missing = sorted(required.difference(fields))
        _require(not missing, "%s is missing fields: %s" % (path, missing))
        _require("success_flag" in fields or "success" in fields, "%s is missing success flag" % path)
        _require("found_flag" in fields or "found" in fields, "%s is missing found flag" % path)
        for row in reader:
            base_seed = _parse_csv_int(row.get("base_seed"), "%s base_seed" % path)
            index = _parse_csv_int(row.get("episode_index"), "%s episode_index" % path)
            episode_seed = _parse_csv_int(row.get("episode_seed"), "%s episode_seed" % path)
            disturbance_seed = _parse_csv_int(row.get("disturbance_seed"), "%s disturbance_seed" % path)
            _require(base_seed == seed, "%s base_seed differs from expected seed" % path)
            _require(0 <= index < expected_rows, "%s episode index is out of range" % path)
            _require(episode_seed == _expected_episode_seed(base_seed, index), "%s episode seed formula mismatch" % path)
            _require(row.get("episode_seed_mode") == PAIRED_SEED_MODE, "%s episode seed mode mismatch" % path)
            _require(disturbance_seed == episode_seed, "%s disturbance seed differs from episode seed" % path)
            _require(row.get("disturbance_rng_mode") == DISTURBANCE_RNG_MODE, "%s disturbance RNG mode mismatch" % path)
            _require(row.get("disturbance_protocol") == DISTURBANCE_PROTOCOL, "%s disturbance protocol mismatch" % path)
            explicitly_applied = _parse_flag(
                row.get("disturbance_explicitly_applied"),
                "%s disturbance_explicitly_applied" % checkpoint_name,
            )
            disturbance_apply_match = _parse_flag(
                row.get("disturbance_apply_match"),
                "%s disturbance_apply_match" % checkpoint_name,
            )
            _require(explicitly_applied == 1, "%s disturbance was not explicitly applied" % path)
            _require(disturbance_apply_match == 1, "%s disturbance apply audit failed" % path)
            key = (seed, index)
            _require(key not in records, "%s contains duplicate episode key %r" % (path, key))
            success_value = row.get("success_flag") if "success_flag" in fields else row.get("success")
            found_value = row.get("found_flag") if "found_flag" in fields else row.get("found")
            success = _parse_flag(success_value, "%s success" % checkpoint_name)
            found = _parse_flag(found_value, "%s found" % checkpoint_name)
            _require(success <= found, "%s episode success cannot exceed found" % checkpoint_name)
            record = {
                "base_seed": base_seed,
                "episode_index": index,
                "episode_seed": episode_seed,
                "disturbance_seed": disturbance_seed,
                "disturbance_protocol": row.get("disturbance_protocol"),
                "disturbance_apply_match": disturbance_apply_match,
                "success": success,
                "found": found,
            }
            for field in (*DISTURBANCE_KEYS, *FLOW_PHASE_KEYS):
                if field == "action_delay_steps":
                    value = _parse_csv_int(row.get(field), "%s %s" % (path, field))
                else:
                    try:
                        value = float(row.get(field, ""))
                    except (TypeError, ValueError):
                        raise SelectionError("%s %s is not numeric" % (path, field))
                    _require(math.isfinite(value), "%s %s is not finite" % (path, field))
                if field in DISTURBANCE_KEYS:
                    low, high = DEFAULT_DISTURBANCE_BOUNDS[field]
                    _require(float(low) <= float(value) <= float(high), "%s %s is outside registry bounds" % (path, field))
                record[field] = value
            for field in CONTINUOUS_FIELDS:
                try:
                    value = float(row.get(field, ""))
                except (TypeError, ValueError):
                    raise SelectionError("%s %s is not numeric" % (path, field))
                _require(math.isfinite(value), "%s %s is not finite" % (path, field))
                record[field] = value
            records[key] = record
    _require(len(records) == expected_rows, "%s row count is %d, expected %d" % (path, len(records), expected_rows))
    _require({key[1] for key in records} == set(range(expected_rows)), "%s episode index set is incomplete" % path)
    return records


def _audit_metric_semantics(metric, allow_zero_found):
    name = metric["checkpoint_name"]
    episodes = _as_int(metric.get("total_episodes"), "%s total episodes" % name)
    success = _as_int(metric.get("total_success"), "%s total success" % name)
    found = _as_int(metric.get("total_found"), "%s total found" % name)
    not_found = _as_int(metric.get("total_not_found"), "%s total not found" % name)
    found_failed = _as_int(metric.get("total_found_but_failed"), "%s total found-but-failed" % name)
    _require(0 <= success <= found <= episodes, "%s counts violate success <= found <= episodes" % name)
    _require(not_found == episodes - found, "%s not-found count is inconsistent" % name)
    _require(found_failed == found - success, "%s found-but-failed count is inconsistent" % name)
    _require(_close(metric["success_rate"], success / float(episodes)), "%s success rate is inconsistent" % name)
    _require(_close(metric["found_rate"], found / float(episodes)), "%s found rate is inconsistent" % name)
    sif = metric.get("succ_if_found")
    if found:
        _require(_finite(sif) and _close(sif, success / float(found)), "%s SIF is inconsistent" % name)
    else:
        _require(success == 0 and sif is None, "%s zero-found SIF must be null" % name)
        _require(allow_zero_found, "%s has zero found episodes outside smoke" % name)
    for field in AGGREGATE_FIELDS:
        if field != "succ_if_found":
            _require(_finite(metric.get(field)), "%s %s must be finite" % (name, field))


def _recompute_episode_metrics(records):
    values = list(records.values())
    total = len(values)
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
    return result


def _audit_episode_aggregates(metric, records, expected_total):
    name = metric["checkpoint_name"]
    _require(len(records) == expected_total, "%s episode count mismatch" % name)
    recomputed = _recompute_episode_metrics(records)
    for field in COUNT_FIELDS:
        _require(metric.get(field) == recomputed[field], "%s CSV %s differs from aggregate" % (name, field))
    for field in AGGREGATE_FIELDS:
        expected = recomputed.get(field)
        actual = metric.get(field)
        if expected is None:
            _require(actual is None, "%s CSV %s should be null" % (name, field))
        else:
            _require(_finite(actual) and _close(actual, expected), "%s CSV %s differs from aggregate" % (name, field))
    metric["csv_aggregate_crosscheck_pass"] = True


def audit_paired_candidate_records(episode_data, expected_names):
    _require(expected_names, "paired audit requires at least one candidate")
    reference = expected_names[0]
    reference_keys = set(episode_data[reference])
    reference_records = episode_data[reference]
    for name in expected_names[1:]:
        _require(set(episode_data[name]) == reference_keys, "paired episode keys differ for %s" % name)
        for key in reference_keys:
            expected = reference_records[key]
            actual = episode_data[name][key]
            _require(actual["episode_seed"] == expected["episode_seed"], "paired episode seeds differ for %s" % name)
            _require(actual["disturbance_seed"] == expected["disturbance_seed"], "paired disturbance seeds differ for %s" % name)
            _require(actual["disturbance_protocol"] == expected["disturbance_protocol"], "paired disturbance protocols differ for %s" % name)
            _require(actual["disturbance_apply_match"] == 1, "paired disturbance apply audit failed for %s" % name)
            for field in DISTURBANCE_KEYS:
                if field == "action_delay_steps":
                    matches = actual[field] == expected[field]
                else:
                    matches = math.isclose(actual[field], expected[field], rel_tol=1e-12, abs_tol=1e-12)
                _require(matches, "paired disturbance field %s differs for %s" % (field, name))
            for field in FLOW_PHASE_KEYS:
                _require(
                    math.isclose(actual[field], expected[field], rel_tol=1e-12, abs_tol=1e-12),
                    "paired flow phase %s differs for %s" % (field, name),
                )
    reference_values = list(reference_records.values())
    distinct_9d = {
        tuple(record[field] for field in DISTURBANCE_KEYS) for record in reference_values
    }
    distinct_full = {
        tuple(record[field] for field in (*DISTURBANCE_KEYS, *FLOW_PHASE_KEYS))
        for record in reference_values
    }
    per_dimension = {
        field: len({record[field] for record in reference_values})
        for field in DISTURBANCE_KEYS
    }
    return {
        "enabled": True,
        "episode_seed_mode": PAIRED_SEED_MODE,
        "episode_seed_formula": PAIRED_SEED_FORMULA,
        "disturbance_protocol": DISTURBANCE_PROTOCOL,
        "disturbance_seed_formula": PAIRED_SEED_FORMULA,
        "episode_index_base": 0,
        "paired_episode_count_per_checkpoint": len(reference_keys),
        "all_candidate_keys_match": True,
        "all_candidate_episode_seeds_match": True,
        "all_candidate_disturbance_seeds_match": True,
        "all_candidate_disturbance_vectors_match": True,
        "all_candidate_flow_phases_match": True,
        "all_episode_disturbance_apply_match": all(
            record["disturbance_apply_match"] == 1
            for name in expected_names
            for record in episode_data[name].values()
        ),
        "bounds_violation_count": 0,
        "distinct_9d_vector_count": len(distinct_9d),
        "distinct_full_disturbance_count": len(distinct_full),
        "per_dimension_distinct_counts": per_dimension,
    }


def audit_fine_sweep(args, summary, fine_root, expected_names, coarse_ranks, model_dir):
    _require(summary.get("overall_pass") is True, "fine sweep overall_pass must be true")
    _require(summary.get("selection_policy") == "legacy", "fine selection_policy must be legacy")
    _require(summary.get("errors") == [], "fine sweep errors must be empty")
    settings = summary.get("settings") or {}
    _require(settings.get("selection_policy") == "legacy", "fine settings selection policy must be legacy")
    _require(settings.get("ablation_mode") == "ch4_uniform_dr", "fine mode must be ch4_uniform_dr")
    _require(settings.get("profile") == "normal_comm", "fine profile must be normal_comm")
    _require(settings.get("paired_episode_seeding") is True, "fine paired seeding must be enabled")
    _require(settings.get("disturbance_protocol") == DISTURBANCE_PROTOCOL, "fine disturbance protocol setting mismatch")
    _require(settings.get("require_paired_disturbances") is True, "fine paired disturbance requirement is disabled")
    _require(settings.get("validation_seeds") == args.expected_validation_seeds, "fine validation seeds mismatch")
    _require(settings.get("episodes_per_seed") == args.expected_episodes_per_seed, "fine episodes per seed mismatch")
    _require(settings.get("max_steps") == args.expected_max_steps, "fine max steps mismatch")
    _require(settings.get("checkpoint_names") == expected_names, "fine configured candidate names mismatch")
    _require(summary.get("validation_seeds") == args.expected_validation_seeds, "fine root validation seeds mismatch")
    _require(summary.get("seed_leakage_detected") is False, "fine sweep reports seed leakage")
    candidate_count = len(expected_names)
    _require(summary.get("candidate_count") == candidate_count, "fine candidate_count mismatch")
    _require(set(args.forbidden_seeds).issubset(set(summary.get("forbidden_test_seeds") or [])), "fine forbidden seed list is incomplete")
    validate_seed_policy(args.expected_validation_seeds, args.forbidden_seeds)
    rows = summary.get("ranked_checkpoints") or []
    row_by_name = {row.get("checkpoint_name"): row for row in rows}
    _require(len(row_by_name) == candidate_count and set(row_by_name) == set(expected_names), "fine checkpoint set mismatch")
    _require(list(expected_names) == sorted(expected_names, key=_checkpoint_episode), "fine candidates are not in episode order")
    expected_total = len(args.expected_validation_seeds) * args.expected_episodes_per_seed
    _audit_disturbance_protocol_summary(
        summary.get("disturbance_protocol_audit"),
        "fine sweep",
        candidate_count,
        expected_total,
    )
    metrics = []
    episode_data = {}
    for name in expected_names:
        row = row_by_name[name]
        aggregate_path = fine_root / name / "aggregate_summary.json"
        aggregate = _load_json(aggregate_path, "%s aggregate summary" % name)
        _require(aggregate.get("summary_name") == name, "%s aggregate name mismatch" % name)
        _require(aggregate.get("method") == "ch4_uniform_dr", "%s aggregate method mismatch" % name)
        _require(aggregate.get("profile") == "normal_comm", "%s aggregate profile mismatch" % name)
        _require(aggregate.get("seeds") == args.expected_validation_seeds, "%s aggregate seeds mismatch" % name)
        _require(aggregate.get("episodes_per_seed") == [args.expected_episodes_per_seed] * len(args.expected_validation_seeds), "%s aggregate episode counts mismatch" % name)
        episode = _as_int(row.get("checkpoint_episode"), "%s checkpoint episode" % name)
        _require(episode == _checkpoint_episode(name), "%s checkpoint episode mismatch" % name)
        expected_checkpoint = (model_dir / (name + ".pt")).resolve()
        actual_checkpoint = Path(str(row.get("checkpoint_path", ""))).resolve()
        _require(actual_checkpoint == expected_checkpoint and actual_checkpoint.is_file(), "%s checkpoint path is invalid" % name)
        metric = {
            "coarse_rank": coarse_ranks[name],
            "checkpoint_name": name,
            "checkpoint_episode": episode,
            "checkpoint_path": str(actual_checkpoint),
            "aggregate_summary_path": str(aggregate_path.resolve()),
        }
        for output_field, aggregate_field in AGGREGATE_FIELDS.items():
            _require(aggregate_field in aggregate, "%s aggregate is missing %s" % (name, aggregate_field))
            metric[output_field] = aggregate.get(aggregate_field)
        for field in COUNT_FIELDS:
            _require(field in aggregate, "%s aggregate is missing %s" % (name, field))
            metric[field] = aggregate.get(field)
        _require(metric["total_episodes"] == expected_total, "%s total episodes mismatch" % name)
        _audit_metric_semantics(metric, args.allow_zero_found and args.smoke)
        records = {}
        for seed in args.expected_validation_seeds:
            seed_dir = fine_root / name / ("seed%d" % seed)
            seed_summary = _load_json(seed_dir / "evaluation_summary.json", "%s seed%d summary" % (name, seed))
            _require(seed_summary.get("method") == "ch4_uniform_dr", "%s seed%d method mismatch" % (name, seed))
            _require(seed_summary.get("profile") == "normal_comm", "%s seed%d profile mismatch" % (name, seed))
            _require(seed_summary.get("seed") == seed, "%s seed summary mismatch" % name)
            _require(seed_summary.get("episodes") == args.expected_episodes_per_seed, "%s seed episode count mismatch" % name)
            _require(seed_summary.get("max_steps") == args.expected_max_steps, "%s seed max steps mismatch" % name)
            _require(seed_summary.get("paired_episode_seeding") is True, "%s paired seeding disabled" % name)
            _require(seed_summary.get("episode_seed_mode") == PAIRED_SEED_MODE, "%s seed mode mismatch" % name)
            _require(seed_summary.get("episode_seed_formula") == PAIRED_SEED_FORMULA, "%s seed formula mismatch" % name)
            _require(seed_summary.get("episode_index_base") == 0, "%s episode index base mismatch" % name)
            _require(seed_summary.get("disturbance_protocol") == DISTURBANCE_PROTOCOL, "%s disturbance protocol mismatch" % name)
            _require(seed_summary.get("disturbance_protocol_version") == DISTURBANCE_PROTOCOL_VERSION, "%s disturbance protocol version mismatch" % name)
            _require(seed_summary.get("explicit_disturbance_application") is True, "%s explicit disturbance application missing" % name)
            _require(seed_summary.get("all_episode_disturbance_apply_match") is True, "%s disturbance apply audit failed" % name)
            _require(seed_summary.get("bounds_violation_count") == 0, "%s disturbance bounds violation reported" % name)
            _require(seed_summary.get("disturbance_keys") == list(DISTURBANCE_KEYS), "%s disturbance keys mismatch" % name)
            _require(seed_summary.get("flow_phase_in_9d_vector") is False, "%s flow phase incorrectly included in 9-D" % name)
            _require(seed_summary.get("first_episode_seed") == _expected_episode_seed(seed, 0), "%s first episode seed mismatch" % name)
            _require(seed_summary.get("last_episode_seed") == _expected_episode_seed(seed, args.expected_episodes_per_seed - 1), "%s last episode seed mismatch" % name)
            seed_records = _load_episode_csv(
                seed_dir / "episode_metrics.csv",
                seed,
                args.expected_episodes_per_seed,
                name,
            )
            records.update(seed_records)
        _audit_episode_aggregates(metric, records, expected_total)
        metrics.append(metric)
        episode_data[name] = records
    paired_audit = audit_paired_candidate_records(episode_data, expected_names)
    _require(paired_audit["paired_episode_count_per_checkpoint"] == expected_total, "paired episode count mismatch")
    return metrics, episode_data, expected_total, paired_audit


def _relative_survivors(metrics, field, tolerance):
    minimum = min(float(row[field]) for row in metrics)
    allowed_delta = max(abs(minimum) * float(tolerance), 1e-12)
    survivors = [
        row for row in metrics if float(row[field]) <= minimum + allowed_delta
    ]
    _require(survivors, "%s gate removed every candidate" % field)
    return minimum, survivors


def select_checkpoint(metrics, args):
    best_success = max(float(row["success_rate"]) for row in metrics)
    success_threshold = best_success - args.success_tie_pp / 100.0
    after_success = [row for row in metrics if float(row["success_rate"]) >= success_threshold - 1e-15]
    _require(after_success, "success gate removed every candidate")
    best_found = max(float(row["found_rate"]) for row in after_success)
    found_threshold = best_found - args.found_tie_pp / 100.0
    after_found = [row for row in after_success if float(row["found_rate"]) >= found_threshold - 1e-15]
    _require(after_found, "found gate removed every candidate")
    finite_sif = [float(row["succ_if_found"]) for row in after_found if _finite(row.get("succ_if_found"))]
    if finite_sif:
        best_sif = max(finite_sif)
        sif_threshold = best_sif - args.sif_tie_pp / 100.0
        after_sif = [row for row in after_found if _finite(row.get("succ_if_found")) and float(row["succ_if_found"]) >= sif_threshold - 1e-15]
        sif_skipped = False
    else:
        _require(args.smoke and args.allow_zero_found, "SIF gate cannot be skipped outside zero-found smoke")
        best_sif = None
        sif_threshold = None
        after_sif = list(after_found)
        sif_skipped = True
    _require(after_sif, "SIF gate removed every candidate")
    min_safety, after_safety = _relative_survivors(after_sif, "avg_safety_cost", args.safety_tie_relative)
    min_recovery, after_recovery = _relative_survivors(after_safety, "avg_recovery_time", args.recovery_tie_relative)
    min_distance, after_distance = _relative_survivors(after_recovery, "avg_final_distance", args.distance_tie_relative)
    selected = min(after_distance, key=lambda row: int(row["checkpoint_episode"]))
    names = lambda rows: [row["checkpoint_name"] for row in rows]
    gates = {
        "best_success": best_success,
        "success_threshold": success_threshold,
        "candidates_after_success_gate": names(after_success),
        "best_found": best_found,
        "found_threshold": found_threshold,
        "candidates_after_found_gate": names(after_found),
        "best_sif": best_sif,
        "sif_threshold": sif_threshold,
        "sif_gate_skipped": sif_skipped,
        "candidates_after_sif_gate": names(after_sif),
        "min_safety": min_safety,
        "candidates_after_safety_gate": names(after_safety),
        "min_recovery_time": min_recovery,
        "candidates_after_recovery_gate": names(after_recovery),
        "min_final_distance": min_distance,
        "candidates_after_distance_gate": names(after_distance),
        "final_tie_break": "smallest checkpoint_episode",
    }
    gate_sets = {
        "passed_success_gate": set(gates["candidates_after_success_gate"]),
        "passed_found_gate": set(gates["candidates_after_found_gate"]),
        "passed_sif_gate": set(gates["candidates_after_sif_gate"]),
        "passed_safety_gate": set(gates["candidates_after_safety_gate"]),
        "passed_recovery_gate": set(gates["candidates_after_recovery_gate"]),
        "passed_distance_gate": set(gates["candidates_after_distance_gate"]),
    }
    for row in metrics:
        for field, survivors in gate_sets.items():
            row[field] = row["checkpoint_name"] in survivors
        row["selected"] = row["checkpoint_name"] == selected["checkpoint_name"]
    return selected, gates


def _percentile(values, probability):
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _sample_statistics(records, keys):
    count = len(keys)
    success = sum(records[key]["success"] for key in keys)
    found = sum(records[key]["found"] for key in keys)
    return {
        "success_rate": success / float(count),
        "found_rate": found / float(count),
        "succ_if_found": success / float(found) if found else None,
        "avg_safety_cost": sum(records[key]["safety_cost"] for key in keys) / float(count),
        "avg_recovery_time": sum(records[key]["recovery_time"] for key in keys) / float(count),
        "avg_final_distance": sum(records[key]["final_distance"] for key in keys) / float(count),
    }


def _metric_delta(metric, selected_stats, candidate_stats):
    left = selected_stats.get(metric)
    right = candidate_stats.get(metric)
    if left is None or right is None:
        return None
    if metric in ("success_rate", "found_rate", "succ_if_found"):
        return float(left) - float(right)
    return float(right) - float(left)


def paired_stratified_bootstrap(selected, metrics, episode_data, seeds, reps, random_seed):
    selected_name = selected["checkpoint_name"]
    selected_records = episode_data[selected_name]
    metric_names = (
        "success_rate",
        "found_rate",
        "succ_if_found",
        "avg_safety_cost",
        "avg_recovery_time",
        "avg_final_distance",
    )
    rng = random.Random(random_seed)
    comparisons = []
    keys_by_seed = {
        seed: sorted([key for key in selected_records if key[0] == seed])
        for seed in seeds
    }
    all_keys = [key for seed in seeds for key in keys_by_seed[seed]]
    for candidate in metrics:
        candidate_name = candidate["checkpoint_name"]
        if candidate_name == selected_name:
            continue
        candidate_records = episode_data[candidate_name]
        selected_point = _sample_statistics(selected_records, all_keys)
        candidate_point = _sample_statistics(candidate_records, all_keys)
        bootstrap_values = {metric: [] for metric in metric_names}
        for _rep in range(reps):
            sampled = []
            for seed in seeds:
                source_keys = keys_by_seed[seed]
                sampled.extend(
                    source_keys[rng.randrange(len(source_keys))]
                    for _ in range(len(source_keys))
                )
            selected_stats = _sample_statistics(selected_records, sampled)
            candidate_stats = _sample_statistics(candidate_records, sampled)
            for metric in metric_names:
                delta = _metric_delta(metric, selected_stats, candidate_stats)
                if delta is not None:
                    bootstrap_values[metric].append(delta)
        metric_results = {}
        for metric in metric_names:
            values = bootstrap_values[metric]
            low = _percentile(values, 0.025)
            high = _percentile(values, 0.975)
            metric_results[metric + "_delta"] = {
                "direction": (
                    "selected_minus_candidate"
                    if metric in ("success_rate", "found_rate", "succ_if_found")
                    else "candidate_minus_selected"
                ),
                "point_estimate": _metric_delta(metric, selected_point, candidate_point),
                "ci95_low": low,
                "ci95_high": high,
                "ci_includes_zero": (
                    None if low is None or high is None else low <= 0.0 <= high
                ),
                "probability_selected_better": (
                    None
                    if not values
                    else sum(value > 0.0 for value in values) / float(len(values))
                ),
                "valid_bootstrap_reps": len(values),
            }
        selected_success_candidate_fail = sum(
            selected_records[key]["success"] == 1
            and candidate_records[key]["success"] == 0
            for key in all_keys
        )
        selected_fail_candidate_success = sum(
            selected_records[key]["success"] == 0
            and candidate_records[key]["success"] == 1
            for key in all_keys
        )
        comparisons.append(
            {
                "selected_checkpoint_name": selected_name,
                "candidate_checkpoint_name": candidate_name,
                "paired_episode_count": len(all_keys),
                "bootstrap_reps": reps,
                "discordant_success_counts": {
                    "selected_success_candidate_fail": selected_success_candidate_fail,
                    "selected_fail_candidate_success": selected_fail_candidate_success,
                },
                "metrics": metric_results,
            }
        )
    return comparisons


def freeze_selected_model(source_path, selected_dir, selected_name, copy_function=shutil.copy2):
    source_path = Path(source_path).resolve()
    selected_dir = Path(selected_dir).resolve()
    _require(source_path.is_file(), "selected source checkpoint is missing: %s" % source_path)
    _require(not selected_dir.exists(), "selected model directory already exists: %s" % selected_dir)
    final_path = selected_dir / selected_name
    temporary = selected_dir / (selected_name + ".tmp")
    source_hash = _sha256(source_path)
    try:
        selected_dir.mkdir(parents=True, exist_ok=False)
        copy_function(str(source_path), str(temporary))
        copied_hash = _sha256(temporary)
        _require(copied_hash == source_hash, "selected model SHA256 does not match source checkpoint")
        os.replace(str(temporary), str(final_path))
        return final_path, source_hash, copied_hash
    except Exception:
        if selected_dir.exists():
            shutil.rmtree(str(selected_dir))
        raise


def _write_csv(path, metrics):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for metric in sorted(metrics, key=lambda row: row["checkpoint_episode"]):
                writer.writerow({field: metric.get(field, "") for field in CSV_FIELDS})
        os.replace(str(temporary), str(path))
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def _fmt(value, digits=6):
    return "null" if value is None else ("%%.%df" % digits) % float(value)


def _markdown(summary):
    gates = summary["selection_gates"]
    paired = summary["paired_randomization_audit"]
    derivation = summary["coarse_candidate_derivation"]
    lines = [
        "# Chapter-4 Uniform DR Independent Final Validation",
        "",
        "- The old reduced-disturbance coarse-v1 was not used for this selection.",
        "- Coarse-v2 explicitly sampled the registered full 9-D disturbance vector.",
        "- Actuator lag, action delay, and action noise all came from the registry bounds.",
        "- Flow phases were paired across checkpoints but were not part of the 9-D vector.",
        "- Candidate count `%s` was derived dynamically from best coarse success minus 1 percentage point."
        % summary["settings"]["candidate_count"],
        "- Validation seeds: `%s`." % summary["settings"]["validation_seeds"],
        "- Test seeds `1/2/3` were not used for selection.",
        "- Paired randomization audit passed: keys=%s, seeds=%s, vectors=%s, phases=%s."
        % (
            paired["all_candidate_keys_match"],
            paired["all_candidate_episode_seeds_match"],
            paired["all_candidate_disturbance_vectors_match"],
            paired["all_candidate_flow_phases_match"],
        ),
        "- This completes validation selection, not a formal test.",
        "",
        "## Candidate Derivation",
        "",
        "- Best coarse success: `%s`." % _fmt(derivation["best_success"]),
        "- One-percentage-point threshold: `%s`." % _fmt(derivation["threshold"]),
        "- Raw margin survivors: `%s`." % derivation["raw_margin_candidates"],
        "- Mode-specific minimum candidate count: `%s`." % derivation["minimum_candidate_count"],
        "- Minimum fill applied: `%s`." % derivation["minimum_fill_applied"],
        "- Minimum fill candidates: `%s`." % derivation["minimum_fill_candidates"],
        "- Final candidates: `%s`." % derivation["derived_candidates"],
        "- Legacy Top-K used for derivation: `False`.",
        "",
        "## Gate Survivors",
        "",
        "- Success: `%s`" % gates["candidates_after_success_gate"],
        "- Found: `%s`" % gates["candidates_after_found_gate"],
        "- Success-if-found: `%s` (skipped=%s)"
        % (gates["candidates_after_sif_gate"], gates["sif_gate_skipped"]),
        "- Safety: `%s`" % gates["candidates_after_safety_gate"],
        "- Recovery: `%s`" % gates["candidates_after_recovery_gate"],
        "- Final distance: `%s`" % gates["candidates_after_distance_gate"],
        "- Final tie-break: `%s`" % gates["final_tie_break"],
        "",
        "## Selected Checkpoint",
        "",
        "- Checkpoint: `%s` (episode %s)"
        % (summary["selected_checkpoint_name"], summary["selected_checkpoint_episode"]),
        "- Frozen model: `%s`" % summary["selected_model_path"],
        "- SHA256: `%s`" % summary["selected_sha256"],
        "- Source/copy hash match: `%s`" % summary["hash_match"],
        "",
        "## Candidate Metrics",
        "",
        "| coarse rank | checkpoint | episode | success | found | SIF | safety | recovery | distance | selected |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in sorted(summary["candidate_metrics"], key=lambda item: item["checkpoint_episode"]):
        lines.append(
            "| %d | %s | %d | %s | %s | %s | %s | %s | %s | %s |"
            % (
                row["coarse_rank"],
                row["checkpoint_name"],
                row["checkpoint_episode"],
                _fmt(row["success_rate"]),
                _fmt(row["found_rate"]),
                _fmt(row["succ_if_found"]),
                _fmt(row["avg_safety_cost"]),
                _fmt(row["avg_recovery_time"]),
                _fmt(row["avg_final_distance"]),
                row["selected"],
            )
        )
    lines.extend(
        [
            "",
            "## Paired Stratified Bootstrap",
            "",
            "Positive deltas always favor the selected checkpoint. Rate deltas are selected minus candidate; cost deltas are candidate minus selected.",
            "",
            "| comparison | metric | point | 95% CI | P(selected better) | CI includes 0 |",
            "| --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    any_ci_contains_zero = False
    for comparison in summary["paired_bootstrap"]:
        label = "%s vs %s" % (
            comparison["selected_checkpoint_name"],
            comparison["candidate_checkpoint_name"],
        )
        for metric_name, result in comparison["metrics"].items():
            includes = result["ci_includes_zero"]
            any_ci_contains_zero = any_ci_contains_zero or includes is True
            lines.append(
                "| %s | %s | %s | [%s, %s] | %s | %s |"
                % (
                    label,
                    metric_name,
                    _fmt(result["point_estimate"]),
                    _fmt(result["ci95_low"]),
                    _fmt(result["ci95_high"]),
                    _fmt(result["probability_selected_better"]),
                    includes,
                )
            )
    if any_ci_contains_zero:
        lines.extend(
            [
                "",
                "At least one 95% CI includes zero; the candidates are not clearly separated statistically under this validation sample.",
            ]
        )
    lines.extend(
        [
            "",
            "## Readiness",
            "",
            "- Checkpoint selection performed: `%s`" % summary["checkpoint_selection_performed"],
            "- Test used for selection: `%s`" % summary["test_used_for_selection"],
            "- Ready for formal robust test: `%s`" % summary["ready_for_formal_robust_test"],
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Finalize independent Chapter-4 Uniform DR checkpoint validation."
    )
    parser.add_argument("--coarse-summary", required=True)
    parser.add_argument("--fine-sweep-summary", required=True)
    parser.add_argument("--fine-result-root", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--training-config", required=True)
    parser.add_argument("--training-audit-summary", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--selected-model-dir", required=True)
    parser.add_argument(
        "--selected-model-name",
        default="selected_by_uniform_dr_validation.pt",
    )
    parser.add_argument("--expected-candidates", nargs="+", required=True)
    parser.add_argument("--expected-validation-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--forbidden-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--expected-episodes-per-seed", type=int, required=True)
    parser.add_argument("--expected-max-steps", type=int, required=True)
    parser.add_argument("--success-tie-pp", type=float, default=1.5)
    parser.add_argument("--found-tie-pp", type=float, default=2.0)
    parser.add_argument("--sif-tie-pp", type=float, default=2.0)
    parser.add_argument("--safety-tie-relative", type=float, default=0.10)
    parser.add_argument("--recovery-tie-relative", type=float, default=0.10)
    parser.add_argument("--distance-tie-relative", type=float, default=0.10)
    parser.add_argument("--bootstrap-reps", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260717)
    parser.add_argument("--allow-zero-found", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def _validate_fixed_protocol(args, coarse_path, out_dir, selected_dir):
    candidate_count = len(args.expected_candidates)
    _require(candidate_count == len(set(args.expected_candidates)), "expected candidates contain duplicates")
    _require(args.expected_candidates == sorted(args.expected_candidates, key=_checkpoint_episode), "expected candidates must be in checkpoint episode order")
    _require(args.selected_model_name == "selected_by_uniform_dr_validation.pt", "selected model filename is invalid")
    _require(_close(args.success_tie_pp, 1.5), "success tie must be 1.5 pp")
    _require(_close(args.found_tie_pp, 2.0), "found tie must be 2.0 pp")
    _require(_close(args.sif_tie_pp, 2.0), "SIF tie must be 2.0 pp")
    _require(_close(args.safety_tie_relative, 0.10), "safety tolerance must be 0.10")
    _require(_close(args.recovery_tie_relative, 0.10), "recovery tolerance must be 0.10")
    _require(_close(args.distance_tie_relative, 0.10), "distance tolerance must be 0.10")
    _require(args.bootstrap_seed == 20260717, "bootstrap seed must be 20260717")
    if args.smoke:
        _require(1 <= candidate_count <= 20, "smoke candidate_count must be between 1 and 20")
        _require(args.expected_validation_seeds == SMOKE_VALIDATION_SEEDS, "smoke validation seed must be 906")
        _require(args.forbidden_seeds == SMOKE_FORBIDDEN_SEEDS, "smoke forbidden seeds are invalid")
        _require(args.expected_episodes_per_seed == 2, "smoke episodes per seed must be 2")
        _require(args.expected_max_steps == 20, "smoke max steps must be 20")
        _require(args.bootstrap_reps == 200, "smoke bootstrap reps must be 200")
        _require(args.allow_zero_found is True, "smoke requires --allow-zero-found")
        _require(coarse_path == (get_smoke_dir("uniform_dr", "ch4_uniform_dr_checkpoint_validation_coarse_full9d_v2_smoke") / "checkpoint_sweep_summary.json").resolve(), "smoke coarse summary path is invalid")
        _require(out_dir == get_smoke_dir("uniform_dr", "ch4_uniform_dr_checkpoint_validation_final_full9d_v2_smoke").resolve(), "smoke out-dir is invalid")
        _require(selected_dir == get_smoke_dir("uniform_dr", "ch4_uniform_dr_selected_full9d_v2_smoke").resolve(), "smoke selected-model-dir is invalid")
    else:
        _require(2 <= candidate_count <= 20, "formal candidate_count must be between 2 and 20")
        _require(args.expected_validation_seeds == FORMAL_VALIDATION_SEEDS, "formal validation seeds must be 106,107,108")
        _require(args.forbidden_seeds == FORMAL_FORBIDDEN_SEEDS, "formal forbidden seeds are invalid")
        _require(args.expected_episodes_per_seed == 200, "formal episodes per seed must be 200")
        _require(args.expected_max_steps == 400, "formal max steps must be 400")
        _require(args.bootstrap_reps == 20000, "formal bootstrap reps must be 20000")
        _require(args.allow_zero_found is False, "formal mode forbids --allow-zero-found")
        _require(coarse_path == (get_selection_dir("uniform_dr", "ch4_uniform_dr_checkpoint_validation_coarse_full9d_v2") / "checkpoint_sweep_summary.json").resolve(), "formal coarse summary path is invalid")
        _require(out_dir == get_selection_dir("uniform_dr", "ch4_uniform_dr_checkpoint_validation_final_full9d_v2").resolve(), "formal out-dir is invalid")
        _require(selected_dir == get_selected_dir("uniform_dr", "ch4_uniform_dr_selected_full9d_v2").resolve(), "formal selected-model-dir is invalid")
    validate_seed_policy(args.expected_validation_seeds, args.forbidden_seeds)


def run(args):
    coarse_path = Path(args.coarse_summary).resolve()
    fine_summary_path = Path(args.fine_sweep_summary).resolve()
    fine_root = Path(args.fine_result_root).resolve()
    model_dir = Path(args.model_dir).resolve()
    training_config_path = Path(args.training_config).resolve()
    training_audit_path = Path(args.training_audit_summary).resolve()
    out_dir = Path(args.out_dir).resolve()
    selected_dir = Path(args.selected_model_dir).resolve()
    _validate_fixed_protocol(args, coarse_path, out_dir, selected_dir)
    _require(model_dir.is_dir(), "model directory is missing: %s" % model_dir)
    _require(fine_root.is_dir(), "fine result root is missing: %s" % fine_root)
    _require(fine_summary_path == fine_root / "checkpoint_sweep_summary.json", "fine summary must be raw_sweep/checkpoint_sweep_summary.json")
    _require(fine_root.parent == out_dir and fine_root.name == "raw_sweep", "fine root must be out-dir/raw_sweep")
    _require(not selected_dir.exists(), "selected model directory already exists: %s" % selected_dir)
    output_paths = [
        out_dir / "final_checkpoint_selection_summary.json",
        out_dir / "final_checkpoint_selection_table.csv",
        out_dir / "final_checkpoint_selection_summary.md",
    ]
    _require(not any(path.exists() for path in output_paths), "one or more final selection outputs already exist")

    coarse = _load_json(coarse_path, "coarse summary")
    derivation, coarse_ranks = audit_coarse_candidate_derivation(
        coarse, args.expected_candidates, smoke=args.smoke
    )
    training_config = _load_json(training_config_path, "training config")
    training_audit = _load_json(training_audit_path, "training audit summary")
    training_source_audit = audit_training_sources(training_config, training_audit)
    fine = _load_json(fine_summary_path, "fine sweep summary")
    metrics, episode_data, expected_total, paired_audit = audit_fine_sweep(
        args,
        fine,
        fine_root,
        args.expected_candidates,
        coarse_ranks,
        model_dir,
    )
    selected, gates = select_checkpoint(metrics, args)
    bootstrap = paired_stratified_bootstrap(
        selected,
        metrics,
        episode_data,
        args.expected_validation_seeds,
        args.bootstrap_reps,
        args.bootstrap_seed,
    )
    _require(
        len(bootstrap) == max(0, len(args.expected_candidates) - 1),
        "paired bootstrap comparison count does not match dynamic candidate_count",
    )

    selected_path = None
    try:
        selected_path, source_hash, selected_hash = freeze_selected_model(
            selected["checkpoint_path"],
            selected_dir,
            args.selected_model_name,
        )
        created_at = datetime.now(timezone.utc).isoformat()
        ready = not args.smoke
        selection_rule = {
            "success_gate_pp": args.success_tie_pp,
            "found_gate_pp": args.found_tie_pp,
            "success_if_found_gate_pp": args.sif_tie_pp,
            "safety_relative_tolerance": args.safety_tie_relative,
            "recovery_relative_tolerance": args.recovery_tie_relative,
            "final_distance_relative_tolerance": args.distance_tie_relative,
            "gate_order": [
                "success",
                "found",
                "success_if_found",
                "safety_cost",
                "recovery_time",
                "final_distance",
            ],
            "final_tie_break": "smallest checkpoint_episode",
            "coarse_rank_used_for_selection": False,
            "legacy_fine_rank_used_for_selection": False,
            "coarse_candidate_rule": (
                "success >= best coarse success - 1 percentage point"
            ),
            "minimum_candidate_fill": (
                "when fewer than the mode-specific minimum survive, fill by success desc then checkpoint episode asc"
            ),
        }
        manifest_path = selected_dir / "selected_model_manifest.json"
        manifest = {
            "experiment_type": "uniform_dr_checkpoint_validation_final",
            "selection_stage": "independent_final_validation",
            "selection_smoke": bool(args.smoke),
            "training_run": TRAINING_RUN,
            "training_seed": 1,
            "warm_start_source_checkpoint": WARM_START_SOURCE_CHECKPOINT,
            "warm_start_source_sha256": WARM_START_SOURCE_SHA256,
            "coarse_validation_seeds": SMOKE_COARSE_SEEDS if args.smoke else COARSE_SEEDS,
            "final_validation_seeds": args.expected_validation_seeds,
            "forbidden_formal_test_seeds": FORMAL_TEST_SEEDS,
            "candidate_names": args.expected_candidates,
            "candidate_count": len(args.expected_candidates),
            "candidate_derivation_dynamic": True,
            "candidate_minimum_fill_applied": derivation["minimum_fill_applied"],
            "candidate_minimum_count": derivation["minimum_candidate_count"],
            "raw_margin_candidate_names": derivation["raw_margin_candidates"],
            "minimum_fill_candidate_names": derivation["minimum_fill_candidates"],
            "evaluation_protocol": DISTURBANCE_PROTOCOL,
            "evaluation_protocol_version": DISTURBANCE_PROTOCOL_VERSION,
            "disturbance_keys": list(DISTURBANCE_KEYS),
            "disturbance_bounds": {
                key: list(DEFAULT_DISTURBANCE_BOUNDS[key]) for key in DISTURBANCE_KEYS
            },
            "flow_phase_keys": list(FLOW_PHASE_KEYS),
            "flow_phase_in_9d_vector": False,
            "paired_disturbance_vectors_verified": paired_audit["all_candidate_disturbance_vectors_match"],
            "paired_flow_phases_verified": paired_audit["all_candidate_flow_phases_match"],
            "legacy_reduced_disturbance_results_used_for_selection": False,
            "selected_checkpoint_name": selected["checkpoint_name"],
            "selected_checkpoint_episode": selected["checkpoint_episode"],
            "source_checkpoint_path": selected["checkpoint_path"],
            "selected_model_path": str(selected_path),
            "source_sha256": source_hash,
            "selected_sha256": selected_hash,
            "hash_match": source_hash == selected_hash,
            "selection_rule": selection_rule,
            "paired_episode_seeding": True,
            "checkpoint_selection_performed": True,
            "test_used_for_selection": False,
            "ready_for_formal_robust_test": ready,
            "formal_robust_test_completed": False,
            "created_at_utc": created_at,
        }
        _atomic_write_json(manifest_path, manifest)
        warnings = []
        ci_contains_zero = any(
            result.get("ci_includes_zero") is True
            for comparison in bootstrap
            for result in comparison["metrics"].values()
        )
        if ci_contains_zero:
            warnings.append(
                "At least one bootstrap CI includes zero; candidates are not clearly separated statistically under this validation sample."
            )
        summary = {
            "overall_pass": True,
            "experiment_type": "uniform_dr_checkpoint_validation_final",
            "selection_stage": "independent_final_validation",
            "selection_smoke": bool(args.smoke),
            "evaluation_protocol": DISTURBANCE_PROTOCOL,
            "evaluation_protocol_version": DISTURBANCE_PROTOCOL_VERSION,
            "disturbance_protocol_audit": fine.get("disturbance_protocol_audit"),
            "legacy_coarse_v1_used": False,
            "candidate_derivation_dynamic": True,
            "candidate_minimum_fill_applied": derivation["minimum_fill_applied"],
            "candidate_minimum_count": derivation["minimum_candidate_count"],
            "coarse_summary": str(coarse_path),
            "fine_sweep_summary": str(fine_summary_path),
            "fine_result_root": str(fine_root),
            "model_dir": str(model_dir),
            "training_config": str(training_config_path),
            "training_audit_summary": str(training_audit_path),
            "coarse_candidate_derivation": derivation,
            "training_source_audit": training_source_audit,
            "settings": {
                "mode": "ch4_uniform_dr",
                "profile": "normal_comm",
                "validation_seeds": args.expected_validation_seeds,
                "forbidden_formal_test_seeds": FORMAL_TEST_SEEDS,
                "forbidden_validation_seeds": args.forbidden_seeds,
                "episodes_per_seed": args.expected_episodes_per_seed,
                "total_episodes_per_checkpoint": expected_total,
                "candidate_count": len(args.expected_candidates),
                "max_steps": args.expected_max_steps,
                "paired_episode_seeding": True,
                "episode_index_base": 0,
                "episode_seed_formula": PAIRED_SEED_FORMULA,
                "disturbance_protocol": DISTURBANCE_PROTOCOL,
                "disturbance_seed_formula": PAIRED_SEED_FORMULA,
                "bootstrap_reps": args.bootstrap_reps,
                "bootstrap_seed": args.bootstrap_seed,
            },
            "candidate_names": args.expected_candidates,
            "candidate_metrics": metrics,
            "paired_randomization_audit": paired_audit,
            "selection_gates": gates,
            "paired_bootstrap": bootstrap,
            "selected_checkpoint_name": selected["checkpoint_name"],
            "selected_checkpoint_episode": selected["checkpoint_episode"],
            "selected_checkpoint_path": selected["checkpoint_path"],
            "selected_model_path": str(selected_path),
            "selected_model_manifest": str(manifest_path),
            "source_sha256": source_hash,
            "selected_sha256": selected_hash,
            "hash_match": source_hash == selected_hash,
            "checkpoint_selection_performed": True,
            "test_used_for_selection": False,
            "ready_for_formal_robust_test": ready,
            "formal_robust_test_completed": False,
            "created_at_utc": created_at,
            "errors": [],
            "warnings": warnings,
        }
        _write_csv(output_paths[1], metrics)
        _atomic_write_text(output_paths[2], _markdown(summary))
        _atomic_write_json(output_paths[0], summary)
        return summary
    except Exception:
        if selected_dir.exists():
            shutil.rmtree(str(selected_dir))
        for path in output_paths:
            if path.exists():
                path.unlink()
        raise


def main():
    derive_modes = {
        "--derive-coarse-candidates": "text",
        "--derive-coarse-candidates-json": "json",
    }
    if len(sys.argv) >= 2 and sys.argv[1] in derive_modes:
        if len(sys.argv) not in (3, 4):
            print("[ERROR] candidate derivation requires a coarse summary and optional --smoke", file=sys.stderr)
            return 2
        smoke = len(sys.argv) == 4 and sys.argv[3] == "--smoke"
        if len(sys.argv) == 4 and not smoke:
            print("[ERROR] candidate derivation accepts only optional --smoke", file=sys.stderr)
            return 2
        try:
            if derive_modes[sys.argv[1]] == "json":
                print(
                    json.dumps(
                        _json_safe(
                            derive_coarse_candidates_json_for_cli(
                                sys.argv[2], smoke=smoke
                            )
                        ),
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                )
            else:
                print(derive_coarse_candidates_for_cli(sys.argv[2], smoke=smoke))
            return 0
        except Exception as exc:
            print(
                "[ERROR] coarse candidate derivation failed: %s: %s"
                % (type(exc).__name__, exc),
                file=sys.stderr,
            )
            return 1
    args = parse_args()
    try:
        summary = run(args)
        print(
            "[PASS] Uniform DR final checkpoint selection: selected=%s episode=%s hash_match=%s"
            % (
                summary["selected_checkpoint_name"],
                summary["selected_checkpoint_episode"],
                summary["hash_match"],
            )
        )
        return 0
    except Exception as exc:
        print(
            "[ERROR] Uniform DR final checkpoint selection failed: %s: %s"
            % (type(exc).__name__, exc),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
