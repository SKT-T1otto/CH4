# -*- coding: utf-8 -*-
"""Evaluate and rank Chapter-4 checkpoints across seeds."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from registry.rbe_disturbance import DEFAULT_DISTURBANCE_BOUNDS, DISTURBANCE_KEYS

EXPERIMENT_TYPE = "clean_nominal_checkpoint_validation_coarse"
SELECTION_STAGE = "coarse"
CLEAN_RANKING_RULE = (
    "success_rate_weighted desc",
    "found_rate_weighted desc",
    "succ_if_found_weighted desc",
    "avg_safety_cost_weighted asc",
    "avg_final_distance_weighted asc",
    "checkpoint_episode asc",
)
WEIGHTED_CONTINUOUS_FIELDS = (
    "avg_reward_weighted",
    "avg_recovery_time_weighted",
    "avg_safety_cost_weighted",
    "avg_final_distance_weighted",
    "avg_final_nav_distance_weighted",
    "avg_action_smoothness_weighted",
    "avg_completion_steps_weighted",
)
TABLE_FIELDS = (
    "rank",
    "checkpoint_name",
    "checkpoint_episode",
    "checkpoint_path",
    "success_rate_weighted",
    "found_rate_weighted",
    "succ_if_found_weighted",
    "not_found_rate_weighted",
    "found_but_failed_rate_weighted",
    "late_found_fail_rate_weighted",
    "early_found_fail_rate_weighted",
    "avg_reward_weighted",
    "avg_recovery_time_weighted",
    "avg_safety_cost_weighted",
    "avg_final_distance_weighted",
    "avg_final_nav_distance_weighted",
    "avg_action_smoothness_weighted",
    "avg_completion_steps_weighted",
    "total_episodes",
    "total_success",
    "total_found",
    "total_not_found",
    "total_found_but_failed",
    "is_top_k",
    "aggregate_summary_path",
)
DISTURBANCE_PROTOCOL_AUTO = "auto"
DISTURBANCE_PROTOCOL_NOMINAL = "nominal_v1"
DISTURBANCE_PROTOCOL_UNIFORM_9D = "uniform_9d_registry_v1"
SUPPORTED_DISTURBANCE_PROTOCOLS = (
    DISTURBANCE_PROTOCOL_AUTO,
    DISTURBANCE_PROTOCOL_NOMINAL,
    DISTURBANCE_PROTOCOL_UNIFORM_9D,
)
FLOW_PHASE_KEYS = ("flow_phase_x", "flow_phase_y")
EPISODE_SEED_MODE = "indexed_common_random_numbers"
EPISODE_SEED_FORMULA = "base_seed * 1000003 + episode_index"
DISTURBANCE_RNG_MODE = "numpy.default_rng(episode_seed)"
REQUIRED_DISTURBANCE_CSV_FIELDS = (
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
)


class SweepError(RuntimeError):
    pass


def _safe_float(value):
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _safe_int(value):
    number = _safe_float(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _parse_flag(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes"}:
        return True
    if text in {"0", "false", "no"}:
        return False
    return None


def _resolved_requested_protocol(args):
    if args.disturbance_protocol != DISTURBANCE_PROTOCOL_AUTO:
        return args.disturbance_protocol
    if args.ablation_mode in {"ch4_uniform_dr", "ch4_reb_only", "ch4_rbe_full"}:
        return DISTURBANCE_PROTOCOL_UNIFORM_9D
    return DISTURBANCE_PROTOCOL_NOMINAL


def _empty_disturbance_audit(args, candidate_count=0):
    return {
        "overall_pass": False,
        "protocol": _resolved_requested_protocol(args),
        "protocol_version": 2,
        "disturbance_keys": list(DISTURBANCE_KEYS),
        "disturbance_bounds": {
            key: list(DEFAULT_DISTURBANCE_BOUNDS[key]) for key in DISTURBANCE_KEYS
        },
        "flow_phase_keys": list(FLOW_PHASE_KEYS),
        "flow_phase_in_9d_vector": False,
        "explicit_application": False,
        "paired_episode_seeding": bool(args.paired_episode_seeding),
        "episode_seed_formula": EPISODE_SEED_FORMULA,
        "disturbance_seed_formula": EPISODE_SEED_FORMULA,
        "candidate_count": int(candidate_count),
        "episode_count_per_checkpoint": int(len(args.seeds) * args.episodes),
        "all_candidate_keys_match": False,
        "all_candidate_episode_seeds_match": False,
        "all_candidate_disturbance_seeds_match": False,
        "all_candidate_disturbance_vectors_match": False,
        "all_candidate_flow_phases_match": False,
        "all_episode_apply_match": False,
        "bounds_violation_count": 0,
        "missing_field_count": 0,
        "distinct_9d_vector_count": 0,
        "distinct_full_disturbance_count": 0,
        "per_dimension_distinct_counts": {key: 0 for key in DISTURBANCE_KEYS},
        "observed_action_delay_values": [],
        "observed_actuator_lag_min": None,
        "observed_actuator_lag_max": None,
        "observed_action_noise_std_min": None,
        "observed_action_noise_std_max": None,
        "errors": [],
    }


def _load_episode_disturbance_rows(path: Path, checkpoint_name, seed):
    if not path.is_file():
        raise SweepError(f"{checkpoint_name}/seed{seed}: episode_metrics.csv is missing")
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            record = dict(row)
            record["_checkpoint_name"] = checkpoint_name
            record["_expected_seed"] = int(seed)
            rows.append(record)
    return rows


def _audit_evaluation_summary(summary, checkpoint_name, seed, args):
    label = f"{checkpoint_name}/seed{seed}"
    checks = (
        (summary.get("method") == args.ablation_mode, "method mismatch"),
        (summary.get("profile") == args.profile, "profile mismatch"),
        (_safe_int(summary.get("seed")) == int(seed), "seed mismatch"),
        (_safe_int(summary.get("episodes")) == int(args.episodes), "episodes mismatch"),
        (_safe_int(summary.get("max_steps")) == int(args.max_steps), "max_steps mismatch"),
        (summary.get("paired_episode_seeding") is bool(args.paired_episode_seeding), "paired seeding mismatch"),
        (summary.get("disturbance_protocol") == _resolved_requested_protocol(args), "disturbance protocol mismatch"),
        (_safe_int(summary.get("disturbance_protocol_version")) == 2, "disturbance protocol version must be 2"),
        (summary.get("explicit_disturbance_application") is True, "explicit disturbance application missing"),
        (summary.get("all_episode_disturbance_apply_match") is True, "episode disturbance apply mismatch"),
        (_safe_int(summary.get("bounds_violation_count")) == 0, "bounds violations reported"),
        (summary.get("disturbance_keys") == list(DISTURBANCE_KEYS), "disturbance key order mismatch"),
        (summary.get("flow_phase_in_9d_vector") is False, "flow phase incorrectly included in 9-D vector"),
    )
    failures = [message for passed, message in checks if not passed]
    distinct = _safe_int(summary.get("distinct_disturbance_vector_count"))
    if distinct is None or distinct < int(args.disturbance_audit_min_distinct_vectors):
        failures.append(
            f"distinct disturbance vector count {distinct} is below {args.disturbance_audit_min_distinct_vectors}"
        )
    if args.require_full_9d_dimension_variation:
        counts = summary.get("per_dimension_distinct_counts") or {}
        invalid = [key for key in DISTURBANCE_KEYS if _safe_int(counts.get(key)) is None or int(counts.get(key, 0)) < 2]
        if invalid:
            failures.append(f"full 9-D dimension variation missing for {invalid}")
    if failures:
        raise SweepError(f"{label}: evaluation summary audit failed: {failures}")


def audit_paired_disturbance_records(
    candidate_rows,
    expected_seeds,
    episodes_per_seed,
    protocol=DISTURBANCE_PROTOCOL_UNIFORM_9D,
    min_distinct_vectors=1,
    require_full_9d_dimension_variation=False,
    paired_episode_seeding=True,
):
    audit_args = argparse.Namespace(
        disturbance_protocol=protocol,
        ablation_mode="ch4_uniform_dr",
        seeds=list(expected_seeds),
        episodes=int(episodes_per_seed),
        paired_episode_seeding=bool(paired_episode_seeding),
    )
    audit = _empty_disturbance_audit(audit_args, candidate_count=len(candidate_rows))
    errors = []
    missing_field_count = 0
    bounds_violation_count = 0
    parsed_by_candidate = {}
    expected_keys = {
        (int(seed), int(index))
        for seed in expected_seeds
        for index in range(int(episodes_per_seed))
    }
    all_episode_apply_match = True
    explicit_application = True

    for checkpoint_name, rows in candidate_rows.items():
        parsed = {}
        for row_number, row in enumerate(rows, start=1):
            missing = [field for field in REQUIRED_DISTURBANCE_CSV_FIELDS if field not in row or row.get(field) in (None, "")]
            missing_field_count += len(missing)
            if missing:
                errors.append(f"{checkpoint_name} row {row_number}: missing fields {missing}")
                continue
            seed = _safe_int(row.get("base_seed"))
            expected_seed = _safe_int(row.get("_expected_seed", seed))
            episode_index = _safe_int(row.get("episode_index"))
            episode_seed = _safe_int(row.get("episode_seed"))
            disturbance_seed = _safe_int(row.get("disturbance_seed"))
            if seed is None or episode_index is None:
                errors.append(f"{checkpoint_name} row {row_number}: invalid episode key")
                continue
            key = (seed, episode_index)
            if key in parsed:
                errors.append(f"{checkpoint_name}: duplicate episode key {key}")
                continue
            if expected_seed is not None and seed != expected_seed:
                errors.append(f"{checkpoint_name} row {row_number}: base_seed {seed} does not match seed directory {expected_seed}")
            expected_episode_seed = seed * 1_000_003 + episode_index
            if paired_episode_seeding and episode_seed != expected_episode_seed:
                errors.append(f"{checkpoint_name} {key}: episode seed formula mismatch")
            if disturbance_seed != expected_episode_seed or disturbance_seed != episode_seed:
                errors.append(f"{checkpoint_name} {key}: disturbance seed mismatch")
            if row.get("episode_seed_mode") != EPISODE_SEED_MODE:
                errors.append(f"{checkpoint_name} {key}: episode seed mode mismatch")
            if row.get("disturbance_rng_mode") != DISTURBANCE_RNG_MODE:
                errors.append(f"{checkpoint_name} {key}: disturbance RNG mode mismatch")
            if row.get("disturbance_protocol") != protocol:
                errors.append(f"{checkpoint_name} {key}: disturbance protocol mismatch")
            applied = _parse_flag(row.get("disturbance_explicitly_applied"))
            matched = _parse_flag(row.get("disturbance_apply_match"))
            explicit_application = explicit_application and applied is True
            all_episode_apply_match = all_episode_apply_match and matched is True
            if applied is not True or matched is not True:
                errors.append(f"{checkpoint_name} {key}: explicit application/apply match is false")

            values = {}
            value_error = False
            for field in (*DISTURBANCE_KEYS, *FLOW_PHASE_KEYS):
                value = _safe_float(row.get(field))
                if value is None:
                    errors.append(f"{checkpoint_name} {key}: non-finite {field}")
                    value_error = True
                else:
                    values[field] = value
            delay = _safe_int(row.get("action_delay_steps"))
            if delay is None:
                errors.append(f"{checkpoint_name} {key}: action_delay_steps is not an integer")
                value_error = True
            else:
                values["action_delay_steps"] = delay
            for field in DISTURBANCE_KEYS:
                if field not in values:
                    continue
                low, high = DEFAULT_DISTURBANCE_BOUNDS[field]
                if values[field] < float(low) or values[field] > float(high):
                    bounds_violation_count += 1
                    errors.append(f"{checkpoint_name} {key}: {field} is outside registry bounds")
            if value_error:
                continue
            parsed[key] = {
                "episode_seed": episode_seed,
                "disturbance_seed": disturbance_seed,
                "disturbance_protocol": row.get("disturbance_protocol"),
                **values,
            }
        if set(parsed) != expected_keys:
            missing_keys = sorted(expected_keys - set(parsed))
            extra_keys = sorted(set(parsed) - expected_keys)
            errors.append(f"{checkpoint_name}: incomplete episode keys; missing={missing_keys}, extra={extra_keys}")
        parsed_by_candidate[checkpoint_name] = parsed

    names = list(candidate_rows)
    reference = parsed_by_candidate.get(names[0], {}) if names else {}
    all_candidate_keys_match = bool(names) and all(set(parsed_by_candidate[name]) == set(reference) for name in names)
    all_episode_seeds_match = all_candidate_keys_match
    all_disturbance_seeds_match = all_candidate_keys_match
    all_vectors_match = all_candidate_keys_match
    all_phases_match = all_candidate_keys_match
    for name in names[1:]:
        candidate = parsed_by_candidate[name]
        if set(candidate) != set(reference):
            continue
        for key in reference:
            left = reference[key]
            right = candidate[key]
            if left["episode_seed"] != right["episode_seed"]:
                all_episode_seeds_match = False
            if left["disturbance_seed"] != right["disturbance_seed"]:
                all_disturbance_seeds_match = False
            for field in DISTURBANCE_KEYS:
                if field == "action_delay_steps":
                    equal = left[field] == right[field]
                else:
                    equal = math.isclose(left[field], right[field], rel_tol=1e-12, abs_tol=1e-12)
                if not equal:
                    all_vectors_match = False
            for field in FLOW_PHASE_KEYS:
                if not math.isclose(left[field], right[field], rel_tol=1e-12, abs_tol=1e-12):
                    all_phases_match = False
            if left["disturbance_protocol"] != right["disturbance_protocol"]:
                all_vectors_match = False

    if not all_candidate_keys_match:
        errors.append("candidate episode key sets do not match")
    if not all_episode_seeds_match:
        errors.append("candidate episode seeds do not match")
    if not all_disturbance_seeds_match:
        errors.append("candidate disturbance seeds do not match")
    if not all_vectors_match:
        errors.append("candidate 9-D disturbance vectors do not match")
    if not all_phases_match:
        errors.append("candidate flow phases do not match")

    reference_records = list(reference.values())
    distinct_vectors = {
        tuple(record[field] for field in DISTURBANCE_KEYS) for record in reference_records
    }
    distinct_full = {
        tuple(record[field] for field in (*DISTURBANCE_KEYS, *FLOW_PHASE_KEYS))
        for record in reference_records
    }
    per_dimension = {
        field: len({record[field] for record in reference_records})
        for field in DISTURBANCE_KEYS
    }
    if len(distinct_vectors) < int(min_distinct_vectors):
        errors.append(f"distinct 9-D vector count {len(distinct_vectors)} is below {min_distinct_vectors}")
    if require_full_9d_dimension_variation:
        insufficient = [field for field, count in per_dimension.items() if count < 2]
        if insufficient:
            errors.append(f"full 9-D variation missing for {insufficient}")
    delays = sorted({int(record["action_delay_steps"]) for record in reference_records})
    lags = [record["actuator_lag"] for record in reference_records]
    noise = [record["action_noise_std"] for record in reference_records]
    audit.update({
        "overall_pass": not errors,
        "explicit_application": explicit_application,
        "all_candidate_keys_match": all_candidate_keys_match,
        "all_candidate_episode_seeds_match": all_episode_seeds_match,
        "all_candidate_disturbance_seeds_match": all_disturbance_seeds_match,
        "all_candidate_disturbance_vectors_match": all_vectors_match,
        "all_candidate_flow_phases_match": all_phases_match,
        "all_episode_apply_match": all_episode_apply_match,
        "bounds_violation_count": bounds_violation_count,
        "missing_field_count": missing_field_count,
        "distinct_9d_vector_count": len(distinct_vectors),
        "distinct_full_disturbance_count": len(distinct_full),
        "per_dimension_distinct_counts": per_dimension,
        "observed_action_delay_values": delays,
        "observed_actuator_lag_min": min(lags) if lags else None,
        "observed_actuator_lag_max": max(lags) if lags else None,
        "observed_action_noise_std_min": min(noise) if noise else None,
        "observed_action_noise_std_max": max(noise) if noise else None,
        "errors": errors,
    })
    return audit


def checkpoint_episode(path: Path, training_config=None):
    match = re.fullmatch(r"snapshot_ep(\d+)", path.stem)
    if match:
        return int(match.group(1))
    if path.name == "maddpg_uavenv_final.pt" and training_config:
        return _safe_int(training_config.get("max_episodes"))
    return None


def _candidate_sort_key(path: Path, training_config=None):
    episode = checkpoint_episode(path, training_config)
    return (episode if episode is not None else 10**12, path.name)


def _load_training_config(
    path: Optional[Path],
):
    if path is None:
        return None
    if not path.is_file():
        raise SweepError(f"training_config does not exist: {path}")
    try:
        with path.open("r", encoding="utf-8") as f:
            config = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise SweepError(f"Unable to read training_config: {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise SweepError(f"training_config must contain a JSON object: {path}")
    return config


def _audit_snapshots(model_dir: Path, training_config):
    snapshot_paths = list(model_dir.glob("snapshot_ep*.pt"))
    valid_entries = []
    invalid_names = []
    for path in snapshot_paths:
        episode = checkpoint_episode(path, training_config)
        if episode is None:
            invalid_names.append(path.name)
        else:
            valid_entries.append((episode, path))
    valid_entries.sort(key=lambda item: (item[0], item[1].name))

    episode_counts = Counter(episode for episode, _path in valid_entries)
    found_episodes = sorted(episode_counts)
    duplicate_episodes = sorted(
        episode for episode, count in episode_counts.items() if count > 1
    )

    max_episodes = _safe_int(training_config.get("max_episodes")) if training_config else None
    snapshot_interval = (
        _safe_int(training_config.get("snapshot_interval")) if training_config else None
    )
    expected_episodes = []
    if max_episodes is not None and snapshot_interval is not None and snapshot_interval > 0:
        expected_episodes = list(range(snapshot_interval, max_episodes + 1, snapshot_interval))
    missing_episodes = sorted(set(expected_episodes) - set(found_episodes))
    extra_episodes = sorted(set(found_episodes) - set(expected_episodes)) if expected_episodes else []
    complete_sequence = bool(expected_episodes) and (
        found_episodes == expected_episodes
        and not duplicate_episodes
        and not invalid_names
    )
    last_snapshot_episode = valid_entries[-1][0] if valid_entries else None

    return {
        "expected_snapshot_episodes": expected_episodes,
        "found_snapshot_episodes": found_episodes,
        "found_snapshot_names": [path.stem for _episode, path in valid_entries],
        "missing_snapshot_episodes": missing_episodes,
        "extra_snapshot_episodes": extra_episodes,
        "duplicate_snapshot_episodes": duplicate_episodes,
        "invalid_snapshot_names": sorted(invalid_names),
        "last_snapshot_episode": last_snapshot_episode,
        "complete_sequence": complete_sequence,
    }, valid_entries


def _audit_final_model(
    model_dir: Path,
    training_config,
    valid_entries,
    allow_dedup,
):
    final_path = model_dir / "maddpg_uavenv_final.pt"
    final_found = final_path.is_file()
    max_episodes = _safe_int(training_config.get("max_episodes")) if training_config else None
    snapshot_interval = (
        _safe_int(training_config.get("snapshot_interval")) if training_config else None
    )
    equivalent_path = next(
        (path for episode, path in valid_entries if episode == max_episodes),
        None,
    )
    final_deduplicated = bool(
        allow_dedup
        and final_found
        and equivalent_path is not None
        and max_episodes is not None
        and snapshot_interval is not None
        and snapshot_interval > 0
        and max_episodes % snapshot_interval == 0
    )
    return {
        "final_model_found": final_found,
        "final_deduplicated": final_deduplicated,
        "final_equivalent_snapshot": equivalent_path.stem if final_deduplicated else None,
        "final_model_path": str(final_path.resolve()) if final_found else str(final_path),
    }, final_path


def _collect_candidates(
    valid_entries,
    final_path,
    final_audit,
    training_config,
    checkpoint_names,
):
    candidates = [path for _episode, path in valid_entries]
    if final_audit["final_model_found"] and not final_audit["final_deduplicated"]:
        candidates.append(final_path)
    candidates.sort(key=lambda path: _candidate_sort_key(path, training_config))

    if checkpoint_names:
        requested = list(checkpoint_names)
        if len(set(requested)) != len(requested):
            raise SweepError(f"Duplicate --checkpoint-names values: {requested}")
        by_name = {path.stem: path for path in candidates}
        missing = [name for name in requested if name not in by_name]
        if missing:
            raise SweepError(f"Requested checkpoints were not found: {missing}")
        candidates = [by_name[name] for name in requested]
        candidates.sort(key=lambda path: _candidate_sort_key(path, training_config))

    if not candidates:
        raise SweepError("No candidate checkpoints were selected.")
    return candidates


def _run(cmd):
    print("[RUN]", " ".join(str(item) for item in cmd), flush=True)
    subprocess.run([str(item) for item in cmd], cwd=str(PROJECT_ROOT), check=True)


def _validate_selection_policy(args):
    if args.selection_policy == "clean_nominal_coarse":
        if args.ablation_mode != "ch4_pse_baseline":
            raise SweepError(
                "clean_nominal_coarse requires ablation_mode=ch4_pse_baseline"
            )
        if args.profile != "normal_comm":
            raise SweepError("clean_nominal_coarse requires profile=normal_comm")


def _validate_aggregate(aggregate, checkpoint_name, expected_seeds, args):
    if aggregate.get("method") != args.ablation_mode:
        raise SweepError(
            f"{checkpoint_name}: aggregate method {aggregate.get('method')} "
            f"does not match {args.ablation_mode}"
        )
    if aggregate.get("profile") != args.profile:
        raise SweepError(
            f"{checkpoint_name}: aggregate profile {aggregate.get('profile')} "
            f"does not match {args.profile}"
        )
    if list(aggregate.get("seeds") or []) != list(expected_seeds):
        raise SweepError(
            f"{checkpoint_name}: aggregate seeds {aggregate.get('seeds')} "
            f"do not match validation seeds {list(expected_seeds)}"
        )
    invalid_rates = [
        field
        for field in ("success_rate_weighted", "found_rate_weighted")
        if _safe_float(aggregate.get(field)) is None
    ]
    if invalid_rates:
        raise SweepError(
            f"{checkpoint_name}: missing or non-finite aggregate rates: {invalid_rates}"
        )

    total_found = _safe_int(aggregate.get("total_found"))
    total_success = _safe_int(aggregate.get("total_success"))
    if total_found is None or total_success is None or total_found < 0 or total_success < 0:
        raise SweepError(
            f"{checkpoint_name}: invalid total_success/total_found: "
            f"{total_success}/{total_found}"
        )
    if total_success > total_found:
        raise SweepError(
            f"{checkpoint_name}: total_success={total_success} exceeds "
            f"total_found={total_found}"
        )
    succ_if_found = _safe_float(aggregate.get("succ_if_found_weighted"))
    if total_found > 0:
        expected_sif = float(total_success) / float(total_found)
        if succ_if_found is None or not math.isclose(
            succ_if_found,
            expected_sif,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise SweepError(
                f"{checkpoint_name}: succ_if_found_weighted={succ_if_found} "
                f"does not match {expected_sif}"
            )
    elif total_success != 0 or aggregate.get("succ_if_found_weighted") is not None:
        raise SweepError(
            f"{checkpoint_name}: zero-found aggregate requires total_success=0 "
            "and succ_if_found_weighted=null"
        )

    if args.selection_policy == "clean_nominal_coarse":
        invalid_continuous = [
            field
            for field in WEIGHTED_CONTINUOUS_FIELDS
            if _safe_float(aggregate.get(field)) is None
        ]
        if invalid_continuous:
            raise SweepError(
                f"{checkpoint_name}: missing or non-finite continuous metrics: "
                f"{invalid_continuous}"
            )
    elif _safe_float(aggregate.get(args.select_metric)) is None:
        raise SweepError(
            f"{checkpoint_name}: legacy select metric {args.select_metric} "
            "is missing or non-finite"
        )
    total_episodes = _safe_int(aggregate.get("total_episodes"))
    if total_episodes is None or total_episodes <= 0:
        raise SweepError(f"{checkpoint_name}: invalid total_episodes={total_episodes}")


def _clean_ranking_key(row):
    def descending(field):
        value = _safe_float(row.get(field))
        return math.inf if value is None else -value

    def ascending(field):
        value = _safe_float(row.get(field))
        return math.inf if value is None else value

    episode = _safe_int(row.get("checkpoint_episode"))
    return (
        descending("success_rate_weighted"),
        descending("found_rate_weighted"),
        descending("succ_if_found_weighted"),
        ascending("avg_safety_cost_weighted"),
        ascending("avg_final_distance_weighted"),
        episode if episode is not None else 10**12,
    )


def _legacy_ranking_key(row, select_metric):
    value = _safe_float(row.get(select_metric))
    return (math.inf if value is None else -value, row.get("checkpoint_name", ""))


def _write_csv(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=TABLE_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in TABLE_FIELDS})


def _fmt(value, digits=6):
    number = _safe_float(value)
    return "" if number is None else f"{number:.{digits}f}"


def _write_md(path: Path, summary, rows):
    is_clean = summary.get("selection_policy") == "clean_nominal_coarse"
    lines = [
        "# Checkpoint Sweep Summary",
        "",
        f"- overall_pass: `{summary.get('overall_pass')}`",
        f"- selection_policy: `{summary.get('selection_policy')}`",
        f"- experiment_type: `{summary.get('experiment_type')}`",
        f"- selection_stage: `{summary.get('selection_stage')}`",
        f"- candidate_count: `{summary.get('candidate_count', 0)}`",
        f"- best_checkpoint_name: `{summary.get('best_checkpoint_name')}`",
        f"- validation_seeds: `{summary.get('settings', {}).get('validation_seeds')}`",
        f"- seed_leakage_detected: `{summary.get('seed_leakage_detected')}`",
        "",
    ]
    if is_clean:
        lines.extend([
            "This is only the coarse rank-1 candidate, not the final selected model.",
            "",
        ])
    if summary.get("errors"):
        lines.extend(["## Errors", ""])
        lines.extend(f"- {error}" for error in summary["errors"])
        lines.append("")
    lines.extend([
        "## Ranking Rule",
        "",
        *[f"- {item}" for item in summary.get("ranking_rule", [])],
        "",
        "## Ranked Checkpoints",
        "",
        "| rank | checkpoint | episode | success | found | success-if-found | safety cost | final distance | top-k |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    for row in rows:
        lines.append(
            "| {rank} | {name} | {episode} | {success} | {found} | {sif} | {safety} | {distance} | {top_k} |".format(
                rank=row.get("rank", ""),
                name=row.get("checkpoint_name", ""),
                episode=row.get("checkpoint_episode", ""),
                success=_fmt(row.get("success_rate_weighted")),
                found=_fmt(row.get("found_rate_weighted")),
                sif=_fmt(row.get("succ_if_found_weighted")),
                safety=_fmt(row.get("avg_safety_cost_weighted")),
                distance=_fmt(row.get("avg_final_distance_weighted")),
                top_k=row.get("is_top_k", False),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _base_summary(args, model_dir, training_config_path, seed_leakage_detected):
    is_clean = args.selection_policy == "clean_nominal_coarse"
    ranking_rule = (
        list(CLEAN_RANKING_RULE)
        if is_clean
        else [f"{args.select_metric} desc"]
    )
    return {
        "overall_pass": False,
        "selection_policy": args.selection_policy,
        "experiment_type": EXPERIMENT_TYPE if is_clean else "checkpoint_sweep",
        "selection_stage": SELECTION_STAGE if is_clean else "legacy",
        "model_dir": str(model_dir),
        "training_config": str(training_config_path) if training_config_path else None,
        "settings": {
            "ablation_mode": args.ablation_mode,
            "profile": args.profile,
            "selection_policy": args.selection_policy,
            "select_metric": args.select_metric,
            "validation_seeds": list(args.seeds),
            "forbidden_test_seeds": list(args.forbid_seeds or []),
            "episodes_per_seed": args.episodes,
            "max_steps": args.max_steps,
            "top_k": args.top_k,
            "checkpoint_names": list(args.checkpoint_names or []),
            "paired_episode_seeding": bool(args.paired_episode_seeding),
            "disturbance_protocol": args.disturbance_protocol,
            "require_paired_disturbances": bool(args.require_paired_disturbances),
            "disturbance_audit_min_distinct_vectors": int(args.disturbance_audit_min_distinct_vectors),
            "require_full_9d_dimension_variation": bool(args.require_full_9d_dimension_variation),
            "require_complete_snapshot_sequence": bool(
                args.require_complete_snapshot_sequence
            ),
        },
        "validation_seeds": list(args.seeds),
        "forbidden_test_seeds": list(args.forbid_seeds or []),
        "seed_leakage_detected": seed_leakage_detected,
        "ranking_rule": ranking_rule,
        "candidate_count": 0,
        "ranked_checkpoints": [],
        "top_k_candidates": [],
        "best_checkpoint_name": None,
        "best_checkpoint_path": None,
        "select_metric": args.select_metric,
        "best_metric_value": None,
        "checkpoints": [],
        "errors": [],
        "disturbance_protocol_audit": _empty_disturbance_audit(args),
    }


def _evaluate_candidates(args, candidates, training_config, result_root):
    rows = []
    aggregates = []
    candidate_episode_rows = {}
    for checkpoint in candidates:
        checkpoint_name = checkpoint.stem
        checkpoint_result_dir = result_root / checkpoint_name
        if checkpoint_result_dir.exists():
            raise SweepError(
                f"Checkpoint result directory already exists: {checkpoint_result_dir}"
            )
        checkpoint_result_dir.mkdir(parents=True)
        checkpoint_episode_rows = []
        for seed in args.seeds:
            seed_dir = checkpoint_result_dir / f"seed{seed}"
            eval_cmd = [
                args.python_exe,
                "evaluate_pse.py",
                "--ablation_mode",
                args.ablation_mode,
                "--model_path",
                str(checkpoint),
                "--episodes",
                str(args.episodes),
                "--max_steps",
                str(args.max_steps),
                "--seed",
                str(seed),
                "--profile",
                args.profile,
                "--result_dir",
                str(seed_dir),
                "--disturbance-protocol",
                args.disturbance_protocol,
            ]
            if args.paired_episode_seeding:
                eval_cmd.append("--paired-episode-seeding")
            _run(eval_cmd)
            if not (seed_dir / "evaluation_summary.json").is_file():
                raise SweepError(
                    f"{checkpoint_name}/seed{seed}: evaluation_summary.json is missing"
                )
            with (seed_dir / "evaluation_summary.json").open("r", encoding="utf-8") as f:
                evaluation_summary = json.load(f)
            _audit_evaluation_summary(evaluation_summary, checkpoint_name, seed, args)
            checkpoint_episode_rows.extend(
                _load_episode_disturbance_rows(
                    seed_dir / "episode_metrics.csv",
                    checkpoint_name,
                    seed,
                )
            )
        _run(
            [
                args.python_exe,
                "tools/summarize_seed_eval_results.py",
                "--result-dir",
                str(checkpoint_result_dir),
                "--name",
                checkpoint_name,
            ]
        )
        aggregate_path = checkpoint_result_dir / "aggregate_summary.json"
        if not aggregate_path.is_file():
            raise SweepError(f"{checkpoint_name}: aggregate_summary.json is missing")
        with aggregate_path.open("r", encoding="utf-8") as f:
            aggregate = json.load(f)
        _validate_aggregate(aggregate, checkpoint_name, args.seeds, args)

        row = {
            "checkpoint_name": checkpoint_name,
            "checkpoint_episode": checkpoint_episode(checkpoint, training_config),
            "checkpoint_path": str(checkpoint),
            "aggregate_summary_path": str(aggregate_path),
        }
        for field in TABLE_FIELDS:
            if field in aggregate:
                row[field] = aggregate[field]
        rows.append(_json_safe(row))
        aggregates.append({"checkpoint": dict(row), "aggregate": aggregate})
        candidate_episode_rows[checkpoint_name] = checkpoint_episode_rows
    return rows, aggregates, candidate_episode_rows


def run_sweep(args, result_root):
    model_dir = Path(args.model_dir).resolve()
    training_config_path = (
        Path(args.training_config).resolve() if args.training_config else None
    )
    forbidden = set(args.forbid_seeds or [])
    leakage = sorted(set(args.seeds).intersection(forbidden))
    summary = _base_summary(
        args,
        model_dir,
        training_config_path,
        seed_leakage_detected=bool(leakage),
    )

    if not model_dir.is_dir():
        raise SweepError(f"model_dir does not exist: {model_dir}")
    if len(set(args.seeds)) != len(args.seeds):
        raise SweepError(f"Duplicate validation seeds are not allowed: {args.seeds}")
    if args.disturbance_audit_min_distinct_vectors < 1:
        raise SweepError("--disturbance-audit-min-distinct-vectors must be >= 1")
    if args.require_paired_disturbances:
        if not args.paired_episode_seeding:
            raise SweepError("--require-paired-disturbances requires --paired-episode-seeding")
        if args.disturbance_protocol != DISTURBANCE_PROTOCOL_UNIFORM_9D:
            raise SweepError(
                "--require-paired-disturbances requires --disturbance-protocol uniform_9d_registry_v1"
            )
        if not args.seeds:
            raise SweepError("--require-paired-disturbances requires a non-empty seed list")
    if leakage:
        raise SweepError(f"Validation seeds overlap forbidden test seeds: {leakage}")
    _validate_selection_policy(args)
    if args.episodes <= 0 or args.max_steps <= 0 or args.top_k <= 0:
        raise SweepError("episodes, max_steps, and top_k must all be positive")

    training_config = _load_training_config(training_config_path)
    if (
        args.selection_policy == "clean_nominal_coarse"
        and training_config
        and training_config.get("mode") != "ch4_pse_baseline"
    ):
        raise SweepError(
            f"training_config mode must be ch4_pse_baseline, got {training_config.get('mode')}"
        )
    if args.require_complete_snapshot_sequence and training_config is None:
        raise SweepError(
            "--require-complete-snapshot-sequence requires --training-config"
        )

    snapshot_audit, valid_entries = _audit_snapshots(model_dir, training_config)
    summary["snapshot_audit"] = snapshot_audit
    if args.require_complete_snapshot_sequence and not snapshot_audit["complete_sequence"]:
        raise SweepError(
            "Snapshot sequence audit failed: "
            f"missing={snapshot_audit['missing_snapshot_episodes']}, "
            f"duplicates={snapshot_audit['duplicate_snapshot_episodes']}, "
            f"invalid={snapshot_audit['invalid_snapshot_names']}, "
            f"extra={snapshot_audit['extra_snapshot_episodes']}"
        )

    final_audit, final_path = _audit_final_model(
        model_dir,
        training_config,
        valid_entries,
        allow_dedup=args.selection_policy == "clean_nominal_coarse",
    )
    summary["final_model_audit"] = final_audit
    candidates = _collect_candidates(
        valid_entries,
        final_path,
        final_audit,
        training_config,
        args.checkpoint_names,
    )
    if (
        args.selection_policy == "clean_nominal_coarse"
        and len(candidates) < args.top_k
    ):
        raise SweepError(
            f"candidate_count={len(candidates)} is smaller than top_k={args.top_k}"
        )

    rows, aggregates, candidate_episode_rows = _evaluate_candidates(
        args,
        candidates,
        training_config,
        result_root,
    )
    disturbance_audit = audit_paired_disturbance_records(
        candidate_episode_rows,
        expected_seeds=args.seeds,
        episodes_per_seed=args.episodes,
        protocol=_resolved_requested_protocol(args),
        min_distinct_vectors=args.disturbance_audit_min_distinct_vectors,
        require_full_9d_dimension_variation=args.require_full_9d_dimension_variation,
        paired_episode_seeding=args.paired_episode_seeding,
    )
    summary["disturbance_protocol_audit"] = disturbance_audit
    if args.require_paired_disturbances and not disturbance_audit["overall_pass"]:
        raise SweepError(f"Paired disturbance audit failed: {disturbance_audit['errors']}")
    if args.selection_policy == "clean_nominal_coarse":
        ranked_rows = sorted(rows, key=_clean_ranking_key)
        effective_top_k = args.top_k
    else:
        ranked_rows = sorted(
            rows,
            key=lambda row: _legacy_ranking_key(row, args.select_metric),
        )
        effective_top_k = min(args.top_k, len(ranked_rows))
    for rank, row in enumerate(ranked_rows, start=1):
        row["rank"] = rank
        row["is_top_k"] = rank <= effective_top_k

    top_rows = ranked_rows[:effective_top_k]
    if (
        args.selection_policy == "clean_nominal_coarse"
        and len(top_rows) != args.top_k
    ):
        raise SweepError(
            f"Expected {args.top_k} top-k candidates, got {len(top_rows)}"
        )
    top_k_candidates = [
        {
            "rank": row["rank"],
            "checkpoint_name": row["checkpoint_name"],
            "checkpoint_episode": row["checkpoint_episode"],
            "checkpoint_path": row["checkpoint_path"],
            "success": row.get("success_rate_weighted"),
            "found": row.get("found_rate_weighted"),
            "success_if_found": row.get("succ_if_found_weighted"),
            "safety_cost": row.get("avg_safety_cost_weighted"),
            "final_distance": row.get("avg_final_distance_weighted"),
        }
        for row in top_rows
    ]

    summary.update(
        {
            "overall_pass": True,
            "candidate_count": len(ranked_rows),
            "ranked_checkpoints": ranked_rows,
            "top_k_candidates": top_k_candidates,
            "checkpoint_aggregates": aggregates,
            "best_checkpoint_name": ranked_rows[0]["checkpoint_name"],
            "best_checkpoint_path": ranked_rows[0]["checkpoint_path"],
            "select_metric": args.select_metric,
            "best_metric_value": _safe_float(
                ranked_rows[0].get(args.select_metric)
            ),
            "checkpoints": aggregates,
            "errors": [],
            "disturbance_protocol_audit": disturbance_audit,
        }
    )
    return _json_safe(summary), ranked_rows


def _write_outputs(result_root, summary, rows):
    json_path = result_root / "checkpoint_sweep_summary.json"
    csv_path = result_root / "checkpoint_sweep_table.csv"
    md_path = result_root / "checkpoint_sweep_summary.md"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(summary), f, indent=2, ensure_ascii=False, allow_nan=False)
    _write_csv(csv_path, rows)
    _write_md(md_path, summary, rows)
    return json_path, csv_path, md_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Sweep Chapter-4 checkpoints across seeds."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--training-config", default=None)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--ablation-mode", required=True)
    parser.add_argument(
        "--selection-policy",
        choices=("legacy", "clean_nominal_coarse"),
        default="legacy",
    )
    parser.add_argument("--select-metric", default="success_rate_weighted")
    parser.add_argument("--profile", default="normal_comm")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--forbid-seeds", type=int, nargs="+", default=[])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--checkpoint-names", nargs="+", default=None)
    parser.add_argument("--paired-episode-seeding", action="store_true")
    parser.add_argument(
        "--disturbance-protocol",
        choices=SUPPORTED_DISTURBANCE_PROTOCOLS,
        default=DISTURBANCE_PROTOCOL_AUTO,
    )
    parser.add_argument("--require-paired-disturbances", action="store_true")
    parser.add_argument(
        "--disturbance-audit-min-distinct-vectors",
        type=int,
        default=1,
    )
    parser.add_argument("--require-full-9d-dimension-variation", action="store_true")
    parser.add_argument("--require-complete-snapshot-sequence", action="store_true")
    parser.add_argument("--python-exe", default=sys.executable)
    return parser.parse_args(argv)


def main():
    args = parse_args()
    result_root = Path(args.result_root).resolve()
    result_root.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir).resolve()
    config_path = Path(args.training_config).resolve() if args.training_config else None
    leakage = bool(set(args.seeds).intersection(set(args.forbid_seeds or [])))
    try:
        summary, rows = run_sweep(args, result_root)
        paths = _write_outputs(result_root, summary, rows)
        if not all(path.is_file() for path in paths):
            raise SweepError("One or more root sweep outputs are missing after write")
        print(
            "[OK] checkpoint sweep: "
            f"selection_policy={summary['selection_policy']} "
            f"candidates={summary['candidate_count']} "
            f"top_k={len(summary['top_k_candidates'])}"
        )
        return 0
    except Exception as exc:
        summary = _base_summary(args, model_dir, config_path, leakage)
        summary["errors"] = [f"{type(exc).__name__}: {exc}"]
        try:
            _write_outputs(result_root, summary, [])
        except Exception as write_exc:
            print(f"[ERROR] Unable to write failure summary: {write_exc}", file=sys.stderr)
        print(f"[ERROR] checkpoint sweep failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
