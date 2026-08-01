# -*- coding: utf-8 -*-
"""Shared utilities for Perf-RBE ablation checkpoint selection and comparison."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from registry.ch4_artifact_layout import source_hash_matches_layout_attestation
EVAL_SCRIPT = PROJECT_ROOT / "evaluate_pse.py"

DISTURBANCE_FIELDS = (
    "episode_seed", "disturbance_seed", "flow_gain", "flow_z_gain",
    "drag_scale", "buoyancy_bias_delta", "a_max_scale", "v_max_scale",
    "actuator_lag", "action_delay_steps", "action_noise_std",
    "flow_phase_x", "flow_phase_y",
)
SOURCE_LOCK_FILES = (
    "env.py", "train.py", "evaluate_pse.py", "algorithms/maddpg.py",
    "map/map_module.py", "registry/rbe_disturbance.py",
    "utils/agents.py", "utils/networks.py", "utils/noise.py",
    "utils/rbe_metrics.py",
)

class ExperimentError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ExperimentError(message)


def sha256(path: Path | str) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path | str) -> Dict[str, Any]:
    path = Path(path)
    require(path.is_file(), f"JSON missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    require(isinstance(data, dict), f"JSON root must be an object: {path}")
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


def write_json(path: Path | str, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(data), handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_csv(path: Path | str) -> List[Dict[str, str]]:
    path = Path(path)
    require(path.is_file(), f"CSV missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path | str, rows: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def safe_int(value: Any) -> int | None:
    number = safe_float(value)
    if number is None or not float(number).is_integer():
        return None
    return int(number)


def parse_flag(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    text = str(value).strip().lower()
    if text in {"1", "1.0", "true", "yes", "y"}:
        return 1
    if text in {"0", "0.0", "false", "no", "n", ""}:
        return 0
    raise ExperimentError(f"invalid boolean flag: {value!r}")


def episode_key(row: Mapping[str, Any]) -> Tuple[int, int]:
    seed = safe_int(row.get("base_seed", row.get("seed")))
    index = safe_int(row.get("episode_index"))
    require(seed is not None and index is not None, "episode row has invalid seed/index")
    return seed, index


def metric_from_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    rows = list(rows)
    episodes = len(rows)
    success = sum(parse_flag(row.get("success_flag", row.get("success", 0))) for row in rows)
    found = sum(parse_flag(row.get("found_flag", row.get("found", 0))) for row in rows)
    found_but_failed = found - success

    def mean(field: str) -> float | None:
        values = [safe_float(row.get(field)) for row in rows]
        values = [value for value in values if value is not None]
        return None if not values else float(sum(values) / len(values))

    return {
        "episodes": episodes,
        "n_success": success,
        "n_found": found,
        "n_not_found": episodes - found,
        "n_found_but_failed": found_but_failed,
        "success_rate": None if episodes == 0 else success / episodes,
        "found_rate": None if episodes == 0 else found / episodes,
        "succ_if_found": None if found == 0 else success / found,
        "not_found_rate": None if episodes == 0 else (episodes - found) / episodes,
        "found_but_failed_rate": None if episodes == 0 else found_but_failed / episodes,
        "avg_reward": mean("reward"),
        "avg_recovery_time": mean("recovery_time"),
        "avg_safety_cost": mean("safety_cost"),
        "avg_final_distance": mean("final_distance"),
        "avg_final_nav_distance": mean("final_nav_distance"),
        "avg_action_smoothness": mean("action_smoothness"),
        "avg_completion_steps": mean("completion_steps"),
    }


def audit_evaluation_unit(
    unit_dir: Path,
    *,
    model_path: Path,
    seed: int,
    episodes: int,
    max_steps: int,
    protocol: str,
) -> Dict[str, Any]:
    summary_path = unit_dir / "evaluation_summary.json"
    csv_path = unit_dir / "episode_metrics.csv"
    summary = read_json(summary_path)
    rows = read_csv(csv_path)
    require(len(rows) == episodes, f"{unit_dir}: expected {episodes} rows, got {len(rows)}")
    require(summary.get("method") == "ch4_rbe_full", f"{unit_dir}: invalid method")
    require(summary.get("profile") == "normal_comm", f"{unit_dir}: invalid profile")
    require(summary.get("episodes") == episodes, f"{unit_dir}: invalid episode count")
    require(summary.get("max_steps") == max_steps, f"{unit_dir}: invalid max_steps")
    require(summary.get("seed") == seed, f"{unit_dir}: invalid seed")
    require(summary.get("paired_episode_seeding") is True, f"{unit_dir}: paired seeding disabled")
    require(summary.get("episode_seed_mode") == "indexed_common_random_numbers", f"{unit_dir}: invalid seed mode")
    require(summary.get("disturbance_protocol") == protocol, f"{unit_dir}: invalid protocol")
    require(Path(str(summary.get("model_path"))).resolve() == model_path.resolve(), f"{unit_dir}: model path mismatch")
    require(summary.get("all_episode_disturbance_apply_match") is True, f"{unit_dir}: disturbance application mismatch")
    require(summary.get("bounds_violation_count") == 0, f"{unit_dir}: disturbance bounds violation")

    seen = set()
    for row in rows:
        key = episode_key(row)
        require(key[0] == seed, f"{unit_dir}: base seed mismatch")
        require(0 <= key[1] < episodes, f"{unit_dir}: episode index out of range")
        require(key not in seen, f"{unit_dir}: duplicate episode key {key}")
        seen.add(key)
        expected_episode_seed = seed * 1_000_003 + key[1]
        require(safe_int(row.get("episode_seed")) == expected_episode_seed, f"{unit_dir}: invalid episode seed")
        success = parse_flag(row.get("success_flag", row.get("success", 0)))
        found = parse_flag(row.get("found_flag", row.get("found", 0)))
        require(success <= found, f"{unit_dir}: success exceeds found")
    require({key[1] for key in seen} == set(range(episodes)), f"{unit_dir}: incomplete episode indexes")
    return {"overall_pass": True, "row_count": len(rows), "summary_sha256": sha256(summary_path), "csv_sha256": sha256(csv_path)}


def unit_is_valid(**kwargs: Any) -> bool:
    try:
        audit_evaluation_unit(**kwargs)
        return True
    except Exception:
        return False


def ensure_evaluation(
    *,
    model_path: Path,
    seed: int,
    episodes: int,
    max_steps: int,
    protocol: str,
    result_dir: Path,
) -> None:
    args = dict(
        unit_dir=result_dir,
        model_path=model_path,
        seed=seed,
        episodes=episodes,
        max_steps=max_steps,
        protocol=protocol,
    )
    if result_dir.exists() and unit_is_valid(**args):
        print(f"[AblationEval] reuse {result_dir}")
        return
    if result_dir.exists():
        shutil.rmtree(result_dir)
    result_dir.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(EVAL_SCRIPT),
        "--ablation_mode", "ch4_rbe_full",
        "--model_path", str(model_path),
        "--episodes", str(episodes),
        "--seed", str(seed),
        "--paired-episode-seeding",
        "--disturbance-protocol", protocol,
        "--max_steps", str(max_steps),
        "--profile", "normal_comm",
        "--result_dir", str(result_dir),
    ]
    print("[AblationEval] start", " ".join(command))
    completed = subprocess.run(command, cwd=str(PROJECT_ROOT))
    require(completed.returncode == 0, f"evaluate_pse failed with exit code {completed.returncode}")
    audit_evaluation_unit(**args)


def pool_units(unit_dirs: Iterable[Path]) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    rows: List[Dict[str, str]] = []
    for unit_dir in unit_dirs:
        rows.extend(read_csv(unit_dir / "episode_metrics.csv"))
    return metric_from_rows(rows), rows


def pairing_audit(rows_by_model: Mapping[str, Sequence[Mapping[str, Any]]]) -> Dict[str, Any]:
    model_ids = sorted(rows_by_model)
    require(model_ids, "pairing audit requires at least one model")
    lookups = {
        model_id: {episode_key(row): row for row in rows}
        for model_id, rows in rows_by_model.items()
    }
    baseline = model_ids[0]
    expected_keys = set(lookups[baseline])
    mismatches: List[Dict[str, Any]] = []
    for model_id in model_ids[1:]:
        if set(lookups[model_id]) != expected_keys:
            mismatches.append({"model_id": model_id, "reason": "episode_key_set"})
            continue
        for key in sorted(expected_keys):
            left = lookups[baseline][key]
            right = lookups[model_id][key]
            for field in DISTURBANCE_FIELDS:
                a, b = left.get(field), right.get(field)
                af, bf = safe_float(a), safe_float(b)
                matches = (
                    math.isclose(af, bf, rel_tol=1e-9, abs_tol=1e-10)
                    if af is not None and bf is not None
                    else str(a) == str(b)
                )
                if not matches:
                    mismatches.append({"model_id": model_id, "episode_key": key, "field": field, "baseline": a, "candidate": b})
                    break
            if len(mismatches) >= 20:
                break
    return {
        "overall_pass": not mismatches,
        "model_count": len(model_ids),
        "paired_episode_count_per_model": len(expected_keys),
        "mismatch_count": len(mismatches),
        "mismatch_examples": mismatches[:20],
    }


def _metric_arrays(rows: Sequence[Mapping[str, Any]], metric: str) -> Dict[int, Dict[str, np.ndarray]]:
    grouped: Dict[int, List[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(episode_key(row)[0], []).append(row)
    output: Dict[int, Dict[str, np.ndarray]] = {}
    for seed, seed_rows in grouped.items():
        seed_rows = sorted(seed_rows, key=lambda row: episode_key(row)[1])
        success = np.asarray([parse_flag(row.get("success_flag", row.get("success", 0))) for row in seed_rows], dtype=np.float64)
        found = np.asarray([parse_flag(row.get("found_flag", row.get("found", 0))) for row in seed_rows], dtype=np.float64)
        if metric == "success":
            value = success
        elif metric == "found":
            value = found
        elif metric == "not_found":
            value = 1.0 - found
        elif metric == "found_but_failed":
            value = found - success
        elif metric == "succ_if_found":
            value = np.zeros_like(success)
        else:
            field = {
                "reward": "reward",
                "recovery_time": "recovery_time",
                "safety_cost": "safety_cost",
                "final_distance": "final_distance",
                "final_nav_distance": "final_nav_distance",
                "action_smoothness": "action_smoothness",
                "completion_steps": "completion_steps",
            }[metric]
            values = [safe_float(row.get(field)) for row in seed_rows]
            require(all(value is not None for value in values), f"non-finite {metric} in seed {seed}")
            value = np.asarray(values, dtype=np.float64)
        output[seed] = {"value": value, "success": success, "found": found}
    return output


def paired_bootstrap(
    candidate_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    metric: str,
    *,
    repetitions: int = 20000,
    seed: int = 20260728,
) -> Dict[str, Any]:
    candidate = _metric_arrays(candidate_rows, metric)
    baseline = _metric_arrays(baseline_rows, metric)
    require(set(candidate) == set(baseline), "paired bootstrap seed sets differ")
    rng = np.random.default_rng(seed)

    if metric == "succ_if_found":
        cand_succ_total = 0.0
        cand_found_total = 0.0
        base_succ_total = 0.0
        base_found_total = 0.0
        boot_cand_succ = np.zeros(repetitions, dtype=np.float64)
        boot_cand_found = np.zeros(repetitions, dtype=np.float64)
        boot_base_succ = np.zeros(repetitions, dtype=np.float64)
        boot_base_found = np.zeros(repetitions, dtype=np.float64)
        for group_seed in sorted(candidate):
            c = candidate[group_seed]
            b = baseline[group_seed]
            require(len(c["success"]) == len(b["success"]), "paired bootstrap row counts differ")
            n = len(c["success"])
            indexes = rng.integers(0, n, size=(repetitions, n), dtype=np.int32)
            cand_succ_total += float(c["success"].sum())
            cand_found_total += float(c["found"].sum())
            base_succ_total += float(b["success"].sum())
            base_found_total += float(b["found"].sum())
            boot_cand_succ += c["success"][indexes].sum(axis=1)
            boot_cand_found += c["found"][indexes].sum(axis=1)
            boot_base_succ += b["success"][indexes].sum(axis=1)
            boot_base_found += b["found"][indexes].sum(axis=1)
        require(cand_found_total > 0 and base_found_total > 0, "SIF unavailable: zero found denominator")
        point = cand_succ_total / cand_found_total - base_succ_total / base_found_total
        valid = (boot_cand_found > 0) & (boot_base_found > 0)
        values = boot_cand_succ[valid] / boot_cand_found[valid] - boot_base_succ[valid] / boot_base_found[valid]
    else:
        point_numerator = 0.0
        point_denominator = 0
        boot_sum = np.zeros(repetitions, dtype=np.float64)
        boot_count = 0
        for group_seed in sorted(candidate):
            c = candidate[group_seed]["value"]
            b = baseline[group_seed]["value"]
            require(len(c) == len(b), "paired bootstrap row counts differ")
            difference = c - b
            n = len(difference)
            indexes = rng.integers(0, n, size=(repetitions, n), dtype=np.int32)
            point_numerator += float(difference.sum())
            point_denominator += n
            boot_sum += difference[indexes].sum(axis=1)
            boot_count += n
        require(point_denominator > 0 and boot_count > 0, "paired bootstrap has no rows")
        point = point_numerator / point_denominator
        values = boot_sum / boot_count

    require(values.size > 0, f"no valid bootstrap repetitions for metric={metric}")
    return {
        "metric": metric,
        "candidate_minus_baseline": float(point),
        "ci95_low": float(np.quantile(values, 0.025)),
        "ci95_high": float(np.quantile(values, 0.975)),
        "probability_delta_positive": float(np.mean(values > 0.0)),
        "bootstrap_method": "seed-stratified paired percentile bootstrap",
        "bootstrap_repetitions_requested": repetitions,
        "bootstrap_repetitions_valid": int(values.size),
        "bootstrap_seed": int(seed),
    }


def model_load_audit(path: Path) -> Dict[str, Any]:
    import torch
    from algorithms.maddpg import MADDPG

    require(path.is_file(), f"model missing: {path}")
    model = MADDPG.init_from_save(str(path), device="cpu")
    observations = [torch.zeros((1, int(params["num_in_pol"])), dtype=torch.float32) for params in model.agent_init_params]
    actions = model.step(observations, explore=False)
    finite = all(bool(torch.isfinite(action).all().item()) for action in actions)
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "sha256": sha256(path),
        "size": path.stat().st_size,
        "agent_count": model.nagents,
        "obs_dims": [int(params["num_in_pol"]) for params in model.agent_init_params],
        "action_dims": [int(params["num_out_pol"]) for params in model.agent_init_params],
        "fixed_forward_pass": finite,
        "overall_pass": finite and model.nagents == 4,
    }


def build_source_lock(extra_files: Sequence[Path] = ()) -> Dict[str, Any]:
    files: Dict[str, str] = {}
    for relative in SOURCE_LOCK_FILES:
        path = PROJECT_ROOT / relative
        require(path.is_file(), f"source-lock file missing: {path}")
        files[relative] = sha256(path)
    for path in extra_files:
        path = Path(path)
        require(path.is_file(), f"source-lock extra file missing: {path}")
        files[str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")] = sha256(path)
    return {"overall_pass": True, "source_files": files}


def verify_source_lock(lock: Mapping[str, Any]) -> None:
    for relative, expected in (lock.get("source_files") or {}).items():
        path = PROJECT_ROOT / relative
        require(path.is_file(), f"source file disappeared: {path}")
        actual = sha256(path)
        require(
            actual == expected
            or source_hash_matches_layout_attestation(relative, expected, actual),
            f"source file changed without a valid layout attestation: {relative}",
        )
