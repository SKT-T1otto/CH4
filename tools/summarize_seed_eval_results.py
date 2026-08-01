# -*- coding: utf-8 -*-
"""Aggregate seed-level evaluate_pse.py summaries into one experiment summary."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


COUNT_FIELDS = (
    "n_episodes",
    "n_success",
    "n_found",
    "n_not_found",
    "n_found_but_failed",
    "n_late_found_fail",
    "n_early_found_fail",
)
MEAN_FIELDS = (
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
    "avg_final_distance_success",
    "avg_final_distance_found_but_failed",
    "avg_final_distance_not_found",
    "avg_safety_cost_success",
    "avg_safety_cost_found_but_failed",
    "avg_safety_cost_not_found",
    "avg_recovery_time_success",
    "avg_recovery_time_found_but_failed",
    "avg_recovery_time_not_found",
    "avg_standby_to_target_dist_success",
    "avg_standby_to_target_dist_found_but_failed",
    "avg_standby_to_target_dist_not_found",
    "avg_belief_entropy_success",
    "avg_belief_entropy_found_but_failed",
    "avg_belief_entropy_not_found",
    "avg_exec_response_cost_success",
    "avg_exec_response_cost_found_but_failed",
    "avg_exec_response_cost_not_found",
    "avg_residual_contribution_ratio_executor_success",
    "avg_residual_contribution_ratio_executor_found_but_failed",
    "avg_residual_contribution_ratio_executor_not_found",
    "avg_residual_contribution_ratio_search_success",
    "avg_residual_contribution_ratio_search_found_but_failed",
    "avg_residual_contribution_ratio_search_not_found",
)
BY_SEED_FIELDS = (
    "seed",
    "method",
    "profile",
    "episodes",
    "max_steps",
    "n_episodes",
    "n_success",
    "n_found",
    "n_not_found",
    "n_found_but_failed",
    "n_late_found_fail",
    "n_early_found_fail",
    "success_rate",
    "found_rate",
    "succ_if_found",
    "not_found_rate",
    "found_but_failed_rate",
    "late_found_fail_rate",
    "early_found_fail_rate",
    "avg_reward",
    "avg_recovery_time",
    "avg_safety_cost",
    "avg_final_distance",
    "avg_final_nav_distance",
    "avg_action_smoothness",
    "avg_completion_steps",
    "avg_final_distance_success",
    "avg_final_distance_found_but_failed",
    "avg_final_distance_not_found",
    "avg_safety_cost_success",
    "avg_safety_cost_found_but_failed",
    "avg_safety_cost_not_found",
    "avg_recovery_time_success",
    "avg_recovery_time_found_but_failed",
    "avg_recovery_time_not_found",
    "avg_standby_to_target_dist_success",
    "avg_standby_to_target_dist_found_but_failed",
    "avg_standby_to_target_dist_not_found",
    "avg_belief_entropy_success",
    "avg_belief_entropy_found_but_failed",
    "avg_belief_entropy_not_found",
    "avg_exec_response_cost_success",
    "avg_exec_response_cost_found_but_failed",
    "avg_exec_response_cost_not_found",
    "avg_residual_contribution_ratio_executor_success",
    "avg_residual_contribution_ratio_executor_found_but_failed",
    "avg_residual_contribution_ratio_executor_not_found",
    "avg_residual_contribution_ratio_search_success",
    "avg_residual_contribution_ratio_search_found_but_failed",
    "avg_residual_contribution_ratio_search_not_found",
    "model_path",
    "summary_path",
)

WEIGHTED_MEAN_FIELDS = (
    "avg_reward",
    "avg_recovery_time",
    "avg_safety_cost",
    "avg_final_distance",
    "avg_final_nav_distance",
    "avg_action_smoothness",
    "avg_completion_steps",
)


def _safe_float(value):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _safe_int(value):
    number = _safe_float(value)
    return 0 if number is None else int(number)


def _mean(values):
    valid = [_safe_float(v) for v in values]
    valid = [v for v in valid if v is not None]
    return sum(valid) / len(valid) if valid else None


def _weighted_mean(rows, field):
    numerator = 0.0
    denominator = 0
    for row in rows:
        value = _safe_float(row.get(field))
        n_episodes = _safe_int(row.get("n_episodes", row.get("episodes")))
        if value is None or n_episodes <= 0:
            continue
        numerator += value * n_episodes
        denominator += n_episodes
    return None if denominator == 0 else numerator / denominator


def _rate(num, den):
    return None if int(den) == 0 else float(num) / float(den)


def _success_if_found_rate(num_success, num_found):
    return (
        None
        if int(num_found) == 0
        else float(num_success) / float(num_found)
    )


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _load_rows(result_dir: Path):
    rows = []
    for summary_path in sorted(result_dir.rglob("evaluation_summary.json")):
        with summary_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        data["summary_path"] = str(summary_path)
        data["result_dir"] = str(summary_path.parent)
        rows.append(data)
    rows.sort(key=lambda r: (_safe_int(r.get("seed")), str(r.get("result_dir", ""))))
    return rows


def build_aggregate(rows, name):
    totals = {field: sum(_safe_int(row.get(field)) for row in rows) for field in COUNT_FIELDS}
    aggregate = {
        "summary_name": name,
        "method": rows[0].get("method") if rows else None,
        "profile": rows[0].get("profile") if rows else None,
        "seeds": [row.get("seed") for row in rows],
        "n_seed_runs": len(rows),
        "episodes_per_seed": [_safe_int(row.get("n_episodes", row.get("episodes"))) for row in rows],
        "source_dirs": [row.get("result_dir") for row in rows],
    }
    aggregate.update({f"total_{field[2:] if field.startswith('n_') else field}": value for field, value in totals.items()})
    aggregate.update(
        {
            "success_rate_weighted": _rate(totals["n_success"], totals["n_episodes"]),
            "found_rate_weighted": _rate(totals["n_found"], totals["n_episodes"]),
            "succ_if_found_weighted": _success_if_found_rate(
                totals["n_success"], totals["n_found"]
            ),
            "not_found_rate_weighted": _rate(totals["n_not_found"], totals["n_episodes"]),
            "found_but_failed_rate_weighted": _rate(totals["n_found_but_failed"], totals["n_episodes"]),
            "late_found_fail_rate_weighted": _rate(totals["n_late_found_fail"], totals["n_found_but_failed"]),
            "early_found_fail_rate_weighted": _rate(totals["n_early_found_fail"], totals["n_found_but_failed"]),
        }
    )
    for field in MEAN_FIELDS:
        aggregate[f"{field}_mean"] = _mean(row.get(field) for row in rows)
    for field in WEIGHTED_MEAN_FIELDS:
        aggregate[f"{field}_weighted"] = _weighted_mean(rows, field)
    return _json_safe(aggregate)


def write_by_seed_csv(rows, out_path: Path):
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=BY_SEED_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in BY_SEED_FIELDS})


def _fmt(value, digits=4):
    number = _safe_float(value)
    return "" if number is None else f"{number:.{digits}f}"


def write_markdown(aggregate, rows, out_path: Path):
    lines = [
        f"# {aggregate.get('summary_name')}",
        "",
        f"- method: {aggregate.get('method')}",
        f"- profile: {aggregate.get('profile')}",
        f"- total episodes: {aggregate.get('total_episodes')}",
        f"- total success: {aggregate.get('total_success')}",
        f"- total found: {aggregate.get('total_found')}",
        f"- total not found: {aggregate.get('total_not_found')}",
        f"- total found but failed: {aggregate.get('total_found_but_failed')}",
        f"- weighted success rate: {_fmt(aggregate.get('success_rate_weighted'), 6)}",
        f"- weighted found rate: {_fmt(aggregate.get('found_rate_weighted'), 6)}",
        f"- weighted success if found: {_fmt(aggregate.get('succ_if_found_weighted'), 6)}",
        f"- weighted average reward: {_fmt(aggregate.get('avg_reward_weighted'), 6)}",
        f"- weighted average recovery time: {_fmt(aggregate.get('avg_recovery_time_weighted'), 6)}",
        f"- weighted average safety cost: {_fmt(aggregate.get('avg_safety_cost_weighted'), 6)}",
        f"- weighted average final distance: {_fmt(aggregate.get('avg_final_distance_weighted'), 6)}",
        f"- weighted average final navigation distance: {_fmt(aggregate.get('avg_final_nav_distance_weighted'), 6)}",
        f"- weighted average action smoothness: {_fmt(aggregate.get('avg_action_smoothness_weighted'), 6)}",
        f"- weighted average completion steps: {_fmt(aggregate.get('avg_completion_steps_weighted'), 6)}",
        "",
        "| seed | success_rate | found_rate | succ_if_found | avg_reward | avg_safety_cost | avg_final_distance | n_success | n_found | n_not_found | n_found_but_failed |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {seed} | {success_rate} | {found_rate} | {succ_if_found} | {avg_reward} | {avg_safety_cost} | {avg_final_distance} | {n_success} | {n_found} | {n_not_found} | {n_found_but_failed} |".format(
                seed=row.get("seed", ""),
                success_rate=_fmt(row.get("success_rate")),
                found_rate=_fmt(row.get("found_rate")),
                succ_if_found=_fmt(row.get("succ_if_found")),
                avg_reward=_fmt(row.get("avg_reward")),
                avg_safety_cost=_fmt(row.get("avg_safety_cost")),
                avg_final_distance=_fmt(row.get("avg_final_distance")),
                n_success=row.get("n_success", ""),
                n_found=row.get("n_found", ""),
                n_not_found=row.get("n_not_found", ""),
                n_found_but_failed=row.get("n_found_but_failed", ""),
            )
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Aggregate seed-level evaluate_pse.py results.")
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    out_dir = Path(args.out_dir) if args.out_dir else result_dir
    if not result_dir.exists():
        raise FileNotFoundError(f"result-dir does not exist: {result_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(result_dir)
    if not rows:
        raise FileNotFoundError(f"No evaluation_summary.json found under {result_dir}")

    name = args.name or result_dir.name
    aggregate = build_aggregate(rows, name)

    json_path = out_dir / "aggregate_summary.json"
    csv_path = out_dir / "summary_by_seed.csv"
    md_path = out_dir / "summary.md"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2, ensure_ascii=False, allow_nan=False)
    write_by_seed_csv(rows, csv_path)
    write_markdown(aggregate, rows, md_path)

    print(f"[OK] aggregate summary: {json_path}")
    print(f"[OK] by-seed csv: {csv_path}")
    print(f"[OK] markdown: {md_path}")
    print(
        "[EvalSummary] aggregate: "
        f"total_success={aggregate.get('total_success')}, "
        f"total_found_but_failed={aggregate.get('total_found_but_failed')}, "
        f"total_not_found={aggregate.get('total_not_found')}, "
        f"succ_if_found_weighted={aggregate.get('succ_if_found_weighted')}"
    )


if __name__ == "__main__":
    main()
