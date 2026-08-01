# -*- coding: utf-8 -*-
"""Recoverable boundary grid evaluation for Chapter-4 RBE checkpoints."""

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.maddpg import MADDPG
from evaluate_pse import _apply_profile
from train import _build_train_env, _configure_ch4_env, get_ablation_config
from registry.rbe_disturbance import (
    DEFAULT_DISTURBANCE_BOUNDS,
    DISTURBANCE_KEYS,
    nominal_disturbance,
)
from utils.rbe_metrics import EpisodeMetricTracker


GRID_SPECS = {
    "flow_drag": ("flow_gain", "drag_scale"),
    "delay_lag": ("action_delay_steps", "actuator_lag"),
}

HEATMAP_METRICS = (
    "success_rate",
    "found_rate",
    "succ_if_found",
    "not_found_rate",
    "found_but_failed_rate",
    "avg_recovery_time",
    "avg_safety_cost",
    "avg_final_distance",
    "avg_action_smoothness",
    "recoverable_flag",
)

EPISODE_FIELDS = (
    "model_name",
    "ablation_mode",
    "model_path",
    "grid_name",
    "axis_x_name",
    "axis_y_name",
    "axis_x_value",
    "axis_y_value",
    "seed",
    "episode_index",
    *DISTURBANCE_KEYS,
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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--models", nargs="+", required=True, help="name,ablation_mode,model_path entries")
    parser.add_argument("--grids", nargs="+", default=["flow_drag", "delay_lag"], choices=sorted(GRID_SPECS))
    parser.add_argument("--grid-size", type=int, default=7)
    parser.add_argument("--episodes-per-cell", type=int, default=3)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--env-device", default="cpu")
    parser.add_argument("--profile", default="normal_comm")
    parser.add_argument("--fixed-other", default="nominal", choices=("nominal", "midpoint"))
    parser.add_argument("--random-flow-phase", type=int, default=1)
    parser.add_argument("--p0", type=float, default=0.50)
    parser.add_argument("--t0", type=float, default=220.0)
    parser.add_argument("--c0", type=float, default=20.0)
    parser.add_argument("--d0", type=float, default=3.0)
    parser.add_argument("--only-model", default=None)
    parser.add_argument("--only-grid", default=None, choices=sorted(GRID_SPECS))
    return parser.parse_args()


def resolve_device(value):
    value = str(value).strip().lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "gpu":
        value = "cuda"
    if value.startswith("cuda") and not torch.cuda.is_available():
        print(f"[GridEval][warning] requested {value}, falling back to cpu")
        return torch.device("cpu")
    return torch.device(value)


def parse_models(model_specs, only_model=None):
    models = []
    for spec in model_specs:
        parts = [part.strip() for part in str(spec).split(",", 2)]
        if len(parts) != 3 or not all(parts):
            raise ValueError(f"Invalid model spec: {spec!r}. Expected name,ablation_mode,model_path")
        model = {"model_name": parts[0], "ablation_mode": parts[1], "model_path": parts[2]}
        if only_model is None or model["model_name"] == only_model:
            models.append(model)
    if not models:
        raise ValueError(f"No models selected. only_model={only_model!r}")
    for model in models:
        if not os.path.exists(model["model_path"]):
            raise FileNotFoundError(f"model_path does not exist: {model['model_path']}")
    return models


def base_disturbance(fixed_other):
    if fixed_other == "nominal":
        return nominal_disturbance()
    return {
        key: (
            int(round((float(bounds[0]) + float(bounds[1])) / 2.0))
            if key == "action_delay_steps"
            else (float(bounds[0]) + float(bounds[1])) / 2.0
        )
        for key, bounds in DEFAULT_DISTURBANCE_BOUNDS.items()
    }


def axis_values(key, grid_size):
    low, high = DEFAULT_DISTURBANCE_BOUNDS[key]
    if key == "action_delay_steps":
        return [int(v) for v in range(int(round(low)), int(round(high)) + 1)]
    return [float(v) for v in np.linspace(float(low), float(high), int(grid_size))]


def grid_cells(grid_name, grid_size):
    axis_x, axis_y = GRID_SPECS[grid_name]
    xs = axis_values(axis_x, grid_size)
    ys = axis_values(axis_y, grid_size)
    cells = []
    for y in ys:
        for x in xs:
            cells.append((x, y))
    return axis_x, axis_y, xs, ys, cells


def make_cell_xi(grid_name, x_value, y_value, fixed_other):
    axis_x, axis_y = GRID_SPECS[grid_name]
    xi = base_disturbance(fixed_other)
    xi[axis_x] = int(round(float(x_value))) if axis_x == "action_delay_steps" else float(x_value)
    xi[axis_y] = int(round(float(y_value))) if axis_y == "action_delay_steps" else float(y_value)
    xi["action_delay_steps"] = int(round(float(xi["action_delay_steps"])))
    return {key: xi[key] for key in DISTURBANCE_KEYS}


def episode_xi(cell_xi, rng, random_flow_phase):
    xi = dict(cell_xi)
    if random_flow_phase:
        xi["flow_phase_x"] = float(rng.uniform(0.0, 2.0 * math.pi))
        xi["flow_phase_y"] = float(rng.uniform(0.0, 2.0 * math.pi))
    return xi


def to_env_actions(actions, env_device):
    return torch.stack([a.detach().to(device=env_device, dtype=torch.float32).view(-1) for a in actions], dim=0)


def set_global_seed(seed):
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_float(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def mean_or_none(values):
    valid = [float(v) for v in values if safe_float(v) is not None]
    return float(np.mean(valid)) if valid else None


def rate(num, den):
    return None if int(den) == 0 else float(num) / float(den)


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe(obj), f, indent=2, ensure_ascii=False, allow_nan=False)


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: "" if row.get(field) is None else row.get(field) for field in fieldnames})


def aggregate_cell(rows, p0, t0, c0, d0):
    n_episodes = len(rows)
    n_success = sum(1 for row in rows if bool(row["success_flag"]))
    n_found = sum(1 for row in rows if bool(row["found_flag"]))
    n_not_found = n_episodes - n_found
    n_found_but_failed = sum(1 for row in rows if bool(row["found_flag"]) and not bool(row["success_flag"]))

    success_rows = [row for row in rows if bool(row["success_flag"])]
    found_failed_rows = [row for row in rows if bool(row["found_flag"]) and not bool(row["success_flag"])]

    out = {
        "model_name": rows[0]["model_name"],
        "ablation_mode": rows[0]["ablation_mode"],
        "model_path": rows[0]["model_path"],
        "grid_name": rows[0]["grid_name"],
        "axis_x_name": rows[0]["axis_x_name"],
        "axis_y_name": rows[0]["axis_y_name"],
        "axis_x_value": rows[0]["axis_x_value"],
        "axis_y_value": rows[0]["axis_y_value"],
        "n_episodes": n_episodes,
        "n_success": n_success,
        "n_found": n_found,
        "n_not_found": n_not_found,
        "n_found_but_failed": n_found_but_failed,
        "success_rate": rate(n_success, n_episodes),
        "found_rate": rate(n_found, n_episodes),
        "succ_if_found": rate(n_success, n_found),
        "not_found_rate": rate(n_not_found, n_episodes),
        "found_but_failed_rate": rate(n_found_but_failed, n_episodes),
        "avg_recovery_time": mean_or_none(row["recovery_time"] for row in rows),
        "avg_recovery_time_success": mean_or_none(row["recovery_time"] for row in success_rows),
        "avg_recovery_time_found_but_failed": mean_or_none(row["recovery_time"] for row in found_failed_rows),
        "avg_safety_cost": mean_or_none(row["safety_cost"] for row in rows),
        "avg_safety_cost_success": mean_or_none(row["safety_cost"] for row in success_rows),
        "avg_safety_cost_found_but_failed": mean_or_none(row["safety_cost"] for row in found_failed_rows),
        "avg_final_distance": mean_or_none(row["final_distance"] for row in rows),
        "avg_final_distance_success": mean_or_none(row["final_distance"] for row in success_rows),
        "avg_final_distance_found_but_failed": mean_or_none(row["final_distance"] for row in found_failed_rows),
        "avg_action_smoothness": mean_or_none(row["action_smoothness"] for row in rows),
        "avg_completion_steps": mean_or_none(row["completion_steps"] for row in rows),
    }
    out["recoverable_flag"] = bool(
        safe_float(out["success_rate"]) is not None
        and out["success_rate"] >= p0
        and safe_float(out["avg_recovery_time"]) is not None
        and out["avg_recovery_time"] <= t0
        and safe_float(out["avg_safety_cost"]) is not None
        and out["avg_safety_cost"] <= c0
        and safe_float(out["avg_final_distance"]) is not None
        and out["avg_final_distance"] <= d0
    )
    out["recoverable_success_only_flag"] = bool(
        n_success > 0
        and safe_float(out["success_rate"]) is not None
        and out["success_rate"] >= p0
        and safe_float(out["avg_recovery_time_success"]) is not None
        and out["avg_recovery_time_success"] <= t0
        and safe_float(out["avg_safety_cost_success"]) is not None
        and out["avg_safety_cost_success"] <= c0
        and safe_float(out["avg_final_distance_success"]) is not None
        and out["avg_final_distance_success"] <= d0
    )
    return out


def matrix_from_rows(cell_rows, xs, ys, metric):
    lookup = {(row["axis_x_value"], row["axis_y_value"]): row for row in cell_rows}
    mat = np.full((len(ys), len(xs)), np.nan, dtype=np.float64)
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            row = lookup.get((x, y))
            if row is None:
                continue
            value = row.get(metric)
            if isinstance(value, bool):
                value = 1.0 if value else 0.0
            value = safe_float(value)
            if value is not None:
                mat[iy, ix] = value
    return mat


def label_value(value):
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    value_f = float(value)
    if abs(value_f - round(value_f)) < 1e-9:
        return str(int(round(value_f)))
    return f"{value_f:.3g}"


def save_heatmaps(out_dir, model_name, grid_name, axis_x, axis_y, xs, ys, cell_rows):
    heatmap_dir = os.path.join(out_dir, "heatmaps")
    os.makedirs(heatmap_dir, exist_ok=True)
    for metric in HEATMAP_METRICS:
        mat = matrix_from_rows(cell_rows, xs, ys, metric)
        if not np.isfinite(mat).any():
            print(f"[GridEval][warning] skip all-missing heatmap: {model_name}/{grid_name}/{metric}")
            continue
        fig, ax = plt.subplots(figsize=(7.2, 5.8))
        kwargs = {}
        if metric in {"success_rate", "found_rate", "succ_if_found", "recoverable_flag"}:
            kwargs.update(vmin=0.0, vmax=1.0)
        im = ax.imshow(mat, origin="lower", aspect="auto", **kwargs)
        ax.set_title(f"{model_name} | {grid_name} | {metric}")
        ax.set_xlabel(axis_x)
        ax.set_ylabel(axis_y)
        ax.set_xticks(np.arange(len(xs)))
        ax.set_xticklabels([label_value(v) for v in xs])
        ax.set_yticks(np.arange(len(ys)))
        ax.set_yticklabels([label_value(v) for v in ys])
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(heatmap_dir, f"{metric}.png"), dpi=160)
        plt.close(fig)


def model_grid_summary(model, grid_name, axis_x, axis_y, xs, ys, cell_rows, args):
    total_cells = len(cell_rows)
    recoverable_cell_count = sum(1 for row in cell_rows if bool(row["recoverable_flag"]))
    recoverable_success_only_cell_count = sum(1 for row in cell_rows if bool(row["recoverable_success_only_flag"]))
    return {
        "model_name": model["model_name"],
        "ablation_mode": model["ablation_mode"],
        "model_path": model["model_path"],
        "grid_name": grid_name,
        "axis_x_name": axis_x,
        "axis_y_name": axis_y,
        "x_values": xs,
        "y_values": ys,
        "total_cells": total_cells,
        "recoverable_cell_count": recoverable_cell_count,
        "recoverable_area_ratio": rate(recoverable_cell_count, total_cells),
        "recoverable_success_only_cell_count": recoverable_success_only_cell_count,
        "recoverable_success_only_area_ratio": rate(recoverable_success_only_cell_count, total_cells),
        "mean_success_rate": mean_or_none(row["success_rate"] for row in cell_rows),
        "mean_found_rate": mean_or_none(row["found_rate"] for row in cell_rows),
        "mean_succ_if_found": mean_or_none(row["succ_if_found"] for row in cell_rows),
        "mean_recovery_time": mean_or_none(row["avg_recovery_time"] for row in cell_rows),
        "mean_safety_cost": mean_or_none(row["avg_safety_cost"] for row in cell_rows),
        "mean_final_distance": mean_or_none(row["avg_final_distance"] for row in cell_rows),
        "thresholds": {"p0": args.p0, "t0": args.t0, "c0": args.c0, "d0": args.d0},
    }


def write_model_grid_markdown(path, summary):
    headers = (
        "model_name",
        "grid_name",
        "recoverable_area_ratio",
        "recoverable_success_only_area_ratio",
        "mean_success_rate",
        "mean_found_rate",
        "mean_succ_if_found",
        "mean_recovery_time",
        "mean_safety_cost",
        "mean_final_distance",
    )
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.append("| " + " | ".join(format_md(summary.get(key)) for key in headers) + " |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def format_md(value):
    value = json_safe(value)
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


def run_episode(env, maddpg, train_device, env_device, xi, max_steps, profile, episode_seed):
    set_global_seed(episode_seed)
    env.set_next_disturbance(xi)
    obs = env.reset()
    _apply_profile(env, profile)
    tracker = EpisodeMetricTracker()
    tracker.reset(env, xi)
    for _step in range(1, int(max_steps) + 1):
        actions = maddpg.step(obs, explore=False)
        env_actions = to_env_actions(actions, env_device)
        obs, rewards, dones = env.step(env_actions)
        rewards_t = rewards.detach().to(dtype=torch.float32) if torch.is_tensor(rewards) else torch.as_tensor(rewards, dtype=torch.float32)
        tracker.step(env, env_actions, rewards_t, dones)
        if all(bool(done) for done in dones):
            break
    return tracker.finalize(env)


def evaluate_model_grid(model, grid_name, args, train_device, env_device):
    axis_x, axis_y, xs, ys, cells = grid_cells(grid_name, args.grid_size)
    out_dir = os.path.join(args.out_root, model["model_name"], grid_name)
    os.makedirs(out_dir, exist_ok=True)

    env_cfg, _ = get_ablation_config(model["ablation_mode"])
    env, env_kwargs = _build_train_env(env_device, int(args.max_steps), env_cfg)
    env, env_kwargs = _configure_ch4_env(env, env_kwargs, model["ablation_mode"])
    _apply_profile(env, args.profile)
    env.use_robust_disturbance = True
    env_kwargs["use_robust_disturbance"] = True

    maddpg = MADDPG.init_from_save(model["model_path"], device=train_device)
    maddpg.prep_rollouts(device=train_device)

    episode_rows = []
    by_cell = defaultdict(list)
    total_cells = len(cells)
    episode_counter = 0
    for cell_i, (x_value, y_value) in enumerate(cells, start=1):
        cell_xi = make_cell_xi(grid_name, x_value, y_value, args.fixed_other)
        for seed in args.seeds:
            rng = np.random.default_rng(int(seed) * 1_000_003 + cell_i)
            for ep_local in range(int(args.episodes_per_cell)):
                episode_counter += 1
                xi = episode_xi(cell_xi, rng, bool(args.random_flow_phase))
                print(
                    f"[GridEval] model={model['model_name']} grid={grid_name} "
                    f"cell {cell_i}/{total_cells} seed={seed} ep={ep_local + 1}/{args.episodes_per_cell}",
                    flush=True,
                )
                episode_seed = int(seed) * 1_000_003 + cell_i * 10_007 + ep_local
                metrics = run_episode(env, maddpg, train_device, env_device, xi, args.max_steps, args.profile, episode_seed)
                row = {
                    "model_name": model["model_name"],
                    "ablation_mode": model["ablation_mode"],
                    "model_path": model["model_path"],
                    "grid_name": grid_name,
                    "axis_x_name": axis_x,
                    "axis_y_name": axis_y,
                    "axis_x_value": x_value,
                    "axis_y_value": y_value,
                    "seed": int(seed),
                    "episode_index": episode_counter,
                }
                for key in DISTURBANCE_KEYS:
                    row[key] = int(metrics[key]) if key == "action_delay_steps" else safe_float(metrics[key])
                row.update(
                    {
                        "success_flag": bool(metrics.get("success_flag", False)),
                        "found_flag": bool(metrics.get("found_flag", False)),
                        "recovery_time": int(metrics.get("recovery_time", args.max_steps)),
                        "safety_cost": safe_float(metrics.get("safety_cost")),
                        "final_distance": safe_float(metrics.get("final_distance")),
                        "final_nav_distance": safe_float(metrics.get("final_nav_distance")),
                        "action_smoothness": safe_float(metrics.get("action_smoothness")),
                        "completion_steps": int(metrics.get("completion_steps", args.max_steps)),
                        "episode_reward_mean": safe_float(metrics.get("episode_reward_mean")),
                        "episode_reward_sum": safe_float(metrics.get("episode_reward_sum")),
                    }
                )
                episode_rows.append(row)
                by_cell[(x_value, y_value)].append(row)

    cell_rows = [aggregate_cell(by_cell[(x_value, y_value)], args.p0, args.t0, args.c0, args.d0) for x_value, y_value in cells]
    cell_fields = list(cell_rows[0].keys()) if cell_rows else []
    write_csv(os.path.join(out_dir, "episode_metrics.csv"), episode_rows, EPISODE_FIELDS)
    write_csv(os.path.join(out_dir, "cell_summary.csv"), cell_rows, cell_fields)
    write_json(os.path.join(out_dir, "cell_summary.json"), cell_rows)
    save_heatmaps(out_dir, model["model_name"], grid_name, axis_x, axis_y, xs, ys, cell_rows)

    summary = model_grid_summary(model, grid_name, axis_x, axis_y, xs, ys, cell_rows, args)
    write_json(os.path.join(out_dir, "model_grid_summary.json"), summary)
    write_model_grid_markdown(os.path.join(out_dir, "model_grid_summary.md"), summary)
    return summary


def build_area_table(summaries):
    rows = []
    for summary in summaries:
        rows.append(
            {
                "model_name": summary["model_name"],
                "grid_name": summary["grid_name"],
                "ablation_mode": summary["ablation_mode"],
                "model_path": summary["model_path"],
                "recoverable_area_ratio": summary["recoverable_area_ratio"],
                "recoverable_success_only_area_ratio": summary["recoverable_success_only_area_ratio"],
                "mean_success_rate": summary["mean_success_rate"],
                "mean_found_rate": summary["mean_found_rate"],
                "mean_succ_if_found": summary["mean_succ_if_found"],
                "mean_recovery_time": summary["mean_recovery_time"],
                "mean_safety_cost": summary["mean_safety_cost"],
                "mean_final_distance": summary["mean_final_distance"],
            }
        )
    return rows


def build_delta_rows(area_rows):
    baselines = {row["grid_name"]: row for row in area_rows if row["model_name"] == "uniform_snapshot"}
    if not baselines:
        print("[GridEval][warning] uniform_snapshot baseline not found; skipping delta_vs_uniform_snapshot.csv")
        return []
    deltas = []
    for row in area_rows:
        if row["model_name"] == "uniform_snapshot":
            continue
        base = baselines.get(row["grid_name"])
        if base is None:
            continue
        deltas.append(
            {
                "model_name": row["model_name"],
                "grid_name": row["grid_name"],
                "delta_recoverable_area_ratio": diff(row["recoverable_area_ratio"], base["recoverable_area_ratio"]),
                "delta_success_rate": diff(row["mean_success_rate"], base["mean_success_rate"]),
                "delta_found_rate": diff(row["mean_found_rate"], base["mean_found_rate"]),
                "delta_succ_if_found": diff(row["mean_succ_if_found"], base["mean_succ_if_found"]),
                "delta_recovery_time": diff(row["mean_recovery_time"], base["mean_recovery_time"]),
                "delta_safety_cost": diff(row["mean_safety_cost"], base["mean_safety_cost"]),
                "delta_final_distance": diff(row["mean_final_distance"], base["mean_final_distance"]),
            }
        )
    return deltas


def diff(value, baseline):
    value = safe_float(value)
    baseline = safe_float(baseline)
    return None if value is None or baseline is None else value - baseline


def write_area_markdown(path, area_rows, delta_rows):
    headers = (
        "model_name",
        "grid_name",
        "recoverable_area_ratio",
        "mean_success_rate",
        "mean_found_rate",
        "mean_succ_if_found",
        "mean_recovery_time",
        "mean_safety_cost",
        "mean_final_distance",
    )
    lines = ["# Recoverable Boundary Grid Summary", "", "| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in area_rows:
        lines.append("| " + " | ".join(format_md(row.get(key)) for key in headers) + " |")
    if delta_rows:
        delta_headers = tuple(delta_rows[0].keys())
        lines.extend(["", "## Delta vs uniform_snapshot", "", "| " + " | ".join(delta_headers) + " |", "| " + " | ".join(["---"] * len(delta_headers)) + " |"])
        for row in delta_rows:
            lines.append("| " + " | ".join(format_md(row.get(key)) for key in delta_headers) + " |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    args = parse_args()
    if args.only_grid:
        args.grids = [args.only_grid]
    args.grid_size = int(max(1, args.grid_size))
    args.episodes_per_cell = int(max(1, args.episodes_per_cell))
    args.out_root = os.path.normpath(args.out_root)
    os.makedirs(args.out_root, exist_ok=True)

    train_device = resolve_device(args.device)
    env_device = resolve_device(args.env_device)
    models = parse_models(args.models, args.only_model)
    print(f"[GridEval] out_root={args.out_root}")
    print(f"[GridEval] models={[m['model_name'] for m in models]} grids={args.grids}")
    print(f"[GridEval] train_device={train_device} env_device={env_device}")

    summaries = []
    for model in models:
        for grid_name in args.grids:
            summaries.append(evaluate_model_grid(model, grid_name, args, train_device, env_device))

    area_rows = build_area_table(summaries)
    delta_rows = build_delta_rows(area_rows)
    area_fields = list(area_rows[0].keys()) if area_rows else ["model_name"]
    write_csv(os.path.join(args.out_root, "recoverable_area_table.csv"), area_rows, area_fields)
    if delta_rows:
        write_csv(os.path.join(args.out_root, "delta_vs_uniform_snapshot.csv"), delta_rows, list(delta_rows[0].keys()))

    summary = {
        "summary_name": os.path.basename(args.out_root.rstrip(os.sep)) or "recoverable_boundary_grid_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "thresholds": {"p0": args.p0, "t0": args.t0, "c0": args.c0, "d0": args.d0},
        "settings": {
            "grids": args.grids,
            "grid_size": args.grid_size,
            "episodes_per_cell": args.episodes_per_cell,
            "seeds": args.seeds,
            "fixed_other": args.fixed_other,
            "random_flow_phase": bool(args.random_flow_phase),
            "max_steps": args.max_steps,
            "profile": args.profile,
        },
        "models": models,
        "model_grid_summaries": summaries,
        "delta_vs_uniform_snapshot": delta_rows,
    }
    write_json(os.path.join(args.out_root, "recoverable_area_summary.json"), summary)
    write_area_markdown(os.path.join(args.out_root, "recoverable_area_summary.md"), area_rows, delta_rows)
    print(f"[GridEval] done: {args.out_root}")


if __name__ == "__main__":
    main()
