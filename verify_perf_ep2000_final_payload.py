#!/usr/bin/env python3
"""Verify logical equality of snapshot_ep2000 and the separately saved final model."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from registry.ch4_artifact_layout import get_training_run_dir, resolve_artifact_path


def load_cpu(path: Path) -> Any:
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def equal(left: Any, right: Any) -> bool:
    if torch.is_tensor(left) or torch.is_tensor(right):
        return (
            torch.is_tensor(left)
            and torch.is_tensor(right)
            and left.dtype == right.dtype
            and tuple(left.shape) == tuple(right.shape)
            and torch.equal(left.detach().cpu(), right.detach().cpu())
        )
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return (
            isinstance(left, np.ndarray)
            and isinstance(right, np.ndarray)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and bool(np.array_equal(left, right, equal_nan=True))
        )
    if isinstance(left, dict) or isinstance(right, dict):
        return (
            isinstance(left, dict)
            and isinstance(right, dict)
            and set(left) == set(right)
            and all(equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(equal(a, b) for a, b in zip(left, right))
        )
    if isinstance(left, float) and isinstance(right, float):
        if math.isnan(left) and math.isnan(right):
            return True
    try:
        return bool(left == right)
    except Exception:
        return False


def main() -> int:
    if len(sys.argv) > 1:
        raise ValueError("artifact root arguments are retired; use CH4_DATA_ROOT")
    overall = True
    for seed in (1, 2, 3):
        model_dir = resolve_artifact_path(
            get_training_run_dir(
                "perf_rbe", f"ch4_rbe_boundary_core_perf_seed{seed}_2000ep_v1"
            )
        )
        snapshot = model_dir / "snapshot_ep2000.pt"
        final = model_dir / "maddpg_uavenv_final.pt"
        if not snapshot.is_file() or not final.is_file():
            print(f"seed{seed}: missing file")
            overall = False
            continue
        same = equal(load_cpu(snapshot), load_cpu(final))
        print(f"seed{seed}: payload_equal={str(same).lower()}")
        overall = overall and same
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
