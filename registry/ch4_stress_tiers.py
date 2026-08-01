"""Deterministic radial full-9D disturbance tiers for Chapter-4 stress tests."""

from __future__ import annotations

import math
from typing import Dict, Mapping, Tuple

import numpy as np

from registry.rbe_disturbance import (
    DEFAULT_DISTURBANCE_BOUNDS,
    DISTURBANCE_KEYS,
    nominal_disturbance,
)


PROTOCOL_ID = "radial_full9d_stress_tiers_v1"
STRESS_LEVELS: Tuple[float, ...] = (0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50)
FLOW_PHASE_KEYS = ("flow_phase_x", "flow_phase_y")

# Positive direction means moving from nominal toward the registry upper bound;
# negative direction means moving toward the lower bound.
_ONE_SIDED_UP = {
    "flow_gain",
    "actuator_lag",
    "action_delay_steps",
    "action_noise_std",
}
_ONE_SIDED_DOWN = {
    "a_max_scale",
    "v_max_scale",
}
_SIGNED = {
    "flow_z_gain",
    "drag_scale",
    "buoyancy_bias_delta",
}


def episode_seed(base_seed: int, episode_index: int) -> int:
    """Common-random-number seed shared across models and stress levels."""
    return int(base_seed) * 1_000_003 + int(episode_index)


def _direction_rng(base_seed: int, episode_index: int) -> np.random.Generator:
    # XOR with a fixed odd constant keeps direction generation independent from
    # environment/global RNG use while remaining deterministic and reproducible.
    seed = episode_seed(base_seed, episode_index) ^ 0x5EED4B1D
    return np.random.default_rng(seed)


def sample_stress_direction(base_seed: int, episode_index: int) -> Dict[str, float]:
    """Sample one adverse full-9D direction and normalize it in L-infinity.

    Every dimension is active. Magnitudes are sampled in [0.35, 1.0], then the
    vector is normalized so at least one continuous dimension reaches its
    registry boundary at rho=1. Signed dimensions independently choose either
    side of the nominal point.
    """
    rng = _direction_rng(base_seed, episode_index)
    raw: Dict[str, float] = {}
    for key in DISTURBANCE_KEYS:
        magnitude = float(rng.uniform(0.35, 1.0))
        if key in _ONE_SIDED_UP:
            sign = 1.0
        elif key in _ONE_SIDED_DOWN:
            sign = -1.0
        elif key in _SIGNED:
            sign = -1.0 if int(rng.integers(0, 2)) == 0 else 1.0
        else:  # Defensive fallback for future registry extensions.
            sign = -1.0 if int(rng.integers(0, 2)) == 0 else 1.0
        raw[key] = sign * magnitude
    scale = max(abs(value) for value in raw.values())
    if not math.isfinite(scale) or scale <= 0.0:
        raise RuntimeError("invalid stress direction scale")
    return {key: float(raw[key] / scale) for key in DISTURBANCE_KEYS}


def flow_phases(base_seed: int, episode_index: int) -> Dict[str, float]:
    seed = episode_seed(base_seed, episode_index) ^ 0x31A5EED7
    rng = np.random.default_rng(seed)
    return {
        "flow_phase_x": float(rng.uniform(0.0, 2.0 * np.pi)),
        "flow_phase_y": float(rng.uniform(0.0, 2.0 * np.pi)),
    }


def disturbance_from_direction(
    direction: Mapping[str, float],
    rho: float,
    *,
    bounds: Mapping[str, Tuple[float, float]] | None = None,
) -> Dict[str, float | int]:
    """Map a normalized direction to a physical disturbance at intensity rho."""
    rho = float(rho)
    if not math.isfinite(rho) or rho < 0.0:
        raise ValueError(f"rho must be finite and non-negative, got {rho!r}")
    bounds = DEFAULT_DISTURBANCE_BOUNDS if bounds is None else bounds
    nominal = nominal_disturbance()
    xi: Dict[str, float | int] = {}
    for key in DISTURBANCE_KEYS:
        if key not in direction:
            raise KeyError(f"stress direction lacks {key}")
        d = float(direction[key])
        if not math.isfinite(d) or abs(d) > 1.0 + 1e-12:
            raise ValueError(f"invalid normalized direction for {key}: {d}")
        low, high = (float(bounds[key][0]), float(bounds[key][1]))
        x0 = float(nominal[key])
        side_span = (high - x0) if d >= 0.0 else (x0 - low)
        value = x0 + rho * d * side_span
        if key == "action_delay_steps":
            value = int(math.floor(value + 0.5))
        else:
            value = float(value)
        xi[key] = value
    return xi


def build_stress_case(base_seed: int, episode_index: int, rho: float) -> Dict[str, object]:
    """Return direction, physical disturbance, phases, and reproducibility data."""
    direction = sample_stress_direction(base_seed, episode_index)
    xi = disturbance_from_direction(direction, rho)
    phases = flow_phases(base_seed, episode_index)
    requested = dict(xi)
    requested.update(phases)
    return {
        "protocol": PROTOCOL_ID,
        "rho": float(rho),
        "base_seed": int(base_seed),
        "episode_index": int(episode_index),
        "episode_seed": episode_seed(base_seed, episode_index),
        "direction": direction,
        "disturbance": xi,
        "phases": phases,
        "requested": requested,
    }


def tier_tag(rho: float) -> str:
    return f"rho_{int(round(float(rho) * 100)):03d}"


def direction_signature(direction: Mapping[str, float]) -> Tuple[float, ...]:
    return tuple(round(float(direction[key]), 12) for key in DISTURBANCE_KEYS)
