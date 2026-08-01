"""Chapter-4 RBE disturbance parameter registry and helpers."""

import numpy as np


DISTURBANCE_KEYS = [
    "flow_gain",
    "flow_z_gain",
    "drag_scale",
    "buoyancy_bias_delta",
    "a_max_scale",
    "v_max_scale",
    "actuator_lag",
    "action_delay_steps",
    "action_noise_std",
]


DEFAULT_DISTURBANCE_BOUNDS = {
    "flow_gain": (0.15, 0.55),
    "flow_z_gain": (-0.08, 0.08),
    "drag_scale": (0.70, 1.50),
    "buoyancy_bias_delta": (-0.04, 0.06),
    "a_max_scale": (0.75, 1.15),
    "v_max_scale": (0.80, 1.15),
    "actuator_lag": (0.00, 0.35),
    "action_delay_steps": (0, 3),
    "action_noise_std": (0.00, 0.08),
}


def nominal_disturbance():
    return {
        "flow_gain": 0.18,
        "flow_z_gain": 0.0,
        "drag_scale": 1.0,
        "buoyancy_bias_delta": 0.0,
        "a_max_scale": 1.0,
        "v_max_scale": 1.0,
        "actuator_lag": 0.0,
        "action_delay_steps": 0,
        "action_noise_std": 0.0,
    }


def _rng_uniform(rng, low, high):
    if rng is None:
        return np.random.uniform(low, high)
    if hasattr(rng, "uniform"):
        return rng.uniform(low, high)
    raise TypeError("rng must provide a uniform(low, high) method")


def _rng_integers(rng, low, high_inclusive):
    if rng is None:
        return np.random.randint(low, high_inclusive + 1)
    if hasattr(rng, "integers"):
        return rng.integers(low, high_inclusive + 1)
    if hasattr(rng, "randint"):
        return rng.randint(low, high_inclusive + 1)
    raise TypeError("rng must provide integers() or randint() for integer sampling")


def sample_uniform_disturbance(rng=None, bounds=None, include_flow_phase=True):
    bounds = DEFAULT_DISTURBANCE_BOUNDS if bounds is None else bounds
    xi = {}
    for key in DISTURBANCE_KEYS:
        low, high = bounds[key]
        if high < low:
            low, high = high, low
        if key == "action_delay_steps":
            xi[key] = int(_rng_integers(rng, int(round(low)), int(round(high))))
        else:
            xi[key] = float(_rng_uniform(rng, float(low), float(high)))
    if include_flow_phase:
        xi["flow_phase_x"] = float(_rng_uniform(rng, 0.0, 2.0 * np.pi))
        xi["flow_phase_y"] = float(_rng_uniform(rng, 0.0, 2.0 * np.pi))
    return xi


def xi_to_array(xi, keys=DISTURBANCE_KEYS):
    return np.asarray([float(xi[key]) for key in keys], dtype=np.float32)


def array_to_xi(arr, keys=DISTURBANCE_KEYS):
    values = np.asarray(arr, dtype=np.float32).reshape(-1)
    if values.size != len(keys):
        raise ValueError(f"Expected {len(keys)} values, got {values.size}")
    xi = {key: float(values[i]) for i, key in enumerate(keys)}
    if "action_delay_steps" in xi:
        xi["action_delay_steps"] = int(round(xi["action_delay_steps"]))
    return xi


def normalize_xi(xi, bounds=None):
    bounds = DEFAULT_DISTURBANCE_BOUNDS if bounds is None else bounds
    out = []
    for key in DISTURBANCE_KEYS:
        low, high = bounds[key]
        denom = float(high) - float(low)
        if abs(denom) < 1e-12:
            out.append(0.0)
        else:
            out.append((float(xi[key]) - float(low)) / denom)
    return np.clip(np.asarray(out, dtype=np.float32), 0.0, 1.0)


def denormalize_xi(x_norm, bounds=None):
    bounds = DEFAULT_DISTURBANCE_BOUNDS if bounds is None else bounds
    values = np.clip(np.asarray(x_norm, dtype=np.float32).reshape(-1), 0.0, 1.0)
    if values.size != len(DISTURBANCE_KEYS):
        raise ValueError(f"Expected {len(DISTURBANCE_KEYS)} values, got {values.size}")
    xi = {}
    for i, key in enumerate(DISTURBANCE_KEYS):
        low, high = bounds[key]
        value = float(low) + float(values[i]) * (float(high) - float(low))
        xi[key] = int(round(value)) if key == "action_delay_steps" else float(value)
    return xi
