# -*- coding: utf-8 -*-
"""REB-guided disturbance sampler for Chapter-4 RBE-full training."""

from __future__ import annotations

import csv
import hashlib
import math
import os
import warnings

import numpy as np

from registry.rbe_disturbance import (
    DEFAULT_DISTURBANCE_BOUNDS,
    DISTURBANCE_KEYS,
    denormalize_xi,
    nominal_disturbance,
    normalize_xi,
    sample_uniform_disturbance,
)


class RBEDisturbanceSampler:
    def __init__(
        self,
        boundary_csv,
        high_risk_csv,
        rng=None,
        boundary_ratio=0.50,
        uniform_ratio=0.30,
        high_risk_ratio=0.20,
        nominal_ratio=0.10,
        jitter_std=0.03,
        jitter_prob=0.80,
        include_flow_phase=True,
        strict_protocol=False,
        expected_boundary_count=None,
        expected_high_risk_count=None,
        expected_boundary_group=None,
        expected_high_risk_group=None,
        require_ratio_sum_one=False,
        high_risk_required=False,
        method_role_labels=None,
        score_rule_sha256=None,
        source_selected_candidates_sha256=None,
    ):
        self.rng = np.random.default_rng() if rng is None else rng
        self.strict_protocol = bool(strict_protocol)
        self.expected_boundary_group = expected_boundary_group
        self.expected_high_risk_group = expected_high_risk_group
        self.method_role_labels = dict(method_role_labels or {})
        self.score_rule_sha256 = score_rule_sha256
        self.source_selected_candidates_sha256 = source_selected_candidates_sha256
        self.boundary_csv = str(boundary_csv)
        self.high_risk_csv = str(high_risk_csv)
        self.boundary_csv_sha256 = self._sha256(boundary_csv) if boundary_csv and os.path.exists(boundary_csv) else None
        self.high_risk_csv_sha256 = self._sha256(high_risk_csv) if high_risk_csv and os.path.exists(high_risk_csv) else None
        self.boundary_rows = self._load_rows(
            boundary_csv,
            required=True,
            expected_count=expected_boundary_count,
            expected_group=expected_boundary_group,
            expected_method_role=self.method_role_labels.get("boundary"),
        )
        self.high_risk_rows = self._load_rows(
            high_risk_csv,
            required=bool(high_risk_required or self.strict_protocol),
            expected_count=expected_high_risk_count,
            expected_group=expected_high_risk_group,
            expected_method_role=self.method_role_labels.get("high_risk"),
        )
        self.jitter_std = float(jitter_std)
        self.jitter_prob = float(jitter_prob)
        if not math.isfinite(self.jitter_std) or self.jitter_std < 0.0:
            raise ValueError(f"RBE sampler jitter_std must be finite and nonnegative, got {jitter_std!r}")
        if not math.isfinite(self.jitter_prob) or not 0.0 <= self.jitter_prob <= 1.0:
            raise ValueError(f"RBE sampler jitter_prob must be within [0, 1], got {jitter_prob!r}")
        self.include_flow_phase = bool(include_flow_phase)
        self.last_source = None
        self.last_rank = None
        self.last_method_role = None
        self.last_candidate_pool_index = None
        self.last_jitter_applied = False
        self.last_jitter_l2 = 0.0
        self.last_action_delay_before_jitter = None
        self.last_action_delay_after_jitter = None

        boundary_ratio = float(boundary_ratio)
        uniform_ratio = float(uniform_ratio)
        high_risk_ratio = float(high_risk_ratio)
        nominal_ratio = float(nominal_ratio)
        raw_ratios = {
            "boundary_core": boundary_ratio,
            "uniform_coverage": uniform_ratio,
            "composite_high_risk_aux": high_risk_ratio,
            "nominal_anchor": nominal_ratio,
        }
        self._validate_ratios(raw_ratios, require_ratio_sum_one=require_ratio_sum_one or self.strict_protocol)
        if not self.high_risk_rows and not self.strict_protocol:
            warnings.warn(
                f"high_risk_csv not found or empty; merging high_risk_ratio into boundary_ratio: {high_risk_csv}",
                RuntimeWarning,
            )
            boundary_ratio += high_risk_ratio
            high_risk_ratio = 0.0
        total = boundary_ratio + uniform_ratio + high_risk_ratio + nominal_ratio
        if total <= 0.0:
            raise ValueError("RBE sampler ratios must sum to a positive value.")
        if self.strict_protocol:
            if abs(total - 1.0) > 1e-12:
                raise ValueError("strict RBE sampler ratios must sum to 1.0 exactly within 1e-12.")
            self.boundary_ratio = boundary_ratio
            self.uniform_ratio = uniform_ratio
            self.high_risk_ratio = high_risk_ratio
            self.nominal_ratio = nominal_ratio
            self.ratios_were_normalized = False
        else:
            self.boundary_ratio = boundary_ratio / total
            self.uniform_ratio = uniform_ratio / total
            self.high_risk_ratio = high_risk_ratio / total
            self.nominal_ratio = nominal_ratio / total
            self.ratios_were_normalized = abs(total - 1.0) > 1e-12
        self.raw_ratios = dict(raw_ratios)
        self.effective_ratios = {
            "boundary_core": self.boundary_ratio,
            "uniform_coverage": self.uniform_ratio,
            "composite_high_risk_aux": self.high_risk_ratio,
            "nominal_anchor": self.nominal_ratio,
        }
        self.boundary_count = len(self.boundary_rows)
        self.high_risk_count = len(self.high_risk_rows)
        if self.strict_protocol:
            self._audit_strict_candidate_sets()

    @staticmethod
    def _sha256(path):
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _strict_csv_int(value, label):
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{label} must be an integer, not bool")
        text = str(value).strip()
        if not text:
            raise ValueError(f"{label} must not be empty")
        try:
            number = float(text)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} must be an integer: {value!r}") from exc
        if not math.isfinite(number) or not number.is_integer():
            raise ValueError(f"{label} must be an integer: {value!r}")
        return int(number)

    @staticmethod
    def _validate_ratios(ratios, require_ratio_sum_one=False):
        total = 0.0
        for name, value in ratios.items():
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"RBE sampler ratio is not finite: {name}={value!r}")
            if value < 0.0:
                raise ValueError(f"RBE sampler ratio is negative: {name}={value!r}")
            total += value
        if require_ratio_sum_one and abs(total - 1.0) > 1e-12:
            raise ValueError(f"RBE sampler ratios must sum to 1.0, got {total}")

    def _load_rows(self, path, required, expected_count=None, expected_group=None, expected_method_role=None):
        if not path or not os.path.exists(path):
            if required:
                raise FileNotFoundError(f"RBE boundary csv does not exist: {path}")
            return []
        rows = []
        with open(path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            fieldnames = set(reader.fieldnames or [])
            missing = [key for key in DISTURBANCE_KEYS if key not in fieldnames]
            if missing:
                raise KeyError(f"RBE candidate csv missing fields {missing}: {path}")
            for row in reader:
                xi = {}
                for key in DISTURBANCE_KEYS:
                    raw_value = row[key]
                    if key == "action_delay_steps":
                        if self.strict_protocol:
                            xi[key] = self._strict_csv_int(raw_value, "strict RBE candidate action_delay_steps")
                        else:
                            number = float(raw_value)
                            if not math.isfinite(number):
                                raise ValueError(f"RBE candidate action_delay_steps is not finite: {raw_value!r}")
                            xi[key] = int(round(number))
                    else:
                        xi[key] = float(raw_value)
                        if not math.isfinite(xi[key]):
                            raise ValueError(f"RBE candidate {key} is not finite: {raw_value!r}")
                    low, high = DEFAULT_DISTURBANCE_BOUNDS[key]
                    if not (float(low) <= float(xi[key]) <= float(high)):
                        raise ValueError(f"RBE candidate {key} outside bounds: {xi[key]} not in [{low}, {high}]")
                rank = None
                if "rank" in row and str(row.get("rank", "")).strip() != "":
                    rank = (
                        self._strict_csv_int(row["rank"], "strict RBE candidate rank")
                        if self.strict_protocol
                        else int(round(float(row["rank"])))
                    )
                candidate_rank = None
                if "candidate_rank" in row and str(row.get("candidate_rank", "")).strip() != "":
                    candidate_rank = (
                        self._strict_csv_int(row["candidate_rank"], "strict RBE candidate candidate_rank")
                        if self.strict_protocol
                        else int(round(float(row["candidate_rank"])))
                    )
                candidate_pool_index = None
                if "candidate_pool_index" in row and str(row.get("candidate_pool_index", "")).strip() != "":
                    candidate_pool_index = (
                        self._strict_csv_int(row["candidate_pool_index"], "strict RBE candidate candidate_pool_index")
                        if self.strict_protocol
                        else int(round(float(row["candidate_pool_index"])))
                    )
                if self.strict_protocol:
                    if rank is None or candidate_rank is None:
                        raise ValueError(f"strict RBE candidate missing rank/candidate_rank: {path}")
                    if candidate_rank != rank:
                        raise ValueError(f"strict RBE candidate rank/candidate_rank mismatch: {path}")
                    if candidate_pool_index is None or candidate_pool_index < 0:
                        raise ValueError(f"strict RBE candidate candidate_pool_index must be a nonnegative integer: {path}")
                    if expected_group is not None and str(row.get("source_group", "")) != str(expected_group):
                        raise ValueError(f"strict RBE candidate source_group mismatch: {row.get('source_group')!r}")
                    if expected_method_role is not None and str(row.get("method_role", "")) != str(expected_method_role):
                        raise ValueError(f"strict RBE candidate method_role mismatch: {row.get('method_role')!r}")
                    if self.score_rule_sha256 is not None and str(row.get("score_rule_sha256", "")) != str(self.score_rule_sha256):
                        raise ValueError("strict RBE candidate score_rule_sha256 mismatch")
                    if (
                        self.source_selected_candidates_sha256 is not None
                        and str(row.get("source_selected_candidates_sha256", "")) != str(self.source_selected_candidates_sha256)
                    ):
                        raise ValueError("strict RBE candidate source_selected_candidates_sha256 mismatch")
                rows.append(
                    {
                        "xi": xi,
                        "rank": rank,
                        "candidate_pool_index": candidate_pool_index,
                        "source_group": row.get("source_group", ""),
                        "method_role": row.get("method_role", ""),
                        "score_rule_sha256": row.get("score_rule_sha256", ""),
                        "source_selected_candidates_sha256": row.get("source_selected_candidates_sha256", ""),
                    }
                )
        if required and not rows:
            raise ValueError(f"RBE boundary csv is empty: {path}")
        if self.strict_protocol and expected_count is not None and len(rows) != int(expected_count):
            raise ValueError(f"strict RBE candidate count mismatch for {path}: {len(rows)} != {expected_count}")
        if self.strict_protocol and rows:
            ranks = [row["rank"] for row in rows]
            expected_ranks = list(range(1, len(rows) + 1))
            if sorted(ranks) != expected_ranks or len(set(ranks)) != len(ranks):
                raise ValueError(f"strict RBE candidate ranks must be complete and unique 1..N: {path}")
            if ranks != expected_ranks:
                raise ValueError(f"strict RBE candidate rows must remain in rank order: {path}")
        return rows

    def _audit_strict_candidate_sets(self):
        boundary_vectors = [tuple(float(row["xi"][key]) for key in DISTURBANCE_KEYS) for row in self.boundary_rows]
        high_vectors = [tuple(float(row["xi"][key]) for key in DISTURBANCE_KEYS) for row in self.high_risk_rows]
        if len(boundary_vectors) != len(set(boundary_vectors)):
            raise ValueError("strict RBE boundary candidates contain duplicate 9D vectors")
        if len(high_vectors) != len(set(high_vectors)):
            raise ValueError("strict RBE high-risk candidates contain duplicate 9D vectors")
        if set(boundary_vectors) & set(high_vectors):
            raise ValueError("strict RBE boundary and high-risk candidates overlap in 9D space")
        if self.ratios_were_normalized:
            raise ValueError("strict RBE sampler must not normalize ratios")
        if self.raw_ratios != self.effective_ratios:
            raise ValueError("strict RBE raw/effective ratios must match exactly")

    def _random(self):
        if hasattr(self.rng, "random"):
            return float(self.rng.random())
        return float(np.random.random())

    def _integers(self, high):
        if hasattr(self.rng, "integers"):
            return int(self.rng.integers(0, int(high)))
        if hasattr(self.rng, "randint"):
            return int(self.rng.randint(0, int(high)))
        return int(np.random.randint(0, int(high)))

    def _normal(self, shape):
        if hasattr(self.rng, "normal"):
            return self.rng.normal(0.0, self.jitter_std, size=shape)
        return np.random.normal(0.0, self.jitter_std, size=shape)

    def _uniform(self, low, high):
        if hasattr(self.rng, "uniform"):
            return float(self.rng.uniform(low, high))
        return float(np.random.uniform(low, high))

    def _choice_row(self, rows):
        return rows[self._integers(len(rows))]

    def _with_phase(self, xi):
        out = dict(xi)
        if self.include_flow_phase:
            out["flow_phase_x"] = self._uniform(0.0, 2.0 * np.pi)
            out["flow_phase_y"] = self._uniform(0.0, 2.0 * np.pi)
        return out

    def _jitter(self, xi, allow_jitter=True):
        out = {key: xi[key] for key in DISTURBANCE_KEYS}
        self.last_jitter_applied = False
        self.last_jitter_l2 = 0.0
        self.last_action_delay_before_jitter = int(out["action_delay_steps"])
        self.last_action_delay_after_jitter = int(out["action_delay_steps"])
        if (not allow_jitter) or self.jitter_std <= 0.0 or self._random() > self.jitter_prob:
            return self._with_phase(out)
        x_norm = normalize_xi(out, bounds=DEFAULT_DISTURBANCE_BOUNDS).astype(np.float32)
        before = x_norm.copy()
        x_norm = np.clip(x_norm + self._normal(x_norm.shape).astype(np.float32), 0.0, 1.0)
        self.last_jitter_applied = True
        self.last_jitter_l2 = float(np.linalg.norm(x_norm.astype(np.float64) - before.astype(np.float64)))
        out = denormalize_xi(x_norm, bounds=DEFAULT_DISTURBANCE_BOUNDS)
        for key in DISTURBANCE_KEYS:
            low, high = DEFAULT_DISTURBANCE_BOUNDS[key]
            if key != "action_delay_steps":
                out[key] = min(max(float(out[key]), float(low)), float(high))
        out["action_delay_steps"] = int(round(out["action_delay_steps"]))
        self.last_action_delay_after_jitter = int(out["action_delay_steps"])
        return self._with_phase(out)

    def audit(self):
        return {
            "boundary_count": self.boundary_count,
            "high_risk_count": self.high_risk_count,
            "raw_ratios": dict(self.raw_ratios),
            "effective_ratios": dict(self.effective_ratios),
            "ratios_were_normalized": bool(self.ratios_were_normalized),
            "strict_protocol": bool(self.strict_protocol),
            "boundary_csv_sha256": self.boundary_csv_sha256,
            "high_risk_csv_sha256": self.high_risk_csv_sha256,
            "jitter_std": self.jitter_std,
            "jitter_prob": self.jitter_prob,
            "include_flow_phase": self.include_flow_phase,
            "last_source": self.last_source,
            "last_method_role": self.last_method_role,
            "last_rank": self.last_rank,
            "last_candidate_pool_index": self.last_candidate_pool_index,
            "last_jitter_applied": self.last_jitter_applied,
            "last_jitter_l2": self.last_jitter_l2,
            "last_action_delay_before_jitter": self.last_action_delay_before_jitter,
            "last_action_delay_after_jitter": self.last_action_delay_after_jitter,
        }

    def _set_last(self, source, row=None):
        self.last_source = source
        self.last_method_role = self.method_role_labels.get(source, "")
        self.last_rank = None if row is None else row.get("rank")
        self.last_candidate_pool_index = None if row is None else row.get("candidate_pool_index")

    def sample(self):
        u = self._random()
        if u < self.boundary_ratio:
            row = self._choice_row(self.boundary_rows)
            self._set_last("boundary", row)
            return self._jitter(row["xi"])
        if u < self.boundary_ratio + self.high_risk_ratio and self.high_risk_rows:
            row = self._choice_row(self.high_risk_rows)
            self._set_last("high_risk", row)
            return self._jitter(row["xi"])
        if u < self.boundary_ratio + self.high_risk_ratio + self.uniform_ratio:
            self._set_last("uniform")
            self.last_jitter_applied = False
            self.last_jitter_l2 = 0.0
            self.last_action_delay_before_jitter = None
            self.last_action_delay_after_jitter = None
            return sample_uniform_disturbance(rng=self.rng, include_flow_phase=self.include_flow_phase)
        self._set_last("nominal")
        return self._jitter(nominal_disturbance(), allow_jitter=False)
