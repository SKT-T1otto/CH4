import csv
import os

import numpy as np
import torch

from registry.rbe_disturbance import DISTURBANCE_KEYS, normalize_xi


OUTCOME_FIELDS = [
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


class REBOutcomeDataset:
    def __init__(self, max_steps, space_diag, keys=None):
        self.max_steps = max(1.0, float(max_steps))
        self.space_diag = max(1e-6, float(space_diag))
        self.keys = list(DISTURBANCE_KEYS if keys is None else keys)
        self.rows = []

    @staticmethod
    def _parse_bool(value):
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float, np.number)):
            return float(value) != 0.0
        if isinstance(value, str):
            text = value.strip().lower()
            if text in ("true", "1", "yes", "y", "t"):
                return True
            if text in ("false", "0", "no", "n", "f", ""):
                return False
        return bool(value)

    @staticmethod
    def _parse_float(value, field):
        try:
            if isinstance(value, str) and value.strip() == "":
                return 0.0
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Could not convert field {field!r}={value!r} to float") from exc

    def _validate_metrics(self, metrics):
        missing = [key for key in self.keys if key not in metrics]
        required = ["success_flag", "recovery_time", "safety_cost", "final_distance"]
        missing.extend(field for field in required if field not in metrics)
        if missing:
            raise KeyError(f"REB outcome metrics missing required fields: {missing}")

    def append(self, metrics: dict):
        self._validate_metrics(metrics)
        row = dict(metrics)
        for key in self.keys:
            row[key] = self._parse_float(row[key], key)
        row["success_flag"] = self._parse_bool(row["success_flag"])
        if "found_flag" in row:
            row["found_flag"] = self._parse_bool(row["found_flag"])
        for field in OUTCOME_FIELDS:
            if field in ("success_flag", "found_flag"):
                continue
            if field in row:
                row[field] = self._parse_float(row[field], field)
        self.rows.append(row)

    def __len__(self):
        return len(self.rows)

    def to_training_arrays(self):
        n = len(self.rows)
        xi_norm = np.zeros((n, len(self.keys)), dtype=np.float32)
        success = np.zeros((n, 1), dtype=np.float32)
        t_rec_norm = np.zeros((n, 1), dtype=np.float32)
        c_safe_log = np.zeros((n, 1), dtype=np.float32)
        d_final_norm = np.zeros((n, 1), dtype=np.float32)

        for i, row in enumerate(self.rows):
            xi = {key: row[key] for key in self.keys}
            xi_norm[i] = normalize_xi(xi, bounds=None)
            success[i, 0] = 1.0 if self._parse_bool(row["success_flag"]) else 0.0
            t_rec_norm[i, 0] = np.clip(float(row["recovery_time"]) / self.max_steps, 0.0, 1.0)
            c_safe_log[i, 0] = np.log1p(max(0.0, float(row["safety_cost"])))
            d_final_norm[i, 0] = np.clip(float(row["final_distance"]) / self.space_diag, 0.0, 1.0)

        return xi_norm, success, t_rec_norm, c_safe_log, d_final_norm

    def sample_batch(self, batch_size, rng=None):
        if len(self.rows) == 0:
            raise ValueError("Cannot sample from an empty REBOutcomeDataset.")
        batch_size = int(max(1, batch_size))
        n = len(self.rows)
        replace = batch_size > n
        if rng is None:
            indices = np.random.choice(n, size=batch_size, replace=replace)
        elif hasattr(rng, "choice"):
            indices = rng.choice(n, size=batch_size, replace=replace)
        else:
            raise TypeError("rng must provide a choice() method")

        xi, success, t_rec, c_safe, d_final = self.to_training_arrays()
        indices = np.asarray(indices, dtype=np.int64)
        return {
            "xi": torch.as_tensor(xi[indices], dtype=torch.float32),
            "success": torch.as_tensor(success[indices], dtype=torch.float32),
            "t_rec": torch.as_tensor(t_rec[indices], dtype=torch.float32),
            "c_safe": torch.as_tensor(c_safe[indices], dtype=torch.float32),
            "d_final": torch.as_tensor(d_final[indices], dtype=torch.float32),
        }

    def _csv_fieldnames(self):
        base = [*self.keys, *OUTCOME_FIELDS]
        extras = sorted({key for row in self.rows for key in row.keys()} - set(base))
        return [*base, *extras]

    def save_csv(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fieldnames = self._csv_fieldnames()
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.rows:
                writer.writerow({key: row.get(key, "") for key in fieldnames})

    def load_csv(self, path):
        self.rows = []
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.append(row)
        return self
