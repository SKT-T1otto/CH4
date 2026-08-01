import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _torch_load_checkpoint(path, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


class REBModel(nn.Module):
    def __init__(self, xi_dim=9, hidden_dim=128):
        super().__init__()
        self.xi_dim = int(xi_dim)
        self.hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(self.xi_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 4),
        )

    def forward(self, xi):
        raw = self.net(xi)
        return {
            "success_logit": raw[:, 0:1],
            "t_rec": torch.sigmoid(raw[:, 1:2]),
            "c_safe": F.softplus(raw[:, 2:3]),
            "d_final": torch.sigmoid(raw[:, 3:4]),
        }


class FoundAwareREBModel(nn.Module):
    def __init__(self, xi_dim=9, hidden_dim=128):
        super().__init__()
        self.xi_dim = int(xi_dim)
        self.hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(self.xi_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 6),
        )

    def forward(self, xi):
        raw = self.net(xi)
        return {
            "found_logit": raw[:, 0:1],
            "success_logit": raw[:, 1:2],
            "success_if_found_logit": raw[:, 2:3],
            "t_rec": torch.sigmoid(raw[:, 3:4]),
            "c_safe": F.softplus(raw[:, 4:5]),
            "d_final": torch.sigmoid(raw[:, 5:6]),
        }


class REBTrainer:
    def __init__(
        self,
        model,
        lr=1e-3,
        device=None,
        lambda_t=1.0,
        lambda_safe=1.0,
        lambda_dist=1.0,
    ):
        self.device = self._resolve_device(device)
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(lr))
        self.lambda_t = float(lambda_t)
        self.lambda_safe = float(lambda_safe)
        self.lambda_dist = float(lambda_dist)

    @staticmethod
    def _resolve_device(device=None):
        if device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(device)
        if device.type == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return device

    def _to_device_batch(self, batch):
        return {
            key: value.to(device=self.device, dtype=torch.float32, non_blocking=True)
            for key, value in batch.items()
        }

    def _loss(self, pred, batch):
        bce = F.binary_cross_entropy_with_logits(pred["success_logit"], batch["success"])
        t_mse = F.mse_loss(pred["t_rec"], batch["t_rec"])
        safe_mse = F.mse_loss(pred["c_safe"], batch["c_safe"])
        dist_mse = F.mse_loss(pred["d_final"], batch["d_final"])
        loss = bce + self.lambda_t * t_mse + self.lambda_safe * safe_mse + self.lambda_dist * dist_mse
        return loss, bce, t_mse, safe_mse, dist_mse

    def update(self, dataset, batch_size=64, updates=1, rng=None):
        if len(dataset) == 0:
            return None

        totals = {
            "reb_loss": 0.0,
            "reb_bce": 0.0,
            "reb_t_mse": 0.0,
            "reb_safe_mse": 0.0,
            "reb_dist_mse": 0.0,
        }
        updates = int(max(1, updates))
        self.model.train()
        for _ in range(updates):
            batch = self._to_device_batch(dataset.sample_batch(batch_size, rng=rng))
            pred = self.model(batch["xi"])
            loss, bce, t_mse, safe_mse, dist_mse = self._loss(pred, batch)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            totals["reb_loss"] += float(loss.detach().item())
            totals["reb_bce"] += float(bce.detach().item())
            totals["reb_t_mse"] += float(t_mse.detach().item())
            totals["reb_safe_mse"] += float(safe_mse.detach().item())
            totals["reb_dist_mse"] += float(dist_mse.detach().item())

        return {key: value / updates for key, value in totals.items()}

    @torch.no_grad()
    def predict(self, xi_norm):
        self.model.eval()
        if torch.is_tensor(xi_norm):
            xi = xi_norm.detach().to(device=self.device, dtype=torch.float32)
        else:
            xi = torch.as_tensor(np.asarray(xi_norm, dtype=np.float32), device=self.device)
        if xi.dim() == 1:
            xi = xi.unsqueeze(0)
        pred = self.model(xi)
        return {
            "p_success": torch.sigmoid(pred["success_logit"]).detach().cpu().numpy(),
            "t_rec": pred["t_rec"].detach().cpu().numpy(),
            "c_safe": pred["c_safe"].detach().cpu().numpy(),
            "d_final": pred["d_final"].detach().cpu().numpy(),
        }

    def save(self, path, extra=None):
        checkpoint = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "extra": {} if extra is None else dict(extra),
        }
        torch.save(checkpoint, path)

    def load(self, path):
        checkpoint = _torch_load_checkpoint(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        if "optimizer_state" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            for state in self.optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = value.to(self.device)
        return checkpoint.get("extra", {})
