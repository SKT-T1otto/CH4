import torch


class OUNoise:
    def __init__(
        self,
        action_dimension,
        mu=0.0,
        theta=0.15,
        sigma=0.2,
        scale=1.0,
        device=None,
        dtype=torch.float32,
    ):
        self.action_dimension = int(action_dimension)
        self.mu = float(mu)
        self.theta = float(theta)
        self.sigma = float(sigma)
        self.scale = float(scale)
        self.dtype = dtype
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.state = torch.full((self.action_dimension,), self.mu, dtype=self.dtype, device=self.device)

    def to(self, device=None, dtype=None):
        if device is not None:
            self.device = torch.device(device)
        if dtype is not None:
            self.dtype = dtype
        self.state = self.state.to(device=self.device, dtype=self.dtype)
        return self

    def reset(self):
        self.state.fill_(self.mu)

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * torch.randn_like(self.state)
        self.state = self.state + dx
        return self.state * self.scale
