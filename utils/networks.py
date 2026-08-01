import torch
import torch.nn as nn
import torch.nn.functional as F


class Identity(nn.Module):
    def forward(self, x):
        return x


class MLPNetwork(nn.Module):
    def __init__(
        self,
        input_dim,
        out_dim,
        hidden_dim=128,
        nonlin=F.relu,
        constrain_out=False,
        norm_in=True,
        discrete_action=True,
        dropout_rate=0.0,
    ):
        super().__init__()
        self.in_fn = nn.LayerNorm(input_dim) if norm_in else Identity()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.nonlin = nonlin
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else Identity()
        if constrain_out and not discrete_action:
            nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
            nn.init.uniform_(self.fc3.bias, -3e-3, 3e-3)
            self.out_fn = torch.tanh
        else:
            self.out_fn = lambda x: x

    def forward(self, x):
        x = self.in_fn(x)
        x = self.nonlin(self.fc1(x))
        x = self.dropout(x)
        x = self.nonlin(self.fc2(x))
        x = self.dropout(x)
        return self.out_fn(self.fc3(x))
