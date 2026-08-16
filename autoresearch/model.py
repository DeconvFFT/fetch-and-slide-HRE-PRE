from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class Actor(nn.Module):
    def __init__(self, input_dim: int = 28, hidden_dim: int = 256, action_dim: int = 4) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        state = F.relu(self.fc1(state))
        state = F.relu(self.fc2(state))
        state = F.relu(self.fc3(state))
        return torch.tanh(self.mu(state))


class Critic(nn.Module):
    def __init__(self, input_dim: int = 32, hidden_dim: int = 256) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.q = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        value = torch.cat([state, action], dim=-1)
        value = F.relu(self.fc1(value))
        value = F.relu(self.fc2(value))
        value = F.relu(self.fc3(value))
        return self.q(value)
