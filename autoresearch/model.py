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
    """TD3 twin critics: two independent Q-networks (clipped double Q-learning).
    forward() returns (q1, q2); q1_only() returns Q1 for the actor loss.
    Twin critics prevent the Q-value overestimation that stalls DDPG under sparse
    -1/0 rewards."""

    def __init__(self, input_dim: int = 32, hidden_dim: int = 256) -> None:
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        value = torch.cat([state, action], dim=-1)
        return self.q1(value), self.q2(value)

    def q1_only(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        value = torch.cat([state, action], dim=-1)
        return self.q1(value)
