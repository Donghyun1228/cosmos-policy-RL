# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import torch.nn as nn


def _mlp(
    in_dim: int,
    out_dim: int,
    hidden_dim: int,
    num_hidden_layers: int,
) -> nn.Sequential:
    """LayerNorm-MLP trunk with ReLU activations.

    LayerNorm stabilizes high-UTD off-policy training (RLPD / DroQ
    style), which is what the residual will be trained under.
    """
    layers: list[nn.Module] = []
    prev = in_dim
    for _ in range(num_hidden_layers):
        layers.append(nn.Linear(prev, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        prev = hidden_dim
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class ResidualActor(nn.Module):
    """6-DOF (delta end-effector) Gaussian residual actor over an action chunk.

    Inputs:
        - ``z_obs``  (B, token_dim): RL token of the current image.
        - ``z_goal`` (B, token_dim): RL token of cosmos's previously
          predicted future image (the goal).

    Outputs:
        - ``mean``    (B, chunk_size, action_dim)
        - ``log_std`` (B, chunk_size, action_dim) -- broadcast from a
          per-action-dim parameter (or buffer if ``learnable_log_std=False``).

    The final linear layer is initialized to zero so that at the start of
    training the actor outputs an all-zero residual, and the wrapping RL
    code passes cosmos's reference action through unchanged.
    """

    def __init__(
        self,
        token_dim: int = 768,
        chunk_size: int = 1,
        action_dim: int = 6,
        hidden_dim: int = 256,
        num_hidden_layers: int = 2,
        log_std_init: float = -1.0,
        learnable_log_std: bool = True,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.chunk_size = chunk_size
        self.action_dim = action_dim

        self.trunk = _mlp(
            in_dim=2 * token_dim,
            out_dim=chunk_size * action_dim,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
        )
        # Pass-through init: residual ~= 0 at step 0.
        nn.init.zeros_(self.trunk[-1].weight)
        nn.init.zeros_(self.trunk[-1].bias)

        log_std = torch.full((action_dim,), float(log_std_init))
        if learnable_log_std:
            self.log_std = nn.Parameter(log_std)
        else:
            self.register_buffer("log_std", log_std)

    def forward(
        self, z_obs: torch.Tensor, z_goal: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([z_obs, z_goal], dim=-1)
        mean = self.trunk(x).reshape(-1, self.chunk_size, self.action_dim)
        log_std = self.log_std.view(1, 1, -1).expand_as(mean)
        return mean, log_std


class ResidualCritic(nn.Module):
    """Twin Q-functions over (obs, goal, action chunk).

    Returns ``(B, num_critics)`` so the wrapping RL code can apply TD3-style
    ``min`` over the ensemble for the target value.
    """

    def __init__(
        self,
        token_dim: int = 768,
        chunk_size: int = 1,
        action_dim: int = 6,
        hidden_dim: int = 256,
        num_hidden_layers: int = 2,
        num_critics: int = 2,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.num_critics = num_critics

        in_dim = 2 * token_dim + chunk_size * action_dim
        self.qs = nn.ModuleList(
            [
                _mlp(
                    in_dim=in_dim,
                    out_dim=1,
                    hidden_dim=hidden_dim,
                    num_hidden_layers=num_hidden_layers,
                )
                for _ in range(num_critics)
            ]
        )

    def forward(
        self,
        z_obs: torch.Tensor,
        z_goal: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        action_flat = action.reshape(action.shape[0], -1)
        x = torch.cat([z_obs, z_goal, action_flat], dim=-1)
        return torch.cat([q(x) for q in self.qs], dim=-1)


class ResidualPolicy(nn.Module):
    """Goal-conditioned tracking residual on top of frozen Cosmos Predict2.

    Cosmos predicts (action_chunk, future_image). After Cosmos's action
    chunk is executed in the environment, this residual takes the new
    observation token (``z_obs``) and the goal token of cosmos's previously
    predicted future image (``z_goal``), and emits a 6-DOF delta-EE
    corrective action chunk that drives the actual obs toward the goal.

    Tokens are produced by the frozen ``RLTokenAutoencoder`` encoder; this
    module is unaware of how they were produced.

    The 7-th action dimension (gripper) is not part of the residual: the
    environment wrapper concatenates Cosmos's gripper prediction onto each
    6-D residual step before sending the full action to the simulator.

    Pure ``nn.Module`` -- action sampling, log-prob, target Q computation,
    BC regularizer toward Cosmos's reference action, etc. live on the
    RLinf side that wraps this.
    """

    def __init__(
        self,
        token_dim: int = 768,
        chunk_size: int = 1,
        action_dim: int = 6,
        hidden_dim: int = 256,
        num_hidden_layers: int = 2,
        num_critics: int = 2,
        log_std_init: float = -1.0,
        learnable_log_std: bool = True,
    ):
        super().__init__()
        self.actor = ResidualActor(
            token_dim=token_dim,
            chunk_size=chunk_size,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            log_std_init=log_std_init,
            learnable_log_std=learnable_log_std,
        )
        self.critic = ResidualCritic(
            token_dim=token_dim,
            chunk_size=chunk_size,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            num_critics=num_critics,
        )

    def forward(
        self, z_obs: torch.Tensor, z_goal: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convenience alias for the actor pass; use ``self.critic`` for Q."""
        return self.actor(z_obs, z_goal)
