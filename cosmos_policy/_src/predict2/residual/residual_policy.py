# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch.nn as nn


class ResidualPolicy(nn.Module):
    """Goal-conditioned tracking residual on top of frozen Cosmos Predict2.

    Cosmos predicts (action_chunk, future_image). After Cosmos's action chunk
    is executed in the environment, the residual takes the new observation
    and the previously predicted future_image (the goal) and emits a
    corrective action chunk that drives the actual obs toward the predicted
    future.

    Pure nn.Module — RL-specific concerns (action sampling, logprob, value
    head) live on the RLinf side, which wraps this module.

    Inputs:
        - obs:  current observation after Cosmos's action chunk executed.
        - goal: Cosmos's previously predicted future image (latent or pixels).
    Output:
        - corrective action chunk, same shape as Cosmos's action.
    """

    def __init__(self, cfg):
        super().__init__()
        raise NotImplementedError

    def forward(self, obs, goal):
        raise NotImplementedError
