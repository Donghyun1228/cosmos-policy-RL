# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Free-function glue between cosmos's prediction pipeline and the residual.

Three pieces:

  - ``GoalState``: lightweight container for the future-image goal that
    cosmos predicts alongside its action chunk. Holds both the raw VAE
    latent (for visualization / debugging) and the pre-computed RL token
    (used directly by the residual every step).

  - ``get_action_with_goal_state``: thin wrapper around
    ``cosmos_utils.get_action`` that additionally extracts the
    future-image VAE latent from cosmos's denoised output and encodes
    it through the frozen RL-token AE. Returns ``(action_chunk,
    GoalState)`` so the env can run cosmos's chunk as-is and still have
    a goal handy for the residual.

  - ``get_residual_action``: one-shot 6-DOF residual correction. Encodes
    the new observation, samples (or argmaxes) the residual actor, and
    composes with cosmos's last action (delta-EE summed; gripper
    passes through unchanged).

This file intentionally does **not** subclass cosmos's model — the three
components (cosmos, AE, residual) stay separate ``nn.Module``s and these
helpers tie them together at call sites, mirroring how cosmos itself
exposes ``get_action`` / ``get_future_state_prediction`` etc. as
free functions in ``cosmos_utils``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from cosmos_policy._src.predict2.residual.residual_policy import ResidualPolicy
from cosmos_policy._src.predict2.residual.rl_token_autoencoder import (
    RLTokenAutoencoder,
)
from cosmos_policy.experiments.robot.cosmos_utils import DEVICE

TEMPORAL_COMPRESSION_FACTOR = 4


@dataclass
class ResidualEvalConfig:
    """Residual-specific config fields, mirroring cosmos's flat dataclass style.

    Compose with cosmos's ``PolicyEvalConfig`` via either multiple
    inheritance (no field overlap) or by adding these fields directly to
    your eval-script dataclass. ``get_residual_components(cfg)`` reads
    these by name.
    """

    rl_token_ae_ckpt_path: str = ""
    residual_policy_ckpt_path: str = ""
    residual_chunk_size: int = 1
    residual_action_dim: int = 6
    residual_hidden_dim: int = 256
    residual_num_hidden_layers: int = 2
    residual_log_std_init: float = -1.0
    residual_learnable_log_std: bool = True
    residual_num_critics: int = 2


_RESIDUAL_KWARG_FIELDS: tuple[tuple[str, str], ...] = (
    ("residual_chunk_size", "chunk_size"),
    ("residual_action_dim", "action_dim"),
    ("residual_hidden_dim", "hidden_dim"),
    ("residual_num_hidden_layers", "num_hidden_layers"),
    ("residual_log_std_init", "log_std_init"),
    ("residual_learnable_log_std", "learnable_log_std"),
    ("residual_num_critics", "num_critics"),
)


def get_residual_components(
    cfg,
) -> tuple[RLTokenAutoencoder, ResidualPolicy]:
    """Load AE (frozen) + residual policy from a flat cfg.

    Mirrors ``cosmos_utils.get_model(cfg)``: takes the eval/training cfg
    and returns ready-to-use modules placed on ``DEVICE``. ``cfg`` only
    needs to expose the ``residual_*`` / ``rl_token_ae_ckpt_path`` /
    ``residual_policy_ckpt_path`` fields; missing optional fields fall
    back to ``ResidualPolicy``'s constructor defaults.

    The AE is moved to ``eval()`` and ``requires_grad_(False)``. The
    residual stays in ``train`` mode so optimizers can pick it up.
    """
    ae = RLTokenAutoencoder()
    if cfg.rl_token_ae_ckpt_path:
        sd = torch.load(cfg.rl_token_ae_ckpt_path, map_location="cpu")
        ae.load_state_dict(sd)
    ae = ae.eval().to(DEVICE)
    for p in ae.parameters():
        p.requires_grad_(False)

    residual_kwargs: dict[str, Any] = {}
    for short, long in _RESIDUAL_KWARG_FIELDS:
        if hasattr(cfg, short):
            residual_kwargs[long] = getattr(cfg, short)

    residual = ResidualPolicy(**residual_kwargs)
    if cfg.residual_policy_ckpt_path:
        sd = torch.load(cfg.residual_policy_ckpt_path, map_location="cpu")
        residual.load_state_dict(sd)
    residual = residual.to(DEVICE)

    return ae, residual


@dataclass
class GoalState:
    """Cached cosmos-predicted goal for one residual rollout.

    ``goal_vae_latent`` is kept for visualization (decode through cosmos
    VAE) and for debugging; ``goal_z_rl`` is what the residual actually
    consumes at every residual step.
    """

    goal_vae_latent: torch.Tensor  # (B, 16, 1, H', W')
    goal_z_rl: torch.Tensor  # (B, 768)


def _wrap_image_to_5frame_video(image: torch.Tensor) -> torch.Tensor:
    """``(B, 3, H, W)`` uint8 → ``(B, 3, 5, H, W)`` uint8 (1 blank + 4 replicas).

    Mirrors ``build_libero_vae_cache.build_video_chunk``: cosmos's WanVAE
    uses 1+T temporal compression with a kernel-3 time conv, so a single
    instant must be wrapped as 1 blank frame followed by 4 replicas of
    the actual image to match the structure cosmos sees at inference.
    """
    if image.dim() != 4:
        raise ValueError(
            f"image must be (B, 3, H, W); got shape {tuple(image.shape)}"
        )
    B, C, H, W = image.shape
    blank = torch.zeros(B, C, 1, H, W, dtype=image.dtype, device=image.device)
    replicas = image.unsqueeze(2).expand(B, C, TEMPORAL_COMPRESSION_FACTOR, H, W)
    return torch.cat([blank, replicas], dim=2)


@torch.no_grad()
def encode_image_to_rl_token(
    cosmos_model: nn.Module,
    rl_token_ae: RLTokenAutoencoder,
    image: torch.Tensor,
) -> torch.Tensor:
    """``(B, 3, H, W)`` uint8 image → ``(B, 768)`` RL token.

    Caller is responsible for matching cosmos's runtime preprocessing
    (resize 224 + 90% area center crop if ``trained_with_image_aug``);
    this helper only wraps the image as a 5-frame video, runs cosmos's
    VAE, drops the leading blank-frame latent, and runs the AE encoder.
    """
    video = _wrap_image_to_5frame_video(image)
    vae_latent = cosmos_model.tokenizer.encode(video)  # (B, 16, 2, 28, 28)
    return rl_token_ae.encode(vae_latent[:, :, 1:2])


def _extract_goal_vae_latent(cosmos_output: dict[str, Any]) -> torch.Tensor:
    """Slice the future-image VAE latent out of cosmos's denoised output.

    cosmos's ``get_action`` returns the full denoised latent sequence in
    ``generated_latent`` and an index dict in ``latent_indices``. The
    future-image latent lives at ``future_image_latent_idx`` along the
    temporal axis.
    """
    gen = cosmos_output["generated_latent"]  # (B, 16, T', 28, 28)
    idx = cosmos_output["latent_indices"]["future_image_latent_idx"]  # (B,)
    batch = torch.arange(gen.shape[0], device=gen.device)
    return gen[batch, :, idx, :, :].unsqueeze(2)  # (B, 16, 1, 28, 28)


def get_action_with_goal_state(
    cfg,
    cosmos_model: nn.Module,
    rl_token_ae: RLTokenAutoencoder,
    dataset_stats: dict,
    observation,
    task_description: str,
    **get_action_kwargs,
) -> tuple[torch.Tensor, GoalState]:
    """Run cosmos's ``get_action`` and bundle its goal output for the residual.

    Returns:
        action_chunk: ``(B, T, 7)`` cosmos reference action chunk
            (already unnormalized by ``get_action``).
        goal_state:  ``GoalState`` with the future-image VAE latent and
            its pre-computed RL token (``rl_token_ae.encode(...)``), so
            every later residual step is a single small MLP forward.
    """
    # Local import to keep this module importable without the eval deps
    # (cosmos_utils imports a fair bit of training/eval-only machinery).
    from cosmos_policy.experiments.robot.cosmos_utils import get_action

    output = get_action(
        cfg=cfg,
        model=cosmos_model,
        dataset_stats=dataset_stats,
        observation=observation,
        task_description=task_description,
        **get_action_kwargs,
    )

    action_chunk: torch.Tensor = output["actions"]
    goal_vae_latent = _extract_goal_vae_latent(output)
    with torch.no_grad():
        goal_z_rl = rl_token_ae.encode(goal_vae_latent)

    return action_chunk, GoalState(
        goal_vae_latent=goal_vae_latent,
        goal_z_rl=goal_z_rl,
    )


def get_residual_action(
    cosmos_model: nn.Module,
    rl_token_ae: RLTokenAutoencoder,
    residual_policy: ResidualPolicy,
    obs_image: torch.Tensor,
    goal_state: GoalState,
    cosmos_action_chunk: torch.Tensor,
    sample: bool = True,
) -> torch.Tensor:
    """One-shot residual correction action after a cosmos chunk completes.

    Args:
        cosmos_model: frozen cosmos LIBERO model (used only for its VAE).
        rl_token_ae:  frozen ``RLTokenAutoencoder`` (used only for its
            ``encode`` path; decoder is unused at RL time).
        residual_policy: trainable actor + twin Q.
        obs_image: ``(B, 3, H, W)`` uint8 — observation **after** cosmos's
            chunk has been executed in the env. Caller is responsible for
            matching cosmos's runtime preprocessing.
        goal_state: returned by ``get_action_with_goal_state``. Holds the
            pre-computed ``goal_z_rl`` that the actor consumes.
        cosmos_action_chunk: ``(B, T, 7)`` reference chunk that was just
            executed. Only the gripper bit of the last step is reused
            here; the delta-EE part is replaced by the residual.
        sample: ``True`` → sample from the actor's Gaussian (training /
            stochastic rollout). ``False`` → take the mean (eval).

    Returns:
        ``(B, 7)`` final corrective action: residual delta-EE summed onto
        cosmos's last action's xyz/rpy, with cosmos's last gripper bit.
    """
    z_obs = encode_image_to_rl_token(cosmos_model, rl_token_ae, obs_image)
    mean, log_std = residual_policy.actor(z_obs, goal_state.goal_z_rl)

    if sample:
        eps = torch.randn_like(mean)
        residual = mean + log_std.exp() * eps
    else:
        residual = mean
    residual = residual.squeeze(1)  # (B, 6); chunk_size=1 default

    last_cosmos = cosmos_action_chunk[:, -1]  # (B, 7)
    delta_ee = last_cosmos[:, :6] + residual
    gripper = last_cosmos[:, 6:7]
    return torch.cat([delta_ee, gripper], dim=-1)
