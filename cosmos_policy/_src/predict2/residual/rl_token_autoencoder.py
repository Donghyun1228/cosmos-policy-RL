# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange


class _TransformerBlock(nn.Module):
    """Pre-norm transformer block (LayerNorm → MHA → LayerNorm → MLP)."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class _Patchify(nn.Module):
    """VAE latent (B, C, T, H, W) → patch tokens (B, T', H', W', model_dim).

    Independent of cosmos's DiT PatchEmbed; trained jointly with the AE.
    """

    def __init__(
        self,
        in_channels: int,
        spatial_patch_size: int,
        temporal_patch_size: int,
        model_dim: int,
    ):
        super().__init__()
        patch_dim = (
            in_channels
            * spatial_patch_size
            * spatial_patch_size
            * temporal_patch_size
        )
        self.proj = nn.Sequential(
            Rearrange(
                "b c (t r) (h m) (w n) -> b t h w (c r m n)",
                r=temporal_patch_size,
                m=spatial_patch_size,
                n=spatial_patch_size,
            ),
            nn.Linear(patch_dim, model_dim),
        )

    def forward(self, vae_latent: torch.Tensor) -> torch.Tensor:
        return self.proj(vae_latent)


class _Unpatchify(nn.Module):
    """patch tokens (B, T', H', W', model_dim) → VAE latent (B, C, T, H, W).

    Mirror of ``_Patchify``; independent weights (not transposed)."""

    def __init__(
        self,
        in_channels: int,
        spatial_patch_size: int,
        temporal_patch_size: int,
        model_dim: int,
    ):
        super().__init__()
        patch_dim = (
            in_channels
            * spatial_patch_size
            * spatial_patch_size
            * temporal_patch_size
        )
        self.proj = nn.Sequential(
            nn.Linear(model_dim, patch_dim),
            Rearrange(
                "b t h w (c r m n) -> b c (t r) (h m) (w n)",
                r=temporal_patch_size,
                m=spatial_patch_size,
                n=spatial_patch_size,
                c=in_channels,
            ),
        )

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        return self.proj(patch_tokens)


class RLTokenEncoder(nn.Module):
    """Compress patch token sequence into a single RL token.

    Adds 3D positional embeddings, appends a learnable readout token at the
    tail, and runs Transformer self-attention. Output at the readout
    position is z_rl.
    """

    def __init__(
        self,
        model_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 6,
        mlp_ratio: float = 4.0,
        max_temporal_patches: int = 2,
        max_spatial_patches: int = 16,
    ):
        super().__init__()
        self.model_dim = model_dim

        self.pos_embed = nn.Parameter(
            torch.zeros(
                1,
                max_temporal_patches,
                max_spatial_patches,
                max_spatial_patches,
                model_dim,
            )
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.rl_token_embed = nn.Parameter(torch.zeros(1, 1, model_dim))
        nn.init.trunc_normal_(self.rl_token_embed, std=0.02)

        self.blocks = nn.ModuleList(
            [
                _TransformerBlock(model_dim, num_heads, mlp_ratio)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        B, Tp, Hp, Wp, D = patch_tokens.shape
        if D != self.model_dim:
            raise ValueError(
                f"patch_tokens has dim {D} but encoder model_dim is "
                f"{self.model_dim}."
            )
        if (
            Tp > self.pos_embed.shape[1]
            or Hp > self.pos_embed.shape[2]
            or Wp > self.pos_embed.shape[3]
        ):
            raise ValueError(
                f"Patch grid ({Tp}, {Hp}, {Wp}) exceeds positional embedding "
                f"capacity {tuple(self.pos_embed.shape[1:4])}."
            )

        x = patch_tokens + self.pos_embed[:, :Tp, :Hp, :Wp, :]
        x = rearrange(x, "b t h w d -> b (t h w) d")

        rl = self.rl_token_embed.expand(B, -1, -1)
        x = torch.cat([x, rl], dim=1)

        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        return x[:, -1, :]


class RLTokenDecoder(nn.Module):
    """Parallel reconstruction decoder via concat + self-attention.

    Concatenates ``z_rl`` with learnable position queries and runs Transformer
    self-attention. The position-query slice of the output is reshaped back
    to the patch grid.
    """

    def __init__(
        self,
        model_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 4,
        mlp_ratio: float = 4.0,
        max_temporal_patches: int = 2,
        max_spatial_patches: int = 16,
    ):
        super().__init__()
        self.model_dim = model_dim

        self.position_queries = nn.Parameter(
            torch.zeros(
                1,
                max_temporal_patches,
                max_spatial_patches,
                max_spatial_patches,
                model_dim,
            )
        )
        nn.init.trunc_normal_(self.position_queries, std=0.02)

        self.blocks = nn.ModuleList(
            [
                _TransformerBlock(model_dim, num_heads, mlp_ratio)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(model_dim)
        self.output_proj = nn.Linear(model_dim, model_dim)

    def forward(
        self,
        rl_token: torch.Tensor,
        target_grid: tuple[int, int, int],
    ) -> torch.Tensor:
        Tp, Hp, Wp = target_grid
        if (
            Tp > self.position_queries.shape[1]
            or Hp > self.position_queries.shape[2]
            or Wp > self.position_queries.shape[3]
        ):
            raise ValueError(
                f"Target grid ({Tp}, {Hp}, {Wp}) exceeds positional query "
                f"capacity {tuple(self.position_queries.shape[1:4])}."
            )
        B = rl_token.shape[0]

        queries = self.position_queries[:, :Tp, :Hp, :Wp, :]
        queries = rearrange(queries, "1 t h w d -> 1 (t h w) d").expand(B, -1, -1)

        x = torch.cat([rl_token.unsqueeze(1), queries], dim=1)

        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        recon = x[:, 1:, :]
        recon = self.output_proj(recon)
        return rearrange(recon, "b (t h w) d -> b t h w d", t=Tp, h=Hp, w=Wp)


class RLTokenAutoencoder(nn.Module):
    """Compress one Cosmos VAE latent frame into a single RL token.

    Self-contained AE: takes a VAE latent ``(B, C, T, H, W)`` directly,
    patchifies with its own learnable patchifier, compresses to a single
    z_rl via the encoder, reconstructs back to a VAE latent via the decoder
    + own depatchifier. Reconstruction loss is computed in VAE latent space.

    Inspired by Xu et al., "RL Token: Bootstrapping Online RL with Vision-
    Language-Action Models" (Physical Intelligence, 2025), but operates on
    Cosmos VAE latents (not Cosmos DiT activations) and uses its own
    patchify/unpatchify modules instead of borrowing Cosmos's PatchEmbed.
    """

    def __init__(
        self,
        in_channels: int = 16,
        spatial_patch_size: int = 2,
        temporal_patch_size: int = 1,
        model_dim: int = 768,
        encoder_num_heads: int = 12,
        encoder_num_layers: int = 6,
        decoder_num_heads: int = 12,
        decoder_num_layers: int = 4,
        mlp_ratio: float = 4.0,
        max_temporal_patches: int = 2,
        max_spatial_patches: int = 16,
    ):
        super().__init__()
        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size

        self.patchify = _Patchify(
            in_channels=in_channels,
            spatial_patch_size=spatial_patch_size,
            temporal_patch_size=temporal_patch_size,
            model_dim=model_dim,
        )
        self.encoder = RLTokenEncoder(
            model_dim=model_dim,
            num_heads=encoder_num_heads,
            num_layers=encoder_num_layers,
            mlp_ratio=mlp_ratio,
            max_temporal_patches=max_temporal_patches,
            max_spatial_patches=max_spatial_patches,
        )
        self.decoder = RLTokenDecoder(
            model_dim=model_dim,
            num_heads=decoder_num_heads,
            num_layers=decoder_num_layers,
            mlp_ratio=mlp_ratio,
            max_temporal_patches=max_temporal_patches,
            max_spatial_patches=max_spatial_patches,
        )
        self.unpatchify = _Unpatchify(
            in_channels=in_channels,
            spatial_patch_size=spatial_patch_size,
            temporal_patch_size=temporal_patch_size,
            model_dim=model_dim,
        )

    def _grid_shape(self, vae_latent: torch.Tensor) -> tuple[int, int, int]:
        _, _, T, H, W = vae_latent.shape
        return (
            T // self.temporal_patch_size,
            H // self.spatial_patch_size,
            W // self.spatial_patch_size,
        )

    def encode(self, vae_latent: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.patchify(vae_latent))

    def decode(
        self,
        rl_token: torch.Tensor,
        target_grid: tuple[int, int, int],
    ) -> torch.Tensor:
        tokens = self.decoder(rl_token, target_grid)
        return self.unpatchify(tokens)

    def reconstruction_loss(self, vae_latent: torch.Tensor) -> torch.Tensor:
        """L_ro = mean(|| AE(vae_latent) - sg(vae_latent) ||^2) in VAE latent space."""
        z_rl = self.encode(vae_latent)
        recon = self.decode(z_rl, target_grid=self._grid_shape(vae_latent))
        target = vae_latent.detach()
        return ((recon - target) ** 2).mean()

    def forward(self, vae_latent: torch.Tensor) -> torch.Tensor:
        return self.encode(vae_latent)
