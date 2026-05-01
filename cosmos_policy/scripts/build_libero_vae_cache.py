# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build a VAE-latent cache from the nvidia/LIBERO-Cosmos-Policy dataset.

For each demo episode, iterates the primary (agentview) image stream, applies
the same preprocessing Cosmos uses at inference (decode JPEG → resize to
224 → 90% center crop) and runs Cosmos's VAE encoder once per sampled
timestep (with the image temporally-replicated 4 times to match cosmos's
1+T temporal compression). The resulting latent for each sample has shape
(16, 1, 28, 28) and is stored in a single HDF5 file for downstream RL-token
autoencoder pretraining.

Usage:
    python -m cosmos_policy.scripts.build_libero_vae_cache \\
        --hf-dataset nvidia/LIBERO-Cosmos-Policy \\
        --download-dir ~/datasets/LIBERO-Cosmos-Policy \\
        --output ~/datasets/libero-cosmos-vae-cache/primary_latents.h5 \\
        --stride 4 \\
        --success-only

Storage layout (output HDF5):
    primary_latents : (N, 16, 1, 28, 28) float16
    episode_index   : (N,) int32   -- index of the source episode
    timestep        : (N,) int32   -- frame index within the episode
"""

from __future__ import annotations

import argparse
import io
import os
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import torch
import torchvision.transforms.functional as TF
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

from cosmos_policy.experiments.robot.cosmos_utils import get_model as cosmos_get_model

COSMOS_IMAGE_SIZE = 224
TEMPORAL_COMPRESSION_FACTOR = 4
LATENT_CHANNELS = 16
LATENT_HW = 28


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--hf-dataset", default="nvidia/LIBERO-Cosmos-Policy")
    p.add_argument("--download-dir", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument(
        "--experiment-name",
        default="cosmos_predict2_2b_480p_libero__inference_only",
        help="Cosmos experiment registered in the ConfigStore; only the "
             "tokenizer is used.",
    )
    p.add_argument(
        "--ckpt-path",
        default="nvidia/Cosmos-Policy-LIBERO-Predict2-2B",
        help="HF repo id or local checkpoint dir for Cosmos.",
    )
    p.add_argument("--stride", type=int, default=4)
    p.add_argument(
        "--success-only",
        action="store_true",
        help="Skip episodes whose filename contains 'success=False'.",
    )
    p.add_argument(
        "--suite",
        default=None,
        help="Optional suite filter, e.g. 'libero_10' or 'libero_goal'.",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Cap number of episodes (debug).",
    )
    return p.parse_args()


def ensure_dataset(hf_dataset: str, download_dir: Path) -> Path:
    """Sanity check that the dataset has been downloaded.

    Auto-download is intentionally not wired up to avoid silent re-downloads
    of a multi-GB dataset; users should pre-download with::

        huggingface-cli download <hf_dataset> --repo-type dataset \\
            --local-dir <download_dir>
    """
    download_dir = download_dir.expanduser()
    if not (download_dir / "all_episodes").is_dir():
        raise FileNotFoundError(
            f"Expected an 'all_episodes/' folder under {download_dir}. "
            f"Pre-download the dataset with `huggingface-cli download "
            f"{hf_dataset} --repo-type dataset --local-dir {download_dir}`."
        )
    return download_dir


def list_episode_files(
    root: Path, success_only: bool, suite: str | None = None
) -> list[Path]:
    """List per-episode HDF5 files from ``all_episodes/`` with optional filters.

    The dataset's ``success_only/`` directory is in a different (per-task,
    LIBERO-native) format, so we always read from ``all_episodes/`` and filter
    by filename — this keeps the per-episode key layout
    (``primary_images_jpeg`` etc.) consistent.
    """
    files = sorted((root / "all_episodes").glob("*.hdf5"))
    if success_only:
        files = [f for f in files if "success=True" in f.name]
    if suite is not None:
        files = [f for f in files if f"suite={suite}" in f.name]
    return files


def decode_jpeg_bytes(buf: bytes) -> np.ndarray:
    """Decode a JPEG byte buffer to (H, W, 3) uint8."""
    return np.array(Image.open(io.BytesIO(buf)).convert("RGB"))


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Match Cosmos inference preprocessing for an already-flipped, JPEG-
    decoded LIBERO frame.

    Stored frames in nvidia/LIBERO-Cosmos-Policy are pre-flipped and JPEG-
    encoded at quality=95, so the only steps left are:
      1. Resize to COSMOS_IMAGE_SIZE.
      2. 90% area center crop + resize back (matches Cosmos's
         apply_image_transforms for trained_with_image_aug=True).
    """
    img = torch.from_numpy(image).permute(2, 0, 1)  # (3, H, W)
    img = TF.resize(
        img.unsqueeze(0).float(),
        [COSMOS_IMAGE_SIZE, COSMOS_IMAGE_SIZE],
        antialias=True,
    ).squeeze(0)
    crop = int(COSMOS_IMAGE_SIZE * (0.9**0.5))
    top = (COSMOS_IMAGE_SIZE - crop) // 2
    img = img[:, top : top + crop, top : top + crop]
    img = TF.resize(
        img.unsqueeze(0),
        [COSMOS_IMAGE_SIZE, COSMOS_IMAGE_SIZE],
        antialias=True,
    ).squeeze(0)
    return img.clamp(0, 255).to(torch.uint8).permute(1, 2, 0).numpy()


def build_video_chunk(image: np.ndarray, device: str) -> torch.Tensor:
    """(H, W, C) uint8 → (1, C, T=5, H, W) uint8 cuda.

    Cosmos's WanVAE uses 1+T temporal compression with a kernel-3 time conv,
    so a single instant is wrapped as 1 blank frame followed by 4 replicas
    of the actual image — matching the structure cosmos uses for every
    ``current_*`` / ``future_*`` slot at inference (see cosmos_utils.py:943).
    The encoder returns 2 latent frames; the second (index 1) is the image's
    latent.
    """
    arr = np.transpose(image, (2, 0, 1))[None, :, None]  # (1, C, 1, H, W)
    arr = np.tile(arr, (1, 1, TEMPORAL_COMPRESSION_FACTOR, 1, 1))  # (1,C,4,H,W)
    blank = np.zeros_like(arr[:, :, :1])  # (1, C, 1, H, W)
    arr = np.concatenate([blank, arr], axis=2)  # (1, C, 5, H, W)
    return torch.from_numpy(arr).to(dtype=torch.uint8, device=device)


def iterate_samples(
    episode_files: Iterable[Path], stride: int
) -> Iterable[tuple[int, int, np.ndarray]]:
    """Yield (episode_idx, timestep, primary_image_uint8) tuples."""
    for ep_idx, ep_file in enumerate(episode_files):
        with h5py.File(ep_file, "r") as f:
            jpeg_bytes = f["primary_images_jpeg"][:]
        for t in range(0, len(jpeg_bytes), stride):
            yield ep_idx, t, decode_jpeg_bytes(bytes(jpeg_bytes[t]))


def count_samples(episode_files: list[Path], stride: int) -> int:
    n = 0
    for ep_file in episode_files:
        with h5py.File(ep_file, "r") as f:
            n += (len(f["primary_images_jpeg"]) + stride - 1) // stride
    return n


def main() -> None:
    args = parse_args()

    # --- 1. Locate dataset on disk (download is left to the user) ---
    download_dir = args.download_dir.expanduser()
    if not download_dir.exists():
        ensure_dataset(args.hf_dataset, download_dir)
    episode_files = list_episode_files(
        download_dir, args.success_only, suite=args.suite
    )
    if args.max_episodes is not None:
        episode_files = episode_files[: args.max_episodes]
    if not episode_files:
        raise RuntimeError(f"No HDF5 episodes found under {download_dir}")
    total = count_samples(episode_files, args.stride)
    print(f"[build-cache] {len(episode_files)} episodes → {total} samples")

    # --- 2. Load Cosmos (only the VAE tokenizer is used) ---
    cosmos_cfg = OmegaConf.create(
        {
            "ckpt_path": args.ckpt_path,
            "config": args.experiment_name,
            "config_file": "cosmos_policy/config/config.py",
        }
    )
    model, _ = cosmos_get_model(cosmos_cfg)
    model = model.to(args.device).eval()
    tokenizer = model.tokenizer

    # --- 3. Allocate output HDF5 ---
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out = h5py.File(args.output, "w")
    latents_ds = out.create_dataset(
        "primary_latents",
        shape=(total, LATENT_CHANNELS, 1, LATENT_HW, LATENT_HW),
        dtype="float16",
        chunks=(1, LATENT_CHANNELS, 1, LATENT_HW, LATENT_HW),
    )
    episode_ds = out.create_dataset("episode_index", shape=(total,), dtype="int32")
    timestep_ds = out.create_dataset("timestep", shape=(total,), dtype="int32")

    # --- 4. Stream encode ---
    write_idx = 0
    pbar = tqdm(total=total, desc="VAE encoding")
    with torch.no_grad():
        for ep_idx, t, raw_image in iterate_samples(episode_files, args.stride):
            image = preprocess_image(raw_image)
            video = build_video_chunk(image, args.device)  # (1, C, 5, H, W) uint8
            latent = tokenizer.encode(video)  # (1, 16, 2, 28, 28)
            # Drop the leading blank-frame latent; keep only the image's slot.
            latent_image = latent[:, :, 1:2]  # (1, 16, 1, 28, 28)
            latents_ds[write_idx] = (
                latent_image.squeeze(0).to(torch.float16).cpu().numpy()
            )
            episode_ds[write_idx] = ep_idx
            timestep_ds[write_idx] = t
            write_idx += 1
            pbar.update(1)
    pbar.close()
    out.close()
    print(f"[build-cache] wrote {write_idx} samples → {args.output}")


if __name__ == "__main__":
    main()
