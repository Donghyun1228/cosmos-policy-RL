# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Train ``RLTokenAutoencoder`` on a cached set of Cosmos VAE latents.

Reads an HDF5 file produced by ``build_libero_vae_cache.py`` and trains the
self-contained AE to reconstruct each latent through the
single-RL-token bottleneck. After training, the encoder is frozen and reused
as the goal/state tokenizer during RL; the decoder is discarded.

Cache layout (input):
    primary_latents : (N, 16, 1, 28, 28) float16
    episode_index   : (N,) int32
    timestep        : (N,) int32

Outputs (under ``--output-dir``):
    best.pt   -- state_dict of the AE with the lowest val loss
    last.pt   -- state_dict of the AE at the end of the final epoch
    log.csv   -- per-epoch (train_loss, val_loss, lr)

Train/val split is per-episode (entire episodes go to val) so that val
measures generalization across scenes, not across timesteps of the same
trajectory.

Usage:
    python -m cosmos_policy.scripts.train_rl_token_ae \\
        --cache ~/datasets/libero-cosmos-vae-cache/primary_latents.h5 \\
        --output-dir ~/checkpoints/rl-token-ae \\
        --epochs 50 --batch-size 256 --lr 1e-4
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from cosmos_policy._src.predict2.residual.rl_token_autoencoder import (
    RLTokenAutoencoder,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument(
        "--val-fraction",
        type=float,
        default=0.05,
        help="Fraction of episodes held out for validation.",
    )
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--precision",
        choices=["fp32", "bf16"],
        default="bf16",
        help="Training compute dtype (autocast).",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Refresh the train-loss line every N steps.",
    )
    p.add_argument(
        "--wandb-project",
        default=None,
        help="If set, log metrics to this Weights & Biases project.",
    )
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-run-name", default=None)
    return p.parse_args()


class _LatentCache(Dataset):
    """HDF5-backed dataset of VAE latents indexed by global sample id.

    The HDF5 file is opened lazily inside ``__getitem__`` (per worker) to
    keep the dataset object pickle-safe for multi-worker DataLoader.
    """

    def __init__(self, h5_path: Path, indices: np.ndarray):
        self._path = str(h5_path)
        self._indices = indices
        self._file: h5py.File | None = None

    def __len__(self) -> int:
        return len(self._indices)

    def _ensure_open(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self._path, "r")
        return self._file

    def __getitem__(self, i: int) -> torch.Tensor:
        f = self._ensure_open()
        arr = f["primary_latents"][int(self._indices[i])]
        return torch.from_numpy(np.asarray(arr))


def compute_latent_stats(
    h5_path: Path, train_idx: np.ndarray, num_channels: int = 16
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel mean / std over the training-split latents.

    Cached next to the HDF5 file as ``<cache>.train_stats.pt`` so that
    re-running training does not re-sweep the cache.
    """
    stats_path = h5_path.with_suffix(h5_path.suffix + ".train_stats.pt")
    if stats_path.exists():
        blob = torch.load(stats_path)
        if int(blob.get("n_train", -1)) == len(train_idx):
            return blob["mean"], blob["std"]

    print(f"[ae-train] computing latent stats over {len(train_idx)} samples")
    s = torch.zeros(num_channels, dtype=torch.float64)
    sq = torch.zeros(num_channels, dtype=torch.float64)
    n = 0
    chunk = 1024
    with h5py.File(h5_path, "r") as f:
        ds = f["primary_latents"]
        sorted_idx = np.sort(train_idx)
        for start in tqdm(
            range(0, len(sorted_idx), chunk),
            desc="latent stats",
            total=(len(sorted_idx) + chunk - 1) // chunk,
        ):
            sel = sorted_idx[start : start + chunk].tolist()
            arr = torch.from_numpy(ds[sel].astype(np.float32)).double()
            s += arr.sum(dim=(0, 2, 3, 4))
            sq += (arr * arr).sum(dim=(0, 2, 3, 4))
            n += arr.numel() // num_channels
    mean = (s / n).float()
    var = (sq / n).float() - mean * mean
    std = var.clamp_min(1e-8).sqrt()
    mean = mean.view(1, num_channels, 1, 1, 1)
    std = std.view(1, num_channels, 1, 1, 1)
    torch.save(
        {"mean": mean, "std": std, "n_train": len(train_idx)}, stats_path
    )
    print(f"[ae-train] cached latent stats → {stats_path}")
    return mean, std


def split_indices_by_episode(
    episode_index: np.ndarray, val_fraction: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Hold out ``val_fraction`` of episodes (not samples) for validation."""
    unique_eps = np.unique(episode_index)
    rng.shuffle(unique_eps)
    n_val = max(1, int(round(len(unique_eps) * val_fraction)))
    val_eps = set(unique_eps[:n_val].tolist())
    is_val = np.array([ep in val_eps for ep in episode_index])
    val_idx = np.nonzero(is_val)[0]
    train_idx = np.nonzero(~is_val)[0]
    return train_idx, val_idx


def lr_lambda(step: int, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def evaluate(
    model: RLTokenAutoencoder,
    loader: DataLoader,
    device: str,
    autocast_dtype: torch.dtype | None,
) -> float:
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=True).float()
            if autocast_dtype is not None:
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    loss = model.reconstruction_loss(batch)
            else:
                loss = model.reconstruction_loss(batch)
            total += loss.item() * batch.size(0)
            n += batch.size(0)
    return total / max(1, n)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    args.output_dir = args.output_dir.expanduser()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache = args.cache.expanduser()

    wandb_run = None
    if args.wandb_project is not None:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            dir=str(args.output_dir),
            config=vars(args),
        )

    # --- 1. Load metadata + build per-episode split ---
    with h5py.File(args.cache, "r") as f:
        episode_index = f["episode_index"][:]
        n_total = f["primary_latents"].shape[0]
    train_idx, val_idx = split_indices_by_episode(
        episode_index, args.val_fraction, rng
    )
    print(
        f"[ae-train] {n_total} samples, "
        f"{len(train_idx)} train / {len(val_idx)} val "
        f"({len(np.unique(episode_index))} episodes)"
    )

    train_loader = DataLoader(
        _LatentCache(args.cache, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        _LatentCache(args.cache, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    # --- 2. Model + optimizer + cosine schedule with warmup ---
    model = RLTokenAutoencoder().to(args.device)
    mean, std = compute_latent_stats(args.cache, train_idx)
    model.set_latent_stats(mean.to(args.device), std.to(args.device))
    print(
        f"[ae-train] latent stats: "
        f"mean range [{mean.min():.3f}, {mean.max():.3f}], "
        f"std range [{std.min():.3f}, {std.max():.3f}]"
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[ae-train] RLTokenAutoencoder: {n_params/1e6:.1f}M params")

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    total_steps = max(1, args.epochs * len(train_loader))
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lambda s: lr_lambda(s, args.warmup_steps, total_steps)
    )
    autocast_dtype = torch.bfloat16 if args.precision == "bf16" else None

    # --- 3. Train loop ---
    log_path = args.output_dir / "log.csv"
    log_f = log_path.open("w", newline="")
    log_writer = csv.writer(log_f)
    log_writer.writerow(["epoch", "train_loss", "val_loss", "lr"])

    best_val = float("inf")
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        running_n = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        for batch in pbar:
            batch = batch.to(args.device, non_blocking=True).float()
            if autocast_dtype is not None:
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    loss = model.reconstruction_loss(batch)
            else:
                loss = model.reconstruction_loss(batch)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()

            running += loss.item() * batch.size(0)
            running_n += batch.size(0)
            global_step += 1
            if global_step % args.log_every == 0:
                pbar.set_postfix(
                    loss=f"{running/running_n:.4e}",
                    lr=f"{sched.get_last_lr()[0]:.2e}",
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train/step_loss": loss.item(),
                            "train/lr": sched.get_last_lr()[0],
                        },
                        step=global_step,
                    )

        train_loss = running / max(1, running_n)
        val_loss = evaluate(model, val_loader, args.device, autocast_dtype)
        lr_now = sched.get_last_lr()[0]
        print(
            f"[ae-train] epoch {epoch}: "
            f"train={train_loss:.4e} val={val_loss:.4e} lr={lr_now:.2e}"
        )
        log_writer.writerow([epoch, train_loss, val_loss, lr_now])
        log_f.flush()
        if wandb_run is not None:
            wandb_run.log(
                {
                    "train/epoch_loss": train_loss,
                    "val/loss": val_loss,
                    "epoch": epoch,
                },
                step=global_step,
            )

        torch.save(model.state_dict(), args.output_dir / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), args.output_dir / "best.pt")
            print(f"[ae-train]   ↳ new best val={val_loss:.4e}")

    log_f.close()
    if wandb_run is not None:
        wandb_run.summary["best_val_loss"] = best_val
        wandb_run.finish()
    print(f"[ae-train] done. best val={best_val:.4e} → {args.output_dir/'best.pt'}")


if __name__ == "__main__":
    main()
