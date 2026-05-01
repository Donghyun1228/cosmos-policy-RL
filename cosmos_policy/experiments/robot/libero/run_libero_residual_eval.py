# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minimal LIBERO eval with the cosmos + residual pipeline.

Mirrors the inner loop of ``run_libero_eval`` but strips away features
we don't need for a sanity-check rollout: no parallel inference, no
best-of-N search, no planning model, no data collection / video saving.
The only addition over a plain cosmos eval is a single residual
correction step after each cosmos chunk has been executed in the env.

For each task in the chosen suite, runs ``--trials-per-task`` rollouts
and prints aggregate success counts. With a random-init residual and
``--residual-mode=passthrough``, this should reproduce the cosmos
baseline (residual=0 every step). With ``--residual-mode=sampled`` it
adds Gaussian noise on top of cosmos's gripper-stripped action.

Run:
    python -m cosmos_policy.experiments.robot.libero.run_libero_residual_eval \\
        --ae-ckpt ~/donghyun/checkpoints/rl-token-ae/best.pt \\
        --task-suite libero_10 \\
        --num-tasks 2 \\
        --trials-per-task 3 \\
        --residual-mode passthrough
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from libero.libero import benchmark

from cosmos_policy.datasets.dataset_utils import apply_jpeg_compression_np
from cosmos_policy.experiments.robot.cosmos_utils import (
    DEVICE,
    get_model,
    init_t5_text_embeddings_cache,
    load_dataset_stats,
)
from cosmos_policy.experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
)
from cosmos_policy.experiments.robot.libero.run_libero_eval import (
    prepare_observation,
)
from cosmos_policy.experiments.robot.residual_utils import (
    get_action_with_goal_state,
    get_residual_action,
    get_residual_components,
)


_TASK_SUITE_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


@dataclass
class _Cfg:
    """Flat cfg matching the fields cosmos_utils.get_action / get_model
    actually read; mirrors ``PolicyEvalConfig`` minus the parallel /
    best-of-N / data-collection knobs we don't use here."""

    config: str = "cosmos_predict2_2b_480p_libero__inference_only"
    ckpt_path: str = "nvidia/Cosmos-Policy-LIBERO-Predict2-2B"
    config_file: str = "cosmos_policy/config/config.py"
    suite: str = "libero"
    chunk_size: int = 16
    num_open_loop_steps: int = 16
    num_denoising_steps_action: int = 5
    use_third_person_image: bool = True
    num_third_person_images: int = 1
    use_wrist_image: bool = True
    num_wrist_images: int = 1
    use_proprio: bool = True
    flip_images: bool = True
    use_jpeg_compression: bool = True
    use_variance_scale: bool = False
    trained_with_image_aug: bool = True
    unnormalize_actions: bool = True
    normalize_proprio: bool = True
    model_family: str = "cosmos"
    dataset_stats_path: str = ""
    t5_text_embeddings_path: str = ""
    rl_token_ae_ckpt_path: str = ""
    residual_policy_ckpt_path: str = ""
    residual_chunk_size: int = 1
    residual_action_dim: int = 6
    residual_hidden_dim: int = 256
    residual_num_hidden_layers: int = 2
    residual_log_std_init: float = -1.0
    residual_learnable_log_std: bool = True
    residual_num_critics: int = 2


def _find_hf_snapshot(repo_id: str) -> Path:
    cache = Path.home() / ".cache/huggingface/hub"
    repo_dir = cache / f"models--{repo_id.replace('/', '--')}"
    snapshots = list((repo_dir / "snapshots").glob("*"))
    if not snapshots:
        raise FileNotFoundError(
            f"No HF snapshot for {repo_id}; run `huggingface-cli download` first."
        )
    return snapshots[0]


def _preprocess_image_for_residual(
    image_np: np.ndarray, image_size: int = 224
) -> torch.Tensor:
    """Match cosmos's runtime image pipeline (which is also what the
    cache + AE were trained on): JPEG q=95 round-trip -> resize 224 ->
    90% area center crop -> resize 224. Returns ``(1, 3, H, W)`` uint8
    on DEVICE.

    The JPEG round-trip is critical: cosmos's WanVAE was trained on
    JPEG-decoded images (the dataset stored JPEG bytes), and the AE
    downstream of it inherits that distribution. Skipping JPEG here
    sends the VAE off-distribution images and cascades into wrong
    latents at the AE.

    ``np.ascontiguousarray`` because ``np.flipud`` (used inside
    ``prepare_observation``) returns a negative-stride view and torch
    refuses to convert that.
    """
    image_np = apply_jpeg_compression_np(image_np, quality=95)
    img = torch.from_numpy(np.ascontiguousarray(image_np))
    img = img.permute(2, 0, 1).unsqueeze(0).float()
    img = TF.resize(img, [image_size, image_size], antialias=True)
    crop = int(image_size * (0.9 ** 0.5))
    top = (image_size - crop) // 2
    img = img[:, :, top : top + crop, top : top + crop]
    img = TF.resize(img, [image_size, image_size], antialias=True)
    return img.clamp(0, 255).to(torch.uint8).to(DEVICE)


def _run_episode(
    cfg: _Cfg,
    env,
    task_description: str,
    cosmos_model,
    rl_token_ae,
    residual_policy,
    dataset_stats,
    max_steps: int,
    residual_mode: str,
) -> bool:
    """One LIBERO episode. Returns True on success (env.step ``done``)."""
    NUM_STEPS_WAIT = 10
    obs = env.reset()

    # Wait for objects to stabilize.
    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))

    sample_residual = residual_mode == "sampled"
    step = 0
    while step < max_steps:
        # 1. Cosmos prediction + goal packaging.
        observation = prepare_observation(
            obs, resize_size=256, flip_images=cfg.flip_images
        )
        action_chunk, goal = get_action_with_goal_state(
            cfg=cfg,
            cosmos_model=cosmos_model,
            rl_token_ae=rl_token_ae,
            dataset_stats=dataset_stats,
            observation=observation,
            task_description=task_description,
            seed=0,
            randomize_seed=False,
            num_denoising_steps_action=cfg.num_denoising_steps_action,
        )

        # 2. Execute cosmos's chunk open-loop in env.
        chunk_np = action_chunk[0].cpu().numpy()  # (T, 7)
        for t in range(min(cfg.num_open_loop_steps, chunk_np.shape[0])):
            obs, _, done, _ = env.step(chunk_np[t].tolist())
            step += 1
            if done:
                return True
            if step >= max_steps:
                return False

        if residual_mode == "none":
            continue  # skip residual entirely (pure cosmos baseline)

        # 3. One-shot residual correction after the chunk completes.
        observation = prepare_observation(
            obs, resize_size=256, flip_images=cfg.flip_images
        )
        img_t = _preprocess_image_for_residual(observation["primary_image"])
        with torch.no_grad():
            final = get_residual_action(
                cosmos_model=cosmos_model,
                rl_token_ae=rl_token_ae,
                residual_policy=residual_policy,
                obs_image=img_t,
                goal_state=goal,
                cosmos_action_chunk=action_chunk,
                sample=sample_residual,
            )
        action_np = final[0].cpu().numpy().tolist()
        obs, _, done, _ = env.step(action_np)
        step += 1
        if done:
            return True

    return False


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ae-ckpt", required=True, type=Path)
    p.add_argument(
        "--task-suite",
        default="libero_10",
        choices=list(_TASK_SUITE_MAX_STEPS.keys()),
    )
    p.add_argument("--num-tasks", type=int, default=2)
    p.add_argument("--trials-per-task", type=int, default=3)
    p.add_argument(
        "--residual-mode",
        default="passthrough",
        choices=["none", "passthrough", "sampled"],
        help="none: skip residual step entirely (pure cosmos). "
             "passthrough: residual=0 (zero-init mean, sample=False). "
             "sampled: stochastic residual (ill-defined for random "
             "init -- mostly noise on top of cosmos).",
    )
    args = p.parse_args()

    snap = _find_hf_snapshot("nvidia/Cosmos-Policy-LIBERO-Predict2-2B")
    cfg = _Cfg(
        dataset_stats_path=str(snap / "libero_dataset_statistics.json"),
        t5_text_embeddings_path=str(snap / "libero_t5_embeddings.pkl"),
        rl_token_ae_ckpt_path=str(args.ae_ckpt.expanduser()),
    )

    print(f"[eval] device: {DEVICE}, residual mode: {args.residual_mode}")
    print(f"[eval] HF snapshot: {snap}")

    # LIBERO env uses EGL for headless rendering.
    os.environ.setdefault("MUJOCO_GL", "egl")

    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)
    cosmos_model, _ = get_model(cfg)
    rl_token_ae, residual_policy = get_residual_components(cfg)
    print("[eval] all models loaded.")

    bench = benchmark.get_benchmark_dict()[args.task_suite]()
    n_tasks = min(args.num_tasks, bench.n_tasks)
    max_steps = _TASK_SUITE_MAX_STEPS[args.task_suite]

    successes = 0
    total = 0
    for task_id in range(n_tasks):
        task = bench.get_task(task_id)
        env, task_description = get_libero_env(
            task, cfg.model_family, resolution=256
        )
        print(
            f"\n[eval] task {task_id+1}/{n_tasks}: {task_description!r} "
            f"(max_steps={max_steps})"
        )
        for trial in range(args.trials_per_task):
            t0 = time.time()
            try:
                success = _run_episode(
                    cfg=cfg,
                    env=env,
                    task_description=task_description,
                    cosmos_model=cosmos_model,
                    rl_token_ae=rl_token_ae,
                    residual_policy=residual_policy,
                    dataset_stats=dataset_stats,
                    max_steps=max_steps,
                    residual_mode=args.residual_mode,
                )
            except Exception as e:
                print(f"  trial {trial}: ERROR {e}")
                success = False
            total += 1
            successes += int(success)
            print(
                f"  trial {trial}: success={success} "
                f"({time.time() - t0:.1f}s; running {successes}/{total})"
            )

        env.close()

    print(
        f"\n[eval] FINAL: {successes}/{total} "
        f"({100 * successes / max(1, total):.1f}%) "
        f"on {args.task_suite}, "
        f"{n_tasks} tasks x {args.trials_per_task} trials, "
        f"residual mode: {args.residual_mode}"
    )


if __name__ == "__main__":
    main()
