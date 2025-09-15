#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Evaluate a policy on an environment by running rollouts and computing metrics.

Usage examples:

You want to evaluate a model from the hub (eg: https://huggingface.co/lerobot/diffusion_pusht)
for 10 episodes.

```
lerobot-eval \
    --policy.path=lerobot/diffusion_pusht \
    --env.type=pusht \
    --eval.batch_size=10 \
    --eval.n_episodes=10 \
    --use_amp=false \
    --device=cuda
```

OR, you want to evaluate a model checkpoint from the LeRobot training script for 10 episodes.
```
lerobot-eval \
    --policy.path=outputs/train/diffusion_pusht/checkpoints/005000/pretrained_model \
    --env.type=pusht \
    --eval.batch_size=10 \
    --eval.n_episodes=10 \
    --use_amp=false \
    --device=cuda
```

Note that in both examples, the repo/folder should contain at least `config.json` and `model.safetensors` files.

You can learn about the CLI options for this script in the `EvalPipelineConfig` in lerobot/configs/eval.py
"""

import datetime
import gc
import json
import logging
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from pprint import pformat

import torch
from termcolor import colored

from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.factory_yaak import make_policy_yaak
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.smolvla.conversion_utils_yaak import (
    __getbatch__,
    load_dataset_stats,
)
from lerobot.policies.smolvla.modeling_smolvla import pad_vector
from lerobot.policies.utils import get_device_from_parameters
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.reye_utils import create_reye_df
from lerobot.utils.utils import (
    init_logging,
)
from lerobot.utils.wandb_utils_yaak import metric_accum_callback


def load_from_wandb_artifact(artifact: str, stats: Path) -> None:
    """rbyte callback specified in yaml"""
    import wandb  # noqa: PLC0415

    dataset_stats = load_dataset_stats(stats)
    run = wandb.run
    artifact_obj = (
        run.use_artifact(artifact)
        if run is not None and not run.disabled
        else wandb.Api().artifact(artifact, type="model")
    )

    artifact_dir = artifact_obj.download()
    # Need to serialize config to be able to instantiate it
    with (Path(artifact_dir) / "train_config.json").open("w") as f:
        json.dump(artifact_obj.logged_by().config, f, indent=4)
    train_cfg = TrainPipelineConfig.from_pretrained(artifact_dir)

    train_cfg.policy.pretrained_path = artifact_dir
    logging.info(pformat(asdict(train_cfg)))

    logging.info("Making policy.")
    policy = make_policy_yaak(
        cfg=train_cfg.policy,
        stats=dataset_stats,
    )
    return policy, train_cfg


def eval_policy_yaak_loop(
    policy: PreTrainedPolicy,
    eval_dataloader: torch.utils.data.DataLoader,
    eval_tracker: MetricsTracker,
    start_seed: int,
) -> MetricsTracker:
    device = get_device_from_parameters(policy)
    policy.eval()
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(start_seed)

    action_dim = policy.config.action_feature.shape[0]
    chunk_size = policy.config.chunk_size
    loss_accumulator = torch.zeros(
        (len(eval_dataloader.dataset), action_dim), dtype=torch.float32, device=device
    )
    pred_actions = torch.zeros(
        (len(eval_dataloader.dataset), chunk_size, action_dim),
        dtype=torch.float32,
        device=device,
    )
    bsize = eval_dataloader.batch_size
    for step, elem in enumerate(eval_dataloader):
        policy.reset()  # Clear queues and reset the policy state
        start_time = time.perf_counter()
        batch = __getbatch__(elem)
        eval_tracker.eval_dataloading_s = time.perf_counter() - start_time
        cur_bsize = elem.shape[0]  # different from bsize if the last batch is smaller
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=True)
        with torch.inference_mode():
            actions_shape = (cur_bsize, policy.config.chunk_size, policy.config.max_action_dim)
            noise = policy.model.sample_noise(actions_shape, device)
            pred_actions[step * bsize : step * bsize + cur_bsize, ...] = (
                policy.predict_action_chunk(batch)
                if step == 0
                else policy.predict_action_chunk(
                    batch,
                    noise=None,
                    x_t_prev=pad_vector(pred_actions[
                        (step - 1) * bsize : (step - 1) * bsize + cur_bsize, ...
                    ], policy.config.max_action_dim),
                )
            )
        eval_tracker.eval_update_s = time.perf_counter() - start_time
        eval_tracker.step()
        del batch
        if step % 50 == 0 and torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()  # Use sparingly
    stability_dict_wandb = {}
    log_images_wandb = []
    pred_actions = pred_actions.cpu()
    metric_accum_callback(
        loss_accumulator.cpu(),
        eval_dataloader.dataset.samples,
        stability_dict_wandb,
        log_images_wandb,
        pred_actions[:, 0, :],
    )
    return eval_tracker, stability_dict_wandb, log_images_wandb, pred_actions


def eval_policy_yaak(
    policy: PreTrainedPolicy,
    eval_dataloader: torch.utils.data.DataLoader,
    eval_tracker: MetricsTracker,
    start_seed: int,
) -> MetricsTracker:
    eval_tracker, stability_dict_wandb, log_images_wandb, _ = eval_policy_yaak_loop(
        policy, eval_dataloader, eval_tracker, start_seed
    )
    return eval_tracker, stability_dict_wandb, log_images_wandb


def predict_policy_yaak(
    policy: PreTrainedPolicy,
    eval_dataloader: torch.utils.data.DataLoader,
    eval_tracker: MetricsTracker,
    start_seed: int,
    reye_output_dir: Path,
) -> MetricsTracker:
    eval_tracker, stability_dict_wandb, _, pred_actions = eval_policy_yaak_loop(
        policy, eval_dataloader, eval_tracker, start_seed
    )
    for key, value in stability_dict_wandb.items():
        logging.info(f"{key}: {value:.4f}")  # noqa: G004
    ts = eval_dataloader.dataset.samples[
        "meta/ImageMetadata.cam_front_left/time_stamp"
    ][0]

    # reye serialization
    # Handle cases with clip and without clip based on the timestamp
    df = create_reye_df(
        eval_dataloader, pred_actions, is_without_clip=isinstance(ts, datetime.datetime)
    )
    reye_path = reye_output_dir / "results17.parquet"
    df.write_parquet(reye_path)
    logging.info(f"Predictions saved to {reye_path}")  # noqa: G004
    return eval_tracker


def predict_main_yaak(
    policy: PreTrainedPolicy,
    eval_dataloader: torch.utils.data.DataLoader,
    train_cfg: TrainPipelineConfig,
    output_dir: Path,
) -> None:
    logging.info(
        colored("Output dir:", "yellow", attrs=["bold"]) + f" {train_cfg.output_dir}"  # noqa: G003
    )

    # Metrics
    eval_metrics = {
        "eval_loss": AverageMeter("eval_loss", ":.3f"),
        "eval_update_s": AverageMeter("eval_updt_s", ":.3f"),
        "eval_dataloading_s": AverageMeter("eval_data_s", ":.3f"),
    }
    num_frames = len(eval_dataloader.dataset.samples)
    num_episodes = num_frames
    eval_tracker = MetricsTracker(
        eval_dataloader.batch_size,
        num_frames,
        num_episodes,
        eval_metrics,
        initial_step=0,
    )
    device = get_device_from_parameters(policy)
    with (
        torch.no_grad(),
        torch.autocast(device_type=device.type)
        if train_cfg.policy.use_amp
        else nullcontext(),
    ):
        predict_policy_yaak(
            policy,
            eval_dataloader,
            eval_tracker,
            start_seed=train_cfg.seed,
            reye_output_dir=output_dir,
        )
    logging.info("End of eval")


if __name__ == "__main__":
    init_logging()
