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

"""This script demonstrates how to train SmolVLA using rbyte.

Once you have trained a model with this script, you can try to evaluate it on
examples/2_evaluate_pretrained_policy.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
from hydra.utils import instantiate

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.conversion_utils_yaak import load_dataset_stats
from lerobot.scripts.train_yaak import train
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import init_logging

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from torch.utils.data import DataLoader


def _make_cfg_smolvla(
    input_features: dict, output_features: dict, normalization_mapping: dict
) -> TrainPipelineConfig:
    cfg = SmolVLAConfig(
        input_features=input_features,
        output_features=output_features,
        normalization_mapping=normalization_mapping,
        load_vlm_weights=True,
        prefix_length=0,
        pad_language_to="max_length",
        num_expert_layers=0,
        scheduler_warmup_steps=16700,
        scheduler_decay_steps=501000,
        device="cuda",
        push_to_hub=False,
    )
    return TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="yaak-ai/L2D"),
        policy=cfg,
        job_name="train_smolvla_rmind_waypoints+speed_chunk_eval",
        batch_size=64,
        steps=334000,
        log_freq=50,
        eval_freq=20000,
    )


def _create_job_name(normalization_mapping: dict, hydra_cfg: DictConfig) -> str:
    """"Create a job name based on the normalization mode and other parameters."""
    # collect normalizations used
    norm_type_action = str(normalization_mapping["ACTION"]).split(".")[-1]
    if hydra_cfg.normalization.quantile_normalization.use_quantiles_instead_of_min_max:
        norm_type_action = "Q00_Q99"
    norm_type_state = str(normalization_mapping["STATE"]).split(".")[-1]

    # collect original name from yaml
    opts = [hydra_cfg.model.train_run]
    # clip or not
    if hydra_cfg.model.use_context:
        opts.append("chunk")
    # frozen vision encoder or not
    if not hydra_cfg.model.freeze_vision_encoder:
        opts.append("vision")

    return f"{'_'.join(opts)}_A_{norm_type_action}_S_{norm_type_state}"  # noqa: DOC201


def make_cfg_smolvla_clip(
    input_features: dict,
    output_features: dict,
    normalization_mapping: dict,
    hydra_cfg: DictConfig,
) -> TrainPipelineConfig:
    # To be done directly in the hydra config: _target_: SmolVLAConfig
    cfg = SmolVLAConfig(
        input_features=input_features,
        output_features=output_features,
        normalization_mapping=normalization_mapping,
        load_vlm_weights=hydra_cfg.model.load_vlm_weights,
        prefix_length=hydra_cfg.model.prefix_length,
        pad_language_to=hydra_cfg.model.pad_language_to,
        num_vlm_layers=hydra_cfg.model.num_vlm_layers,  # different for Eagle & SmolVLM
        # optimizer params
        optimizer_lr=hydra_cfg.model.optimizer_lr,
        optimizer_grad_clip_norm=hydra_cfg.model.optimizer_grad_clip_norm,
        optimizer_betas=tuple(hydra_cfg.model.optimizer_betas),
        optimizer_weight_decay=hydra_cfg.model.optimizer_weight_decay,
        scheduler_warmup_steps=hydra_cfg.model.scheduler_warmup_steps,
        scheduler_decay_steps=hydra_cfg.model.scheduler_decay_steps,
        scheduler_decay_lr=hydra_cfg.model.scheduler_decay_lr,
        # horizon length (only 1st action or more)
        chunk_size=hydra_cfg.model.chunk_size,
        n_action_steps=hydra_cfg.model.n_action_steps,
        device="cuda",
        # yaak tuning options
        use_context=hydra_cfg.model.use_context,
        use_separate_intent=hydra_cfg.model.use_separate_intent,
        max_action_dim=hydra_cfg.model.max_action_dim,
        max_state_dim=hydra_cfg.model.max_state_dim,
        max_intent_dim=hydra_cfg.model.max_intent_dim,
        use_image_norm=hydra_cfg.model.use_image_norm,
        use_masked_loss=hydra_cfg.model.use_masked_loss,
        use_acc_loss=hydra_cfg.model.use_acc_loss,
        use_state_masking=hydra_cfg.model.use_state_masking,
        state_masking_probability=hydra_cfg.model.state_masking_probability,
        push_to_hub=False,
        # vision encoder options
        freeze_vision_encoder=hydra_cfg.model.freeze_vision_encoder,
    )

    return TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="yaak-ai/L2D"),
        policy=cfg,
        job_name=_create_job_name(normalization_mapping, hydra_cfg),
        batch_size=hydra_cfg.model.batch_size,
        steps=hydra_cfg.model.steps,
        log_freq=hydra_cfg.model.log_freq,
        eval_freq=hydra_cfg.model.eval_freq,
    )


def _train_smolvla_rbyte(hydra_cfg: DictConfig) -> None:
    init_logging()

    dataset_stats = load_dataset_stats(hydra_cfg.paths.lerobot_stats)

    # Substitute min-max normalization with quantile normalization if specified
    if hydra_cfg.normalization.quantile_normalization.use_quantiles_instead_of_min_max:
        # monkey patching: replace min-max values with quantile values to keep the
        # original code
        for k, dim in hydra_cfg.normalization.quantile_normalization.quantile_keys_dims:
            if k in dataset_stats:
                dataset_stats[k]["max"][dim] = dataset_stats[k]["q99"][dim]
                dataset_stats[k]["min"][dim] = dataset_stats[k]["q01"][dim]

    input_features = {
        "observation.images.front_left": PolicyFeature(
            type=FeatureType.VISUAL, shape=(3, 324, 576)
        ),
        # Waypoints and speed are used as state features: see src/lerobot/constants_yaak.py  # noqa: E501
        "observation.state.waypoints": PolicyFeature(
            type=FeatureType.STATE,
            shape=dataset_stats["observation.state.waypoints"]["mean"].shape,
        ),
        "observation.state.vehicle": PolicyFeature(
            type=FeatureType.STATE,
            shape=dataset_stats["observation.state.vehicle"]["mean"].shape,
        ),
    }

    output_features = {
        "action.continuous": PolicyFeature(
            type=FeatureType.ACTION,
            shape=dataset_stats["action.continuous"]["mean"].shape,
        )
    }

    normalization_mapping = {
        "VISUAL": instantiate(hydra_cfg.normalization.visual),
        "STATE": instantiate(hydra_cfg.normalization.state),
        "ACTION": instantiate(hydra_cfg.normalization.action),
    }

    train_cfg = make_cfg_smolvla_clip(
        input_features, output_features, normalization_mapping, hydra_cfg
    )
    if hydra_cfg.model.resume:
        logging.info(
            f"Resuming training from a checkpoint {hydra_cfg.model.config_path}"
        )  # noqa: G004
        train_cfg = TrainPipelineConfig.from_pretrained(hydra_cfg.model.config_path)
        train_cfg.resume = hydra_cfg.model.resume
        # Draccus monkey patching
        sys.argv.append(f"--config_path={hydra_cfg.model.config_path}")
    logging.info(f"Train job: {train_cfg.job_name}")  # noqa: G004

    train_cfg.wandb.enable = True

    json_cfg = f"{train_cfg.job_name}.json"
    with Path(json_cfg).open("w") as f:  # noqa: PLW1514
        train_cfg_json = train_cfg.to_dict()
        json.dump(train_cfg_json, f, indent=4)
    logging.info(f"Train config saved to {json_cfg}")  # noqa: G004

    set_seed(train_cfg.seed)
    dataloader: DataLoader = instantiate(hydra_cfg.datamodule)
    logging.info(f"Train dataloader: {len(dataloader)}")  # noqa: G004
    logging.info(f"Dataset pipeline {hydra_cfg.datamodule.dataset.samples.run_folder}")  # noqa: G004
    dataloader_test: DataLoader = instantiate(hydra_cfg.datamodule_val)
    train(train_cfg, dataloader, dataset_stats, dataloader_test)


@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    return _train_smolvla_rbyte(cfg)


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_forkserver_preload(["polars", "duckdb"])
    main()
