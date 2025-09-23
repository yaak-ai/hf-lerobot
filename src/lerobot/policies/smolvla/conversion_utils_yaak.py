from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
import numpy as np
import torch
from hydra.utils import instantiate

import lerobot.constants_yaak as constants_yaak
from lerobot.configs.types import NormalizationMode
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import init_logging

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from torch import Tensor

    from rbyte import Dataset


def patch_norm_mode(
    norm_mode: NormalizationMode, key: str, batch: dict[str, Tensor]
) -> NormalizationMode:
    # Monkey patch for the case where different STATE features have
    # different normalization modes

    # Note: alternative would be adding a special feature type for intent
    # but feature types are used extensivelly through codebase
    # and would need to be extra careful to not break anything
    if "observation.state.waypoints" not in batch:
        return norm_mode
    return norm_mode


def load_dataset_stats(stats_path: str | Path) -> dict[str, dict[str, np.ndarray]]:
    stats_path = Path(stats_path)
    with stats_path.open() as f:
        stats_json = json.load(f)
    return {
        k: {metric: np.array(vals) for metric, vals in v.items()}
        for k, v in stats_json.items()
    }


def compute_stats(samples: Dataset, stat_file: str = "dataset_stats.json") -> None:
    samples.write_parquet("whole_samples_30_not_rotated.parquet")
    cnt = len(samples)
    columns = [
        "meta/VehicleMotion/gas_pedal_normalized",
        "meta/VehicleMotion/brake_pedal_normalized",
        "meta/VehicleMotion/steering_angle_normalized",
    ]

    action_stats = []
    for column in columns:
        values = np.stack(samples[column].to_numpy())
        stats = [
            values.min(),
            values.max(),
            values.mean(),
            values.std(),
            np.percentile(values, 1),
            np.percentile(values, 99.99),
        ]
        del values
        action_stats.append(stats)
    action_stats = np.stack(action_stats).T

    dataset_stats = {
        "action.continuous": {},
        "observation.state.waypoints": {},
        "observation.state.vehicle": {},
    }
    for key in dataset_stats:  # noqa: PLC0206
        dataset_stats[key]["count"] = [cnt]
    keys = ["min", "max", "mean", "std", "q01", "q99"]

    for i, key in enumerate(keys):
        dataset_stats["action.continuous"][key] = action_stats[i, :].tolist()

    waypoints = np.stack(samples["observation.state.waypoints"].to_numpy()).reshape(
        -1, 2
    )
    wp_cnt = 10
    wps = [
        np.tile(waypoints.min(), wp_cnt * 2),
        np.tile(waypoints.max(), wp_cnt * 2),
        # mean/std in valid and because of different x/y scaling
        np.tile(waypoints.mean(axis=0), wp_cnt),
        np.tile(waypoints.std(axis=0), wp_cnt),
        np.tile(np.percentile(waypoints, 1), wp_cnt * 2),
        np.tile(np.percentile(waypoints, 99), wp_cnt * 2),
    ]
    for i, key in enumerate(keys):
        dataset_stats["observation.state.waypoints"][key] = wps[i].tolist()

    column = "observation.state.vehicle"
    values = np.stack(samples[column])
    state_stats = np.array([
        values.min(),
        values.max(),
        values.mean(),
        values.std(),
        np.percentile(values, 1),
        np.percentile(values, 99),
    ])[:, None]
    for i, key in enumerate(keys):
        dataset_stats[column][key] = state_stats[i, :].tolist()

    with Path(stat_file).open("w") as f:  # noqa: PLW1514
        json.dump(dataset_stats, f, indent=4)
    print(dataset_stats)  # noqa: T201
    print(f"Dataset stats saved to {stat_file}")  # noqa: T201


@torch.no_grad
def merge_waypoints_speed_as_state(
    batch: dict[str, torch.Tensor], key: str
) -> dict[str, torch.Tensor]:
    """Used when waypoint and speed"""
    # compare inputs (lerobot or yaak constants) to yaak constants
    if key != constants_yaak.OBS_STATE:
        # lerobot's OBS_STATE != yaak's OBS_STATE
        return batch[key]  # noqa: DOC201
    return torch.cat((batch[key], batch[constants_yaak.OBS_STATE_VEHICLE]), dim=-1)


def __getbatch__(a: dict) -> dict:  # noqa: N807
    """Used inside every Dataloader loop to conform Lerobot format"""
    batch = {}
    batch["meta/ImageMetadata.cam_front_left/time_stamp"] = a.data[
        "meta/ImageMetadata.cam_front_left/time_stamp"
    ]
    batch["task"] = a.data["task"]
    batch["action.continuous"] = torch.stack(
        (
            a.data["meta/VehicleMotion/gas_pedal_normalized"],
            a.data["meta/VehicleMotion/brake_pedal_normalized"],
            a.data["meta/VehicleMotion/steering_angle_normalized"],
        ),
        dim=-1,
    ).to(dtype=torch.float32)

    # a.data["action.continuous"].to(dtype=torch.float32)  # noqa: ERA001

    # Handling longer contexts (ndim == 5) and single images (ndim == 4)
    img = (
        a.data["cam_front_left"][:, None, :, :, :]
        if a.data["cam_front_left"].ndim == 4  # noqa: PLR2004
        else a.data["cam_front_left"]
    )
    batch["observation.images.front_left"] = (
        torch.zeros_like(img).permute(0, 1, 4, 2, 3).type(torch.float32) / 255
    )
    # Handling longer contexts (ndim == 3) and single timestamps (ndim == 2)
    batch["observation.state.vehicle"] = (
        a.data["observation.state.vehicle"][:, :, None]
        if a.data["observation.state.vehicle"].ndim == 2  # noqa: PLR2004
        else a.data["observation.state.vehicle"][:, None, None]
    ).to(dtype=torch.float32)
    seq_len = batch["observation.state.vehicle"].shape[1]
    batch["observation.state.waypoints"] = (
        a.data["observation.state.waypoints"][:, None, :].expand(-1, seq_len, -1)
        .to(dtype=torch.float32)
    )
    return batch  # noqa: DOC201


def _conversion_main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)

    logging.info(f"instantiating datamodule {cfg.datamodule._target_}")  # noqa: G004

    # Compute dataset stats
    dataset: Dataset = instantiate(cfg.datamodule.dataset)
    stat_file = "dataset_stats_rotated_30.json"
    compute_stats(dataset.samples, stat_file)

    # datamodule: DataLoader = instantiate(cfg.datamodule)  # noqa: ERA001
    # a = next(iter(datamodule))  # noqa: ERA001
    # batch = __getbatch__(a)  # noqa: ERA001
    # print(batch)  # noqa: ERA001
    # return datamodule  # noqa: ERA001


@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    return _conversion_main(cfg)


if __name__ == "__main__":
    init_logging()
    main()
