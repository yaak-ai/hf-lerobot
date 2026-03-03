from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
import matplotlib.pyplot as plt
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

logger = logging.getLogger(__name__)



class MuLawScaler:
    def __init__(self, mu=255.0):
        """
        mu dictates the compression level. 255 is standard for 8-bit audio,
        but for steering, you might experiment with lower values like 10 or 50.
        """
        self.mu = torch.tensor(mu)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Applies mu-law scaling to inputs in [-1, 1]."""
        # Ensure mu is on the same device as x
        mu = self.mu.to(x.device)
        return torch.sign(x) * torch.log1p(mu * torch.abs(x)) / torch.log1p(mu)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Inverts the mu-law scaling back to [-1, 1]."""
        mu = self.mu.to(y.device)
        # log1p is ln(1+x), so the inverse uses exp(y * ln(1+mu)) - 1
        return torch.sign(y) * (torch.exp(torch.abs(y) * torch.log1p(mu)) - 1.0) / mu


class PowerScaler:
    def __init__(self, alpha=0.5):
        """
        alpha < 1.0 stretches values near 0 and compresses values near 1/-1.
        Standard choice is 0.5 (square root).
        """
        self.alpha = alpha

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Applies scaling. Use this on your dataset before training."""
        # x is assumed to be normalized to [-1, 1]
        # Added a tiny epsilon for numerical stability if backpropping through it
        return torch.sign(x) * torch.pow(torch.abs(x) + 1e-8, self.alpha)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Applies inverse. Use this on your generated actions at inference."""
        return torch.sign(y) * torch.pow(torch.abs(y) + 1e-8, 1.0 / self.alpha)



def unscale_steering_action(
    actions: torch.Tensor, mu: float = 255.0
) -> torch.Tensor:
    """Inverse MuLaw transform on the steering dimension (index 2) of an action tensor.

    The dataset pipeline applies MuLaw scaling to steering at the SQL level.
    Call this on predicted actions before comparing against raw steering ground truth
    or writing results for downstream consumption (e.g. reye).

    Implemented with ``torch.cat`` instead of index assignment to avoid
    ScatterND operations that are not supported in ONNX export.
    Clamping uses tensor constants for the same reason.

    Args:
        actions: Float tensor of shape ``[..., action_dim]`` with
            ``action_dim >= 3`` (gas, brake, steering, ...).
        mu: MuLaw compression factor — must match the value used in the SQL
            ``list_transform`` expression (default 255.0).

    Returns:
        Action tensor with the same shape as ``actions``, with the steering
        dimension inverse-transformed back to raw ``[-1, 1]`` space.
    """
    mulaw = MuLawScaler(mu=mu)
    neg_one = torch.tensor(-1.0, dtype=actions.dtype, device=actions.device)
    pos_one = torch.tensor(1.0, dtype=actions.dtype, device=actions.device)
    unscaled_steering = mulaw.inverse(
        torch.clamp(actions[..., 2:3], neg_one, pos_one)
    )
    return torch.cat([actions[..., :2], unscaled_steering, actions[..., 3:]], dim=-1)


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
    return (
        NormalizationMode.ZERO_ONE if key == "observation.state.vehicle" else norm_mode
    )


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
        img.permute(0, 1, 4, 2, 3).type(torch.float32) / 255
    )
    # Handling longer contexts (ndim == 3) and single timestamps (ndim == 2)
    batch["observation.state.vehicle"] = (
        a.data["observation.state.vehicle"][:, :, None]
        if a.data["observation.state.vehicle"].ndim == 2  # noqa: PLR2004
        else a.data["observation.state.vehicle"][:, None, None]
    ).to(dtype=torch.float32)
    seq_len = batch["observation.state.vehicle"].shape[1]
    batch["observation.state.waypoints"] = (
        a.data["observation.state.waypoints"][:, None, :]
        .expand(-1, seq_len, -1)
        .to(dtype=torch.float32)
    )
    return batch  # noqa: DOC201


def _plot_distribution(
    ax: plt.Axes,
    values: np.ndarray,
    mu_law_scaled: np.ndarray,
    power_scaled: np.ndarray,
    mu: float,
) -> None:
    ax.hist(values, bins=100, alpha=0.5, label="Raw", density=True)
    ax.hist(mu_law_scaled, bins=100, alpha=0.5, label=f"Mu-Law (mu={mu})", density=True)
    ax.hist(power_scaled, bins=100, alpha=0.5, label="Power (a=0.5)", density=True)
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of steering_angle_normalized")
    ax.legend()
    ax.grid(visible=True, alpha=0.3)


def _plot_mapping_curve(
    ax: plt.Axes,
    mu_law_scaler: MuLawScaler,
    power_scaler: PowerScaler,
    mu: float,
) -> None:
    x = np.linspace(-1.0, 1.0, 500, dtype=np.float32)
    ax.plot(x, x, label="Raw (identity)", linestyle="--")
    ax.plot(
        x, mu_law_scaler.transform(torch.tensor(x)).numpy(), label=f"Mu-Law (mu={mu})"
    )
    ax.plot(x, power_scaler.transform(torch.tensor(x)).numpy(), label="Power (a=0.5)")
    ax.set_xlabel("Input value")
    ax.set_ylabel("Scaled value")
    ax.set_title("Scaling function mapping")
    ax.legend()
    ax.grid(visible=True, alpha=0.3)


def _plot_timeseries(
    ax: plt.Axes,
    values: np.ndarray,
    mu_law_scaled: np.ndarray,
    power_scaled: np.ndarray,
    mu: float,
    n: int = 1000,
) -> None:
    n = min(n, len(values))
    t = np.arange(n)
    ax.plot(t, values[:n], label="Raw", alpha=0.8)
    ax.plot(t, mu_law_scaled[:n], label=f"Mu-Law (mu={mu})", alpha=0.8)
    ax.plot(t, power_scaled[:n], label="Power (a=0.5)", alpha=0.8)
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Steering angle")
    ax.set_title(f"Timeseries (first {n} samples)")
    ax.legend()
    ax.grid(visible=True, alpha=0.3)


def eval_dataset_scaling(
    samples: Dataset,
    output_dir: str = "tmp/scaling_mu_drive",
    timeseries_n: int = 1000,
) -> None:
    """Evaluate MuLaw and Power scaling on the steering angle feature.

    Saves a 3-panel PNG (histogram, mapping curve, timeseries) and a markdown
    report embedding the image.
    """
    column = "meta/VehicleMotion/steering_angle_normalized"
    mu = 255.0

    values: np.ndarray = (
        np.stack(samples[column].to_numpy()).flatten().astype(np.float32)
    )
    mu_law_scaler = MuLawScaler(mu=mu)
    power_scaler = PowerScaler(alpha=0.5)
    mu_law_scaled: np.ndarray = mu_law_scaler.transform(torch.tensor(values)).numpy()
    power_scaled: np.ndarray = power_scaler.transform(torch.tensor(values)).numpy()

    output_path = Path(f"{output_dir}_{int(mu)}")
    output_path.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(21, 5))
    _plot_distribution(axes[0], values, mu_law_scaled, power_scaled, mu)
    _plot_mapping_curve(axes[1], mu_law_scaler, power_scaler, mu)
    _plot_timeseries(axes[2], values, mu_law_scaled, power_scaled, mu, n=timeseries_n)
    fig.tight_layout()

    png_path = output_path / "steering_angle_scaling.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Scaling plot saved to %s", png_path)

    md_path = output_path / "steering_angle_scaling.md"

    def _row(name: str, arr: np.ndarray) -> str:
        return (
            f"| {name} | {arr.min():.4f} | {arr.max():.4f}"
            f" | {arr.mean():.4f} | {arr.std():.4f} |"
        )

    rows = [
        _row("Raw", values),
        _row(f"Mu-Law (mu={mu})", mu_law_scaled),
        _row("Power (a=0.5)", power_scaled),
    ]
    md_path.write_text(
        "# Steering Angle Scaling Evaluation\n\n"
        f"Feature: `{column}`  \n"
        f"Samples: {len(values)}\n\n"
        "| Scaler | min | max | mean | std |\n"
        "|--------|-----|-----|------|-----|\n"
        + "\n".join(rows)
        + "\n\n![Scaling comparison](steering_angle_scaling.png)\n"
    )
    logger.info("Scaling report saved to %s", md_path)


def _conversion_main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)

    logging.info(f"instantiating datamodule {cfg.datamodule._target_}")  # noqa: G004

    # Compute dataset stats
    dataset: Dataset = instantiate(cfg.datamodule.dataset)
    # stat_file = "dataset_stats_rotated_30.json"
    # compute_stats(dataset.samples, stat_file)

    eval_dataset_scaling(dataset.samples)

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
