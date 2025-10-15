from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
from wandb import Image

if TYPE_CHECKING:
    from torch import Tensor


def _gt_factory(samples: pl.DataFrame) -> tuple[list[str], np.ndarray]:
    # Extract GT from DataFrame
    action_names = [
        "meta/VehicleMotion/gas_pedal_normalized",
        "meta/VehicleMotion/brake_pedal_normalized",
        "meta/VehicleMotion/steering_angle_normalized",
    ]
    gt_actions = np.stack([np.stack(samples[name]) for name in action_names], axis=2)[
        :, 0, :
    ]
    return action_names, gt_actions


def _get_nnz_actions(gt_actions):
    """Check if the actions Bx3 are non-zero to zoom in on the non-zero values."""
    # Thresholds are determied ad hoc
    gas_cond = gt_actions[:, :1] > 0.04  # (1.0 / 255 + 0.001)
    brake_cond = gt_actions[:, 1:2] > (1.0 / 164 + 0.001)
    steer_cond = (
        gt_actions[:, -1:].abs()
        if isinstance(gt_actions, torch.Tensor)
        else np.abs(gt_actions[:, -1:])
    ) > 0.05
    return gas_cond, brake_cond, steer_cond


def tracking_callback(
    loss_dict: dict[str, torch.Tensor],
    losses: torch.Tensor,
    actions: torch.Tensor,
    mode: str = "train",
) -> None:
    """Callback to log losses during training."""
    if mode not in {"train", "eval"}:
        raise ValueError(mode)  # noqa: DOC501
    with torch.no_grad():
        # monitor the first action in the batch as it will be used
        loss_first_timestamp = losses[:, 0, :]

        gas_cond, brake_cond, steer_cond = _get_nnz_actions(actions[:, 0, :])

        if mode == "train":
            # Compute std dev of loss for each action dimension
            gas_loss_nnz_std = (
                loss_first_timestamp[gas_cond[:, 0], 0].std().item()
                if gas_cond.sum() > 1
                else -0.1
            )
            brake_loss_nnz_std = (
                loss_first_timestamp[brake_cond[:, 0], 1].std().item()
                if brake_cond.sum() > 1
                else -0.1
            )
            steer_loss_nnz_std = (
                loss_first_timestamp[steer_cond[:, 0], -1].std().item()
                if steer_cond.sum() > 1
                else -0.1
            )

            # use the following for direct logging to wandb
            loss_dict["loss_std"] = loss_first_timestamp.std().item()  # scalar

            # Per action loss std for nnz values (where the "action happes")
            loss_dict["gas_loss_nnz_std"] = gas_loss_nnz_std  # scalar
            loss_dict["brake_loss_nnz_std"] = brake_loss_nnz_std  # scalar
            loss_dict["steer_loss_nnz_std"] = steer_loss_nnz_std  # scalar

            # Tracking loss jumps wrt action jumps
            loss_dict["gas_nnz_std"] = (
                actions[gas_cond[:, 0], 0, 0].std().item()
                if gas_cond.sum() > 1
                else -0.1
            )  # scalar
            loss_dict["brake_nnz_std"] = (
                actions[brake_cond[:, 0], 0, 1].std().item()
                if brake_cond.sum() > 1
                else -0.1
            )  # scalar
            loss_dict["steer_nnz_std"] = (
                actions[steer_cond[:, 0], 0, -1].std().item()
                if steer_cond.sum() > 1
                else -0.1
            )  # scalar
        else:
            # use the dict for accumulation during eval and inference
            # this not logged directly to wandb
            loss_dict["loss_first_timestamp"] = loss_first_timestamp  # (batch_size, 3)


def metric_accum_callback(
    loss_accumulator: torch.Tensor,
    samples: pl.DataFrame,
    wandb_dict: dict,
    log_images: list[Image],
    pred_actions: torch.Tensor | None = None,
) -> None:
    """At the end of eval step: collect the loss metrics and plot the losses."""
    # Done at the end (not for each batch) to plot actions over entire drives
    action_dim = loss_accumulator.shape[-1]
    loss_accumulator = loss_accumulator.cpu()

    # Extract GT from DataFrame
    action_names, gt_actions = _gt_factory(samples)
    nnz_accumulator = np.concatenate(_get_nnz_actions(gt_actions), axis=-1)

    # Collect scalar metrics for each action dimension
    action_names = [a.split("/")[-1].split("_")[0] for a in action_names]
    for i in range(action_dim):
        if nnz_accumulator[:, i].sum() > 1:
            wandb_dict[f"{action_names[i]}_loss_nnz_std"] = (
                loss_accumulator[nnz_accumulator[:, i], i].std().item()
            )
            wandb_dict[f"{action_names[i]}_loss_nnz_mean"] = (
                loss_accumulator[nnz_accumulator[:, i], i].mean().item()
            )
            wandb_dict[f"{action_names[i]}_loss_nnz_cnt"] = (
                nnz_accumulator[:, i].sum().item()
            )
        else:
            wandb_dict[f"{action_names[i]}_loss_nnz_std"] = -0.1
            if gt_actions is not None:
                wandb_dict[f"{action_names[i]}_nnz_std"] = -0.1
            wandb_dict[f"{action_names[i]}_loss_nnz_mean"] = -0.1
            wandb_dict[f"{action_names[i]}_loss_nnz_cnt"] = 0

    drives = samples["__input_id"]
    unique_drives = drives.unique().to_list()
    for drive in unique_drives:
        drive_selector = drives == drive
        images = _plot_losses_callback(
            loss_accumulator[drive_selector, :],
            nnz_accumulator[drive_selector, :],
            gt_actions[drive_selector, :],
            action_names,
            pred_actions[drive_selector, :],
            caption=drive,
        )
        for img in images:
            log_images.append(img)  # noqa: PERF402


def _plot_losses_callback(
    loss_accumulator: torch.Tensor,
    nnz_accumulator: torch.Tensor,
    gt_actions: torch.Tensor,
    action_names: list[str],
    pred_actions: torch.Tensor | None = None,
    caption: str | None = None,
) -> Image:
    """Plot the losses and actions over time for each action dimension."""
    action_dim = (
        loss_accumulator.shape[-1]
        if loss_accumulator is not None
        else gt_actions.shape[-1]
    )
    n_rows = 1 + int(gt_actions is not None)
    fig_nnz, ax_nnz = plt.subplots(action_dim * n_rows, 1, figsize=(20, 20))
    fig_full, ax_full = plt.subplots(action_dim, 1, figsize=(20, 8))
    is_nnz = nnz_accumulator is not None
    for i in range(action_dim):
        if is_nnz:
            ax_nnz[i * n_rows].plot(
                loss_accumulator[nnz_accumulator[:, i], i].cpu().numpy(),
                label=f"{action_names[i]} loss",
            )
            ax_nnz[i * n_rows].grid(True)
            ax_nnz[i * n_rows].set_title(f"{action_names[i]} loss nnz")
        if gt_actions is not None:
            # plot actions in the nex row (zoom to nnz)
            if is_nnz:
                ax_nnz[i * n_rows + 1].plot(
                    gt_actions[nnz_accumulator[:, i], i],
                    label=f"{action_names[i]} gt",
                )
                ax_nnz[i * n_rows + 1].set_title(f"{action_names[i]} action nnz")
                ax_nnz[i * n_rows + 1].set_ylim([0, 1] if i < 2 else [-1, 1])
                ax_nnz[i * n_rows + 1].grid(True)
            # separate plot for all actions
            ax_full[i].set_title(f"{action_names[i]} action")
            ax_full[i].set_ylim([0, 1] if i < 2 else [-1, 1])
            ax_full[i].plot(
                gt_actions[:, i],
                label=f"{action_names[i]} gt",
            )
        if pred_actions is not None:
            # plot predictions on the same graph as GT
            if is_nnz:
                ax_nnz[i * n_rows + 1].plot(
                    pred_actions[nnz_accumulator[:, i], i],
                    label=f"{action_names[i]} pred",
                    alpha=0.7,
                )
            ax_full[i].plot(
                pred_actions[:, i],
                label=f"{action_names[i]} pred",
                alpha=0.7,
            )
            ax_full[i].grid(True)
    fig_full.set_tight_layout(True)
    fig_nnz.set_tight_layout(True)
    plt.tight_layout()
    return Image(fig_nnz, caption=f"{caption}_nnz"), Image(  # noqa: DOC201
        fig_full, caption=f"{caption}_complete"
    )


def action_callback(
    samples: pl.DataFrame,
    log_images: list[Image],
    pred_actions: torch.Tensor | None = None,
) -> None:
    """At the end of eval step: collect the loss metrics and plot the losses."""

    # Extract GT from DataFrame
    # action_names, gt_actions = _gt_factory(samples)
    action_names = [
        "meta/VehicleMotion/gas_pedal_normalized",
        "meta/VehicleMotion/brake_pedal_normalized",
        "meta/VehicleMotion/steering_angle_normalized",
    ]
    gt_actions = np.stack([np.stack(samples[name]) for name in action_names]).T

    drives = samples["input_id"]
    unique_drives = drives.unique().to_list()
    for drive in unique_drives:
        drive_selector = drives == drive
        images = _plot_losses_callback(
            None,
            None,
            gt_actions[drive_selector, :],
            action_names,
            pred_actions[drive_selector, :],
            caption=drive,
        )
        for img in images:
            log_images.append(img)  # noqa: PERF402
