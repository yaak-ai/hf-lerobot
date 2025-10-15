from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from hydra.utils import instantiate

from lerobot.utils.action_predictability import MLP, batch_mlp_corr
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import init_logging

if TYPE_CHECKING:
    import polars as pl
    from omegaconf import DictConfig

    from rbyte import Dataset


def _plot_mse_metrics(
    learned_train_loss: list,
    learned_val_loss: list,
    expert_train_loss: list,
    expert_val_loss: list,
    expert_train_run_name: str,
) -> None:
    _, ax = plt.subplots(2, 1, figsize=(12, 6))
    if learned_train_loss:
        ax[1].plot(learned_train_loss, label="Learned Train Loss")
        ax[1].plot(learned_val_loss, label="Learned Val Loss")
        ax[1].set_title("Learned: train vs val")
        ax[1].set_xlabel("Epochs")
        ax[1].set_ylabel("Loss")
        ax[1].legend()
        ax[1].grid(True)  # noqa: FBT003
    if expert_train_loss:
        ax[0].plot(expert_train_loss, label="Expert Train Loss")
        ax[0].plot(expert_val_loss, label="Expert Val Loss")
        ax[0].set_title("Expert: train vs val losses")
        ax[0].set_xlabel("Epochs")
        ax[0].set_ylabel("Loss")
        ax[0].legend()
        ax[0].grid(True)  # noqa: FBT003
    plt.tight_layout()
    Path(f"tmp/{expert_train_run_name}").mkdir(parents=True, exist_ok=True)
    plt.savefig(f"tmp/{expert_train_run_name}/train_vs_val.png")

    _, ax = plt.subplots(2, 1, figsize=(12, 12))
    if expert_val_loss:
        ax[0].plot(
            expert_val_loss, label="Expert Judge Val Loss", color="blue", linestyle="--"
        )
    if learned_val_loss:
        ax[0].plot(
            learned_val_loss,
            label="Learned Judge Val Loss",
            color="orange",
            linestyle="--",
        )
    ax[0].set_title("Val loss: Expert vs Learned")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("MSE Loss")
    ax[0].legend()
    ax[0].grid(True)  # noqa: FBT003
    if expert_val_loss:
        ax[1].plot(
            expert_val_loss, label="Expert Judge Val Loss", color="blue", linestyle="--"
        )
    ax[1].set_title("Val loss: Expert")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("MSE Loss")
    ax[1].legend()
    ax[1].grid(True)  # noqa: FBT003
    plt.savefig(
        f"tmp/{expert_train_run_name}/action_predictability_loss_comparison.png"
    )


def drive_versatility_measure(samples: pl.DataFrame) -> None:
    action_names = [
        "meta/VehicleMotion/gas_pedal_normalized",
        "meta/VehicleMotion/brake_pedal_normalized",
        "meta/VehicleMotion/steering_angle_normalized",
    ]
    th = [0.08, 0.12, 0.08]
    drives = samples["__input_id"]
    unique_drives = drives.unique().to_list()
    drive_stats = np.zeros((len(unique_drives), len(action_names) + 1), dtype=np.int32)
    for i, drive in enumerate(unique_drives):
        drive_rows = samples.filter(pl.col("__input_id") == drive)
        logging.info(f"Processing drive: {drive}")  # noqa: G004, LOG015
        drive_changes = []
        for col_name, t in zip(action_names, th, strict=True):
            gt_actions = np.stack(drive_rows[col_name])[:, 0]
            nnz_values = np.where(np.abs(gt_actions) > t)[0]
            diff = nnz_values[1:] - nnz_values[:-1]

            change_th = 10
            change = diff > change_th
            if change.sum() == 0:
                continue
            change = np.concatenate((change, np.zeros(1, dtype=bool)))
            change_shift = np.concatenate((np.zeros(1, dtype=bool), change[:-1]))
            change_indices = np.unique(
                np.concatenate((nnz_values[change_shift], nnz_values[change]))
            )
            drive_changes.append(len(change_indices))
        if len(drive_changes) < 1:
            logging.warning(f"Drive {drive} has no valid actions.")  # noqa: G004, LOG015
            continue
        drive_stats[i, 0] = np.sum(drive_changes)
        drive_stats[i, 1:] = np.array(drive_changes)

    inds = np.argsort(drive_stats[:, 0])[::-1]
    unique_drives = np.array(unique_drives, dtype=object)
    for drive, stats in zip(unique_drives[inds], drive_stats[inds], strict=True):
        logging.info(  # noqa: LOG015
            f"Drive: {drive}, Total Changes: {stats[0]}, "  # noqa: G004
            f"Gas Pedal Changes: {stats[1]}, "
            f"Brake Pedal Changes: {stats[2]}, "
            f"Steering Angle Changes: {stats[3]}"
        )


def _eval_action_observability(hydra_cfg: DictConfig) -> None:
    init_logging()
    set_seed(hydra_cfg.seed)

    # points to holdout_clip
    # dataset_val: Dataset = instantiate(hydra_cfg.datamodule_val.dataset)
    policy_learned = instantiate(hydra_cfg.model.judge)
    learner_checkpoint = Path(
        "outputs/train/2025-10-14/16-00-00_mlp_learner_rmind_t_10/checkpoints"
    )
    expert_checkpoint = Path(
        "outputs/train/2025-10-14/16-00-51_mlp_expert_rmind_t_10/checkpoints"
    )
    policy_learned.from_pretrained(learner_checkpoint)
    policy_expert = instantiate(hydra_cfg.model.judge)
    policy_expert.from_pretrained(expert_checkpoint)
    import torch

    #  constant braking
    data_past = torch.zeros(1, 5, 3, dtype=torch.float32)
    data_past[:, :, 1] = torch.tensor(0.3, dtype=torch.float32)
    data_past = data_past.reshape(-1, 15)

    results_learned = policy_learned(data_past)
    results_expert = policy_expert(data_past)
    logging.info(f"Learned results: {results_learned}")  # noqa: G004, LOG015
    logging.info(f"Expert results: {results_expert}")  # noqa: G004, LOG015

    # increasing gas pedal
    data_past = torch.zeros(1, 5, 3, dtype=torch.float32)
    data_past[:, :, 0] = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=torch.float32)
    data_past = data_past.reshape(-1, 15)
    results_learned = policy_learned(data_past)
    results_expert = policy_expert(data_past)
    logging.info(f"Learned results: {results_learned}")  # noqa: G004, LOG015
    logging.info(f"Expert results: {results_expert}")  # noqa: G004, LOG015

    # increasing brake pedal
    data_past = torch.zeros(1, 5, 3, dtype=torch.float32)
    data_past[:, :, 1] = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=torch.float32)
    data_past = data_past.reshape(-1, 15)
    results_learned = policy_learned(data_past)
    results_expert = policy_expert(data_past)
    logging.info(f"Learned results: {results_learned}")  # noqa: G004, LOG015
    logging.info(f"Expert results: {results_expert}")  # noqa: G004, LOG015

    # decreasing brake pedal
    data_past = torch.zeros(1, 5, 3, dtype=torch.float32)
    data_past[:, :, 1] = torch.arange(0.5, 0, -0.1, dtype=torch.float32)
    data_past = data_past.reshape(-1, 15)
    results_learned = policy_learned(data_past)
    results_expert = policy_expert(data_past)
    logging.info(f"Learned results: {results_learned}")  # noqa: G004, LOG015
    logging.info(f"Expert results: {results_expert}")  # noqa: G004, LOG015

    # decreasing gas pedal
    data_past = torch.zeros(1, 5, 3, dtype=torch.float32)
    data_past[:, :, 0] = torch.arange(0.5, 0, -0.1, dtype=torch.float32)
    data_past = data_past.reshape(-1, 15)
    results_learned = policy_learned(data_past)
    results_expert = policy_expert(data_past)
    logging.info(f"Learned results: {results_learned}")  # noqa: G004, LOG015
    logging.info(f"Expert results: {results_expert}")  # noqa: G004, LOG015


def _train_action_observability(hydra_cfg: DictConfig) -> None:
    init_logging()

    if "learner" in hydra_cfg.model:
        policy_learned = instantiate(hydra_cfg.model.judge)
        learned_train_loss, learned_val_loss = _train_learned_judge(
            hydra_cfg, policy_learned
        )
    else:
        learned_train_loss, learned_val_loss = None, None

    policy_expert = instantiate(hydra_cfg.model.judge)
    expert_train_loss, expert_val_loss = _train_expert_judge(
        hydra_cfg, policy_expert
    )
    _plot_mse_metrics(
        learned_train_loss,
        learned_val_loss,
        expert_train_loss,
        expert_val_loss,
        hydra_cfg.model.expert.train_run,
    )
    logging.info(f"Expert judge MSE : {expert_val_loss[-1]:e}")  # noqa: G004, LOG015

    if learned_val_loss:
        logging.info(f"Learned judge MSE : {learned_val_loss[-1]:e}")  # noqa: G004, LOG015

        action_predictability_ratio = expert_val_loss[-1] / learned_val_loss[-1]

        logging.info(  # noqa: LOG015
            f"Action predictability ratio (Expert / Learned): {action_predictability_ratio:.4f}"  # noqa: E501, G004
        )


def _train_expert_judge(
    hydra_cfg: DictConfig, policy: MLP
) -> tuple[list[float], list[float]]:
    df_train, df_holdout = instantiate(hydra_cfg.datamodule.dataset.samples_smolvla_expert)
    # df_train, df_holdout = instantiate(hydra_cfg.datamodule.dataset.samples_rmind_expert)
    # instantiate(hydra_cfg.dataset)  # noqa: ERA001
    # dataset = dataset.samples
    logging.info(f"Samples in the expert train set: {len(df_train)}")  # noqa: G004, LOG015
    logging.info(f"Samples in the expert holdout set: {len(df_holdout)}")  # noqa: G004, LOG015
    # drive_versatility_measure(dataset.samples)

    _, train_losses, val_losses = batch_mlp_corr(
        policy,
        df_train,
        df_holdout,
        hydra_cfg.model.expert,
        hydra_cfg.model.reye_dest,
    )
    return train_losses, val_losses


def _train_learned_judge(
    hydra_cfg: DictConfig, policy: MLP
) -> tuple[list[float], list[float]]:
    # points to bc_clip
    df_train, df_holdout = instantiate(hydra_cfg.datamodule.dataset.samples_smolvla_learner)
    # df_train, df_holdout = instantiate(hydra_cfg.datamodule.dataset.samples_rmind_learner)
    logging.info(f"Samples in the learner train set: {len(df_train)}")  # noqa: G004, LOG015
    logging.info(f"Samples in the learner holdout set: {len(df_holdout)}")  # noqa: G004, LOG015
    _, train_losses, val_losses = batch_mlp_corr(
        policy,
        df_train,
        df_holdout,
        hydra_cfg.model.learner,
        hydra_cfg.model.reye_dest,
    )
    return train_losses, val_losses


@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    # _eval_action_observability(cfg)
    return _train_action_observability(cfg)


if __name__ == "__main__":
    main()
