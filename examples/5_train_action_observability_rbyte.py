from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
from hydra.utils import instantiate

from lerobot.utils.action_predictability import MLP, batch_mlp_corr
from lerobot.utils.utils import init_logging
import matplotlib.pyplot as plt  # noqa: PLC0415

if TYPE_CHECKING:
    import polars as pl
    from omegaconf import DictConfig

    from rbyte import Dataset


def _plot_mse_metrics(
    learned_train_loss: list,
    learned_val_loss: list,
    expert_train_loss: list,
    expert_val_loss: list,
) -> None:
    _, ax = plt.subplots(2, 1, figsize=(12, 6))
    ax[0].plot(learned_train_loss, label="Learned Judge Train Loss")
    ax[0].plot(learned_val_loss, label="Learned Judge Val Loss")
    ax[0].set_title("Learned Judge Losses")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Loss")
    ax[0].legend()
    ax[0].grid(True)  # noqa: FBT003
    ax[1].plot(expert_train_loss, label="Expert Judge Train Loss")
    ax[1].plot(expert_val_loss, label="Expert Judge Val Loss")
    ax[1].set_title("Expert Judge Losses")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Loss")
    ax[1].legend()
    ax[1].grid(True)  # noqa: FBT003
    plt.tight_layout()
    Path("tmp").mkdir(parents=True, exist_ok=True)
    plt.savefig("tmp/action_observability_losses.png")

    _, ax = plt.subplots(2, 1, figsize=(12, 12))
    ax[0].plot(
        expert_val_loss, label="Expert Judge Val Loss", color="blue", linestyle="--"
    )
    ax[0].plot(
        learned_val_loss, label="Learned Judge Val Loss", color="orange", linestyle="--"
    )
    ax[0].set_title("Action Predictability Loss Comparison")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Loss")
    ax[0].legend()
    ax[0].grid(True)  # noqa: FBT003
    ax[1].plot(
        expert_val_loss, label="Expert Judge Val Loss", color="blue", linestyle="--"
    )
    ax[1].set_title("Action Predictability Expert Loss")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Loss")
    ax[1].legend()
    ax[1].grid(True)  # noqa: FBT003
    plt.savefig("tmp/action_predictability_loss_comparison.png")


def _reye_monkey_patcher(pt: Path) -> pl.DataFrame:
    import polars as pl  # noqa: PLC0415
    from rbyte.io import (  # noqa: PLC0415
        DataFrameConcater,
        DataFrameDuckDbQuery,
        DataFrameGroupByDynamic,
    )

    parquet_files = Path(pt).glob("*.parquet")
    df = pl.read_parquet(list(parquet_files))
    frame_col = "batch/data/meta/ImageMetadata.cam_front_left/frame_idx"
    gby = DataFrameGroupByDynamic(
        index_column=frame_col,
        every="10i",
        period="101i",
        gather_every=1,
    )
    dbquery = DataFrameDuckDbQuery()
    query = """
        SELECT list_slice("predictions/policy/prediction_value/continuous/gas_pedal"::FLOAT[101], 1, 60, 10) AS "meta/VehicleMotion/gas_pedal_history",
            list_slice("predictions/policy/prediction_value/continuous/brake_pedal"::FLOAT[101], 1, 60, 10) AS "meta/VehicleMotion/brake_pedal_history",
            list_slice("predictions/policy/prediction_value/continuous/steering_angle"::FLOAT[101], 1, 60, 10) AS "meta/VehicleMotion/steering_angle_history",
            ("predictions/policy/prediction_value/continuous/gas_pedal")[52] AS "meta/VehicleMotion/gas_pedal_normalized",
            ("predictions/policy/prediction_value/continuous/brake_pedal")[52] AS "meta/VehicleMotion/brake_pedal_normalized",
            ("predictions/policy/prediction_value/continuous/steering_angle")[52] AS "meta/VehicleMotion/steering_angle_normalized",
        FROM samples
        WHERE len("batch/data/meta/ImageMetadata.cam_front_left/frame_idx") = 101
        AND list_last("batch/data/meta/ImageMetadata.cam_front_left/frame_idx")
        - list_first("batch/data/meta/ImageMetadata.cam_front_left/frame_idx") == 100
        """
    input_col = "batch/meta/input_id"
    drives = df[input_col].unique().to_list()
    samples = []
    for drive in drives:
        logging.info(f"Processing drive: {drive}")  # noqa: G004, LOG015
        samples.append(
            dbquery(query=query, samples=gby(df.filter(pl.col(input_col) == drive)))
        )

    cctr = DataFrameConcater(
        key_column="input_id",
    )
    return cctr(
        keys=drives,
        values=samples,
    )


def _train_action_observability(hydra_cfg: DictConfig) -> None:
    init_logging()

    dataset_val: Dataset = instantiate(hydra_cfg.datamodule_val.dataset)
    logging.info(f"Samples in the holdout set: {len(dataset_val.samples)}")  # noqa: G004, LOG015
    policy = instantiate(hydra_cfg.model.judge)
    learned_train_loss, learned_val_loss = _train_learned_judge(
        hydra_cfg.model.learner.reye_path, dataset_val, policy
    )
    expert_train_loss, expert_val_loss = _train_expert_judge(
        hydra_cfg, dataset_val, policy
    )
    _plot_mse_metrics(
        learned_train_loss,
        learned_val_loss,
        expert_train_loss,
        expert_val_loss,
    )
    logging.info(f"Expert judge MSE : {expert_val_loss[-1]:e}")  # noqa: G004, LOG015
    logging.info(f"Learned judge MSE : {learned_val_loss[-1]:e}")  # noqa: G004, LOG015

    action_predictability_ratio = expert_val_loss[-1] / learned_val_loss[-1]

    logging.info(  # noqa: LOG015
        f"Action predictability ratio (Expert / Learned): {action_predictability_ratio:.4f}"  # noqa: E501, G004
    )


def _train_expert_judge(
    hydra_cfg: DictConfig, dataset_val: Dataset, policy: MLP
) -> tuple[list[float], list[float]]:
    dataset: Dataset = instantiate(hydra_cfg.dataset)
    logging.info(f"Samples in the train set: {len(dataset.samples)}")  # noqa: G004, LOG015

    _, train_losses, val_losses = batch_mlp_corr(
        policy, dataset.samples, dataset_val.samples, "train_judge_expert_policy"
    )
    return train_losses, val_losses


def _train_learned_judge(
    reye_pred_path: Path, dataset_val: Dataset, policy: MLP
) -> tuple[list[float], list[float]]:
    df_pred = _reye_monkey_patcher(reye_pred_path)
    _, train_losses, val_losses = batch_mlp_corr(
        policy, df_pred, dataset_val.samples, "train_judge_learned_policy"
    )
    return train_losses, val_losses


@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    return _train_action_observability(cfg)


if __name__ == "__main__":
    main()
