from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from hydra.utils import instantiate

from lerobot.utils.action_predictability import MLP, batch_mlp_corr
from lerobot.utils.learned_pipeline import (
    RbyteColumns,
    ReyeColumnRmind,
    ReyeColumns,
    filter_rmind_by_copycat,
    reye_eval_dataset,
)
from lerobot.utils.utils import init_logging

if TYPE_CHECKING:
    import polars as pl
    from omegaconf import DictConfig

    from rbyte import Dataset


def _plot_sanity_check(
    dataset_val: Dataset, dest: Path, pred_actions: np.array = None
) -> None:
    samples = dataset_val.samples
    timestamps = np.stack(samples["meta/ImageMetadata.cam_front_left/time_stamp"])
    if "meta/ImageMetadata.cam_front_left_future/time_stamp" in samples.columns:
        timestamps_future = np.stack(
            samples["meta/ImageMetadata.cam_front_left_future/time_stamp"]
        )[:, :1]
        timestamps = np.concatenate([timestamps, timestamps_future], axis=-1)

    actions = np.stack(
        [
            np.stack(samples["meta/VehicleMotion/gas_pedal_normalized"]),
            np.stack(samples["meta/VehicleMotion/brake_pedal_normalized"]),
            np.stack(samples["meta/VehicleMotion/steering_angle_normalized"]),
        ],
        axis=-1,
    )
    history = (
        np.stack(
            (
                np.stack(samples["meta/VehicleMotion/gas_pedal_history"]),
                np.stack(samples["meta/VehicleMotion/brake_pedal_history"]),
                np.stack(samples["meta/VehicleMotion/steering_angle_history"]),
            ),
            axis=-1,
        )
        if "meta/VehicleMotion/gas_pedal_history" in samples
        else actions[:, :-1, :]
    )
    if "meta/VehicleMotion/gas_pedal_history" not in samples:
        actions = actions[:, -1:, :]
    _plot_across_actions(
        actions,
        history,
        timestamps,
        pred_actions,
        dest,
    )


def _plot_sanity_check_rmind(
    samples: pl.DataFrame, dest: Path, pred_actions: np.array = None
) -> None:
    columns = ReyeColumnRmind()
    new_samples = samples.with_columns(
        pl.from_epoch(pl.col(columns.col_timestamp).arr.get(0), time_unit="us").alias(
            "t0"
        ),
        pl.from_epoch(pl.col(columns.col_timestamp).arr.get(1), time_unit="us").alias(
            "t1"
        ),
        pl.from_epoch(pl.col(columns.col_timestamp).arr.get(2), time_unit="us").alias(
            "t2"
        ),
        pl.from_epoch(pl.col(columns.col_timestamp).arr.get(3), time_unit="us").alias(
            "t3"
        ),
        pl.from_epoch(pl.col(columns.col_timestamp).arr.get(4), time_unit="us").alias(
            "t4"
        ),
        pl.from_epoch(pl.col(columns.col_timestamp).arr.get(5), time_unit="us").alias(
            "t5"
        ),
    )
    timestamps = np.stack(
        [
            np.stack(new_samples["t0"]),
            np.stack(new_samples["t1"]),
            np.stack(new_samples["t2"]),
            np.stack(new_samples["t3"]),
            np.stack(new_samples["t4"]),
            np.stack(new_samples["t5"]),
        ],
        axis=-1,
    )
    actions = np.stack(
        [
            np.stack(samples[columns.col_gas_pred]),
            np.stack(samples[columns.col_brake_pred]),
            np.stack(samples[columns.col_steering_pred]),
        ],
        axis=-1,
    )[:, -1:, :]
    history = (
        np.stack(
            (
                np.stack(samples[columns.col_gas_gt]),
                np.stack(samples[columns.col_brake_gt]),
                np.stack(samples[columns.col_steering_gt]),
            ),
            axis=-1,
        )
    )[:, :, 0, :]
    _plot_across_actions(
        actions,
        history,
        timestamps,
        pred_actions,
        dest,
    )


def _plot_across_actions(
    actions: np.array,
    history: np.array,
    timestamps: np.array,
    pred_actions: np.array | None,
    dest: Path,
) -> None:
    a = 1
    short_names = ["gas", "brake", "steering"]
    inds = np.arange(len(actions))
    rng = np.arange(0, len(inds), len(inds) // min(len(inds), 19))

    _, ax = plt.subplots(len(rng), 1, figsize=(16, 4 * len(rng)))
    for i, ri in enumerate(rng):
        ax[i].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        ind = inds[ri]
        ax[i].plot(
            timestamps[ind, -1],
            actions[ind, 0, a],
            label=f"{short_names[a]}",
            marker="o",
        )
        if pred_actions is not None:
            ax[i].plot(
                timestamps[ind, -1],
                pred_actions[ind],
                label=f"{short_names[a]} pred",
                marker="o",
            )
        ax[i].plot(
            (
                timestamps[ind, :-1]
                if len(history[ind, :, a]) < len(timestamps[ind, ...])
                else timestamps[ind, ...]
            ),
            history[ind, :, a],
            label=f"{short_names[a]} history",
        )
        ax[i].set_title(f"Episode {ind}: {np.std(history[ind, :, a]):e}")
        ax[i].grid(True)
        ax[i].legend()
        ax[i].set_xlabel("Time step")
        ax[i].set_ylabel("Normalized value")
        ax[i].set_xticks(timestamps[ind, :])
    plt.tight_layout()
    plt.savefig(dest)


# _dir: /nasa/team-space/artifacts/predictions/yaak/cargpt
# _dir_lerobot: /nasa/team-space/artifacts/predictions/lerobot
# models:
#   mlp-rmind-train-set: ${_dir_lerobot}/mlp-train_expert_rmind_t_10
#   rmind: ${_dir}/model-ujqd8m57:v5/2025-10-10--15-38-43
#   smolvla-no-speed: ${_dir_lerobot}/policy_smolvla-seed_1000-dataset_yaak-ai_L2D-300000:v14/all_stride_10
#   smolvla-full: ${_dir_lerobot}/policy_smolvla-seed_1000-dataset_yaak-ai_L2D-560000:v3/all_mlp_stride_10


def compute_error(dataset: pl.DataFrame):
    pass


def _train_action_observability(hydra_cfg: DictConfig) -> None:
    init_logging()

    _dir = Path("/nasa/team-space/artifacts/predictions/yaak/cargpt")
    _dir_lerobot = Path("/nasa/team-space/artifacts/predictions/lerobot")

    # Copycat model
    dataset_val: Dataset = instantiate(hydra_cfg.datamodule_val.dataset)
    reye_pred_path = _dir_lerobot / "mlp-train_expert_rmind_t_10"
    _plot_sanity_check(dataset_val, Path(f"tmp/sanity_check_{reye_pred_path.name}.png"))
    eval_dataset = reye_eval_dataset(
        reye_pred_path,
        dataset_val.samples,
    )
    reye_columns = ReyeColumns()
    actions = np.stack(eval_dataset[reye_columns.col_brake_pred])
    gt_actions = np.stack(eval_dataset[reye_columns.col_brake_gt])
    mse_copycat = np.mean((actions - gt_actions) ** 2)
    l1_copycat = abs(actions - gt_actions)
    nr_samples_copycat = len(actions)
    logging.info(f"Mean Squared Error: {mse_copycat:e}")  # noqa: G004, LOG015

    _plot_sanity_check(
        dataset_val, Path(f"tmp/pred_sanity_check_{reye_pred_path.name}.png"), actions
    )

    # rmind
    dataset_rmind: Dataset = instantiate(hydra_cfg.dataset)
    reye_pred_path = _dir / "model-ujqd8m57:v5/2025-10-10--15-38-43"
    reye_key = "rmind"
    _plot_sanity_check(dataset_rmind, Path(f"tmp/sanity_check_{reye_key}.png"))
    eval_dataset_rmind = reye_eval_dataset(
        reye_pred_path,
        dataset_rmind.samples,
    )
    eval_dataset = filter_rmind_by_copycat(eval_dataset_rmind, eval_dataset)
    _plot_sanity_check_rmind(
        eval_dataset, Path(f"tmp/pred_sanity_check_{reye_key}.png"), None
    )

    reye_columns = ReyeColumnRmind()
    actions = np.stack(eval_dataset[reye_columns.col_brake_pred])[:, -1]
    gt_actions = np.stack(eval_dataset[reye_columns.col_brake_gt])[:, -1].flatten()
    mse_rmind = np.mean((actions - gt_actions) ** 2)
    l1_rmind = abs(actions - gt_actions)
    nr_samples_rmind = len(actions)
    logging.info(f"Mean Squared Error: {mse_rmind:e}")  # noqa: G004, LOG015

    # SmolVLA model
    reye_pred_path = (
        _dir_lerobot
        / "policy_smolvla-seed_1000-dataset_yaak-ai_L2D-560000:v3/all_mlp_stride_10"
    )
    reye_key = "smolvla-full"
    _plot_sanity_check(dataset_val, Path(f"tmp/sanity_check_{reye_key}.png"))
    eval_dataset = reye_eval_dataset(
        reye_pred_path,
        dataset_val.samples,
    )
    reye_columns = ReyeColumns()
    actions = np.stack(eval_dataset[reye_columns.col_brake_pred])
    gt_actions = np.stack(eval_dataset[reye_columns.col_brake_gt])
    mse_smol_vla = np.mean((actions - gt_actions) ** 2)
    l1_smol_vla = abs(actions - gt_actions)
    nr_samples_smol_vla = len(actions)
    logging.info(f"Mean Squared Error: {mse_smol_vla:e}")  # noqa: G004, LOG015

    _plot_sanity_check(
        dataset_val, Path(f"tmp/pred_sanity_check_{reye_pred_path.name}.png"), actions
    )

    ratio_rmind = mse_copycat / mse_rmind
    ratio_smolvla = mse_copycat / mse_smol_vla
    eps = 0.1
    eps_pp_copycat = l1_copycat * eps
    success_rate_rmind = (
        np.sum((l1_copycat - l1_rmind) > eps_pp_copycat) / len(l1_rmind) * 100
    )
    success_rate_smolvla = (
        np.sum((l1_copycat - l1_smol_vla) > eps_pp_copycat) / len(l1_smol_vla) * 100
    )

    cond = l1_copycat > 0.09
    cond = l1_copycat > -10
    success_rate_rmind_significant = (
        np.sum((l1_copycat[cond] - l1_rmind[cond]) > eps_pp_copycat[cond])
        / len(l1_copycat[cond])
        * 100
    )
    success_rate_smolvla_significant = (
        np.sum((l1_copycat[cond] - l1_smol_vla[cond]) > eps_pp_copycat[cond])
        / len(l1_copycat[cond])
        * 100
    )

    plt.figure(figsize=(16, 8))
    plt.plot(
        np.arange(len(l1_copycat[cond])),
        l1_copycat[cond],
        label="Copycat",
        marker="o",
        alpha=0.5,
        linestyle="--",
    )
    plt.plot(
        np.arange(len(l1_rmind[cond])),
        l1_rmind[cond],
        label="Rmind",
        marker="o",
        alpha=0.5,
    )
    plt.plot(
        np.arange(len(l1_smol_vla[cond])),
        l1_smol_vla[cond],
        label="SmolVLA",
        marker="o",
        alpha=0.5,
    )
    plt.title(
        f"L1 error rates: LOWER IS BETTER\n" +
        f"Success rate Rmind: {success_rate_rmind:.2f}%, significant {success_rate_rmind_significant:.2f}%, success rate Smol VLA {success_rate_smolvla:.2f}%, significant {success_rate_smolvla_significant:.2f}%"
    )  # noqa: G004, LOG015
    plt.xlabel("Sample index")
    plt.ylabel("L1 error")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(Path(f"tmp/l1_error_comparison_all.png"))

    logging.info(f"Mean Squared Error copycat: {mse_copycat:e} ({nr_samples_copycat})")  # noqa: G004, LOG015
    logging.info(f"Mean Squared Error rmind: {mse_rmind:e} ({nr_samples_rmind})")  # noqa: G004, LOG015
    logging.info(  # noqa: LOG015
        f"Mean Squared Error SmolVLA: {mse_smol_vla:e} ({nr_samples_smol_vla})"  # noqa: G004
    )
    logging.info(f"Copycat to Rmind ratio: {ratio_rmind:.4f}")  # noqa: G004, LOG015
    logging.info(f"Copycat to Smolvla ratio: {ratio_smolvla:.4f}")  # noqa: G004, LOG015
    logging.info(  # noqa: LOG015
        f"Success rate Rmind: {success_rate_rmind:.2f}%"  # noqa: G004
    )
    logging.info(  # noqa: LOG015
        f"Success rate Smolvla: {success_rate_smolvla:.2f}%"  # noqa: G004
    )
    logging.info(  # noqa: LOG015
        f"Success rate Rmind significant: {success_rate_rmind_significant:.2f}%"  # noqa: G004
    )
    logging.info(  # noqa: LOG015
        f"Success rate Smolvla significant: {success_rate_smolvla_significant:.2f}%"  # noqa: G004
    )


@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    return _train_action_observability(cfg)


if __name__ == "__main__":
    main()
