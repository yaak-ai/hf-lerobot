import logging
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from lerobot.utils.utils import init_logging


def plot_errors(short_names, d_orig, d, gt_actions_orig, drive):
    fig, ax = plt.subplots(len(short_names), 1, figsize=(20, 12))

    for i in range(len(short_names)):
        ax[i].plot(d_orig[:, i], label=f"error orig {short_names[i]}", color="red")
        ax[i].plot(
            d[:, i],
            label=f"error approx {short_names[i]}",
            color="orange",
            linestyle="--",
        )
        ax[i].plot(gt_actions_orig[:, i], label=f"{short_names[i]}", alpha=0.7)
        ax[i].legend()

    plt.legend()
    plt.tight_layout()
    ax[0].set_title(f"Errors for drive {drive}, should be 0")
    dest = Path(f"tmp/{drive}/tmp_errors.png")
    dest.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(dest)
    logging.info(f"Saved plot to {dest}")  # noqa: G004, LOG015


def generate_error_table(l1_approx, l1_orig, short_names):
    print("Negative values are good for the 1st two rows")  # noqa: T201
    print("| Error per action | " + " | ".join(short_names) + " |")  # noqa: T201
    print("|--------|" + "|".join(["--------"] * len(short_names)) + "|")  # noqa: T201

    # Print rows
    rows = [
        "approx/orig-1",
        "nnz(approx/orig-1)",
        "approx",
        "orig",
        "nnz(approx)",
        "nnz(orig)",
    ]

    percentage = [
        l1_approx[short_names[c]]["mean_all"] / l1_orig[short_names[c]]["mean_all"] - 1
        for c in range(3)
    ]
    print(  # noqa: T201
        f"| {rows[0]} | " + " | ".join([f"{s * 100:.2f}%" for s in percentage]) + " |"
    )
    percentage_nnz = [
        l1_approx[short_names[c]]["mean"] / l1_orig[short_names[c]]["mean"] - 1
        for c in range(3)
    ]
    print(  # noqa: T201
        f"| {rows[1]} | "
        + " | ".join([f"{s * 100:.2f}%" for s in percentage_nnz])
        + " |"
    )
    print(  # noqa: T201
        f"| {rows[2]} | "
        + " | ".join([f"{l1_approx[short_names[c]]['mean_all']:.5f}" for c in range(3)])
        + " |"
    )
    print(  # noqa: T201
        f"| {rows[3]} | "
        + " | ".join([f"{l1_orig[short_names[c]]['mean_all']:.5f}" for c in range(3)])
        + " |"
    )
    print(  # noqa: T201
        f"| {rows[4]} | "
        + " | ".join([f"{l1_approx[short_names[c]]['mean']:.5f}" for c in range(3)])
        + " |"
    )
    print(  # noqa: T201
        f"| {rows[5]} | "
        + " | ".join([f"{l1_orig[short_names[c]]['mean']:.5f}" for c in range(3)])
        + " |"
    )


def compute_l1(actions, gt_actions, short_names, verbose=True):
    zero_conds = [0.04, 1.0 / 164 + 0.001, 1.0 / 164 + 0.001]

    l1_stats = {}
    for act_ind in [0, 1, 2]:
        cond = gt_actions[:, act_ind] > zero_conds[act_ind]
        ae_all = np.abs(actions[:, act_ind] - gt_actions[:, act_ind])
        ae = np.abs(actions[cond, act_ind] - gt_actions[cond, act_ind])
        if verbose:
            logging.info(  # noqa: LOG015
                f"{short_names[act_ind]} MSE: {np.mean(ae):.3f} median {np.median(ae):.3f}, std: {np.std(ae):.3f}, nr_samples: {len(ae)} , {sum(cond)}"  # noqa: G004
            )
            logging.info(  # noqa: LOG015
                f"{short_names[act_ind]} MSE all: {np.mean(ae_all):.3f} median {np.median(ae_all):.3f}, std: {np.std(ae_all):.3f}, nr_samples: {len(ae_all)}"  # noqa: G004
            )
        l1_stats[short_names[act_ind]] = {
            "mean": np.mean(ae),
            "median": np.median(ae),
            "max": np.max(ae),
            "std": np.std(ae),
            "nr_samples": len(ae),
            "mean_all": np.mean(ae_all),
            "median_all": np.median(ae_all),
            "max_all": np.max(ae_all),
            "std_all": np.std(ae_all),
            "nr_samples_all": len(ae_all),
        }

    logging.info(f"Zero conditions {short_names}: {zero_conds}")  # noqa: G004, LOG015
    return l1_stats


def load_actions(df):
    action_names = [
        "predictions/policy/ground_truth/continuous/gas_pedal",
        "predictions/policy/ground_truth/continuous/brake_pedal",
        "predictions/policy/ground_truth/continuous/steering_angle",
    ]
    short_names = [name.split("/")[-1] for name in action_names]
    gt_actions = np.stack([np.stack(df[name]) for name in action_names], axis=-1)
    action_names = [
        "predictions/policy/prediction_value/continuous/gas_pedal",
        "predictions/policy/prediction_value/continuous/brake_pedal",
        "predictions/policy/prediction_value/continuous/steering_angle",
    ]
    actions = np.stack([np.stack(df[name]) for name in action_names], axis=-1)
    return actions, gt_actions, short_names


def test_fp16_tradeoff(cfg: DictConfig) -> None:
    orig = instantiate(cfg.export.models.bfloat16, _recursive_=True, _convert_="all")
    converted = instantiate(
        cfg.export.models.float16, _recursive_=True, _convert_="all"
    )
    logging.info("Models intatntiated")  # noqa: LOG015
    drive_column = cfg.export.drive_column
    drives = orig[drive_column].unique()
    for drive in drives:
        test_fp16_tradeoff_drive(
            orig.filter(pl.col(drive_column) == drive),
            converted.filter(pl.col(drive_column) == drive),
            drive,
        )


def test_fp16_tradeoff_drive(orig, converted, drive):
    logging.info(f"============ Testing drive {drive }===============")  # noqa: G004, LOG015
    actions_orig, gt_actions_orig, _ = load_actions(orig)
    actions, gt_actions, short_names = load_actions(converted)

    l1_approx = compute_l1(actions, gt_actions, short_names, False)  # noqa: FBT003
    l1_orig = compute_l1(actions_orig, gt_actions_orig, short_names, False)  # noqa: FBT003
    generate_error_table(l1_approx, l1_orig, short_names)
    d_orig = np.abs(actions_orig - gt_actions_orig)
    d = np.abs(actions - gt_actions)
    plot_errors(short_names, d_orig, d, gt_actions_orig, drive)

    l1 = np.abs(actions_orig - actions)
    logging.info(f"Max total L1 {np.max(l1, axis=0)}")  # noqa: G004, LOG015
    logging.info(f"Mean total L1 {np.mean(l1, axis=0)}")  # noqa: G004, LOG015

    logging.info("L1 for non zero action values")  # noqa: LOG015
    non_zero_l1_stats = compute_l1(actions, actions_orig, short_names)
    for act, act_stat in non_zero_l1_stats.items():
        for k, v in act_stat.items():
            logging.info(f"{act}: {k}: {v:.4f}")  # noqa: G004, LOG015

    logging.info("Cosine similarity")  # noqa: LOG015
    metric = torch.nn.CosineSimilarity(dim=-1)
    for i, act in enumerate(short_names):
        cos_sim = metric(
            torch.from_numpy(actions_orig[:, i]), torch.from_numpy(actions_orig[:, i])
        )
        logging.info(f"{act}: cos sim = {cos_sim.item():.3f}")  # noqa: G004, LOG015


@hydra.main(version_base=None)
@torch.inference_mode()
def main(cfg: DictConfig) -> None:
    test_fp16_tradeoff(cfg)


if __name__ == "__main__":
    init_logging()
    main()
