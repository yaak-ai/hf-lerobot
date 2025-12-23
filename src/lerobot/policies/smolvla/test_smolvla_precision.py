import logging

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from lerobot.utils.utils import init_logging


def compute_l1(actions, gt_actions, short_names):
    zero_conds = [0.04, 1.0 / 164 + 0.001, 1.0 / 164 + 0.001]

    l1_stats = {}
    for act_ind in [0, 1, 2]:
        cond = gt_actions[:, act_ind] > zero_conds[act_ind]
        ae_all = np.abs(actions[:, act_ind] - gt_actions[:, act_ind])
        ae = np.abs(actions[cond, act_ind] - gt_actions[cond, act_ind])
        logging.info(  # noqa: LOG015
            f"{short_names[act_ind]} MSE: {np.mean(ae):.3f} median {np.median(ae):.3f}, std: {np.std(ae):.3f}, nr_samples: {len(ae)} , {sum(cond)}",  # noqa: E501, G004
            f"MSE all: {np.mean(ae_all):.3f} median {np.median(ae_all):.3f}, std: {np.std(ae_all):.3f}, nr_samples: {len(ae_all)}",  # noqa: E501
        )
        l1_stats[short_names[act_ind]] = {
            "mean": np.mean(ae),
            "median": np.median(ae),
            "std": np.std(ae),
            "nr_samples": len(ae),
            "mean_all": np.mean(ae_all),
            "median_all": np.median(ae_all),
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

    actions_orig, gt_actions_orig, _ = load_actions(orig)
    actions, gt_actions, short_names = load_actions(converted)

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
