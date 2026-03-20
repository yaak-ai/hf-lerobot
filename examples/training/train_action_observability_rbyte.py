from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
from hydra.utils import instantiate

from lerobot.utils.action_predictability import MLP, batch_mlp_corr
from lerobot.utils.utils import init_logging

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from rbyte import Dataset


def _train_action_observability(hydra_cfg: DictConfig) -> None:
    init_logging()

    dataset: Dataset = instantiate(hydra_cfg.dataset)
    logging.info(f"Samples in the train set: {len(dataset.samples)}")  # noqa: G004, LOG015
    dataset_val: Dataset = instantiate(hydra_cfg.datamodule_val.dataset)
    logging.info(f"Samples in the test set: {len(dataset_val.samples)}")  # noqa: G004, LOG015
    policy = instantiate(hydra_cfg.model)

    batch_mlp_corr(
        policy, dataset.samples, dataset_val.samples, "train_judge_expert_policy"
    )


@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    return _train_action_observability(cfg)


if __name__ == "__main__":
    main()
