from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
from hydra.utils import instantiate

from lerobot.scripts.eval_yaak import (
    predict_main_yaak,
)
from lerobot.utils.utils import init_logging

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from torch.utils.data import DataLoader


def _eval_yaak(hydra_cfg: DictConfig) -> None:
    init_logging()

    dataloader_test: DataLoader = instantiate(hydra_cfg.datamodule)
    samples = dataloader_test.dataset.samples
    logging.info(f"Samples in the test set: {len(samples)}")  # noqa: G004

    output_dir = (
        Path("/nasa/team-space/artifacts/predictions/lerobot/")
        / Path(hydra_cfg.model.artifact).name / "same_noise"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    # samples.write_parquet(output_dir / "dataset.parquet")  # noqa: ERA001
    logging.info(f"Writing results to: {output_dir}")  # noqa: G004
    policy, train_cfg = instantiate(hydra_cfg.model)
    predict_main_yaak(policy, dataloader_test, train_cfg, output_dir)
    logging.info(f"Results written to: {output_dir}")  # noqa: G004
    print("Evaluation completed.")  # noqa: T201


@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    return _eval_yaak(cfg)


if __name__ == "__main__":
    main()
