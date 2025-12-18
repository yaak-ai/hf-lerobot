import logging
from typing import TYPE_CHECKING

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from lerobot.utils.utils import init_logging

if TYPE_CHECKING:
    from omegaconf import DictConfig

@hydra.main(version_base=None)
@torch.inference_mode()
def main(cfg: DictConfig) -> None:
    export_dynamo(cfg)


def export_dynamo(cfg: DictConfig) -> None:
    logging.debug("instantiating policy")  # noqa: LOG015
    dtype = torch.float16 if cfg.dtype == "torch.float16" else torch.float32
    logging.info(f"instantiating the model of {dtype}")  # noqa: G004, LOG015
    policy = instantiate(cfg.model)
    policy = policy.to(dtype).to(cfg.device)
    policy.eval()

    image = instantiate(cfg.image_load)
    image_processor = instantiate(cfg.image_processor)
    args = image_processor.preprocess(image).to(
        cfg.device, dtype=dtype
    )
    _ = policy(args)
    logging.info("torch exporting")  # noqa: LOG015

    dynamo_kwargs = instantiate(cfg.dynamo_kwargs)
    exported_program = torch.export.export(mod=policy, args=(args,), **dynamo_kwargs)
    onnx_kwargs = instantiate(cfg.onnx_kwargs)
    _ = torch.onnx.export(model=exported_program, **onnx_kwargs)

    logging.info(f"exported {onnx_kwargs['f']}")  # noqa: G004, LOG015


if __name__ == "__main__":
    init_logging()
    main()
