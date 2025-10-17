import logging

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.testing import make_tensor

from lerobot.policies.utils import get_device_from_parameters
from lerobot.utils.utils import init_logging


def batch_dict(device: torch.device) -> dict:
    return {
        "meta/ImageMetadata.cam_front_left/time_stamp": torch.tensor(
            [
                1670067032648344,
                1670067032981700,
                1670067033314001,
                1670067033646234,
                1670067033978448,
                1670067034310751,
            ],
            dtype=torch.int64,
            device=device,
        ),
        "task": [
            "Given the waypoints and current vehicle speed, follow the waypoints while adhering to traffic rules and regulations."
        ],
        "action.continuous": make_tensor(
            (1, 50, 3), dtype=torch.float32, device=device, low=0.0, high=1.0
        ),
        "observation.images.front_left": make_tensor(
            (1, 6, 3, 324, 576),
            dtype=torch.float32,
            device=device,
            low=0,
            high=1,
        ),
        "observation.state.vehicle": make_tensor(
            (1, 6, 1), dtype=torch.float32, device=device, low=0.0, high=130.0
        ),
        "observation.state.waypoints": make_tensor(
            (1, 6, 20),
            dtype=torch.float32,
            device=device,
            low=-148.0924835205078,
            high=146.4394073486328,
        ),
    }


@hydra.main(version_base=None)
@torch.inference_mode()
def main(cfg: DictConfig) -> None:
    export_onnx_only(cfg)
    export_dynamo(cfg)


def export_onnx_only(cfg: DictConfig) -> None:
    logging.debug("instantiating policy")  # noqa: LOG015
    policy, _ = instantiate(cfg.model)
    device = get_device_from_parameters(policy)
    policy.eval()

    logging.debug("torch exporting")  # noqa: LOG015
    batch = batch_dict(device)
    # 2nd arg
    noise = torch.normal(
        mean=0.0,
        std=1.0,
        size=(1, 50, 32),
        dtype=torch.float32,
        device=device,
    )
    # 3rd arg
    time = torch.tensor([0.5], device=device, dtype=torch.float32)
    torch.onnx.export(
        policy,
        (batch, noise, time),
        "policy.onnx",
        export_params=True,
        do_constant_folding=True,
        input_names=["batch", "noise", "time"],
        output_names=["loss"],
        dynamic_axes={
            "batch": {0: "batch_size"},
            "noise": {0: "batch_size"},
        },
    )


def export_dynamo(cfg: DictConfig) -> None:
    logging.debug("instantiating policy")  # noqa: LOG015
    policy, _ = instantiate(cfg.model)
    device = get_device_from_parameters(policy)
    args = [batch_dict(device)]
    policy.eval()

    logging.debug("torch exporting")  # noqa: LOG015
    exported_program = torch.export.export(mod=policy, args=tuple(args), strict=True)
    logging.debug("onnx exporting")  # noqa: LOG015
    model = torch.onnx.export(
        model=exported_program,
        external_data=False,
        dynamo=True,
        optimize=True,
        verify=True,
    )

    logging.debug("exported")  # noqa: LOG015


if __name__ == "__main__":
    init_logging()
    main()
