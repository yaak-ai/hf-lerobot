import logging
from pathlib import Path

import hydra
import pytest
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.testing import make_tensor

from lerobot.policies.utils import get_device_from_parameters
from lerobot.utils.utils import init_logging


def dummy_input(device: torch.device, dtype: torch.dtype) -> dict:
    batch = {
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
            (1, 50, 3), dtype=dtype, device=device, low=0.0, high=1.0
        ),
        "observation.images.front_left": make_tensor(
            (1, 6, 3, 324, 576),
            dtype=dtype,
            device=device,
            low=0,
            high=1,
        ),
        "observation.state.vehicle": make_tensor(
            (1, 6, 1), dtype=dtype, device=device, low=0.0, high=130.0
        ),
        "observation.state.waypoints": make_tensor(
            (1, 6, 20),
            dtype=dtype,
            device=device,
            low=-148.0924835205078,
            high=146.4394073486328,
        ),
    }
    # 2nd arg
    noise = torch.normal(
        mean=0.0,
        std=1.0,
        size=(1, 50, 32),
        dtype=dtype,
        device=device,
    )
    # 3rd arg: always torch.float32
    time = torch.tensor([0.5], device=device, dtype=torch.float32)
    return batch, noise, time


@hydra.main(version_base=None)
@torch.inference_mode()
def main(cfg: DictConfig) -> None:
    # export_dynamo(cfg)  # noqa: ERA001
    test_fp16_tradeoff(cfg)


def export_onnx_only(cfg: DictConfig) -> None:
    logging.debug("instantiating policy")  # noqa: LOG015
    policy, _ = instantiate(cfg.model)
    device = get_device_from_parameters(policy)
    policy.eval()

    logging.debug("torch exporting")  # noqa: LOG015
    batch, noise, time = dummy_input(device)
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


def prepare_model_data(cfg: DictConfig, dtype: torch.dtype) -> None:
    logging.debug("instantiating policy")  # noqa: LOG015
    policy, _ = instantiate(cfg.model)

    policy = policy.to(dtype).to(cfg.device)
    policy.eval()

    device = get_device_from_parameters(policy)
    args = dummy_input(device, dtype)
    return policy, args


def test_fp16_tradeoff(cfg: DictConfig) -> None:
    policy, _ = prepare_model_data(cfg, torch.float32)

    policy_export, _ = prepare_model_data(cfg, torch.float16)
    n_rollouts = 10
    device = get_device_from_parameters(policy)
    comparison = torch.zeros((n_rollouts, 50, 3), dtype=torch.float32, device=cfg.device)
    average_comparison = torch.zeros((n_rollouts,), dtype=torch.float32, device=cfg.device)
    for i in range(n_rollouts):
        args = dummy_input(device, torch.float32)
        result = policy(*args)
        # simulate exporting
        with torch.inference_mode(), pytest.MonkeyPatch.context() as m:
            m.setattr("torch.compiler._is_exporting_flag", True)
            args_export = dummy_input(device, torch.float16)
            result_export = policy_export(*args_export)

            average_comparison[i] = torch.abs(result_export[0] - result[0])
            comparison[i, ...] = torch.abs(result_export[1] - result[1])[..., :3]
    logging.info(f"Per element mean {comparison.mean():.4f}")  # noqa: G004, LOG015
    logging.info(f"95th quantile {comparison.quantile(0.95):.4f}")  # noqa: G004, LOG015
    logging.info(f"mean of means {average_comparison.mean():.4f}")  # noqa: G004, LOG015


def export_dynamo(cfg: DictConfig) -> None:
    logging.debug("instantiating policy")  # noqa: LOG015
    policy, args = prepare_model_data(cfg, torch.float16)

    with torch.inference_mode(), pytest.MonkeyPatch.context() as m:
        m.setattr("torch.compiler._is_exporting_flag", True)
        result = policy(args)

    logging.info("torch exporting")  # noqa: LOG015
    exported_program = torch.export.export(mod=policy, args=tuple(args), strict=True)

    dest = Path(cfg["f"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.export.save(exported_program, dest.parent / "dynamo_exported.pt2")
    # exported_program = torch.export.load(dest.parent / "dynamo_exported.pt2")  # noqa: E501, ERA001
    # result = exported_program.module()(batch_dict(device, dtype), noise, time)  # noqa: ERA001
    logging.info("torch export done")  # noqa: LOG015

    logging.info("onnx exporting")  # noqa: LOG015
    model = torch.onnx.export(
        model=exported_program,
        args=tuple(args),
        f=cfg["f"],
        artifacts_dir=cfg["artifacts_dir"],
        external_data=False,
        dynamo=True,
        optimize=True,
        verify=True,
        report=True,
    )

    logging.info(f"exported to {cfg['artifacts_dir']}")  # noqa: G004, LOG015


if __name__ == "__main__":
    init_logging()
    main()
