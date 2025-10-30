import logging
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
import onnxruntime as ort
import pytest
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.testing import make_tensor

from lerobot.policies.smolvla.conversion_utils_yaak import __getbatch__
from lerobot.policies.smolvla.modeling_smolvla import resize_with_pad
from lerobot.policies.utils import get_device_from_parameters
from lerobot.utils.utils import init_logging

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from torch.utils.data import DataLoader


class ExportPolicy(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, batch, noise):
        return self.policy.predict_action_chunk(batch, noise)


def dummy_input(device: torch.device, dtype: torch.dtype) -> dict:
    batch = {
        "meta/ImageMetadata.cam_front_left/time_stamp": torch.tensor(
            [
                [
                    1670067032648344,
                    1670067032981700,
                    1670067033314001,
                    1670067033646234,
                    1670067033978448,
                    1670067034310751,
                ]
            ],
            dtype=torch.int64,
            device=device,
        ),
        "task": [
            "Given the waypoints and current vehicle speed, follow the waypoints while adhering to traffic rules and regulations"
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


def episode_input(cfg: DictConfig, device: torch.device, dtype: torch.dtype) -> dict:
    dataloader_test: DataLoader = instantiate(cfg.datamodule)
    batch = __getbatch__(next(iter(dataloader_test)))
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            if v.dtype != dtype:
                batch[k] = v.to(dtype)
            batch[k] = batch[k].to(device)
    im_key = "observation.images.front_left"
    batch[im_key] = resize_with_pad(batch[im_key][0], 512, 512, pad_value=0)[None, ...]
    _, noise, time = dummy_input(torch.device(cfg.device), dtype)
    batch.pop("meta/ImageMetadata.cam_front_left/time_stamp", None)
    batch.pop("action.continuous", None)
    return batch, noise, time


@hydra.main(version_base=None)
@torch.inference_mode()
def main(cfg: DictConfig) -> None:
    export_dynamo(cfg)
    # test_fp16_tradeoff(cfg)
    # test_onnx_export(cfg)


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

    device = get_device_from_parameters(policy)
    args = dummy_input(device, dtype)
    return policy, args


def test_fp16_tradeoff(cfg: DictConfig) -> None:  # noqa: PLR0914
    policy, _ = prepare_model_data(cfg, torch.float32)
    dtype = torch.float16 if cfg.dtype == "torch.float16" else torch.float32
    policy_export, _ = prepare_model_data(cfg, dtype)
    n_rollouts = 10
    device = get_device_from_parameters(policy)
    metric = torch.nn.CosineSimilarity(dim=-1)
    l1_over_horizon = torch.zeros(
        (n_rollouts, 50, 3), dtype=torch.float32, device=cfg.device
    )
    cosine_flat = torch.zeros((n_rollouts,), dtype=torch.float32, device=cfg.device)
    cosine_over_horizon = torch.zeros(
        (n_rollouts, 50), dtype=torch.float32, device=cfg.device
    )
    l1_mean = torch.zeros((n_rollouts,), dtype=torch.float32, device=cfg.device)
    for i in range(n_rollouts):
        args = dummy_input(device, torch.float32)
        args_export = [
            {
                k: deepcopy(v).to(dtype) if isinstance(v, torch.Tensor) else deepcopy(v)
                for k, v in args[0].items()
            }
        ]
        args_export.extend([args[1].to(dtype), args[-1]])
        result = policy(*args)
        # simulate exporting
        with torch.inference_mode(), pytest.MonkeyPatch.context() as m:
            m.setattr("torch.compiler._is_exporting_flag", True)
            result_export = policy_export(*args_export)

            l1_mean[i] = torch.abs(result_export[0] - result[0])
            l1_over_horizon[i, ...] = torch.abs(result_export[1] - result[1])[..., :3]
            # cosine similarity: compare each of 50 action vectors separately
            horizon_tensorrt = result_export[1].to(torch.float32)[..., :3]
            horizon_torch = result[1][..., :3]
            cosine_over_horizon[i, ...] = metric(horizon_tensorrt, horizon_torch)
            # cosine similarity: just flat out 50x3 vectors into a single vector
            # (Gr00t style) & compute the angle
            cosine_flat[i] = metric(horizon_tensorrt.flatten(), horizon_torch.flatten())

    logging.info(f"Per element l1 mean {l1_over_horizon.mean():.4f}")  # noqa: G004, LOG015
    logging.info(f"l1 95th quantile {l1_over_horizon.quantile(0.95):.4f}")  # noqa: G004, LOG015
    logging.info(f"l1 mean of means {l1_mean.mean():.4f}")  # noqa: G004, LOG015

    stat = cosine_over_horizon.mean()
    logging.info(  # noqa: LOG015
        f"Cosine similarity per horizon mean {torch.rad2deg(torch.acos(stat)):.4f} deg ({stat:.4f})"  # noqa: G004
    )
    stat = cosine_over_horizon.quantile(0.95)
    logging.info(  # noqa: LOG015
        f"Cosine similarity per horizon 95th quantile {torch.rad2deg(torch.acos(stat)):.4f} deg ({stat:.4f})"  # noqa: E501, G004
    )
    stat = cosine_flat.mean()
    logging.info(  # noqa: LOG015
        f"Cosine similarity flat mean {torch.rad2deg(torch.acos(stat)):.4f} deg ({stat:.4f})"  # noqa: E501, G004
    )
    stat = cosine_flat.quantile(0.95)
    logging.info(  # noqa: LOG015
        f"Cosine similarity flat 95th quantile {torch.rad2deg(torch.acos(stat)):.4f} deg ({stat:.4f})"  # noqa: E501, G004
    )


def export_dynamo(cfg: DictConfig) -> None:
    logging.debug("instantiating policy")  # noqa: LOG015
    dtype = torch.float16 if cfg.dtype == "torch.float16" else torch.float32
    policy_vla, _ = prepare_model_data(cfg, dtype)

    # temporary hack
    policy_vla.config.num_steps = 1
    policy_vla.config.resize_imgs_with_padding = None

    args = episode_input(cfg, torch.device(cfg.device), dtype)

    policy: ExportPolicy = ExportPolicy(policy_vla)
    policy = torch.compile(policy)
    policy.eval()

    # with torch.inference_mode(), pytest.MonkeyPatch.context() as m:
    #     m.setattr("torch.compiler._is_exporting_flag", True)  # noqa: ERA001
    #     result = policy(*args[:-1])  # noqa: ERA001

    logging.info("torch exporting")  # noqa: LOG015
    exported_program = torch.export.export(
        mod=policy, args=tuple(args[:-1]), strict=True
    )

    dest = Path(cfg["f"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.export.save(exported_program, dest.parent / "dynamo_exported.pt2")
    # exported_program = torch.export.load(dest.parent / "dynamo_exported.pt2")  # noqa: E501, ERA001
    # result = exported_program.module()(*args)  # noqa: ERA001
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
        dump_exported_program=True,
    )

    logging.info(f"exported to {cfg['artifacts_dir']}")  # noqa: G004, LOG015


def test_onnx_export(cfg: DictConfig) -> None:
    session = ort.InferenceSession(cfg.test.path)
    dtype = torch.float16 if cfg.dtype == "torch.float16" else torch.float32
    args = dummy_input(torch.device(cfg.device), dtype)

    inputs = session.get_inputs()
    names = [input_arg.name for input_arg in inputs]
    # Assign inputs by name if possible
    input_feed_dict = {
        "batch_" + k.lower().replace("/", "_").replace(".", "_"): v.numpy()
        for k, v in args[0].items()
        if "batch_" + k.lower().replace("/", "_").replace(".", "_") in names
    }
    not_assigned = {
        k: v
        for k, v in args[0].items()
        if "batch_" + k.lower().replace("/", "_").replace(".", "_") not in names
    }
    if "noise" in names:
        input_feed_dict["noise"] = args[1].numpy()
    if "time" in names:
        input_feed_dict["time"] = args[2].numpy()

    # Assign remaining inputs by shape
    for input_val in inputs:
        if input_val.name in input_feed_dict:
            continue
        for k, v in not_assigned.items():
            if isinstance(v, torch.Tensor) and torch.Size(input_val.shape) == v.shape:
                input_feed_dict[input_val.name] = v.numpy()
                not_assigned.pop(k)
                break

    # Run inference
    outputs = session.run(None, input_feed_dict)
    logging.info(f"Outputs: {outputs}")  # noqa: G004, LOG015


if __name__ == "__main__":
    init_logging()
    main()
