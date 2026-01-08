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
from transformers.models.smolvlm.modeling_smolvlm import SmolVLMVisionTransformer

from lerobot.policies.smolvla.conversion_utils_yaak import __getbatch__
from lerobot.policies.smolvla.modeling_smolvla import resize_with_pad
from lerobot.policies.utils import get_device_from_parameters
from lerobot.utils.utils import init_logging
from lerobot.utils.wandb_utils import WandBLogger

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from torch.utils.data import DataLoader


class ExportVisionTower(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy.model.vlm_with_expert.vlm.model.vision_model
        self.connector = policy.model.vlm_with_expert.vlm.model.connector

    def forward(self, images, patch_attention_mask):
        return self.connector(
            self.policy(
                pixel_values=images,
                patch_attention_mask=patch_attention_mask,
            ).last_hidden_state
        )


class ExportPolicy(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy
        # self.policy = policy.model.vlm_with_expert.vlm.model.vision_model
        # self.policy = policy.model

    def forward(self, batch, noise):
        return self.policy.predict_action_chunk(batch, noise)

    # def forward_vision(self, images):
    #     return self.policy.embed_image(images)

    # def forward(self, images, patch_attention_mask):
    #     return self.policy(
    #             pixel_values=images,
    #             patch_attention_mask=patch_attention_mask,
    #         ).last_hidden_state

    # def forward(self, current_image, obs_hist_1, obs_hist2, obs_hist3, obs_hist4, obs_hist5, patch_attention_mask):
    #     current_emb = self.policy(
    #             pixel_values=current_image,
    #             patch_attention_mask=patch_attention_mask,
    #         ).last_hidden_state
    #     hist1 = self.policy(
    #             pixel_values=obs_hist_1,
    #             patch_attention_mask=patch_attention_mask,
    #         ).last_hidden_state
    #     hist2 = self.policy(
    #             pixel_values=obs_hist2,
    #             patch_attention_mask=patch_attention_mask,
    #         ).last_hidden_state
    #     hist3 = self.policy(
    #             pixel_values=obs_hist3,
    #             patch_attention_mask=patch_attention_mask,
    #         ).last_hidden_state
    #     hist4 = self.policy(
    #             pixel_values=obs_hist4,
    #             patch_attention_mask=patch_attention_mask,
    #         ).last_hidden_state
    #     hist5 = self.policy(
    #             pixel_values=obs_hist5,
    #             patch_attention_mask=patch_attention_mask,
    #         ).last_hidden_state
    #     return torch.cat([current_emb, hist1, hist2, hist3, hist4, hist5], dim=0)

    # def forward_fwd(
    #     self,
    #     attention_mask: torch.Tensor | None = None,
    #     position_ids: torch.LongTensor | None = None,
    #     past_key_values: list[torch.FloatTensor] | None = None,
    #     inputs_embeds: list[torch.FloatTensor] = None,
    #     use_cache: bool | None = None,
    #     fill_kv_cache: bool | None = None,
    # ):
    #     return self.policy(
    #         attention_mask,
    #         position_ids,
    #         past_key_values,
    #         inputs_embeds,
    #         use_cache,
    #         fill_kv_cache,
    #     )

    # def forward(
    #     self,
    #     prefix_att_2d_masks,
    #     prefix_position_ids,
    #     prefix_embs,
    #     noise,
    #     prefix_pad_masks,
    # ):
    #     _, past_key_values = self.policy.vlm_with_expert.forward(
    #         attention_mask=prefix_att_2d_masks,
    #         position_ids=prefix_position_ids,
    #         past_key_values=None,
    #         inputs_embeds=[prefix_embs, None],
    #         use_cache=True,
    #         fill_kv_cache=True,
    #     )
    #     bsize = prefix_att_2d_masks.shape[0]
    #     device = prefix_att_2d_masks.device
    #     dt = -1.0 / self.policy.config.num_steps
    #     dtype = (
    #         torch.float32
    #         if not torch.compiler.is_exporting()
    #         else self.policy.action_in_proj.weight.dtype
    #     )
    #     dt = torch.tensor(dt, dtype=dtype, device=device)
    #     x_t = noise
    #     for time in torch.arange(
    #         1.0, 0, -1.0 / self.policy.config.num_steps, dtype=dtype, device=device
    #     ):
    #         expanded_time = time.expand(bsize)
    #         v_t = self.policy.denoise_step_orig(
    #             prefix_pad_masks,
    #             past_key_values,
    #             x_t,
    #             expanded_time,
    #         )
    #         # Euler step
    #         x_t += dt * v_t
    #     return x_t

    # def forward(
    #     self,
    #     prefix_att_2d_masks,
    #     past_key_values,
    #     noise,
    #     prefix_pad_masks,
    # ):
    #     bsize = prefix_att_2d_masks.shape[0]
    #     device = prefix_att_2d_masks.device
    #     dt = -1.0 / self.policy.config.num_steps
    #     dtype = (
    #         torch.float32
    #         if not torch.compiler.is_exporting()
    #         else self.policy.action_in_proj.weight.dtype
    #     )
    #     dt = torch.tensor(dt, dtype=dtype, device=device)
    #     x_t = noise
    #     for time in torch.arange(
    #         1.0, 0, -1.0 / self.policy.config.num_steps, dtype=dtype, device=device
    #     ):
    #         expanded_time = time.expand(bsize)
    #         v_t = self.policy.denoise_step_orig(
    #             prefix_pad_masks,
    #             past_key_values,
    #             x_t,
    #             expanded_time,
    #         )
    #         # Euler step
    #         x_t += dt * v_t
    #     return x_t

    # def forward(
    #     self,
    #     prefix_att_2d_masks,
    #     prefix_position_ids,
    #     prefix_embs,
    # ):
    #     _, past_key_values = self.policy.vlm_with_expert.forward(
    #         attention_mask=prefix_att_2d_masks,
    #         position_ids=prefix_position_ids,
    #         past_key_values=None,
    #         inputs_embeds=[prefix_embs, None],
    #         use_cache=True,
    #         fill_kv_cache=True,
    #     )
    #     return past_key_values


def dummy_vlmexpert(device: torch.device, dtype: torch.dtype, bsize: int = 1) -> dict:
    attention_mask = torch.load("tmp/attention_mask.pt", map_location=device)
    position_ids = torch.load("tmp/position_ids.pt", map_location=device)
    past_key_values = None
    inputs_embeds_0 = torch.load("tmp/inputs_embeds_0.pt", map_location=device).to(
        dtype
    )
    inputs_embeds = [inputs_embeds_0, None]
    use_cache = True
    fill_kv_cache = True
    return (  # pyright: ignore[reportReturnType]
        attention_mask,
        position_ids,
        past_key_values,
        inputs_embeds,
        use_cache,
        fill_kv_cache,
    )


def dummy_vla(device: torch.device, dtype: torch.dtype, bsize: int = 1) -> dict:
    prefix_att_2d_masks = torch.load("tmp/prefix_att_2d_masks.pt", map_location=device)
    prefix_position_ids = torch.load("tmp/prefix_position_ids.pt", map_location=device)
    prefix_embs = torch.load("tmp/prefix_embs.pt", map_location=device).to(dtype)
    noise = torch.load("tmp/noise.pt", map_location=device).to(dtype)
    prefix_pad_masks = torch.load("tmp/prefix_pad_masks.pt", map_location=device)
    return (  # pyright: ignore[reportReturnType]
        prefix_att_2d_masks,
        prefix_position_ids,
        prefix_embs,
        noise,
        prefix_pad_masks,
    )


def dummy_obs(device: torch.device, dtype: torch.dtype, bsize: int = 1) -> dict:
    prefix_att_2d_masks = torch.load("tmp/prefix_att_2d_masks.pt", map_location=device)
    prefix_position_ids = torch.load("tmp/prefix_position_ids.pt", map_location=device)
    prefix_embs = torch.load("tmp/prefix_embs.pt", map_location=device).to(dtype)
    return (  # pyright: ignore[reportReturnType]
        prefix_att_2d_masks,
        prefix_position_ids,
        prefix_embs,
    )


def dummy_denoise(device: torch.device, dtype: torch.dtype, bsize: int = 1) -> dict:
    prefix_att_2d_masks = torch.load("tmp/prefix_att_2d_masks.pt", map_location=device)
    past_key_values = torch.load("tmp/past_key_values.pt", map_location=device)
    noise = torch.load("tmp/noise.pt", map_location=device).to(dtype)
    prefix_pad_masks = torch.load("tmp/prefix_pad_masks.pt", map_location=device)
    return (  # pyright: ignore[reportReturnType]
        prefix_att_2d_masks,
        past_key_values,
        noise,
        prefix_pad_masks,
    )


def dummy_input(device: torch.device, dtype: torch.dtype, bsize: int = 1) -> dict:
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
            (bsize, 50, 3), dtype=dtype, device=device, low=0.0, high=1.0
        ),
        "observation.images.front_left": make_tensor(
            (bsize, 6, 3, 324, 576),
            dtype=dtype,
            device=device,
            low=0,
            high=1,
        ),
        "observation.state.vehicle": make_tensor(
            (bsize, 6, 1), dtype=dtype, device=device, low=0.0, high=130.0
        ),
        "observation.state.waypoints": make_tensor(
            (bsize, 6, 20),
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
        size=(bsize, 50, 32),
        dtype=dtype,
        device=device,
    )
    # 3rd arg: always torch.float32
    time = torch.tensor([0.5], device=device, dtype=torch.float32)
    return batch, noise, time  # pyright: ignore[reportReturnType]


def episode_input(
    cfg: DictConfig, device: torch.device, dtype: torch.dtype, normalize_image=False
) -> dict:
    dataloader_test: DataLoader = instantiate(cfg.datamodule)
    batch = __getbatch__(next(iter(dataloader_test)))
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            if v.dtype != dtype:
                batch[k] = v.to(dtype)
            batch[k] = batch[k].to(device)
    im_key = "observation.images.front_left"
    batch[im_key] = resize_with_pad(
        torch.reshape(batch[im_key], (-1, *batch[im_key].shape[-3:])),
        512,
        512,
        pad_value=0,
    ).reshape((*batch[im_key].shape[:-2], 512, 512))
    if normalize_image:
        batch[im_key] *= 2.0
        batch[im_key] -= 1.0
    _, noise, time = dummy_input(
        torch.device(cfg.device), dtype, bsize=batch[im_key].shape[0]
    )
    batch.pop("meta/ImageMetadata.cam_front_left/time_stamp", None)
    batch.pop("action.continuous", None)
    return batch, noise, time


@hydra.main(version_base=None)
@torch.inference_mode()
def main(cfg: DictConfig) -> None:
    export_dynamo(cfg)


def prepare_model_data(cfg: DictConfig, dtype: torch.dtype) -> None:
    logging.debug("instantiating policy")  # noqa: LOG015
    with torch.inference_mode(), pytest.MonkeyPatch.context() as m:
        m.setattr("torch.compiler._is_exporting_flag", True)
        policy, train_cfg = instantiate(cfg.model)

    policy = policy.to(dtype).to(cfg.device)
    policy.eval()

    device = get_device_from_parameters(policy)
    args = dummy_input(device, dtype)
    return policy, train_cfg, args


def export_lm_expert(policy_vla, args, dynamo_kwargs, onnx_kwargs):
    pass


def export_vision(policy_vla, args, dynamo_kwargs, onnx_kwargs, wandb_logger):
    policy = ExportVisionTower(policy_vla)

    # SigLip args only (policy.model.vlm_with_expert.vlm.model.vision_model)
    images = args[0]["observation.images.front_left"][:1, :, ...]
    images = images.reshape((
        -1,
        *images.shape[2:],
    ))
    args = (
        images,
        torch.ones((images.shape[0], 32, 32), dtype=torch.bool, device=images.device),
    )

    # with torch.inference_mode(), pytest.MonkeyPatch.context() as m:  # noqa: SIM117
    #     m.setattr("torch.compiler._is_exporting_flag", True)  # noqa: ERA001
    #     result = policy(*args)  # noqa: ERA001

    exported_program = torch.export.export(mod=policy, args=(args), **dynamo_kwargs)
    _ = torch.onnx.export(model=exported_program, args=tuple(args), **onnx_kwargs)
    wandb_logger.log_onnx(Path(onnx_kwargs["artifacts_dir"]))


def export_dynamo(cfg: DictConfig) -> None:
    logging.debug("instantiating policy")  # noqa: LOG015
    dtype = torch.bfloat16

    # dtype = torch.float16 if cfg.dtype == "torch.float16" else torch.float32
    policy_vla, policy_cfg, _ = prepare_model_data(cfg, dtype)
    policy_vla.normalize_inputs
    policy_cfg.job_name = cfg.job_name
    policy_cfg.output_dir = cfg.artifacts_dir
    wandb_logger = WandBLogger(policy_cfg)

    dynamo_kwargs = instantiate(cfg.dynamo_kwargs)
    onnx_kwargs = instantiate(cfg.onnx_kwargs)
    vision_kwargs = instantiate(cfg.vision_kwargs)
    lm_expert_kwargs = instantiate(cfg.lm_expert_kwargs)
    shape_kwargs = instantiate(cfg.shape_kwargs)

    # temporary hack
    # policy_vla.config.num_steps = 1  # noqa: ERA001
    policy_vla.config.resize_imgs_with_padding = None

    policy: ExportPolicy = ExportPolicy(policy_vla)
    # policy = torch.compile(policy)  # noqa: ERA001
    policy.eval()
    # LLM fwd pass only (forward_fwd)
    # args = dummy_vlmexpert(torch.device(cfg.device), dtype, bsize=1)  # noqa: ERA001

    # LLM fwd and BWD passes (policy.model w denoising)
    # args = dummy_vla(torch.device(cfg.device), dtype, bsize=1)  # noqa: ERA001

    # LLM denoise
    # args = dummy_denoise(torch.device(cfg.device), dtype, bsize=1)  # noqa: ERA001

    # LLM obs
    # args = dummy_obs(torch.device(cfg.device), dtype, bsize=1)  # noqa: ERA001

    args = episode_input(
        cfg,
        torch.device(cfg.device),
        dtype,
        isinstance(policy.policy, SmolVLMVisionTransformer),
    )[:-1]

    # SigLip args only (policy.model.vlm_with_expert.vlm.model.vision_model)
    # args = args[0]["observation.images.front_left"][:1, :, ...]  # noqa: ERA001
    # args = (args.reshape((-1, *args.shape[2:])), None)  # remove batch dim  # noqa: ERA001

    # Sequential SigLip args
    # args = args[0]["observation.images.front_left"][0, :, ...]  # noqa: ERA001
    # args = torch.split(args, 1)  # noqa: ERA001
    # args = list(args)  # noqa: ERA001
    # args.append(None)  # noqa: ERA001

    export_vision(
        policy_vla,
        args,
        # dynamo_kwargs,
        {**dynamo_kwargs, **shape_kwargs},
        {**onnx_kwargs, **vision_kwargs},
        wandb_logger,
    )

    # with torch.inference_mode(), pytest.MonkeyPatch.context() as m:  # noqa: SIM117
    #     m.setattr("torch.compiler._is_exporting_flag", True)  # noqa: ERA001
    #     result = policy(*args)  # noqa: ERA001
    logging.info("torch exporting")  # noqa: LOG015

    exported_program = torch.export.export(mod=policy, args=(args), **dynamo_kwargs)

    _ = torch.onnx.export(model=exported_program, args=tuple(args), **onnx_kwargs)
    # wandb_logger.log_onnx(cfg["artifacts_dir"])

    logging.info(f"exported to {cfg['artifacts_dir']}")  # noqa: G004, LOG015


if __name__ == "__main__":
    init_logging()
    main()
