import datetime
import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
import numpy as np
import onnxruntime as ort
import pytest
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from tqdm import tqdm

from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.types import FeatureType, NormalizationMode
from lerobot.constants_yaak import ACTION, OBS_IMAGE, OBS_STATE, OBS_STATE_VEHICLE
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.smolvla.conversion_utils_yaak import __getbatch__, patch_norm_mode
from lerobot.policies.smolvla.modeling_smolvla import resize_with_pad
from lerobot.policies.smolvla.modeling_smolvlm import (
    ExportSmolVLMVisionEmbeddings,
    ExportSmolVLMVisionTransformer,
)
from lerobot.policies.utils import get_device_from_parameters
from lerobot.utils.reye_utils import create_reye_df
from lerobot.utils.utils import init_logging
from lerobot.utils.wandb_utils import WandBLogger

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


class ExportActionModel(torch.nn.Module):
    def __init__(self, policy: PreTrainedPolicy, action_dim: int) -> None:
        super().__init__()
        self.policy = policy
        self.action_dim = action_dim

    def forward(
        self,
        prefix_embs: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        prefix_att_masks: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        actions = self.policy.model.sample_actions_embeddings(
            prefix_embs, prefix_pad_masks, prefix_att_masks, noise
        )[:, :, : self.action_dim]
        # Clamp gas and brake, as denoising (with smaller no of steps) can produce negative values
        # For steering, clamping may not be necasrry but just to be on the safe side
        return torch.cat(
            [
                torch.clamp(
                    actions[:, :, :-1],
                    min=torch.tensor(0.0).to(actions.device),
                    max=torch.tensor(1.0).to(actions.device),
                ),
                torch.clamp(
                    actions[:, :, -1:],
                    min=torch.tensor(-1.0).to(actions.device),
                    max=torch.tensor(1.0).to(actions.device),
                ),
            ],
            dim=-1,
        )


class ExportEmbeddingModelFull(torch.nn.Module):
    def __init__(
        self, policy: PreTrainedPolicy, lang_emb: torch.Tensor, lang_masks: torch.Tensor
    ) -> None:
        super().__init__()
        self.policy = policy

        # replace SmolVLM components with export compatible versions
        vision_tower = policy.model.vlm_with_expert.vlm.model.vision_model
        dtype = next(vision_tower.parameters()).dtype
        device = get_device_from_parameters(vision_tower)
        export_tower = ExportSmolVLMVisionTransformer(vision_tower.config)
        export_tower.load_state_dict(vision_tower.state_dict())
        export_tower = export_tower.to(device).to(dtype)
        policy.model.vlm_with_expert.vlm.model.vision_model = export_tower

        vision_embeddings = (
            policy.model.vlm_with_expert.vlm.model.vision_model.embeddings
        )
        dtype = next(vision_embeddings.parameters()).dtype
        export_embeddings = ExportSmolVLMVisionEmbeddings(
            policy.model.vlm_with_expert.vlm.model.vision_model.config
        )
        export_embeddings.load_state_dict(vision_embeddings.state_dict())
        export_embeddings = export_embeddings.to(device).to(dtype)
        policy.model.vlm_with_expert.vlm.model.vision_model.embeddings = (
            export_embeddings
        )

        # Save language embeddings into a buffer
        self.policy.register_buffer("lang_emb", lang_emb)
        self.policy.register_buffer("lang_masks", lang_masks)

    def forward(self, batch: dict) -> tuple:
        # Normalization in episode construction
        bsize, seq_len = batch[OBS_IMAGE].shape[:2]
        device = batch[OBS_IMAGE].device
        images, img_masks = (
            batch[OBS_IMAGE],  # B, L, C, W, H
            torch.ones((bsize, seq_len, 1), dtype=torch.bool, device=device),  # B, L, 1
        )
        state = (
            self.policy.prepare_state(batch)
            if not self.policy.use_context
            else self.policy.prepare_state_context(batch)
        )
        prefix_embs, prefix_pad_masks, prefix_att_masks = (
            self.policy.model.embed_prefix(
                images,
                img_masks,
                self.policy.lang_emb,
                self.policy.lang_masks,
                state=state,
            )
        )
        return prefix_embs, prefix_pad_masks, prefix_att_masks


class ExportEmbeddingModelIncremental(torch.nn.Module):
    def __init__(
        self,
        policy: PreTrainedPolicy,
        lang_emb: torch.Tensor,
        lang_masks: torch.Tensor,
        context_length: int,
        num_tokens: list,
    ) -> None:
        super().__init__()
        self.policy = policy
        self.context_length = context_length
        self.num_tokens = num_tokens

        # replace SmolVLM components with export compatible versions
        vision_tower = policy.model.vlm_with_expert.vlm.model.vision_model
        dtype = next(vision_tower.parameters()).dtype
        device = get_device_from_parameters(vision_tower)
        export_tower = ExportSmolVLMVisionTransformer(vision_tower.config)
        export_tower.load_state_dict(vision_tower.state_dict())
        export_tower = export_tower.to(device).to(dtype)
        policy.model.vlm_with_expert.vlm.model.vision_model = export_tower

        vision_embeddings = (
            policy.model.vlm_with_expert.vlm.model.vision_model.embeddings
        )
        dtype = next(vision_embeddings.parameters()).dtype
        export_embeddings = ExportSmolVLMVisionEmbeddings(
            policy.model.vlm_with_expert.vlm.model.vision_model.config
        )
        export_embeddings.load_state_dict(vision_embeddings.state_dict())
        export_embeddings = export_embeddings.to(device).to(dtype)
        policy.model.vlm_with_expert.vlm.model.vision_model.embeddings = (
            export_embeddings
        )

        # Save language embeddings into a buffer
        self.policy.register_buffer("lang_emb", lang_emb)
        self.policy.register_buffer("lang_masks", lang_masks)

    def forward(
        self,
        batch: dict,
        prefix_embs_cache: torch.Tensor,
        prefix_pad_masks_cache: torch.Tensor,
        prefix_att_masks_cache: torch.Tensor,
    ) -> tuple:
        # Normalization in episode construction
        bsize, seq_len = batch[OBS_IMAGE].shape[:2]
        device = batch[OBS_IMAGE].device
        images, img_masks = (
            batch[OBS_IMAGE],  # B, L, C, W, H
            torch.ones((bsize, seq_len, 1), dtype=torch.bool, device=device),  # B, L, 1
        )
        state = (
            self.policy.prepare_state(batch)
            if not self.policy.use_context
            else self.policy.prepare_state_context(batch)
        )
        prefix_embs, _, _ = self.policy.model.embed_prefix(
            images,
            img_masks,
            self.policy.lang_emb,
            self.policy.lang_masks,
            state=state,
        )
        prefix = torch.cat(
            [
                prefix_embs_cache[
                    :,
                    self.num_tokens[0] : self.num_tokens[0] * self.context_length,
                    ...,
                ],
                prefix_embs[:, : self.num_tokens[0], ...],
                # language
                prefix_embs_cache[
                    :,
                    self.num_tokens[0] * self.context_length : self.num_tokens[0]
                    * self.context_length  # images
                    + self.num_tokens[1],  # language
                    ...,
                ],
                # waypoints / intent
                prefix_embs[
                    :,
                    -self.num_tokens[-2] - self.num_tokens[-1] : -self.num_tokens[-1],
                    ...,
                ].expand(-1, self.context_length, -1),
                # state / speed
                prefix_embs_cache[
                    :,
                    self.num_tokens[0] * self.context_length  # images
                    + self.num_tokens[1]  # language
                    + self.num_tokens[2] * self.context_length  # intent
                    + self.num_tokens[3] : self.num_tokens[0]
                    * self.context_length  # images
                    + self.num_tokens[1]  # language
                    + self.num_tokens[2] * self.context_length  # intent
                    + self.num_tokens[3] * self.context_length,  # state
                    ...,
                ],
                prefix_embs[
                    :,
                    -self.num_tokens[-1] :,
                    ...,
                ],
            ],
            dim=1,
        )
        # attention and padding masks are fixed
        return prefix, prefix_pad_masks_cache, prefix_att_masks_cache


class ExportModel(torch.nn.Module):
    def __init__(
        self,
        policy: PreTrainedPolicy,
        lang_emb: torch.Tensor,
        lang_masks: torch.Tensor,
        action_dim: int,
    ) -> None:
        super().__init__()
        self.embedding_model = ExportEmbeddingModelFull(policy, lang_emb, lang_masks)
        self.action_model = ExportActionModel(policy, action_dim)

    def forward(self, batch: dict, noise: torch.Tensor) -> torch.Tensor:
        (prefix_embs, prefix_pad_masks, prefix_att_masks) = self.embedding_model(batch)
        return (
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
            self.action_model(prefix_embs, prefix_pad_masks, prefix_att_masks, noise),
        )


class ExportModelIncremental(torch.nn.Module):
    def __init__(
        self,
        policy: PreTrainedPolicy,
        lang_emb: torch.Tensor,
        lang_masks: torch.Tensor,
        context_length: int,
        num_tokens: list,
        action_dim: int,
    ) -> None:
        super().__init__()
        self.embedding_model = ExportEmbeddingModelIncremental(
            policy, lang_emb, lang_masks, context_length, num_tokens
        )
        self.action_model = ExportActionModel(policy, action_dim)

    def forward(
        self,
        batch: dict,
        prefix_embs_cache: torch.Tensor,
        prefix_pad_masks_cache: torch.Tensor,
        prefix_att_masks_cache: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        (prefix_embs, prefix_pad_masks, prefix_att_masks) = self.embedding_model(
            batch, prefix_embs_cache, prefix_pad_masks_cache, prefix_att_masks_cache
        )
        return (
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
            self.action_model(prefix_embs, prefix_pad_masks, prefix_att_masks, noise),
        )


def build_episode(
    cfg: DictConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    dataloader_test: DataLoader = instantiate(cfg.datamodule)
    batch = __getbatch__(next(iter(dataloader_test)))
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            if v.dtype != dtype:
                batch[k] = v.to(dtype)
            batch[k] = batch[k].to(device)
    batch.pop("meta/ImageMetadata.cam_front_left/time_stamp", None)
    batch.pop(ACTION, None)
    return batch


def _normalize_min_max(
    buffer: nn.ParameterDict, input_tensor: torch.Tensor
) -> torch.Tensor:
    # normalize to [0,1]
    input_tensor = (input_tensor - buffer["min"]) / (
        buffer["max"] - buffer["min"] + 1e-8
    )
    # normalize to [-1, 1]
    input_tensor = input_tensor * 2 - 1
    return torch.clamp(input_tensor, -1, 1)


def _normalize_zero_one(
    buffer: nn.ParameterDict, input_tensor: torch.Tensor
) -> torch.Tensor:
    # normalize to [0,1]
    return (input_tensor - buffer["min"]) / (buffer["max"] - buffer["min"] + 1e-8)


def _normalize_state(normalization_parameters: dict, batch: dict) -> None:
    if normalization_parameters["batch_observation_state_vehicle"]["norm_mode"] != str(
        NormalizationMode.ZERO_ONE
    ):
        msg = f" Add support for {normalization_parameters['batch_observation_state_vehicle']['norm_mode']}"
        raise ValueError(msg)
    batch[OBS_STATE_VEHICLE] = _normalize_zero_one(
        normalization_parameters["batch_observation_state_vehicle"]["buffer"],
        batch[OBS_STATE_VEHICLE],
    )
    if normalization_parameters["batch_observation_state_waypoints"][
        "norm_mode"
    ] != str(NormalizationMode.MIN_MAX):
        msg = f" Add support for {normalization_parameters['batch_observation_state_waypoints']['norm_mode']}"
        raise ValueError(msg)
    batch[OBS_STATE] = _normalize_min_max(
        normalization_parameters["batch_observation_state_waypoints"]["buffer"],
        batch[OBS_STATE],
    )


def update_episode_for_policy(
    embedding_kwargs: DictConfig,
    policy_vla: PreTrainedPolicy,
    batch: dict,
    device: torch.device,
    dtype: torch.dtype,
    normalization_parameters: dict,
) -> tuple:
    """
    1. Update batch and other model inputs based on policy config and embedding_kwargs
    2. Update policy config based on embedding_kwargs
    """  # noqa: DOC201
    # Images
    if embedding_kwargs["resize_in_episode_construction"]:
        batch[OBS_IMAGE] = resize_with_pad(
            torch.reshape(batch[OBS_IMAGE], (-1, *batch[OBS_IMAGE].shape[-3:])),
            *policy_vla.config.resize_imgs_with_padding,
            pad_value=0,
        ).reshape((
            *batch[OBS_IMAGE].shape[:-2],
            *policy_vla.config.resize_imgs_with_padding,
        ))
        # Do not resize in model, it's resized here
        policy_vla.config.resize_imgs_with_padding = None
        # Siglip normalization
        batch[OBS_IMAGE] *= 2.0
        batch[OBS_IMAGE] -= 1.0
    # Remove resize_in_episode_construction because it's not supported by torch.export
    embedding_kwargs.pop("resize_in_episode_construction")

    # Normalize state
    _normalize_state(normalization_parameters, batch)

    # Language
    # keep the tokenization outside the ONNX model since it produces errors
    # also keep it outside of episode construction since it is policy-dependent
    lang_tokens, lang_masks = policy_vla.prepare_language(batch)
    lang_emb = policy_vla.model.vlm_with_expert.embed_language_tokens(lang_tokens)
    # Normalize language embeddings
    lang_emb_dim = lang_emb.shape[-1]
    lang_emb *= math.sqrt(lang_emb_dim)
    # after tokenization, task text is no longer needed in the batch
    batch.pop("task")
    # Noise
    bsize = batch[OBS_IMAGE].shape[0]
    noise = torch.normal(
        mean=0.0,
        std=1.0,
        size=(bsize, policy_vla.config.chunk_size, policy_vla.config.max_action_dim),
        dtype=dtype,
        device=device,
    )
    return lang_emb, lang_masks, noise


def prepare_data_for_accuracy_test(
    cfg: DictConfig,
    device: torch.device,
    dtype: torch.dtype,
    policy_vla: PreTrainedPolicy,
    lang_emb: torch.Tensor,
    lang_masks: torch.Tensor,
    normalization_parameters: dict,
    noise: torch.Tensor,
) -> None:
    dataloader_test = instantiate(cfg.datamodule)
    w, h = 512, 512
    stats = torch.zeros((len(dataloader_test), 3), dtype=torch.float16, device="cuda")
    for step, elem in tqdm(enumerate(dataloader_test)):
        batch = __getbatch__(elem)
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                if v.dtype != dtype:
                    batch[k] = v.to(dtype)
                batch[k] = batch[k].to(device)
        batch.pop("meta/ImageMetadata.cam_front_left/time_stamp", None)
        batch[OBS_IMAGE] = resize_with_pad(
            torch.reshape(batch[OBS_IMAGE], (-1, *batch[OBS_IMAGE].shape[-3:])),
            w,
            h,
            pad_value=0,
        ).reshape((
            *batch[OBS_IMAGE].shape[:-2],
            w,
            h,
        ))
        # Siglip normalization
        batch[OBS_IMAGE] *= 2.0
        batch[OBS_IMAGE] -= 1.0
        wp_before = batch[OBS_STATE][0, -1, ...].clone()
        a = wp_before.reshape(10, 2)
        d = a[1:, :] - a[:-1, :]
        norms = d.norm(dim=1)
        stats[step, ...] = torch.tensor([norms.median(), norms.min(), norms.max()])
        _normalize_state(normalization_parameters, batch)
        wp = batch[OBS_STATE]
        print("Step")
    print("Done")
    return

    def torch_dtype_to_numpy(torch_dtype):
        mapping = {
            torch.float32: np.float32,
            torch.float16: np.float16,
            torch.float64: np.float64,
            torch.int32: np.int32,
            torch.int64: np.int64,
            torch.int8: np.int8,
            torch.uint8: np.uint8,
            torch.bool: np.bool_,
        }
        return mapping.get(torch_dtype, np.float32)

    dataloader_test = instantiate(cfg.datamodule)
    w, h = 512, 512
    # policy_dual = ExportDualModel(policy_vla, lang_emb, lang_masks, 3)
    policy_emb = ExportEmbeddingModelFull(policy_vla, lang_emb, lang_masks)
    policy_action = ExportActionModelIncremental(policy_vla, 3)

    pred_actions = torch.zeros(
        (len(dataloader_test.dataset), 3), dtype=dtype, device=device
    )
    onnx_actions = np.zeros(
        (len(dataloader_test.dataset), 3), dtype=torch_dtype_to_numpy(dtype)
    )
    emb_onnx_path = Path("outputs/2026-01-27/16-03-51/vision/embedding_dynamic.onnx")
    onnx_path = Path("outputs/2026-01-27/16-03-51/action/action3.onnx")
    noise_cpu = noise.clone().cpu()
    collected_data = []
    session = ort.InferenceSession(onnx_path)
    session_emb = ort.InferenceSession(emb_onnx_path)
    for step, elem in tqdm(enumerate(dataloader_test)):
        batch = __getbatch__(elem)
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                if v.dtype != dtype:
                    batch[k] = v.to(dtype)
                batch[k] = batch[k].to(device)
        batch.pop("meta/ImageMetadata.cam_front_left/time_stamp", None)
        batch[OBS_IMAGE] = resize_with_pad(
            torch.reshape(batch[OBS_IMAGE], (-1, *batch[OBS_IMAGE].shape[-3:])),
            w,
            h,
            pad_value=0,
        ).reshape((
            *batch[OBS_IMAGE].shape[:-2],
            w,
            h,
        ))
        # Siglip normalization
        batch[OBS_IMAGE] *= 2.0
        batch[OBS_IMAGE] -= 1.0

        _normalize_state(normalization_parameters, batch)

        # task = batch.pop("task")  # noqa: ERA001
        with torch.inference_mode(), pytest.MonkeyPatch.context() as m:
            m.setattr("torch.compiler._is_exporting_flag", True)
            # result = policy_dual(batch, noise.clone())  # noqa: ERA001
            # prefix_embs, prefix_pad_masks, prefix_att_masks = policy_emb(batch)  # noqa: E501, ERA001
            prefix_embs, prefix_pad_masks, prefix_att_masks = policy_emb({
                OBS_IMAGE: batch[OBS_IMAGE].clone(),
                OBS_STATE_VEHICLE: batch[OBS_STATE_VEHICLE].clone(),
                OBS_STATE: batch[OBS_STATE].clone(),
            })

            # prefix_embs_onnx, prefix_pad_masks_onnx, prefix_att_masks_onnx = (
            #     session_emb.run(
            #         None,
            #         {
            #             "batch_observation_images_front_left": batch[OBS_IMAGE]
            #             .clone()
            #             .cpu()
            #             .numpy(),
            #             "batch_observation_state_vehicle": batch[OBS_STATE_VEHICLE]
            #             .clone()
            #             .cpu()
            #             .numpy(),
            #             "batch_observation_state_waypoints": batch[OBS_STATE]
            #             .clone()
            #             .cpu()
            #             .numpy(),
            #         },
            #     )
            # )

            actions_onnx = session.run(
                None,
                {
                    "prefix_embs": prefix_embs.clone().cpu().numpy(),
                    "prefix_pad_masks": prefix_pad_masks.clone().cpu().numpy(),
                    "prefix_att_masks": prefix_att_masks.clone().cpu().numpy(),
                    "noise": noise.clone().cpu().numpy(),
                },
            )
            actions = policy_action(
                prefix_embs.clone(),
                prefix_pad_masks.clone(),
                prefix_att_masks.clone(),
                noise.clone(),
            )
            pred_actions[step, ...] = actions[0, 0, ...]
            onnx_actions[step, ...] = actions_onnx[0][0, 0, ...]
        # embs_cpu = prefix_embs.cpu().numpy()
        # embedding_error = (
        #     prefix_embs_onnx[0, : 64 * 6, ...] - embs_cpu[0, : 64 * 6, ...]
        # ).flatten()
        # state_lang_error = np.sum(
        #     np.abs(prefix_embs_onnx[0, 64 * 6 :, ...] - embs_cpu[0, 64 * 6 :, ...])
        # )
        # action_error = np.abs(
        #     actions_onnx[0][0, 0, ...] - actions[0, 0, ...].cpu().numpy()
        # )
        # logging.info(f"Action error {action_error}")  # noqa: G004, LOG015
        # logging.info(  # noqa: LOG015
        #     f"Vision error [{np.max(embedding_error)}, {np.median(embedding_error)}, {embedding_error.min()}], S={np.sum(np.abs(embedding_error))}"  # noqa: E501, G004
        # )
        # logging.info(f"State lang error {state_lang_error}")  # noqa: G004, LOG015
        # action_error_horizon = np.abs((actions_onnx[0][0, :4, ...] - actions[0, :4, ...].cpu().numpy()))  # noqa: ERA001
        # logging.info(f"Action error horizon {action_error_horizon}")  # noqa: ERA001
        # Collect batch and noise, moving to CPU for portability
        batch_cpu = {
            k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in batch.items()
        }
        collected_data.append({
            "batch": batch_cpu,
            "prefix_embs": prefix_embs.cpu(),
            "prefix_pad_masks": prefix_pad_masks.cpu(),
            "prefix_att_masks": prefix_att_masks.cpu(),
            "actions": actions.cpu(),
        })
    # Serialize collected data to file
    output_path = Path(cfg.artifacts_dir) / "accuracy_test_data.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "data": collected_data,
            "noise": noise_cpu,
            "lang_emb": lang_emb.cpu(),
            "lang_masks": lang_masks.cpu(),
            "metadata": {
                "dtype": str(dtype),
                "device": str(device),
                "num_batches": len(collected_data),
                "image_size": (w, h),
                "chunk_size": policy_vla.config.chunk_size,
                "max_action_dim": policy_vla.config.max_action_dim,
            },
        },
        output_path,
    )

    logging.info(f"Saved {len(collected_data)} batches to {output_path}")  # noqa: G004, LOG015

    # reye serialization
    # Handle cases with clip and without clip based on the timestamp
    ts = dataloader_test.dataset.samples[
        "meta/ImageMetadata.cam_front_left/time_stamp"
    ][0]
    df = create_reye_df(
        dataloader_test,
        pred_actions.cpu(),
        is_without_clip=isinstance(ts, datetime.datetime),
    )
    reye_torch = Path(cfg.reye_torch)
    reye_torch.mkdir(parents=True, exist_ok=True)
    reye_path = reye_torch / "results.parquet"
    df.write_parquet(reye_path)

    df_onnx = create_reye_df(
        dataloader_test, onnx_actions, is_without_clip=isinstance(ts, datetime.datetime)
    )
    reye_onnx = Path(cfg.reye_onnx)
    reye_onnx.mkdir(parents=True, exist_ok=True)
    reye_path = reye_onnx / "results.parquet"
    df_onnx.write_parquet(reye_path)

    delta_input = Path(cfg.artifacts_dir) / "samples.parquet"
    dataloader_test.dataset.samples.write_parquet(delta_input)

    logging.info(f"""
    cd delta_accuracy
    rsync -av valentina@berghain:{output_path.resolve()} .
    rsync -av {output_path.name} valentina@delta:/home/valentina/data
    rsync -av valentina@berghain:{delta_input.resolve()} .
    rsync -av {delta_input.name} valentina@delta:/home/valentina/data
    rsync -av {delta_input.name} nvidia@delta-emc1:/home/nvidia/accuracy
    """)  # noqa: G004, LOG015


def prepare_model_data(cfg: DictConfig, dtype: torch.dtype) -> None:
    logging.debug("instantiating policy")  # noqa: LOG015
    with torch.inference_mode(), pytest.MonkeyPatch.context() as m:
        m.setattr("torch.compiler._is_exporting_flag", True)
        policy, train_cfg = instantiate(cfg.model)

    policy = policy.to(dtype).to(cfg.device)
    policy.eval()
    return policy, train_cfg


def export_normalization_params(
    policy_vla: PreTrainedPolicy, normalization_kwargs: dict, batch: dict
) -> dict:
    normalization_parameters = {}

    norm_mode = policy_vla.normalize_inputs.norm_map.get(
        FeatureType.STATE, NormalizationMode.IDENTITY
    )
    norm_mode = patch_norm_mode(norm_mode, OBS_STATE_VEHICLE, batch)
    normalization_parameters["batch_observation_state_vehicle"] = {
        "buffer": policy_vla.normalize_inputs.buffer_observation_state_vehicle,
        "norm_mode": str(norm_mode),
    }

    norm_mode = policy_vla.normalize_inputs.norm_map.get(
        FeatureType.STATE, NormalizationMode.IDENTITY
    )
    norm_mode = patch_norm_mode(norm_mode, OBS_STATE, batch)
    normalization_parameters["batch_observation_state_waypoints"] = {
        "buffer": policy_vla.normalize_inputs.buffer_observation_state_waypoints,
        "norm_mode": str(norm_mode),
    }
    norm_path = Path(normalization_kwargs["f"])
    norm_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(normalization_parameters, norm_path)

    logging.info(f"""
    mkdir -p {Path(norm_path).stem}
    cd {norm_path.stem}
    rsync -av valentina@berghain:{norm_path.resolve()} .
    rsync -av {norm_path.name} valentina@delta:/home/valentina
    rsync -av {norm_path.name} nvidia@delta-emc1:/home/nvidia/normalization
    """)  # noqa: G004, LOG015
    return normalization_parameters


def export_embedding_model_full(
    policy_vla: PreTrainedPolicy,
    args: tuple,
    dynamo_kwargs: dict,
    onnx_kwargs: dict,
    wandb_logger: WandBLogger,
) -> tuple:
    batch, lang_emb, lang_masks = args
    policy = ExportEmbeddingModelFull(policy_vla, lang_emb, lang_masks)
    policy.eval()
    exported_program = torch.export.export(mod=policy, args=(batch,), **dynamo_kwargs)
    _ = torch.onnx.export(
        model=exported_program,
        args=(batch,),
        **onnx_kwargs,
    )
    wandb_logger.log_onnx(Path(onnx_kwargs["artifacts_dir"]))
    logging.info(f"""
    mkdir -p {Path(onnx_kwargs["f"]).stem}
    cd {Path(onnx_kwargs["f"]).stem}
    rsync -av valentina@berghain:{Path(onnx_kwargs["f"]).resolve()} .
    rsync -av {Path(onnx_kwargs["f"]).name} valentina@delta:/home/valentina
    rsync -av {Path(onnx_kwargs["f"]).name} nvidia@delta-emc1:/home/nvidia/onnx_models
    """)  # noqa: G004, LOG015

    with torch.inference_mode(), pytest.MonkeyPatch.context() as m:
        m.setattr("torch.compiler._is_exporting_flag", True)
        (prefix_embs, prefix_pad_masks, prefix_att_masks) = policy(batch)
        return prefix_embs, prefix_pad_masks, prefix_att_masks


def export_embedding_model_incremental(
    policy_vla: PreTrainedPolicy,
    args: tuple,
    dynamo_kwargs: dict,
    onnx_kwargs: dict,
    wandb_logger: WandBLogger,
) -> tuple:
    (
        batch,
        prefix_embs_cache,
        prefix_pad_masks_cache,
        prefix_att_masks_cache,
        lang_emb,
        lang_masks,
    ) = args
    context_length = onnx_kwargs["context_length"]
    onnx_kwargs.pop("context_length")
    num_tokens = onnx_kwargs["num_tokens"]
    onnx_kwargs.pop("num_tokens")
    policy = ExportEmbeddingModelIncremental(
        policy_vla, lang_emb, lang_masks, context_length, num_tokens
    )
    policy.eval()
    args_inc = (
        batch,
        prefix_embs_cache,
        prefix_pad_masks_cache,
        prefix_att_masks_cache,
    )
    exported_program = torch.export.export(
        mod=policy,
        args=(args_inc),
        **dynamo_kwargs,
    )
    _ = torch.onnx.export(
        model=exported_program,
        args=(args_inc),
        **onnx_kwargs,
    )
    wandb_logger.log_onnx(Path(onnx_kwargs["artifacts_dir"]))
    logging.info(f"""
    mkdir -p {Path(onnx_kwargs["f"]).stem}
    cd {Path(onnx_kwargs["f"]).stem}
    rsync -av valentina@berghain:{Path(onnx_kwargs["f"]).resolve()} .
    rsync -av {Path(onnx_kwargs["f"]).name} valentina@delta:/home/valentina
    rsync -av {Path(onnx_kwargs["f"]).name} nvidia@delta-emc1:/home/nvidia/onnx_models
    """)  # noqa: G004, LOG015

    with torch.inference_mode(), pytest.MonkeyPatch.context() as m:
        m.setattr("torch.compiler._is_exporting_flag", True)
        (prefix_embs, prefix_pad_masks, prefix_att_masks) = policy(*args_inc)
        return prefix_embs, prefix_pad_masks, prefix_att_masks


def export_action_model(
    policy_vla: PreTrainedPolicy,
    args: tuple,
    dynamo_kwargs,
    onnx_kwargs,
    wandb_logger: WandBLogger,
) -> tuple:
    # Overwrite the number of denoising steps
    policy_vla.config.num_steps = onnx_kwargs["num_steps"]
    onnx_kwargs.pop("num_steps")
    action_dim = onnx_kwargs["action_dim"]
    onnx_kwargs.pop("action_dim")
    policy = ExportActionModel(policy_vla, action_dim)
    policy.eval()
    # with torch.inference_mode(), pytest.MonkeyPatch.context() as m:  # noqa: SIM117
    #     m.setattr("torch.compiler._is_exporting_flag", True)  # noqa: ERA001
    #     result = policy(*args)  # noqa: ERA001
    exported_program = torch.export.export(mod=policy, args=(args), **dynamo_kwargs)
    _ = torch.onnx.export(
        model=exported_program,
        args=(args),
        **onnx_kwargs,
    )
    wandb_logger.log_onnx(Path(onnx_kwargs["artifacts_dir"]))
    logging.info(f"""
    mkdir -p {Path(onnx_kwargs["f"]).stem}
    cd {Path(onnx_kwargs["f"]).stem}
    rsync -av valentina@berghain:{Path(onnx_kwargs["f"]).resolve()} .
    rsync -av {Path(onnx_kwargs["f"]).name} valentina@delta:/home/valentina
    rsync -av {Path(onnx_kwargs["f"]).name} nvidia@delta-emc1:/home/nvidia/onnx_models
    """)  # noqa: G004, LOG015


def export_model_full(
    policy_vla: PreTrainedPolicy,
    args: tuple,
    dynamo_kwargs: dict,
    onnx_kwargs: dict,
    wandb_logger: WandBLogger,
) -> None:
    batch, lang_emb, lang_masks, noise = args

    # Overwrite the number of denoising steps
    policy_vla.config.num_steps = onnx_kwargs["num_steps"]
    onnx_kwargs.pop("num_steps")
    action_dim = onnx_kwargs["action_dim"]
    onnx_kwargs.pop("action_dim")

    policy = ExportModel(policy_vla, lang_emb, lang_masks, action_dim)
    policy.eval()
    args_full = (batch, noise)
    exported_program = torch.export.export(
        mod=policy, args=(args_full), **dynamo_kwargs
    )
    _ = torch.onnx.export(
        model=exported_program,
        args=(args_full),
        **onnx_kwargs,
    )
    wandb_logger.log_onnx(Path(onnx_kwargs["artifacts_dir"]))
    logging.info(f"""
    mkdir -p {Path(onnx_kwargs["f"]).stem}
    cd {Path(onnx_kwargs["f"]).stem}
    rsync -av valentina@berghain:{Path(onnx_kwargs["f"]).resolve()} .
    rsync -av {Path(onnx_kwargs["f"]).name} valentina@delta:/home/valentina
    rsync -av {Path(onnx_kwargs["f"]).name} nvidia@delta-emc1:/home/nvidia/onnx_models
    """)  # noqa: G004, LOG015

    with torch.inference_mode(), pytest.MonkeyPatch.context() as m:
        m.setattr("torch.compiler._is_exporting_flag", True)
        (prefix_embs, prefix_pad_masks, prefix_att_masks, _) = policy(*args_full)
        return prefix_embs, prefix_pad_masks, prefix_att_masks


def export_model_incremental(
    policy_vla: PreTrainedPolicy,
    args: tuple,
    dynamo_kwargs: dict,
    onnx_kwargs: dict,
    wandb_logger: WandBLogger,
) -> None:
    (
        batch,
        prefix_embs_cache,
        prefix_pad_masks_cache,
        prefix_att_masks_cache,
        lang_emb,
        lang_masks,
        noise,
    ) = args
    context_length = onnx_kwargs["context_length"]
    onnx_kwargs.pop("context_length")
    num_tokens = onnx_kwargs["num_tokens"]
    onnx_kwargs.pop("num_tokens")

    # Overwrite the number of denoising steps
    policy_vla.config.num_steps = onnx_kwargs["num_steps"]
    onnx_kwargs.pop("num_steps")
    action_dim = onnx_kwargs["action_dim"]
    onnx_kwargs.pop("action_dim")

    policy = ExportModelIncremental(
        policy_vla, lang_emb, lang_masks, context_length, num_tokens, action_dim
    )
    policy.eval()
    args_inc = (
        batch,
        prefix_embs_cache,
        prefix_pad_masks_cache,
        prefix_att_masks_cache,
        noise,
    )
    # with torch.inference_mode(), pytest.MonkeyPatch.context() as m:
    #     m.setattr("torch.compiler._is_exporting_flag", True)  # noqa: ERA001
    #     result = policy(*args_inc)  # noqa: ERA001
    exported_program = torch.export.export(mod=policy, args=(args_inc), **dynamo_kwargs)
    _ = torch.onnx.export(
        model=exported_program,
        args=(args_inc),
        **onnx_kwargs,
    )
    wandb_logger.log_onnx(Path(onnx_kwargs["artifacts_dir"]))
    logging.info(f"""
    mkdir -p {Path(onnx_kwargs["f"]).stem}
    cd {Path(onnx_kwargs["f"]).stem}
    rsync -av valentina@berghain:{Path(onnx_kwargs["f"]).resolve()} .
    rsync -av {Path(onnx_kwargs["f"]).name} valentina@delta:/home/valentina
    rsync -av {Path(onnx_kwargs["f"]).name} nvidia@delta-emc1:/home/nvidia/onnx_models
    """)  # noqa: G004, LOG015


def init_wandb(cfg: DictConfig, policy_cfg: TrainPipelineConfig) -> WandBLogger:
    # Monkey patching: use policy config to initialize wandb logging
    policy_cfg.job_name = cfg.job_name
    policy_cfg.output_dir = cfg.artifacts_dir
    return WandBLogger(policy_cfg)


def export_dynamo(cfg: DictConfig) -> None:  # noqa: PLR0914
    logging.debug("instantiating policy")  # noqa: LOG015

    # torch.backends.cudnn.benchmark = True
    # torch.backends.cuda.matmul.allow_tf32 = True

    dtype = torch.float16 if cfg.dtype == "torch.float16" else torch.float32
    policy_vla, policy_cfg = prepare_model_data(cfg, dtype)
    logging.info(f"instantiating the model of {dtype}")  # noqa: G004, LOG015

    wandb_logger = init_wandb(cfg, policy_cfg)

    # model specific kwargs
    embedding_full_kwargs = instantiate(cfg.embedding_full_kwargs)
    embedding_inc_kwargs = instantiate(cfg.embedding_inc_kwargs)
    action_kwargs = instantiate(cfg.action_kwargs)

    model_full_kwargs = instantiate(cfg.model_full_kwargs)
    model_inc_kwargs = instantiate(cfg.model_inc_kwargs)

    batch = build_episode(cfg, torch.device(cfg.device), dtype)

    normalization_kwargs = instantiate(cfg.normalization_kwargs)
    normalization_parameters = export_normalization_params(
        policy_vla, normalization_kwargs, batch
    )

    lang_emb, lang_masks, noise = update_episode_for_policy(
        embedding_full_kwargs,
        policy_vla,
        batch,
        torch.device(cfg.device),
        dtype,
        normalization_parameters,
    )

    prepare_data_for_accuracy_test(
        cfg,
        torch.device(cfg.device),
        dtype,
        policy_vla,
        lang_emb,
        lang_masks,
        normalization_parameters,
        noise,
    )

    args_embedding = (batch, lang_emb.clone(), lang_masks.clone())
    batch1 = {k: v[:, :1, :].clone() for k, v in batch.items()}
    batch_model = {k: v.clone() for k, v in batch.items()}

    dynamo_kwargs = instantiate(cfg.dynamo_kwargs)
    onnx_kwargs = instantiate(cfg.onnx_kwargs)

    logging.info("Exporting the embedding model")  # noqa: LOG015
    prefix_embs, prefix_pad_masks, prefix_att_masks = export_embedding_model_full(
        policy_vla,
        args_embedding,
        dynamo_kwargs,
        {**onnx_kwargs, **embedding_full_kwargs},
        wandb_logger,
    )
    logging.info("Exported the embedding model")  # noqa: LOG015

    # incremental embedding model
    args_embedding_inc = (
        batch1,
        prefix_embs.clone(),
        prefix_pad_masks.clone(),
        prefix_att_masks.clone(),
        lang_emb.clone(),
        lang_masks.clone(),
    )
    logging.info("Exporting the incremental embedding model")  # noqa: LOG015
    prefix_embs, prefix_pad_masks, prefix_att_masks = (
        export_embedding_model_incremental(
            policy_vla,
            args_embedding_inc,
            dynamo_kwargs,
            {**onnx_kwargs, **embedding_inc_kwargs},
            wandb_logger,
        )
    )
    logging.info("Exported the incremental embedding model")  # noqa: LOG015

    args_action = (
        prefix_embs.clone(),
        prefix_pad_masks.clone(),
        prefix_att_masks.clone(),
        noise.clone(),
    )

    export_action_model(
        policy_vla,
        args_action,
        dynamo_kwargs,
        {**onnx_kwargs, **action_kwargs},
        wandb_logger,
    )
    logging.info("Exported the action model")  # noqa: LOG015
    args_full = (batch_model, lang_emb.clone(), lang_masks.clone(), noise.clone())

    prefix_embs, prefix_pad_masks, prefix_att_masks = export_model_full(
        policy_vla,
        args_full,
        dynamo_kwargs,
        {**onnx_kwargs, **model_full_kwargs},
        wandb_logger,
    )
    logging.info("Exported the full model")  # noqa: LOG015

    args_inc = (
        batch1,
        prefix_embs.clone(),
        prefix_pad_masks.clone(),
        prefix_att_masks.clone(),
        lang_emb.clone(),
        lang_masks.clone(),
        noise.clone(),
    )
    export_model_incremental(
        policy_vla,
        args_inc,
        dynamo_kwargs,
        {**onnx_kwargs, **model_inc_kwargs},
        wandb_logger,
    )
    logging.info("Exported the incremental embedding model")  # noqa: LOG015


@hydra.main(version_base=None)
@torch.inference_mode()
def main(cfg: DictConfig) -> None:
    export_dynamo(cfg)


if __name__ == "__main__":
    init_logging()
    main()
