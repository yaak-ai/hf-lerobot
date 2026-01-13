import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
import pytest
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.smolvla.conversion_utils_yaak import __getbatch__
from lerobot.policies.smolvla.modeling_smolvla import resize_with_pad
from lerobot.policies.smolvla.modeling_smolvlm import (
    ExportSmolVLMVisionEmbeddings,
    ExportSmolVLMVisionTransformer,
)
from lerobot.policies.utils import get_device_from_parameters
from lerobot.utils.utils import init_logging
from lerobot.utils.wandb_utils import WandBLogger

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


class ExportEmbeddingModel(torch.nn.Module):
    def __init__(self, policy: PreTrainedPolicy, lang_emb, lang_masks):
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

    def forward(self, batch: dict):
        batch = self.policy._prepare_batch(batch)  # noqa: SLF001
        images, img_masks = (
            self.policy.prepare_images(batch)
            if not self.policy.use_context
            else self.policy.prepare_images_context(batch)
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
    batch.pop("action.continuous", None)
    return batch


def update_episode_for_policy(
    embedding_kwargs: DictConfig,
    policy_vla: PreTrainedPolicy,
    batch: dict,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple:
    """
    1. Update batch and other model inputs based on policy config and embedding_kwargs
    2. Update policy config based on embedding_kwargs
    """  # noqa: DOC201
    # Images
    im_key = "observation.images.front_left"
    if embedding_kwargs["resize_in_episode_construction"]:
        batch[im_key] = resize_with_pad(
            torch.reshape(batch[im_key], (-1, *batch[im_key].shape[-3:])),
            *policy_vla.config.resize_imgs_with_padding,
            pad_value=0,
        ).reshape((
            *batch[im_key].shape[:-2],
            *policy_vla.config.resize_imgs_with_padding,
        ))
        # Do not resize in model, it's resized here
        policy_vla.config.resize_imgs_with_padding = None
        # Siglip normalization
        batch[im_key] *= 2.0
        batch[im_key] -= 1.0
    # Remove resize_in_episode_construction because it's not supported by torch.export
    embedding_kwargs.pop("resize_in_episode_construction")

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
    bsize = batch[im_key].shape[0]
    noise = torch.normal(
        mean=0.0,
        std=1.0,
        size=(bsize, policy_vla.config.chunk_size, policy_vla.config.max_action_dim),
        dtype=dtype,
        device=device,
    )
    return lang_emb, lang_masks, noise


def prepare_model_data(cfg: DictConfig, dtype: torch.dtype) -> None:
    logging.debug("instantiating policy")  # noqa: LOG015
    with torch.inference_mode(), pytest.MonkeyPatch.context() as m:
        m.setattr("torch.compiler._is_exporting_flag", True)
        policy, train_cfg = instantiate(cfg.model)

    policy = policy.to(dtype).to(cfg.device)
    policy.eval()
    return policy, train_cfg


def export_action(policy_vla, args, dynamo_kwargs, onnx_kwargs):
    pass


def export_embedding(policy_vla, args, dynamo_kwargs, onnx_kwargs, wandb_logger):
    batch, lang_emb, lang_masks = args
    policy = ExportEmbeddingModel(policy_vla, lang_emb, lang_masks)
    policy.eval()
    # with torch.inference_mode(), pytest.MonkeyPatch.context() as m:  # noqa: SIM117
    #     m.setattr("torch.compiler._is_exporting_flag", True)  # noqa: ERA001
    #     result = policy(batch)  # noqa: ERA001

    exported_program = torch.export.export(mod=policy, args=(batch,), **dynamo_kwargs)
    _ = torch.onnx.export(
        model=exported_program,
        args=(batch,),
        **onnx_kwargs,
    )
    wandb_logger.log_onnx(Path(onnx_kwargs["artifacts_dir"]))
    print(f"mkdir -p {Path(onnx_kwargs['f']).stem}")  # noqa: T201
    print(f"cd {Path(onnx_kwargs['f']).stem}")  # noqa: T201
    print(f"rsync -av valentina@berghain:{Path(onnx_kwargs['f']).resolve()} .")  # noqa: T201
    print(f"rsync -av {Path(onnx_kwargs['f']).name} valentina@delta:/home/valentina")  # noqa: T201


def init_wandb(cfg: DictConfig, policy_cfg: TrainPipelineConfig):
    # Monkey patching: use policy config to initialize wandb logging
    policy_cfg.job_name = cfg.job_name
    policy_cfg.output_dir = cfg.artifacts_dir
    return WandBLogger(policy_cfg)


def export_dynamo(cfg: DictConfig) -> None:
    logging.debug("instantiating policy")  # noqa: LOG015
    dtype = torch.float16 if cfg.dtype == "torch.float16" else torch.float32
    policy_vla, policy_cfg = prepare_model_data(cfg, dtype)
    logging.info(f"instantiating the model of {dtype}")  # noqa: G004, LOG015

    wandb_logger = init_wandb(cfg, policy_cfg)

    # model specific kwargs
    embedding_kwargs = instantiate(cfg.embedding_kwargs)
    lm_expert_kwargs = instantiate(cfg.lm_expert_kwargs)
    shape_kwargs = instantiate(cfg.shape_kwargs)

    batch = build_episode(cfg, torch.device(cfg.device), dtype)
    lang_emb, lang_masks, noise = update_episode_for_policy(
        embedding_kwargs, policy_vla, batch, torch.device(cfg.device), dtype
    )
    args_embedding = (batch, lang_emb, lang_masks)

    dynamo_kwargs = instantiate(cfg.dynamo_kwargs)
    onnx_kwargs = instantiate(cfg.onnx_kwargs)

    export_embedding(
        policy_vla,
        args_embedding,
        # dynamo_kwargs,
        {**dynamo_kwargs, **shape_kwargs},
        {**onnx_kwargs, **embedding_kwargs},
        wandb_logger,
    )

    logging.info(f"exported to {cfg['artifacts_dir']}")  # noqa: G004, LOG015


@hydra.main(version_base=None)
@torch.inference_mode()
def main(cfg: DictConfig) -> None:
    export_dynamo(cfg)


if __name__ == "__main__":
    init_logging()
    main()
