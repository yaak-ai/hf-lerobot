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
from lerobot.constants_yaak import ACTION, OBS_IMAGE
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
                torch.clamp(actions[:, :, :-1], min=0, max=1),
                torch.clamp(actions[:, :, -1:], min=-1, max=1),
            ],
            dim=-1,
        )


class ExportEmbeddingModel(torch.nn.Module):
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
        batch = self.policy._prepare_batch(batch)  # noqa: SLF001
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


class ExportDualModel(torch.nn.Module):
    def __init__(
        self,
        policy: PreTrainedPolicy,
        lang_emb: torch.Tensor,
        lang_masks: torch.Tensor,
        action_dim: int,
    ) -> None:
        super().__init__()
        self.embedding_model = ExportEmbeddingModel(policy, lang_emb, lang_masks)
        self.action_model = ExportActionModel(policy, action_dim)

    def forward(self, batch: dict, noise: torch.Tensor) -> torch.Tensor:
        (prefix_embs, prefix_pad_masks, prefix_att_masks) = self.embedding_model(batch)
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
    noise: torch.Tensor,
) -> None:
    dataloader_test: DataLoader = instantiate(cfg.datamodule)
    w, h = 512, 512
    policy_dual = ExportDualModel(policy_vla, lang_emb, lang_masks, 3)
    policy_emb = ExportEmbeddingModel(policy_vla, lang_emb, lang_masks)
    noise_cpu = noise.clone().cpu()
    collected_data = []
    for _, elem in enumerate(dataloader_test):  # noqa: FURB148
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
        # task = batch.pop("task")  # noqa: ERA001
        with torch.inference_mode(), pytest.MonkeyPatch.context() as m:
            m.setattr("torch.compiler._is_exporting_flag", True)
            # result = policy_dual(batch, noise.clone())  # noqa: ERA001
            prefix_embs, prefix_pad_masks, prefix_att_masks = policy_emb(batch)

        # Collect batch and noise, moving to CPU for portability
        batch_cpu = {
            k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in batch.items()
        }
        collected_data.append({
            "batch": batch_cpu,
            "prefix_embs": prefix_embs.cpu(),
            "prefix_pad_masks": prefix_pad_masks.cpu(),
            "prefix_att_masks": prefix_att_masks.cpu()
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

    logging.info(f"""
    cd delta_accuracy
    rsync -av valentina@berghain:{output_path.resolve()} .
    rsync -av {output_path.name} valentina@delta:/home/valentina/data
    """)  # noqa: G004, LOG015


def prepare_model_data(cfg: DictConfig, dtype: torch.dtype) -> None:
    logging.debug("instantiating policy")  # noqa: LOG015
    with torch.inference_mode(), pytest.MonkeyPatch.context() as m:
        m.setattr("torch.compiler._is_exporting_flag", True)
        policy, train_cfg = instantiate(cfg.model)

    policy = policy.to(dtype).to(cfg.device)
    policy.eval()
    return policy, train_cfg


def export_embedding_model(
    policy_vla: PreTrainedPolicy,
    args: tuple,
    dynamo_kwargs,
    onnx_kwargs,
    wandb_logger: WandBLogger,
) -> tuple:
    batch, lang_emb, lang_masks = args
    policy = ExportEmbeddingModel(policy_vla, lang_emb, lang_masks)
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
    """)  # noqa: G004, LOG015

    with torch.inference_mode(), pytest.MonkeyPatch.context() as m:
        m.setattr("torch.compiler._is_exporting_flag", True)
        (prefix_embs, prefix_pad_masks, prefix_att_masks) = policy(batch)
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
    embedding_kwargs = instantiate(cfg.embedding_kwargs)
    action_kwargs = instantiate(cfg.action_kwargs)
    shape_kwargs = instantiate(cfg.shape_kwargs)

    batch = build_episode(cfg, torch.device(cfg.device), dtype)
    lang_emb, lang_masks, noise = update_episode_for_policy(
        embedding_kwargs, policy_vla, batch, torch.device(cfg.device), dtype
    )

    prepare_data_for_accuracy_test(
        cfg, torch.device(cfg.device), dtype, policy_vla, lang_emb, lang_masks, noise
    )

    args_embedding = (batch, lang_emb, lang_masks)

    dynamo_kwargs = instantiate(cfg.dynamo_kwargs)
    onnx_kwargs = instantiate(cfg.onnx_kwargs)

    logging.info("Exporting the embedding model")  # noqa: LOG015
    prefix_embs, prefix_pad_masks, prefix_att_masks = export_embedding_model(
        policy_vla,
        args_embedding,
        {**dynamo_kwargs, **shape_kwargs},
        {**onnx_kwargs, **embedding_kwargs},
        wandb_logger,
    )
    logging.info("Exported the embedding model")  # noqa: LOG015

    args_action = (prefix_embs, prefix_pad_masks, prefix_att_masks, noise)
    logging.info("Exported the action model")  # noqa: LOG015
    export_action_model(
        policy_vla,
        args_action,
        dynamo_kwargs,
        {**onnx_kwargs, **action_kwargs},
        wandb_logger,
    )
    logging.info("Exported the action model")  # noqa: LOG015


@hydra.main(version_base=None)
@torch.inference_mode()
def main(cfg: DictConfig) -> None:
    export_dynamo(cfg)


if __name__ == "__main__":
    init_logging()
    main()
