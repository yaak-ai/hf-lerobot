import getpass
import json
import logging
import math
import socket
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import hydra
import pytest
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.types import NormalizationMode  # noqa: F401
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.smolvla.conversion_utils_yaak import __getbatch__
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, resize_with_pad
from lerobot.policies.smolvla.modeling_smolvlm import (
    ExportSmolVLMVisionEmbeddings,
    ExportSmolVLMVisionTransformer,
)
from lerobot.policies.utils import get_device_from_parameters
from lerobot.processor.normalize_processor import NormalizerProcessorStep, patch_norm_mode
from lerobot.processor.pipeline import PolicyProcessorPipeline
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.utils.constants import (
    ACTION,
    OBS_IMAGE,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
)
from lerobot.utils.utils import init_logging

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


class ExportEmbeddingModelFull(torch.nn.Module):
    def __init__(
        self,
        policy: PreTrainedPolicy,
        lang_emb: torch.Tensor | None = None,
        lang_masks: torch.Tensor | None = None,
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

        vision_embeddings = policy.model.vlm_with_expert.vlm.model.vision_model.embeddings
        dtype = next(vision_embeddings.parameters()).dtype
        export_embeddings = ExportSmolVLMVisionEmbeddings(
            policy.model.vlm_with_expert.vlm.model.vision_model.config
        )
        export_embeddings.load_state_dict(vision_embeddings.state_dict(), strict=False)
        export_embeddings = export_embeddings.to(device).to(dtype)
        policy.model.vlm_with_expert.vlm.model.vision_model.embeddings = export_embeddings

        # None buffers are absent from state_dict() — from_local re-registers with real tensors.
        self.policy.register_buffer("lang_emb", lang_emb)
        self.policy.register_buffer("lang_masks", lang_masks)

    @classmethod
    def from_wandb(cls, artifact: str) -> "ExportEmbeddingModelFull":
        """Instantiate from a WandB .pt artifact with no other parameters."""
        import wandb  # noqa: PLC0415

        run = wandb.run
        artifact_obj = (
            run.use_artifact(artifact)
            if run is not None and not run.disabled
            else wandb.Api().artifact(artifact, type="model")
        )
        return cls.from_local(Path(artifact_obj.download()))

    @classmethod
    def from_local(cls, directory: Path) -> "ExportEmbeddingModelFull":
        """Instantiate from a local directory produced by an export run."""

        config = PreTrainedConfig.from_pretrained(directory)
        config.device = "cpu"  # override; caller can .to("cuda") after
        policy = SmolVLAPolicy(config)
        model = cls(policy)  # lang_emb=None, lang_masks=None — absent from state_dict()
        state_dict = torch.load(next(directory.glob("embedding_full*.pt")), map_location="cpu")
        lang_emb = state_dict.pop("policy.lang_emb")
        lang_masks = state_dict.pop("policy.lang_masks")
        model.load_state_dict(state_dict, assign=True)  # strict=True — both sides now lack lang keys
        # assign = True to get favorize the state_dict dtype
        model.policy.register_buffer("lang_emb", lang_emb)
        model.policy.register_buffer("lang_masks", lang_masks)
        return model

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
            if not self.policy.config.use_context
            else self.policy.prepare_state_wrapper(batch)
        )
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.policy.model.embed_prefix(
            images,
            img_masks,
            self.policy.lang_emb,
            self.policy.lang_masks,
            state=state,
        )
        return prefix_embs, prefix_pad_masks, prefix_att_masks


class ExportEmbeddingModelIncremental(torch.nn.Module):
    def __init__(
        self,
        policy: PreTrainedPolicy,
        lang_emb: torch.Tensor | None = None,
        lang_masks: torch.Tensor | None = None,
        context_length: int = 0,
        num_tokens: list | None = None,
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

        vision_embeddings = policy.model.vlm_with_expert.vlm.model.vision_model.embeddings
        dtype = next(vision_embeddings.parameters()).dtype
        export_embeddings = ExportSmolVLMVisionEmbeddings(
            policy.model.vlm_with_expert.vlm.model.vision_model.config
        )
        export_embeddings.load_state_dict(vision_embeddings.state_dict())
        export_embeddings = export_embeddings.to(device).to(dtype)
        policy.model.vlm_with_expert.vlm.model.vision_model.embeddings = export_embeddings

        # None buffers are absent from state_dict() — from_local re-registers with real tensors.
        self.policy.register_buffer("lang_emb", lang_emb)
        self.policy.register_buffer("lang_masks", lang_masks)

    @classmethod
    def from_wandb(cls, artifact: str) -> "ExportEmbeddingModelIncremental":
        """Instantiate from a WandB .pt artifact with no other parameters."""
        import wandb  # noqa: PLC0415

        run = wandb.run
        artifact_obj = (
            run.use_artifact(artifact)
            if run is not None and not run.disabled
            else wandb.Api().artifact(artifact, type="model")
        )
        return cls.from_local(Path(artifact_obj.download()))

    @classmethod
    def from_local(cls, directory: Path) -> "ExportEmbeddingModelIncremental":
        """Instantiate from a local directory produced by an export run."""

        config = PreTrainedConfig.from_pretrained(directory)
        config.device = "cpu"  # override; caller can .to("cuda") after
        policy = SmolVLAPolicy(config)
        export_cfg = json.loads((directory / "export_config.json").read_text())
        model = cls(policy, context_length=export_cfg["context_length"], num_tokens=export_cfg["num_tokens"])
        state_dict = torch.load(next(directory.glob("embedding_inc*.pt")), map_location="cpu")
        lang_emb = state_dict.pop("policy.lang_emb")
        lang_masks = state_dict.pop("policy.lang_masks")
        model.load_state_dict(state_dict, assign=True)  # strict=True — both sides now lack lang keys
        model.policy.register_buffer("lang_emb", lang_emb)
        model.policy.register_buffer("lang_masks", lang_masks)
        return model

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
            if not self.policy.config.use_context
            else self.policy.prepare_state_wrapper(batch)
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
                    + self.num_tokens[3] : self.num_tokens[0] * self.context_length  # images
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

    @classmethod
    def from_wandb(cls, artifact: str) -> "ExportActionModel":
        """Instantiate from a WandB .pt artifact with no other parameters."""
        import wandb  # noqa: PLC0415

        run = wandb.run
        artifact_obj = (
            run.use_artifact(artifact)
            if run is not None and not run.disabled
            else wandb.Api().artifact(artifact, type="model")
        )
        return cls.from_local(Path(artifact_obj.download()))

    @classmethod
    def from_local(cls, directory: Path) -> "ExportActionModel":
        """Instantiate from a local directory produced by an export run."""

        config = PreTrainedConfig.from_pretrained(directory)
        config.device = "cpu"  # override; caller can .to("cuda") after
        policy = SmolVLAPolicy(config)
        export_cfg = json.loads((directory / "export_config.json").read_text())
        model = cls(policy, action_dim=export_cfg["action_dim"])
        state_dict = torch.load(next(directory.glob("action*.pt")), map_location="cpu")
        model.load_state_dict(state_dict, assign=True)  # assign=True to get favorize the state_dict dtype
        return model


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


def _normalize_min_max(buffer: nn.ParameterDict, input_tensor: torch.Tensor) -> torch.Tensor:
    # normalize to [0,1]
    input_tensor = (input_tensor - buffer["min"]) / (buffer["max"] - buffer["min"] + 1e-8)
    # normalize to [-1, 1]
    input_tensor = input_tensor * 2 - 1
    return torch.clamp(input_tensor, -1, 1)


def _normalize_zero_one(buffer: nn.ParameterDict, input_tensor: torch.Tensor) -> torch.Tensor:
    # normalize to [0,1]
    return (input_tensor - buffer["min"]) / (buffer["max"] - buffer["min"] + 1e-8)


def torch_constants_to_trt(torch_constant: str) -> str:
    return "batch_" + torch_constant.replace(".", "_")


def _normalize_state(normalization_parameters: dict, batch: dict) -> None:
    for k in batch.keys():
        trt_key = torch_constants_to_trt(k)
        if trt_key not in normalization_parameters:
            continue
        norm_mode = normalization_parameters[trt_key]["norm_mode"]

        if norm_mode == str(NormalizationMode.MIN_MAX):
            batch[k] = _normalize_min_max(
                normalization_parameters[trt_key]["buffer"],
                batch[k],
            )
        if norm_mode == str(NormalizationMode.ZERO_ONE):
            batch[k] = _normalize_zero_one(
                normalization_parameters[trt_key]["buffer"],
                batch[k],
            )
            continue


def update_episode_for_policy(
    embedding_kwargs: DictConfig,
    policy_vla: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    batch: dict,
    device: torch.device,
    dtype: torch.dtype,
    normalization_parameters: dict,
    keys_to_keep: list[str],
) -> tuple:
    """
    1. Update batch and other model inputs based on policy config and embedding_kwargs
    2. Update policy config based on embedding_kwargs
    """  # noqa: DOC201
    batch = preprocessor(batch)

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

    # Normalize state done via preprocessor(batch)
    # _normalize_state(normalization_parameters, batch)

    # Language
    # keep the tokenization outside the ONNX model since it produces errors
    # also keep it outside of episode construction since it is policy-dependent
    lang_tokens = batch[OBS_LANGUAGE_TOKENS]
    lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
    lang_emb = policy_vla.model.vlm_with_expert.embed_language_tokens(lang_tokens)
    # Normalize language embeddings
    lang_emb_dim = lang_emb.shape[-1]
    lang_emb *= math.sqrt(lang_emb_dim)
    # after tokenization, task text is no longer needed in the batch
    for k in list(batch.keys()):
        if k not in keys_to_keep:
            batch.pop(k)
    # Noise
    bsize = batch[OBS_IMAGE].shape[0]
    noise = torch.normal(
        mean=0.0,
        std=1.0,
        size=(bsize, policy_vla.config.chunk_size, policy_vla.config.max_action_dim),
        dtype=dtype,
        device=device,
    )
    return lang_emb, lang_masks, noise, batch


def prepare_model_data(cfg: DictConfig, dtype: torch.dtype) -> None:
    logging.debug("instantiating policy")  # noqa: LOG015
    with torch.inference_mode(), pytest.MonkeyPatch.context() as m:
        m.setattr("torch.compiler._is_exporting_flag", True)
        policy, train_cfg, preprocessor, postprocessor = instantiate(cfg.model)

    policy = policy.to(dtype).to(cfg.device)
    policy.eval()
    return policy, train_cfg, preprocessor, postprocessor


def export_normalization_params(
    preprocessor: PolicyProcessorPipeline,
    normalization_kwargs: dict,
    device: torch.device,
    dtype: torch.dtype,
    wandb_logger: WandBLogger,
) -> dict:
    """Build normalization parameter buffers from dataset statistics JSON.

    Replaces the earlier approach of reading buffers from policy.normalize_inputs
    (which no longer exists on v0.4.4).  The stats JSON path is supplied via
    normalization_kwargs["stats_path"].

    Norm modes are hardcoded to match the validation in _normalize_state:
      - OBS_STATE_VEHICLE  → ZERO_ONE
      - OBS_STATE          → MIN_MAX
    """

    normalization_step = [step for step in preprocessor.steps if isinstance(step, NormalizerProcessorStep)]
    if len(normalization_step) == 0:
        raise ValueError("No NormalizerProcessorStep found in preprocessor pipeline.")
    elif len(normalization_step) > 1:
        raise ValueError("Multiple NormalizerProcessorStep found in preprocessor pipeline.")
    normalization_step = normalization_step[0]

    normalization_parameters = {}
    for key, feature in normalization_step.features.items():
        norm_mode = normalization_step.norm_map.get(feature.type, NormalizationMode.IDENTITY)
        norm_mode = patch_norm_mode(norm_mode, key)
        if norm_mode == NormalizationMode.IDENTITY:
            continue
        key_stats = normalization_step.stats[key]
        normalization_parameters[torch_constants_to_trt(key)] = {
            "buffer": {
                "min": torch.tensor(key_stats["min"], dtype=dtype, device=device),
                "max": torch.tensor(key_stats["max"], dtype=dtype, device=device),
            },
            "norm_mode": str(norm_mode),
        }

    norm_path = Path(normalization_kwargs["f"])
    norm_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(normalization_parameters, norm_path)
    wandb_logger.log_pt(norm_path.parent, "normalization")

    _user = getpass.getuser()
    _host = socket.gethostname()
    logging.info(f"""
    mkdir -p {Path(norm_path).stem}
    cd {norm_path.stem}
    rsync -av {_user}@{_host}:{norm_path.resolve()} .
    rsync -av {norm_path.name} {_user}@delta:/home/{_user}
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
    # Log PyTorch models to WandB
    pt_path = Path(onnx_kwargs["artifacts_dir"]) / "embedding_full.pt"
    pt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), pt_path)
    policy.policy.config._save_pretrained(Path(onnx_kwargs["artifacts_dir"]))
    artifact_ref = wandb_logger.log_pt(Path(onnx_kwargs["artifacts_dir"]), "embedding_full")
    if artifact_ref is not None:
        logging.info(f"ExportEmbeddingModelFull artifact: {artifact_ref}")  # noqa: G004, LOG015

    exported_program = torch.export.export(mod=policy, args=(batch,), **dynamo_kwargs)
    _ = torch.onnx.export(
        model=exported_program,
        args=(batch,),
        **onnx_kwargs,
    )
    onnx_refs = wandb_logger.log_onnx(Path(onnx_kwargs["artifacts_dir"]))
    for kind, ref in onnx_refs.items():
        logging.info(f"ExportEmbeddingModelFull {kind} artifact: {ref}")  # noqa: G004, LOG015
    _user = getpass.getuser()
    _host = socket.gethostname()
    logging.info(f"""
    mkdir -p {Path(onnx_kwargs["f"]).stem}
    cd {Path(onnx_kwargs["f"]).stem}
    rsync -av {_user}@{_host}:{Path(onnx_kwargs["f"]).resolve()} .
    rsync -av {Path(onnx_kwargs["f"]).name} {_user}@delta:/home/{_user}
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
    policy = ExportEmbeddingModelIncremental(policy_vla, lang_emb, lang_masks, context_length, num_tokens)
    policy.eval()
    args_inc = (
        batch,
        prefix_embs_cache,
        prefix_pad_masks_cache,
        prefix_att_masks_cache,
    )
    pt_path = Path(onnx_kwargs["artifacts_dir"]) / "embedding_inc.pt"
    pt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), pt_path)
    policy.policy.config._save_pretrained(Path(onnx_kwargs["artifacts_dir"]))
    (Path(onnx_kwargs["artifacts_dir"]) / "export_config.json").write_text(
        json.dumps({"context_length": context_length, "num_tokens": num_tokens})
    )
    artifact_ref = wandb_logger.log_pt(Path(onnx_kwargs["artifacts_dir"]), "embedding_inc")
    if artifact_ref is not None:
        logging.info(f"ExportEmbeddingModelIncremental artifact: {artifact_ref}")  # noqa: G004, LOG015
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
    onnx_refs = wandb_logger.log_onnx(Path(onnx_kwargs["artifacts_dir"]))
    for kind, ref in onnx_refs.items():
        logging.info(f"ExportEmbeddingModelIncremental {kind} artifact: {ref}")  # noqa: G004, LOG015
    _user = getpass.getuser()
    _host = socket.gethostname()
    logging.info(f"""
    mkdir -p {Path(onnx_kwargs["f"]).stem}
    cd {Path(onnx_kwargs["f"]).stem}
    rsync -av {_user}@{_host}:{Path(onnx_kwargs["f"]).resolve()} .
    rsync -av {Path(onnx_kwargs["f"]).name} {_user}@delta:/home/{_user}
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
    #  Log PyTorch to WandB
    pt_path = Path(onnx_kwargs["artifacts_dir"]) / "action.pt"
    pt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), pt_path)
    policy.policy.config._save_pretrained(Path(onnx_kwargs["artifacts_dir"]))
    (Path(onnx_kwargs["artifacts_dir"]) / "export_config.json").write_text(
        json.dumps({"action_dim": action_dim})
    )
    artifact_ref = wandb_logger.log_pt(Path(onnx_kwargs["artifacts_dir"]), "action")
    if artifact_ref is not None:
        logging.info(f"ExportActionModel artifact: {artifact_ref}")  # noqa: G004, LOG015
    exported_program = torch.export.export(mod=policy, args=(args), **dynamo_kwargs)
    _ = torch.onnx.export(
        model=exported_program,
        args=(args),
        **onnx_kwargs,
    )
    onnx_refs = wandb_logger.log_onnx(Path(onnx_kwargs["artifacts_dir"]))
    for kind, ref in onnx_refs.items():
        logging.info(f"ExportActionModel {kind} artifact: {ref}")  # noqa: G004, LOG015
    _user = getpass.getuser()
    _host = socket.gethostname()
    logging.info(f"""
    mkdir -p {Path(onnx_kwargs["f"]).stem}
    cd {Path(onnx_kwargs["f"]).stem}
    rsync -av {_user}@{_host}:{Path(onnx_kwargs["f"]).resolve()} .
    rsync -av {Path(onnx_kwargs["f"]).name} {_user}@delta:/home/{_user}
    rsync -av {Path(onnx_kwargs["f"]).name} nvidia@delta-emc1:/home/nvidia/onnx_models
    """)  # noqa: G004, LOG015


def load_from_wandb_onnx(artifact: str) -> Path:
    """Download an ONNX artifact from WandB and return the local path.

    Args:
        artifact: WandB artifact path, e.g. "entity/project/name:version".

    Returns:
        Path to the downloaded .onnx file.
    """
    import wandb  # noqa: PLC0415

    run = wandb.run
    artifact_obj = (
        run.use_artifact(artifact)
        if run is not None and not run.disabled
        else wandb.Api().artifact(artifact, type="model")
    )
    artifact_dir = Path(artifact_obj.download())
    onnx_files = list(artifact_dir.glob("*.onnx"))
    if not onnx_files:
        raise FileNotFoundError(f"No .onnx file found in downloaded artifact at {artifact_dir}")
    return onnx_files[0]


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
    device = torch.device(cfg.device)
    policy_vla, policy_cfg, preprocessor, postprocessor = prepare_model_data(cfg, dtype)
    logging.info(f"instantiating the model of {dtype}")  # noqa: G004, LOG015

    wandb_logger = init_wandb(cfg, policy_cfg)

    # model specific kwargs
    embedding_full_kwargs = instantiate(cfg.embedding_full_kwargs)
    embedding_inc_kwargs = instantiate(cfg.embedding_inc_kwargs)
    action_kwargs = instantiate(cfg.action_kwargs)

    batch = build_episode(cfg, device, dtype)

    normalization_kwargs = instantiate(cfg.normalization_kwargs)
    # Load normalization parameters from dataset stats (no longer from policy.normalize_inputs)
    normalization_parameters = export_normalization_params(
        preprocessor, normalization_kwargs, device, dtype, wandb_logger
    )

    # skip all the batch keys like REWARD that proprocessor might add
    export_batch_keys = instantiate(cfg.export_batch_keys)

    lang_emb, lang_masks, noise, batch = update_episode_for_policy(
        embedding_full_kwargs,
        policy_vla,
        preprocessor,
        batch,
        device,
        dtype,
        normalization_parameters,
        export_batch_keys,
    )

    args_embedding = (batch, lang_emb.clone(), lang_masks.clone())
    batch1 = {k: v[:, :1, :].clone() for k, v in batch.items()}
    batch_model = {k: v.clone() for k, v in batch.items()}

    dynamo_kwargs = instantiate(cfg.dynamo_kwargs)
    onnx_kwargs = instantiate(cfg.onnx_kwargs)

    logging.info("Exporting the embedding model")  # noqa: LOG015
    prefix_embs, prefix_pad_masks, prefix_att_masks = export_embedding_model_full(
        deepcopy(policy_vla),  # Embedding model registers buffers
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
    prefix_embs, prefix_pad_masks, prefix_att_masks = export_embedding_model_incremental(
        deepcopy(policy_vla),  # Embedding model registers buffers
        args_embedding_inc,
        dynamo_kwargs,
        {**onnx_kwargs, **embedding_inc_kwargs},
        wandb_logger,
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


@hydra.main(version_base=None)
@torch.inference_mode()
def main(cfg: DictConfig) -> None:
    export_dynamo(cfg)


if __name__ == "__main__":
    init_logging()
    main()
