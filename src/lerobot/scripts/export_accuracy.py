import datetime
import logging
from pathlib import Path

import hydra
import pytest
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from PIL import Image
from tqdm import tqdm

from lerobot.constants_yaak import ACTION, OBS_IMAGE
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.smolvla.conversion_utils_yaak import __getbatch__
from lerobot.policies.smolvla.modeling_smolvla import resize_with_pad
from lerobot.scripts.export_onnx import (
    ExportActionModel,
    ExportEmbeddingModelFull,
    ExportEmbeddingModelIncremental,
    _normalize_state,
    build_episode,
    export_normalization_params,
    prepare_model_data,
    update_episode_for_policy,
)
from lerobot.utils.reye_utils import create_reye_df
from lerobot.utils.utils import init_logging


def _intantiate_prod_models(policy_vla, lang_emb, lang_masks, cfg):
    policy_emb = ExportEmbeddingModelFull(policy_vla, lang_emb, lang_masks)
    policy_emb.eval()
    policy_vla.config.num_steps = cfg.action_kwargs["num_steps"]
    action_dim = cfg.action_kwargs["action_dim"]
    policy_action = ExportActionModel(policy_vla, action_dim)
    policy_action.eval()

    context_length = cfg.embedding_inc_kwargs["context_length"]
    num_tokens = cfg.embedding_inc_kwargs["num_tokens"]
    policy_emb_inc = ExportEmbeddingModelIncremental(
        policy_vla, lang_emb, lang_masks, context_length, num_tokens
    )
    policy_emb_inc.eval()

    return policy_emb, policy_action, policy_emb_inc


def _eval_prod_models(policy_emb, policy_action, policy_emb_inc, batch, noise):
    with torch.inference_mode(), pytest.MonkeyPatch.context() as m:
        m.setattr("torch.compiler._is_exporting_flag", True)
        prefix_embs, prefix_pad_masks, prefix_att_masks = policy_emb(batch)
        actions = policy_action(
            prefix_embs, prefix_pad_masks, prefix_att_masks, noise.clone()
        )
        return actions[0, 0, ...]


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
    policy_emb, policy_action, policy_emb_inc = _intantiate_prod_models(
        policy_vla, lang_emb, lang_masks, cfg
    )
    w, h = 512, 512
    real_actions = torch.zeros(
        (len(dataloader_test), 3), dtype=torch.float32, device=device
    )
    images = []
    for step, elem in tqdm(enumerate(dataloader_test)):
        images.append(
            Image.fromarray(elem.data["cam_front_left"].cpu().numpy()[0, 0, ...])
        )
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
        real_actions[step, ...] = _eval_prod_models(
            policy_emb, policy_action, policy_emb_inc, batch, noise.clone()
        )

    drive = dataloader_test.dataset.samples["__input_id"][0]
    delta_dir = Path(f"tmp/debug_steering/gt_angles/{drive}")
    delta_dir.mkdir(parents=True, exist_ok=True)

    # reye serialization
    # Handle cases with clip and without clip based on the timestamp
    ts = dataloader_test.dataset.samples[
        "meta/ImageMetadata.cam_front_left/time_stamp"
    ][0]
    df = create_reye_df(
        dataloader_test,
        real_actions.cpu(),
        is_without_clip=isinstance(ts, datetime.datetime),
    )
    reye_torch = Path(cfg.reye_torch)
    reye_torch.mkdir(parents=True, exist_ok=True)
    reye_path = reye_torch / "results.parquet"
    df.write_parquet(reye_path)

    samples_path = reye_torch / "samples"
    samples_path.mkdir(parents=True, exist_ok=True)
    delta_input = samples_path / "samples.parquet"
    dataloader_test.dataset.samples.write_parquet(delta_input)

    logging.info(f"""
    cd delta_accuracy
    rsync -av valentina@berghain:{reye_torch} .
    rsync -av {reye_path.name} valentina@delta:/home/valentina/data/
    rsync -av samples/{delta_input.name} valentina@delta:/home/valentina/data/
    rsync -av {reye_path.name} nvidia@delta-emc1:/home/nvidia/accuracy/
    rsync -av samples/{delta_input.name} nvidia@delta-emc1:/home/nvidia/accuracy/
    """)  # noqa: G004, LOG015


def export_accuracy(cfg: DictConfig) -> None:
    dtype = torch.float16 if cfg.dtype == "torch.float16" else torch.float32
    policy_vla, policy_cfg = prepare_model_data(cfg, dtype)
    batch = build_episode(cfg, torch.device(cfg.device), dtype)

    # model specific kwargs
    embedding_full_kwargs = instantiate(cfg.embedding_full_kwargs)

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


@hydra.main(version_base=None)
@torch.inference_mode()
def main(cfg: DictConfig) -> None:
    export_accuracy(cfg)


if __name__ == "__main__":
    init_logging()
    main()
