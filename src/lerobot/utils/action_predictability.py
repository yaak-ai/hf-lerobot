import collections
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from omegaconf import DictConfig
from safetensors.torch import load_model as load_model_as_safetensor
from safetensors.torch import save_model as save_model_as_safetensor
from torch.utils.data import DataLoader, TensorDataset

from lerobot.constants import CHECKPOINTS_DIR, PRETRAINED_MODEL_DIR
from lerobot.policies.utils import get_device_from_parameters
from lerobot.utils.hub import HubMixin
from lerobot.utils.reye_utils import (
    create_reye_df_from_dataset,
    create_reye_df_from_reye,
)
from lerobot.utils.train_utils import save_training_state
from lerobot.utils.wandb_utils import WandBLogger
from lerobot.utils.wandb_utils_yaak import action_callback


class MLP(nn.Module, HubMixin):
    def __init__(self, input_dim, hidden_dim, output_dim, nr_hidden_layers):
        super().__init__()
        layers = collections.OrderedDict()
        layers["input"] = nn.Linear(input_dim, hidden_dim)
        layers["relu_input"] = nn.ReLU()
        for i in range(nr_hidden_layers):
            layers[f"hidden_{i}"] = nn.Linear(hidden_dim, hidden_dim)
            layers[f"relu_{i}"] = nn.ReLU()
        layers["output"] = nn.Linear(hidden_dim, output_dim)
        self.net = nn.Sequential(layers)

    def forward(self, x):
        return self.net(x)

    def _save_pretrained(self, save_directory: Path) -> None:
        model_to_save = self.module if hasattr(self, "module") else self
        save_model_as_safetensor(
            model_to_save, str(save_directory / SAFETENSORS_SINGLE_FILE)
        )

    def from_pretrained(
        self,
        checkpoint_dir: str | Path,
    ):
        pretrained_model_name_or_path = (
            checkpoint_dir / PRETRAINED_MODEL_DIR / SAFETENSORS_SINGLE_FILE
        )
        load_model_as_safetensor(self, pretrained_model_name_or_path, device="cpu")


def __getdataloader__(
    dataset: pl.DataFrame, batch_size=2048, input_dim=18, do_shuffle=True
) -> DataLoader:
    src_dim = (dataset["meta/VehicleMotion/gas_pedal_history"].list.len() * 3).unique()[
        0
    ] // 3
    target_dim = input_dim // 3
    stride = src_dim - target_dim
    x = (
        torch.from_numpy(
            np.stack(
                (
                    np.stack(dataset["meta/VehicleMotion/gas_pedal_history"]),
                    np.stack(dataset["meta/VehicleMotion/brake_pedal_history"]),
                    np.stack(dataset["meta/VehicleMotion/steering_angle_history"]),
                ),
                axis=-1,
            )[:, stride:, :]
        )
        .to(dtype=torch.float32)
        .reshape(-1, input_dim)
    )  # Reshape to (N*(S-1), 3)

    y = torch.from_numpy(
        np.stack(
            (
                np.stack(dataset["meta/VehicleMotion/gas_pedal_normalized"]),
                np.stack(dataset["meta/VehicleMotion/brake_pedal_normalized"]),
                np.stack(dataset["meta/VehicleMotion/steering_angle_normalized"]),
            ),
            axis=-1,
        )
    ).to(dtype=torch.float32)
    if y.ndim == 3:
        y = y[:, 0, :]
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=do_shuffle)


def batch_mlp_corr(
    judge_policy: nn.Module,
    df_train: pl.DataFrame,
    df_holdout: pl.DataFrame,
    policy_cfg: DictConfig,
    reye_dest: Path,
) -> tuple[Path, list[float], list[float]]:
    train_dataloader = __getdataloader__(
        df_train, policy_cfg.batch_size, judge_policy.net.input.in_features
    )
    val_dataloader = __getdataloader__(
        df_holdout,
        policy_cfg.batch_size,
        judge_policy.net.input.in_features,
        do_shuffle=False,
    )

    optimizer = optim.Adam(
        judge_policy.parameters(),
        lr=policy_cfg.lr,
        weight_decay=policy_cfg.weight_decay,
    )
    criterion = nn.MSELoss()

    device = get_device_from_parameters(judge_policy)
    train_losses = []
    val_losses = []
    for epoch in range(policy_cfg.num_epochs):
        judge_policy.train()
        total_loss = 0.0
        for xb, yb in train_dataloader:
            xb, yb = xb.to(device), yb.to(device)  # noqa: PLW2901
            optimizer.zero_grad()
            preds = judge_policy(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        train_losses.append(avg_loss)

        judge_policy.eval()
        total_loss_val = 0.0
        # action serialization
        pred_actions = torch.zeros(
            (len(val_dataloader.dataset), 3), dtype=torch.float32, device=device
        )
        bsize = val_dataloader.batch_size
        with torch.no_grad():
            for step, (xb, yb) in enumerate(val_dataloader):
                cur_bsize = xb.shape[0]
                xb, yb = xb.to(device), yb.to(device)  # noqa: PLW2901
                preds = judge_policy(xb)
                pred_actions[step * bsize : step * bsize + cur_bsize, :] = preds
                loss = criterion(preds, yb)
                total_loss_val += loss.item()
        avg_loss_val = total_loss_val / len(val_dataloader)
        val_losses.append(avg_loss_val)
        logging.info(  # noqa: LOG015
            f"Epoch {epoch + 1}/{policy_cfg.num_epochs}, Loss: {avg_loss:e} Val Loss: {avg_loss_val:e}"  # noqa: E501, G004
        )
        if epoch == policy_cfg.num_epochs - 1:
            log_images = []
            # Log wandb metrics
            action_callback(df_holdout, log_images, pred_actions)
            Path(f"tmp/{policy_cfg.train_run}").mkdir(parents=True, exist_ok=True)
            log_images[-1].image.save(
                f"tmp/{policy_cfg.train_run}/action_plot_epoch_{epoch}.png"
            )
            # Log reye metrics
            df_res = create_reye_df_from_reye(df_holdout, pred_actions)
            reye_endpoint = f"mlp-{policy_cfg.train_run}"
            dest = reye_dest / Path(reye_endpoint)
            logging.info(  # noqa: LOG015
                "".join([
                    f"mlp-{policy_cfg.train_run}: ",
                    "${_dir_lerobot}/",
                    f"mlp-{policy_cfg.train_run}",
                ])
            )
            dest.mkdir(parents=True, exist_ok=True)
            df_res.write_parquet(dest / f"val_results_epoch_{epoch}.parquet")

    timestamp = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")  # noqa: DTZ005
    checkpoint_dir = Path(
        f"outputs/train/{timestamp}_{policy_cfg.train_run}/{CHECKPOINTS_DIR}"
    )
    pretrained_dir = checkpoint_dir / PRETRAINED_MODEL_DIR
    judge_policy.save_pretrained(pretrained_dir)
    logging.info(  # noqa: LOG015
        f"Model saved to {pretrained_dir} and {checkpoint_dir}"  # noqa: G004
    )
    save_training_state(
        checkpoint_dir, policy_cfg.num_epochs * len(train_dataloader), optimizer, None
    )
    return checkpoint_dir, train_losses, val_losses
