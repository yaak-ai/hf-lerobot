import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from safetensors.torch import save_model as save_model_as_safetensor
from torch.utils.data import DataLoader, TensorDataset

from lerobot.constants import CHECKPOINTS_DIR, PRETRAINED_MODEL_DIR
from lerobot.policies.utils import get_device_from_parameters
from lerobot.utils.hub import HubMixin
from lerobot.utils.train_utils import save_training_state
from lerobot.utils.wandb_utils import WandBLogger


class MLP(nn.Module, HubMixin):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)

    def _save_pretrained(self, save_directory: Path) -> None:
        model_to_save = self.module if hasattr(self, "module") else self
        save_model_as_safetensor(
            model_to_save, str(save_directory / SAFETENSORS_SINGLE_FILE)
        )


def __getdataloader__(dataset: pl.DataFrame, batch_size=2048):
    x = (
        torch.from_numpy(
            np.stack(
                (
                    np.stack(dataset["meta/VehicleMotion/gas_pedal_history"]),
                    np.stack(dataset["meta/VehicleMotion/brake_pedal_history"]),
                    np.stack(dataset["meta/VehicleMotion/steering_angle_history"]),
                ),
                axis=-1,
            )
        )
        .to(dtype=torch.float32)
        .reshape(-1, 18)
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
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)


def batch_mlp_corr(
    judge_policy: nn.Module,
    dataset: pl.DataFrame,
    dataset_val: pl.DataFrame,
    train_run: str,
    num_epochs: int = 10,
    batch_size: int = 2048,
) -> tuple[Path, list[float], list[float]]:
    train_dataloader = __getdataloader__(dataset, batch_size)
    val_dataloader = __getdataloader__(dataset_val, batch_size)

    optimizer = optim.Adam(judge_policy.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    device = get_device_from_parameters(judge_policy)
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
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
        with torch.no_grad():
            for xb, yb in val_dataloader:
                xb, yb = xb.to(device), yb.to(device)  # noqa: PLW2901
                preds = judge_policy(xb)
                loss = criterion(preds, yb)
                total_loss_val += loss.item()
        avg_loss_val = total_loss_val / len(val_dataloader)
        logging.info(f"Validation Loss: {avg_loss_val:.4f}")  # noqa: G004, LOG015
        val_losses.append(avg_loss_val)
        logging.info(  # noqa: LOG015
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f} Val Loss: {avg_loss_val:.4f}"  # noqa: E501, G004
        )

    timestamp = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")  # noqa: DTZ005
    checkpoint_dir = Path(f"outputs/train/{timestamp}_{train_run}/{CHECKPOINTS_DIR}")
    pretrained_dir = checkpoint_dir / PRETRAINED_MODEL_DIR
    judge_policy.save_pretrained(pretrained_dir)
    save_training_state(
        checkpoint_dir, num_epochs * len(train_dataloader), optimizer, None
    )
    return checkpoint_dir, train_losses, val_losses
