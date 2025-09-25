from typing import final, override

import torch
from torch import Tensor
from torch.nn import Linear, Module


@final
class ModalityDropout(Module):
    def __init__(
        self, max_state_dim: int, hidden_size: int, probability: float
    ) -> None:
        super().__init__()
        self.state_proj = Linear(max_state_dim, hidden_size)
        self.probability = probability
        self.mask_embedding_dim = hidden_size
        self.mask_embedding = torch.nn.Parameter(torch.randn(hidden_size))

    @override
    def forward(self, inputs: Tensor) -> Tensor:
        embeddings = self.state_proj(inputs)
        mask = self.mask_embedding.expand(embeddings.shape).to(
            dtype=embeddings.dtype, device=embeddings.device
        )

        sample_mask = (torch.rand(embeddings.shape[0], device=embeddings.device) < self.probability).view(
            -1, *([1] * (embeddings.ndim - 1))
        )

        return torch.where(sample_mask, mask, embeddings)
