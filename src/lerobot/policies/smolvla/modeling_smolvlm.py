from typing import Optional, Union

import torch
import torch.nn.functional as F
from transformers.models.smolvlm.modeling_smolvlm import (
    BaseModelOutput,
    SmolVLMVisionAttention,
    SmolVLMVisionEmbeddings,
    SmolVLMVisionTransformer,
)


class ExportSmolVLMVisionAttention(SmolVLMVisionAttention):
    """Override to avoid a float32 attention scale constant during ONNX export.

    The parent stores `self.scale = head_dim**-0.5` 
    This only creates problems for torch== 2.10.x and this is not a nice solution
    since attention implementation in the original model might change.
    Also ONNX surgery to fix this post-export is possible
    but non-trivial due to the constant being embedded in a local function (not visible in the main graph).
    Hence, torch is downgraded to 2.8.0 in pyproject.toml to avoid this issue.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not torch.compiler.is_exporting():
            return super().forward(hidden_states, attention_mask, **kwargs)

        batch_size, seq_length, embed_dim = hidden_states.shape

        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        queries = queries.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        # Derive scale from the query tensor so the constant matches the model dtype
        # (avoids a float32 Constant node when exporting to float16)
        scale = queries.new_tensor(self.head_dim**-0.5)
        attn_weights = torch.matmul(queries, keys.transpose(-1, -2)) * scale
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(queries.dtype)

        attn_output = torch.matmul(attn_weights, values)
        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(batch_size, seq_length, embed_dim).contiguous()
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights


class ExportSmolVLMVisionEmbeddings(SmolVLMVisionEmbeddings):
    """Override to avoid data-dependent control flow during ONNX export.

    During export, position_ids are fixed to 1024 (max sequence length) to
    avoid dynamic shapes from patch_attention_mask iteration. Outside export,
    delegates to the upstream implementation unchanged.
    """

    def forward(self, pixel_values: torch.FloatTensor, patch_attention_mask: Optional[torch.BoolTensor] = None) -> torch.Tensor:
        if not torch.compiler.is_exporting():
            return super().forward(pixel_values, patch_attention_mask)

        patch_embeds = self.patch_embedding(pixel_values)
        embeddings = patch_embeds.flatten(2).transpose(1, 2)

        # Fixed position_ids: circumvents data-dependent control flow over patch_attention_mask
        position_ids = torch.arange(0, 1024, dtype=torch.int64).unsqueeze(0).repeat(pixel_values.shape[0], 1)
        position_ids = position_ids.to(self.position_embedding.weight.device)
        embeddings += self.position_embedding(position_ids)
        return embeddings


class ExportSmolVLMVisionTransformer(SmolVLMVisionTransformer):
    """Override to remove data-dependent attention mask allocation during ONNX export.

    During export, patch_attention_mask is set to None (attend to full sequence)
    to avoid dynamic control flow. Outside export, delegates to the upstream
    implementation unchanged.
    """

    def forward(
        self,
        pixel_values,
        patch_attention_mask: Optional[torch.BoolTensor] = None,
        **kwargs,
    ) -> Union[tuple, BaseModelOutput]:
        if not torch.compiler.is_exporting():
            return super().forward(pixel_values, patch_attention_mask, **kwargs)

        # Export path: skip patch_attention_mask entirely to avoid data-dependent shapes
        hidden_states = self.embeddings(pixel_values=pixel_values)

        encoder_outputs: BaseModelOutput = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=None,
        )

        last_hidden_state = encoder_outputs.last_hidden_state
        last_hidden_state = self.post_layernorm(last_hidden_state)

        return BaseModelOutput(last_hidden_state=last_hidden_state)
