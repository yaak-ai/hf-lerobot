"""Export safeguard tests for SmolVLM vision override classes.

Two concerns are tested here:

1. **API compatibility** (``test_upstream_*``): inspect the upstream transformers classes to
   verify that the contracts our export-path code relies on have not changed. These tests are
   intentionally fast (no heavy model instantiation) and act as a pre-flight check before
   running ``export_onnx.py``.

2. **Behavioral correctness** (``test_embeddings_*`` / ``test_transformer_*``): verify that
   the override classes produce the same output as their base classes in non-export mode.

All tests run on CPU only and are a prerequisite for ``just export-onnx``.
"""

import inspect

import pytest
import torch
from transformers.models.smolvlm.modeling_smolvlm import (
    SmolVLMVisionEmbeddings,
    SmolVLMVisionTransformer,
)

from lerobot.policies.smolvla.modeling_smolvlm import (
    ExportSmolVLMVisionEmbeddings,
    ExportSmolVLMVisionTransformer,
)
from tests.utils import require_package


# ---------------------------------------------------------------------------
# Config fixture — loaded from the VLM backbone referenced in onnx.yaml
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def production_vision_config():
    """SmolVLMVisionConfig loaded from the VLM backbone used in onnx.yaml.

    ``SmolVLAConfig.vlm_model_name`` resolves to
    ``"HuggingFaceTB/SmolVLM2-500M-Video-Instruct"``.  Only ``config.json``
    (~10 KB) is fetched; cached locally by HuggingFace after the first run.
    """
    transformers = pytest.importorskip("transformers")
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    vlm_config = transformers.AutoConfig.from_pretrained(SmolVLAConfig().vlm_model_name)
    return vlm_config.vision_config


def _fast_config(vision_config):
    """Return a copy of vision_config with num_hidden_layers=1 for fast CPU tests."""
    import copy

    cfg = copy.deepcopy(vision_config)
    cfg.num_hidden_layers = 1
    return cfg


# ---------------------------------------------------------------------------
# Group 1: API compatibility — detect upstream drift before export
# ---------------------------------------------------------------------------


@require_package("transformers")
def test_upstream_embeddings_super_call_viable():
    """SmolVLMVisionEmbeddings.forward must have pixel_values then patch_attention_mask as its
    first two positional parameters (after self).

    ExportSmolVLMVisionEmbeddings delegates via
    ``super().forward(pixel_values, patch_attention_mask)`` positionally.
    If upstream renames or reorders these parameters, the wrong values are passed silently.
    """
    params = list(inspect.signature(SmolVLMVisionEmbeddings.forward).parameters.keys())
    assert params[:3] == ["self", "pixel_values", "patch_attention_mask"], (
        f"SmolVLMVisionEmbeddings.forward parameter order has changed to {params}. "
        "Update ExportSmolVLMVisionEmbeddings.forward's super() call."
    )


@require_package("transformers")
def test_upstream_transformer_super_call_viable():
    """SmolVLMVisionTransformer.forward must have pixel_values then patch_attention_mask as its
    first two positional parameters (after self), and must accept **kwargs.

    ExportSmolVLMVisionTransformer delegates via
    ``super().forward(pixel_values, patch_attention_mask, **kwargs)``.
    If upstream removes the **kwargs slot the call will raise TypeError.
    """
    sig = inspect.signature(SmolVLMVisionTransformer.forward)
    params = list(sig.parameters.keys())
    assert params[:3] == ["self", "pixel_values", "patch_attention_mask"], (
        f"SmolVLMVisionTransformer.forward parameter order has changed to {params}. "
        "Update ExportSmolVLMVisionTransformer.forward's super() call."
    )
    var_keyword_params = [k for k, v in sig.parameters.items() if v.kind is inspect.Parameter.VAR_KEYWORD]
    assert var_keyword_params, (
        "SmolVLMVisionTransformer.forward no longer accepts **kwargs. "
        "Update ExportSmolVLMVisionTransformer.forward's super() call."
    )


@require_package("transformers")
def test_upstream_transformer_export_path_attributes(production_vision_config):
    """SmolVLMVisionTransformer must expose embeddings, encoder, and post_layernorm.

    The export path in ExportSmolVLMVisionTransformer accesses all three directly.
    """
    model = SmolVLMVisionTransformer(_fast_config(production_vision_config))
    for attr in ("embeddings", "encoder", "post_layernorm"):
        assert hasattr(model, attr), (
            f"SmolVLMVisionTransformer no longer has attribute '{attr}'. "
            "Update the export path in ExportSmolVLMVisionTransformer."
        )


@require_package("transformers")
def test_upstream_embeddings_export_path_attributes(production_vision_config):
    """SmolVLMVisionEmbeddings must expose patch_embedding and position_embedding (with .weight).

    The export path in ExportSmolVLMVisionEmbeddings accesses these directly.
    """
    model = SmolVLMVisionEmbeddings(production_vision_config)
    assert hasattr(model, "patch_embedding"), (
        "SmolVLMVisionEmbeddings no longer has attribute 'patch_embedding'. "
        "Update the export path in ExportSmolVLMVisionEmbeddings."
    )
    assert hasattr(model, "position_embedding"), (
        "SmolVLMVisionEmbeddings no longer has attribute 'position_embedding'. "
        "Update the export path in ExportSmolVLMVisionEmbeddings."
    )
    assert hasattr(model.position_embedding, "weight"), (
        "SmolVLMVisionEmbeddings.position_embedding no longer has a 'weight' attribute. "
        "Update the export path in ExportSmolVLMVisionEmbeddings."
    )


@require_package("transformers")
def test_upstream_encoder_export_call_viable(production_vision_config):
    """SmolVLMEncoder.forward(inputs_embeds=..., attention_mask=None) must work and return an
    object with a .last_hidden_state field.

    This is the exact call made by ExportSmolVLMVisionTransformer's export path, and the source
    of the breakage when upstream removed output_attentions / output_hidden_states / return_dict.

    The encoder is obtained from a fully initialised SmolVLMVisionTransformer (a PretrainedModel)
    so that _attn_implementation is properly set on the shared config object before use.
    """
    config = _fast_config(production_vision_config)
    # SmolVLMVisionTransformer.__init__ (PretrainedModel) sets config._attn_implementation;
    # extracting its encoder gives us a runnable SmolVLMEncoder.
    encoder = SmolVLMVisionTransformer(config).encoder.eval()
    dummy_embeds = torch.zeros(1, 4, config.hidden_size)
    with torch.no_grad():
        out = encoder(inputs_embeds=dummy_embeds, attention_mask=None)
    assert hasattr(out, "last_hidden_state"), (
        f"SmolVLMEncoder.forward no longer returns an object with 'last_hidden_state': {type(out)}. "
        "Update ExportSmolVLMVisionTransformer's export path."
    )


# ---------------------------------------------------------------------------
# Group 2: Behavioral consistency — super() delegation must match base class
# ---------------------------------------------------------------------------


@require_package("transformers")
def test_embeddings_non_export_matches_base(production_vision_config):
    """Outside export, ExportSmolVLMVisionEmbeddings must produce identical output to
    SmolVLMVisionEmbeddings.

    Validates that the non-export super() delegation is correct.
    """
    config = production_vision_config
    torch.manual_seed(0)
    base = SmolVLMVisionEmbeddings(config)
    override = ExportSmolVLMVisionEmbeddings(config)
    override.load_state_dict(base.state_dict())

    B, H, W = 2, config.image_size, config.image_size
    pixel_values = torch.randn(B, config.num_channels, H, W)
    nh, nw = H // config.patch_size, W // config.patch_size
    patch_attention_mask = torch.ones(B, nh, nw, dtype=torch.bool)

    base.eval()
    override.eval()
    with torch.no_grad():
        expected = base(pixel_values, patch_attention_mask)
        actual = override(pixel_values, patch_attention_mask)

    torch.testing.assert_close(actual, expected)


@require_package("transformers")
def test_transformer_non_export_matches_base(production_vision_config):
    """Outside export, ExportSmolVLMVisionTransformer must produce identical output to
    SmolVLMVisionTransformer.

    Validates that the non-export super() delegation is correct.
    """
    config = _fast_config(production_vision_config)
    torch.manual_seed(0)
    base = SmolVLMVisionTransformer(config)
    override = ExportSmolVLMVisionTransformer(config)
    override.load_state_dict(base.state_dict(), strict=False)

    pixel_values = torch.randn(1, config.num_channels, config.image_size, config.image_size)

    base.eval()
    override.eval()
    with torch.no_grad():
        expected = base(pixel_values)
        actual = override(pixel_values)

    torch.testing.assert_close(actual.last_hidden_state, expected.last_hidden_state)


