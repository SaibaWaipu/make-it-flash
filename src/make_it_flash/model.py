"""Gated DeltaNet construction helpers for Qwen3-MoE pilot conversion."""

from __future__ import annotations

from typing import Any


def get_decoder_layers(model: Any) -> Any:
    """Return the Qwen-style decoder layer list or fail with a useful error."""
    candidates = (
        getattr(getattr(model, "model", None), "layers", None),
        getattr(getattr(getattr(model, "model", None), "model", None), "layers", None),
    )
    for layers in candidates:
        if layers is not None:
            return layers
    raise ValueError("expected a Qwen decoder with model.layers")


def make_gdn_config(base_config: Any, *, key_heads: int | None = None, value_heads: int | None = None) -> Any:
    """Build a Qwen3-Next GDN config sized to the LLM-jp hidden width.

    GDN head counts are deliberately an explicit design choice rather than
    copied from full attention's KV count. Defaults preserve the teacher's
    query head width and total key width, with twice as many value heads.
    """
    if getattr(base_config, "model_type", None) != "qwen3_moe":
        raise ValueError(f"unsupported teacher model_type: {getattr(base_config, 'model_type', None)!r}")
    from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig

    hidden_size = int(base_config.hidden_size)
    attention_heads = int(base_config.num_attention_heads)
    head_dim = int(getattr(base_config, "head_dim", hidden_size // attention_heads))
    key_heads = key_heads or max(1, attention_heads // 2)
    value_heads = value_heads or attention_heads
    if key_heads <= 0 or value_heads <= 0 or value_heads % key_heads:
        raise ValueError("GDN value-head count must be a positive multiple of key-head count")
    if key_heads * head_dim != hidden_size:
        raise ValueError(
            f"key_heads * head_dim must equal hidden_size ({key_heads} * {head_dim} != {hidden_size})"
        )

    return Qwen3NextConfig(
        vocab_size=int(base_config.vocab_size),
        hidden_size=hidden_size,
        intermediate_size=int(getattr(base_config, "intermediate_size", hidden_size * 4)),
        num_hidden_layers=int(base_config.num_hidden_layers),
        num_attention_heads=attention_heads,
        head_dim=head_dim,
        rms_norm_eps=float(getattr(base_config, "rms_norm_eps", 1e-6)),
        linear_key_head_dim=head_dim,
        linear_value_head_dim=head_dim,
        linear_num_key_heads=key_heads,
        linear_num_value_heads=value_heads,
        linear_conv_kernel_dim=4,
        layer_types=["linear_attention"] * int(base_config.num_hidden_layers),
        use_cache=False,
    )


def make_gdn(base_config: Any, layer_idx: int, *, key_heads: int | None = None, value_heads: int | None = None) -> Any:
    """Instantiate a Transformers Qwen3-Next GDN module for one teacher layer."""
    from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextGatedDeltaNet

    config = make_gdn_config(base_config, key_heads=key_heads, value_heads=value_heads)
    return Qwen3NextGatedDeltaNet(config, layer_idx)
