"""Gated DeltaNet construction helpers for Qwen3-MoE pilot conversion."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class _SigmoidRMSNormGated(nn.Module):
    """Qwen4Exp-style gated RMSNorm used by Qwen3.8-Flash-Next GDN blocks."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.activation = "sigmoid"

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * torch.sigmoid(gate.to(torch.float32))
        return hidden_states.to(input_dtype)


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

    Defaults use the Qwen3.8-Flash-Next GDN layout (16 QK heads, 48 value
    heads, 128 dimensions each). The value projection maps back to the teacher
    hidden width, so the GDN key/value widths need not equal hidden_size.
    """
    if getattr(base_config, "model_type", None) != "qwen3_moe":
        raise ValueError(f"unsupported teacher model_type: {getattr(base_config, 'model_type', None)!r}")
    from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig

    hidden_size = int(base_config.hidden_size)
    attention_heads = int(base_config.num_attention_heads)
    head_dim = int(getattr(base_config, "head_dim", hidden_size // attention_heads))
    key_heads = 16 if key_heads is None else key_heads
    value_heads = 48 if value_heads is None else value_heads
    if head_dim <= 0 or key_heads <= 0 or value_heads <= 0 or value_heads % key_heads:
        raise ValueError("GDN dimensions must be positive and value-head count a multiple of key-head count")

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
    gdn = Qwen3NextGatedDeltaNet(config, layer_idx)
    # Qwen3.8-Flash-Next separates the GDN output gate from the SiLU conv activation.
    gdn.norm = _SigmoidRMSNormGated(gdn.head_v_dim, eps=gdn.layer_norm_epsilon)
    return gdn
