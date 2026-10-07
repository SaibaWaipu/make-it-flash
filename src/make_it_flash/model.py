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
        getattr(model, "layers", None),
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


def configure_hybrid_cache(model: Any, gdn_layers: list[int] | tuple[int, ...]) -> list[str]:
    """Configure Transformers' DynamicCache for GDN and full-attention layers.

    Call this before the first cached forward pass. GDN indices use the same
    decoder-layer numbering as ``model.layers``; all other layers use standard
    attention KV-cache layers.
    """
    decoder_layers = get_decoder_layers(model)
    layer_count = len(decoder_layers)
    indices = tuple(gdn_layers)
    if any(not isinstance(index, int) or isinstance(index, bool) for index in indices):
        raise ValueError("GDN layer indices must be integers")
    if len(set(indices)) != len(indices):
        raise ValueError("GDN layer indices must not contain duplicates")
    if any(index < 0 or index >= layer_count for index in indices):
        raise ValueError(f"GDN layer indices must be in [0, {layer_count})")

    config = getattr(model, "config", None)
    if config is None:
        raise ValueError("model must expose a Transformers config")
    decoder_config = config.get_text_config(decoder=True) if hasattr(config, "get_text_config") else config
    if getattr(decoder_config, "model_type", None) != "qwen3_moe":
        raise ValueError(f"unsupported cache model_type: {getattr(decoder_config, 'model_type', None)!r}")
    gdn_indices = set(indices)
    layer_types = [
        "linear_attention" if layer_idx in gdn_indices else "full_attention"
        for layer_idx in range(layer_count)
    ]
    decoder_config.layer_types = layer_types
    return layer_types


class GDNQwen3MoeAttentionAdapter(nn.Module):
    """Expose GDN through Qwen3-MoE's attention and hybrid-cache interfaces.

    Supports no mask, a 2-D binary padding mask, or standard 4-D causal masks
    with optional binary key padding. For cached generation, configure the
    model with :func:`configure_hybrid_cache` before creating DynamicCache.
    Nonstandard/sparse masks and incompatible cache layouts fail closed.
    """

    def __init__(self, gdn: nn.Module) -> None:
        super().__init__()
        self.gdn = gdn

    @staticmethod
    def _padding_mask(
        hidden_states: torch.Tensor, attention_mask: torch.Tensor | None
    ) -> torch.Tensor | None:
        if attention_mask is None:
            return None
        batch_size, query_length = hidden_states.shape[:2]
        attention_mask = attention_mask.to(device=hidden_states.device)
        if attention_mask.ndim == 2:
            if attention_mask.shape[0] != batch_size or attention_mask.shape[1] < query_length:
                raise NotImplementedError("2-D padding masks must cover the current GDN input")
            return attention_mask[:, -query_length:].to(dtype=torch.bool)
        if (
            attention_mask.ndim != 4
            or attention_mask.shape[0] != batch_size
            or attention_mask.shape[1] != 1
            or attention_mask.shape[-2] != query_length
            or attention_mask.shape[-1] < query_length
        ):
            raise NotImplementedError("unsupported attention mask shape for GDN adapter")
        if attention_mask.dtype == torch.bool:
            allowed = attention_mask
        elif attention_mask.is_floating_point():
            allowed = attention_mask == 0
        else:
            raise NotImplementedError("4-D attention masks must be boolean or additive floating point")

        key_length = allowed.shape[-1]
        key_valid = allowed.any(dim=-2).squeeze(1)
        query_valid = allowed.any(dim=-1).squeeze(1)
        query_positions = torch.arange(query_length, device=hidden_states.device) + key_length - query_length
        key_positions = torch.arange(key_length, device=hidden_states.device)
        causal = key_positions[None, :] <= query_positions[:, None]
        expected = causal[None, None, :, :] & query_valid[:, None, :, None] & key_valid[:, None, None, :]
        if not torch.equal(allowed, expected):
            raise NotImplementedError("only standard causal masks with binary padding are supported")
        return key_valid[:, -query_length:]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Any | None = None,
        use_cache: bool | None = False,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        if use_cache and past_key_values is None:
            raise NotImplementedError("cached GDN calls require a hybrid DynamicCache")
        if kwargs.get("output_attentions", False):
            raise NotImplementedError("GDN does not provide attention weights")
        if past_key_values is not None:
            from transformers.cache_utils import DynamicCache

            if not isinstance(past_key_values, DynamicCache):
                raise NotImplementedError("only Transformers DynamicCache is supported for cached GDN calls")
            layer_idx = getattr(self.gdn, "layer_idx", None)
            cache_layers = getattr(past_key_values, "layers", None)
            required_methods = ("has_previous_state", "update_conv_state", "update_recurrent_state")
            if (
                not isinstance(layer_idx, int)
                or cache_layers is None
                or layer_idx < 0
                or layer_idx >= len(cache_layers)
                or not all(callable(getattr(past_key_values, name, None)) for name in required_methods)
                or not all(
                    hasattr(cache_layers[layer_idx], name)
                    for name in ("has_previous_state", "update_conv_state", "update_recurrent_state")
                )
            ):
                raise NotImplementedError(
                    "past_key_values must be a DynamicCache configured with a linear_attention layer at this index"
                )
        padding_mask = self._padding_mask(hidden_states, attention_mask)
        return self.gdn(hidden_states, cache_params=past_key_values, attention_mask=padding_mask), None
