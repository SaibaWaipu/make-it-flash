"""Custom Transformers loader for the integrated LLM-jp Flash-Next checkpoint.

Load with ``trust_remote_code=True``. The checkpoint stores all tensors in the
ordinary Qwen3-MoE shard/index layout, but its attention modules are custom:
24 GDN layers and 8 QSA layers.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM, Qwen3MoeModel

from .configuration_flashnext import FlashNextQwen3MoeConfig
# Direct relative imports ensure Hugging Face's dynamic-module copier ships
# the transitive runtime helpers referenced lazily by model.py.
from .flash_next import QSAQwen3MoeAttentionAdapter as _QSAQwen3MoeAttentionAdapter
from .hybrid_cache import FlashNextDynamicCache as _FlashNextDynamicCache
from .model import (
    GDNQwen3MoeAttentionAdapter,
    create_flash_next_cache,
    flash_next_attention_schedule,
    make_gdn,
    make_qsa,
)


def _integration(config: FlashNextQwen3MoeConfig) -> tuple[tuple[int, ...], tuple[int, ...], dict, dict]:
    metadata = getattr(config, "flash_next_integration", None)
    if not isinstance(metadata, dict) or metadata.get("format") != "make_it_flash.integrated_full_checkpoint.v1":
        raise ValueError("config is missing the integrated Flash-Next provenance and module configs")

    expected_gdn, expected_qsa = flash_next_attention_schedule(int(config.num_hidden_layers))
    gdn_layers = tuple(metadata.get("gdn_layers", ()))
    qsa_layers = tuple(metadata.get("qsa_layers", ()))
    if gdn_layers != expected_gdn or qsa_layers != expected_qsa:
        raise ValueError("checkpoint attention schedule is not the supported 3-GDN/1-QSA layout")

    gdn_config = metadata.get("gdn_config")
    qsa_config = metadata.get("qsa_config")
    if not isinstance(gdn_config, dict) or not isinstance(qsa_config, dict):
        raise ValueError("config must include both gdn_config and qsa_config")
    expected_head_dim = int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
    if (
        int(gdn_config.get("linear_key_head_dim", -1)) != expected_head_dim
        or int(gdn_config.get("linear_value_head_dim", -1)) != expected_head_dim
        or int(gdn_config.get("linear_conv_kernel_dim", -1)) != 4
    ):
        raise ValueError("GDN config does not match the integrated checkpoint dimensions")
    return gdn_layers, qsa_layers, gdn_config, qsa_config


def _configured_dtype(config: FlashNextQwen3MoeConfig) -> torch.dtype | None:
    dtype: Any = getattr(config, "dtype", None)
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype, None)
    if not isinstance(dtype, torch.dtype):
        dtype = getattr(config, "torch_dtype", None)
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype, None)
    return dtype if isinstance(dtype, torch.dtype) else None


def _install_flash_next_attention(model: nn.Module) -> tuple[tuple[int, ...], tuple[int, ...]]:
    config = model.config
    gdn_layers, qsa_layers, gdn_config, qsa_config = _integration(config)
    decoder_layers = getattr(model, "layers", None)
    if decoder_layers is None:
        decoder_layers = getattr(getattr(model, "model", None), "layers", None)
    if decoder_layers is None or len(decoder_layers) != config.num_hidden_layers:
        raise ValueError("expected a Qwen3-MoE decoder with the configured number of layers")

    dtype = _configured_dtype(config)
    for layer_idx in gdn_layers:
        gdn = make_gdn(
            config,
            layer_idx,
            key_heads=int(gdn_config["linear_num_key_heads"]),
            value_heads=int(gdn_config["linear_num_value_heads"]),
        )
        attention = GDNQwen3MoeAttentionAdapter(gdn)
        decoder_layers[layer_idx].self_attn = attention.to(dtype=dtype) if dtype is not None else attention

    for layer_idx in qsa_layers:
        attention = make_qsa(config, layer_idx=layer_idx, **qsa_config)
        decoder_layers[layer_idx].self_attn = attention.to(dtype=dtype) if dtype is not None else attention
    return gdn_layers, qsa_layers


def _ensure_flash_next_cache(
    model: nn.Module,
    *,
    past_key_values: Any | None,
    use_cache: bool,
) -> Any | None:
    if not use_cache or past_key_values is not None:
        return past_key_values
    gdn_layers, qsa_layers, _, _ = _integration(model.config)
    return create_flash_next_cache(model, gdn_layers=gdn_layers, qsa_layers=qsa_layers)


class FlashNextQwen3MoeModel(Qwen3MoeModel):
    """Qwen3-MoE body with the integrated GDN/QSA modules installed."""

    config_class = FlashNextQwen3MoeConfig

    def __init__(self, config: FlashNextQwen3MoeConfig):
        super().__init__(config)
        _install_flash_next_attention(self)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Any | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Any,
    ):
        if use_cache is None:
            use_cache = bool(getattr(self.config, "use_cache", False))
        past_key_values = _ensure_flash_next_cache(
            self, past_key_values=past_key_values, use_cache=bool(use_cache)
        )
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )


class FlashNextQwen3MoeForCausalLM(Qwen3MoeForCausalLM):
    """Causal-LM wrapper that installs custom attention before weight loading."""

    config_class = FlashNextQwen3MoeConfig

    def __init__(self, config: FlashNextQwen3MoeConfig):
        super().__init__(config)
        _install_flash_next_attention(self.model)

    @classmethod
    def _supports_default_dynamic_cache(cls) -> bool:
        # GenerationMixin's DynamicCache lacks the QSA indexer state. The model
        # forward creates FlashNextDynamicCache instead when caching is enabled.
        return False

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Any | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_router_logits: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Any,
    ):
        if use_cache is None:
            use_cache = bool(getattr(self.config, "use_cache", False))
        past_key_values = _ensure_flash_next_cache(
            self, past_key_values=past_key_values, use_cache=bool(use_cache)
        )
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_router_logits=output_router_logits,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )


__all__ = ["FlashNextQwen3MoeModel", "FlashNextQwen3MoeForCausalLM"]
