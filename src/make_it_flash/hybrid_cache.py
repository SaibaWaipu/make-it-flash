"""Hybrid DynamicCache support for GDN, QSA indexer state, and dense KV layers."""

from __future__ import annotations

import torch
from transformers.cache_utils import DynamicCache, DynamicLayer, LinearAttentionLayer


class _IndexedDynamicLayer(DynamicLayer):
    """Dynamic KV layer with the extra raw-key/position history used by QSA."""

    def __init__(self) -> None:
        super().__init__()
        self.indexer_keys: torch.Tensor | None = None
        self.indexer_position_ids: torch.Tensor | None = None

    def update_indexer(
        self, indexer_keys: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if indexer_keys.ndim != 3 or position_ids.ndim != 2:
            raise ValueError("QSA indexer keys and positions must have shapes (batch, seq, dim) and (batch, seq)")
        if indexer_keys.shape[:2] != position_ids.shape:
            raise ValueError("QSA indexer keys and positions have incompatible batch/sequence shapes")
        if self.indexer_keys is None:
            self.indexer_keys = indexer_keys
            self.indexer_position_ids = position_ids
        else:
            if self.indexer_keys.shape[0] != indexer_keys.shape[0] or self.indexer_keys.shape[-1] != indexer_keys.shape[-1]:
                raise ValueError("QSA indexer cache batch size or head dimension changed")
            if self.indexer_position_ids is None:
                raise RuntimeError("QSA indexer position cache is missing")
            self.indexer_keys = torch.cat((self.indexer_keys, indexer_keys), dim=1)
            self.indexer_position_ids = torch.cat((self.indexer_position_ids, position_ids), dim=1)
        return self.indexer_keys, self.indexer_position_ids

    def reset(self) -> None:
        super().reset()
        # DynamicLayer.reset zeroes but retains its token length. QSA requires
        # the KV and indexer histories to both restart at length zero.
        if self.keys.ndim >= 2:
            self.keys = self.keys[..., :0, :]
            self.values = self.values[..., :0, :]
        self.indexer_keys = None
        self.indexer_position_ids = None

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        super().reorder_cache(beam_idx)
        if self.indexer_keys is not None:
            self.indexer_keys = self.indexer_keys.index_select(0, beam_idx.to(self.indexer_keys.device))
        if self.indexer_position_ids is not None:
            self.indexer_position_ids = self.indexer_position_ids.index_select(
                0, beam_idx.to(self.indexer_position_ids.device)
            )

    def crop(self, max_length: int) -> None:
        super().crop(max_length)
        if self.indexer_keys is None:
            return
        if max_length < 0:
            max_length = self.indexer_keys.shape[1] - abs(max_length)
        self.indexer_keys = self.indexer_keys[:, :max_length]
        if self.indexer_position_ids is not None:
            self.indexer_position_ids = self.indexer_position_ids[:, :max_length]

    def batch_repeat_interleave(self, repeats: int) -> None:
        super().batch_repeat_interleave(repeats)
        if self.indexer_keys is not None:
            self.indexer_keys = self.indexer_keys.repeat_interleave(repeats, dim=0)
        if self.indexer_position_ids is not None:
            self.indexer_position_ids = self.indexer_position_ids.repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        super().batch_select_indices(indices)
        if self.indexer_keys is not None:
            self.indexer_keys = self.indexer_keys[indices.to(self.indexer_keys.device)]
        if self.indexer_position_ids is not None:
            self.indexer_position_ids = self.indexer_position_ids[indices.to(self.indexer_position_ids.device)]


class FlashNextDynamicCache(DynamicCache):
    """A DynamicCache whose per-layer types include indexed-attention state."""

    supports_qsa_indexer_positions = True

    def __init__(self, layer_types: list[str] | tuple[str, ...]) -> None:
        if not layer_types:
            raise ValueError("layer_types must not be empty")
        allowed = {"linear_attention", "indexed_attention", "full_attention"}
        unknown = set(layer_types) - allowed
        if unknown:
            raise ValueError(f"unsupported Flash-Next cache layer types: {sorted(unknown)}")
        super().__init__()
        self.layers = []
        for layer_type in layer_types:
            if layer_type == "linear_attention":
                self.layers.append(LinearAttentionLayer())
            elif layer_type == "indexed_attention":
                self.layers.append(_IndexedDynamicLayer())
            else:
                self.layers.append(DynamicLayer())
        self.layer_class_to_replicate = None
        self.layer_types = tuple(layer_types)

    def update_indexer(
        self, indexer_keys: torch.Tensor, position_ids: torch.Tensor, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if layer_idx < 0 or layer_idx >= len(self.layers):
            raise ValueError(f"QSA indexer layer index {layer_idx} is outside the cache")
        layer = self.layers[layer_idx]
        if not isinstance(layer, _IndexedDynamicLayer):
            raise ValueError(f"cache layer {layer_idx} is not configured for indexed attention")
        return layer.update_indexer(indexer_keys, position_ids)
