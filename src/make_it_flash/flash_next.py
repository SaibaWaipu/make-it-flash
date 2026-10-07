"""Composable Flash-Next-inspired components for Qwen3-MoE experiments.

These are building blocks, not a complete model conversion. Architecture-level
integration and quality evaluation remain separate steps.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class _GroupedRMSNorm(nn.Module):
    """RMSNorm that normalizes each residual stream independently."""

    def __init__(self, width: int, group_size: int, eps: float) -> None:
        super().__init__()
        if width <= 0 or group_size <= 0 or width % group_size:
            raise ValueError("width must be a positive multiple of group_size")
        self.weight = nn.Parameter(torch.zeros(width))
        self.group_size = group_size
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.shape[-1] != self.weight.numel():
            raise ValueError(f"expected last dimension {self.weight.numel()}, got {hidden_states.shape[-1]}")
        grouped = hidden_states.float().reshape(*hidden_states.shape[:-1], -1, self.group_size)
        normalized = grouped * torch.rsqrt(grouped.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        normalized = normalized.flatten(-2) * (1.0 + self.weight.float())
        return normalized.to(dtype=hidden_states.dtype)


class GatedResidualMixer(nn.Module):
    """Qwen4Exp-style low-rank mixer for widened residual streams.

    ``forward`` returns the mixed single-width branch input, the original
    concatenated stream state, and (unless ``use_combine=False``) one
    data-dependent write gate per stream. ``inject`` applies a branch output
    back into those streams. The module is not wired into Qwen3-MoE decoder
    layers by itself; replacing residual/norm paths requires distillation.
    """

    def __init__(
        self,
        hidden_size: int,
        *,
        stream_count: int = 4,
        low_rank: int = 320,
        eps: float = 1e-6,
        initializer_range: float = 0.02,
        use_combine: bool = True,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or stream_count <= 0 or low_rank <= 0:
            raise ValueError("hidden_size, stream_count, and low_rank must be positive")
        if initializer_range <= 0:
            raise ValueError("initializer_range must be positive")
        self.hidden_size = hidden_size
        self.stream_count = stream_count
        self.hc_hidden_size = hidden_size * stream_count
        self.hc_norm = _GroupedRMSNorm(self.hc_hidden_size, hidden_size, eps)
        self.input_mix_weight_down = nn.Linear(self.hc_hidden_size, low_rank, bias=False)
        self.input_mix_weight_up = nn.Linear(low_rank, self.hc_hidden_size, bias=False)
        self.block_inject_weight = (
            nn.Linear(self.hc_hidden_size, stream_count, bias=False) if use_combine else None
        )
        for module in (self.input_mix_weight_down, self.input_mix_weight_up, self.block_inject_weight):
            if module is not None:
                nn.init.normal_(module.weight, mean=0.0, std=initializer_range)

    def forward(
        self, hyper_input: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if hyper_input.shape[-1] != self.hc_hidden_size:
            raise ValueError(
                f"expected {self.hc_hidden_size} hyper-connection features, got {hyper_input.shape[-1]}"
            )
        hyper_input_normed = self.hc_norm(hyper_input)
        input_mix_weight = F.silu(self.input_mix_weight_down(hyper_input_normed) / self.stream_count)
        input_mix_weight = torch.sigmoid(self.input_mix_weight_up(input_mix_weight))
        input_mix_weight = input_mix_weight.unflatten(-1, (self.stream_count, self.hidden_size))
        streams = hyper_input_normed.unflatten(-1, (self.stream_count, self.hidden_size))
        mixed_input = (input_mix_weight * streams).mean(dim=-2)
        if self.block_inject_weight is None:
            return mixed_input
        injection_weights = 2 * torch.sigmoid(self.block_inject_weight(hyper_input_normed) / self.stream_count)
        return mixed_input, hyper_input, injection_weights

    def inject(
        self, hyper_input: torch.Tensor, branch_output: torch.Tensor, injection_weights: torch.Tensor
    ) -> torch.Tensor:
        """Add a single-width attention/MLP output back into all streams."""
        expected_input = self.hc_hidden_size
        expected_gate = self.stream_count
        if hyper_input.shape[-1] != expected_input:
            raise ValueError(f"expected residual state width {expected_input}")
        if branch_output.shape[:-1] != hyper_input.shape[:-1] or branch_output.shape[-1] != self.hidden_size:
            raise ValueError("branch_output must match residual leading dimensions and hidden_size")
        if injection_weights.shape != (*hyper_input.shape[:-1], expected_gate):
            raise ValueError("injection_weights must have one value per residual stream")
        injection = branch_output.unsqueeze(-2) * injection_weights.unsqueeze(-1)
        return hyper_input + injection.flatten(-2)


def _normalize_position_ids(
    position_ids: torch.Tensor | None, batch_size: int, seq_len: int, device: torch.device
) -> torch.Tensor:
    if position_ids is None:
        return torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    position_ids = position_ids.to(device=device, dtype=torch.long)
    if position_ids.ndim == 1:
        position_ids = position_ids.unsqueeze(0)
    if position_ids.ndim != 2 or position_ids.shape[-1] != seq_len or position_ids.shape[0] not in (1, batch_size):
        raise ValueError("position_ids must have shape (seq_len), (1, seq_len), or (batch, seq_len)")
    return position_ids.expand(batch_size, -1)


def _apply_partial_rope(
    hidden_states: torch.Tensor, position_ids: torch.Tensor, rotary_dim: int, rope_theta: float
) -> torch.Tensor:
    if rotary_dim == 0:
        return hidden_states
    if rotary_dim % 2 or rotary_dim > hidden_states.shape[-1]:
        raise ValueError("rotary_dim must be even and no larger than the head dimension")
    inv_freq = 1.0 / (
        rope_theta
        ** (torch.arange(0, rotary_dim, 2, device=hidden_states.device, dtype=torch.float32) / rotary_dim)
    )
    frequencies = position_ids.to(device=hidden_states.device, dtype=torch.float32).unsqueeze(-1) * inv_freq
    angles = torch.cat((frequencies, frequencies), dim=-1)
    cos = angles.cos().to(dtype=hidden_states.dtype).unsqueeze(-2)
    sin = angles.sin().to(dtype=hidden_states.dtype).unsqueeze(-2)
    rotated = hidden_states[..., :rotary_dim]
    half = rotary_dim // 2
    rotated_half = torch.cat((-rotated[..., half:], rotated[..., :half]), dim=-1)
    rotated = rotated * cos + rotated_half * sin
    if rotary_dim == hidden_states.shape[-1]:
        return rotated
    return torch.cat((rotated, hidden_states[..., rotary_dim:]), dim=-1)


class QSAIndexer(nn.Module):
    """Qwen Sparse Attention micro-block selector for uncached sequences."""

    def __init__(
        self,
        hidden_size: int,
        *,
        index_n_heads: int = 4,
        index_head_dim: int = 128,
        token_budget: int = 2048,
        compress_ratio: int = 4,
        rotary_dim: int = 64,
        rope_theta: float = 10_000_000.0,
        eps: float = 1e-6,
        initializer_range: float = 0.02,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or index_n_heads <= 0 or index_head_dim <= 0:
            raise ValueError("hidden_size and indexer head dimensions must be positive")
        if token_budget < compress_ratio or compress_ratio <= 0:
            raise ValueError("token_budget must cover at least one positive-sized micro-block")
        if rotary_dim < 0 or rotary_dim % 2 or rotary_dim > index_head_dim or rope_theta <= 0:
            raise ValueError("invalid rotary configuration")
        if initializer_range <= 0:
            raise ValueError("initializer_range must be positive")
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.token_budget = token_budget
        self.compress_ratio = compress_ratio
        self.block_topk = token_budget // compress_ratio
        self.rotary_dim = rotary_dim
        self.rope_theta = rope_theta
        self.index_qk_proj = nn.Linear(hidden_size, (index_n_heads + 1) * index_head_dim, bias=False)
        self.q_layernorm = _GroupedRMSNorm(index_head_dim, index_head_dim, eps)
        self.k_layernorm = _GroupedRMSNorm(index_head_dim, index_head_dim, eps)
        nn.init.normal_(self.index_qk_proj.weight, mean=0.0, std=initializer_range)

    def _project_queries_keys(
        self, hidden_states: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, query_length, _ = hidden_states.shape
        projected = self.index_qk_proj(hidden_states)
        query, raw_keys = torch.split(
            projected,
            [self.index_n_heads * self.index_head_dim, self.index_head_dim],
            dim=-1,
        )
        query = query.reshape(batch_size, query_length, self.index_n_heads, self.index_head_dim)
        raw_keys = raw_keys.reshape(batch_size, query_length, self.index_head_dim)
        query = self.q_layernorm(query)
        query = _apply_partial_rope(query, positions, self.rotary_dim, self.rope_theta)
        return query, raw_keys

    def _score_visible_blocks(
        self,
        query: torch.Tensor,
        raw_keys: torch.Tensor,
        positions: torch.Tensor,
        local_visible_indices: torch.Tensor,
        *,
        smooth: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_complete_blocks = local_visible_indices.numel() // self.compress_ratio
        block_token_indices = local_visible_indices[: num_complete_blocks * self.compress_ratio].view(
            num_complete_blocks, self.compress_ratio
        )
        if num_complete_blocks == 0:
            return raw_keys.new_empty((0,), dtype=torch.float32), block_token_indices
        key_groups = raw_keys.index_select(0, block_token_indices.flatten())
        pooled_keys = key_groups.view(num_complete_blocks, self.compress_ratio, self.index_head_dim)
        pooled_keys = pooled_keys.float().mean(dim=1).to(dtype=raw_keys.dtype)
        pooled_keys = self.k_layernorm(pooled_keys).unsqueeze(0).unsqueeze(2)
        block_start_positions = positions.index_select(0, block_token_indices[:, 0]).unsqueeze(0)
        block_keys = _apply_partial_rope(
            pooled_keys, block_start_positions, self.rotary_dim, self.rope_theta
        ).squeeze(0).squeeze(1)
        scores = torch.matmul(query.float(), block_keys.float().transpose(-1, -2)).transpose(-1, -2)
        activation = F.softplus(scores) if smooth else F.relu(scores)
        scores = activation.sum(dim=-1) / math.sqrt(self.index_head_dim)
        return scores, block_token_indices

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor | None,
        attention_mask: torch.Tensor,
        *,
        past_key_values: object | None = None,
        layer_idx: int | None = None,
    ) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape (batch, query, hidden)")
        batch_size, query_length, _ = hidden_states.shape
        attention_mask = attention_mask.to(device=hidden_states.device)
        if attention_mask.ndim != 4 or attention_mask.shape[:3] != (batch_size, 1, query_length):
            raise ValueError("QSA requires a 4-D causal/padding mask with shape (batch, 1, query, key)")
        if attention_mask.dtype == torch.bool:
            visible_token_indices = attention_mask
        elif attention_mask.is_floating_point():
            visible_token_indices = attention_mask == 0
        else:
            raise ValueError("attention_mask must be boolean or additive floating point")
        key_length = attention_mask.shape[-1]
        if key_length < query_length:
            raise ValueError("attention mask key length cannot be shorter than query length")
        query_positions = _normalize_position_ids(position_ids, batch_size, query_length, hidden_states.device)
        query, current_raw_keys = self._project_queries_keys(hidden_states, query_positions)
        if past_key_values is None:
            raw_keys = current_raw_keys
            key_positions = query_positions
        else:
            if layer_idx is None or not getattr(past_key_values, "supports_qsa_indexer_positions", False):
                raise NotImplementedError("cached QSA requires FlashNextDynamicCache and a valid layer_idx")
            raw_keys, key_positions = past_key_values.update_indexer(current_raw_keys, query_positions, layer_idx)
        if raw_keys.shape[:2] != (batch_size, key_length) or key_positions.shape != (batch_size, key_length):
            raise ValueError("QSA indexer cache length does not match attention_mask")

        max_selected_tokens = self.token_budget + self.compress_ratio - 1
        selected_token_indices = torch.full(
            (batch_size, query_length, max_selected_tokens),
            -1,
            dtype=torch.long,
            device=hidden_states.device,
        )
        visible = visible_token_indices[:, 0]
        for batch_idx in range(batch_size):
            for query_idx in range(query_length):
                local_visible_indices = torch.nonzero(visible[batch_idx, query_idx], as_tuple=False).flatten()
                num_complete_blocks = local_visible_indices.numel() // self.compress_ratio
                scores, block_token_indices = self._score_visible_blocks(
                    query[batch_idx, query_idx], raw_keys[batch_idx], key_positions[batch_idx], local_visible_indices
                )
                if num_complete_blocks:
                    chosen_blocks = scores.topk(min(self.block_topk, num_complete_blocks), dim=0).indices
                    selected_tokens = block_token_indices.index_select(0, chosen_blocks).flatten()
                else:
                    selected_tokens = local_visible_indices.new_empty((0,))
                tail = local_visible_indices[num_complete_blocks * self.compress_ratio :]
                selected_tokens = torch.cat((selected_tokens, tail))
                selected_token_indices[batch_idx, query_idx, : selected_tokens.numel()] = selected_tokens

        selected_token_mask = torch.zeros(
            (batch_size, query_length, key_length + 1), dtype=torch.bool, device=attention_mask.device
        )
        scatter_indices = torch.where(selected_token_indices >= 0, selected_token_indices, key_length)
        selected_token_mask.scatter_(-1, scatter_indices, True)
        return selected_token_mask[..., :key_length].unsqueeze(1)

    def selection_loss(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor | None,
        attention_mask: torch.Tensor,
        teacher_attention: torch.Tensor,
    ) -> torch.Tensor:
        """Train block scores toward teacher attention mass (top-k itself has no gradient).

        ``teacher_attention`` is a dense teacher probability tensor shaped
        ``(batch, heads, query, key)`` or ``(batch, query, key)``. This auxiliary
        loss uses a smooth softplus score surrogate; inference keeps the exact
        ReLU/top-k reference selector.
        """
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape (batch, query, hidden)")
        batch_size, query_length, _ = hidden_states.shape
        attention_mask = attention_mask.to(device=hidden_states.device)
        if attention_mask.ndim != 4 or attention_mask.shape[:3] != (batch_size, 1, query_length):
            raise ValueError("QSA requires a 4-D causal/padding mask")
        if attention_mask.shape[-1] != query_length:
            raise NotImplementedError("cached QSA selection loss is not implemented yet")
        if attention_mask.dtype == torch.bool:
            visible = attention_mask[:, 0]
        elif attention_mask.is_floating_point():
            visible = attention_mask[:, 0] == 0
        else:
            raise ValueError("attention_mask must be boolean or additive floating point")
        teacher = teacher_attention.detach().to(device=hidden_states.device, dtype=torch.float32)
        if teacher.ndim == 3:
            teacher = teacher.unsqueeze(1)
        if teacher.ndim != 4 or teacher.shape[0] != batch_size or teacher.shape[-2:] != (query_length, query_length):
            raise ValueError("teacher_attention must have shape (batch, heads, query, key)")
        if not torch.isfinite(teacher).all() or torch.any(teacher < 0):
            raise ValueError("teacher_attention must contain finite nonnegative probabilities")
        teacher = teacher.mean(dim=1)
        positions = _normalize_position_ids(position_ids, batch_size, query_length, hidden_states.device)
        query, raw_keys = self._project_queries_keys(hidden_states, positions)
        losses = []
        for batch_idx in range(batch_size):
            for query_idx in range(query_length):
                local_visible_indices = torch.nonzero(visible[batch_idx, query_idx], as_tuple=False).flatten()
                num_complete_blocks = local_visible_indices.numel() // self.compress_ratio
                if num_complete_blocks == 0:
                    continue
                scores, block_token_indices = self._score_visible_blocks(
                    query[batch_idx, query_idx],
                    raw_keys[batch_idx],
                    positions[batch_idx],
                    local_visible_indices,
                    smooth=True,
                )
                teacher_mass = teacher[batch_idx, query_idx].index_select(
                    0, block_token_indices.flatten()
                ).view(num_complete_blocks, self.compress_ratio).sum(dim=-1)
                total_mass = teacher_mass.sum()
                if total_mass > 0:
                    target_distribution = teacher_mass / total_mass
                    losses.append(-(target_distribution * F.log_softmax(scores, dim=0)).sum())
        if not losses:
            return hidden_states.sum() * 0.0
        return torch.stack(losses).mean()


class QSAQwen3MoeAttentionAdapter(nn.Module):
    """Reference-shaped sparse GQA attention for Qwen3-MoE layer grafting.

    Cached decoding requires ``FlashNextDynamicCache`` and a ``layer_idx``.
    The standard DynamicCache is deliberately rejected because it lacks the
    QSA raw-key and position-id cache state.
    """

    def __init__(
        self,
        hidden_size: int,
        *,
        layer_idx: int | None = None,
        num_heads: int = 24,
        num_key_value_heads: int = 2,
        head_dim: int = 256,
        rotary_dim: int = 64,
        rope_theta: float = 10_000_000.0,
        index_n_heads: int = 4,
        index_head_dim: int = 128,
        token_budget: int = 2048,
        compress_ratio: int = 4,
        attention_dropout: float = 0.0,
        eps: float = 1e-6,
        initializer_range: float = 0.02,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or num_heads <= 0 or num_key_value_heads <= 0 or head_dim <= 0:
            raise ValueError("attention dimensions must be positive")
        if num_heads % num_key_value_heads:
            raise ValueError("num_heads must be divisible by num_key_value_heads")
        if not 0 <= attention_dropout < 1:
            raise ValueError("attention_dropout must be in [0, 1)")
        if rotary_dim < 0 or rotary_dim % 2 or rotary_dim > head_dim or rope_theta <= 0:
            raise ValueError("invalid rotary configuration")
        if initializer_range <= 0:
            raise ValueError("initializer_range must be positive")
        self.hidden_size = hidden_size
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_heads // num_key_value_heads
        self.head_dim = head_dim
        self.scaling = head_dim**-0.5
        self.rotary_dim = rotary_dim
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim * 2, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.q_norm = _GroupedRMSNorm(head_dim, head_dim, eps)
        self.k_norm = _GroupedRMSNorm(head_dim, head_dim, eps)
        self.indexer = QSAIndexer(
            hidden_size,
            index_n_heads=index_n_heads,
            index_head_dim=index_head_dim,
            token_budget=token_budget,
            compress_ratio=compress_ratio,
            rotary_dim=min(rotary_dim, index_head_dim),
            rope_theta=rope_theta,
            eps=eps,
            initializer_range=initializer_range,
        )
        for module in (self.q_proj, self.k_proj, self.v_proj, self.o_proj):
            nn.init.normal_(module.weight, mean=0.0, std=initializer_range)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: object | None = None,
        use_cache: bool | None = False,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if use_cache:
            if past_key_values is None or not getattr(past_key_values, "supports_qsa_indexer_positions", False):
                raise NotImplementedError("cached QSA requires FlashNextDynamicCache from create_flash_next_cache")
        elif past_key_values is not None:
            raise NotImplementedError("QSA cache was supplied while use_cache=False")
        if past_key_values is not None and self.layer_idx is None:
            raise NotImplementedError("cached QSA requires the decoder layer_idx")
        if attention_mask is None:
            raise NotImplementedError("QSA requires an explicit causal/padding attention mask")
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError("hidden_states must match (batch, sequence, hidden_size)")
        batch_size, seq_len, _ = hidden_states.shape
        positions = _normalize_position_ids(position_ids, batch_size, seq_len, hidden_states.device)
        attention_mask = attention_mask.to(device=hidden_states.device)
        selected_mask = self.indexer(
            hidden_states,
            positions,
            attention_mask,
            past_key_values=past_key_values,
            layer_idx=self.layer_idx,
        )
        if attention_mask.dtype == torch.bool:
            combined_mask = attention_mask & selected_mask
            visible = combined_mask
        elif attention_mask.is_floating_point():
            min_dtype = torch.finfo(attention_mask.dtype).min
            sparse_additive_mask = torch.where(
                selected_mask,
                torch.zeros((), dtype=attention_mask.dtype, device=attention_mask.device),
                min_dtype,
            )
            combined_mask = attention_mask + sparse_additive_mask
            visible = (attention_mask == 0) & selected_mask
        else:
            raise ValueError("attention_mask must be boolean or additive floating point")

        input_shape = hidden_states.shape[:-1]
        projected_q = self.q_proj(hidden_states).view(*input_shape, self.num_heads, self.head_dim * 2)
        query_states, gate = torch.chunk(projected_q, 2, dim=-1)
        gate = gate.flatten(-2)
        query_states = self.q_norm(query_states).contiguous()
        key_states = self.k_norm(self.k_proj(hidden_states).view(*input_shape, self.num_key_value_heads, self.head_dim))
        value_states = self.v_proj(hidden_states).view(*input_shape, self.num_key_value_heads, self.head_dim)
        query_states = _apply_partial_rope(query_states, positions, self.rotary_dim, self.rope_theta)
        key_states = _apply_partial_rope(key_states, positions, self.rotary_dim, self.rope_theta)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)
        attention_scores = torch.matmul(query_states.float(), key_states.float().transpose(-1, -2)) * self.scaling
        if combined_mask.dtype == torch.bool:
            attention_scores = attention_scores.masked_fill(~combined_mask, torch.finfo(attention_scores.dtype).min)
        else:
            attention_scores = attention_scores + combined_mask.to(dtype=attention_scores.dtype)
        attention_probs = torch.softmax(attention_scores, dim=-1)
        valid_queries = visible.any(dim=-1, keepdim=True)
        attention_probs = attention_probs * valid_queries.to(dtype=attention_probs.dtype)
        if self.training and self.attention_dropout:
            attention_probs = F.dropout(attention_probs, p=self.attention_dropout)
        attention_output = torch.matmul(attention_probs.to(value_states.dtype), value_states)
        attention_output = attention_output.transpose(1, 2).reshape(*input_shape, self.num_heads * self.head_dim)
        attention_output = self.o_proj(attention_output * torch.sigmoid(gate))
        output_attentions = bool(kwargs.get("output_attentions", False))
        return attention_output, attention_probs if output_attentions else None


_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_HASH_PRIME = 10007


def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    for divisor in range(3, math.isqrt(value) + 1, 2):
        if value % divisor == 0:
            return False
    return True


def _nth_prime_after(start: int, count: int) -> int:
    prime = start
    for _ in range(count):
        prime += 1
        while not _is_prime(prime):
            prime += 1
    return prime


def _ngram_hash_multipliers(vocab_size: int, ngram_size: int, layer_index: int, seed: int) -> torch.Tensor:
    multiplier_max = ((1 << 63) - 1) // max(vocab_size, 1)
    half_bound = max(1, multiplier_max // 2)
    base_seed = seed + _HASH_PRIME * layer_index
    values = []
    for index in range(ngram_size):
        value = (base_seed + _SPLITMIX_GAMMA * (index + 1)) & _MASK64
        values.append(2 * (_splitmix64(value) % half_bound) + 1)
    return torch.tensor(values, dtype=torch.long)


class CompactNGramEmbedding(nn.Module):
    """Hashed n-gram table sized for a retained tokenizer and small adaptation.

    The hash/reset mechanics follow the public Qwen4Exp reference, while
    ``bucket_count`` deliberately replaces its very large embedding allocation.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        *,
        ngram_size: int = 3,
        heads_per_ngram: int = 2,
        bucket_count: int = 8192,
        eos_token_id: int,
        layer_index: int = 0,
        seed: int = 1234,
        padding_multiple: int = 128,
    ) -> None:
        super().__init__()
        if vocab_size <= 0 or embedding_dim <= 0 or ngram_size < 2 or heads_per_ngram <= 0:
            raise ValueError("vocab_size, embedding_dim, heads_per_ngram must be positive; ngram_size must be >= 2")
        if bucket_count < 2 or layer_index < 0 or padding_multiple <= 0:
            raise ValueError("bucket_count and padding_multiple must be positive; layer_index cannot be negative")
        if not 0 <= eos_token_id < vocab_size:
            raise ValueError("eos_token_id must be inside the tokenizer vocabulary")
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.ngram_size = ngram_size
        self.context_len = ngram_size - 1
        self.heads_per_ngram = heads_per_ngram
        self.ngram_heads = (ngram_size - 1) * heads_per_ngram
        self.eos_token_id = eos_token_id
        if embedding_dim % self.ngram_heads:
            raise ValueError("embedding_dim must be divisible by the total n-gram head count")
        self.head_dim = embedding_dim // self.ngram_heads

        head_vocab_sizes = [
            _nth_prime_after(bucket_count - 1, layer_index * self.ngram_heads + head_index + 1)
            for head_index in range(self.ngram_heads)
        ]
        head_offsets = []
        total_vocab_size = 0
        for size in head_vocab_sizes:
            head_offsets.append(total_vocab_size)
            total_vocab_size += size
        padded_vocab_size = math.ceil(total_vocab_size / padding_multiple) * padding_multiple
        self.ngram_embedding = nn.Embedding(padded_vocab_size, self.head_dim)
        self.register_buffer("layer_multipliers", _ngram_hash_multipliers(vocab_size, ngram_size, layer_index, seed))
        self.register_buffer("ngram_heads_vocab_sizes", torch.tensor(head_vocab_sizes, dtype=torch.long))
        self.register_buffer("ngram_heads_offsets", torch.tensor(head_offsets, dtype=torch.long))

    def _shift_right_ignore_eos(self, token_ids: torch.Tensor, shift: int) -> torch.Tensor:
        if shift == 0:
            return token_ids
        batch_size, seq_len = token_ids.shape
        positions = torch.arange(seq_len, device=token_ids.device, dtype=torch.long)
        eos_positions = torch.where(token_ids == self.eos_token_id, positions, -1)
        previous_eos_inclusive = torch.cummax(eos_positions, dim=1).values
        previous_eos = torch.cat(
            [eos_positions.new_full((batch_size, 1), -1), previous_eos_inclusive[:, :-1]], dim=1
        )
        segment_start = previous_eos + 1
        position_in_segment = positions.unsqueeze(0) - segment_start
        source_positions = positions - shift
        gather_positions = source_positions.clamp_min(0).unsqueeze(0).expand(batch_size, -1)
        shifted = token_ids.gather(dim=1, index=gather_positions)
        valid = (position_in_segment >= shift) & (source_positions.unsqueeze(0) >= 0)
        return torch.where(valid, shifted, token_ids.new_full((), self.eos_token_id))

    def hash_ids(self, input_ids: torch.Tensor, previous_context: torch.Tensor | None = None) -> torch.Tensor:
        if input_ids.ndim != 2 or input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("input_ids must be a rank-2 integer tensor")
        if input_ids.numel() and (input_ids.min() < 0 or input_ids.max() >= self.vocab_size):
            raise ValueError("input_ids contain a token outside vocab_size")
        input_ids = input_ids.long()
        if input_ids.shape[1] == 0:
            return torch.empty(
                (input_ids.shape[0], 0, self.ngram_heads), dtype=torch.long, device=input_ids.device
            )
        if previous_context is None:
            previous_context = input_ids.new_full((input_ids.shape[0], self.context_len), self.eos_token_id)
        elif previous_context.shape != (input_ids.shape[0], self.context_len):
            raise ValueError(f"previous_context must have shape (batch, {self.context_len})")
        token_history = torch.cat([previous_context.to(input_ids.device).long(), input_ids], dim=-1)
        multipliers = self.layer_multipliers.to(input_ids.device)
        head_vocab_sizes = self.ngram_heads_vocab_sizes.to(input_ids.device)
        head_offsets = self.ngram_heads_offsets.to(input_ids.device)
        shifted_tokens = [
            self._shift_right_ignore_eos(token_history, shift) for shift in range(self.ngram_size)
        ]
        blocks = []
        for ngram in range(2, self.ngram_size + 1):
            head_start = (ngram - 2) * self.heads_per_ngram
            head_end = head_start + self.heads_per_ngram
            mixed_ids = shifted_tokens[0] * multipliers[0]
            for position in range(1, ngram):
                mixed_ids = torch.bitwise_xor(mixed_ids, shifted_tokens[position] * multipliers[position])
            vocab_sizes = head_vocab_sizes[head_start:head_end]
            offsets = head_offsets[head_start:head_end]
            ngram_ids = torch.remainder(mixed_ids.unsqueeze(-1), vocab_sizes.view(1, 1, -1))
            blocks.append(ngram_ids + offsets.view(1, 1, -1))
        return torch.cat(blocks, dim=-1)[:, -input_ids.shape[1] :]

    def update_context(self, input_ids: torch.Tensor, previous_context: torch.Tensor | None = None) -> torch.Tensor:
        """Return the fixed-size token history needed for incremental decoding."""
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be rank 2")
        if previous_context is None:
            previous_context = input_ids.new_full((input_ids.shape[0], self.context_len), self.eos_token_id)
        elif previous_context.shape != (input_ids.shape[0], self.context_len):
            raise ValueError(f"previous_context must have shape (batch, {self.context_len})")
        return torch.cat([previous_context.to(input_ids.device), input_ids.long()], dim=-1)[:, -self.context_len :]

    def forward(self, input_ids: torch.Tensor, previous_context: torch.Tensor | None = None) -> torch.Tensor:
        ngram_ids = self.hash_ids(input_ids, previous_context)
        execution_device = self.ngram_embedding.weight.device
        return self.ngram_embedding(ngram_ids.to(execution_device)).to(input_ids.device).flatten(-2)


class CompactPLELayer(nn.Module):
    """Compact, single-layer PLE prototype with neutral initialization.

    The value projection and depthwise convolution start at zero so attaching
    the feature initially leaves an existing language model's stream state
    unchanged. It currently handles full sequences; cached convolution state
    is a separate integration step.
    """

    def __init__(
        self,
        hidden_size: int,
        *,
        stream_count: int = 4,
        ple_embed_dim: int = 256,
        vocab_size: int,
        eos_token_id: int,
        ngram_size: int = 3,
        heads_per_ngram: int = 2,
        bucket_count: int = 8192,
        conv_kernel_size: int = 4,
        layer_index: int = 0,
        seed: int = 1234,
        eps: float = 1e-6,
        initializer_range: float = 0.02,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or stream_count <= 0 or ple_embed_dim <= 0 or conv_kernel_size <= 0:
            raise ValueError("hidden_size, stream_count, ple_embed_dim, and conv_kernel_size must be positive")
        if initializer_range <= 0:
            raise ValueError("initializer_range must be positive")
        self.hidden_size = hidden_size
        self.stream_count = stream_count
        self.hc_hidden_size = hidden_size * stream_count
        self.ngram_size = ngram_size
        self.short_conv_state_len = (conv_kernel_size - 1) * ngram_size
        self.ple_embedding = CompactNGramEmbedding(
            vocab_size,
            ple_embed_dim,
            ngram_size=ngram_size,
            heads_per_ngram=heads_per_ngram,
            bucket_count=bucket_count,
            eos_token_id=eos_token_id,
            layer_index=layer_index,
            seed=seed,
        )
        self.key_proj = nn.Linear(ple_embed_dim, self.hc_hidden_size, bias=False)
        self.value_proj = nn.Linear(ple_embed_dim, hidden_size, bias=False)
        self.norm_key = _GroupedRMSNorm(self.hc_hidden_size, hidden_size, eps)
        self.norm_query = _GroupedRMSNorm(self.hc_hidden_size, hidden_size, eps)
        self.norm_conv = _GroupedRMSNorm(self.hc_hidden_size, hidden_size, eps)
        self.conv1d = nn.Conv1d(
            self.hc_hidden_size,
            self.hc_hidden_size,
            kernel_size=conv_kernel_size,
            groups=self.hc_hidden_size,
            dilation=ngram_size,
            bias=False,
        )
        nn.init.normal_(self.key_proj.weight, mean=0.0, std=initializer_range)
        nn.init.zeros_(self.value_proj.weight)
        nn.init.zeros_(self.conv1d.weight)

    def forward(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("input_ids must be an integer tensor")
        if hidden_states.shape[:2] != input_ids.shape or hidden_states.shape[-1] != self.hc_hidden_size:
            raise ValueError("hidden_states and input_ids have incompatible shapes")
        input_ids = input_ids.to(device=hidden_states.device)
        if attention_mask is not None and attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must have the same shape as input_ids")
        if attention_mask is not None:
            valid = attention_mask.to(dtype=torch.bool, device=input_ids.device)
            input_ids = torch.where(valid, input_ids, input_ids.new_full((), self.ple_embedding.eos_token_id))
        else:
            valid = None

        embeddings = self.ple_embedding(input_ids)
        keys = self.norm_key(self.key_proj(embeddings)).unflatten(-1, (self.stream_count, self.hidden_size))
        values = self.value_proj(embeddings)
        queries = self.norm_query(hidden_states).unflatten(-1, (self.stream_count, self.hidden_size))
        gate = (keys * queries).sum(dim=-1, keepdim=True) / math.sqrt(self.hidden_size)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gated_value = torch.sigmoid(gate) * values.unsqueeze(-2)
        gated_value_normed = self.norm_conv(gated_value.flatten(-2))
        gated_value = gated_value.flatten(-2)
        if valid is not None:
            mask = valid.unsqueeze(-1).to(dtype=gated_value.dtype)
            gated_value = gated_value * mask
            gated_value_normed = gated_value_normed * mask

        seq_len = gated_value.shape[1]
        conv_input = gated_value_normed.transpose(1, 2)
        conv_input = F.pad(conv_input, (self.short_conv_state_len, 0))
        conv_input = conv_input[..., -(self.short_conv_state_len + seq_len) :]
        conv_output = F.silu(self.conv1d(conv_input)).transpose(1, 2)
        return gated_value + conv_output


class GatedSharedExpert(nn.Module):
    """Optional gated shared FFN added beside, not instead of, the routed MoE."""

    def __init__(
        self, hidden_size: int, intermediate_size: int = 640, *, initializer_range: float = 0.02
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or intermediate_size <= 0 or initializer_range <= 0:
            raise ValueError("hidden_size, intermediate_size, and initializer_range must be positive")
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.shared_expert_gate = nn.Linear(hidden_size, 1, bias=False)
        nn.init.normal_(self.gate_proj.weight, mean=0.0, std=initializer_range)
        nn.init.normal_(self.up_proj.weight, mean=0.0, std=initializer_range)
        nn.init.zeros_(self.down_proj.weight)
        nn.init.normal_(self.shared_expert_gate.weight, mean=0.0, std=initializer_range)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        expert = self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))
        return torch.sigmoid(self.shared_expert_gate(hidden_states)) * expert

    def add_to(self, routed_output: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        """Preserve the routed-expert result and add the shared contribution."""
        if routed_output.shape != hidden_states.shape:
            raise ValueError("routed_output and hidden_states must have the same shape")
        return routed_output + self(hidden_states)


class MultiTokenPredictionHead(nn.Module):
    """Small auxiliary head that predicts future token logits without replacing the main head."""

    def __init__(self, hidden_size: int, vocab_size: int, horizon: int = 1) -> None:
        super().__init__()
        if hidden_size <= 0 or vocab_size <= 0 or horizon <= 0:
            raise ValueError("hidden_size, vocab_size, and horizon must be positive")
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.horizon = horizon
        self.projections = nn.ModuleList(nn.Linear(hidden_size, hidden_size, bias=False) for _ in range(horizon))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_size) for _ in range(horizon))

    def forward(self, hidden_states: torch.Tensor, embedding_weight: torch.Tensor) -> torch.Tensor:
        """Return tied-embedding logits shaped ``(horizon, batch, seq, vocab)``."""
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError("hidden_states width does not match this prediction head")
        if tuple(embedding_weight.shape) != (self.vocab_size, self.hidden_size):
            raise ValueError("embedding_weight must match (vocab_size, hidden_size)")
        logits = []
        for projection, norm in zip(self.projections, self.norms, strict=True):
            future_hidden = norm(hidden_states + projection(hidden_states))
            logits.append(F.linear(future_hidden, embedding_weight))
        return torch.stack(logits, dim=0)

    @staticmethod
    def loss(
        logits: torch.Tensor, input_ids: torch.Tensor, *, ignore_index: int = -100
    ) -> torch.Tensor:
        """Compute per-horizon next-token loss, ignoring positions without future targets."""
        if logits.ndim != 4 or input_ids.ndim != 2 or logits.shape[1:3] != input_ids.shape:
            raise ValueError("logits and input_ids have incompatible batch/sequence shapes")
        horizon, _, sequence_length, vocab_size = logits.shape
        if sequence_length <= horizon:
            raise ValueError("sequence must be longer than the prediction horizon")
        losses = []
        for index in range(horizon):
            target = input_ids[:, index + 1 : sequence_length - horizon + index + 1]
            prediction = logits[index, :, : target.shape[1], :]
            losses.append(
                F.cross_entropy(
                    prediction.reshape(-1, vocab_size), target.reshape(-1), ignore_index=ignore_index
                )
            )
        return torch.stack(losses).mean()
