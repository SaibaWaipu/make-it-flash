from types import SimpleNamespace

import pytest
import torch
from torch import nn
from safetensors.torch import load_file, save_file
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeModel

from make_it_flash.model import (
    GDNQwen3MoeAttentionAdapter,
    _SigmoidRMSNormGated,
    configure_hybrid_cache,
    create_flash_next_cache,
    flash_next_attention_schedule,
    graft_attention_modules,
    make_gdn,
    make_qsa,
)


def test_gdn_shapes_match_hidden_size():
    config = SimpleNamespace(
        model_type="qwen3_moe",
        hidden_size=64,
        vocab_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        head_dim=16,
        intermediate_size=128,
        rms_norm_eps=1e-6,
    )
    gdn = make_gdn(config, 0, key_heads=4, value_heads=8)
    result = gdn(torch.randn(1, 8, 64))
    assert result.shape == (1, 8, 64)


def test_flash_next_schedule_matches_reference_3_to_1_ratio():
    gdn_layers, qsa_layers = flash_next_attention_schedule(32)
    assert len(gdn_layers) == 24
    assert len(qsa_layers) == 8
    assert qsa_layers == tuple(range(3, 32, 4))
    assert set(gdn_layers).isdisjoint(qsa_layers)
    assert sorted((*gdn_layers, *qsa_layers)) == list(range(32))


def test_make_qsa_and_graft_attention_modules_validate_before_mutating():
    config = SimpleNamespace(model_type="qwen3_moe", hidden_size=8)
    qsa = make_qsa(
        config,
        num_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        rotary_dim=4,
        index_n_heads=2,
        index_head_dim=4,
        token_budget=4,
        compress_ratio=2,
    )
    model = Qwen3MoeModel(_tiny_qwen3_moe_config(num_hidden_layers=2))
    qsa.double()
    original = model.layers[0].self_attn
    with pytest.raises(ValueError, match="outside"):
        graft_attention_modules(model, {0: qsa, 2: nn.Identity()})
    assert model.layers[0].self_attn is original

    restored = graft_attention_modules(model, {0: qsa})
    assert restored == {0: original}
    assert model.layers[0].self_attn is qsa
    assert next(qsa.parameters()).dtype == next(original.parameters()).dtype


def test_graft_attention_modules_does_not_partially_mutate_on_placement_failure():
    class FailingPlacement(nn.Module):
        def to(self, *args, **kwargs):
            raise RuntimeError("synthetic placement failure")

    model = Qwen3MoeModel(_tiny_qwen3_moe_config(num_hidden_layers=2))
    original = [layer.self_attn for layer in model.layers]
    qsa = make_qsa(
        SimpleNamespace(model_type="qwen3_moe", hidden_size=64),
        num_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        rotary_dim=8,
        index_n_heads=2,
        index_head_dim=8,
        token_budget=4,
        compress_ratio=2,
    )

    with pytest.raises(RuntimeError, match="synthetic placement failure"):
        graft_attention_modules(model, {0: qsa, 1: FailingPlacement()})

    assert all(layer.self_attn is before for layer, before in zip(model.layers, original))


def test_graft_attention_modules_rolls_back_assignment_failure():
    class LayerWithFailingAttentionSetter(nn.Module):
        def __init__(self, attention, *, fail_assignment=False):
            super().__init__()
            self._fail_attention_assignment = False
            self.self_attn = attention
            self._fail_attention_assignment = fail_assignment

        def __setattr__(self, name, value):
            if name == "self_attn" and getattr(self, "_fail_attention_assignment", False):
                raise RuntimeError("synthetic graft assignment failure")
            super().__setattr__(name, value)

    class TinyDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(model_type="qwen3_moe")
            self.layers = nn.ModuleList(
                [
                    LayerWithFailingAttentionSetter(nn.Linear(8, 8)),
                    LayerWithFailingAttentionSetter(nn.Linear(8, 8), fail_assignment=True),
                ]
            )

    model = TinyDecoder()
    original = [layer.self_attn for layer in model.layers]
    replacements = {0: nn.Linear(8, 8), 1: nn.Linear(8, 8)}

    with pytest.raises(RuntimeError, match="synthetic graft assignment failure"):
        graft_attention_modules(model, replacements)

    assert all(layer.self_attn is before for layer, before in zip(model.layers, original))


def test_gdn_forward_at_llm_jp_41_hidden_width():
    config = SimpleNamespace(
        model_type="qwen3_moe",
        hidden_size=2560,
        vocab_size=196608,
        num_hidden_layers=32,
        num_attention_heads=40,
        head_dim=128,
        intermediate_size=7680,
        rms_norm_eps=1e-6,
    )
    gdn = make_gdn(config, 0)
    result = gdn(torch.randn(1, 2, 2560))
    assert (gdn.num_k_heads, gdn.num_v_heads) == (16, 48)
    assert (gdn.key_dim, gdn.value_dim) == (2048, 6144)
    assert gdn.norm.activation == "sigmoid"
    assert result.shape == (1, 2, 2560)


def test_qwen38_output_gate_uses_sigmoid():
    norm = _SigmoidRMSNormGated(2, eps=0.0)
    hidden = torch.ones(1, 2)
    gate = torch.zeros(1, 2)
    assert torch.allclose(norm(hidden, gate), torch.full_like(hidden, 0.5))


def _tiny_qwen3_moe_config(num_hidden_layers=1):
    return Qwen3MoeConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
    )


def test_gdn_attention_adapter_grafts_and_reloads_in_tiny_model(tmp_path):
    torch.manual_seed(17)
    config = _tiny_qwen3_moe_config()
    base_config = SimpleNamespace(**config.to_dict())
    model = Qwen3MoeModel(config).eval()
    layer = model.layers[0]
    original_mlp = layer.mlp
    original_input_norm = layer.input_layernorm
    original_post_attention_norm = layer.post_attention_layernorm

    gdn = make_gdn(base_config, 0, key_heads=4, value_heads=8).eval()
    layer.self_attn = GDNQwen3MoeAttentionAdapter(gdn)
    outer_state = {
        name: tensor.detach().clone()
        for name, tensor in layer.state_dict().items()
        if not name.startswith("self_attn.")
    }
    hidden = torch.randn(1, 3, config.hidden_size)
    input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    input_mask = torch.tensor([[1, 1, 1, 0], [0, 1, 1, 1]])
    position_ids = torch.arange(hidden.shape[1]).unsqueeze(0)

    with torch.no_grad():
        attention_result = layer.self_attn(
            hidden,
            attention_mask=None,
            position_ids=position_ids,
            position_embeddings=None,
            past_key_values=None,
            use_cache=False,
        )
        output_before_reload = layer(
            hidden,
            attention_mask=None,
            position_ids=position_ids,
            position_embeddings=None,
            past_key_values=None,
            use_cache=False,
        )
        model_output_before_reload = model(
            input_ids=input_ids, attention_mask=input_mask, use_cache=False
        ).last_hidden_state

    assert isinstance(attention_result, tuple) and attention_result[1] is None
    assert attention_result[0].shape == hidden.shape
    assert output_before_reload.shape == hidden.shape
    assert torch.isfinite(output_before_reload).all()
    assert model_output_before_reload.shape == (2, 4, config.hidden_size)
    assert torch.isfinite(model_output_before_reload).all()
    assert layer.mlp is original_mlp
    assert layer.input_layernorm is original_input_norm
    assert layer.post_attention_layernorm is original_post_attention_norm

    checkpoint = tmp_path / "gdn.safetensors"
    save_file(
        {name: value.detach().cpu().contiguous() for name, value in gdn.state_dict().items()},
        str(checkpoint),
    )
    reloaded_gdn = make_gdn(base_config, 0, key_heads=4, value_heads=8).eval()
    reloaded_gdn.load_state_dict(load_file(str(checkpoint), device="cpu"))
    layer.self_attn = GDNQwen3MoeAttentionAdapter(reloaded_gdn)

    with torch.no_grad():
        output_after_reload = layer(
            hidden,
            attention_mask=None,
            position_ids=position_ids,
            position_embeddings=None,
            past_key_values=None,
            use_cache=False,
        )
        model_output_after_reload = model(
            input_ids=input_ids, attention_mask=input_mask, use_cache=False
        ).last_hidden_state

    torch.testing.assert_close(output_before_reload, output_after_reload)
    torch.testing.assert_close(model_output_before_reload, model_output_after_reload)
    for name, tensor in layer.state_dict().items():
        if not name.startswith("self_attn."):
            torch.testing.assert_close(outer_state[name], tensor)


def test_gdn_attention_adapter_fails_closed_on_cache_and_nonstandard_masks():
    config = SimpleNamespace(
        model_type="qwen3_moe",
        hidden_size=64,
        vocab_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        head_dim=16,
        intermediate_size=128,
        rms_norm_eps=1e-6,
    )
    adapter = GDNQwen3MoeAttentionAdapter(make_gdn(config, 0, key_heads=4, value_heads=8))
    hidden = torch.randn(1, 3, 64)

    with pytest.raises(NotImplementedError, match="cache"):
        adapter(hidden, use_cache=True)

    prepared_mask = torch.tensor(
        [[[[True, False, False], [True, True, False], [True, True, False]]]]
    )
    mapped_mask = adapter._padding_mask(hidden, prepared_mask)
    assert torch.equal(mapped_mask, torch.tensor([[True, True, False]]))
    additive_mask = torch.where(
        prepared_mask, torch.tensor(0.0), torch.tensor(torch.finfo(torch.float32).min)
    )
    assert torch.equal(adapter._padding_mask(hidden, additive_mask), mapped_mask)
    with pytest.raises(NotImplementedError, match="standard causal masks"):
        adapter(hidden, attention_mask=torch.ones(1, 1, 3, 3, dtype=torch.bool))

    output, weights = adapter(hidden, attention_mask=torch.tensor([[1, 1, 0]]))
    assert output.shape == hidden.shape
    assert weights is None
    assert torch.isfinite(output).all()


def test_qsa_gdn_hybrid_cache_preserves_incremental_decode_parity():
    torch.manual_seed(61)
    config = _tiny_qwen3_moe_config(num_hidden_layers=2)
    model = Qwen3MoeModel(config).eval()
    qsa = make_qsa(
        SimpleNamespace(**config.to_dict()),
        layer_idx=0,
        num_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        rotary_dim=8,
        rope_theta=100.0,
        index_n_heads=2,
        index_head_dim=8,
        token_budget=4,
        compress_ratio=2,
    )
    gdn = GDNQwen3MoeAttentionAdapter(
        make_gdn(SimpleNamespace(**config.to_dict()), 1, key_heads=4, value_heads=8)
    )
    graft_attention_modules(model, {0: qsa, 1: gdn})
    cache = create_flash_next_cache(model, gdn_layers=[1], qsa_layers=[0])
    assert cache.layer_types == ("indexed_attention", "linear_attention")

    prefix_ids = torch.tensor([[1, 2, 3], [0, 5, 6]])
    prefix_mask = torch.tensor([[1, 1, 1], [0, 1, 1]])
    next_ids = torch.tensor([[4], [7]])
    full_ids = torch.cat((prefix_ids, next_ids), dim=1)
    full_mask = torch.cat((prefix_mask, torch.ones(2, 1, dtype=prefix_mask.dtype)), dim=1)
    with torch.no_grad():
        full_output = model(input_ids=full_ids, attention_mask=full_mask, use_cache=False).last_hidden_state
        prefix_output = model(
            input_ids=prefix_ids,
            attention_mask=prefix_mask,
            past_key_values=cache,
            use_cache=True,
        )
        cached_output = model(
            input_ids=next_ids,
            attention_mask=full_mask,
            past_key_values=prefix_output.past_key_values,
            use_cache=True,
        ).last_hidden_state

    assert torch.isfinite(cached_output).all()
    torch.testing.assert_close(cached_output, full_output[:, -1:], atol=3e-4, rtol=3e-4)


def test_gdn_adapter_uses_hybrid_dynamic_cache_for_incremental_decode():
    torch.manual_seed(29)
    config = _tiny_qwen3_moe_config(num_hidden_layers=2)
    model = Qwen3MoeModel(config)
    gdn = make_gdn(SimpleNamespace(**config.to_dict()), 0, key_heads=4, value_heads=8)
    model.layers[0].self_attn = GDNQwen3MoeAttentionAdapter(gdn)
    model.eval()

    assert configure_hybrid_cache(model, [0]) == ["linear_attention", "full_attention"]
    prefix_ids = torch.tensor([[1, 2, 3, 4], [0, 5, 6, 7]])
    prefix_mask = torch.tensor([[1, 1, 1, 1], [0, 1, 1, 1]])
    next_ids = torch.tensor([[8], [9]])
    full_ids = torch.cat((prefix_ids, next_ids), dim=1)
    full_mask = torch.cat((prefix_mask, torch.ones(2, 1, dtype=prefix_mask.dtype)), dim=1)

    with torch.no_grad():
        full_output = model(input_ids=full_ids, attention_mask=full_mask, use_cache=False).last_hidden_state
        prefix_output = model(input_ids=prefix_ids, attention_mask=prefix_mask, use_cache=True)
        cached_output = model(
            input_ids=next_ids,
            attention_mask=full_mask,
            past_key_values=prefix_output.past_key_values,
            use_cache=True,
        ).last_hidden_state

    assert cached_output.shape == (2, 1, config.hidden_size)
    assert torch.isfinite(cached_output).all()
    torch.testing.assert_close(cached_output, full_output[:, -1:], atol=2e-4, rtol=2e-4)
