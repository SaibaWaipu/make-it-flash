from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeModel

from make_it_flash.model import GDNQwen3MoeAttentionAdapter, _SigmoidRMSNormGated, make_gdn


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


def _tiny_qwen3_moe_config():
    return Qwen3MoeConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
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
    input_ids = torch.tensor([[1, 2, 3]])
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
            input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False
        ).last_hidden_state

    assert isinstance(attention_result, tuple) and attention_result[1] is None
    assert attention_result[0].shape == hidden.shape
    assert output_before_reload.shape == hidden.shape
    assert torch.isfinite(output_before_reload).all()
    assert model_output_before_reload.shape == hidden.shape
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
            input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False
        ).last_hidden_state

    torch.testing.assert_close(output_before_reload, output_after_reload)
    torch.testing.assert_close(model_output_before_reload, model_output_after_reload)
    for name, tensor in layer.state_dict().items():
        if not name.startswith("self_attn."):
            torch.testing.assert_close(outer_state[name], tensor)


def test_gdn_attention_adapter_fails_closed_on_cache_and_causal_masks():
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
    with pytest.raises(NotImplementedError, match="2-D padding mask"):
        adapter(hidden, attention_mask=torch.zeros(1, 1, 3, 3))

    output, weights = adapter(hidden, attention_mask=torch.tensor([[1, 1, 0]]))
    assert output.shape == hidden.shape
    assert weights is None
    assert torch.isfinite(output).all()
