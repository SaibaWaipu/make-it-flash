import torch
import pytest
from transformers import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeModel

from make_it_flash.cache import _capture_local_attention


def _tiny_model():
    config = Qwen3MoeConfig(
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
    config._attn_implementation = "sdpa"
    return Qwen3MoeModel(config).eval()


def test_selective_attention_capture_returns_compact_block_mass_and_restores_backend():
    torch.manual_seed(71)
    model = _tiny_model()
    attention = model.layers[0].self_attn
    original_forward = attention.forward
    input_ids = torch.tensor([[1, 2, 3, 4]])

    captures = _capture_local_attention(model, input_ids, (0,), (0,))

    assert captures[0]["input"].shape == (4, 64)
    assert captures[0]["target"].shape == (4, 64)
    assert captures[0]["block_mass"].shape == (4, 1)
    assert torch.isfinite(captures[0]["block_mass"]).all()
    assert torch.all(captures[0]["block_mass"] >= 0)
    torch.testing.assert_close(
        captures[0]["block_mass"][:, 0], torch.full((4,), 4.0, dtype=torch.bfloat16),
        atol=0.02, rtol=0.02,
    )
    assert model.config._attn_implementation == "sdpa"
    assert attention.forward == original_forward


def test_selective_attention_capture_without_maps_keeps_sdpa_backend():
    model = _tiny_model()
    captures = _capture_local_attention(model, torch.tensor([[1, 2, 3]]), (0,), ())
    assert set(captures[0]) == {"input", "target"}
    assert model.config._attn_implementation == "sdpa"


def test_selective_attention_capture_requires_selected_layer_subset():
    model = _tiny_model()
    with pytest.raises(ValueError, match="subset"):
        _capture_local_attention(model, torch.tensor([[1, 2]]), (0,), (1,))
