from types import SimpleNamespace

import torch

from make_it_flash.model import _SigmoidRMSNormGated, make_gdn


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
