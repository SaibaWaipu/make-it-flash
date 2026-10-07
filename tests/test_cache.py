import json
from types import SimpleNamespace

import torch
import pytest
from transformers import AutoConfig, AutoModelForCausalLM, Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeModel

import make_it_flash.cache as cache_module
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


def test_qsa_cache_preflight_rejects_short_data_before_gpu_or_model_load(tmp_path, monkeypatch):
    data_file = tmp_path / "calibration.jsonl"
    data_file.write_text(
        json.dumps({"sample_id": "short-1", "input_ids": [1, 2, 3]})
        + "\n"
        + json.dumps({"sample_id": "short-2", "input_ids": [4, 5, 6]})
        + "\n",
        encoding="utf-8",
    )
    data_file.with_name("data_manifest.json").write_text(
        json.dumps({"model_id": "fixture/model", "model_revision": "fixture-sha"}),
        encoding="utf-8",
    )
    output_dir = tmp_path / "qsa-cache"
    monkeypatch.setattr(
        AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: pytest.fail("QSA length preflight must happen before config/model loading"),
    )
    monkeypatch.setattr(
        cache_module,
        "_check_gpu_memory",
        lambda *_: pytest.fail("QSA length preflight must happen before GPU checks"),
    )

    with pytest.raises(ValueError, match="needs at least two sequences of 2052 tokens"):
        cache_module.cache_teacher_outputs(
            data_file=data_file,
            output_dir=output_dir,
            layers=(3,),
            attention_layers=(3,),
        )

    assert not output_dir.exists()


def test_prune_stale_cache_shards_keeps_only_current_sequence_range(tmp_path):
    for name in (
        "sample_000000.safetensors",
        "sample_000001.safetensors",
        "sample_000002.safetensors",
        "sample_unexpected.safetensors",
    ):
        (tmp_path / name).write_bytes(b"fixture")

    cache_module._prune_stale_cache_shards(tmp_path, emitted=2)

    assert (tmp_path / "sample_000000.safetensors").is_file()
    assert (tmp_path / "sample_000001.safetensors").is_file()
    assert not (tmp_path / "sample_000002.safetensors").exists()
    assert not (tmp_path / "sample_unexpected.safetensors").exists()


def test_failed_cache_overwrite_invalidates_old_manifest(tmp_path, monkeypatch):
    data_file = tmp_path / "calibration.jsonl"
    data_file.write_text(json.dumps({"sample_id": "new", "input_ids": [1, 2]}) + "\n", encoding="utf-8")
    data_file.with_name("data_manifest.json").write_text(
        json.dumps({"model_id": "fixture/model", "model_revision": "fixture-sha"}),
        encoding="utf-8",
    )
    output_dir = tmp_path / "cache"
    output_dir.mkdir()
    marker = output_dir / "cache_manifest.json"
    marker.write_text(json.dumps({"model_id": "old/model", "model_revision": "old-sha"}), encoding="utf-8")
    old_shard = output_dir / "sample_000000.safetensors"
    old_shard.write_bytes(b"old cached sample")

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

    class FakeTeacher:
        def __init__(self):
            self.config = config
            self.layers = [SimpleNamespace(self_attn=torch.nn.Linear(64, 64))]
            self.embedding = torch.nn.Embedding(64, 64)

        def eval(self):
            return self

        def get_input_embeddings(self):
            return self.embedding

    monkeypatch.setattr(AutoConfig, "from_pretrained", lambda *args, **kwargs: config)
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", lambda *args, **kwargs: FakeTeacher())
    monkeypatch.setattr(cache_module, "_check_gpu_memory", lambda *_: None)
    monkeypatch.setattr(
        cache_module,
        "_capture_local_attention",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("synthetic capture interruption")),
    )

    with pytest.raises(RuntimeError, match="synthetic capture interruption"):
        cache_module.cache_teacher_outputs(
            data_file=data_file,
            output_dir=output_dir,
            layers=(0,),
            overwrite=True,
        )

    assert not marker.exists()
    assert old_shard.is_file()
