import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

import make_it_flash.fit_qsa as fit_qsa_module
from make_it_flash.fit_qsa import fit_qsa_layer
from make_it_flash.provenance import checkpoint_provenance, sha256_file


QSA_CONFIG = {
    "num_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "rotary_dim": 4,
    "rope_theta": 100.0,
    "index_n_heads": 2,
    "index_head_dim": 4,
    "token_budget": 4,
    "compress_ratio": 2,
}


def _write_qsa_cache_example(cache_dir: Path, index: int, seq_len: int):
    num_blocks = seq_len // QSA_CONFIG["compress_ratio"]
    save_file(
        {
            "layer_00_input": torch.randn(seq_len, 8).to(torch.bfloat16).contiguous(),
            "layer_00_target": torch.randn(seq_len, 8).to(torch.bfloat16).contiguous(),
            "layer_00_block_mass": torch.rand(seq_len, num_blocks).to(torch.bfloat16).contiguous(),
        },
        str(cache_dir / f"sample_{index:06d}.safetensors"),
        metadata={"sample_id": str(index), "config": "fixture"},
    )


def test_fit_qsa_layer_writes_standalone_checkpoint_from_teacher_maps(tmp_path: Path, monkeypatch):
    torch.manual_seed(73)
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "fit-qsa"
    cache_dir.mkdir()
    manifest = {
        "model_id": "fixture/qwen3-moe",
        "model_revision": "fixture-sha",
        "layers": [0],
        "attention_layers": [0],
        "base_config": {"model_type": "qwen3_moe", "hidden_size": 8},
        "qsa_config": QSA_CONFIG,
    }
    (cache_dir / "cache_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    causal_mask = torch.ones(1, 1, 6, 6, dtype=torch.bool).tril()
    for index in range(2):
        hidden = torch.randn(1, 6, 8)
        target = torch.randn(1, 6, 8)
        teacher_attention = torch.softmax(
            torch.randn(1, 4, 6, 6).masked_fill(~causal_mask, -1e4), dim=-1
        )
        teacher_block_mass = teacher_attention.sum(dim=1)[..., :6].view(1, 6, 3, 2).sum(dim=-1)
        save_file(
            {
                "layer_00_input": hidden[0].to(torch.bfloat16).contiguous(),
                "layer_00_target": target[0].to(torch.bfloat16).contiguous(),
                "layer_00_block_mass": teacher_block_mass[0].to(torch.bfloat16).contiguous(),
            },
            str(cache_dir / f"sample_{index:06d}.safetensors"),
            metadata={"sample_id": str(index), "config": "fixture"},
        )

    monkeypatch.setattr(
        fit_qsa_module,
        "_split_value",
        lambda path: 0 if path.name == "sample_000000.safetensors" else 2**64 - 1,
    )
    metrics = fit_qsa_layer(
        cache_dir=cache_dir,
        output_dir=output_dir,
        layer=0,
        epochs=1,
        max_steps=1,
        validation_fraction=0.49,
        allow_cpu=True,
    )

    assert metrics["steps"] == 1
    assert metrics["validation_uses_train_fallback"] is False
    assert metrics["num_validation_sequences"] == 1
    assert metrics["selector_pruning_examples"] == 1
    assert metrics["selector_pruning_validation_examples"] == 1
    assert metrics["initial_validation"]["selection_loss"] >= 0
    checkpoint = output_dir / "qsa_layer_00.safetensors"
    assert checkpoint.is_file()
    assert (output_dir / "fit_qsa_layer_00.json").is_file()
    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
    assert all(metadata[key] == value for key, value in checkpoint_provenance().items())
    assert metrics["checkpoint_sha256"] == sha256_file(checkpoint)


def test_fit_qsa_preflights_training_sequence_pruning_length(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "out"
    cache_dir.mkdir()
    manifest = {
        "model_id": "fixture/model",
        "model_revision": "fixture",
        "layers": [0],
        "attention_layers": [0],
        "base_config": {"model_type": "qwen3_moe", "hidden_size": 8},
        "qsa_config": QSA_CONFIG,
    }
    (cache_dir / "cache_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    _write_qsa_cache_example(cache_dir, 0, seq_len=5)

    import pytest

    with pytest.raises(ValueError, match="no QSA training sequence reaches 6 tokens"):
        fit_qsa_layer(
            cache_dir=cache_dir,
            output_dir=output_dir,
            layer=0,
            validation_fraction=0,
            allow_cpu=True,
        )
    assert not output_dir.exists()


def test_fit_qsa_preflights_independent_validation_pruning_length(tmp_path: Path, monkeypatch):
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "out"
    cache_dir.mkdir()
    manifest = {
        "model_id": "fixture/model",
        "model_revision": "fixture",
        "layers": [0],
        "attention_layers": [0],
        "base_config": {"model_type": "qwen3_moe", "hidden_size": 8},
        "qsa_config": QSA_CONFIG,
    }
    (cache_dir / "cache_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    _write_qsa_cache_example(cache_dir, 0, seq_len=5)
    _write_qsa_cache_example(cache_dir, 1, seq_len=6)
    monkeypatch.setattr(
        fit_qsa_module,
        "_split_value",
        lambda path: 0 if path.name == "sample_000000.safetensors" else 2**64 - 1,
    )

    import pytest

    with pytest.raises(ValueError, match="no independent QSA validation sequence reaches 6 tokens"):
        fit_qsa_layer(
            cache_dir=cache_dir,
            output_dir=output_dir,
            layer=0,
            validation_fraction=0.49,
            allow_cpu=True,
        )
    assert not output_dir.exists()


def test_fit_qsa_preflights_missing_independent_validation_split(tmp_path: Path, monkeypatch):
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "out"
    cache_dir.mkdir()
    manifest = {
        "model_id": "fixture/model",
        "model_revision": "fixture",
        "layers": [0],
        "attention_layers": [0],
        "base_config": {"model_type": "qwen3_moe", "hidden_size": 8},
        "qsa_config": QSA_CONFIG,
    }
    (cache_dir / "cache_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    _write_qsa_cache_example(cache_dir, 0, seq_len=6)
    monkeypatch.setattr(fit_qsa_module, "_split_value", lambda _path: 0)

    import pytest

    with pytest.raises(ValueError, match="no independent QSA validation examples"):
        fit_qsa_layer(
            cache_dir=cache_dir,
            output_dir=output_dir,
            layer=0,
            validation_fraction=0.1,
            allow_cpu=True,
        )
    assert not output_dir.exists()


def test_fit_qsa_requires_positive_selector_loss_weight(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    manifest = {
        "model_id": "fixture/model",
        "model_revision": "fixture",
        "layers": [0],
        "attention_layers": [0],
        "base_config": {"model_type": "qwen3_moe", "hidden_size": 8},
        "qsa_config": QSA_CONFIG,
    }
    (cache_dir / "cache_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    import pytest

    with pytest.raises(ValueError, match="selector_loss_weight must all be positive"):
        fit_qsa_layer(
            cache_dir=cache_dir,
            output_dir=tmp_path / "out",
            layer=0,
            selector_loss_weight=0,
            allow_cpu=True,
        )


def test_fit_qsa_requires_teacher_attention_maps(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    manifest = {
        "model_id": "fixture/model",
        "model_revision": "fixture",
        "layers": [0],
        "attention_layers": [],
        "base_config": {"model_type": "qwen3_moe", "hidden_size": 8},
        "qsa_config": QSA_CONFIG,
    }
    (cache_dir / "cache_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    import pytest

    with pytest.raises(ValueError, match="block masses"):
        fit_qsa_layer(cache_dir=cache_dir, output_dir=tmp_path / "out", layer=0, allow_cpu=True)
