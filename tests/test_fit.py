import json
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from make_it_flash.fit import fit_one_layer
from make_it_flash.model import make_gdn
from make_it_flash.provenance import checkpoint_provenance, sha256_file


def test_fit_writes_standalone_checkpoint(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "fit"
    cache_dir.mkdir()
    manifest = {
        "model_id": "fixture/model",
        "model_revision": "fixture-sha",
        "layers": [0],
        "base_config": {
            "model_type": "qwen3_moe",
            "hidden_size": 64,
            "vocab_size": 128,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "head_dim": 16,
            "rms_norm_eps": 1e-6,
        },
        "gdn_config": {"linear_num_key_heads": 4, "linear_num_value_heads": 8},
    }
    (cache_dir / "cache_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    teacher = make_gdn(SimpleNamespace(**manifest["base_config"]), 0, key_heads=4, value_heads=8)
    teacher.eval()
    for index in range(2):
        x = torch.randn(1, 6, 64)
        with torch.no_grad():
            y = teacher(x)
        save_file(
            {
                "layer_00_input": x[0].to(torch.bfloat16).contiguous(),
                "layer_00_target": y[0].to(torch.bfloat16).contiguous(),
            },
            str(cache_dir / f"sample_{index:06d}.safetensors"),
            metadata={"sample_id": str(index), "config": "fixture"},
        )

    metrics = fit_one_layer(
        cache_dir=cache_dir,
        output_dir=output_dir,
        layer=0,
        epochs=1,
        max_steps=1,
        validation_fraction=0,
        allow_cpu=True,
    )
    checkpoint = output_dir / "gdn_layer_00.safetensors"
    assert metrics["steps"] == 1
    assert checkpoint.is_file()
    assert (output_dir / "fit_layer_00.json").is_file()
    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
    assert all(metadata[key] == value for key, value in checkpoint_provenance().items())
    assert metrics["checkpoint_sha256"] == sha256_file(checkpoint)


def test_fit_one_step_at_llm_jp_41_width(tmp_path: Path):
    cache_dir = tmp_path / "cache-llmjp41"
    output_dir = tmp_path / "fit-llmjp41"
    cache_dir.mkdir()
    manifest = {
        "model_id": "llm-jp/llm-jp-4.1-32b-a3b-thinking",
        "model_revision": "cda260706786758045e5e96bf4d738bbc01155b5",
        "layers": [0],
        "base_config": {
            "model_type": "qwen3_moe",
            "hidden_size": 2560,
            "vocab_size": 196608,
            "num_hidden_layers": 32,
            "num_attention_heads": 40,
            "head_dim": 128,
            "intermediate_size": 7680,
            "rms_norm_eps": 1e-6,
        },
        "gdn_config": {"linear_num_key_heads": 16, "linear_num_value_heads": 48},
    }
    (cache_dir / "cache_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    save_file(
        {
            "layer_00_input": torch.randn(2, 2560).to(torch.bfloat16).contiguous(),
            "layer_00_target": torch.randn(2, 2560).to(torch.bfloat16).contiguous(),
        },
        str(cache_dir / "sample_000000.safetensors"),
        metadata={"sample_id": "synthetic-4.1-width", "config": "synthetic"},
    )

    metrics = fit_one_layer(
        cache_dir=cache_dir,
        output_dir=output_dir,
        layer=0,
        epochs=1,
        max_steps=1,
        validation_fraction=0,
        allow_cpu=True,
    )
    assert metrics["steps"] == 1
    assert metrics["model_id"] == "llm-jp/llm-jp-4.1-32b-a3b-thinking"
    assert (output_dir / "gdn_layer_00.safetensors").is_file()
