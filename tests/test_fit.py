import json
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from make_it_flash.fit import fit_one_layer
from make_it_flash.model import make_gdn


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
    assert metrics["steps"] == 1
    assert (output_dir / "gdn_layer_00.safetensors").is_file()
    assert (output_dir / "fit_layer_00.json").is_file()
