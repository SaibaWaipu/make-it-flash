import torch
from safetensors.torch import load_file, save_file

from make_it_flash.full_checkpoint import (
    cast_safetensors_to_bfloat16,
    rewrite_safetensors_shard,
    safetensors_tensor_sizes,
)


def test_cast_module_checkpoint_to_bfloat16_preserves_keys_and_metadata(tmp_path):
    source = tmp_path / "module-f32.safetensors"
    converted = tmp_path / "module-bf16.safetensors"
    save_file(
        {"weight": torch.tensor([1.0, 2.0]), "index": torch.tensor([1, 2], dtype=torch.int64)},
        str(source),
        metadata={"source": "fit"},
    )
    result = cast_safetensors_to_bfloat16(
        source,
        converted,
        metadata_updates={"integrated_dtype": "BF16"},
    )
    tensors = load_file(str(converted), device="cpu")
    assert tensors["weight"].dtype == torch.bfloat16
    assert torch.equal(tensors["weight"].float(), torch.tensor([1.0, 2.0]))
    assert tensors["index"].dtype == torch.int64
    assert result["source_dtypes"] == ["torch.float32", "torch.int64"]
    assert result["output_dtypes"] == ["torch.bfloat16", "torch.int64"]


def test_rewrite_safetensors_replaces_attention_payloads_without_changing_values(tmp_path):
    source = tmp_path / "source.safetensors"
    adapter = tmp_path / "adapter.safetensors"
    output = tmp_path / "output.safetensors"
    base_tensors = {
        "model.embed_tokens.weight": torch.arange(12, dtype=torch.float32).reshape(4, 3),
        "model.layers.0.self_attn.q_proj.weight": torch.full((2, 3), 1.0),
        "model.layers.0.self_attn.o_proj.weight": torch.full((3, 2), 2.0),
        "model.layers.0.mlp.gate_proj.weight": torch.full((3, 3), 3.0),
        "model.layers.1.self_attn.q_proj.weight": torch.full((2, 3), 4.0),
    }
    fitted_tensors = {
        "q_proj.weight": torch.full((2, 3), 9.0),
        "indexer.proj.weight": torch.full((2, 3), 10.0),
    }
    save_file(base_tensors, str(source), metadata={"format": "pt", "base": "pinned"})
    save_file(fitted_tensors, str(adapter), metadata={"layer": "0"})

    result = rewrite_safetensors_shard(
        source,
        output,
        remove_prefixes=("model.layers.0.self_attn.",),
        additions=((adapter, "model.layers.0.self_attn."),),
        metadata_updates={"integrated": "true"},
    )

    merged = load_file(str(output), device="cpu")
    assert set(merged) == {
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.indexer.proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.1.self_attn.q_proj.weight",
    }
    assert torch.equal(merged["model.layers.0.self_attn.q_proj.weight"], fitted_tensors["q_proj.weight"])
    assert torch.equal(merged["model.layers.0.self_attn.indexer.proj.weight"], fitted_tensors["indexer.proj.weight"])
    assert torch.equal(merged["model.embed_tokens.weight"], base_tensors["model.embed_tokens.weight"])
    assert set(result["removed_keys"]) == {
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
    }
    assert set(result["added_keys"]) == {
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.indexer.proj.weight",
    }
    assert result["sha256"]
    assert result["tensor_count"] == len(merged)
    assert result["removed_tensor_bytes"] == 48
    assert result["added_tensor_bytes"] == 48
    assert result["source_tensor_bytes"] == 156
    assert result["tensor_bytes"] == 156
    assert safetensors_tensor_sizes(output)["model.layers.0.self_attn.q_proj.weight"] == 24


def test_rewrite_safetensors_rejects_duplicate_keys_and_overwrite(tmp_path):
    source = tmp_path / "source.safetensors"
    adapter = tmp_path / "adapter.safetensors"
    output = tmp_path / "output.safetensors"
    save_file({"keep.weight": torch.ones(1)}, str(source))
    save_file({"keep.weight": torch.zeros(1)}, str(adapter))

    try:
        rewrite_safetensors_shard(
            source,
            output,
            remove_prefixes=("drop.",),
            additions=((adapter, ""),),
        )
    except ValueError as error:
        assert "duplicate tensor key" in str(error)
    else:
        raise AssertionError("duplicate tensor key must be rejected")

    rewrite_safetensors_shard(source, output, remove_prefixes=("drop.",))
    try:
        rewrite_safetensors_shard(source, output, remove_prefixes=("drop.",))
    except FileExistsError:
        pass
    else:
        raise AssertionError("existing output file must not be overwritten")
