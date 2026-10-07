import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM, Qwen3MoeModel

from make_it_flash.model import (
    GDNQwen3MoeAttentionAdapter,
    create_flash_next_cache,
    flash_next_attention_schedule,
    make_gdn,
    make_qsa,
)
from make_it_flash.overlay import assemble_flash_next_overlay, load_flash_next_overlay
from make_it_flash.provenance import checkpoint_provenance, sha256_file
from make_it_flash.flash_next import QSAQwen3MoeAttentionAdapter


MODEL_ID = "fixture/llm-jp-qwen3-moe"
MODEL_REVISION = "a" * 40
GDN_CONFIG = {
    "linear_key_head_dim": 16,
    "linear_value_head_dim": 16,
    "linear_num_key_heads": 4,
    "linear_num_value_heads": 8,
    "linear_conv_kernel_dim": 4,
}
QSA_CONFIG = {
    "num_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "rotary_dim": 8,
    "rope_theta": 100.0,
    "index_n_heads": 2,
    "index_head_dim": 8,
    "token_budget": 4,
    "compress_ratio": 2,
}


def _tiny_config():
    config = Qwen3MoeConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
    )
    config._name_or_path = MODEL_ID
    config._commit_hash = MODEL_REVISION
    return config


def _write_fit_artifacts(gdn_dir: Path, qsa_dir: Path):
    base_config = SimpleNamespace(**_tiny_config().to_dict())
    base_config._name_or_path = MODEL_ID
    base_config._commit_hash = MODEL_REVISION
    gdn_layers, qsa_layers = flash_next_attention_schedule(base_config.num_hidden_layers)
    for layer in gdn_layers:
        module = make_gdn(base_config, layer, key_heads=4, value_heads=8)
        checkpoint_path = gdn_dir / f"gdn_layer_{layer:02d}.safetensors"
        save_file(
            {name: value.detach().contiguous() for name, value in module.state_dict().items()},
            str(checkpoint_path),
            metadata={
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "teacher_layer": str(layer),
                "module": "transformers.models.qwen3_next.Qwen3NextGatedDeltaNet",
                "gdn_config": json.dumps(GDN_CONFIG, sort_keys=True),
                **checkpoint_provenance(),
            },
        )
        (gdn_dir / f"fit_layer_{layer:02d}.json").write_text(
            json.dumps(
                {
                    "model_id": MODEL_ID,
                    "model_revision": MODEL_REVISION,
                    "teacher_layer": layer,
                    "steps": 2,
                    "num_train_sequences": 10,
                    "num_validation_sequences": 2,
                    "validation_uses_train_fallback": False,
                    "initial_validation_mse": 1.0,
                    "best_validation_mse": 0.8,
                    "checkpoint": checkpoint_path.name,
                    "checkpoint_sha256": sha256_file(checkpoint_path),
                }
            ),
            encoding="utf-8",
        )
    for layer in qsa_layers:
        module = make_qsa(base_config, layer_idx=layer, **QSA_CONFIG)
        checkpoint_path = qsa_dir / f"qsa_layer_{layer:02d}.safetensors"
        save_file(
            {name: value.detach().contiguous() for name, value in module.state_dict().items()},
            str(checkpoint_path),
            metadata={
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "teacher_layer": str(layer),
                "module": "make_it_flash.QSAQwen3MoeAttentionAdapter",
                "qsa_config": json.dumps(QSA_CONFIG, sort_keys=True),
                **checkpoint_provenance(),
            },
        )
        (qsa_dir / f"fit_qsa_layer_{layer:02d}.json").write_text(
            json.dumps(
                {
                    "model_id": MODEL_ID,
                    "model_revision": MODEL_REVISION,
                    "teacher_layer": layer,
                    "steps": 2,
                    "num_train_sequences": 10,
                    "num_validation_sequences": 2,
                    "validation_uses_train_fallback": False,
                    "selector_loss_weight": 0.1,
                    "initial_validation": {"total_loss": 1.0, "selection_loss": 1.0},
                    "best_validation_total_loss": 0.8,
                    "final_validation": {"total_loss": 0.8, "selection_loss": 0.8},
                    "selected_checkpoint_validation_selection_loss": 0.8,
                    "selector_pruning_examples": 3,
                    "selector_pruning_validation_examples": 1,
                    "checkpoint": checkpoint_path.name,
                    "checkpoint_sha256": sha256_file(checkpoint_path),
                }
            ),
            encoding="utf-8",
        )
    return base_config


def _assemble(tmp_path: Path):
    gdn_dir = tmp_path / "gdn-fits"
    qsa_dir = tmp_path / "qsa-fits"
    overlay_dir = tmp_path / "overlay"
    gdn_dir.mkdir()
    qsa_dir.mkdir()
    base_config = _write_fit_artifacts(gdn_dir, qsa_dir)
    manifest = assemble_flash_next_overlay(
        base_config=base_config,
        base_model_id=MODEL_ID,
        base_model_revision=MODEL_REVISION,
        gdn_fit_dir=gdn_dir,
        qsa_fit_dir=qsa_dir,
        output_dir=overlay_dir,
    )
    return overlay_dir, manifest


@pytest.mark.parametrize(
    ("attribute", "value"),
    [("_name_or_path", "wrong/model"), ("_commit_hash", "b" * 40)],
)
def test_overlay_assembly_requires_verified_base_config(tmp_path, attribute, value):
    gdn_dir = tmp_path / "gdn-fits"
    qsa_dir = tmp_path / "qsa-fits"
    gdn_dir.mkdir()
    qsa_dir.mkdir()
    base_config = _write_fit_artifacts(gdn_dir, qsa_dir)
    setattr(base_config, attribute, value)

    with pytest.raises(ValueError, match="base config (source|revision) is unverified"):
        assemble_flash_next_overlay(
            base_config=base_config,
            base_model_id=MODEL_ID,
            base_model_revision=MODEL_REVISION,
            gdn_fit_dir=gdn_dir,
            qsa_fit_dir=qsa_dir,
            output_dir=tmp_path / "overlay",
        )


def test_overlay_assembly_refuses_unimproved_fit_metrics(tmp_path):
    gdn_dir = tmp_path / "gdn-fits"
    qsa_dir = tmp_path / "qsa-fits"
    overlay_dir = tmp_path / "overlay"
    gdn_dir.mkdir()
    qsa_dir.mkdir()
    base_config = _write_fit_artifacts(gdn_dir, qsa_dir)
    metrics_path = gdn_dir / "fit_layer_00.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["best_validation_mse"] = metrics["initial_validation_mse"]
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")

    with pytest.raises(ValueError, match="did not improve"):
        assemble_flash_next_overlay(
            base_config=base_config,
            base_model_id=MODEL_ID,
            base_model_revision=MODEL_REVISION,
            gdn_fit_dir=gdn_dir,
            qsa_fit_dir=qsa_dir,
            output_dir=overlay_dir,
        )

    assert not overlay_dir.exists()


def test_overlay_assembly_requires_qsa_pruning_evidence(tmp_path):
    gdn_dir = tmp_path / "gdn-fits"
    qsa_dir = tmp_path / "qsa-fits"
    overlay_dir = tmp_path / "overlay"
    gdn_dir.mkdir()
    qsa_dir.mkdir()
    base_config = _write_fit_artifacts(gdn_dir, qsa_dir)
    metrics_path = qsa_dir / "fit_qsa_layer_03.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["selector_pruning_examples"] = 0
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")

    with pytest.raises(ValueError, match="top-k pruning"):
        assemble_flash_next_overlay(
            base_config=base_config,
            base_model_id=MODEL_ID,
            base_model_revision=MODEL_REVISION,
            gdn_fit_dir=gdn_dir,
            qsa_fit_dir=qsa_dir,
            output_dir=overlay_dir,
        )

    assert not overlay_dir.exists()


def test_overlay_assembly_requires_validation_topk_pruning_evidence(tmp_path):
    gdn_dir = tmp_path / "gdn-fits"
    qsa_dir = tmp_path / "qsa-fits"
    gdn_dir.mkdir()
    qsa_dir.mkdir()
    base_config = _write_fit_artifacts(gdn_dir, qsa_dir)
    metrics_path = qsa_dir / "fit_qsa_layer_03.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["selector_pruning_validation_examples"] = 0
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")

    with pytest.raises(ValueError, match="independent validation example"):
        assemble_flash_next_overlay(
            base_config=base_config,
            base_model_id=MODEL_ID,
            base_model_revision=MODEL_REVISION,
            gdn_fit_dir=gdn_dir,
            qsa_fit_dir=qsa_dir,
            output_dir=tmp_path / "overlay",
        )


@pytest.mark.parametrize(
    ("selector_weight", "selected_loss", "match"),
    [
        (0.0, 0.8, "positive selector_loss_weight"),
        (0.1, 1.0, "selector validation loss did not improve"),
    ],
)
def test_overlay_assembly_rejects_untrained_qsa_selector(
    tmp_path, selector_weight, selected_loss, match
):
    gdn_dir = tmp_path / "gdn-fits"
    qsa_dir = tmp_path / "qsa-fits"
    overlay_dir = tmp_path / "overlay"
    gdn_dir.mkdir()
    qsa_dir.mkdir()
    base_config = _write_fit_artifacts(gdn_dir, qsa_dir)
    metrics_path = qsa_dir / "fit_qsa_layer_03.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["selector_loss_weight"] = selector_weight
    metrics["final_validation"]["selection_loss"] = selected_loss
    metrics["selected_checkpoint_validation_selection_loss"] = selected_loss
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        assemble_flash_next_overlay(
            base_config=base_config,
            base_model_id=MODEL_ID,
            base_model_revision=MODEL_REVISION,
            gdn_fit_dir=gdn_dir,
            qsa_fit_dir=qsa_dir,
            output_dir=overlay_dir,
        )

    assert not overlay_dir.exists()


def test_four_layer_overlay_round_trip_grafts_full_schedule_and_preserves_causal_lm_assets(tmp_path):
    overlay_dir, manifest = _assemble(tmp_path)
    config = _tiny_config()
    model = Qwen3MoeForCausalLM(config).eval()
    decoder_layers = model.model.layers
    non_attention_before = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
        if ".self_attn." not in name
    }
    mlps = [layer.mlp for layer in decoder_layers]
    input_embedding = model.get_input_embeddings().weight.detach().clone()
    output_embedding = model.get_output_embeddings().weight.detach().clone()
    vocab_size = model.config.vocab_size

    cache = load_flash_next_overlay(
        model,
        overlay_dir,
        base_model_id=MODEL_ID,
        base_model_revision=MODEL_REVISION,
    )

    assert manifest["schedule"] == {"gdn_layers": [0, 1, 2], "qsa_layers": [3]}
    assert cache.layer_types == ("linear_attention",) * 3 + ("indexed_attention",)
    assert isinstance(decoder_layers[0].self_attn, GDNQwen3MoeAttentionAdapter)
    assert isinstance(decoder_layers[3].self_attn, QSAQwen3MoeAttentionAdapter)
    assert [layer.mlp for layer in decoder_layers] == mlps
    assert model.config.vocab_size == vocab_size
    torch.testing.assert_close(model.get_input_embeddings().weight, input_embedding)
    torch.testing.assert_close(model.get_output_embeddings().weight, output_embedding)
    for name, value in non_attention_before.items():
        torch.testing.assert_close(model.state_dict()[name], value)

    input_ids = torch.tensor([[1, 2, 3, 4]])
    with pytest.raises(NotImplementedError, match="FlashNextDynamicCache"):
        model(input_ids=input_ids[:, :3], use_cache=True)

    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids[:, :3],
            past_key_values=cache,
            max_new_tokens=2,
            do_sample=False,
            use_cache=True,
        )
        fresh_cache = create_flash_next_cache(
            model, gdn_layers=(0, 1, 2), qsa_layers=(3,)
        )
        result = model(input_ids=input_ids, use_cache=False).logits
        prefix = model(input_ids=input_ids[:, :3], past_key_values=fresh_cache, use_cache=True)
        cached = model(
            input_ids=input_ids[:, 3:],
            past_key_values=prefix.past_key_values,
            use_cache=True,
        ).logits
    assert generated.shape == (1, 5)
    assert result.shape == (1, 4, vocab_size)
    assert torch.isfinite(result).all()
    torch.testing.assert_close(cached, result[:, -1:], atol=3e-4, rtol=3e-4)


def test_overlay_loader_rejects_revision_mismatch_without_mutating_model(tmp_path):
    overlay_dir, _ = _assemble(tmp_path)
    model = Qwen3MoeModel(_tiny_config()).eval()
    original = [layer.self_attn for layer in model.layers]

    with pytest.raises(ValueError, match="ID/revision"):
        load_flash_next_overlay(
            model,
            overlay_dir,
            base_model_id=MODEL_ID,
            base_model_revision="b" * 40,
        )

    assert all(layer.self_attn is before for layer, before in zip(model.layers, original))


@pytest.mark.parametrize("bad_revision", [None, "b" * 40])
def test_overlay_loader_rejects_unverified_loaded_revision(tmp_path, bad_revision):
    overlay_dir, _ = _assemble(tmp_path)
    config = _tiny_config()
    config._commit_hash = bad_revision
    model = Qwen3MoeModel(config).eval()
    original = [layer.self_attn for layer in model.layers]

    with pytest.raises(ValueError, match="base config revision is unverified"):
        load_flash_next_overlay(
            model,
            overlay_dir,
            base_model_id=MODEL_ID,
            base_model_revision=MODEL_REVISION,
        )

    assert all(layer.self_attn is before for layer, before in zip(model.layers, original))


def test_overlay_overwrite_builds_off_to_the_side_before_replacing(tmp_path, monkeypatch):
    import make_it_flash.overlay as overlay_module

    gdn_dir = tmp_path / "gdn-fits"
    qsa_dir = tmp_path / "qsa-fits"
    overlay_dir = tmp_path / "overlay"
    gdn_dir.mkdir()
    qsa_dir.mkdir()
    base_config = _write_fit_artifacts(gdn_dir, qsa_dir)
    first_manifest = assemble_flash_next_overlay(
        base_config=base_config,
        base_model_id=MODEL_ID,
        base_model_revision=MODEL_REVISION,
        gdn_fit_dir=gdn_dir,
        qsa_fit_dir=qsa_dir,
        output_dir=overlay_dir,
    )
    manifest_path = overlay_dir / "flash_next_overlay.json"
    original_manifest = manifest_path.read_bytes()
    original_gdn_file = overlay_dir / first_manifest["modules"]["gdn"]["0"]["file"]
    original_qsa_file = overlay_dir / first_manifest["modules"]["qsa"]["3"]["file"]
    real_copyfile = overlay_module.shutil.copyfile

    def fail_during_qsa_copy(source, destination):
        if Path(destination).parent.name == "qsa":
            raise OSError("synthetic interrupted copy")
        return real_copyfile(source, destination)

    monkeypatch.setattr(overlay_module.shutil, "copyfile", fail_during_qsa_copy)
    with pytest.raises(OSError, match="interrupted copy"):
        assemble_flash_next_overlay(
            base_config=base_config,
            base_model_id=MODEL_ID,
            base_model_revision=MODEL_REVISION,
            gdn_fit_dir=gdn_dir,
            qsa_fit_dir=qsa_dir,
            output_dir=overlay_dir,
            overwrite=True,
        )

    assert manifest_path.read_bytes() == original_manifest
    assert original_gdn_file.is_file() and original_qsa_file.is_file()
    assert not list(tmp_path.glob(".overlay.staging-*"))


def test_overlay_loader_rejects_corrupted_module_before_grafting(tmp_path):
    overlay_dir, manifest = _assemble(tmp_path)
    relative = manifest["modules"]["gdn"]["0"]["file"]
    module_path = overlay_dir / relative
    module_path.write_bytes(module_path.read_bytes() + b"corruption")
    model = Qwen3MoeModel(_tiny_config()).eval()
    original = [layer.self_attn for layer in model.layers]

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_flash_next_overlay(
            model,
            overlay_dir,
            base_model_id=MODEL_ID,
            base_model_revision=MODEL_REVISION,
        )

    assert all(layer.self_attn is before for layer, before in zip(model.layers, original))
