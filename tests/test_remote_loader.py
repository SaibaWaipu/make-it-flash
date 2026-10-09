import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_MODULES = PROJECT_ROOT / "src" / "make_it_flash"
REMOTE_MODULES = {
    "configuration_flashnext.py": PROJECT_ROOT / "remote_code" / "configuration_flashnext.py",
    "modeling_flashnext.py": PROJECT_ROOT / "remote_code" / "modeling_flashnext.py",
    "model.py": PROJECT_MODULES / "model.py",
    "flash_next.py": PROJECT_MODULES / "flash_next.py",
    "hybrid_cache.py": PROJECT_MODULES / "hybrid_cache.py",
}
AUTO_MAP = {
    "AutoConfig": "configuration_flashnext.FlashNextQwen3MoeConfig",
    "AutoModel": "modeling_flashnext.FlashNextQwen3MoeModel",
    "AutoModelForCausalLM": "modeling_flashnext.FlashNextQwen3MoeForCausalLM",
}


def _tiny_repo(tmp_path):
    for name, source in REMOTE_MODULES.items():
        shutil.copyfile(source, tmp_path / name)

    base = Qwen3MoeConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        moe_intermediate_size=16,
        num_experts=4,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
        max_position_embeddings=64,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    config = base.to_dict()
    config.update(
        {
            "architectures": ["FlashNextQwen3MoeForCausalLM"],
            "auto_map": AUTO_MAP,
            "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
            "flash_next_integration": {
                "format": "make_it_flash.integrated_full_checkpoint.v1",
                "gdn_layers": [0, 1, 2],
                "qsa_layers": [3],
                "gdn_config": {
                    "linear_conv_kernel_dim": 4,
                    "linear_key_head_dim": 16,
                    "linear_num_key_heads": 2,
                    "linear_num_value_heads": 4,
                    "linear_value_head_dim": 16,
                },
                "qsa_config": {
                    "compress_ratio": 2,
                    "head_dim": 16,
                    "index_head_dim": 8,
                    "index_n_heads": 2,
                    "num_heads": 4,
                    "num_key_value_heads": 2,
                    "rope_theta": 100.0,
                    "rotary_dim": 8,
                    "token_budget": 4,
                },
            },
        }
    )
    (tmp_path / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return tmp_path


def test_remote_auto_config_and_tiny_from_pretrained_round_trip(tmp_path):
    repo = _tiny_repo(tmp_path)
    config = AutoConfig.from_pretrained(repo, trust_remote_code=True, local_files_only=True)
    assert type(config).__name__ == "FlashNextQwen3MoeConfig"
    assert config.flash_next_integration["qsa_layers"] == [3]
    assert config.layer_types[-1] == "full_attention"

    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True).eval()
    state = model.state_dict()
    gdn_keys = [key for key in state if ".layers.0.self_attn.gdn." in key]
    qsa_keys = [key for key in state if ".layers.3.self_attn." in key]
    assert len(gdn_keys) == 7
    assert len(qsa_keys) == 9
    assert not any("layers.0.self_attn.q_proj" in key for key in state)

    model.save_pretrained(repo, safe_serialization=True)
    checkpoint_state = {}
    for name, value in state.items():
        if name.endswith(".mlp.experts.gate_up_proj"):
            prefix = name[: -len("gate_up_proj")]
            gate, up = value.chunk(2, dim=1)
            for expert_idx in range(config.num_experts):
                checkpoint_state[f"{prefix}{expert_idx}.gate_proj.weight"] = gate[expert_idx].contiguous()
                checkpoint_state[f"{prefix}{expert_idx}.up_proj.weight"] = up[expert_idx].contiguous()
        elif name.endswith(".mlp.experts.down_proj"):
            prefix = name[: -len("down_proj")]
            for expert_idx in range(config.num_experts):
                checkpoint_state[f"{prefix}{expert_idx}.down_proj.weight"] = value[expert_idx].contiguous()
        else:
            checkpoint_state[name] = value.contiguous()
    save_file(checkpoint_state, str(repo / "model.safetensors"))

    loaded, info = AutoModelForCausalLM.from_pretrained(
        repo,
        trust_remote_code=True,
        local_files_only=True,
        output_loading_info=True,
    )
    assert not info["missing_keys"]
    assert not info["unexpected_keys"]
    assert set(loaded.state_dict()) == set(state)

    input_ids = torch.tensor([[1, 4, 5]], dtype=torch.long)
    with torch.no_grad():
        output = loaded(input_ids=input_ids, use_cache=False)
        generated = loaded.generate(input_ids=input_ids, max_new_tokens=1, do_sample=False, use_cache=True)
    assert output.logits.shape == (1, 3, config.vocab_size)
    assert torch.isfinite(output.logits).all()
    assert generated.shape == (1, 4)
    assert loaded.config.layer_types[-1] == "full_attention"
    loaded.config.validate()
