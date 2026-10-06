"""Teacher-forced capture of local attention input/output targets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from .model import get_decoder_layers, make_gdn_config


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc


def _check_gpu_memory(min_free_gib: float) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "teacher activation capture requires a CUDA GPU; run the Kaggle GPU notebook or an HF A100 job"
        )
    free_bytes = sum(torch.cuda.mem_get_info(index)[0] for index in range(torch.cuda.device_count()))
    free_gib = free_bytes / (1024**3)
    if free_gib < min_free_gib:
        raise RuntimeError(
            f"only {free_gib:.1f} GiB CUDA memory is free; need at least {min_free_gib:.1f} GiB "
            "for the BF16 32B teacher pilot. No model weights were downloaded."
        )


def cache_teacher_outputs(
    *,
    data_file: str | Path,
    output_dir: str | Path,
    layers: list[int] | tuple[int, ...] = (0,),
    max_examples: int | None = None,
    min_free_gib: float = 66.0,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run the base model and save input/output pairs at selected attention blocks."""
    try:
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError("install make-it-flash with its runtime dependencies first") from exc

    source_path = Path(data_file)
    data_manifest_path = source_path.with_name("data_manifest.json")
    if not source_path.is_file() or not data_manifest_path.is_file():
        raise FileNotFoundError("data_file and its adjacent data_manifest.json are required")
    data_manifest = json.loads(data_manifest_path.read_text(encoding="utf-8"))
    model_id = data_manifest["model_id"]
    revision = data_manifest.get("model_revision", "main")

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    marker = target_dir / "cache_manifest.json"
    if not overwrite and (marker.exists() or any(target_dir.glob("*.safetensors"))):
        raise FileExistsError(f"cache output already exists under {target_dir}; pass --overwrite")

    _check_gpu_memory(min_free_gib)
    config = AutoConfig.from_pretrained(model_id, revision=revision, trust_remote_code=False)
    if getattr(config, "model_type", None) != "qwen3_moe":
        raise ValueError(f"expected a qwen3_moe base model, got {getattr(config, 'model_type', None)!r}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
        trust_remote_code=False,
    )
    model.eval()
    decoder_layers = get_decoder_layers(model)
    selected = tuple(sorted(set(int(index) for index in layers)))
    if not selected or min(selected) < 0 or max(selected) >= len(decoder_layers):
        raise ValueError(f"layer indices must be in [0, {len(decoder_layers) - 1}]")
    for index in selected:
        if not hasattr(decoder_layers[index], "self_attn"):
            raise ValueError(f"layer {index} has no self_attn module")

    gdn_config = make_gdn_config(config)
    captures: dict[int, dict[str, torch.Tensor]] = {index: {} for index in selected}
    hooks = []
    for index in selected:
        attention = decoder_layers[index].self_attn

        def pre_hook(module, args, kwargs, layer_index=index):
            hidden = kwargs.get("hidden_states")
            if hidden is None and args:
                hidden = args[0]
            if hidden is None:
                raise RuntimeError(f"could not read layer {layer_index} attention input")
            captures[layer_index]["input"] = hidden.detach()[0].to(device="cpu", dtype=torch.bfloat16).contiguous()

        def post_hook(module, args, output, layer_index=index):
            result = output[0] if isinstance(output, (tuple, list)) else output
            if not isinstance(result, torch.Tensor):
                raise RuntimeError(f"unexpected attention output type at layer {layer_index}")
            captures[layer_index]["target"] = result.detach()[0].to(device="cpu", dtype=torch.bfloat16).contiguous()

        hooks.append(attention.register_forward_pre_hook(pre_hook, with_kwargs=True))
        hooks.append(attention.register_forward_hook(post_hook))

    embedding_device = model.get_input_embeddings().weight.device
    emitted = 0
    token_count = 0
    try:
        with torch.inference_mode():
            for row in _read_jsonl(source_path):
                ids = row.get("input_ids")
                if not isinstance(ids, list) or not ids:
                    continue
                if len(ids) > int(getattr(config, "max_position_embeddings", len(ids))):
                    raise ValueError(f"sample {row.get('sample_id')} exceeds model position limit")
                input_ids = torch.tensor([ids], dtype=torch.long, device=embedding_device)
                for layer in selected:
                    captures[layer].clear()
                model(input_ids=input_ids, use_cache=False, output_attentions=False)
                tensors: dict[str, torch.Tensor] = {
                    "input_ids": input_ids[0].detach().to(device="cpu", dtype=torch.int32).contiguous()
                }
                for layer in selected:
                    capture = captures[layer]
                    if "input" not in capture or "target" not in capture:
                        raise RuntimeError(f"attention hook did not capture layer {layer}")
                    tensors[f"layer_{layer:02d}_input"] = capture["input"]
                    tensors[f"layer_{layer:02d}_target"] = capture["target"]
                metadata = {
                    "sample_id": str(row.get("sample_id", emitted)),
                    "category": str(row.get("category", "unknown")),
                    "config": str(row.get("config", "unknown")),
                    "model_id": model_id,
                    "model_revision": revision,
                }
                save_file(tensors, str(target_dir / f"sample_{emitted:06d}.safetensors"), metadata=metadata)
                emitted += 1
                token_count += len(ids)
                if max_examples is not None and emitted >= max_examples:
                    break
    finally:
        for hook in hooks:
            hook.remove()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    base_fields = {
        "model_type": config.model_type,
        "hidden_size": int(config.hidden_size),
        "vocab_size": int(config.vocab_size),
        "num_hidden_layers": int(config.num_hidden_layers),
        "num_attention_heads": int(config.num_attention_heads),
        "head_dim": int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)),
        "rms_norm_eps": float(getattr(config, "rms_norm_eps", 1e-6)),
        "max_position_embeddings": int(getattr(config, "max_position_embeddings", 0)),
    }
    gdn_fields = {
        "linear_key_head_dim": int(gdn_config.linear_key_head_dim),
        "linear_value_head_dim": int(gdn_config.linear_value_head_dim),
        "linear_num_key_heads": int(gdn_config.linear_num_key_heads),
        "linear_num_value_heads": int(gdn_config.linear_num_value_heads),
        "linear_conv_kernel_dim": int(gdn_config.linear_conv_kernel_dim),
    }
    manifest = {
        "model_id": model_id,
        "model_revision": revision,
        "data_file": str(source_path),
        "layers": list(selected),
        "num_examples": emitted,
        "num_tokens": token_count,
        "storage_dtype": "bfloat16",
        "base_config": base_fields,
        "gdn_config": gdn_fields,
        "note": "cache contains one teacher-forced local attention input/output pair per sequence; it is not a merged hybrid checkpoint",
    }
    marker.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
