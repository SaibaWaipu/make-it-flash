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
            "teacher activation capture requires a CUDA GPU; run the documented Hugging Face Jobs workflow on an A100"
        )
    free_bytes = sum(torch.cuda.mem_get_info(index)[0] for index in range(torch.cuda.device_count()))
    free_gib = free_bytes / (1024**3)
    if free_gib < min_free_gib:
        raise RuntimeError(
            f"only {free_gib:.1f} GiB CUDA memory is free; need at least {min_free_gib:.1f} GiB "
            "for the BF16 32B teacher pilot. No model weights were downloaded."
        )


def _capture_local_attention(
    model: Any,
    input_ids: torch.Tensor,
    layers: tuple[int, ...],
    attention_layers: tuple[int, ...],
    compress_ratio: int = 4,
) -> dict[int, dict[str, torch.Tensor]]:
    """Capture local targets and compact teacher mass for complete micro-blocks."""
    if compress_ratio <= 0:
        raise ValueError("compress_ratio must be positive")
    decoder_layers = get_decoder_layers(model)
    attention_set = set(attention_layers)
    if not attention_set.issubset(layers):
        raise ValueError("attention_layers must be a subset of the cached layers")
    captures: dict[int, dict[str, torch.Tensor]] = {index: {} for index in layers}
    hooks = []
    original_forwards: dict[int, Any] = {}
    model_config = getattr(model, "config", None)
    original_implementation = getattr(model_config, "_attn_implementation", "sdpa") or "sdpa"
    try:
        for index in layers:
            attention = decoder_layers[index].self_attn

            def pre_hook(module, args, kwargs, layer_index=index):
                hidden = kwargs.get("hidden_states")
                if hidden is None and args:
                    hidden = args[0]
                if hidden is None:
                    raise RuntimeError(f"could not read layer {layer_index} attention input")
                captures[layer_index]["input"] = hidden.detach()[0].to(
                    device="cpu", dtype=torch.bfloat16
                ).contiguous()

            def post_hook(module, args, output, layer_index=index):
                if not isinstance(output, (tuple, list)) or not output:
                    raise RuntimeError(f"unexpected attention output at layer {layer_index}")
                result = output[0]
                if not isinstance(result, torch.Tensor):
                    raise RuntimeError(f"unexpected attention output type at layer {layer_index}")
                captures[layer_index]["target"] = result.detach()[0].to(
                    device="cpu", dtype=torch.bfloat16
                ).contiguous()
                if layer_index in attention_set:
                    if len(output) < 2 or not isinstance(output[1], torch.Tensor):
                        raise RuntimeError(f"attention weights were not returned for layer {layer_index}")
                    weights = output[1].detach()[0]
                    seq_len, key_len = weights.shape[-2:]
                    if seq_len != key_len:
                        raise RuntimeError("teacher attention map must be square for unpadded calibration sequences")
                    num_blocks = key_len // compress_ratio
                    if num_blocks:
                        head_summed = weights.sum(dim=0, dtype=torch.float32)
                        block_mass = head_summed[:, : num_blocks * compress_ratio]
                        block_mass = block_mass.view(seq_len, num_blocks, compress_ratio).sum(dim=-1)
                    else:
                        block_mass = torch.empty((seq_len, 0), dtype=torch.float32, device=weights.device)
                    captures[layer_index]["block_mass"] = block_mass.to(
                        device="cpu", dtype=torch.bfloat16
                    ).contiguous()

            hooks.append(attention.register_forward_pre_hook(pre_hook, with_kwargs=True))
            hooks.append(attention.register_forward_hook(post_hook))
        if attention_set:
            # Eager mask construction is needed because SDPA may express causality
            # with is_causal=True and pass no explicit mask to the local module.
            # Keep non-target layers on SDPA while the target emits its probabilities.
            model_config._attn_implementation = "eager"
            for index, decoder_layer in enumerate(decoder_layers):
                attention = decoder_layer.self_attn
                original_forward = attention.forward
                config = attention.config
                layer_is_target = index in attention_set

                def layer_forward(
                    *args,
                    _forward=original_forward,
                    _config=config,
                    _target=layer_is_target,
                    **kwargs,
                ):
                    previous_implementation = getattr(_config, "_attn_implementation", original_implementation)
                    _config._attn_implementation = "eager" if _target else original_implementation
                    if _target:
                        kwargs["output_attentions"] = True
                    try:
                        return _forward(*args, **kwargs)
                    finally:
                        _config._attn_implementation = previous_implementation

                original_forwards[index] = original_forward
                attention.forward = layer_forward

        with torch.inference_mode():
            model(input_ids=input_ids, use_cache=False, output_attentions=False)
    finally:
        for hook in hooks:
            hook.remove()
        for index, original_forward in original_forwards.items():
            decoder_layers[index].self_attn.forward = original_forward
        if attention_set and model_config is not None:
            model_config._attn_implementation = original_implementation
    return captures


def cache_teacher_outputs(
    *,
    data_file: str | Path,
    output_dir: str | Path,
    layers: list[int] | tuple[int, ...] = (0,),
    attention_layers: list[int] | tuple[int, ...] = (),
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
    layer_count = int(config.num_hidden_layers)
    selected = tuple(sorted(set(int(index) for index in layers)))
    attention_selected = tuple(sorted(set(int(index) for index in attention_layers)))
    if not selected or min(selected) < 0 or max(selected) >= layer_count:
        raise ValueError(f"layer indices must be in [0, {layer_count - 1}]")
    if not set(attention_selected).issubset(selected):
        raise ValueError("attention_layers must be included in layers")
    if len(attention_selected) > 1:
        raise ValueError("capture one dense teacher attention layer at a time to bound memory")

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
    if len(decoder_layers) != layer_count:
        raise RuntimeError("loaded model layer count differs from AutoConfig")
    for index in selected:
        if not hasattr(decoder_layers[index], "self_attn"):
            raise ValueError(f"layer {index} has no self_attn module")

    gdn_config = make_gdn_config(config)
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
                captures = _capture_local_attention(model, input_ids, selected, attention_selected)
                tensors: dict[str, torch.Tensor] = {
                    "input_ids": input_ids[0].detach().to(device="cpu", dtype=torch.int32).contiguous()
                }
                for layer in selected:
                    capture = captures[layer]
                    if "input" not in capture or "target" not in capture:
                        raise RuntimeError(f"attention hook did not capture layer {layer}")
                    tensors[f"layer_{layer:02d}_input"] = capture["input"]
                    tensors[f"layer_{layer:02d}_target"] = capture["target"]
                    if layer in attention_selected:
                        if "block_mass" not in capture:
                            raise RuntimeError(f"attention hook did not capture block masses for layer {layer}")
                        tensors[f"layer_{layer:02d}_block_mass"] = capture["block_mass"]
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
    qsa_fields = {
        "num_heads": 24,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "rotary_dim": 64,
        "rope_theta": 10_000_000.0,
        "index_n_heads": 4,
        "index_head_dim": 128,
        "token_budget": 2048,
        "compress_ratio": 4,
    }
    manifest = {
        "model_id": model_id,
        "model_revision": revision,
        "data_file": str(source_path),
        "layers": list(selected),
        "attention_layers": list(attention_selected),
        "num_examples": emitted,
        "num_tokens": token_count,
        "storage_dtype": "bfloat16",
        "base_config": base_fields,
        "gdn_config": gdn_fields,
        "qsa_config": qsa_fields,
        "note": "cache contains teacher-forced local attention input/output pairs; optional teacher attention is reduced to complete micro-block masses for attention_layers; this is not a merged hybrid checkpoint",
    }
    marker.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
