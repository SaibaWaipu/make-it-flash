"""Fit one block-indexed QSA attention adapter from teacher caches."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

from .fit import _split_value
from .model import make_qsa
from .provenance import checkpoint_provenance, sha256_file


def _qsa_sample(
    path: Path, layer: int, device: torch.device, compress_ratio: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    input_key = f"layer_{layer:02d}_input"
    target_key = f"layer_{layer:02d}_target"
    block_mass_key = f"layer_{layer:02d}_block_mass"
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        missing = {input_key, target_key, block_mass_key} - keys
        if missing:
            raise KeyError(f"{path.name} is missing QSA fitting tensors: {sorted(missing)}")
        hidden = handle.get_tensor(input_key).to(device=device, dtype=torch.float32).unsqueeze(0)
        target = handle.get_tensor(target_key).to(device=device, dtype=torch.float32).unsqueeze(0)
        teacher_block_mass = handle.get_tensor(block_mass_key).to(device=device, dtype=torch.float32).unsqueeze(0)
    if hidden.shape != target.shape or hidden.ndim != 3:
        raise ValueError(f"teacher input/output shape mismatch in {path.name}")
    seq_len = hidden.shape[1]
    expected_blocks = seq_len // compress_ratio
    if teacher_block_mass.shape != (1, seq_len, expected_blocks):
        raise ValueError(f"teacher block mass has an invalid shape in {path.name}")
    if not torch.isfinite(teacher_block_mass).all() or torch.any(teacher_block_mass < 0):
        raise ValueError(f"teacher block masses must be finite and nonnegative in {path.name}")
    positions = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
    causal_mask = torch.ones((1, 1, seq_len, seq_len), dtype=torch.bool, device=device).tril()
    return hidden, target, teacher_block_mass, positions, causal_mask


def _evaluate_qsa(
    qsa: torch.nn.Module,
    paths: list[Path],
    layer: int,
    device: torch.device,
    selector_loss_weight: float,
    compress_ratio: int,
) -> dict[str, float]:
    if not paths:
        return {"mse": float("nan"), "selection_loss": float("nan"), "total_loss": float("nan")}
    qsa.eval()
    squared_error = 0.0
    elements = 0
    selector_sum = 0.0
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            for path in paths:
                hidden, target, teacher_attention, positions, causal_mask = _qsa_sample(path, layer, device, compress_ratio)
                prediction, _ = qsa(
                    hidden,
                    attention_mask=causal_mask,
                    position_ids=positions,
                    use_cache=False,
                )
                selector_loss = qsa.indexer.selection_loss(
                    hidden, positions, causal_mask, teacher_attention
                )
                squared_error += float(F.mse_loss(prediction.float(), target, reduction="sum").item())
                elements += target.numel()
                selector_sum += float(selector_loss.item())
    mse = squared_error / max(elements, 1)
    selection_loss = selector_sum / len(paths)
    return {
        "mse": mse,
        "selection_loss": selection_loss,
        "total_loss": mse + selector_loss_weight * selection_loss,
    }


def fit_qsa_layer(
    *,
    cache_dir: str | Path,
    output_dir: str | Path,
    layer: int,
    epochs: int = 3,
    max_steps: int = 200,
    learning_rate: float = 1e-4,
    selector_loss_weight: float = 0.1,
    validation_fraction: float = 0.1,
    seed: int = 17,
    overwrite: bool = False,
    allow_cpu: bool = False,
) -> dict[str, Any]:
    """Fit a QSA adapter using output MSE plus teacher-block selection loss.

    This consumes compact per-layer teacher block masses captured on request by
    the cache stage. It trains a single local layer; it neither trains nor merges a full LM.
    """
    cache_path = Path(cache_dir)
    manifest_path = cache_path / "cache_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if layer not in manifest.get("layers", []):
        raise ValueError(f"layer {layer} was not included in cache; cached layers={manifest.get('layers', [])}")
    if layer not in manifest.get("attention_layers", []):
        raise ValueError(f"teacher attention block masses for layer {layer} were not cached")
    qsa_config = dict(manifest.get("qsa_config", {}))
    compress_ratio = int(qsa_config.get("compress_ratio", 4))
    token_budget = int(qsa_config.get("token_budget", 2048))
    if compress_ratio <= 0 or token_budget < compress_ratio:
        raise ValueError("qsa_config must have a positive compress_ratio and token_budget covering one block")
    if (
        epochs < 1
        or max_steps < 1
        or not math.isfinite(learning_rate)
        or learning_rate <= 0
        or not math.isfinite(selector_loss_weight)
        or selector_loss_weight <= 0
    ):
        raise ValueError("epochs, max_steps, learning_rate, and selector_loss_weight must all be positive finite values")
    if not 0.0 <= validation_fraction < 0.5:
        raise ValueError("validation_fraction must be in [0, 0.5)")
    if not torch.cuda.is_available() and not allow_cpu:
        raise RuntimeError("QSA fitting requires CUDA; use --allow-cpu only for tiny smoke tests")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_path = Path(output_dir)
    checkpoint_path = output_path / f"qsa_layer_{layer:02d}.safetensors"
    metrics_path = output_path / f"fit_qsa_layer_{layer:02d}.json"
    if not overwrite and (checkpoint_path.exists() or metrics_path.exists()):
        raise FileExistsError(f"QSA fit output already exists under {output_path}; pass --overwrite")

    all_paths = sorted(cache_path.glob("sample_*.safetensors"))
    if not all_paths:
        raise FileNotFoundError(f"no activation shards in {cache_path}")
    train_paths: list[Path] = []
    validation_paths: list[Path] = []
    threshold = int(validation_fraction * (2**64 - 1))
    for path in all_paths:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
        if f"layer_{layer:02d}_block_mass" not in keys:
            continue
        (validation_paths if validation_fraction and _split_value(path) < threshold else train_paths).append(path)
    if not train_paths and validation_paths:
        train_paths.append(validation_paths.pop())
    if not train_paths:
        raise ValueError(f"no cached QSA examples for layer {layer}")
    validation_uses_train_fallback = not validation_paths
    if validation_uses_train_fallback:
        validation_paths = train_paths[: min(4, len(train_paths))]

    input_key = f"layer_{layer:02d}_input"

    def _sequence_lengths(paths: list[Path]) -> list[int]:
        lengths = []
        for path in paths:
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                lengths.append(handle.get_slice(input_key).get_shape()[0])
        return lengths

    train_sequence_lengths = _sequence_lengths(train_paths)
    validation_sequence_lengths = _sequence_lengths(validation_paths)
    max_train_sequence_length = max(train_sequence_lengths)
    pruning_threshold = (token_budget // compress_ratio + 1) * compress_ratio
    selector_pruning_examples = sum(length >= pruning_threshold for length in train_sequence_lengths)
    selector_pruning_validation_examples = sum(
        length >= pruning_threshold for length in validation_sequence_lengths
    )
    if selector_pruning_examples == 0:
        raise ValueError(
            f"no QSA training sequence reaches {pruning_threshold} tokens required for top-k pruning; "
            "increase prepare --max-seq-len and recache"
        )
    if validation_uses_train_fallback:
        raise ValueError(
            "no independent QSA validation examples are available; collect more calibration data and recache"
        )
    if selector_pruning_validation_examples == 0:
        raise ValueError(
            f"no independent QSA validation sequence reaches {pruning_threshold} tokens required for top-k pruning; "
            "increase prepare --max-seq-len or collect more long validation examples"
        )
    output_path.mkdir(parents=True, exist_ok=True)

    base = SimpleNamespace(**manifest["base_config"])
    torch.manual_seed(seed)
    random.seed(seed)
    qsa = make_qsa(base, layer_idx=layer, **qsa_config).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(qsa.parameters(), lr=learning_rate, weight_decay=0.01)

    initial_validation = _evaluate_qsa(
        qsa, validation_paths, layer, device, selector_loss_weight, compress_ratio
    )
    best_validation_loss = initial_validation["total_loss"]
    best_state = {name: tensor.detach().cpu().clone() for name, tensor in qsa.state_dict().items()}
    steps = 0
    recent_losses: list[float] = []
    use_amp = device.type == "cuda"
    for _epoch in range(epochs):
        order = list(train_paths)
        random.shuffle(order)
        for path in order:
            qsa.train()
            hidden, target, teacher_attention, positions, causal_mask = _qsa_sample(path, layer, device, compress_ratio)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                prediction, _ = qsa(
                    hidden,
                    attention_mask=causal_mask,
                    position_ids=positions,
                    use_cache=False,
                )
                mse_loss = F.mse_loss(prediction.float(), target.float())
                selector_loss = qsa.indexer.selection_loss(
                    hidden, positions, causal_mask, teacher_attention
                )
                loss = mse_loss + selector_loss_weight * selector_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite QSA loss at step {steps + 1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(qsa.parameters(), 1.0)
            optimizer.step()
            steps += 1
            recent_losses.append(float(loss.detach().item()))

            if steps % 10 == 0 or steps == 1:
                current = _evaluate_qsa(qsa, validation_paths, layer, device, selector_loss_weight, compress_ratio)
                if math.isfinite(current["total_loss"]) and current["total_loss"] < best_validation_loss:
                    best_validation_loss = current["total_loss"]
                    best_state = {name: tensor.detach().cpu().clone() for name, tensor in qsa.state_dict().items()}
            if steps >= max_steps:
                break
        if steps >= max_steps:
            break

    current_validation = _evaluate_qsa(qsa, validation_paths, layer, device, selector_loss_weight, compress_ratio)
    if math.isfinite(current_validation["total_loss"]) and current_validation["total_loss"] < best_validation_loss:
        best_validation_loss = current_validation["total_loss"]
        best_state = {name: tensor.detach().cpu().clone() for name, tensor in qsa.state_dict().items()}
    qsa.load_state_dict(best_state)
    final_validation = _evaluate_qsa(
        qsa, validation_paths, layer, device, selector_loss_weight, compress_ratio
    )
    save_file(
        {name: tensor.detach().cpu().contiguous() for name, tensor in qsa.state_dict().items()},
        str(checkpoint_path),
        metadata={
            "model_id": str(manifest["model_id"]),
            "model_revision": str(manifest["model_revision"]),
            "teacher_layer": str(layer),
            "module": "make_it_flash.QSAQwen3MoeAttentionAdapter",
            "qsa_config": json.dumps(qsa_config, sort_keys=True),
            **checkpoint_provenance(),
        },
    )
    checkpoint_sha256 = sha256_file(checkpoint_path)
    metrics = {
        "model_id": manifest["model_id"],
        "model_revision": manifest["model_revision"],
        "teacher_layer": layer,
        "num_train_sequences": len(train_paths),
        "num_validation_sequences": len(validation_paths),
        "validation_uses_train_fallback": validation_uses_train_fallback,
        "max_train_sequence_length": max_train_sequence_length,
        "selector_pruning_threshold_tokens": pruning_threshold,
        "selector_pruning_examples": selector_pruning_examples,
        "selector_pruning_validation_examples": selector_pruning_validation_examples,
        "selector_pruning_warning": (
            None if selector_pruning_examples else "training sequences did not exceed token_budget by a full micro-block"
        ),
        "qsa_config": qsa_config,
        "steps": steps,
        "epochs_requested": epochs,
        "learning_rate": learning_rate,
        "selector_loss_weight": selector_loss_weight,
        "initial_validation": initial_validation,
        "best_validation_total_loss": best_validation_loss,
        "final_validation": final_validation,
        "selected_checkpoint_validation_selection_loss": final_validation["selection_loss"],
        "mean_recent_train_loss": sum(recent_losses[-min(20, len(recent_losses)):]) / max(1, min(20, len(recent_losses))),
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": checkpoint_sha256,
        "warning": "this is a standalone fitted QSA layer, not a merged/reloadable hybrid language model checkpoint",
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics
