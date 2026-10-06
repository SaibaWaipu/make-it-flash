"""Fit one Gated DeltaNet to cached teacher attention outputs."""

from __future__ import annotations

import hashlib
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

from .model import make_gdn


def _metadata(path: Path) -> dict[str, str]:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return handle.metadata() or {}


def _split_value(path: Path) -> int:
    meta = _metadata(path)
    stable_id = f"{meta.get('config', '')}:{meta.get('sample_id', path.name)}"
    return int.from_bytes(hashlib.sha256(stable_id.encode()).digest()[:8], "big")


def _tensor_pair(path: Path, layer: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    input_key = f"layer_{layer:02d}_input"
    target_key = f"layer_{layer:02d}_target"
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        if input_key not in keys or target_key not in keys:
            raise KeyError(f"{path.name} does not contain cached layer {layer}")
        x = handle.get_tensor(input_key).to(device=device, dtype=torch.float32).unsqueeze(0)
        y = handle.get_tensor(target_key).to(device=device, dtype=torch.float32).unsqueeze(0)
    if x.shape != y.shape:
        raise ValueError(f"teacher pair shape mismatch in {path.name}: {tuple(x.shape)} vs {tuple(y.shape)}")
    return x, y


def _evaluate(model: Any, paths: list[Path], layer: int, device: torch.device) -> float:
    if not paths:
        return float("nan")
    squared_error = 0.0
    elements = 0
    model.eval()
    with torch.no_grad():
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            for path in paths:
                x, target = _tensor_pair(path, layer, device)
                pred = model(x)
                squared_error += float(F.mse_loss(pred.float(), target, reduction="sum").item())
                elements += target.numel()
    return squared_error / max(elements, 1)


def fit_one_layer(
    *,
    cache_dir: str | Path,
    output_dir: str | Path,
    layer: int = 0,
    epochs: int = 3,
    max_steps: int = 200,
    learning_rate: float = 1e-4,
    validation_fraction: float = 0.1,
    seed: int = 17,
    overwrite: bool = False,
    allow_cpu: bool = False,
) -> dict[str, Any]:
    """Train only the new GDN block; teacher and other model weights stay frozen."""
    cache_path = Path(cache_dir)
    manifest_path = cache_path / "cache_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if layer not in manifest["layers"]:
        raise ValueError(f"layer {layer} was not included in cache; cached layers={manifest['layers']}")
    if epochs < 1 or max_steps < 1 or learning_rate <= 0:
        raise ValueError("epochs, max_steps, and learning_rate must be positive")
    if not 0.0 <= validation_fraction < 0.5:
        raise ValueError("validation_fraction must be in [0, 0.5)")
    if not torch.cuda.is_available() and not allow_cpu:
        raise RuntimeError("GDN fitting requires CUDA; use --allow-cpu only for tiny smoke tests")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out / f"gdn_layer_{layer:02d}.safetensors"
    metrics_path = out / f"fit_layer_{layer:02d}.json"
    if not overwrite and (checkpoint_path.exists() or metrics_path.exists()):
        raise FileExistsError(f"fit output already exists under {out}; pass --overwrite")

    all_paths = sorted(cache_path.glob("sample_*.safetensors"))
    if not all_paths:
        raise FileNotFoundError(f"no activation shards in {cache_path}")
    train_paths: list[Path] = []
    val_paths: list[Path] = []
    threshold = int(validation_fraction * (2**64 - 1))
    for path in all_paths:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
        if f"layer_{layer:02d}_input" not in keys:
            continue
        (val_paths if validation_fraction and _split_value(path) < threshold else train_paths).append(path)
    if not train_paths and val_paths:
        train_paths.append(val_paths.pop())
    if not train_paths:
        raise ValueError(f"no cached examples for layer {layer}")
    validation_uses_train_fallback = not val_paths
    if validation_uses_train_fallback:
        val_paths = train_paths[: min(4, len(train_paths))]

    base = SimpleNamespace(**manifest["base_config"])
    gdn_kwargs = {
        "key_heads": int(manifest["gdn_config"]["linear_num_key_heads"]),
        "value_heads": int(manifest["gdn_config"]["linear_num_value_heads"]),
    }
    torch.manual_seed(seed)
    random.seed(seed)
    gdn = make_gdn(base, layer, **gdn_kwargs).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(gdn.parameters(), lr=learning_rate, weight_decay=0.01)

    initial_val_loss = _evaluate(gdn, val_paths, layer, device)
    best_val_loss = initial_val_loss
    best_state = {name: tensor.detach().cpu().clone() for name, tensor in gdn.state_dict().items()}
    steps = 0
    losses: list[float] = []
    use_amp = device.type == "cuda"
    for epoch in range(epochs):
        order = list(train_paths)
        random.shuffle(order)
        for path in order:
            gdn.train()
            x, target = _tensor_pair(path, layer, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                prediction = gdn(x)
                loss = F.mse_loss(prediction.float(), target.float())
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {steps + 1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gdn.parameters(), 1.0)
            optimizer.step()
            steps += 1
            losses.append(float(loss.detach().item()))

            if steps % 10 == 0 or steps == 1:
                current_val = _evaluate(gdn, val_paths, layer, device)
                if math.isfinite(current_val) and current_val < best_val_loss:
                    best_val_loss = current_val
                    best_state = {name: tensor.detach().cpu().clone() for name, tensor in gdn.state_dict().items()}
            if steps >= max_steps:
                break
        if steps >= max_steps:
            break

    current_val_loss = _evaluate(gdn, val_paths, layer, device)
    if math.isfinite(current_val_loss) and current_val_loss < best_val_loss:
        best_val_loss = current_val_loss
        best_state = {name: tensor.detach().cpu().clone() for name, tensor in gdn.state_dict().items()}
    gdn.load_state_dict(best_state)
    final_val_loss = _evaluate(gdn, val_paths, layer, device)
    save_file(
        {name: tensor.detach().cpu().contiguous() for name, tensor in gdn.state_dict().items()},
        str(checkpoint_path),
        metadata={
            "model_id": str(manifest["model_id"]),
            "model_revision": str(manifest["model_revision"]),
            "teacher_layer": str(layer),
            "module": "transformers.models.qwen3_next.Qwen3NextGatedDeltaNet",
            "gdn_config": json.dumps(manifest["gdn_config"], sort_keys=True),
        },
    )
    metrics = {
        "model_id": manifest["model_id"],
        "model_revision": manifest["model_revision"],
        "teacher_layer": layer,
        "num_train_sequences": len(train_paths),
        "num_validation_sequences": len(val_paths),
        "validation_uses_train_fallback": validation_uses_train_fallback,
        "base_config": manifest["base_config"],
        "gdn_config": manifest["gdn_config"],
        "steps": steps,
        "epochs_requested": epochs,
        "learning_rate": learning_rate,
        "initial_validation_mse": initial_val_loss,
        "best_validation_mse": best_val_loss,
        "final_validation_mse": final_val_loss,
        "mean_recent_train_mse": sum(losses[-min(20, len(losses)):]) / max(1, min(20, len(losses))),
        "checkpoint": checkpoint_path.name,
        "warning": "this is a standalone local GDN block, not a merged/reloadable hybrid causal LM checkpoint",
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics
