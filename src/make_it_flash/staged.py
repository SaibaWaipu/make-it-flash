"""One-layer-at-a-time work with immutable calibration provenance.

No Hub writes occur here. Each invocation caches and fits only one attention
layer, then the separate publisher can upload its checkpoint and metrics.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .cache import cache_teacher_outputs
from .fit import fit_one_layer
from .fit_qsa import fit_qsa_layer
from .full_run import _verify_cache, _verify_fit, preflight_full_run
from .model import flash_next_attention_schedule
from .provenance import validate_base_config_provenance


def run_staged_layer(
    *, data_file: str | Path, output_dir: str | Path, layer: int,
    validation_fraction: float = 0.1, epochs: int = 3, max_steps: int = 200,
    min_free_gib: float = 66.0, dry_run: bool = False, allow_cpu: bool = False,
) -> dict[str, Any]:
    """Capture/fit exactly one layer; fail closed on partial or stale outputs."""
    if type(layer) is not int or not 0 <= layer < 32:
        raise ValueError("layer must be an integer from 0 to 31")
    if type(dry_run) is not bool or type(allow_cpu) is not bool:
        raise ValueError("dry_run and allow_cpu must be boolean")
    if epochs < 1 or max_steps < 1 or not math.isfinite(min_free_gib) or min_free_gib < 0:
        raise ValueError("epochs/max_steps must be positive and min_free_gib nonnegative")
    expected = preflight_full_run(data_file, validation_fraction=validation_fraction)
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(expected["model_id"], revision=expected["model_revision"],
                                         trust_remote_code=False)
    validate_base_config_provenance(config, expected["model_id"], expected["model_revision"])
    gdn, qsa = flash_next_attention_schedule(32)
    if getattr(config, "model_type", None) != "qwen3_moe" or getattr(config, "num_hidden_layers", None) != 32:
        raise ValueError("staged conversion requires a pinned 32-layer Qwen3-MoE base")
    kind = "qsa" if layer in qsa else "gdn"
    root = Path(output_dir)
    source = Path(data_file).resolve()
    if source.is_relative_to(root.resolve()):
        raise ValueError("layer output cannot contain calibration inputs")
    cache_dir = root / "cache"
    fit_dir = root / "fit"
    checkpoint = fit_dir / f"{kind}_layer_{layer:02d}.safetensors"
    metrics = fit_dir / (f"fit_layer_{layer:02d}.json" if kind == "gdn" else f"fit_qsa_layer_{layer:02d}.json")
    report = {"layer": layer, "kind": kind, "model_id": expected["model_id"],
              "model_revision": expected["model_revision"], "calibration_data_sha256": expected["data_sha256"],
              "data_manifest_sha256": expected["manifest_sha256"], "counts": expected["counts"],
              "checkpoint": str(checkpoint), "metrics": str(metrics), "dry_run": dry_run}
    if dry_run:
        return report
    if checkpoint.exists() and metrics.exists():
        _verify_fit(fit_dir, kind, layer, expected)
        report["status"] = "already_fitted"
        return report
    if checkpoint.exists() or metrics.exists():
        raise FileExistsError("partial fit exists; inspect it before retrying this layer")
    if (cache_dir / "cache_manifest.json").is_file():
        _verify_cache(cache_dir, expected, kind, layer)
    else:
        if cache_dir.exists() and any(cache_dir.iterdir()):
            raise FileExistsError("incomplete cache exists; inspect it before retrying this layer")
        if not allow_cpu:
            from .cache import _check_gpu_memory

            _check_gpu_memory(min_free_gib)
        cache_teacher_outputs(data_file=data_file, output_dir=cache_dir, layers=(layer,),
                              attention_layers=(layer,) if kind == "qsa" else (), min_free_gib=min_free_gib)
        _verify_cache(cache_dir, expected, kind, layer)
    kwargs = dict(cache_dir=cache_dir, output_dir=fit_dir, layer=layer, epochs=epochs,
                  max_steps=max_steps, validation_fraction=validation_fraction, allow_cpu=allow_cpu)
    if kind == "qsa":
        fit_qsa_layer(**kwargs)
    else:
        fit_one_layer(**kwargs)
    _verify_fit(fit_dir, kind, layer, expected)
    report["status"] = "fitted"
    return report
