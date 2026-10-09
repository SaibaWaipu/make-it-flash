"""Resumable 24-GDN/8-QSA local fitting and complete overlay assembly.

The calibration corpus is prepared separately. This stage does not submit paid
Jobs, upload weights, or claim that the locally fitted model preserves quality.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
from typing import Any, Callable

from .cache import QSA_COMPRESS_RATIO, QSA_TOKEN_BUDGET, cache_teacher_outputs
from .fit import fit_one_layer
from .fit_qsa import fit_qsa_layer
from .model import flash_next_attention_schedule
from .overlay import assemble_flash_next_overlay
from .provenance import sha256_file, validate_calibration_data_provenance, validate_model_revision


def _split_value(config: str, sample_id: str) -> int:
    import hashlib

    return int.from_bytes(hashlib.sha256(f"{config}:{sample_id}".encode()).digest()[:8], "big")


def _release_cuda_memory() -> None:
    """Return model/fit allocations to the driver before the next 32B reload."""
    gc.collect()
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def preflight_full_run(
    data_file: str | Path, *, validation_fraction: float = 0.1, min_qsa_sequence_length: int | None = None
) -> dict[str, Any]:
    """Check the actual stable train/validation split before any teacher load."""
    if not math.isfinite(validation_fraction) or not 0 < validation_fraction < 0.5:
        raise ValueError("full-layer fitting requires an independent validation_fraction in (0, 0.5)")
    source = Path(data_file)
    manifest_path = source.with_name("data_manifest.json")
    if not source.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("calibration JSONL and adjacent data_manifest.json are required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("data_file") not in (None, source.name):
        raise ValueError("invalid calibration data manifest or mismatched data_file")
    revision = manifest["model_revision"]
    validate_model_revision(revision)
    threshold = min_qsa_sequence_length or (QSA_TOKEN_BUDGET // QSA_COMPRESS_RATIO + 1) * QSA_COMPRESS_RATIO
    if not isinstance(threshold, int) or isinstance(threshold, bool) or threshold < 2:
        raise ValueError("min_qsa_sequence_length must be >= 2")
    split_boundary = int(validation_fraction * (2**64 - 1))
    counts = {"train": 0, "validation": 0, "qsa_train": 0, "qsa_validation": 0}
    ids: dict[tuple[str, str], str] = {}
    token_owners: dict[str, tuple[str, str]] = {}
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at calibration line {line_number}") from exc
            tokens = row.get("input_ids") if isinstance(row, dict) else None
            if not isinstance(tokens, list) or len(tokens) < 2 or any(type(token) is not int for token in tokens):
                raise ValueError(f"invalid input_ids at calibration line {line_number}")
            config, sample_id = row.get("config"), row.get("sample_id")
            if not isinstance(config, str) or not isinstance(sample_id, str) or not sample_id:
                raise ValueError(f"calibration line {line_number} requires config and sample_id")
            # Hashing the token sequence catches an accidental copy across split IDs.
            import hashlib

            token_digest = hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()
            group = "validation" if _split_value(config, sample_id) < split_boundary else "train"
            owner = token_owners.setdefault(token_digest, (config, sample_id))
            if owner != (config, sample_id) and ids.get(owner) != group:
                raise ValueError("identical calibration token sequences cross train/validation IDs")
            previous = ids.setdefault((config, sample_id), group)
            if previous != group:
                raise RuntimeError("sample ID crosses the validation boundary")
            counts[group] += 1
            if len(tokens) >= threshold:
                counts[f"qsa_{group}"] += 1
    if any(not counts[key] for key in counts):
        raise ValueError(
            f"full run requires independent train/validation and QSA-pruned train/validation sequences "
            f"(length >= {threshold}); observed {counts}. Prepare more/longer examples or select a different seed."
        )
    return {
        "data_file": str(source.resolve()),
        "data_sha256": sha256_file(source),
        "manifest_sha256": sha256_file(manifest_path),
        "model_id": manifest["model_id"],
        "model_revision": revision,
        "counts": counts,
        "token_sha256": sorted(token_owners),
        "qsa_min_length": threshold,
    }


def _verify_cache(cache_dir: Path, expected: dict[str, Any], kind: str, layer: int) -> None:
    from safetensors import safe_open

    manifest_path = cache_dir / "cache_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing cache manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    calibration = validate_calibration_data_provenance(manifest.get("calibration_data"))
    if (manifest.get("model_id"), manifest.get("model_revision")) != (
        expected["model_id"], expected["model_revision"]
    ) or any(calibration[key] != expected[key] for key in ("data_sha256", "manifest_sha256")):
        raise ValueError(f"cached teacher provenance does not match the current calibration data: {cache_dir}")
    if layer not in manifest.get("layers", []) or (kind == "qsa" and layer not in manifest.get("attention_layers", [])):
        raise ValueError(f"cache lacks {kind} layer {layer}: {cache_dir}")
    shards = sorted(cache_dir.glob("sample_*.safetensors"))
    if not shards or len(shards) != manifest.get("num_examples"):
        raise ValueError(f"cache shard count differs from manifest: {cache_dir}")
    from .provenance import validate_cache_sample_metadata

    for index, shard in enumerate(shards):
        if shard.name != f"sample_{index:06d}.safetensors":
            raise ValueError(f"cache shard numbering is not contiguous: {cache_dir}")
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            validate_cache_sample_metadata(handle.metadata(), path=shard, model_id=expected["model_id"],
                                           model_revision=expected["model_revision"], calibration_data=calibration)
            required = {f"layer_{layer:02d}_input", f"layer_{layer:02d}_target"}
            if kind == "qsa":
                required.add(f"layer_{layer:02d}_block_mass")
            if not required.issubset(handle.keys()):
                raise ValueError(f"cache shard {shard.name} lacks {kind} targets")


def _verify_fit(fit_dir: Path, kind: str, layer: int, expected: dict[str, Any]) -> None:
    from safetensors import safe_open

    stem = f"{kind}_layer_{layer:02d}.safetensors"
    metrics_name = f"fit_layer_{layer:02d}.json" if kind == "gdn" else f"fit_qsa_layer_{layer:02d}.json"
    checkpoint, metrics_path = fit_dir / stem, fit_dir / metrics_name
    if not checkpoint.is_file() or not metrics_path.is_file():
        raise FileNotFoundError(f"incomplete fit for layer {layer} under {fit_dir}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    calibration = validate_calibration_data_provenance(metrics.get("calibration_data"))
    if (metrics.get("model_id"), metrics.get("model_revision"), metrics.get("teacher_layer")) != (
        expected["model_id"], expected["model_revision"], layer
    ) or any(calibration[key] != expected[key] for key in ("data_sha256", "manifest_sha256")):
        raise ValueError(f"fit provenance mismatch: {metrics_path}")
    if metrics.get("checkpoint") != checkpoint.name or metrics.get("checkpoint_sha256") != sha256_file(checkpoint):
        raise ValueError(f"fit checkpoint hash mismatch: {checkpoint}")
    if metrics.get("validation_uses_train_fallback") is not False:
        raise ValueError(f"fit has no independent validation: {metrics_path}")
    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    if metadata.get("teacher_layer") != str(layer) or metadata.get("model_revision") != expected["model_revision"]:
        raise ValueError(f"fit checkpoint metadata mismatch: {checkpoint}")


def run_full_conversion(
    *, data_file: str | Path, output_dir: str | Path, validation_fraction: float = 0.1,
    epochs: int = 3, max_steps: int = 200, min_free_gib: float = 66.0,
    resume: bool = False, allow_cpu: bool = False, dry_run: bool = False,
    assemble: bool = True,
    on_layer_complete: Callable[[str, int, Path, dict[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    """Capture/fit every layer; optionally publish each verified fit immediately."""
    if (not isinstance(resume, bool) or not isinstance(dry_run, bool) or not isinstance(allow_cpu, bool)
            or not isinstance(assemble, bool)):
        raise ValueError("resume, dry_run, allow_cpu, and assemble must be boolean")
    if on_layer_complete is not None and not callable(on_layer_complete):
        raise ValueError("on_layer_complete must be callable")
    if epochs < 1 or max_steps < 1 or not math.isfinite(min_free_gib) or min_free_gib < 0:
        raise ValueError("epochs/max_steps must be positive and min_free_gib must be nonnegative")
    expected = preflight_full_run(data_file, validation_fraction=validation_fraction)
    from transformers import AutoConfig
    from .provenance import validate_base_config_provenance

    base_config = AutoConfig.from_pretrained(expected["model_id"], revision=expected["model_revision"],
                                              trust_remote_code=False)
    validate_base_config_provenance(base_config, expected["model_id"], expected["model_revision"])
    num_layers = int(base_config.num_hidden_layers)
    gdn, qsa = flash_next_attention_schedule(num_layers)
    if base_config.model_type != "qwen3_moe" or num_layers != 32 or (len(gdn), len(qsa)) != (24, 8):
        raise ValueError(f"expected 32 Qwen3-MoE layers (24 GDN + 8 QSA), found {num_layers}")
    root = Path(output_dir)
    if root.resolve() == Path(data_file).resolve().parent or Path(data_file).resolve().is_relative_to(root.resolve()):
        raise ValueError("full-run output_dir must not contain or replace the calibration data")
    plan_data = {key: value for key, value in expected.items() if key != "token_sha256"}
    plan = {"model_id": expected["model_id"], "model_revision": expected["model_revision"],
            "qsa_layers": list(qsa), "gdn_layers": list(gdn), "data": plan_data,
            "output_dir": str(root), "estimated_teacher_loads": len(qsa), "dry_run": dry_run,
            "assemble": assemble, "progressive_publish": on_layer_complete is not None}
    if dry_run:
        return plan
    if not allow_cpu:
        from .cache import _check_gpu_memory

        _check_gpu_memory(min_free_gib)
    if root.exists() and not resume:
        raise FileExistsError(f"full-run output exists: {root}; use --resume to validate and continue")
    if resume and (root / "overlay" / "flash_next_overlay.json").is_file():
        raise FileExistsError("complete overlay already exists; do not overwrite or re-upload a completed run")
    if (root / "overlay").exists():
        raise FileExistsError("incomplete overlay directory exists; inspect it before resuming")
    root.mkdir(parents=True, exist_ok=True)
    gdn_dir, qsa_dir = root / "gdn", root / "qsa"
    # Fit in numeric layer order so each completed layer can be committed
    # immediately. The first GDN capture also stores QSA layer 3, preserving
    # the eight-teacher-load optimization (one shared capture plus seven QSA).
    for layer in range(num_layers):
        kind = "qsa" if layer in qsa else "gdn"
        fit_dir = qsa_dir if kind == "qsa" else gdn_dir
        checkpoint = fit_dir / f"{kind}_layer_{layer:02d}.safetensors"
        metrics = fit_dir / (f"fit_layer_{layer:02d}.json" if kind == "gdn" else f"fit_qsa_layer_{layer:02d}.json")
        if resume and checkpoint.is_file() and metrics.is_file():
            _verify_fit(fit_dir, kind, layer, expected)
            continue
        cache_layer = qsa[0] if kind == "gdn" else layer
        cache_dir = root / "cache" / f"qsa_layer_{cache_layer:02d}"
        cache_manifest = cache_dir / "cache_manifest.json"
        if cache_manifest.is_file():
            _verify_cache(cache_dir, expected, kind, layer)
        else:
            capture_layers = (*gdn, qsa[0]) if kind == "gdn" or layer == qsa[0] else (layer,)
            capture_attention = (qsa[0],) if kind == "gdn" else (layer,)
            cache_teacher_outputs(data_file=data_file, output_dir=cache_dir, layers=capture_layers,
                                  attention_layers=capture_attention, min_free_gib=min_free_gib,
                                  overwrite=cache_dir.exists())
            # cache_teacher_outputs drops its model on return, but its CUDA
            # allocator cache is still reserved until explicitly released here.
            _release_cuda_memory()
            _verify_cache(cache_dir, expected, kind, layer)
        kwargs = dict(cache_dir=cache_dir, output_dir=fit_dir, layer=layer,
                      epochs=epochs, max_steps=max_steps, validation_fraction=validation_fraction,
                      overwrite=checkpoint.exists() or metrics.exists(), allow_cpu=allow_cpu)
        if kind == "gdn":
            fit_one_layer(**kwargs)
        else:
            fit_qsa_layer(**kwargs)
        # Fit modules/optimizer tensors are out of scope now; release their
        # cached CUDA blocks before a later layer reloads the 32B teacher.
        _release_cuda_memory()
        _verify_fit(fit_dir, kind, layer, expected)
        if on_layer_complete is not None:
            if (sha256_file(data_file) != expected["data_sha256"]
                    or sha256_file(Path(data_file).with_name("data_manifest.json")) != expected["manifest_sha256"]):
                raise RuntimeError("calibration data changed during the full run; refuse the next layer upload")
            on_layer_complete(kind, layer, fit_dir, expected)
    if sha256_file(data_file) != expected["data_sha256"] or sha256_file(Path(data_file).with_name("data_manifest.json")) != expected["manifest_sha256"]:
        raise RuntimeError("calibration data changed during the full run")
    if not assemble:
        plan.update({"dry_run": False, "completed_layers": num_layers, "assembled": False})
        return plan
    assembled = root / "overlay"
    result = assemble_flash_next_overlay(base_config=base_config, base_model_id=expected["model_id"],
                                         base_model_revision=expected["model_revision"], gdn_fit_dir=gdn_dir,
                                         qsa_fit_dir=qsa_dir, output_dir=assembled, overwrite=resume)
    result["calibration_data"]["token_sha256"] = expected["token_sha256"]
    from .overlay import OVERLAY_FILENAME

    manifest_file = assembled / OVERLAY_FILENAME
    manifest_file.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    plan.update({"dry_run": False, "overlay_dir": str(assembled), "overlay_manifest": result})
    return plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fit and assemble all 32 Flash-Next-inspired attention layers")
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--min-free-gib", type=float, default=66.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true", help="tiny fixture tests only")
    args = parser.parse_args(argv)
    report = run_full_conversion(**vars(args))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
