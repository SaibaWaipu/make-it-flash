"""Portable adapter-overlay assembly and loading for Qwen3-MoE Flash-Next."""

from __future__ import annotations

import json
import math
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

from safetensors import safe_open
from safetensors.torch import load_file

from .model import (
    GDNQwen3MoeAttentionAdapter,
    create_flash_next_cache,
    flash_next_attention_schedule,
    get_decoder_layers,
    graft_attention_modules,
    make_gdn,
    make_qsa,
)
from .provenance import (
    checkpoint_provenance,
    sha256_file,
    validate_base_config_provenance,
    validate_calibration_data_provenance,
    validate_model_revision,
)

OVERLAY_FILENAME = "flash_next_overlay.json"
OVERLAY_KIND = "make_it_flash.qwen3_moe_flash_next_overlay"
OVERLAY_VERSION = 1
_PROVENANCE_FIELDS = (
    "transformers_version",
    "make_it_flash_version",
    "checkpoint_schema_version",
    "implementation_fingerprint",
)

_BASE_FIELDS = (
    "model_type",
    "hidden_size",
    "vocab_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "rms_norm_eps",
    "max_position_embeddings",
    "num_experts",
    "num_experts_per_tok",
    "moe_intermediate_size",
)


def _config_fields(config: Any) -> dict[str, Any]:
    if getattr(config, "model_type", None) != "qwen3_moe":
        raise ValueError(f"expected qwen3_moe base config, got {getattr(config, 'model_type', None)!r}")
    fields: dict[str, Any] = {"model_type": "qwen3_moe"}
    for name in _BASE_FIELDS[1:]:
        value = getattr(config, name, None)
        if value is not None:
            if name in {"rms_norm_eps"}:
                fields[name] = float(value)
            else:
                fields[name] = int(value)
    required = {"hidden_size", "vocab_size", "num_hidden_layers", "num_attention_heads"}
    missing = required - fields.keys()
    if missing:
        raise ValueError(f"base config is missing required fields: {sorted(missing)}")
    return fields


def _read_checkpoint_metadata(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"missing fitted layer checkpoint: {path}")
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    required = {"model_id", "model_revision", "teacher_layer", "module", "calibration_data", *_PROVENANCE_FIELDS}
    missing = required - metadata.keys()
    if missing:
        raise ValueError(f"{path.name} lacks checkpoint metadata: {sorted(missing)}")
    return metadata


def _read_json_metadata(metadata: dict[str, str], key: str, path: Path) -> dict[str, Any]:
    try:
        value = json.loads(metadata[key])
    except (KeyError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path.name} has invalid {key} metadata") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} {key} metadata must be an object")
    return value


def _validate_fit_metrics(
    *,
    kind: str,
    fit_dir: Path,
    checkpoint_path: Path,
    layer: int,
    base_model_id: str,
    base_model_revision: str,
) -> dict[str, Any]:
    metrics_path = fit_dir / (
        f"fit_layer_{layer:02d}.json" if kind == "gdn" else f"fit_qsa_layer_{layer:02d}.json"
    )
    if not metrics_path.is_file():
        raise FileNotFoundError(f"missing fit metrics needed for overlay quality gate: {metrics_path}")
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid fit metrics JSON: {metrics_path}") from exc
    if not isinstance(metrics, dict):
        raise ValueError(f"fit metrics must be an object: {metrics_path}")
    if (
        metrics.get("model_id") != base_model_id
        or metrics.get("model_revision") != base_model_revision
        or metrics.get("teacher_layer") != layer
    ):
        raise ValueError(f"fit metrics identity does not match base layer {layer}")
    calibration_data = validate_calibration_data_provenance(metrics.get("calibration_data"))
    if (
        metrics.get("checkpoint") != checkpoint_path.name
        or metrics.get("checkpoint_sha256") != sha256_file(checkpoint_path)
    ):
        raise ValueError(f"fit metrics do not authenticate the {kind.upper()} checkpoint for layer {layer}")
    if int(metrics.get("steps", 0)) <= 0:
        raise ValueError(f"layer {layer} has no completed fitting steps")
    if int(metrics.get("num_train_sequences", 0)) <= 0 or int(metrics.get("num_validation_sequences", 0)) <= 0:
        raise ValueError(f"layer {layer} needs nonempty train and independent validation sets")
    if metrics.get("validation_uses_train_fallback") is not False:
        raise ValueError(f"layer {layer} lacks an independent validation split")

    selector_validation: dict[str, float] | None = None
    if kind == "gdn":
        initial = float(metrics.get("initial_validation_mse", float("nan")))
        best = float(metrics.get("best_validation_mse", float("nan")))
    else:
        initial_record = metrics.get("initial_validation")
        final_record = metrics.get("final_validation")
        if not isinstance(initial_record, dict) or not isinstance(final_record, dict):
            raise ValueError(f"QSA layer {layer} lacks initial/final validation metrics")
        initial = float(initial_record.get("total_loss", float("nan")))
        best = float(metrics.get("best_validation_total_loss", float("nan")))
        selector_weight = float(metrics.get("selector_loss_weight", 0.0))
        initial_selector = float(initial_record.get("selection_loss", float("nan")))
        final_selector = float(final_record.get("selection_loss", float("nan")))
        selected_selector = float(metrics.get("selected_checkpoint_validation_selection_loss", float("nan")))
        if not math.isfinite(selector_weight) or selector_weight <= 0:
            raise ValueError(f"QSA layer {layer} needs a positive selector_loss_weight")
        if not math.isfinite(initial_selector) or not math.isfinite(final_selector) or initial_selector <= 0 or final_selector < 0:
            raise ValueError(f"QSA layer {layer} has invalid selector validation losses")
        initial_mse = float(initial_record.get("mse", float("nan")))
        final_mse = float(final_record.get("mse", float("nan")))
        if not math.isfinite(initial_mse) or not math.isfinite(final_mse) or initial_mse < 0 or final_mse < 0:
            raise ValueError(f"QSA layer {layer} lacks finite independent validation output MSE")
        if final_mse > initial_mse + max(1e-9, 1e-6 * initial_mse):
            raise ValueError(f"QSA layer {layer} output MSE degraded despite improving selector/total loss")
        if not math.isclose(final_selector, selected_selector, rel_tol=1e-6, abs_tol=1e-9):
            raise ValueError(f"QSA layer {layer} selected-checkpoint selector metric is inconsistent")
        selector_improvement = (initial_selector - final_selector) / initial_selector
        if selector_improvement <= 1e-6:
            raise ValueError(f"QSA layer {layer} selector validation loss did not improve")
        if int(metrics.get("selector_pruning_examples", 0)) <= 0:
            raise ValueError(
                f"QSA layer {layer} was not trained on sequences long enough to exercise top-k pruning"
            )
        pruning_validation_examples = int(metrics.get("selector_pruning_validation_examples", 0))
        if pruning_validation_examples <= 0:
            raise ValueError(
                f"QSA layer {layer} has no independent validation example that exercises top-k pruning"
            )
        selector_validation = {
            "selector_loss_weight": selector_weight,
            "initial_selection_loss": initial_selector,
            "selected_checkpoint_selection_loss": final_selector,
            "selection_relative_improvement": selector_improvement,
            "selector_pruning_validation_examples": pruning_validation_examples,
        }
    if not math.isfinite(initial) or not math.isfinite(best) or initial <= 0 or best < 0:
        raise ValueError(f"layer {layer} has invalid validation losses")
    relative_improvement = (initial - best) / initial
    if relative_improvement <= 1e-6:
        raise ValueError(f"layer {layer} validation loss did not improve; refusing untrained/degraded overlay")
    result = {
        "metrics_file": metrics_path.name,
        "calibration_data": calibration_data,
        "steps": int(metrics["steps"]),
        "num_train_sequences": int(metrics["num_train_sequences"]),
        "num_validation_sequences": int(metrics["num_validation_sequences"]),
        "initial_validation_loss": initial,
        "best_validation_loss": best,
        "relative_improvement": relative_improvement,
    }
    if selector_validation is not None:
        result.update(selector_validation)
    return result


def _sha256(path: Path) -> str:
    return sha256_file(path)


def _validate_revision(revision: str) -> None:
    validate_model_revision(revision)


def _validate_base_provenance(config: Any, model_id: str, revision: str) -> None:
    validate_base_config_provenance(config, model_id, revision)


def _publish_staged_overlay(stage: Path, target: Path, *, overwrite: bool) -> None:
    backup: Path | None = None
    if target.exists():
        if target.is_symlink() or not target.is_dir():
            raise ValueError(f"overlay destination must be a regular directory: {target}")
        if not overwrite:
            raise FileExistsError(f"overlay output already exists under {target}; pass overwrite=True")
        backup = target.with_name(f".{target.name}.backup-{uuid.uuid4().hex}")
        target.rename(backup)
    try:
        stage.rename(target)
    except Exception:
        if backup is not None and backup.exists():
            backup.rename(target)
        raise
    if backup is not None:
        shutil.rmtree(backup, ignore_errors=True)


def assemble_flash_next_overlay(
    *,
    base_config: Any,
    base_model_id: str,
    base_model_revision: str,
    gdn_fit_dir: str | Path,
    qsa_fit_dir: str | Path,
    output_dir: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Package all scheduled per-layer fits into a base-model adapter overlay.

    This copies only the fitted GDN/QSA attention modules and a manifest. It does
    not copy or merge the 32B base weights, tokenizer, MoE, or language-model head.
    """
    if not isinstance(overwrite, bool):
        raise ValueError("overwrite must be a bool")
    if not base_model_id or not base_model_revision:
        raise ValueError("base_model_id and base_model_revision must be pinned")
    _validate_revision(base_model_revision)
    _validate_base_provenance(base_config, base_model_id, base_model_revision)
    base_fields = _config_fields(base_config)
    layer_count = base_fields["num_hidden_layers"]
    gdn_layers, qsa_layers = flash_next_attention_schedule(layer_count)
    gdn_root = Path(gdn_fit_dir)
    qsa_root = Path(qsa_fit_dir)
    target_root = Path(output_dir)
    if target_root.is_symlink() or (target_root.exists() and not target_root.is_dir()):
        raise ValueError(f"overlay destination must be a regular directory: {target_root}")
    if target_root.exists() and not overwrite:
        raise FileExistsError(f"overlay output already exists under {target_root}; pass overwrite=True")
    target_root.parent.mkdir(parents=True, exist_ok=True)

    source_files: dict[str, dict[int, Path]] = {"gdn": {}, "qsa": {}}
    fit_validation: dict[str, dict[str, dict[str, Any]]] = {"gdn": {}, "qsa": {}}
    gdn_config: dict[str, Any] | None = None
    qsa_config: dict[str, Any] | None = None
    expected_metadata = {
        "gdn": "transformers.models.qwen3_next.Qwen3NextGatedDeltaNet",
        "qsa": "make_it_flash.QSAQwen3MoeAttentionAdapter",
    }
    runtime_provenance = checkpoint_provenance()
    observed_provenance: dict[str, str] | None = None
    observed_calibration_data: dict[str, Any] | None = None
    for kind, indices, source_root, config_key in (
        ("gdn", gdn_layers, gdn_root, "gdn_config"),
        ("qsa", qsa_layers, qsa_root, "qsa_config"),
    ):
        for index in indices:
            checkpoint = source_root / f"{kind}_layer_{index:02d}.safetensors"
            metadata = _read_checkpoint_metadata(checkpoint)
            if metadata["model_id"] != base_model_id or metadata["model_revision"] != base_model_revision:
                raise ValueError(f"{checkpoint.name} was fitted against a different base model/revision")
            if metadata["teacher_layer"] != str(index):
                raise ValueError(f"{checkpoint.name} teacher_layer metadata does not match layer {index}")
            if metadata["module"] != expected_metadata[kind]:
                raise ValueError(f"{checkpoint.name} contains an unsupported {kind.upper()} module type")
            checkpoint_provenance_record = {key: metadata[key] for key in _PROVENANCE_FIELDS}
            if checkpoint_provenance_record != runtime_provenance:
                raise ValueError(f"{checkpoint.name} was produced by an incompatible runtime")
            if observed_provenance is not None and checkpoint_provenance_record != observed_provenance:
                raise ValueError("fit checkpoints were produced by different runtimes")
            observed_provenance = checkpoint_provenance_record
            config_value = _read_json_metadata(metadata, config_key, checkpoint)
            current_config = gdn_config if kind == "gdn" else qsa_config
            if current_config is None:
                current_config = config_value
            elif current_config != config_value:
                raise ValueError(f"{kind.upper()} fit checkpoints use inconsistent module configurations")
            if kind == "gdn":
                gdn_config = current_config
            else:
                qsa_config = current_config
            checkpoint_calibration_data = validate_calibration_data_provenance(
                _read_json_metadata(metadata, "calibration_data", checkpoint)
            )
            fit_record = _validate_fit_metrics(
                kind=kind,
                fit_dir=source_root,
                checkpoint_path=checkpoint,
                layer=index,
                base_model_id=base_model_id,
                base_model_revision=base_model_revision,
            )
            if fit_record["calibration_data"] != checkpoint_calibration_data:
                raise ValueError(f"{checkpoint.name} calibration provenance differs from fit metrics")
            if observed_calibration_data is not None and checkpoint_calibration_data != observed_calibration_data:
                raise ValueError("fit layers were trained on different calibration data")
            observed_calibration_data = checkpoint_calibration_data
            fit_record.pop("calibration_data")
            fit_validation[kind][str(index)] = fit_record
            source_files[kind][index] = checkpoint

    if gdn_config is None or qsa_config is None:
        raise ValueError("the requested schedule requires at least one fitted GDN and one fitted QSA layer")
    if observed_calibration_data is None:
        raise ValueError("overlay lacks calibration-data provenance")
    expected_gdn_config = {
        "linear_key_head_dim": int(getattr(base_config, "head_dim", base_fields["hidden_size"] // base_fields["num_attention_heads"])),
        "linear_value_head_dim": int(getattr(base_config, "head_dim", base_fields["hidden_size"] // base_fields["num_attention_heads"])),
        "linear_conv_kernel_dim": 4,
    }
    for name, expected in expected_gdn_config.items():
        if int(gdn_config.get(name, -1)) != expected:
            raise ValueError(f"GDN fit config {name}={gdn_config.get(name)!r} does not match loader value {expected}")
    required_qsa_fields = {
        "num_heads", "num_key_value_heads", "head_dim", "rotary_dim", "rope_theta",
        "index_n_heads", "index_head_dim", "token_budget", "compress_ratio",
    }
    missing_qsa_fields = required_qsa_fields - qsa_config.keys()
    if missing_qsa_fields:
        raise ValueError(f"QSA fit config is missing fields: {sorted(missing_qsa_fields)}")

    with tempfile.TemporaryDirectory(
        prefix=f".{target_root.name}.staging-", dir=target_root.parent
    ) as staging_path:
        staging_root = Path(staging_path)
        module_entries: dict[str, dict[str, dict[str, str]]] = {"gdn": {}, "qsa": {}}
        for kind in ("gdn", "qsa"):
            for index, source in source_files[kind].items():
                relative = Path("modules") / kind / source.name
                destination = staging_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
                module_entries[kind][str(index)] = {
                    "file": relative.as_posix(),
                    "sha256": _sha256(destination),
                }

        manifest = {
            "format": OVERLAY_KIND,
            "format_version": OVERLAY_VERSION,
            "base_model_id": base_model_id,
            "base_model_revision": base_model_revision,
            "calibration_data": observed_calibration_data,
            "provenance": runtime_provenance,
            "base_config": base_fields,
            "schedule": {"gdn_layers": list(gdn_layers), "qsa_layers": list(qsa_layers)},
            "gdn_config": gdn_config,
            "qsa_config": qsa_config,
            "modules": module_entries,
            "fit_validation": fit_validation,
            "note": "adapter overlay only; each module has independent validation improvement and QSA pruning evidence; load the exact base model separately; no base weights or tokenizer are included",
        }
        (staging_root / OVERLAY_FILENAME).write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        _publish_staged_overlay(staging_root, target_root, overwrite=overwrite)
    return manifest


def _overlay_file(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    resolved_root = root.resolve()
    if candidate != resolved_root and resolved_root not in candidate.parents:
        raise ValueError(f"overlay module path escapes its root: {relative!r}")
    return candidate


def load_flash_next_overlay(
    model: Any,
    overlay_dir: str | Path,
    *,
    base_model_id: str,
    base_model_revision: str,
    verify_sha256: bool = True,
) -> Any:
    """Load and graft an overlay onto an already-loaded, exact base model.

    Returns an empty ``FlashNextDynamicCache`` configured for the 3-GDN/1-QSA
    schedule. Pass it explicitly as ``past_key_values`` on the first cached
    forward or ``generate`` call. Transformers' default ``DynamicCache`` lacks
    QSA indexer-position state and is intentionally unsupported. The returned
    cache is mutable and should be fresh for each generation request.
    """
    if not isinstance(verify_sha256, bool):
        raise ValueError("verify_sha256 must be a bool")
    _validate_revision(base_model_revision)
    root = Path(overlay_dir)
    manifest_path = root / OVERLAY_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing overlay manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != OVERLAY_KIND or manifest.get("format_version") != OVERLAY_VERSION:
        raise ValueError("unsupported Flash-Next overlay format")
    if manifest.get("base_model_id") != base_model_id or manifest.get("base_model_revision") != base_model_revision:
        raise ValueError("overlay base model ID/revision does not match the requested base")
    runtime_provenance = checkpoint_provenance()
    if manifest.get("provenance") != runtime_provenance:
        raise ValueError("overlay runtime versions do not match the fitted module provenance")

    config = getattr(model, "config", None)
    if config is None:
        raise ValueError("base model must expose a Transformers config")
    decoder_config = config.get_text_config(decoder=True) if hasattr(config, "get_text_config") else config
    actual_fields = _config_fields(decoder_config)
    expected_fields = manifest.get("base_config", {})
    for name, expected in expected_fields.items():
        if actual_fields.get(name) != expected:
            raise ValueError(f"base model config mismatch for {name}: expected {expected!r}, got {actual_fields.get(name)!r}")
    _validate_base_provenance(config, base_model_id, base_model_revision)

    decoder_layers = get_decoder_layers(model)
    if len(decoder_layers) != expected_fields["num_hidden_layers"]:
        raise ValueError("loaded base model layer count does not match the overlay")
    gdn_layers, qsa_layers = flash_next_attention_schedule(len(decoder_layers))
    schedule = manifest.get("schedule", {})
    if schedule.get("gdn_layers") != list(gdn_layers) or schedule.get("qsa_layers") != list(qsa_layers):
        raise ValueError("overlay does not contain the expected complete 3-GDN/1-QSA schedule")

    module_entries = manifest.get("modules", {})
    replacements: dict[int, Any] = {}
    for kind, indices in (("gdn", gdn_layers), ("qsa", qsa_layers)):
        entries = module_entries.get(kind, {})
        if set(entries) != {str(index) for index in indices}:
            raise ValueError(f"overlay is missing one or more scheduled {kind.upper()} modules")
        for index in indices:
            entry = entries[str(index)]
            checkpoint = _overlay_file(root, entry["file"])
            if not checkpoint.is_file():
                raise FileNotFoundError(f"missing overlay module file: {checkpoint}")
            if verify_sha256 and _sha256(checkpoint) != entry.get("sha256"):
                raise ValueError(f"SHA-256 mismatch for overlay module {entry['file']}")
            metadata = _read_checkpoint_metadata(checkpoint)
            expected_module = (
                "transformers.models.qwen3_next.Qwen3NextGatedDeltaNet"
                if kind == "gdn"
                else "make_it_flash.QSAQwen3MoeAttentionAdapter"
            )
            if (
                metadata["model_id"] != base_model_id
                or metadata["model_revision"] != base_model_revision
                or metadata["teacher_layer"] != str(index)
                or metadata["module"] != expected_module
                or {key: metadata[key] for key in _PROVENANCE_FIELDS} != runtime_provenance
            ):
                raise ValueError(f"overlay module metadata does not match base layer {index}")
            config_key = "gdn_config" if kind == "gdn" else "qsa_config"
            if _read_json_metadata(metadata, config_key, checkpoint) != manifest[config_key]:
                raise ValueError(f"overlay module config metadata mismatch at layer {index}")
            state_dict = load_file(str(checkpoint), device="cpu")
            if kind == "gdn":
                gdn_config = manifest["gdn_config"]
                gdn = make_gdn(
                    decoder_config,
                    index,
                    key_heads=int(gdn_config["linear_num_key_heads"]),
                    value_heads=int(gdn_config["linear_num_value_heads"]),
                )
                gdn.load_state_dict(state_dict, strict=True)
                replacements[index] = GDNQwen3MoeAttentionAdapter(gdn)
            else:
                qsa = make_qsa(decoder_config, layer_idx=index, **manifest["qsa_config"])
                qsa.load_state_dict(state_dict, strict=True)
                replacements[index] = qsa

    had_layer_types = hasattr(decoder_config, "layer_types")
    previous_layer_types = getattr(decoder_config, "layer_types", None)
    try:
        cache = create_flash_next_cache(model, gdn_layers=gdn_layers, qsa_layers=qsa_layers)
        graft_attention_modules(model, replacements)
    except Exception:
        if had_layer_types:
            decoder_config.layer_types = previous_layer_types
        else:
            delattr(decoder_config, "layer_types")
        raise
    return cache
