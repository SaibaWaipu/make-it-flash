"""Integrate staged GDN/QSA tensors into the complete pinned base checkpoint.

The exporter rewrites one Safetensors shard at a time and uploads it directly to
an approved private Hub repo. It does not instantiate the 32B model or run
inference; model loadability and quality are intentionally not claimed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from huggingface_hub import CommitOperationAdd, HfApi, get_token
from huggingface_hub.errors import RepositoryNotFoundError

from make_it_flash.full_checkpoint import (
    cast_safetensors_to_bfloat16,
    read_safetensors_header,
    rewrite_safetensors_shard,
)
from make_it_flash.model import flash_next_attention_schedule
from make_it_flash.provenance import sha256_file

BASE_REPO = "llm-jp/llm-jp-4.1-32b-a3b-thinking"
BASE_REVISION = "cda260706786758045e5e96bf4d738bbc01155b5"
STAGED_REPO = "RemydreScarlet/llm-jp-4.1-flash-next"
STAGED_REVISION = "58c691aff794b1a1c458421f8beb152d5e11c619"
TARGET_REPO = "RemydreScarlet/llm-jp-4.1-flash-prepreview"
RUN_ID = "flashnext-32-run01"
DEFAULT_OVERLAY_DIR = Path("artifacts/assembled/flashnext-32-run01")
DEFAULT_OUTPUT_DIR = Path("artifacts/full-checkpoint-export")
_INDEX_NAME = "model.safetensors.index.json"
_CONFIG_NAME = "config.json"
_README_NAME = "README.md"
_FULL_MANIFEST_NAME = "flash_next_full_checkpoint.json"
_SOURCE_MANIFEST_NAME = "flash_next_source_overlay_manifest.json"
_REMOTE_CODE_FILES = {
    "configuration_flashnext.py": Path("remote_code/configuration_flashnext.py"),
    "modeling_flashnext.py": Path("remote_code/modeling_flashnext.py"),
    "model.py": Path("src/make_it_flash/model.py"),
    "flash_next.py": Path("src/make_it_flash/flash_next.py"),
    "hybrid_cache.py": Path("src/make_it_flash/hybrid_cache.py"),
}
_REMOTE_AUTO_MAP = {
    "AutoConfig": "configuration_flashnext.FlashNextQwen3MoeConfig",
    "AutoModel": "modeling_flashnext.FlashNextQwen3MoeModel",
    "AutoModelForCausalLM": "modeling_flashnext.FlashNextQwen3MoeForCausalLM",
}
_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _attention_prefix(layer: int) -> str:
    return f"model.layers.{layer}.self_attn."


def _module_prefix(kind: str, layer: int) -> str:
    prefix = _attention_prefix(layer)
    return f"{prefix}gdn." if kind == "gdn" else prefix


def _verify_overlay(overlay_dir: Path) -> tuple[dict[str, Any], dict[int, Path], str]:
    manifest_path = overlay_dir / "flash_next_overlay.json"
    manifest = _read_json(manifest_path)
    if manifest.get("base_model_id") != BASE_REPO or manifest.get("base_model_revision") != BASE_REVISION:
        raise ValueError("overlay was fitted against a different base model or revision")
    if manifest.get("run_id", RUN_ID) != RUN_ID:
        raise ValueError("overlay run ID does not match the approved staged run")
    gdn_layers, qsa_layers = flash_next_attention_schedule(32)
    schedule = manifest.get("schedule", {})
    if schedule.get("gdn_layers") != list(gdn_layers) or schedule.get("qsa_layers") != list(qsa_layers):
        raise ValueError("overlay does not match the expected 24 GDN / 8 QSA schedule")
    module_paths: dict[int, Path] = {}
    for kind, layers in (("gdn", gdn_layers), ("qsa", qsa_layers)):
        entries = manifest.get("modules", {}).get(kind, {})
        if set(entries) != {str(layer) for layer in layers}:
            raise ValueError(f"overlay manifest has an incomplete {kind} layer set")
        for layer in layers:
            entry = entries[str(layer)]
            path = overlay_dir / entry["file"]
            if not path.is_file() or sha256_file(path) != entry.get("sha256"):
                raise ValueError(f"overlay module layer {layer} is missing or has a bad hash")
            module_paths[layer] = path
    return manifest, module_paths, sha256_file(manifest_path)


def _load_base_metadata(api: HfApi, scratch_parent: Path) -> tuple[dict, dict, list[str]]:
    with TemporaryDirectory(prefix="full-checkpoint-meta-", dir=scratch_parent) as temp:
        cache = Path(temp) / "cache"
        config_path = Path(api.hf_hub_download(
            repo_id=BASE_REPO, filename=_CONFIG_NAME, repo_type="model",
            revision=BASE_REVISION, cache_dir=cache,
        ))
        index_path = Path(api.hf_hub_download(
            repo_id=BASE_REPO, filename=_INDEX_NAME, repo_type="model",
            revision=BASE_REVISION, cache_dir=cache,
        ))
        config = _read_json(config_path)
        index = _read_json(index_path)
    files = api.list_repo_files(BASE_REPO, repo_type="model", revision=BASE_REVISION)
    return config, index, files


def _build_plan(
    *, config: dict, index: dict, source_files: list[str],
    overlay_manifest: dict, overlay_dir: Path, module_paths: dict[int, Path],
) -> dict[str, Any]:
    if config.get("model_type") != "qwen3_moe" or config.get("num_hidden_layers") != 32:
        raise ValueError("pinned base config is not the expected 32-layer Qwen3-MoE model")
    base_fields = overlay_manifest.get("base_config", {})
    for field in ("model_type", "hidden_size", "vocab_size", "num_hidden_layers"):
        if base_fields.get(field) != config.get(field):
            raise ValueError(f"overlay and base model differ on {field}")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("base model index lacks a weight_map")
    shards = sorted(set(weight_map.values()))
    if len(shards) != 13 or not all(shard in source_files for shard in shards):
        raise ValueError("pinned base model does not contain its expected 13 Safetensors shards")

    gdn_layers, qsa_layers = flash_next_attention_schedule(32)
    kinds = {layer: "gdn" for layer in gdn_layers} | {layer: "qsa" for layer in qsa_layers}
    layer_shards: dict[int, str] = {}
    additions_by_shard: dict[str, list[tuple[Path, str]]] = {shard: [] for shard in shards}
    added_weight_map: dict[str, str] = {}
    source_module_tensor_bytes = 0
    added_tensor_bytes = 0
    for layer in range(32):
        query_key = f"{_attention_prefix(layer)}q_proj.weight"
        shard = weight_map.get(query_key)
        if shard not in additions_by_shard:
            raise ValueError(f"base index does not map {query_key} to a known shard")
        layer_shards[layer] = shard
        kind = kinds[layer]
        module_path = module_paths[layer]
        prefix = _module_prefix(kind, layer)
        additions_by_shard[shard].append((module_path, prefix))
        header, _ = read_safetensors_header(module_path)
        for tensor_name, entry in header.items():
            if tensor_name == "__metadata__":
                continue
            merged_name = f"{prefix}{tensor_name}"
            if merged_name in added_weight_map:
                raise ValueError(f"duplicate fitted tensor name: {merged_name}")
            added_weight_map[merged_name] = shard
            start, end = entry["data_offsets"]
            source_size = end - start
            if entry.get("dtype") != "F32" or source_size % 2:
                raise ValueError(f"expected an F32 fitted tensor for BF16 integration: {module_path}:{tensor_name}")
            source_module_tensor_bytes += source_size
            added_tensor_bytes += source_size // 2

    remove_prefixes = tuple(_attention_prefix(layer) for layer in range(32))
    removed_weight_keys = {
        name for name in weight_map
        if any(name.startswith(prefix) for prefix in remove_prefixes)
    }
    if not removed_weight_keys:
        raise ValueError("base index has no attention tensors to replace")
    retained_weight_map = {
        name: shard for name, shard in weight_map.items()
        if name not in removed_weight_keys
    }
    duplicate = set(retained_weight_map).intersection(added_weight_map)
    if duplicate:
        raise ValueError(f"fitted tensors collide with retained base weights: {sorted(duplicate)[:5]}")
    integrated_weight_map = retained_weight_map | added_weight_map
    if set(integrated_weight_map.values()) != set(shards):
        raise ValueError("integrated checkpoint index leaves an empty or unreferenced model shard")
    missing = set(shards) - set(source_files)
    if missing:
        raise ValueError(f"base repo is missing checkpoint shards: {sorted(missing)}")
    non_weight_files = [
        name for name in source_files
        if name not in set(shards) and name not in {_CONFIG_NAME, _INDEX_NAME}
    ]
    return {
        "shards": shards,
        "layer_shards": layer_shards,
        "additions_by_shard": additions_by_shard,
        "added_weight_map": added_weight_map,
        "weight_map": integrated_weight_map,
        "removed_weight_keys": sorted(removed_weight_keys),
        "remove_prefixes": remove_prefixes,
        "added_tensor_bytes": added_tensor_bytes,
        "source_module_tensor_bytes_f32": source_module_tensor_bytes,
        "non_weight_files": non_weight_files,
        "base_tensor_bytes": int(index.get("metadata", {}).get("total_size", 0)),
        "gdn_layers": list(gdn_layers),
        "qsa_layers": list(qsa_layers),
        "kinds": kinds,
        "overlay_dir": str(overlay_dir),
    }


def _repo_state(api: HfApi, *, create: bool, allowed_files: set[str]) -> tuple[str | None, set[str]]:
    try:
        info = api.model_info(TARGET_REPO, files_metadata=False)
    except RepositoryNotFoundError:
        if not create:
            return None, set()
        api.create_repo(repo_id=TARGET_REPO, repo_type="model", private=True, exist_ok=False)
        info = api.model_info(TARGET_REPO, files_metadata=False)
    if not info.private:
        raise ValueError(f"target repo must remain private: {TARGET_REPO}")
    files = set(api.list_repo_files(TARGET_REPO, repo_type="model"))
    unknown = files - allowed_files
    if unknown:
        raise ValueError(f"refusing to overwrite unexpected files in target repo: {sorted(unknown)[:10]}")
    return info.sha, files


def _upload_base_metadata(api: HfApi, files: list[str], scratch_parent: Path) -> list[str]:
    existing = set(api.list_repo_files(TARGET_REPO, repo_type="model"))
    operations = []
    uploaded_names: list[str] = []
    with TemporaryDirectory(prefix="full-checkpoint-small-files-", dir=scratch_parent) as temp:
        cache = Path(temp) / "cache"
        for source_name in files:
            destination_name = "UPSTREAM_README.md" if source_name == _README_NAME else source_name
            if destination_name in existing:
                continue
            path = api.hf_hub_download(
                repo_id=BASE_REPO, filename=source_name, repo_type="model",
                revision=BASE_REVISION, cache_dir=cache,
            )
            operations.append(CommitOperationAdd(path_in_repo=destination_name, path_or_fileobj=path))
            uploaded_names.append(destination_name)
        if operations:
            parent = api.model_info(TARGET_REPO, files_metadata=False).sha
            api.create_commit(
                repo_id=TARGET_REPO, repo_type="model", parent_commit=parent,
                operations=operations,
                commit_message="Copy pinned base tokenizer and support files",
            )
    return uploaded_names


def _upload_metadata_if_changed(api: HfApi, payloads: list[tuple[str, bytes]], scratch_parent: Path) -> None:
    """Upload final small metadata files without creating empty resume commits."""
    current = api.model_info(TARGET_REPO, files_metadata=False)
    existing = set(api.list_repo_files(TARGET_REPO, repo_type="model"))
    operations = []
    with TemporaryDirectory(prefix="full-checkpoint-verify-meta-", dir=scratch_parent) as temp:
        cache = Path(temp) / "cache"
        for name, payload in payloads:
            if name in existing:
                current_path = api.hf_hub_download(
                    repo_id=TARGET_REPO,
                    filename=name,
                    repo_type="model",
                    revision=current.sha,
                    cache_dir=cache,
                )
                if Path(current_path).read_bytes() == payload:
                    continue
            operations.append(CommitOperationAdd(path_in_repo=name, path_or_fileobj=payload))
    if operations:
        api.create_commit(
            repo_id=TARGET_REPO,
            repo_type="model",
            parent_commit=current.sha,
            operations=operations,
            commit_message="Publish integrated 32-layer full checkpoint metadata",
        )


def _metadata_bytes(
    *, config: dict, index: dict, plan: dict[str, Any], overlay_manifest: dict,
    overlay_sha256: str, shard_results: dict[str, dict[str, Any]],
) -> tuple[bytes, bytes, bytes, bytes]:
    removed_keys = set(plan["removed_weight_keys"])
    actually_removed = {
        key for result in shard_results.values() for key in result["removed_keys"]
    }
    if actually_removed != removed_keys:
        missing = sorted(removed_keys - actually_removed)
        unexpected = sorted(actually_removed - removed_keys)
        raise ValueError(f"rewritten attention keys mismatch (missing={missing[:5]}, unexpected={unexpected[:5]})")
    expected_added = set(plan["added_weight_map"])
    actually_added = {
        key for result in shard_results.values() for key in result["added_keys"]
    }
    if actually_added != expected_added:
        missing = sorted(expected_added - actually_added)
        unexpected = sorted(actually_added - expected_added)
        raise ValueError(f"rewritten fitted keys mismatch (missing={missing[:5]}, unexpected={unexpected[:5]})")

    removed_bytes = sum(int(result["removed_tensor_bytes"]) for result in shard_results.values())
    added_bytes = sum(int(result["added_tensor_bytes"]) for result in shard_results.values())
    source_bytes = sum(int(result["source_tensor_bytes"]) for result in shard_results.values())
    output_bytes = sum(int(result["tensor_bytes"]) for result in shard_results.values())
    if source_bytes != plan["base_tensor_bytes"]:
        raise ValueError(
            f"source shard tensor bytes ({source_bytes}) differ from index total_size ({plan['base_tensor_bytes']})"
        )
    if added_bytes != plan["added_tensor_bytes"]:
        raise ValueError("integrated BF16 tensor byte count differs from the module index plan")
    expected_output_bytes = source_bytes - removed_bytes + added_bytes
    if output_bytes != expected_output_bytes:
        raise ValueError("rewritten shard tensor bytes do not match removed/added tensor totals")

    loader_code_hashes = {
        name: sha256_file(_PROJECT_ROOT / source)
        for name, source in _REMOTE_CODE_FILES.items()
    }
    index_copy = json.loads(json.dumps(index))
    index_copy["weight_map"] = dict(sorted(plan["weight_map"].items()))
    index_copy.setdefault("metadata", {})["total_size"] = output_bytes

    config_copy = json.loads(json.dumps(config))
    config_copy["architectures"] = ["FlashNextQwen3MoeForCausalLM"]
    config_copy["auto_map"] = dict(_REMOTE_AUTO_MAP)
    config_copy["layer_types"] = [
        "linear_attention" if layer in plan["gdn_layers"] else "full_attention"
        for layer in range(32)
    ]
    config_copy["use_cache"] = True
    config_copy["flash_next_integration"] = {
        "format": "make_it_flash.integrated_full_checkpoint.v1",
        "run_id": RUN_ID,
        "source_overlay_repo": STAGED_REPO,
        "source_overlay_revision": STAGED_REVISION,
        "base_model_id": BASE_REPO,
        "base_model_revision": BASE_REVISION,
        "gdn_layers": plan["gdn_layers"],
        "qsa_layers": plan["qsa_layers"],
        "gdn_config": overlay_manifest["gdn_config"],
        "qsa_config": overlay_manifest["qsa_config"],
        "custom_cache": "FlashNextDynamicCache",
        "remote_loader_code_sha256": loader_code_hashes,
        "synthetic_loader_round_trip_tested": True,
        "full_checkpoint_load_tested": False,
        "inference_tested": False,
        "quality_evaluation_tested": False,
    }

    integration_manifest = {
        "format": "make_it_flash.integrated_full_checkpoint.v1",
        "run_id": RUN_ID,
        "target_repo": TARGET_REPO,
        "base_model_id": BASE_REPO,
        "base_model_revision": BASE_REVISION,
        "base_index_tensor_bytes": source_bytes,
        "removed_base_attention_tensor_bytes": removed_bytes,
        "source_fitted_tensor_dtype": "F32",
        "integrated_fitted_tensor_dtype": "BF16",
        "source_fitted_attention_tensor_bytes_f32": plan["source_module_tensor_bytes_f32"],
        "added_fitted_attention_tensor_bytes": added_bytes,
        "integrated_tensor_bytes": output_bytes,
        "replaced_base_attention_tensor_count": len(actually_removed),
        "integrated_fitted_tensor_count": len(actually_added),
        "gdn_layers": plan["gdn_layers"],
        "qsa_layers": plan["qsa_layers"],
        "base_attention_tensors_removed": True,
        "fitted_tensors_integrated_into_full_model_shards": True,
        "source_overlay_repo": STAGED_REPO,
        "source_overlay_revision": STAGED_REVISION,
        "source_overlay_manifest_sha256": overlay_sha256,
        "runtime_provenance": overlay_manifest.get("provenance"),
        "calibration_data": overlay_manifest.get("calibration_data"),
        "custom_loader": {
            "auto_map": _REMOTE_AUTO_MAP,
            "code_sha256": loader_code_hashes,
            "synthetic_round_trip_and_cache_generation_tested": True,
            "full_checkpoint_loaded": False,
        },
        "shards": {
            name: {
                "sha256": result["sha256"],
                "file_size": result["file_size"],
                "tensor_bytes": result["tensor_bytes"],
                "removed_tensor_bytes": result["removed_tensor_bytes"],
                "added_tensor_bytes": result["added_tensor_bytes"],
                "removed_tensor_count": len(result["removed_keys"]),
                "added_tensor_count": len(result["added_keys"]),
            }
            for name, result in sorted(shard_results.items())
        },
        "inference_tested": False,
        "quality_evaluation_tested": False,
        "note": "Full base weights are included with the 24 GDN/8 QSA attention replacement. Custom remote-code loading passed tiny synthetic round-trip/cache-generation tests and full index/key/shape checks; the 66 GB checkpoint itself was not loaded and inference quality remains unvalidated.",
    }
    readme = f"""---
license: other
base_model: {BASE_REPO}
---

# LLM-jp 4.1 Flash Prepreview — integrated 32-layer checkpoint

This private preview contains the complete pinned base checkpoint with the fitted Flash-Next attention tensors integrated into the full-model Safetensors shards. It is **not an adapter-only repository**: the original attention tensors at all 32 layers were removed and replaced by 24 GDN and 8 QSA modules. Fitted F32 tensors were cast to BF16 to match the base attention dtype used by the runtime graft; the remaining base tensors, tokenizer, and support files come from the pinned base revision.

- Base: `{BASE_REPO}` at `{BASE_REVISION}`
- Fit run: `{RUN_ID}`
- Staged source: `{STAGED_REPO}` at `{STAGED_REVISION}`
- Schedule: GDN layers {plan['gdn_layers']}; QSA layers {plan['qsa_layers']}
- Transformers/checkpoint provenance: `{overlay_manifest.get('provenance', {})}`

This is an experimental prepreview. It includes custom remote code for the Qwen3-MoE + GDN/QSA architecture; load it with Transformers 5.5.x and `trust_remote_code=True`, for example `AutoModelForCausalLM.from_pretrained(repo_id, trust_remote_code=True, dtype="auto")`. The custom config/model path and a tiny synthetic save/load smoke test are verified, and the expected full-checkpoint tensor names/shapes were checked without loading the 66 GB weights. **The full checkpoint itself has not been loaded or generated from, and no perplexity or Japanese held-out quality evaluation was run.** Runtime quality and performance are unvalidated.

The original upstream model card is preserved as [`UPSTREAM_README.md`](UPSTREAM_README.md). Exact tensor replacement, per-shard hashes, and data provenance are in [`{_FULL_MANIFEST_NAME}`]({_FULL_MANIFEST_NAME}) and [`{_SOURCE_MANIFEST_NAME}`]({_SOURCE_MANIFEST_NAME}).
"""
    return (
        (json.dumps(config_copy, indent=2, ensure_ascii=False) + "\n").encode(),
        (json.dumps(index_copy, indent=2, ensure_ascii=False) + "\n").encode(),
        (json.dumps(integration_manifest, indent=2, ensure_ascii=False) + "\n").encode(),
        readme.encode("utf-8"),
    )


def export_full_checkpoint(
    *, overlay_dir: str | Path = DEFAULT_OVERLAY_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    launch: bool = False,
    api: HfApi | None = None,
) -> dict[str, Any]:
    overlay_root = Path(overlay_dir)
    output_root = Path(output_dir)
    overlay_manifest, module_paths, overlay_sha256 = _verify_overlay(overlay_root)
    hub = api or HfApi(token=get_token())
    config, base_index, source_files = _load_base_metadata(hub, output_root.parent)
    plan = _build_plan(
        config=config, index=base_index, source_files=source_files,
        overlay_manifest=overlay_manifest, overlay_dir=overlay_root,
        module_paths=module_paths,
    )
    base_weight_map = base_index["weight_map"]
    allowed_files = set(source_files) | {
        "UPSTREAM_README.md", _FULL_MANIFEST_NAME, _SOURCE_MANIFEST_NAME,
        *_REMOTE_CODE_FILES.keys(),
    }
    progress_path = output_root / "progress.json"
    progress: dict[str, Any] = {
        "format": "make_it_flash.full_export_progress.v1",
        "target_repo": TARGET_REPO,
        "base_revision": BASE_REVISION,
        "overlay_revision": STAGED_REVISION,
        "shards": {},
    }
    if progress_path.is_file():
        old_progress = _read_json(progress_path)
        if any(old_progress.get(key) != progress[key] for key in ("format", "target_repo", "base_revision", "overlay_revision")):
            raise ValueError("existing export progress belongs to a different checkpoint")
        progress["shards"] = old_progress.get("shards", {})

    summary = {
        "target_repo": TARGET_REPO,
        "target_private": True,
        "base_repo": BASE_REPO,
        "base_revision": BASE_REVISION,
        "staged_repo": STAGED_REPO,
        "staged_revision": STAGED_REVISION,
        "run_id": RUN_ID,
        "base_tensor_bytes": plan["base_tensor_bytes"],
        "expected_removed_attention_keys": len(plan["removed_weight_keys"]),
        "expected_added_module_keys": len(plan["added_weight_map"]),
        "source_module_tensor_bytes_f32": plan["source_module_tensor_bytes_f32"],
        "integrated_module_tensor_bytes_bf16": plan["added_tensor_bytes"],
        "model_shards": len(plan["shards"]),
        "gdn_layers": plan["gdn_layers"],
        "qsa_layers": plan["qsa_layers"],
        "source_overlay_manifest_sha256": overlay_sha256,
        "launch": launch,
    }
    if not launch:
        return summary

    output_root.mkdir(parents=True, exist_ok=True)
    existed_before = False
    try:
        hub.model_info(TARGET_REPO, files_metadata=False)
        existed_before = True
    except RepositoryNotFoundError:
        pass
    if existed_before and not progress_path.is_file():
        existing = set(hub.list_repo_files(TARGET_REPO, repo_type="model"))
        if existing:
            raise ValueError("target repo already has files but no matching local export progress; refusing overwrite")
    _repo_state(hub, create=True, allowed_files=allowed_files)
    _upload_base_metadata(hub, plan["non_weight_files"], output_root.parent)

    # Verify already uploaded shards against the local resume record before skipping.
    remote_info = hub.model_info(TARGET_REPO, files_metadata=True)
    remote_shards = {item.rfilename: item for item in (remote_info.siblings or [])}
    for shard in plan["shards"]:
        saved = progress["shards"].get(shard)
        remote = remote_shards.get(shard)
        if saved and remote and remote.size == saved.get("file_size") and remote.lfs and remote.lfs.sha256 == saved.get("sha256"):
            continue
        with TemporaryDirectory(prefix="full-checkpoint-shard-", dir=output_root.parent) as temp:
            scratch = Path(temp)
            cache_dir = scratch / "cache"
            source_path = Path(hub.hf_hub_download(
                repo_id=BASE_REPO, filename=shard, repo_type="model",
                revision=BASE_REVISION, cache_dir=cache_dir,
            ))
            source_header, _ = read_safetensors_header(source_path)
            layers_in_shard = [layer for layer, layer_shard in plan["layer_shards"].items() if layer_shard == shard]
            for layer in layers_in_shard:
                query_key = f"{_attention_prefix(layer)}q_proj.weight"
                if source_header.get(query_key, {}).get("dtype") != "BF16":
                    raise ValueError(f"expected BF16 base attention weights at layer {layer}")
            converted_additions = []
            converted_dir = scratch / "bf16-modules"
            converted_dir.mkdir()
            for module_path, prefix in plan["additions_by_shard"][shard]:
                converted_path = converted_dir / module_path.name
                conversion = cast_safetensors_to_bfloat16(
                    module_path,
                    converted_path,
                    metadata_updates={
                        "make_it_flash_integrated_dtype": "BF16",
                        "make_it_flash_runtime_dtype": "BF16",
                    },
                )
                if conversion["output_dtypes"] != ["torch.bfloat16"]:
                    raise ValueError(f"unexpected converted module dtype: {conversion['output_dtypes']}")
                converted_additions.append((converted_path, prefix))
            output_path = scratch / shard
            result = rewrite_safetensors_shard(
                source_path,
                output_path,
                remove_prefixes=plan["remove_prefixes"],
                additions=converted_additions,
                metadata_updates={
                    "make_it_flash_integrated": "true",
                    "make_it_flash_module_dtype": "BF16",
                    "make_it_flash_run_id": RUN_ID,
                    "make_it_flash_base_revision": BASE_REVISION,
                    "make_it_flash_overlay_manifest_sha256": overlay_sha256,
                },
            )
            expected_removed_here = {
                key for key in plan["removed_weight_keys"]
                if base_weight_map[key] == shard
            }
            if set(result["removed_keys"]) != expected_removed_here:
                raise ValueError(f"base attention key removal mismatch in shard {shard}")
            expected_added_here = {
                key for key, destination_shard in plan["added_weight_map"].items()
                if destination_shard == shard
            }
            if set(result["added_keys"]) != expected_added_here:
                raise ValueError(f"fitted tensor insertion mismatch in shard {shard}")
            parent = hub.model_info(TARGET_REPO, files_metadata=False).sha
            hub.upload_file(
                path_or_fileobj=output_path,
                path_in_repo=shard,
                repo_id=TARGET_REPO,
                repo_type="model",
                parent_commit=parent,
                commit_message=f"Integrate fitted attention weights into {shard}",
            )
            progress["shards"][shard] = result
            progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
            print(json.dumps({"uploaded_shard": shard, "sha256": result["sha256"], "file_size": result["file_size"]}), flush=True)

    if set(progress["shards"]) != set(plan["shards"]):
        raise ValueError("not all base model shards have been integrated")
    config_bytes, index_bytes, manifest_bytes, readme_bytes = _metadata_bytes(
        config=config, index=base_index, plan=plan, overlay_manifest=overlay_manifest,
        overlay_sha256=overlay_sha256, shard_results=progress["shards"],
    )
    output_root.mkdir(parents=True, exist_ok=True)
    config_path = output_root / _CONFIG_NAME
    index_path = output_root / _INDEX_NAME
    manifest_path = output_root / _FULL_MANIFEST_NAME
    readme_path = output_root / _README_NAME
    source_manifest_path = output_root / _SOURCE_MANIFEST_NAME
    config_path.write_bytes(config_bytes)
    index_path.write_bytes(index_bytes)
    manifest_path.write_bytes(manifest_bytes)
    readme_path.write_bytes(readme_bytes)
    source_manifest_path.write_bytes((overlay_root / "flash_next_overlay.json").read_bytes())
    remote_code_payloads: list[tuple[str, bytes]] = []
    for repo_name, source_relative in _REMOTE_CODE_FILES.items():
        source_path = _PROJECT_ROOT / source_relative
        if not source_path.is_file():
            raise FileNotFoundError(f"missing remote loader source file: {source_path}")
        payload = source_path.read_bytes()
        (output_root / repo_name).write_bytes(payload)
        remote_code_payloads.append((repo_name, payload))
    _upload_metadata_if_changed(
        hub,
        [
            (_CONFIG_NAME, config_bytes),
            (_INDEX_NAME, index_bytes),
            (_FULL_MANIFEST_NAME, manifest_bytes),
            (_SOURCE_MANIFEST_NAME, source_manifest_path.read_bytes()),
            (_README_NAME, readme_bytes),
            *remote_code_payloads,
        ],
        output_root.parent,
    )

    final_info = hub.model_info(TARGET_REPO, files_metadata=True)
    if not final_info.private:
        raise ValueError("target repository unexpectedly became public")
    remote_files = {
        item.rfilename: item
        for item in hub.list_repo_tree(TARGET_REPO, repo_type="model", recursive=True, expand=True)
    }
    for shard, result in progress["shards"].items():
        remote = remote_files.get(shard)
        if remote is None or remote.size != result["file_size"] or not remote.lfs or remote.lfs.sha256 != result["sha256"]:
            raise ValueError(f"uploaded shard verification failed: {shard}")
    required = set(plan["shards"]) | {
        _CONFIG_NAME, _INDEX_NAME, _README_NAME, _FULL_MANIFEST_NAME,
        _SOURCE_MANIFEST_NAME, "UPSTREAM_README.md", *_REMOTE_CODE_FILES.keys(),
    }
    missing = required - set(remote_files)
    if missing:
        raise ValueError(f"final private repo is missing files: {sorted(missing)}")
    result = {
        **summary,
        "launch": True,
        "repo_sha": final_info.sha,
        "private": final_info.private,
        "files": len(remote_files),
        "integrated_tensor_bytes": json.loads(manifest_bytes)["integrated_tensor_bytes"],
        "verified_shards": len(progress["shards"]),
        "output_dir": str(output_root.resolve()),
    }
    (output_root / "export_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stream-integrate all fitted modules into a complete pinned base checkpoint")
    parser.add_argument("--overlay-dir", type=Path, default=DEFAULT_OVERLAY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--launch", action="store_true", help="create/update the private full-checkpoint repo")
    parser.add_argument("--resume", action="store_true", help="resume a prior private export from local progress")
    args = parser.parse_args(argv)
    if args.launch and args.resume:
        parser.error("choose either --launch or --resume")
    if args.resume and not (args.output_dir / "progress.json").is_file():
        parser.error("--resume requires the matching local progress.json")
    print(json.dumps(export_full_checkpoint(
        overlay_dir=args.overlay_dir,
        output_dir=args.output_dir,
        launch=args.launch or args.resume,
    ), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
