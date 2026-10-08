"""Independent-tokenized-corpus perplexity checks for a base/overlay pair.

This module only evaluates already-tokenized data. It does not prepare datasets,
launch remote jobs, fine-tune weights, or modify the base tokenizer/model assets.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .overlay import OVERLAY_FILENAME, OVERLAY_KIND, OVERLAY_VERSION, load_flash_next_overlay
from .provenance import (
    sha256_file,
    validate_base_config_provenance,
    validate_calibration_data_provenance,
    validate_model_revision,
)


def _require_eval_gpu_memory(min_free_gib: float) -> None:
    if not math.isfinite(min_free_gib) or min_free_gib <= 0:
        raise ValueError("min_free_gib must be a positive finite value")
    if not torch.cuda.is_available():
        raise RuntimeError("full 32B perplexity evaluation requires CUDA; no model weights were loaded")
    free_bytes = sum(torch.cuda.mem_get_info(index)[0] for index in range(torch.cuda.device_count()))
    free_gib = free_bytes / (1024**3)
    if free_gib < min_free_gib:
        raise RuntimeError(
            f"only {free_gib:.1f} GiB CUDA memory is free; evaluation requires {min_free_gib:.1f} GiB; "
            "no model weights were loaded"
        )


def _evaluation_data_provenance(
    data_file: str | Path,
    *,
    model_id: str,
    model_revision: str,
    calibration_data: dict[str, object],
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_path = Path(data_file)
    manifest_path = source_path.with_name("data_manifest.json")
    if not source_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("evaluation JSONL and adjacent data_manifest.json are required")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("evaluation data manifest must be a JSON object")
    if manifest.get("data_file") not in (None, source_path.name):
        raise ValueError("evaluation data_manifest.json data_file does not match the requested JSONL")
    if manifest.get("model_id") != model_id or manifest.get("model_revision") != model_revision:
        raise ValueError("evaluation data was tokenized for a different base model/revision")
    validate_model_revision(model_revision)
    training_data = validate_calibration_data_provenance(calibration_data)
    provenance = {
        "data_sha256": sha256_file(source_path),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "dataset_id": manifest.get("dataset_id"),
        "dataset_revision": manifest.get("dataset_revision"),
        "split": manifest.get("split"),
        "seed": manifest.get("seed"),
        "actual_tokens": manifest.get("actual_tokens"),
    }
    if (
        provenance["data_sha256"] == training_data["data_sha256"]
        or provenance["manifest_sha256"] == training_data["manifest_sha256"]
    ):
        raise ValueError("evaluation corpus is identical to the calibration corpus; use held-out data")
    training_dataset = (training_data.get("dataset_id"), training_data.get("dataset_revision"), training_data.get("split"))
    evaluation_dataset = (provenance["dataset_id"], provenance["dataset_revision"], provenance["split"])
    if evaluation_dataset[0] is not None and evaluation_dataset == training_dataset:
        raise ValueError("evaluation dataset/split matches calibration data; use an independent dataset or split")
    # A different manifest/split name alone cannot prove sample-level independence.
    # The full-run overlay records a digest set, so evaluation need not retain SFT
    # training text or depend on an ephemeral job filesystem after upload.
    digests = training_data.get("token_sha256")
    if digests is not None:
        if not isinstance(digests, list) or any(not isinstance(value, str) or len(value) != 64 for value in digests):
            raise ValueError("invalid calibration token digest set")
        with source_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                tokens = row.get("input_ids") if isinstance(row, dict) else None
                if not isinstance(tokens, list) or not tokens:
                    raise ValueError(f"evaluation corpus lacks input_ids: {source_path}")
                digest = hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()
                if digest in digests:
                    raise ValueError("evaluation and calibration corpora share token sequences; use genuinely held-out data")
    return manifest, provenance


def _preflight_evaluation_sequences(
    data_file: str | Path,
    *,
    vocab_size: int,
    context_limit: int | None,
    pruning_threshold: int,
    max_examples: int | None = None,
    require_pruning: bool = True,
) -> dict[str, int]:
    if vocab_size <= 1 or pruning_threshold <= 1:
        raise ValueError("vocab_size and pruning_threshold must be greater than one")
    examples = 0
    pruning_examples = 0
    max_sequence_length = 0
    source_path = Path(data_file)
    with source_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid evaluation JSON at {source_path}:{line_number}") from exc
            token_ids = row.get("input_ids") if isinstance(row, dict) else None
            if not isinstance(token_ids, list) or any(
                not isinstance(token, int) or isinstance(token, bool) for token in token_ids
            ):
                raise ValueError(f"evaluation row {line_number} must contain integer input_ids")
            if len(token_ids) < 2:
                continue
            if context_limit and len(token_ids) > context_limit:
                raise ValueError(
                    f"evaluation row {line_number} has {len(token_ids)} tokens, exceeding context limit {context_limit}"
                )
            if min(token_ids) < 0 or max(token_ids) >= vocab_size:
                raise ValueError(f"evaluation row {line_number} contains token IDs outside vocabulary size {vocab_size}")
            examples += 1
            max_sequence_length = max(max_sequence_length, len(token_ids))
            if len(token_ids) >= pruning_threshold:
                pruning_examples += 1
            if max_examples is not None and examples >= max_examples:
                break
    if examples == 0:
        raise ValueError("evaluation corpus contains no sequences with at least two tokens")
    if require_pruning and pruning_examples == 0:
        raise ValueError(
            f"held-out evaluation needs at least one sequence of {pruning_threshold} tokens to exercise QSA pruning; "
            f"found none among {examples} scored examples (max length {max_sequence_length})"
        )
    return {
        "scored_examples": examples,
        "qsa_pruning_examples": pruning_examples,
        "required_sequence_length": pruning_threshold,
        "max_sequence_length": max_sequence_length,
    }


def _score_tokenized_jsonl(
    model: Any,
    data_file: str | Path,
    *,
    vocab_size: int,
    context_limit: int,
    max_examples: int | None = None,
    target_chunk_tokens: int = 64,
) -> dict[str, Any]:
    if vocab_size <= 1 or context_limit <= 1:
        raise ValueError("vocab_size and context_limit must be greater than one")
    if max_examples is not None and max_examples < 1:
        raise ValueError("max_examples must be positive when provided")
    if target_chunk_tokens < 1:
        raise ValueError("target_chunk_tokens must be positive")
    embedding = model.get_input_embeddings()
    input_device = embedding.weight.device
    model.eval()

    total_nll = 0.0
    total_tokens = 0
    examples = 0
    skipped_short = 0
    categories: dict[str, dict[str, float | int]] = {}
    source_path = Path(data_file)
    with source_path.open(encoding="utf-8") as source, torch.inference_mode():
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid evaluation JSON at {source_path}:{line_number}") from exc
            token_ids = row.get("input_ids") if isinstance(row, dict) else None
            if not isinstance(token_ids, list) or any(
                not isinstance(token, int) or isinstance(token, bool) for token in token_ids
            ):
                raise ValueError(f"evaluation row {line_number} must contain integer input_ids")
            if len(token_ids) < 2:
                skipped_short += 1
                continue
            if len(token_ids) > context_limit:
                raise ValueError(
                    f"evaluation row {line_number} has {len(token_ids)} tokens, exceeding context limit {context_limit}"
                )
            if min(token_ids) < 0 or max(token_ids) >= vocab_size:
                raise ValueError(f"evaluation row {line_number} contains token IDs outside vocabulary size {vocab_size}")

            input_ids = torch.tensor([token_ids], dtype=torch.long, device=input_device)
            output = model(input_ids=input_ids, use_cache=False)
            logits = output.logits
            if logits.ndim != 3 or logits.shape[:2] != input_ids.shape or logits.shape[-1] != vocab_size:
                raise ValueError(f"model returned unexpected logits shape {tuple(logits.shape)}")
            sample_nll = 0.0
            target_count = len(token_ids) - 1
            for start in range(0, target_count, target_chunk_tokens):
                end = min(start + target_chunk_tokens, target_count)
                target_logits = logits[:, start:end, :].float().reshape(-1, vocab_size)
                labels = input_ids[:, start + 1 : end + 1].to(logits.device).reshape(-1)
                sample_nll += float(F.cross_entropy(target_logits, labels, reduction="sum").item())
            total_nll += sample_nll
            total_tokens += target_count
            examples += 1
            category = str(row.get("category", "unknown"))
            bucket = categories.setdefault(category, {"examples": 0, "target_tokens": 0, "nll_sum": 0.0})
            bucket["examples"] += 1
            bucket["target_tokens"] += target_count
            bucket["nll_sum"] += sample_nll
            del output, logits, input_ids
            if max_examples is not None and examples >= max_examples:
                break

    if total_tokens == 0:
        raise ValueError("evaluation corpus contains no sequences with at least two tokens")

    def _summary(nll_sum: float, count: int) -> dict[str, float | int]:
        mean_nll = nll_sum / count
        if not math.isfinite(mean_nll) or mean_nll > math.log(float.fromhex("0x1.fffffffffffffp+1023")):
            raise FloatingPointError("evaluation produced a non-finite perplexity")
        return {"target_tokens": count, "nll_sum": nll_sum, "mean_nll": mean_nll, "perplexity": math.exp(mean_nll)}

    return {
        "examples": examples,
        "skipped_short_examples": skipped_short,
        **_summary(total_nll, total_tokens),
        "by_category": {
            name: {"examples": values["examples"], **_summary(float(values["nll_sum"]), int(values["target_tokens"]))}
            for name, values in sorted(categories.items())
        },
    }


def _japanese_tokenizer_probe(tokenizer: Any, vocab_size: int) -> dict[str, Any]:
    """Verify and record Japanese text and chat-template tokenization."""
    text = "日本語の能力・tokenizer・MoE資産を維持します。"

    def _normalize_ids(value: Any, label: str) -> list[int]:
        if isinstance(value, dict):
            value = value.get("input_ids")
        if isinstance(value, torch.Tensor):
            value = value.tolist()
        if isinstance(value, tuple):
            value = list(value)
        if isinstance(value, list) and value and isinstance(value[0], list):
            if len(value) != 1:
                raise ValueError(f"{label} unexpectedly returned a batch")
            value = value[0]
        if not isinstance(value, list) or not value or any(
            not isinstance(token, int) or isinstance(token, bool) for token in value
        ):
            raise ValueError(f"{label} must return non-empty integer token IDs")
        if min(value) < 0 or max(value) >= vocab_size:
            raise ValueError(f"{label} returned token IDs outside vocabulary size {vocab_size}")
        return value

    try:
        text_ids = _normalize_ids(tokenizer.encode(text, add_special_tokens=False), "Japanese encode probe")
        chat_ids = _normalize_ids(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True
            ),
            "Japanese chat-template probe",
        )
        decoded = tokenizer.decode(text_ids, skip_special_tokens=False)
    except Exception as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError("pinned tokenizer failed the Japanese text/chat-template probe") from exc
    if not isinstance(decoded, str):
        raise ValueError("tokenizer.decode must return text for the Japanese round-trip probe")
    return {
        "text": text,
        "token_ids": text_ids,
        "decoded_text": decoded,
        "roundtrip_exact": decoded == text,
        "chat_template_token_ids": chat_ids,
    }


def evaluate_flash_next(
    *,
    data_file: str | Path,
    overlay_dir: str | Path,
    output_file: str | Path,
    base_model_id: str,
    base_model_revision: str,
    max_examples: int | None = None,
    target_chunk_tokens: int = 64,
    min_free_gib: float = 66.0,
    require_qsa_pruning: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Compare next-token perplexity for a pinned base and its complete overlay.

    The held-out JSONL must have been tokenized by the exact pinned base tokenizer,
    and its data/content provenance must differ from the overlay's calibration data.
    Models are loaded and scored sequentially to limit peak memory.
    """
    validate_model_revision(base_model_revision)
    if max_examples is not None and max_examples < 1:
        raise ValueError("max_examples must be positive when provided")
    if target_chunk_tokens < 1:
        raise ValueError("target_chunk_tokens must be positive")
    if not math.isfinite(min_free_gib) or min_free_gib <= 0:
        raise ValueError("min_free_gib must be a positive finite value")
    output_path = Path(output_file)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"evaluation report already exists: {output_path}; pass overwrite=True")
    overlay_root = Path(overlay_dir)
    overlay_manifest_path = overlay_root / OVERLAY_FILENAME
    input_paths = {
        Path(data_file).resolve(),
        Path(data_file).with_name("data_manifest.json").resolve(),
        overlay_manifest_path.resolve(),
    }
    if output_path.resolve() in input_paths or output_path.resolve().is_relative_to(overlay_root.resolve()):
        raise ValueError("evaluation report must not overwrite evaluation inputs or files inside the overlay")
    if not overlay_manifest_path.is_file():
        raise FileNotFoundError(f"missing overlay manifest: {overlay_manifest_path}")
    overlay_manifest_bytes = overlay_manifest_path.read_bytes()
    overlay_manifest_sha256 = hashlib.sha256(overlay_manifest_bytes).hexdigest()
    overlay_manifest = json.loads(overlay_manifest_bytes.decode("utf-8"))
    if not isinstance(overlay_manifest, dict):
        raise ValueError("overlay manifest must be a JSON object")
    if overlay_manifest.get("format") != OVERLAY_KIND or overlay_manifest.get("format_version") != OVERLAY_VERSION:
        raise ValueError("unsupported Flash-Next overlay format")
    if (
        overlay_manifest.get("base_model_id") != base_model_id
        or overlay_manifest.get("base_model_revision") != base_model_revision
    ):
        raise ValueError("overlay base model ID/revision does not match evaluation arguments")
    calibration_data = validate_calibration_data_provenance(overlay_manifest.get("calibration_data"))
    _, evaluation_data = _evaluation_data_provenance(
        data_file,
        model_id=base_model_id,
        model_revision=base_model_revision,
        calibration_data=calibration_data,
    )
    if not isinstance(require_qsa_pruning, bool):
        raise ValueError("require_qsa_pruning must be a bool")
    qsa_config = overlay_manifest.get("qsa_config")
    base_config = overlay_manifest.get("base_config")
    if not isinstance(qsa_config, dict) or not isinstance(base_config, dict):
        raise ValueError("overlay lacks QSA/base config needed for evaluation preflight")
    compress_ratio = int(qsa_config.get("compress_ratio", 0))
    token_budget = int(qsa_config.get("token_budget", 0))
    if compress_ratio <= 0 or token_budget < compress_ratio:
        raise ValueError("overlay qsa_config has an invalid token_budget/compress_ratio")
    pruning_threshold = (token_budget // compress_ratio + 1) * compress_ratio
    context_limit = int(base_config.get("max_position_embeddings", 0)) or None
    pruning_preflight = _preflight_evaluation_sequences(
        data_file,
        vocab_size=int(base_config["vocab_size"]),
        context_limit=context_limit,
        pruning_threshold=pruning_threshold,
        max_examples=max_examples,
        require_pruning=require_qsa_pruning,
    )

    try:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("install make-it-flash with its runtime dependencies first") from exc

    config = AutoConfig.from_pretrained(base_model_id, revision=base_model_revision, trust_remote_code=False)
    validate_base_config_provenance(config, base_model_id, base_model_revision)
    if getattr(config, "model_type", None) != "qwen3_moe":
        raise ValueError("Japanese perplexity evaluator supports only the pinned qwen3_moe base model")
    vocab_size = int(config.vocab_size)
    context_limit = int(getattr(config, "max_position_embeddings", 0))
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id, revision=base_model_revision, trust_remote_code=False
    )
    tokenizer_commit = getattr(tokenizer, "_commit_hash", None)
    if tokenizer_commit is None:
        tokenizer_commit = getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
    if tokenizer_commit is not None and tokenizer_commit != base_model_revision:
        raise ValueError("loaded tokenizer commit does not match the pinned base revision")
    tokenizer_vocab_size = len(tokenizer)
    vocabulary = tokenizer.get_vocab()
    if tokenizer_vocab_size > vocab_size or (vocabulary and max(vocabulary.values()) >= vocab_size):
        raise ValueError("tokenizer has token IDs outside the base model vocabulary")
    special_token_ids = [int(value) for value in getattr(tokenizer, "all_special_ids", [])]
    if any(value < 0 or value >= vocab_size for value in special_token_ids):
        raise ValueError("tokenizer special tokens are outside the base model vocabulary")
    tokenizer_record = {
        "model_id": base_model_id,
        "model_revision": base_model_revision,
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_name_or_path": str(getattr(tokenizer, "name_or_path", base_model_id)),
        "tokenizer_vocab_size": tokenizer_vocab_size,
        "model_vocab_size": vocab_size,
        "commit_hash": tokenizer_commit or base_model_revision,
        "special_token_ids": special_token_ids,
    }
    tokenizer_probe = _japanese_tokenizer_probe(tokenizer, vocab_size)
    del tokenizer

    _require_eval_gpu_memory(min_free_gib)

    def _load_base_model():
        loaded = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            revision=base_model_revision,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
            trust_remote_code=False,
        )
        validate_base_config_provenance(loaded.config, base_model_id, base_model_revision)
        loaded.eval()
        return loaded

    model = _load_base_model()
    base_metrics = _score_tokenized_jsonl(
        model,
        data_file,
        vocab_size=vocab_size,
        context_limit=context_limit,
        max_examples=max_examples,
        target_chunk_tokens=target_chunk_tokens,
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = _load_base_model()
    load_flash_next_overlay(
        model,
        overlay_root,
        base_model_id=base_model_id,
        base_model_revision=base_model_revision,
    )
    hybrid_metrics = _score_tokenized_jsonl(
        model,
        data_file,
        vocab_size=vocab_size,
        context_limit=context_limit,
        max_examples=max_examples,
        target_chunk_tokens=target_chunk_tokens,
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    source_path = Path(data_file)
    if evaluation_data["data_sha256"] != sha256_file(source_path):
        raise RuntimeError("evaluation JSONL changed while scoring; refusing to publish results")
    manifest_path = source_path.with_name("data_manifest.json")
    if evaluation_data["manifest_sha256"] != sha256_file(manifest_path):
        raise RuntimeError("evaluation data manifest changed while scoring; refusing to publish results")
    if overlay_manifest_sha256 != sha256_file(overlay_manifest_path):
        raise RuntimeError("overlay manifest changed while scoring; refusing to publish results")
    report = {
        "base_model_id": base_model_id,
        "base_model_revision": base_model_revision,
        "overlay_manifest_sha256": overlay_manifest_sha256,
        "tokenizer": tokenizer_record,
        "japanese_tokenizer_probe": tokenizer_probe,
        "calibration_data": calibration_data,
        "evaluation_data": evaluation_data,
        "requires_qsa_pruning_evidence": require_qsa_pruning,
        "qsa_pruning_preflight": pruning_preflight,
        "max_examples": max_examples,
        "target_chunk_tokens": target_chunk_tokens,
        "base": base_metrics,
        "hybrid": hybrid_metrics,
        "relative_perplexity_change": hybrid_metrics["perplexity"] / base_metrics["perplexity"] - 1.0,
        "note": "evaluation only; this does not fine-tune, merge, or upload model weights",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
