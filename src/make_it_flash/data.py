"""Deterministic, streaming preparation of a small calibration token mix."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

DATASET_ID = "llm-jp/llm-jp-4.1-thinking-sft-data"


@dataclass(frozen=True)
class MixSource:
    category: str
    weight: float
    configs: tuple[str, ...]


DEFAULT_MIX = (
    MixSource("japanese", 0.35, ("jaster_v1.4.1", "llmjp_extraction_wiki_ja_v0.3")),
    MixSource("english", 0.25, ("daring_anteater", "flan")),
    MixSource("math", 0.15, ("logical_math_coding_wizard8x22b", "nemotron_post_v3_math", "nemotron3_sft_multilingual_v2_math_ja_stackoverflow")),
    MixSource("code", 0.15, ("synthetic_jp_en_coding",)),
    MixSource("tool_agent", 0.10, ("nemotron_agentic_mix_v0.1.1", "nemotron_agentic_mix_v0.1.1_ja")),
)


def allocate_quotas(total: int, mix: Iterable[MixSource] = DEFAULT_MIX) -> dict[str, int]:
    """Allocate an exact token budget using largest-remainder rounding."""
    sources = tuple(mix)
    if total <= 0:
        raise ValueError("token budget must be positive")
    weight_sum = sum(item.weight for item in sources)
    if not sources or not math.isclose(weight_sum, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("mix weights must sum to 1.0")
    exact = [total * item.weight for item in sources]
    quotas = [math.floor(value) for value in exact]
    remainder = total - sum(quotas)
    order = sorted(range(len(sources)), key=lambda i: exact[i] - quotas[i], reverse=True)
    for index in order[:remainder]:
        quotas[index] += 1
    return {item.category: quota for item, quota in zip(sources, quotas)}


def normalize_messages(value: Any) -> list[dict[str, Any]]:
    """Decode the dataset's JSON-encoded Harmony-like messages."""
    if isinstance(value, str):
        value = json.loads(value)
    if isinstance(value, dict):
        value = value.get("messages")
        if isinstance(value, str):
            value = json.loads(value)
    if not isinstance(value, list) or not value:
        raise ValueError("row has no non-empty messages list")

    normalized: list[dict[str, Any]] = []
    for message in value:
        if not isinstance(message, dict) or not isinstance(message.get("role"), str):
            raise ValueError("message must be an object with a role")
        item = dict(message)
        content = item.get("content", "")
        if content is None:
            content = ""
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict):
                    text = part.get("text")
                    if text is not None:
                        parts.append(str(text))
                else:
                    parts.append(str(part))
            content = "".join(parts)
        elif not isinstance(content, str):
            content = str(content)
        item["content"] = content
        normalized.append(item)
    return normalized


def tokenize_messages(tokenizer: Any, value: Any) -> list[int]:
    messages = normalize_messages(value)
    try:
        token_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False
        )
    except Exception as original_error:
        # LLM-jp's tokenizer template rejects orphaned Harmony tool-role messages
        # in this dataset. Preserve the observation text as a tagged user turn.
        if not any(message.get("role") == "tool" for message in messages):
            raise ValueError(
                "tokenizer chat template rejected a dataset conversation; check the base model tokenizer and dataset revision"
            ) from original_error
        compatible: list[dict[str, Any]] = []
        for message in messages:
            item = dict(message)
            if item.get("role") == "tool":
                tool_name = item.get("name") or item.get("tool_name") or "tool"
                item["role"] = "user"
                item["content"] = f"[tool observation: {tool_name}]\n{item.get('content', '')}"
                for field in ("name", "tool_name", "tool_call_id", "channel", "recipient"):
                    item.pop(field, None)
            compatible.append(item)
        try:
            token_ids = tokenizer.apply_chat_template(
                compatible, tokenize=True, add_generation_prompt=False
            )
        except Exception as fallback_error:
            raise ValueError(
                "tokenizer chat template rejected a tool trajectory after converting tool observations to tagged user turns"
            ) from fallback_error
    if isinstance(token_ids, Mapping):
        token_ids = token_ids.get("input_ids")
        if token_ids is None:
            raise ValueError("tokenizer returned a mapping without input_ids")
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if token_ids and isinstance(token_ids[0], list):
        if len(token_ids) != 1:
            raise ValueError("expected one tokenized conversation")
        token_ids = token_ids[0]
    result = [int(token) for token in token_ids]
    if not result:
        raise ValueError("tokenizer returned an empty conversation")
    return result


def _stable_seed(seed: int, config: str) -> int:
    digest = hashlib.sha256(config.encode("utf-8")).digest()
    return (seed + int.from_bytes(digest[:4], "big")) % (2**32)


def prepare_calibration_data(
    *,
    output_dir: str | Path,
    model_id: str = "llm-jp/llm-jp-4.1-32b-a3b-thinking",
    model_revision: str = "main",
    dataset_id: str = DATASET_ID,
    dataset_revision: str = "main",
    split: str = "reasoning_medium",
    max_tokens: int = 100_000,
    max_seq_len: int = 2048,
    seed: int = 17,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Stream a weighted sample and write tokenized JSONL plus provenance."""
    if max_seq_len < 2:
        raise ValueError("max_seq_len must be at least 2")
    if max_tokens < len(DEFAULT_MIX):
        raise ValueError(f"max_tokens must be at least {len(DEFAULT_MIX)}")

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("install make-it-flash with its runtime dependencies first") from exc

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "calibration.jsonl"
    manifest_path = out_dir / "data_manifest.json"
    if not overwrite and (jsonl_path.exists() or manifest_path.exists()):
        raise FileExistsError(f"output already exists under {out_dir}; pass --overwrite to replace it")

    api = HfApi()
    resolved_model_revision = api.model_info(model_id, revision=model_revision, files_metadata=False).sha
    resolved_dataset_revision = api.dataset_info(
        dataset_id, revision=dataset_revision, files_metadata=False
    ).sha
    quotas = allocate_quotas(max_tokens)
    actual_by_category = {item.category: 0 for item in DEFAULT_MIX}
    actual_by_config: dict[str, int] = {}
    sequences_by_category = {item.category: 0 for item in DEFAULT_MIX}
    skipped_by_config: dict[str, int] = {}
    candidate_rows_by_config: dict[str, int] = {}
    worker_shutdown_warnings: list[dict[str, Any]] = []
    temp_path = jsonl_path.with_suffix(".jsonl.tmp")
    if temp_path.exists():
        raise FileExistsError(f"temporary output already exists: {temp_path}")

    try:
        with tempfile.TemporaryDirectory(dir=out_dir, prefix=".prepare-") as staging_dir:
            staging = Path(staging_dir)
            source_jobs: list[dict[str, Any]] = []
            for category in DEFAULT_MIX:
                category_target = quotas[category.category]
                per_source = [category_target // len(category.configs)] * len(category.configs)
                for index in range(category_target % len(category.configs)):
                    per_source[index] += 1
                for source_index, (config_name, source_target) in enumerate(
                    zip(category.configs, per_source)
                ):
                    if source_target <= 0:
                        continue
                    raw_path = staging / f"{category.category}-{source_index}-raw.jsonl"
                    summary_path = staging / f"{category.category}-{source_index}.json"
                    spec_path = staging / f"{category.category}-{source_index}-spec.json"
                    is_tool = category.category == "tool_agent"
                    spec = {
                        "dataset_id": dataset_id,
                        "dataset_revision": resolved_dataset_revision,
                        "split": split,
                        "config_name": config_name,
                        "seed": _stable_seed(seed, config_name),
                        "candidate_char_budget": max(8192, source_target * (128 if is_tool else 8)),
                        "max_examples": 2048 if is_tool else 512,
                        "output_file": str(raw_path.resolve()),
                        "summary_file": str(summary_path.resolve()),
                    }
                    spec_path.write_text(json.dumps(spec), encoding="utf-8")
                    completed = subprocess.run(
                        [sys.executable, "-m", "make_it_flash.worker", str(spec_path.resolve())],
                        capture_output=True,
                        text=True,
                    )
                    if completed.returncode != 0:
                        if not raw_path.is_file() or not summary_path.is_file():
                            details = (completed.stderr or completed.stdout)[-2000:]
                            raise RuntimeError(
                                f"source worker failed for {config_name} (exit={completed.returncode}): {details}"
                            )
                        worker_shutdown_warnings.append({
                            "config": config_name,
                            "returncode": completed.returncode,
                            "reason": "worker wrote and closed its raw shard and summary before a native streaming-reader shutdown signal",
                        })
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    candidate_rows = int(summary["candidate_rows"])
                    with raw_path.open("r", encoding="utf-8") as raw_source:
                        written_rows = sum(1 for _ in raw_source)
                    if written_rows != candidate_rows:
                        raise RuntimeError(
                            f"raw candidate shard for {config_name} is incomplete: "
                            f"{written_rows} rows on disk, {candidate_rows} in worker summary"
                        )
                    candidate_rows_by_config[config_name] = candidate_rows
                    source_jobs.append({
                        "category": category.category,
                        "config": config_name,
                        "target_tokens": source_target,
                        "raw_path": raw_path,
                        "summary_path": summary_path,
                    })

            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                model_id, revision=resolved_model_revision, trust_remote_code=False
            )
            with temp_path.open("w", encoding="utf-8") as sink:
                for job in source_jobs:
                    category_name = job["category"]
                    config_name = job["config"]
                    remaining = int(job["target_tokens"])
                    skipped = 0
                    sequences = 0
                    with Path(job["raw_path"]).open("r", encoding="utf-8") as source:
                        for line_number, line in enumerate(source, 1):
                            if remaining <= 0:
                                break
                            try:
                                raw = json.loads(line)
                                ids = tokenize_messages(tokenizer, raw.get("messages"))
                            except (ValueError, TypeError, json.JSONDecodeError):
                                skipped += 1
                                continue
                            ids = ids[:remaining]
                            sample_id = str(raw.get("sample_id") or f"{config_name}:{line_number}")
                            for chunk_no, offset in enumerate(range(0, len(ids), max_seq_len)):
                                chunk = ids[offset : offset + max_seq_len]
                                if not chunk:
                                    continue
                                record = {
                                    "category": category_name,
                                    "config": config_name,
                                    "sample_id": sample_id,
                                    "chunk": chunk_no,
                                    "input_ids": chunk,
                                }
                                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                                actual_by_category[category_name] += len(chunk)
                                sequences_by_category[category_name] += 1
                                sequences += 1
                            remaining -= len(ids)
                    actual_by_config[config_name] = int(job["target_tokens"]) - remaining
                    skipped_by_config[config_name] = skipped
                    if sequences == 0:
                        print(f"warning: no valid tokenized samples found in {config_name}")
            del tokenizer
            temp_path.replace(jsonl_path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise

    manifest = {
        "model_id": model_id,
        "model_revision": resolved_model_revision,
        "dataset_id": dataset_id,
        "dataset_revision": resolved_dataset_revision,
        "split": split,
        "seed": seed,
        "target_tokens": max_tokens,
        "max_seq_len": max_seq_len,
        "target_mix": {item.category: item.weight for item in DEFAULT_MIX},
        "target_tokens_by_category": quotas,
        "actual_tokens_by_category": actual_by_category,
        "actual_tokens_by_config": actual_by_config,
        "sequences_by_category": sequences_by_category,
        "skipped_rows_by_config": skipped_by_config,
        "candidate_rows_by_config": candidate_rows_by_config,
        "worker_shutdown_warnings": worker_shutdown_warnings,
        "actual_tokens": sum(actual_by_category.values()),
        "data_file": jsonl_path.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest
