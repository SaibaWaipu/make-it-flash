"""Fixed, three-source SFT calibration mix for staged LLM-jp conversion.

Selections are pinned to the LLM-jp 4.1 SFT dataset revision and model
revision. Only the approved English, Japanese and coding configs are read.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .data import (
    DATASET_ID,
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    MixSource,
    _stable_seed,
    allocate_quotas,
    tokenize_messages,
)
from .full_run import _split_value, preflight_full_run
from .provenance import validate_model_revision


SOURCES = (
    MixSource("japanese", 0.4, ("llmjp_extraction_wiki_ja_v0.3",)),
    MixSource("english", 0.3, ("daring_anteater",)),
    MixSource("code", 0.3, ("synthetic_jp_en_coding",)),
)
DATASET_REVISION = "cb4210190af6a0000fd91cb5bb8361ad6d6ade01"
SPLIT = "reasoning_medium"


def prepare_cherry_picked_calibration(
    *, output_dir: str | Path, max_tokens: int = 100_000, max_seq_len: int = 4096,
    seed: int = 17, model_id: str = DEFAULT_MODEL_ID,
    model_revision: str = DEFAULT_MODEL_REVISION, dataset_revision: str = DATASET_REVISION,
    validation_fraction: float = 0.1, max_candidates_per_config: int = 1200,
) -> dict[str, Any]:
    """Select, tokenize and validate one fixed three-source calibration corpus.

    Failures leave no published JSONL/manifest, so low-cost local retries with
    a larger candidate pool do not contaminate a future immutable Hub upload.
    """
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer
    from .worker import fetch_one_source

    if max_tokens < len(SOURCES) or max_seq_len < 2052 or max_candidates_per_config < 1:
        raise ValueError("max_tokens must cover all sources, max_seq_len >=2052, and candidate limit positive")
    validate_model_revision(model_revision)
    validate_model_revision(dataset_revision)
    if dataset_revision != DATASET_REVISION or model_id != DEFAULT_MODEL_ID:
        raise ValueError("cherry-pick uses the reviewed fixed 4.1 dataset and base model")
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite an existing calibration directory: {output}")
    api = HfApi()
    if api.model_info(model_id, revision=model_revision).sha != model_revision:
        raise ValueError("pinned model revision does not exist")
    if api.dataset_info(DATASET_ID, revision=dataset_revision).sha != dataset_revision:
        raise ValueError("pinned dataset revision does not exist")
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=model_revision, trust_remote_code=False)
    quotas = allocate_quotas(max_tokens, SOURCES)
    candidate_limit = max_candidates_per_config
    # Work in a private temporary directory, moving only validated text to output.
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".cherry-pick-", dir=output.parent) as scratch:
        scratch_path = Path(scratch)
        selected: list[dict[str, Any]] = []
        chosen_hashes: set[tuple[int, ...]] = set()
        selected_ids: set[tuple[str, str]] = set()
        skipped = {}
        source_summary = {}
        for source in SOURCES:
            config = source.configs[0]
            raw_file, summary_file = scratch_path / f"{config}.jsonl", scratch_path / f"{config}.summary.json"
            summary = fetch_one_source(dataset_id=DATASET_ID, dataset_revision=dataset_revision,
                                       split=SPLIT, config_name=config, seed=_stable_seed(seed, config),
                                       candidate_char_budget=max(20_000_000, quotas[source.category] * 24),
                                       max_examples=candidate_limit, output_file=str(raw_file),
                                       summary_file=str(summary_file))
            source_summary[config] = summary
            remaining = quotas[source.category]
            skip_count = 0
            candidates: list[dict[str, Any]] = []
            candidate_ids: set[str] = set()
            with raw_file.open(encoding="utf-8") as handle:
                for line in handle:
                    raw = json.loads(line)
                    try:
                        ids = tokenize_messages(tokenizer, raw.get("messages"))
                    except (ValueError, TypeError):
                        skip_count += 1
                        continue
                    source_sample_id = str(raw.get("sample_id") or "")
                    if not source_sample_id or source_sample_id in candidate_ids:
                        skip_count += 1
                        continue
                    candidate_ids.add(source_sample_id)
                    tokens = ids[:max_seq_len]
                    if len(tokens) < 2:
                        skip_count += 1
                        continue
                    signature = tuple(tokens)
                    if signature in chosen_hashes:
                        skip_count += 1
                        continue
                    group = ("validation" if _split_value(config, source_sample_id)
                             < int(validation_fraction * (2**64 - 1)) else "train")
                    candidates.append({"config": config, "sample_id": source_sample_id,
                                       "category": source.category, "input_ids": tokens,
                                       "split": group, "signature": signature})
            # Reserve at least one genuinely long sequence from each stable split
            # and each category; short-only Japanese calibration undertrains the
            # long-context selector even when other categories satisfy its gate.
            reserved = []
            for group in ("validation", "train"):
                candidate = next((item for item in candidates
                                  if item["split"] == group and len(item["input_ids"]) >= 2052), None)
                if candidate is None:
                    raise ValueError(f"{config} has no >=2052-token {group} example in the candidate pool")
                reserved.append(candidate)
            ordered = reserved + [item for item in candidates if item not in reserved]
            used_ids: set[str] = set()
            for candidate in ordered:
                if remaining <= 0:
                    break
                if candidate["sample_id"] in used_ids:
                    continue
                tokens = candidate["input_ids"]
                take = min(len(tokens), remaining)
                if take < 2:
                    continue
                tokens = tokens[:take]
                signature = tuple(tokens)
                if signature in chosen_hashes:
                    skip_count += 1
                    continue
                chosen_hashes.add(signature)
                used_ids.add(candidate["sample_id"])
                selected_ids.add((config, candidate["sample_id"]))
                selected.append({"category": source.category, "config": config,
                                 "sample_id": candidate["sample_id"], "chunk": 0, "input_ids": tokens})
                remaining -= len(tokens)
            skipped[config] = skip_count
            if remaining:
                raise ValueError(f"{config} is short by {remaining} tokens; increase candidate budget or retry")
        stage = scratch_path / "publish"
        stage.mkdir()
        calibration = stage / "calibration.jsonl"
        calibration.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected), encoding="utf-8")
        actual = {entry.category: sum(len(row["input_ids"]) for row in selected if row["category"] == entry.category)
                  for entry in SOURCES}
        manifest = {
            "model_id": model_id, "model_revision": model_revision, "dataset_id": DATASET_ID,
            "dataset_revision": dataset_revision, "split": SPLIT, "seed": seed,
            "target_tokens": max_tokens, "max_seq_len": max_seq_len,
            "max_candidates_per_config": candidate_limit,
            "long_context_threshold": 2052,
            "selection_strategy": (
                "reserve one >=threshold example from each category in each stable train/validation split; "
                "then fill each category quota from the deterministically shuffled candidate stream"
            ),
            "target_mix": {entry.category: entry.weight for entry in SOURCES},
            "sources": {entry.category: {"config": entry.configs[0], "license": {
                "japanese": "Apache-2.0", "english": "CC BY 4.0", "code": "Apache-2.0"}[entry.category]}
                for entry in SOURCES},
            "actual_tokens_by_category": actual, "actual_tokens": sum(actual.values()),
            "long_context_sequences_by_category": {
                entry.category: sum(1 for row in selected if row["category"] == entry.category
                                    and len(row["input_ids"]) >= 2052)
                for entry in SOURCES
            },
            "skipped_rows_by_config": skipped, "candidate_rows_by_config": {
                key: value["candidate_rows"] for key, value in source_summary.items()},
            "data_file": calibration.name,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        (stage / "data_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        split = preflight_full_run(calibration, validation_fraction=validation_fraction)
        manifest["split_preflight"] = split["counts"]
        (stage / "data_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        stage.rename(output)
        return {"manifest": manifest, "preflight": split,
                "data_file": str(output / "calibration.jsonl")}
