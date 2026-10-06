"""Isolated raw-row fetcher; tokenization stays out of open Arrow readers."""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path
from typing import Any


def fetch_one_source(
    *,
    dataset_id: str,
    dataset_revision: str,
    split: str,
    config_name: str,
    seed: int,
    candidate_char_budget: int,
    max_examples: int,
    output_file: str,
    summary_file: str,
) -> dict[str, Any]:
    from datasets import load_dataset

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(
        dataset_id,
        name=config_name,
        split=split,
        streaming=True,
        revision=dataset_revision,
    )
    stream = dataset.shuffle(seed=seed, buffer_size=32)
    iterator = iter(stream)
    rows = 0
    characters = 0
    try:
        with output_path.open("w", encoding="utf-8") as sink:
            for row in iterator:
                messages = row.get("messages")
                sample_id = str(row.get("ID") or f"{config_name}:{rows}")
                record = {"sample_id": sample_id, "messages": messages}
                serialized = json.dumps(record, ensure_ascii=False)
                sink.write(serialized + "\n")
                rows += 1
                characters += len(serialized)
                if rows >= max_examples or characters >= candidate_char_budget:
                    break
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            close()
        del iterator, stream, dataset
        gc.collect()

    summary = {"config": config_name, "candidate_rows": rows, "candidate_characters": characters}
    Path(summary_file).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        raise SystemExit("internal worker expects one JSON spec path")
    spec = json.loads(Path(args[0]).read_text(encoding="utf-8"))
    fetch_one_source(**spec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
