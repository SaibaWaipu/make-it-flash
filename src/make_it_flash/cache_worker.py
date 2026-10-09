"""Run one teacher capture in an isolated process to release CUDA at exit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .cache import cache_teacher_outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Capture teacher activations in an isolated process")
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--attention-layers", type=int, nargs="*", default=[])
    parser.add_argument("--min-free-gib", type=float, default=66.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    manifest = cache_teacher_outputs(
        data_file=args.data_file,
        output_dir=args.output_dir,
        layers=args.layers,
        attention_layers=args.attention_layers,
        min_free_gib=args.min_free_gib,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
