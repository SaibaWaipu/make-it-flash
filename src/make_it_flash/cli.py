"""Command-line stages for the layer-local GDN conversion pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION

DEFAULT_MODEL = DEFAULT_MODEL_ID


def _json_output(value):
    print(json.dumps(value, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="make-it-flash", description="Prepare and locally align one GDN block.")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="stream and tokenize a balanced calibration subset")
    prepare.add_argument("--output-dir", type=Path, default=Path("artifacts/data"))
    prepare.add_argument("--model", default=DEFAULT_MODEL)
    prepare.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    prepare.add_argument("--dataset", default="llm-jp/llm-jp-4.1-thinking-sft-data")
    prepare.add_argument("--dataset-revision", default="main")
    prepare.add_argument("--split", default="reasoning_medium")
    prepare.add_argument("--max-tokens", type=int, default=100_000)
    prepare.add_argument("--max-seq-len", type=int, default=2048)
    prepare.add_argument("--seed", type=int, default=17)
    prepare.add_argument("--overwrite", action="store_true")

    cache = commands.add_parser("cache", help="capture teacher attention activations and optional QSA block masses")
    cache.add_argument("--data-file", type=Path, default=Path("artifacts/data/calibration.jsonl"))
    cache.add_argument("--output-dir", type=Path, default=Path("artifacts/cache"))
    cache.add_argument("--layers", default="0", help="comma-separated decoder layer indices")
    cache.add_argument(
        "--attention-layers",
        default="",
        help="subset of --layers whose dense teacher attention should be reduced to QSA block masses (e.g. 3)",
    )
    cache.add_argument("--max-examples", type=int)
    cache.add_argument("--min-free-gib", type=float, default=66.0)
    cache.add_argument("--overwrite", action="store_true")

    fit = commands.add_parser("fit", help="fit one Qwen3-Next GDN to cached teacher targets")
    fit.add_argument("--cache-dir", type=Path, default=Path("artifacts/cache"))
    fit.add_argument("--output-dir", type=Path, default=Path("artifacts/gdn"))
    fit.add_argument("--layer", type=int, default=0)
    fit.add_argument("--epochs", type=int, default=3)
    fit.add_argument("--max-steps", type=int, default=200)
    fit.add_argument("--learning-rate", type=float, default=1e-4)
    fit.add_argument("--validation-fraction", type=float, default=0.1)
    fit.add_argument("--seed", type=int, default=17)
    fit.add_argument("--allow-cpu", action="store_true", help="allow slow CPU fitting for tiny smoke tests only")
    fit.add_argument("--overwrite", action="store_true")

    fit_qsa = commands.add_parser("fit-qsa", help="fit one QSA block from cached teacher block masses")
    fit_qsa.add_argument("--cache-dir", type=Path, default=Path("artifacts/cache"))
    fit_qsa.add_argument("--output-dir", type=Path, default=Path("artifacts/qsa"))
    fit_qsa.add_argument("--layer", type=int, required=True)
    fit_qsa.add_argument("--epochs", type=int, default=3)
    fit_qsa.add_argument("--max-steps", type=int, default=200)
    fit_qsa.add_argument("--learning-rate", type=float, default=1e-4)
    fit_qsa.add_argument("--selector-loss-weight", type=float, default=0.1)
    fit_qsa.add_argument("--validation-fraction", type=float, default=0.1)
    fit_qsa.add_argument("--seed", type=int, default=17)
    fit_qsa.add_argument("--allow-cpu", action="store_true", help="allow slow CPU fitting for tiny smoke tests only")
    fit_qsa.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        from .data import prepare_calibration_data

        result = prepare_calibration_data(
            output_dir=args.output_dir,
            model_id=args.model,
            model_revision=args.model_revision,
            dataset_id=args.dataset,
            dataset_revision=args.dataset_revision,
            split=args.split,
            max_tokens=args.max_tokens,
            max_seq_len=args.max_seq_len,
            seed=args.seed,
            overwrite=args.overwrite,
        )
    elif args.command == "cache":
        from .cache import cache_teacher_outputs

        try:
            layers = [int(value.strip()) for value in args.layers.split(",") if value.strip()]
            attention_layers = [int(value.strip()) for value in args.attention_layers.split(",") if value.strip()]
        except ValueError as exc:
            raise SystemExit("--layers and --attention-layers must be comma-separated integers, e.g. 0,3,7") from exc
        result = cache_teacher_outputs(
            data_file=args.data_file,
            output_dir=args.output_dir,
            layers=layers,
            attention_layers=attention_layers,
            max_examples=args.max_examples,
            min_free_gib=args.min_free_gib,
            overwrite=args.overwrite,
        )
    elif args.command == "fit":
        from .fit import fit_one_layer

        result = fit_one_layer(
            cache_dir=args.cache_dir,
            output_dir=args.output_dir,
            layer=args.layer,
            epochs=args.epochs,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
            allow_cpu=args.allow_cpu,
            overwrite=args.overwrite,
        )
    else:
        from .fit_qsa import fit_qsa_layer

        result = fit_qsa_layer(
            cache_dir=args.cache_dir,
            output_dir=args.output_dir,
            layer=args.layer,
            epochs=args.epochs,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            selector_loss_weight=args.selector_loss_weight,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
            allow_cpu=args.allow_cpu,
            overwrite=args.overwrite,
        )
    _json_output(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
