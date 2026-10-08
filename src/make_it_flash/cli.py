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

    assemble = commands.add_parser("assemble", help="package complete GDN/QSA layer fits as a base-model overlay")
    assemble.add_argument("--model", default=DEFAULT_MODEL)
    assemble.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    assemble.add_argument("--gdn-dir", type=Path, required=True)
    assemble.add_argument("--qsa-dir", type=Path, required=True)
    assemble.add_argument("--output-dir", type=Path, default=Path("artifacts/flash_next_overlay"))
    assemble.add_argument("--overwrite", action="store_true")

    evaluate = commands.add_parser("evaluate", help="compare base/hybrid token-level perplexity on held-out data")
    evaluate.add_argument("--model", default=DEFAULT_MODEL)
    evaluate.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    evaluate.add_argument("--data-file", type=Path, required=True)
    evaluate.add_argument("--overlay-dir", type=Path, required=True)
    evaluate.add_argument("--output-file", type=Path, default=Path("artifacts/evaluation/report.json"))
    evaluate.add_argument("--max-examples", type=int)
    evaluate.add_argument("--target-chunk-tokens", type=int, default=64)
    evaluate.add_argument("--min-free-gib", type=float, default=66.0)
    evaluate.add_argument(
        "--allow-no-qsa-pruning",
        action="store_true",
        help="allow short-context PPL that does not exercise QSA top-k pruning",
    )
    evaluate.add_argument("--overwrite", action="store_true")

    full_run = commands.add_parser("full-run", help="fit all 24 GDN + 8 QSA layers and assemble an overlay")
    full_run.add_argument("--data-file", type=Path, required=True)
    full_run.add_argument("--output-dir", type=Path, required=True)
    full_run.add_argument("--validation-fraction", type=float, default=0.1)
    full_run.add_argument("--epochs", type=int, default=3)
    full_run.add_argument("--max-steps", type=int, default=200)
    full_run.add_argument("--min-free-gib", type=float, default=66.0)
    full_run.add_argument("--resume", action="store_true")
    full_run.add_argument("--dry-run", action="store_true")
    full_run.add_argument("--allow-cpu", action="store_true", help="tiny fixture tests only")

    staged = commands.add_parser("staged-layer", help="capture and fit one of 32 scheduled layers for incremental publishing")
    staged.add_argument("--data-file", type=Path, required=True)
    staged.add_argument("--output-dir", type=Path, required=True)
    staged.add_argument("--layer", type=int, required=True)
    staged.add_argument("--validation-fraction", type=float, default=0.1)
    staged.add_argument("--epochs", type=int, default=3)
    staged.add_argument("--max-steps", type=int, default=200)
    staged.add_argument("--min-free-gib", type=float, default=66.0)
    staged.add_argument("--dry-run", action="store_true")
    staged.add_argument("--allow-cpu", action="store_true", help="tiny fixture tests only")
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
    elif args.command == "fit-qsa":
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
    elif args.command == "staged-layer":
        from .staged import run_staged_layer

        result = run_staged_layer(data_file=args.data_file, output_dir=args.output_dir, layer=args.layer,
                                  validation_fraction=args.validation_fraction, epochs=args.epochs,
                                  max_steps=args.max_steps, min_free_gib=args.min_free_gib,
                                  dry_run=args.dry_run, allow_cpu=args.allow_cpu)
    elif args.command == "full-run":
        from .full_run import run_full_conversion

        result = run_full_conversion(data_file=args.data_file, output_dir=args.output_dir,
                                     validation_fraction=args.validation_fraction, epochs=args.epochs,
                                     max_steps=args.max_steps, min_free_gib=args.min_free_gib,
                                     resume=args.resume, dry_run=args.dry_run, allow_cpu=args.allow_cpu)
    elif args.command == "evaluate":
        from .evaluation import evaluate_flash_next

        result = evaluate_flash_next(
            data_file=args.data_file,
            overlay_dir=args.overlay_dir,
            output_file=args.output_file,
            base_model_id=args.model,
            base_model_revision=args.model_revision,
            max_examples=args.max_examples,
            target_chunk_tokens=args.target_chunk_tokens,
            min_free_gib=args.min_free_gib,
            require_qsa_pruning=not args.allow_no_qsa_pruning,
            overwrite=args.overwrite,
        )
    else:
        from transformers import AutoConfig
        from .overlay import assemble_flash_next_overlay

        base_config = AutoConfig.from_pretrained(
            args.model, revision=args.model_revision, trust_remote_code=False
        )
        result = assemble_flash_next_overlay(
            base_config=base_config,
            base_model_id=args.model,
            base_model_revision=args.model_revision,
            gdn_fit_dir=args.gdn_dir,
            qsa_fit_dir=args.qsa_dir,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )
    _json_output(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
