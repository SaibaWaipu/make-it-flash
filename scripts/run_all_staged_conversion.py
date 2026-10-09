"""Run all 32 fits in one process, publishing each verified layer immediately.

The helper is intended for one long-running HF Job. A completed layer is stored
in one atomic Hub commit before fitting the next layer; a timeout therefore
leaves already committed layer artifacts intact and can be resumed with the
same run ID and the current model-repo SHA.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _import_path in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(_import_path) not in sys.path:
        sys.path.insert(0, str(_import_path))

from huggingface_hub import HfApi, hf_hub_download

from make_it_flash.full_run import preflight_full_run, run_full_conversion
from make_it_flash.model import flash_next_attention_schedule
from make_it_flash.provenance import validate_calibration_data_provenance, validate_model_revision
from scripts.publish_staged_layer import OUTPUT_REPO, artifact_paths, publish_staged_layer


def _restore_published_layers(
    *, api: HfApi, repo_id: str, repo_sha: str, run_id: str, source_commit: str,
    output_dir: Path, expected: dict[str, Any],
) -> list[int]:
    files = set(api.list_repo_files(repo_id, repo_type="model"))
    expected_paths: set[str] = set()
    restored: list[int] = []
    _, qsa_layers = flash_next_attention_schedule(32)
    calibration = {
        "data_sha256": expected["data_sha256"],
        "manifest_sha256": expected["manifest_sha256"],
    }
    for layer in range(32):
        prefix, checkpoint_name, metrics_name = artifact_paths(run_id, layer)
        paths = (f"{prefix}/{checkpoint_name}", f"{prefix}/{metrics_name}", f"{prefix}/record.json")
        expected_paths.update(paths)
        present = [path in files for path in paths]
        if any(present) and not all(present):
            raise ValueError(f"remote layer {layer} has a partial atomic record")
        if not all(present):
            continue
        record_path = hf_hub_download(repo_id, filename=paths[2], repo_type="model", revision=repo_sha)
        record = json.loads(Path(record_path).read_text(encoding="utf-8"))
        expected_kind = "qsa" if layer in qsa_layers else "gdn"
        if (record.get("format") != "make_it_flash.staged_layer.v1"
                or record.get("run_id") != run_id or record.get("layer") != layer
                or record.get("kind") != expected_kind
                or record.get("checkpoint_file") != checkpoint_name
                or record.get("metrics_file") != metrics_name
                or record.get("source_commit") != source_commit):
            raise ValueError(f"remote layer {layer} run/source/artifact provenance mismatch")
        if record.get("model_id") != expected["model_id"] or record.get("model_revision") != expected["model_revision"]:
            raise ValueError(f"remote layer {layer} base model provenance mismatch")
        record_calibration = validate_calibration_data_provenance(record.get("calibration_data"))
        if any(record_calibration.get(key) != value for key, value in calibration.items()):
            raise ValueError(f"remote layer {layer} calibration provenance mismatch")
        fit_dir = output_dir / ("qsa" if layer in qsa_layers else "gdn")
        fit_dir.mkdir(parents=True, exist_ok=True)
        local_checkpoint, local_metrics = fit_dir / checkpoint_name, fit_dir / metrics_name
        if local_checkpoint.exists() or local_metrics.exists():
            raise FileExistsError(f"local fit path already exists while restoring layer {layer}")
        for remote_path, local_path, digest_key in (
            (paths[0], local_checkpoint, "checkpoint_sha256"),
            (paths[1], local_metrics, "metrics_sha256"),
        ):
            downloaded = hf_hub_download(repo_id, filename=remote_path, repo_type="model", revision=repo_sha)
            payload = Path(downloaded).read_bytes()
            if hashlib.sha256(payload).hexdigest() != record.get(digest_key):
                raise ValueError(f"remote layer {layer} file hash mismatch: {remote_path}")
            local_path.write_bytes(payload)
        # Re-run the same strict metrics, checkpoint metadata and quality checks
        # used before publication; do not trust remote records merely by path.
        checked = publish_staged_layer(
            fit_dir=fit_dir, run_id=run_id, layer=layer,
            expected_repo_sha=repo_sha, expected_source_commit=source_commit,
            repo_id=repo_id, dry_run=True,
        )
        if checked["checkpoint_sha256"] != record["checkpoint_sha256"]:
            raise ValueError(f"restored layer {layer} checkpoint does not match its commit record")
        restored.append(layer)
    run_prefix = f"staged/{run_id}/"
    unexpected = {path for path in files if path.startswith(run_prefix) and path not in expected_paths}
    if unexpected:
        raise ValueError(f"unexpected files under this run ID: {sorted(unexpected)[:3]}")
    return restored


def run_progressive_conversion(
    *, data_file: str | Path, output_dir: str | Path, run_id: str,
    expected_repo_sha: str, source_commit: str, epochs: int = 3, max_steps: int = 200,
    validation_fraction: float = 0.1, min_free_gib: float = 66.0,
    resume: bool = False, dry_run: bool = False, api: HfApi | None = None,
) -> dict[str, Any]:
    """Fit the full 32-layer schedule in one run and commit each layer in turn."""
    if type(resume) is not bool or type(dry_run) is not bool:
        raise ValueError("resume and dry_run must be booleans")
    if epochs < 1 or max_steps < 1 or min_free_gib < 0:
        raise ValueError("epochs/max_steps must be positive and min_free_gib nonnegative")
    # Validate run ID and layer path before any Hub access.
    artifact_paths(run_id, 0)
    validate_model_revision(expected_repo_sha)
    validate_model_revision(source_commit)
    expected_data = preflight_full_run(data_file, validation_fraction=validation_fraction)
    output = Path(output_dir)
    if Path(data_file).resolve().is_relative_to(output.resolve()):
        raise ValueError("output directory must not contain calibration data")
    if dry_run:
        plan = run_full_conversion(
            data_file=data_file, output_dir=output, validation_fraction=validation_fraction,
            epochs=epochs, max_steps=max_steps, min_free_gib=min_free_gib,
            dry_run=True, assemble=False,
        )
        plan.update({"repo_id": OUTPUT_REPO, "initial_repo_sha": expected_repo_sha,
                     "run_id": run_id, "resume": resume, "layers_to_publish": 32})
        return plan

    hub = api if api is not None else HfApi()
    info = hub.model_info(OUTPUT_REPO, files_metadata=False)
    if not info.private or info.sha != expected_repo_sha:
        raise ValueError("model repo must be private and match the pinned initial parent commit")
    restored: list[int] = []
    if resume:
        restored = _restore_published_layers(
            api=hub, repo_id=OUTPUT_REPO, repo_sha=expected_repo_sha,
            run_id=run_id, source_commit=source_commit, output_dir=output,
            expected=expected_data,
        )
    else:
        files = set(hub.list_repo_files(OUTPUT_REPO, repo_type="model"))
        if any(path.startswith(f"staged/{run_id}/") for path in files):
            raise FileExistsError("run ID already has remote layer records; use --resume with the current repo SHA")

    current_repo_sha = expected_repo_sha
    committed: list[dict[str, Any]] = []

    def publish_completed_layer(kind: str, layer: int, fit_dir: Path, _expected: dict[str, Any]) -> None:
        nonlocal current_repo_sha
        result = publish_staged_layer(
            fit_dir=fit_dir, run_id=run_id, layer=layer,
            expected_repo_sha=current_repo_sha, expected_source_commit=source_commit,
            repo_id=OUTPUT_REPO, dry_run=False, api=hub,
        )
        current_repo_sha = result["new_repo_sha"]
        committed.append({"kind": kind, "layer": layer, "repo_sha": current_repo_sha,
                          "record_sha256": result["record_sha256"]})
        print(json.dumps({"event": "layer_committed", **committed[-1]}, sort_keys=True), flush=True)

    report = run_full_conversion(
        data_file=data_file, output_dir=output, validation_fraction=validation_fraction,
        epochs=epochs, max_steps=max_steps, min_free_gib=min_free_gib,
        resume=resume, dry_run=False, assemble=False,
        on_layer_complete=publish_completed_layer,
    )
    report.update({"initial_repo_sha": expected_repo_sha, "final_repo_sha": current_repo_sha,
                   "restored_layers": restored, "newly_committed_layers": committed,
                   "completed_layers": len(restored) + len(committed), "assembled": False})
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit 32 layers in one run and commit each as soon as it passes validation")
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--expected-repo-sha", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--min-free-gib", type=float, default=66.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    report = run_progressive_conversion(**vars(args))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
