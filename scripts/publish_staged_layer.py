"""Publish one validated layer to an existing private Hub repo atomically.

Only a fitted checkpoint, its metrics, and calibration provenance metadata are
published. Teacher activations and SFT token sequences are never uploaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from huggingface_hub import CommitOperationAdd, HfApi
from safetensors import safe_open

from make_it_flash.model import flash_next_attention_schedule
from make_it_flash.overlay import _validate_fit_metrics
from make_it_flash.provenance import (
    checkpoint_provenance,
    sha256_file,
    validate_calibration_data_provenance,
    validate_model_revision,
)

OUTPUT_REPO = "RemydreScarlet/llm-jp-4.1-flash-next"


def artifact_paths(run_id: str, layer: int) -> tuple[str, str, str]:
    if not re.fullmatch(r"[A-Za-z0-9_-]{4,64}", run_id):
        raise ValueError("run_id must use 4-64 ASCII letters, digits, underscore, or hyphen")
    if type(layer) is not int or not 0 <= layer < 32:
        raise ValueError("layer must be an integer from 0 to 31")
    _, qsa = flash_next_attention_schedule(32)
    kind = "qsa" if layer in qsa else "gdn"
    checkpoint = f"{kind}_layer_{layer:02d}.safetensors"
    metrics = f"fit_qsa_layer_{layer:02d}.json" if kind == "qsa" else f"fit_layer_{layer:02d}.json"
    return f"staged/{run_id}/layers/{kind}/{layer:02d}", checkpoint, metrics


def publish_staged_layer(
    *, fit_dir: str | Path, run_id: str, layer: int, expected_repo_sha: str,
    expected_source_commit: str, execution_commit: str | None = None,
    repo_id: str = OUTPUT_REPO, dry_run: bool = True,
    api: Any | None = None,
) -> dict[str, Any]:
    """Validate and upload one layer in a single CAS-guarded Hub commit."""
    prefix, checkpoint_name, metrics_name = artifact_paths(run_id, layer)
    if repo_id != OUTPUT_REPO:
        raise ValueError("staged publisher only writes to the approved RemydreScarlet repo")
    validate_model_revision(expected_repo_sha)
    validate_model_revision(expected_source_commit)
    execution_commit = execution_commit or expected_source_commit
    validate_model_revision(execution_commit)
    root = Path(fit_dir)
    checkpoint, metrics_file = root / checkpoint_name, root / metrics_name
    if not checkpoint.is_file() or not metrics_file.is_file():
        raise FileNotFoundError(f"missing fitted checkpoint and metrics for layer {layer}")
    metrics_bytes = metrics_file.read_bytes()
    metrics = json.loads(metrics_bytes)
    if not isinstance(metrics, dict):
        raise ValueError("fit metrics must be a JSON object")
    base_id = metrics.get("model_id")
    base_revision = metrics.get("model_revision")
    if not isinstance(base_id, str):
        raise ValueError("fit metrics lack the base model ID")
    validate_model_revision(base_revision)
    kind = "qsa" if "qsa" in checkpoint_name else "gdn"
    quality = _validate_fit_metrics(kind=kind, fit_dir=root, checkpoint_path=checkpoint,
                                    layer=layer, base_model_id=base_id,
                                    base_model_revision=base_revision)
    calibration = validate_calibration_data_provenance(metrics.get("calibration_data"))
    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    expected_module = ("make_it_flash.QSAQwen3MoeAttentionAdapter" if kind == "qsa"
                       else "transformers.models.qwen3_next.Qwen3NextGatedDeltaNet")
    for key, value in {
        "model_id": base_id, "model_revision": base_revision,
        "teacher_layer": str(layer), "module": expected_module,
        "calibration_data": json.dumps(calibration, sort_keys=True),
        **checkpoint_provenance(),
    }.items():
        if metadata.get(key) != value:
            raise ValueError(f"checkpoint metadata mismatch: {key}")
    checkpoint_hash = sha256_file(checkpoint)
    metrics_hash = hashlib.sha256(metrics_bytes).hexdigest()
    record = {
        "format": "make_it_flash.staged_layer.v1",
        "kind": kind, "layer": layer, "run_id": run_id,
        "model_id": base_id, "model_revision": base_revision,
        "calibration_data": calibration,
        "checkpoint_file": checkpoint_name, "checkpoint_sha256": checkpoint_hash,
        "metrics_file": metrics_name, "metrics_sha256": metrics_hash,
        "source_commit": expected_source_commit,
        "execution_commit": execution_commit,
        "runtime_provenance": checkpoint_provenance(),
        "quality": quality,
    }
    record_bytes = (json.dumps(record, sort_keys=True, indent=2) + "\n").encode()
    result = {"repo_id": repo_id, "prefix": prefix, "parent_commit": expected_repo_sha,
              "checkpoint_sha256": checkpoint_hash, "metrics_sha256": metrics_hash,
              "record_sha256": hashlib.sha256(record_bytes).hexdigest(), "dry_run": dry_run}
    if dry_run:
        return result
    hub = api if api is not None else HfApi()
    info = hub.model_info(repo_id, files_metadata=False)
    if not info.private or info.sha != expected_repo_sha:
        raise ValueError("private repo status or expected parent commit has changed")
    files = set(hub.list_repo_files(repo_id, repo_type="model"))
    destinations = (f"{prefix}/{checkpoint_name}", f"{prefix}/{metrics_name}", f"{prefix}/record.json")
    if any(path in files for path in destinations):
        raise FileExistsError("this staged layer path already exists; never overwrite fitted artifacts")
    commit = hub.create_commit(
        repo_id=repo_id, repo_type="model", parent_commit=expected_repo_sha,
        operations=[
            CommitOperationAdd(path_in_repo=destinations[0], path_or_fileobj=checkpoint),
            CommitOperationAdd(path_in_repo=destinations[1], path_or_fileobj=metrics_bytes),
            CommitOperationAdd(path_in_repo=destinations[2], path_or_fileobj=record_bytes),
        ],
        commit_message=f"Store unassembled {kind.upper()} fit for layer {layer:02d}",
    )
    result.update({"dry_run": False, "new_repo_sha": commit.oid})
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate and publish one staged layer")
    parser.add_argument("fit_dir", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--expected-repo-sha", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--execution-commit", help="actual source checkout commit; defaults to expected-source-commit")
    parser.add_argument("--launch", action="store_true", help="actually upload a private Hub commit")
    args = parser.parse_args(argv)
    print(json.dumps(publish_staged_layer(
        fit_dir=args.fit_dir, run_id=args.run_id, layer=args.layer,
        expected_repo_sha=args.expected_repo_sha,
        expected_source_commit=args.expected_source_commit,
        execution_commit=args.execution_commit,
        dry_run=not args.launch,
    ), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
