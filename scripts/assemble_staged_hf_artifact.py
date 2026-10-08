"""Download 32 immutable layer records and assemble a complete overlay.

This is read-only against Hub: publication of the assembled overlay is a
separate, explicitly authorized operation after quality/evaluation checks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from huggingface_hub import HfApi

from make_it_flash.model import flash_next_attention_schedule
from make_it_flash.overlay import assemble_flash_next_overlay
from make_it_flash.provenance import checkpoint_provenance, sha256_file

from publish_staged_layer import OUTPUT_REPO, artifact_paths


def assemble_staged_run(*, run_id: str, output_dir: str | Path, repo_id: str = OUTPUT_REPO,
                        revision: str | None = None, api: HfApi | None = None) -> dict:
    if repo_id != OUTPUT_REPO:
        raise ValueError("only the approved RemydreScarlet repo is supported")
    hub = api if api is not None else HfApi()
    info = hub.model_info(repo_id, files_metadata=False)
    if not info.private:
        raise ValueError("staged run must stay private")
    from make_it_flash.provenance import validate_model_revision

    pinned_revision = revision or info.sha
    validate_model_revision(pinned_revision)
    _, qsa = flash_next_attention_schedule(32)
    runtime = checkpoint_provenance()
    records = []
    with TemporaryDirectory(prefix="make-it-flash-assemble-") as scratch:
        root = Path(scratch)
        gdn_dir, qsa_dir = root / "gdn", root / "qsa"
        gdn_dir.mkdir()
        qsa_dir.mkdir()
        for layer in range(32):
            prefix, checkpoint_name, metrics_name = artifact_paths(run_id, layer)
            record_file = hub.hf_hub_download(repo_id, filename=f"{prefix}/record.json", repo_type="model", revision=pinned_revision)
            record = json.loads(Path(record_file).read_text(encoding="utf-8"))
            kind = "qsa" if layer in qsa else "gdn"
            if (record.get("format"), record.get("kind"), record.get("layer"), record.get("run_id")) != (
                "make_it_flash.staged_layer.v1", kind, layer, run_id
            ) or record.get("checkpoint_file") != checkpoint_name or record.get("metrics_file") != metrics_name:
                raise ValueError(f"inconsistent staged record for layer {layer}")
            if record.get("runtime_provenance") != runtime:
                raise ValueError("staged records require identical current runtime fingerprints")
            target = qsa_dir if kind == "qsa" else gdn_dir
            for name, hash_key in ((checkpoint_name, "checkpoint_sha256"), (metrics_name, "metrics_sha256")):
                downloaded = hub.hf_hub_download(repo_id, filename=f"{prefix}/{name}", repo_type="model", revision=pinned_revision)
                if sha256_file(downloaded) != record[hash_key]:
                    raise ValueError(f"staged layer {layer} {name} has changed")
                (target / name).write_bytes(Path(downloaded).read_bytes())
            records.append(record)
        first = records[0]
        for record in records[1:]:
            for key in ("model_id", "model_revision", "calibration_data", "runtime_provenance", "source_commit"):
                if record.get(key) != first.get(key):
                    raise ValueError(f"staged layers have inconsistent {key}")
        from transformers import AutoConfig
        from make_it_flash.provenance import validate_base_config_provenance

        base = AutoConfig.from_pretrained(first["model_id"], revision=first["model_revision"], trust_remote_code=False)
        validate_base_config_provenance(base, first["model_id"], first["model_revision"])
        manifest = assemble_flash_next_overlay(base_config=base, base_model_id=first["model_id"],
                                               base_model_revision=first["model_revision"],
                                               gdn_fit_dir=gdn_dir, qsa_fit_dir=qsa_dir,
                                               output_dir=output_dir)
    return {"repo_id": repo_id, "run_id": run_id, "pinned_repo_revision": pinned_revision,
            "overlay_dir": str(output_dir), "manifest": manifest}


def main() -> int:
    parser = argparse.ArgumentParser(description="Read staged layer records and assemble a verified 32-layer overlay")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--revision", help="pinned model repo commit SHA")
    args = parser.parse_args()
    print(json.dumps(assemble_staged_run(run_id=args.run_id, output_dir=args.output_dir,
                                         revision=args.revision), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
