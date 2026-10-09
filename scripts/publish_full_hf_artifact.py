"""Publish a complete validated overlay to an existing private model repo."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

from make_it_flash.model import flash_next_attention_schedule
from make_it_flash.overlay import OVERLAY_FILENAME, OVERLAY_KIND, OVERLAY_VERSION
from make_it_flash.provenance import sha256_file


def publish(overlay_dir: Path, evaluation_file: Path | None = None) -> dict[str, str]:
    repo_id = os.environ.get("MIF_OUTPUT_REPO")
    expected_sha = os.environ.get("MIF_EXPECTED_OUTPUT_REPO_SHA")
    if repo_id != "RemydreScarlet/llm-jp-4.1-flash-next" or not expected_sha:
        raise ValueError("pin the approved private output repo and its expected SHA")
    api = HfApi()
    repo = api.model_info(repo_id)
    if not repo.private or repo.sha != expected_sha:
        raise ValueError("refusing upload: private status or repo commit changed")
    manifest = json.loads((overlay_dir / OVERLAY_FILENAME).read_text(encoding="utf-8"))
    gdn, qsa = flash_next_attention_schedule(32)
    if (manifest.get("format"), manifest.get("format_version")) != (OVERLAY_KIND, OVERLAY_VERSION):
        raise ValueError("invalid overlay format")
    if manifest.get("schedule") != {"gdn_layers": list(gdn), "qsa_layers": list(qsa)}:
        raise ValueError("overlay is missing scheduled layers")
    if manifest.get("base_config", {}).get("num_hidden_layers") != 32:
        raise ValueError("overlay does not target a 32-layer model")
    from make_it_flash.provenance import checkpoint_provenance

    if manifest.get("provenance") != checkpoint_provenance():
        raise ValueError("overlay runtime fingerprint changed")
    for kind, layers in (("gdn", gdn), ("qsa", qsa)):
        entries = manifest.get("modules", {}).get(kind, {})
        if set(entries) != {str(layer) for layer in layers}:
            raise ValueError(f"overlay is missing {kind} weights")
        for layer in layers:
            entry = entries[str(layer)]
            relative = Path(entry["file"])
            if relative.is_absolute() or ".." in relative.parts or relative.parts[:2] != ("modules", kind):
                raise ValueError("overlay module path must stay inside its kind directory")
            file = overlay_dir / relative
            if not file.is_file() or sha256_file(file) != entry.get("sha256"):
                raise ValueError(f"overlay module hash mismatch: {relative}")
    # Publish only adapter files. Base weights, raw SFT, and teacher activations stay local.
    prefix = f"runs/{sha256_file(overlay_dir / OVERLAY_FILENAME)[:16]}"
    api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=str(overlay_dir),
                      path_in_repo=prefix, allow_patterns=[OVERLAY_FILENAME, "modules/**/*.safetensors"],
                      commit_message="Upload complete 24-GDN/8-QSA adapter overlay")
    result = {"repo_id": repo_id, "overlay_path": prefix}
    if evaluation_file is not None and evaluation_file.is_file():
        api.upload_file(repo_id=repo_id, repo_type="model", path_or_fileobj=str(evaluation_file),
                        path_in_repo=f"{prefix}/evaluation.json", commit_message="Upload held-out evaluation metrics")
        result["evaluation_path"] = f"{prefix}/evaluation.json"
    return result


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) not in (1, 2):
        raise SystemExit("usage: publish_full_hf_artifact.py OVERLAY_DIR [EVALUATION_JSON]")
    print(json.dumps(publish(Path(args[0]), Path(args[1]) if len(args) == 2 else None), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
