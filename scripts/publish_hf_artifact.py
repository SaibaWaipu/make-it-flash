"""Upload only the private fitted GDN block and metrics from an HF Job."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: publish_hf_artifact.py FIT_OUTPUT_DIR")
    repo_id = os.environ.get("MIF_OUTPUT_REPO")
    if not repo_id:
        raise SystemExit("MIF_OUTPUT_REPO must be a private Hugging Face model repo id")
    folder = Path(sys.argv[1])
    if not folder.is_dir():
        raise SystemExit(f"fit output directory not found: {folder}")
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(folder),
        allow_patterns=["gdn_layer_*.safetensors", "fit_layer_*.json"],
        commit_message="Upload one-layer make_it_flash pilot artifact",
    )
    print(f"Uploaded GDN weights and metrics to private repo {repo_id}")


if __name__ == "__main__":
    main()
