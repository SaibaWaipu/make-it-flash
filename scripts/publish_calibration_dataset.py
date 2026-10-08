"""Publish a pinned calibration JSONL to a separate private dataset repo.

This uploads tokenized SFT-derived data. Do not run it without checking the
source dataset redistribution rights and the target repo's privacy settings.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

from make_it_flash.full_run import preflight_full_run
from make_it_flash.provenance import sha256_file


def publish_calibration(*, data_file: str | Path, repo_id: str, expected_repo_sha: str,
                        dry_run: bool = True, replace_existing: bool = False,
                        api: HfApi | None = None) -> dict:
    if not re.fullmatch(r"RemydreScarlet/[A-Za-z0-9_.-]+", repo_id):
        raise ValueError("calibration data must go to an explicit RemydreScarlet dataset repo")
    if re.fullmatch(r"[0-9a-fA-F]{40}", expected_repo_sha) is None:
        raise ValueError("pin a 40-character dataset repo parent SHA")
    source = Path(data_file)
    manifest = source.with_name("data_manifest.json")
    expected = preflight_full_run(source)
    result = {"repo_id": repo_id, "data_sha256": expected["data_sha256"],
              "manifest_sha256": expected["manifest_sha256"],
              "parent_commit": expected_repo_sha, "dry_run": dry_run,
              "qsa_pruning_train": expected["counts"]["qsa_train"],
              "qsa_pruning_validation": expected["counts"]["qsa_validation"]}
    if dry_run:
        return result
    hub = api if api is not None else HfApi()
    info = hub.dataset_info(repo_id)
    if not info.private or info.sha != expected_repo_sha:
        raise ValueError("dataset repo is public or its parent commit changed")
    paths = set(hub.list_repo_files(repo_id, repo_type="dataset"))
    existing = {"calibration.jsonl", "data_manifest.json", "README.md"} & paths
    if existing and not replace_existing:
        raise FileExistsError("calibration data already exists; use a new repo revision and explicit --replace-existing")
    manifest_record = json.loads(manifest.read_text(encoding="utf-8"))
    source_url = ("https://huggingface.co/datasets/llm-jp/llm-jp-4.1-thinking-sft-data/tree/"
                  + str(manifest_record.get("dataset_revision", "")))
    card = ("---\n" "language:\n- ja\n- en\n" "license: other\n"
            "license_name: mixed-cc-by-4.0-and-apache-2.0\n" "---\n\n"
            "# Private LLM-jp 4.1 Flash-Next calibration subset\n\n"
            "Tokenized SFT-derived calibration data for local attention-block fitting, not an evaluation set. "
            "Private use only; do not redistribute without reviewing every upstream license.\n\n"
            f"Source: [{manifest_record.get('dataset_id', '')}]({source_url}) "
            f"at `{manifest_record.get('dataset_revision', '')}`, split `{manifest_record.get('split', '')}`.\n\n"
            "| Category | Source config | Upstream terms |\n|---|---|---|\n"
            "| English | [nvidia/Daring-Anteater](https://huggingface.co/datasets/nvidia/Daring-Anteater) (`daring_anteater`) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/): attribution required |\n"
            "| Japanese | [llm-jp-4.1 thinking SFT](https://huggingface.co/datasets/llm-jp/llm-jp-4.1-thinking-sft-data) (`llmjp_extraction_wiki_ja_v0.3`) | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0): retain copyright/license/notices |\n"
            "| Code | [llm-jp/Synthetic-JP-EN-Coding-Dataset](https://huggingface.co/datasets/llm-jp/Synthetic-JP-EN-Coding-Dataset) (`synthetic_jp_en_coding`) | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0): retain copyright/license/notices |\n\n"
            "Individual source licenses and any additional notices must be checked before wider reuse. "
            "The original SFT responses may contain model-generated inaccuracies. "
            "Use the matching pinned tokenizer/model revision in `data_manifest.json`.\n")
    commit = hub.create_commit(repo_id=repo_id, repo_type="dataset", parent_commit=expected_repo_sha,
                               operations=[
                                   CommitOperationAdd(path_in_repo="calibration.jsonl", path_or_fileobj=source),
                                   CommitOperationAdd(path_in_repo="data_manifest.json", path_or_fileobj=manifest),
                                   CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=card.encode("utf-8")),
                               ], commit_message="Store attributed private English/Japanese/code calibration corpus")
    result.update({"dry_run": False, "new_repo_sha": commit.oid})
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Private, explicitly authorized calibration dataset publication")
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--expected-repo-sha", required=True)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--replace-existing", action="store_true",
                        help="publish an explicitly reviewed new dataset version at the root paths")
    args = parser.parse_args()
    print(json.dumps(publish_calibration(data_file=args.data_file, repo_id=args.repo_id,
                                         expected_repo_sha=args.expected_repo_sha,
                                         dry_run=not args.launch, replace_existing=args.replace_existing), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
