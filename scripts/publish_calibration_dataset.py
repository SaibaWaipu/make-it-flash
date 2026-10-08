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
                        dry_run: bool = True, api: HfApi | None = None) -> dict:
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
    if {"calibration.jsonl", "data_manifest.json"} & paths:
        raise FileExistsError("calibration data already exists; do not overwrite immutable run data")
    commit = hub.create_commit(repo_id=repo_id, repo_type="dataset", parent_commit=expected_repo_sha,
                               operations=[
                                   CommitOperationAdd(path_in_repo="calibration.jsonl", path_or_fileobj=source),
                                   CommitOperationAdd(path_in_repo="data_manifest.json", path_or_fileobj=manifest),
                               ], commit_message="Store fixed private calibration corpus for staged fitting")
    result.update({"dry_run": False, "new_repo_sha": commit.oid})
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Private, explicitly authorized calibration dataset publication")
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--expected-repo-sha", required=True)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    print(json.dumps(publish_calibration(data_file=args.data_file, repo_id=args.repo_id,
                                         expected_repo_sha=args.expected_repo_sha,
                                         dry_run=not args.launch), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
