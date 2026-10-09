import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from make_it_flash import staged
from make_it_flash.provenance import checkpoint_provenance, sha256_file
from test_full_run import make_data, REVISION
from test_overlay import _write_fit_artifacts, MODEL_ID, MODEL_REVISION
from scripts.publish_staged_layer import artifact_paths, publish_staged_layer


def test_staged_dry_run_does_not_capture_or_fit(tmp_path, monkeypatch):
    from transformers import AutoConfig

    source = make_data(tmp_path)
    config = SimpleNamespace(model_type="qwen3_moe", num_hidden_layers=32, _name_or_path="fixture/base", _commit_hash=REVISION)
    monkeypatch.setattr(AutoConfig, "from_pretrained", lambda *args, **kwargs: config)
    monkeypatch.setattr(staged, "cache_teacher_outputs", lambda **kwargs: pytest.fail("dry-run captured"))
    result = staged.run_staged_layer(data_file=source, output_dir=tmp_path / "out", layer=3, dry_run=True)
    assert result["kind"] == "qsa" and result["layer"] == 3
    assert not (tmp_path / "out").exists()


def test_staged_captures_and_fits_only_one_layer(tmp_path, monkeypatch):
    from transformers import AutoConfig

    source = make_data(tmp_path)
    config = SimpleNamespace(model_type="qwen3_moe", num_hidden_layers=32, _name_or_path="fixture/base", _commit_hash=REVISION)
    monkeypatch.setattr(AutoConfig, "from_pretrained", lambda *args, **kwargs: config)
    calls = []

    def capture(**kwargs):
        calls.append(("capture", kwargs))
        kwargs["output_dir"].mkdir(parents=True)
        (kwargs["output_dir"] / "cache_manifest.json").write_text("{}")

    def fit(**kwargs):
        calls.append(("fit", kwargs))
        kwargs["output_dir"].mkdir(parents=True)
        (kwargs["output_dir"] / "qsa_layer_03.safetensors").write_bytes(b"fixture")
        (kwargs["output_dir"] / "fit_qsa_layer_03.json").write_text("{}")

    monkeypatch.setattr(staged, "cache_teacher_outputs", capture)
    monkeypatch.setattr(staged, "fit_qsa_layer", fit)
    monkeypatch.setattr(staged, "_verify_cache", lambda *args: None)
    monkeypatch.setattr(staged, "_verify_fit", lambda *args: None)
    result = staged.run_staged_layer(data_file=source, output_dir=tmp_path / "out", layer=3, allow_cpu=True)
    assert result["status"] == "fitted"
    assert calls[0][1]["layers"] == (3,) and calls[0][1]["attention_layers"] == (3,)
    assert calls[1][1]["layer"] == 3 and len(calls) == 2


def test_staged_publisher_dry_run_validates_fit_without_hub_write(tmp_path):
    gdn, qsa = tmp_path / "gdn", tmp_path / "qsa"
    gdn.mkdir()
    qsa.mkdir()
    _write_fit_artifacts(gdn, qsa)
    result = publish_staged_layer(fit_dir=qsa, run_id="run-001", layer=3,
                                  expected_repo_sha="b" * 40, expected_source_commit="c" * 40)
    assert result["dry_run"] is True and result["prefix"] == "staged/run-001/layers/qsa/03"


def test_staged_publisher_uploads_one_atomic_cas_commit(tmp_path):
    gdn, qsa = tmp_path / "gdn", tmp_path / "qsa"
    gdn.mkdir()
    qsa.mkdir()
    _write_fit_artifacts(gdn, qsa)

    class FakeHub:
        def __init__(self):
            self.commits = []
        def model_info(self, repo_id, files_metadata=False):
            return SimpleNamespace(private=True, sha="b" * 40)
        def list_repo_files(self, repo_id, repo_type):
            return ["README.md"]
        def create_commit(self, **kwargs):
            self.commits.append(kwargs)
            return SimpleNamespace(oid="d" * 40)

    hub = FakeHub()
    result = publish_staged_layer(fit_dir=qsa, run_id="run-001", layer=3,
                                  expected_repo_sha="b" * 40, expected_source_commit="c" * 40,
                                  dry_run=False, api=hub)
    assert result["new_repo_sha"] == "d" * 40
    assert len(hub.commits) == 1
    assert hub.commits[0]["parent_commit"] == "b" * 40
    assert len(hub.commits[0]["operations"]) == 3
    assert all(op.path_in_repo.startswith("staged/run-001/layers/qsa/03/") for op in hub.commits[0]["operations"])
    record = json.loads(hub.commits[0]["operations"][2].path_or_fileobj.decode("utf-8"))
    assert record["source_commit"] == "c" * 40
    assert record["execution_commit"] == "c" * 40


def test_staged_publisher_rejects_existing_layer_path(tmp_path):
    gdn, qsa = tmp_path / "gdn", tmp_path / "qsa"
    gdn.mkdir()
    qsa.mkdir()
    _write_fit_artifacts(gdn, qsa)
    class FakeHub:
        def model_info(self, *args, **kwargs):
            return SimpleNamespace(private=True, sha="b" * 40)
        def list_repo_files(self, *args, **kwargs):
            return ["staged/run-001/layers/qsa/03/record.json"]
        def create_commit(self, **kwargs):
            pytest.fail("existing layer must not be overwritten")
    with pytest.raises(FileExistsError, match="already exists"):
        publish_staged_layer(fit_dir=qsa, run_id="run-001", layer=3,
                             expected_repo_sha="b" * 40, expected_source_commit="c" * 40,
                             dry_run=False, api=FakeHub())


def test_staged_paths_reject_path_traversal():
    with pytest.raises(ValueError, match="run_id"):
        artifact_paths("../not-private", 0)
