from types import SimpleNamespace

from make_it_flash import full_run
from scripts import run_all_staged_conversion as progressive


REVISION = "a" * 40
PARENT = "b" * 40


class FakeHub:
    def model_info(self, repo_id, files_metadata=False):
        return SimpleNamespace(private=True, sha=PARENT)

    def list_repo_files(self, repo_id, repo_type="model"):
        return []


def test_one_progressive_invocation_commits_every_layer_in_order(tmp_path, monkeypatch):
    source = tmp_path / "calibration.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    (tmp_path / "data_manifest.json").write_text(
        '{"model_id":"fixture/base","model_revision":"' + REVISION + '"}', encoding="utf-8"
    )
    expected = {"data_sha256": "c" * 64, "manifest_sha256": "d" * 64,
                "model_id": "fixture/base", "model_revision": REVISION}
    monkeypatch.setattr(progressive, "preflight_full_run", lambda *args, **kwargs: expected)
    calls = []
    parents = []

    def fake_publish(**kwargs):
        parents.append(kwargs["expected_repo_sha"])
        calls.append((kwargs["run_id"], kwargs["layer"], kwargs["expected_source_commit"], kwargs["execution_commit"]))
        return {"new_repo_sha": f"{len(calls):040x}", "record_sha256": "e" * 64}

    monkeypatch.setattr(progressive, "publish_staged_layer", fake_publish)

    def fake_conversion(**kwargs):
        assert kwargs["assemble"] is False
        assert kwargs["resume"] is False
        for layer in range(32):
            kind = "qsa" if layer in (3, 7, 11, 15, 19, 23, 27, 31) else "gdn"
            kwargs["on_layer_complete"](kind, layer, tmp_path / kind, expected)
        return {"completed_layers": 32, "assembled": False}

    monkeypatch.setattr(progressive, "run_full_conversion", fake_conversion)
    result = progressive.run_progressive_conversion(
        data_file=source, output_dir=tmp_path / "out", run_id="pilot-all",
        expected_repo_sha=PARENT, source_commit=REVISION, api=FakeHub(),
    )
    assert [layer for _, layer, _, _ in calls] == list(range(32))
    assert len({run_id for run_id, _, _, _ in calls}) == 1
    assert all(source == REVISION and execution == REVISION for _, _, source, execution in calls)
    assert parents == [PARENT, *[f"{i:040x}" for i in range(1, 32)]]
    assert result["final_repo_sha"] == f"{32:040x}"
    assert result["completed_layers"] == 32
    assert result["assembled"] is False


def test_progressive_resume_restores_remote_layers_then_publishes_remaining(tmp_path, monkeypatch):
    source = tmp_path / "calibration.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    (tmp_path / "data_manifest.json").write_text(
        '{"model_id":"fixture/base","model_revision":"' + REVISION + '"}', encoding="utf-8"
    )
    expected = {"data_sha256": "c" * 64, "manifest_sha256": "d" * 64,
                "model_id": "fixture/base", "model_revision": REVISION}
    monkeypatch.setattr(progressive, "preflight_full_run", lambda *args, **kwargs: expected)
    monkeypatch.setattr(progressive, "_restore_published_layers", lambda **kwargs: [0, 1])
    commits = []

    execution = "f" * 40

    def fake_publish(**kwargs):
        commits.append((kwargs["layer"], kwargs["expected_source_commit"], kwargs["execution_commit"]))
        return {"new_repo_sha": f"{len(commits):040x}", "record_sha256": "f" * 64}

    monkeypatch.setattr(progressive, "publish_staged_layer", fake_publish)

    def fake_conversion(**kwargs):
        assert kwargs["resume"] is True
        for layer in range(2, 32):
            kind = "qsa" if layer in (3, 7, 11, 15, 19, 23, 27, 31) else "gdn"
            kwargs["on_layer_complete"](kind, layer, tmp_path / kind, expected)
        return {"completed_layers": 32, "assembled": False}

    monkeypatch.setattr(progressive, "run_full_conversion", fake_conversion)
    result = progressive.run_progressive_conversion(
        data_file=source, output_dir=tmp_path / "out", run_id="pilot-all",
        expected_repo_sha=PARENT, source_commit=REVISION, execution_commit=execution,
        resume=True, api=FakeHub(),
    )
    assert result["restored_layers"] == [0, 1]
    assert [layer for layer, _, _ in commits] == list(range(2, 32))
    assert all(source == REVISION and executed == execution for _, source, executed in commits)
    assert result["execution_commit"] == execution
    assert result["completed_layers"] == 32


def test_progressive_dry_run_does_not_check_or_write_hub(tmp_path, monkeypatch):
    source = tmp_path / "calibration.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    (tmp_path / "data_manifest.json").write_text(
        '{"model_id":"fixture/base","model_revision":"' + REVISION + '"}', encoding="utf-8"
    )
    monkeypatch.setattr(progressive, "preflight_full_run", lambda *args, **kwargs: {
        "data_sha256": "c" * 64, "manifest_sha256": "d" * 64,
        "model_id": "fixture/base", "model_revision": REVISION,
    })
    monkeypatch.setattr(progressive, "run_full_conversion", lambda **kwargs: {
        "dry_run": True, "estimated_teacher_loads": 8, "gdn_layers": list(range(32)), "qsa_layers": [3, 7, 11, 15, 19, 23, 27, 31]
    })
    result = progressive.run_progressive_conversion(
        data_file=source, output_dir=tmp_path / "out", run_id="pilot-all",
        expected_repo_sha=PARENT, source_commit=REVISION, dry_run=True,
    )
    assert result["layers_to_publish"] == 32
    assert result["estimated_teacher_loads"] == 8
