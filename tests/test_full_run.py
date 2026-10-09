import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from make_it_flash import full_run


REVISION = "a" * 40


def make_data(tmp_path: Path, *, short_validation: bool = False) -> Path:
    source = tmp_path / "calibration.jsonl"
    rows = []
    for sample in range(20):
        sample_id = f"sample-{sample}"
        validation = full_run._split_value("fixture", sample_id) < int(0.1 * (2**64 - 1))
        count = 10 if (short_validation and validation) else 2052
        rows.append({"config": "fixture", "sample_id": sample_id, "input_ids": [sample + 1] * count})
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    (tmp_path / "data_manifest.json").write_text(
        json.dumps({"data_file": source.name, "model_id": "fixture/base", "model_revision": REVISION}), encoding="utf-8"
    )
    return source


def test_release_cuda_memory_clears_cached_allocator(monkeypatch):
    calls = []
    monkeypatch.setattr(full_run.gc, "collect", lambda: calls.append("gc"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: calls.append("sync"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty"))
    full_run._release_cuda_memory()
    assert calls == ["gc", "sync", "empty"]


def test_preflight_proves_actual_long_train_validation_split(tmp_path):
    source = make_data(tmp_path)
    result = full_run.preflight_full_run(source)
    assert result["counts"]["qsa_train"] > 0
    assert result["counts"]["qsa_validation"] > 0
    assert result["data_sha256"] == full_run.sha256_file(source)


def test_preflight_rejects_short_validation_before_teacher_load(tmp_path):
    source = make_data(tmp_path, short_validation=True)
    with pytest.raises(ValueError, match="QSA-pruned train/validation"):
        full_run.preflight_full_run(source)


def test_preflight_rejects_duplicates_across_split_ids(tmp_path):
    source = make_data(tmp_path)
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    first_train = next(row for row in rows if full_run._split_value("fixture", row["sample_id"]) >= int(0.1 * (2**64 - 1)))
    first_validation = next(row for row in rows if full_run._split_value("fixture", row["sample_id"]) < int(0.1 * (2**64 - 1)))
    first_validation["input_ids"] = first_train["input_ids"]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="cross train/validation"):
        full_run.preflight_full_run(source)


def test_full_run_dry_run_checks_schedule_without_loading_teacher(tmp_path, monkeypatch):
    from transformers import AutoConfig

    source = make_data(tmp_path)
    config = SimpleNamespace(num_hidden_layers=32, model_type="qwen3_moe", _name_or_path="fixture/base", _commit_hash=REVISION)
    monkeypatch.setattr(AutoConfig, "from_pretrained", lambda *args, **kwargs: config)
    monkeypatch.setattr(full_run, "cache_teacher_outputs", lambda **kwargs: pytest.fail("dry-run loaded teacher"))
    result = full_run.run_full_conversion(data_file=source, output_dir=tmp_path / "out", dry_run=True)
    assert result["estimated_teacher_loads"] == 8
    assert result["qsa_layers"] == [3, 7, 11, 15, 19, 23, 27, 31]
    assert len(result["gdn_layers"]) == 24
    assert not (tmp_path / "out").exists()


def test_full_run_uses_eight_teacher_captures_and_32_fits(tmp_path, monkeypatch):
    from transformers import AutoConfig

    source = make_data(tmp_path)
    config = SimpleNamespace(num_hidden_layers=32, model_type="qwen3_moe", _name_or_path="fixture/base", _commit_hash=REVISION)
    monkeypatch.setattr(AutoConfig, "from_pretrained", lambda *args, **kwargs: config)
    captures = []
    fits = []
    published = []
    assembled = []

    def capture(**kwargs):
        captures.append(kwargs)
        folder = kwargs["output_dir"]
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "cache_manifest.json").write_text("{}", encoding="utf-8")

    def fit(**kwargs):
        fits.append(kwargs)
        folder = kwargs["output_dir"]
        folder.mkdir(parents=True, exist_ok=True)
        layer = kwargs["layer"]
        kind = "gdn" if folder.name == "gdn" else "qsa"
        save_file({"weight": torch.zeros(2, 2)}, str(folder / f"{kind}_layer_{layer:02d}.safetensors"))
        (folder / (f"fit_layer_{layer:02d}.json" if kind == "gdn" else f"fit_qsa_layer_{layer:02d}.json")).write_text("{}")

    monkeypatch.setattr(full_run, "cache_teacher_outputs", capture)
    monkeypatch.setattr(full_run, "fit_one_layer", fit)
    monkeypatch.setattr(full_run, "fit_qsa_layer", fit)
    monkeypatch.setattr(full_run, "_verify_cache", lambda *args: None)
    monkeypatch.setattr(full_run, "_verify_fit", lambda *args: None)
    memory_releases = []
    monkeypatch.setattr(full_run, "_release_cuda_memory", lambda: memory_releases.append(True))
    def assemble(**kwargs):
        assembled.append(kwargs)
        kwargs["output_dir"].mkdir(parents=True)
        return {"assembled": True, "calibration_data": {}}

    monkeypatch.setattr(full_run, "assemble_flash_next_overlay", assemble)

    def after_layer(kind, layer, fit_dir, expected):
        published.append((kind, layer, fit_dir, expected["data_sha256"]))
        assert (fit_dir / f"{kind}_layer_{layer:02d}.safetensors").is_file()

    result = full_run.run_full_conversion(data_file=source, output_dir=tmp_path / "out", allow_cpu=True,
                                          on_layer_complete=after_layer)
    assert len(captures) == 8
    assert len(fits) == 32
    assert len(memory_releases) == len(captures) + len(fits)
    assert [call["layer"] for call in fits] == list(range(32))
    assert set(captures[0]["layers"]) == set(range(32)) - {7, 11, 15, 19, 23, 27, 31}
    assert all(len(call["layers"]) == 1 for call in captures[1:])
    assert [layer for _, layer, _, _ in published] == list(range(32))
    assert len(assembled) == 1 and result["overlay_manifest"]["assembled"] is True
    assert len(result["overlay_manifest"]["calibration_data"]["token_sha256"]) == 20
