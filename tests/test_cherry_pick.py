import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from make_it_flash import cherry_pick
from make_it_flash.full_run import preflight_full_run
from make_it_flash.provenance import sha256_file


class Tokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return list(range(int(messages[0]["content"])))


def test_cherry_pick_three_sources_and_provenance(tmp_path, monkeypatch):
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer

    monkeypatch.setattr(HfApi, "model_info", lambda *args, **kwargs: SimpleNamespace(sha=cherry_pick.DEFAULT_MODEL_REVISION))
    monkeypatch.setattr(HfApi, "dataset_info", lambda *args, **kwargs: SimpleNamespace(sha=cherry_pick.DATASET_REVISION))
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: Tokenizer())
    calls = []

    def fetch(**kwargs):
        calls.append(kwargs["config_name"])
        Path(kwargs["output_file"]).write_text("".join(
            json.dumps({"sample_id": f"{kwargs['config_name']}-{i}", "messages": [{"role": "user", "content": str(2052 + i)}]}) + "\n"
            for i in range(80)), encoding="utf-8")
        return {"candidate_rows": 80}

    from make_it_flash import worker

    monkeypatch.setattr(worker, "fetch_one_source", fetch)
    result = cherry_pick.prepare_cherry_picked_calibration(output_dir=tmp_path / "output", max_tokens=24_000)
    manifest = result["manifest"]
    assert set(calls) == {"daring_anteater", "llmjp_extraction_wiki_ja_v0.3", "synthetic_jp_en_coding"}
    assert manifest["actual_tokens_by_category"] == {"japanese": 9600, "english": 7200, "code": 7200}
    assert all(result["preflight"]["counts"].values())
    assert all(count >= 2 for count in manifest["long_context_sequences_by_category"].values())
    from make_it_flash.full_run import _split_value

    for category in ("japanese", "english", "code"):
        long_rows = [row for row in (json.loads(line) for line in Path(result["data_file"]).read_text().splitlines())
                     if row["category"] == category and len(row["input_ids"]) >= 2052]
        assert any(_split_value(row["config"], row["sample_id"]) < int(0.1 * (2**64 - 1)) for row in long_rows)
        assert any(_split_value(row["config"], row["sample_id"]) >= int(0.1 * (2**64 - 1)) for row in long_rows)
    assert sha256_file(result["data_file"]) == preflight_full_run(result["data_file"])["data_sha256"]
    assert not any("messages" in json.loads(line) for line in Path(result["data_file"]).read_text().splitlines())


def test_cherry_pick_refuses_invalid_sequence_length_without_writing(tmp_path):
    with pytest.raises(ValueError, match="max_seq_len"):
        cherry_pick.prepare_cherry_picked_calibration(output_dir=tmp_path / "output", max_seq_len=2048)
    assert not (tmp_path / "output").exists()
