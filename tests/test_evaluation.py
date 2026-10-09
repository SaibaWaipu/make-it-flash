import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from make_it_flash import evaluation as evaluation_module
from make_it_flash.evaluation import (
    _evaluation_data_provenance,
    _japanese_tokenizer_probe,
    _preflight_evaluation_sequences,
    _score_tokenized_jsonl,
)
from make_it_flash.overlay import OVERLAY_FILENAME, OVERLAY_KIND, OVERLAY_VERSION
from make_it_flash.provenance import sha256_file


MODEL_ID = "fixture/model"
MODEL_REVISION = "a" * 40
TRAINING_PROVENANCE = {
    "data_sha256": "1" * 64,
    "manifest_sha256": "2" * 64,
    "dataset_id": "fixture/dataset",
    "dataset_revision": "3" * 40,
    "split": "train",
}


class ProbeTokenizer:
    text = "日本語の能力・tokenizer・MoE資産を維持します。"

    def encode(self, text, *, add_special_tokens):
        assert text == self.text
        assert add_special_tokens is False
        return [1, 2, 3]

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert messages == [{"role": "user", "content": self.text}]
        assert tokenize is True
        assert add_generation_prompt is True
        return torch.tensor([[4, 5, 6, 7]])

    def decode(self, token_ids, *, skip_special_tokens):
        assert token_ids == [1, 2, 3]
        assert skip_special_tokens is False
        return self.text


class UniformLM(torch.nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, 4)
        self.vocab_size = vocab_size

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, *, input_ids, use_cache):
        assert use_cache is False
        return SimpleNamespace(
            logits=torch.zeros((*input_ids.shape, self.vocab_size), device=input_ids.device)
        )


def _write_eval_corpus(tmp_path: Path, *, split: str = "validation") -> Path:
    data_file = tmp_path / "evaluation.jsonl"
    data_file.write_text(
        json.dumps({"sample_id": "jp-1", "category": "japanese", "input_ids": [0, 1, 2]})
        + "\n"
        + json.dumps({"sample_id": "en-1", "category": "english", "input_ids": [3, 4]})
        + "\n",
        encoding="utf-8",
    )
    data_file.with_name("data_manifest.json").write_text(
        json.dumps(
            {
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "dataset_id": "fixture/dataset",
                "dataset_revision": "3" * 40,
                "split": split,
                "data_file": data_file.name,
            }
        ),
        encoding="utf-8",
    )
    return data_file


def test_japanese_tokenizer_probe_records_text_and_chat_ids():
    report = _japanese_tokenizer_probe(ProbeTokenizer(), vocab_size=8)

    assert report["roundtrip_exact"] is True
    assert report["token_ids"] == [1, 2, 3]
    assert report["chat_template_token_ids"] == [4, 5, 6, 7]


def test_japanese_tokenizer_probe_rejects_out_of_vocab_chat_ids():
    class BadProbeTokenizer(ProbeTokenizer):
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            return [1, 2, 8]

    with pytest.raises(ValueError, match="outside vocabulary size 8"):
        _japanese_tokenizer_probe(BadProbeTokenizer(), vocab_size=8)


def test_evaluation_preflight_requires_qsa_pruning_in_scored_examples(tmp_path):
    data_file = tmp_path / "heldout.jsonl"
    data_file.write_text(
        json.dumps({"input_ids": [0, 1, 2, 3, 4]})
        + "\n"
        + json.dumps({"input_ids": [0, 1, 2, 3, 4, 5]})
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="at least one sequence of 6 tokens"):
        _preflight_evaluation_sequences(
            data_file,
            vocab_size=8,
            context_limit=8,
            pruning_threshold=6,
            max_examples=1,
        )

    summary = _preflight_evaluation_sequences(
        data_file,
        vocab_size=8,
        context_limit=8,
        pruning_threshold=6,
    )
    assert summary == {
        "scored_examples": 2,
        "qsa_pruning_examples": 1,
        "required_sequence_length": 6,
        "max_sequence_length": 6,
    }


def test_score_tokenized_jsonl_reports_perplexity_and_category_counts(tmp_path):
    data_file = _write_eval_corpus(tmp_path)

    report = _score_tokenized_jsonl(
        UniformLM(vocab_size=8),
        data_file,
        vocab_size=8,
        context_limit=8,
        max_examples=2,
        target_chunk_tokens=1,
    )

    assert report["examples"] == 2
    assert report["target_tokens"] == 3
    assert report["perplexity"] == pytest.approx(8.0)
    assert report["by_category"]["japanese"]["target_tokens"] == 2
    assert report["by_category"]["english"]["examples"] == 1


def test_evaluation_provenance_accepts_different_heldout_split(tmp_path):
    data_file = _write_eval_corpus(tmp_path)

    manifest, provenance = _evaluation_data_provenance(
        data_file,
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        calibration_data=TRAINING_PROVENANCE,
    )

    assert manifest["split"] == "validation"
    assert provenance["split"] == "validation"
    assert provenance["data_sha256"] == sha256_file(data_file)
    assert provenance["manifest_sha256"] != TRAINING_PROVENANCE["manifest_sha256"]


def test_evaluation_rejects_reused_token_sequence_even_with_new_split(tmp_path):
    import hashlib

    data_file = _write_eval_corpus(tmp_path)
    reused = [0, 1, 2]
    digest = hashlib.sha256(json.dumps(reused, separators=(",", ":")).encode()).hexdigest()
    training = {**TRAINING_PROVENANCE, "token_sha256": [digest]}
    with pytest.raises(ValueError, match="share token sequences"):
        _evaluation_data_provenance(data_file, model_id=MODEL_ID,
                                    model_revision=MODEL_REVISION, calibration_data=training)


def test_evaluation_provenance_rejects_same_dataset_split(tmp_path):
    data_file = _write_eval_corpus(tmp_path, split="train")
    training = dict(TRAINING_PROVENANCE)

    with pytest.raises(ValueError, match="dataset/split matches calibration"):
        _evaluation_data_provenance(
            data_file,
            model_id=MODEL_ID,
            model_revision=MODEL_REVISION,
            calibration_data=training,
        )


def test_evaluation_provenance_rejects_exact_calibration_corpus(tmp_path):
    data_file = _write_eval_corpus(tmp_path, split="train")
    manifest_path = data_file.with_name("data_manifest.json")
    training = {
        **TRAINING_PROVENANCE,
        "data_sha256": sha256_file(data_file),
        "manifest_sha256": sha256_file(manifest_path),
    }

    with pytest.raises(ValueError, match="identical to the calibration corpus"):
        _evaluation_data_provenance(
            data_file,
            model_id=MODEL_ID,
            model_revision=MODEL_REVISION,
            calibration_data=training,
        )


def test_evaluate_flash_next_runs_base_then_overlay_without_real_gpu_or_weights(tmp_path, monkeypatch):
    import transformers

    overlay_dir = tmp_path / "overlay"
    overlay_dir.mkdir()
    calibration_data = {
        "data_sha256": "1" * 64,
        "manifest_sha256": "2" * 64,
        "dataset_id": "fixture/dataset",
        "dataset_revision": "3" * 40,
        "split": "train",
    }
    (overlay_dir / OVERLAY_FILENAME).write_text(
        json.dumps(
            {
                "format": OVERLAY_KIND,
                "format_version": OVERLAY_VERSION,
                "base_model_id": MODEL_ID,
                "base_model_revision": MODEL_REVISION,
                "calibration_data": calibration_data,
                "qsa_config": {"token_budget": 4, "compress_ratio": 2},
                "base_config": {"vocab_size": 8, "max_position_embeddings": 10},
            }
        ),
        encoding="utf-8",
    )
    data_dir = tmp_path / "heldout"
    data_dir.mkdir()
    data_file = data_dir / "evaluation.jsonl"
    data_file.write_text(
        json.dumps({"sample_id": "jp-long", "category": "japanese", "input_ids": [0, 1, 2, 3, 4, 5]})
        + "\n",
        encoding="utf-8",
    )
    data_file.with_name("data_manifest.json").write_text(
        json.dumps(
            {
                "data_file": data_file.name,
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "dataset_id": "fixture/dataset",
                "dataset_revision": "3" * 40,
                "split": "validation",
            }
        ),
        encoding="utf-8",
    )

    def config():
        return SimpleNamespace(
            _name_or_path=MODEL_ID,
            _commit_hash=MODEL_REVISION,
            model_type="qwen3_moe",
            vocab_size=8,
            max_position_embeddings=10,
        )

    class FakeTokenizer(ProbeTokenizer):
        name_or_path = MODEL_ID
        _commit_hash = MODEL_REVISION
        init_kwargs = {"_commit_hash": MODEL_REVISION}
        all_special_ids = [0]

        def __len__(self):
            return 8

        def get_vocab(self):
            return {f"token-{index}": index for index in range(8)}

    model_loads = []

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            model_loads.append((args, kwargs))
            model = UniformLM(vocab_size=8)
            model.config = config()
            return model

    monkeypatch.setattr(transformers, "AutoConfig", SimpleNamespace(from_pretrained=lambda *a, **k: config()))
    monkeypatch.setattr(transformers, "AutoTokenizer", SimpleNamespace(from_pretrained=lambda *a, **k: FakeTokenizer()))
    monkeypatch.setattr(transformers, "AutoModelForCausalLM", FakeAutoModel)
    monkeypatch.setattr(evaluation_module, "_require_eval_gpu_memory", lambda min_free_gib: None)
    monkeypatch.setattr(evaluation_module, "load_flash_next_overlay", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    output_file = tmp_path / "report.json"
    report = evaluation_module.evaluate_flash_next(
        data_file=data_file,
        overlay_dir=overlay_dir,
        output_file=output_file,
        base_model_id=MODEL_ID,
        base_model_revision=MODEL_REVISION,
        target_chunk_tokens=2,
    )

    assert len(model_loads) == 2
    assert report["base"]["perplexity"] == pytest.approx(8.0)
    assert report["hybrid"]["perplexity"] == pytest.approx(8.0)
    assert report["qsa_pruning_preflight"]["qsa_pruning_examples"] == 1
    assert report["japanese_tokenizer_probe"]["roundtrip_exact"] is True
    assert json.loads(output_file.read_text(encoding="utf-8"))["relative_perplexity_change"] == pytest.approx(0.0)


def test_score_tokenized_jsonl_rejects_token_ids_outside_base_vocab(tmp_path):
    data_file = tmp_path / "bad.jsonl"
    data_file.write_text(json.dumps({"input_ids": [1, 9]}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="outside vocabulary size 8"):
        _score_tokenized_jsonl(
            UniformLM(vocab_size=8), data_file, vocab_size=8, context_limit=8
        )
