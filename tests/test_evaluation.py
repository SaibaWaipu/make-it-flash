import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from make_it_flash.evaluation import (
    _evaluation_data_provenance,
    _preflight_evaluation_sequences,
    _score_tokenized_jsonl,
)
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


def test_score_tokenized_jsonl_rejects_token_ids_outside_base_vocab(tmp_path):
    data_file = tmp_path / "bad.jsonl"
    data_file.write_text(json.dumps({"input_ids": [1, 9]}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="outside vocabulary size 8"):
        _score_tokenized_jsonl(
            UniformLM(vocab_size=8), data_file, vocab_size=8, context_limit=8
        )
