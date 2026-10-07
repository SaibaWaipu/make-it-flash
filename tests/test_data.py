import json

import pytest

from make_it_flash.cli import DEFAULT_MODEL, build_parser
from make_it_flash.data import allocate_quotas, normalize_messages, prepare_calibration_data, tokenize_messages


def test_default_teacher_is_llm_jp_41_thinking():
    from inspect import signature

    expected = "llm-jp/llm-jp-4.1-32b-a3b-thinking"
    assert DEFAULT_MODEL == expected
    assert build_parser().parse_args(["prepare"]).model == expected
    assert signature(prepare_calibration_data).parameters["model_id"].default == expected


def test_default_quotas_match_planned_mix():
    quotas = allocate_quotas(100_000)
    assert quotas == {
        "japanese": 35_000,
        "english": 25_000,
        "math": 15_000,
        "code": 15_000,
        "tool_agent": 10_000,
    }
    assert sum(quotas.values()) == 100_000


def test_normalize_json_encoded_harmony_messages():
    raw = json.dumps({"messages": [{"role": "user", "content": ["計算 ", {"text": "1+1"}]}]})
    result = normalize_messages(raw)
    assert result[0]["content"] == "計算 1+1"


def test_tokenize_uses_chat_template_without_open_generation():
    class FakeTokenizer:
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            assert tokenize is True
            assert add_generation_prompt is False
            assert messages[0]["content"] == "hello"
            return [11, 12, 13]

    assert tokenize_messages(FakeTokenizer(), [{"role": "user", "content": ["hello"]}]) == [11, 12, 13]


def test_tokenize_accepts_batch_encoding_mapping():
    class MappingTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return {"input_ids": torch.tensor([[21, 22]])}

    import torch

    assert tokenize_messages(MappingTokenizer(), [{"role": "user", "content": "hi"}]) == [21, 22]


def test_tool_observation_falls_back_to_tagged_user_turn():
    class ToolTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            if any(message["role"] == "tool" for message in messages):
                raise ValueError("orphaned tool role")
            assert messages[1]["role"] == "user"
            assert messages[1]["content"].startswith("[tool observation: python]")
            return [31, 32]

    messages = [
        {"role": "assistant", "content": "running code"},
        {"role": "tool", "name": "python", "content": "2"},
    ]
    assert tokenize_messages(ToolTokenizer(), messages) == [31, 32]


def test_invalid_mix_is_rejected():
    from make_it_flash.data import MixSource

    with pytest.raises(ValueError, match="sum to 1.0"):
        allocate_quotas(10, [MixSource("x", 0.5, ("a",))])
