"""Config class for the integrated LLM-jp Flash-Next checkpoint.

The config keeps Transformers-recognized layer types for compatibility. The
actual GDN/QSA schedule and dimensions are stored in ``flash_next_integration``.
"""

from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig


class FlashNextQwen3MoeConfig(Qwen3MoeConfig):
    """Qwen3-MoE config carrying the integrated Flash-Next module metadata."""

    model_type = "qwen3_moe"


__all__ = ["FlashNextQwen3MoeConfig"]
