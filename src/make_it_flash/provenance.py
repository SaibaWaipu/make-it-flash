"""Version tags needed to safely reinstantiate saved GDN/QSA modules."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

CHECKPOINT_SCHEMA_VERSION = "1"
_IMPLEMENTATION_FILES = (
    "model.py",
    "flash_next.py",
    "cache.py",
    "fit.py",
    "fit_qsa.py",
    "overlay.py",
    "provenance.py",
)


def validate_model_revision(revision: str) -> None:
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-fA-F]{40}", revision) is None:
        raise ValueError("base_model_revision must be a pinned 40-character commit SHA")


def validate_base_config_provenance(config, model_id: str, revision: str) -> None:
    validate_model_revision(revision)
    actual_model_id = getattr(config, "_name_or_path", None)
    actual_revision = getattr(config, "_commit_hash", None)
    if actual_model_id != model_id:
        raise ValueError(
            f"base config source is unverified: expected _name_or_path={model_id!r}, got {actual_model_id!r}"
        )
    if actual_revision != revision:
        raise ValueError(
            f"base config revision is unverified: expected _commit_hash={revision!r}, got {actual_revision!r}"
        )


def _implementation_fingerprint() -> str:
    package_root = Path(__file__).parent
    digest = hashlib.sha256()
    for name in _IMPLEMENTATION_FILES:
        path = package_root / name
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_provenance() -> dict[str, str]:
    """Return exact runtime versions serialized into every fitted module file."""
    import transformers

    from . import __version__

    return {
        "transformers_version": str(transformers.__version__),
        "make_it_flash_version": str(__version__),
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "implementation_fingerprint": _implementation_fingerprint(),
    }
