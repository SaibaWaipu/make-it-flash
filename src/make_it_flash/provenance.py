"""Version tags needed to safely reinstantiate saved GDN/QSA modules."""

from __future__ import annotations

import hashlib
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
