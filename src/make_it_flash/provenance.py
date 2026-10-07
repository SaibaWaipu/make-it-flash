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


def validate_calibration_data_provenance(value: dict[str, object] | None) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("cache manifest lacks calibration-data provenance")
    for field in ("data_sha256", "manifest_sha256"):
        digest = value.get(field)
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"calibration-data provenance has an invalid {field}")
    return dict(value)


def validate_cache_sample_metadata(
    metadata: dict[str, str] | None,
    *,
    path: str | Path,
    model_id: str,
    model_revision: str,
    calibration_data: dict[str, object],
) -> None:
    if not isinstance(metadata, dict):
        raise ValueError(f"cache shard {Path(path).name} has no provenance metadata")
    if metadata.get("model_id") != model_id or metadata.get("model_revision") != model_revision:
        raise ValueError(
            f"cache shard {Path(path).name} base ID/revision does not match cache manifest"
        )
    if not isinstance(metadata.get("sample_id"), str) or not metadata["sample_id"]:
        raise ValueError(f"cache shard {Path(path).name} has no sample_id provenance")
    expected_data = validate_calibration_data_provenance(calibration_data)
    if (
        metadata.get("calibration_data_sha256") != expected_data["data_sha256"]
        or metadata.get("data_manifest_sha256") != expected_data["manifest_sha256"]
    ):
        raise ValueError(f"cache shard {Path(path).name} calibration provenance does not match cache manifest")


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
