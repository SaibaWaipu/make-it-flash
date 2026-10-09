"""Streaming Safetensors shard rewrite helpers for full checkpoint export.

The helpers preserve tensor payload bytes exactly while allowing selected
parameter names to be removed and fitted module tensors to be inserted without
materializing a multi-billion-parameter model in RAM or VRAM.
"""

from __future__ import annotations

import hashlib
import json
import struct
from contextlib import ExitStack
from pathlib import Path
from typing import Mapping, Sequence

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

_HEADER_LENGTH_BYTES = 8
_COPY_CHUNK_BYTES = 8 * 1024 * 1024


def read_safetensors_header(path: str | Path) -> tuple[dict, int]:
    """Return the parsed header and absolute start offset of tensor data."""
    source = Path(path)
    size = source.stat().st_size
    with source.open("rb") as stream:
        raw_length = stream.read(_HEADER_LENGTH_BYTES)
        if len(raw_length) != _HEADER_LENGTH_BYTES:
            raise ValueError(f"{source} is shorter than a Safetensors header")
        header_length = struct.unpack("<Q", raw_length)[0]
        if header_length > size - _HEADER_LENGTH_BYTES:
            raise ValueError(f"invalid Safetensors header length in {source}")
        raw_header = stream.read(header_length)
        if len(raw_header) != header_length:
            raise ValueError(f"truncated Safetensors header in {source}")
    try:
        header = json.loads(raw_header.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid Safetensors JSON header in {source}") from error
    if not isinstance(header, dict):
        raise ValueError(f"Safetensors header must be a JSON object: {source}")
    data_start = _HEADER_LENGTH_BYTES + header_length
    data_size = size - data_start
    for name, entry in header.items():
        if name == "__metadata__":
            if not isinstance(entry, dict) or any(
                not isinstance(key, str) or not isinstance(value, str)
                for key, value in entry.items()
            ):
                raise ValueError(f"invalid Safetensors metadata in {source}")
            continue
        if not isinstance(entry, dict):
            raise ValueError(f"invalid tensor header entry {name!r} in {source}")
        offsets = entry.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(value) is not int for value in offsets)
            or offsets[0] < 0
            or offsets[1] < offsets[0]
            or offsets[1] > data_size
        ):
            raise ValueError(f"invalid tensor data offsets for {name!r} in {source}")
    return header, data_start


def safetensors_tensor_sizes(path: str | Path) -> dict[str, int]:
    """Return tensor payload byte lengths without reading payloads into RAM."""
    header, _ = read_safetensors_header(path)
    return {
        name: entry["data_offsets"][1] - entry["data_offsets"][0]
        for name, entry in header.items()
        if name != "__metadata__"
    }


def cast_safetensors_to_bfloat16(
    source_path: str | Path,
    destination_path: str | Path,
    *,
    metadata_updates: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Cast floating tensors in one small module checkpoint to BF16.

    This matches the runtime graft, which casts fitted modules to the BF16 base
    attention dtype. It is intended for one adapter module at a time, not a full
    model shard.
    """
    source = Path(source_path)
    destination = Path(destination_path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite converted checkpoint {destination}")
    with safe_open(str(source), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
        tensors = {name: handle.get_tensor(name) for name in handle.keys()}
    source_dtypes = {str(tensor.dtype) for tensor in tensors.values()}
    converted = {
        name: tensor.to(dtype=torch.bfloat16) if tensor.is_floating_point() else tensor
        for name, tensor in tensors.items()
    }
    for key, value in (metadata_updates or {}).items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("Safetensors metadata updates must be string pairs")
        metadata[key] = value
    save_file(converted, str(destination), metadata=metadata or None)
    with safe_open(str(destination), framework="pt", device="cpu") as handle:
        if set(handle.keys()) != set(converted):
            raise ValueError("converted Safetensors keys do not match the source")
    output_dtypes = {str(tensor.dtype) for tensor in converted.values()}
    return {
        "path": str(destination),
        "source_dtypes": sorted(source_dtypes),
        "output_dtypes": sorted(output_dtypes),
        "tensor_count": len(converted),
        "file_size": destination.stat().st_size,
        "sha256": _sha256_file(destination),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_COPY_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rewrite_safetensors_shard(
    source_path: str | Path,
    destination_path: str | Path,
    *,
    remove_prefixes: Sequence[str] = (),
    additions: Sequence[tuple[str | Path, str]] = (),
    metadata_updates: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Rewrite one shard by streaming original and prefixed tensor byte ranges.

    ``remove_prefixes`` removes every source tensor whose name begins with any
    listed prefix. Each ``additions`` entry is ``(file_path, key_prefix)``; its
    tensor names are prefixed and appended to the output. Tensor dtypes, shapes,
    and raw payloads are preserved exactly.
    """
    source = Path(source_path)
    destination = Path(destination_path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite checkpoint shard {destination}")
    if not remove_prefixes or any(not isinstance(prefix, str) or not prefix for prefix in remove_prefixes):
        raise ValueError("remove_prefixes must contain non-empty parameter prefixes")

    source_header, source_data_start = read_safetensors_header(source)
    sources: list[tuple[Path, int, int, int]] = []
    header: dict[str, object] = {}
    removed_keys: list[str] = []
    added_keys: list[str] = []
    removed_tensor_bytes = 0
    added_tensor_bytes = 0
    data_cursor = 0

    def append_tensor(
        name: str,
        entry: dict,
        payload_path: Path,
        payload_data_start: int,
    ) -> None:
        nonlocal data_cursor
        if name in header:
            raise ValueError(f"duplicate tensor key while rewriting shard: {name}")
        start, end = entry["data_offsets"]
        length = end - start
        rewritten_entry = {key: value for key, value in entry.items() if key != "data_offsets"}
        rewritten_entry["data_offsets"] = [data_cursor, data_cursor + length]
        header[name] = rewritten_entry
        sources.append((payload_path, payload_data_start, start, end))
        data_cursor += length

    for name, entry in source_header.items():
        if name == "__metadata__":
            continue
        if any(name.startswith(prefix) for prefix in remove_prefixes):
            removed_keys.append(name)
            removed_tensor_bytes += entry["data_offsets"][1] - entry["data_offsets"][0]
            continue
        append_tensor(name, entry, source, source_data_start)

    for addition_path, prefix in additions:
        addition = Path(addition_path)
        if not isinstance(prefix, str):
            raise ValueError("tensor key prefixes must be strings")
        addition_header, addition_data_start = read_safetensors_header(addition)
        for name, entry in addition_header.items():
            if name == "__metadata__":
                continue
            prefixed_name = f"{prefix}{name}"
            append_tensor(prefixed_name, entry, addition, addition_data_start)
            added_keys.append(prefixed_name)
            added_tensor_bytes += entry["data_offsets"][1] - entry["data_offsets"][0]

    if not header:
        raise ValueError("rewritten Safetensors shard would contain no tensors")
    output_metadata = dict(source_header.get("__metadata__", {}))
    for key, value in (metadata_updates or {}).items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("Safetensors metadata updates must be string pairs")
        output_metadata[key] = value
    if output_metadata:
        header["__metadata__"] = output_metadata

    encoded_header = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    padding = (-len(encoded_header)) % 8
    encoded_header += b" " * padding
    prefix_bytes = struct.pack("<Q", len(encoded_header)) + encoded_header
    digest = hashlib.sha256()
    digest.update(prefix_bytes)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with ExitStack() as stack, destination.open("xb") as output:
        handles = {source: stack.enter_context(source.open("rb"))}
        for payload_path, _, _, _ in sources:
            if payload_path not in handles:
                handles[payload_path] = stack.enter_context(payload_path.open("rb"))
        output.write(prefix_bytes)
        for payload_path, payload_data_start, start, end in sources:
            input_stream = handles[payload_path]
            input_stream.seek(payload_data_start + start)
            remaining = end - start
            while remaining:
                chunk = input_stream.read(min(_COPY_CHUNK_BYTES, remaining))
                if not chunk:
                    raise ValueError(f"truncated tensor payload in {payload_path}")
                output.write(chunk)
                digest.update(chunk)
                remaining -= len(chunk)
        output.flush()

    expected_size = len(prefix_bytes) + data_cursor
    actual_size = destination.stat().st_size
    if actual_size != expected_size:
        raise ValueError(f"rewritten shard size mismatch: expected {expected_size}, got {actual_size}")
    with safe_open(str(destination), framework="pt", device="cpu") as handle:
        if set(handle.keys()) != set(header) - {"__metadata__"}:
            raise ValueError("rewritten Safetensors keys do not match the generated header")
    return {
        "path": str(destination),
        "sha256": digest.hexdigest(),
        "file_size": actual_size,
        "tensor_bytes": data_cursor,
        "removed_tensor_bytes": removed_tensor_bytes,
        "added_tensor_bytes": added_tensor_bytes,
        "source_tensor_bytes": data_cursor + removed_tensor_bytes - added_tensor_bytes,
        "removed_keys": removed_keys,
        "added_keys": added_keys,
        "tensor_count": len(header) - (1 if "__metadata__" in header else 0),
    }
