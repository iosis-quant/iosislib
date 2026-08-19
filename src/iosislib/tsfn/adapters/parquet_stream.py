"""Stream a logical Parquet file composed of subdivided objects.

A logical file is an ordered list of Parquet objects (local files or S3
objects) pinned by a chunk manifest. Each chunk's content SHA-256 is computed
by the producer at upload time and recorded in the manifest, so the source can
verify identity without downloading chunk bytes. The source loads the tiny
manifest object, verifies a pinned Merkle root derived from the per-chunk
digests, then lazily scans the chunk objects through ``pl.scan_parquet`` with
projection and predicate pushdown.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from iosislib.core.tsfn import FrameSignature, TSFN, TSFNConfig
from iosislib.core.utils import current_s3_credentials
from iosislib.tsfn.adapters.local_sources import (
    PathLike,
    _normalize_parquet_location,
    _normalize_pathlike,
    _open_parquet_filesystem,
    _project_declared_columns,
    _resolve_output_signature,
    _validate_output_signature,
    _UNRESOLVED_SIGNATURE,
)

_MANIFEST_FORMAT = "iosis.parquet-chunk-manifest"
_MANIFEST_VERSION = 1
_MANIFEST_FILE = "manifest.json"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MERKLE_ALGORITHM = "sha256"
_MERKLE_TREE = "balanced-binary"
_MERKLE_TAG = b"iosislib-parquet-chunk-merkle-v1\0"
_READ_BLOCK = 1024 * 1024


def _normalize_merkle_digest(digest: str) -> str:
    if not isinstance(digest, str):
        raise TypeError("content_sha256 must be a string")
    normalized = digest.lower()
    if normalized and not _SHA256.fullmatch(normalized):
        raise ValueError(
            "content_sha256 must be a 64-character hexadecimal digest or empty "
            "to stream without verification"
        )
    return normalized


@dataclass(frozen=True)
class ParquetChunk:
    """One subdivided Parquet object of a logical file.

    ``key`` is relative to the manifest location. ``sha256`` is the content
    digest of the object's exact bytes as computed by the producer at upload.
    """

    key: str
    sha256: str
    size: int
    rows: int

    def __post_init__(self) -> None:
        key = self.key
        if not isinstance(key, str) or not key.strip():
            raise ValueError("chunk key must be a non-empty string")
        normalized_key = key.replace("\\", "/")
        if (
            normalized_key.startswith("/")
            or any(part == ".." for part in normalized_key.split("/"))
            or "?" in normalized_key
            or "#" in normalized_key
        ):
            raise ValueError("chunk key must be a safe relative object path")
        object.__setattr__(self, "key", normalized_key)
        if not isinstance(self.sha256, str) or not _SHA256.fullmatch(self.sha256):
            raise ValueError("chunk sha256 must be a 64-character hexadecimal digest")
        if isinstance(self.size, bool) or not isinstance(self.size, int) or self.size < 1:
            raise ValueError("chunk size must be a positive integer")
        if isinstance(self.rows, bool) or not isinstance(self.rows, int) or self.rows < 1:
            raise ValueError("chunk rows must be a positive integer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "sha256": self.sha256,
            "size": self.size,
            "rows": self.rows,
        }

    @classmethod
    def from_dict(cls, value: object) -> ParquetChunk:
        data = _check_keys(value, "chunk", required=frozenset({"key", "sha256", "size", "rows"}))
        try:
            return cls(
                key=data["key"],
                sha256=data["sha256"],
                size=data["size"],
                rows=data["rows"],
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid chunk entry: {exc}") from exc


def _merkle_leaf(digest: bytes) -> bytes:
    return hashlib.sha256(_MERKLE_TAG + b"leaf\0" + digest).digest()


def _merkle_internal(left: bytes, right: bytes | None) -> bytes:
    if right is None:
        return left
    return hashlib.sha256(_MERKLE_TAG + b"node\0" + left + right).digest()


def chunk_merkle_root(chunks: Sequence[ParquetChunk]) -> str:
    """Return the balanced-binary Merkle root over ordered per-chunk digests.

    Leaves are the chunks' content SHA-256 digests in manifest order. Internal
    nodes hash their two children with a domain-separated tag; a single child
    at an odd level is promoted. Reordering or replacing any chunk changes the
    root, so the root pins the logical file's object layout and content.
    """
    leaves = [bytes.fromhex(chunk.sha256) for chunk in chunks]
    if not leaves:
        raise ValueError("chunk_merkle_root requires at least one chunk")
    level = [_merkle_leaf(digest) for digest in leaves]
    while len(level) > 1:
        next_level = []
        for index in range(0, len(level), 2):
            left = level[index]
            right = level[index + 1] if index + 1 < len(level) else None
            next_level.append(_merkle_internal(left, right))
        level = next_level
    return level[0].hex()


@dataclass(frozen=True)
class ChunkManifest:
    """The ordered chunk list and Merkle root of one logical Parquet file."""

    chunks: tuple[ParquetChunk, ...] = ()
    format: str = _MANIFEST_FORMAT
    version: int = _MANIFEST_VERSION

    def __post_init__(self) -> None:
        if self.format != _MANIFEST_FORMAT:
            raise ValueError(
                f"chunk manifest format must be {_MANIFEST_FORMAT!r}, got {self.format!r}"
            )
        if self.version != _MANIFEST_VERSION:
            raise ValueError(
                f"unsupported chunk manifest version: {self.version!r}"
            )
        if not isinstance(self.chunks, tuple) or not all(
            isinstance(chunk, ParquetChunk) for chunk in self.chunks
        ):
            raise TypeError("chunks must be a tuple of ParquetChunk values")
        if not self.chunks:
            raise ValueError("chunk manifest must declare at least one chunk")
        chunk_keys = [chunk.key for chunk in self.chunks]
        duplicate_keys = sorted(
            {key for key in chunk_keys if chunk_keys.count(key) > 1}
        )
        if duplicate_keys:
            raise ValueError(f"Duplicate chunk keys: {duplicate_keys}")

    @property
    def merkle_root(self) -> str:
        return chunk_merkle_root(self.chunks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "version": self.version,
            "merkle": {
                "algorithm": _MERKLE_ALGORITHM,
                "tree": _MERKLE_TREE,
                "root": self.merkle_root,
            },
            "chunks": [chunk.to_dict() for chunk in self.chunks],
        }

    def to_json_bytes(self) -> bytes:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    @classmethod
    def from_dict(cls, value: object) -> ChunkManifest:
        data = _check_keys(
            value,
            "chunk manifest",
            required=frozenset({"format", "version", "merkle", "chunks"}),
        )
        merkle = _check_keys(
            data["merkle"],
            "chunk manifest merkle",
            required=frozenset({"algorithm", "tree", "root"}),
        )
        chunks_value = data["chunks"]
        if not isinstance(chunks_value, list) or not all(
            isinstance(item, dict) for item in chunks_value
        ):
            raise ValueError("chunk manifest chunks must be an array of objects")
        try:
            manifest = cls(
                chunks=tuple(ParquetChunk.from_dict(item) for item in chunks_value),
                format=data["format"],
                version=data["version"],
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid chunk manifest: {exc}") from exc
        if merkle["algorithm"] != _MERKLE_ALGORITHM:
            raise ValueError(
                f"chunk manifest merkle algorithm must be {_MERKLE_ALGORITHM!r}"
            )
        if merkle["tree"] != _MERKLE_TREE:
            raise ValueError(
                f"chunk manifest merkle tree must be {_MERKLE_TREE!r}"
            )
        if not isinstance(merkle["root"], str) or not _SHA256.fullmatch(merkle["root"]):
            raise ValueError("chunk manifest merkle root must be a 64-hex digest")
        if merkle["root"] != manifest.merkle_root:
            raise ValueError(
                f"chunk manifest merkle root does not match its chunks: "
                f"declared {merkle['root']}, computed {manifest.merkle_root}"
            )
        return manifest

    @classmethod
    def from_bytes(cls, raw: bytes) -> ChunkManifest:
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("chunk manifest is not valid UTF-8 JSON") from exc
        return cls.from_dict(value)


def _check_keys(
    value: object,
    label: str,
    *,
    required: frozenset[str],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    missing = sorted(required - value.keys())
    if missing:
        raise ValueError(f"{label} is missing: {', '.join(missing)}")
    extra = sorted(value.keys() - required)
    if extra:
        raise ValueError(f"{label} has unknown field(s): {', '.join(extra)}")
    return value


def build_parquet_chunk_manifest(path: PathLike) -> str:
    """Author a chunk manifest for a local directory of Parquet files.

    Streams every ``.parquet`` file once in 1 MiB blocks to compute its content
    SHA-256, reads the row count from the Parquet metadata footer, and writes
    ``manifest.json`` (with the Merkle root) into the directory. Returns the
    root digest. S3 producers should hash each chunk object at upload and write
    the equivalent manifest alongside the objects.
    """
    from pyarrow.parquet import read_metadata

    directory = Path(_normalize_pathlike(path))
    if not directory.is_dir():
        raise ValueError(f"Parquet chunk directory does not exist: {directory!r}")
    parquet_files = sorted(directory.rglob("*.parquet"))
    if not parquet_files:
        raise ValueError(
            f"Parquet chunk directory contains no .parquet files: {directory!r}"
        )
    chunks: list[ParquetChunk] = []
    for file in parquet_files:
        key = str(file.relative_to(directory)).replace("\\", "/")
        digest = hashlib.sha256()
        with file.open("rb") as handle:
            for block in iter(lambda: handle.read(_READ_BLOCK), b""):
                digest.update(block)
        metadata = read_metadata(str(file))
        chunks.append(
            ParquetChunk(
                key=key,
                sha256=digest.hexdigest(),
                size=file.stat().st_size,
                rows=metadata.num_rows,
            )
        )
    manifest = ChunkManifest(chunks=tuple(chunks))
    (directory / _MANIFEST_FILE).write_bytes(manifest.to_json_bytes())
    return manifest.merkle_root


def _resolve_manifest_location(path: str, manifest: str | None) -> str:
    if manifest is not None:
        return manifest
    return f"{path.rstrip('/')}/{_MANIFEST_FILE}"


def _manifest_base(location: str) -> str:
    if location.startswith("s3://"):
        return location.rpartition("/")[0]
    return os.path.dirname(location)


def _join_location(base: str, key: str) -> str:
    if base.startswith("s3://"):
        return f"{base.rstrip('/')}/{key}"
    return os.path.join(base, key)


def _load_chunk_manifest(location: str) -> ChunkManifest:
    from pyarrow.fs import FileType

    filesystem, path = _open_parquet_filesystem(location)
    if filesystem.get_file_info(path).type != FileType.File:
        raise ValueError(f"Streaming parquet manifest does not exist: {location!r}")
    with filesystem.open_input_file(path) as file:
        raw = file.read()
    return ChunkManifest.from_bytes(raw)


def _verify_chunk_layout(manifest: ChunkManifest, base: str) -> None:
    from pyarrow.fs import FileType

    filesystem, fs_path = _open_parquet_filesystem(base)
    for chunk in manifest.chunks:
        info = filesystem.get_file_info(f"{fs_path}/{chunk.key}")
        if info.type != FileType.File:
            raise ValueError(
                f"Streaming parquet chunk is missing: {chunk.key!r}"
            )
        if info.size != chunk.size:
            raise ValueError(
                f"Streaming parquet chunk size mismatch for {chunk.key!r}. "
                f"Expected {chunk.size}, got {info.size}"
            )


def _scan_storage_options() -> dict[str, str]:
    credentials = current_s3_credentials()
    if credentials is None:
        return {}
    options = {
        "aws_access_key_id": credentials.access_key,
        "aws_secret_access_key": credentials.secret_key,
    }
    if credentials.session_token is not None:
        options["aws_session_token"] = credentials.session_token
    if credentials.region is not None:
        options["aws_region"] = credentials.region
    return options


@dataclass(frozen=True)
class StreamingParquetSourceConfig(TSFNConfig):
    path: PathLike
    output_signature: FrameSignature = _UNRESOLVED_SIGNATURE
    content_sha256: str = ""
    schema: Mapping[str, object] | None = None
    manifest: str | None = None
    verify_layout: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _normalize_parquet_location(self.path))
        signature = _resolve_output_signature(self.output_signature, self.schema)
        _validate_output_signature(signature)
        object.__setattr__(self, "output_signature", signature)
        object.__setattr__(self, "schema", None)
        object.__setattr__(
            self,
            "content_sha256",
            _normalize_merkle_digest(self.content_sha256),
        )
        if self.manifest is not None:
            object.__setattr__(
                self,
                "manifest",
                _normalize_parquet_location(self.manifest),
            )
        if not isinstance(self.verify_layout, bool):
            raise TypeError("verify_layout must be a boolean")


class StreamingParquetSource(TSFN):
    """Stream a logical Parquet file composed of subdivided local/S3 objects.

    Reads the chunk manifest (the only object fully loaded), verifies the
    pinned Merkle root against the producer-computed per-chunk content digests,
    then lazily scans the chunk objects in manifest order with projection and
    predicate pushdown. Optional ``verify_layout`` stats object sizes without
    downloading chunk bytes.
    """

    VERSION = "0.1.0"
    CONFIG_CLS = StreamingParquetSourceConfig

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        return FrameSignature.empty(), self.parameters.output_signature

    def apply(self) -> pl.LazyFrame:
        params = self.parameters
        manifest_location = _resolve_manifest_location(params.path, params.manifest)
        manifest = _load_chunk_manifest(manifest_location)
        if params.content_sha256 and manifest.merkle_root != params.content_sha256:
            raise ValueError(
                f"Streaming parquet manifest content digest mismatch for "
                f"{params.path!r}. Expected {params.content_sha256}, got "
                f"{manifest.merkle_root}"
            )
        base = _manifest_base(manifest_location)
        if params.verify_layout:
            _verify_chunk_layout(manifest, base)
        uris = [_join_location(base, chunk.key) for chunk in manifest.chunks]
        storage_options = _scan_storage_options() if base.startswith("s3://") else None
        frame = pl.scan_parquet(uris, storage_options=storage_options)
        return _project_declared_columns(frame, params.output_signature)


def merkle_sha256_parquet_source(path: PathLike) -> str:
    """Return the pinned Merkle root for a location's chunk manifest."""
    location = _normalize_parquet_location(path)
    manifest_location = _resolve_manifest_location(location, None)
    return _load_chunk_manifest(manifest_location).merkle_root


__all__ = [
    "ChunkManifest",
    "ParquetChunk",
    "StreamingParquetSource",
    "StreamingParquetSourceConfig",
    "build_parquet_chunk_manifest",
    "chunk_merkle_root",
    "merkle_sha256_parquet_source",
]
