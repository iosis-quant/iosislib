from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from iosislib.core.tsfn import (
    FrameSignature,
    TimeAxis,
    TSFN,
    TSFNConfig,
    _column_signatures,
    _frame_physical_schema,
)

if TYPE_CHECKING:
    from pyarrow.fs import FileInfo, FileSystem


PathLike = str | os.PathLike[str]
ParquetSnapshot = tuple[tuple[str, bytes], ...]


def sha256_file(path: PathLike) -> str:
    """Read a local file and return its lowercase SHA-256 content digest."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_pathlike(path: PathLike) -> str:
    if not isinstance(path, (str, os.PathLike)):
        raise TypeError("path must be a string or path-like object")
    normalized = os.fspath(path)
    if not isinstance(normalized, str):
        raise TypeError("path must resolve to a string, not bytes")
    if not normalized:
        raise ValueError("path must be non-empty")
    return normalized


def _normalize_path(path: PathLike) -> str:
    return str(Path(_normalize_pathlike(path)))


def _normalize_parquet_location(path: PathLike) -> str:
    normalized = _normalize_pathlike(path)
    if "://" not in normalized:
        return str(Path(normalized))
    if not normalized.startswith("s3://"):
        raise ValueError("ParquetSource supports only local paths and s3:// locations")
    without_scheme = normalized.removeprefix("s3://")
    bucket, separator, key = without_scheme.partition("/")
    if not bucket:
        raise ValueError("s3:// location must include a bucket name")
    if any(character in bucket for character in "?#"):
        raise ValueError("s3:// bucket name must not contain a query or fragment")
    if "?" in key or "#" in key:
        raise ValueError("s3:// location must not contain a query or fragment")
    normalized_key = key.rstrip("/")
    return (
        f"s3://{bucket}/{normalized_key}"
        if separator and normalized_key
        else f"s3://{bucket}"
    )


def _normalize_content_sha256(content_sha256: str) -> str:
    if not isinstance(content_sha256, str):
        raise TypeError("content_sha256 must be a string")
    normalized = content_sha256.lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError("content_sha256 must be a 64-character hexadecimal digest")
    return normalized


def _validate_output_signature(output_signature: FrameSignature) -> None:
    if not isinstance(output_signature, FrameSignature):
        raise TypeError("output_signature must be a FrameSignature")
    if output_signature.is_empty():
        raise ValueError("output_signature must declare a time axis")


_STRING_DTYPES = {
    "bool": pl.Boolean,
    "float32": pl.Float32,
    "float64": pl.Float64,
    "int32": pl.Int32,
    "int64": pl.Int64,
    "string": pl.String,
}


def _dtype_from_string(value: object) -> pl.DataType:
    if not isinstance(value, str) or value not in _STRING_DTYPES:
        raise ValueError(
            f"Unsupported schema dtype {value!r}; expected one of "
            f"{sorted(_STRING_DTYPES)}"
        )
    return _STRING_DTYPES[value]


def _signature_from_schema(schema: Mapping[str, object]) -> FrameSignature:
    """Build a FrameSignature from the portable source DSL schema form.

    The portable form is ``{"time": <time column name>, "columns":
    {<name>: <dtype string>}}`` and matches the source nodes emitted by the
    web strategy model.
    """
    extra = sorted(schema.keys() - {"time", "columns"})
    if extra:
        raise ValueError(f"Unsupported schema field(s): {extra}")
    time_value = schema.get("time")
    if not isinstance(time_value, str) or not time_value.strip():
        raise ValueError("schema.time must be the time column name")
    columns_value = schema.get("columns", {})
    if not isinstance(columns_value, Mapping):
        raise TypeError("schema.columns must be a mapping")
    columns = tuple(
        (name, _dtype_from_string(dtype_value))
        for name, dtype_value in sorted(columns_value.items())
    )
    return FrameSignature(time=TimeAxis(column=time_value), columns=columns)


_UNRESOLVED_SIGNATURE = FrameSignature(time=None, columns=())


def _resolve_output_signature(
    output_signature: FrameSignature,
    schema: Mapping[str, object] | None,
) -> FrameSignature:
    if (output_signature is _UNRESOLVED_SIGNATURE) == (schema is None):
        raise ValueError(
            "Source configuration requires exactly one of 'output_signature' or 'schema'"
        )
    if output_signature is not _UNRESOLVED_SIGNATURE:
        return output_signature
    assert schema is not None
    return _signature_from_schema(schema)


def _verify_content_sha256(
    *,
    source: str,
    location: str,
    expected_sha256: str,
    actual_sha256: str,
) -> None:
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{source} content digest mismatch for {location!r}. "
            f"Expected {expected_sha256}, got {actual_sha256}"
        )


def _read_verified_snapshot(path: str, expected_sha256: str) -> bytes:
    """Read, verify, and return the one byte snapshot this execution consumes."""
    snapshot = Path(path).read_bytes()
    _verify_content_sha256(
        source="Local file",
        location=path,
        expected_sha256=expected_sha256,
        actual_sha256=hashlib.sha256(snapshot).hexdigest(),
    )
    return snapshot


def _open_parquet_filesystem(location: str) -> tuple[FileSystem, str]:
    from pyarrow.fs import FileSystem, LocalFileSystem

    if location.startswith("s3://"):
        return FileSystem.from_uri(location)
    filesystem = LocalFileSystem()
    return filesystem, filesystem.normalize_path(location)


def _relative_name(path: str, base_path: str) -> str:
    prefix = f"{base_path.rstrip('/')}/" if base_path else ""
    return path[len(prefix) :] if prefix and path.startswith(prefix) else path


def _parquet_file_infos(
    filesystem: FileSystem,
    path: str,
    location: str,
) -> tuple[tuple[str, FileInfo], ...]:
    from pyarrow.fs import FileSelector, FileType

    info = filesystem.get_file_info(path)
    if info.type == FileType.File:
        candidates = (info,)
        base_path = info.path.rpartition("/")[0]
        explicit_file = True
    else:
        if info.type == FileType.NotFound and not location.startswith("s3://"):
            raise FileNotFoundError(f"Parquet location does not exist: {location!r}")
        candidates = tuple(
            filesystem.get_file_info(
                FileSelector(path, recursive=True, allow_not_found=True)
            )
        )
        base_path = path.rstrip("/")
        explicit_file = False

    selected = tuple(
        sorted(
            (
                (_relative_name(candidate.path, base_path), candidate)
                for candidate in candidates
                if candidate.type == FileType.File
                and (explicit_file or candidate.path.lower().endswith(".parquet"))
            ),
            key=lambda item: item[0],
        )
    )
    if not selected:
        raise ValueError(
            f"Parquet location contains no .parquet files or objects: {location!r}"
        )
    return selected


def _read_parquet_snapshot(location: str) -> ParquetSnapshot:
    filesystem, path = _open_parquet_filesystem(location)
    infos = _parquet_file_infos(filesystem, path, location)
    objects: list[tuple[str, bytes]] = []
    for relative_name, info in infos:
        with filesystem.open_input_file(info.path) as file:
            objects.append((relative_name, file.read()))
    return tuple(objects)


def _parquet_snapshot_sha256(snapshot: ParquetSnapshot) -> str:
    # A single object intentionally retains sha256_file compatibility, including
    # when reached through a directory/prefix. Multi-object identity includes the
    # physical manifest, so repartitioning intentionally changes the digest.
    if len(snapshot) == 1:
        return hashlib.sha256(snapshot[0][1]).hexdigest()

    digest = hashlib.sha256(b"iosislib-parquet-dataset-v1\0")
    for name, content in snapshot:
        encoded_name = name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(hashlib.sha256(content).digest())
    return digest.hexdigest()


def sha256_parquet_source(path: PathLike) -> str:
    """Return the digest for a local or S3 Parquet file/dataset snapshot."""
    location = _normalize_parquet_location(path)
    return _parquet_snapshot_sha256(_read_parquet_snapshot(location))


def _read_verified_parquet_snapshot(
    location: str,
    expected_sha256: str,
) -> ParquetSnapshot:
    snapshot = _read_parquet_snapshot(location)
    actual_sha256 = _parquet_snapshot_sha256(snapshot)
    _verify_content_sha256(
        source="Parquet",
        location=location,
        expected_sha256=expected_sha256,
        actual_sha256=actual_sha256,
    )
    return snapshot


def _project_declared_columns(
    frame: pl.LazyFrame,
    signature: FrameSignature,
) -> pl.LazyFrame:
    return frame.select(*_frame_physical_schema(signature))


@dataclass(frozen=True)
class CSVSourceConfig(TSFNConfig):
    path: PathLike
    content_sha256: str
    output_signature: FrameSignature = _UNRESOLVED_SIGNATURE
    schema: Mapping[str, object] | None = None
    separator: str = ","

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _normalize_path(self.path))
        signature = _resolve_output_signature(self.output_signature, self.schema)
        _validate_output_signature(signature)
        object.__setattr__(self, "output_signature", signature)
        object.__setattr__(self, "schema", None)
        object.__setattr__(
            self,
            "content_sha256",
            _normalize_content_sha256(self.content_sha256),
        )
        if not isinstance(self.separator, str):
            raise TypeError("separator must be a string")
        if len(self.separator.encode("utf-8")) != 1:
            raise ValueError("separator must be a single-byte character")
        if any(column.shape for column in _column_signatures(signature)):
            raise ValueError(
                "CSVSource does not support shaped columns; use ParquetSource"
            )


class CSVSource(TSFN):
    """Read one verified in-memory snapshot, then expose parsed data lazily.

    Execution holds the complete source bytes plus the parsed frame in memory. No
    staging file or other persistent artifact is created.
    """

    VERSION = "0.2.0"
    CONFIG_CLS = CSVSourceConfig

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        return FrameSignature.empty(), self.parameters.output_signature

    def apply(self) -> pl.LazyFrame:
        params = self.parameters
        snapshot = _read_verified_snapshot(params.path, params.content_sha256)
        frame = pl.read_csv(
            snapshot,
            has_header=True,
            separator=params.separator,
            schema_overrides=_frame_physical_schema(params.output_signature),
            try_parse_dates=True,
        ).lazy()
        return _project_declared_columns(frame, params.output_signature)


@dataclass(frozen=True)
class ParquetSourceConfig(TSFNConfig):
    path: PathLike
    content_sha256: str
    output_signature: FrameSignature = _UNRESOLVED_SIGNATURE
    schema: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _normalize_parquet_location(self.path))
        signature = _resolve_output_signature(self.output_signature, self.schema)
        _validate_output_signature(signature)
        object.__setattr__(self, "output_signature", signature)
        object.__setattr__(self, "schema", None)
        object.__setattr__(
            self,
            "content_sha256",
            _normalize_content_sha256(self.content_sha256),
        )


class ParquetSource(TSFN):
    """Read a local or S3 Parquet snapshot, then expose parsed data lazily.

    A location may be a file/object or a directory/prefix. Dataset files are read
    recursively in lexicographic path order. Execution holds all source bytes plus
    parsed frames in memory and creates no staging artifacts.
    """

    VERSION = "0.3.0"
    CONFIG_CLS = ParquetSourceConfig

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        return FrameSignature.empty(), self.parameters.output_signature

    def apply(self) -> pl.LazyFrame:
        params = self.parameters
        snapshot = _read_verified_parquet_snapshot(
            params.path,
            params.content_sha256,
        )
        frames = [pl.read_parquet(content) for _, content in snapshot]
        frame = (
            frames[0] if len(frames) == 1 else pl.concat(frames, how="vertical")
        ).lazy()
        return _project_declared_columns(frame, params.output_signature)


__all__ = [
    "CSVSource",
    "CSVSourceConfig",
    "ParquetSource",
    "ParquetSourceConfig",
    "sha256_file",
    "sha256_parquet_source",
]
