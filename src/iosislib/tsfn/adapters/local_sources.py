from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from iosislib.core.tsfn import (
    FrameSignature,
    TSFN,
    TSFNConfig,
    _column_signatures,
    _frame_physical_schema,
)


PathLike = str | os.PathLike[str]


def sha256_file(path: PathLike) -> str:
    """Read a local file and return its lowercase SHA-256 content digest."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_path(path: PathLike) -> str:
    if not isinstance(path, (str, os.PathLike)):
        raise TypeError("path must be a string or path-like object")
    normalized = os.fspath(path)
    if not isinstance(normalized, str):
        raise TypeError("path must resolve to a string, not bytes")
    if not normalized:
        raise ValueError("path must be non-empty")
    return str(Path(normalized))


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


def _verify_snapshot(path: str, expected_sha256: str) -> None:
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"Local file content digest mismatch for {path!r}. "
            f"Expected {expected_sha256}, got {actual_sha256}"
        )


def _project_declared_columns(
    frame: pl.LazyFrame,
    signature: FrameSignature,
) -> pl.LazyFrame:
    return frame.select(*_frame_physical_schema(signature))


@dataclass(frozen=True)
class CSVSourceConfig(TSFNConfig):
    path: PathLike
    output_signature: FrameSignature
    content_sha256: str
    separator: str = ","

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _normalize_path(self.path))
        _validate_output_signature(self.output_signature)
        object.__setattr__(
            self,
            "content_sha256",
            _normalize_content_sha256(self.content_sha256),
        )
        if not isinstance(self.separator, str):
            raise TypeError("separator must be a string")
        if len(self.separator.encode("utf-8")) != 1:
            raise ValueError("separator must be a single-byte character")
        if any(column.shape for column in _column_signatures(self.output_signature)):
            raise ValueError(
                "CSVSource does not support shaped columns; use ParquetSource"
            )


class CSVSource(TSFN):
    VERSION = "0.1.0"
    CONFIG_CLS = CSVSourceConfig

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        return FrameSignature.empty(), self.parameters.output_signature

    def apply(self) -> pl.LazyFrame:
        params = self.parameters
        _verify_snapshot(params.path, params.content_sha256)
        frame = pl.scan_csv(
            params.path,
            has_header=True,
            separator=params.separator,
            schema_overrides=_frame_physical_schema(params.output_signature),
            try_parse_dates=True,
            glob=False,
        )
        return _project_declared_columns(frame, params.output_signature)


@dataclass(frozen=True)
class ParquetSourceConfig(TSFNConfig):
    path: PathLike
    output_signature: FrameSignature
    content_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _normalize_path(self.path))
        _validate_output_signature(self.output_signature)
        object.__setattr__(
            self,
            "content_sha256",
            _normalize_content_sha256(self.content_sha256),
        )


class ParquetSource(TSFN):
    VERSION = "0.1.0"
    CONFIG_CLS = ParquetSourceConfig

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        return FrameSignature.empty(), self.parameters.output_signature

    def apply(self) -> pl.LazyFrame:
        params = self.parameters
        _verify_snapshot(params.path, params.content_sha256)
        frame = pl.scan_parquet(params.path, glob=False)
        return _project_declared_columns(frame, params.output_signature)


__all__ = [
    "CSVSource",
    "CSVSourceConfig",
    "ParquetSource",
    "ParquetSourceConfig",
    "sha256_file",
]
