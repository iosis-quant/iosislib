"""Scan a partitioned Parquet dataset as a single logical table.

Uses Polars' native ``scan_parquet`` with ``hive_partitioning=True`` to
treat a directory of Parquet files as one unified table.  Any ``key=value``
directories in the path are automatically detected as partition columns -
no explicit declaration needed.

Just pass a glob path and Polars does the rest:

- ``s3://bucket/data/**/*.parquet`` - flat files, no partitioning
- ``s3://bucket/data/year=*/month=*/*.parquet`` - hive-partitioned
- ``/local/path/**/*.parquet`` - works locally too

Optional time-range filtering pushes predicates down to the Parquet reader
so entire partition directories and row groups are skipped.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from iosislib.core.tsfn import (
    FrameSignature,
    TSFN,
    TSFNConfig,
    _time_axis_physical_dtype,
)
from iosislib.core.utils import current_s3_credentials
from iosislib.tsfn.adapters.local_sources import (
    _project_declared_columns,
    _resolve_output_signature,
    _validate_output_signature,
    _UNRESOLVED_SIGNATURE,
)

_MANIFEST_FORMAT = "iosis.cloud-dataset-v1"


def _normalize_path(path: str) -> str:
    normalized = path.rstrip("/")
    if not normalized:
        raise ValueError("path must be non-empty")
    if normalized.startswith("s3://"):
        without_scheme = normalized.removeprefix("s3://")
        bucket, separator, key = without_scheme.partition("/")
        if not bucket:
            raise ValueError("s3:// location must include a bucket name")
        if any(c in bucket for c in "?#"):
            raise ValueError("s3:// bucket name must not contain a query or fragment")
        if "?" in key or "#" in key:
            raise ValueError("s3:// location must not contain a query or fragment")
        normalized_key = key.rstrip("/")
        return (
            f"s3://{bucket}/{normalized_key}"
            if separator and normalized_key
            else f"s3://{bucket}"
        )
    return str(Path(normalized))


def _resolve_storage_options(path: str) -> dict[str, str] | None:
    """Resolve Polars storage_options from URL scheme and scoped credentials."""
    if not path.startswith("s3://"):
        return None
    credentials = current_s3_credentials()
    if credentials is None:
        return {}
    options: dict[str, str] = {
        "aws_access_key_id": credentials.access_key,
        "aws_secret_access_key": credentials.secret_key,
    }
    if credentials.session_token is not None:
        options["aws_session_token"] = credentials.session_token
    if credentials.region is not None:
        options["aws_region"] = credentials.region
    return options


def _validate_time_range(value: object) -> tuple[str, str] | None:
    if value is None:
        return None
    if isinstance(value, str) or not isinstance(value, Sequence) or len(value) != 2:
        raise TypeError("time_range must be a (start, end) tuple of ISO-8601 strings")
    start, end = value
    if not isinstance(start, str) or not start:
        raise ValueError("time_range start must be a non-empty string")
    if not isinstance(end, str) or not end:
        raise ValueError("time_range end must be a non-empty string")
    return (start, end)


@dataclass(frozen=True, slots=True)
class CloudDatasetManifest:
    """Metadata for a partitioned dataset."""

    format: str
    path: str
    schema: dict[str, Any]
    time_range: tuple[str, str] | None = None
    resolution: str = ""
    row_count: int = 0
    bytes: int = 0

    def __post_init__(self) -> None:
        if self.format != _MANIFEST_FORMAT:
            raise ValueError(
                f"manifest format must be {_MANIFEST_FORMAT!r}, got {self.format!r}"
            )
        if not isinstance(self.path, str) or not self.path.strip():
            raise ValueError("path must be a non-empty string")
        if not isinstance(self.schema, dict):
            raise TypeError("schema must be a mapping")

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "format": self.format,
            "path": self.path,
            "schema": self.schema,
        }
        if self.time_range is not None:
            result["time_range"] = {
                "start": self.time_range[0],
                "end": self.time_range[1],
            }
        if self.resolution:
            result["resolution"] = self.resolution
        if self.row_count:
            result["row_count"] = self.row_count
        if self.bytes:
            result["bytes"] = self.bytes
        return result

    def to_json_bytes(self) -> bytes:
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")

    @classmethod
    def from_dict(cls, value: object) -> CloudDatasetManifest:
        if not isinstance(value, dict):
            raise ValueError("manifest must be a JSON object")
        required = {"format", "path", "schema"}
        missing = sorted(required - value.keys())
        if missing:
            raise ValueError(f"manifest is missing: {', '.join(missing)}")
        time_range_raw = value.get("time_range")
        time_range: tuple[str, str] | None = None
        if time_range_raw is not None:
            if not isinstance(time_range_raw, dict):
                raise TypeError("time_range must be an object with start/end")
            time_range = (time_range_raw["start"], time_range_raw["end"])
        return cls(
            format=value["format"],
            path=value["path"],
            schema=value["schema"],
            time_range=time_range,
            resolution=value.get("resolution", ""),
            row_count=value.get("row_count", 0),
            bytes=value.get("bytes", 0),
        )

    @classmethod
    def from_bytes(cls, raw: bytes) -> CloudDatasetManifest:
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("manifest is not valid UTF-8 JSON") from exc
        return cls.from_dict(parsed)


@dataclass(frozen=True)
class CloudDatasetSourceConfig(TSFNConfig):
    """Configuration for scanning a Parquet dataset.

    ``path`` is a glob pattern pointing to Parquet files (local or S3).
    Polars auto-detects hive partition columns from ``key=value`` directories:

    - ``s3://bucket/data/**/*.parquet`` - no partitioning
    - ``s3://bucket/data/year=*/month=*/*.parquet`` - hive-partitioned
    - ``/local/path/year=*/month=*/*.parquet`` - works locally too

    ``time_range`` is an optional ``(start, end)`` pair of ISO-8601 timestamp
    strings.  When provided the adapter pushes a filter down to the Parquet
    reader so that irrelevant row groups and partition directories are skipped.
    """

    path: str
    output_signature: FrameSignature = _UNRESOLVED_SIGNATURE
    schema: Mapping[str, object] | None = None
    time_range: tuple[str, str] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _normalize_path(self.path))
        signature = _resolve_output_signature(self.output_signature, self.schema)
        _validate_output_signature(signature)
        object.__setattr__(self, "output_signature", signature)
        object.__setattr__(self, "schema", None)
        object.__setattr__(self, "time_range", _validate_time_range(self.time_range))


class CloudDatasetSource(TSFN):
    """Scan a Parquet dataset as a single logical table.

    Uses Polars' native ``scan_parquet`` with ``hive_partitioning=True``.
    Any ``key=value`` directories in the path are automatically detected
    as partition columns.

    No data is downloaded until ``collect()``.  Projection pushdown ensures
    only requested columns are fetched, and optional time-range filtering
    prunes entire partition directories.
    """

    VERSION = "0.1.0"
    CONFIG_CLS = CloudDatasetSourceConfig

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        return FrameSignature.empty(), self.parameters.output_signature

    def apply(self, lf: pl.LazyFrame | None = None) -> pl.LazyFrame:
        params = self.parameters

        storage_options = _resolve_storage_options(params.path)

        lazy_table = pl.scan_parquet(
            params.path,
            hive_partitioning=True,
            use_statistics=True,
            storage_options=storage_options,
        )

        if params.time_range is not None:
            time_col = params.output_signature.time
            if time_col is not None:
                col_name = time_col.column
                start, end = params.time_range
                time_dtype = _time_axis_physical_dtype(time_col)
                lazy_table = lazy_table.filter(
                    (pl.col(col_name).cast(time_dtype) >= pl.lit(start).cast(time_dtype))
                    & (pl.col(col_name).cast(time_dtype) < pl.lit(end).cast(time_dtype))
                )

        return _project_declared_columns(lazy_table, params.output_signature)


__all__ = [
    "CloudDatasetManifest",
    "CloudDatasetSource",
    "CloudDatasetSourceConfig",
]
