"""Scan a partitioned Parquet dataset as a single logical table.

Uses Polars' native ``scan_parquet`` with ``hive_partitioning=True`` to
treat a directory of Parquet files as one unified table.  Any ``key=value``
directories in the path are automatically detected as partition columns -
no explicit declaration needed.

Just pass a glob path and Polars does the rest:

- ``s3://bucket/data/**/*.parquet`` - flat files, no partitioning
- ``s3://bucket/data/year=*/month=*/*.parquet`` - hive-partitioned
- ``/local/path/**/*.parquet`` - works locally too

Optional time-range filtering (inclusive on both bounds) pushes predicates
down to the Parquet reader so entire partition directories and row groups
are skipped.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import polars as pl

from iosislib.core.tsfn import (
    FrameSignature,
    TimeAxis,
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
_BARE_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


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


def _parse_time_bound(value: str, *, label: str) -> datetime:
    """Parse an inclusive time_range bound, accepting bare dates ("2021-08-01").

    Timezone-aware bounds are normalized to naive UTC so they compare by
    instant against time columns in any timezone.
    """
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(
            f"time_range {label} must be ISO-8601 ({value!r} is not parseable)"
        ) from exc
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def _time_range_predicates(
    time_col: TimeAxis,
    start_dt: datetime,
    end_dt: datetime,
    end: str,
) -> tuple[pl.Expr, pl.Expr]:
    """Build inclusive ``start``/``end`` predicates for one time axis.

    Both bounds are inclusive: a bare-date end (``"2026-01-03"``) covers that
    whole day, a timestamp end includes rows exactly at the bound. Bounds are
    naive UTC after parsing; timezone-aware columns are compared in naive UTC.
    """
    time_dtype = _time_axis_physical_dtype(time_col)
    compare_dtype = time_dtype
    column = pl.col(time_col.column)
    if isinstance(time_dtype, pl.Datetime) and time_dtype.time_zone:
        compare_dtype = pl.Datetime(time_dtype.time_unit)
        column = column.dt.convert_time_zone("UTC").dt.replace_time_zone(None)
    column = column.cast(compare_dtype)
    start_expr = column >= pl.lit(start_dt).cast(compare_dtype)
    if _BARE_DATE.fullmatch(end):
        day_end = end_dt + timedelta(days=1)
        if start_dt >= day_end:
            raise ValueError("time_range start must not be after end")
        return start_expr, column < pl.lit(day_end).cast(compare_dtype)
    if start_dt > end_dt:
        raise ValueError("time_range start must not be after end")
    return start_expr, column <= pl.lit(end_dt).cast(compare_dtype)


@dataclass(frozen=True, slots=True)
class DatasetManifest:
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
    def from_dict(cls, value: object) -> DatasetManifest:
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
    def from_bytes(cls, raw: bytes) -> DatasetManifest:
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("manifest is not valid UTF-8 JSON") from exc
        return cls.from_dict(parsed)


@dataclass(frozen=True)
class DatasetSourceConfig(TSFNConfig):
    """Configuration for scanning a Parquet dataset.

    ``path`` is a glob pattern pointing to Parquet files (local or S3).
    Polars auto-detects hive partition columns from ``key=value`` directories:

    - ``s3://bucket/data/**/*.parquet`` - no partitioning
    - ``s3://bucket/data/year=*/month=*/*.parquet`` - hive-partitioned
    - ``/local/path/year=*/month=*/*.parquet`` - works locally too

    ``time_range`` is an optional ``(start, end)`` pair of ISO-8601 strings —
    bare dates (``"2021-08-01"``) or timestamps (``"2021-08-01T12:00:00"``,
    ``"2021-08-01T12:00:00Z"``). Both bounds are inclusive: a bare-date end
    covers that whole day, a timestamp end includes rows exactly at the bound.
    Timezone-aware bounds are compared as instants against timezone-aware
    columns. When provided, the adapter pushes a filter down to the Parquet
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


class DatasetSource(TSFN):
    """Scan a Parquet dataset as a single logical table.

    Uses Polars' native ``scan_parquet`` with ``hive_partitioning=True``.
    Any ``key=value`` directories in the path are automatically detected
    as partition columns.

    No data is downloaded until ``collect()``.  Projection pushdown ensures
    only requested columns are fetched, and optional time-range filtering
    prunes entire partition directories.

    Declared columns use the shared source coercions: ``Datetime`` unit casts
    (timezones must still match), ``String`` timestamp parsing, and fixed-width
    ``List`` to ``Array`` conversion. All other mismatches fail loudly.
    ``time_range`` bounds are inclusive on both ends and accept bare dates
    (``"2021-08-01"``) and ISO-8601 timestamps.
    """

    VERSION = "0.3.0"
    CONFIG_CLS = DatasetSourceConfig

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
                start, end = params.time_range
                start_dt = _parse_time_bound(start, label="start")
                end_dt = _parse_time_bound(end, label="end")
                start_expr, end_expr = _time_range_predicates(
                    time_col, start_dt, end_dt, end
                )
                lazy_table = lazy_table.filter(start_expr & end_expr)

        return _project_declared_columns(lazy_table, params.output_signature)


__all__ = [
    "DatasetManifest",
    "DatasetSource",
    "DatasetSourceConfig",
]
