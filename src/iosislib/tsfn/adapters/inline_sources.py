from __future__ import annotations

import io
from dataclasses import dataclass

import polars as pl

from iosislib.core.tsfn import (
    FrameSignature,
    TSFN,
    TSFNConfig,
    _frame_physical_schema,
)
from iosislib.tsfn.adapters.local_sources import (
    _validate_output_signature,
    _project_declared_columns,
)


def _frame_to_bytes(frame: pl.DataFrame) -> bytes:
    """Serialize a DataFrame to Arrow IPC bytes for deterministic identity."""
    buffer = io.BytesIO()
    frame.write_ipc(buffer)
    return buffer.getvalue()


def _frame_from_bytes(data: bytes) -> pl.DataFrame:
    """Deserialize a DataFrame from Arrow IPC bytes."""
    return pl.read_ipc(io.BytesIO(data))


@dataclass(frozen=True)
class DataFrameSourceConfig(TSFNConfig):
    """Configuration for an in-memory DataFrame source.

    The frame is captured at construction time and serialized to Arrow IPC
    bytes for deterministic identity. No content hash is needed because the
    frame is not re-read from disk.
    """

    frame_bytes: bytes
    output_signature: FrameSignature

    def __post_init__(self) -> None:
        if not isinstance(self.frame_bytes, bytes):
            raise TypeError("frame_bytes must be bytes (use DataFrameSourceConfig.from_frame() to create from a DataFrame)")
        _validate_output_signature(self.output_signature)
        frame = _frame_from_bytes(self.frame_bytes)
        expected = _frame_physical_schema(self.output_signature)
        actual = {col: dtype for col, dtype in zip(frame.columns, frame.dtypes)}
        mismatched = [
            name
            for name in expected
            if name not in actual or not _dtypes_compatible(actual[name], expected[name])
        ]
        if mismatched:
            raise ValueError(
                f"Frame columns/dtypes do not match output_signature. "
                f"Mismatched: {mismatched}. "
                f"Expected: {expected}. "
                f"Got: {actual}"
            )

    @classmethod
    def from_frame(
        cls, frame: pl.DataFrame, output_signature: FrameSignature
    ) -> DataFrameSourceConfig:
        """Create a config from a Polars DataFrame."""
        if not isinstance(frame, pl.DataFrame):
            raise TypeError("frame must be a polars DataFrame")
        return cls(frame_bytes=_frame_to_bytes(frame), output_signature=output_signature)

    @property
    def frame(self) -> pl.DataFrame:
        """Reconstruct the DataFrame from stored bytes."""
        return _frame_from_bytes(self.frame_bytes)


class DataFrameSource(TSFN):
    """A source that wraps an in-memory Polars DataFrame.

    Unlike CSVSource and ParquetSource, no content hash is needed because
    the frame is captured at construction time and never re-read.
    """

    VERSION = "0.1.0"
    CONFIG_CLS = DataFrameSourceConfig

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        return FrameSignature.empty(), self.parameters.output_signature

    def apply(self) -> pl.LazyFrame:
        frame = self.parameters.frame.lazy()
        return _project_declared_columns(frame, self.parameters.output_signature)


def _dtypes_compatible(actual: pl.DataType, expected: pl.DataType) -> bool:
    return actual == expected
