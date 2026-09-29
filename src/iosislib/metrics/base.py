"""Reduce-style metric TSFNs: one time-indexed frame in, one metric row out."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from math import isfinite
from typing import ClassVar

import polars as pl

from iosislib.core.tsfn import BatchTSFN, FrameSignature, TSFNConfig, TimeAxis


def _validate_column_name(value: object, *, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value:
        raise ValueError(f"{field_name} must be non-empty")


@dataclass(frozen=True)
class MetricConfig(TSFNConfig):
    """Shared parameters for every metric TSFN."""

    timestamp_column: str = "timestamp"
    drop_nonfinite: bool = False

    def __post_init__(self) -> None:
        _validate_column_name(self.timestamp_column, field_name="timestamp_column")
        if not isinstance(self.drop_nonfinite, bool):
            raise TypeError("drop_nonfinite must be a boolean")


def _nonfinite(column_name: str) -> pl.Expr:
    return ~pl.col(column_name).is_finite().fill_null(False)


class MetricTSFN(BatchTSFN[MetricConfig], abc.ABC):
    """Reduce a time-indexed frame to exactly one row carrying one metric.

    Subclasses declare ``METRIC_NAME``, ``INPUT_COLUMNS``, and optionally
    ``MIN_ROWS``, and implement :meth:`metric_value`. Nulls follow each input's
    null policy (default ERROR). NaN/inf inputs raise ``ValueError`` unless
    ``drop_nonfinite`` drops rows where any input is not finite. The frame is
    sorted by time first, keeping input order among equal timestamps, and the
    output row is stamped with the last input timestamp regardless of how many
    rows the reduction filters out.
    """

    CONFIG_CLS = MetricConfig
    METRIC_NAME: ClassVar[str]
    INPUT_COLUMNS: ClassVar[tuple[str, ...]]
    MIN_ROWS: ClassVar[int] = 1

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        metric_name = self.METRIC_NAME
        input_columns = self.INPUT_COLUMNS
        if not isinstance(metric_name, str):
            raise TypeError(
                f"{type(self).__name__}.METRIC_NAME must be a string"
            )
        if not metric_name:
            raise ValueError(
                f"{type(self).__name__}.METRIC_NAME must be non-empty"
            )
        if not isinstance(input_columns, tuple):
            raise TypeError(
                f"{type(self).__name__}.INPUT_COLUMNS must be a tuple of column names"
            )
        if not all(isinstance(name, str) for name in input_columns):
            raise TypeError(
                f"{type(self).__name__}.INPUT_COLUMNS must contain only strings"
            )
        if not input_columns or not all(input_columns):
            raise ValueError(
                f"{type(self).__name__}.INPUT_COLUMNS must be a non-empty tuple "
                "of non-empty column names"
            )
        duplicates = sorted(
            {name for name in input_columns if input_columns.count(name) > 1}
        )
        if duplicates:
            raise ValueError(
                f"{type(self).__name__}.INPUT_COLUMNS contains duplicates: {duplicates}"
            )
        min_rows = self.MIN_ROWS
        if isinstance(min_rows, bool) or not isinstance(min_rows, int):
            raise TypeError(f"{type(self).__name__}.MIN_ROWS must be an integer")
        if min_rows < 1:
            raise ValueError(
                f"{type(self).__name__}.MIN_ROWS must be a positive integer"
            )
        time_column = self.parameters.timestamp_column
        if metric_name == time_column:
            raise ValueError(
                f"{type(self).__name__}: metric column {metric_name!r} must differ "
                f"from the timestamp column {time_column!r}"
            )
        if time_column in input_columns:
            raise ValueError(
                f"{type(self).__name__}: input column {time_column!r} must differ "
                "from the timestamp column"
            )
        axis = TimeAxis(column=time_column)
        return (
            FrameSignature(
                time=axis,
                columns=tuple((name, pl.Float64) for name in input_columns),
            ),
            FrameSignature(time=axis, columns=((metric_name, pl.Float64),)),
        )

    def batch(self, frame: pl.DataFrame) -> pl.DataFrame:
        params = self.parameters
        time_column = params.timestamp_column
        nonfinite = pl.any_horizontal(
            *(_nonfinite(name) for name in self.INPUT_COLUMNS)
        )
        ordered = frame.sort(time_column, maintain_order=True)
        stamp = ordered.tail(1)
        nonfinite_rows = ordered.select(nonfinite.alias("__nonfinite")).to_series()
        if nonfinite_rows.any():
            if not params.drop_nonfinite:
                counts = ordered.select(
                    tuple(
                        _nonfinite(name).sum().alias(name)
                        for name in self.INPUT_COLUMNS
                    )
                ).row(0, named=True)
                detail = ", ".join(
                    f"{name}: {count}" for name, count in counts.items() if count
                )
                raise ValueError(
                    f"{type(self).__name__} found non-finite input value(s) "
                    f"({detail}); set drop_nonfinite: true to drop rows whose "
                    "inputs are not finite"
                )
            ordered = ordered.filter(~nonfinite_rows)
        if ordered.height < self.MIN_ROWS:
            raise ValueError(
                f"{type(self).__name__} requires at least {self.MIN_ROWS} row(s), "
                f"got {ordered.height}"
            )
        value = self.metric_value(ordered)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(
                f"{type(self).__name__}.metric_value must return a float, "
                f"got {type(value).__name__}"
            )
        result = float(value)
        if not isfinite(result):
            raise ValueError(
                f"{type(self).__name__} produced a non-finite value ({result})"
            )
        return stamp.select(
            pl.col(time_column),
            pl.lit(result, dtype=pl.Float64).alias(self.METRIC_NAME),
        )

    @abc.abstractmethod
    def metric_value(self, frame: pl.DataFrame) -> float:
        """Return the metric computed over a validated, time-sorted frame."""


__all__ = ["MetricConfig", "MetricTSFN"]
