"""Concrete post-hoc metric extractors.

Both extractors assume the input frame is already time-sorted and validate
their required columns strictly: a missing column, a non-numeric column, nulls,
non-finite values, or insufficient rows all raise ``ValueError``/``TypeError``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from iosislib.metrics.extractor import MetricExtractor


def _validate_column_name(name: str, label: str) -> None:
    if not isinstance(name, str) or not name:
        raise ValueError(f"{label} must be a non-empty string")


def _numeric_column(frame: pl.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        raise ValueError(f"frame is missing required column {column!r}")
    series = frame.get_column(column)
    if not series.dtype.is_numeric():
        raise TypeError(f"column {column!r} must be numeric, got {series.dtype}")
    if series.is_null().any():
        raise ValueError(f"column {column!r} contains null values")
    array = series.cast(pl.Float64).to_numpy()
    if not np.isfinite(array).all():
        raise ValueError(f"column {column!r} must contain only finite values")
    return array


@dataclass(frozen=True)
class MeanSquaredError(MetricExtractor):
    """Return the mean squared error between prediction and target columns."""

    VERSION = "1.0.0"

    prediction_column: str = "prediction"
    target_column: str = "target"

    def __post_init__(self) -> None:
        _validate_column_name(self.prediction_column, "prediction_column")
        _validate_column_name(self.target_column, "target_column")
        if self.prediction_column == self.target_column:
            raise ValueError("prediction_column and target_column must differ")

    def required_columns(self) -> tuple[str, ...]:
        return (self.prediction_column, self.target_column)

    def metric_names(self) -> tuple[str, ...]:
        return ("mse",)

    def extract(self, frame: pl.DataFrame) -> dict[str, float]:
        prediction = _numeric_column(frame, self.prediction_column)
        target = _numeric_column(frame, self.target_column)
        if frame.height < 1:
            raise ValueError("MeanSquaredError requires at least one row")
        if prediction.shape != target.shape:
            raise ValueError("prediction and target must have equal row counts")
        return {"mse": float(np.mean((prediction - target) ** 2))}


@dataclass(frozen=True)
class MaxDrawdown(MetricExtractor):
    """Return the largest peak-to-trough decline in an equity column.

    Drawdown for each row is measured against the running peak; rows whose
    running peak is not positive contribute no drawdown. A monotonically
    non-declining series therefore yields ``0.0``.
    """

    VERSION = "1.0.0"

    equity_column: str = "equity"

    def __post_init__(self) -> None:
        _validate_column_name(self.equity_column, "equity_column")

    def required_columns(self) -> tuple[str, ...]:
        return (self.equity_column,)

    def metric_names(self) -> tuple[str, ...]:
        return ("max_drawdown",)

    def extract(self, frame: pl.DataFrame) -> dict[str, float]:
        equity = _numeric_column(frame, self.equity_column)
        if frame.height < 2:
            raise ValueError("MaxDrawdown requires at least two rows")
        peak = np.maximum.accumulate(equity)
        with np.errstate(divide="ignore", invalid="ignore"):
            drawdown = np.where(peak > 0.0, (peak - equity) / peak, 0.0)
        return {"max_drawdown": float(np.max(drawdown))}


__all__ = ["MaxDrawdown", "MeanSquaredError"]
