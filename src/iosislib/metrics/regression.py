"""Regression-quality metrics: mean squared and mean absolute error."""

from __future__ import annotations

import polars as pl

from iosislib.metrics.base import MetricTSFN


class Mse(MetricTSFN):
    """Mean squared error between prediction and target columns."""

    VERSION = "1.0.0"
    METRIC_NAME = "mse"
    INPUT_COLUMNS = ("prediction", "target")

    def metric_value(self, frame: pl.DataFrame) -> float:
        return frame.select(
            ((pl.col("prediction") - pl.col("target")) ** 2).mean()
        ).item()


class Mae(MetricTSFN):
    """Mean absolute error between prediction and target columns."""

    VERSION = "1.0.0"
    METRIC_NAME = "mae"
    INPUT_COLUMNS = ("prediction", "target")

    def metric_value(self, frame: pl.DataFrame) -> float:
        return frame.select(
            (pl.col("prediction") - pl.col("target")).abs().mean()
        ).item()


__all__ = ["Mae", "Mse"]
