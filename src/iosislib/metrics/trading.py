"""Trading evaluation metrics over equity curves and return series."""

from __future__ import annotations

import polars as pl

from iosislib.metrics.base import MetricTSFN


class MaxDrawdown(MetricTSFN):
    """Largest peak-to-trough relative decline of an equity column.

    Drawdown per row is measured against the running peak; rows whose running
    peak is not positive contribute no drawdown, so a monotonically
    non-declining curve yields ``0.0``.
    """

    VERSION = "1.0.0"
    METRIC_NAME = "max_drawdown"
    INPUT_COLUMNS = ("equity",)
    MIN_ROWS = 2

    def metric_value(self, frame: pl.DataFrame) -> float:
        equity = pl.col("equity")
        peak = equity.cum_max()
        drawdown = pl.when(peak > 1e-12).then((peak - equity) / peak).otherwise(0.0)
        return frame.select(drawdown.max()).item()


class Sharpe(MetricTSFN):
    """Mean divided by sample standard deviation of a returns column.

    Not annualized: multiply externally when a scaling factor is required.
    Requires a non-zero standard deviation.
    """

    VERSION = "1.0.0"
    METRIC_NAME = "sharpe"
    INPUT_COLUMNS = ("returns",)
    MIN_ROWS = 2

    def metric_value(self, frame: pl.DataFrame) -> float:
        if frame.get_column("returns").n_unique() == 1:
            raise ValueError(
                "Sharpe requires a non-zero standard deviation of returns"
            )
        mean, std = frame.select(
            pl.col("returns").mean().alias("mean"),
            pl.col("returns").std(ddof=1).alias("std"),
        ).row(0)
        if std is None:
            raise ValueError(
                "Sharpe requires a non-zero standard deviation of returns"
            )
        return float(mean) / float(std)


class TotalReturn(MetricTSFN):
    """Relative change from the first to the last value of an equity column.

    ``(last / first) - 1``; the first value must be non-zero.
    """

    VERSION = "1.0.0"
    METRIC_NAME = "total_return"
    INPUT_COLUMNS = ("equity",)
    MIN_ROWS = 2

    def metric_value(self, frame: pl.DataFrame) -> float:
        first, last = frame.select(
            pl.col("equity").first().alias("first"),
            pl.col("equity").last().alias("last"),
        ).row(0)
        if first == 0.0:
            raise ValueError(
                "TotalReturn requires a non-zero first equity value"
            )
        return float(last) / float(first) - 1.0


__all__ = ["MaxDrawdown", "Sharpe", "TotalReturn"]
