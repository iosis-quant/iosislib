from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from numbers import Real
from typing import Any

import polars as pl

from src.classes import FrameSignature, TSFN, TSFNConfig, TimeAxis


YFINANCE_INTERVALS = frozenset(
    {
        "1m",
        "2m",
        "5m",
        "15m",
        "30m",
        "60m",
        "90m",
        "1h",
        "1d",
        "5d",
        "1wk",
        "1mo",
        "3mo",
    }
)
OHLCV_FIELDS = ("Open", "High", "Low", "Close", "Volume")


@dataclass(frozen=True)
class YFinanceOHLCVConfig(TSFNConfig):
    ticker: str
    date_range: tuple[str, str] | list[str]
    interval: str = "1d"
    auto_adjust: bool = False
    timeout_seconds: float = 10.0
    timestamp_column: str = "timestamp"
    open_column: str = "open"
    high_column: str = "high"
    low_column: str = "low"
    close_column: str = "close"
    volume_column: str = "volume"

    def __post_init__(self) -> None:
        if not isinstance(self.ticker, str) or not self.ticker.strip():
            raise ValueError("ticker must be a non-empty string")
        ticker = self.ticker.strip()
        if any(separator in ticker for separator in (",", " ")):
            raise ValueError("ticker must identify a single ticker")
        object.__setattr__(self, "ticker", ticker)

        if isinstance(self.date_range, str) or not isinstance(self.date_range, Sequence):
            raise TypeError("date_range must be a two-item sequence of date strings")
        date_range = tuple(self.date_range)
        if len(date_range) != 2:
            raise ValueError("date_range must contain exactly start and end dates")
        if not all(isinstance(value, str) for value in date_range):
            raise TypeError("date_range must contain only strings")
        if any(not value for value in date_range):
            raise ValueError("date_range values must be non-empty")

        start, end = date_range
        if _parse_iso_datetime(start) > _parse_iso_datetime(end):
            raise ValueError("date_range start must be less than or equal to end")
        object.__setattr__(self, "date_range", date_range)

        if self.interval not in YFINANCE_INTERVALS:
            raise ValueError(f"interval must be one of {sorted(YFINANCE_INTERVALS)}")
        if not isinstance(self.auto_adjust, bool):
            raise TypeError("auto_adjust must be a bool")
        if isinstance(self.timeout_seconds, bool) or not isinstance(self.timeout_seconds, Real):
            raise TypeError("timeout_seconds must be a number")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        output_columns = (
            self.timestamp_column,
            self.open_column,
            self.high_column,
            self.low_column,
            self.close_column,
            self.volume_column,
        )
        if any(not isinstance(column, str) for column in output_columns):
            raise TypeError("output column names must be strings")
        if any(not column for column in output_columns):
            raise ValueError("output column names must be non-empty")

        duplicate_columns = sorted(
            {column for column in output_columns if output_columns.count(column) > 1}
        )
        if duplicate_columns:
            raise ValueError(f"Duplicate output column names: {duplicate_columns}")


class YFinanceOHLCV(TSFN):
    VERSION = "0.1.0"
    CONFIG_CLS = YFinanceOHLCVConfig

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        params = self.parameters
        output = FrameSignature(
            time=TimeAxis(column=params.timestamp_column),
            columns=(
                (params.open_column, pl.Float64),
                (params.high_column, pl.Float64),
                (params.low_column, pl.Float64),
                (params.close_column, pl.Float64),
                (params.volume_column, pl.Int64),
            ),
        )
        return FrameSignature.empty(), output

    def apply(self) -> pl.LazyFrame:
        frame = _download_ohlcv(self.parameters)
        return _ohlcv_frame_to_lazyframe(frame, config=self.parameters)


def _download_ohlcv(config: YFinanceOHLCVConfig) -> Any:
    try:
        import inspect

        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError(
            "yfinance is required for YFinanceOHLCV; install it to fetch OHLCV data"
        ) from exc

    kwargs: dict[str, Any] = {
        "tickers": config.ticker,
        "start": config.date_range[0],
        "end": config.date_range[1],
        "interval": config.interval,
        "auto_adjust": config.auto_adjust,
        "actions": False,
        "threads": False,
        "progress": False,
        "timeout": config.timeout_seconds,
    }
    if "multi_level_index" in inspect.signature(yf.download).parameters:
        kwargs["multi_level_index"] = False

    return yf.download(**kwargs)


def _ohlcv_frame_to_lazyframe(
    frame: Any,
    *,
    config: YFinanceOHLCVConfig,
) -> pl.LazyFrame:
    pd = _import_pandas()
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("yfinance OHLCV payload must be a pandas DataFrame")

    schema = _ohlcv_schema(config)
    if frame.empty:
        return pl.DataFrame([], schema=schema).lazy()
    if isinstance(frame.index, pd.MultiIndex):
        raise ValueError("yfinance OHLCV dataframe must have a single time index")

    columns = _select_ohlcv_columns(frame, ticker=config.ticker)
    normalized = frame.loc[:, list(columns.values())].copy()
    normalized.columns = [
        config.open_column,
        config.high_column,
        config.low_column,
        config.close_column,
        config.volume_column,
    ]
    normalized.insert(0, config.timestamp_column, frame.index)
    normalized[config.timestamp_column] = _normalize_timestamp_series(
        normalized[config.timestamp_column]
    )

    return (
        pl.DataFrame(normalized.to_dict("list"))
        .lazy()
        .select(
            pl.col(config.timestamp_column).cast(pl.Datetime(time_unit="us")),
            pl.col(config.open_column).cast(pl.Float64, strict=False),
            pl.col(config.high_column).cast(pl.Float64, strict=False),
            pl.col(config.low_column).cast(pl.Float64, strict=False),
            pl.col(config.close_column).cast(pl.Float64, strict=False),
            pl.col(config.volume_column).cast(pl.Int64, strict=False),
        )
        .sort(config.timestamp_column)
    )


def _select_ohlcv_columns(frame: Any, *, ticker: str) -> dict[str, Any]:
    selected = {}
    for field_name in OHLCV_FIELDS:
        selected[field_name] = _find_column(frame.columns, field_name, ticker=ticker)
    return selected


def _find_column(columns: Any, field_name: str, *, ticker: str) -> Any:
    field_key = _normalize_column_label(field_name)
    ticker_key = _normalize_column_label(ticker)
    matches = []
    for column in columns:
        labels = column if isinstance(column, tuple) else (column,)
        normalized_labels = tuple(_normalize_column_label(label) for label in labels)
        if field_key not in normalized_labels:
            continue
        if len(normalized_labels) > 1 and ticker_key not in normalized_labels:
            continue
        matches.append(column)

    if not matches:
        raise ValueError(f"yfinance OHLCV dataframe missing required column '{field_name}'")
    if len(matches) > 1:
        raise ValueError(f"yfinance OHLCV dataframe has ambiguous column '{field_name}'")
    return matches[0]


def _normalize_timestamp_series(series: Any) -> Any:
    pd = _import_pandas()
    timestamps = pd.to_datetime(series, utc=True, errors="raise")
    return timestamps.dt.tz_convert("UTC").dt.tz_localize(None)


def _ohlcv_schema(config: YFinanceOHLCVConfig) -> dict[str, pl.DataType]:
    return {
        config.timestamp_column: pl.Datetime,
        config.open_column: pl.Float64,
        config.high_column: pl.Float64,
        config.low_column: pl.Float64,
        config.close_column: pl.Float64,
        config.volume_column: pl.Int64,
    }


def _parse_iso_datetime(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(
            "date_range values must be ISO date or datetime strings"
        ) from exc
    if parsed.tzinfo is not None:
        return parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def _normalize_column_label(value: Any) -> str:
    return str(value).strip().lower().replace(" ", "")


def _import_pandas() -> Any:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "pandas is required to normalize yfinance OHLCV data"
        ) from exc
    return pd
