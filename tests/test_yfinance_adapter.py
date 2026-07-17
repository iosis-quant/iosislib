from __future__ import annotations

from datetime import datetime

import pandas as pd
import polars as pl
import pytest

from src.core.graph import Graph
from src.core.node import Node
from src.tsfn.adapters import YFinanceOHLCV, YFinanceOHLCVConfig
from src.tsfn.adapters import yfinance


def make_config(**overrides) -> YFinanceOHLCVConfig:
    params = {
        "ticker": "AAPL",
        "date_range": ("2026-01-01", "2026-01-03"),
        "interval": "1d",
    }
    params.update(overrides)
    return YFinanceOHLCVConfig(**params)


def test_yfinance_ohlcv_config_validation() -> None:
    with pytest.raises(ValueError, match="ticker must be"):
        make_config(ticker="")

    with pytest.raises(ValueError, match="single ticker"):
        make_config(ticker="AAPL MSFT")

    with pytest.raises(TypeError, match="two-item sequence"):
        make_config(date_range="2026-01-01")

    with pytest.raises(ValueError, match="exactly start and end"):
        make_config(date_range=("2026-01-01",))

    with pytest.raises(TypeError, match="only strings"):
        make_config(date_range=("2026-01-01", 20260102))

    with pytest.raises(ValueError, match="ISO date"):
        make_config(date_range=("not-a-date", "2026-01-03"))

    with pytest.raises(ValueError, match="less than or equal"):
        make_config(date_range=("2026-01-03", "2026-01-01"))

    with pytest.raises(ValueError, match="interval must be one of"):
        make_config(interval="7m")

    with pytest.raises(TypeError, match="auto_adjust must be a bool"):
        make_config(auto_adjust="no")

    with pytest.raises(TypeError, match="timeout_seconds must be a number"):
        make_config(timeout_seconds=True)

    with pytest.raises(ValueError, match="timeout_seconds must be positive"):
        make_config(timeout_seconds=0)

    with pytest.raises(ValueError, match="Duplicate output column names"):
        make_config(open_column="timestamp")


def test_yfinance_ohlcv_download_uses_expected_yfinance_parameters(monkeypatch) -> None:
    calls = []

    class FakeYFinance:
        @staticmethod
        def download(**kwargs):
            calls.append(kwargs)
            return pd.DataFrame()

    monkeypatch.setitem(__import__("sys").modules, "yfinance", FakeYFinance)
    config = make_config(interval="1h", auto_adjust=True, timeout_seconds=3.0)

    frame = yfinance._download_ohlcv(config)

    assert frame.empty
    assert calls == [
        {
            "tickers": "AAPL",
            "start": "2026-01-01",
            "end": "2026-01-03",
            "interval": "1h",
            "auto_adjust": True,
            "actions": False,
            "threads": False,
            "progress": False,
            "timeout": 3.0,
        }
    ]


def test_yfinance_ohlcv_dataframe_becomes_sorted_lazyframe() -> None:
    frame = pd.DataFrame(
        {
            "Open": [20, 10],
            "High": [22, 12],
            "Low": [19, 9],
            "Close": [21, 11],
            "Volume": [2000, 1000],
        },
        index=pd.to_datetime(["2026-01-02", "2026-01-01"]),
    )

    result = yfinance._ohlcv_frame_to_lazyframe(frame, config=make_config()).collect()

    assert result.schema == {
        "timestamp": pl.Datetime(time_unit="us"),
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close": pl.Float64,
        "volume": pl.Int64,
    }
    assert result["timestamp"].to_list() == [
        datetime(2026, 1, 1),
        datetime(2026, 1, 2),
    ]
    assert result["open"].to_list() == [10.0, 20.0]
    assert result["high"].to_list() == [12.0, 22.0]
    assert result["low"].to_list() == [9.0, 19.0]
    assert result["close"].to_list() == [11.0, 21.0]
    assert result["volume"].to_list() == [1000, 2000]


def test_yfinance_ohlcv_dataframe_accepts_multiindex_columns() -> None:
    columns = pd.MultiIndex.from_tuples(
        [
            ("Open", "AAPL"),
            ("High", "AAPL"),
            ("Low", "AAPL"),
            ("Close", "AAPL"),
            ("Volume", "AAPL"),
            ("Adj Close", "AAPL"),
        ]
    )
    frame = pd.DataFrame(
        [[10, 12, 9, 11, 1000, 10.5]],
        columns=columns,
        index=pd.to_datetime(["2026-01-01"]),
    )

    result = yfinance._ohlcv_frame_to_lazyframe(frame, config=make_config()).collect()

    assert result.columns == ["timestamp", "open", "high", "low", "close", "volume"]
    assert result.row(0) == (datetime(2026, 1, 1), 10.0, 12.0, 9.0, 11.0, 1000)


def test_yfinance_ohlcv_dataframe_normalizes_timezone_aware_index() -> None:
    frame = pd.DataFrame(
        {
            "Open": [10],
            "High": [12],
            "Low": [9],
            "Close": [11],
            "Volume": [1000],
        },
        index=pd.DatetimeIndex(["2026-01-01 09:30"], tz="America/New_York"),
    )

    result = yfinance._ohlcv_frame_to_lazyframe(frame, config=make_config()).collect()

    assert result["timestamp"].to_list() == [datetime(2026, 1, 1, 14, 30)]


def test_yfinance_ohlcv_dataframe_validation_is_explicit() -> None:
    with pytest.raises(ValueError, match="must be a pandas DataFrame"):
        yfinance._ohlcv_frame_to_lazyframe([], config=make_config())

    missing_close = pd.DataFrame(
        {
            "Open": [10],
            "High": [12],
            "Low": [9],
            "Volume": [1000],
        },
        index=pd.to_datetime(["2026-01-01"]),
    )
    with pytest.raises(ValueError, match="missing required column 'Close'"):
        yfinance._ohlcv_frame_to_lazyframe(missing_close, config=make_config())

    multi_index = missing_close.copy()
    multi_index.index = pd.MultiIndex.from_tuples(
        [("AAPL", pd.Timestamp("2026-01-01"))]
    )
    with pytest.raises(ValueError, match="single time index"):
        yfinance._ohlcv_frame_to_lazyframe(multi_index, config=make_config())


def test_yfinance_ohlcv_tsfn_executes_through_graph(monkeypatch) -> None:
    calls = []

    def fake_download(config: YFinanceOHLCVConfig):
        calls.append(config)
        return pd.DataFrame(
            {
                "Open": [10],
                "High": [12],
                "Low": [9],
                "Close": [11],
                "Volume": [1000],
            },
            index=pd.to_datetime(["2026-01-01"]),
        )

    monkeypatch.setattr(yfinance, "_download_ohlcv", fake_download)
    node = Node(
        YFinanceOHLCV,
        parameters={
            "ticker": "AAPL",
            "date_range": ["2026-01-01", "2026-01-03"],
            "interval": "1d",
        },
    )

    result = Graph(node).execute()

    assert calls == [make_config()]
    assert node.outputs == {
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close": pl.Float64,
        "volume": pl.Int64,
    }
    assert result.row(0) == (datetime(2026, 1, 1), 10.0, 12.0, 9.0, 11.0, 1000)
