from __future__ import annotations

from datetime import datetime
from urllib.parse import parse_qs, urlparse

import polars as pl
import pytest

from src.classes import Graph, Node
from src.tsfn.adapters import PolymarketPriceHistory, PolymarketPriceHistoryConfig
from src.tsfn.adapters import polymarket


def test_polymarket_price_history_url_uses_expected_clob_query_params() -> None:
    config = PolymarketPriceHistoryConfig(
        token_id="123",
        start_ts=10,
        end_ts=20,
        interval="1h",
        fidelity=5,
        base_url="https://example.test/",
    )

    url = polymarket._price_history_url(config)
    parsed = urlparse(url)
    query = parse_qs(parsed.query)

    assert parsed.scheme == "https"
    assert parsed.netloc == "example.test"
    assert parsed.path == "/prices-history"
    assert query == {
        "market": ["123"],
        "interval": ["1h"],
        "startTs": ["10"],
        "endTs": ["20"],
        "fidelity": ["5"],
    }


def test_polymarket_price_history_payload_becomes_sorted_lazyframe() -> None:
    lf = polymarket._price_history_payload_to_lazyframe(
        {
            "history": [
                {"t": 60, "p": "0.62"},
                {"t": 0, "p": 0.5},
            ]
        },
        timestamp_column="timestamp",
        price_column="price",
    )

    result = lf.collect()

    assert result.schema["timestamp"] == pl.Datetime
    assert result.schema["price"] == pl.Float64
    assert result["timestamp"].to_list() == [
        datetime(1970, 1, 1, 0, 0),
        datetime(1970, 1, 1, 0, 1),
    ]
    assert result["price"].to_list() == [0.5, 0.62]


def test_polymarket_price_history_payload_validation_is_explicit() -> None:
    with pytest.raises(ValueError, match="must be a JSON object"):
        polymarket._price_history_payload_to_lazyframe(
            [],
            timestamp_column="timestamp",
            price_column="price",
        )

    with pytest.raises(ValueError, match="'history' must be a list"):
        polymarket._price_history_payload_to_lazyframe(
            {"history": {}},
            timestamp_column="timestamp",
            price_column="price",
        )

    with pytest.raises(ValueError, match="must contain 't' and 'p'"):
        polymarket._price_history_payload_to_lazyframe(
            {"history": [{"t": 0}]},
            timestamp_column="timestamp",
            price_column="price",
        )

    with pytest.raises(ValueError, match="Invalid Polymarket timestamp"):
        polymarket._price_history_payload_to_lazyframe(
            {"history": [{"t": "bad", "p": 0.5}]},
            timestamp_column="timestamp",
            price_column="price",
        )


def test_polymarket_price_history_config_validation() -> None:
    with pytest.raises(ValueError, match="token_id must be non-empty"):
        PolymarketPriceHistoryConfig(token_id="")

    with pytest.raises(ValueError, match="interval must be one of"):
        PolymarketPriceHistoryConfig(token_id="123", interval="bad")

    with pytest.raises(ValueError, match="fidelity must be at least 1 minute"):
        PolymarketPriceHistoryConfig(token_id="123", fidelity=0)

    with pytest.raises(ValueError, match="start_ts must be less than or equal to end_ts"):
        PolymarketPriceHistoryConfig(token_id="123", start_ts=20, end_ts=10)

    with pytest.raises(ValueError, match="must be different"):
        PolymarketPriceHistoryConfig(
            token_id="123",
            timestamp_column="value",
            price_column="value",
        )


def test_polymarket_price_history_tsfn_executes_through_graph(monkeypatch) -> None:
    calls = []

    def fake_fetch_json(url: str, *, timeout_seconds: float):
        calls.append((url, timeout_seconds))
        return {"history": [{"t": 0, "p": 0.5}, {"t": 60, "p": "0.6"}]}

    monkeypatch.setattr(polymarket, "_fetch_json", fake_fetch_json)
    node = Node(
        PolymarketPriceHistory,
        parameters={
            "token_id": "123",
            "interval": "1h",
            "base_url": "https://example.test",
            "timeout_seconds": 3.0,
        },
    )

    result = Graph(node).execute()

    assert calls == [
        (
            "https://example.test/prices-history?market=123&interval=1h",
            3.0,
        )
    ]
    assert result["timestamp"].to_list() == [
        datetime(1970, 1, 1, 0, 0),
        datetime(1970, 1, 1, 0, 1),
    ]
    assert result["price"].to_list() == [0.5, 0.6]
