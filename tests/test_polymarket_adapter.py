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


def test_polymarket_event_url_uses_slug_query_param() -> None:
    config = PolymarketPriceHistoryConfig(
        event_slug="highest-temperature-in-shanghai-on-july-1-2026",
        gamma_base_url="https://gamma.example.test/",
    )

    url = polymarket._event_url(config)
    parsed = urlparse(url)

    assert parsed.scheme == "https"
    assert parsed.netloc == "gamma.example.test"
    assert parsed.path == "/events"
    assert parse_qs(parsed.query) == {
        "slug": ["highest-temperature-in-shanghai-on-july-1-2026"]
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


def test_polymarket_event_payload_extracts_markets_with_ordered_outcomes() -> None:
    markets = polymarket._markets_from_event_payload(
        [
            {
                "slug": "highest-temperature-in-shanghai-on-july-1-2026",
                "markets": [
                    {
                        "id": "2725880",
                        "slug": "highest-temperature-in-shanghai-on-july-1-2026-21corbelow",
                        "question": "Will the highest temperature be 21C or below?",
                        "outcomes": '["21C or below", "22C", "23C or above"]',
                        "clobTokenIds": '["low-token", "mid-token", "high-token"]',
                    }
                ],
            }
        ],
        requested_slug="highest-temperature-in-shanghai-on-july-1-2026",
    )

    assert markets == (
        polymarket.PolymarketMarket(
            column_name="0",
            outcomes=("21C or below", "22C", "23C or above"),
            token_ids=("low-token", "mid-token", "high-token"),
        ),
    )


def test_polymarket_event_payload_accepts_explicit_market_column_names() -> None:
    markets = polymarket._markets_from_event_payload(
        {
            "slug": "weather-event",
            "markets": [
                {
                    "outcomes": '["Low", "High"]',
                    "clobTokenIds": '["low-token", "high-token"]',
                },
                {
                    "outcomes": '["Rain", "Dry"]',
                    "clobTokenIds": '["rain-token", "dry-token"]',
                },
            ],
        },
        requested_slug="weather-event",
        column_names=("temperature", "rainfall"),
    )

    assert tuple(market.column_name for market in markets) == (
        "temperature",
        "rainfall",
    )


def test_polymarket_event_payload_validation_is_explicit() -> None:
    with pytest.raises(ValueError, match="No Polymarket event found"):
        polymarket._markets_from_event_payload([], requested_slug="missing")

    with pytest.raises(ValueError, match="must contain a markets list"):
        polymarket._markets_from_event_payload({"slug": "x"}, requested_slug="x")

    with pytest.raises(ValueError, match="must have equal length"):
        polymarket._markets_from_event_payload(
            {
                "slug": "x",
                "markets": [
                    {
                        "outcomes": '["Yes", "No"]',
                        "clobTokenIds": '["only-one"]',
                    }
                ],
            },
            requested_slug="x",
        )

    with pytest.raises(ValueError, match="field 'outcomes' must be a JSON array"):
        polymarket._markets_from_event_payload(
            {
                "slug": "x",
                "markets": [
                    {
                        "outcomes": "not-json",
                        "clobTokenIds": '["id"]',
                    }
                ],
            },
            requested_slug="x",
        )

    with pytest.raises(ValueError, match="at least one outcome"):
        polymarket._markets_from_event_payload(
            {
                "slug": "x",
                "markets": [
                    {
                        "id": "market-id",
                        "outcomes": "[]",
                        "clobTokenIds": "[]",
                    }
                ],
            },
            requested_slug="x",
        )

    with pytest.raises(ValueError, match="market_column_names length"):
        polymarket._markets_from_event_payload(
            {
                "slug": "x",
                "markets": [
                    {
                        "outcomes": '["Yes", "No"]',
                        "clobTokenIds": '["yes-token", "no-token"]',
                    }
                ],
            },
            requested_slug="x",
            column_names=("a", "b"),
        )


def test_polymarket_price_history_config_validation() -> None:
    with pytest.raises(ValueError, match="Exactly one"):
        PolymarketPriceHistoryConfig()

    with pytest.raises(ValueError, match="Exactly one"):
        PolymarketPriceHistoryConfig(token_id="123", event_slug="slug")

    with pytest.raises(ValueError, match="token_id must be non-empty"):
        PolymarketPriceHistoryConfig(token_id="")

    with pytest.raises(ValueError, match="event_slug must be non-empty"):
        PolymarketPriceHistoryConfig(event_slug="")

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

    with pytest.raises(ValueError, match="price_column is only used with token_id"):
        PolymarketPriceHistoryConfig(
            event_slug="slug",
            price_column="ignored_price_name",
        )

    with pytest.raises(ValueError, match="alignment_tolerance must be non-empty"):
        PolymarketPriceHistoryConfig(token_id="123", alignment_tolerance="")

    with pytest.raises(ValueError, match="alignment_tolerance must be non-negative"):
        PolymarketPriceHistoryConfig(token_id="123", alignment_tolerance=-1)

    with pytest.raises(TypeError, match="alignment_tolerance must be"):
        PolymarketPriceHistoryConfig(token_id="123", alignment_tolerance=True)

    with pytest.raises(ValueError, match="can only be used with event_slug"):
        PolymarketPriceHistoryConfig(token_id="123", market_column_names=("a",))

    with pytest.raises(TypeError, match="sequence of strings"):
        PolymarketPriceHistoryConfig(event_slug="slug", market_column_names="a")

    with pytest.raises(TypeError, match="only strings"):
        PolymarketPriceHistoryConfig(event_slug="slug", market_column_names=("a", 1))

    with pytest.raises(ValueError, match="non-empty strings"):
        PolymarketPriceHistoryConfig(event_slug="slug", market_column_names=("a", ""))

    with pytest.raises(ValueError, match="Duplicate market_column_names"):
        PolymarketPriceHistoryConfig(event_slug="slug", market_column_names=("a", "a"))


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


def test_polymarket_event_slug_tsfn_fetches_market_price_vectors(
    monkeypatch,
) -> None:
    calls = []

    def fake_fetch_json(url: str, *, timeout_seconds: float):
        calls.append((url, timeout_seconds))
        if "gamma.example.test/events" in url:
            return [
                {
                    "slug": "highest-temperature-in-shanghai-on-july-1-2026",
                    "markets": [
                        {
                            "id": "2725880",
                            "slug": "highest-temperature-in-shanghai-on-july-1-2026-21corbelow",
                            "question": "Will the highest temperature be 21C or below?",
                            "outcomes": '["21C or below", "22C", "23C or above"]',
                            "clobTokenIds": '["low-token", "mid-token", "high-token"]',
                        },
                        {
                            "id": "2725881",
                            "slug": "highest-temperature-in-shanghai-on-july-1-2026-22c",
                            "question": "Will the highest temperature be 22C?",
                            "outcomes": '["Yes", "No"]',
                            "clobTokenIds": '["yes-token-2", "no-token-2"]',
                        }
                    ],
                }
            ]
        if "market=low-token" in url:
            return {"history": [{"t": 0, "p": 0.1}, {"t": 60, "p": 0.2}]}
        if "market=mid-token" in url:
            return {"history": [{"t": 0, "p": 0.3}, {"t": 60, "p": 0.4}]}
        if "market=high-token" in url:
            return {"history": [{"t": 0, "p": 0.6}, {"t": 60, "p": 0.4}]}
        if "market=yes-token-2" in url:
            return {"history": [{"t": 0, "p": 0.3}, {"t": 60, "p": 0.4}]}
        if "market=no-token-2" in url:
            return {"history": [{"t": 0, "p": 0.7}, {"t": 60, "p": 0.6}]}
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(polymarket, "_fetch_json", fake_fetch_json)
    node = Node(
        PolymarketPriceHistory,
        parameters={
            "event_slug": "highest-temperature-in-shanghai-on-july-1-2026",
            "interval": "1m",
            "fidelity": 10,
            "base_url": "https://clob.example.test",
            "gamma_base_url": "https://gamma.example.test",
            "alignment_tolerance": "1m",
            "market_column_names": ["low_mid_high", "exact_22"],
        },
    )

    result = Graph(node).execute()
    column_21_or_below = "low_mid_high"
    column_22 = "exact_22"
    event_url = (
        "https://gamma.example.test/events?"
        "slug=highest-temperature-in-shanghai-on-july-1-2026"
    )

    assert calls == [
        (
            event_url,
            10.0,
        ),
        (
            "https://clob.example.test/prices-history?market=low-token&interval=1m&fidelity=10",
            10.0,
        ),
        (
            "https://clob.example.test/prices-history?market=mid-token&interval=1m&fidelity=10",
            10.0,
        ),
        (
            "https://clob.example.test/prices-history?market=high-token&interval=1m&fidelity=10",
            10.0,
        ),
        (
            "https://clob.example.test/prices-history?market=yes-token-2&interval=1m&fidelity=10",
            10.0,
        ),
        (
            "https://clob.example.test/prices-history?market=no-token-2&interval=1m&fidelity=10",
            10.0,
        ),
    ]
    assert node.outputs == {
        column_21_or_below: pl.List(pl.Float64),
        column_22: pl.List(pl.Float64),
    }
    assert result.columns == ["timestamp", column_21_or_below, column_22]
    assert result["timestamp"].to_list() == [
        datetime(1970, 1, 1, 0, 0),
        datetime(1970, 1, 1, 0, 1),
    ]
    assert result[column_21_or_below].to_list() == [
        [0.1, 0.3, 0.6],
        [0.2, 0.4, 0.4],
    ]
    assert result[column_22].to_list() == [[0.3, 0.7], [0.4, 0.6]]


def test_polymarket_event_slug_asof_aligns_market_vectors_with_tolerance(
    monkeypatch,
) -> None:
    def fake_fetch_json(url: str, *, timeout_seconds: float):
        if "gamma.example.test/events" in url:
            return [
                {
                    "slug": "example-event",
                    "markets": [
                        {
                            "id": "market-a",
                            "slug": "market-a",
                            "outcomes": '["Up", "Down"]',
                            "clobTokenIds": '["a-up", "a-down"]',
                        },
                        {
                            "id": "market-b",
                            "slug": "market-b",
                            "outcomes": '["Warm", "Cold"]',
                            "clobTokenIds": '["b-warm", "b-cold"]',
                        },
                    ],
                }
            ]

        market = parse_qs(urlparse(url).query)["market"][0]
        histories = {
            "a-up": [{"t": 0, "p": 0.1}, {"t": 60, "p": 0.2}],
            "a-down": [{"t": 0, "p": 0.9}, {"t": 60, "p": 0.8}],
            "b-warm": [{"t": 3, "p": 0.3}, {"t": 63, "p": 0.4}],
            "b-cold": [{"t": 3, "p": 0.7}, {"t": 63, "p": 0.6}],
        }
        return {"history": histories[market]}

    monkeypatch.setattr(polymarket, "_fetch_json", fake_fetch_json)
    node = Node(
        PolymarketPriceHistory,
        parameters={
            "event_slug": "example-event",
            "base_url": "https://clob.example.test",
            "gamma_base_url": "https://gamma.example.test",
            "alignment_tolerance": "5s",
        },
    )

    result = Graph(node).execute()

    assert result.columns == ["timestamp", "0", "1"]
    assert result["timestamp"].to_list() == [
        datetime(1970, 1, 1, 0, 0),
        datetime(1970, 1, 1, 0, 0, 3),
        datetime(1970, 1, 1, 0, 1),
        datetime(1970, 1, 1, 0, 1, 3),
    ]
    assert result["0"].to_list() == [
        [0.1, 0.9],
        [0.1, 0.9],
        [0.2, 0.8],
        [0.2, 0.8],
    ]
    assert result["1"].to_list() == [
        None,
        [0.3, 0.7],
        None,
        [0.4, 0.6],
    ]
