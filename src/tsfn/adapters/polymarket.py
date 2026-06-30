from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import polars as pl

from src.classes import FrameSignature, TSFN, TSFNConfig, TimeAxis


POLYMARKET_CLOB_URL = "https://clob.polymarket.com"
PRICE_HISTORY_INTERVALS = frozenset({"max", "all", "1m", "1h", "6h", "1d", "1w"})


@dataclass(frozen=True)
class PolymarketPriceHistoryConfig(TSFNConfig):
    token_id: str
    start_ts: int | None = None
    end_ts: int | None = None
    interval: str = "1m"
    fidelity: int | None = None
    base_url: str = POLYMARKET_CLOB_URL
    timeout_seconds: float = 10.0
    timestamp_column: str = "timestamp"
    price_column: str = "price"

    def __post_init__(self) -> None:
        if not self.token_id:
            raise ValueError("token_id must be non-empty")
        if self.interval not in PRICE_HISTORY_INTERVALS:
            raise ValueError(
                f"interval must be one of {sorted(PRICE_HISTORY_INTERVALS)}"
            )
        if self.fidelity is not None and self.fidelity < 1:
            raise ValueError("fidelity must be at least 1 minute")
        if (
            self.start_ts is not None
            and self.end_ts is not None
            and self.start_ts > self.end_ts
        ):
            raise ValueError("start_ts must be less than or equal to end_ts")
        if not self.timestamp_column:
            raise ValueError("timestamp_column must be non-empty")
        if not self.price_column:
            raise ValueError("price_column must be non-empty")
        if self.timestamp_column == self.price_column:
            raise ValueError("timestamp_column and price_column must be different")


class PolymarketPriceHistory(TSFN):
    VERSION = "0.1.0"
    CONFIG_CLS = PolymarketPriceHistoryConfig

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        params = self.parameters
        output = FrameSignature(
            time=TimeAxis(column=params.timestamp_column),
            columns=((params.price_column, pl.Float64),),
        )
        return FrameSignature.empty(), output

    def apply(self) -> pl.LazyFrame:
        params = self.parameters
        payload = _fetch_json(
            _price_history_url(params),
            timeout_seconds=params.timeout_seconds,
        )
        return _price_history_payload_to_lazyframe(
            payload,
            timestamp_column=params.timestamp_column,
            price_column=params.price_column,
        )


def _price_history_url(config: PolymarketPriceHistoryConfig) -> str:
    query: dict[str, str | int] = {
        "market": config.token_id,
        "interval": config.interval,
    }
    if config.start_ts is not None:
        query["startTs"] = config.start_ts
    if config.end_ts is not None:
        query["endTs"] = config.end_ts
    if config.fidelity is not None:
        query["fidelity"] = config.fidelity

    return f"{config.base_url.rstrip('/')}/prices-history?{urlencode(query)}"


def _fetch_json(url: str, *, timeout_seconds: float) -> Any:
    request = Request(url, headers={"User-Agent": "iosislib/0.1"})
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise RuntimeError(f"Polymarket API returned HTTP {exc.code} for {url}") from exc
    except URLError as exc:
        raise RuntimeError(f"Polymarket API request failed for {url}: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Polymarket API returned invalid JSON for {url}") from exc


def _price_history_payload_to_lazyframe(
    payload: Any,
    *,
    timestamp_column: str,
    price_column: str,
) -> pl.LazyFrame:
    if not isinstance(payload, dict):
        raise ValueError("Polymarket price history payload must be a JSON object")

    history = payload.get("history", [])
    if not isinstance(history, list):
        raise ValueError("Polymarket price history payload 'history' must be a list")

    rows = []
    for item in history:
        if not isinstance(item, dict):
            raise ValueError("Polymarket price history entries must be JSON objects")
        if "t" not in item or "p" not in item:
            raise ValueError("Polymarket price history entries must contain 't' and 'p'")

        rows.append(
            {
                timestamp_column: _timestamp_from_unix_seconds(item["t"]),
                price_column: float(item["p"]),
            }
        )

    schema = {
        timestamp_column: pl.Datetime,
        price_column: pl.Float64,
    }
    return pl.DataFrame(rows, schema=schema).lazy().sort(timestamp_column)


def _timestamp_from_unix_seconds(value: Any) -> datetime:
    try:
        timestamp = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid Polymarket timestamp: {value!r}") from exc
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).replace(tzinfo=None)
