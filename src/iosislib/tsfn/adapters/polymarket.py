from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from numbers import Real
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import polars as pl

from iosislib.core.tsfn import FrameSignature, TSFN, TSFNConfig, TimeAxis


POLYMARKET_CLOB_URL = "https://clob.polymarket.com"
POLYMARKET_GAMMA_URL = "https://gamma-api.polymarket.com"
PRICE_HISTORY_INTERVALS = frozenset({"max", "all", "1m", "1h", "6h", "1d", "1w"})


@dataclass(frozen=True)
class PolymarketMarket:
    column_name: str
    outcomes: tuple[str, ...]
    token_ids: tuple[str, ...]


@dataclass(frozen=True)
class PolymarketPriceHistoryConfig(TSFNConfig):
    token_id: str | None = None
    event_slug: str | None = None
    start_ts: int | None = None
    end_ts: int | None = None
    interval: str = "1m"
    fidelity: int | None = None
    base_url: str = POLYMARKET_CLOB_URL
    gamma_base_url: str = POLYMARKET_GAMMA_URL
    timeout_seconds: float = 10.0
    timestamp_column: str = "timestamp"
    price_column: str = "price"
    alignment_tolerance: str | int | float | None = None
    market_column_names: tuple[str, ...] | list[str] | None = None
    include_unpriced_markets: bool = False

    def __post_init__(self) -> None:
        if self.token_id is None and self.event_slug is None:
            raise ValueError("Exactly one of token_id or event_slug must be provided")
        if self.token_id is not None and self.event_slug is not None:
            raise ValueError("Exactly one of token_id or event_slug must be provided")
        if self.token_id is not None and not self.token_id:
            raise ValueError("token_id must be non-empty")
        if self.event_slug is not None and not self.event_slug:
            raise ValueError("event_slug must be non-empty")
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
        if self.token_id is not None:
            if not self.price_column:
                raise ValueError("price_column must be non-empty")
            if self.timestamp_column == self.price_column:
                raise ValueError("timestamp_column and price_column must be different")
        elif self.price_column != "price":
            raise ValueError(
                "price_column is only used with token_id; use market_column_names "
                "for event_slug outputs"
            )
        if isinstance(self.alignment_tolerance, bool) or not isinstance(
            self.alignment_tolerance,
            (str, Real, type(None)),
        ):
            raise TypeError(
                "alignment_tolerance must be a Polars asof tolerance string, "
                "number, or None"
            )
        if isinstance(self.alignment_tolerance, str) and not self.alignment_tolerance:
            raise ValueError("alignment_tolerance must be non-empty when provided")
        if isinstance(self.alignment_tolerance, Real) and self.alignment_tolerance < 0:
            raise ValueError("alignment_tolerance must be non-negative")
        if not isinstance(self.include_unpriced_markets, bool):
            raise TypeError("include_unpriced_markets must be a bool")
        if self.market_column_names is not None:
            if self.event_slug is None:
                raise ValueError("market_column_names can only be used with event_slug")
            if isinstance(self.market_column_names, str) or not isinstance(
                self.market_column_names,
                Sequence,
            ):
                raise TypeError("market_column_names must be a sequence of strings")

            market_column_names = tuple(self.market_column_names)
            if not all(isinstance(name, str) for name in market_column_names):
                raise TypeError("market_column_names must contain only strings")
            if any(not name for name in market_column_names):
                raise ValueError("market_column_names must contain non-empty strings")

            duplicate_names = sorted(
                {name for name in market_column_names if market_column_names.count(name) > 1}
            )
            if duplicate_names:
                raise ValueError(
                    f"Duplicate market_column_names are not allowed: {duplicate_names}"
                )
            object.__setattr__(self, "market_column_names", market_column_names)


class PolymarketPriceHistory(TSFN):
    VERSION = "0.1.0"
    CONFIG_CLS = PolymarketPriceHistoryConfig

    def _event_markets(self) -> tuple[PolymarketMarket, ...]:
        markets = getattr(self, "_resolved_event_markets", None)
        if markets is None:
            payload = _fetch_json(
                _event_url(self.parameters),
                timeout_seconds=self.parameters.timeout_seconds,
            )
            markets = _markets_from_event_payload(
                payload,
                requested_slug=self.parameters.event_slug,
                column_names=self.parameters.market_column_names,
                include_unpriced_markets=self.parameters.include_unpriced_markets,
            )
            self._resolved_event_markets = markets
        return markets

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        params = self.parameters
        columns = (
            tuple((market.column_name, pl.List(pl.Float64)) for market in self._event_markets())
            if params.event_slug is not None
            else ((params.price_column, pl.Float64),)
        )
        output = FrameSignature(
            time=TimeAxis(column=params.timestamp_column),
            columns=columns,
        )
        return FrameSignature.empty(), output

    def apply(self) -> pl.LazyFrame:
        params = self.parameters
        if params.event_slug is not None:
            return _event_price_history_to_lazyframe(
                self._event_markets(),
                config=params,
            )

        payload = _fetch_json(
            _price_history_url(params),
            timeout_seconds=params.timeout_seconds,
        )
        return _price_history_payload_to_lazyframe(
            payload,
            timestamp_column=params.timestamp_column,
            price_column=params.price_column,
        )


def _event_url(config: PolymarketPriceHistoryConfig) -> str:
    if config.event_slug is None:
        raise ValueError("event_slug is required to build a Polymarket event URL")
    return f"{config.gamma_base_url.rstrip('/')}/events?{urlencode({'slug': config.event_slug})}"


def _price_history_url(
    config: PolymarketPriceHistoryConfig,
    *,
    token_id: str | None = None,
) -> str:
    market = token_id if token_id is not None else config.token_id
    if market is None:
        raise ValueError("token_id is required to build a Polymarket price history URL")

    query: dict[str, str | int] = {
        "market": market,
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


def _event_price_history_to_lazyframe(
    markets: tuple[PolymarketMarket, ...],
    *,
    config: PolymarketPriceHistoryConfig,
) -> pl.LazyFrame:
    if not markets:
        return _empty_event_price_history_lazyframe(
            timestamp_column=config.timestamp_column,
        )

    frames = []
    for market in markets:
        outcome_frames = []
        outcome_price_columns = []
        for index, token_id in enumerate(market.token_ids):
            price_column = f"__outcome_price_{index}"
            payload = _fetch_json(
                _price_history_url(config, token_id=token_id),
                timeout_seconds=config.timeout_seconds,
            )
            outcome_frames.append(
                _price_history_payload_to_lazyframe(
                    payload,
                    timestamp_column=config.timestamp_column,
                    price_column=price_column,
                )
            )
            outcome_price_columns.append(price_column)

        market_lf = _asof_join_on_union_timeline(
            tuple(outcome_frames),
            timestamp_column=config.timestamp_column,
            tolerance=config.alignment_tolerance,
        )
        frames.append(
            market_lf.with_columns(
                pl.concat_list(outcome_price_columns).alias(market.column_name)
            ).select(config.timestamp_column, market.column_name)
        )

    return _asof_join_on_union_timeline(
        tuple(frames),
        timestamp_column=config.timestamp_column,
        tolerance=config.alignment_tolerance,
    )


def _asof_join_on_union_timeline(
    frames: tuple[pl.LazyFrame, ...],
    *,
    timestamp_column: str,
    tolerance: str | int | float | None,
) -> pl.LazyFrame:
    if not frames:
        return _empty_event_price_history_lazyframe(timestamp_column=timestamp_column)

    output = (
        pl.concat(
            [frame.select(timestamp_column) for frame in frames],
            how="vertical",
        )
        .unique()
        .sort(timestamp_column)
    )
    for frame in frames:
        output = output.join_asof(
            frame.sort(timestamp_column),
            on=timestamp_column,
            strategy="backward",
            tolerance=tolerance,
        )
    return output.sort(timestamp_column)


def _markets_from_event_payload(
    payload: Any,
    *,
    requested_slug: str | None,
    column_names: tuple[str, ...] | None = None,
    include_unpriced_markets: bool = False,
) -> tuple[PolymarketMarket, ...]:
    event = _event_from_payload(payload, requested_slug=requested_slug)

    markets = event.get("markets", event.get("market"))
    if not isinstance(markets, list):
        raise ValueError("Polymarket event payload must contain a markets list")

    included_markets = [
        market
        for market in markets
        if include_unpriced_markets or not _market_is_explicitly_unpriced(market)
    ]
    output_column_names = _event_market_column_names(
        market_count=len(included_markets),
        column_names=column_names,
    )
    event_markets = []
    for index, market in enumerate(included_markets):
        if not isinstance(market, dict):
            raise ValueError("Polymarket event markets must be JSON objects")
        outcomes = _json_array_field(market, "outcomes")
        clob_token_ids = _json_array_field(market, "clobTokenIds")
        if len(outcomes) != len(clob_token_ids):
            raise ValueError(
                "Polymarket market outcomes and clobTokenIds must have equal length"
            )
        if not outcomes:
            raise ValueError("Polymarket event markets must expose at least one outcome")

        event_markets.append(
            PolymarketMarket(
                column_name=output_column_names[index],
                outcomes=tuple(str(outcome) for outcome in outcomes),
                token_ids=tuple(str(token_id) for token_id in clob_token_ids),
            )
        )

    return tuple(event_markets)


def _market_is_explicitly_unpriced(market: Any) -> bool:
    if not isinstance(market, dict):
        return False

    if "outcomePrices" not in market:
        has_lifecycle_flags = any(key in market for key in ("active", "funded", "ready"))
        return (
            has_lifecycle_flags
            and market.get("active") is False
            and market.get("funded") is False
            and market.get("ready") is False
        )

    outcome_prices = market.get("outcomePrices")
    if outcome_prices is None or outcome_prices == "":
        return True
    if isinstance(outcome_prices, str):
        try:
            outcome_prices = json.loads(outcome_prices)
        except json.JSONDecodeError:
            return False
    return isinstance(outcome_prices, list) and len(outcome_prices) == 0


def _event_market_column_names(
    *,
    market_count: int,
    column_names: tuple[str, ...] | None,
) -> tuple[str, ...]:
    if column_names is None:
        return tuple(str(index) for index in range(market_count))
    if len(column_names) != market_count:
        raise ValueError(
            "market_column_names length must match the number of Polymarket event markets"
        )
    return column_names


def _event_from_payload(payload: Any, *, requested_slug: str | None) -> dict[str, Any]:
    if isinstance(payload, list):
        if not payload:
            raise ValueError(f"No Polymarket event found for slug '{requested_slug}'")
        event = payload[0]
    elif isinstance(payload, dict):
        event = payload
    else:
        raise ValueError("Polymarket event payload must be a JSON object or list")

    if not isinstance(event, dict):
        raise ValueError("Polymarket event payload entries must be JSON objects")
    return event


def _json_array_field(payload: dict[str, Any], field_name: str) -> list[Any]:
    value = payload.get(field_name)
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Polymarket field '{field_name}' must be a JSON array") from exc
        if isinstance(parsed, list):
            return parsed
    raise ValueError(f"Polymarket field '{field_name}' must be a list")


def _empty_event_price_history_lazyframe(
    *,
    timestamp_column: str,
) -> pl.LazyFrame:
    schema = {
        timestamp_column: pl.Datetime,
    }
    return pl.DataFrame([], schema=schema).lazy()


def _timestamp_from_unix_seconds(value: Any) -> datetime:
    try:
        timestamp = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid Polymarket timestamp: {value!r}") from exc
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).replace(tzinfo=None)
