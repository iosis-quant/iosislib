from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from iosislib.backtest import BacktestConfig, BacktestTSFN, SignalPolicy, Venue
from iosislib.backtest.backtest import _feed_from_declaration
from iosislib.backtest.feeds import (
    L2Feed,
    dense_l2_side_from_sparse,
    l2_interpolate_volumes,
    l2_ladder_prices,
)

START = datetime(2026, 1, 1)
LEVELS = 4


def venue(width: int = 1) -> Venue:
    return Venue("test", tuple(f"A{i}" for i in range(width)))


def frame(
    bid_price: np.ndarray,
    bid_volume: np.ndarray,
    ask_price: np.ndarray,
    ask_volume: np.ndarray,
    signal: list[list[float]] | None = None,
) -> pl.DataFrame:
    rows, width, levels = bid_price.shape
    assert levels == LEVELS
    sig = signal if signal is not None else [[0.0] * width for _ in range(rows)]
    return pl.DataFrame(
        [
            pl.Series(
                "timestamp",
                [START + timedelta(minutes=r) for r in range(rows)],
                dtype=pl.Datetime,
            ),
            pl.Series(
                "bid_price",
                bid_price.reshape(rows, -1).tolist(),
                dtype=pl.Array(pl.Float64, width * levels),
            ),
            pl.Series(
                "bid_volume",
                bid_volume.reshape(rows, -1).tolist(),
                dtype=pl.Array(pl.Float64, width * levels),
            ),
            pl.Series(
                "ask_price",
                ask_price.reshape(rows, -1).tolist(),
                dtype=pl.Array(pl.Float64, width * levels),
            ),
            pl.Series(
                "ask_volume",
                ask_volume.reshape(rows, -1).tolist(),
                dtype=pl.Array(pl.Float64, width * levels),
            ),
            pl.Series("signal", sig, dtype=pl.Array(pl.Float64, width)),
        ]
    )


def empty_book(rows: int, width: int) -> tuple[np.ndarray, ...]:
    bid_price = np.tile(
        np.array([0.40, 0.30, 0.20, 0.10]), (rows, width, 1)
    )
    ask_price = np.tile(
        np.array([0.50, 0.60, 0.70, 0.80]), (rows, width, 1)
    )
    return (
        bid_price.copy(),
        np.zeros((rows, width, LEVELS)),
        ask_price.copy(),
        np.zeros((rows, width, LEVELS)),
    )


def test_l2_feed_declares_four_columns() -> None:
    feed = L2Feed(venue(2), depth_levels=LEVELS)
    assert feed.VERSION == "2.0.0"
    assert feed.columns == (
        ("bid_price", pl.Float64, (2, LEVELS)),
        ("bid_volume", pl.Float64, (2, LEVELS)),
        ("ask_price", pl.Float64, (2, LEVELS)),
        ("ask_volume", pl.Float64, (2, LEVELS)),
    )
    assert feed.to_dict()["depth_levels"] == LEVELS


def test_l2_feed_rejects_bad_config() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        L2Feed(venue(), depth_levels=0)
    with pytest.raises(ValueError, match="distinct"):
        L2Feed(venue(), bid_price_column="same", bid_volume_column="same")
    with pytest.raises(ValueError, match="cannot be empty"):
        L2Feed(venue(), ask_price_column="")


def test_l2_ladder_is_linear_and_inclusive() -> None:
    ladder = l2_ladder_prices(0.5, 0.0, 4)
    assert ladder.tolist() == pytest.approx([0.5, 1 / 3, 1 / 6, 0.0])
    assert l2_ladder_prices(0.5, 1.0, 1).tolist() == [0.5]
    with pytest.raises(ValueError, match="positive integer"):
        l2_ladder_prices(0.5, 0.0, 0)


def test_l2_interpolation_is_linear_with_zero_outside() -> None:
    ladder = np.array([0.5, 0.6, 0.7, 0.8])
    vols = l2_interpolate_volumes(
        ladder, np.array([0.5, 0.7]), np.array([10.0, 30.0])
    )
    assert vols.tolist() == pytest.approx([10.0, 20.0, 30.0, 0.0])
    assert l2_interpolate_volumes(ladder, np.array([]), np.array([])).tolist() == [
        0.0,
        0.0,
        0.0,
        0.0,
    ]


def test_dense_side_combines_ladder_and_volumes() -> None:
    prices, volumes = dense_l2_side_from_sparse(
        0.5, 1.0, 3, np.array([0.5, 0.6]), np.array([4.0, 8.0])
    )
    assert prices.tolist() == pytest.approx([0.5, 0.75, 1.0])
    assert volumes.tolist() == pytest.approx([4.0, 0.0, 0.0])


def test_quotes_use_first_resting_level_with_empty_fallback() -> None:
    bid_price, bid_volume, ask_price, ask_volume = empty_book(1, 1)
    bid_volume[0, 0, 2] = 7.0
    ask_volume[0, 0, 1] = 3.0
    feed = L2Feed(venue(), depth_levels=LEVELS)
    bid, ask = feed.quotes(
        frame(bid_price, bid_volume, ask_price, ask_volume)
    )
    assert bid.to_list() == [[0.20]]
    assert ask.to_list() == [[0.60]]

    bid_price, bid_volume, ask_price, ask_volume = empty_book(1, 1)
    bid, ask = feed.quotes(
        frame(bid_price, bid_volume, ask_price, ask_volume)
    )
    assert bid.to_list() == [[0.40]]
    assert ask.to_list() == [[0.50]]


def test_backtest_walks_explicit_prices_across_levels() -> None:
    bid_price, bid_volume, ask_price, ask_volume = empty_book(1, 1)
    ask_volume[0, 0, 0] = 4.0
    ask_volume[0, 0, 1] = 4.0
    ask_price[0, 0, 0] = 0.50
    ask_price[0, 0, 1] = 0.60
    feed = L2Feed(venue(), depth_levels=LEVELS)
    result = BacktestTSFN(
        BacktestConfig(feed=feed, policy=SignalPolicy(), initial_cash=100.0)
    ).batch(frame(bid_price, bid_volume, ask_price, ask_volume, [[6.0]]))
    assert result.get_column("fill_price").to_list()[0] == pytest.approx(
        [(4 * 0.5 + 2 * 0.6) / 6]
    )
    assert result.get_column("unfilled").to_list() == [[0.0]]


def test_backtest_limit_uses_explicit_prices() -> None:
    bid_price, bid_volume, ask_price, ask_volume = empty_book(1, 1)
    ask_volume[0, 0, 0] = 10.0
    ask_volume[0, 0, 1] = 10.0
    ask_price[0, 0] = np.array([0.50, 0.90, 0.95, 1.00])
    feed = L2Feed(venue(), depth_levels=LEVELS)
    node = BacktestTSFN(
        BacktestConfig(
            feed=feed,
            policy=SignalPolicy(),
            initial_cash=100.0,
            limit_price_column="limit",
        )
    )
    values = frame(bid_price, bid_volume, ask_price, ask_volume, [[5.0]])
    limited = values.with_columns(
        pl.Series("limit", [[0.60]], dtype=pl.Array(pl.Float64, 1))
    )
    result = node.batch(limited)
    assert result.get_column("order").to_list() == [[5.0]]
    assert result.get_column("fill_price").to_list()[0] == pytest.approx([0.50])

    unlimited = values.with_columns(
        pl.Series("limit", [[0.40]], dtype=pl.Array(pl.Float64, 1))
    )
    missed = node.batch(unlimited)
    assert missed.get_column("order").to_list() == [[0.0]]
    assert missed.get_column("unfilled").to_list() == [[5.0]]


def test_backtest_rejects_unsorted_ladder_and_bad_values() -> None:
    feed = L2Feed(venue(), depth_levels=LEVELS)
    node = BacktestTSFN(
        BacktestConfig(feed=feed, policy=SignalPolicy(), initial_cash=10.0)
    )
    bid_price, bid_volume, ask_price, ask_volume = empty_book(1, 1)
    bid_price[0, 0] = np.array([0.30, 0.40, 0.20, 0.10])
    with pytest.raises(ValueError, match="best-first"):
        node.batch(frame(bid_price, bid_volume, ask_price, ask_volume))

    bid_price, bid_volume, ask_price, ask_volume = empty_book(1, 1)
    bid_volume[0, 0, 0] = -1.0
    with pytest.raises(ValueError, match="non-negative"):
        node.batch(frame(bid_price, bid_volume, ask_price, ask_volume))

    bid_price, bid_volume, ask_price, ask_volume = empty_book(1, 1)
    bid_volume[0, 0, 0] = 1.0
    ask_volume[0, 0, 0] = 1.0
    ask_price[0, 0, 1] = -0.5
    with pytest.raises(ValueError, match="non-negative"):
        node.batch(frame(bid_price, bid_volume, ask_price, ask_volume))


def test_declaration_uses_new_keys_and_rejects_legacy() -> None:
    feed = _feed_from_declaration(
        {"kind": "l2", "venue": {"name": "v", "universe": ["A"]}, "depth_levels": 4}
    )
    assert isinstance(feed, L2Feed)
    assert feed.depth_levels == 4
    with pytest.raises(ValueError, match="legacy L2 keys"):
        _feed_from_declaration(
            {
                "kind": "l2",
                "venue": {"name": "v", "universe": ["A"]},
                "bid_depth_column": "bid_depth",
                "tick": 0.01,
            }
        )
