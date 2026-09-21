from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl

from iosislib.backtest import (
    BacktestConfig,
    BacktestTSFN,
    L1Feed,
    L2Feed,
    SignalPolicy,
    Venue,
)
from _floats import assert_float_close, assert_lists_close


START = datetime(2026, 1, 1)
LEVELS = 4


def l1_frame(
    bid: list[list[float]],
    ask: list[list[float]],
    signal: list[list[float]],
) -> pl.DataFrame:
    width = len(bid[0])
    rows = len(bid)
    return pl.DataFrame(
        [
            pl.Series(
                "timestamp",
                [START + timedelta(minutes=row) for row in range(rows)],
                dtype=pl.Datetime,
            ),
            pl.Series("bid", bid, dtype=pl.Array(pl.Float64, width)),
            pl.Series("ask", ask, dtype=pl.Array(pl.Float64, width)),
            pl.Series("signal", signal, dtype=pl.Array(pl.Float64, width)),
        ]
    )


def l2_frame(
    bid_price: np.ndarray,
    bid_volume: np.ndarray,
    ask_price: np.ndarray,
    ask_volume: np.ndarray,
    signal: list[list[float]],
) -> pl.DataFrame:
    rows, width, levels = bid_price.shape
    assert levels == LEVELS
    assert bid_volume.shape == (rows, width, levels)
    assert ask_price.shape == (rows, width, levels)
    assert ask_volume.shape == (rows, width, levels)
    return pl.DataFrame(
        [
            pl.Series(
                "timestamp",
                [START + timedelta(minutes=row) for row in range(rows)],
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
            pl.Series("signal", signal, dtype=pl.Array(pl.Float64, width)),
        ]
    )


def l2_book(
    rows: int,
    width: int,
    *,
    bid_head: float = 0.40,
    ask_head: float = 0.50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return empty best-first ladders with valid prices and zero volumes."""
    bid_price = np.zeros((rows, width, LEVELS))
    ask_price = np.zeros((rows, width, LEVELS))
    for i in range(LEVELS):
        bid_price[:, :, i] = max(bid_head - 0.10 * i, 0.01)
        ask_price[:, :, i] = min(ask_head + 0.10 * i, 1.0)
    return bid_price, np.zeros((rows, width, LEVELS)), ask_price, np.zeros(
        (rows, width, LEVELS)
    )


def run_l1(
    frame: pl.DataFrame, width: int, initial_cash: float
) -> pl.DataFrame:
    feed = L1Feed(Venue("test", tuple(f"A{i}" for i in range(width))))
    node = BacktestTSFN(
        BacktestConfig(feed=feed, policy=SignalPolicy(), initial_cash=initial_cash)
    )
    return node.batch(frame)


def run_l2(
    frame: pl.DataFrame, width: int, initial_cash: float
) -> pl.DataFrame:
    feed = L2Feed(
        Venue("test", tuple(f"A{i}" for i in range(width))),
        depth_levels=LEVELS,
    )
    node = BacktestTSFN(
        BacktestConfig(feed=feed, policy=SignalPolicy(), initial_cash=initial_cash)
    )
    return node.batch(frame)


def test_l1_blocks_buy_that_would_breach() -> None:
    frame = l1_frame(
        bid=[[0.4], [0.4]],
        ask=[[0.5], [0.5]],
        signal=[[50.0], [101.0]],
    )
    result = run_l1(frame, 1, 10.0)
    assert_lists_close(result.get_column("order").to_list(), [[50.0], [0.0]])
    assert_lists_close(result.get_column("cash").to_list(), [10.0 - 25.0, 10.0 - 25.0])
    assert result.get_column("equity").to_list()[-1] >= 0.0


def test_l1_zero_equity_boundary_triggers_liquidation() -> None:
    frame = l1_frame(
        bid=[[0.4], [0.4]],
        ask=[[0.5], [0.5]],
        signal=[[100.0], [101.0]],
    )
    result = run_l1(frame, 1, 10.0)
    assert_lists_close(result.get_column("order").to_list(), [[100.0], [-100.0]])
    assert_lists_close(result.get_column("cash").to_list(), [10.0 - 50.0, 0.0])
    assert_lists_close(result.get_column("balance").to_list(), [[100.0], [0.0]])


def test_l1_liquidates_on_ruin_then_freezes() -> None:
    frame = l1_frame(
        bid=[[0.9], [0.09], [0.09]],
        ask=[[1.0], [0.1], [0.1]],
        signal=[[10.0], [5.0], [7.0]],
    )
    result = run_l1(frame, 1, 1.0)
    assert_lists_close(result.get_column("order").to_list(), [[10.0], [-10.0], [0.0]])
    assert_lists_close(result.get_column("proposed_order").to_list(), [[10.0], [5.0], [0.0]])
    assert_lists_close(result.get_column("balance").to_list(), [[10.0], [0.0], [0.0]])
    assert_lists_close(result.get_column("cash").to_list(), [-9.0, -8.1, -8.1])


def test_l1_wide_liquidates_longs_and_shorts() -> None:
    width = 8
    frame = l1_frame(
        bid=[[0.9] * width, [0.01] * width, [0.01] * width],
        ask=[[1.0] * width, [0.02] * width, [0.02] * width],
        signal=[[10.0, -10.0] + [0.0] * 6, [0.0] * width, [5.0] * width],
    )
    result = run_l1(frame, width, 1.0)
    assert_lists_close(result.get_column("order").to_list()[1], [-10.0, 10.0] + [0.0] * 6)
    assert_lists_close(result.get_column("order").to_list()[2], [0.0] * width)
    assert_lists_close(result.get_column("balance").to_list()[1], [0.0] * width)
    assert_float_close(result.get_column("cash").to_list()[1], 0.1 - 0.2)
    assert_lists_close(result.get_column("proposed_order").to_list()[2], [0.0] * width)


def test_l1_ruin_with_flat_book_early_returns() -> None:
    frame = l1_frame(
        bid=[[0.9], [0.9]],
        ask=[[1.0], [1.0]],
        signal=[[5.0], [5.0]],
    )
    result = run_l1(frame, 1, 0.0)
    assert_lists_close(result.get_column("order").to_list(), [[0.0], [0.0]])
    assert_lists_close(result.get_column("proposed_order").to_list(), [[5.0], [0.0]])
    assert_lists_close(result.get_column("cash").to_list(), [0.0, 0.0])


def test_l1_wide_gates_assets_in_order() -> None:
    width = 8
    frame = l1_frame(
        bid=[[0.4] * width],
        ask=[[0.5] * width],
        signal=[[101.0] * width],
    )
    result = run_l1(frame, width, 10.5)
    assert_lists_close(result.get_column("order").to_list(), [[101.0] + [0.0] * 7])
    assert_lists_close(result.get_column("cash").to_list(), [10.5 - 50.5])
    assert result.get_column("equity").to_list()[-1] >= 0.0


def test_l2_blocks_fill_on_slippage_breach() -> None:
    bid_price, bid_volume, ask_price, ask_volume = l2_book(1, 1)
    bid_volume[0, 0, 0] = 10.0
    ask_volume[0, 0, 0] = 5.0
    ask_volume[0, 0, 1] = 5.0
    frame = l2_frame(bid_price, bid_volume, ask_price, ask_volume, [[10.0]])
    result = run_l2(frame, 1, 1.0)
    assert_lists_close(result.get_column("order").to_list(), [[0.0]])
    assert_lists_close(result.get_column("unfilled").to_list(), [[10.0]])
    assert_lists_close(result.get_column("cash").to_list(), [1.0])


def test_l2_liquidates_on_ruin_then_freezes() -> None:
    bid_price, bid_volume, ask_price, ask_volume = l2_book(3, 1)
    bid_volume[0, 0, 0] = 10.0
    ask_volume[0, 0, 0] = 10.0
    bid_price[1, 0] = np.array([0.01, 0.01, 0.01, 0.01])
    bid_volume[1, 0, 0] = 10.0
    ask_volume[1, 0, 0] = 10.0
    bid_price[2, 0] = np.array([0.01, 0.01, 0.01, 0.01])
    bid_volume[2, 0, 0] = 10.0
    ask_volume[2, 0, 0] = 10.0
    frame = l2_frame(
        bid_price, bid_volume, ask_price, ask_volume, [[10.0], [3.0], [3.0]]
    )
    result = run_l2(frame, 1, 1.0)
    assert_lists_close(result.get_column("order").to_list(), [[10.0], [-10.0], [0.0]])
    assert_lists_close(result.get_column("proposed_order").to_list(), [[10.0], [3.0], [0.0]])
    assert_lists_close(result.get_column("unfilled").to_list(), [[0.0], [0.0], [0.0]])
    assert_lists_close(result.get_column("fill_price").to_list(), [[0.5], [0.01], [0.0]])
    assert_lists_close(result.get_column("balance").to_list(), [[10.0], [0.0], [0.0]])
    assert_lists_close(result.get_column("cash").to_list(), [-4.0, -3.9, -3.9])


def test_l2_wide_liquidates_through_dispatcher() -> None:
    width = 8
    bid_price, bid_volume, ask_price, ask_volume = l2_book(3, width)
    bid_volume[0, 0, 0] = 10.0
    ask_volume[0, 0, 0] = 10.0
    bid_price[1, 0] = np.array([0.01, 0.01, 0.01, 0.01])
    bid_volume[1, 0, 0] = 10.0
    ask_volume[1, 0, 0] = 10.0
    bid_price[2, 0] = np.array([0.01, 0.01, 0.01, 0.01])
    bid_volume[2, 0, 0] = 10.0
    ask_volume[2, 0, 0] = 10.0
    signal = [[10.0] + [0.0] * 7, [3.0] + [0.0] * 7, [3.0] + [0.0] * 7]
    frame = l2_frame(bid_price, bid_volume, ask_price, ask_volume, signal)
    result = run_l2(frame, width, 1.0)
    assert_float_close(result.get_column("order").to_list()[1][0], -10.0)
    assert_lists_close(result.get_column("order").to_list()[2], [0.0] * width)
    assert_lists_close(result.get_column("proposed_order").to_list()[2], [0.0] * width)
    assert_lists_close(result.get_column("balance").to_list()[1], [0.0] * width)
    assert_lists_close(result.get_column("cash").to_list()[1:], [-3.9, -3.9])


def test_l2_wide_voids_breaching_asset_only() -> None:
    width = 8
    bid_price, bid_volume, ask_price, ask_volume = l2_book(1, width)
    bid_volume[0, :, 0] = 10.0
    bid_price[0, 1, 0] = 0.49
    bid_volume[0, 1, 0] = 10.0
    ask_price[0, 0] = np.array([0.90, 0.95, 1.00, 1.00])
    ask_volume[0, 0, 0] = 50.0
    ask_volume[0, 1, 0] = 10.0
    signal = [[100.0, 1.0] + [0.0] * 6]
    frame = l2_frame(bid_price, bid_volume, ask_price, ask_volume, signal)
    result = run_l2(frame, width, 5.0)
    assert_lists_close(result.get_column("order").to_list(), [[0.0, 1.0] + [0.0] * 6])
    assert_float_close(result.get_column("unfilled").to_list()[0][0], 100.0)
    assert_lists_close(result.get_column("cash").to_list(), [4.5])
    assert result.get_column("equity").to_list()[-1] >= 0.0
