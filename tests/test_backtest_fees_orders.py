from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from iosislib.backtest import (
    BacktestConfig,
    BacktestTSFN,
    FixedFeeSchedule,
    L1Feed,
    L2Feed,
    SignalPolicy,
    Venue,
    list_fee_schedules,
    register_fee_schedule,
)
from iosislib.backtest.fees import FeeSchedule, _fee_from_declaration
from _floats import assert_float_close, assert_lists_close


START = datetime(2026, 1, 1)
LEVELS = 4


def venue(width: int = 1) -> Venue:
    return Venue("test", tuple(f"A{i}" for i in range(width)))


def l1_frame(
    bid: list[list[float]],
    ask: list[list[float]],
    signal: list[list[float]],
    limit: list[list[float]] | None = None,
    cancel: list[list[float]] | None = None,
) -> pl.DataFrame:
    width = len(bid[0])
    rows = len(bid)
    cols = [
        pl.Series(
            "timestamp",
            [START + timedelta(minutes=r) for r in range(rows)],
            dtype=pl.Datetime,
        ),
        pl.Series("bid", bid, dtype=pl.Array(pl.Float64, width)),
        pl.Series("ask", ask, dtype=pl.Array(pl.Float64, width)),
        pl.Series("signal", signal, dtype=pl.Array(pl.Float64, width)),
    ]
    if limit is not None:
        cols.append(pl.Series("limit", limit, dtype=pl.Array(pl.Float64, width)))
    if cancel is not None:
        cols.append(pl.Series("cancel", cancel, dtype=pl.Array(pl.Float64, width)))
    return pl.DataFrame(cols)


def l2_book(rows: int, width: int) -> tuple[np.ndarray, ...]:
    bid_px = np.zeros((rows, width, LEVELS))
    ask_px = np.zeros((rows, width, LEVELS))
    for i in range(LEVELS):
        bid_px[:, :, i] = max(0.40 - 0.10 * i, 0.01)
        ask_px[:, :, i] = min(0.50 + 0.10 * i, 1.0)
    return (
        bid_px,
        np.zeros((rows, width, LEVELS)),
        ask_px,
        np.zeros((rows, width, LEVELS)),
    )


def l2_frame(
    bid_px: np.ndarray,
    bid_vol: np.ndarray,
    ask_px: np.ndarray,
    ask_vol: np.ndarray,
    signal: list[list[float]],
    limit: list[list[float]] | None = None,
    cancel: list[list[float]] | None = None,
) -> pl.DataFrame:
    rows, width, levels = bid_px.shape
    assert levels == LEVELS
    cols = [
        pl.Series(
            "timestamp",
            [START + timedelta(minutes=r) for r in range(rows)],
            dtype=pl.Datetime,
        ),
        pl.Series(
            "bid_price",
            bid_px.reshape(rows, -1).tolist(),
            dtype=pl.Array(pl.Float64, width * levels),
        ),
        pl.Series(
            "bid_volume",
            bid_vol.reshape(rows, -1).tolist(),
            dtype=pl.Array(pl.Float64, width * levels),
        ),
        pl.Series(
            "ask_price",
            ask_px.reshape(rows, -1).tolist(),
            dtype=pl.Array(pl.Float64, width * levels),
        ),
        pl.Series(
            "ask_volume",
            ask_vol.reshape(rows, -1).tolist(),
            dtype=pl.Array(pl.Float64, width * levels),
        ),
        pl.Series("signal", signal, dtype=pl.Array(pl.Float64, width)),
    ]
    if limit is not None:
        cols.append(pl.Series("limit", limit, dtype=pl.Array(pl.Float64, width)))
    if cancel is not None:
        cols.append(pl.Series("cancel", cancel, dtype=pl.Array(pl.Float64, width)))
    return pl.DataFrame(cols)


def run_l1(
    frame: pl.DataFrame, width: int, initial_cash: float = 100.0, **cfg
) -> pl.DataFrame:
    feed = L1Feed(venue(width))
    return BacktestTSFN(
        BacktestConfig(
            feed=feed, policy=SignalPolicy(), initial_cash=initial_cash, **cfg
        )
    ).batch(frame)


def run_l2(
    frame: pl.DataFrame, width: int, initial_cash: float = 100.0, **cfg
) -> pl.DataFrame:
    feed = L2Feed(venue(width), depth_levels=LEVELS)
    return BacktestTSFN(
        BacktestConfig(
            feed=feed, policy=SignalPolicy(), initial_cash=initial_cash, **cfg
        )
    ).batch(frame)


def test_l1_taker_fee_deducted_from_cash() -> None:
    result = run_l1(
        l1_frame([[9.0]], [[10.0]], [[2.0]]),
        1,
        fee_schedule=FixedFeeSchedule(taker_rate=0.01, maker_rate=0.0),
    )
    assert_lists_close(result.get_column("cash").to_list(), [100.0 - 20.0 - 0.2])
    assert_lists_close(result.get_column("fees").to_list(), [0.2])
    assert_lists_close(result.get_column("balance").to_list(), [[2.0]])


def test_no_fee_schedule_reports_zero_fees() -> None:
    result = run_l1(l1_frame([[9.0]], [[10.0]], [[2.0]]), 1)
    assert_lists_close(result.get_column("fees").to_list(), [0.0])
    assert_lists_close(result.get_column("cash").to_list(), [80.0])


def test_maker_rebate_pays_cash_on_rested_fill() -> None:
    fees = FixedFeeSchedule(taker_rate=0.002, maker_rate=-0.001)
    bid_px, bid_vol, ask_px, ask_vol = l2_book(2, 1)
    ask_vol[0, 0, 0] = 10.0
    bid_vol[0, 0, 0] = 10.0
    ask_px[1, 0] = np.array([0.44, 0.50, 0.60, 0.70])
    ask_vol[1, 0, 0] = 10.0
    result = run_l2(
        l2_frame(
            bid_px, bid_vol, ask_px, ask_vol,
            [[10.0], [0.0]], limit=[[0.45], [float("nan")]],
        ),
        1,
        fee_schedule=fees,
        limit_price_column="limit",
    )
    assert_lists_close(result.get_column("order").to_list(), [[0.0], [10.0]])
    assert_lists_close(result.get_column("unfilled").to_list(), [[10.0], [0.0]])
    assert_lists_close(
        result.get_column("open_orders").to_list(),
        [[10.0, 0.0], [0.0, 0.0]],
    )
    assert_lists_close(result.get_column("fill_price").to_list(), [[0.0], [0.44]])
    assert_float_close(result.get_column("fees").to_list()[-1], -10.0 * 0.44 * 0.001)
    assert_float_close(
        result.get_column("cash").to_list()[-1], 100.0 - 4.4 + 0.0044
    )


def test_lean_and_full_paths_agree_without_features() -> None:
    """All-NaN limits force the full path; absent limits take the lean path."""
    rng = np.random.default_rng(11)
    rows, width = 50, 8
    bid = np.round(rng.uniform(0.3, 0.7, (rows, width)), 4).tolist()
    ask = (np.round(np.array(bid) + 0.05, 4)).tolist()
    signal = np.round(rng.normal(0, 8, (rows, width)), 2).tolist()
    plain = l1_frame(bid, ask, signal)
    limited = l1_frame(
        bid, ask, signal, limit=[[float("nan")] * width for _ in range(rows)]
    )
    lean = run_l1(plain, width)
    full = run_l1(limited, width, limit_price_column="limit")
    for column in (
        "cash", "equity", "balance", "order", "proposed_order",
        "fill_price", "unfilled", "fees", "open_orders",
    ):
        assert_lists_close(
            lean.get_column(column).to_list(), full.get_column(column).to_list()
        )


def test_fee_schedule_declaration_and_registry() -> None:
    config = BacktestConfig(
        feed=L1Feed(venue()),
        policy=SignalPolicy(),
        initial_cash=1.0,
        fee_schedule={"kind": "fixed", "taker_rate": 0.01, "maker_rate": -0.002},
    )
    assert isinstance(config.fee_schedule, FixedFeeSchedule)
    assert config.fee_schedule.taker_rate == 0.01
    assert "fixed" in list_fee_schedules()
    with pytest.raises(ValueError, match="Unknown fee schedule kind"):
        _fee_from_declaration({"kind": "nope"})
    with pytest.raises(ValueError, match="already registered"):
        register_fee_schedule("fixed", FixedFeeSchedule)
    with pytest.raises(TypeError, match="must be a number"):
        FixedFeeSchedule(taker_rate=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be finite"):
        FixedFeeSchedule(maker_rate=float("inf"))
    with pytest.raises(TypeError, match="fee_schedule must be"):
        BacktestConfig(
            feed=L1Feed(venue()), policy=SignalPolicy(), initial_cash=1.0,
            fee_schedule="nope",  # type: ignore[arg-type]
        )


def test_fee_schedule_enters_node_identity() -> None:
    from iosislib.core.node import Node

    base = Node(
        BacktestTSFN,
        config=BacktestConfig(
            feed=L1Feed(venue()), policy=SignalPolicy(), initial_cash=100.0
        ),
    )
    with_fees = Node(
        BacktestTSFN,
        config=BacktestConfig(
            feed=L1Feed(venue()), policy=SignalPolicy(), initial_cash=100.0,
            fee_schedule=FixedFeeSchedule(taker_rate=0.01),
        ),
    )
    same_fees = Node(
        BacktestTSFN,
        config=BacktestConfig(
            feed=L1Feed(venue()), policy=SignalPolicy(), initial_cash=100.0,
            fee_schedule={"kind": "fixed", "taker_rate": 0.01},
        ),
    )
    assert base.ID != with_fees.ID
    assert with_fees == same_fees
    assert isinstance(list_fee_schedules()["fixed"], type)
    assert issubclass(list_fee_schedules()["fixed"], FeeSchedule)


def test_l1_spread_fraction_slippage() -> None:
    result = run_l1(
        l1_frame([[9.0], [9.0]], [[10.0], [10.0]], [[2.0], [-1.0]]),
        1,
        slippage_spread_fraction=0.5,
    )
    assert_lists_close(result.get_column("fill_price").to_list(), [[10.5], [8.5]])
    assert_lists_close(
        result.get_column("cash").to_list(), [100.0 - 21.0, 100.0 - 21.0 + 8.5]
    )
    with pytest.raises(ValueError, match="non-negative"):
        run_l1(l1_frame([[9.0]], [[10.0]], [[1.0]]), 1, slippage_spread_fraction=-0.1)
    with pytest.raises(ValueError, match="non-negative"):
        run_l1(
            l1_frame([[9.0]], [[10.0]], [[1.0]]), 1,
            slippage_spread_fraction=float("nan"),
        )
    with pytest.raises(TypeError, match="must be a number"):
        run_l1(
            l1_frame([[9.0]], [[10.0]], [[1.0]]), 1,
            slippage_spread_fraction="x",  # type: ignore[arg-type]
        )


def test_l1_equity_gate_accounts_for_slippage() -> None:
    frame = l1_frame([[0.4]], [[0.5]], [[20.0]])
    fills = run_l1(frame, 1, 3.0, slippage_spread_fraction=0.0)
    assert_lists_close(fills.get_column("order").to_list(), [[20.0]])
    blocked = run_l1(frame, 1, 3.0, slippage_spread_fraction=1.0)
    assert_lists_close(blocked.get_column("order").to_list(), [[0.0]])
    assert_lists_close(blocked.get_column("unfilled").to_list(), [[20.0]])


def test_l1_limit_rests_then_fills() -> None:
    result = run_l1(
        l1_frame(
            [[0.4], [0.4]], [[0.5], [0.42]], [[10.0], [0.0]],
            limit=[[0.45], [float("nan")]],
        ),
        1,
        limit_price_column="limit",
    )
    assert_lists_close(result.get_column("order").to_list(), [[0.0], [10.0]])
    assert_lists_close(result.get_column("unfilled").to_list(), [[10.0], [0.0]])
    assert_lists_close(
        result.get_column("open_orders").to_list(),
        [[10.0, 0.0], [0.0, 0.0]],
    )
    assert_lists_close(result.get_column("fill_price").to_list(), [[0.0], [0.45]])
    assert_lists_close(result.get_column("cash").to_list(), [100.0, 95.5])


def test_l1_limit_capped_by_slippage() -> None:
    result = run_l1(
        l1_frame([[9.0]], [[10.0]], [[2.0]], limit=[[10.2]]),
        1,
        limit_price_column="limit",
        slippage_spread_fraction=0.5,
    )
    assert_lists_close(result.get_column("order").to_list(), [[0.0]])
    assert_lists_close(
        result.get_column("open_orders").to_list(), [[2.0, 0.0]]
    )


def test_l1_sell_limit_mirror() -> None:
    result = run_l1(
        l1_frame(
            [[9.0], [10.5]], [[10.0], [11.0]], [[-4.0], [0.0]],
            limit=[[10.2], [float("nan")]],
        ),
        1,
        limit_price_column="limit",
    )
    assert_lists_close(result.get_column("order").to_list(), [[0.0], [-4.0]])
    assert_lists_close(result.get_column("fill_price").to_list(), [[0.0], [10.2]])
    assert_lists_close(result.get_column("cash").to_list(), [100.0, 100.0 + 40.8])


def test_cancel_clears_working_orders() -> None:
    result = run_l1(
        l1_frame(
            [[0.4], [0.4]], [[0.5], [0.42]], [[10.0], [0.0]],
            limit=[[0.45], [float("nan")]], cancel=[[0.0], [1.0]],
        ),
        1,
        limit_price_column="limit",
        cancel_column="cancel",
    )
    assert_lists_close(
        result.get_column("open_orders").to_list(),
        [[10.0, 0.0], [0.0, 0.0]],
    )
    assert_lists_close(result.get_column("balance").to_list(), [[0.0], [0.0]])
    assert_lists_close(result.get_column("cash").to_list(), [100.0, 100.0])


def test_cancel_column_validation() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        run_l1(l1_frame([[0.4]], [[0.5]], [[1.0]]), 1, cancel_column="")
    with pytest.raises(ValueError, match="must differ"):
        run_l1(
            l1_frame([[0.4]], [[0.5]], [[1.0]]), 1,
            limit_price_column="x", cancel_column="x",
        )
    function = BacktestTSFN(
        BacktestConfig(
            feed=L1Feed(venue()), policy=SignalPolicy(), initial_cash=100.0,
            cancel_column="cancel",
        )
    )
    with pytest.raises(ValueError, match="Missing required input column: 'cancel'"):
        function.batch(l1_frame([[0.4]], [[0.5]], [[1.0]]))


def test_cancel_replace_supersedes_working() -> None:
    bid_px, bid_vol, ask_px, ask_vol = l2_book(2, 1)
    ask_vol[:, 0, 0] = 10.0
    result = run_l2(
        l2_frame(
            bid_px, bid_vol, ask_px, ask_vol,
            [[10.0], [4.0]], limit=[[0.45], [0.46]],
        ),
        1,
        limit_price_column="limit",
    )
    assert_lists_close(
        result.get_column("open_orders").to_list(),
        [[10.0, 0.0], [4.0, 0.0]],
    )
    assert_lists_close(result.get_column("order").to_list(), [[0.0], [0.0]])


def test_l2_partial_working_fill_keeps_remainder() -> None:
    bid_px, bid_vol, ask_px, ask_vol = l2_book(2, 1)
    ask_vol[0, 0, 0] = 10.0
    ask_px[1, 0] = np.array([0.44, 0.50, 0.60, 0.70])
    ask_vol[1, 0, 0] = 4.0
    result = run_l2(
        l2_frame(
            bid_px, bid_vol, ask_px, ask_vol,
            [[10.0], [0.0]], limit=[[0.45], [float("nan")]],
        ),
        1,
        limit_price_column="limit",
    )
    assert_lists_close(result.get_column("order").to_list(), [[0.0], [4.0]])
    assert_lists_close(
        result.get_column("open_orders").to_list(),
        [[10.0, 0.0], [6.0, 0.0]],
    )
    assert_lists_close(result.get_column("balance").to_list(), [[0.0], [4.0]])


def test_l2_working_fill_blocked_on_breach_stays_open() -> None:
    bid_px, bid_vol, ask_px, ask_vol = l2_book(2, 1)
    bid_px[1, 0] = np.array([0.01, 0.01, 0.01, 0.01])
    ask_px[1] = np.array([0.40, 0.50, 0.60, 0.70])
    ask_vol[1, 0, 0] = 10.0
    result = run_l2(
        l2_frame(
            bid_px, bid_vol, ask_px, ask_vol,
            [[10.0], [0.0]], limit=[[0.45], [float("nan")]],
        ),
        1,
        1.0,
        limit_price_column="limit",
    )
    assert_lists_close(result.get_column("order").to_list(), [[0.0], [0.0]])
    assert_lists_close(
        result.get_column("open_orders").to_list(),
        [[10.0, 0.0], [10.0, 0.0]],
    )
    assert_lists_close(result.get_column("balance").to_list(), [[0.0], [0.0]])


def test_ruin_liquidates_and_clears_working() -> None:
    bid_px, bid_vol, ask_px, ask_vol = l2_book(2, 1)
    ask_vol[0, 0, 0] = 4.0
    bid_px[1, 0] = np.array([0.01, 0.01, 0.01, 0.01])
    bid_vol[1, 0, 0] = 100.0
    ask_vol[1, 0, 0] = 100.0
    result = BacktestTSFN(
        BacktestConfig(
            feed=L2Feed(venue(), depth_levels=LEVELS),
            policy=SignalPolicy(),
            initial_cash=1.0,
            fee_schedule=FixedFeeSchedule(taker_rate=0.01),
            limit_price_column="limit",
        )
    ).batch(
        l2_frame(
            bid_px, bid_vol, ask_px, ask_vol,
            [[10.0], [0.0]], limit=[[0.50], [float("nan")]],
        )
    )
    assert_lists_close(result.get_column("order").to_list(), [[4.0], [-4.0]])
    assert_lists_close(result.get_column("balance").to_list(), [[4.0], [0.0]])
    assert_lists_close(
        result.get_column("open_orders").to_list(),
        [[6.0, 0.0], [0.0, 0.0]],
    )
    assert_float_close(result.get_column("fees").to_list()[-1], 4 * 0.5 * 0.01 + 4 * 0.01 * 0.01)


def test_wide_l1_limits_fees_and_working() -> None:
    width = 8
    result = run_l1(
        l1_frame(
            [[0.4] * width, [0.4] * width],
            [[0.5] * width, [0.42] * width],
            [[10.0] + [0.0] * 7, [0.0] * width],
            limit=[[0.45] + [float("nan")] * 7, [float("nan")] * width],
        ),
        width,
        limit_price_column="limit",
        fee_schedule=FixedFeeSchedule(taker_rate=0.01, maker_rate=0.005),
    )
    assert_lists_close(
        result.get_column("open_orders").to_list()[0], [10.0, 0.0] + [0.0] * 14
    )
    assert_lists_close(
        result.get_column("open_orders").to_list()[1], [0.0] * (2 * width)
    )
    assert_lists_close(result.get_column("order").to_list()[1], [10.0] + [0.0] * 7)
    assert_float_close(result.get_column("fees").to_list()[-1], 10 * 0.45 * 0.005)
    assert_float_close(result.get_column("cash").to_list()[-1], 100.0 - 4.5 - 0.0225)


def test_wide_l2_working_and_maker_fees() -> None:
    width = 8
    bid_px, bid_vol, ask_px, ask_vol = l2_book(2, width)
    ask_vol[0, 0, 0] = 10.0
    ask_px[1, 0] = np.array([0.44, 0.50, 0.60, 0.70])
    ask_vol[1, 0, 0] = 10.0
    signal = [[10.0] + [0.0] * 7, [0.0] * width]
    limit = [[0.45] + [float("nan")] * 7, [float("nan")] * width]
    result = run_l2(
        l2_frame(bid_px, bid_vol, ask_px, ask_vol, signal, limit=limit),
        width,
        limit_price_column="limit",
        fee_schedule=FixedFeeSchedule(taker_rate=0.002, maker_rate=-0.001),
    )
    assert_lists_close(result.get_column("order").to_list()[1], [10.0] + [0.0] * 7)
    assert_float_close(result.get_column("fill_price").to_list()[1][0], 0.44)
    assert_float_close(result.get_column("fees").to_list()[-1], -10.0 * 0.44 * 0.001)
    assert_lists_close(
        result.get_column("open_orders").to_list()[1], [0.0] * (2 * width)
    )
