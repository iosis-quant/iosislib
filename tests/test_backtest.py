from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from iosislib.backtest import (
    Array,
    BacktestConfig,
    BacktestTSFN,
    Feed,
    L1Feed,
    MarketState,
    Order,
    Policy,
    PolicyState,
    StatefulPolicy,
    Venue,
)
from iosislib.core.graph import Graph
from iosislib.core.node import Node
from iosislib.core.tsfn import FrameSignature, TSFN, TSFNConfig, TimeAxis


START = datetime(2026, 1, 1)


def market_frame(
    bid: list[list[float]],
    ask: list[list[float]],
    signal: list[list[float]],
    *,
    timestamps: list[datetime] | None = None,
) -> pl.DataFrame:
    width = len(bid[0]) if bid else 1
    rows = len(bid)
    return pl.DataFrame(
        [
            pl.Series(
                "timestamp",
                timestamps or [START + timedelta(minutes=row) for row in range(rows)],
                dtype=pl.Datetime,
            ),
            pl.Series("bid", bid, dtype=pl.Array(pl.Float64, width)),
            pl.Series("ask", ask, dtype=pl.Array(pl.Float64, width)),
            pl.Series("signal", signal, dtype=pl.Array(pl.Float64, width)),
        ]
    )


def l1_feed(width: int = 1, *, time_axis: TimeAxis = TimeAxis()) -> L1Feed:
    return L1Feed(
        Venue("test", tuple(f"A{index}" for index in range(width))),
        time_axis=time_axis,
    )


@dataclass(frozen=True)
class InvalidQuoteFeed(Feed):
    VERSION = "1.0.0"

    @property
    def columns(self) -> tuple[tuple[str, pl.DataType, tuple[int, ...]], ...]:
        return (("bid", pl.Float64, (self.width,)), ("ask", pl.Float64, (self.width,)))

    def quotes(self, frame: pl.DataFrame) -> tuple[pl.Series, pl.Series]:
        return frame.get_column("bid"), "not-a-series"  # type: ignore[return-value]


@dataclass(frozen=True)
class SignalOrderPolicy(Policy):
    VERSION = "1.0.0"

    def decide(self, policy_state, state: MarketState, cash: float, balances: Array, orders: Array, row: int) -> None:
        del policy_state, cash, balances
        orders[row] = state.information


@dataclass(frozen=True)
class CashAwarePolicy(Policy):
    VERSION = "1.0.0"

    def decide(self, policy_state, state: MarketState, cash: float, balances: Array, orders: Array, row: int) -> None:
        del policy_state, state, balances
        orders[row] = np.array([1.0 if cash == 100.0 else 2.0], dtype=np.float64)


@dataclass(frozen=True)
class WrongWidthPolicy(Policy):
    VERSION = "1.0.0"

    def decide(self, policy_state, state: MarketState, cash: float, balances: Array, orders: Array, row: int) -> None:
        del policy_state, state, cash, balances
        orders[row] = np.array([1.0, 2.0], dtype=np.float64)


@dataclass(frozen=True)
class MixedEightAssetPolicy(Policy):
    VERSION = "1.0.0"

    def decide(self, policy_state, state: MarketState, cash: float, balances: Array, orders: Array, row: int) -> None:
        del policy_state, state, cash, balances
        orders[row] = np.array([1.0, -1.0] * 4, dtype=np.float64)


@dataclass(frozen=True)
class CounterState(PolicyState):
    tick: int = 0


@dataclass(frozen=True)
class CounterPolicy(StatefulPolicy):
    VERSION = "1.0.0"

    def initial_state(self) -> PolicyState:
        return CounterState()

    def decide(self, policy_state, state: MarketState, cash: float, balances: Array, orders: Array, row: int) -> PolicyState:
        del state, cash, balances
        assert isinstance(policy_state, CounterState)
        orders[row] = float(policy_state.tick + 1)
        return CounterState(policy_state.tick + 1)


@dataclass(frozen=True)
class InvalidStatePolicy(StatefulPolicy):
    VERSION = "1.0.0"

    def initial_state(self) -> PolicyState:
        return None  # type: ignore[return-value]

    def decide(self, policy_state, state: MarketState, cash: float, balances: Array, orders: Array, row: int) -> PolicyState:
        del policy_state, state, cash, balances, orders, row
        return PolicyState()


@dataclass(frozen=True)
class InvalidNextStatePolicy(StatefulPolicy):
    VERSION = "1.0.0"

    def initial_state(self) -> PolicyState:
        return PolicyState()

    def decide(self, policy_state, state: MarketState, cash: float, balances: Array, orders: Array, row: int) -> PolicyState:
        del policy_state, state, cash, balances, orders, row
        return None  # type: ignore[return-value]


@dataclass(frozen=True)
class MutatingBalancePolicy(Policy):
    VERSION = "1.0.0"

    def decide(self, policy_state, state: MarketState, cash: float, balances: Array, orders: Array, row: int) -> None:
        del policy_state, state, cash
        balances[0] = 1.0


@dataclass(frozen=True)
class MutatingMarketPolicy(Policy):
    VERSION = "1.0.0"

    def decide(self, policy_state, state: MarketState, cash: float, balances: Array, orders: Array, row: int) -> None:
        del policy_state, cash, balances
        state.bid[0] = 1.0


@dataclass(frozen=True)
class TapeConfig(TSFNConfig):
    pass


class Tape(TSFN[TapeConfig]):
    VERSION = "1.0.0"
    CONFIG_CLS = TapeConfig

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        return (
            FrameSignature.empty(),
            FrameSignature(
                columns=(
                    ("bid", pl.Float64, (1,)),
                    ("ask", pl.Float64, (1,)),
                    ("signal", pl.Float64, (1,)),
                )
            ),
        )

    def apply(self) -> pl.LazyFrame:
        return market_frame([[9.0], [10.0]], [[10.0], [11.0]], [[1.0], [-1.0]]).lazy()


def backtest(
    policy: Policy, *, width: int = 1, initial_cash: float = 100.0
) -> BacktestTSFN:
    return BacktestTSFN(
        BacktestConfig(feed=l1_feed(width), policy=policy, initial_cash=initial_cash)
    )


def test_venue_validates_an_ordered_non_empty_unique_universe() -> None:
    venue = Venue("markets", ("yes", "no"))

    assert venue.width == 2
    assert venue.to_dict()["universe"] == ["yes", "no"]
    assert '"name":"markets"' in str(venue)

    with pytest.raises(ValueError, match="non-empty string"):
        Venue("", ("yes",))
    with pytest.raises(ValueError, match="non-empty tuple"):
        Venue("markets", ())
    with pytest.raises(ValueError, match="unique"):
        Venue("markets", ("yes", "yes"))
    with pytest.raises(ValueError, match="non-empty strings"):
        Venue("markets", ("",))


def test_l1_feed_declares_quotes_extracts_them_and_serializes_configuration() -> None:
    feed = L1Feed(
        Venue("markets", ("a", "b")), bid_column="best_bid", ask_column="best_ask"
    )
    values = pl.DataFrame(
        {"best_bid": [[1.0, 2.0]], "best_ask": [[1.1, 2.1]]},
        schema={
            "best_bid": pl.Array(pl.Float64, 2),
            "best_ask": pl.Array(pl.Float64, 2),
        },
    )

    assert feed.columns == (
        ("best_bid", pl.Float64, (2,)),
        ("best_ask", pl.Float64, (2,)),
    )
    assert feed.frame_signature().time == TimeAxis()
    assert [series.name for series in feed.quotes(values)] == ["best_bid", "best_ask"]
    assert feed.to_dict()["bid_column"] == "best_bid"
    assert "iosislib.backtest.feeds.L1Feed" in str(feed)

    with pytest.raises(ValueError, match="cannot be empty"):
        L1Feed(Venue("markets", ("a",)), bid_column="")
    with pytest.raises(ValueError, match="must differ"):
        L1Feed(Venue("markets", ("a",)), bid_column="quote", ask_column="quote")


def test_backtest_node_identity_tracks_executable_configuration() -> None:
    first = Node(
        BacktestTSFN,
        config=BacktestConfig(
            feed=l1_feed(), policy=SignalOrderPolicy(), initial_cash=100.0
        ),
    )
    equivalent = Node(
        BacktestTSFN,
        config=BacktestConfig(
            feed=l1_feed(), policy=SignalOrderPolicy(), initial_cash=100.0
        ),
    )
    changed_cash = Node(
        BacktestTSFN,
        config=BacktestConfig(
            feed=l1_feed(), policy=SignalOrderPolicy(), initial_cash=101.0
        ),
    )
    changed_venue = Node(
        BacktestTSFN,
        config=BacktestConfig(
            feed=L1Feed(Venue("other", ("A0",))),
            policy=SignalOrderPolicy(),
            initial_cash=100.0,
        ),
    )

    assert first == equivalent
    assert first.ID == equivalent.ID
    assert first.ID != changed_cash.ID
    assert first.ID != changed_venue.ID


def test_backtest_config_rejects_invalid_runtime_declarations() -> None:
    with pytest.raises(TypeError, match="feed must be a Feed or a declarative mapping"):
        BacktestConfig(feed=None, policy=SignalOrderPolicy(), initial_cash=1.0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="policy must be a Policy or a declarative mapping"):
        BacktestConfig(feed=l1_feed(), policy=None, initial_cash=1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="initial_cash must be finite"):
        BacktestConfig(
            feed=l1_feed(), policy=SignalOrderPolicy(), initial_cash=float("nan")
        )


def test_immediate_orders_update_cash_balances_and_equity() -> None:
    result = backtest(SignalOrderPolicy()).batch(
        market_frame([[9.0], [10.0]], [[10.0], [11.0]], [[2.0], [-1.0]])
    )

    assert result.get_column("cash").to_list() == [80.0, 90.0]
    assert result.get_column("balance").to_list() == [[2.0], [1.0]]
    assert result.get_column("order").to_list() == [[2.0], [-1.0]]
    assert result.get_column("equity").to_list() == [98.0, 100.0]


def test_policy_observes_the_previous_completed_portfolio() -> None:
    result = backtest(CashAwarePolicy()).batch(
        market_frame([[9.0], [10.0]], [[10.0], [11.0]], [[0.0], [0.0]])
    )

    assert result.get_column("cash").to_list() == [90.0, 68.0]
    assert result.get_column("balance").to_list() == [[1.0], [3.0]]


def test_stateful_policy_state_is_fresh_for_each_execution() -> None:
    values = market_frame([[9.0], [10.0]], [[10.0], [11.0]], [[0.0], [0.0]])
    function = backtest(CounterPolicy())

    first = function.batch(values)
    second = function.batch(values)

    assert first.equals(second)
    assert first.get_column("order").to_list() == [[1.0], [2.0]]


def test_policies_cannot_mutate_market_input_arrays() -> None:
    values = market_frame([[9.0]], [[10.0]], [[0.0]])

    with pytest.raises(ValueError, match="read-only"):
        backtest(MutatingMarketPolicy()).batch(values)


def test_no_risk_constraints_are_implicitly_applied() -> None:
    result = backtest(SignalOrderPolicy()).batch(
        market_frame([[9.0]], [[10.0]], [[20.0]])
    )

    assert result.get_column("cash").to_list() == [-100.0]
    assert result.get_column("balance").to_list() == [[20.0]]


def test_wide_execution_uses_bid_for_sells_and_ask_for_buys() -> None:
    bid = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
    ask = bid + 1.0
    result = backtest(MixedEightAssetPolicy(), width=8, initial_cash=1_000.0).batch(
        market_frame([bid.tolist()], [ask.tolist()], [np.zeros(8).tolist()])
    )

    quantities = np.array([1.0, -1.0] * 4)
    expected_cash = 1_000.0 - float(
        np.dot(quantities, np.where(quantities >= 0.0, ask, bid))
    )
    assert result.get_column("cash").to_list() == [expected_cash]
    assert result.get_column("balance").to_list() == [quantities.tolist()]
    assert result.get_column("equity").to_list() == [
        expected_cash + float(np.dot(quantities, bid))
    ]


def test_policy_wrong_width_write_fails_loudly() -> None:
    with pytest.raises(ValueError, match="could not broadcast"):
        backtest(WrongWidthPolicy()).batch(market_frame([[9.0]], [[10.0]], [[0.0]]))


def test_stateful_policy_contracts_fail_loudly() -> None:
    with pytest.raises(TypeError, match="initial_state"):
        backtest(InvalidStatePolicy()).batch(market_frame([[9.0]], [[10.0]], [[0.0]]))


def test_graph_execution_uses_the_backtest_batch_loop() -> None:
    tape = Node(Tape)
    simulation = Node(
        BacktestTSFN,
        bindings={"bid": tape.bid, "ask": tape.ask, "signal": tape.signal},
        config=BacktestConfig(
            feed=l1_feed(), policy=SignalOrderPolicy(), initial_cash=100.0
        ),
    )

    result = Graph(simulation).execute()

    assert result.columns == [
        "timestamp",
        "cash",
        "equity",
        "balance",
        "order",
        "proposed_order",
    ]
    assert result.get_column("balance").to_list() == [[1.0], [0.0]]


def test_backtest_rejects_malformed_custom_feed_quotes() -> None:
    function = BacktestTSFN(
        BacktestConfig(
            feed=InvalidQuoteFeed(Venue("test", ("A0",))),
            policy=SignalOrderPolicy(),
            initial_cash=100.0,
        )
    )

    with pytest.raises(TypeError, match="Feed.quotes must return"):
        function.batch(market_frame([[9.0]], [[10.0]], [[0.0]]))


def test_backtest_validates_time_order_and_quote_signal_values() -> None:
    function = backtest(SignalOrderPolicy())
    good = market_frame([[9.0], [10.0]], [[10.0], [11.0]], [[0.0], [0.0]])

    with pytest.raises(ValueError, match="strictly increasing"):
        function.batch(
            good.with_columns(pl.Series("timestamp", [START, START], dtype=pl.Datetime))
        )
    with pytest.raises(ValueError, match="non-null and sorted"):
        function.batch(good.sort("timestamp", descending=True))
    with pytest.raises(ValueError, match="bid cannot exceed ask"):
        function.batch(market_frame([[11.0]], [[10.0]], [[0.0]]))
    with pytest.raises(ValueError, match="bid and ask must be positive"):
        function.batch(market_frame([[0.0]], [[10.0]], [[0.0]]))
    with pytest.raises(ValueError, match="signal must be finite"):
        function.batch(market_frame([[9.0]], [[10.0]], [[np.nan]]))


def test_backtest_validates_input_schema_and_empty_batches() -> None:
    function = backtest(SignalOrderPolicy())
    good = market_frame([[9.0]], [[10.0]], [[0.0]])

    with pytest.raises(ValueError, match="Missing required input column: 'signal'"):
        function.batch(good.drop("signal"))
    with pytest.raises(TypeError, match="Column 'signal' type mismatch"):
        function.batch(
            good.with_columns(pl.col("signal").cast(pl.Array(pl.Float32, 1)))
        )
    with pytest.raises(TypeError, match="Time column 'timestamp' type mismatch"):
        function.batch(good.with_columns(pl.col("timestamp").cast(pl.Date)))

    empty = pl.DataFrame(
        [
            pl.Series("timestamp", [], dtype=pl.Datetime),
            pl.Series("bid", [], dtype=pl.Array(pl.Float64, 1)),
            pl.Series("ask", [], dtype=pl.Array(pl.Float64, 1)),
            pl.Series("signal", [], dtype=pl.Array(pl.Float64, 1)),
        ]
    )
    result = function.batch(empty)
    assert dict(result.schema) == {
        "timestamp": pl.Datetime,
        "cash": pl.Float64,
        "equity": pl.Float64,
        "balance": pl.Array(pl.Float64, 1),
        "order": pl.Array(pl.Float64, 1),
        "proposed_order": pl.Array(pl.Float64, 1),
    }
    assert result.height == 0
