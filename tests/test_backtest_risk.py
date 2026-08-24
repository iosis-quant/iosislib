from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from iosislib.backtest import (
    BacktestConfig,
    BacktestTSFN,
    FractionalKellyPolicy,
    FractionalLimitPolicy,
    NO_OP_RISK,
    NoOpRiskPolicy,
    Order,
    Policy,
    PolicyState,
    RiskDecision,
    RiskPolicy,
    RiskReason,
    StatefulRiskPolicy,
    Venue,
    classify_reason,
)
from iosislib.backtest.feeds import L1Feed
from iosislib.backtest.policy import Array, MarketState
from iosislib.core.graph import Graph
from iosislib.core.node import Node
from iosislib.core.tsfn import FrameSignature, TSFN, TSFNConfig, TimeAxis

START = datetime(2026, 1, 1)


def market_frame(
    bid: list[list[float]],
    ask: list[list[float]],
    signal: list[list[float]],
) -> pl.DataFrame:
    width = len(bid[0])
    return pl.DataFrame(
        [
            pl.Series(
                "timestamp",
                [START + timedelta(minutes=row) for row in range(len(bid))],
                dtype=pl.Datetime,
            ),
            pl.Series("bid", bid, dtype=pl.Array(pl.Float64, width)),
            pl.Series("ask", ask, dtype=pl.Array(pl.Float64, width)),
            pl.Series("signal", signal, dtype=pl.Array(pl.Float64, width)),
        ]
    )


def l1_feed(width: int = 1) -> L1Feed:
    return L1Feed(
        Venue("test", tuple(f"A{index}" for index in range(width))),
        time_axis=TimeAxis(),
    )


@dataclass(frozen=True)
class SignalOrderPolicy(Policy):
    VERSION = "1.0.0"

    def decide(self, policy_state, state: MarketState, cash: float, balances: Array, orders: Array, row: int) -> None:
        del policy_state, cash, balances
        orders[row] = state.information


def backtest(
    policy: Policy,
    *,
    risk_policy: RiskPolicy | None = None,
    width: int = 1,
    initial_cash: float = 100.0,
    validate: bool = True,
) -> BacktestTSFN:
    return BacktestTSFN(
        BacktestConfig(
            feed=l1_feed(width),
            policy=policy,
            initial_cash=initial_cash,
            risk_policy=risk_policy,
            validate=validate,
        )
    )


def test_no_risk_policy_passes_orders_through_unchanged() -> None:
    result = backtest(SignalOrderPolicy()).batch(
        market_frame([[9.0], [10.0]], [[10.0], [11.0]], [[2.0], [-1.0]])
    )

    assert result.get_column("order").to_list() == [[2.0], [-1.0]]
    assert result.get_column("proposed_order").to_list() == [[2.0], [-1.0]]


def test_explicit_noop_risk_policy_equals_null_risk_path() -> None:
    values = market_frame([[9.0], [10.0]], [[10.0], [11.0]], [[2.0], [-1.0]])

    implicit = backtest(SignalOrderPolicy()).batch(values)
    explicit = backtest(
        SignalOrderPolicy(), risk_policy=NoOpRiskPolicy()
    ).batch(values)
    singleton = backtest(SignalOrderPolicy(), risk_policy=NO_OP_RISK).batch(values)

    assert implicit.equals(explicit)
    assert implicit.equals(singleton)


def test_fractional_limit_policy_clamps_positions_to_fraction_of_equity() -> None:
    result = backtest(
        SignalOrderPolicy(), risk_policy=FractionalLimitPolicy(0.5)
    ).batch(market_frame([[9.0]], [[10.0]], [[20.0]]))

    assert result.get_column("proposed_order").to_list() == [[20.0]]
    assert result.get_column("order").to_list() == [[5.0]]
    assert result.get_column("balance").to_list() == [[5.0]]
    assert result.get_column("cash").to_list() == [50.0]
    assert result.get_column("equity").to_list() == [95.0]


def test_fractional_limit_policy_clamps_only_exceeding_assets() -> None:
    result = backtest(
        SignalOrderPolicy(), risk_policy=FractionalLimitPolicy(0.5), width=2
    ).batch(
        market_frame([[9.0, 19.0]], [[10.0, 20.0]], [[20.0, 1.0]])
    )

    assert result.get_column("order").to_list() == [[5.0, 1.0]]


def test_fractional_limit_policy_no_change_when_within_limit() -> None:
    result = backtest(
        SignalOrderPolicy(), risk_policy=FractionalLimitPolicy(0.5)
    ).batch(market_frame([[9.0]], [[10.0]], [[2.0]]))

    assert result.get_column("order").to_list() == [[2.0]]


def test_fractional_limit_policy_zeroes_orders_at_nonpositive_equity() -> None:
    result = backtest(
        SignalOrderPolicy(),
        risk_policy=FractionalLimitPolicy(0.5),
        initial_cash=-10.0,
    ).batch(market_frame([[9.0]], [[10.0]], [[20.0]]))

    assert result.get_column("order").to_list() == [[0.0]]
    assert result.get_column("proposed_order").to_list() == [[20.0]]
    assert result.get_column("cash").to_list() == [-10.0]


def test_fractional_limit_policy_validates_fraction() -> None:
    with pytest.raises(ValueError, match="finite"):
        FractionalLimitPolicy(float("nan"))
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        FractionalLimitPolicy(0.0)
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        FractionalLimitPolicy(1.5)


def test_fractional_kelly_policy_sizes_from_probability_and_ask() -> None:
    result = backtest(
        SignalOrderPolicy(), risk_policy=FractionalKellyPolicy(1.0)
    ).batch(market_frame([[0.5]], [[0.6]], [[0.7]]))

    expected = 25.0 / 0.6
    assert result.get_column("order").to_list() == [[pytest.approx(expected)]]
    assert result.get_column("proposed_order").to_list() == [[0.7]]


def test_fractional_kelly_policy_scales_by_custom_fraction_and_is_deterministic() -> None:
    values = market_frame([[0.5]], [[0.6]], [[0.7]])
    half = backtest(
        SignalOrderPolicy(), risk_policy=FractionalKellyPolicy(0.5)
    ).batch(values)
    full = backtest(
        SignalOrderPolicy(), risk_policy=FractionalKellyPolicy(1.0)
    ).batch(values)
    repeat = backtest(
        SignalOrderPolicy(), risk_policy=FractionalKellyPolicy(0.5)
    ).batch(values)

    assert half.equals(repeat)
    assert half.get_column("order").to_list() == [
        [pytest.approx(full.get_column("order")[0] / 2.0)]
    ]


def test_fractional_kelly_policy_zeroes_orders_without_edge() -> None:
    result = backtest(
        SignalOrderPolicy(), risk_policy=FractionalKellyPolicy(1.0)
    ).batch(market_frame([[0.5]], [[0.7]], [[0.5]]))

    assert result.get_column("order").to_list() == [[0.0]]


def test_fractional_kelly_policy_validates_fraction() -> None:
    with pytest.raises(ValueError, match="finite"):
        FractionalKellyPolicy(float("nan"))
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        FractionalKellyPolicy(0.0)
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        FractionalKellyPolicy(2.0)


@dataclass(frozen=True)
class ZeroingState(PolicyState):
    tick: int = 0


@dataclass(frozen=True)
class FirstTickZeroPolicy(StatefulRiskPolicy):
    VERSION = "1.0.0"

    def initial_state(self) -> PolicyState:
        return ZeroingState()

    def decide(
        self,
        risk_state: PolicyState | None,
        state: MarketState,
        proposed: Array,
        cash: float,
        balances: Array,
        orders: Array,
        row: int,
    ) -> PolicyState:
        del state, cash, balances
        assert isinstance(risk_state, ZeroingState)
        target = orders[row]
        if risk_state.tick == 0:
            target.fill(0.0)
        else:
            target[:] = proposed[row]
        return ZeroingState(risk_state.tick + 1)


def test_stateful_risk_policy_state_is_fresh_for_each_execution() -> None:
    values = market_frame([[9.0], [10.0]], [[10.0], [11.0]], [[2.0], [-1.0]])
    function = backtest(SignalOrderPolicy(), risk_policy=FirstTickZeroPolicy())

    first = function.batch(values)
    second = function.batch(values)

    assert first.equals(second)
    assert first.get_column("order").to_list() == [[0.0], [-1.0]]
    assert first.get_column("proposed_order").to_list() == [[2.0], [-1.0]]


def test_classify_reason_posthoc() -> None:
    proposed = np.array([2.0, -1.0])
    assert classify_reason(proposed, proposed) == RiskReason.NO_CHANGE
    assert classify_reason(proposed, np.zeros(2)) == RiskReason.ZEROED
    assert classify_reason(proposed, np.array([2.0, -0.5])) == RiskReason.CLAMPED


def test_risk_decision_derives_modified_from_reason() -> None:
    order = Order(np.zeros(1, dtype=np.float64))

    assert RiskDecision(order=order, reason=RiskReason.NO_CHANGE).modified is False
    assert RiskDecision(order=order, reason=RiskReason.CLAMPED).modified is True
    assert RiskDecision(order=order, reason=RiskReason.ZEROED).modified is True


def test_risk_decision_validates_contracts() -> None:
    with pytest.raises(TypeError, match="order must be an Order"):
        RiskDecision(order="nope")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="reason must be a RiskReason"):
        RiskDecision(order=Order(np.zeros(1, dtype=np.float64)), reason=3)  # type: ignore[arg-type]


def test_backtest_node_identity_tracks_risk_policy() -> None:
    def node(risk_policy: RiskPolicy | None) -> Node:
        return Node(
            BacktestTSFN,
            config=BacktestConfig(
                feed=l1_feed(),
                policy=SignalOrderPolicy(),
                initial_cash=100.0,
                risk_policy=risk_policy,
            ),
        )

    no_risk = node(None)
    limit_half = node(FractionalLimitPolicy(0.5))
    limit_quarter = node(FractionalLimitPolicy(0.25))
    kelly = node(FractionalKellyPolicy(0.5))

    assert no_risk.ID != limit_half.ID
    assert limit_half.ID != limit_quarter.ID
    assert limit_half.ID != kelly.ID
    assert node(FractionalLimitPolicy(0.5)).ID == limit_half.ID


def test_backtest_config_rejects_invalid_risk_policy() -> None:
    with pytest.raises(TypeError, match="risk_policy must be a RiskPolicy or a declarative mapping"):
        BacktestConfig(
            feed=l1_feed(),
            policy=SignalOrderPolicy(),
            initial_cash=1.0,
            risk_policy="risky",  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class BadReturnRiskPolicy(RiskPolicy):
    VERSION = "1.0.0"

    def decide(
        self,
        risk_state: PolicyState | None,
        state: MarketState,
        proposed: Array,
        cash: float,
        balances: Array,
        orders: Array,
        row: int,
    ) -> PolicyState | None:
        del risk_state, proposed, state, cash, balances, orders, row
        return None


@dataclass(frozen=True)
class WrongWidthRiskPolicy(RiskPolicy):
    VERSION = "1.0.0"

    def decide(
        self,
        risk_state: PolicyState | None,
        state: MarketState,
        proposed: Array,
        cash: float,
        balances: Array,
        orders: Array,
        row: int,
    ) -> PolicyState | None:
        del risk_state, proposed, state, cash, balances
        orders[row] = np.array([1.0, 2.0], dtype=np.float64)
        return None


@dataclass(frozen=True)
class MutatingBalanceRiskPolicy(RiskPolicy):
    VERSION = "1.0.0"

    def decide(
        self,
        risk_state: PolicyState | None,
        state: MarketState,
        proposed: Array,
        cash: float,
        balances: Array,
        orders: Array,
        row: int,
    ) -> PolicyState | None:
        del risk_state, proposed, state, cash
        balances[0] = 1.0
        return None


@dataclass(frozen=True)
class MutatingMarketRiskPolicy(RiskPolicy):
    VERSION = "1.0.0"

    def decide(
        self,
        risk_state: PolicyState | None,
        state: MarketState,
        proposed: Array,
        cash: float,
        balances: Array,
        orders: Array,
        row: int,
    ) -> PolicyState | None:
        del risk_state, proposed, cash, balances, orders, row
        state.bid[0] = 1.0
        return None


def test_risk_wrong_width_fails_loudly() -> None:
    with pytest.raises(ValueError, match="could not broadcast"):
        backtest(SignalOrderPolicy(), risk_policy=WrongWidthRiskPolicy()).batch(
            market_frame([[9.0]], [[10.0]], [[0.0]])
        )


def test_risk_policies_cannot_mutate_market_input_arrays() -> None:
    values = market_frame([[9.0]], [[10.0]], [[0.0]])

    with pytest.raises(ValueError, match="read-only"):
        backtest(
            SignalOrderPolicy(), risk_policy=MutatingMarketRiskPolicy()
        ).batch(values)


def test_validate_false_skips_frame_validation() -> None:
    values = market_frame([[9.0]], [[10.0]], [[0.0]])
    result = backtest(SignalOrderPolicy(), validate=False).batch(values)
    assert result.height == 1


def test_backtest_graph_runs_with_a_risk_policy_bound() -> None:
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
            return market_frame([[9.0]], [[10.0]], [[20.0]]).lazy()

    tape = Node(Tape)
    simulation = Node(
        BacktestTSFN,
        bindings={"bid": tape.bid, "ask": tape.ask, "signal": tape.signal},
        config=BacktestConfig(
            feed=l1_feed(),
            policy=SignalOrderPolicy(),
            initial_cash=100.0,
            risk_policy=FractionalLimitPolicy(0.5),
        ),
    )

    result = Graph(simulation).execute()

    assert result.get_column("order").to_list() == [[5.0]]
