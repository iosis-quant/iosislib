"""A minimal graph-native, immediate-execution backtest TSFN."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import cast

import numpy as np
import polars as pl

from iosislib.backtest.feeds import Feed
from iosislib.backtest.policy import (
    Array,
    MarketState,
    ModelPolicy,
    Order,
    Policy,
    PolicyState,
    StatefulPolicy,
)
from iosislib.core.tsfn import BatchTSFN, FrameSignature, TSFNConfig, _column_signatures
from iosislib.core.utils import (
    _datetime_dtype_without_timezone,
    _dtype_matches,
    numpy_to_series,
    series_to_numpy,
)


@dataclass(frozen=True)
class BacktestConfig(TSFNConfig):
    """Configuration for one immediate-execution simulation."""

    feed: Feed
    policy: Policy
    initial_cash: float

    def __post_init__(self) -> None:
        if not isinstance(self.feed, Feed):
            raise TypeError("feed must be a Feed")
        if not isinstance(self.policy, Policy):
            raise TypeError("policy must be a Policy")
        if not isfinite(self.initial_cash):
            raise ValueError("initial_cash must be finite")


class _History:
    """Append-only, preallocated history for one known-size batch input."""

    def __init__(self, rows: int, width: int) -> None:
        self._row = 0
        self.cash = np.empty(rows, dtype=np.float64)
        self.equity = np.empty(rows, dtype=np.float64)
        self.balances = np.empty((rows, width), dtype=np.float64)
        self.orders = np.empty((rows, width), dtype=np.float64)

    def append(
        self,
        cash: float,
        balances: Array,
        quantities: Array,
        bid: Array,
        market_row: int,
    ) -> None:
        if self._row >= len(self.cash):
            raise RuntimeError("history capacity exhausted")
        self.cash[self._row] = cash
        self.equity[self._row] = cash + float(np.dot(balances, bid[market_row]))
        self.balances[self._row] = balances
        self.orders[self._row] = quantities
        self._row += 1

    def assert_complete(self) -> None:
        if self._row != len(self.cash):
            raise RuntimeError("history is incomplete")


class BacktestTSFN(BatchTSFN[BacktestConfig]):
    """Simulate policy orders against a feed's executable quotes, row by row."""

    VERSION = "1.0.0"
    CONFIG_CLS = BacktestConfig

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        config = self.parameters
        width = config.feed.width
        feature_shape = (
            config.policy.feature_shape
            if isinstance(config.policy, ModelPolicy)
            else (width,)
        )
        signal_column = (
            ("signal", pl.Float64)
            if not feature_shape
            else ("signal", pl.Float64, feature_shape)
        )
        input_columns = (*config.feed.columns, signal_column)
        if isinstance(config.policy, ModelPolicy):
            target_column = (
                ("target", pl.Float64)
                if not config.policy.target_shape
                else ("target", pl.Float64, config.policy.target_shape)
            )
            input_columns = (*input_columns, target_column)
        return (
            FrameSignature(time=config.feed.time_axis, columns=input_columns),
            FrameSignature(
                time=config.feed.time_axis,
                columns=(
                    ("cash", pl.Float64),
                    ("equity", pl.Float64),
                    ("balance", pl.Float64, (width,)),
                    ("order", pl.Float64, (width,)),
                ),
            ),
        )

    def batch(self, frame: pl.DataFrame) -> pl.DataFrame:
        self._validate_frame(frame)
        config = self.parameters
        width = config.feed.width
        bid, ask = self._quotes(frame)
        feature_shape = (
            config.policy.feature_shape
            if isinstance(config.policy, ModelPolicy)
            else (width,)
        )
        information = self._value_column(
            frame.get_column("signal"), feature_shape, "signal"
        )
        target = (
            self._value_column(
                frame.get_column("target"), config.policy.target_shape, "target"
            )
            if isinstance(config.policy, ModelPolicy)
            else None
        )
        timestamps = frame.get_column(config.feed.time_axis.column)

        cash = config.initial_cash
        balances = np.zeros(width, dtype=np.float64)
        history = _History(frame.height, width)
        state = MarketState(information, bid, ask, target)
        policy_state = self._initial_policy_state(config.policy)
        for row, timestamp in enumerate(timestamps):
            state.move_to(timestamp, row)
            order, policy_state = self._decide(
                config.policy, policy_state, state, cash, balances, width
            )
            cash = self._execute(cash, balances, order, bid, ask, row)
            history.append(cash, balances, order.quantities, bid, row)
        history.assert_complete()
        return pl.DataFrame(
            [
                timestamps,
                numpy_to_series("cash", history.cash),
                numpy_to_series("equity", history.equity),
                self._history_array_series("balance", history.balances, width),
                self._history_array_series("order", history.orders, width),
            ]
        )

    def _history_array_series(self, name: str, values: Array, width: int) -> pl.Series:
        if len(values) == 0:
            return pl.Series(name, [], dtype=pl.Array(pl.Float64, width))
        return numpy_to_series(name, values, shape=(width,))

    def _quotes(self, frame: pl.DataFrame) -> tuple[Array, Array]:
        quote_series = self.parameters.feed.quotes(frame)
        if (
            not isinstance(quote_series, tuple)
            or len(quote_series) != 2
            or not all(isinstance(series, pl.Series) for series in quote_series)
        ):
            raise TypeError(
                "Feed.quotes must return a (bid, ask) pair of Polars Series"
            )
        bid_series, ask_series = quote_series
        width = self.parameters.feed.width
        bid = self._array_column(bid_series, width, "bid")
        ask = self._array_column(ask_series, width, "ask")
        if (bid <= 0.0).any() or (ask <= 0.0).any():
            raise ValueError("bid and ask must be positive")
        if (bid > ask).any():
            raise ValueError("bid cannot exceed ask")
        return bid, ask

    def _array_column(self, series: pl.Series, width: int, name: str) -> Array:
        return self._value_column(series, (width,), name)

    def _value_column(
        self,
        series: pl.Series,
        shape: tuple[int, ...],
        name: str,
    ) -> Array:
        if series.null_count():
            raise ValueError(f"{name} cannot contain nulls")
        if len(series) == 0:
            return np.empty(
                (0, *shape) if shape else (0,),
                dtype=np.float64,
            )
        values = cast(
            Array,
            series_to_numpy(
                series,
                shape=shape or None,
                allow_copy=True,
            ),
        )
        expected_shape = (len(series), *shape) if shape else (len(series),)
        if values.shape != expected_shape:
            raise TypeError(
                f"{name} must have shape {shape or '()'} per row"
            )
        if values.dtype != np.dtype(np.float64):
            raise TypeError(f"{name} must expose Float64 values")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} must be finite")
        return values

    @staticmethod
    def _initial_policy_state(policy: Policy) -> PolicyState | None:
        if not isinstance(policy, StatefulPolicy):
            return None
        policy_state = policy.initial_state()
        if not isinstance(policy_state, PolicyState):
            raise TypeError("StatefulPolicy.initial_state must return a PolicyState")
        return policy_state

    @classmethod
    def _decide(
        cls,
        policy: Policy,
        policy_state: PolicyState | None,
        state: MarketState,
        cash: float,
        balances: Array,
        width: int,
    ) -> tuple[Order, PolicyState | None]:
        balances.setflags(write=False)
        try:
            if isinstance(policy, StatefulPolicy):
                if policy_state is None:
                    raise RuntimeError("StatefulPolicy requires an initial PolicyState")
                result = policy.decide_stateful(policy_state, state, cash, balances)
                if not isinstance(result, tuple) or len(result) != 2:
                    raise TypeError(
                        "StatefulPolicy.decide_stateful must return (Order, PolicyState)"
                    )
                order, next_state = result
                if not isinstance(next_state, PolicyState):
                    raise TypeError(
                        "StatefulPolicy.decide_stateful must return a PolicyState"
                    )
            else:
                order = policy.decide(state, cash, balances)
                next_state = None
        finally:
            balances.setflags(write=True)
        return cls._order(order, width), next_state

    @staticmethod
    def _order(value: Order, width: int) -> Order:
        if not isinstance(value, Order):
            raise TypeError("Policy.decide must return an Order")
        if len(value.quantities) != width:
            raise ValueError(f"Order width must be {width}")
        return value

    @staticmethod
    def _execute(
        cash: float, balances: Array, order: Order, bid: Array, ask: Array, row: int
    ) -> float:
        quantities = order.quantities
        if quantities.size < 8:
            for asset, quantity in enumerate(quantities):
                if quantity >= 0.0:
                    cash -= quantity * ask[row, asset]
                else:
                    cash -= quantity * bid[row, asset]
                balances[asset] += quantity
            return cash
        prices = np.where(quantities >= 0.0, ask[row], bid[row])
        cash -= float(np.dot(quantities, prices))
        balances += quantities
        return cash

    def _validate_frame(self, frame: pl.DataFrame) -> None:
        config = self.parameters
        time = config.feed.time_axis
        if time.column not in frame.schema:
            raise ValueError(f"Missing required time column: '{time.column}'")
        actual_time = frame.schema[time.column]
        if not _dtype_matches(
            _datetime_dtype_without_timezone(actual_time),
            _datetime_dtype_without_timezone(cast(pl.DataType, time.dtype)),
        ):
            raise TypeError(f"Time column '{time.column}' type mismatch")
        if getattr(actual_time, "time_zone", None) != time.timezone:
            raise TypeError(f"Time column '{time.column}' timezone mismatch")
        timestamps = frame.get_column(time.column)
        if timestamps.null_count() or not timestamps.is_sorted():
            raise ValueError(f"{time.column} must be non-null and sorted")
        if timestamps.n_unique() != frame.height:
            raise ValueError(f"{time.column} must be strictly increasing")
        for column in _column_signatures(self.type_signature()[0]):
            actual = frame.schema.get(column.name)
            if actual is None:
                raise ValueError(f"Missing required input column: '{column.name}'")
            if actual != column.physical_dtype:
                raise TypeError(f"Column '{column.name}' type mismatch")


__all__ = ["BacktestConfig", "BacktestTSFN"]
