"""A minimal graph-native, immediate-execution backtest TSFN."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import cast

import numpy as np
import numpy.typing as npt
import polars as pl

from iosislib.backtest.feeds import Feed
from iosislib.backtest.policy import (
    Array,
    MarketState,
    ModelPolicy,
    Policy,
    PolicyState,
    StatefulPolicy,
)
from iosislib.backtest.risk import (
    NO_OP_RISK,
    RiskPolicy,
    StatefulRiskPolicy,
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
    risk_policy: RiskPolicy | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.feed, Feed):
            raise TypeError("feed must be a Feed")
        if not isinstance(self.policy, Policy):
            raise TypeError("policy must be a Policy")
        if not isfinite(self.initial_cash):
            raise ValueError("initial_cash must be finite")
        if self.risk_policy is not None and not isinstance(
            self.risk_policy, RiskPolicy
        ):
            raise TypeError("risk_policy must be a RiskPolicy or None")


class BacktestTSFN(BatchTSFN[BacktestConfig]):
    """Simulate policy orders against a feed's executable quotes, row by row."""

    VERSION = "1.2.0"
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
                    ("proposed_order", pl.Float64, (width,)),
                    ("risk_reason", pl.Int8),
                ),
            ),
        )

    def batch(self, frame: pl.DataFrame) -> pl.DataFrame:
        self._validate_frame(frame)
        config = self.parameters
        width = config.feed.width
        rows = frame.height
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

        policy = config.policy
        policy_stateful = isinstance(policy, StatefulPolicy)
        policy_state = self._initial_policy_state(policy)

        risk_policy = (
            config.risk_policy if config.risk_policy is not None else NO_OP_RISK
        )
        risk_stateful = isinstance(risk_policy, StatefulRiskPolicy)
        risk_state = self._initial_risk_state(risk_policy)

        proposed_order = np.empty((rows, width), dtype=np.float64)
        order = np.empty((rows, width), dtype=np.float64)
        balance = np.empty((rows, width), dtype=np.float64)
        cash_col = np.empty(rows, dtype=np.float64)
        reason = np.empty(rows, dtype=np.int8)

        price = np.empty(width, dtype=np.float64)
        price_mask: npt.NDArray[np.bool_] = np.empty(width, dtype=np.bool_)

        state = MarketState(information, bid, ask, target)
        running_cash = config.initial_cash
        running_balances = np.zeros(width, dtype=np.float64)

        _move_to = state.move_to
        if policy_stateful:
            _policy_decide = policy.decide_stateful
        else:
            _policy_decide = policy.decide
        if risk_stateful:
            _risk_decide = risk_policy.decide_stateful
        else:
            _risk_decide = risk_policy.decide
        _execute = self._execute

        for row, timestamp in enumerate(timestamps):
            _move_to(timestamp, row)
            if policy_stateful:
                policy_state = _policy_decide(
                    policy_state, state, running_cash, running_balances,
                    proposed_order, row,
                )
            else:
                _policy_decide(
                    state, running_cash, running_balances, proposed_order, row,
                )

            if risk_stateful:
                risk_reason, risk_state = _risk_decide(
                    risk_state, state, proposed_order, running_cash,
                    running_balances, order, row,
                )
            else:
                risk_reason = _risk_decide(
                    proposed_order, state, running_cash, running_balances,
                    order, row,
                )
            reason[row] = int(risk_reason)

            running_cash = _execute(
                running_cash, running_balances, order[row], bid, ask,
                row, price, price_mask,
            )
            cash_col[row] = running_cash
            balance[row] = running_balances

        equity = cash_col + np.einsum("ij,ij->i", balance, bid)
        return pl.DataFrame(
            [
                timestamps,
                numpy_to_series("cash", cash_col),
                numpy_to_series("equity", equity),
                self._array_series("balance", balance, width),
                self._array_series("order", order, width),
                self._array_series("proposed_order", proposed_order, width),
                pl.Series("risk_reason", reason, dtype=pl.Int8),
            ]
        )

    def _array_series(self, name: str, values: Array, width: int) -> pl.Series:
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

    @staticmethod
    def _initial_risk_state(
        risk_policy: RiskPolicy | None,
    ) -> PolicyState | None:
        if risk_policy is None or not isinstance(risk_policy, StatefulRiskPolicy):
            return None
        risk_state = risk_policy.initial_state()
        if not isinstance(risk_state, PolicyState):
            raise TypeError(
                "StatefulRiskPolicy.initial_state must return a PolicyState"
            )
        return risk_state

    @staticmethod
    def _execute(
        cash: float,
        balances: Array,
        quantities: Array,
        bid: Array,
        ask: Array,
        row: int,
        price: Array,
        price_mask: npt.NDArray[np.bool_],
    ) -> float:
        if quantities.size < 8:
            for asset, quantity in enumerate(quantities):
                if quantity >= 0.0:
                    cash -= quantity * ask[row, asset]
                else:
                    cash -= quantity * bid[row, asset]
                balances[asset] += quantity
            return cash
        np.greater_equal(quantities, 0.0, out=price_mask)
        np.copyto(price, bid[row])
        np.copyto(price, ask[row], where=price_mask)
        cash -= float(np.dot(quantities, price))
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
