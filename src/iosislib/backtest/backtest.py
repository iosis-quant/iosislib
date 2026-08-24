"""A minimal graph-native, immediate-execution backtest TSFN."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import polars as pl

from iosislib.backtest.feeds import Feed, L1Feed
from iosislib.backtest.policy import (
    Array,
    MarketState,
    ModelPolicy,
    Policy,
    PolicyState,
    SignalPolicy,
    StatefulPolicy,
    ThresholdPolicy,
)
from iosislib.backtest.risk import (
    NO_OP_RISK,
    FractionalKellyPolicy,
    FractionalLimitPolicy,
    RiskPolicy,
    StatefulRiskPolicy,
)
from iosislib.backtest.venue import Venue
from iosislib.core.tsfn import BatchTSFN, FrameSignature, TSFNConfig, _column_signatures
from iosislib.core.utils import (
    _datetime_dtype_without_timezone,
    _dtype_matches,
    numpy_to_series,
    series_to_numpy,
)


_POLICY_REGISTRY: dict[str, type[Policy]] = {
    "signal": SignalPolicy,
    "threshold": ThresholdPolicy,
}

_RISK_REGISTRY: dict[str, type[RiskPolicy]] = {
    "fractional_limit": FractionalLimitPolicy,
    "fractional_kelly": FractionalKellyPolicy,
}


def _feed_from_declaration(value: Mapping[str, Any]) -> Feed:
    """Resolve a declarative feed mapping into a concrete ``Feed`` instance."""
    kind = value.get("kind", "l1")
    if kind != "l1":
        raise ValueError(f"Unsupported feed kind: {kind!r}; only 'l1' is supported")
    venue_data = value.get("venue")
    if not isinstance(venue_data, Mapping):
        raise ValueError("feed.venue must be a mapping with 'name' and 'universe'")
    universe = venue_data.get("universe")
    if isinstance(universe, (list, tuple)):
        universe = tuple(universe)
    else:
        raise ValueError("feed.venue.universe must be a list of asset names")
    venue = Venue(name=venue_data["name"], universe=universe)
    return L1Feed(
        venue=venue,
        bid_column=value.get("bid_column", "bid"),
        ask_column=value.get("ask_column", "ask"),
    )


def _policy_from_declaration(value: Mapping[str, Any]) -> Policy:
    """Resolve a declarative policy mapping into a concrete ``Policy`` instance."""
    kind = value.get("kind")
    if kind is None:
        raise ValueError("policy declaration must declare a 'kind'")
    cls = _POLICY_REGISTRY.get(kind)
    if cls is None:
        raise ValueError(
            f"Unknown policy kind: {kind!r}; "
            f"available: {sorted(_POLICY_REGISTRY)}"
        )
    params = {k: v for k, v in value.items() if k != "kind"}
    return cls(**params)


def _risk_policy_from_declaration(value: Mapping[str, Any]) -> RiskPolicy:
    """Resolve a declarative risk policy mapping into a ``RiskPolicy``."""
    kind = value.get("kind")
    if kind is None:
        raise ValueError("risk_policy declaration must declare a 'kind'")
    cls = _RISK_REGISTRY.get(kind)
    if cls is None:
        raise ValueError(
            f"Unknown risk_policy kind: {kind!r}; "
            f"available: {sorted(_RISK_REGISTRY)}"
        )
    params = {k: v for k, v in value.items() if k != "kind"}
    return cls(**params)


def _normalize_feed(value: Feed | Mapping[str, Any]) -> Feed:
    if isinstance(value, Feed):
        return value
    if isinstance(value, Mapping):
        return _feed_from_declaration(value)
    raise TypeError("feed must be a Feed or a declarative mapping")


def _normalize_policy(value: Policy | Mapping[str, Any]) -> Policy:
    if isinstance(value, Policy):
        return value
    if isinstance(value, Mapping):
        return _policy_from_declaration(value)
    raise TypeError("policy must be a Policy or a declarative mapping")


def _normalize_risk_policy(value: RiskPolicy | Mapping[str, Any]) -> RiskPolicy:
    if isinstance(value, RiskPolicy):
        return value
    if isinstance(value, Mapping):
        return _risk_policy_from_declaration(value)
    raise TypeError("risk_policy must be a RiskPolicy or a declarative mapping")


@dataclass(frozen=True)
class BacktestConfig(TSFNConfig):
    """Configuration for one immediate-execution simulation.

    ``feed`` and ``policy`` accept either live Python objects or declarative
    YAML-compatible mappings.  ``risk_policy`` follows the same convention.
    """

    feed: Feed
    policy: Policy
    initial_cash: float
    risk_policy: RiskPolicy | None = None
    validate: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "feed", _normalize_feed(self.feed)
        )
        object.__setattr__(
            self, "policy", _normalize_policy(self.policy)
        )
        if self.risk_policy is not None:
            object.__setattr__(
                self, "risk_policy", _normalize_risk_policy(self.risk_policy)
            )
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
                ),
            ),
        )

    def batch(self, frame: pl.DataFrame) -> pl.DataFrame:
        if self.parameters.validate:
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
        policy_state = self._initial_policy_state(policy)

        risk_policy = (
            config.risk_policy if config.risk_policy is not None else NO_OP_RISK
        )
        risk_state = self._initial_risk_state(risk_policy)

        proposed_order = np.empty((rows, width), dtype=np.float64)
        order = np.empty((rows, width), dtype=np.float64)
        balance = np.empty((rows, width), dtype=np.float64)
        cash_col = np.empty(rows, dtype=np.float64)

        price = np.empty(width, dtype=np.float64)
        price_mask: npt.NDArray[np.bool_] = np.empty(width, dtype=np.bool_)

        state = MarketState(information, bid, ask, target)
        running_cash = config.initial_cash
        running_balances = np.zeros(width, dtype=np.float64)

        _move_to = state.move_to
        _policy_decide = policy.decide
        _risk_decide = risk_policy.decide
        _execute = self._execute

        for row, timestamp in enumerate(timestamps):
            _move_to(timestamp, row)
            policy_state = _policy_decide(
                policy_state, state, running_cash, running_balances,
                proposed_order, row,
            )
            risk_state = _risk_decide(
                risk_state, state, proposed_order, running_cash,
                running_balances, order, row,
            )

            running_cash = _execute(
                running_cash, running_balances, order, bid, ask,
                row, price, price_mask,
            )
            cash_col[row] = running_cash
            balance[row] = running_balances

        equity = cash_col + (balance * bid).sum(axis=1)
        return pl.DataFrame(
            [
                timestamps,
                numpy_to_series("cash", cash_col),
                numpy_to_series("equity", equity),
                self._array_series("balance", balance, width),
                self._array_series("order", order, width),
                self._array_series("proposed_order", proposed_order, width),
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
        orders: Array,
        bid: Array,
        ask: Array,
        row: int,
        price: Array,
        price_mask: npt.NDArray[np.bool_],
    ) -> float:
        order_row = orders[row]
        ask_row = ask[row]
        bid_row = bid[row]
        width = order_row.shape[0]
        if width < 8:
            for asset in range(width):
                quantity = order_row[asset]
                if quantity != 0.0:
                    if quantity >= 0.0:
                        cash -= quantity * ask_row[asset]
                    else:
                        cash -= quantity * bid_row[asset]
                    balances[asset] += quantity
            return cash
        np.greater_equal(order_row, 0.0, out=price_mask)
        np.copyto(price, bid_row)
        np.copyto(price, ask_row, where=price_mask)
        cash -= float(np.dot(order_row, price))
        balances += order_row
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
