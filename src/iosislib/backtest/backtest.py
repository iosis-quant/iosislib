"""A minimal graph-native, immediate-execution backtest TSFN."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import polars as pl

from iosislib.backtest.feeds import Feed, L1Feed, L2Feed
from iosislib.backtest._tolerances import (
    FILL_ABS_EPS,
    cost_eps,
    has_fill,
    is_ruined,
    is_zero_fill,
    is_zero_qty,
)
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


def register_policy(kind: str, cls: type[Policy]) -> None:
    """Register a custom Policy class under a declarative ``kind`` name.

    The ``kind`` value is used in YAML strategy declarations as
    ``policy: {kind: "<kind>", ...}``. The class must be a concrete
    ``Policy`` subclass.
    """
    if not isinstance(kind, str) or not kind.strip():
        raise ValueError("policy kind must be a non-empty string")
    if not isinstance(cls, type) or not issubclass(cls, Policy):
        raise TypeError("cls must be a concrete Policy subclass")
    if kind in _POLICY_REGISTRY:
        raise ValueError(f"policy kind {kind!r} is already registered")
    _POLICY_REGISTRY[kind] = cls


def register_risk_policy(kind: str, cls: type[RiskPolicy]) -> None:
    """Register a custom RiskPolicy class under a declarative ``kind`` name.

    The ``kind`` value is used in YAML strategy declarations as
    ``risk_policy: {kind: "<kind>", ...}``. The class must be a concrete
    ``RiskPolicy`` subclass.
    """
    if not isinstance(kind, str) or not kind.strip():
        raise ValueError("risk_policy kind must be a non-empty string")
    if not isinstance(cls, type) or not issubclass(cls, RiskPolicy):
        raise TypeError("cls must be a concrete RiskPolicy subclass")
    if kind in _RISK_REGISTRY:
        raise ValueError(f"risk_policy kind {kind!r} is already registered")
    _RISK_REGISTRY[kind] = cls


def list_policies() -> dict[str, type[Policy]]:
    """Return the current policy registry."""
    return dict(_POLICY_REGISTRY)


def list_risk_policies() -> dict[str, type[RiskPolicy]]:
    """Return the current risk policy registry."""
    return dict(_RISK_REGISTRY)


def _feed_from_declaration(value: Mapping[str, Any]) -> Feed:
    """Resolve a declarative feed mapping into a concrete ``Feed`` instance."""
    kind = value.get("kind", "l1")
    venue_data = value.get("venue")
    if not isinstance(venue_data, Mapping):
        raise ValueError("feed.venue must be a mapping with 'name' and 'universe'")
    universe = venue_data.get("universe")
    if isinstance(universe, (list, tuple)):
        universe = tuple(universe)
    else:
        raise ValueError("feed.venue.universe must be a list of asset names")
    venue = Venue(name=venue_data["name"], universe=universe)
    if kind == "l1":
        return L1Feed(
            venue=venue,
            bid_column=value.get("bid_column", "bid"),
            ask_column=value.get("ask_column", "ask"),
        )
    if kind == "l2":
        legacy = {"bid_depth_column", "ask_depth_column", "tick"} & set(value)
        if legacy:
            raise ValueError(
                f"legacy L2 keys {sorted(legacy)} were removed in L2Feed 2.0.0; "
                "use bid_price_column/bid_volume_column/ask_price_column/"
                "ask_volume_column with depth_levels"
            )
        return L2Feed(
            venue=venue,
            bid_price_column=value.get("bid_price_column", "bid_price"),
            bid_volume_column=value.get("bid_volume_column", "bid_volume"),
            ask_price_column=value.get("ask_price_column", "ask_price"),
            ask_volume_column=value.get("ask_volume_column", "ask_volume"),
            depth_levels=int(value.get("depth_levels", 25)),
        )
    raise ValueError(f"Unsupported feed kind: {kind!r}; expected 'l1' or 'l2'")


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
    limit_price_column: str | None = None

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
        if self.limit_price_column is not None:
            if not isinstance(self.limit_price_column, str):
                raise TypeError("limit_price_column must be a string or None")
            if not self.limit_price_column:
                raise ValueError("limit_price_column cannot be empty")


class BacktestTSFN(BatchTSFN[BacktestConfig]):
    """Simulate policy orders against a feed's executable quotes, row by row."""

    VERSION = "2.0.0"
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
        if config.limit_price_column is not None:
            input_columns = (
                *input_columns,
                (config.limit_price_column, pl.Float64, (width,)),
            )
        if isinstance(config.policy, ModelPolicy):
            target_column = (
                ("target", pl.Float64)
                if not config.policy.target_shape
                else ("target", pl.Float64, config.policy.target_shape)
            )
            input_columns = (*input_columns, target_column)
        output_columns: tuple[Any, ...] = (
            ("cash", pl.Float64),
            ("equity", pl.Float64),
            ("balance", pl.Float64, (width,)),
            ("order", pl.Float64, (width,)),
            ("proposed_order", pl.Float64, (width,)),
        )
        if isinstance(config.feed, L2Feed):
            output_columns = (
                *output_columns,
                ("fill_price", pl.Float64, (width,)),
                ("unfilled", pl.Float64, (width,)),
            )
        return (
            FrameSignature(time=config.feed.time_axis, columns=input_columns),
            FrameSignature(
                time=config.feed.time_axis,
                columns=output_columns,
            ),
        )

    def batch(self, frame: pl.DataFrame) -> pl.DataFrame:
        if self.parameters.validate:
            self._validate_frame(frame)
        config = self.parameters
        width = config.feed.width
        rows = frame.height
        bid, ask = self._quotes(frame)
        is_l2 = isinstance(config.feed, L2Feed)
        bid_price: Array | None = None
        bid_volume: Array | None = None
        ask_price: Array | None = None
        ask_volume: Array | None = None
        limits: Array | None = None
        if is_l2:
            feed = cast(L2Feed, config.feed)
            bid_price, bid_volume, ask_price, ask_volume = self._depth(frame, feed)
            limits = self._limits(frame, width, rows)
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
        fill_price = (
            np.empty((rows, width), dtype=np.float64) if is_l2 else None
        )
        unfilled = (
            np.empty((rows, width), dtype=np.float64) if is_l2 else None
        )

        price = np.empty(width, dtype=np.float64)
        price_mask: npt.NDArray[np.bool_] = np.empty(width, dtype=np.bool_)

        state = MarketState(information, bid, ask, target)
        running_cash = config.initial_cash
        running_balances = np.zeros(width, dtype=np.float64)

        _move_to = state.move_to
        _policy_decide = policy.decide
        _risk_decide = risk_policy.decide
        _execute = self._execute
        _execute_l2 = self._execute_l2

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

            if is_l2:
                assert (
                    bid_price is not None
                    and bid_volume is not None
                    and ask_price is not None
                    and ask_volume is not None
                )
                assert fill_price is not None and unfilled is not None
                running_cash, ruined = _execute_l2(
                    running_cash, running_balances, order, bid_price,
                    bid_volume, ask_price, ask_volume, row, limits,
                    bid, fill_price, unfilled,
                )
            else:
                running_cash, ruined = _execute(
                    running_cash, running_balances, order, bid, ask,
                    row, price, price_mask,
                )
            cash_col[row] = running_cash
            balance[row] = running_balances
            if ruined:
                cash_col[row + 1 :] = running_cash
                balance[row + 1 :] = running_balances
                order[row + 1 :] = 0.0
                proposed_order[row + 1 :] = 0.0
                if is_l2:
                    assert fill_price is not None and unfilled is not None
                    fill_price[row + 1 :] = 0.0
                    unfilled[row + 1 :] = 0.0
                break

        equity = cash_col + (balance * bid).sum(axis=1)
        columns = [
            timestamps,
            numpy_to_series("cash", cash_col),
            numpy_to_series("equity", equity),
            self._array_series("balance", balance, width),
            self._array_series("order", order, width),
            self._array_series("proposed_order", proposed_order, width),
        ]
        if is_l2:
            assert fill_price is not None and unfilled is not None
            columns.append(self._array_series("fill_price", fill_price, width))
            columns.append(self._array_series("unfilled", unfilled, width))
        return pl.DataFrame(columns)

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
    ) -> tuple[float, bool]:
        order_row = orders[row]
        ask_row = ask[row]
        bid_row = bid[row]
        width = order_row.shape[0]
        equity = cash + float(np.dot(balances, bid_row))
        if is_ruined(equity, cash):
            if width < 8:
                for asset in range(width):
                    position = float(balances[asset])
                    if position > 0.0:
                        cash += position * float(bid_row[asset])
                    elif position < 0.0:
                        cash += position * float(ask_row[asset])
                    balances[asset] = 0.0
                    order_row[asset] = -position
                return cash, True
            liquidation = -balances
            np.greater_equal(balances, 0.0, out=price_mask)
            np.copyto(price, ask_row)
            np.copyto(price, bid_row, where=price_mask)
            cash += float(np.dot(balances, price))
            order_row[:] = liquidation
            balances[:] = 0.0
            return cash, True
        if width < 8:
            for asset in range(width):
                quantity = float(order_row[asset])
                if is_zero_qty(quantity):
                    order_row[asset] = 0.0
                    continue
                if quantity > 0.0:
                    spread = float(ask_row[asset]) - float(bid_row[asset])
                    if equity - quantity * spread < -cost_eps(equity):
                        order_row[asset] = 0.0
                        continue
                    equity -= quantity * spread
                    cash -= quantity * float(ask_row[asset])
                else:
                    cash -= quantity * float(bid_row[asset])
                balances[asset] += quantity
            return cash, False
        executed = np.array(order_row, dtype=np.float64, copy=True)
        executed[np.abs(executed) <= 1e-12] = 0.0
        run = equity
        for asset in range(width):
            quantity = float(executed[asset])
            if quantity > 0.0:
                spread = float(ask_row[asset]) - float(bid_row[asset])
                if run - quantity * spread < -cost_eps(run):
                    executed[asset] = 0.0
                else:
                    run -= quantity * spread
        np.greater_equal(executed, 0.0, out=price_mask)
        np.copyto(price, bid_row)
        np.copyto(price, ask_row, where=price_mask)
        cash -= float(np.dot(executed, price))
        balances += executed
        order_row[:] = executed
        return cash, False

    def _depth(
        self, frame: pl.DataFrame, feed: L2Feed
    ) -> tuple[Array, Array, Array, Array]:
        depth_series = feed.depth(frame)
        if (
            not isinstance(depth_series, tuple)
            or len(depth_series) != 4
            or not all(isinstance(series, pl.Series) for series in depth_series)
        ):
            raise TypeError(
                "Feed.depth must return a (bid_price, bid_volume, "
                "ask_price, ask_volume) tuple of Polars Series"
            )
        bid_price_series, bid_volume_series, ask_price_series, ask_volume_series = (
            depth_series
        )
        shape = (feed.width, feed.depth_levels)
        bid_price = self._value_column(
            bid_price_series, shape, feed.bid_price_column
        )
        bid_volume = self._value_column(
            bid_volume_series, shape, feed.bid_volume_column
        )
        ask_price = self._value_column(
            ask_price_series, shape, feed.ask_price_column
        )
        ask_volume = self._value_column(
            ask_volume_series, shape, feed.ask_volume_column
        )
        if (bid_volume < 0.0).any() or (ask_volume < 0.0).any():
            raise ValueError("depth volumes must be non-negative")
        if (bid_price < 0.0).any() or (ask_price < 0.0).any():
            raise ValueError("depth prices must be non-negative")
        if (np.diff(bid_price, axis=-1) > 0.0).any():
            raise ValueError("bid_price ladder must be best-first (non-increasing)")
        if (np.diff(ask_price, axis=-1) < 0.0).any():
            raise ValueError("ask_price ladder must be best-first (non-decreasing)")
        return bid_price, bid_volume, ask_price, ask_volume

    def _limits(self, frame: pl.DataFrame, width: int, rows: int) -> Array | None:
        column = self.parameters.limit_price_column
        if column is None:
            return None
        series = frame.get_column(column)
        if series.null_count():
            raise ValueError(f"{column} cannot contain nulls")
        values = cast(
            Array,
            series_to_numpy(series, shape=(width,), allow_copy=True),
        )
        if values.shape != (rows, width):
            raise TypeError(f"{column} must have shape ({width},) per row")
        if values.dtype != np.dtype(np.float64):
            raise TypeError(f"{column} must expose Float64 values")
        if np.isposinf(values).any() or np.isneginf(values).any():
            raise ValueError(f"{column} must be finite or NaN")
        return values

    @staticmethod
    def _execute_l2_narrow(
        cash: float,
        balances: Array,
        orders: Array,
        bid_price: Array,
        bid_volume: Array,
        ask_price: Array,
        ask_volume: Array,
        row: int,
        limits: Array | None,
        marks: Array,
        fill_price: Array,
        unfilled: Array,
        force: bool = False,
    ) -> tuple[float, bool]:
        requested = orders[row]
        width = requested.shape[0]
        mark_row = marks[row]
        equity = cash + float(np.dot(balances, mark_row))
        if is_ruined(equity, cash) and not force:
            requested[:] = -balances
            limits = None
            force = True
        for asset in range(width):
            quantity = float(requested[asset])
            if is_zero_qty(quantity):
                fill_price[row, asset] = 0.0
                unfilled[row, asset] = 0.0
                requested[asset] = 0.0
                continue
            buying = quantity > 0.0
            limit = float(limits[row, asset]) if limits is not None else float("nan")
            need = abs(quantity)
            filled = 0.0
            cost = 0.0
            if buying:
                prices = ask_price[row, asset]
                volumes = ask_volume[row, asset]
                if limit != limit:
                    eligible = volumes
                    eligible_px = prices
                else:
                    mask = prices <= limit
                    eligible = volumes * mask
                    eligible_px = prices
                cum = np.cumsum(eligible)
                take = np.minimum(eligible, np.maximum(need - cum + eligible, 0.0))
                filled = float(take.sum())
                cost = float(np.dot(take, eligible_px))
            else:
                prices = bid_price[row, asset]
                volumes = bid_volume[row, asset]
                if limit != limit:
                    eligible = volumes
                    eligible_px = prices
                else:
                    mask = prices >= limit
                    eligible = volumes * mask
                    eligible_px = prices
                cum = np.cumsum(eligible)
                take = np.minimum(eligible, np.maximum(need - cum + eligible, 0.0))
                filled = float(take.sum())
                cost = float(np.dot(take, eligible_px))
            if has_fill(filled):
                average = cost / filled
                signed_fill = filled if buying else -filled
                if (
                    not force
                    and equity + signed_fill * (float(mark_row[asset]) - average)
                    < -cost_eps(equity)
                ):
                    requested[asset] = 0.0
                    fill_price[row, asset] = 0.0
                    unfilled[row, asset] = quantity
                    continue
                equity += signed_fill * (float(mark_row[asset]) - average)
                balances[asset] += signed_fill
                cash -= signed_fill * average
                requested[asset] = signed_fill
                fill_price[row, asset] = average
                unfilled[row, asset] = quantity - signed_fill
            else:
                requested[asset] = 0.0
                fill_price[row, asset] = 0.0
                unfilled[row, asset] = quantity
        return cash, force

    @staticmethod
    def _execute_l2(
        cash: float,
        balances: Array,
        orders: Array,
        bid_price: Array,
        bid_volume: Array,
        ask_price: Array,
        ask_volume: Array,
        row: int,
        limits: Array | None,
        marks: Array,
        fill_price: Array,
        unfilled: Array,
    ) -> tuple[float, bool]:
        requested = orders[row]
        width = requested.shape[0]
        if width < 8:
            return BacktestTSFN._execute_l2_narrow(
                cash, balances, orders, bid_price, bid_volume,
                ask_price, ask_volume, row, limits,
                marks, fill_price, unfilled,
            )
        mark_row = marks[row]
        equity = cash + float(np.dot(balances, mark_row))
        if is_ruined(equity, cash):
            requested[:] = -balances
            cash, _ = BacktestTSFN._execute_l2_narrow(
                cash, balances, orders, bid_price, bid_volume,
                ask_price, ask_volume, row, None,
                marks, fill_price, unfilled, True,
            )
            return cash, True
        quantity = requested.copy()
        quantity[np.abs(quantity) <= 1e-12] = 0.0
        if not np.any(np.abs(quantity) > 1e-12):
            fill_price[row] = 0.0
            unfilled[row] = 0.0
            requested[:] = 0.0
            return cash, False
        need = np.abs(quantity)
        filled = np.zeros(width, dtype=np.float64)
        cost = np.zeros(width, dtype=np.float64)
        buy = quantity > 0.0
        buy_px = ask_price[row][buy]
        buy_vol = ask_volume[row][buy]
        if limits is not None:
            lim = limits[row][buy][:, None]
            buy_vol = np.where(np.isnan(lim) | (buy_px <= lim), buy_vol, 0.0)
        cum = np.cumsum(buy_vol, axis=-1)
        take = np.minimum(buy_vol, np.maximum(need[buy][:, None] - cum + buy_vol, 0.0))
        filled[buy] = take.sum(axis=-1)
        cost[buy] = (take * buy_px).sum(axis=-1)
        sell = quantity < 0.0
        sell_px = bid_price[row][sell]
        sell_vol = bid_volume[row][sell]
        if limits is not None:
            lim = limits[row][sell][:, None]
            sell_vol = np.where(np.isnan(lim) | (sell_px >= lim), sell_vol, 0.0)
        cum = np.cumsum(sell_vol, axis=-1)
        take = np.minimum(sell_vol, np.maximum(need[sell][:, None] - cum + sell_vol, 0.0))
        filled[sell] = take.sum(axis=-1)
        cost[sell] = (take * sell_px).sum(axis=-1)
        run = equity
        for asset in range(width):
            fill = float(filled[asset])
            if is_zero_fill(fill):
                filled[asset] = 0.0
                cost[asset] = 0.0
                continue
            avg = float(cost[asset]) / fill
            signed = fill if float(quantity[asset]) > 0.0 else -fill
            if run + signed * (float(mark_row[asset]) - avg) < -cost_eps(run):
                filled[asset] = 0.0
                cost[asset] = 0.0
            else:
                run += signed * (float(mark_row[asset]) - avg)
        has_fill = filled > FILL_ABS_EPS
        average = np.zeros(width, dtype=np.float64)
        average[has_fill] = cost[has_fill] / filled[has_fill]
        signed_fill = np.where(quantity > 0.0, filled, -filled)
        signed_fill[~has_fill] = 0.0
        balances += signed_fill
        cash -= float(np.dot(signed_fill, average))
        requested[:] = signed_fill
        fill_price[row] = average
        unfilled[row] = quantity - signed_fill
        return cash, False

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
