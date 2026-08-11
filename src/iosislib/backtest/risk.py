"""Risk-policy contracts for the minimal event backtest."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields, is_dataclass
from enum import IntEnum
from math import isfinite
from typing import Any, ClassVar, cast

import numpy as np

from iosislib.backtest.policy import Array, MarketState, Order, PolicyState
from iosislib.core.utils import _canonical_json, _serialize_value


class RiskReason(IntEnum):
    """Stable integer reason describing what a risk policy did to an order."""

    NO_CHANGE = 0
    CLAMPED = 1
    ZEROED = 2


@dataclass(frozen=True)
class RiskDecision:
    """The risk-adjusted result for one tick: an order plus its reason."""

    order: Order
    modified: bool = True
    reason: RiskReason = RiskReason.CLAMPED

    def __post_init__(self) -> None:
        if not isinstance(self.order, Order):
            raise TypeError("order must be an Order")
        if not isinstance(self.modified, bool):
            raise TypeError("modified must be a boolean")
        if not isinstance(self.reason, RiskReason):
            raise TypeError("reason must be a RiskReason")
        object.__setattr__(self, "modified", self.reason is not RiskReason.NO_CHANGE)


def _decision(original: Array, effective: Array) -> RiskDecision:
    """Classify an effective order against the proposed one into a RiskDecision."""
    if original.shape != effective.shape:
        raise ValueError("effective order must match the proposed order width")
    order = Order(np.ascontiguousarray(effective))
    if np.array_equal(original, effective):
        return RiskDecision(order=order, reason=RiskReason.NO_CHANGE)
    if not np.any(effective):
        return RiskDecision(order=order, reason=RiskReason.ZEROED)
    return RiskDecision(order=order, reason=RiskReason.CLAMPED)


class RiskPolicy(ABC):
    """Transform a proposed order into a risk-adjusted order before execution."""

    _SERIALIZE_WITH_TO_DICT: ClassVar[bool] = True
    VERSION: ClassVar[str]

    @abstractmethod
    def derisk(
        self,
        order: Order,
        state: MarketState,
        cash: float,
        balances: Array,
    ) -> RiskDecision:
        """Return the effective order: as-is, clamped, or clamped to zero.

        ``balances`` is executor-owned persistent storage and is read-only while
        the method runs. Implementations must not retain ``state`` or balances.
        """

    def to_dict(self) -> dict[str, Any]:
        values: dict[str, Any] = (
            {
                item.name: _serialize_value(getattr(self, item.name))
                for item in fields(cast(Any, self))
            }
            if is_dataclass(self)
            else {}
        )
        return {
            "type": f"{type(self).__module__}.{type(self).__qualname__}",
            "version": self.VERSION,
            **values,
        }

    def __str__(self) -> str:
        return _canonical_json(self.to_dict())


class StatefulRiskPolicy(RiskPolicy, ABC):
    """A risk policy whose immutable state remains local to one execution."""

    @abstractmethod
    def initial_state(self) -> PolicyState:
        """Return fresh state for one run; never retain it on the policy."""

    @abstractmethod
    def derisk_stateful(
        self,
        policy_state: PolicyState,
        order: Order,
        state: MarketState,
        cash: float,
        balances: Array,
    ) -> tuple[RiskDecision, PolicyState]:
        """Return the risk decision and successor state for the current tick."""

    def derisk(
        self,
        order: Order,
        state: MarketState,
        cash: float,
        balances: Array,
    ) -> RiskDecision:
        del order, state, cash, balances
        raise TypeError("StatefulRiskPolicy must be driven via derisk_stateful")


@dataclass(frozen=True)
class FractionalLimitPolicy(RiskPolicy):
    """Cap each position's notional at a fraction of current equity."""

    VERSION = "1.0.0"

    fraction: float

    def __post_init__(self) -> None:
        if not isfinite(self.fraction):
            raise ValueError("fraction must be finite")
        if self.fraction <= 0.0 or self.fraction > 1.0:
            raise ValueError("fraction must be in (0, 1]")

    def derisk(
        self,
        order: Order,
        state: MarketState,
        cash: float,
        balances: Array,
    ) -> RiskDecision:
        equity = cash + float(np.dot(balances, state.bid))
        max_notional = self.fraction * equity
        quantities = order.quantities.copy()
        for index, quantity in enumerate(quantities):
            if quantity == 0.0:
                continue
            price = state.ask[index] if quantity > 0.0 else state.bid[index]
            if price <= 0.0:
                raise ValueError("quote price must be positive")
            if max_notional <= 0.0:
                quantities[index] = 0.0
                continue
            projected = balances[index] + quantity
            if abs(projected) * price > max_notional:
                target_position = np.copysign(max_notional / price, projected)
                quantities[index] = float(target_position - balances[index])
        return _decision(order.quantities, quantities)


@dataclass(frozen=True)
class FractionalKellyPolicy(RiskPolicy):
    """Size long exposure to each outcome from probability and market price."""

    VERSION = "1.0.0"

    custom_fraction: float

    def __post_init__(self) -> None:
        if not isfinite(self.custom_fraction):
            raise ValueError("custom_fraction must be finite")
        if self.custom_fraction <= 0.0 or self.custom_fraction > 1.0:
            raise ValueError("custom_fraction must be in (0, 1]")

    def derisk(
        self,
        order: Order,
        state: MarketState,
        cash: float,
        balances: Array,
    ) -> RiskDecision:
        proposed = order.quantities
        probabilities = np.asarray(state.information, dtype=np.float64)
        prices = np.asarray(state.ask, dtype=np.float64)
        if probabilities.shape != prices.shape:
            raise ValueError("information and ask must have equal width")
        if not np.isfinite(probabilities).all():
            raise ValueError("predicted probabilities must be finite")
        if ((probabilities < 0.0) | (probabilities > 1.0)).any():
            raise ValueError("predicted probabilities must be within [0, 1]")
        if (prices <= 0.0).any():
            raise ValueError("ask prices must be positive")
        equity = cash + float(np.dot(balances, state.bid))
        if equity <= 0.0:
            effective = np.zeros_like(probabilities)
        else:
            denominator = 1.0 - prices
            with np.errstate(divide="ignore", invalid="ignore"):
                kelly = probabilities - (1.0 - probabilities) * prices / denominator
            kelly = np.where((denominator > 0.0) & (kelly > 0.0), kelly, 0.0)
            stake_cash = kelly * self.custom_fraction * equity
            effective = stake_cash / prices
        return _decision(proposed, effective)


__all__ = [
    "FractionalKellyPolicy",
    "FractionalLimitPolicy",
    "RiskDecision",
    "RiskPolicy",
    "RiskReason",
    "StatefulRiskPolicy",
]
