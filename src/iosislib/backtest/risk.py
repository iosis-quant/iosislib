"""Risk-policy contracts for the minimal event backtest."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields, is_dataclass
from enum import IntEnum
from math import copysign, isfinite
from typing import Any, ClassVar, cast

import numpy as np

from iosislib.backtest.policy import Array, MarketState, Order, PolicyState
from iosislib.core.utils import _canonical_json, _serialize_value

_SCALAR_WIDTH_THRESHOLD = 16


class RiskReason(IntEnum):
    """Stable integer reason describing what a risk policy did to an order."""

    NO_CHANGE = 0
    CLAMPED = 1
    ZEROED = 2


@dataclass(frozen=True)
class RiskDecision:
    """The risk-adjusted result for one tick: an order plus its reason.

    A passive value type for analytics and serialization; the backtest loop
    itself never constructs decisions.
    """

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


def classify_reason(proposed: Array, effective: Array) -> RiskReason:
    """Post-hoc classification: infer what a risk policy did to an order.

    Not used in the hot loop. Provided for post-batch analytics and tracing.
    """
    if np.array_equal(proposed, effective):
        return RiskReason.NO_CHANGE
    if not np.any(effective):
        return RiskReason.ZEROED
    return RiskReason.CLAMPED


class RiskPolicy(ABC):
    """Transform a proposed order into a risk-adjusted order before execution."""

    _SERIALIZE_WITH_TO_DICT: ClassVar[bool] = True
    VERSION: ClassVar[str]

    @abstractmethod
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
        """Write the effective order into ``orders[row]``.

        No validation, no classification. Just write the quantities and return
        the next risk state (or None for non-stateful policies).
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


@dataclass(frozen=True)
class NoOpRiskPolicy(RiskPolicy):
    """Pass-through risk policy; the executor uses it when none is configured."""

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
        del risk_state, state, cash, balances
        orders[row] = proposed[row]
        return None


NO_OP_RISK = NoOpRiskPolicy()


class StatefulRiskPolicy(RiskPolicy, ABC):
    """A risk policy whose immutable state remains local to one execution."""

    @abstractmethod
    def initial_state(self) -> PolicyState:
        """Return fresh state for one run; never retain it on the policy."""

    @abstractmethod
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
        """Write effective order into ``orders[row]`` and return new state."""


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
        bid_row = state.bid
        ask_row = state.ask
        proposed_row = proposed[row]
        target = orders[row]
        width = len(bid_row)
        equity = cash
        for i in range(width):
            equity += balances[i] * bid_row[i]
        max_notional = self.fraction * equity
        if max_notional <= 0.0:
            for i in range(width):
                target[i] = 0.0
            return None
        if width <= _SCALAR_WIDTH_THRESHOLD:
            for i in range(width):
                target[i] = proposed_row[i]
            for i in range(width):
                qty = target[i]
                if qty != 0.0:
                    price = ask_row[i] if qty > 0.0 else bid_row[i]
                    projected = balances[i] + qty
                    notional = abs(projected) * price
                    if notional > max_notional:
                        target[i] = (
                            copysign(max_notional / price, projected) - balances[i]
                        )
        else:
            target[:] = proposed_row
            prices = np.where(target > 0.0, ask_row, bid_row)
            projected = balances + target
            notional = np.abs(projected) * prices
            exceeds = (target != 0.0) & (notional > max_notional)
            if exceeds.any():
                target[exceeds] = (
                    np.copysign(max_notional / prices[exceeds], projected[exceeds])
                    - balances[exceeds]
                )
        return None


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
        probabilities = state.information
        ask_row = state.ask
        bid_row = state.bid
        target = orders[row]
        width = len(probabilities)
        equity = cash
        for i in range(width):
            equity += balances[i] * bid_row[i]
        if equity <= 0.0:
            for i in range(width):
                target[i] = 0.0
        elif width <= _SCALAR_WIDTH_THRESHOLD:
            fraction = self.custom_fraction * equity
            for i in range(width):
                denom = 1.0 - ask_row[i]
                if denom > 0.0:
                    k = (
                        probabilities[i]
                        - (1.0 - probabilities[i]) * ask_row[i] / denom
                    )
                    if k > 0.0:
                        target[i] = k * fraction / ask_row[i]
                        continue
                target[i] = 0.0
        else:
            denominator = 1.0 - ask_row
            kelly = probabilities - (1.0 - probabilities) * ask_row / denominator
            kelly = np.where((denominator > 0.0) & (kelly > 0.0), kelly, 0.0)
            stake_cash = kelly * self.custom_fraction * equity
            np.divide(stake_cash, ask_row, out=target)
        return None


__all__ = [
    "FractionalKellyPolicy",
    "FractionalLimitPolicy",
    "NO_OP_RISK",
    "NoOpRiskPolicy",
    "RiskDecision",
    "RiskPolicy",
    "RiskReason",
    "StatefulRiskPolicy",
    "classify_reason",
]
