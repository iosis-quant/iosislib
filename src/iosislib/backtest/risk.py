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
    """The risk-adjusted result for one tick: an order plus its reason.

    A passive value type for analytics and serialization; the backtest loop
    itself never constructs decisions and instead reasons are written per row.
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


def _reason_for(proposed: Array, effective: Array) -> RiskReason:
    """Classify an effective order against the proposed one."""
    if proposed.shape != effective.shape:
        raise ValueError("effective order must match the proposed order width")
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
    ) -> tuple[RiskReason, PolicyState | None]:
        """Write the effective order into ``orders[row]``.

        ``proposed`` is the full batch buffer of policy-requested quantities.
        Returns ``(reason, new_risk_state_or_None)``.
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
    ) -> tuple[RiskReason, PolicyState | None]:
        del risk_state, state, cash, balances
        orders[row] = proposed[row]
        return (RiskReason.NO_CHANGE, None)


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
    ) -> tuple[RiskReason, PolicyState]:
        """Write effective order into ``orders[row]`` and return (reason, new_state)."""


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
    ) -> tuple[RiskReason, PolicyState | None]:
        equity = cash + float(np.dot(balances, state.bid))
        max_notional = self.fraction * equity
        target = orders[row]
        target[:] = proposed[row]
        if max_notional <= 0.0:
            target[:] = 0.0
            return (_reason_for(proposed[row], target), None)
        prices = np.where(target > 0.0, state.ask, state.bid)
        projected = balances + target
        notional = np.abs(projected) * prices
        exceeds = (target != 0.0) & (notional > max_notional)
        if exceeds.any():
            target[exceeds] = (
                np.copysign(max_notional / prices[exceeds], projected[exceeds])
                - balances[exceeds]
            )
        return (_reason_for(proposed[row], target), None)


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
    ) -> tuple[RiskReason, PolicyState | None]:
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
        target = orders[row]
        if equity <= 0.0:
            target.fill(0.0)
        else:
            denominator = 1.0 - prices
            with np.errstate(divide="ignore", invalid="ignore"):
                kelly = probabilities - (1.0 - probabilities) * prices / denominator
            kelly = np.where((denominator > 0.0) & (kelly > 0.0), kelly, 0.0)
            stake_cash = kelly * self.custom_fraction * equity
            np.divide(stake_cash, prices, out=target)
        return (_reason_for(proposed[row], target), None)


__all__ = [
    "FractionalKellyPolicy",
    "FractionalLimitPolicy",
    "NO_OP_RISK",
    "NoOpRiskPolicy",
    "RiskDecision",
    "RiskPolicy",
    "RiskReason",
    "StatefulRiskPolicy",
]
