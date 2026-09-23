"""Fixed maker/taker fee schedules for graph-native backtests."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from math import isfinite
from typing import Any, ClassVar, cast

from iosislib.core.utils import _canonical_json, _serialize_value


class FeeSchedule(ABC):
    """Quote a signed per-notional fee rate by liquidity role.

    Rates are fractions of fill notional: ``fee = |qty| * price * rate``.
    Positive rates charge cash; negative rates pay rebates.  Maker rates
    apply to rested (GTC working-order) fills; taker rates apply to
    market orders and crossing limit orders.
    """

    _SERIALIZE_WITH_TO_DICT: ClassVar[bool] = True
    VERSION: ClassVar[str]

    @property
    @abstractmethod
    def taker_rate(self) -> float:
        """Return the signed per-notional taker rate."""

    @property
    @abstractmethod
    def maker_rate(self) -> float:
        """Return the signed per-notional maker rate."""

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
class FixedFeeSchedule(FeeSchedule):
    """One signed taker rate and one signed maker rate for every fill."""

    VERSION = "1.0.0"
    taker_rate: float = 0.0
    maker_rate: float = 0.0

    def __post_init__(self) -> None:
        for name in ("taker_rate", "maker_rate"):
            rate = getattr(self, name)
            if isinstance(rate, bool) or not isinstance(rate, (int, float)):
                raise TypeError(f"{name} must be a number")
            if not isfinite(rate):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, float(rate))


_FEE_REGISTRY: dict[str, type[FeeSchedule]] = {
    "fixed": FixedFeeSchedule,
}


def register_fee_schedule(kind: str, cls: type[FeeSchedule]) -> None:
    """Register a custom FeeSchedule class under a declarative ``kind`` name."""
    if not isinstance(kind, str) or not kind.strip():
        raise ValueError("fee schedule kind must be a non-empty string")
    if not isinstance(cls, type) or not issubclass(cls, FeeSchedule):
        raise TypeError("cls must be a concrete FeeSchedule subclass")
    if kind in _FEE_REGISTRY:
        raise ValueError(f"fee schedule kind {kind!r} is already registered")
    _FEE_REGISTRY[kind] = cls


def list_fee_schedules() -> dict[str, type[FeeSchedule]]:
    """Return the current fee-schedule registry."""
    return dict(_FEE_REGISTRY)


def _fee_from_declaration(value: Mapping[str, Any]) -> FeeSchedule:
    """Resolve a declarative fee-schedule mapping into a concrete instance."""
    kind = value.get("kind", "fixed")
    cls = _FEE_REGISTRY.get(kind)
    if cls is None:
        raise ValueError(
            f"Unknown fee schedule kind: {kind!r}; "
            f"available: {sorted(_FEE_REGISTRY)}"
        )
    params = {k: v for k, v in value.items() if k != "kind"}
    return cls(**params)


def _normalize_fee_schedule(value: FeeSchedule | Mapping[str, Any] | None) -> FeeSchedule | None:
    if value is None:
        return None
    if isinstance(value, FeeSchedule):
        return value
    if isinstance(value, Mapping):
        return _fee_from_declaration(value)
    raise TypeError("fee_schedule must be a FeeSchedule, a declarative mapping, or None")


__all__ = [
    "FeeSchedule",
    "FixedFeeSchedule",
    "list_fee_schedules",
    "register_fee_schedule",
]
