"""Stable market-universe declarations for backtests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from iosislib.core.utils import _canonical_json


@dataclass(frozen=True)
class Venue:
    """The ordered asset universe shared by a feed and a backtest."""

    _SERIALIZE_WITH_TO_DICT: ClassVar[bool] = True
    VERSION: ClassVar[str] = "1.0.0"

    name: str
    universe: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string")
        if not isinstance(self.universe, tuple) or not self.universe:
            raise ValueError("universe must be a non-empty tuple")
        if any(
            not isinstance(asset, str) or not asset.strip() for asset in self.universe
        ):
            raise ValueError("universe entries must be non-empty strings")
        if len(set(self.universe)) != len(self.universe):
            raise ValueError("universe entries must be unique")

    @property
    def width(self) -> int:
        """Return the fixed number of assets in the ordered universe."""
        return len(self.universe)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": f"{type(self).__module__}.{type(self).__qualname__}",
            "version": self.VERSION,
            "name": self.name,
            "universe": list(self.universe),
        }

    def __str__(self) -> str:
        return _canonical_json(self.to_dict())


__all__ = ["Venue"]
