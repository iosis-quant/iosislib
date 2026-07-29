"""Market-feed declarations for graph-native backtests."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

import polars as pl

from iosislib.core.tsfn import ColumnEntry, FrameSignature, TimeAxis
from iosislib.core.utils import _canonical_json

from iosislib.backtest.venue import Venue


@dataclass(frozen=True)
class Feed(ABC):
    """Declare a market schema and extract executable quotes from a frame."""

    _SERIALIZE_WITH_TO_DICT: ClassVar[bool] = True
    VERSION: ClassVar[str]

    venue: Venue
    time_axis: TimeAxis = TimeAxis()

    def __post_init__(self) -> None:
        if not isinstance(self.venue, Venue):
            raise TypeError("venue must be a Venue")
        if not isinstance(self.time_axis, TimeAxis):
            raise TypeError("time_axis must be a TimeAxis")

    @property
    def width(self) -> int:
        """Return the number of quote values expected on each row."""
        return self.venue.width

    @property
    @abstractmethod
    def columns(self) -> tuple[ColumnEntry, ...]:
        """Return physical graph columns required by this feed."""

    @abstractmethod
    def quotes(self, frame: pl.DataFrame) -> tuple[pl.Series, pl.Series]:
        """Return one array-valued bid and ask series for every frame row."""

    def frame_signature(self) -> FrameSignature:
        """Return the full physical input contract for this feed."""
        return FrameSignature(time=self.time_axis, columns=self.columns)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": f"{type(self).__module__}.{type(self).__qualname__}",
            "version": self.VERSION,
            "venue": self.venue.to_dict(),
            "time_axis": {
                "column": self.time_axis.column,
                "dtype": str(self.time_axis.dtype),
                "timezone": self.time_axis.timezone,
            },
        }

    def __str__(self) -> str:
        return _canonical_json(self.to_dict())


@dataclass(frozen=True)
class L1Feed(Feed):
    """Best bid and ask quote vectors for every asset and timestamp."""

    VERSION = "1.0.0"
    bid_column: str = "bid"
    ask_column: str = "ask"

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.bid_column or not self.ask_column:
            raise ValueError("quote column names cannot be empty")
        if self.bid_column == self.ask_column:
            raise ValueError("bid and ask columns must differ")

    @property
    def columns(self) -> tuple[ColumnEntry, ...]:
        shape = (self.width,)
        return (
            (self.bid_column, pl.Float64, shape),
            (self.ask_column, pl.Float64, shape),
        )

    def quotes(self, frame: pl.DataFrame) -> tuple[pl.Series, pl.Series]:
        return frame.get_column(self.bid_column), frame.get_column(self.ask_column)

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "bid_column": self.bid_column,
            "ask_column": self.ask_column,
        }


__all__ = ["Feed", "L1Feed"]
