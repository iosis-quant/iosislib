"""Market-feed declarations for graph-native backtests."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

import polars as pl

from iosislib.core.tsfn import ColumnEntry, FrameSignature, TimeAxis
from iosislib.core.utils import _canonical_json, series_to_numpy

from iosislib.backtest.venue import Venue

import numpy as np


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


@dataclass(frozen=True)
class L2Feed(Feed):
    VERSION = "1.0.0"
    bid_depth_column: str = "bid_depth"
    ask_depth_column: str = "ask_depth"
    depth_levels: int = 101
    tick: float = 0.01

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.bid_depth_column or not self.ask_depth_column:
            raise ValueError("depth column names cannot be empty")
        if self.bid_depth_column == self.ask_depth_column:
            raise ValueError("bid and ask depth columns must differ")
        if (
            isinstance(self.depth_levels, bool)
            or not isinstance(self.depth_levels, int)
            or self.depth_levels < 1
        ):
            raise ValueError("depth_levels must be a positive integer")
        if isinstance(self.tick, bool) or not isinstance(
            self.tick, (int, float)
        ):
            raise TypeError("tick must be a number")
        if not np.isfinite(self.tick):
            raise ValueError("tick must be finite")
        if self.tick <= 0.0 or self.tick > 1.0:
            raise ValueError("tick must be in (0, 1]")

    @property
    def columns(self) -> tuple[ColumnEntry, ...]:
        shape = (self.width, self.depth_levels)
        return (
            (self.bid_depth_column, pl.Float64, shape),
            (self.ask_depth_column, pl.Float64, shape),
        )

    def depth(self, frame: pl.DataFrame) -> tuple[pl.Series, pl.Series]:
        return (
            frame.get_column(self.bid_depth_column),
            frame.get_column(self.ask_depth_column),
        )

    def quotes(self, frame: pl.DataFrame) -> tuple[pl.Series, pl.Series]:
        from iosislib.core.utils import numpy_to_series

        bid_series, ask_series = self.depth(frame)
        shape = (self.width, self.depth_levels)
        bid = np.asarray(
            series_to_numpy(bid_series, shape=shape, allow_copy=True),
            dtype=np.float64,
        )
        ask = np.asarray(
            series_to_numpy(ask_series, shape=shape, allow_copy=True),
            dtype=np.float64,
        )
        levels = self.depth_levels
        grid = np.arange(levels, dtype=np.float64) * self.tick
        bid_mask = bid > 0.0
        ask_mask = ask > 0.0
        bid_idx = levels - 1 - bid_mask[:, :, ::-1].argmax(axis=-1)
        ask_idx = ask_mask.argmax(axis=-1)
        best_bid_mat = np.where(
            bid_mask.any(axis=-1),
            np.clip(grid[bid_idx], self.tick, 1.0),
            self.tick,
        )
        best_ask_mat = np.where(
            ask_mask.any(axis=-1),
            np.clip(grid[ask_idx], self.tick, 1.0),
            1.0,
        )
        return (
            numpy_to_series("bid", best_bid_mat, shape=(self.width,), allow_copy=True),
            numpy_to_series("ask", best_ask_mat, shape=(self.width,), allow_copy=True),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "bid_depth_column": self.bid_depth_column,
            "ask_depth_column": self.ask_depth_column,
            "depth_levels": self.depth_levels,
            "tick": self.tick,
        }


__all__ = ["Feed", "L1Feed", "L2Feed"]
