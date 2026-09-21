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
    """Level-2 book with N price+volume levels per side of the midpoint.

    Each side is stored best-first as a fixed-size ladder: ``bid_price``
    descends from the best bid toward the price floor, ``ask_price``
    ascends from the best ask toward the price cap.  Prices are always
    populated (a linear ladder by convention, see :func:`l2_ladder_prices`);
    empty outer levels simply carry zero volume, so no price padding or
    null sentinel is needed.
    """

    VERSION = "2.0.0"
    bid_price_column: str = "bid_price"
    bid_volume_column: str = "bid_volume"
    ask_price_column: str = "ask_price"
    ask_volume_column: str = "ask_volume"
    depth_levels: int = 25

    def __post_init__(self) -> None:
        super().__post_init__()
        names = (
            self.bid_price_column,
            self.bid_volume_column,
            self.ask_price_column,
            self.ask_volume_column,
        )
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("L2 column names cannot be empty")
        if len(set(names)) != 4:
            raise ValueError("L2 column names must be distinct")
        if (
            isinstance(self.depth_levels, bool)
            or not isinstance(self.depth_levels, int)
            or self.depth_levels < 1
        ):
            raise ValueError("depth_levels must be a positive integer")

    @property
    def columns(self) -> tuple[ColumnEntry, ...]:
        shape = (self.width, self.depth_levels)
        return (
            (self.bid_price_column, pl.Float64, shape),
            (self.bid_volume_column, pl.Float64, shape),
            (self.ask_price_column, pl.Float64, shape),
            (self.ask_volume_column, pl.Float64, shape),
        )

    def depth(
        self, frame: pl.DataFrame
    ) -> tuple[pl.Series, pl.Series, pl.Series, pl.Series]:
        return (
            frame.get_column(self.bid_price_column),
            frame.get_column(self.bid_volume_column),
            frame.get_column(self.ask_price_column),
            frame.get_column(self.ask_volume_column),
        )

    def quotes(self, frame: pl.DataFrame) -> tuple[pl.Series, pl.Series]:
        from iosislib.core.utils import numpy_to_series

        bid_price, bid_volume, ask_price, ask_volume = self.depth(frame)
        shape = (self.width, self.depth_levels)
        bid_px = np.asarray(
            series_to_numpy(bid_price, shape=shape, allow_copy=True),
            dtype=np.float64,
        )
        bid_vol = np.asarray(
            series_to_numpy(bid_volume, shape=shape, allow_copy=True),
            dtype=np.float64,
        )
        ask_px = np.asarray(
            series_to_numpy(ask_price, shape=shape, allow_copy=True),
            dtype=np.float64,
        )
        ask_vol = np.asarray(
            series_to_numpy(ask_volume, shape=shape, allow_copy=True),
            dtype=np.float64,
        )
        # Best-first ladders: first level with resting volume is executable.
        # A fully empty side falls back to the ladder head so marks stay defined.
        bid_hit = bid_vol > 1e-12
        ask_hit = ask_vol > 1e-12
        bid_idx = np.where(
            bid_hit.any(axis=-1), bid_hit.argmax(axis=-1), 0
        )
        ask_idx = np.where(
            ask_hit.any(axis=-1), ask_hit.argmax(axis=-1), 0
        )
        best_bid_mat = np.where(
            bid_hit.any(axis=-1),
            np.take_along_axis(bid_px, bid_idx[:, :, None], axis=-1)[:, :, 0],
            bid_px[:, :, 0],
        )
        best_ask_mat = np.where(
            ask_hit.any(axis=-1),
            np.take_along_axis(ask_px, ask_idx[:, :, None], axis=-1)[:, :, 0],
            ask_px[:, :, 0],
        )
        return (
            numpy_to_series("bid", best_bid_mat, shape=(self.width,), allow_copy=True),
            numpy_to_series("ask", best_ask_mat, shape=(self.width,), allow_copy=True),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "bid_price_column": self.bid_price_column,
            "ask_price_column": self.ask_price_column,
            "bid_volume_column": self.bid_volume_column,
            "ask_volume_column": self.ask_volume_column,
            "depth_levels": self.depth_levels,
        }


def l2_ladder_prices(best: float, bound: float, levels: int) -> np.ndarray:
    """Return a fixed-size linear price ladder from ``best`` to ``bound``.

    Both endpoints are included, so an empty outer level still carries a
    valid price with zero volume.  Bids typically use ``bound=0.0`` and
    asks ``bound=1.0`` for Polymarket-style binary books.
    """
    if not np.isfinite(best) or not np.isfinite(bound):
        raise ValueError("best and bound must be finite")
    if isinstance(levels, bool) or not isinstance(levels, int) or levels < 1:
        raise ValueError("levels must be a positive integer")
    if levels == 1:
        return np.array([float(best)], dtype=np.float64)
    return np.linspace(float(best), float(bound), levels, dtype=np.float64)


def l2_interpolate_volumes(
    ladder: np.ndarray,
    known_prices: np.ndarray,
    known_volumes: np.ndarray,
) -> np.ndarray:
    """Linearly interpolate sparse ``known`` volumes onto a fixed ladder.

    Volumes outside the observed price range map to zero.  ``ladder`` may
    ascend or descend; interpolation is performed in ascending price order.
    """
    ladder_arr = np.asarray(ladder, dtype=np.float64)
    xp = np.asarray(known_prices, dtype=np.float64)
    fp = np.asarray(known_volumes, dtype=np.float64)
    if ladder_arr.ndim != 1 or xp.ndim != 1 or fp.ndim != 1:
        raise ValueError("ladder, known_prices and known_volumes must be 1-D")
    if xp.shape != fp.shape:
        raise ValueError("known_prices and known_volumes must have equal length")
    if xp.size == 0:
        return np.zeros_like(ladder_arr)
    if not np.isfinite(ladder_arr).all():
        raise ValueError("ladder prices must be finite")
    order = np.argsort(xp, kind="stable")
    return np.interp(ladder_arr, xp[order], np.maximum(fp[order], 0.0), left=0.0, right=0.0)


def dense_l2_side_from_sparse(
    best: float,
    bound: float,
    levels: int,
    known_prices: np.ndarray,
    known_volumes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a (prices, volumes) ladder from a sparse side of a book."""
    prices = l2_ladder_prices(best, bound, levels)
    volumes = l2_interpolate_volumes(prices, known_prices, known_volumes)
    return prices, volumes


__all__ = [
    "Feed",
    "L1Feed",
    "L2Feed",
    "dense_l2_side_from_sparse",
    "l2_interpolate_volumes",
    "l2_ladder_prices",
]
