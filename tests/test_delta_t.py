from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import ClassVar

import polars as pl
import pytest

from iosislib.core.graph import Graph
from iosislib.core.node import Node
from iosislib.core.tsfn import (
    DELTA_T_COLUMN,
    FrameSignature,
    TSFN,
    TSFNConfig,
    TimeAxis,
    delta_t_dtype,
)
from iosislib.perturb import BlockPermute
from iosislib.perturb.block_permute import _permute_order
from iosislib.tsfn.transforms import Lag, Negate

MINUTES = (0, 1, 3, 6)
VALUES = (10.0, 11.0, 12.0, 13.0)
SOURCE_DELTAS = [
    None,
    timedelta(minutes=1),
    timedelta(minutes=2),
    timedelta(minutes=3),
]
PAIR_FRAME = FrameSignature(
    columns=(("left", pl.Float64), ("right", pl.Float64))
)


def _timestamps() -> list[datetime]:
    return [datetime(2026, 1, 1) + timedelta(minutes=minute) for minute in MINUTES]


@dataclass(frozen=True)
class SeriesConfig(TSFNConfig):
    minutes: tuple[int, ...] = MINUTES
    values: tuple[float, ...] = VALUES


class SeriesSource(TSFN[SeriesConfig]):
    VERSION = "1.0.0"
    CONFIG_CLS = SeriesConfig

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        return FrameSignature.empty(), FrameSignature(columns=(("value", pl.Float64),))

    def apply(self) -> pl.LazyFrame:
        return pl.DataFrame(
            {
                "timestamp": [
                    datetime(2026, 1, 1) + timedelta(minutes=minute)
                    for minute in self.parameters.minutes
                ],
                "value": list(self.parameters.values),
            },
            schema={"timestamp": pl.Datetime, "value": pl.Float64},
        ).lazy()


@dataclass(frozen=True)
class SpyConfig(TSFNConfig):
    input_column: str = "value"
    output_column: str = "value"


class InputSpy(TSFN[SpyConfig]):
    """Terminal node that records the aligned frame it receives."""

    VERSION = "1.0.0"
    CONFIG_CLS = SpyConfig
    seen: ClassVar[list[pl.DataFrame]] = []

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        params = self.parameters
        axis = TimeAxis()
        return (
            FrameSignature(time=axis, columns=((params.input_column, pl.Float64),)),
            FrameSignature(time=axis, columns=((params.output_column, pl.Float64),)),
        )

    def apply(self, lf: pl.LazyFrame | None = None) -> pl.LazyFrame:
        if lf is None:
            raise ValueError("InputSpy requires an input frame")
        InputSpy.seen.append(lf.collect())
        params = self.parameters
        return lf.select(
            "timestamp", pl.col(params.input_column).alias(params.output_column)
        )


class PairSpy(TSFN):
    """Two-input terminal node that records its aligned union-timeline frame."""

    VERSION = "1.0.0"
    seen: ClassVar[list[pl.DataFrame]] = []

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        return PAIR_FRAME, PAIR_FRAME

    def apply(self, lf: pl.LazyFrame | None = None) -> pl.LazyFrame:
        if lf is None:
            raise ValueError("PairSpy requires an input frame")
        PairSpy.seen.append(lf.collect())
        return lf.select("timestamp", "left", "right")


class EchoDelta(TSFN):
    """Returns declared outputs plus a correctly-typed trailing delta_t."""

    VERSION = "1.0.0"

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        frame = FrameSignature(columns=(("value", pl.Float64),))
        return frame, frame

    def apply(self, lf: pl.LazyFrame | None = None) -> pl.LazyFrame:
        if lf is None:
            raise ValueError("EchoDelta requires an input frame")
        return lf.select(
            "timestamp",
            "value",
            pl.col("timestamp").diff().alias(DELTA_T_COLUMN),
        )


class EchoDeltaMiddle(EchoDelta):
    """Places delta_t between declared columns instead of at the end."""

    VERSION = "1.0.1"

    def apply(self, lf: pl.LazyFrame | None = None) -> pl.LazyFrame:
        if lf is None:
            raise ValueError("EchoDeltaMiddle requires an input frame")
        return lf.select(
            "timestamp",
            pl.col("timestamp").diff().alias(DELTA_T_COLUMN),
            "value",
        )


class EchoDeltaBadDtype(EchoDelta):
    """Emits delta_t with a mismatched Duration unit."""

    VERSION = "1.0.2"

    def apply(self, lf: pl.LazyFrame | None = None) -> pl.LazyFrame:
        if lf is None:
            raise ValueError("EchoDeltaBadDtype requires an input frame")
        return lf.select(
            "timestamp",
            "value",
            pl.lit(None, dtype=pl.Duration("ms")).alias(DELTA_T_COLUMN),
        )


def _plain_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {"timestamp": _timestamps(), "value": VALUES},
        schema={"timestamp": pl.Datetime, "value": pl.Float64},
    )


@pytest.fixture(autouse=True)
def _clear_spies():
    InputSpy.seen.clear()
    PairSpy.seen.clear()
    yield
    InputSpy.seen.clear()
    PairSpy.seen.clear()


def test_delta_t_dtype_follows_the_time_axis() -> None:
    assert delta_t_dtype(TimeAxis()) == pl.Duration("us")
    assert delta_t_dtype(TimeAxis(dtype=pl.Datetime("ms"))) == pl.Duration("ms")
    assert delta_t_dtype(TimeAxis(dtype=pl.Int64)) == pl.Int64


def test_frame_signature_rejects_delta_t_as_value_column() -> None:
    with pytest.raises(ValueError, match="read-only"):
        FrameSignature(columns=((DELTA_T_COLUMN, pl.Duration("us")),))


def test_frame_signature_rejects_delta_t_as_time_column() -> None:
    with pytest.raises(ValueError, match="read-only"):
        FrameSignature(
            time=TimeAxis(column=DELTA_T_COLUMN),
            columns=(("value", pl.Float64),),
        )


def test_source_terminal_returns_declared_columns_only() -> None:
    result = Graph(Node(SeriesSource)).execute()

    assert result.columns == ["timestamp", "value"]
    assert result.height == len(MINUTES)


def test_delta_t_is_injected_after_the_source_and_reaches_the_child() -> None:
    source = Node(SeriesSource)
    spy = Node(InputSpy, bindings={"value": source.output("value")})

    result = Graph(spy).execute()

    assert len(InputSpy.seen) == 1
    frame = InputSpy.seen[0]
    assert frame.schema[DELTA_T_COLUMN] == pl.Duration("us")
    assert frame[DELTA_T_COLUMN].to_list() == SOURCE_DELTAS
    assert frame["timestamp"].to_list() == _timestamps()
    assert result.columns == ["timestamp", "value"]


def test_delta_t_is_carried_unchanged_through_a_transform() -> None:
    source = Node(SeriesSource)
    lag = Node(
        Lag,
        bindings={"value": source.output("value")},
        parameters={"periods": 1},
    )
    spy = Node(
        InputSpy,
        bindings={"lag": lag.output("lag")},
        parameters={"input_column": "lag"},
    )

    result = Graph(spy).execute()

    frame = InputSpy.seen[0]
    assert frame[DELTA_T_COLUMN].to_list() == SOURCE_DELTAS
    assert frame["lag"].to_list() == [None, 10.0, 11.0, 12.0]
    assert result.columns == ["timestamp", "value"]


def test_block_permute_gathers_delta_t_with_the_payload() -> None:
    source = Node(SeriesSource)
    permute = Node(
        BlockPermute,
        bindings={"value": source.output("value")},
        parameters={"block_length": 2, "seed": 7},
    )
    spy = Node(InputSpy, bindings={"value": permute.output("value")})

    result = Graph(spy).execute()

    order = _permute_order(len(MINUTES), 2, 7)
    frame = InputSpy.seen[0]
    assert frame["timestamp"].to_list() == _timestamps()
    assert frame["value"].to_list() == [VALUES[i] for i in order]
    assert frame[DELTA_T_COLUMN].to_list() == [SOURCE_DELTAS[i] for i in order]
    assert result.columns == ["timestamp", "value"]


def test_permuted_delta_t_survives_an_itemwise_chain() -> None:
    source = Node(SeriesSource)
    permute = Node(
        BlockPermute,
        bindings={"value": source.output("value")},
        parameters={"block_length": 2, "seed": 7},
    )
    negate = Node(Negate, bindings={"value": permute.output("value")})
    spy = Node(
        InputSpy,
        bindings={"negated": negate.output("negated")},
        parameters={"input_column": "negated"},
    )

    Graph(spy).execute()

    order = _permute_order(len(MINUTES), 2, 7)
    frame = InputSpy.seen[0]
    assert frame[DELTA_T_COLUMN].to_list() == [SOURCE_DELTAS[i] for i in order]
    assert frame["negated"].to_list() == [-VALUES[i] for i in order]


def test_multi_parent_join_recomputes_delta_t_over_the_union_timeline() -> None:
    left = Node(
        SeriesSource,
        parameters={"minutes": (0, 2, 5), "values": (1.0, 2.0, 3.0)},
    )
    right = Node(
        SeriesSource,
        parameters={"minutes": (1, 3), "values": (10.0, 20.0)},
    )
    pair = Node(
        PairSpy,
        bindings={"left": left.output("value"), "right": right.output("value")},
    )

    result = Graph(pair).execute()

    frame = PairSpy.seen[0]
    assert frame["timestamp"].to_list() == [
        datetime(2026, 1, 1),
        datetime(2026, 1, 1, 0, 1),
        datetime(2026, 1, 1, 0, 2),
        datetime(2026, 1, 1, 0, 3),
        datetime(2026, 1, 1, 0, 5),
    ]
    assert frame[DELTA_T_COLUMN].to_list() == [
        None,
        timedelta(minutes=1),
        timedelta(minutes=1),
        timedelta(minutes=1),
        timedelta(minutes=2),
    ]
    assert result.columns == ["timestamp", "left", "right"]


def test_permuted_delta_t_dies_at_a_multi_parent_join() -> None:
    left = Node(
        SeriesSource,
        parameters={"minutes": (0, 2, 5), "values": (1.0, 2.0, 3.0)},
    )
    permute = Node(
        BlockPermute,
        bindings={"value": left.output("value")},
        parameters={"block_length": 3, "seed": 11},
    )
    right = Node(
        SeriesSource,
        parameters={"minutes": (1, 3), "values": (10.0, 20.0)},
    )
    pair = Node(
        PairSpy,
        bindings={"left": permute.output("value"), "right": right.output("value")},
    )

    Graph(pair).execute()

    frame = PairSpy.seen[0]
    assert frame[DELTA_T_COLUMN].to_list() == [
        None,
        timedelta(minutes=1),
        timedelta(minutes=1),
        timedelta(minutes=1),
        timedelta(minutes=2),
    ]


def test_output_validation_accepts_delta_t_as_the_last_column() -> None:
    fn = EchoDelta({})

    out = fn(_plain_frame().lazy()).collect()

    assert out.columns == ["timestamp", "value", DELTA_T_COLUMN]


def test_output_validation_rejects_delta_t_in_the_wrong_position() -> None:
    fn = EchoDeltaMiddle({})

    with pytest.raises(ValueError, match="must exactly match"):
        fn(_plain_frame().lazy())


def test_output_validation_rejects_delta_t_with_a_wrong_dtype() -> None:
    fn = EchoDeltaBadDtype({})

    with pytest.raises(TypeError, match=f"'{DELTA_T_COLUMN}' type mismatch"):
        fn(_plain_frame().lazy())
