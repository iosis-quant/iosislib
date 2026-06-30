from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import polars as pl
import pytest

from src.classes import FrameSignature, Graph, Node, TSFN, TSFNConfig, TimeAxis

VALUE_FRAME = FrameSignature(columns=(("value", pl.Int64),))
COMBINED_FRAME = FrameSignature(columns=(("left", pl.Int64), ("right", pl.Int64)))
TRIPLE_FRAME = FrameSignature(
    columns=(("left", pl.Int64), ("right", pl.Int64), ("anchor", pl.Int64))
)


@dataclass(frozen=True)
class SeriesConfig(TSFNConfig):
    minutes: tuple[int, ...]
    values: tuple[int, ...]


class SeriesSource(TSFN):
    VERSION = "1.0.0"
    CONFIG_CLS = SeriesConfig

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        return FrameSignature.empty(), VALUE_FRAME

    def apply(self) -> pl.LazyFrame:
        return pl.DataFrame(
            {
                "timestamp": [dt(minute) for minute in self.parameters.minutes],
                "value": list(self.parameters.values),
            }
        ).lazy()


class CombineValues(TSFN):
    VERSION = "1.0.0"

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        return COMBINED_FRAME, COMBINED_FRAME

    def apply(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.select("timestamp", "left", "right")


class CombineThreeValues(TSFN):
    VERSION = "1.0.0"

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        return TRIPLE_FRAME, TRIPLE_FRAME

    def apply(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.select("timestamp", "left", "right", "anchor")


class NeedsInput(TSFN):
    VERSION = "1.0.0"

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        return VALUE_FRAME, VALUE_FRAME

    def apply(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.select("timestamp", "value")


class EmptyInputTransform(TSFN):
    VERSION = "1.0.0"

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        return FrameSignature.empty(), VALUE_FRAME

    def apply(self) -> pl.LazyFrame:
        return pl.DataFrame({"timestamp": [dt(0)], "value": [1]}).lazy()


class MissingTimeSource(TSFN):
    VERSION = "1.0.0"

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        return FrameSignature.empty(), VALUE_FRAME

    def apply(self) -> pl.LazyFrame:
        return pl.DataFrame({"value": [1]}).lazy()


class WrongTimeSource(TSFN):
    VERSION = "1.0.0"

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        return FrameSignature.empty(), VALUE_FRAME

    def apply(self) -> pl.LazyFrame:
        return pl.DataFrame({"timestamp": ["2026-01-01"], "value": [1]}).lazy()


class MissingValueSource(TSFN):
    VERSION = "1.0.0"

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        return FrameSignature.empty(), VALUE_FRAME

    def apply(self) -> pl.LazyFrame:
        return pl.DataFrame({"timestamp": [dt(0)]}).lazy()


class FloatSource(TSFN):
    VERSION = "1.0.0"

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        frame = FrameSignature(columns=(("value", pl.Float64),))
        return FrameSignature.empty(), frame

    def apply(self) -> pl.LazyFrame:
        return pl.DataFrame({"timestamp": [dt(0)], "value": [1.0]}).lazy()


def dt(minute: int) -> datetime:
    return datetime(2026, 1, 1, 0, minute)


def source_node(
    name: str,
    minutes: tuple[int, ...],
    values: tuple[int, ...],
) -> Node:
    return Node(
        SeriesSource,
        parameters={"minutes": minutes, "values": values},
        name=name,
    )


def test_frame_signature_empty_accepts_only_no_time_and_no_columns() -> None:
    assert FrameSignature.empty().is_empty()

    with pytest.raises(ValueError, match="cannot declare value columns"):
        FrameSignature(time=None, columns=(("value", pl.Int64),))


def test_inputless_tsfn_apply_is_called_without_frame() -> None:
    source = source_node("source", (0, 1), (10, 20))

    result = Graph(source).execute()

    assert result["timestamp"].to_list() == [dt(0), dt(1)]
    assert result["value"].to_list() == [10, 20]


def test_graph_compile_returns_and_caches_root_lazyframe() -> None:
    source = source_node("source", (0, 1), (10, 20))
    graph = Graph(source)

    compiled = graph.compile()

    assert isinstance(compiled, pl.LazyFrame)
    assert graph.compile() is compiled
    assert graph._compiled_root_lf is compiled
    assert graph.execute()["value"].to_list() == [10, 20]


def test_inputful_tsfn_receives_lazyframe() -> None:
    source = source_node("source", (0, 1), (10, 20))
    transform = Node(
        NeedsInput,
        bindings={"value": source.value},
        name="transform",
    )

    result = Graph(transform).execute()

    assert result["timestamp"].to_list() == [dt(0), dt(1)]
    assert result["value"].to_list() == [10, 20]


def test_no_predecessor_node_with_non_empty_input_signature_fails_validation() -> None:
    with pytest.raises(ValueError, match="empty input signature"):
        Graph(Node(NeedsInput, name="bad_source"))


def test_bound_node_with_empty_input_signature_fails_validation() -> None:
    source = source_node("source", (0,), (1,))

    with pytest.raises(ValueError, match="Unexpected bound inputs"):
        Graph(Node(EmptyInputTransform, bindings={"value": source.value}))


def test_output_schema_validation_catches_missing_or_wrong_columns() -> None:
    with pytest.raises(RuntimeError, match="Execution failed.*Missing required output time column"):
        Graph(Node(MissingTimeSource)).execute()

    with pytest.raises(RuntimeError, match="Execution failed.*Time column 'timestamp' type mismatch"):
        Graph(Node(WrongTimeSource)).execute()

    with pytest.raises(RuntimeError, match="Execution failed.*Missing required output column: 'value'"):
        Graph(Node(MissingValueSource)).execute()


def test_union_timeline_uses_backward_asof_without_lookahead() -> None:
    left = source_node("left", (0, 2), (10, 20))
    right = source_node("right", (1, 3), (100, 300))
    combined = Node(
        CombineValues,
        bindings={
            "left": left.value,
            "right": right.value,
        },
        name="combined",
    )

    result = Graph(combined).execute()

    assert result["timestamp"].to_list() == [dt(0), dt(1), dt(2), dt(3)]
    assert result["left"].to_list() == [10, 10, 20, 20]
    assert result["right"].to_list() == [None, 100, 100, 300]


def test_default_input_tolerance_is_unbounded_for_asof_alignment() -> None:
    left = source_node("left", (0,), (10,))
    right = source_node("right", (3,), (300,))
    combined = Node(
        CombineValues,
        bindings={
            "left": left.value,
            "right": right.value,
        },
        name="combined",
    )

    result = Graph(combined).execute()

    assert combined.tolerances == {}
    assert result["timestamp"].to_list() == [dt(0), dt(3)]
    assert result["left"].to_list() == [10, 10]
    assert result["right"].to_list() == [None, 300]


def test_consumer_input_tolerance_limits_backward_asof_alignment() -> None:
    left = source_node("left", (0,), (10,))
    right = source_node("right", (2, 3), (200, 300))
    combined = Node(
        CombineValues,
        bindings={
            "left": left.value,
            "right": right.value,
        },
        tolerances={"left": timedelta(minutes=1)},
        name="combined",
    )

    result = Graph(combined).execute()

    assert result["timestamp"].to_list() == [dt(0), dt(2), dt(3)]
    assert result["left"].to_list() == [10, None, None]
    assert result["right"].to_list() == [None, 200, 300]


def test_same_parent_can_feed_inputs_with_different_consumer_tolerances() -> None:
    parent = source_node("parent", (0,), (10,))
    anchor = source_node("anchor", (2,), (200,))
    combined = Node(
        CombineThreeValues,
        bindings={
            "left": parent.value,
            "right": parent.value,
            "anchor": anchor.value,
        },
        tolerances={"left": timedelta(minutes=1)},
        name="combined",
    )

    result = Graph(combined).execute()

    assert result["timestamp"].to_list() == [dt(0), dt(2)]
    assert result["left"].to_list() == [10, None]
    assert result["right"].to_list() == [10, 10]
    assert result["anchor"].to_list() == [None, 200]


def test_binding_validation_still_catches_missing_extra_unknown_and_wrong_typed_edges() -> None:
    left = source_node("left", (0,), (1,))
    right = source_node("right", (0,), (2,))

    with pytest.raises(ValueError, match="Missing expected inputs"):
        Graph(Node(CombineValues, bindings={"left": left.value}))

    with pytest.raises(ValueError, match="Unexpected bound inputs"):
        Graph(
            Node(
                CombineValues,
                bindings={
                    "left": left.value,
                    "right": right.value,
                    "extra": right.value,
                },
            )
        )

    with pytest.raises(ValueError, match="does not expose output 'missing'"):
        Graph(
            Node(
                CombineValues,
                bindings={
                    "left": (left, "missing"),
                    "right": right.value,
                },
            )
        )

    with pytest.raises(TypeError, match="Type mismatch"):
        Graph(
            Node(
                CombineValues,
                bindings={
                    "left": left.value,
                    "right": Node(FloatSource).value,
                },
            )
        )

    with pytest.raises(ValueError, match="Unexpected tolerances"):
        Node(
            CombineValues,
            bindings={
                "left": left.value,
                "right": right.value,
            },
            tolerances={"missing": "1m"},
        )
