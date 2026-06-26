from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass, field
from datetime import datetime
import json

import polars as pl
import pytest

from src.classes import (
    FrameSignature,
    Graph,
    Node,
    TSFN,
    TSFNConfig,
    TimeAxis,
    _dtype_matches,
)


def dt(minute: int) -> datetime:
    return datetime(2026, 1, 1, 0, minute)


VALUE_FRAME = FrameSignature(columns=(("value", pl.Int64),))
FLOAT_FRAME = FrameSignature(columns=(("value", pl.Float64),))
PAIR_FRAME = FrameSignature(columns=(("left", pl.Int64), ("right", pl.Int64)))


@dataclass(frozen=True)
class SeriesConfig(TSFNConfig):
    minutes: tuple[int, ...] = (0,)
    values: tuple[int, ...] = (1,)
    output: str = "value"
    time_column: str = "timestamp"


class SeriesSource(TSFN):
    CONFIG_CLS = SeriesConfig

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        output = FrameSignature(
            time=TimeAxis(
                column=self.parameters.time_column,
            ),
            columns=((self.parameters.output, pl.Int64),),
        )
        return FrameSignature.empty(), output

    def apply(self) -> pl.LazyFrame:
        return pl.DataFrame(
            {
                self.parameters.time_column: [dt(minute) for minute in self.parameters.minutes],
                self.parameters.output: list(self.parameters.values),
            }
        ).lazy()


class FloatSource(TSFN):
    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        return FrameSignature.empty(), FLOAT_FRAME

    def apply(self) -> pl.LazyFrame:
        return pl.DataFrame({"timestamp": [dt(0)], "value": [1.5]}).lazy()


class DateSignatureSource(TSFN):
    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        frame = FrameSignature(
            time=TimeAxis(dtype=pl.Date),
            columns=(("value", pl.Int64),),
        )
        return FrameSignature.empty(), frame

    def apply(self) -> pl.LazyFrame:
        return pl.DataFrame({"timestamp": [dt(0)], "value": [1]}).lazy()


class UtcSignatureSource(TSFN):
    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        frame = FrameSignature(
            time=TimeAxis(dtype=pl.Datetime, timezone="UTC"),
            columns=(("value", pl.Int64),),
        )
        return FrameSignature.empty(), frame

    def apply(self) -> pl.LazyFrame:
        return pl.DataFrame({"timestamp": [dt(0)], "value": [1]}).lazy()


class Identity(TSFN):
    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        return VALUE_FRAME, VALUE_FRAME

    def apply(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.select("timestamp", "value")


class Pair(TSFN):
    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        return PAIR_FRAME, PAIR_FRAME

    def apply(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.select("timestamp", "left", "right")


class TimeOnlyInput(TSFN):
    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        return FrameSignature(), VALUE_FRAME

    def apply(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.select(pl.col("timestamp"), pl.lit(1).alias("value"))


class EmptyInputButBound(TSFN):
    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        return FrameSignature.empty(), VALUE_FRAME

    def apply(self) -> pl.LazyFrame:
        return pl.DataFrame({"timestamp": [dt(0)], "value": [1]}).lazy()


class MissingOutputTime(TSFN):
    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        return FrameSignature.empty(), VALUE_FRAME

    def apply(self) -> pl.LazyFrame:
        return pl.DataFrame({"value": [1]}).lazy()


class WrongOutputValueType(TSFN):
    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        return FrameSignature.empty(), VALUE_FRAME

    def apply(self) -> pl.LazyFrame:
        return pl.DataFrame({"timestamp": [dt(0)], "value": ["not-int"]}).lazy()


@dataclass(frozen=True)
class RequiredConfig(TSFNConfig):
    required: int
    optional: int = 2
    labels: tuple[str, ...] = field(default_factory=lambda: ("x",))


class ConfiguredSource(TSFN):
    CONFIG_CLS = RequiredConfig

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        return FrameSignature.empty(), VALUE_FRAME

    def apply(self) -> pl.LazyFrame:
        return pl.DataFrame(
            {
                "timestamp": [dt(0)],
                "value": [self.parameters.required + self.parameters.optional],
            }
        ).lazy()


@dataclass(frozen=True)
class DTypeConfig(TSFNConfig):
    dtype: pl.DataType


@dataclass(frozen=True)
class BadConfig(TSFNConfig):
    payload: object


def source(
    name: str = "source",
    minutes: tuple[int, ...] = (0,),
    values: tuple[int, ...] = (1,),
    output: str = "value",
    time_column: str = "timestamp",
) -> Node:
    return Node(
        SeriesSource,
        parameters={
            "minutes": minutes,
            "values": values,
            "output": output,
            "time_column": time_column,
        },
        name=name,
    )


def test_frame_signature_invariants_are_strict() -> None:
    default_signature = FrameSignature()

    assert default_signature.time == TimeAxis()
    assert default_signature.columns == ()
    assert FrameSignature.empty().is_empty()

    with pytest.raises(TypeError, match="columns must be a tuple"):
        FrameSignature(columns=[("value", pl.Int64)])  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="cannot declare value columns"):
        FrameSignature(time=None, columns=(("value", pl.Int64),))

    with pytest.raises(ValueError, match="must not be listed as a value column"):
        FrameSignature(columns=(("timestamp", pl.Datetime),))

    with pytest.raises(FrozenInstanceError):
        default_signature.columns = (("mutated", pl.Int64),)  # type: ignore[misc]


def test_dtype_matching_matrix_covers_classes_instances_and_parameterized_types() -> None:
    assert _dtype_matches(pl.Int64, pl.Int64)
    assert _dtype_matches(pl.Int64(), pl.Int64)
    assert _dtype_matches(pl.Int64, pl.Int64())
    assert _dtype_matches(pl.Datetime(time_unit="us"), pl.Datetime)
    assert _dtype_matches(pl.Datetime(time_unit="us"), pl.Datetime(time_unit="us"))
    assert _dtype_matches(pl.List(pl.Int64), pl.List(pl.Int64))
    assert _dtype_matches(pl.List(pl.List(pl.Int64)), pl.List(pl.List(pl.Int64)))

    assert not _dtype_matches(pl.Int64, pl.Float64)
    assert not _dtype_matches(pl.Datetime(time_unit="us"), pl.Datetime(time_unit="ns"))
    assert not _dtype_matches(pl.Datetime(time_unit="ns"), pl.Datetime)
    assert not _dtype_matches(pl.Datetime(time_unit="us", time_zone="UTC"), pl.Datetime)
    assert not _dtype_matches(pl.Date, pl.Datetime)
    assert not _dtype_matches(pl.List(pl.Int64), pl.List(pl.Float64))
    assert not _dtype_matches(pl.List(pl.List(pl.Int64)), pl.List(pl.List(pl.Float64)))
    assert not _dtype_matches(pl.List(pl.Int64), pl.List)
    assert not _dtype_matches(pl.List, pl.List(pl.Int64))


def test_time_axis_timezone_is_checked_separately_from_datetime_family() -> None:
    frame = FrameSignature(
        time=TimeAxis(dtype=pl.Datetime, timezone="UTC"),
        columns=(("value", pl.Int64),),
    )
    fn = Identity({})

    with pytest.raises(TypeError, match="timezone mismatch"):
        fn._validate_schema(
            pl.DataFrame({"timestamp": [dt(0)], "value": [1]}).lazy(),
            frame,
            "input",
        )


def test_schema_validation_rejects_list_inner_type_mismatches() -> None:
    frame = FrameSignature(columns=(("items", pl.List(pl.Int64)),))
    fn = Identity({})

    fn._validate_schema(
        pl.DataFrame({"timestamp": [dt(0)], "items": [[1, 2]]}).lazy(),
        frame,
        "input",
    )

    with pytest.raises(TypeError, match="Column 'items' type mismatch"):
        fn._validate_schema(
            pl.DataFrame({"timestamp": [dt(0)], "items": [[1.5, 2.5]]}).lazy(),
            frame,
            "input",
        )


def test_config_binding_validates_extra_missing_defaults_and_json_representation() -> None:
    with pytest.raises(ValueError, match="Unexpected parameters"):
        ConfiguredSource({"required": 1, "surprise": 99})

    with pytest.raises(ValueError, match="Missing required parameters"):
        ConfiguredSource({})

    fn = ConfiguredSource({"required": 3})

    assert fn.parameters == RequiredConfig(required=3)
    assert Graph(Node(ConfiguredSource, parameters={"required": 3})).execute()["value"].to_list() == [5]
    assert json.loads(str(fn.parameters)) == {
        "labels": ["x"],
        "optional": 2,
        "required": 3,
    }


def test_config_string_serializes_polars_dtypes_and_rejects_opaque_objects() -> None:
    assert json.loads(str(DTypeConfig(dtype=pl.Int64()))) == {"dtype": "Int64"}

    with pytest.raises(TypeError, match="not serializable"):
        str(BadConfig(payload=object()))


def test_tsfn_call_contract_for_inputless_and_inputful_functions() -> None:
    inputless = SeriesSource({"minutes": (0,), "values": (10,)})
    inputful = Identity({})

    with pytest.raises(ValueError, match="Expected no input frame"):
        inputless(pl.DataFrame({"timestamp": [dt(0)], "value": [1]}).lazy())

    with pytest.raises(ValueError, match="Missing required input frame"):
        inputful()

    with pytest.raises(ValueError, match="Missing required input column"):
        inputful(pl.DataFrame({"timestamp": [dt(0)]}).lazy())

    with pytest.raises(TypeError, match="Column 'value' type mismatch"):
        inputful(pl.DataFrame({"timestamp": [dt(0)], "value": ["bad"]}).lazy())


def test_output_schema_validation_catches_missing_time_and_wrong_value_type() -> None:
    with pytest.raises(ValueError, match="Missing required output time column"):
        Graph(Node(MissingOutputTime)).execute()

    with pytest.raises(TypeError, match="Column 'value' type mismatch"):
        Graph(Node(WrongOutputValueType)).execute()


def test_node_outputs_exclude_time_and_getattr_only_exposes_value_outputs() -> None:
    node = source(minutes=(0,), values=(1,))

    assert node.outputs == {"value": pl.Int64}
    assert node.value == (node, "value")

    with pytest.raises(AttributeError, match="does not expose output"):
        _ = node.timestamp


def test_node_ids_are_deterministic_and_change_with_material_definition_parts() -> None:
    first = source(name="same", minutes=(0,), values=(1,))
    second = source(name="same", minutes=(0,), values=(1,))
    different_name = source(name="different", minutes=(0,), values=(1,))
    different_params = source(name="same", minutes=(0,), values=(2,))
    transform_a = Node(Identity, bindings={"value": first.value}, name="identity")
    transform_b = Node(Identity, bindings={"value": second.value}, name="identity")

    assert first.ID == second.ID
    assert transform_a.ID == transform_b.ID
    assert first.ID != different_name.ID
    assert first.ID != different_params.ID
    assert first.ID != transform_a.ID


def test_graph_ids_are_deterministic_and_node_list_is_topological() -> None:
    left = source("left", (0,), (1,))
    right = source("right", (0,), (2,))
    root = Node(Pair, bindings={"left": left.value, "right": right.value}, name="pair")
    graph = Graph(root)

    left2 = source("left", (0,), (1,))
    right2 = source("right", (0,), (2,))
    root2 = Node(Pair, bindings={"left": left2.value, "right": right2.value}, name="pair")

    assert [node.name for node in graph.node_list] == ["left", "right", "pair"]
    assert graph.ID == Graph(root2).ID


def test_same_parent_can_feed_multiple_inputs_without_duplicate_execution_nodes() -> None:
    parent = source("parent", (0, 2), (10, 20))
    root = Node(Pair, bindings={"left": parent.value, "right": parent.value})
    graph = Graph(root)

    result = graph.execute()

    assert graph.node_list == [parent, root]
    assert result["timestamp"].to_list() == [dt(0), dt(2)]
    assert result["left"].to_list() == [10, 20]
    assert result["right"].to_list() == [10, 20]


def test_graph_validation_rejects_time_only_roots_and_empty_signature_bound_nodes() -> None:
    with pytest.raises(ValueError, match="empty input signature"):
        Graph(Node(TimeOnlyInput))

    upstream = source()
    with pytest.raises(ValueError, match="Unexpected bound inputs"):
        Graph(Node(EmptyInputButBound, bindings={"value": upstream.value}))


def test_graph_validation_rejects_unknown_outputs_wrong_types_and_orphaned_mutated_edges() -> None:
    upstream = source("upstream", (0,), (1,))
    other = source("other", (0,), (2,))

    with pytest.raises(ValueError, match="does not expose output 'missing'"):
        Graph(Node(Identity, bindings={"value": (upstream, "missing")}))

    with pytest.raises(TypeError, match="Type mismatch"):
        Graph(Node(Identity, bindings={"value": Node(FloatSource).value}))

    mutated = Node(Identity, bindings={"value": upstream.value})
    mutated.bindings = {"value": other.value}

    with pytest.raises(ValueError, match="binds to parent"):
        Graph(mutated)


def test_graph_validation_rejects_time_axis_column_dtype_and_timezone_mismatches() -> None:
    wrong_time_name = source("event_source", (0,), (1,), time_column="event_time")
    wrong_dtype = Node(DateSignatureSource)
    wrong_timezone = Node(UtcSignatureSource)

    with pytest.raises(ValueError, match="Time axis mismatch"):
        Graph(Node(Identity, bindings={"value": wrong_time_name.value}))

    with pytest.raises(TypeError, match="Time axis dtype mismatch"):
        Graph(Node(Identity, bindings={"value": wrong_dtype.value}))

    with pytest.raises(TypeError, match="Time axis timezone mismatch"):
        Graph(Node(Identity, bindings={"value": wrong_timezone.value}))


def test_cycle_detection_runs_before_validation_when_node_inputs_form_a_cycle() -> None:
    start = source("start", (0,), (1,))
    root = Node(Identity, bindings={"value": start.value}, name="root")

    start.bindings = {"value": root.value}
    start.inputs = (root,)

    with pytest.raises(ValueError, match="Cycle detected"):
        Graph(root)


def test_union_asof_alignment_sorts_unsorted_parents_and_never_looks_forward() -> None:
    left = source("left", (3, 0), (30, 10))
    right = source("right", (2,), (200,))
    root = Node(Pair, bindings={"left": left.value, "right": right.value})

    result = Graph(root).execute()

    assert result["timestamp"].to_list() == [dt(0), dt(2), dt(3)]
    assert result["left"].to_list() == [10, 10, 30]
    assert result["right"].to_list() == [None, 200, 200]


def test_asof_alignment_handles_independent_output_names_from_different_parents() -> None:
    alpha = source("alpha", (0, 2), (1, 3), output="alpha_value")
    beta = source("beta", (1,), (10,), output="beta_value")
    root = Node(Pair, bindings={"left": alpha.alpha_value, "right": beta.beta_value})

    result = Graph(root).execute()

    assert result["timestamp"].to_list() == [dt(0), dt(1), dt(2)]
    assert result["left"].to_list() == [1, 1, 3]
    assert result["right"].to_list() == [None, 10, 10]
