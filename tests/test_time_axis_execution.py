from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import polars as pl
import pytest

from iosislib.core.graph import Executor, Graph, LocalExecutor
from iosislib.core.node import Node
from iosislib.core.tsfn import FrameSignature, TSFN, TSFNConfig

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


class RequiredMaterialization(TSFN):
    VERSION = "1.0.0"
    REQUIRES_MATERIALIZATION = True

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


class RecordingExecutor(LocalExecutor):
    def __init__(self) -> None:
        super().__init__()
        self.aligned_nodes: list[str] = []
        self.lowered_nodes: list[str] = []
        self.materialized_nodes: list[str] = []

    def lower_node(
        self,
        node: Node,
        input_lf: pl.LazyFrame | None,
    ) -> pl.LazyFrame:
        self.lowered_nodes.append(node.ID)
        return super().lower_node(node, input_lf)

    def align_inputs(
        self,
        node: Node,
        results: dict[str, pl.LazyFrame],
    ) -> pl.LazyFrame:
        self.aligned_nodes.append(node.ID)
        return super().align_inputs(node, results)

    def materialize(self, node: Node, lf: pl.LazyFrame) -> pl.DataFrame:
        self.materialized_nodes.append(node.ID)
        return super().materialize(node, lf)


def test_frame_signature_empty_accepts_only_no_time_and_no_columns() -> None:
    assert FrameSignature.empty().is_empty()

    with pytest.raises(ValueError, match="cannot declare value columns"):
        FrameSignature(time=None, columns=(("value", pl.Int64),))


def test_inputless_tsfn_apply_is_called_without_frame() -> None:
    source = source_node("source", (0, 1), (10, 20))

    result = Graph(source).execute()

    assert result["timestamp"].to_list() == [dt(0), dt(1)]
    assert result["value"].to_list() == [10, 20]


def test_graph_verify_checks_contracts_without_lowering_or_materializing() -> None:
    graph = Graph(Node(MissingTimeSource))

    assert graph.verify() is None
    assert not hasattr(graph, "_compiled_root_lf")

    with pytest.raises(RuntimeError, match="Missing required output time column"):
        graph.execute()


def test_execute_and_describe_consume_the_stored_validated_declaration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = source_node("source", (0, 1), (10, 20))
    graph = Graph(source)
    original_id = graph.ID
    original_nodes = graph.node_list

    def unexpected_revalidation(*args: object, **kwargs: object) -> object:
        raise AssertionError("validated graph state must be reused")

    monkeypatch.setattr(Graph, "verify", unexpected_revalidation)
    monkeypatch.setattr(
        Graph,
        "_validated_declaration",
        classmethod(unexpected_revalidation),
    )
    monkeypatch.setattr(
        Graph,
        "_dependency_order",
        classmethod(unexpected_revalidation),
    )

    description = graph.describe()
    result = graph.execute()

    assert graph.ID == original_id
    assert graph.node_list == original_nodes
    assert description["id"] == original_id
    assert [node["id"] for node in description["nodes"]] == [source.ID]
    assert result["value"].to_list() == [10, 20]


def test_graph_execute_uses_the_supplied_runtime_executor() -> None:
    source = source_node("source", (0, 1), (10, 20))
    transform = Node(NeedsInput, bindings={"value": source.value}, name="transform")
    executor = RecordingExecutor()

    graph = Graph(transform)
    result = graph.execute(executor=executor)

    assert result["value"].to_list() == [10, 20]
    assert not hasattr(graph, "executor")
    assert executor.lowered_nodes == [source.ID, transform.ID]
    assert executor.aligned_nodes == [transform.ID]
    assert executor.materialized_nodes == [transform.ID]


def test_each_execute_can_select_an_independent_runtime_executor() -> None:
    source = source_node("source", (0,), (10,))
    first = RecordingExecutor()
    second = RecordingExecutor()
    graph = Graph(source)

    first_result = graph.execute(executor=first)
    second_result = graph.execute(executor=second)

    assert first_result.equals(second_result)
    assert first.materialized_nodes == [source.ID]
    assert second.materialized_nodes == [source.ID]


def test_executor_is_abstract_and_graph_rejects_non_executors() -> None:
    with pytest.raises(TypeError, match="abstract"):
        Executor()  # type: ignore[abstract]
    assert not hasattr(LocalExecutor(), "lower")

    source = source_node("source", (0,), (10,))
    with pytest.raises(TypeError, match="unexpected keyword argument 'executor'"):
        Graph(source, executor=LocalExecutor())  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="must be an Executor"):
        Graph(source).execute(executor=object())  # type: ignore[arg-type]


def test_graph_materialization_points_split_lazy_execution_regions() -> None:
    source = source_node("source", (0, 1), (10, 20))
    middle = Node(
        NeedsInput,
        bindings={"value": source.value},
        name="middle",
        materialize=True,
    )
    root = Node(NeedsInput, bindings={"value": middle.value}, name="root")
    executor = RecordingExecutor()
    graph = Graph(root)

    result = graph.execute(executor=executor)

    assert result["value"].to_list() == [10, 20]
    assert graph.materialized_node_ids == frozenset({middle.ID})
    assert executor.materialized_nodes == [middle.ID, root.ID]
    assert graph.ID == Graph(root).ID


def test_explicit_root_materialization_is_not_performed_twice() -> None:
    root = Node(
        SeriesSource,
        parameters={"minutes": (0,), "values": (10,)},
        name="source",
        materialize=True,
    )
    executor = RecordingExecutor()

    Graph(root).execute(executor=executor)

    assert executor.materialized_nodes == [root.ID]


def test_node_materialization_does_not_affect_identity() -> None:
    source = source_node("source", (0,), (10,))
    lazy_middle = Node(
        NeedsInput,
        bindings={"value": source.value},
        materialize=False,
    )
    material_middle = Node(
        NeedsInput,
        bindings={"value": source.value},
        materialize=True,
    )
    lazy_root = Node(NeedsInput, bindings={"value": lazy_middle.value})
    material_root = Node(NeedsInput, bindings={"value": material_middle.value})

    lazy_graph = Graph(lazy_root)
    material_graph = Graph(material_root)

    assert lazy_middle.ID == material_middle.ID
    assert lazy_root.ID == material_root.ID
    assert lazy_graph.ID == material_graph.ID
    # Materialization is execution state only; the root is not forced.
    assert lazy_graph.materialized_node_ids == frozenset()
    assert material_graph.materialized_node_ids == frozenset({material_middle.ID})


def test_same_node_identity_across_materialization_flags() -> None:
    lazy = Node(
        SeriesSource,
        parameters={"minutes": (0,), "values": (10,)},
        materialize=False,
    )
    materialized = Node(
        SeriesSource,
        parameters={"minutes": (0,), "values": (10,)},
        materialize=True,
    )
    root = Node(
        NeedsInput,
        bindings={"value": materialized.value},
        name="root",
    )
    executor = RecordingExecutor()
    graph = Graph(root)

    result = graph.execute(executor=executor)

    assert lazy.ID == materialized.ID
    assert graph.node_list == (materialized, root)
    assert executor.lowered_nodes == [materialized.ID, root.ID]
    assert executor.materialized_nodes == [materialized.ID, root.ID]
    assert result["value"].to_list() == [10]


def test_tsfn_materialization_requirement_defaults_node_intent_and_is_verified() -> None:
    source = source_node("source", (0,), (10,))
    default_node = Node(
        RequiredMaterialization,
        bindings={"value": source.value},
    )
    disabled_node = Node(
        RequiredMaterialization,
        bindings={"value": source.value},
        materialize=False,
    )

    assert default_node.materialize is True
    assert default_node.function.requires_materialization is True
    assert Graph(default_node).materialized_node_ids == frozenset({default_node.ID})
    with pytest.raises(AttributeError):
        default_node.function.requires_materialization = False  # type: ignore[misc]
    with pytest.raises(ValueError, match="requires materialization"):
        Graph(disabled_node)


def test_materialization_metadata_must_be_boolean() -> None:
    with pytest.raises(TypeError, match="Node materialize must be a boolean"):
        Node(SeriesSource, materialize=1)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="REQUIRES_MATERIALIZATION must be a boolean"):
        class InvalidMaterializationMetadata(TSFN):
            VERSION = "1.0.0"
            REQUIRES_MATERIALIZATION = "yes"  # type: ignore[assignment]

            def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
                return FrameSignature.empty(), VALUE_FRAME

            def apply(self) -> pl.LazyFrame:
                return pl.DataFrame(
                    {"timestamp": [dt(0)], "value": [1]}
                ).lazy()

    with pytest.raises(TypeError, match="DEFAULT_NULL_POLICY must be a NullPolicy"):
        class InvalidDefaultNullPolicy(TSFN):
            VERSION = "1.0.0"
            DEFAULT_NULL_POLICY = "error"  # type: ignore[assignment]

            def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
                return FrameSignature.empty(), VALUE_FRAME

            def apply(self) -> pl.LazyFrame:
                return pl.DataFrame(
                    {"timestamp": [dt(0)], "value": [1]}
                ).lazy()


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

    with pytest.raises(ValueError, match="Type mismatch"):
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
