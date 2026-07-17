from dataclasses import FrozenInstanceError, dataclass
import json

import polars as pl
import pytest

from iosislib.core.graph import (
    Graph,
    GraphValidationError,
    ValidationIssue,
    ValidationReport,
)
from iosislib.core.node import Node
from iosislib.core.tsfn import FrameSignature, NullPolicy, TSFN, TSFNConfig


VALUE_FRAME = FrameSignature(columns=(("value", pl.Int64),))
FLOAT_FRAME = FrameSignature(columns=(("value", pl.Float64),))
PAIR_FRAME = FrameSignature(
    columns=(("left", pl.Int64), ("right", pl.Int64))
)
APPLY_CALLS = 0


@dataclass(frozen=True)
class SourceConfig(TSFNConfig):
    value: int


class IntSource(TSFN):
    VERSION = "8.0.0"
    CONFIG_CLS = SourceConfig

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        return FrameSignature.empty(), VALUE_FRAME

    def apply(self) -> pl.LazyFrame:
        global APPLY_CALLS
        APPLY_CALLS += 1
        raise AssertionError("inspection must not execute sources")


class FloatSource(TSFN):
    VERSION = "8.0.1"

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        return FrameSignature.empty(), FLOAT_FRAME

    def apply(self) -> pl.LazyFrame:
        global APPLY_CALLS
        APPLY_CALLS += 1
        raise AssertionError("inspection must not execute sources")


class Passthrough(TSFN):
    VERSION = "8.1.0"

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        return VALUE_FRAME, VALUE_FRAME

    def apply(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        global APPLY_CALLS
        APPLY_CALLS += 1
        raise AssertionError("inspection must not lower transforms")


class Pair(TSFN):
    VERSION = "8.2.0"

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        return PAIR_FRAME, PAIR_FRAME

    def apply(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        global APPLY_CALLS
        APPLY_CALLS += 1
        raise AssertionError("inspection must not lower transforms")


def int_source(value: int, name: str) -> Node:
    return Node(IntSource, parameters={"value": value}, name=name)


def invalid_root(*, reverse_bindings: bool) -> Node:
    integer = int_source(1, "integer")
    floating = Node(FloatSource, name="floating")
    binding_items = [
        ("left", (integer, "missing")),
        ("right", floating.value),
        ("extra", integer.value),
    ]
    if reverse_bindings:
        binding_items.reverse()
    return Node(
        Pair,
        bindings=dict(binding_items),
        null_handlers={"left": NullPolicy.FILL},
        name="invalid_pair",
    )


def inspectable_graph(*, reverse_bindings: bool) -> Graph:
    left = int_source(1, "left_source")
    right = int_source(2, "right_source")
    middle = Node(
        Passthrough,
        bindings={"value": left.value},
        name="middle",
        materialize=True,
    )
    binding_items = [("left", middle.value), ("right", right.value)]
    if reverse_bindings:
        binding_items.reverse()
    root = Node(
        Pair,
        bindings=dict(binding_items),
        tolerances={"left": "5m"},
        null_handlers={"left": NullPolicy.FILL},
        null_fill_values={"left": 0},
        name="pair",
    )
    return Graph(root)


def test_validation_aggregates_independent_issues_deterministically() -> None:
    first = Graph.validate(invalid_root(reverse_bindings=False))
    second = Graph.validate(invalid_root(reverse_bindings=True))

    assert isinstance(first, ValidationReport)
    assert first.is_valid is False
    assert first == second
    assert [issue.code for issue in first.issues] == [
        "INPUT_TYPE_MISMATCH",
        "MISSING_NULL_FILL_VALUE",
        "PARENT_OUTPUT_MISSING",
        "UNEXPECTED_BINDING",
    ]
    assert all(isinstance(issue, ValidationIssue) for issue in first.issues)
    assert all(issue.node_name == "invalid_pair" for issue in first.issues)
    assert all(issue.tsfn_class.endswith(".Pair") for issue in first.issues)
    assert all(issue.tsfn_version == "8.2.0" for issue in first.issues)
    assert not any(
        issue.code == "INPUT_TYPE_MISMATCH" and issue.input_name == "left"
        for issue in first.issues
    )
    json.dumps(first.to_dict(), sort_keys=True, allow_nan=False)

    with pytest.raises(FrozenInstanceError):
        first.issues = ()  # type: ignore[misc]


def test_invalid_construction_raises_one_error_with_the_complete_report() -> None:
    root = invalid_root(reverse_bindings=False)
    report = Graph.validate(root)

    with pytest.raises(GraphValidationError) as exc_info:
        Graph(root)

    assert exc_info.value.report == report
    assert "Graph validation failed with 4 issue(s)" in str(exc_info.value)
    assert "Type mismatch" in str(exc_info.value)
    assert "does not expose output 'missing'" in str(exc_info.value)


def test_cycles_are_reported_clearly_without_unsafe_downstream_checks() -> None:
    source = int_source(1, "source")
    root = Node(Passthrough, bindings={"value": source.value}, name="root")
    object.__setattr__(source, "bindings", {"value": root.value})
    object.__setattr__(source, "inputs", (root,))

    report = Graph.validate(root)

    assert [issue.code for issue in report.issues] == ["CYCLE"]
    assert "root -> source -> root" in report.issues[0].message
    with pytest.raises(GraphValidationError, match="Cycle detected"):
        Graph(root)


def test_describe_is_canonical_json_data_and_never_executes_tsfns() -> None:
    global APPLY_CALLS
    APPLY_CALLS = 0

    first_graph = inspectable_graph(reverse_bindings=False)
    second_graph = inspectable_graph(reverse_bindings=True)
    assert Graph.validate(first_graph.root_node).is_valid

    first = first_graph.describe()
    second = second_graph.describe()
    serialized = json.dumps(first, sort_keys=True, allow_nan=False)

    assert first == second
    assert APPLY_CALLS == 0
    assert "0x" not in serialized
    assert first["id"] == first_graph.ID
    assert first["root_id"] == first_graph.root_node.ID

    node_positions = {
        description["id"]: index
        for index, description in enumerate(first["nodes"])
    }
    for index, description in enumerate(first["nodes"]):
        for binding in description["bindings"].values():
            assert node_positions[binding["parent_id"]] < index

    by_name = {description["name"]: description for description in first["nodes"]}
    root = by_name["pair"]
    assert root["function"]["qualname"] == "Pair"
    assert root["function"]["version"] == "8.2.0"
    assert root["input_signature"]["columns"] == [
        ("left", "Int64"),
        ("right", "Int64"),
    ]
    assert list(root["bindings"]) == ["left", "right"]
    assert root["tolerances"] == {
        "left": {"type": "str", "value": "5m"},
        "right": None,
    }
    assert root["null_handlers"] == {
        "left": {"kind": "policy", "value": "fill"},
        "right": {"kind": "policy", "value": "propagate"},
    }
    assert root["null_fill_values"] == {"left": 0}
    assert root["materialization"]["reasons"] == ["root_result"]
    assert by_name["middle"]["materialization"]["reasons"] == [
        "node_requested"
    ]
    assert by_name["left_source"]["materialization"]["boundary"] is False
