from __future__ import annotations

from datetime import datetime
from types import MappingProxyType

import polars as pl
import pytest

from iosislib.core.graph import Graph, GraphValidationError
from iosislib.core.node import Node
from iosislib.core.tsfn import FrameSignature, NullHandler, NullPolicy, TSFN


class IntegerSource(TSFN):
    VERSION = "1.0.0"

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        return FrameSignature.empty(), FrameSignature(columns=(("value", pl.Int64),))

    def apply(self) -> pl.LazyFrame:
        return pl.DataFrame(
            {"timestamp": [datetime(2026, 1, 1)], "value": [1]}
        ).lazy()


class Passthrough(TSFN):
    VERSION = "1.0.0"

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        signature = FrameSignature(columns=(("value", pl.Int64),))
        return signature, signature

    def apply(self, lf: pl.LazyFrame | None = None) -> pl.LazyFrame:
        assert lf is not None
        return lf.select("timestamp", "value")


class ErrorByDefault(Passthrough):
    VERSION = "1.0.0"
    DEFAULT_NULL_POLICY = NullPolicy.ERROR


class Pair(TSFN):
    VERSION = "1.0.0"

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        return (
            FrameSignature(columns=(("left", pl.Int64), ("right", pl.Int64))),
            FrameSignature(columns=(("pair", pl.Int64, (2,)),)),
        )

    def apply(self, lf: pl.LazyFrame | None = None) -> pl.LazyFrame:
        assert lf is not None
        return lf.select(
            "timestamp",
            pl.concat_arr("left", "right").alias("pair"),
        )


def fill_nulls(series: pl.Series) -> pl.Series:
    return series.fill_null(0)


def test_omitted_and_explicit_default_handlers_have_one_effective_identity() -> None:
    source = Node(IntegerSource)
    omitted = Node(Passthrough, bindings={"value": source.value})
    explicit_policy = Node(
        Passthrough,
        bindings={"value": source.value},
        null_policies={"value": NullPolicy.PROPAGATE},
    )
    explicit_handler = Node(
        Passthrough,
        bindings={"value": source.value},
        null_handlers={"value": NullHandler.from_policy("propagate")},
    )

    assert omitted.ID == explicit_policy.ID == explicit_handler.ID
    assert omitted.definition == explicit_policy.definition == explicit_handler.definition
    assert omitted.null_handlers["value"].policy is NullPolicy.PROPAGATE
    assert omitted.function.input_null_policy("value") is NullPolicy.PROPAGATE
    assert omitted.definition["null_handlers"] == {
        "value": {"kind": "policy", "value": "propagate"}
    }


def test_non_propagate_tsfn_default_is_canonical_effective_policy() -> None:
    source = Node(IntegerSource)
    omitted = Node(ErrorByDefault, bindings={"value": source.value})
    explicit = Node(
        ErrorByDefault,
        bindings={"value": source.value},
        null_policies={"value": NullPolicy.ERROR},
    )

    assert omitted.ID == explicit.ID
    assert omitted.null_handlers["value"].policy is NullPolicy.ERROR
    assert omitted.function.input_null_policy("value") is NullPolicy.ERROR
    assert omitted.definition["null_handlers"]["value"]["value"] == "error"


def test_only_relevant_fill_values_enter_identity() -> None:
    source = Node(IntegerSource)
    zero = Node(
        Passthrough,
        bindings={"value": source.value},
        null_policies={"value": NullPolicy.FILL},
        null_fill_values={"value": 0},
    )
    one = Node(
        Passthrough,
        bindings={"value": source.value},
        null_policies={"value": NullPolicy.FILL},
        null_fill_values={"value": 1},
    )

    assert zero.ID != one.ID
    assert zero.definition["null_fill_values"] == {"value": 0}
    assert one.definition["null_fill_values"] == {"value": 1}


@pytest.mark.parametrize(
    "handler",
    [NullPolicy.PROPAGATE, NullPolicy.ERROR, NullHandler.from_function(fill_nulls, version="1")],
)
def test_fill_value_is_rejected_for_non_fill_effective_handler(
    handler: NullPolicy | NullHandler,
) -> None:
    source = Node(IntegerSource)

    with pytest.raises(ValueError, match="only valid.*NullPolicy.FILL"):
        Node(
            Passthrough,
            bindings={"value": source.value},
            null_handlers={"value": handler},
            null_fill_values={"value": 0},
        )


def test_unknown_fill_metadata_reaches_graph_validation() -> None:
    source = Node(IntegerSource)
    invalid = Node(
        Passthrough,
        bindings={"value": source.value},
        null_fill_values={"missing": 0},
    )

    with pytest.raises(GraphValidationError) as error:
        Graph(invalid)

    assert [issue.code for issue in error.value.report.issues] == [
        "UNEXPECTED_NULL_FILL_VALUE"
    ]


def test_materialization_is_execution_state_not_identity() -> None:
    lazy = Node(IntegerSource, materialize=False)
    materialized = Node(IntegerSource, materialize=True)

    # Materialization is execution state only; it never contributes to node or
    # graph identity.
    assert lazy.ID == materialized.ID
    assert lazy.definition == materialized.definition
    assert Graph(lazy).ID == Graph(materialized).ID


def test_mixed_materialization_declarations_share_identity() -> None:
    lazy = Node(IntegerSource, materialize=False)
    materialized = Node(IntegerSource, materialize=True)
    root = Node(Pair, bindings={"left": lazy.value, "right": materialized.value})
    graph = Graph(root)

    # Same node identity regardless of the materialization flag.
    assert lazy.ID == materialized.ID
    assert len(graph.node_list) == 2
    assert graph.execute()["pair"].to_list() == [[1, 1]]


def test_custom_handler_versions_remain_authoritative_identity() -> None:
    source = Node(IntegerSource)
    one = Node(
        Passthrough,
        bindings={"value": source.value},
        null_handlers={"value": fill_nulls},
        null_handler_versions={"value": "1.0.0"},
    )
    repeated = Node(
        Passthrough,
        bindings={"value": source.value},
        null_handlers={"value": fill_nulls},
        null_handler_versions={"value": "1.0.0"},
    )
    changed = Node(
        Passthrough,
        bindings={"value": source.value},
        null_handlers={"value": fill_nulls},
        null_handler_versions={"value": "2.0.0"},
    )

    assert one.ID == repeated.ID
    assert one.ID != changed.ID


def test_effective_policy_identity_ignores_mapping_insertion_order() -> None:
    source = Node(IntegerSource)
    first = Node(
        Pair,
        bindings={"left": source.value, "right": source.value},
        null_policies={"left": NullPolicy.FILL, "right": NullPolicy.FILL},
        null_fill_values={"left": 0, "right": 1},
    )
    reversed_order = Node(
        Pair,
        bindings={"right": source.value, "left": source.value},
        null_policies={"right": NullPolicy.FILL, "left": NullPolicy.FILL},
        null_fill_values={"right": 1, "left": 0},
    )

    assert first.ID == reversed_order.ID
    assert first.definition == reversed_order.definition
    assert tuple(first.bindings) == ("left", "right")
    assert tuple(first.null_handlers) == ("left", "right")
    assert tuple(first.null_fill_values) == ("left", "right")


def test_effective_policy_mappings_and_definition_are_immutable() -> None:
    caller_handlers = {"value": NullPolicy.FILL}
    caller_fills = {"value": 0}
    node = Node(
        Passthrough,
        bindings={"value": Node(IntegerSource).value},
        null_policies=caller_handlers,
        null_fill_values=caller_fills,
    )
    original_id = node.ID

    caller_handlers["value"] = NullPolicy.ERROR
    caller_fills["value"] = 99

    assert node.ID == original_id
    assert node.null_handlers["value"].policy is NullPolicy.FILL
    assert node.null_fill_values["value"] == 0
    assert isinstance(node.null_handlers, MappingProxyType)
    assert isinstance(node.null_fill_values, MappingProxyType)
    with pytest.raises(TypeError):
        node.null_handlers["value"] = NullHandler.from_policy(NullPolicy.ERROR)
    with pytest.raises(TypeError):
        node.null_fill_values["value"] = 99
    with pytest.raises(TypeError):
        node.definition["materialize"] = False
