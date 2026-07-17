from __future__ import annotations

from collections.abc import Mapping
from dataclasses import FrozenInstanceError, dataclass
from datetime import datetime
from typing import Any

import polars as pl
import pytest

from iosislib.core.graph import Graph
from iosislib.core.node import Node
from iosislib.core.tsfn import FrameSignature, TSFN, TSFNConfig
from iosislib.core.utils import _canonical_identity_json


@dataclass(frozen=True)
class IdentitySourceConfig(TSFNConfig):
    token: Any
    payload: Any = ()


class IdentitySource(TSFN):
    VERSION = "1.0.0"
    CONFIG_CLS = IdentitySourceConfig

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        return (
            FrameSignature.empty(),
            FrameSignature(columns=(("value", pl.Int64),)),
        )

    def apply(self) -> pl.LazyFrame:
        value = 1 if self.parameters.token is pl.Float64 else 2
        payload = self.parameters.payload
        if isinstance(payload, Mapping):
            value += len(payload)
        return pl.DataFrame(
            {"timestamp": [datetime(2026, 1, 1)], "value": [value]}
        ).lazy()


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


class Passthrough(TSFN):
    VERSION = "1.0.0"

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        signature = FrameSignature(columns=(("value", pl.Int64),))
        return signature, signature

    def apply(self, lf: pl.LazyFrame | None = None) -> pl.LazyFrame:
        assert lf is not None
        return lf.select("timestamp", "value")


class StatefulSource(IdentitySource):
    VERSION = "1.0.0"

    def __init__(self, parameters: dict[str, Any]) -> None:
        super().__init__(parameters)
        self.multiplier = self.parameters.payload

    def apply(self) -> pl.LazyFrame:
        return pl.DataFrame(
            {
                "timestamp": [datetime(2026, 1, 1)],
                "value": [self.multiplier],
            }
        ).lazy()


def fill_nulls(series: pl.Series) -> pl.Series:
    return series.fill_null(0)


def test_canonical_identity_distinguishes_dtype_categories_and_graph_execution() -> None:
    dtype_source = Node(IdentitySource, parameters={"token": pl.Float64})
    string_source = Node(IdentitySource, parameters={"token": "Float64"})
    root = Node(
        Pair,
        bindings={"left": dtype_source.value, "right": string_source.value},
    )

    assert _canonical_identity_json(pl.Float64) != _canonical_identity_json("Float64")
    assert _canonical_identity_json(pl.Float64) != _canonical_identity_json(
        pl.Float64()
    )
    assert dtype_source.ID != string_source.ID
    assert len(Graph(root).node_list) == 3
    assert Graph(root).execute()["pair"].to_list() == [[1, 2]]


def test_caller_collections_are_copied_and_deeply_immutable() -> None:
    caller_list = [1, 2]
    caller_set = {"alpha", "beta"}
    caller_dict = {
        "items": caller_list,
        "nested": {"groups": caller_set},
    }
    node = Node(
        IdentitySource,
        parameters={"token": "token", "payload": caller_dict},
    )
    original_id = node.ID
    original_definition = node.definition

    caller_list.append(3)
    caller_set.add("gamma")
    caller_dict["new"] = [4]
    caller_dict["nested"]["groups"] = set()

    payload = node.parameters.payload
    assert tuple(payload["items"]) == (1, 2)
    assert payload["nested"]["groups"] == frozenset({"alpha", "beta"})
    assert "new" not in payload
    assert node.ID == original_id
    assert node.definition == original_definition
    assert Graph(node).execute()["value"].to_list() == [4]

    with pytest.raises(TypeError):
        payload["new"] = ()
    with pytest.raises(AttributeError):
        payload["items"].append(3)
    with pytest.raises(AttributeError):
        payload["nested"]["groups"].add("gamma")


def test_list_and_tuple_syntax_normalize_to_the_same_definition() -> None:
    from_list = Node(
        IdentitySource,
        parameters={"token": "token", "payload": {"items": [1, 2]}},
    )
    from_tuple = Node(
        IdentitySource,
        parameters={"payload": {"items": (1, 2)}, "token": "token"},
    )

    assert from_list.ID == from_tuple.ID
    assert from_list.definition == from_tuple.definition
    assert from_list.parameters == from_tuple.parameters


def test_node_freezes_resolved_tsfn_identity_state() -> None:
    node = Node(IdentitySource, parameters={"token": "token"})

    assert node.function.__class__ is IdentitySource

    with pytest.raises(AttributeError, match="immutable"):
        node.function.signature = (FrameSignature.empty(), FrameSignature.empty())
    with pytest.raises(AttributeError, match="immutable"):
        node.function.parameters = IdentitySourceConfig(token=pl.Float64)
    with pytest.raises(AttributeError, match="immutable"):
        node.function.version = "2.0.0"
    with pytest.raises(FrozenInstanceError):
        node.parameters.token = pl.Float64
    with pytest.raises(TypeError):
        node.definition["parameters"] = "changed"


def test_subclass_definition_state_is_hashed_and_frozen() -> None:
    one = Node(StatefulSource, parameters={"token": "token", "payload": 1})
    two = Node(StatefulSource, parameters={"token": "token", "payload": 2})

    assert one.ID != two.ID
    assert one.definition["function"]["state"]["multiplier"] == 1
    with pytest.raises(AttributeError, match="immutable"):
        one.function.multiplier = 3


def test_custom_null_handler_versions_participate_in_identity() -> None:
    source = Node(IdentitySource, parameters={"token": "token"})
    version_one = Node(
        Passthrough,
        bindings={"value": source.value},
        null_handlers={"value": fill_nulls},
        null_handler_versions={"value": "1.0.0"},
    )
    version_two = Node(
        Passthrough,
        bindings={"value": source.value},
        null_handlers={"value": fill_nulls},
        null_handler_versions={"value": "2.0.0"},
    )
    repeated = Node(
        Passthrough,
        bindings={"value": source.value},
        null_handlers={"value": fill_nulls},
        null_handler_versions={"value": "1.0.0"},
    )

    assert version_one.ID != version_two.ID
    assert version_one.ID == repeated.ID
    assert version_one.definition == repeated.definition


def test_custom_null_handler_requires_an_explicit_behavior_version() -> None:
    source = Node(IdentitySource, parameters={"token": "token"})

    with pytest.raises(ValueError, match="must declare a behavior version"):
        Node(
            Passthrough,
            bindings={"value": source.value},
            null_handlers={"value": fill_nulls},
        )


def test_semantically_equal_parents_still_deduplicate_by_authoritative_id() -> None:
    first = Node(
        IdentitySource,
        parameters={"token": "token", "payload": {"items": [1, 2]}},
    )
    equivalent = Node(
        IdentitySource,
        parameters={"payload": {"items": (1, 2)}, "token": "token"},
    )
    root = Node(Pair, bindings={"left": first.value, "right": equivalent.value})
    graph = Graph(root)

    assert first.ID == equivalent.ID
    assert first.definition == equivalent.definition
    assert len(graph.node_list) == 2
    assert graph.execute()["pair"].to_list() == [[3, 3]]
