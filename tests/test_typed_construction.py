from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

import polars as pl
import pytest

from iosislib.core.node import Node
from iosislib.core.tsfn import FrameSignature, TSFN, TSFNConfig
from iosislib.tsfn.transforms import (
    Logit,
    LogitConfig,
    SpreadConfig,
)


@dataclass(frozen=True)
class NestedConfig(TSFNConfig):
    payload: Mapping[str, object]


class NestedSource(TSFN[NestedConfig]):
    VERSION = "1.0.0"
    CONFIG_CLS = NestedConfig

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        return (
            FrameSignature.empty(),
            FrameSignature(columns=(("value", pl.Int64),)),
        )

    def apply(self, lf: pl.LazyFrame | None = None) -> pl.LazyFrame:
        assert lf is None
        return pl.DataFrame(
            {"timestamp": [datetime(2026, 1, 1)], "value": [1]}
        ).lazy()


def logit_config() -> LogitConfig:
    return LogitConfig(
        input_column="probability",
        output_column="score",
        timestamp_column="observed_at",
    )


def logit_parameters() -> dict[str, object]:
    return {
        "input_column": "probability",
        "output_column": "score",
        "timestamp_column": "observed_at",
    }


def test_config_and_mapping_construction_normalize_to_one_node_definition() -> None:
    from_config = Node(Logit, config=logit_config())
    from_mapping = Node(Logit, parameters=logit_parameters())

    assert isinstance(from_config.parameters, LogitConfig)
    assert from_config.parameters == from_mapping.parameters
    assert from_config.function.signature == from_mapping.function.signature
    assert from_config.outputs == from_mapping.outputs
    assert from_config.definition == from_mapping.definition
    assert from_config.ID == from_mapping.ID


def test_direct_tsfn_construction_accepts_exact_config_or_mapping() -> None:
    from_config = Logit(logit_config())
    from_mapping = Logit(logit_parameters())

    assert from_config.parameters == from_mapping.parameters
    assert from_config.signature == from_mapping.signature


def test_wrong_config_class_is_rejected_by_node_and_direct_tsfn() -> None:
    wrong_config = SpreadConfig()

    with pytest.raises(
        TypeError,
        match=r"Logit requires config LogitConfig, got SpreadConfig",
    ):
        Node(Logit, config=wrong_config)  # type: ignore[arg-type]

    with pytest.raises(
        TypeError,
        match=r"Logit requires config LogitConfig, got SpreadConfig",
    ):
        Logit(wrong_config)  # type: ignore[arg-type]


def test_parameters_and_config_are_mutually_exclusive_even_for_empty_mapping() -> None:
    with pytest.raises(ValueError, match="parameters and config are mutually exclusive"):
        Node(Logit, parameters={}, config=LogitConfig())


def test_config_construction_copies_and_deeply_freezes_nested_values() -> None:
    values = [1, 2]
    payload: dict[str, object] = {"values": values, "nested": {"enabled": True}}
    config = NestedConfig(payload=payload)

    node = Node(NestedSource, config=config)
    direct = NestedSource(config)
    values.append(3)
    payload["new"] = "caller mutation"
    nested = payload["nested"]
    assert isinstance(nested, dict)
    nested["enabled"] = False

    for normalized in (node.parameters, direct.parameters):
        assert normalized.payload["values"] == (1, 2)
        assert normalized.payload["nested"] == {"enabled": True}
        assert "new" not in normalized.payload
        with pytest.raises(TypeError):
            normalized.payload["new"] = "mutation"  # type: ignore[index]
        with pytest.raises(AttributeError):
            normalized.payload["values"].append(3)  # type: ignore[union-attr]


def test_explicit_output_binding_matches_dynamic_sugar_and_validates_names() -> None:
    node = Node(Logit, config=logit_config())

    assert node.output("score") == node.score

    with pytest.raises(TypeError, match="Node output name must be a string"):
        node.output(1)  # type: ignore[arg-type]
    with pytest.raises(
        ValueError,
        match=r"Logit.*does not expose output 'missing'.*\['score'\]",
    ):
        node.output("missing")
