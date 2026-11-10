from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import polars as pl
import pytest

from iosislib.core.graph import Graph
from iosislib.core.tsfn import FrameSignature, TSFN, TSFNConfig
from iosislib.strategy import (
    EXPANSION_FORMAT,
    EXPANSION_VERSION,
    Input,
    OperationRegistry,
    Reference,
    StrategySyntaxError,
    StrategyValidationError,
    builtin_registry,
    expansion_schema,
    load_expansion,
    loads_expansion,
    lower,
)


@dataclass(frozen=True)
class SeriesConfig(TSFNConfig):
    xs: tuple[float, ...]
    ys: tuple[float, ...]


class SeriesSource(TSFN):
    VERSION = "1.0.0"
    CONFIG_CLS = SeriesConfig

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        return (
            FrameSignature.empty(),
            FrameSignature(columns=(("x", pl.Float64), ("y", pl.Float64))),
        )

    def apply(self) -> pl.LazyFrame:
        n = len(self.parameters.xs)
        return pl.DataFrame(
            {
                "timestamp": [
                    datetime(2026, 1, 1) + timedelta(minutes=i) for i in range(n)
                ],
                "x": list(self.parameters.xs),
                "y": list(self.parameters.ys),
            },
            schema={
                "timestamp": pl.Datetime,
                "x": pl.Float64,
                "y": pl.Float64,
            },
        ).lazy()


def registry() -> OperationRegistry:
    return OperationRegistry(
        {
            **builtin_registry().operations,
            ("test.series_source", "1.0.0"): SeriesSource,
        }
    )


EXPANSION = """\
format: iosis.expansion
version: 0.1.0
name: demo-expansion
description: Clone a source strategy into block-permute variants.
nodes:
  data:
    op: test.series_source
    version: 1.0.0
    params:
      xs: [1.0, 2.0, 3.0, 4.0]
      ys: [10.0, 20.0, 30.0, 40.0]
  x_lag:
    op: transform.lag
    version: 0.2.0
    inputs:
      x: data.x
    params:
      input_column: x
      output_column: x_lag
      periods: 1
outputs:
  target: data.y
  temporal: x_lag.x_lag
expand:
  at: data
  template:
    op: perturb.block_permute
    version: 0.1.0
    inputs:
      x: {from: data.x, nulls: propagate}
      y: {from: data.y, nulls: propagate}
    params:
      seed: 42
  variants:
    - label: b1
      params: {block_length: 1}
    - label: b4
      params: {block_length: 4}
"""


VARIANTS_BLOCK = """\
  variants:
    - label: b1
      params: {block_length: 1}
    - label: b4
      params: {block_length: 4}
"""


def test_loads_expansion_builds_expanded_strategy() -> None:
    expanded = loads_expansion(EXPANSION)

    assert set(expanded.strategy.nodes) == {
        "data",
        "x_lag",
        "data__b1",
        "data__b4",
        "x_lag__b1",
        "x_lag__b4",
    }
    assert set(expanded.strategy.outputs) == {
        "target",
        "temporal",
        "b1_target",
        "b1_temporal",
        "b4_target",
        "b4_temporal",
    }
    assert expanded.variant_outputs["target"] is None
    assert expanded.variant_outputs["temporal"] is None
    assert expanded.variant_outputs["b1_target"] == "b1"
    assert expanded.variant_outputs["b4_temporal"] == "b4"


def test_loads_expansion_merges_template_and_variant_params() -> None:
    expanded = loads_expansion(EXPANSION)

    assert expanded.strategy.nodes["data__b1"].params == {
        "seed": 42,
        "block_length": 1,
    }
    assert expanded.strategy.nodes["data__b4"].params == {
        "seed": 42,
        "block_length": 4,
    }


def test_loads_expansion_rewires_downstream_to_the_variant() -> None:
    expanded = loads_expansion(EXPANSION)

    assert expanded.strategy.nodes["data__b1"].inputs["x"] == Input(
        Reference("data", "x"), nulls="propagate"
    )
    assert expanded.strategy.nodes["x_lag__b1"].inputs["x"].source == Reference(
        "data__b1", "x"
    )
    assert expanded.strategy.nodes["x_lag"].inputs["x"].source == Reference(
        "data", "x"
    )


def test_loads_expansion_keeps_descriptions_and_base_outputs() -> None:
    expanded = loads_expansion(EXPANSION)

    assert expanded.strategy.name == "demo-expansion"
    assert expanded.strategy.description == (
        "Clone a source strategy into block-permute variants."
    )
    assert expanded.strategy.outputs["target"] == Reference("data", "y")
    assert expanded.strategy.outputs["temporal"] == Reference("x_lag", "x_lag")


def test_expansion_document_is_rejected_by_the_strategy_loader() -> None:
    from iosislib.strategy import loads

    with pytest.raises(StrategyValidationError) as error:
        loads(EXPANSION)

    assert error.value.path == "$"
    assert "expand" in error.value.message


def test_load_expansion_reads_a_utf8_file(tmp_path) -> None:
    path = tmp_path / "expansion.yaml"
    path.write_text(EXPANSION, encoding="utf-8")

    expanded = load_expansion(path)

    assert set(expanded.strategy.nodes) == set(
        loads_expansion(EXPANSION).strategy.nodes
    )


def test_expanded_strategy_lowers_and_executes_every_output() -> None:
    expanded = loads_expansion(EXPANSION)
    lowered = lower(expanded.strategy, registry())

    frames = {}
    for name in lowered.outputs:
        node, _column = lowered.outputs[name]
        frames[name] = Graph(node).execute()

    assert set(frames) == set(expanded.strategy.outputs)
    assert sorted(frames["target"]["y"].to_list()) == [10.0, 20.0, 30.0, 40.0]

    baseline_pairs = sorted(
        zip(frames["target"]["x"].to_list(), frames["target"]["y"].to_list())
    )
    variant_pairs = sorted(
        zip(frames["b1_target"]["x"].to_list(), frames["b1_target"]["y"].to_list())
    )
    assert variant_pairs == baseline_pairs
    assert frames["b1_target"]["y"].to_list() != frames["target"]["y"].to_list()

    baseline_temporal = frames["temporal"]["x_lag"].to_list()
    variant_temporal = frames["b1_temporal"]["x_lag"].to_list()
    assert variant_temporal != baseline_temporal


def test_single_multi_terminal_execute_collects_terminal_outputs() -> None:
    expanded = loads_expansion(EXPANSION)
    lowered = lower(expanded.strategy, registry())

    result = Graph(list(lowered.nodes.values())).execute()

    assert isinstance(result, dict)

    def frame(name: str) -> pl.DataFrame:
        node, _column = lowered.outputs[name]
        return result[node.ID]

    assert frame("temporal")["x_lag"].to_list() != frame("b1_temporal")[
        "x_lag"
    ].to_list()
    assert frame("b4_temporal")["x_lag"].to_list() == frame("temporal")[
        "x_lag"
    ].to_list()


@pytest.mark.parametrize(
    ("text", "path", "message"),
    [
        (
            EXPANSION.replace("format: iosis.expansion", "format: iosis.strategy", 1),
            "$",
            "expected format",
        ),
        (
            EXPANSION.replace("version: 0.1.0", "version: 0.2.0", 1),
            "$",
            "expected version",
        ),
        (
            EXPANSION.replace(
                "name: demo-expansion", "name: demo-expansion\nsurprise: true", 1
            ),
            "$",
            "unknown field",
        ),
        (
            EXPANSION.replace("expand:\n  at: data", "surprise:\n  at: data", 1),
            "$",
            "missing required field(s): expand",
        ),
        (
            EXPANSION.replace("at: data", "at: missing", 1),
            "$.expand.at",
            "references unknown node",
        ),
        (
            EXPANSION.replace(VARIANTS_BLOCK, "  variants: []\n", 1),
            "$.expand.variants",
            "at least one variant",
        ),
        (
            EXPANSION.replace(
                "    - label: b1\n      params: {block_length: 1}",
                "    - params: {block_length: 1}",
                1,
            ),
            "$.expand.variants[0]",
            "missing required field(s): label",
        ),
        (
            EXPANSION.replace("- label: b4", "- label: b1", 1),
            "$.expand.variants[1].label",
            "duplicate variant label",
        ),
        (
            EXPANSION.replace(
                "op: perturb.block_permute\n    version: 0.1.0",
                "op: perturb.block_permute",
                1,
            ),
            "$.expand.template",
            "missing required field(s): version",
        ),
        (
            EXPANSION.replace("params: {block_length: 1}", "params: [1, 2]", 1),
            "$.expand.variants[0].params",
            "expected a mapping",
        ),
        (
            EXPANSION.replace(
                "    - label: b1\n      params: {block_length: 1}",
                "    - label: b1\n      surprise: true",
                1,
            ),
            "$.expand.variants[0]",
            "unknown field",
        ),
        (
            EXPANSION.replace("x: data.x", "x: missing.x", 1),
            "$.nodes.x_lag.inputs.x",
            "unknown node",
        ),
    ],
)
def test_expansion_validation_errors_point_to_the_bad_declaration(
    text: str, path: str, message: str
) -> None:
    with pytest.raises(StrategyValidationError) as error:
        loads_expansion(text)

    assert error.value.path == path
    assert message in error.value.message


def test_empty_expansion_document_is_rejected() -> None:
    with pytest.raises(StrategyValidationError) as error:
        loads_expansion("")

    assert error.value.path == "$"
    assert "empty" in error.value.message


def test_anchors_are_rejected_in_expansion_documents() -> None:
    with pytest.raises(StrategySyntaxError, match="anchors and aliases"):
        loads_expansion(EXPANSION.replace("name: demo-expansion", "name: &n demo", 1))


def test_packaged_expansion_json_schema_describes_current_format() -> None:
    document = expansion_schema()

    assert document["$schema"].endswith("2020-12/schema")
    assert document["properties"]["format"]["const"] == EXPANSION_FORMAT
    assert document["properties"]["version"]["const"] == EXPANSION_VERSION
    assert set(document["required"]) == {
        "format",
        "version",
        "name",
        "nodes",
        "outputs",
        "expand",
    }
    assert set(document["$defs"]["expand"]["required"]) == {"at", "template", "variants"}
    assert set(document["$defs"]["node"]["required"]) == {"op", "version"}
