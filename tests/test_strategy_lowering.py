from __future__ import annotations


from dataclasses import dataclass
from datetime import datetime
from types import ModuleType

import polars as pl
import pytest

from iosislib.core.tsfn import FrameSignature, NullPolicy, TSFN, TSFNConfig, TimeAxis
from iosislib.strategy import (
    OperationRegistry,
    StrategyLoweringError,
    builtin_registry,
    loads,
    lower,
    registry_from_exports,
)
from iosislib.backtest import BacktestTSFN
from iosislib.models import DenseMLP
from iosislib.tsfn.adapters import CSVSource
from iosislib.tsfn.transforms import Delta


@dataclass(frozen=True)
class SourceConfig(TSFNConfig):
    value: float = 1.0


class Source(TSFN[SourceConfig]):
    VERSION = "1.0.0"
    CONFIG_CLS = SourceConfig

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        return (
            FrameSignature.empty(),
            FrameSignature(time=TimeAxis(), columns=(("value", pl.Float64),)),
        )

    def apply(self, lf: pl.LazyFrame | None = None) -> pl.LazyFrame:
        del lf
        return pl.DataFrame(
            {
                "timestamp": [datetime(2026, 1, 1)],
                "value": [self.parameters.value],
            },
            schema={"timestamp": pl.Datetime, "value": pl.Float64},
        ).lazy()

@dataclass(frozen=True)
class ScaleConfig(TSFNConfig):
    input_column: str = "value"
    output_column: str = "scaled"
    factor: float = 1.0


class Scale(TSFN[ScaleConfig]):
    VERSION = "1.0.0"
    CONFIG_CLS = ScaleConfig

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        return (
            FrameSignature(
                time=TimeAxis(),
                columns=((self.parameters.input_column, pl.Float64),),
            ),
            FrameSignature(
                time=TimeAxis(),
                columns=((self.parameters.output_column, pl.Float64),),
            ),
        )

    def apply(self, lf: pl.LazyFrame | None = None) -> pl.LazyFrame:
        if lf is None:
            raise ValueError("Scale requires an input frame")
        params = self.parameters
        return lf.select(
            "timestamp",
            (pl.col(params.input_column) * params.factor).alias(params.output_column),
        )


REGISTRY = OperationRegistry(
    {
        ("test.source", "1.0.0"): Source,
        ("test.scale", "1.0.0"): Scale,
    }
)

STRATEGY = """
format: iosis.strategy
version: 0.1.0
name: shared-roots
nodes:
  source:
    op: test.source
    version: 1.0.0
    params:
      value: 2.0

  alpha:
    op: test.scale
    version: 1.0.0
    inputs:
      value:
        from: source.value
        tolerance: 1m
        nulls: fill
        fill: 0.0
    params:
      input_column: value
      output_column: alpha
      factor: 2.0

  beta:
    op: test.scale
    version: 1.0.0
    inputs:
      value: source.value
    params:
      input_column: value
      output_column: beta
      factor: 3.0
outputs:
  alpha: alpha.alpha
  beta: beta.beta
"""

def test_lower_creates_shared_nodes_and_one_graph_per_named_output() -> None:
    lowered = lower(loads(STRATEGY), REGISTRY)

    assert lowered.nodes["alpha"].bindings["value"][0] is lowered.nodes["source"]
    assert lowered.nodes["beta"].bindings["value"][0] is lowered.nodes["source"]
    assert lowered.output("alpha") == lowered.nodes["alpha"].output("alpha")
    assert tuple(node.name for node in lowered.graph("alpha").node_list) == (
        "source",
        "alpha",
    )
    assert tuple(node.name for node in lowered.graph("beta").node_list) == (
        "source",
        "beta",
    )
    assert lowered.graph("alpha").execute().get_column("alpha").to_list() == [4.0]
    assert lowered.graph("beta").execute().get_column("beta").to_list() == [6.0]
    alpha = lowered.nodes["alpha"]
    assert alpha.tolerances == {"value": "1m"}
    assert alpha.null_handlers["value"].policy is NullPolicy.FILL
    assert alpha.null_fill_values == {"value": 0.0}

def test_lower_reports_missing_registry_entries_and_unknown_outputs() -> None:
    missing_operation = loads(STRATEGY.replace("test.source", "test.missing"))

    with pytest.raises(StrategyLoweringError, match="no TSFN is registered") as error:
        lower(missing_operation, REGISTRY)
    assert error.value.path == "$.nodes.source"

    with pytest.raises(ValueError, match="does not declare an output"):
        lower(loads(STRATEGY), REGISTRY).graph("unknown")

def test_registry_from_exports_uses_only_public_concrete_tsfns() -> None:
    package = ModuleType("test_package")
    package.__all__ = ("Source", "SourceConfig", "helper")
    package.Source = Source
    package.SourceConfig = SourceConfig
    package.helper = lambda: None

    registry = registry_from_exports({"test": package})

    assert registry.resolve("test.source", Source.VERSION) is Source
    assert len(registry.operations) == 1


def test_builtin_registry_discovers_public_library_tsfns() -> None:
    registry = builtin_registry()

    assert registry.resolve("transform.delta", Delta.VERSION) is Delta
    assert registry.resolve("source.csv_source", CSVSource.VERSION) is CSVSource
    assert registry.resolve("model.dense_mlp", DenseMLP.VERSION) is DenseMLP
    assert registry.resolve("backtest.backtest", BacktestTSFN.VERSION) is BacktestTSFN
