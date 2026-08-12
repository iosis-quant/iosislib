from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

import polars as pl
import pytest

from iosislib.core.graph import Graph
from iosislib.core.tsfn import FrameSignature, TSFN, TSFNConfig, TimeAxis
from iosislib.metrics import MaxDrawdown, MeanSquaredError, extract_metrics
from iosislib.metrics.strategy import (
    MetricSpec,
    ResolvedMetricSpec,
    parse_metric_specs,
    resolve_metric_specs,
)
from iosislib.strategy import (
    OperationRegistry,
    Strategy,
    StrategyLoweringError,
    StrategyValidationError,
    lower,
)
from iosislib.strategy.ir import Reference


@dataclass(frozen=True)
class SourceConfig(TSFNConfig):
    value: float = 1.0
    output_column: str = "value"


class Source(TSFN[SourceConfig]):
    VERSION = "1.0.0"
    CONFIG_CLS = SourceConfig

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        return (
            FrameSignature.empty(),
            FrameSignature(
                time=TimeAxis(),
                columns=((self.parameters.output_column, pl.Float64),),
            ),
        )

    def apply(self, lf: pl.LazyFrame | None = None) -> pl.LazyFrame:
        del lf
        column = self.parameters.output_column
        return pl.DataFrame(
            {
                "timestamp": [
                    datetime(2026, 1, 1, 0),
                    datetime(2026, 1, 1, 1),
                    datetime(2026, 1, 1, 2),
                ],
                column: [
                    self.parameters.value,
                    self.parameters.value,
                    self.parameters.value,
                ],
            },
            schema={"timestamp": pl.Datetime, column: pl.Float64},
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


def _nodes() -> dict[str, object]:
    return {
        "base": {"op": "test.source", "version": "1.0.0", "params": {"value": 1.0}},
        "target": {
            "op": "test.scale",
            "version": "1.0.0",
            "inputs": {"value": "base.value"},
            "params": {
                "input_column": "value",
                "output_column": "target",
                "factor": 2.0,
            },
            "materialize": True,
        },
        "signal": {
            "op": "test.scale",
            "version": "1.0.0",
            "inputs": {"target": "target.target"},
            "params": {
                "input_column": "target",
                "output_column": "equity",
                "factor": 3.0,
            },
            "materialize": True,
        },
        "prediction": {
            "op": "test.scale",
            "version": "1.0.0",
            "inputs": {"equity": "signal.equity"},
            "params": {
                "input_column": "equity",
                "output_column": "prediction",
                "factor": 0.5,
            },
        },
    }


def _strategy(metrics: Mapping[str, object] | None = None) -> Strategy:
    data: dict[str, object] = {
        "format": "iosis.strategy",
        "version": "0.1.0",
        "name": "metrics",
        "nodes": _nodes(),
        "outputs": {"prediction": "prediction.prediction"},
    }
    if metrics is not None:
        data["metadata"] = {"metrics": metrics}
    return Strategy.from_data(data)


def _lowered(metrics: Mapping[str, object] | None = None):
    return lower(_strategy(metrics), REGISTRY)


METRICS = {
    "mse": {
        "prediction": "prediction.prediction",
        "target": "target.target",
    },
    "max_drawdown": "signal.equity",
}


def test_parse_without_metrics_returns_no_specs() -> None:
    assert parse_metric_specs(_strategy().metadata) == ()


def test_parse_accepts_shorthand_and_mapping_forms() -> None:
    specs = parse_metric_specs(_strategy(METRICS).metadata)

    assert [spec.name for spec in specs] == ["max_drawdown", "mse"]
    by_name = {spec.name: spec for spec in specs}
    assert {column: str(reference) for column, reference in by_name["mse"].references.items()} == {
        "prediction": "prediction.prediction",
        "target": "target.target",
    }
    assert by_name["mse"].metric_names == ("mse",)
    assert {column: str(reference) for column, reference in by_name["max_drawdown"].references.items()} == {
        "equity": "signal.equity",
    }
    assert by_name["max_drawdown"].metric_names == ("max_drawdown",)


def test_parse_unknown_extractor_raises() -> None:
    with pytest.raises(StrategyValidationError, match="unknown metric extractor"):
        parse_metric_specs(_strategy({"sharpe": "signal.equity"}).metadata)


def test_parse_unknown_required_column_raises() -> None:
    with pytest.raises(StrategyValidationError, match="does not require column"):
        parse_metric_specs(
            _strategy(
                {
                    "mse": {
                        "prediction": "prediction.prediction",
                        "label": "target.target",
                    }
                }
            ).metadata
        )


def test_parse_missing_required_column_raises() -> None:
    with pytest.raises(StrategyValidationError, match="requires column source"):
        parse_metric_specs(
            _strategy({"mse": {"prediction": "prediction.prediction"}}).metadata
        )


def test_parse_rejects_output_mismatching_column_name() -> None:
    with pytest.raises(StrategyValidationError, match="must match column name"):
        parse_metric_specs(
            _strategy(
                {
                    "mse": {
                        "prediction": "signal.equity",
                        "target": "target.target",
                    }
                }
            ).metadata
        )


def test_parse_rejects_shorthand_for_multi_column_extractor() -> None:
    with pytest.raises(StrategyValidationError, match="requires columns"):
        parse_metric_specs(_strategy({"mse": "prediction.prediction"}).metadata)


def test_parse_rejects_non_mapping_metrics_block() -> None:
    strategy = Strategy.from_data(
        {
            "format": "iosis.strategy",
            "version": "0.1.0",
            "name": "block",
            "nodes": _nodes(),
            "outputs": {"prediction": "prediction.prediction"},
            "metadata": {"metrics": "nope"},
        }
    )
    with pytest.raises(StrategyValidationError, match="expected a mapping"):
        parse_metric_specs(strategy.metadata)


def test_parse_rejects_bad_reference() -> None:
    with pytest.raises(StrategyValidationError, match="node.output"):
        parse_metric_specs(_strategy({"max_drawdown": "just-one"}).metadata)


def test_resolve_resolves_node_ids_and_preserves_sources() -> None:
    lowered = _lowered()
    specs = resolve_metric_specs(parse_metric_specs(_strategy(METRICS).metadata), lowered)

    assert [spec.name for spec in specs] == ["max_drawdown", "mse"]
    mse = specs[1]
    assert mse.sources == {
        "prediction": (lowered.nodes["prediction"].ID, "prediction"),
        "target": (lowered.nodes["target"].ID, "target"),
    }
    assert mse.metric_names == ("mse",)
    assert specs[0].sources == {"equity": (lowered.nodes["signal"].ID, "equity")}


def test_resolve_unknown_node_raises() -> None:
    metrics = {"max_drawdown": "missing.equity"}
    with pytest.raises(StrategyLoweringError, match="unknown node"):
        resolve_metric_specs(parse_metric_specs(_strategy(metrics).metadata), _lowered())


def test_resolve_missing_output_raises() -> None:
    metrics = {"max_drawdown": "base.equity"}
    with pytest.raises(StrategyLoweringError, match="does not expose output"):
        resolve_metric_specs(parse_metric_specs(_strategy(metrics).metadata), _lowered())


def test_resolve_rejects_node_that_will_not_be_cached() -> None:
    uncached = {
        "base": {"op": "test.source", "version": "1.0.0", "params": {"value": 1.0}},
        "target": {
            "op": "test.scale",
            "version": "1.0.0",
            "inputs": {"value": "base.value"},
            "params": {
                "input_column": "value",
                "output_column": "target",
                "factor": 2.0,
            },
        },
        "prediction": {
            "op": "test.scale",
            "version": "1.0.0",
            "inputs": {"target": "target.target"},
            "params": {
                "input_column": "target",
                "output_column": "prediction",
                "factor": 0.5,
            },
        },
    }
    strategy = Strategy.from_data(
        {
            "format": "iosis.strategy",
            "version": "0.1.0",
            "name": "uncached",
            "nodes": uncached,
            "outputs": {"prediction": "prediction.prediction"},
            "metadata": {
                "metrics": {
                    "mse": {
                        "prediction": "prediction.prediction",
                        "target": "target.target",
                    }
                }
            },
        }
    )
    lowered = lower(strategy, REGISTRY)

    with pytest.raises(StrategyLoweringError, match="must be materialized"):
        resolve_metric_specs(parse_metric_specs(strategy.metadata), lowered)


def test_resolve_accepts_root_and_source_nodes() -> None:
    strategy = Strategy.from_data(
        {
            "format": "iosis.strategy",
            "version": "0.1.0",
            "name": "roots",
            "nodes": {
                "equity_source": {
                    "op": "test.source",
                    "version": "1.0.0",
                    "params": {"value": 1.0, "output_column": "equity"},
                },
                "target": {
                    "op": "test.scale",
                    "version": "1.0.0",
                    "inputs": {"equity": "equity_source.equity"},
                    "params": {
                        "input_column": "equity",
                        "output_column": "target",
                        "factor": 2.0,
                    },
                },
                "prediction": {
                    "op": "test.scale",
                    "version": "1.0.0",
                    "inputs": {"target": "target.target"},
                    "params": {
                        "input_column": "target",
                        "output_column": "prediction",
                        "factor": 0.5,
                    },
                },
            },
            "outputs": {
                "prediction": "prediction.prediction",
                "target": "target.target",
            },
            "metadata": {
                "metrics": {
                    "max_drawdown": "equity_source.equity",
                    "mse": {
                        "prediction": "prediction.prediction",
                        "target": "target.target",
                    },
                }
            },
        }
    )
    lowered = lower(strategy, REGISTRY)
    specs = resolve_metric_specs(parse_metric_specs(strategy.metadata), lowered)

    by_name = {spec.name: spec for spec in specs}
    assert by_name["max_drawdown"].sources["equity"][0] == lowered.nodes["equity_source"].ID
    assert by_name["mse"].sources["prediction"][0] == lowered.nodes["prediction"].ID
    assert by_name["mse"].sources["target"][0] == lowered.nodes["target"].ID


def test_resolve_rejects_duplicate_metric_names_across_specs() -> None:
    lowered = _lowered()
    references = {
        "prediction": "prediction.prediction",
        "target": "target.target",
    }
    specs = (
        MetricSpec(
            name="mse",
            extractor=MeanSquaredError(),
            references={
                column: Reference.parse(reference)
                for column, reference in references.items()
            },
        ),
        MetricSpec(
            name="second_mse",
            extractor=MeanSquaredError(),
            references={
                column: Reference.parse(reference)
                for column, reference in references.items()
            },
        ),
    )

    with pytest.raises(StrategyLoweringError, match="duplicate metric name"):
        resolve_metric_specs(specs, lowered)


def test_resolve_requires_a_lowered_strategy() -> None:
    with pytest.raises(TypeError, match="LoweredStrategy"):
        resolve_metric_specs((), None)


def test_resolve_requires_metric_specs() -> None:
    with pytest.raises(TypeError, match="MetricSpec"):
        resolve_metric_specs((object(),), _lowered())


def test_metric_config_does_not_change_node_or_graph_identity() -> None:
    plain = _lowered()
    configured = _lowered(METRICS)

    for name in _nodes():
        assert configured.nodes[name].ID == plain.nodes[name].ID
    assert configured.graph("prediction").ID == plain.graph("prediction").ID


def test_parse_resolve_extract_end_to_end() -> None:
    strategy = _strategy(METRICS)
    lowered = lower(strategy, REGISTRY)
    specs = resolve_metric_specs(parse_metric_specs(strategy.metadata), lowered)

    frames: dict[str, pl.DataFrame] = {}
    for node_id in {source[0] for spec in specs for source in spec.sources.values()}:
        node = next(
            candidate for candidate in lowered.nodes.values() if candidate.ID == node_id
        )
        frames[node_id] = Graph(node).execute()

    values: dict[str, float] = {}
    for spec in specs:
        unique_frames = [
            frames[node_id]
            for node_id in dict.fromkeys(
                source[0] for source in spec.sources.values()
            )
        ]
        row = extract_metrics(unique_frames, spec.extractor).to_dicts()
        if row:
            values.update(row[0])

    assert values == {
        "mse": 1.0,
        "max_drawdown": 0.0,
    }


def test_metric_spec_validation() -> None:
    with pytest.raises(ValueError, match="cover exactly"):
        MetricSpec(
            name="mse",
            extractor=MeanSquaredError(),
            references={"prediction": Reference.parse("a.prediction")},
        )
    with pytest.raises(TypeError, match="must map column names to References"):
        MetricSpec(
            name="mse",
            extractor=MeanSquaredError(),
            references={"prediction": None},
        )
    with pytest.raises(ValueError, match="non-empty"):
        MetricSpec(name="", extractor=MaxDrawdown(), references={})
    with pytest.raises(TypeError, match="MetricExtractor"):
        MetricSpec(name="x", extractor=object(), references={})


def test_resolved_metric_spec_validation() -> None:
    with pytest.raises(ValueError, match="cover exactly"):
        ResolvedMetricSpec(
            name="mse",
            extractor=MeanSquaredError(),
            sources={"prediction": ("a" * 64, "prediction")},
        )
    with pytest.raises(TypeError, match="node_id, output_name"):
        ResolvedMetricSpec(
            name="mse",
            extractor=MeanSquaredError(),
            sources={"prediction": ("a",), "target": ("b",)},
        )
