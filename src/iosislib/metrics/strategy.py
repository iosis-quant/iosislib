"""Declarative metric configuration and its resolution against a strategy.

Metrics are configured in a strategy's free-form ``metadata.metrics`` block:

.. code-block:: yaml

    metadata:
      metrics:
        mse:
          prediction: model.prediction
          target: target.source
        max_drawdown: backtest.equity

Each key is the extractor name from the registry, each value either a bare
``node.output`` reference (when the extractor requires a single column) or a
mapping from each required column to its ``node.output`` source. The referenced
output name must match the configured column name; renaming is not supported.

Metric configuration never affects graph identity: it is validated during
compilation but contributes nothing to node definitions or output graphs.
Resolution only guarantees that each referenced node is one whose output the
executor will cache, so the metrics can be computed after the fact from cached
frames without re-execution.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Never

from iosislib.core.node import Node as CoreNode
from iosislib.metrics.extractor import MetricExtractor
from iosislib.metrics.extractors import MaxDrawdown, MeanSquaredError
from iosislib.strategy.ir import Reference, StrategyValidationError
from iosislib.strategy.lowering import LoweredStrategy, StrategyLoweringError


_METRIC_EXTRACTORS: Mapping[str, type[MetricExtractor]] = {
    "mse": MeanSquaredError,
    "max_drawdown": MaxDrawdown,
}


@dataclass(frozen=True, slots=True)
class MetricSpec:
    """One declarative metric: a named extractor and its column sources."""

    name: str
    extractor: MetricExtractor
    references: Mapping[str, Reference]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("MetricSpec.name must be a non-empty string")
        if not isinstance(self.extractor, MetricExtractor):
            raise TypeError("MetricSpec.extractor must be a MetricExtractor")
        if not isinstance(self.references, Mapping) or not all(
            isinstance(column, str) and isinstance(reference, Reference)
            for column, reference in self.references.items()
        ):
            raise TypeError(
                "MetricSpec.references must map column names to References"
            )
        if set(self.references) != set(self.extractor.required_columns()):
            raise ValueError(
                "MetricSpec.references must cover exactly the extractor's "
                f"required columns {self.extractor.required_columns()}"
            )

    @property
    def metric_names(self) -> tuple[str, ...]:
        return self.extractor.metric_names()


@dataclass(frozen=True, slots=True)
class ResolvedMetricSpec:
    """A metric whose column sources are content-addressed node outputs."""

    name: str
    extractor: MetricExtractor
    sources: Mapping[str, tuple[str, str]]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("ResolvedMetricSpec.name must be a non-empty string")
        if not isinstance(self.extractor, MetricExtractor):
            raise TypeError("ResolvedMetricSpec.extractor must be a MetricExtractor")
        if not isinstance(self.sources, Mapping) or not all(
            isinstance(column, str)
            and isinstance(value, tuple)
            and len(value) == 2
            and all(isinstance(item, str) and item for item in value)
            for column, value in self.sources.items()
        ):
            raise TypeError(
                "ResolvedMetricSpec.sources must map column names to "
                "(node_id, output_name) pairs"
            )
        if set(self.sources) != set(self.extractor.required_columns()):
            raise ValueError(
                "ResolvedMetricSpec.sources must cover exactly the extractor's "
                f"required columns {self.extractor.required_columns()}"
            )

    @property
    def metric_names(self) -> tuple[str, ...]:
        return self.extractor.metric_names()


def _fail(path: str, message: str) -> Never:
    raise StrategyValidationError(path, message)


def _reference(value: object, path: str) -> Reference:
    if not isinstance(value, str):
        _fail(path, "expected a 'node.output' reference string")
    return Reference.parse(value, path)


def _column_sources(
    extractor: MetricExtractor,
    value: object,
    path: str,
) -> Mapping[str, Reference]:
    required = extractor.required_columns()
    if isinstance(value, str):
        if len(required) != 1:
            _fail(
                path,
                f"{type(extractor).__name__} requires columns "
                f"{sorted(required)}; provide a mapping of each column to its "
                "'node.output' reference",
            )
        reference = _reference(value, path)
        return {required[0]: reference}
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            _fail(path, "mapping keys must be column names")
        sources: dict[str, Reference] = {}
        for column, reference_value in sorted(value.items()):
            column = str(column)
            if column not in required:
                _fail(
                    path,
                    f"{type(extractor).__name__} does not require column "
                    f"{column!r}; required columns are {sorted(required)}",
                )
            sources[column] = _reference(reference_value, f"{path}.{column}")
        missing = sorted(set(required) - set(sources))
        if missing:
            _fail(
                path,
                f"{type(extractor).__name__} requires column source(s) for "
                f"{missing}",
            )
        return sources
    _fail(
        path,
        "expected a 'node.output' reference or a mapping of column names to "
        "references",
    )


def _validate_output_matches_column(
    sources: Mapping[str, Reference],
    path: str,
) -> None:
    for column, reference in sources.items():
        if reference.output != column:
            _fail(
                f"{path}.{column}",
                f"referenced output {reference.output!r} must match column name "
                f"{column!r}; column renaming is not supported",
            )


def parse_metric_specs(
    metadata: Mapping[str, object],
    *,
    path: str = "$.metadata.metrics",
) -> tuple[MetricSpec, ...]:
    """Validate and parse a strategy's ``metadata.metrics`` block.

    ``metadata`` is the strategy's free-form metadata mapping. An absent
    ``metrics`` key yields no specs. Invalid blocks raise
    :class:`StrategyValidationError` with a path into the document.
    """
    if not isinstance(metadata, Mapping):
        _fail(path, "expected a mapping")
    metrics = metadata.get("metrics")
    if metrics is None:
        return ()
    if not isinstance(metrics, Mapping) or not all(
        isinstance(key, str) for key in metrics
    ):
        _fail(path, "expected a mapping of extractor names to column sources")

    specs: list[MetricSpec] = []
    for extractor_name, value in sorted(metrics.items()):
        extractor_name = str(extractor_name)
        extractor_cls = _METRIC_EXTRACTORS.get(extractor_name)
        if extractor_cls is None:
            _fail(
                f"{path}.{extractor_name}",
                f"unknown metric extractor {extractor_name!r}; available: "
                f"{sorted(_METRIC_EXTRACTORS)}",
            )
        extractor = extractor_cls()
        sources = _column_sources(extractor, value, f"{path}.{extractor_name}")
        _validate_output_matches_column(sources, f"{path}.{extractor_name}")
        specs.append(
            MetricSpec(
                name=extractor_name,
                extractor=extractor,
                references=dict(sorted(sources.items())),
            )
        )
    return tuple(specs)


def _is_cached(node: CoreNode[Any], root_nodes: set[CoreNode[Any]]) -> bool:
    if node.materialize:
        return True
    if node in root_nodes:
        return True
    return not node.bindings


def resolve_metric_specs(
    specs: Sequence[MetricSpec],
    lowered: LoweredStrategy,
) -> tuple[ResolvedMetricSpec, ...]:
    """Resolve parsed specs against one lowered strategy.

    Each ``node.output`` reference is resolved to its content-addressed node
    output and validated against the strategy's executable graph. A referenced
    node that the executor would not cache raises :class:`StrategyLoweringError`
    so metrics are never silently skipped.
    """
    if not isinstance(lowered, LoweredStrategy):
        raise TypeError("resolve_metric_specs requires a LoweredStrategy")
    if not isinstance(specs, Sequence):
        raise TypeError("specs must be a sequence of MetricSpec instances")

    root_nodes = {node for node, _ in lowered.outputs.values()}
    resolved: list[ResolvedMetricSpec] = []
    seen_names: set[str] = set()
    for spec in specs:
        if not isinstance(spec, MetricSpec):
            raise TypeError("specs must be MetricSpec instances")
        path = f"$.metadata.metrics.{spec.name}"
        sources: dict[str, tuple[str, str]] = {}
        for column, reference in sorted(spec.references.items()):
            node = lowered.nodes.get(reference.node)
            if node is None:
                raise StrategyLoweringError(
                    path,
                    f"references unknown node {reference.node!r}",
                )
            if reference.output not in node.outputs:
                raise StrategyLoweringError(
                    f"{path}.{column}",
                    f"node {reference.node!r} does not expose output "
                    f"{reference.output!r}",
                )
            if not _is_cached(node, root_nodes):
                raise StrategyLoweringError(
                    f"{path}.{column}",
                    f"node {reference.node!r} must be materialized to extract "
                    "metrics; set 'materialize: true' on it",
                )
            sources[column] = (node.ID, reference.output)
        for metric_name in spec.metric_names:
            if metric_name in seen_names:
                raise StrategyLoweringError(
                    path,
                    f"duplicate metric name {metric_name!r}",
                )
            seen_names.add(metric_name)
        resolved.append(
            ResolvedMetricSpec(
                name=spec.name,
                extractor=spec.extractor,
                sources=dict(sorted(sources.items())),
            )
        )
    return tuple(resolved)


__all__ = [
    "MetricSpec",
    "ResolvedMetricSpec",
    "parse_metric_specs",
    "resolve_metric_specs",
]
