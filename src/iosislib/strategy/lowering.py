"""Lower portable strategy declarations into verified core graph roots."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import re
from types import MappingProxyType, ModuleType
from typing import Any, TypeAlias, cast

from iosislib.core.graph import Graph
from iosislib.core.node import Node as CoreNode
from iosislib.core.tsfn import TSFN
from iosislib.strategy.ir import Strategy


OperationKey: TypeAlias = tuple[str, str]
OutputBinding: TypeAlias = tuple[CoreNode[Any], str]


class StrategyLoweringError(ValueError):
    """Raised when a valid strategy cannot be lowered to core declarations."""

    def __init__(self, path: str, message: str):
        self.path = path
        self.message = message
        super().__init__(f"{path}: {message}")


@dataclass(frozen=True, slots=True)
class OperationRegistry:
    """Explicit operation/version mapping used to lower a strategy."""

    operations: Mapping[OperationKey, type[TSFN[Any]]]

    def __post_init__(self) -> None:
        if not isinstance(self.operations, Mapping):
            raise TypeError("OperationRegistry.operations must be a mapping")

        normalized: dict[OperationKey, type[TSFN[Any]]] = {}
        for key, function_cls in self.operations.items():
            if (
                not isinstance(key, tuple)
                or len(key) != 2
                or not all(isinstance(item, str) and item for item in key)
            ):
                raise TypeError(
                    "OperationRegistry keys must be non-empty (operation, version) tuples"
                )
            if not isinstance(function_cls, type) or not issubclass(function_cls, TSFN):
                raise TypeError("OperationRegistry values must be concrete TSFN classes")
            if function_cls.VERSION != key[1]:
                raise ValueError(
                    f"Operation {key[0]!r} is registered as {key[1]!r}, but "
                    f"{function_cls.__name__}.VERSION is {function_cls.VERSION!r}"
                )
            normalized[key] = function_cls
        object.__setattr__(
            self,
            "operations",
            MappingProxyType(dict(sorted(normalized.items()))),
        )

    def resolve(self, operation: str, version: str) -> type[TSFN[Any]]:
        try:
            return self.operations[(operation, version)]
        except KeyError as error:
            raise StrategyLoweringError(
                "registry",
                f"no TSFN is registered for {operation!r}@{version}",
            ) from error


_OPERATION_NAMESPACE = re.compile(r"^[a-z][a-z0-9_.-]*$")
_FIRST_CAPITAL = re.compile(r"(.)([A-Z][a-z]+)")
_ALL_CAPITAL = re.compile(r"([a-z0-9])([A-Z])")


def _operation_name(export_name: str) -> str:
    name = export_name.removesuffix("TSFN")
    first_pass = _FIRST_CAPITAL.sub(r"\1_\2", name)
    return _ALL_CAPITAL.sub(r"\1_\2", first_pass).lower()


def registry_from_exports(
    packages: Mapping[str, ModuleType],
) -> OperationRegistry:
    """Discover concrete public TSFNs from package __all__ declarations.

    Each mapping key is a DSL operation namespace. A public TSFN export becomes
    namespace.snake_case_export_name at the class's declared version.
    Non-TSFN exports such as configs and helpers are ignored.
    """
    if not isinstance(packages, Mapping):
        raise TypeError("registry_from_exports requires a mapping of namespaces to modules")

    operations: dict[OperationKey, type[TSFN[Any]]] = {}
    for namespace, package in sorted(packages.items()):
        if (
            not isinstance(namespace, str)
            or not _OPERATION_NAMESPACE.fullmatch(namespace)
        ):
            raise ValueError("Registry namespaces must be lowercase DSL operation names")
        if not isinstance(package, ModuleType):
            raise TypeError("Registry package values must be modules")
        exports = getattr(package, "__all__", None)
        if not isinstance(exports, (tuple, list)) or not all(
            isinstance(name, str) for name in exports
        ):
            raise TypeError(f"{package.__name__} must define __all__ as strings")

        for export_name in exports:
            try:
                candidate = getattr(package, export_name)
            except AttributeError as error:
                raise ValueError(
                    f"{package.__name__}.__all__ names missing attribute {export_name!r}"
                ) from error
            if not isinstance(candidate, type) or not issubclass(candidate, TSFN):
                continue
            operation = f"{namespace}.{_operation_name(export_name)}"
            key = (operation, candidate.VERSION)
            existing = operations.get(key)
            if existing is not None and existing is not candidate:
                raise ValueError(
                    f"Duplicate exported operation {operation!r}@{candidate.VERSION}"
                )
            operations[key] = candidate
    return OperationRegistry(operations)


def builtin_registry() -> OperationRegistry:
    """Return entries for public built-in transforms, sources, models, and backtests."""
    import iosislib.backtest as backtest
    import iosislib.models as models
    import iosislib.tsfn.adapters as adapters
    import iosislib.tsfn.transforms as transforms

    return registry_from_exports(
        {
            "backtest": backtest,
            "model": models,
            "source": adapters,
            "transform": transforms,
        }
    )


@dataclass(frozen=True, slots=True)
class LoweredStrategy:
    """Shared core nodes and named output bindings produced from one strategy."""

    nodes: Mapping[str, CoreNode[Any]]
    outputs: Mapping[str, OutputBinding]

    def __post_init__(self) -> None:
        object.__setattr__(self, "nodes", MappingProxyType(dict(self.nodes)))
        object.__setattr__(self, "outputs", MappingProxyType(dict(self.outputs)))

    def output(self, name: str) -> OutputBinding:
        try:
            return self.outputs[name]
        except KeyError as error:
            raise ValueError(f"Strategy does not declare an output named {name!r}") from error

    def graph(self, output_name: str) -> Graph:
        """Return the graph rooted at the node that produces one named output."""
        root_node, _ = self.output(output_name)
        return Graph(root_node)

def lower(strategy: Strategy, registry: OperationRegistry) -> LoweredStrategy:
    """Resolve one backend-independent strategy into shared core Node objects."""
    if not isinstance(strategy, Strategy):
        raise TypeError("lower requires a Strategy")
    if not isinstance(registry, OperationRegistry):
        raise TypeError("lower requires an OperationRegistry")

    nodes: dict[str, CoreNode[Any]] = {}
    for node_name in strategy.topological_order:
        declaration = strategy.nodes[node_name]
        path = f"$.nodes.{node_name}"
        try:
            bindings: dict[str, OutputBinding] = {}
            tolerances: dict[str, str | int | float] = {}
            null_policies: dict[str, str] = {}
            null_fill_values: dict[str, object] = {}
            for input_name, input_declaration in declaration.inputs.items():
                parent = nodes[input_declaration.source.node]
                bindings[input_name] = parent.output(input_declaration.source.output)
                if input_declaration.tolerance is not None:
                    tolerances[input_name] = input_declaration.tolerance
                if input_declaration.nulls is not None:
                    null_policies[input_name] = input_declaration.nulls
                if input_declaration.fill is not None:
                    null_fill_values[input_name] = input_declaration.fill

            nodes[node_name] = CoreNode(
                registry.resolve(declaration.op, declaration.version),
                bindings=bindings,
                parameters=cast(Mapping[str, object], declaration.params),
                name=node_name,
                materialize=declaration.materialize,
                tolerances=tolerances,
                null_policies=null_policies,
                null_fill_values=null_fill_values,
            )
        except StrategyLoweringError as error:
            raise StrategyLoweringError(path, error.message) from error
        except (KeyError, TypeError, ValueError) as error:
            raise StrategyLoweringError(path, str(error)) from error
    outputs: dict[str, OutputBinding] = {}
    for output_name, reference in strategy.outputs.items():
        try:
            outputs[output_name] = nodes[reference.node].output(reference.output)
        except (KeyError, TypeError, ValueError) as error:
            raise StrategyLoweringError(
                f"$.outputs.{output_name}", str(error)
            ) from error
    return LoweredStrategy(nodes=nodes, outputs=outputs)


__all__ = [
    "LoweredStrategy",
    "OperationRegistry",
    "OperationKey",
    "OutputBinding",
    "StrategyLoweringError",
    "builtin_registry",
    "lower",
    "registry_from_exports",
]
