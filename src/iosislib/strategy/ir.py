from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Any, Never, TypeAlias, cast


STRATEGY_FORMAT = "iosis.strategy"
STRATEGY_VERSION = "0.1.0"

Scalar: TypeAlias = str | int | float | bool | None
Value: TypeAlias = Scalar | tuple["Value", ...] | Mapping[str, "Value"]
Tolerance: TypeAlias = str | int | float | None

_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")
_OPERATION = re.compile(r"^[a-z][a-z0-9_.-]*$")
_SEMVER = re.compile(
    r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)"
    r"(?:-((?:0|[1-9][0-9]*|[0-9]*[A-Za-z-][0-9A-Za-z-]*)"
    r"(?:\.(?:0|[1-9][0-9]*|[0-9]*[A-Za-z-][0-9A-Za-z-]*))*))?"
    r"(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$"
)
_NULL_POLICIES = frozenset({"error", "propagate", "drop", "fill", "pass"})


class StrategyValidationError(ValueError):
    """Raised when a strategy document is structurally invalid."""

    def __init__(self, path: str, message: str):
        self.path = path
        self.message = message
        super().__init__(f"{path}: {message}")


def _fail(path: str, message: str) -> Never:
    raise StrategyValidationError(path, message)


def _mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        _fail(path, "expected a mapping")
    if not all(isinstance(key, str) for key in value):
        _fail(path, "mapping keys must be strings")
    return cast(Mapping[str, object], value)


def _check_keys(
    value: Mapping[str, object],
    path: str,
    *,
    allowed: set[str],
    required: Set[str] = frozenset(),
) -> None:
    missing = sorted(required - value.keys())
    if missing:
        _fail(path, f"missing required field(s): {', '.join(missing)}")
    extra = sorted(value.keys() - allowed)
    if extra:
        _fail(path, f"unknown field(s): {', '.join(extra)}")


def _identifier(value: object, path: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        _fail(
            path,
            "expected an identifier beginning with a letter and containing only "
            "letters, digits, '_' or '-'",
        )
    return value


def _semver(value: object, path: str) -> str:
    if not isinstance(value, str) or not _SEMVER.fullmatch(value):
        _fail(path, "expected a semantic version such as '0.1.0'")
    return value


def _freeze_value(value: object, path: str) -> Value:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            _fail(path, "non-finite numbers are not supported")
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            _fail(path, "mapping keys must be strings")
        return MappingProxyType(
            {
                key: _freeze_value(item, f"{path}.{key}")
                for key, item in sorted(value.items())
            }
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(
            _freeze_value(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    _fail(
        path,
        f"unsupported value type {type(value).__name__}; use JSON-compatible values",
    )


def _thaw_value(value: Value) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_value(item) for item in value]
    return value


@dataclass(frozen=True, slots=True, order=True)
class Reference:
    node: str
    output: str

    def __post_init__(self) -> None:
        _identifier(self.node, "reference.node")
        _identifier(self.output, "reference.output")

    @classmethod
    def parse(cls, value: object, path: str = "reference") -> Reference:
        if not isinstance(value, str):
            _fail(path, "expected a 'node.output' string")
        parts = value.split(".")
        if len(parts) != 2:
            _fail(path, "expected exactly one '.' in a 'node.output' reference")
        return cls(
            node=_identifier(parts[0], f"{path}.node"),
            output=_identifier(parts[1], f"{path}.output"),
        )

    def __str__(self) -> str:
        return f"{self.node}.{self.output}"


@dataclass(frozen=True, slots=True)
class Input:
    source: Reference
    tolerance: Tolerance = None
    nulls: str | None = None
    fill: Scalar = None

    def __post_init__(self) -> None:
        if not isinstance(self.source, Reference):
            raise TypeError("Input.source must be a Reference")
        tolerance = self.tolerance
        if isinstance(tolerance, bool) or not isinstance(
            tolerance, (str, int, float, type(None))
        ):
            raise TypeError("Input.tolerance must be a string, number, or None")
        if isinstance(tolerance, str) and not tolerance:
            raise ValueError("Input.tolerance must be non-empty when provided")
        if isinstance(tolerance, (int, float)) and tolerance < 0:
            raise ValueError("Input.tolerance must be non-negative")
        if isinstance(tolerance, float) and not isfinite(tolerance):
            raise ValueError("Input.tolerance must be finite")
        if self.nulls is not None and self.nulls not in _NULL_POLICIES:
            raise ValueError(
                f"Input.nulls must be one of {sorted(_NULL_POLICIES)}"
            )
        if self.nulls == "fill" and self.fill is None:
            raise ValueError("Input.fill is required when nulls is 'fill'")
        if self.nulls != "fill" and self.fill is not None:
            raise ValueError("Input.fill is only valid when nulls is 'fill'")
        _freeze_value(self.fill, "input.fill")

    @classmethod
    def from_data(cls, value: object, path: str) -> Input:
        if isinstance(value, str):
            return cls(source=Reference.parse(value, path))

        data = _mapping(value, path)
        _check_keys(
            data,
            path,
            allowed={"from", "tolerance", "nulls", "fill"},
            required={"from"},
        )
        nulls = data.get("nulls")
        if "nulls" in data and not isinstance(nulls, str):
            _fail(f"{path}.nulls", "expected a string")
        assert nulls is None or isinstance(nulls, str)
        if "tolerance" in data and data["tolerance"] is None:
            _fail(f"{path}.tolerance", "omit this field instead of using null")
        if "fill" in data and data["fill"] is None:
            _fail(f"{path}.fill", "expected a non-null scalar value")
        fill = data.get("fill")
        if isinstance(fill, (Mapping, list, tuple)):
            _fail(f"{path}.fill", "expected a scalar value")
        try:
            return cls(
                source=Reference.parse(data["from"], f"{path}.from"),
                tolerance=cast(Tolerance, data.get("tolerance")),
                nulls=nulls,
                fill=cast(Scalar, fill),
            )
        except (TypeError, ValueError) as exc:
            _fail(path, str(exc))

    def to_data(self) -> str | dict[str, Scalar]:
        if self.tolerance is None and self.nulls is None and self.fill is None:
            return str(self.source)
        result: dict[str, Scalar] = {"from": str(self.source)}
        if self.tolerance is not None:
            result["tolerance"] = self.tolerance
        if self.nulls is not None:
            result["nulls"] = self.nulls
        if self.fill is not None:
            result["fill"] = self.fill
        return result


@dataclass(frozen=True, slots=True)
class Node:
    op: str
    version: str
    inputs: Mapping[str, Input] = field(default_factory=dict)
    params: Mapping[str, Value] = field(default_factory=dict)
    materialize: bool | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.op, str) or not _OPERATION.fullmatch(self.op):
            raise ValueError(
                "Node.op must be a lowercase operation name such as "
                "'transform.logit'"
            )
        _semver(self.version, "node.version")
        inputs = _mapping(self.inputs, "node.inputs")
        normalized_inputs: dict[str, Input] = {}
        for name, input_value in sorted(inputs.items()):
            _identifier(name, f"node.inputs.{name}")
            if not isinstance(input_value, Input):
                raise TypeError(f"Node input '{name}' must be an Input")
            normalized_inputs[name] = input_value
        object.__setattr__(self, "inputs", MappingProxyType(normalized_inputs))

        params = _freeze_value(self.params, "node.params")
        if not isinstance(params, Mapping):
            raise TypeError("Node.params must be a mapping")
        object.__setattr__(self, "params", params)
        if self.materialize is not None and not isinstance(self.materialize, bool):
            raise TypeError("Node.materialize must be a boolean or None")

    @classmethod
    def from_data(cls, value: object, path: str) -> Node:
        data = _mapping(value, path)
        _check_keys(
            data,
            path,
            allowed={"op", "version", "inputs", "params", "materialize"},
            required={"op", "version"},
        )
        op = data["op"]
        if not isinstance(op, str):
            _fail(f"{path}.op", "expected a string")
        version = _semver(data["version"], f"{path}.version")
        raw_inputs = _mapping(data.get("inputs", {}), f"{path}.inputs")
        inputs = {
            _identifier(name, f"{path}.inputs.{name}"): Input.from_data(
                input_value, f"{path}.inputs.{name}"
            )
            for name, input_value in raw_inputs.items()
        }
        params = _mapping(data.get("params", {}), f"{path}.params")
        materialize = data.get("materialize")
        if "materialize" in data and not isinstance(materialize, bool):
            _fail(f"{path}.materialize", "expected a boolean")
        assert materialize is None or isinstance(materialize, bool)
        try:
            return cls(
                op=op,
                version=version,
                inputs=inputs,
                params=cast(Mapping[str, Value], params),
                materialize=materialize,
            )
        except (TypeError, ValueError) as exc:
            _fail(path, str(exc))

    def to_data(self) -> dict[str, Any]:
        result: dict[str, Any] = {"op": self.op, "version": self.version}
        if self.inputs:
            result["inputs"] = {
                name: input_value.to_data()
                for name, input_value in self.inputs.items()
            }
        if self.params:
            result["params"] = _thaw_value(cast(Value, self.params))
        if self.materialize is not None:
            result["materialize"] = self.materialize
        return result


@dataclass(frozen=True, slots=True)
class Strategy:
    name: str
    nodes: Mapping[str, Node]
    outputs: Mapping[str, Reference]
    description: str | None = None
    metadata: Mapping[str, Value] = field(default_factory=dict)
    format: str = STRATEGY_FORMAT
    version: str = STRATEGY_VERSION

    def __post_init__(self) -> None:
        if self.format != STRATEGY_FORMAT:
            raise ValueError(
                f"Strategy.format must be {STRATEGY_FORMAT!r}, got {self.format!r}"
            )
        _semver(self.version, "strategy.version")
        if self.version != STRATEGY_VERSION:
            raise ValueError(
                f"Strategy.version must be {STRATEGY_VERSION!r}, got {self.version!r}"
            )
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Strategy.name must be a non-empty string")
        if self.description is not None and not isinstance(self.description, str):
            raise TypeError("Strategy.description must be a string or None")

        nodes = _mapping(self.nodes, "nodes")
        normalized_nodes: dict[str, Node] = {}
        for node_id, node in nodes.items():
            _identifier(node_id, f"nodes.{node_id}")
            if not isinstance(node, Node):
                raise TypeError(f"Strategy node '{node_id}' must be a Node")
            normalized_nodes[node_id] = node
        if not normalized_nodes:
            raise ValueError("Strategy.nodes must not be empty")
        object.__setattr__(self, "nodes", MappingProxyType(normalized_nodes))

        outputs = _mapping(self.outputs, "outputs")
        normalized_outputs: dict[str, Reference] = {}
        for name, reference in sorted(outputs.items()):
            _identifier(name, f"outputs.{name}")
            if not isinstance(reference, Reference):
                raise TypeError(f"Strategy output '{name}' must be a Reference")
            normalized_outputs[name] = reference
        if not normalized_outputs:
            raise ValueError("Strategy.outputs must not be empty")
        object.__setattr__(self, "outputs", MappingProxyType(normalized_outputs))

        metadata = _freeze_value(self.metadata, "metadata")
        if not isinstance(metadata, Mapping):
            raise TypeError("Strategy.metadata must be a mapping")
        object.__setattr__(self, "metadata", metadata)
        self._validate_graph()

    @classmethod
    def from_data(cls, value: object) -> Strategy:
        data = _mapping(value, "$")
        _check_keys(
            data,
            "$",
            allowed={
                "format",
                "version",
                "name",
                "description",
                "metadata",
                "nodes",
                "outputs",
            },
            required={"format", "version", "name", "nodes", "outputs"},
        )
        if data["format"] != STRATEGY_FORMAT:
            _fail(
                "$.format",
                f"expected {STRATEGY_FORMAT!r}, got {data['format']!r}",
            )
        version = _semver(data["version"], "$.version")
        if version != STRATEGY_VERSION:
            _fail(
                "$.version",
                f"expected supported version {STRATEGY_VERSION!r}, got {version!r}",
            )
        name = data["name"]
        if not isinstance(name, str) or not name.strip():
            _fail("$.name", "expected a non-empty string")
        description = data.get("description")
        if "description" in data and not isinstance(description, str):
            _fail("$.description", "expected a string")
        assert description is None or isinstance(description, str)
        raw_nodes = _mapping(data["nodes"], "$.nodes")
        nodes = {
            _identifier(node_id, f"$.nodes.{node_id}"): Node.from_data(
                node_value, f"$.nodes.{node_id}"
            )
            for node_id, node_value in raw_nodes.items()
        }
        raw_outputs = _mapping(data["outputs"], "$.outputs")
        outputs = {
            _identifier(output_name, f"$.outputs.{output_name}"): Reference.parse(
                reference, f"$.outputs.{output_name}"
            )
            for output_name, reference in raw_outputs.items()
        }
        metadata = _mapping(data.get("metadata", {}), "$.metadata")
        try:
            return cls(
                name=name,
                nodes=nodes,
                outputs=outputs,
                description=description,
                metadata=cast(Mapping[str, Value], metadata),
                version=version,
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, StrategyValidationError):
                raise
            _fail("$", str(exc))

    def _validate_graph(self) -> None:
        dependencies: dict[str, set[str]] = {node_id: set() for node_id in self.nodes}
        for node_id, node in self.nodes.items():
            for input_name, input_value in node.inputs.items():
                parent = input_value.source.node
                if parent not in self.nodes:
                    _fail(
                        f"$.nodes.{node_id}.inputs.{input_name}",
                        f"references unknown node {parent!r}",
                    )
                dependencies[node_id].add(parent)
        for output_name, reference in self.outputs.items():
            if reference.node not in self.nodes:
                _fail(
                    f"$.outputs.{output_name}",
                    f"references unknown node {reference.node!r}",
                )

        visiting: list[str] = []
        visited: set[str] = set()

        def visit(node_id: str) -> None:
            if node_id in visiting:
                start = visiting.index(node_id)
                cycle = visiting[start:] + [node_id]
                _fail("$.nodes", f"cycle detected: {' -> '.join(cycle)}")
            if node_id in visited:
                return
            visiting.append(node_id)
            for parent in sorted(dependencies[node_id]):
                visit(parent)
            visiting.pop()
            visited.add(node_id)

        for node_id in sorted(self.nodes):
            visit(node_id)

        used: set[str] = set()

        def mark_used(node_id: str) -> None:
            if node_id in used:
                return
            used.add(node_id)
            for parent in dependencies[node_id]:
                mark_used(parent)

        for reference in self.outputs.values():
            mark_used(reference.node)
        unused = sorted(self.nodes.keys() - used)
        if unused:
            _fail("$.nodes", f"nodes are not reachable from outputs: {unused}")

    @property
    def topological_order(self) -> tuple[str, ...]:
        order: list[str] = []
        visited: set[str] = set()

        def visit(node_id: str) -> None:
            if node_id in visited:
                return
            node = self.nodes[node_id]
            for input_value in node.inputs.values():
                visit(input_value.source.node)
            visited.add(node_id)
            order.append(node_id)

        for reference in self.outputs.values():
            visit(reference.node)
        return tuple(order)

    def to_data(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "format": self.format,
            "version": self.version,
            "name": self.name,
        }
        if self.description is not None:
            result["description"] = self.description
        if self.metadata:
            result["metadata"] = _thaw_value(cast(Value, self.metadata))
        result["nodes"] = {
            node_id: self.nodes[node_id].to_data()
            for node_id in self.topological_order
        }
        result["outputs"] = {
            name: str(reference) for name, reference in self.outputs.items()
        }
        return result

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_data(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


__all__ = [
    "Input",
    "Node",
    "Reference",
    "STRATEGY_FORMAT",
    "STRATEGY_VERSION",
    "Strategy",
    "StrategyValidationError",
    "Value",
]
