from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any

import polars as pl

from iosislib.core.tsfn import (
    ColumnSignature,
    NullHandler,
    NullPolicy,
    TSFN,
    _column_signature_map,
    _column_signatures,
    _normalize_null_handler,
)
from iosislib.core.utils import (
    AsofTolerance,
    _canonical_identity_json,
    _normalize_identity_value,
)


def _format_null_handler(
    handler: NullHandler,
    version: str | None,
) -> dict[str, str]:
    if handler.policy is not None:
        return {"kind": "policy", "value": handler.policy.value}

    assert handler.function is not None
    assert version is not None
    return {
        "kind": "function",
        "module": handler.function.__module__,
        "qualname": handler.function.__qualname__,
        "version": version,
    }


def _format_null_handlers(
    handlers: Mapping[str, NullHandler],
    versions: Mapping[str, str],
) -> dict[str, dict[str, str]]:
    return {
        name: _format_null_handler(handler, versions.get(name))
        for name, handler in sorted(handlers.items())
    }


def _declared_null_handler_version(
    input_name: str,
    handler: NullHandler,
    configured_versions: Mapping[str, str],
) -> str | None:
    if handler.function is None:
        return None

    configured = configured_versions.get(input_name)
    wrapped = handler.version
    attributed = getattr(handler.function, "__iosis_version__", None)
    declared = [
        (source, value)
        for source, value in (
            ("NullHandler", wrapped),
            ("Node.null_handler_versions", configured),
            ("function.__iosis_version__", attributed),
        )
        if value is not None
    ]
    versions = {value for _, value in declared}
    if len(versions) > 1:
        raise ValueError(
            f"Conflicting custom null handler versions for input '{input_name}'"
        )
    version = declared[0][1] if declared else None
    if version is None:
        raise ValueError(
            f"Custom null handler for input '{input_name}' must declare a behavior "
            "version through NullHandler.from_function(..., version=...), "
            "Node.null_handler_versions, or function.__iosis_version__"
        )
    if not isinstance(version, str) or not version.strip():
        raise ValueError(
            f"Custom null handler version for input '{input_name}' must be "
            "a non-empty string"
        )
    return version


def _normalize_function_state(function: TSFN) -> None:
    for name, value in tuple(vars(function).items()):
        object.__setattr__(function, name, _normalize_identity_value(value))


def _format_function_identity(
    function_cls: type[TSFN],
    function: TSFN,
) -> dict[str, Any]:
    input_signature, output_signature = function.signature
    core_state = {
        "version",
        "parameters",
        "signature",
        "_input_null_handlers",
        "_input_null_fill_values",
        "_node_definition_frozen",
    }
    return {
        "module": function_cls.__module__,
        "qualname": function_cls.__qualname__,
        "version": function.version,
        "input_signature": input_signature,
        "output_signature": output_signature,
        "state": MappingProxyType(
            {
                name: value
                for name, value in sorted(vars(function).items())
                if name not in core_state
            }
        ),
    }


class Node:
    def __setattr__(self, name: str, value: Any) -> None:
        if object.__getattribute__(self, "__dict__").get("_frozen", False):
            raise AttributeError("Node instances are immutable")
        object.__setattr__(self, name, value)

    def __init__(
        self,
        function_cls: type[TSFN],
        bindings: dict[str, tuple[Node, str]] | None = None,
        parameters: dict[str, Any] | None = None,
        name: str | None = None,
        materialize: bool | None = None,
        tolerances: dict[str, AsofTolerance] | None = None,
        null_handlers: dict[
            str,
            NullHandler | NullPolicy | str | Callable[[pl.Series], pl.Series],
        ]
        | None = None,
        null_policies: dict[str, NullPolicy | str] | None = None,
        null_fill_values: dict[str, Any] | None = None,
        null_handler_versions: dict[str, str] | None = None,
    ):
        if materialize is not None and not isinstance(materialize, bool):
            raise TypeError("Node materialize must be a boolean or None")
        self.name = name
        self.function_cls = function_cls
        normalized_parameters = _normalize_identity_value(parameters or {})
        self.function = function_cls(normalized_parameters)
        _normalize_function_state(self.function)
        self.parameters = self.function.parameters
        self.materialize = (
            self.function.requires_materialization
            if materialize is None
            else materialize
        )

        self.bindings = MappingProxyType(dict(bindings or {}))
        self.tolerances = MappingProxyType(dict(tolerances or {}))
        configured_handlers = dict(null_handlers or {})
        policy_handlers = dict(null_policies or {})
        duplicated_handlers = set(configured_handlers) & set(policy_handlers)
        if duplicated_handlers:
            raise ValueError(
                "Inputs cannot be configured in both null_handlers and null_policies: "
                f"{sorted(duplicated_handlers)}"
            )
        configured_handlers.update(policy_handlers)
        self.null_handlers = MappingProxyType(
            {
                input_name: _normalize_null_handler(handler)
                for input_name, handler in configured_handlers.items()
            }
        )
        self.null_policies = self.null_handlers
        normalized_fill_values = _normalize_identity_value(null_fill_values or {})
        self.null_fill_values = normalized_fill_values
        configured_versions = dict(null_handler_versions or {})
        unexpected_versions = set(configured_versions) - set(self.null_handlers)
        if unexpected_versions:
            raise ValueError(
                "Unexpected custom null handler versions for unconfigured inputs: "
                f"{sorted(unexpected_versions)}"
            )
        self.null_handler_versions = MappingProxyType(
            {
                input_name: version
                for input_name, handler in self.null_handlers.items()
                if (
                    version := _declared_null_handler_version(
                        input_name,
                        handler,
                        configured_versions,
                    )
                )
                is not None
            }
        )
        policy_versions = set(configured_versions) - set(self.null_handler_versions)
        if policy_versions:
            raise ValueError(
                "Custom null handler versions require function handlers: "
                f"{sorted(policy_versions)}"
            )
        unexpected_tolerances = set(self.tolerances) - set(self.bindings)
        if unexpected_tolerances:
            raise ValueError(
                f"Unexpected tolerances for unbound inputs: {sorted(unexpected_tolerances)}"
            )

        self.inputs = tuple(
            dict.fromkeys(parent for parent, _ in self.bindings.values())
        )

        bound_input_columns: dict[str, ColumnSignature] = {}
        for input_name, (parent, parent_column) in self.bindings.items():
            parent_outputs = _column_signature_map(parent.function.signature[1])
            if parent_column in parent_outputs:
                bound_input_columns[input_name] = parent_outputs[parent_column]
        self.function.signature = _normalize_identity_value(
            self.function.resolve_signature(bound_input_columns)
        )
        self.function._configure_input_nulls(
            self.null_handlers,
            self.null_fill_values,
        )

        self.outputs = MappingProxyType(
            {
                column.name: column.physical_dtype
                for column in _column_signatures(self.function.signature[1])
            }
        )

        self.function._freeze_definition()
        self.definition = self._normalized_definition()
        self.ID = self._generate_persistent_id()
        self._frozen = True

    def __getattr__(self, name: str) -> tuple[Node, str]:
        """Provide syntactic sugar for output references such as ``node.value``."""
        attrs = object.__getattribute__(self, "__dict__")
        outputs = attrs.get("outputs")
        if isinstance(outputs, Mapping) and name in outputs:
            return (self, name)
        function_cls = attrs.get("function_cls")
        function_name = (
            function_cls.__name__
            if isinstance(function_cls, type)
            else "<uninitialized>"
        )
        raise AttributeError(
            f"'{self.__class__.__name__}' or its configured TSFN '{function_name}' "
            f"does not expose output: '{name}'"
        )

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Node) and self.ID == other.ID

    def __hash__(self) -> int:
        return hash(self.ID)

    def __repr__(self) -> str:
        attrs = object.__getattribute__(self, "__dict__")
        node_id = attrs.get("ID", "<uninitialized>")
        function_cls = attrs.get("function_cls")
        function = attrs.get("function")
        function_name = (
            function_cls.__name__
            if isinstance(function_cls, type)
            else "<uninitialized>"
        )
        version = getattr(function, "version", "<uninitialized>")
        return (
            f"Node(name={attrs.get('name')!r}, id={str(node_id)[:8]!r}, "
            f"fn={function_name}@{version}, "
            f"materialize={attrs.get('materialize', '<uninitialized>')!r})"
        )

    def _generate_persistent_id(self) -> str:
        serialized_data = _canonical_identity_json(self.definition)
        return hashlib.sha256(serialized_data.encode("utf-8")).hexdigest()

    def _normalized_definition(self) -> Mapping[str, Any]:
        serialized_bindings = {
            input_name: {
                "parent_id": parent.ID,
                "parent_output": parent_column,
                "tolerance": self.tolerances.get(input_name),
            }
            for input_name, (parent, parent_column) in sorted(self.bindings.items())
        }

        node_definition = {
            "bindings": serialized_bindings,
            "function": _format_function_identity(self.function_cls, self.function),
            "null_fill_values": self.null_fill_values,
            "null_handlers": _format_null_handlers(
                self.null_handlers,
                self.null_handler_versions,
            ),
            "parameters": self.parameters,
            "outputs": tuple(sorted(self.outputs.items())),
        }
        return _normalize_identity_value(node_definition)


__all__ = ["Node"]
