from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any

import polars as pl

from src.core.tsfn import (
    ColumnSignature,
    NullHandler,
    NullPolicy,
    TSFN,
    _column_signature_map,
    _column_signatures,
    _format_frame_signature,
    _normalize_null_handler,
)
from src.core.utils import (
    AsofTolerance,
    _canonical_json,
    _format_tolerance,
    _serialize_value,
)


def _format_null_handler(handler: NullHandler) -> dict[str, str]:
    if handler.policy is not None:
        return {"kind": "policy", "value": handler.policy.value}

    assert handler.function is not None
    return {
        "kind": "function",
        "module": handler.function.__module__,
        "qualname": handler.function.__qualname__,
    }


def _format_null_handlers(
    handlers: Mapping[str, NullHandler],
) -> dict[str, dict[str, str]]:
    return {
        name: _format_null_handler(handler)
        for name, handler in sorted(handlers.items())
    }


def _format_null_fill_values(values: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: _serialize_value(value)
        for name, value in sorted(values.items())
    }


def _format_function_identity(function: TSFN) -> dict[str, Any]:
    input_signature, output_signature = function.signature
    return {
        "module": function.__class__.__module__,
        "qualname": function.__class__.__qualname__,
        "version": function.version,
        "input_signature": _format_frame_signature(input_signature),
        "output_signature": _format_frame_signature(output_signature),
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
    ):
        if materialize is not None and not isinstance(materialize, bool):
            raise TypeError("Node materialize must be a boolean or None")
        self.name = name
        self.function_cls = function_cls
        self.function = function_cls(parameters or {})
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
        self.null_fill_values = MappingProxyType(dict(null_fill_values or {}))
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
        self.function.signature = self.function.resolve_signature(bound_input_columns)
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
        serialized_bindings = {
            input_name: {
                "parent_id": parent.ID,
                "parent_output": parent_column,
                "tolerance": _format_tolerance(self.tolerances.get(input_name)),
            }
            for input_name, (parent, parent_column) in sorted(self.bindings.items())
        }

        node_definition = {
            "bindings": serialized_bindings,
            "function": _format_function_identity(self.function),
            "null_fill_values": _format_null_fill_values(self.null_fill_values),
            "null_handlers": _format_null_handlers(self.null_handlers),
            "parameters": self.parameters.to_dict(),
            "outputs": sorted(
                (name, str(dtype)) for name, dtype in self.outputs.items()
            ),
        }

        serialized_data = _canonical_json(node_definition)
        return hashlib.sha256(serialized_data.encode("utf-8")).hexdigest()


__all__ = ["Node"]
