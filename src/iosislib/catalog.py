"""Dump the built-in TSFN operation registry as a JSON-serializable catalog."""

from __future__ import annotations

import math
import typing
from dataclasses import MISSING, Field, fields
from enum import Enum
from typing import Any

from iosislib.core.tsfn import TSFN, _format_frame_signature
from iosislib.strategy import OperationRegistry, builtin_registry

_PROBE_PARAMS: dict[str, dict[str, object]] = {
    "source.polymarket_price_history": {"token_id": "PLACEHOLDER"},
    "source.y_finance_ohlcv": {
        "ticker": "PLACEHOLDER",
        "date_range": ["2026-01-01", "2026-01-02"],
    },
    "transform.feature_packer": {"input_columns": ["a", "b"]},
}


def dump_tsfn_catalog(registry: OperationRegistry | None = None) -> dict[str, Any]:
    """Serialize every registered TSFN and its attributes to plain JSON data.

    Parameter metadata comes from each TSFN's ``CONFIG_CLS`` dataclass fields.
    Input/output signatures come from ``type_signature()`` on a default (or
    minimally probed) instance; TSFNs whose signatures depend on files or
    live objects resolve to ``signature: null``.
    """
    registry = registry if registry is not None else builtin_registry()
    entries = [
        _entry(op, version, function_cls)
        for (op, version), function_cls in registry.operations.items()
    ]
    entries.sort(key=lambda entry: (entry["op"], entry["version"]))
    return _json_safe({
        "format": "iosis.tsfn-catalog",
        "version": "0.1.0",
        "tsfns": entries,
    })


def _entry(op: str, version: str, function_cls: type[TSFN[Any]]) -> dict[str, Any]:
    return {
        "op": op,
        "version": version,
        "category": op.split(".", 1)[0],
        "class": function_cls.__name__,
        "module": function_cls.__module__,
        "description": _description(function_cls),
        "configClass": function_cls.CONFIG_CLS.__name__,
        "requiresMaterialization": bool(function_cls.REQUIRES_MATERIALIZATION),
        "defaultNullPolicy": _enum_value(function_cls.DEFAULT_NULL_POLICY),
        "lookahead": bool(function_cls.LOOKAHEAD),
        "allowLookaheadInputs": sorted(function_cls.ALLOW_LOOKAHEAD_INPUTS),
        "parameters": [_parameter(field_) for field_ in fields(function_cls.CONFIG_CLS)],
        "signature": _signature(op, function_cls),
    }


def _parameter(field_: Field[Any]) -> dict[str, Any]:
    required = field_.default is MISSING and field_.default_factory is MISSING
    return {
        "name": field_.name,
        "type": _type_str(field_.type),
        "required": required,
        "default": None if required else _json_safe(field_.default),
    }


def _signature(op: str, function_cls: type[TSFN[Any]]) -> dict[str, Any] | None:
    try:
        instance = function_cls(_PROBE_PARAMS.get(op, {}))
        input_signature, output_signature = instance.type_signature()
    except Exception:
        return None
    return {
        "input": _format_frame_signature(input_signature),
        "output": _format_frame_signature(output_signature),
    }


def _description(function_cls: type[TSFN[Any]]) -> str:
    doc = function_cls.__doc__
    if not doc:
        return ""
    lines = [line.strip() for line in doc.splitlines() if line.strip()]
    return lines[0] if lines else ""


def _enum_value(value: Any) -> str:
    return value.value if isinstance(value, Enum) else str(value)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Enum):
        return _json_safe(value.value)
    text = str(value)
    return text if text and not text.startswith("<") else None


def _type_str(type_: Any) -> str:
    if type_ is None or type_ is type(None):
        return "None"
    origin = typing.get_origin(type_)
    if origin is None:
        if isinstance(type_, type):
            return type_.__name__
        return str(type_)
    args = [_type_str(argument) for argument in typing.get_args(type_)]
    if origin is typing.Union:
        return " | ".join(args)
    origin_name = getattr(origin, "__name__", str(origin))
    return f"{origin_name}[{', '.join(args)}]"


__all__ = ["dump_tsfn_catalog"]
