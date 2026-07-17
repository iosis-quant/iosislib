from __future__ import annotations

import json
from copy import copy
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from datetime import date, datetime, time, timedelta
from enum import Enum
from math import isfinite, prod
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import polars as pl
import pyarrow as pa


AsofTolerance = str | int | float | timedelta | None


def _normalize_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    if not isinstance(shape, tuple):
        raise TypeError("column shape must be a tuple of positive integers")
    for dimension in shape:
        if (
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension < 1
        ):
            raise ValueError("column shape dimensions must be positive integers")
    return shape


def _flat_size(shape: tuple[int, ...]) -> int:
    return prod(shape) if shape else 1


def _series_null_count(series: pl.Series) -> int:
    count = int(series.null_count())
    if _is_array_instance(series.dtype):
        inner_count = series.to_frame().select(
            pl.col(series.name)
            .arr.eval(pl.element().is_null())
            .arr.sum()
            .sum()
            .alias("inner_nulls")
        )["inner_nulls"][0]
        count += int(inner_count or 0)
    return count


def _validate_numpy_arrow_alias(
    name: str,
    array: np.ndarray,
    arrow_array: pa.Array,
) -> None:
    if array.size == 0:
        return
    data_buffer = arrow_array.buffers()[1]
    numpy_address = array.__array_interface__["data"][0]
    if data_buffer is None or data_buffer.address != numpy_address:
        raise ValueError(
            f"Conversion for Series '{name}' required a NumPy/Arrow buffer copy"
        )


def _dtype_matches(actual: pl.DataType, expected: pl.DataType) -> bool:
    actual_is_class = _is_dtype_class(actual)
    expected_is_class = _is_dtype_class(expected)

    if actual_is_class or expected_is_class:
        if actual_is_class and expected_is_class:
            return actual is expected

        if expected_is_class:
            return _matches_default_dtype_instance(actual, expected)

        return _matches_default_dtype_instance(expected, actual)

    if _is_list_instance(actual) and _is_list_instance(expected):
        return _dtype_matches(_list_inner_dtype(actual), _list_inner_dtype(expected))

    return actual == expected


def _is_dtype_class(dtype: pl.DataType) -> bool:
    return isinstance(dtype, type) and issubclass(dtype, pl.DataType)


def _is_list_instance(dtype: pl.DataType) -> bool:
    return not _is_dtype_class(dtype) and isinstance(dtype, pl.List)


def _is_array_instance(dtype: pl.DataType) -> bool:
    return not _is_dtype_class(dtype) and isinstance(dtype, pl.Array)


def _list_inner_dtype(dtype: pl.DataType) -> pl.DataType:
    return dtype.inner


def _datetime_dtype_without_timezone(dtype: pl.DataType) -> pl.DataType:
    if not _is_dtype_class(dtype) and isinstance(dtype, pl.Datetime):
        return pl.Datetime(time_unit=dtype.time_unit)
    return dtype


def _matches_default_dtype_instance(
    dtype_instance: pl.DataType,
    dtype_cls: type[pl.DataType],
) -> bool:
    if dtype_cls is pl.List or _is_list_instance(dtype_instance):
        return False

    try:
        default_instance = dtype_cls()
    except TypeError:
        return False

    return (
        type(dtype_instance) is type(default_instance)
        and dtype_instance == default_instance
    )


def _format_tolerance(tolerance: AsofTolerance) -> dict[str, str] | None:
    if tolerance is None:
        return None
    return {
        "type": type(tolerance).__name__,
        "value": str(tolerance),
    }


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _serialize_value(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _canonical_identity_json(value: Any) -> str:
    """Return the injective canonical encoding used by persistent identities."""
    return json.dumps(
        _serialize_identity_value(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _qualified_type_name(value: Any) -> str:
    value_type = value if isinstance(value, type) else type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _normalize_identity_value(value: Any) -> Any:
    """Copy supported identity input into recursively immutable canonical state."""
    if isinstance(value, Enum):
        return value

    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("Non-finite floats are not valid identity values")
        return value

    if isinstance(value, np.generic):
        return _normalize_identity_value(value.item())

    if _is_dtype_class(value) or isinstance(value, pl.DataType):
        return value

    if isinstance(value, (datetime, date, time, timedelta, Path, bytes)):
        return value

    if is_dataclass(value) and not isinstance(value, type):
        params = getattr(type(value), "__dataclass_params__", None)
        if params is None or not params.frozen:
            raise TypeError(
                f"Dataclass identity value {_qualified_type_name(value)} must be frozen"
            )
        field_names = {item.name for item in fields(value)}
        undeclared_state = set(getattr(value, "__dict__", {})) - field_names
        if undeclared_state:
            raise TypeError(
                f"Dataclass identity value {_qualified_type_name(value)} has "
                f"undeclared state: {sorted(undeclared_state)}"
            )
        normalized = copy(value)
        for item in fields(value):
            object.__setattr__(
                normalized,
                item.name,
                _normalize_identity_value(getattr(value, item.name)),
            )
        return normalized

    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                _normalize_identity_value(key): _normalize_identity_value(item)
                for key, item in value.items()
            }
        )

    if isinstance(value, (list, tuple)):
        return tuple(_normalize_identity_value(item) for item in value)

    if isinstance(value, (set, frozenset)):
        return frozenset(_normalize_identity_value(item) for item in value)

    raise TypeError(f"Type {type(value)} is not a supported identity value")


def _serialize_identity_value(value: Any) -> Any:
    """Return a type-tagged JSON value with no supported cross-category aliases."""
    if isinstance(value, Enum):
        return {
            "type": "enum",
            "class": _qualified_type_name(value),
            "value": _serialize_identity_value(value.value),
        }

    if isinstance(value, np.generic):
        return {
            "type": "numpy_scalar",
            "dtype": value.dtype.str,
            "value": _serialize_identity_value(value.item()),
        }

    if value is None:
        return {"type": "none"}

    if isinstance(value, bool):
        return {"type": "bool", "value": value}

    if isinstance(value, str):
        return {"type": "str", "value": value}

    if isinstance(value, int):
        return {"type": "int", "value": str(value)}

    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("Non-finite floats are not valid identity values")
        return {"type": "float", "value": value.hex()}

    if _is_dtype_class(value):
        return {
            "type": "polars_dtype_class",
            "class": _qualified_type_name(value),
        }

    if isinstance(value, pl.DataType):
        return {
            "type": "polars_dtype_instance",
            "class": _qualified_type_name(value),
            "value": str(value),
        }

    if isinstance(value, datetime):
        return {"type": "datetime", "value": value.isoformat()}

    if isinstance(value, date):
        return {"type": "date", "value": value.isoformat()}

    if isinstance(value, time):
        return {"type": "time", "value": value.isoformat()}

    if isinstance(value, timedelta):
        return {
            "type": "timedelta",
            "days": value.days,
            "seconds": value.seconds,
            "microseconds": value.microseconds,
        }

    if isinstance(value, Path):
        return {"type": "path", "value": str(value)}

    if isinstance(value, bytes):
        return {"type": "bytes", "value": value.hex()}

    if getattr(type(value), "_SERIALIZE_WITH_TO_DICT", False):
        return {
            "type": "to_dict",
            "class": _qualified_type_name(value),
            "value": _serialize_identity_value(value.to_dict()),
        }

    if is_dataclass(value) and not isinstance(value, type):
        return {
            "type": "dataclass",
            "class": _qualified_type_name(value),
            "fields": [
                [item.name, _serialize_identity_value(getattr(value, item.name))]
                for item in fields(value)
            ],
        }

    if isinstance(value, Mapping):
        serialized_items = [
            (
                _serialize_identity_value(key),
                _serialize_identity_value(item),
            )
            for key, item in value.items()
        ]
        serialized_items.sort(key=lambda item: _identity_sort_key(item[0]))
        return {
            "type": "mapping",
            "items": [[key, item] for key, item in serialized_items],
        }

    if isinstance(value, list):
        return {
            "type": "list",
            "items": [_serialize_identity_value(item) for item in value],
        }

    if isinstance(value, tuple):
        return {
            "type": "tuple",
            "items": [_serialize_identity_value(item) for item in value],
        }

    if isinstance(value, (set, frozenset)):
        serialized_items = [_serialize_identity_value(item) for item in value]
        serialized_items.sort(key=_identity_sort_key)
        return {"type": "set", "items": serialized_items}

    raise TypeError(f"Type {type(value)} is not serializable for identity")


def _identity_sort_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _serialize_value(value: Any) -> Any:
    """Return a deterministic, JSON-compatible representation of a value."""
    if isinstance(value, Enum):
        return _serialize_value(value.value)

    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("Non-finite floats are not serializable")
        return value

    if isinstance(value, np.generic):
        return _serialize_value(value.item())

    if _is_dtype_class(value) or isinstance(value, pl.DataType):
        return str(value)

    if isinstance(value, datetime):
        return {"__type__": "datetime", "value": value.isoformat()}

    if isinstance(value, date):
        return {"__type__": "date", "value": value.isoformat()}

    if isinstance(value, time):
        return {"__type__": "time", "value": value.isoformat()}

    if isinstance(value, timedelta):
        return {
            "__type__": "timedelta",
            "days": value.days,
            "seconds": value.seconds,
            "microseconds": value.microseconds,
        }

    if isinstance(value, Path):
        return {"__type__": "path", "value": str(value)}

    if isinstance(value, bytes):
        return {"__type__": "bytes", "value": value.hex()}

    if getattr(type(value), "_SERIALIZE_WITH_TO_DICT", False):
        return value.to_dict()

    if is_dataclass(value) and not isinstance(value, type):
        return {
            "__type__": _qualified_type_name(value),
            "fields": {
                item.name: _serialize_value(getattr(value, item.name))
                for item in fields(value)
            },
        }

    if isinstance(value, Mapping):
        if all(isinstance(key, str) for key in value):
            return {
                key: _serialize_value(item)
                for key, item in sorted(value.items())
            }

        serialized_items = [
            (_serialize_value(key), _serialize_value(item))
            for key, item in value.items()
        ]
        serialized_items.sort(key=lambda item: _canonical_json(item[0]))
        return {
            "__type__": "mapping",
            "items": [[key, item] for key, item in serialized_items],
        }

    if isinstance(value, (list, tuple)):
        return [_serialize_value(item) for item in value]

    if isinstance(value, (set, frozenset)):
        serialized_items = [_serialize_value(item) for item in value]
        serialized_items.sort(key=_canonical_json)
        return {"__type__": "set", "items": serialized_items}

    raise TypeError(f"Type {type(value)} not serializable")


__all__ = ["AsofTolerance"]
