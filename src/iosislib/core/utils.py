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
from typing import Any, TypeAlias, cast

import numpy as np
import numpy.typing as npt
import polars as pl
import pyarrow as pa
import torch
from torch.utils.dlpack import from_dlpack as torch_from_dlpack


AsofTolerance = str | int | float | timedelta | None
PolarsDataType: TypeAlias = pl.DataType | type[pl.DataType]


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


def _prepare_series_for_bridge(series: pl.Series) -> pl.Series:
    null_count = _series_null_count(series)
    if null_count:
        raise ValueError(
            f"Cannot bridge column '{series.name}' with {null_count} null "
            "value(s); input null policy must resolve them before conversion"
        )
    return series


def series_to_numpy(
    series: pl.Series,
    shape: tuple[int, ...] | None = None,
    *,
    allow_copy: bool = False,
) -> npt.NDArray[Any]:
    """Expose a non-null Polars Series as a NumPy array."""
    series = _prepare_series_for_bridge(series)
    array = series.to_numpy(allow_copy=allow_copy)
    if shape:
        reshaped = array.reshape((len(series), *_normalize_shape(shape)))
        if not allow_copy and not np.shares_memory(array, reshaped):
            raise ValueError(
                f"Cannot reshape Series '{series.name}' to {shape} without copying"
            )
        return reshaped
    return array


def numpy_to_series(
    name: str,
    array: npt.NDArray[Any],
    *,
    dtype: PolarsDataType = pl.Float64,
    shape: tuple[int, ...] | None = None,
    allow_copy: bool = False,
) -> pl.Series:
    """Expose a NumPy buffer as a Polars Series through Arrow when possible."""
    if not isinstance(array, np.ndarray):
        if not allow_copy:
            raise TypeError(
                f"Cannot create Series '{name}' without copying from "
                f"{type(array).__name__}; expected a NumPy ndarray"
            )
        array = np.asarray(array)

    if shape:
        shape = _normalize_shape(shape)
        width = _flat_size(shape)
        reshaped = array.reshape((len(array), width))
        if not allow_copy and not np.shares_memory(array, reshaped):
            raise ValueError(
                f"Cannot reshape output for Series '{name}' without copying"
            )
        array = reshaped
        if not array.flags.c_contiguous:
            if not allow_copy:
                raise ValueError(
                    f"Cannot create shaped Series '{name}' without copying: "
                    "NumPy array is not C-contiguous"
                )
            array = np.ascontiguousarray(array)

        flat = array.ravel()
        arrow_values = pa.array(flat)
        if not allow_copy:
            _validate_numpy_arrow_alias(name, flat, arrow_values)
        arrow_array = pa.FixedSizeListArray.from_arrays(arrow_values, width)
        series = cast(pl.Series, pl.from_arrow(arrow_array)).alias(name)
        expected_dtype = pl.Array(dtype, width)
        if series.dtype != expected_dtype:
            if not allow_copy:
                raise TypeError(
                    f"Cannot create Series '{name}' as {expected_dtype} without "
                    f"copying or casting; inferred {series.dtype}"
                )
            series = series.cast(expected_dtype, strict=False)
        if not allow_copy:
            _validate_numpy_arrow_alias(name, flat, series.to_arrow().values)
        return series

    if not array.flags.c_contiguous:
        if not allow_copy:
            raise ValueError(
                f"Cannot create Series '{name}' without copying: "
                "NumPy array is not C-contiguous"
            )
        array = np.ascontiguousarray(array)
    arrow_array = pa.array(array)
    if not allow_copy:
        _validate_numpy_arrow_alias(name, array, arrow_array)
    series = cast(pl.Series, pl.from_arrow(arrow_array)).alias(name)
    if series.dtype != dtype:
        if not allow_copy:
            raise TypeError(
                f"Cannot create Series '{name}' as {dtype} without copying or "
                f"casting; inferred {series.dtype}"
            )
        series = series.cast(dtype, strict=False)
    if not allow_copy:
        _validate_numpy_arrow_alias(name, array, series.to_arrow())
    return series


def series_to_torch(
    series: pl.Series,
    shape: tuple[int, ...] | None = None,
    *,
    allow_copy: bool = False,
) -> torch.Tensor:
    """Expose a Series buffer to Torch through Arrow DLPack when possible."""
    series = _prepare_series_for_bridge(series)
    if series.n_chunks() > 1:
        if not allow_copy:
            raise ValueError(
                f"Cannot expose multi-chunk Series '{series.name}' to Torch "
                "without rechunking"
            )
        series = series.rechunk()

    arrow_array = series.to_arrow()
    arrow_values = (
        arrow_array.flatten()
        if pa.types.is_fixed_size_list(arrow_array.type)
        else arrow_array
    )
    try:
        tensor = torch_from_dlpack(arrow_values)
    except (BufferError, RuntimeError, TypeError):
        if not allow_copy:
            raise ValueError(
                f"Cannot expose Series '{series.name}' to Torch through DLPack "
                "without copying"
            ) from None
        tensor = torch.from_numpy(series.to_numpy(allow_copy=True))

    if shape:
        normalized_shape = _normalize_shape(shape)
        expected_values = len(series) * _flat_size(normalized_shape)
        if tensor.numel() != expected_values:
            raise ValueError(
                f"Series '{series.name}' contains {tensor.numel()} values; "
                f"shape {normalized_shape} requires {expected_values}"
            )
        tensor = tensor.reshape((len(series), *normalized_shape))
    return tensor


def torch_to_series(
    name: str,
    tensor: torch.Tensor,
    *,
    dtype: PolarsDataType = pl.Float64,
    shape: tuple[int, ...] | None = None,
    allow_copy: bool = False,
) -> pl.Series:
    """Expose CPU Torch storage as a Polars Series through Arrow."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("torch_to_series requires a torch.Tensor")

    tensor = tensor.detach()
    if tensor.device.type != "cpu":
        if not allow_copy:
            raise ValueError(
                f"Cannot create Series '{name}' from a {tensor.device.type} "
                "tensor without copying it to CPU"
            )
        tensor = tensor.cpu()
    if not tensor.is_contiguous():
        if not allow_copy:
            raise ValueError(
                f"Cannot create Series '{name}' from a non-contiguous tensor "
                "without copying"
            )
        tensor = tensor.contiguous()

    try:
        array = tensor.numpy()
    except RuntimeError:
        if not allow_copy:
            raise ValueError(
                f"Cannot expose tensor storage for Series '{name}' without copying"
            ) from None
        array = tensor.resolve_conj().resolve_neg().numpy()
    return numpy_to_series(
        name,
        array,
        dtype=dtype,
        shape=shape,
        allow_copy=allow_copy,
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
