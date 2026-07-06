from __future__ import annotations

import abc
import hashlib
import json
from datetime import timedelta
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from enum import Enum
from math import prod
from typing import Any, ClassVar, Type
from collections import defaultdict
from collections.abc import Callable, Mapping
import inspect
from types import MappingProxyType

import polars as pl
import numpy as np
import pyarrow as pa
import torch

AsofTolerance = str | int | float | timedelta | None
_MISSING_FILL_VALUE = object()


class NullPolicy(str, Enum):
    ERROR = "error"
    PROPAGATE = "propagate"
    DROP = "drop"
    FILL = "fill"
    PASS = "pass"


@dataclass(frozen=True)
class NullHandler:
    policy: NullPolicy | None = None
    function: Callable[[pl.Series], pl.Series] | None = None

    def __post_init__(self) -> None:
        has_policy = self.policy is not None
        has_function = self.function is not None
        if has_policy == has_function:
            raise ValueError("NullHandler must wrap exactly one policy or function")
        if has_policy:
            object.__setattr__(self, "policy", _normalize_null_policy(self.policy))
        if has_function:
            _validate_null_handler_function(self.function)

    @classmethod
    def from_policy(cls, policy: NullPolicy | str) -> NullHandler:
        return cls(policy=_normalize_null_policy(policy))

    @classmethod
    def from_function(cls, function: Callable[[pl.Series], pl.Series]) -> NullHandler:
        return cls(function=function)

    @property
    def is_custom(self) -> bool:
        return self.function is not None


@dataclass(frozen=True)
class TimeAxis:
    column: str = "timestamp"
    dtype: pl.DataType = pl.Datetime
    timezone: str | None = None


@dataclass(frozen=True)
class ColumnSignature:
    name: str
    dtype: pl.DataType
    shape: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError("column name must be a string")
        shape = _normalize_shape(self.shape)
        _validate_column_dtype(self.dtype, shape)
        object.__setattr__(self, "shape", shape)

    @property
    def physical_dtype(self) -> pl.DataType:
        return _column_physical_dtype(self)


ColumnEntry = tuple[str, pl.DataType] | tuple[str, pl.DataType, tuple[int, ...]]


@dataclass(frozen=True)
class FrameSignature:
    time: TimeAxis | None = field(default_factory=TimeAxis)
    columns: tuple[ColumnEntry | ColumnSignature, ...] = ()

    @classmethod
    def empty(cls) -> FrameSignature:
        """Return the explicit signature for a TSFN that consumes no input frame."""
        return cls(time=None, columns=())

    def __post_init__(self) -> None:
        if not isinstance(self.columns, tuple):
            raise TypeError("columns must be a tuple, not a list")
        normalized_columns = tuple(
            _column_entry(_column_signature(column)) for column in self.columns
        )
        object.__setattr__(self, "columns", normalized_columns)

        columns = _column_signatures(self)

        column_names = [column.name for column in columns]
        duplicate_names = sorted(
            {name for name in column_names if column_names.count(name) > 1}
        )
        if duplicate_names:
            raise ValueError(f"Duplicate value columns: {duplicate_names}")

        if self.time is None and columns:
            raise ValueError("Inputless frame signatures cannot declare value columns")
        if self.time is not None and any(
            column.name == self.time.column for column in columns
        ):
            raise ValueError(
                f"Time column '{self.time.column}' must not be listed as a value column"
            )

    def is_empty(self) -> bool:
        return self.time is None and not self.columns


def _column_signature(column: ColumnEntry | ColumnSignature) -> ColumnSignature:
    if isinstance(column, ColumnSignature):
        shape = _normalize_shape(column.shape)
        _validate_column_dtype(column.dtype, shape)
        return ColumnSignature(column.name, column.dtype, shape)

    if (
        not isinstance(column, tuple)
        or len(column) not in (2, 3)
        or not isinstance(column[0], str)
    ):
        raise TypeError(
            "columns must contain (name, dtype) or (name, dtype, shape) entries"
        )

    name = column[0]
    dtype = column[1]
    shape = _normalize_shape(column[2]) if len(column) == 3 else ()
    _validate_column_dtype(dtype, shape)
    return ColumnSignature(name, dtype, shape)


def _column_entry(column: ColumnSignature) -> ColumnEntry:
    if column.shape:
        return (column.name, column.dtype, column.shape)
    return (column.name, column.dtype)


def _column_signatures(signature: FrameSignature) -> tuple[ColumnSignature, ...]:
    return tuple(_column_signature(column) for column in signature.columns)


def _column_signature_map(signature: FrameSignature) -> dict[str, ColumnSignature]:
    return {column.name: column for column in _column_signatures(signature)}


def _replace_column(
    signature: FrameSignature,
    column_name: str,
    replacement: ColumnSignature,
) -> tuple[ColumnEntry, ...]:
    return tuple(
        _column_entry(replacement if column.name == column_name else column)
        for column in _column_signatures(signature)
    )


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


def _validate_column_dtype(dtype: pl.DataType, shape: tuple[int, ...]) -> None:
    if not (_is_dtype_class(dtype) or isinstance(dtype, pl.DataType)):
        raise TypeError("column dtype must be a Polars data type")
    if shape and (
        dtype is pl.Array
        or dtype is pl.List
        or _is_array_instance(dtype)
        or _is_list_instance(dtype)
    ):
        raise TypeError("shaped columns must declare an element dtype, not Array/List")


def _flat_size(shape: tuple[int, ...]) -> int:
    return prod(shape) if shape else 1


def _column_physical_dtype(column: ColumnEntry | ColumnSignature) -> pl.DataType:
    column = _column_signature(column)
    if not column.shape:
        return column.dtype
    return pl.Array(column.dtype, _flat_size(column.shape))


def _format_column_signature(column: ColumnEntry | ColumnSignature) -> str:
    column = _column_signature(column)
    if not column.shape:
        return str(column.dtype)
    return f"{column.dtype} shape={column.shape}"


def _column_signature_matches(
    actual: ColumnSignature,
    expected: ColumnSignature,
) -> bool:
    return actual.shape == expected.shape and _dtype_matches(actual.dtype, expected.dtype)


def _is_array_dtype(dtype: pl.DataType) -> bool:
    return _is_array_instance(dtype)


def _column_null_expr(column_name: str, dtype: pl.DataType) -> pl.Expr:
    column = pl.col(column_name)
    if _is_array_dtype(dtype):
        inner_nulls = column.arr.eval(pl.element().is_null()).arr.any().fill_null(False)
        return column.is_null() | inner_nulls
    return column.is_null()


def _fill_column_null_expr(
    column_name: str,
    dtype: pl.DataType,
    fill_value: Any,
) -> pl.Expr:
    column = pl.col(column_name)
    if _is_array_dtype(dtype):
        fill_array = [fill_value] * dtype.size
        return (
            pl.when(column.is_null())
            .then(pl.lit(fill_array, dtype=dtype))
            .otherwise(column)
            .arr.eval(pl.element().fill_null(fill_value))
            .alias(column_name)
        )
    return column.fill_null(fill_value).alias(column_name)


def _series_null_count(series: pl.Series) -> int:
    count = int(series.null_count())
    if _is_array_dtype(series.dtype):
        inner_count = series.to_frame().select(
            pl.col(series.name)
            .arr.eval(pl.element().is_null())
            .arr.sum()
            .sum()
            .alias("inner_nulls")
        )["inner_nulls"][0]
        count += int(inner_count or 0)
    return count


def _drop_series_nulls(series: pl.Series) -> pl.Series:
    return (
        series.to_frame()
        .filter(~_column_null_expr(series.name, series.dtype))
        .to_series()
    )


def _fill_series_nulls(series: pl.Series, fill_value: Any) -> pl.Series:
    return (
        series.to_frame()
        .select(_fill_column_null_expr(series.name, series.dtype, fill_value))
        .to_series()
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


def _format_frame_signature(signature: FrameSignature) -> dict[str, Any]:
    if signature.is_empty():
        return {"time": None, "columns": []}

    assert signature.time is not None
    return {
        "time": {
            "column": signature.time.column,
            "dtype": str(signature.time.dtype),
            "timezone": signature.time.timezone,
        },
        "columns": [
            (column.name, str(column.dtype))
            if not column.shape
            else {
                "name": column.name,
                "dtype": str(column.dtype),
                "shape": column.shape,
                "physical_dtype": str(column.physical_dtype),
            }
            for column in _column_signatures(signature)
        ],
    }


def _format_tolerance(tolerance: AsofTolerance) -> dict[str, str] | None:
    if tolerance is None:
        return None
    return {
        "type": type(tolerance).__name__,
        "value": str(tolerance),
    }


def _normalize_null_policy(policy: NullPolicy | str) -> NullPolicy:
    if isinstance(policy, NullPolicy):
        return policy
    try:
        return NullPolicy(policy)
    except ValueError as exc:
        valid = [item.value for item in NullPolicy]
        raise ValueError(f"Unknown null policy {policy!r}. Expected one of {valid}") from exc


def _validate_null_handler_function(function: Callable[[pl.Series], pl.Series]) -> None:
    if not inspect.isfunction(function):
        raise TypeError("Custom null handlers must be named Python functions")
    if function.__name__ == "<lambda>" or "<locals>" in function.__qualname__:
        raise ValueError(
            "Custom null handlers must be named top-level functions so node IDs "
            "remain deterministic"
        )


def _normalize_null_handler(
    handler: NullHandler | NullPolicy | str | Callable[[pl.Series], pl.Series],
) -> NullHandler:
    if isinstance(handler, NullHandler):
        return handler
    if isinstance(handler, NullPolicy) or isinstance(handler, str):
        return NullHandler.from_policy(handler)
    if callable(handler):
        return NullHandler.from_function(handler)
    raise TypeError(
        "Null handlers must be a NullPolicy, policy string, NullHandler, or named function"
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
    return {name: value for name, value in sorted(values.items())}


def _format_function_identity(function: TSFN) -> dict[str, Any]:
    input_sig, output_sig = function.signature
    return {
        "module": function.__class__.__module__,
        "qualname": function.__class__.__qualname__,
        "version": function.version,
        "input_signature": _format_frame_signature(input_sig),
        "output_signature": _format_frame_signature(output_sig),
    }


@dataclass(frozen=True)
class TSFNConfig(abc.ABC):
    def __str__(self) -> str:
        def serializer(obj):
            if isinstance(obj, pl.DataType):
                return str(obj)
            raise TypeError(f"Type {type(obj)} not serializable")
            
        data = {f.name: getattr(self, f.name) for f in fields(self)}
        return json.dumps(data, sort_keys=True, default=serializer)


class TSFN(abc.ABC):
    CONFIG_CLS: ClassVar[Type[TSFNConfig]] = TSFNConfig
    VERSION: ClassVar[str]

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if not inspect.isabstract(cls):
            cls._validate_config_cls()
            cls._validate_version()

    @classmethod
    def _validate_version(cls) -> str:
        if "VERSION" not in cls.__dict__:
            raise TypeError(f"TSFN subclass '{cls.__name__}' must define VERSION")
        if not isinstance(cls.VERSION, str):
            raise TypeError(f"TSFN subclass '{cls.__name__}' VERSION must be a string")
        if not cls.VERSION.strip():
            raise ValueError(f"TSFN subclass '{cls.__name__}' VERSION must be non-empty")
        return cls.VERSION

    @classmethod
    def _validate_config_cls(cls) -> type[TSFNConfig]:
        config_cls = cls.CONFIG_CLS
        if not isinstance(config_cls, type):
            raise TypeError(f"{cls.__name__}.CONFIG_CLS must be a class")
        if not is_dataclass(config_cls):
            raise TypeError(f"{cls.__name__}.CONFIG_CLS must be a dataclass")
        if not issubclass(config_cls, TSFNConfig):
            raise TypeError(f"{cls.__name__}.CONFIG_CLS must inherit TSFNConfig")
        return config_cls

    def __init__(
        self,
        parameters: dict[str, Any],
    ):
        self._validate_config_cls()
        self.version = self._validate_version()
        self.parameters = self._bind_and_validate_config(parameters)
        self.signature = self.type_signature()
        self._bridge_null_handlers = MappingProxyType({})
        self._bridge_null_fill_values = MappingProxyType({})

    def _bind_and_validate_config(self, params: dict[str, Any]) -> TSFNConfig:
        config_fields = fields(self.CONFIG_CLS)
        allowed_fields = {f.name for f in config_fields}

        # 1. Check for unexpected parameters (prevents silent typos)
        extra_keys = params.keys() - allowed_fields
        if extra_keys:
            raise ValueError(
                f"Unexpected parameters for {self.__class__.__name__}: {sorted(extra_keys)}"
            )

        # 2. Check for missing required parameters
        required_fields = {
            f.name for f in config_fields
            if f.default is MISSING and f.default_factory is MISSING
        }
        missing_keys = required_fields - params.keys()

        if missing_keys:
            raise ValueError(
                f"Parameter validation failed for {self.__class__.__name__}. "
                f"Missing required parameters: {sorted(missing_keys)}. "
                f"Expected schema: {self.CONFIG_CLS.__annotations__}"
            )

        return self.CONFIG_CLS(**params)

    @abc.abstractmethod
    def type_signature(self) -> tuple[
        FrameSignature,
        FrameSignature,
    ]:
        """Return (input_frame_signature, output_frame_signature)."""
        pass

    def __str__(self) -> str:
        input_sig, output_sig = self.signature
        return (
            f"{self.__class__.__module__}.{self.__class__.__qualname__}@{self.version}"
            f"(in:{json.dumps(_format_frame_signature(input_sig), sort_keys=True)}, "
            f"out:{json.dumps(_format_frame_signature(output_sig), sort_keys=True)})"
        )

    def validate_input_schema(self, lf: pl.LazyFrame | None) -> None:
        """Validates that the incoming LazyFrame matches the expected input signature."""
        self._validate_schema(lf, self.signature[0], "input")

    def validate_output_schema(self, lf: pl.LazyFrame) -> None:
        """Validates that the returned LazyFrame matches the expected output signature."""
        self._validate_schema(lf, self.signature[1], "output")

    def _validate_schema(
        self,
        lf: pl.LazyFrame | None,
        signature: FrameSignature,
        schema_name: str,
    ) -> None:
        if signature.is_empty():
            if lf is not None:
                raise ValueError(f"Expected no {schema_name} frame for inputless signature")
            return

        if lf is None:
            raise ValueError(f"Missing required {schema_name} frame")

        if signature.time is None:
            raise ValueError(f"{schema_name.capitalize()} signature must declare a time axis")

        current_schema = lf.collect_schema()
        time_axis = signature.time

        if time_axis.column not in current_schema:
            raise ValueError(f"Missing required {schema_name} time column: '{time_axis.column}'")

        actual_time_type = current_schema[time_axis.column]
        if not _dtype_matches(
            _datetime_dtype_without_timezone(actual_time_type),
            _datetime_dtype_without_timezone(time_axis.dtype),
        ):
            raise TypeError(
                f"Time column '{time_axis.column}' type mismatch. "
                f"Expected {time_axis.dtype}, got {actual_time_type}"
            )

        actual_timezone = getattr(actual_time_type, "time_zone", None)
        if actual_timezone != time_axis.timezone:
            raise TypeError(
                f"Time column '{time_axis.column}' timezone mismatch. "
                f"Expected {time_axis.timezone}, got {actual_timezone}"
            )

        for column in _column_signatures(signature):
            if column.name not in current_schema:
                raise ValueError(
                    f"Missing required {schema_name} column: '{column.name}'"
                )
            
            actual_type = current_schema[column.name]
            expected_type = column.physical_dtype
            if not _dtype_matches(actual_type, expected_type):
                raise TypeError(
                    f"Column '{column.name}' type mismatch. "
                    f"Expected {expected_type} ({_format_column_signature(column)}), "
                    f"got {actual_type}"
                )

    def resolve_signature(
        self,
        bound_input_columns: Mapping[str, ColumnSignature],
    ) -> tuple[FrameSignature, FrameSignature]:
        """Resolve shape-dependent signatures from already-bound parent columns."""
        return self.signature

    def input_column_signature(self, column_name: str) -> ColumnSignature:
        return _column_signature_map(self.signature[0])[column_name]

    def output_column_signature(self, column_name: str) -> ColumnSignature:
        return _column_signature_map(self.signature[1])[column_name]

    def bridge_null_handler(
        self,
        column_name: str,
        explicit_handler: (
            NullHandler | NullPolicy | str | Callable[[pl.Series], pl.Series] | None
        ) = None,
    ) -> NullHandler:
        if explicit_handler is not None:
            return _normalize_null_handler(explicit_handler)
        return self._bridge_null_handlers.get(
            column_name,
            NullHandler.from_policy(NullPolicy.ERROR),
        )

    def bridge_null_policy(
        self,
        column_name: str,
        explicit_policy: NullPolicy | str | None = None,
    ) -> NullPolicy:
        handler = self.bridge_null_handler(column_name, explicit_policy)
        if handler.policy is None:
            raise TypeError(
                f"Column '{column_name}' uses a custom null handler, not a NullPolicy"
            )
        return handler.policy

    def bridge_null_fill_value(
        self,
        column_name: str,
        explicit_fill_value: Any = _MISSING_FILL_VALUE,
    ) -> Any:
        if explicit_fill_value is not _MISSING_FILL_VALUE:
            return explicit_fill_value
        return self._bridge_null_fill_values.get(column_name, _MISSING_FILL_VALUE)

    def __call__(self, lf: pl.LazyFrame | None = None) -> pl.LazyFrame:
        input_signature = self.signature[0]
        self.validate_input_schema(lf)

        if input_signature.is_empty():
            output_lf = self.apply()
        else:
            output_lf = self.apply(lf)

        self.validate_output_schema(output_lf)
        return output_lf

    def series_to_numpy(
        self,
        series: pl.Series,
        shape: tuple[int, ...] | None = None,
        *,
        null_policy: (
            NullHandler | NullPolicy | str | Callable[[pl.Series], pl.Series] | None
        ) = None,
        fill_value: Any = _MISSING_FILL_VALUE,
        allow_copy: bool = True,
    ):
        series = self._prepare_series_for_bridge(
            series,
            null_policy=null_policy,
            fill_value=fill_value,
        )
        array = series.to_numpy(allow_copy=allow_copy)
        if shape:
            return array.reshape((len(series), *_normalize_shape(shape)))
        return array

    def numpy_to_series(
        self,
        name: str,
        array,
        *,
        dtype: pl.DataType = pl.Float64,
        shape: tuple[int, ...] | None = None,
        allow_copy: bool = True,
    ) -> pl.Series:

        array = np.asarray(array)
        if shape:
            shape = _normalize_shape(shape)
            width = _flat_size(shape)
            array = array.reshape((len(array), width))
            if not array.flags.c_contiguous:
                if not allow_copy:
                    raise ValueError(
                        f"Cannot create shaped Series '{name}' without copying: "
                        "NumPy array is not C-contiguous"
                    )
                array = np.ascontiguousarray(array)

            flat = array.ravel()
            arrow_values = pa.array(flat)
            arrow_array = pa.FixedSizeListArray.from_arrays(arrow_values, width)
            series = pl.from_arrow(arrow_array).alias(name)
            expected_dtype = pl.Array(dtype, width)
            if series.dtype != expected_dtype:
                if not allow_copy:
                    raise TypeError(
                        f"Cannot create Series '{name}' as {expected_dtype} without "
                        f"copying or casting; inferred {series.dtype}"
                    )
                series = series.cast(expected_dtype, strict=False)
            return series

        if not array.flags.c_contiguous:
            if not allow_copy:
                raise ValueError(
                    f"Cannot create Series '{name}' without copying: "
                    "NumPy array is not C-contiguous"
                )
            array = np.ascontiguousarray(array)
        series = pl.from_arrow(pa.array(array)).alias(name)
        if series.dtype != dtype:
            if not allow_copy:
                raise TypeError(
                    f"Cannot create Series '{name}' as {dtype} without copying or "
                    f"casting; inferred {series.dtype}"
                )
            series = series.cast(dtype, strict=False)
        return series

    def series_to_torch(
        self,
        series: pl.Series,
        shape: tuple[int, ...] | None = None,
        *,
        null_policy: (
            NullHandler | NullPolicy | str | Callable[[pl.Series], pl.Series] | None
        ) = None,
        fill_value: Any = _MISSING_FILL_VALUE,
        allow_copy: bool = True,
    ):

        array = self.series_to_numpy(
            series,
            shape,
            null_policy=null_policy,
            fill_value=fill_value,
            allow_copy=allow_copy,
        )
        return torch.from_numpy(array)

    def torch_to_series(
        self,
        name: str,
        tensor,
        *,
        dtype: pl.DataType = pl.Float64,
        shape: tuple[int, ...] | None = None,
        allow_copy: bool = True,
    ) -> pl.Series:
        tensor = tensor.detach().cpu()
        return self.numpy_to_series(
            name,
            tensor.numpy(),
            dtype=dtype,
            shape=shape,
            allow_copy=allow_copy,
        )

    def series_to_pytorch(
        self,
        series: pl.Series,
        shape: tuple[int, ...] | None = None,
        *,
        null_policy: (
            NullHandler | NullPolicy | str | Callable[[pl.Series], pl.Series] | None
        ) = None,
        fill_value: Any = _MISSING_FILL_VALUE,
        allow_copy: bool = True,
    ):
        return self.series_to_torch(
            series,
            shape,
            null_policy=null_policy,
            fill_value=fill_value,
            allow_copy=allow_copy,
        )

    def pytorch_to_series(
        self,
        name: str,
        tensor,
        *,
        dtype: pl.DataType = pl.Float64,
        shape: tuple[int, ...] | None = None,
        allow_copy: bool = True,
    ) -> pl.Series:
        return self.torch_to_series(
            name,
            tensor,
            dtype=dtype,
            shape=shape,
            allow_copy=allow_copy,
        )

    def _prepare_series_for_bridge(
        self,
        series: pl.Series,
        *,
        null_policy: (
            NullHandler | NullPolicy | str | Callable[[pl.Series], pl.Series] | None
        ) = None,
        fill_value: Any = _MISSING_FILL_VALUE,
    ) -> pl.Series:
        handler = self.bridge_null_handler(series.name, null_policy)
        policy = handler.policy
        if policy is None:
            null_count = _series_null_count(series)
            if null_count == 0:
                return series
            handled = handler.function(series)
            if not isinstance(handled, pl.Series):
                raise TypeError(
                    f"Custom null handler for column '{series.name}' must return a "
                    "Polars Series"
                )
            if handled.name != series.name:
                handled = handled.rename(series.name)
            return handled

        if policy is NullPolicy.PASS:
            return series
        if policy is NullPolicy.PROPAGATE:
            raise ValueError(
                "NullPolicy.PROPAGATE cannot be used by direct NumPy/Torch bridge "
                f"helpers for column '{series.name}'. Use a Polars-native transform "
                "or implement propagation in the TSFN batch method."
            )

        null_count = _series_null_count(series)
        if null_count == 0:
            return series

        if policy is NullPolicy.ERROR:
            raise ValueError(
                f"NullPolicy.ERROR failed for column '{series.name}': "
                f"{null_count} null value(s) encountered before NumPy/Torch conversion"
            )

        if policy is NullPolicy.DROP:
            return _drop_series_nulls(series)

        if policy is NullPolicy.FILL:
            fill_value = self.bridge_null_fill_value(series.name, fill_value)
            if fill_value is _MISSING_FILL_VALUE:
                raise ValueError(
                    f"NullPolicy.FILL for column '{series.name}' requires a fill value"
                )
            return _fill_series_nulls(series, fill_value)

        raise ValueError(f"Unhandled null policy {policy!r}")

    @abc.abstractmethod
    def apply(self, lf: pl.LazyFrame | None = None) -> pl.LazyFrame:
        """All transformations should be done lazily here."""
        pass


class ItemwiseUnaryTSFN(TSFN, abc.ABC):
    """Base for one-input transforms that operate independently per value."""

    @abc.abstractmethod
    def itemwise_input_column(self) -> str:
        pass

    @abc.abstractmethod
    def itemwise_output_column(self) -> str:
        pass

    @abc.abstractmethod
    def itemwise_expr(self, value: pl.Expr) -> pl.Expr:
        pass

    def resolve_signature(
        self,
        bound_input_columns: Mapping[str, ColumnSignature],
    ) -> tuple[FrameSignature, FrameSignature]:
        input_signature, output_signature = self.signature
        input_columns = _column_signature_map(input_signature)
        output_columns = _column_signature_map(output_signature)
        input_name = self.itemwise_input_column()
        output_name = self.itemwise_output_column()

        if input_name not in input_columns:
            raise ValueError(f"Itemwise input column '{input_name}' is not declared")
        if output_name not in output_columns:
            raise ValueError(f"Itemwise output column '{output_name}' is not declared")
        if len(input_columns) != 1 or len(output_columns) != 1:
            raise ValueError("ItemwiseUnaryTSFN requires exactly one input and one output")

        bound_input = bound_input_columns.get(input_name)
        if bound_input is None:
            return self.signature

        declared_input = input_columns[input_name]
        declared_output = output_columns[output_name]
        resolved_input = ColumnSignature(
            declared_input.name,
            declared_input.dtype,
            bound_input.shape,
        )
        resolved_output = ColumnSignature(
            declared_output.name,
            declared_output.dtype,
            bound_input.shape,
        )
        return (
            FrameSignature(
                time=input_signature.time,
                columns=_replace_column(input_signature, input_name, resolved_input),
            ),
            FrameSignature(
                time=output_signature.time,
                columns=_replace_column(output_signature, output_name, resolved_output),
            ),
        )

    def apply(self, lf: pl.LazyFrame | None = None) -> pl.LazyFrame:
        if lf is None:
            raise ValueError("ItemwiseUnaryTSFN requires an input frame")

        input_signature, output_signature = self.signature
        if input_signature.time is None:
            raise ValueError("ItemwiseUnaryTSFN input signature must declare a time axis")

        input_column = _column_signature_map(input_signature)[self.itemwise_input_column()]
        output_column = _column_signature_map(output_signature)[self.itemwise_output_column()]
        value = pl.col(input_column.name)

        if input_column.shape:
            result = value.arr.eval(self.itemwise_expr(pl.element()))
        else:
            result = self.itemwise_expr(value)

        return lf.select(
            input_signature.time.column,
            result.cast(output_column.physical_dtype).alias(output_column.name),
        )


class BatchTSFN(TSFN, abc.ABC):
    """Base for one-output batch UDF transforms over Polars Series chunks."""

    BATCH_IS_ELEMENTWISE: ClassVar[bool] = False

    @abc.abstractmethod
    def batch_input_columns(self) -> tuple[str, ...]:
        pass

    @abc.abstractmethod
    def batch_output_column(self) -> str:
        pass

    @abc.abstractmethod
    def batch(self, fields: Mapping[str, pl.Series]) -> pl.Series:
        pass

    def _batch_from_struct(self, struct_series: pl.Series) -> pl.Series:
        fields = {
            column_name: struct_series.struct.field(column_name)
            for column_name in self.batch_input_columns()
        }
        return self.batch(fields)

    def apply(self, lf: pl.LazyFrame | None = None) -> pl.LazyFrame:
        if lf is None:
            raise ValueError("BatchTSFN requires an input frame")

        input_signature, output_signature = self.signature
        if input_signature.time is None:
            raise ValueError("BatchTSFN input signature must declare a time axis")

        prepared_lf = self._apply_lazy_null_policies(lf)
        output_columns = _column_signature_map(output_signature)
        output_column = output_columns[self.batch_output_column()]
        input_columns = list(self.batch_input_columns())
        batch_expr = (
            pl.struct(input_columns)
            .map_batches(
                self._batch_from_struct,
                return_dtype=output_column.physical_dtype,
                is_elementwise=self.BATCH_IS_ELEMENTWISE,
            )
            .alias(output_column.name)
        )
        return prepared_lf.select(input_signature.time.column, batch_expr)

    def _apply_lazy_null_policies(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        input_columns = _column_signature_map(self.signature[0])
        prepared_lf = lf

        for column_name in self.batch_input_columns():
            handler = self.bridge_null_handler(column_name)
            if handler.policy is None:
                continue
            policy = handler.policy
            if policy not in (NullPolicy.DROP, NullPolicy.FILL):
                continue

            column = input_columns[column_name]
            if policy is NullPolicy.DROP:
                prepared_lf = prepared_lf.filter(
                    ~_column_null_expr(column.name, column.physical_dtype)
                )
                continue

            fill_value = self.bridge_null_fill_value(column.name)
            if fill_value is _MISSING_FILL_VALUE:
                raise ValueError(
                    f"NullPolicy.FILL for column '{column.name}' requires a fill value"
                )
            prepared_lf = prepared_lf.with_columns(
                _fill_column_null_expr(column.name, column.physical_dtype, fill_value)
            )

        return prepared_lf


class ItemwiseStructTSFN(BatchTSFN, abc.ABC):
    """Base for n-ary itemwise transforms lowered through a struct batch."""

    BATCH_IS_ELEMENTWISE: ClassVar[bool] = True

    def resolve_signature(
        self,
        bound_input_columns: Mapping[str, ColumnSignature],
    ) -> tuple[FrameSignature, FrameSignature]:
        input_signature, output_signature = self.signature
        input_columns = _column_signature_map(input_signature)
        output_columns = _column_signature_map(output_signature)
        output_name = self.batch_output_column()

        if output_name not in output_columns:
            raise ValueError(f"Itemwise output column '{output_name}' is not declared")

        resolved_input_columns: tuple[ColumnEntry, ...] = input_signature.columns
        input_shapes = []
        for input_name in self.batch_input_columns():
            if input_name not in input_columns:
                raise ValueError(f"Itemwise input column '{input_name}' is not declared")
            declared_input = input_columns[input_name]
            bound_input = bound_input_columns.get(input_name)
            resolved_shape = (
                declared_input.shape if bound_input is None else bound_input.shape
            )
            input_shapes.append(resolved_shape)
            resolved_input_columns = _replace_column(
                FrameSignature(time=input_signature.time, columns=resolved_input_columns),
                input_name,
                ColumnSignature(
                    declared_input.name,
                    declared_input.dtype,
                    resolved_shape,
                ),
            )

        output_shape = self.resolve_output_shape(tuple(input_shapes))
        declared_output = output_columns[output_name]
        resolved_output = ColumnSignature(
            declared_output.name,
            declared_output.dtype,
            output_shape,
        )
        return (
            FrameSignature(time=input_signature.time, columns=resolved_input_columns),
            FrameSignature(
                time=output_signature.time,
                columns=_replace_column(output_signature, output_name, resolved_output),
            ),
        )

    def resolve_output_shape(
        self,
        input_shapes: tuple[tuple[int, ...], ...],
    ) -> tuple[int, ...]:
        if not input_shapes:
            return ()

        first_shape = input_shapes[0]
        mismatched_shapes = sorted({shape for shape in input_shapes if shape != first_shape})
        if mismatched_shapes:
            raise ValueError(
                "Itemwise struct inputs must share the same shape unless "
                "resolve_output_shape() is overridden"
            )
        return first_shape


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
        tolerances: dict[str, AsofTolerance] | None = None,
        null_handlers: dict[
            str,
            NullHandler | NullPolicy | str | Callable[[pl.Series], pl.Series],
        ]
        | None = None,
        null_policies: dict[str, NullPolicy | str] | None = None,
        null_fill_values: dict[str, Any] | None = None,
    ):
        self.name = name
        self.function_cls = function_cls
        self.function = function_cls(parameters or {})
        self.parameters = self.function.parameters
        
        # bindings map TSFN semantic input names to (parent_node, parent_output_column)
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
        
        # Deduplicate and extract unique parent Nodes preserving order
        self.inputs = tuple(dict.fromkeys(parent for parent, _ in self.bindings.values()))

        bound_input_columns: dict[str, ColumnSignature] = {}
        for input_name, (parent, parent_col) in self.bindings.items():
            parent_outputs = _column_signature_map(parent.function.signature[1])
            if parent_col in parent_outputs:
                bound_input_columns[input_name] = parent_outputs[parent_col]
        self.function.signature = self.function.resolve_signature(bound_input_columns)
        self.function._bridge_null_handlers = self.null_handlers
        self.function._bridge_null_fill_values = self.null_fill_values
        
        # Map exposed output names to their corresponding data types
        self.outputs = MappingProxyType(
            {
                column.name: column.physical_dtype
                for column in _column_signatures(self.function.signature[1])
            }
        )
        
        self.ID = self._generate_persistent_id()
        self._frozen = True

    def __getattr__(self, name: str) -> tuple[Node, str]:
        """Provides syntactical sugar to reference outputs (e.g., node.lagged)."""
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
            f"fn={function_name}@{version})"
        )

    def _generate_persistent_id(self) -> str:
        # Serialize bindings deterministically by sorting semantic input keys
        serialized_bindings = {
            input_name: {
                "parent_id": parent.ID,
                "parent_output": parent_col,
                "tolerance": _format_tolerance(self.tolerances.get(input_name)),
            }
            for input_name, (parent, parent_col) in sorted(self.bindings.items())
        }
        
        node_definition = {
            "bindings": serialized_bindings,
            "function": _format_function_identity(self.function),
            "null_fill_values": _format_null_fill_values(self.null_fill_values),
            "null_handlers": _format_null_handlers(self.null_handlers),
            "parameters": json.loads(str(self.parameters)),
            "outputs": sorted([(name, str(dtype)) for name, dtype in self.outputs.items()]),
        }

        serialized_data = json.dumps(node_definition, sort_keys=True)
        return hashlib.sha256(serialized_data.encode("utf-8")).hexdigest()


class Graph:
    def __init__(self, root_node: Node):
        self.root_node = root_node
        self.node_list = self.get_subgraph_execution_order(self.root_node)
        
        # Perform global static validation
        self._validate_graph()
        
        self.ID = self._generate_persistent_id()
        self._compiled_root_lf: pl.LazyFrame | None = None

    def __repr__(self) -> str:
        return (
            f"Graph(id={self.ID[:8]!r}, root={self.root_node.ID[:8]!r}, "
            f"nodes={len(self.node_list)})"
        )

    def get_subgraph_execution_order(self, target_node: Node) -> list[Node]:
        visited: set[str] = set()
        visiting: set[str] = set()
        ordered_execution_nodes: list[Node] = []

        def dfs(node: Node):
            if node.ID in visiting:
                raise ValueError(
                    f"Cycle detected at node '{node.name or node.ID}'! "
                    f"Graphs must be acyclic."
                )

            if node.ID in visited:
                return

            visiting.add(node.ID)
            for parent_node in node.inputs:
                dfs(parent_node)
            visiting.remove(node.ID)
            visited.add(node.ID)
            ordered_execution_nodes.append(node)

        dfs(target_node)
        return ordered_execution_nodes

    def _validate_graph(self) -> None:
        """Validates bindings, outputs, and type compatibility at construction."""
        node_ids = {graph_node.ID for graph_node in self.node_list}

        for node in self.node_list:
            input_signature = node.function.signature[0]
            input_columns = _column_signature_map(input_signature)
            expected_inputs = set(input_columns)
            bound_inputs = set(node.bindings.keys())
            tolerance_inputs = set(node.tolerances.keys())
            null_handler_inputs = set(node.null_handlers.keys())
            null_fill_inputs = set(node.null_fill_values.keys())

            if not node.bindings and not input_signature.is_empty():
                raise ValueError(
                    f"Binding validation failed for node '{node.name or node.ID}'. "
                    "Nodes with no predecessors must declare an empty input signature."
                )

            extra_tolerances = tolerance_inputs - bound_inputs
            if extra_tolerances:
                raise ValueError(
                    f"Binding validation failed for node '{node.name or node.ID}'. "
                    f"Unexpected tolerances for unbound inputs: {sorted(extra_tolerances)}"
                )

            extra_null_handlers = null_handler_inputs - expected_inputs
            if extra_null_handlers:
                raise ValueError(
                    f"Binding validation failed for node '{node.name or node.ID}'. "
                    f"Unexpected null handlers for inputs: {sorted(extra_null_handlers)}"
                )

            extra_null_fill_values = null_fill_inputs - expected_inputs
            if extra_null_fill_values:
                raise ValueError(
                    f"Binding validation failed for node '{node.name or node.ID}'. "
                    f"Unexpected null fill values for inputs: {sorted(extra_null_fill_values)}"
                )

            missing_fill_values = {
                input_name
                for input_name, handler in node.null_handlers.items()
                if handler.policy is NullPolicy.FILL
                and input_name not in node.null_fill_values
            }
            if missing_fill_values:
                raise ValueError(
                    f"Binding validation failed for node '{node.name or node.ID}'. "
                    "NullPolicy.FILL requires null_fill_values for inputs: "
                    f"{sorted(missing_fill_values)}"
                )

            # 1. Binding completeness
            missing = expected_inputs - bound_inputs
            if missing:
                raise ValueError(
                    f"Binding validation failed for node '{node.name or node.ID}'. "
                    f"Missing expected inputs: {sorted(missing)}"
                )
            
            extra = bound_inputs - expected_inputs
            if extra:
                raise ValueError(
                    f"Binding validation failed for node '{node.name or node.ID}'. "
                    f"Unexpected bound inputs: {sorted(extra)}"
                )

            if node.bindings and input_signature.time is None:
                raise ValueError(
                    f"Binding validation failed for node '{node.name or node.ID}'. "
                    "Bound nodes must declare an input time axis."
                )

            # 2. Output existence & 3. Type compatibility
            for input_name, (parent_node, parent_col) in node.bindings.items():
                if parent_node.ID not in node_ids:
                    raise ValueError(
                        f"Node '{node.name or node.ID}' binds to parent "
                        f"'{parent_node.name or parent_node.ID}' which is not in this graph."
                    )
                
                # Check output existence
                parent_output_columns = _column_signature_map(
                    parent_node.function.signature[1]
                )
                if parent_col not in parent_output_columns:
                    raise ValueError(
                        f"Binding validation failed for node '{node.name or node.ID}'. "
                        f"Parent node '{parent_node.name or parent_node.ID}' does not expose "
                        f"output '{parent_col}'. Available outputs: {list(parent_output_columns)}"
                    )

                # Check type compatibility
                expected_column = input_columns[input_name]
                actual_column = parent_output_columns[parent_col]
                if not _column_signature_matches(actual_column, expected_column):
                    raise TypeError(
                        f"Type mismatch at node '{node.name or node.ID}' for input '{input_name}': "
                        f"expected {_format_column_signature(expected_column)}, "
                        f"got {_format_column_signature(actual_column)} "
                        f"from '{parent_node.name or parent_node.ID}.{parent_col}'"
                    )

                self._validate_time_axis_compatibility(node, parent_node)

    def _validate_time_axis_compatibility(self, node: Node, parent_node: Node) -> None:
        child_time = node.function.signature[0].time
        parent_time = parent_node.function.signature[1].time

        if child_time is None:
            raise ValueError(
                f"Bound node '{node.name or node.ID}' must declare an input time axis"
            )
        if parent_time is None:
            raise ValueError(
                f"Parent node '{parent_node.name or parent_node.ID}' must declare an output time axis"
            )

        if parent_time.column != child_time.column:
            raise ValueError(
                f"Time axis mismatch at node '{node.name or node.ID}': "
                f"expected parent time column '{child_time.column}', "
                f"got '{parent_time.column}' from '{parent_node.name or parent_node.ID}'"
            )

        if not _dtype_matches(parent_time.dtype, child_time.dtype):
            raise TypeError(
                f"Time axis dtype mismatch at node '{node.name or node.ID}': "
                f"expected {child_time.dtype}, got {parent_time.dtype} "
                f"from '{parent_node.name or parent_node.ID}'"
            )

        if parent_time.timezone != child_time.timezone:
            raise TypeError(
                f"Time axis timezone mismatch at node '{node.name or node.ID}': "
                f"expected {child_time.timezone}, got {parent_time.timezone} "
                f"from '{parent_node.name or parent_node.ID}'"
            )

    def _generate_persistent_id(self) -> str:
        graph_definition = {
            "root_id": self.root_node.ID,
            "nodes": sorted(node.ID for node in self.node_list),
        }

        serialized_data = json.dumps(graph_definition, sort_keys=True)
        return hashlib.sha256(serialized_data.encode("utf-8")).hexdigest()

    def compile(self) -> pl.LazyFrame:
        if self._compiled_root_lf is not None:
            return self._compiled_root_lf

        results: dict[str, pl.LazyFrame] = {}

        for node in self.node_list:
            try:
                if not node.bindings:
                    results[node.ID] = node.function()
                else:
                    # Construct TSFN input dataframe by grouping projections per parent
                    parent_to_bindings = defaultdict(list)
                    for input_name, (parent_node, parent_col) in node.bindings.items():
                        tolerance = node.tolerances.get(input_name)
                        parent_to_bindings[(parent_node, tolerance)].append((parent_col, input_name))

                    input_time = node.function.signature[0].time
                    if input_time is None:
                        raise ValueError(
                            f"Bound node '{node.name or node.ID}' must declare an input time axis"
                        )
                    time_col = input_time.column
                    parent_frames = []
                    for (parent_node, tolerance), binds in parent_to_bindings.items():
                        parent_lf = results[parent_node.ID]
                        parent_time = parent_node.function.signature[1].time
                        if parent_time is None:
                            raise ValueError(
                                f"Parent node '{parent_node.name or parent_node.ID}' "
                                "must declare an output time axis"
                            )
                        # Select expected columns and alias them directly to the TSFN's semantic input name.
                        select_exprs = [pl.col(parent_time.column).alias(time_col)]
                        select_exprs.extend(pl.col(p_col).alias(i_name) for p_col, i_name in binds)
                        parent_frames.append(
                            (parent_lf.select(select_exprs).sort(time_col), tolerance)
                        )

                    # Preserve every parent timestamp, then align each parent without lookahead.
                    node_input_lf = (
                        pl.concat(
                            [parent_lf.select(time_col) for parent_lf, _ in parent_frames],
                            how="vertical",
                        )
                        .unique()
                        .sort(time_col)
                    )
                    for parent_lf, tolerance in parent_frames:
                        node_input_lf = node_input_lf.join_asof(
                            parent_lf,
                            on=time_col,
                            strategy="backward",
                            tolerance=tolerance,
                        )
                    results[node.ID] = node.function(node_input_lf)
            except Exception as exc:
                raise RuntimeError(
                    f"Execution failed at node '{node.name or node.ID[:8]}' "
                    f"({node.function_cls.__name__}@{node.function.version}): {exc}"
                ) from exc

        self._compiled_root_lf = results[self.root_node.ID]
        return self._compiled_root_lf

    def execute(self) -> pl.DataFrame:
        root_lf = self._compiled_root_lf
        if root_lf is None:
            root_lf = self.compile()

        try:
            return root_lf.collect()
        except Exception as exc:
            node = self.root_node
            raise RuntimeError(
                f"Execution failed while collecting root node '{node.name or node.ID[:8]}' "
                f"({node.function_cls.__name__}@{node.function.version}): {exc}"
            ) from exc
