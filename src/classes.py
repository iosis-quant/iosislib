from __future__ import annotations

import abc
import hashlib
import json
from datetime import date, datetime, time, timedelta
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from enum import Enum
from math import ceil, isfinite, prod
from pathlib import Path
from typing import Any, ClassVar, Type
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping
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


def _time_axis_physical_dtype(time_axis: TimeAxis) -> pl.DataType:
    dtype = time_axis.dtype
    if time_axis.timezone is None:
        return dtype
    if dtype is pl.Datetime:
        return pl.Datetime(time_zone=time_axis.timezone)
    if isinstance(dtype, pl.Datetime):
        return pl.Datetime(
            time_unit=dtype.time_unit,
            time_zone=time_axis.timezone,
        )
    raise TypeError("A timezone can only be declared for a Datetime time axis")


def _frame_physical_schema(signature: FrameSignature) -> dict[str, pl.DataType]:
    if signature.time is None:
        raise ValueError("A physical frame schema requires a time axis")
    return {
        signature.time.column: _time_axis_physical_dtype(signature.time),
        **{
            column.name: column.physical_dtype
            for column in _column_signatures(signature)
        },
    }


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
    return {
        name: _serialize_value(value)
        for name, value in sorted(values.items())
    }


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
    def to_dict(self) -> dict[str, Any]:
        return {
            config_field.name: _serialize_value(getattr(self, config_field.name))
            for config_field in fields(self)
        }

    def __str__(self) -> str:
        return _canonical_json(self.to_dict())


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _serialize_value(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _qualified_type_name(value: Any) -> str:
    value_type = value if isinstance(value, type) else type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


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

    if isinstance(value, Model):
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


class Dataset(abc.ABC):
    """A finite, repeatable source of schema-stable Polars batches."""

    @property
    @abc.abstractmethod
    def row_count(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def schema(self) -> pl.Schema:
        pass

    def batches(self, *, epoch: int = 0, seed: int = 0) -> Iterator[pl.DataFrame]:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("Dataset epoch must be a non-negative integer")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("Dataset seed must be an integer")

        declared_rows = self.row_count
        if (
            isinstance(declared_rows, bool)
            or not isinstance(declared_rows, int)
            or declared_rows < 0
        ):
            raise ValueError("Dataset row_count must be a non-negative integer")
        declared_schema = self.schema
        if not isinstance(declared_schema, pl.Schema):
            raise TypeError("Dataset schema must be a Polars Schema")

        yielded_rows = 0
        for batch in self._batches(epoch=epoch, seed=seed):
            if not isinstance(batch, pl.DataFrame):
                raise TypeError("Dataset batches must be Polars DataFrames")
            if batch.is_empty():
                raise ValueError("Dataset batches cannot be empty")
            if batch.schema != declared_schema:
                raise TypeError(
                    "Dataset batch schema does not match the declared dataset schema"
                )
            yielded_rows += batch.height
            yield batch

        if declared_rows and not yielded_rows:
            raise RuntimeError("Non-empty Dataset yielded no batches")
        if yielded_rows > declared_rows:
            raise RuntimeError("Dataset yielded more rows than its declared row count")

    @abc.abstractmethod
    def _batches(self, *, epoch: int, seed: int) -> Iterator[pl.DataFrame]:
        pass


@dataclass(frozen=True)
class FrameDataset(Dataset):
    """A materialized frame exposed as deterministic mini-batches."""

    frame: pl.DataFrame
    batch_size: int | None = None
    shuffle: bool = False
    drop_last: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.frame, pl.DataFrame):
            raise TypeError("FrameDataset.frame must be a Polars DataFrame")
        if self.frame.is_empty():
            raise ValueError("FrameDataset.frame cannot be empty")
        if self.batch_size is not None and (
            isinstance(self.batch_size, bool)
            or not isinstance(self.batch_size, int)
            or self.batch_size < 1
        ):
            raise ValueError("FrameDataset.batch_size must be positive or None")
        if not isinstance(self.shuffle, bool):
            raise TypeError("FrameDataset.shuffle must be a boolean")
        if not isinstance(self.drop_last, bool):
            raise TypeError("FrameDataset.drop_last must be a boolean")
        if self.drop_last and self.batch_size is None:
            raise ValueError("drop_last requires a finite batch_size")
        if self.drop_last and self.frame.height < self.batch_size:
            raise ValueError("drop_last would discard the entire dataset")

    @property
    def row_count(self) -> int:
        return self.frame.height

    @property
    def schema(self) -> pl.Schema:
        return self.frame.schema

    def _batches(self, *, epoch: int, seed: int) -> Iterator[pl.DataFrame]:
        frame = self.frame
        if self.shuffle:
            digest = hashlib.sha256(f"{seed}:{epoch}".encode("ascii")).digest()
            epoch_seed = int.from_bytes(digest[:4], "big")
            frame = frame.sample(
                fraction=1.0,
                shuffle=True,
                seed=epoch_seed,
            )

        batch_size = self.batch_size or frame.height
        for offset in range(0, frame.height, batch_size):
            batch = frame.slice(offset, batch_size)
            if self.drop_last and batch.height < batch_size:
                break
            yield batch

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": _qualified_type_name(self),
            "rows": self.row_count,
            "schema": {name: str(dtype) for name, dtype in self.schema.items()},
            "batch_size": self.batch_size,
            "shuffle": self.shuffle,
            "drop_last": self.drop_last,
        }

    def __str__(self) -> str:
        return _canonical_json(self.to_dict())

    def __repr__(self) -> str:
        return (
            f"FrameDataset(rows={self.row_count}, batch_size={self.batch_size!r}, "
            f"shuffle={self.shuffle}, drop_last={self.drop_last})"
        )


@dataclass(frozen=True)
class DatasetSplit:
    """The supervised datasets produced for one fitting operation."""

    train: Dataset
    validation: Dataset | None = None
    test: Dataset | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.train, Dataset):
            raise TypeError("DatasetSplit.train must be a Dataset")
        expected_schema = self.train.schema
        if list(expected_schema.names()) != ["features", "target"]:
            raise ValueError(
                "DatasetSplit requires canonical ['features', 'target'] columns"
            )
        for name in ("validation", "test"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, Dataset):
                raise TypeError(f"DatasetSplit.{name} must be a Dataset or None")
            if value is not None and value.schema != expected_schema:
                raise TypeError(
                    f"DatasetSplit.{name} schema must match the training schema"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_rows": self.train.row_count,
            "validation_rows": (
                None if self.validation is None else self.validation.row_count
            ),
            "test_rows": None if self.test is None else self.test.row_count,
        }

    def __str__(self) -> str:
        return _canonical_json(self.to_dict())


class DatasetSplitter(abc.ABC):
    """A deterministic partition of already ordered supervised examples."""

    def split(self, frame: pl.DataFrame, *, seed: int = 0) -> DatasetSplit:
        if not isinstance(frame, pl.DataFrame):
            raise TypeError("DatasetSplitter requires a Polars DataFrame")
        if frame.is_empty():
            raise ValueError("DatasetSplitter cannot split an empty frame")
        if frame.columns != ["features", "target"]:
            raise ValueError(
                "DatasetSplitter requires canonical ['features', 'target'] columns"
            )
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("DatasetSplitter seed must be an integer")
        result = self._split(frame, seed=seed)
        if not isinstance(result, DatasetSplit):
            raise TypeError("DatasetSplitter._split must return a DatasetSplit")
        split_rows = sum(
            dataset.row_count
            for dataset in (result.train, result.validation, result.test)
            if dataset is not None
        )
        if split_rows > frame.height:
            raise ValueError("DatasetSplitter produced more rows than its input frame")
        return result

    @abc.abstractmethod
    def _split(self, frame: pl.DataFrame, *, seed: int) -> DatasetSplit:
        pass

    def __str__(self) -> str:
        return _canonical_json(self)


@dataclass(frozen=True)
class ChronologicalSplitter(DatasetSplitter):
    """Tail holdouts over the row order established by the feature graph."""

    validation_size: int | float = 0
    test_size: int | float = 0
    gap: int = 0
    batch_size: int | None = None
    shuffle_train: bool = False
    drop_last: bool = False

    def __post_init__(self) -> None:
        for name in ("validation_size", "test_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be an integer count or float fraction")
            if isinstance(value, int) and value < 0:
                raise ValueError(f"{name} cannot be negative")
            if isinstance(value, float) and not 0.0 <= value < 1.0:
                raise ValueError(f"{name} fraction must be in [0, 1)")
        if isinstance(self.gap, bool) or not isinstance(self.gap, int) or self.gap < 0:
            raise ValueError("gap must be a non-negative integer")
        if self.batch_size is not None and (
            isinstance(self.batch_size, bool)
            or not isinstance(self.batch_size, int)
            or self.batch_size < 1
        ):
            raise ValueError("batch_size must be positive or None")
        if not isinstance(self.shuffle_train, bool):
            raise TypeError("shuffle_train must be a boolean")
        if not isinstance(self.drop_last, bool):
            raise TypeError("drop_last must be a boolean")
        if self.drop_last and self.batch_size is None:
            raise ValueError("drop_last requires a finite batch_size")

    @staticmethod
    def _size(value: int | float, total: int) -> int:
        if isinstance(value, int):
            return value
        return ceil(total * value) if value else 0

    def _split(self, frame: pl.DataFrame, *, seed: int) -> DatasetSplit:
        del seed
        validation_size = self._size(self.validation_size, frame.height)
        test_size = self._size(self.test_size, frame.height)
        cursor = frame.height

        test_frame: pl.DataFrame | None = None
        if test_size:
            cursor -= test_size
            if cursor < 0:
                raise ValueError("test_size exceeds the available rows")
            test_frame = frame.slice(cursor, test_size)
            cursor -= self.gap

        validation_frame: pl.DataFrame | None = None
        if validation_size:
            cursor -= validation_size
            if cursor < 0:
                raise ValueError("validation_size and test_size leave no training rows")
            validation_frame = frame.slice(cursor, validation_size)
            cursor -= self.gap

        if cursor <= 0:
            raise ValueError("Split sizes and gaps leave no training rows")
        train_frame = frame.slice(0, cursor)

        return DatasetSplit(
            train=FrameDataset(
                train_frame,
                batch_size=self.batch_size,
                shuffle=self.shuffle_train,
                drop_last=self.drop_last,
            ),
            validation=(
                None
                if validation_frame is None
                else FrameDataset(validation_frame, batch_size=self.batch_size)
            ),
            test=(
                None
                if test_frame is None
                else FrameDataset(test_frame, batch_size=self.batch_size)
            ),
        )


@dataclass(frozen=True, kw_only=True)
class Model(abc.ABC):
    """An immutable, serializable inference checkpoint."""

    VERSION: ClassVar[str]

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if not inspect.isabstract(cls):
            cls._validate_version()

    @classmethod
    def _validate_version(cls) -> str:
        if "VERSION" not in cls.__dict__:
            raise TypeError(f"Model subclass '{cls.__name__}' must define VERSION")
        if not isinstance(cls.VERSION, str):
            raise TypeError(f"Model subclass '{cls.__name__}' VERSION must be a string")
        if not cls.VERSION.strip():
            raise ValueError(f"Model subclass '{cls.__name__}' VERSION must be non-empty")
        return cls.VERSION

    def __post_init__(self) -> None:
        self._validate_version()

    @property
    def version(self) -> str:
        return self._validate_version()

    def predict(self, features: pl.Series) -> pl.Series:
        if not isinstance(features, pl.Series):
            raise TypeError("Model.predict requires a Polars Series")
        if features.is_empty():
            raise ValueError("Model.predict cannot predict an empty feature batch")
        feature_nulls = _series_null_count(features)
        if feature_nulls:
            raise ValueError(
                f"Model.predict received {feature_nulls} null feature value(s)"
            )
        output = self._predict(features)
        if not isinstance(output, pl.Series):
            raise TypeError("Model._predict must return a Polars Series")
        if len(output) != len(features):
            raise ValueError("Model prediction must preserve feature row count")
        output_nulls = _series_null_count(output)
        if output_nulls:
            raise ValueError(
                f"Model._predict returned {output_nulls} null prediction value(s)"
            )
        return output.rename("prediction")

    @abc.abstractmethod
    def _predict(self, features: pl.Series) -> pl.Series:
        pass

    def to_dict(self) -> dict[str, Any]:
        return {
            "module": type(self).__module__,
            "qualname": type(self).__qualname__,
            "version": self.version,
            "state": {
                item.name: _serialize_value(getattr(self, item.name))
                for item in fields(self)
            },
        }

    def __str__(self) -> str:
        return _canonical_json(self.to_dict())

    def __repr__(self) -> str:
        state = ", ".join(
            f"{item.name}={getattr(self, item.name)!r}" for item in fields(self)
        )
        suffix = f", {state}" if state else ""
        return f"{type(self).__name__}(version={self.version!r}{suffix})"


@dataclass(frozen=True, kw_only=True)
class SupervisedModel(Model, abc.ABC):
    """A checkpoint that can fit supervised data and return a new checkpoint."""

    def fit(self, datasets: DatasetSplit, *, seed: int) -> SupervisedModel:
        if not isinstance(datasets, DatasetSplit):
            raise TypeError("SupervisedModel.fit requires a DatasetSplit")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("SupervisedModel fit seed must be an integer")

        checkpoint = self._fit(
            datasets.train,
            datasets.validation,
            seed=seed,
        )
        if not isinstance(checkpoint, SupervisedModel):
            raise TypeError("SupervisedModel._fit must return a SupervisedModel")
        if checkpoint is self:
            raise ValueError("SupervisedModel._fit must return a new checkpoint")
        return checkpoint

    @abc.abstractmethod
    def _fit(
        self,
        train: Dataset,
        validation: Dataset | None,
        *,
        seed: int,
    ) -> SupervisedModel:
        pass


MetricItems = tuple[tuple[str, float], ...]


def _normalize_metrics(metrics: Mapping[str, float]) -> MetricItems:
    if not isinstance(metrics, Mapping):
        raise TypeError("Segment metrics must be a mapping")
    normalized: list[tuple[str, float]] = []
    for name, value in metrics.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Metric names must be non-empty strings")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"Metric '{name}' must be numeric")
        numeric = float(value)
        if not isfinite(numeric):
            raise ValueError(f"Metric '{name}' must be finite")
        normalized.append((name, numeric))
    return tuple(sorted(normalized))


@dataclass(frozen=True)
class ScheduleContext:
    """Immutable state observed when selecting the next prediction segment."""

    total_rows: int
    rows_seen: int
    rows_since_retrain: int
    retrain_count: int
    metrics: MetricItems = ()

    def __post_init__(self) -> None:
        for name in (
            "total_rows",
            "rows_seen",
            "rows_since_retrain",
            "retrain_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"ScheduleContext.{name} must be non-negative")
        if self.rows_seen > self.total_rows:
            raise ValueError("ScheduleContext.rows_seen cannot exceed total_rows")
        if self.rows_since_retrain > self.rows_seen:
            raise ValueError(
                "ScheduleContext.rows_since_retrain cannot exceed rows_seen"
            )
        normalized = _normalize_metrics(dict(self.metrics))
        if len(normalized) != len(self.metrics):
            raise ValueError("ScheduleContext metric names must be unique")
        object.__setattr__(self, "metrics", normalized)

    def metric(self, name: str) -> float | None:
        return dict(self.metrics).get(name)

    def __str__(self) -> str:
        return _canonical_json(self)


@dataclass(frozen=True)
class ScheduleDecision:
    """Whether to retrain now and the exclusive end of the next batch."""

    retrain: bool
    predict_until: int

    def __post_init__(self) -> None:
        if not isinstance(self.retrain, bool):
            raise TypeError("ScheduleDecision.retrain must be a boolean")
        if (
            isinstance(self.predict_until, bool)
            or not isinstance(self.predict_until, int)
            or self.predict_until < 1
        ):
            raise ValueError("ScheduleDecision.predict_until must be positive")

    def __str__(self) -> str:
        return _canonical_json(self)


class Scheduler(abc.ABC):
    """A pure policy over model-transition boundaries, never individual rows."""

    def decide(self, context: ScheduleContext) -> ScheduleDecision:
        if not isinstance(context, ScheduleContext):
            raise TypeError("Scheduler requires a ScheduleContext")
        if context.rows_seen >= context.total_rows:
            raise ValueError("Scheduler cannot advance a completed execution")
        decision = self._decide(context)
        if not isinstance(decision, ScheduleDecision):
            raise TypeError("Scheduler._decide must return a ScheduleDecision")
        if decision.predict_until <= context.rows_seen:
            raise ValueError("Scheduler decision must advance the row cursor")
        if decision.predict_until > context.total_rows:
            raise ValueError("Scheduler decision exceeds the available rows")
        if decision.retrain and context.rows_seen == 0:
            raise ValueError("Scheduler cannot retrain without historical rows")
        return decision

    @abc.abstractmethod
    def _decide(self, context: ScheduleContext) -> ScheduleDecision:
        pass

    def __str__(self) -> str:
        return _canonical_json(self)


@dataclass(frozen=True)
class FrozenScheduler(Scheduler):
    def _decide(self, context: ScheduleContext) -> ScheduleDecision:
        return ScheduleDecision(False, context.total_rows)


@dataclass(frozen=True)
class EveryNTicksScheduler(Scheduler):
    every: int

    def __post_init__(self) -> None:
        if isinstance(self.every, bool) or not isinstance(self.every, int) or self.every < 1:
            raise ValueError("EveryNTicksScheduler.every must be a positive integer")

    def _decide(self, context: ScheduleContext) -> ScheduleDecision:
        retrain = (
            context.rows_seen > 0
            and context.rows_since_retrain >= self.every
        )
        rows_to_boundary = (
            self.every if retrain else self.every - context.rows_since_retrain
        )
        return ScheduleDecision(
            retrain,
            min(context.rows_seen + rows_to_boundary, context.total_rows),
        )


@dataclass(frozen=True)
class MetricThresholdScheduler(Scheduler):
    """Retrain when the previous segment's named metric exceeds a threshold."""

    metric_name: str
    threshold: float
    check_every: int

    def __post_init__(self) -> None:
        if not isinstance(self.metric_name, str) or not self.metric_name:
            raise ValueError("metric_name must be a non-empty string")
        if isinstance(self.threshold, bool) or not isinstance(
            self.threshold, (int, float)
        ):
            raise TypeError("threshold must be numeric")
        if not isfinite(float(self.threshold)):
            raise ValueError("threshold must be finite")
        if (
            isinstance(self.check_every, bool)
            or not isinstance(self.check_every, int)
            or self.check_every < 1
        ):
            raise ValueError("check_every must be a positive integer")

    def _decide(self, context: ScheduleContext) -> ScheduleDecision:
        observed = context.metric(self.metric_name)
        return ScheduleDecision(
            retrain=(observed is not None and observed > self.threshold),
            predict_until=min(
                context.rows_seen + self.check_every,
                context.total_rows,
            ),
        )


@dataclass(frozen=True)
class AnyScheduler(Scheduler):
    schedulers: tuple[Scheduler, ...]

    def __post_init__(self) -> None:
        if not self.schedulers:
            raise ValueError("AnyScheduler requires at least one scheduler")
        if not all(isinstance(item, Scheduler) for item in self.schedulers):
            raise TypeError("AnyScheduler entries must be Scheduler instances")

    def _decide(self, context: ScheduleContext) -> ScheduleDecision:
        decisions = tuple(item.decide(context) for item in self.schedulers)
        return ScheduleDecision(
            retrain=any(item.retrain for item in decisions),
            predict_until=min(item.predict_until for item in decisions),
        )


class TSFN(abc.ABC):
    CONFIG_CLS: ClassVar[Type[TSFNConfig]] = TSFNConfig
    VERSION: ClassVar[str]
    REQUIRES_MATERIALIZATION: ClassVar[bool] = False
    DEFAULT_NULL_POLICY: ClassVar[NullPolicy] = NullPolicy.PROPAGATE

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if not inspect.isabstract(cls):
            cls._validate_config_cls()
            cls._validate_version()
            cls._validate_materialization_requirement()
            cls._validate_default_null_policy()

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

    @classmethod
    def _validate_materialization_requirement(cls) -> bool:
        required = cls.REQUIRES_MATERIALIZATION
        if not isinstance(required, bool):
            raise TypeError(
                f"{cls.__name__}.REQUIRES_MATERIALIZATION must be a boolean"
            )
        return required

    @classmethod
    def _validate_default_null_policy(cls) -> NullPolicy:
        policy = cls.DEFAULT_NULL_POLICY
        if not isinstance(policy, NullPolicy):
            raise TypeError(
                f"{cls.__name__}.DEFAULT_NULL_POLICY must be a NullPolicy"
            )
        return policy

    @property
    def requires_materialization(self) -> bool:
        return self._validate_materialization_requirement()

    def __init__(
        self,
        parameters: dict[str, Any],
    ):
        self._validate_config_cls()
        self.version = self._validate_version()
        self._validate_materialization_requirement()
        self._validate_default_null_policy()
        self.parameters = self._bind_and_validate_config(parameters)
        self.signature = self.type_signature()
        self._input_null_handlers = MappingProxyType({})
        self._input_null_fill_values = MappingProxyType({})

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

    def input_null_handler(
        self,
        column_name: str,
    ) -> NullHandler:
        return self._input_null_handlers.get(
            column_name,
            NullHandler.from_policy(self.DEFAULT_NULL_POLICY),
        )

    def input_null_policy(self, column_name: str) -> NullPolicy:
        handler = self.input_null_handler(column_name)
        if handler.policy is None:
            raise TypeError(
                f"Column '{column_name}' uses a custom null handler, not a NullPolicy"
            )
        return handler.policy

    def input_null_fill_value(self, column_name: str) -> Any:
        return self._input_null_fill_values.get(column_name, _MISSING_FILL_VALUE)

    def _configure_input_nulls(
        self,
        handlers: Mapping[str, NullHandler],
        fill_values: Mapping[str, Any],
    ) -> None:
        self._input_null_handlers = MappingProxyType(dict(handlers))
        self._input_null_fill_values = MappingProxyType(dict(fill_values))

    def __call__(self, lf: pl.LazyFrame | None = None) -> pl.LazyFrame:
        input_signature = self.signature[0]
        self.validate_input_schema(lf)

        if input_signature.is_empty():
            output_lf = self.apply()
        else:
            if lf is None:
                raise ValueError("Missing required input frame")
            output_lf = self.apply(self._prepare_input_nulls(lf))

        self.validate_output_schema(output_lf)
        return output_lf

    def series_to_numpy(
        self,
        series: pl.Series,
        shape: tuple[int, ...] | None = None,
        *,
        allow_copy: bool = False,
    ):
        series = self._prepare_series_for_bridge(series)
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
        self,
        name: str,
        array,
        *,
        dtype: pl.DataType = pl.Float64,
        shape: tuple[int, ...] | None = None,
        allow_copy: bool = False,
    ) -> pl.Series:
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
            series = pl.from_arrow(arrow_array).alias(name)
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
        series = pl.from_arrow(arrow_array).alias(name)
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
        self,
        series: pl.Series,
        shape: tuple[int, ...] | None = None,
        *,
        allow_copy: bool = False,
    ):
        """Expose a Series buffer to Torch; the returned tensor is borrowed input."""
        series = self._prepare_series_for_bridge(series)

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
            tensor = torch.from_dlpack(arrow_values)
        except (BufferError, RuntimeError, TypeError):
            if not allow_copy:
                raise ValueError(
                    f"Cannot expose Series '{series.name}' to Torch through DLPack "
                    "without copying"
                ) from None
            array = series.to_numpy(allow_copy=True)
            tensor = torch.from_numpy(array)

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
        self,
        name: str,
        tensor,
        *,
        dtype: pl.DataType = pl.Float64,
        shape: tuple[int, ...] | None = None,
        allow_copy: bool = False,
    ) -> pl.Series:
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
        return self.numpy_to_series(
            name,
            array,
            dtype=dtype,
            shape=shape,
            allow_copy=allow_copy,
        )

    def series_to_pytorch(
        self,
        series: pl.Series,
        shape: tuple[int, ...] | None = None,
        *,
        allow_copy: bool = False,
    ):
        return self.series_to_torch(
            series,
            shape,
            allow_copy=allow_copy,
        )

    def pytorch_to_series(
        self,
        name: str,
        tensor,
        *,
        dtype: pl.DataType = pl.Float64,
        shape: tuple[int, ...] | None = None,
        allow_copy: bool = False,
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
    ) -> pl.Series:
        null_count = _series_null_count(series)
        if null_count:
            raise ValueError(
                f"Cannot bridge column '{series.name}' with {null_count} null "
                "value(s); input null policy must resolve them before conversion"
            )
        return series

    def _prepare_input_nulls(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Apply every configured input null policy before subclass execution."""
        input_columns = _column_signature_map(self.signature[0])
        prepared_lf = lf
        requires_eager_handling = False

        for column_name in input_columns:
            column = input_columns[column_name]
            handler = self.input_null_handler(column_name)
            if handler.policy is None:
                requires_eager_handling = True
                continue
            policy = handler.policy
            if policy is NullPolicy.ERROR:
                requires_eager_handling = True
                continue
            if policy in (NullPolicy.PROPAGATE, NullPolicy.PASS):
                continue

            if policy is NullPolicy.DROP:
                prepared_lf = prepared_lf.filter(
                    ~_column_null_expr(column.name, column.physical_dtype)
                )
                continue

            fill_value = self.input_null_fill_value(column.name)
            if fill_value is _MISSING_FILL_VALUE:
                raise ValueError(
                    f"NullPolicy.FILL for column '{column.name}' requires a fill value"
                )
            prepared_lf = prepared_lf.with_columns(
                _fill_column_null_expr(column.name, column.physical_dtype, fill_value)
            )

        if requires_eager_handling:
            schema = prepared_lf.collect_schema()
            prepared_lf = prepared_lf.map_batches(
                self._apply_eager_null_handlers,
                predicate_pushdown=False,
                projection_pushdown=False,
                slice_pushdown=False,
                schema=schema,
                validate_output_schema=True,
                streamable=False,
            )
        return prepared_lf

    def _apply_eager_null_handlers(self, frame: pl.DataFrame) -> pl.DataFrame:
        prepared = frame
        for column in _column_signatures(self.signature[0]):
            handler = self.input_null_handler(column.name)
            if handler.policy not in (None, NullPolicy.ERROR):
                continue

            series = prepared.get_column(column.name)
            null_count = _series_null_count(series)
            if null_count == 0:
                continue
            if handler.policy is NullPolicy.ERROR:
                raise ValueError(
                    f"NullPolicy.ERROR failed for column '{column.name}': "
                    f"{null_count} null value(s) encountered"
                )

            handled = handler.function(series)
            if not isinstance(handled, pl.Series):
                raise TypeError(
                    f"Custom null handler for column '{column.name}' must return a "
                    "Polars Series"
                )
            if len(handled) != len(series):
                raise ValueError(
                    f"Custom null handler for column '{column.name}' must preserve "
                    "row count"
                )
            if handled.name != column.name:
                handled = handled.rename(column.name)
            prepared = prepared.with_columns(handled)

        return prepared

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
    """Base for non-streaming, full-frame batch UDF transformations.

    Polars lends the callback a DataFrame backed by its existing buffers. Batch
    implementations must treat that input as immutable and return a frame matching
    the complete output FrameSignature.
    """

    REQUIRES_MATERIALIZATION: ClassVar[bool] = True
    DEFAULT_NULL_POLICY: ClassVar[NullPolicy] = NullPolicy.ERROR

    @abc.abstractmethod
    def batch(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Transform one complete input frame into the declared output frame."""
        pass

    def execute_batch(self, frame: pl.DataFrame) -> pl.DataFrame:
        output = self.batch(frame)
        if not isinstance(output, pl.DataFrame):
            raise TypeError(
                f"{self.__class__.__name__}.batch must return a Polars DataFrame, "
                f"got {type(output).__name__}"
            )
        return output

    def prepare_batch_input(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        input_signature = self.signature[0]
        if input_signature.time is None:
            raise ValueError("BatchTSFN input signature must declare a time axis")
        input_schema = _frame_physical_schema(input_signature)
        return lf.select(*input_schema.keys())

    def apply(self, lf: pl.LazyFrame | None = None) -> pl.LazyFrame:
        if lf is None:
            raise ValueError("BatchTSFN requires an input frame")

        output_signature = self.signature[1]
        output_schema = _frame_physical_schema(output_signature)
        return self.prepare_batch_input(lf).map_batches(
            self.execute_batch,
            predicate_pushdown=False,
            projection_pushdown=False,
            slice_pushdown=False,
            schema=output_schema,
            validate_output_schema=True,
            streamable=False,
        )


class SupervisedModelTSFN(BatchTSFN, abc.ABC):
    """Batched walk-forward orchestration for supervised checkpoints.

    The graph supplies exactly one packed ``features`` column and one ``target``
    column. Python advances between scheduler boundaries; prediction and fitting
    operate on whole Series or Dataset batches rather than individual rows.
    """

    FEATURE_COLUMN: ClassVar[str] = "features"
    TARGET_COLUMN: ClassVar[str] = "target"
    PREDICTION_COLUMN: ClassVar[str] = "prediction"

    def __init__(self, parameters: dict[str, Any]) -> None:
        super().__init__(parameters)
        self._validate_supervised_contract()

    @abc.abstractmethod
    def initial_model(self) -> SupervisedModel:
        pass

    @abc.abstractmethod
    def scheduler(self) -> Scheduler:
        pass

    @abc.abstractmethod
    def splitter(self) -> DatasetSplitter:
        pass

    @abc.abstractmethod
    def training_seed(self, retrain_count: int) -> int:
        pass

    def segment_metrics(
        self,
        target: pl.Series,
        prediction: pl.Series,
    ) -> Mapping[str, float]:
        del target, prediction
        return {}

    def batch(self, frame: pl.DataFrame) -> pl.DataFrame:
        if frame.is_empty():
            return pl.DataFrame(schema=_frame_physical_schema(self.signature[1]))

        time_column = self._time_column()
        ordered = (
            frame
            if frame.get_column(time_column).is_sorted()
            else frame.sort(time_column)
        )
        supervised = ordered.select(self.FEATURE_COLUMN, self.TARGET_COLUMN)

        scheduler = self.scheduler()
        splitter = self.splitter()
        if not isinstance(scheduler, Scheduler):
            raise TypeError("SupervisedModelTSFN.scheduler must return a Scheduler")
        if not isinstance(splitter, DatasetSplitter):
            raise TypeError(
                "SupervisedModelTSFN.splitter must return a DatasetSplitter"
            )

        active_model = self.initial_model()
        if not isinstance(active_model, SupervisedModel):
            raise TypeError(
                "SupervisedModelTSFN.initial_model must return a SupervisedModel"
            )

        outputs: list[pl.DataFrame] = []
        cursor = 0
        last_retrain_at = 0
        retrain_count = 0
        metrics: MetricItems = ()

        while cursor < ordered.height:
            context = ScheduleContext(
                total_rows=ordered.height,
                rows_seen=cursor,
                rows_since_retrain=cursor - last_retrain_at,
                retrain_count=retrain_count,
                metrics=metrics,
            )
            decision = scheduler.decide(context)

            if decision.retrain:
                seed = self.training_seed(retrain_count)
                if isinstance(seed, bool) or not isinstance(seed, int):
                    raise TypeError("training_seed must return an integer")
                datasets = splitter.split(supervised.slice(0, cursor), seed=seed)
                active_model = active_model.fit(datasets, seed=seed)
                last_retrain_at = cursor
                retrain_count += 1

            segment_length = decision.predict_until - cursor
            segment = ordered.slice(cursor, segment_length)
            prediction = active_model.predict(
                segment.get_column(self.FEATURE_COLUMN)
            )
            expected_prediction = self.output_column_signature(
                self.PREDICTION_COLUMN
            ).physical_dtype
            if not _dtype_matches(prediction.dtype, expected_prediction):
                raise TypeError(
                    f"Model prediction type mismatch. Expected "
                    f"{expected_prediction}, got {prediction.dtype}"
                )

            metrics = _normalize_metrics(
                self.segment_metrics(
                    segment.get_column(self.TARGET_COLUMN),
                    prediction,
                )
            )
            outputs.append(
                pl.DataFrame(
                    [
                        segment.get_column(time_column),
                        prediction.rename(self.PREDICTION_COLUMN),
                    ]
                )
            )
            cursor = decision.predict_until

        return pl.concat(outputs, how="vertical", rechunk=False)

    def _validate_supervised_contract(self) -> None:
        input_signature, output_signature = self.signature
        input_names = tuple(entry[0] for entry in input_signature.columns)
        output_names = tuple(entry[0] for entry in output_signature.columns)
        if input_names != (self.FEATURE_COLUMN, self.TARGET_COLUMN):
            raise ValueError(
                "SupervisedModelTSFN input signature must declare exactly "
                "('features', 'target')"
            )
        if output_names != (self.PREDICTION_COLUMN,):
            raise ValueError(
                "SupervisedModelTSFN output signature must declare exactly "
                "('prediction',)"
            )
        if input_signature.time is None or output_signature.time is None:
            raise ValueError("SupervisedModelTSFN requires input and output time axes")
        if input_signature.time != output_signature.time:
            raise ValueError("SupervisedModelTSFN must preserve its input time axis")

    def _time_column(self) -> str:
        time_axis = self.signature[0].time
        if time_axis is None:
            raise ValueError("SupervisedModelTSFN requires an input time axis")
        return time_axis.column


class ItemwiseStructTSFN(TSFN, abc.ABC):
    """Base for n-ary itemwise transforms lowered through a struct batch."""

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
        output = self.batch(fields)
        if not isinstance(output, pl.Series):
            raise TypeError(
                f"{self.__class__.__name__}.batch must return a Polars Series, "
                f"got {type(output).__name__}"
            )
        return output

    def apply(self, lf: pl.LazyFrame | None = None) -> pl.LazyFrame:
        if lf is None:
            raise ValueError("ItemwiseStructTSFN requires an input frame")

        input_signature, output_signature = self.signature
        if input_signature.time is None:
            raise ValueError("ItemwiseStructTSFN input signature must declare a time axis")

        input_columns = self.batch_input_columns()
        output_column = _column_signature_map(output_signature)[
            self.batch_output_column()
        ]
        batch_expr = (
            pl.struct(list(input_columns))
            .map_batches(
                self._batch_from_struct,
                return_dtype=output_column.physical_dtype,
                is_elementwise=True,
            )
            .alias(output_column.name)
        )
        return lf.select(input_signature.time.column, batch_expr)

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
        self.function._configure_input_nulls(
            self.null_handlers,
            self.null_fill_values,
        )

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
            f"fn={function_name}@{version}, "
            f"materialize={attrs.get('materialize', '<uninitialized>')!r})"
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
            "parameters": self.parameters.to_dict(),
            "outputs": sorted([(name, str(dtype)) for name, dtype in self.outputs.items()]),
        }

        serialized_data = _canonical_json(node_definition)
        return hashlib.sha256(serialized_data.encode("utf-8")).hexdigest()


class Executor(abc.ABC):
    """Lower and execute a verified graph.

    The base executor owns the Polars lowering contract, including temporal input
    alignment. Concrete executors decide how a lazy result is materialized.
    """

    def execute(self, graph: Graph) -> pl.DataFrame:
        graph.verify()
        root_lf = self._evaluate_to_root(graph)
        return self.materialize(graph.root_node, root_lf)

    def _evaluate_to_root(self, graph: Graph) -> pl.LazyFrame:
        """Evaluate required boundaries and return the root lazy value."""
        graph.verify()
        results: dict[str, pl.LazyFrame] = {}

        for node in graph.node_list:
            try:
                node_input_lf = (
                    None
                    if not node.bindings
                    else self.align_inputs(node, results)
                )
                results[node.ID] = self.lower_node(node, node_input_lf)
            except Exception as exc:
                raise RuntimeError(
                    f"Execution failed at node '{node.name or node.ID[:8]}' "
                    f"({node.function_cls.__name__}@{node.function.version}): {exc}"
                ) from exc

            if (
                node.ID in graph.materialized_node_ids
                and node.ID != graph.root_node.ID
            ):
                results[node.ID] = self.materialize(node, results[node.ID]).lazy()

        return results[graph.root_node.ID]

    def lower_node(
        self,
        node: Node,
        input_lf: pl.LazyFrame | None,
    ) -> pl.LazyFrame:
        if input_lf is None:
            return node.function()
        return node.function(input_lf)

    def align_inputs(
        self,
        node: Node,
        results: Mapping[str, pl.LazyFrame],
    ) -> pl.LazyFrame:
        """Build a union timeline and backward-asof align a node's inputs."""
        parent_to_bindings: dict[
            tuple[Node, AsofTolerance],
            list[tuple[str, str]],
        ] = defaultdict(list)
        for input_name, (parent_node, parent_col) in node.bindings.items():
            tolerance = node.tolerances.get(input_name)
            parent_to_bindings[(parent_node, tolerance)].append(
                (parent_col, input_name)
            )

        input_time = node.function.signature[0].time
        if input_time is None:
            raise ValueError(
                f"Bound node '{node.name or node.ID}' must declare an input time axis"
            )
        time_col = input_time.column

        parent_frames: list[tuple[pl.LazyFrame, AsofTolerance]] = []
        for (parent_node, tolerance), binds in parent_to_bindings.items():
            parent_lf = results[parent_node.ID]
            parent_time = parent_node.function.signature[1].time
            if parent_time is None:
                raise ValueError(
                    f"Parent node '{parent_node.name or parent_node.ID}' "
                    "must declare an output time axis"
                )
            select_exprs = [pl.col(parent_time.column).alias(time_col)]
            select_exprs.extend(
                pl.col(parent_col).alias(input_name)
                for parent_col, input_name in binds
            )
            parent_frames.append(
                (parent_lf.select(select_exprs).sort(time_col), tolerance)
            )

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
        return node_input_lf

    @abc.abstractmethod
    def materialize(self, node: Node, lf: pl.LazyFrame) -> pl.DataFrame:
        """Materialize one graph boundary into the executor's local table type."""
        pass


class LocalExecutor(Executor):
    """Execute a graph on one machine using Polars' local query engine."""

    def materialize(self, node: Node, lf: pl.LazyFrame) -> pl.DataFrame:
        try:
            return lf.collect()
        except Exception as exc:
            raise RuntimeError(
                f"Execution failed while materializing node "
                f"'{node.name or node.ID[:8]}' "
                f"({node.function_cls.__name__}@{node.function.version}): {exc}"
            ) from exc


class Graph:
    def __init__(
        self,
        root_node: Node,
        *,
        executor: Executor | None = None,
    ):
        if not isinstance(root_node, Node):
            raise TypeError("Graph root_node must be a Node")
        if executor is not None and not isinstance(executor, Executor):
            raise TypeError("Graph executor must be an Executor")

        self.root_node = root_node
        self._declared_nodes = self.get_declared_nodes(self.root_node)
        self.node_list = self.get_subgraph_execution_order(self.root_node)
        self.executor = LocalExecutor() if executor is None else executor
        self.materialized_node_ids = frozenset(
            node.ID for node in self._declared_nodes if node.materialize
        )

        self.verify()
        self.ID = self._generate_persistent_id()

    def __repr__(self) -> str:
        return (
            f"Graph(id={self.ID[:8]!r}, root={self.root_node.ID[:8]!r}, "
            f"nodes={len(self.node_list)}, "
            f"executor={self.executor.__class__.__name__})"
        )

    def verify(self) -> None:
        """Verify graph structure and type contracts without lowering or executing."""
        self._validate_materializations()
        self._validate_graph()

    def _validate_materializations(self) -> None:
        for node in self._declared_nodes:
            if node.function.requires_materialization and not node.materialize:
                raise ValueError(
                    f"Materialization validation failed for node "
                    f"'{node.name or node.ID}': {node.function_cls.__name__} "
                    "requires materialization"
                )

    def get_declared_nodes(self, target_node: Node) -> tuple[Node, ...]:
        """Return every concrete node declaration, including semantic duplicates."""
        visited_objects: set[int] = set()
        declared_nodes: list[Node] = []

        def visit(node: Node) -> None:
            object_id = id(node)
            if object_id in visited_objects:
                return
            visited_objects.add(object_id)
            for parent, _ in node.bindings.values():
                visit(parent)
            declared_nodes.append(node)

        visit(target_node)
        return tuple(declared_nodes)

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

    def execute(self, executor: Executor | None = None) -> pl.DataFrame:
        selected_executor = self.executor if executor is None else executor
        if not isinstance(selected_executor, Executor):
            raise TypeError("Graph executor must be an Executor")
        return selected_executor.execute(self)
