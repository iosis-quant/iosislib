from __future__ import annotations

import abc
import json
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from enum import Enum
from typing import Any, ClassVar, Generic, TypeAlias, TypeVar, cast
from collections.abc import Callable, Mapping
import inspect
from types import MappingProxyType

import polars as pl

from iosislib.core.utils import (
    _canonical_json,
    _datetime_dtype_without_timezone,
    _dtype_matches,
    _flat_size,
    _is_array_instance,
    _is_dtype_class,
    _is_list_instance,
    _normalize_identity_value,
    _normalize_shape,
    _serialize_value,
    _series_null_count,
)


PolarsDataType: TypeAlias = pl.DataType | type[pl.DataType]

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
    version: str | None = None

    def __post_init__(self) -> None:
        has_policy = self.policy is not None
        has_function = self.function is not None
        if has_policy == has_function:
            raise ValueError("NullHandler must wrap exactly one policy or function")
        if has_policy:
            policy = self.policy
            assert policy is not None
            if self.version is not None:
                raise ValueError("Policy null handlers cannot declare a version")
            object.__setattr__(self, "policy", _normalize_null_policy(policy))
        if has_function:
            function = self.function
            assert function is not None
            _validate_null_handler_function(function)
            if self.version is not None and (
                not isinstance(self.version, str) or not self.version.strip()
            ):
                raise ValueError(
                    "Custom null handler version must be a non-empty string"
                )

    @classmethod
    def from_policy(cls, policy: NullPolicy | str) -> NullHandler:
        return cls(policy=_normalize_null_policy(policy))

    @classmethod
    def from_function(
        cls,
        function: Callable[[pl.Series], pl.Series],
        *,
        version: str | None = None,
    ) -> NullHandler:
        return cls(function=function, version=version)

    @property
    def is_custom(self) -> bool:
        return self.function is not None


@dataclass(frozen=True)
class TimeAxis:
    column: str = "timestamp"
    dtype: PolarsDataType = pl.Datetime
    timezone: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.column, str):
            raise TypeError("time column must be a string")
        if not self.column.strip():
            raise ValueError("time column must be non-empty")
        dtype = cast(pl.DataType, self.dtype)
        if not (_is_dtype_class(dtype) or isinstance(dtype, pl.DataType)):
            raise TypeError("time axis dtype must be a Polars data type")
        if self.timezone is not None and not isinstance(self.timezone, str):
            raise TypeError("time axis timezone must be a string or None")
        if self.timezone is not None and not (
            self.dtype is pl.Datetime
            or (
                not _is_dtype_class(dtype)
                and isinstance(self.dtype, pl.Datetime)
            )
        ):
            raise TypeError("A timezone can only be declared for a Datetime time axis")


@dataclass(frozen=True)
class ColumnSignature:
    name: str
    dtype: PolarsDataType
    shape: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError("column name must be a string")
        if not self.name.strip():
            raise ValueError("column name must be non-empty")
        shape = _normalize_shape(self.shape)
        _validate_column_dtype(self.dtype, shape)
        object.__setattr__(self, "shape", shape)

    @property
    def physical_dtype(self) -> PolarsDataType:
        return _column_physical_dtype(self)


ColumnEntry = (
    tuple[str, PolarsDataType]
    | tuple[str, PolarsDataType, tuple[int, ...]]
)


@dataclass(frozen=True)
class FrameSignature:
    time: TimeAxis | None = field(default_factory=TimeAxis)
    columns: tuple[ColumnEntry | ColumnSignature, ...] = ()

    @classmethod
    def empty(cls) -> FrameSignature:
        """Return the explicit signature for a TSFN that consumes no input frame."""
        return cls(time=None, columns=())

    def __post_init__(self) -> None:
        if self.time is not None and not isinstance(self.time, TimeAxis):
            raise TypeError("time must be a TimeAxis or None")
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


def _validate_column_dtype(dtype: PolarsDataType, shape: tuple[int, ...]) -> None:
    dtype_value = cast(pl.DataType, dtype)
    if not (_is_dtype_class(dtype_value) or isinstance(dtype_value, pl.DataType)):
        raise TypeError("column dtype must be a Polars data type")
    if shape and (
        dtype is pl.Array
        or dtype is pl.List
        or _is_array_instance(dtype_value)
        or _is_list_instance(dtype_value)
    ):
        raise TypeError("shaped columns must declare an element dtype, not Array/List")


def _column_physical_dtype(column: ColumnEntry | ColumnSignature) -> PolarsDataType:
    column = _column_signature(column)
    if not column.shape:
        return column.dtype
    return pl.Array(column.dtype, _flat_size(column.shape))


def _time_axis_physical_dtype(time_axis: TimeAxis) -> PolarsDataType:
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


def _frame_physical_schema(signature: FrameSignature) -> dict[str, PolarsDataType]:
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
    return actual.shape == expected.shape and _dtype_matches(
        cast(pl.DataType, actual.dtype),
        cast(pl.DataType, expected.dtype),
    )


def _is_array_dtype(dtype: PolarsDataType) -> bool:
    return _is_array_instance(cast(pl.DataType, dtype))


def _column_null_expr(column_name: str, dtype: PolarsDataType) -> pl.Expr:
    column = pl.col(column_name)
    if _is_array_dtype(dtype):
        inner_nulls = column.arr.eval(pl.element().is_null()).arr.any().fill_null(False)
        return column.is_null() | inner_nulls
    return column.is_null()


def _fill_column_null_expr(
    column_name: str,
    dtype: PolarsDataType,
    fill_value: Any,
) -> pl.Expr:
    column = pl.col(column_name)
    if _is_array_dtype(dtype):
        array_dtype = cast(pl.Array, dtype)
        fill_array = [fill_value] * array_dtype.size
        return (
            pl.when(column.is_null())
            .then(pl.lit(fill_array, dtype=dtype))
            .otherwise(column)
            .arr.eval(pl.element().fill_null(fill_value))
            .alias(column_name)
        )
    return column.fill_null(fill_value).alias(column_name)


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


@dataclass(frozen=True)
class TSFNConfig(abc.ABC):
    def to_dict(self) -> dict[str, Any]:
        return {
            config_field.name: _serialize_value(getattr(self, config_field.name))
            for config_field in fields(self)
        }

    def __str__(self) -> str:
        return _canonical_json(self.to_dict())


ConfigT = TypeVar("ConfigT", bound=TSFNConfig)


class TSFN(abc.ABC, Generic[ConfigT]):
    CONFIG_CLS: ClassVar[type[TSFNConfig]] = TSFNConfig
    VERSION: ClassVar[str]
    REQUIRES_MATERIALIZATION: ClassVar[bool] = False
    DEFAULT_NULL_POLICY: ClassVar[NullPolicy] = NullPolicy.PROPAGATE

    def __setattr__(self, name: str, value: Any) -> None:
        if self.__dict__.get("_node_definition_frozen", False):
            raise AttributeError("Configured TSFN definitions are immutable")
        object.__setattr__(self, name, value)

    def _freeze_definition(self) -> None:
        object.__setattr__(self, "_node_definition_frozen", True)

    def __init_subclass__(cls, **kwargs: Any) -> None:
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
        parameters: Mapping[str, object] | ConfigT,
    ) -> None:
        self._validate_config_cls()
        self.version = self._validate_version()
        self._validate_materialization_requirement()
        self._validate_default_null_policy()
        self.parameters = self._bind_and_validate_config(parameters)
        self.signature = self._validate_type_signature(self.type_signature())
        self._input_null_handlers: Mapping[str, NullHandler] = MappingProxyType({})
        self._input_null_fill_values: Mapping[str, Any] = MappingProxyType({})

    def _validate_type_signature(self, signature: Any) -> tuple[
        FrameSignature,
        FrameSignature,
    ]:
        class_name = self.__class__.__name__
        if not isinstance(signature, tuple):
            raise TypeError(
                f"{class_name}.type_signature() must return a tuple of exactly two "
                "FrameSignature values"
            )
        if len(signature) != 2:
            raise ValueError(
                f"{class_name}.type_signature() must return exactly two items, "
                f"got {len(signature)}"
            )
        for index, frame_signature in enumerate(signature):
            if not isinstance(frame_signature, FrameSignature):
                raise TypeError(
                    f"{class_name}.type_signature() item {index} must be a "
                    f"FrameSignature, got {type(frame_signature).__name__}"
                )
        return signature

    def _bind_and_validate_config(
        self,
        params: Mapping[str, object] | ConfigT,
    ) -> ConfigT:
        config_cls = self._validate_config_cls()
        if isinstance(params, TSFNConfig):
            if type(params) is not config_cls:
                raise TypeError(
                    f"{self.__class__.__name__} requires config "
                    f"{config_cls.__name__}, got {type(params).__name__}"
                )
            return cast(ConfigT, _normalize_identity_value(params))
        if not isinstance(params, Mapping):
            raise TypeError(
                f"{self.__class__.__name__} parameters must be a mapping or "
                f"{config_cls.__name__}"
            )

        normalized_params = cast(
            Mapping[str, object],
            _normalize_identity_value(params),
        )
        config_fields = fields(config_cls)
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

        return cast(
            ConfigT,
            config_cls(**cast(dict[str, Any], dict(normalized_params))),
        )

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
            _datetime_dtype_without_timezone(cast(pl.DataType, time_axis.dtype)),
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
            if not _dtype_matches(actual_type, cast(pl.DataType, expected_type)):
                raise TypeError(
                    f"Column '{column.name}' type mismatch. "
                    f"Expected {expected_type} ({_format_column_signature(column)}), "
                    f"got {actual_type}"
                )

        if schema_name == "output":
            expected_columns = list(_frame_physical_schema(signature))
            actual_columns = list(current_schema)
            if actual_columns != expected_columns:
                raise ValueError(
                    "Output schema columns must exactly match the declared order. "
                    f"Expected {expected_columns}, got {actual_columns}"
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

        if not isinstance(output_lf, pl.LazyFrame):
            raise TypeError(
                f"{self.__class__.__name__}.apply() must return a Polars LazyFrame, "
                f"got {type(output_lf).__name__}"
            )
        self.validate_output_schema(output_lf)
        return output_lf

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

            function = handler.function
            assert function is not None
            handled = function(series)
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


class ItemwiseUnaryTSFN(TSFN[ConfigT], abc.ABC):
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


class BatchTSFN(TSFN[ConfigT], abc.ABC):
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


class ItemwiseStructTSFN(TSFN[ConfigT], abc.ABC):
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

        resolved_input_columns = tuple(
            _column_entry(column) for column in _column_signatures(input_signature)
        )
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


class RollingUnaryTSFN(TSFN[ConfigT], abc.ABC):
    """Base for unary transforms computed over a fixed rolling window.

    Concrete subclasses name the scalar input/output columns, expose the fixed
    window size and minimum sample count, and build the lazy windowed
    expression. Inputs are scalar Float64; windowed operations over shaped
    Array columns are out of scope. The base sorts by the declared time axis
    before applying the window, matching ``Delta`` and ``Lag``.
    """

    @abc.abstractmethod
    def rolling_input_column(self) -> str:
        pass

    @abc.abstractmethod
    def rolling_output_column(self) -> str:
        pass

    @abc.abstractmethod
    def rolling_periods(self) -> int:
        pass

    @abc.abstractmethod
    def rolling_min_samples(self) -> int:
        pass

    @abc.abstractmethod
    def rolling_expr(
        self,
        value: pl.Expr,
        *,
        periods: int,
        min_samples: int,
    ) -> pl.Expr:
        pass

    def apply(self, lf: pl.LazyFrame | None = None) -> pl.LazyFrame:
        if lf is None:
            raise ValueError("RollingUnaryTSFN requires an input frame")

        input_signature, output_signature = self.signature
        if input_signature.time is None:
            raise ValueError(
                "RollingUnaryTSFN input signature must declare a time axis"
            )

        input_column = _column_signature_map(input_signature)[
            self.rolling_input_column()
        ]
        output_column = _column_signature_map(output_signature)[
            self.rolling_output_column()
        ]
        result = self.rolling_expr(
            pl.col(input_column.name),
            periods=self.rolling_periods(),
            min_samples=self.rolling_min_samples(),
        )
        return lf.sort(input_signature.time.column).select(
            input_signature.time.column,
            result.cast(output_column.physical_dtype).alias(output_column.name),
        )


class EwmUnaryTSFN(TSFN[ConfigT], abc.ABC):
    """Base for unary exponentially-weighted transforms.

    Concrete subclasses name the scalar input/output columns, expose the decay
    ``alpha`` and adjustment mode, and build the lazy exponentially-weighted
    expression. The base sorts by the declared time axis before applying the
    window.
    """

    @abc.abstractmethod
    def ewm_input_column(self) -> str:
        pass

    @abc.abstractmethod
    def ewm_output_column(self) -> str:
        pass

    @abc.abstractmethod
    def ewm_alpha(self) -> float:
        pass

    @abc.abstractmethod
    def ewm_min_samples(self) -> int:
        pass

    @abc.abstractmethod
    def ewm_adjust(self) -> bool:
        pass

    @abc.abstractmethod
    def ewm_expr(
        self,
        value: pl.Expr,
        *,
        alpha: float,
        min_samples: int,
        adjust: bool,
    ) -> pl.Expr:
        pass

    def apply(self, lf: pl.LazyFrame | None = None) -> pl.LazyFrame:
        if lf is None:
            raise ValueError("EwmUnaryTSFN requires an input frame")

        input_signature, output_signature = self.signature
        if input_signature.time is None:
            raise ValueError("EwmUnaryTSFN input signature must declare a time axis")

        input_column = _column_signature_map(input_signature)[
            self.ewm_input_column()
        ]
        output_column = _column_signature_map(output_signature)[
            self.ewm_output_column()
        ]
        result = self.ewm_expr(
            pl.col(input_column.name),
            alpha=self.ewm_alpha(),
            min_samples=self.ewm_min_samples(),
            adjust=self.ewm_adjust(),
        )
        return lf.sort(input_signature.time.column).select(
            input_signature.time.column,
            result.cast(output_column.physical_dtype).alias(output_column.name),
        )


__all__ = [
    "BatchTSFN",
    "ColumnSignature",
    "EwmUnaryTSFN",
    "FrameSignature",
    "ItemwiseStructTSFN",
    "ItemwiseUnaryTSFN",
    "NullHandler",
    "NullPolicy",
    "PolarsDataType",
    "RollingUnaryTSFN",
    "TSFN",
    "TSFNConfig",
    "TimeAxis",
]
