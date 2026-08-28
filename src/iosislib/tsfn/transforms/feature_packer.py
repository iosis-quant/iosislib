from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import polars as pl

from iosislib.core.tsfn import (
    ColumnSignature,
    FrameSignature,
    PolarsDataType,
    TSFN,
    TSFNConfig,
    TimeAxis,
    _column_entry,
    _column_signature_map,
)
from iosislib.core.utils import _flat_size, _is_dtype_class
from iosislib.tsfn.transforms._validation import (
    validate_column_name,
    validate_distinct_columns,
)


_NUMERIC_DTYPE_CLASSES = frozenset(
    {
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.Int128,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
        pl.UInt128,
        pl.Float16,
        pl.Float32,
        pl.Float64,
    }
)

_FLOAT_DTYPE_CLASSES = frozenset({pl.Float32, pl.Float64})


def _is_numeric_dtype(dtype: PolarsDataType) -> bool:
    if _is_dtype_class(cast(pl.DataType, dtype)):
        return dtype in _NUMERIC_DTYPE_CLASSES
    return type(dtype) in _NUMERIC_DTYPE_CLASSES


def _is_boolean_dtype(dtype: PolarsDataType) -> bool:
    if _is_dtype_class(cast(pl.DataType, dtype)):
        return dtype is pl.Boolean
    return type(dtype) is pl.Boolean


def _is_float_dtype(dtype: PolarsDataType) -> bool:
    if _is_dtype_class(cast(pl.DataType, dtype)):
        return dtype in _FLOAT_DTYPE_CLASSES
    return type(dtype) in _FLOAT_DTYPE_CLASSES


def _validate_castable(
    dtype: PolarsDataType,
    *,
    output_dtype: PolarsDataType,
    column: str,
) -> None:
    if not (_is_numeric_dtype(dtype) or _is_boolean_dtype(dtype)):
        raise TypeError(
            f"FeaturePacker input column '{column}' must be numeric or boolean "
            f"so it can be packed as {output_dtype}, got {dtype}"
        )


@dataclass(frozen=True)
class FeaturePackerConfig(TSFNConfig):
    input_columns: tuple[str, ...]
    output_column: str = "features"
    timestamp_column: str = "timestamp"
    output_dtype: PolarsDataType = pl.Float64

    def __post_init__(self) -> None:
        if isinstance(self.input_columns, str) or not isinstance(
            self.input_columns, Sequence
        ):
            raise TypeError("input_columns must be a sequence of strings")
        columns = tuple(self.input_columns)
        if not columns:
            raise ValueError("input_columns must contain at least one column")
        if not all(isinstance(name, str) for name in columns):
            raise TypeError("input_columns must contain only strings")
        if any(not name for name in columns):
            raise ValueError("input_columns must contain non-empty strings")
        duplicate_names = sorted(
            {name for name in columns if columns.count(name) > 1}
        )
        if duplicate_names:
            raise ValueError(
                f"Duplicate input_columns are not allowed: {duplicate_names}"
            )
        object.__setattr__(self, "input_columns", columns)
        validate_column_name(self.output_column, field_name="output_column")
        validate_column_name(self.timestamp_column, field_name="timestamp_column")
        validate_distinct_columns(self.timestamp_column, self.output_column)
        if self.output_column in columns:
            raise ValueError("output_column must not be one of the input columns")
        if self.timestamp_column in columns:
            raise ValueError("timestamp_column must not be one of the input columns")
        if not _is_float_dtype(self.output_dtype):
            raise TypeError("output_dtype must be a Float32 or Float64 Polars dtype")


class FeaturePacker(TSFN[FeaturePackerConfig]):
    """Pack multiple scalar columns into a single fixed-width array column.

    The output column name is set by the ``output_column`` parameter
    (default ``"features"``). In the strategy YAML, the input mapping keys
    must match the ``input_columns`` entries, and the upstream node must
    produce columns with those names.

    Example YAML::

        nodes:
          packer:
            op: transform.feature_packer
            version: 0.2.0
            inputs:
              rolling_mean: momentum.rolling_mean
              spread: spread.spread
            params:
              input_columns: [rolling_mean, spread]
              output_column: features
    """

    VERSION = "0.2.0"
    CONFIG_CLS = FeaturePackerConfig

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        params = self.parameters
        width = len(params.input_columns)
        time = TimeAxis(column=params.timestamp_column)
        input_frame = FrameSignature(
            time=time,
            columns=tuple((name, params.output_dtype) for name in params.input_columns),
        )
        output_frame = FrameSignature(
            time=time,
            columns=((params.output_column, params.output_dtype, (width,)),),
        )
        return input_frame, output_frame

    def resolve_signature(
        self,
        bound_input_columns: Mapping[str, ColumnSignature],
    ) -> tuple[FrameSignature, FrameSignature]:
        params = self.parameters
        input_signature, output_signature = self.signature
        if input_signature.time is None or output_signature.time is None:
            raise ValueError("FeaturePacker requires input and output time axes")

        declared_columns = _column_signature_map(input_signature)
        resolved_columns = []
        total_width = 0
        for name in params.input_columns:
            declared = declared_columns[name]
            bound = bound_input_columns.get(name)
            if bound is not None:
                _validate_castable(
                    bound.dtype,
                    output_dtype=params.output_dtype,
                    column=name,
                )
                dtype = bound.dtype
                shape = bound.shape
            else:
                dtype = declared.dtype
                shape = declared.shape
            total_width += _flat_size(shape)
            resolved_columns.append(_column_entry(ColumnSignature(name, dtype, shape)))

        return (
            FrameSignature(
                time=input_signature.time,
                columns=tuple(resolved_columns),
            ),
            FrameSignature(
                time=output_signature.time,
                columns=((params.output_column, params.output_dtype, (total_width,)),),
            ),
        )

    def apply(self, lf: pl.LazyFrame | None = None) -> pl.LazyFrame:
        if lf is None:
            raise ValueError("FeaturePacker requires an input frame")
        params = self.parameters
        input_signature, _ = self.signature
        if input_signature.time is None:
            raise ValueError("FeaturePacker input signature must declare a time axis")

        input_columns = _column_signature_map(input_signature)
        output_dtype = cast(pl.DataType, params.output_dtype)

        packed = []
        for name in params.input_columns:
            column = input_columns[name]
            if column.shape:
                packed.append(
                    pl.col(name).cast(pl.Array(output_dtype, _flat_size(column.shape)))
                )
            else:
                packed.append(pl.col(name).cast(output_dtype))

        features = pl.concat_arr(*packed).alias(params.output_column)
        return lf.select(params.timestamp_column, features)
