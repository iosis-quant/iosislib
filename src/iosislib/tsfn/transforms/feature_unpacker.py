from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import polars as pl

from iosislib.core.tsfn import (
    ColumnSignature,
    FrameSignature,
    TSFN,
    TSFNConfig,
    TimeAxis,
    _column_signature_map,
    _replace_column,
)
from iosislib.core.utils import _flat_size
from iosislib.tsfn.transforms._validation import (
    validate_column_name,
    validate_distinct_columns,
)


@dataclass(frozen=True)
class FeatureUnpackerConfig(TSFNConfig):
    input_column: str = "features"
    base_name: str = "feature"
    timestamp_column: str = "timestamp"
    features: int | None = None

    def __post_init__(self) -> None:
        validate_column_name(self.input_column, field_name="input_column")
        validate_column_name(self.base_name, field_name="base_name")
        validate_column_name(self.timestamp_column, field_name="timestamp_column")
        validate_distinct_columns(self.timestamp_column, self.input_column)
        if self.features is not None:
            if isinstance(self.features, bool) or not isinstance(self.features, int):
                raise TypeError("features must be a positive integer or None")
            if self.features < 1:
                raise ValueError("features must be a positive integer")


class FeatureUnpacker(TSFN[FeatureUnpackerConfig]):
    VERSION = "0.1.0"
    CONFIG_CLS = FeatureUnpackerConfig

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        params = self.parameters
        width = params.features if params.features is not None else 1
        time = TimeAxis(column=params.timestamp_column)
        input_frame = FrameSignature(
            time=time,
            columns=((params.input_column, pl.Float64, (width,)),),
        )
        output_frame = FrameSignature(
            time=time,
            columns=tuple(
                (f"{params.base_name}_{index}", pl.Float64)
                for index in range(width)
            ),
        )
        return input_frame, output_frame

    def resolve_signature(
        self,
        bound_input_columns: Mapping[str, ColumnSignature],
    ) -> tuple[FrameSignature, FrameSignature]:
        params = self.parameters
        input_signature, output_signature = self.signature
        if input_signature.time is None or output_signature.time is None:
            raise ValueError("FeatureUnpacker requires input and output time axes")

        declared_input = _column_signature_map(input_signature)[params.input_column]
        bound = bound_input_columns.get(params.input_column)
        if bound is not None:
            shape = bound.shape
            dtype = bound.dtype
            width = _flat_size(shape)
            if params.features is not None and params.features != width:
                raise ValueError(
                    f"Configured features {params.features} does not match the "
                    f"bound width {width}"
                )
        else:
            if params.features is None:
                raise ValueError(
                    f"{params.input_column} must be connected in the graph or have "
                    "configured features"
                )
            width = params.features
            shape = (width,)
            dtype = declared_input.dtype

        resolved_input = ColumnSignature(declared_input.name, dtype, shape)
        output_columns = tuple(
            (f"{params.base_name}_{index}", pl.Float64)
            for index in range(width)
        )
        return (
            FrameSignature(
                time=input_signature.time,
                columns=_replace_column(
                    input_signature, params.input_column, resolved_input
                ),
            ),
            FrameSignature(time=output_signature.time, columns=output_columns),
        )

    def apply(self, lf: pl.LazyFrame | None = None) -> pl.LazyFrame:
        if lf is None:
            raise ValueError("FeatureUnpacker requires an input frame")
        params = self.parameters
        input_signature, _ = self.signature
        if input_signature.time is None:
            raise ValueError(
                "FeatureUnpacker input signature must declare a time axis"
            )

        input_column = _column_signature_map(input_signature)[params.input_column]
        width = _flat_size(input_column.shape)
        outputs = []
        for index in range(width):
            if input_column.shape:
                expr = pl.col(params.input_column).arr.get(index)
            else:
                expr = pl.col(params.input_column)
            outputs.append(
                expr.cast(pl.Float64).alias(f"{params.base_name}_{index}")
            )

        return lf.sort(input_signature.time.column).select(
            input_signature.time.column, *outputs
        )


__all__ = ["FeatureUnpacker", "FeatureUnpackerConfig"]
