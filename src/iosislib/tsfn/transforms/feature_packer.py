from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import polars as pl

from iosislib.core.tsfn import FrameSignature, TSFN, TSFNConfig, TimeAxis
from iosislib.tsfn.transforms._validation import (
    validate_column_name,
    validate_distinct_columns,
)


@dataclass(frozen=True)
class FeaturePackerConfig(TSFNConfig):
    input_columns: tuple[str, ...]
    output_column: str = "features"
    timestamp_column: str = "timestamp"

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


class FeaturePacker(TSFN[FeaturePackerConfig]):
    VERSION = "0.1.0"
    CONFIG_CLS = FeaturePackerConfig

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        params = self.parameters
        width = len(params.input_columns)
        input_frame = FrameSignature(
            time=TimeAxis(column=params.timestamp_column),
            columns=tuple((name, pl.Float64) for name in params.input_columns),
        )
        output_frame = FrameSignature(
            time=TimeAxis(column=params.timestamp_column),
            columns=((params.output_column, pl.Float64, (width,)),),
        )
        return input_frame, output_frame

    def apply(self, lf: pl.LazyFrame | None = None) -> pl.LazyFrame:
        if lf is None:
            raise ValueError("FeaturePacker requires an input frame")
        params = self.parameters
        features = (
            pl.concat_list([pl.col(name) for name in params.input_columns])
            .cast(pl.Array(pl.Float64, len(params.input_columns)))
            .alias(params.output_column)
        )
        return lf.select(params.timestamp_column, features)
