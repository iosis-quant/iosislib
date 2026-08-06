from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import polars as pl

from iosislib.core.tsfn import FrameSignature, ItemwiseStructTSFN, TSFNConfig, TimeAxis
from iosislib.tsfn.transforms._validation import (
    validate_column_name,
    validate_distinct_columns,
)


@dataclass(frozen=True)
class LogRatioConfig(TSFNConfig):
    numerator_column: str = "left"
    denominator_column: str = "right"
    output_column: str = "log_ratio"
    timestamp_column: str = "timestamp"

    def __post_init__(self) -> None:
        validate_column_name(self.numerator_column, field_name="numerator_column")
        validate_column_name(self.denominator_column, field_name="denominator_column")
        validate_column_name(self.output_column, field_name="output_column")
        validate_column_name(self.timestamp_column, field_name="timestamp_column")
        validate_distinct_columns(self.numerator_column, self.denominator_column)
        validate_distinct_columns(self.timestamp_column, self.numerator_column)
        validate_distinct_columns(self.timestamp_column, self.denominator_column)
        validate_distinct_columns(self.timestamp_column, self.output_column)


class LogRatio(ItemwiseStructTSFN[LogRatioConfig]):
    VERSION = "0.1.0"
    CONFIG_CLS = LogRatioConfig

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        params = self.parameters
        input_frame = FrameSignature(
            time=TimeAxis(column=params.timestamp_column),
            columns=(
                (params.numerator_column, pl.Float64),
                (params.denominator_column, pl.Float64),
            ),
        )
        output_frame = FrameSignature(
            time=TimeAxis(column=params.timestamp_column),
            columns=((params.output_column, pl.Float64),),
        )
        return input_frame, output_frame

    def batch_input_columns(self) -> tuple[str, ...]:
        params = self.parameters
        return (params.numerator_column, params.denominator_column)

    def batch_output_column(self) -> str:
        return self.parameters.output_column

    def batch(self, fields: Mapping[str, pl.Series]) -> pl.Series:
        params = self.parameters
        numerator = fields[params.numerator_column]
        denominator = fields[params.denominator_column]
        log_ratio = (numerator / denominator).log()
        invalid = (
            (numerator <= 0.0)
            | (denominator <= 0.0)
            | log_ratio.is_infinite()
            | log_ratio.is_nan()
        ).fill_null(False)
        output_shape = self.output_column_signature(params.output_column).shape

        if output_shape:
            log_ratio = log_ratio.arr.eval(
                pl.when(pl.element().is_infinite() | pl.element().is_nan())
                .then(None)
                .otherwise(pl.element())
            )
        else:
            log_ratio = log_ratio.set(invalid, None)

        return log_ratio.rename(params.output_column)
