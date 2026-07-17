from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from src.core.tsfn import FrameSignature, ItemwiseStructTSFN, TSFNConfig, TimeAxis
from src.tsfn.transforms._validation import validate_column_name, validate_distinct_columns


@dataclass(frozen=True)
class RatioConfig(TSFNConfig):
    numerator_column: str = "left"
    denominator_column: str = "right"
    output_column: str = "ratio"
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


class Ratio(ItemwiseStructTSFN):
    VERSION = "0.1.0"
    CONFIG_CLS = RatioConfig

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

    def batch(self, fields: dict[str, pl.Series]) -> pl.Series:
        params = self.parameters
        ratio = fields[params.numerator_column] / fields[params.denominator_column]
        output_shape = self.output_column_signature(params.output_column).shape

        if output_shape:
            ratio = ratio.arr.eval(
                pl.when(pl.element().is_infinite())
                .then(None)
                .otherwise(pl.element())
            )
        else:
            ratio = ratio.set(ratio.is_infinite(), None)

        return ratio.rename(params.output_column)
