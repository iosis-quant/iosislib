from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from src.classes import FrameSignature, TSFN, TSFNConfig, TimeAxis
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


class Ratio(TSFN):
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

    def apply(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        params = self.parameters
        denominator = pl.col(params.denominator_column)
        ratio = (
            pl.when((denominator.is_null() | (denominator == 0.0)))
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise(pl.col(params.numerator_column) / denominator)
            .cast(pl.Float64)
            .alias(params.output_column)
        )
        return lf.select(params.timestamp_column, ratio)
