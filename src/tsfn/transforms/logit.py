from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from src.core.tsfn import FrameSignature, ItemwiseUnaryTSFN, TSFNConfig, TimeAxis
from src.tsfn.transforms._validation import validate_column_name, validate_distinct_columns


@dataclass(frozen=True)
class LogitConfig(TSFNConfig):
    input_column: str = "value"
    output_column: str = "logit"
    timestamp_column: str = "timestamp"

    def __post_init__(self) -> None:
        validate_column_name(self.input_column, field_name="input_column")
        validate_column_name(self.output_column, field_name="output_column")
        validate_column_name(self.timestamp_column, field_name="timestamp_column")
        validate_distinct_columns(self.timestamp_column, self.input_column)
        validate_distinct_columns(self.timestamp_column, self.output_column)


class Logit(ItemwiseUnaryTSFN):
    VERSION = "0.1.0"
    CONFIG_CLS = LogitConfig

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        params = self.parameters
        input_frame = FrameSignature(
            time=TimeAxis(column=params.timestamp_column),
            columns=((params.input_column, pl.Float64),),
        )
        output_frame = FrameSignature(
            time=TimeAxis(column=params.timestamp_column),
            columns=((params.output_column, pl.Float64),),
        )
        return input_frame, output_frame

    def itemwise_input_column(self) -> str:
        return self.parameters.input_column

    def itemwise_output_column(self) -> str:
        return self.parameters.output_column

    def itemwise_expr(self, value: pl.Expr) -> pl.Expr:
        return (
            pl.when((value > 0.0) & (value < 1.0))
            .then((value / (1.0 - value)).log())
            .otherwise(pl.lit(None, dtype=pl.Float64))
        )
