from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from iosislib.core.tsfn import FrameSignature, TSFN, TSFNConfig, TimeAxis
from iosislib.tsfn.transforms._validation import (
    validate_column_name,
    validate_distinct_columns,
)


@dataclass(frozen=True)
class LogReturnConfig(TSFNConfig):
    input_column: str = "value"
    output_column: str = "log_return"
    timestamp_column: str = "timestamp"
    periods: int = 1

    def __post_init__(self) -> None:
        validate_column_name(self.input_column, field_name="input_column")
        validate_column_name(self.output_column, field_name="output_column")
        validate_column_name(self.timestamp_column, field_name="timestamp_column")
        validate_distinct_columns(
            self.timestamp_column, self.input_column, self.output_column
        )
        if isinstance(self.periods, bool) or not isinstance(self.periods, int):
            raise TypeError("periods must be an integer")
        if self.periods < 1:
            raise ValueError("periods must be at least 1")


class LogReturn(TSFN[LogReturnConfig]):
    VERSION = "0.1.0"
    CONFIG_CLS = LogReturnConfig

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

    def apply(self, lf: pl.LazyFrame | None = None) -> pl.LazyFrame:
        if lf is None:
            raise ValueError("LogReturn requires an input frame")
        params = self.parameters
        value = pl.col(params.input_column)
        prior = value.shift(params.periods)
        result = pl.when((value > 0.0) & (prior > 0.0)).then(
            value.log() - prior.log()
        ).otherwise(pl.lit(None, dtype=pl.Float64))
        return lf.sort(params.timestamp_column).select(
            params.timestamp_column, result.alias(params.output_column)
        )
