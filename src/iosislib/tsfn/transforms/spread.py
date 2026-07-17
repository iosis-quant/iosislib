from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from iosislib.core.tsfn import FrameSignature, ItemwiseStructTSFN, TSFNConfig, TimeAxis
from iosislib.tsfn.transforms._validation import validate_column_name, validate_distinct_columns


@dataclass(frozen=True)
class SpreadConfig(TSFNConfig):
    left_column: str = "left"
    right_column: str = "right"
    output_column: str = "spread"
    timestamp_column: str = "timestamp"

    def __post_init__(self) -> None:
        validate_column_name(self.left_column, field_name="left_column")
        validate_column_name(self.right_column, field_name="right_column")
        validate_column_name(self.output_column, field_name="output_column")
        validate_column_name(self.timestamp_column, field_name="timestamp_column")
        validate_distinct_columns(self.left_column, self.right_column)
        validate_distinct_columns(self.timestamp_column, self.left_column)
        validate_distinct_columns(self.timestamp_column, self.right_column)
        validate_distinct_columns(self.timestamp_column, self.output_column)


class Spread(ItemwiseStructTSFN):
    VERSION = "0.1.0"
    CONFIG_CLS = SpreadConfig

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        params = self.parameters
        input_frame = FrameSignature(
            time=TimeAxis(column=params.timestamp_column),
            columns=(
                (params.left_column, pl.Float64),
                (params.right_column, pl.Float64),
            ),
        )
        output_frame = FrameSignature(
            time=TimeAxis(column=params.timestamp_column),
            columns=((params.output_column, pl.Float64),),
        )
        return input_frame, output_frame

    def batch_input_columns(self) -> tuple[str, ...]:
        params = self.parameters
        return (params.left_column, params.right_column)

    def batch_output_column(self) -> str:
        return self.parameters.output_column

    def batch(self, fields: dict[str, pl.Series]) -> pl.Series:
        params = self.parameters
        return (fields[params.left_column] - fields[params.right_column]).rename(
            params.output_column
        )
