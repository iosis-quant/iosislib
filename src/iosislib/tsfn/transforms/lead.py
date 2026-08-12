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
from iosislib.tsfn.transforms._validation import validate_column_name, validate_distinct_columns


@dataclass(frozen=True)
class LeadConfig(TSFNConfig):
    input_column: str = "value"
    output_column: str = "lead"
    timestamp_column: str = "timestamp"
    periods: int = 1

    def __post_init__(self) -> None:
        validate_column_name(self.input_column, field_name="input_column")
        validate_column_name(self.output_column, field_name="output_column")
        validate_column_name(self.timestamp_column, field_name="timestamp_column")
        validate_distinct_columns(self.timestamp_column, self.input_column)
        validate_distinct_columns(self.timestamp_column, self.output_column)
        if isinstance(self.periods, bool) or not isinstance(self.periods, int):
            raise TypeError("periods must be an integer")
        if self.periods < 1:
            raise ValueError("periods must be at least 1")


class Lead(TSFN[LeadConfig]):
    """Forward-shift a value to expose a future observation.

    Lead is look-ahead by construction; graph verification only allows its
    output to feed a declared target/label input of a supervised model.
    """

    VERSION = "0.1.0"
    CONFIG_CLS = LeadConfig
    LOOKAHEAD = True

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

    def resolve_signature(
        self,
        bound_input_columns: Mapping[str, ColumnSignature],
    ) -> tuple[FrameSignature, FrameSignature]:
        input_signature, output_signature = self.signature
        input_columns = _column_signature_map(input_signature)
        output_columns = _column_signature_map(output_signature)
        input_name = self.parameters.input_column
        output_name = self.parameters.output_column

        bound_input = bound_input_columns.get(input_name)
        if bound_input is None:
            return self.signature

        resolved_input = ColumnSignature(
            input_columns[input_name].name,
            input_columns[input_name].dtype,
            bound_input.shape,
        )
        resolved_output = ColumnSignature(
            output_columns[output_name].name,
            output_columns[output_name].dtype,
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
            raise ValueError("Lead requires an input frame")
        params = self.parameters
        input_signature, output_signature = self.signature
        if input_signature.time is None:
            raise ValueError("Lead input signature must declare a time axis")
        output_column = _column_signature_map(output_signature)[params.output_column]

        lead = pl.col(params.input_column).shift(-params.periods)
        return lf.sort(params.timestamp_column).select(
            params.timestamp_column,
            lead.cast(output_column.physical_dtype).alias(params.output_column),
        )
