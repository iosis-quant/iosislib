from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import polars as pl

from iosislib.core.tsfn import (
    EwmUnaryTSFN,
    FrameSignature,
    TSFNConfig,
    TimeAxis,
)
from iosislib.tsfn.transforms._validation import (
    validate_column_name,
    validate_distinct_columns,
)


@dataclass(frozen=True)
class EwmConfig(TSFNConfig):
    input_column: str = "value"
    output_column: str = "ewm"
    timestamp_column: str = "timestamp"
    alpha: float = 0.1
    adjust: bool = True
    min_samples: int = 1

    def __post_init__(self) -> None:
        validate_column_name(self.input_column, field_name="input_column")
        validate_column_name(self.output_column, field_name="output_column")
        validate_column_name(self.timestamp_column, field_name="timestamp_column")
        validate_distinct_columns(
            self.timestamp_column, self.input_column, self.output_column
        )
        if isinstance(self.alpha, bool) or not isinstance(self.alpha, Real):
            raise TypeError("alpha must be a number")
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha must be between 0 (exclusive) and 1")
        if not isinstance(self.adjust, bool):
            raise TypeError("adjust must be a bool")
        if isinstance(self.min_samples, bool) or not isinstance(self.min_samples, int):
            raise TypeError("min_samples must be an integer")
        if self.min_samples < 1:
            raise ValueError("min_samples must be at least 1")


class EwmTransform(EwmUnaryTSFN[EwmConfig]):
    def ewm_input_column(self) -> str:
        return self.parameters.input_column

    def ewm_output_column(self) -> str:
        return self.parameters.output_column

    def ewm_alpha(self) -> float:
        return self.parameters.alpha

    def ewm_min_samples(self) -> int:
        return self.parameters.min_samples

    def ewm_adjust(self) -> bool:
        return self.parameters.adjust


@dataclass(frozen=True)
class EwmMeanConfig(EwmConfig):
    output_column: str = "ewm_mean"


class EwmMean(EwmTransform):
    VERSION = "0.1.0"
    CONFIG_CLS = EwmMeanConfig

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        params = self.parameters
        return (
            FrameSignature(
                time=TimeAxis(column=params.timestamp_column),
                columns=((params.input_column, pl.Float64),),
            ),
            FrameSignature(
                time=TimeAxis(column=params.timestamp_column),
                columns=((params.output_column, pl.Float64),),
            ),
        )

    def ewm_expr(
        self,
        value: pl.Expr,
        *,
        alpha: float,
        min_samples: int,
        adjust: bool,
    ) -> pl.Expr:
        return value.ewm_mean(alpha=alpha, adjust=adjust, min_samples=min_samples)


@dataclass(frozen=True)
class EwmStdConfig(EwmConfig):
    output_column: str = "ewm_std"


class EwmStd(EwmTransform):
    VERSION = "0.1.0"
    CONFIG_CLS = EwmStdConfig

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        params = self.parameters
        return (
            FrameSignature(
                time=TimeAxis(column=params.timestamp_column),
                columns=((params.input_column, pl.Float64),),
            ),
            FrameSignature(
                time=TimeAxis(column=params.timestamp_column),
                columns=((params.output_column, pl.Float64),),
            ),
        )

    def ewm_expr(
        self,
        value: pl.Expr,
        *,
        alpha: float,
        min_samples: int,
        adjust: bool,
    ) -> pl.Expr:
        return value.ewm_std(alpha=alpha, adjust=adjust, min_samples=min_samples)
