from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from iosislib.core.tsfn import (
    FrameSignature,
    RollingUnaryTSFN,
    TSFNConfig,
    TimeAxis,
)
from iosislib.tsfn.transforms._validation import (
    validate_column_name,
    validate_distinct_columns,
)


@dataclass(frozen=True)
class RollingConfig(TSFNConfig):
    input_column: str = "value"
    output_column: str = "rolling"
    timestamp_column: str = "timestamp"
    periods: int = 3
    min_samples: int = 1

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
        if isinstance(self.min_samples, bool) or not isinstance(self.min_samples, int):
            raise TypeError("min_samples must be an integer")
        if not 1 <= self.min_samples <= self.periods:
            raise ValueError("min_samples must be between 1 and periods")


class RollingTransform(RollingUnaryTSFN[RollingConfig]):
    def rolling_input_column(self) -> str:
        return self.parameters.input_column

    def rolling_output_column(self) -> str:
        return self.parameters.output_column

    def rolling_periods(self) -> int:
        return self.parameters.periods

    def rolling_min_samples(self) -> int:
        return self.parameters.min_samples


@dataclass(frozen=True)
class RollingMeanConfig(RollingConfig):
    output_column: str = "rolling_mean"


class RollingMean(RollingTransform):
    VERSION = "0.1.0"
    CONFIG_CLS = RollingMeanConfig

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

    def rolling_expr(
        self,
        value: pl.Expr,
        *,
        periods: int,
        min_samples: int,
    ) -> pl.Expr:
        return value.rolling_mean(window_size=periods, min_samples=min_samples)


@dataclass(frozen=True)
class RollingStdConfig(RollingConfig):
    output_column: str = "rolling_std"


class RollingStd(RollingTransform):
    VERSION = "0.1.0"
    CONFIG_CLS = RollingStdConfig

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

    def rolling_expr(
        self,
        value: pl.Expr,
        *,
        periods: int,
        min_samples: int,
    ) -> pl.Expr:
        return value.rolling_std(window_size=periods, min_samples=min_samples)


@dataclass(frozen=True)
class RollingSumConfig(RollingConfig):
    output_column: str = "rolling_sum"


class RollingSum(RollingTransform):
    VERSION = "0.1.0"
    CONFIG_CLS = RollingSumConfig

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

    def rolling_expr(
        self,
        value: pl.Expr,
        *,
        periods: int,
        min_samples: int,
    ) -> pl.Expr:
        return value.rolling_sum(window_size=periods, min_samples=min_samples)


@dataclass(frozen=True)
class RollingMaxConfig(RollingConfig):
    output_column: str = "rolling_max"


class RollingMax(RollingTransform):
    VERSION = "0.1.0"
    CONFIG_CLS = RollingMaxConfig

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

    def rolling_expr(
        self,
        value: pl.Expr,
        *,
        periods: int,
        min_samples: int,
    ) -> pl.Expr:
        return value.rolling_max(window_size=periods, min_samples=min_samples)


@dataclass(frozen=True)
class RollingMinConfig(RollingConfig):
    output_column: str = "rolling_min"


class RollingMin(RollingTransform):
    VERSION = "0.1.0"
    CONFIG_CLS = RollingMinConfig

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

    def rolling_expr(
        self,
        value: pl.Expr,
        *,
        periods: int,
        min_samples: int,
    ) -> pl.Expr:
        return value.rolling_min(window_size=periods, min_samples=min_samples)


@dataclass(frozen=True)
class RollingMedianConfig(RollingConfig):
    output_column: str = "rolling_median"


class RollingMedian(RollingTransform):
    VERSION = "0.1.0"
    CONFIG_CLS = RollingMedianConfig

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

    def rolling_expr(
        self,
        value: pl.Expr,
        *,
        periods: int,
        min_samples: int,
    ) -> pl.Expr:
        return value.rolling_median(window_size=periods, min_samples=min_samples)


@dataclass(frozen=True)
class RollingZScoreConfig(RollingConfig):
    output_column: str = "rolling_z_score"


class RollingZScore(RollingTransform):
    VERSION = "0.1.0"
    CONFIG_CLS = RollingZScoreConfig

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

    def rolling_expr(
        self,
        value: pl.Expr,
        *,
        periods: int,
        min_samples: int,
    ) -> pl.Expr:
        mean = value.rolling_mean(window_size=periods, min_samples=min_samples)
        std = value.rolling_std(window_size=periods, min_samples=min_samples)
        return pl.when(std > 0.0).then(
            (value - mean) / std
        ).otherwise(pl.lit(None, dtype=pl.Float64))
