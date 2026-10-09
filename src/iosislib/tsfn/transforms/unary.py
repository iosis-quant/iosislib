from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from math import isfinite

import polars as pl

from iosislib.core.tsfn import (
    FrameSignature,
    ItemwiseUnaryTSFN,
    TSFNConfig,
    TimeAxis,
)
from iosislib.tsfn.transforms._validation import (
    validate_column_name,
    validate_distinct_columns,
)


def _validate_unary_columns(
    input_column: str,
    output_column: str,
    timestamp_column: str,
) -> None:
    validate_column_name(input_column, field_name="input_column")
    validate_column_name(output_column, field_name="output_column")
    validate_column_name(timestamp_column, field_name="timestamp_column")
    validate_distinct_columns(timestamp_column, input_column)
    validate_distinct_columns(timestamp_column, output_column)


def _unary_signature(
    input_column: str,
    output_column: str,
    timestamp_column: str,
) -> tuple[FrameSignature, FrameSignature]:
    time = TimeAxis(column=timestamp_column)
    return (
        FrameSignature(time=time, columns=((input_column, pl.Float64),)),
        FrameSignature(time=time, columns=((output_column, pl.Float64),)),
    )


@dataclass(frozen=True)
class AbsConfig(TSFNConfig):
    input_column: str = "value"
    output_column: str = "abs"
    timestamp_column: str = "timestamp"

    def __post_init__(self) -> None:
        _validate_unary_columns(
            self.input_column, self.output_column, self.timestamp_column
        )


class Abs(ItemwiseUnaryTSFN[AbsConfig]):
    VERSION = "0.1.0"
    CONFIG_CLS = AbsConfig

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        params = self.parameters
        return _unary_signature(
            params.input_column, params.output_column, params.timestamp_column
        )

    def itemwise_input_column(self) -> str:
        return self.parameters.input_column

    def itemwise_output_column(self) -> str:
        return self.parameters.output_column

    def itemwise_expr(self, value: pl.Expr) -> pl.Expr:
        return value.abs()


@dataclass(frozen=True)
class SignConfig(TSFNConfig):
    input_column: str = "value"
    output_column: str = "sign"
    timestamp_column: str = "timestamp"

    def __post_init__(self) -> None:
        _validate_unary_columns(
            self.input_column, self.output_column, self.timestamp_column
        )


class Sign(ItemwiseUnaryTSFN[SignConfig]):
    VERSION = "0.1.0"
    CONFIG_CLS = SignConfig

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        params = self.parameters
        return _unary_signature(
            params.input_column, params.output_column, params.timestamp_column
        )

    def itemwise_input_column(self) -> str:
        return self.parameters.input_column

    def itemwise_output_column(self) -> str:
        return self.parameters.output_column

    def itemwise_expr(self, value: pl.Expr) -> pl.Expr:
        return value.sign()


@dataclass(frozen=True)
class SqrtConfig(TSFNConfig):
    input_column: str = "value"
    output_column: str = "sqrt"
    timestamp_column: str = "timestamp"

    def __post_init__(self) -> None:
        _validate_unary_columns(
            self.input_column, self.output_column, self.timestamp_column
        )


class Sqrt(ItemwiseUnaryTSFN[SqrtConfig]):
    VERSION = "0.1.0"
    CONFIG_CLS = SqrtConfig

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        params = self.parameters
        return _unary_signature(
            params.input_column, params.output_column, params.timestamp_column
        )

    def itemwise_input_column(self) -> str:
        return self.parameters.input_column

    def itemwise_output_column(self) -> str:
        return self.parameters.output_column

    def itemwise_expr(self, value: pl.Expr) -> pl.Expr:
        # The square root of a negative value is undefined; surface it as
        # null so missing stays missing instead of becoming NaN.
        return (
            pl.when(value >= 0.0)
            .then(value.sqrt())
            .otherwise(pl.lit(None, dtype=pl.Float64))
        )


@dataclass(frozen=True)
class PowerConfig(TSFNConfig):
    input_column: str = "value"
    output_column: str = "power"
    timestamp_column: str = "timestamp"
    exponent: float = 2.0

    def __post_init__(self) -> None:
        _validate_unary_columns(
            self.input_column, self.output_column, self.timestamp_column
        )
        if isinstance(self.exponent, bool) or not isinstance(self.exponent, Real):
            raise TypeError("exponent must be a number")
        if not isfinite(float(self.exponent)):
            raise ValueError("exponent must be finite")


class Power(ItemwiseUnaryTSFN[PowerConfig]):
    VERSION = "0.1.0"
    CONFIG_CLS = PowerConfig

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        params = self.parameters
        return _unary_signature(
            params.input_column, params.output_column, params.timestamp_column
        )

    def itemwise_input_column(self) -> str:
        return self.parameters.input_column

    def itemwise_output_column(self) -> str:
        return self.parameters.output_column

    def itemwise_expr(self, value: pl.Expr) -> pl.Expr:
        # Fractional exponents over negative bases and overflow both produce
        # non-finite values; normalize them to null like Ratio does.
        result = value.pow(float(self.parameters.exponent))
        return (
            pl.when(result.is_finite())
            .then(result)
            .otherwise(pl.lit(None, dtype=pl.Float64))
        )


@dataclass(frozen=True)
class ClipConfig(TSFNConfig):
    input_column: str = "value"
    output_column: str = "clip"
    timestamp_column: str = "timestamp"
    lower: float | None = None
    upper: float | None = None

    def __post_init__(self) -> None:
        _validate_unary_columns(
            self.input_column, self.output_column, self.timestamp_column
        )
        if self.lower is None and self.upper is None:
            raise ValueError("At least one of lower and upper must be provided")
        for name, bound in (("lower", self.lower), ("upper", self.upper)):
            if bound is None:
                continue
            if isinstance(bound, bool) or not isinstance(bound, Real):
                raise TypeError(f"{name} must be a number or None")
            if not isfinite(float(bound)):
                raise ValueError(f"{name} must be finite")
        if (
            self.lower is not None
            and self.upper is not None
            and float(self.lower) > float(self.upper)
        ):
            raise ValueError("lower must not exceed upper")


class Clip(ItemwiseUnaryTSFN[ClipConfig]):
    VERSION = "0.1.0"
    CONFIG_CLS = ClipConfig

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        params = self.parameters
        return _unary_signature(
            params.input_column, params.output_column, params.timestamp_column
        )

    def itemwise_input_column(self) -> str:
        return self.parameters.input_column

    def itemwise_output_column(self) -> str:
        return self.parameters.output_column

    def itemwise_expr(self, value: pl.Expr) -> pl.Expr:
        params = self.parameters
        return value.clip(params.lower, params.upper)


__all__ = [
    "Abs",
    "AbsConfig",
    "Clip",
    "ClipConfig",
    "Power",
    "PowerConfig",
    "Sign",
    "SignConfig",
    "Sqrt",
    "SqrtConfig",
]
