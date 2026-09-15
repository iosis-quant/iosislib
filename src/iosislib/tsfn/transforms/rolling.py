from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import polars as pl

from iosislib.core.tsfn import (
    FrameSignature,
    RollingUnaryTSFN,
    TSFN,
    TSFNConfig,
    TimeAxis,
    _column_signature_map,
)
from iosislib.tsfn.transforms._validation import (
    validate_column_name,
    validate_distinct_columns,
)


def _validate_window(periods: int, min_samples: int) -> None:
    if isinstance(periods, bool) or not isinstance(periods, int):
        raise TypeError("periods must be an integer")
    if periods < 1:
        raise ValueError("periods must be at least 1")
    if isinstance(min_samples, bool) or not isinstance(min_samples, int):
        raise TypeError("min_samples must be an integer")
    if not 1 <= min_samples <= periods:
        raise ValueError("min_samples must be between 1 and periods")


def _validate_ddof(ddof: int, periods: int) -> None:
    if isinstance(ddof, bool) or not isinstance(ddof, int):
        raise TypeError("ddof must be an integer")
    if not 0 <= ddof < periods:
        raise ValueError("ddof must be between 0 and periods - 1")


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
        _validate_window(self.periods, self.min_samples)


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


@dataclass(frozen=True)
class RollingVarConfig(RollingConfig):
    output_column: str = "rolling_var"
    ddof: int = 1

    def __post_init__(self) -> None:
        super().__post_init__()
        _validate_ddof(self.ddof, self.periods)


class RollingVar(RollingTransform):
    """Rolling variance over a window.

    The default ``ddof=1`` matches the divisor used by ``RollingStd``, so
    ``rolling_var`` equals ``rolling_std`` squared under default parameters.
    """

    VERSION = "0.1.0"
    CONFIG_CLS = RollingVarConfig

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
        ddof = cast(RollingVarConfig, self.parameters).ddof
        return value.rolling_var(
            window_size=periods, min_samples=min_samples, ddof=ddof
        )


@dataclass(frozen=True)
class RollingCovConfig(TSFNConfig):
    left_column: str = "left"
    right_column: str = "right"
    output_column: str = "rolling_cov"
    timestamp_column: str = "timestamp"
    periods: int = 3
    min_samples: int = 1
    ddof: int = 1

    def __post_init__(self) -> None:
        validate_column_name(self.left_column, field_name="left_column")
        validate_column_name(self.right_column, field_name="right_column")
        validate_column_name(self.output_column, field_name="output_column")
        validate_column_name(self.timestamp_column, field_name="timestamp_column")
        validate_distinct_columns(
            self.timestamp_column,
            self.left_column,
            self.right_column,
            self.output_column,
        )
        _validate_window(self.periods, self.min_samples)
        _validate_ddof(self.ddof, self.periods)


class RollingCov(TSFN[RollingCovConfig]):
    """Rolling covariance between two columns over a shared window.

    Both inputs align on the graph union timeline before windowing, so the
    covariance at each timestamp only observes rows at or before it. Chain
    ``EwmMean`` over the scalar output when an exponentially-weighted
    smoothing of the covariance is needed.

    Degenerate windows (fewer observations than ``ddof`` demands, or a window
    holding missing inputs) surface as NaN from Polars; they are normalized
    to null so missing stays missing and downstream smoothing is not
    poisoned by NaN.
    """

    VERSION = "0.1.0"
    CONFIG_CLS = RollingCovConfig

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        params = self.parameters
        return (
            FrameSignature(
                time=TimeAxis(column=params.timestamp_column),
                columns=(
                    (params.left_column, pl.Float64),
                    (params.right_column, pl.Float64),
                ),
            ),
            FrameSignature(
                time=TimeAxis(column=params.timestamp_column),
                columns=((params.output_column, pl.Float64),),
            ),
        )

    def apply(self, lf: pl.LazyFrame | None = None) -> pl.LazyFrame:
        if lf is None:
            raise ValueError("RollingCov requires an input frame")

        input_signature, output_signature = self.signature
        if input_signature.time is None:
            raise ValueError("RollingCov input signature must declare a time axis")

        params = self.parameters
        output_column = _column_signature_map(output_signature)[params.output_column]
        cov = (
            pl.rolling_cov(
                pl.col(params.left_column),
                pl.col(params.right_column),
                window_size=params.periods,
                min_samples=params.min_samples,
                ddof=params.ddof,
            ).fill_nan(None)
        )
        return lf.sort(input_signature.time.column).select(
            input_signature.time.column,
            cov.cast(output_column.physical_dtype).alias(output_column.name),
        )


@dataclass(frozen=True)
class RollingCovMatrixConfig(TSFNConfig):
    input_columns: tuple[str, ...]
    output_column: str = "cov_matrix"
    timestamp_column: str = "timestamp"
    periods: int = 3
    min_samples: int = 1
    ddof: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.input_columns, str) or not isinstance(
            self.input_columns, Sequence
        ):
            raise TypeError("input_columns must be a sequence of strings")
        columns = tuple(self.input_columns)
        if len(columns) < 2:
            raise ValueError("input_columns must contain at least two columns")
        if not all(isinstance(name, str) for name in columns):
            raise TypeError("input_columns must contain only strings")
        if any(not name for name in columns):
            raise ValueError("input_columns must contain non-empty strings")
        duplicate_names = sorted(
            {name for name in columns if columns.count(name) > 1}
        )
        if duplicate_names:
            raise ValueError(
                f"Duplicate input_columns are not allowed: {duplicate_names}"
            )
        object.__setattr__(self, "input_columns", columns)
        validate_column_name(self.output_column, field_name="output_column")
        validate_column_name(self.timestamp_column, field_name="timestamp_column")
        validate_distinct_columns(self.timestamp_column, self.output_column)
        if self.output_column in columns:
            raise ValueError("output_column must not be one of the input columns")
        if self.timestamp_column in columns:
            raise ValueError("timestamp_column must not be one of the input columns")
        _validate_window(self.periods, self.min_samples)
        _validate_ddof(self.ddof, self.periods)


class RollingCovMatrix(TSFN[RollingCovMatrixConfig]):
    """Rolling covariance matrix over N input columns for Value-at-Risk.

    The output is one flat ``pl.Array(pl.Float64, n * n)`` cell per row in
    row-major order: entry ``i * n + j`` holds ``cov(columns[i], columns[j])``
    and the diagonal holds each column's rolling variance. The matrix is
    symmetric by construction. Portfolio variance follows as ``w^T Σ w``
    downstream. To smooth a matrix, unpack it, apply ``EwmMean`` per element,
    and repack it.

    Covariance cells normalize Polars NaN (degenerate or missing-input
    windows) to null; variance cells ignore missing inputs like the other
    rolling unary transforms.
    """

    VERSION = "0.1.0"
    CONFIG_CLS = RollingCovMatrixConfig

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        params = self.parameters
        width = len(params.input_columns)
        time = TimeAxis(column=params.timestamp_column)
        return (
            FrameSignature(
                time=time,
                columns=tuple((name, pl.Float64) for name in params.input_columns),
            ),
            FrameSignature(
                time=time,
                columns=((params.output_column, pl.Float64, (width * width,)),),
            ),
        )

    def apply(self, lf: pl.LazyFrame | None = None) -> pl.LazyFrame:
        if lf is None:
            raise ValueError("RollingCovMatrix requires an input frame")

        input_signature, output_signature = self.signature
        if input_signature.time is None:
            raise ValueError(
                "RollingCovMatrix input signature must declare a time axis"
            )

        params = self.parameters
        columns = list(params.input_columns)
        cells: dict[tuple[int, int], pl.Expr] = {}
        ordered: list[pl.Expr] = []
        for i, left in enumerate(columns):
            for j, right in enumerate(columns):
                if j < i:
                    ordered.append(cells[(j, i)])
                    continue
                if i == j:
                    cell = pl.col(left).rolling_var(
                        window_size=params.periods,
                        min_samples=params.min_samples,
                        ddof=params.ddof,
                    )
                else:
                    cell = pl.rolling_cov(
                        pl.col(left),
                        pl.col(right),
                        window_size=params.periods,
                        min_samples=params.min_samples,
                        ddof=params.ddof,
                    ).fill_nan(None)
                cells[(i, j)] = cell
                ordered.append(cell)

        output_column = _column_signature_map(output_signature)[params.output_column]
        matrix = (
            pl.concat_arr(*ordered)
            .cast(output_column.physical_dtype)
            .alias(output_column.name)
        )
        return lf.sort(input_signature.time.column).select(
            input_signature.time.column,
            matrix,
        )
