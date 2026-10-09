from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from math import isfinite

import numpy as np
import polars as pl

from iosislib.core.model import (
    ChronologicalSplitter,
    Dataset,
    DatasetSplitter,
    EveryNTicksScheduler,
    Scheduler,
    SupervisedModel,
    SupervisedModelTSFN,
    scheduler_from_declaration,
    shape_width,
    splitter_from_declaration,
    validate_optional_width,
)
from iosislib.core.tsfn import (
    ColumnSignature,
    FrameSignature,
    TSFNConfig,
    TimeAxis,
    _column_signature_map,
)
from iosislib.models._regression import (
    collect_dataset,
    feature_matrix,
    mean_squared_error,
    resolve_shape,
)


def _require_finite_arrays(features: np.ndarray, target: np.ndarray) -> None:
    if not np.isfinite(features).all() or not np.isfinite(target).all():
        raise ValueError("Linear regression training data must contain only finite values")


def _solve(features: np.ndarray, target: np.ndarray, l2: float) -> np.ndarray:
    """Closed-form OLS/ridge solution with an unpenalized intercept column."""
    design = np.hstack([np.ones((features.shape[0], 1)), features])
    if l2 > 0.0:
        penalty = np.eye(features.shape[1] + 1, dtype=np.float64) * np.sqrt(l2)
        penalty[0, 0] = 0.0
        design = np.vstack([design, penalty])
        target = np.concatenate([target, np.zeros(features.shape[1] + 1)])
    solution, *_ = np.linalg.lstsq(design, target, rcond=None)
    if not np.isfinite(solution).all():
        raise ValueError("Linear regression produced non-finite coefficients")
    return solution


@dataclass(frozen=True, kw_only=True)
class LinearRegressionModel(SupervisedModel):
    """Immutable ordinary least squares checkpoint with optional ridge penalty."""

    VERSION = "0.1.0"

    feature_width: int
    target_width: int
    l2: float = 0.0
    coefficients: tuple[float, ...] | None = None
    intercept: float = 0.0

    @property
    def is_trained(self) -> bool:
        return self.coefficients is not None

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in ("feature_width", "target_width"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.target_width != 1:
            raise ValueError(
                "Linear regression supports a single output; target_width must be 1"
            )
        if isinstance(self.l2, bool) or not isinstance(self.l2, Real):
            raise TypeError("l2 must be a number")
        if not isfinite(float(self.l2)) or float(self.l2) < 0.0:
            raise ValueError("l2 must be finite and non-negative")
        if not isfinite(float(self.intercept)):
            raise ValueError("intercept must be finite")
        if self.coefficients is not None:
            coefficients = tuple(
                float(value) for value in self.coefficients
            )
            if len(coefficients) != self.feature_width:
                raise ValueError(
                    "coefficients must hold exactly feature_width values"
                )
            if not all(isfinite(value) for value in coefficients):
                raise ValueError("coefficients must be finite")
            object.__setattr__(self, "coefficients", coefficients)

    def _fit(
        self,
        train: Dataset,
        validation: Dataset | None,
        *,
        seed: int,
    ) -> SupervisedModel:
        del validation, seed
        train_features, train_target = collect_dataset(
            train,
            feature_width=self.feature_width,
            target_width=self.target_width,
        )
        _require_finite_arrays(train_features, train_target)
        solution = _solve(
            train_features, train_target.ravel(), float(self.l2)
        )
        return LinearRegressionModel(
            feature_width=self.feature_width,
            target_width=self.target_width,
            l2=self.l2,
            coefficients=tuple(float(value) for value in solution[1:]),
            intercept=float(solution[0]),
        )

    def _predict(self, features: pl.Series) -> pl.Series:
        if self.coefficients is None:
            return pl.Series(
                "prediction",
                [[float("nan")] * self.target_width for _ in range(len(features))],
                dtype=pl.Array(pl.Float64, self.target_width),
            )
        values = feature_matrix(features, width=self.feature_width)
        coefficients = np.asarray(self.coefficients, dtype=np.float64)
        with np.errstate(over="ignore"):
            prediction = values @ coefficients + self.intercept
        if not np.isfinite(prediction).all():
            raise ValueError("LinearRegressionModel produced non-finite predictions")
        return pl.Series(
            "prediction",
            prediction.reshape(-1, self.target_width),
            dtype=pl.Array(pl.Float64, self.target_width),
        )


@dataclass(frozen=True)
class LinearRegressionConfig(TSFNConfig):
    """Configuration for a linear regression TSFN.

    ``feature_width`` and ``target_width`` are derived from the graph bindings
    unless explicitly configured. ``scheduler`` and ``splitter`` accept
    declarative mappings.
    """

    feature_width: int | None = None
    target_width: int | None = None
    scheduler: Scheduler | Mapping[str, object] | None = None
    splitter: DatasetSplitter | Mapping[str, object] | None = None
    seed: int = 0
    l2: float = 0.0
    timestamp_column: str = "timestamp"

    def __post_init__(self) -> None:
        validate_optional_width("feature_width", self.feature_width)
        validate_optional_width("target_width", self.target_width)
        object.__setattr__(
            self,
            "scheduler",
            scheduler_from_declaration(
                self.scheduler,
                default=EveryNTicksScheduler(100),
            ),
        )
        object.__setattr__(
            self,
            "splitter",
            splitter_from_declaration(
                self.splitter,
                default=ChronologicalSplitter(validation_size=0.2),
            ),
        )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if not isinstance(self.timestamp_column, str) or not self.timestamp_column:
            raise ValueError("timestamp_column must be a non-empty string")
        if self.feature_width is not None and self.target_width is not None:
            LinearRegressionModel(
                feature_width=self.feature_width,
                target_width=self.target_width,
                l2=self.l2,
            )


class LinearRegression(SupervisedModelTSFN):
    """Walk-forward ordinary least squares regression with optional ridge."""

    VERSION = "0.1.0"
    CONFIG_CLS = LinearRegressionConfig

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        params = self.parameters
        feature_shape = (params.feature_width,) if params.feature_width else ()
        target_shape = (params.target_width,) if params.target_width else ()
        time = TimeAxis(params.timestamp_column)
        return (
            FrameSignature(
                time=time,
                columns=(
                    ("features", pl.Float64, feature_shape),
                    ("target", pl.Float64, target_shape),
                ),
            ),
            FrameSignature(
                time=time,
                columns=(("prediction", pl.Float64, target_shape),),
            ),
        )

    def resolve_signature(
        self,
        bound_input_columns: Mapping[str, ColumnSignature],
    ) -> tuple[FrameSignature, FrameSignature]:
        params = self.parameters
        feature_shape = resolve_shape(
            "features", bound_input_columns, params.feature_width
        )
        target_shape = resolve_shape(
            "target", bound_input_columns, params.target_width
        )
        time = self.signature[0].time
        return (
            FrameSignature(
                time=time,
                columns=(
                    ("features", pl.Float64, feature_shape),
                    ("target", pl.Float64, target_shape),
                ),
            ),
            FrameSignature(
                time=time,
                columns=(
                    ("prediction", pl.Float64, (max(shape_width(target_shape), 1),)),
                ),
            ),
        )

    def initial_model(self) -> SupervisedModel:
        feature_width, target_width = self._resolved_widths()
        return LinearRegressionModel(
            feature_width=feature_width,
            target_width=target_width,
            l2=self.parameters.l2,
        )

    def _resolved_widths(self) -> tuple[int, int]:
        columns = _column_signature_map(self.signature[0])
        return max(shape_width(columns["features"].shape), 1), max(
            shape_width(columns["target"].shape), 1
        )

    def scheduler(self) -> Scheduler:
        return self.parameters.scheduler

    def splitter(self) -> DatasetSplitter:
        return self.parameters.splitter

    def training_seed(self, retrain_count: int) -> int:
        return self.parameters.seed + retrain_count

    def segment_metrics(
        self,
        target: pl.Series,
        prediction: pl.Series,
    ) -> Mapping[str, float]:
        return {"mse": mean_squared_error(target, prediction)}


__all__ = ["LinearRegression", "LinearRegressionConfig", "LinearRegressionModel"]
