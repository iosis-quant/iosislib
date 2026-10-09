from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from numbers import Real

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

_PROBABILITY_EPSILON = 1e-15


def _require_finite_arrays(features: np.ndarray, target: np.ndarray) -> None:
    if not np.isfinite(features).all() or not np.isfinite(target).all():
        raise ValueError(
            "Logistic regression training data must contain only finite values"
        )


def _require_binary_target(target: np.ndarray) -> None:
    unique = np.unique(target)
    if not np.isin(unique, (0.0, 1.0)).all():
        raise ValueError(
            "Logistic regression target must contain only 0 and 1; got values "
            f"{sorted(float(value) for value in unique)}"
        )


def _sigmoid(values: np.ndarray) -> np.ndarray:
    """Element-wise logistic function, overflow-safe for large magnitudes."""
    absolute = np.abs(values)
    positive = 1.0 / (1.0 + np.exp(-absolute))
    negative = np.exp(-absolute) / (1.0 + np.exp(-absolute))
    return np.where(values >= 0.0, positive, negative)


def _irls(
    features: np.ndarray,
    target: np.ndarray,
    *,
    l2: float,
    max_iter: int,
    tol: float,
) -> np.ndarray:
    """Iteratively reweighted least squares with an unpenalized intercept."""
    design = np.hstack([np.ones((features.shape[0], 1)), features])
    penalty = np.zeros(
        (features.shape[1] + 1, features.shape[1] + 1), dtype=np.float64
    )
    if l2 > 0.0:
        penalty = np.eye(features.shape[1] + 1, dtype=np.float64) * np.sqrt(l2)
        penalty[0, 0] = 0.0
    augmented_target = np.zeros(features.shape[1] + 1, dtype=np.float64)
    beta = np.zeros(features.shape[1] + 1, dtype=np.float64)
    for _ in range(max_iter):
        eta = design @ beta
        probability = np.clip(
            _sigmoid(eta), _PROBABILITY_EPSILON, 1.0 - _PROBABILITY_EPSILON
        )
        weight = probability * (1.0 - probability)
        working = eta + (target - probability) / weight
        sqrt_weight = np.sqrt(weight)
        weighted_design = design * sqrt_weight[:, None]
        weighted_working = working * sqrt_weight
        updated, *_ = np.linalg.lstsq(
            np.vstack([weighted_design, penalty]),
            np.concatenate([weighted_working, augmented_target]),
            rcond=None,
        )
        if not np.isfinite(updated).all():
            raise ValueError("Logistic regression produced non-finite coefficients")
        step = float(np.max(np.abs(updated - beta)))
        beta = updated
        if step < tol:
            break
    return beta


@dataclass(frozen=True, kw_only=True)
class LogisticRegressionModel(SupervisedModel):
    """Immutable binary logistic regression checkpoint.

    ``coefficients`` hold the feature weights and ``intercept`` the bias;
    predictions are probabilities of class 1 produced by the sigmoid of the
    linear score.
    """

    VERSION = "0.1.0"

    feature_width: int
    target_width: int
    l2: float = 0.0
    max_iter: int = 100
    tol: float = 1e-8
    coefficients: tuple[float, ...] | None = None
    intercept: float = 0.0

    @property
    def is_trained(self) -> bool:
        return self.coefficients is not None

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in ("feature_width", "target_width", "max_iter"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.target_width != 1:
            raise ValueError(
                "Logistic regression supports a single output; target_width must be 1"
            )
        if isinstance(self.l2, bool) or not isinstance(self.l2, Real):
            raise TypeError("l2 must be a number")
        if not isfinite(float(self.l2)) or float(self.l2) < 0.0:
            raise ValueError("l2 must be finite and non-negative")
        if isinstance(self.tol, bool) or not isinstance(self.tol, Real):
            raise TypeError("tol must be a number")
        if not isfinite(float(self.tol)) or float(self.tol) <= 0.0:
            raise ValueError("tol must be finite and positive")
        if not isfinite(float(self.intercept)):
            raise ValueError("intercept must be finite")
        if self.coefficients is not None:
            coefficients = tuple(float(value) for value in self.coefficients)
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
        target = train_target.ravel()
        _require_binary_target(target)
        beta = _irls(
            train_features,
            target,
            l2=float(self.l2),
            max_iter=self.max_iter,
            tol=float(self.tol),
        )
        return LogisticRegressionModel(
            feature_width=self.feature_width,
            target_width=self.target_width,
            l2=self.l2,
            max_iter=self.max_iter,
            tol=self.tol,
            coefficients=tuple(float(value) for value in beta[1:]),
            intercept=float(beta[0]),
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
            probability = _sigmoid(values @ coefficients + self.intercept)
        if not np.isfinite(probability).all():
            raise ValueError(
                "LogisticRegressionModel produced non-finite predictions"
            )
        return pl.Series(
            "prediction",
            probability.reshape(-1, self.target_width),
            dtype=pl.Array(pl.Float64, self.target_width),
        )


@dataclass(frozen=True)
class LogisticRegressionConfig(TSFNConfig):
    """Configuration for a logistic regression TSFN.

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
    max_iter: int = 100
    tol: float = 1e-8
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
            LogisticRegressionModel(
                feature_width=self.feature_width,
                target_width=self.target_width,
                l2=self.l2,
                max_iter=self.max_iter,
                tol=self.tol,
            )


class LogisticRegression(SupervisedModelTSFN):
    """Walk-forward logistic regression emitting class probabilities."""

    VERSION = "0.1.0"
    CONFIG_CLS = LogisticRegressionConfig

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
        return LogisticRegressionModel(
            feature_width=feature_width,
            target_width=target_width,
            l2=self.parameters.l2,
            max_iter=self.parameters.max_iter,
            tol=self.parameters.tol,
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


__all__ = ["LogisticRegression", "LogisticRegressionConfig", "LogisticRegressionModel"]
