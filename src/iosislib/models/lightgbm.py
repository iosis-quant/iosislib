from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import polars as pl

from iosislib.core.model import (
    Dataset,
    DatasetSplitter,
    Scheduler,
    SupervisedModel,
    SupervisedModelTSFN,
)
from iosislib.core.tsfn import FrameSignature, TSFNConfig, TimeAxis
from iosislib.models._regression import (
    collect_dataset,
    feature_matrix,
    mean_squared_error,
)


def _import_lightgbm() -> Any:
    try:
        import lightgbm
    except ImportError as error:
        raise ImportError(
            "LightGBM models require the optional 'lightgbm' dependency; "
            "install iosislib[lightgbm]"
        ) from error
    return lightgbm


@dataclass(frozen=True, kw_only=True)
class LightGBMModel(SupervisedModel):
    """Immutable LightGBM scalar-regression checkpoint."""

    VERSION = "0.1.0"

    feature_width: int
    num_boost_round: int = 100
    learning_rate: float = 0.05
    num_leaves: int = 31
    max_depth: int = -1
    min_data_in_leaf: int = 20
    early_stopping_rounds: int = 0
    model_text: str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in ("feature_width", "num_boost_round", "num_leaves"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.max_depth, bool)
            or not isinstance(self.max_depth, int)
            or self.max_depth == 0
            or self.max_depth < -1
        ):
            raise ValueError("max_depth must be -1 or a positive integer")
        if (
            isinstance(self.min_data_in_leaf, bool)
            or not isinstance(self.min_data_in_leaf, int)
            or self.min_data_in_leaf < 1
        ):
            raise ValueError("min_data_in_leaf must be a positive integer")
        if (
            isinstance(self.early_stopping_rounds, bool)
            or not isinstance(self.early_stopping_rounds, int)
            or self.early_stopping_rounds < 0
        ):
            raise ValueError("early_stopping_rounds must be a non-negative integer")
        if (
            isinstance(self.learning_rate, bool)
            or not isinstance(self.learning_rate, (int, float))
            or self.learning_rate <= 0
        ):
            raise ValueError("learning_rate must be positive")

    def _fit(
        self,
        train: Dataset,
        validation: Dataset | None,
        *,
        seed: int,
    ) -> SupervisedModel:
        lightgbm = _import_lightgbm()
        train_features, train_target = collect_dataset(
            train,
            width=self.feature_width,
            seed=seed,
        )
        train_data = lightgbm.Dataset(
            train_features,
            label=train_target,
            free_raw_data=False,
        )
        valid_sets = None
        valid_names = None
        callbacks = []
        if validation is not None:
            validation_features, validation_target = collect_dataset(
                validation,
                width=self.feature_width,
                seed=seed,
            )
            validation_data = lightgbm.Dataset(
                validation_features,
                label=validation_target,
                reference=train_data,
                free_raw_data=False,
            )
            valid_sets = [validation_data]
            valid_names = ["validation"]
            if self.early_stopping_rounds:
                callbacks.append(
                    lightgbm.early_stopping(
                        self.early_stopping_rounds,
                        verbose=False,
                    )
                )

        booster = lightgbm.train(
            {
                "objective": "regression",
                "metric": "l2",
                "learning_rate": float(self.learning_rate),
                "num_leaves": self.num_leaves,
                "max_depth": self.max_depth,
                "min_data_in_leaf": self.min_data_in_leaf,
                "seed": seed,
                "feature_fraction_seed": seed,
                "bagging_seed": seed,
                "data_random_seed": seed,
                "deterministic": True,
                "force_col_wise": True,
                "num_threads": 1,
                "verbosity": -1,
            },
            train_data,
            num_boost_round=self.num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )
        iteration = booster.best_iteration or booster.current_iteration()
        return LightGBMModel(
            feature_width=self.feature_width,
            num_boost_round=self.num_boost_round,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            max_depth=self.max_depth,
            min_data_in_leaf=self.min_data_in_leaf,
            early_stopping_rounds=self.early_stopping_rounds,
            model_text=booster.model_to_string(num_iteration=iteration),
        )

    def _predict(self, features: pl.Series) -> pl.Series:
        if self.model_text is None:
            return pl.repeat(
                0.0,
                len(features),
                dtype=pl.Float64,
                eager=True,
            )
        lightgbm = _import_lightgbm()
        values = feature_matrix(features, width=self.feature_width)
        booster = lightgbm.Booster(model_str=self.model_text)
        prediction = booster.predict(values)
        return pl.Series("prediction", prediction, dtype=pl.Float64)


@dataclass(frozen=True)
class LightGBMConfig(TSFNConfig):
    feature_width: int
    scheduler: Scheduler
    splitter: DatasetSplitter
    seed: int = 0
    num_boost_round: int = 100
    learning_rate: float = 0.05
    num_leaves: int = 31
    max_depth: int = -1
    min_data_in_leaf: int = 20
    early_stopping_rounds: int = 0
    timestamp_column: str = "timestamp"

    def __post_init__(self) -> None:
        if not isinstance(self.scheduler, Scheduler):
            raise TypeError("scheduler must be a Scheduler")
        if not isinstance(self.splitter, DatasetSplitter):
            raise TypeError("splitter must be a DatasetSplitter")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if not isinstance(self.timestamp_column, str) or not self.timestamp_column:
            raise ValueError("timestamp_column must be a non-empty string")
        LightGBMModel(
            feature_width=self.feature_width,
            num_boost_round=self.num_boost_round,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            max_depth=self.max_depth,
            min_data_in_leaf=self.min_data_in_leaf,
            early_stopping_rounds=self.early_stopping_rounds,
        )


class LightGBM(SupervisedModelTSFN):
    """Walk-forward LightGBM regression using the L2/MSE objective."""

    VERSION = "0.1.0"
    CONFIG_CLS = LightGBMConfig

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        params = self.parameters
        time = TimeAxis(params.timestamp_column)
        return (
            FrameSignature(
                time=time,
                columns=(
                    ("features", pl.Float64, (params.feature_width,)),
                    ("target", pl.Float64),
                ),
            ),
            FrameSignature(
                time=time,
                columns=(("prediction", pl.Float64),),
            ),
        )

    def initial_model(self) -> SupervisedModel:
        params = self.parameters
        return LightGBMModel(
            feature_width=params.feature_width,
            num_boost_round=params.num_boost_round,
            learning_rate=params.learning_rate,
            num_leaves=params.num_leaves,
            max_depth=params.max_depth,
            min_data_in_leaf=params.min_data_in_leaf,
            early_stopping_rounds=params.early_stopping_rounds,
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


__all__ = ["LightGBM", "LightGBMConfig", "LightGBMModel"]
