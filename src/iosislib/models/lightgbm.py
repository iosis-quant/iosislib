from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

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
from iosislib.core.tsfn import FrameSignature, TSFNConfig, TimeAxis, _column_signature_map
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
    """Immutable LightGBM multi-output regression checkpoint."""

    VERSION = "0.2.0"

    feature_width: int
    target_width: int
    num_boost_round: int = 100
    learning_rate: float = 0.05
    num_leaves: int = 31
    max_depth: int = -1
    min_data_in_leaf: int = 20
    early_stopping_rounds: int = 0
    model_text: str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in ("feature_width", "target_width", "num_boost_round", "num_leaves"):
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
            feature_width=self.feature_width,
            target_width=self.target_width,
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
                feature_width=self.feature_width,
                target_width=self.target_width,
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
            target_width=self.target_width,
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
            return pl.Series(
                "prediction",
                [[0.0] * self.target_width for _ in range(len(features))],
                dtype=pl.Array(pl.Float64, self.target_width),
            )
        lightgbm = _import_lightgbm()
        values = feature_matrix(features, width=self.feature_width)
        booster = lightgbm.Booster(model_str=self.model_text)
        prediction = booster.predict(values)
        return pl.Series(
            "prediction",
            prediction,
            dtype=pl.Array(pl.Float64, self.target_width),
        )


@dataclass(frozen=True)
class LightGBMConfig(TSFNConfig):
    """Configuration for a LightGBM regression TSFN.

    ``feature_width`` and ``target_width`` are derived from the graph bindings
    unless explicitly configured. ``scheduler`` and ``splitter`` accept
    declarative mappings.
    """

    feature_width: int | None = None
    target_width: int | None = None
    scheduler: Scheduler | Mapping[str, object] | None = None
    splitter: DatasetSplitter | Mapping[str, object] | None = None
    seed: int = 0
    num_boost_round: int = 100
    learning_rate: float = 0.05
    num_leaves: int = 31
    max_depth: int = -1
    min_data_in_leaf: int = 20
    early_stopping_rounds: int = 0
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
            LightGBMModel(
                feature_width=self.feature_width,
                target_width=self.target_width,
                num_boost_round=self.num_boost_round,
                learning_rate=self.learning_rate,
                num_leaves=self.num_leaves,
                max_depth=self.max_depth,
                min_data_in_leaf=self.min_data_in_leaf,
                early_stopping_rounds=self.early_stopping_rounds,
            )


class LightGBM(SupervisedModelTSFN):
    """Walk-forward LightGBM regression using the L2/MSE objective."""

    VERSION = "0.2.0"
    CONFIG_CLS = LightGBMConfig

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        params = self.parameters
        feature_width = params.feature_width or 0
        target_width = params.target_width or 0
        time = TimeAxis(params.timestamp_column)
        return (
            FrameSignature(
                time=time,
                columns=(
                    ("features", pl.Float64, (feature_width,)),
                    ("target", pl.Float64, (target_width,)),
                ),
            ),
            FrameSignature(
                time=time,
                columns=(("prediction", pl.Float64, (target_width,)),),
            ),
        )

    def resolve_signature(
        self,
        bound_input_columns: Mapping[str, object],
    ) -> tuple[FrameSignature, FrameSignature]:
        params = self.parameters
        feature_width = self._derive_width(
            "features", bound_input_columns, params.feature_width
        )
        target_width = self._derive_width(
            "target", bound_input_columns, params.target_width
        )
        time = self.signature[0].time
        return (
            FrameSignature(
                time=time,
                columns=(
                    ("features", pl.Float64, (feature_width,)),
                    ("target", pl.Float64, (target_width,)),
                ),
            ),
            FrameSignature(
                time=time,
                columns=(("prediction", pl.Float64, (target_width,)),),
            ),
        )

    @staticmethod
    def _derive_width(
        name: str,
        bound_input_columns: Mapping[str, object],
        configured: int | None,
    ) -> int:
        bound = bound_input_columns.get(name)
        if bound is not None:
            width = shape_width(getattr(bound, "shape", None))
            if configured is not None and configured != width:
                raise ValueError(
                    f"Configured {name} width {configured} does not match the "
                    f"bound width {width}"
                )
            return width
        if configured is not None:
            return configured
        raise ValueError(
            f"{name} must be connected in the graph or have a configured width"
        )

    def initial_model(self) -> SupervisedModel:
        params = self.parameters
        feature_width, target_width = self._resolved_widths()
        return LightGBMModel(
            feature_width=feature_width,
            target_width=target_width,
            num_boost_round=params.num_boost_round,
            learning_rate=params.learning_rate,
            num_leaves=params.num_leaves,
            max_depth=params.max_depth,
            min_data_in_leaf=params.min_data_in_leaf,
            early_stopping_rounds=params.early_stopping_rounds,
        )

    def _resolved_widths(self) -> tuple[int, int]:
        columns = _column_signature_map(self.signature[0])
        feature_width = shape_width(columns["features"].shape)
        target_width = shape_width(columns["target"].shape)
        if feature_width == 0 or target_width == 0:
            raise ValueError(
                "features and target widths must be resolved from the graph "
                "before fitting"
            )
        return feature_width, target_width

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
