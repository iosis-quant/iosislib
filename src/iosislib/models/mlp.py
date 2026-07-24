from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import polars as pl
import torch
from torch import nn

from iosislib.core.model import (
    Dataset,
    DatasetSplitter,
    Scheduler,
    SupervisedModel,
    SupervisedModelTSFN,
)
from iosislib.core.tsfn import FrameSignature, TSFNConfig, TimeAxis
from iosislib.models._regression import (
    dataset_arrays,
    feature_matrix,
    mean_squared_error,
)


ParameterState = tuple[tuple[float, ...], ...]


def _validate_layers(layers: tuple[int, ...]) -> None:
    if len(layers) < 2:
        raise ValueError("layers must contain at least input and output widths")
    if any(isinstance(width, bool) or not isinstance(width, int) for width in layers):
        raise TypeError("layer widths must be integers")
    if any(width < 1 for width in layers):
        raise ValueError("layer widths must be positive")
    if layers[-1] != 1:
        raise ValueError("The final dense layer must have width 1 for regression")


def _network(layers: tuple[int, ...]) -> nn.Sequential:
    modules: list[nn.Module] = []
    for index, (input_width, output_width) in enumerate(
        zip(layers[:-1], layers[1:], strict=True)
    ):
        modules.append(nn.Linear(input_width, output_width, dtype=torch.float64))
        if index < len(layers) - 2:
            modules.append(nn.ReLU())
    return nn.Sequential(*modules)


def _parameter_state(model: nn.Module) -> ParameterState:
    return tuple(
        tuple(float(value) for value in parameter.detach().cpu().reshape(-1).tolist())
        for parameter in model.parameters()
    )


def _load_parameter_state(model: nn.Module, state: ParameterState) -> None:
    parameters = tuple(model.parameters())
    if len(parameters) != len(state):
        raise ValueError("Dense MLP checkpoint does not match its layer architecture")
    with torch.no_grad():
        for parameter, values in zip(parameters, state, strict=True):
            if parameter.numel() != len(values):
                raise ValueError(
                    "Dense MLP checkpoint does not match its layer architecture"
                )
            parameter.copy_(
                torch.tensor(values, dtype=torch.float64).reshape(parameter.shape)
            )


def _validation_mse(
    model: nn.Module,
    dataset: Dataset,
    *,
    feature_width: int,
    seed: int,
) -> float:
    total_squared_error = 0.0
    total_rows = 0
    model.eval()
    with torch.no_grad():
        for features, target in dataset_arrays(
            dataset,
            width=feature_width,
            seed=seed,
        ):
            feature_tensor = torch.from_numpy(features)
            target_tensor = torch.from_numpy(target)
            prediction = model(feature_tensor).reshape(-1)
            total_squared_error += float(
                torch.sum((prediction - target_tensor) ** 2).item()
            )
            total_rows += len(target)
    if not total_rows:
        raise ValueError("Validation dataset cannot be empty")
    return total_squared_error / total_rows


@dataclass(frozen=True, kw_only=True)
class DenseMLPModel(SupervisedModel):
    """Immutable PyTorch dense-regression checkpoint."""

    VERSION = "0.1.0"

    layers: tuple[int, ...]
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    state: ParameterState | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        _validate_layers(self.layers)
        if (
            isinstance(self.epochs, bool)
            or not isinstance(self.epochs, int)
            or self.epochs < 1
        ):
            raise ValueError("epochs must be a positive integer")
        if (
            isinstance(self.learning_rate, bool)
            or not isinstance(self.learning_rate, (int, float))
            or self.learning_rate <= 0
        ):
            raise ValueError("learning_rate must be positive")
        if (
            isinstance(self.weight_decay, bool)
            or not isinstance(self.weight_decay, (int, float))
            or self.weight_decay < 0
        ):
            raise ValueError("weight_decay must be non-negative")

    def _fit(
        self,
        train: Dataset,
        validation: Dataset | None,
        *,
        seed: int,
    ) -> SupervisedModel:
        feature_width = self.layers[0]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = _network(self.layers)
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=float(self.learning_rate),
                weight_decay=float(self.weight_decay),
            )
            loss_function = nn.MSELoss()
            best_state: ParameterState | None = None
            best_validation_loss = float("inf")

            for epoch in range(self.epochs):
                model.train()
                saw_batch = False
                for features, target in dataset_arrays(
                    train,
                    width=feature_width,
                    epoch=epoch,
                    seed=seed,
                ):
                    saw_batch = True
                    feature_tensor = torch.from_numpy(features)
                    target_tensor = torch.from_numpy(target)
                    optimizer.zero_grad(set_to_none=True)
                    prediction = model(feature_tensor).reshape(-1)
                    loss = loss_function(prediction, target_tensor)
                    loss.backward()
                    optimizer.step()
                if not saw_batch:
                    raise ValueError("Cannot fit a dense MLP on an empty dataset")

                if validation is not None:
                    validation_loss = _validation_mse(
                        model,
                        validation,
                        feature_width=feature_width,
                        seed=seed,
                    )
                    if validation_loss < best_validation_loss:
                        best_validation_loss = validation_loss
                        best_state = _parameter_state(model)

            trained_state = (
                _parameter_state(model) if best_state is None else best_state
            )

        return DenseMLPModel(
            layers=self.layers,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            state=trained_state,
        )

    def _predict(self, features: pl.Series) -> pl.Series:
        if self.state is None:
            return pl.repeat(
                0.0,
                len(features),
                dtype=pl.Float64,
                eager=True,
            )
        values = feature_matrix(features, width=self.layers[0])
        model = _network(self.layers)
        _load_parameter_state(model, self.state)
        model.eval()
        with torch.no_grad():
            prediction = model(torch.from_numpy(values)).reshape(-1).numpy()
        return pl.Series("prediction", prediction, dtype=pl.Float64)


@dataclass(frozen=True)
class DenseMLPConfig(TSFNConfig):
    """Configuration for a scalar dense-regression TSFN.

    ``layers`` includes the input width and must end in one, for example
    ``(8, 32, 16, 1)``.
    """

    layers: tuple[int, ...]
    scheduler: Scheduler
    splitter: DatasetSplitter
    seed: int = 0
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    timestamp_column: str = "timestamp"

    def __post_init__(self) -> None:
        _validate_layers(self.layers)
        if not isinstance(self.scheduler, Scheduler):
            raise TypeError("scheduler must be a Scheduler")
        if not isinstance(self.splitter, DatasetSplitter):
            raise TypeError("splitter must be a DatasetSplitter")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if not isinstance(self.timestamp_column, str) or not self.timestamp_column:
            raise ValueError("timestamp_column must be a non-empty string")
        DenseMLPModel(
            layers=self.layers,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
        )


class DenseMLP(SupervisedModelTSFN):
    """Walk-forward PyTorch dense regression trained with MSE loss."""

    VERSION = "0.1.0"
    CONFIG_CLS = DenseMLPConfig

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        params = self.parameters
        time = TimeAxis(params.timestamp_column)
        return (
            FrameSignature(
                time=time,
                columns=(
                    ("features", pl.Float64, (params.layers[0],)),
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
        return DenseMLPModel(
            layers=params.layers,
            epochs=params.epochs,
            learning_rate=params.learning_rate,
            weight_decay=params.weight_decay,
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


__all__ = ["DenseMLP", "DenseMLPConfig", "DenseMLPModel"]
