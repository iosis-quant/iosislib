from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import polars as pl
import torch
from torch import nn

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


def _validate_hidden_layers(hidden_layers: tuple[int, ...]) -> None:
    if any(isinstance(width, bool) or not isinstance(width, int) for width in hidden_layers):
        raise TypeError("hidden layer widths must be integers")
    if any(width < 1 for width in hidden_layers):
        raise ValueError("hidden layer widths must be positive")


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
    target_width: int,
    seed: int,
) -> float:
    total_squared_error = 0.0
    total_values = 0
    model.eval()
    with torch.no_grad():
        for features, target in dataset_arrays(
            dataset,
            feature_width=feature_width,
            target_width=target_width,
            seed=seed,
        ):
            feature_tensor = torch.from_numpy(features)
            target_tensor = torch.from_numpy(target)
            prediction = model(feature_tensor)
            total_squared_error += float(
                torch.sum((prediction - target_tensor) ** 2).item()
            )
            total_values += target.size
    if not total_values:
        raise ValueError("Validation dataset cannot be empty")
    return total_squared_error / total_values


@dataclass(frozen=True, kw_only=True)
class DenseMLPModel(SupervisedModel):
    """Immutable PyTorch dense-regression checkpoint.

    ``layers`` spans input width to output width, for example ``(8, 32, 16, 3)``
    where the input width comes from the bound features and the output width
    from the bound target.
    """

    VERSION = "0.2.0"

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
        target_width = self.layers[-1]
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
                    feature_width=feature_width,
                    target_width=target_width,
                    epoch=epoch,
                    seed=seed,
                ):
                    saw_batch = True
                    feature_tensor = torch.from_numpy(features)
                    target_tensor = torch.from_numpy(target)
                    optimizer.zero_grad(set_to_none=True)
                    prediction = model(feature_tensor)
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
                        target_width=target_width,
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
        output_width = self.layers[-1]
        if self.state is None:
            return pl.Series(
                "prediction",
                [[0.0] * output_width for _ in range(len(features))],
                dtype=pl.Array(pl.Float64, output_width),
            )
        values = feature_matrix(features, width=self.layers[0])
        model = _network(self.layers)
        _load_parameter_state(model, self.state)
        model.eval()
        with torch.no_grad():
            prediction = model(torch.from_numpy(values)).numpy()
        return pl.Series(
            "prediction",
            prediction,
            dtype=pl.Array(pl.Float64, output_width),
        )


@dataclass(frozen=True)
class DenseMLPConfig(TSFNConfig):
    """Configuration for a dense-regression TSFN.

    ``hidden_layers`` are the widths of the interior dense layers. The input
    width is the bound ``features`` width and the output width is the bound
    ``target`` width; both are derived from the graph unless explicitly
    configured. ``scheduler`` and ``splitter`` accept declarative mappings.
    """

    feature_width: int | None = None
    target_width: int | None = None
    hidden_layers: tuple[int, ...] = ()
    scheduler: Scheduler | Mapping[str, object] | None = None
    splitter: DatasetSplitter | Mapping[str, object] | None = None
    seed: int = 0
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    timestamp_column: str = "timestamp"

    def __post_init__(self) -> None:
        object.__setattr__(self, "hidden_layers", tuple(self.hidden_layers))
        _validate_hidden_layers(self.hidden_layers)
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
            DenseMLPModel(
                layers=(self.feature_width, *self.hidden_layers, self.target_width),
                epochs=self.epochs,
                learning_rate=self.learning_rate,
                weight_decay=self.weight_decay,
            )


class DenseMLP(SupervisedModelTSFN):
    """Walk-forward PyTorch dense regression trained with MSE loss."""

    VERSION = "0.2.0"
    CONFIG_CLS = DenseMLPConfig

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
        bound_input_columns: Mapping[str, object],
    ) -> tuple[FrameSignature, FrameSignature]:
        params = self.parameters
        feature_shape = self._resolve_shape(
            "features", bound_input_columns, params.feature_width
        )
        target_shape = self._resolve_shape(
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
                columns=(("prediction", pl.Float64, (shape_width(target_shape),)),),
            ),
        )

    @staticmethod
    def _resolve_shape(
        name: str,
        bound_input_columns: Mapping[str, object],
        configured: int | None,
    ) -> tuple[int, ...]:
        bound = bound_input_columns.get(name)
        if bound is not None:
            shape = getattr(bound, "shape", None) or ()
            width = shape_width(shape)
            if configured is not None and configured != width:
                raise ValueError(
                    f"Configured {name} width {configured} does not match the "
                    f"bound width {width}"
                )
            return shape
        if configured is not None:
            return (configured,)
        raise ValueError(
            f"{name} must be connected in the graph or have a configured width"
        )

    def initial_model(self) -> SupervisedModel:
        params = self.parameters
        feature_width, target_width = self._resolved_widths()
        return DenseMLPModel(
            layers=(feature_width, *params.hidden_layers, target_width),
            epochs=params.epochs,
            learning_rate=params.learning_rate,
            weight_decay=params.weight_decay,
        )

    def _resolved_widths(self) -> tuple[int, int]:
        columns = _column_signature_map(self.signature[0])
        return shape_width(columns["features"].shape), shape_width(
            columns["target"].shape
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
