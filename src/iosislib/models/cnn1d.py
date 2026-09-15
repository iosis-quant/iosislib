from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import polars as pl
import torch
import torch.nn.functional as F
from torch import nn

from iosislib.core.model import (
    ChronologicalSplitter,
    Dataset,
    DatasetSplitter,
    EveryNTicksScheduler,
    FrameDataset,
    MetricItems,
    ScheduleContext,
    Scheduler,
    SupervisedModel,
    SupervisedModelTSFN,
    _normalize_metrics,
    scheduler_from_declaration,
    shape_width,
    splitter_from_declaration,
    validate_optional_width,
)
from iosislib.core.tsfn import FrameSignature, TSFNConfig, TimeAxis, _column_signature_map
from iosislib.core.utils import _dtype_matches
from iosislib.core.tsfn import _frame_physical_schema
from iosislib.models._regression import (
    collect_dataset,
    feature_matrix,
    mean_squared_error,
)


ParameterState = tuple[tuple[float, ...], ...]
FloatArray = npt.NDArray[np.float64]


def _validate_sequence_length(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("sequence_length must be a positive integer")
    return value


def _validate_cnn_stack(
    channels: tuple[int, ...],
    kernel_sizes: tuple[int, ...],
    dilations: tuple[int, ...],
    sequence_length: int,
) -> None:
    if len(channels) < 1:
        raise ValueError("channels must contain at least one layer")
    if len(kernel_sizes) != len(channels):
        raise ValueError("kernel_sizes must have the same length as channels")
    if len(dilations) != len(channels):
        raise ValueError("dilations must have the same length as channels")
    for name, values in (
        ("channel widths", channels),
        ("kernel sizes", kernel_sizes),
        ("dilations", dilations),
    ):
        if any(isinstance(v, bool) or not isinstance(v, int) for v in values):
            raise TypeError(f"{name} must be integers")
        if any(v < 1 for v in values):
            raise ValueError(f"{name} must be positive")
    for kernel, dilation in zip(kernel_sizes, dilations, strict=True):
        effective = (kernel - 1) * dilation + 1
        if kernel > sequence_length or effective > sequence_length:
            raise ValueError(
                f"kernel {kernel} with dilation {dilation} needs "
                f"effective width {effective} but sequence_length is {sequence_length}"
            )


def _validate_dense_hidden(dense_hidden: tuple[int, ...]) -> None:
    if any(isinstance(w, bool) or not isinstance(w, int) for w in dense_hidden):
        raise TypeError("dense_hidden widths must be integers")
    if any(w < 1 for w in dense_hidden):
        raise ValueError("dense_hidden widths must be positive")


def _validate_dropout(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("dropout must be numeric")
    numeric = float(value)
    if not 0.0 <= numeric < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    return numeric


class _CausalConv1d(nn.Module):
    """Conv1d with left-only padding so output at t sees only past inputs."""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, dilation: int
    ) -> None:
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            dtype=torch.float64,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad:
            x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class _TCNRegressor(nn.Module):
    """Causal TCN: dilated conv stack over time, read last timestep, dense head."""

    def __init__(
        self,
        feature_width: int,
        channels: tuple[int, ...],
        kernel_sizes: tuple[int, ...],
        dilations: tuple[int, ...],
        dropout: float,
        dense_hidden: tuple[int, ...],
        output_width: int,
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        in_channels = feature_width
        for out_channels, kernel, dilation in zip(
            channels, kernel_sizes, dilations, strict=True
        ):
            blocks.append(_CausalConv1d(in_channels, out_channels, kernel, dilation))
            blocks.append(nn.ReLU())
            if dropout > 0:
                blocks.append(nn.Dropout(dropout))
            in_channels = out_channels
        self.conv_stack = nn.Sequential(*blocks)
        head_sizes = (in_channels, *dense_hidden, output_width)
        head: list[nn.Module] = []
        for index, (in_w, out_w) in enumerate(
            zip(head_sizes[:-1], head_sizes[1:], strict=True)
        ):
            head.append(nn.Linear(in_w, out_w, dtype=torch.float64))
            if index < len(head_sizes) - 2:
                head.append(nn.ReLU())
        self.head = nn.Sequential(*head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.conv_stack(x)
        last_step = hidden[:, :, -1]
        return self.head(last_step)


def _build_network(
    feature_width: int,
    channels: tuple[int, ...],
    kernel_sizes: tuple[int, ...],
    dilations: tuple[int, ...],
    dropout: float,
    dense_hidden: tuple[int, ...],
    output_width: int,
) -> nn.Module:
    return _TCNRegressor(
        feature_width,
        channels,
        kernel_sizes,
        dilations,
        dropout,
        dense_hidden,
        output_width,
    )


def _parameter_state(model: nn.Module) -> ParameterState:
    return tuple(
        tuple(float(value) for value in parameter.detach().cpu().reshape(-1).tolist())
        for parameter in model.parameters()
    )


def _load_parameter_state(model: nn.Module, state: ParameterState) -> None:
    parameters = tuple(model.parameters())
    if len(parameters) != len(state):
        raise ValueError("CNN1D checkpoint does not match its architecture")
    with torch.no_grad():
        for parameter, values in zip(parameters, state, strict=True):
            if parameter.numel() != len(values):
                raise ValueError("CNN1D checkpoint does not match its architecture")
            parameter.copy_(
                torch.tensor(values, dtype=torch.float64).reshape(parameter.shape)
            )


def _require_finite(a: FloatArray, b: FloatArray) -> None:
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("CNN1D training data must contain only finite values")


def _reject_shuffled(dataset: Dataset) -> int | None:
    if isinstance(dataset, FrameDataset):
        if dataset.shuffle:
            raise ValueError(
                "CNN1D requires shuffle_train=False: shuffling rows destroys "
                "temporal contiguity needed for sequence windows"
            )
        return dataset.batch_size
    return None


def _make_windows(
    features: FloatArray, target: FloatArray, sequence_length: int
) -> tuple[FloatArray, FloatArray]:
    count = features.shape[0]
    if count < sequence_length:
        raise ValueError(
            f"Need at least {sequence_length} rows for one window, got {count}"
        )
    starts = range(count - sequence_length + 1)
    windows = np.ascontiguousarray(
        np.stack([features[start : start + sequence_length].T for start in starts])
    )
    targets = np.ascontiguousarray(target[sequence_length - 1 :])
    return windows, targets


def _window_batches(
    inputs: FloatArray,
    targets: FloatArray,
    batch_size: int | None,
) -> list[tuple[FloatArray, FloatArray]]:
    size = batch_size or inputs.shape[0]
    return [
        (inputs[offset : offset + size], targets[offset : offset + size])
        for offset in range(0, inputs.shape[0], size)
    ]


def _fit_windows(
    model: nn.Module,
    inputs: FloatArray,
    targets: FloatArray,
    *,
    batch_size: int | None,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
) -> None:
    model.train()
    for batch_x, batch_y in _window_batches(inputs, targets, batch_size):
        _require_finite(batch_x.reshape(batch_x.shape[0], -1), batch_y)
        feature_tensor = torch.from_numpy(batch_x)
        target_tensor = torch.from_numpy(batch_y)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(model(feature_tensor), target_tensor)
        loss.backward()
        optimizer.step()


def _windows_mse(
    model: nn.Module,
    dataset: Dataset,
    *,
    feature_width: int,
    target_width: int,
    sequence_length: int,
    seed: int,
) -> float:
    features, target = collect_dataset(
        dataset,
        feature_width=feature_width,
        target_width=target_width,
        seed=seed,
    )
    inputs, targets = _make_windows(features, target, sequence_length)
    total_squared_error = 0.0
    total_values = 0
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in _window_batches(inputs, targets, None):
            prediction = model(torch.from_numpy(batch_x)).numpy()
            total_squared_error += float(np.sum((prediction - batch_y) ** 2))
            total_values += batch_y.size
    if not total_values:
        raise ValueError("Validation dataset cannot be empty")
    return total_squared_error / total_values


@dataclass(frozen=True, kw_only=True)
class CNN1DModel(SupervisedModel):
    """Immutable causal-TCN regression checkpoint over time windows.

    Each training example is a window of ``sequence_length`` consecutive rows;
    the label is the target at the window's last row. Convolution runs over
    the time axis with left-only (causal) padding and dilations, so a
    prediction at time ``t`` never sees rows after ``t``.
    """

    VERSION = "0.2.0"

    feature_width: int
    sequence_length: int
    channels: tuple[int, ...]
    kernel_sizes: tuple[int, ...]
    dilations: tuple[int, ...]
    dense_hidden: tuple[int, ...]
    output_width: int
    dropout: float = 0.0
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    state: ParameterState | None = None

    @property
    def is_trained(self) -> bool:
        return self.state is not None

    def __post_init__(self) -> None:
        super().__post_init__()
        validate_optional_width("feature_width", self.feature_width)
        _validate_sequence_length(self.sequence_length)
        _validate_cnn_stack(
            self.channels, self.kernel_sizes, self.dilations, self.sequence_length
        )
        _validate_dense_hidden(self.dense_hidden)
        _validate_dropout(self.dropout)
        if self.output_width < 1:
            raise ValueError("output_width must be positive")
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

    def _network(self) -> nn.Module:
        return _build_network(
            self.feature_width,
            self.channels,
            self.kernel_sizes,
            self.dilations,
            self.dropout,
            self.dense_hidden,
            self.output_width,
        )

    def _fit(
        self,
        train: Dataset,
        validation: Dataset | None,
        *,
        seed: int,
    ) -> SupervisedModel:
        batch_size = _reject_shuffled(train)
        if validation is not None:
            _reject_shuffled(validation)
        features, target = collect_dataset(
            train,
            feature_width=self.feature_width,
            target_width=self.output_width,
            seed=seed,
        )
        inputs, targets = _make_windows(features, target, self.sequence_length)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = self._network()
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=float(self.learning_rate),
                weight_decay=float(self.weight_decay),
            )
            loss_function = nn.MSELoss()
            best_state: ParameterState | None = None
            best_validation_loss = float("inf")

            for _ in range(self.epochs):
                _fit_windows(
                    model,
                    inputs,
                    targets,
                    batch_size=batch_size,
                    optimizer=optimizer,
                    loss_function=loss_function,
                )
                if validation is not None:
                    validation_loss = _windows_mse(
                        model,
                        validation,
                        feature_width=self.feature_width,
                        target_width=self.output_width,
                        sequence_length=self.sequence_length,
                        seed=seed,
                    )
                    if validation_loss < best_validation_loss:
                        best_validation_loss = validation_loss
                        best_state = _parameter_state(model)

            trained_state = (
                _parameter_state(model) if best_state is None else best_state
            )

        return CNN1DModel(
            feature_width=self.feature_width,
            sequence_length=self.sequence_length,
            channels=self.channels,
            kernel_sizes=self.kernel_sizes,
            dilations=self.dilations,
            dense_hidden=self.dense_hidden,
            output_width=self.output_width,
            dropout=self.dropout,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            state=trained_state,
        )

    def _predict(self, features: pl.Series) -> pl.Series:
        if self.state is None:
            return pl.Series(
                "prediction",
                [[float("nan")] * self.output_width for _ in range(len(features))],
                dtype=pl.Array(pl.Float64, self.output_width),
            )
        values = feature_matrix(features, width=self.feature_width)
        count = values.shape[0]
        predictions: list[list[float]] = [
            [float("nan")] * self.output_width for _ in range(count)
        ]
        if count >= self.sequence_length:
            model = self._network()
            _load_parameter_state(model, self.state)
            model.eval()
            full, _ = _make_windows(
                values,
                np.zeros((count, self.output_width), dtype=np.float64),
                self.sequence_length,
            )
            with torch.no_grad():
                output = model(torch.from_numpy(full)).numpy()
            if not np.isfinite(output).all():
                raise ValueError("CNN1DModel produced non-finite predictions")
            for row, pred in enumerate(output, start=self.sequence_length - 1):
                predictions[row] = [float(v) for v in pred.tolist()]
        return pl.Series(
            "prediction",
            predictions,
            dtype=pl.Array(pl.Float64, self.output_width),
        )


@dataclass(frozen=True)
class CNN1DConfig(TSFNConfig):
    """Configuration for a causal 1D-CNN (TCN) regression TSFN.

    ``sequence_length`` is the lookback window: each prediction at time ``t``
    sees rows ``t-L+1..t``. ``channels``/``kernel_sizes``/``dilations`` define
    the causal conv stack over time (channels = per-layer output widths).
    ``dense_hidden`` sizes the head applied to the last timestep; the final
    output width always comes from the bound target.

    Gap/purge embargo: ``splitter`` ``gap`` rows are dropped between
    train/validation/test and ``purge_window`` drops label-unavailable tail
    rows. Windows are built separately inside each split, so no window ever
    spans a split boundary and the embargo holds in window space too.
    """

    feature_width: int | None = None
    target_width: int | None = None
    sequence_length: int = 32
    channels: tuple[int, ...] = (16,)
    kernel_sizes: tuple[int, ...] = (3,)
    dilations: tuple[int, ...] = ()
    dense_hidden: tuple[int, ...] = ()
    dropout: float = 0.0
    scheduler: Scheduler | Mapping[str, object] | None = None
    splitter: DatasetSplitter | Mapping[str, object] | None = None
    seed: int = 0
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    timestamp_column: str = "timestamp"

    def __post_init__(self) -> None:
        _validate_sequence_length(self.sequence_length)
        object.__setattr__(self, "channels", tuple(self.channels))
        object.__setattr__(self, "kernel_sizes", tuple(self.kernel_sizes))
        dilations = tuple(self.dilations) if self.dilations else (1,) * len(
            self.channels
        )
        object.__setattr__(self, "dilations", dilations)
        object.__setattr__(self, "dense_hidden", tuple(self.dense_hidden))
        _validate_cnn_stack(
            self.channels, self.kernel_sizes, self.dilations, self.sequence_length
        )
        _validate_dense_hidden(self.dense_hidden)
        object.__setattr__(self, "dropout", _validate_dropout(self.dropout))
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
        shuffle = getattr(self.splitter, "shuffle_train", False)
        if shuffle:
            raise ValueError(
                "CNN1D requires shuffle_train=False: shuffling rows destroys "
                "temporal contiguity needed for sequence windows"
            )
        if self.feature_width is not None and self.target_width is not None:
            CNN1DModel(
                feature_width=self.feature_width,
                sequence_length=self.sequence_length,
                channels=self.channels,
                kernel_sizes=self.kernel_sizes,
                dilations=self.dilations,
                dense_hidden=self.dense_hidden,
                output_width=self.target_width,
                dropout=self.dropout,
                epochs=self.epochs,
                learning_rate=self.learning_rate,
                weight_decay=self.weight_decay,
            )


class CNN1D(SupervisedModelTSFN):
    """Walk-forward causal-TCN regression trained with MSE loss.

    Overrides ``batch()`` to be history-aware: each predicted segment is
    prepended with its ``sequence_length - 1`` predecessors so windows at the
    segment start still see full history, while training still uses only rows
    before the cursor. The first ``sequence_length - 1`` rows overall emit
    NaN predictions (warmup).
    """

    VERSION = "0.2.0"
    CONFIG_CLS = CNN1DConfig

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
                columns=(
                    ("prediction", pl.Float64, (max(shape_width(target_shape), 1),)),
                ),
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
        return CNN1DModel(
            feature_width=feature_width,
            sequence_length=params.sequence_length,
            channels=params.channels,
            kernel_sizes=params.kernel_sizes,
            dilations=params.dilations,
            dense_hidden=params.dense_hidden,
            output_width=target_width,
            dropout=params.dropout,
            epochs=params.epochs,
            learning_rate=params.learning_rate,
            weight_decay=params.weight_decay,
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

    def batch(self, frame: pl.DataFrame) -> pl.DataFrame:
        if frame.is_empty():
            return pl.DataFrame(schema=_frame_physical_schema(self.signature[1]))

        time_column = self._time_column()
        ordered = (
            frame
            if frame.get_column(time_column).is_sorted()
            else frame.sort(time_column)
        )
        supervised = ordered.select(self.FEATURE_COLUMN, self.TARGET_COLUMN)

        scheduler = self.scheduler()
        splitter = self.splitter()
        if not isinstance(scheduler, Scheduler):
            raise TypeError("SupervisedModelTSFN.scheduler must return a Scheduler")
        if not isinstance(splitter, DatasetSplitter):
            raise TypeError(
                "SupervisedModelTSFN.splitter must return a DatasetSplitter"
            )
        if getattr(splitter, "shuffle_train", False):
            raise ValueError(
                "CNN1D requires shuffle_train=False: shuffling rows destroys "
                "temporal contiguity needed for sequence windows"
            )

        active_model = self.initial_model()
        if not isinstance(active_model, SupervisedModel):
            raise TypeError(
                "SupervisedModelTSFN.initial_model must return a SupervisedModel"
            )

        lookback = max(int(self.parameters.sequence_length) - 1, 0)
        outputs: list[pl.DataFrame] = []
        cursor = 0
        last_retrain_at = 0
        retrain_count = 0
        metrics: MetricItems = ()

        while cursor < ordered.height:
            context = ScheduleContext(
                total_rows=ordered.height,
                rows_seen=cursor,
                rows_since_retrain=cursor - last_retrain_at,
                retrain_count=retrain_count,
                metrics=metrics,
            )
            decision = scheduler.decide(context)

            if decision.retrain:
                seed = self.training_seed(retrain_count)
                if isinstance(seed, bool) or not isinstance(seed, int):
                    raise TypeError("training_seed must return an integer")
                datasets = splitter.split(supervised.slice(0, cursor), seed=seed)
                active_model = active_model.fit(datasets, seed=seed)
                last_retrain_at = cursor
                retrain_count += 1

            segment_length = decision.predict_until - cursor
            segment = ordered.slice(cursor, segment_length)
            history_start = max(0, cursor - lookback)
            window_frame = ordered.slice(
                history_start, segment_length + cursor - history_start
            )
            full_prediction = active_model.predict(
                window_frame.get_column(self.FEATURE_COLUMN)
            )
            prediction = full_prediction.slice(
                cursor - history_start, segment_length
            )
            expected_prediction = self.output_column_signature(
                self.PREDICTION_COLUMN
            ).physical_dtype
            if not _dtype_matches(prediction.dtype, expected_prediction):
                raise TypeError(
                    "Model prediction type mismatch. Expected "
                    f"{expected_prediction}, got {prediction.dtype}"
                )

            metrics = ()
            if active_model.is_trained:
                metrics = _normalize_metrics(
                    self.segment_metrics(
                        segment.get_column(self.TARGET_COLUMN),
                        prediction,
                    )
                )
            outputs.append(
                pl.DataFrame(
                    [
                        segment.get_column(time_column),
                        prediction.rename(self.PREDICTION_COLUMN),
                    ]
                )
            )
            cursor = decision.predict_until

        return pl.concat(outputs, how="vertical", rechunk=False)


__all__ = ["CNN1D", "CNN1DConfig", "CNN1DModel"]
