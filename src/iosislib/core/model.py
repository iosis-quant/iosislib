from __future__ import annotations

import abc
import hashlib
import importlib
import inspect
import json
import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, fields
from datetime import date, datetime, time, timedelta, timezone
from math import ceil, isfinite
from pathlib import Path
from typing import Any, ClassVar, cast

import polars as pl

from iosislib.core.tsfn import BatchTSFN, _frame_physical_schema
from iosislib.core.utils import (
    _canonical_json,
    _dtype_matches,
    _flat_size,
    _qualified_type_name,
    _serialize_value,
    _series_null_count,
    current_s3_credentials,
    resolve_model_dir,
)


class Dataset(abc.ABC):
    """A finite, repeatable source of schema-stable Polars batches."""

    @property
    @abc.abstractmethod
    def row_count(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def schema(self) -> pl.Schema:
        pass

    def batches(self, *, epoch: int = 0, seed: int = 0) -> Iterator[pl.DataFrame]:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("Dataset epoch must be a non-negative integer")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("Dataset seed must be an integer")

        declared_rows = self.row_count
        if (
            isinstance(declared_rows, bool)
            or not isinstance(declared_rows, int)
            or declared_rows < 0
        ):
            raise ValueError("Dataset row_count must be a non-negative integer")
        declared_schema = self.schema
        if not isinstance(declared_schema, pl.Schema):
            raise TypeError("Dataset schema must be a Polars Schema")

        yielded_rows = 0
        for batch in self._batches(epoch=epoch, seed=seed):
            if not isinstance(batch, pl.DataFrame):
                raise TypeError("Dataset batches must be Polars DataFrames")
            if batch.is_empty():
                raise ValueError("Dataset batches cannot be empty")
            if batch.schema != declared_schema:
                raise TypeError(
                    "Dataset batch schema does not match the declared dataset schema"
                )
            yielded_rows += batch.height
            yield batch

        if declared_rows and not yielded_rows:
            raise RuntimeError("Non-empty Dataset yielded no batches")
        if yielded_rows > declared_rows:
            raise RuntimeError("Dataset yielded more rows than its declared row count")

    @abc.abstractmethod
    def _batches(self, *, epoch: int, seed: int) -> Iterator[pl.DataFrame]:
        pass


@dataclass(frozen=True)
class FrameDataset(Dataset):
    """A materialized frame exposed as deterministic mini-batches."""

    frame: pl.DataFrame
    batch_size: int | None = None
    shuffle: bool = False
    drop_last: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.frame, pl.DataFrame):
            raise TypeError("FrameDataset.frame must be a Polars DataFrame")
        if self.frame.is_empty():
            raise ValueError("FrameDataset.frame cannot be empty")
        if self.batch_size is not None and (
            isinstance(self.batch_size, bool)
            or not isinstance(self.batch_size, int)
            or self.batch_size < 1
        ):
            raise ValueError("FrameDataset.batch_size must be positive or None")
        if not isinstance(self.shuffle, bool):
            raise TypeError("FrameDataset.shuffle must be a boolean")
        if not isinstance(self.drop_last, bool):
            raise TypeError("FrameDataset.drop_last must be a boolean")
        if self.drop_last and self.batch_size is None:
            raise ValueError("drop_last requires a finite batch_size")
        if self.drop_last and self.frame.height < self.batch_size:
            raise ValueError("drop_last would discard the entire dataset")

    @property
    def row_count(self) -> int:
        return self.frame.height

    @property
    def schema(self) -> pl.Schema:
        return self.frame.schema

    def _batches(self, *, epoch: int, seed: int) -> Iterator[pl.DataFrame]:
        frame = self.frame
        if self.shuffle:
            digest = hashlib.sha256(f"{seed}:{epoch}".encode("ascii")).digest()
            epoch_seed = int.from_bytes(digest[:4], "big")
            frame = frame.sample(
                fraction=1.0,
                shuffle=True,
                seed=epoch_seed,
            )

        batch_size = self.batch_size or frame.height
        for offset in range(0, frame.height, batch_size):
            batch = frame.slice(offset, batch_size)
            if self.drop_last and batch.height < batch_size:
                break
            yield batch

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": _qualified_type_name(self),
            "rows": self.row_count,
            "schema": {name: str(dtype) for name, dtype in self.schema.items()},
            "batch_size": self.batch_size,
            "shuffle": self.shuffle,
            "drop_last": self.drop_last,
        }

    def __str__(self) -> str:
        return _canonical_json(self.to_dict())

    def __repr__(self) -> str:
        return (
            f"FrameDataset(rows={self.row_count}, batch_size={self.batch_size!r}, "
            f"shuffle={self.shuffle}, drop_last={self.drop_last})"
        )


@dataclass(frozen=True)
class DatasetSplit:
    """The supervised datasets produced for one fitting operation."""

    train: Dataset
    validation: Dataset | None = None
    test: Dataset | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.train, Dataset):
            raise TypeError("DatasetSplit.train must be a Dataset")
        expected_schema = self.train.schema
        if list(expected_schema.names()) != ["features", "target"]:
            raise ValueError(
                "DatasetSplit requires canonical ['features', 'target'] columns"
            )
        for name in ("validation", "test"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, Dataset):
                raise TypeError(f"DatasetSplit.{name} must be a Dataset or None")
            if value is not None and value.schema != expected_schema:
                raise TypeError(
                    f"DatasetSplit.{name} schema must match the training schema"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_rows": self.train.row_count,
            "validation_rows": (
                None if self.validation is None else self.validation.row_count
            ),
            "test_rows": None if self.test is None else self.test.row_count,
        }

    def __str__(self) -> str:
        return _canonical_json(self.to_dict())


class DatasetSplitter(abc.ABC):
    """A deterministic partition of already ordered supervised examples."""

    def split(self, frame: pl.DataFrame, *, seed: int = 0) -> DatasetSplit:
        if not isinstance(frame, pl.DataFrame):
            raise TypeError("DatasetSplitter requires a Polars DataFrame")
        if frame.is_empty():
            raise ValueError("DatasetSplitter cannot split an empty frame")
        if frame.columns != ["features", "target"]:
            raise ValueError(
                "DatasetSplitter requires canonical ['features', 'target'] columns"
            )
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("DatasetSplitter seed must be an integer")
        result = self._split(frame, seed=seed)
        if not isinstance(result, DatasetSplit):
            raise TypeError("DatasetSplitter._split must return a DatasetSplit")
        split_rows = sum(
            dataset.row_count
            for dataset in (result.train, result.validation, result.test)
            if dataset is not None
        )
        if split_rows > frame.height:
            raise ValueError("DatasetSplitter produced more rows than its input frame")
        return result

    @abc.abstractmethod
    def _split(self, frame: pl.DataFrame, *, seed: int) -> DatasetSplit:
        pass

    def __str__(self) -> str:
        return _canonical_json(self)


@dataclass(frozen=True)
class ChronologicalSplitter(DatasetSplitter):
    """Tail holdouts over the row order established by the feature graph.

    ``purge_window`` treats the final rows of the input as label-unavailable:
    they are excluded from the whole split because their targets only become
    observable after a retraining boundary.
    """

    validation_size: int | float = 0
    test_size: int | float = 0
    gap: int = 0
    batch_size: int | None = None
    shuffle_train: bool = False
    drop_last: bool = False
    purge_window: int = 0

    def __post_init__(self) -> None:
        for name in ("validation_size", "test_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be an integer count or float fraction")
            if isinstance(value, int) and value < 0:
                raise ValueError(f"{name} cannot be negative")
            if isinstance(value, float) and not 0.0 <= value < 1.0:
                raise ValueError(f"{name} fraction must be in [0, 1)")
        if isinstance(self.gap, bool) or not isinstance(self.gap, int) or self.gap < 0:
            raise ValueError("gap must be a non-negative integer")
        if (
            isinstance(self.purge_window, bool)
            or not isinstance(self.purge_window, int)
            or self.purge_window < 0
        ):
            raise ValueError("purge_window must be a non-negative integer")
        if self.batch_size is not None and (
            isinstance(self.batch_size, bool)
            or not isinstance(self.batch_size, int)
            or self.batch_size < 1
        ):
            raise ValueError("batch_size must be positive or None")
        if not isinstance(self.shuffle_train, bool):
            raise TypeError("shuffle_train must be a boolean")
        if not isinstance(self.drop_last, bool):
            raise TypeError("drop_last must be a boolean")
        if self.drop_last and self.batch_size is None:
            raise ValueError("drop_last requires a finite batch_size")

    @staticmethod
    def _size(value: int | float, total: int) -> int:
        if isinstance(value, int):
            return value
        return ceil(total * value) if value else 0

    def _split(self, frame: pl.DataFrame, *, seed: int) -> DatasetSplit:
        del seed
        if self.purge_window >= frame.height:
            raise ValueError("purge_window leaves no training rows")
        if self.purge_window:
            frame = frame.slice(0, frame.height - self.purge_window)
        validation_size = self._size(self.validation_size, frame.height)
        test_size = self._size(self.test_size, frame.height)
        cursor = frame.height

        test_frame: pl.DataFrame | None = None
        if test_size:
            cursor -= test_size
            if cursor < 0:
                raise ValueError("test_size exceeds the available rows")
            test_frame = frame.slice(cursor, test_size)
            cursor -= self.gap

        validation_frame: pl.DataFrame | None = None
        if validation_size:
            cursor -= validation_size
            if cursor < 0:
                raise ValueError("validation_size and test_size leave no training rows")
            validation_frame = frame.slice(cursor, validation_size)
            cursor -= self.gap

        if cursor <= 0:
            raise ValueError("Split sizes and gaps leave no training rows")
        train_frame = frame.slice(0, cursor)

        return DatasetSplit(
            train=FrameDataset(
                train_frame,
                batch_size=self.batch_size,
                shuffle=self.shuffle_train,
                drop_last=self.drop_last,
            ),
            validation=(
                None
                if validation_frame is None
                else FrameDataset(validation_frame, batch_size=self.batch_size)
            ),
            test=(
                None
                if test_frame is None
                else FrameDataset(test_frame, batch_size=self.batch_size)
            ),
        )


@dataclass(frozen=True, kw_only=True)
class Model(abc.ABC):
    """An immutable, serializable inference checkpoint."""

    VERSION: ClassVar[str]
    _SERIALIZE_WITH_TO_DICT: ClassVar[bool] = True

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if not inspect.isabstract(cls):
            cls._validate_version()

    @classmethod
    def _validate_version(cls) -> str:
        if "VERSION" not in cls.__dict__:
            raise TypeError(f"Model subclass '{cls.__name__}' must define VERSION")
        if not isinstance(cls.VERSION, str):
            raise TypeError(f"Model subclass '{cls.__name__}' VERSION must be a string")
        if not cls.VERSION.strip():
            raise ValueError(f"Model subclass '{cls.__name__}' VERSION must be non-empty")
        return cls.VERSION

    def __post_init__(self) -> None:
        self._validate_version()

    @property
    def version(self) -> str:
        return self._validate_version()

    @property
    def is_trained(self) -> bool:
        """Whether this checkpoint can produce a real prediction.

        Checkpoints that have not been trained yet return ``False`` so a
        walk-forward loop can refuse to emit model metrics for rows it has no
        model for, instead of scoring against a placeholder prediction.
        """
        return True

    def predict(self, features: pl.Series) -> pl.Series:
        if not isinstance(features, pl.Series):
            raise TypeError("Model.predict requires a Polars Series")
        if features.is_empty():
            raise ValueError("Model.predict cannot predict an empty feature batch")
        feature_nulls = _series_null_count(features)
        if feature_nulls:
            raise ValueError(
                f"Model.predict received {feature_nulls} null feature value(s)"
            )
        output = self._predict(features)
        if not isinstance(output, pl.Series):
            raise TypeError("Model._predict must return a Polars Series")
        if len(output) != len(features):
            raise ValueError("Model prediction must preserve feature row count")
        output_nulls = _series_null_count(output)
        if output_nulls:
            raise ValueError(
                f"Model._predict returned {output_nulls} null prediction value(s)"
            )
        return output.rename("prediction")

    @abc.abstractmethod
    def _predict(self, features: pl.Series) -> pl.Series:
        pass

    def to_dict(self) -> dict[str, Any]:
        return {
            "module": type(self).__module__,
            "qualname": type(self).__qualname__,
            "version": self.version,
            "state": {
                item.name: _serialize_value(getattr(self, item.name))
                for item in fields(self)
            },
        }

    def __str__(self) -> str:
        return _canonical_json(self.to_dict())

    def __repr__(self) -> str:
        state = ", ".join(
            f"{item.name}={getattr(self, item.name)!r}" for item in fields(self)
        )
        suffix = f", {state}" if state else ""
        return f"{type(self).__name__}(version={self.version!r}{suffix})"


@dataclass(frozen=True, kw_only=True)
class SupervisedModel(Model, abc.ABC):
    """A checkpoint that can fit supervised data and return a new checkpoint."""

    def fit(self, datasets: DatasetSplit, *, seed: int) -> SupervisedModel:
        if not isinstance(datasets, DatasetSplit):
            raise TypeError("SupervisedModel.fit requires a DatasetSplit")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("SupervisedModel fit seed must be an integer")

        checkpoint = self._fit(
            datasets.train,
            datasets.validation,
            seed=seed,
        )
        if not isinstance(checkpoint, SupervisedModel):
            raise TypeError("SupervisedModel._fit must return a SupervisedModel")
        if checkpoint is self:
            raise ValueError("SupervisedModel._fit must return a new checkpoint")
        return checkpoint

    @abc.abstractmethod
    def _fit(
        self,
        train: Dataset,
        validation: Dataset | None,
        *,
        seed: int,
    ) -> SupervisedModel:
        pass


MetricItems = tuple[tuple[str, float], ...]


def _normalize_metrics(metrics: Mapping[str, float]) -> MetricItems:
    if not isinstance(metrics, Mapping):
        raise TypeError("Segment metrics must be a mapping")
    normalized: list[tuple[str, float]] = []
    for name, value in metrics.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Metric names must be non-empty strings")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"Metric '{name}' must be numeric")
        numeric = float(value)
        if not isfinite(numeric):
            raise ValueError(f"Metric '{name}' must be finite")
        normalized.append((name, numeric))
    return tuple(sorted(normalized))


@dataclass(frozen=True)
class ScheduleContext:
    """Immutable state observed when selecting the next prediction segment."""

    total_rows: int
    rows_seen: int
    rows_since_retrain: int
    retrain_count: int
    metrics: MetricItems = ()

    def __post_init__(self) -> None:
        for name in (
            "total_rows",
            "rows_seen",
            "rows_since_retrain",
            "retrain_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"ScheduleContext.{name} must be non-negative")
        if self.rows_seen > self.total_rows:
            raise ValueError("ScheduleContext.rows_seen cannot exceed total_rows")
        if self.rows_since_retrain > self.rows_seen:
            raise ValueError(
                "ScheduleContext.rows_since_retrain cannot exceed rows_seen"
            )
        normalized = _normalize_metrics(dict(self.metrics))
        if len(normalized) != len(self.metrics):
            raise ValueError("ScheduleContext metric names must be unique")
        object.__setattr__(self, "metrics", normalized)

    def metric(self, name: str) -> float | None:
        return dict(self.metrics).get(name)

    def __str__(self) -> str:
        return _canonical_json(self)


@dataclass(frozen=True)
class ScheduleDecision:
    """Whether to retrain now and the exclusive end of the next batch."""

    retrain: bool
    predict_until: int

    def __post_init__(self) -> None:
        if not isinstance(self.retrain, bool):
            raise TypeError("ScheduleDecision.retrain must be a boolean")
        if (
            isinstance(self.predict_until, bool)
            or not isinstance(self.predict_until, int)
            or self.predict_until < 1
        ):
            raise ValueError("ScheduleDecision.predict_until must be positive")

    def __str__(self) -> str:
        return _canonical_json(self)


class Scheduler(abc.ABC):
    """A pure policy over model-transition boundaries, never individual rows."""

    def decide(self, context: ScheduleContext) -> ScheduleDecision:
        if not isinstance(context, ScheduleContext):
            raise TypeError("Scheduler requires a ScheduleContext")
        if context.rows_seen >= context.total_rows:
            raise ValueError("Scheduler cannot advance a completed execution")
        decision = self._decide(context)
        if not isinstance(decision, ScheduleDecision):
            raise TypeError("Scheduler._decide must return a ScheduleDecision")
        if decision.predict_until <= context.rows_seen:
            raise ValueError("Scheduler decision must advance the row cursor")
        if decision.predict_until > context.total_rows:
            raise ValueError("Scheduler decision exceeds the available rows")
        if decision.retrain and context.rows_seen == 0:
            raise ValueError("Scheduler cannot retrain without historical rows")
        return decision

    @abc.abstractmethod
    def _decide(self, context: ScheduleContext) -> ScheduleDecision:
        pass

    def __str__(self) -> str:
        return _canonical_json(self)


@dataclass(frozen=True)
class FrozenScheduler(Scheduler):
    def _decide(self, context: ScheduleContext) -> ScheduleDecision:
        return ScheduleDecision(False, context.total_rows)


@dataclass(frozen=True)
class EveryNTicksScheduler(Scheduler):
    every: int

    def __post_init__(self) -> None:
        if isinstance(self.every, bool) or not isinstance(self.every, int) or self.every < 1:
            raise ValueError("EveryNTicksScheduler.every must be a positive integer")

    def _decide(self, context: ScheduleContext) -> ScheduleDecision:
        retrain = (
            context.rows_seen > 0
            and context.rows_since_retrain >= self.every
        )
        rows_to_boundary = (
            self.every if retrain else self.every - context.rows_since_retrain
        )
        return ScheduleDecision(
            retrain,
            min(context.rows_seen + rows_to_boundary, context.total_rows),
        )


@dataclass(frozen=True)
class MetricThresholdScheduler(Scheduler):
    """Retrain when the previous segment's named metric exceeds a threshold."""

    metric_name: str
    threshold: float
    check_every: int

    def __post_init__(self) -> None:
        if not isinstance(self.metric_name, str) or not self.metric_name:
            raise ValueError("metric_name must be a non-empty string")
        if isinstance(self.threshold, bool) or not isinstance(
            self.threshold, (int, float)
        ):
            raise TypeError("threshold must be numeric")
        if not isfinite(float(self.threshold)):
            raise ValueError("threshold must be finite")
        if (
            isinstance(self.check_every, bool)
            or not isinstance(self.check_every, int)
            or self.check_every < 1
        ):
            raise ValueError("check_every must be a positive integer")

    def _decide(self, context: ScheduleContext) -> ScheduleDecision:
        observed = context.metric(self.metric_name)
        return ScheduleDecision(
            retrain=(observed is not None and observed > self.threshold),
            predict_until=min(
                context.rows_seen + self.check_every,
                context.total_rows,
            ),
        )


@dataclass(frozen=True)
class AnyScheduler(Scheduler):
    schedulers: tuple[Scheduler, ...]

    def __post_init__(self) -> None:
        if not self.schedulers:
            raise ValueError("AnyScheduler requires at least one scheduler")
        if not all(isinstance(item, Scheduler) for item in self.schedulers):
            raise TypeError("AnyScheduler entries must be Scheduler instances")

    def _decide(self, context: ScheduleContext) -> ScheduleDecision:
        decisions = tuple(item.decide(context) for item in self.schedulers)
        return ScheduleDecision(
            retrain=any(item.retrain for item in decisions),
            predict_until=min(item.predict_until for item in decisions),
        )


def _reject_keys(declaration: Mapping[str, object], allowed: set[str], name: str) -> None:
    unexpected = set(declaration) - allowed
    if unexpected:
        raise ValueError(f"Unexpected keys for {name}: {sorted(unexpected)}")


def scheduler_from_declaration(value: object, *, default: Scheduler) -> Scheduler:
    """Normalize a scheduler declaration into a concrete ``Scheduler``.

    Accepts a ``Scheduler`` instance, ``None`` (returns ``default``), or a
    declarative mapping selecting one of ``frozen``, ``every``, ``metric``, or
    ``any``.
    """
    if value is None:
        return default
    if isinstance(value, Scheduler):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("scheduler must be a Scheduler or a declarative mapping")

    if "frozen" in value:
        _reject_keys(value, {"frozen"}, "scheduler.frozen")
        if value["frozen"] is not True:
            raise ValueError("scheduler.frozen must declare 'frozen': true")
        return FrozenScheduler()

    if "every" in value:
        _reject_keys(value, {"every"}, "scheduler.every")
        every = value["every"]
        if isinstance(every, bool) or not isinstance(every, int):
            raise TypeError("scheduler.every must be an integer")
        return EveryNTicksScheduler(every)

    if "metric" in value:
        _reject_keys(value, {"metric"}, "scheduler.metric")
        return metric_threshold_scheduler_from_declaration(value["metric"])

    if "any" in value:
        _reject_keys(value, {"any"}, "scheduler.any")
        entries = value["any"]
        if not isinstance(entries, (list, tuple)) or not entries:
            raise ValueError("scheduler.any must be a non-empty list of schedulers")
        return AnyScheduler(
            tuple(
                scheduler_from_declaration(entry, default=default)
                for entry in entries
            )
        )

    raise ValueError(
        "scheduler declaration must declare exactly one of 'frozen', 'every', "
        "'metric', or 'any'"
    )


def metric_threshold_scheduler_from_declaration(
    value: object,
) -> MetricThresholdScheduler:
    if not isinstance(value, Mapping):
        raise TypeError("scheduler.metric must be a mapping")
    allowed = {"name", "metric_name", "threshold", "check_every"}
    _reject_keys(value, allowed, "scheduler.metric")
    if "metric_name" in value and "name" in value:
        raise ValueError(
            "scheduler.metric must declare exactly one of 'metric_name' or 'name'"
        )
    metric_name = value.get("metric_name", value.get("name"))
    return MetricThresholdScheduler(
        metric_name=metric_name,
        threshold=cast(Any, value.get("threshold")),
        check_every=cast(Any, value.get("check_every")),
    )


def splitter_from_declaration(
    value: object,
    *,
    default: DatasetSplitter,
) -> DatasetSplitter:
    """Normalize a splitter declaration into a concrete ``DatasetSplitter``.

    Accepts a ``DatasetSplitter`` instance, ``None`` (returns ``default``), or
    a declarative mapping of ``ChronologicalSplitter`` fields.
    """
    if value is None:
        return default
    if isinstance(value, DatasetSplitter):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("splitter must be a DatasetSplitter or a declarative mapping")
    allowed = {
        "validation_size",
        "test_size",
        "gap",
        "batch_size",
        "shuffle_train",
        "drop_last",
        "purge_window",
    }
    _reject_keys(value, allowed, "splitter")
    return ChronologicalSplitter(
        validation_size=value.get("validation_size", 0),
        test_size=value.get("test_size", 0),
        gap=value.get("gap", 0),
        batch_size=value.get("batch_size"),
        shuffle_train=value.get("shuffle_train", False),
        drop_last=value.get("drop_last", False),
        purge_window=value.get("purge_window", 0),
    )


def validate_optional_width(name: str, value: object) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer or None")


def shape_width(shape: tuple[int, ...] | None) -> int:
    return _flat_size(shape or ())


class SupervisedModelTSFN(BatchTSFN, abc.ABC):
    """Batched walk-forward orchestration for supervised checkpoints.

    The graph supplies exactly one packed ``features`` column and one ``target``
    column. Python advances between scheduler boundaries; prediction and fitting
    operate on whole Series or Dataset batches rather than individual rows.
    """

    FEATURE_COLUMN: ClassVar[str] = "features"
    TARGET_COLUMN: ClassVar[str] = "target"
    PREDICTION_COLUMN: ClassVar[str] = "prediction"
    ALLOW_LOOKAHEAD_INPUTS: ClassVar[frozenset[str]] = frozenset({TARGET_COLUMN})

    def __init__(self, parameters: dict[str, Any]) -> None:
        super().__init__(parameters)
        self._validate_supervised_contract()

    @abc.abstractmethod
    def initial_model(self) -> SupervisedModel:
        pass

    @abc.abstractmethod
    def scheduler(self) -> Scheduler:
        pass

    @abc.abstractmethod
    def splitter(self) -> DatasetSplitter:
        pass

    @abc.abstractmethod
    def training_seed(self, retrain_count: int) -> int:
        pass

    def segment_metrics(
        self,
        target: pl.Series,
        prediction: pl.Series,
    ) -> Mapping[str, float]:
        del target, prediction
        return {}

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

        active_model = self.initial_model()
        if not isinstance(active_model, SupervisedModel):
            raise TypeError(
                "SupervisedModelTSFN.initial_model must return a SupervisedModel"
            )

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
            prediction = active_model.predict(
                segment.get_column(self.FEATURE_COLUMN)
            )
            expected_prediction = self.output_column_signature(
                self.PREDICTION_COLUMN
            ).physical_dtype
            if not _dtype_matches(prediction.dtype, expected_prediction):
                raise TypeError(
                    f"Model prediction type mismatch. Expected "
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

    def _validate_supervised_contract(self) -> None:
        input_signature, output_signature = self.signature
        input_names = tuple(entry[0] for entry in input_signature.columns)
        output_names = tuple(entry[0] for entry in output_signature.columns)
        if input_names != (self.FEATURE_COLUMN, self.TARGET_COLUMN):
            raise ValueError(
                "SupervisedModelTSFN input signature must declare exactly "
                "('features', 'target')"
            )
        if output_names != (self.PREDICTION_COLUMN,):
            raise ValueError(
                "SupervisedModelTSFN output signature must declare exactly "
                "('prediction',)"
            )
        if input_signature.time is None or output_signature.time is None:
            raise ValueError("SupervisedModelTSFN requires input and output time axes")
        if input_signature.time != output_signature.time:
            raise ValueError("SupervisedModelTSFN must preserve its input time axis")

    def _time_column(self) -> str:
        time_axis = self.signature[0].time
        if time_axis is None:
            raise ValueError("SupervisedModelTSFN requires an input time axis")
        return time_axis.column


@dataclass(frozen=True)
class TrainingResult:
    """The finished model and submodel checkpoints from one training run."""

    finished: Model
    checkpoints: tuple[Model, ...] = ()
    metrics: MetricItems = ()

    def __post_init__(self) -> None:
        if not isinstance(self.finished, Model):
            raise TypeError("TrainingResult.finished must be a Model")
        object.__setattr__(self, "checkpoints", tuple(self.checkpoints))
        if not all(isinstance(item, Model) for item in self.checkpoints):
            raise TypeError("TrainingResult.checkpoints must all be Models")
        object.__setattr__(self, "metrics", _normalize_metrics(dict(self.metrics)))


def _import_qualname(dotted: str) -> type:
    """Resolve a ``module.QualName`` string to a class object."""
    parts = dotted.split(".")
    for index in range(len(parts) - 1, 0, -1):
        try:
            module = importlib.import_module(".".join(parts[:index]))
        except ImportError:
            continue
        target: Any = module
        try:
            for part in parts[index:]:
                target = getattr(target, part)
        except AttributeError:
            continue
        if isinstance(target, type):
            return target
    raise ImportError(f"Cannot resolve type {dotted!r}")


def _decode_state_value(value: Any) -> Any:
    """Restore a JSON-compatible ``Model.to_dict()`` state value.

    Mirrors ``_serialize_value`` tagged mappings and tuple-encoded lists so a
    checkpoint round-trips through its JSON artifact form.
    """
    if isinstance(value, list):
        return tuple(_decode_state_value(item) for item in value)
    if not isinstance(value, dict):
        return value
    type_tag = value.get("__type__")
    if not isinstance(type_tag, str):
        return {key: _decode_state_value(item) for key, item in value.items()}
    if type_tag == "mapping":
        return {
            _decode_state_value(key): _decode_state_value(item)
            for key, item in value["items"]
        }
    if type_tag == "datetime":
        return datetime.fromisoformat(value["value"])
    if type_tag == "date":
        return date.fromisoformat(value["value"])
    if type_tag == "time":
        return time.fromisoformat(value["value"])
    if type_tag == "timedelta":
        return timedelta(
            days=value["days"],
            seconds=value["seconds"],
            microseconds=value["microseconds"],
        )
    if type_tag == "path":
        return Path(value["value"])
    if type_tag == "bytes":
        return bytes.fromhex(value["value"])
    nested_fields = value.get("fields")
    if isinstance(nested_fields, dict):
        cls = _import_qualname(type_tag)
        return cls(
            **{
                name: _decode_state_value(item)
                for name, item in nested_fields.items()
            }
        )
    raise ValueError(f"Unsupported serialized value tag: {type_tag!r}")


def model_from_dict(data: Mapping[str, Any]) -> Model:
    """Rebuild an immutable checkpoint from ``Model.to_dict()`` JSON state."""
    if not isinstance(data, Mapping):
        raise TypeError("model_from_dict requires a mapping")
    for key in ("module", "qualname", "state"):
        if key not in data:
            raise ValueError(f"Model payload is missing {key!r}")
    cls = _import_qualname(f"{data['module']}.{data['qualname']}")
    if not (isinstance(cls, type) and issubclass(cls, Model)):
        raise TypeError(
            f"Model payload does not resolve to a Model subclass: "
            f"{data['module']}.{data['qualname']}"
        )
    expected_version = data.get("version")
    if expected_version != cls.VERSION:
        raise ValueError(
            f"Model payload version {expected_version!r} does not match "
            f"{cls.__name__}.VERSION {cls.VERSION!r}"
        )
    state = data["state"]
    if not isinstance(state, Mapping):
        raise TypeError("Model payload state must be a mapping")
    checkpoint = cls(
        **{name: _decode_state_value(item) for name, item in state.items()}
    )
    if not isinstance(checkpoint, Model):
        raise TypeError("Model payload did not produce a Model")
    return checkpoint


def _validate_model_group(group: object) -> str:
    """Validate a model group name used as a store directory segment."""
    if not isinstance(group, str) or not group.strip():
        raise ValueError("model_group must be a non-empty string")
    if (
        group != group.strip()
        or "/" in group
        or "\\" in group
        or group in (".", "..")
        or ".." in group.split(os.sep)
    ):
        raise ValueError(
            f"Invalid model_group {group!r}: must be a single path segment "
            "without separators"
        )
    return group


def _manifest_run_trained_at(run: Mapping[str, Any]) -> datetime:
    trained_at = datetime.fromisoformat(run["trained_at"])
    if trained_at.tzinfo is None:
        raise ValueError("Manifest trained_at timestamps must carry a timezone")
    return trained_at


def _validate_manifest(manifest: Any) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        raise ValueError("Model manifest must be a mapping")
    runs = manifest.get("runs")
    if not isinstance(runs, list):
        raise ValueError("Model manifest requires a 'runs' list")
    ordered_at: list[datetime] = []
    for run in runs:
        if not isinstance(run, dict):
            raise ValueError("Model manifest runs must be mappings")
        for key in ("run_id", "trained_at", "finished_model_id", "checkpoints"):
            if key not in run:
                raise ValueError(f"Model manifest run is missing {key!r}")
        ordered_at.append(_manifest_run_trained_at(run))
    if ordered_at != sorted(ordered_at):
        raise ValueError("Model manifest runs must be sorted by trained_at")
    return manifest


@dataclass(frozen=True)
class ModelStore:
    """Executor-owned artifact store for finished models and checkpoints.

    Layout per group: ``<root>/<group>/manifest.json`` plus one
    ``<model_id>.json`` artifact per finished model or checkpoint, where the id
    is the SHA-256 of the canonical checkpoint payload. Identical retrains are
    idempotent: saving an already-recorded finished model is a no-op.

    ``root`` is either a local path or an ``s3://bucket/prefix`` URI, mirroring
    ``LocalExecutor`` cache handling. S3 state is addressed through
    ``pyarrow.fs.S3FileSystem`` with the execution-scoped ``S3Credentials``.
    """

    root: Path | str

    def __post_init__(self) -> None:
        root = self.root
        if isinstance(root, Path):
            root.mkdir(parents=True, exist_ok=True)
            return
        if isinstance(root, str):
            if root.startswith("s3://"):
                normalized = root.rstrip("/")
                without_scheme = normalized.removeprefix("s3://")
                bucket, _, key = without_scheme.partition("/")
                if not bucket:
                    raise ValueError(
                        "s3:// model location must include a bucket name"
                    )
                stripped_key = key.rstrip("/")
                object.__setattr__(
                    self,
                    "root",
                    f"s3://{bucket}/{stripped_key}"
                    if stripped_key
                    else f"s3://{bucket}",
                )
                return
            local = Path(root)
            object.__setattr__(self, "root", local)
            local.mkdir(parents=True, exist_ok=True)
            return
        if isinstance(root, os.PathLike):
            local = Path(os.fspath(root))
            object.__setattr__(self, "root", local)
            local.mkdir(parents=True, exist_ok=True)
            return
        raise TypeError("ModelStore.root must be a pathlib Path or s3:// URI")

    @property
    def _is_s3(self) -> bool:
        return isinstance(self.root, str) and self.root.startswith("s3://")

    @property
    def _s3_prefix(self) -> str:
        assert isinstance(self.root, str)
        return self.root.rstrip("/")

    def _s3_key(self, group: str, filename: str) -> str:
        return f"{self._s3_prefix}/{_validate_model_group(group)}/{filename}"

    def _open_s3_filesystem(self):  # type: ignore[no-untyped-def]
        from pyarrow.fs import S3FileSystem

        credentials = current_s3_credentials()
        if credentials is None:
            return S3FileSystem()
        return S3FileSystem(
            access_key=credentials.access_key,
            secret_key=credentials.secret_key,
            session_token=credentials.session_token,
            region=credentials.region,
        )

    @classmethod
    def from_context(
        cls, explicit: str | os.PathLike[str] | None = None
    ) -> ModelStore:
        """Resolve the active store: explicit override, scope, then env."""
        resolved = resolve_model_dir(explicit)
        if resolved is None:
            raise RuntimeError(
                "No model store is configured. Pass model_dir to the executor "
                "or set the IOSIS_MODEL_DIR environment variable."
            )
        return cls(resolved)

    def group_dir(self, group: str) -> Path | str:
        if self._is_s3:
            return f"{self._s3_prefix}/{_validate_model_group(group)}"
        assert isinstance(self.root, Path)
        return self.root / _validate_model_group(group)

    def manifest_path(self, group: str) -> Path | str:
        location = self.group_dir(group)
        if isinstance(location, str):
            return f"{location}/manifest.json"
        return location / "manifest.json"

    def read_manifest(self, group: str) -> dict[str, Any]:
        """Read and validate a group's manifest; missing means no runs yet."""
        if self._is_s3:
            return self._read_manifest_s3(group)
        path = self.manifest_path(group)
        assert isinstance(path, Path)
        if not path.exists():
            return {"group": group, "runs": []}
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"Model manifest for group {group!r} is unreadable: {exc}"
            ) from exc
        return _validate_manifest(manifest)

    def _read_manifest_s3(self, group: str) -> dict[str, Any]:
        from pyarrow.fs import FileType

        key = self._s3_key(group, "manifest.json")
        bare = key.removeprefix("s3://")
        filesystem = self._open_s3_filesystem()
        try:
            info = filesystem.get_file_info(bare)
            if info.type != FileType.File:
                return {"group": group, "runs": []}
            with filesystem.open_input_stream(bare) as stream:
                manifest = json.loads(stream.read().decode("utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"Model manifest for group {group!r} is unreadable: {exc}"
            ) from exc
        return _validate_manifest(manifest)

    def _write_manifest(self, group: str, manifest: dict[str, Any]) -> None:
        validated = _validate_manifest(manifest)
        if self._is_s3:
            key = self._s3_key(group, "manifest.json")
            bare = key.removeprefix("s3://")
            filesystem = self._open_s3_filesystem()
            filesystem.create_dir(bare.rpartition("/")[0], recursive=True)
            payload = json.dumps(validated, sort_keys=True, indent=2).encode(
                "utf-8"
            )
            with filesystem.open_output_stream(bare) as stream:
                stream.write(payload)
            return
        directory = self.group_dir(group)
        assert isinstance(directory, Path)
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / "manifest.json"
        staging = directory / f".manifest.{os.getpid()}.tmp"
        staging.write_text(
            json.dumps(validated, sort_keys=True, indent=2), encoding="utf-8"
        )
        os.replace(staging, target)

    @staticmethod
    def model_id(checkpoint: Model) -> str:
        """Content-address a checkpoint from its canonical payload."""
        if not isinstance(checkpoint, Model):
            raise TypeError("ModelStore.model_id requires a Model")
        payload = _canonical_json(checkpoint.to_dict())
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _write_checkpoint(self, group: str, checkpoint: Model) -> str:
        model_id = self.model_id(checkpoint)
        if self._is_s3:
            from pyarrow.fs import FileType

            key = self._s3_key(group, f"{model_id}.json")
            bare = key.removeprefix("s3://")
            filesystem = self._open_s3_filesystem()
            info = filesystem.get_file_info(bare)
            if info.type != FileType.File:
                filesystem.create_dir(bare.rpartition("/")[0], recursive=True)
                payload = json.dumps(
                    checkpoint.to_dict(), sort_keys=True, indent=2
                ).encode("utf-8")
                with filesystem.open_output_stream(bare) as stream:
                    stream.write(payload)
            return model_id
        location = self.group_dir(group)
        assert isinstance(location, Path)
        path = location / f"{model_id}.json"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(checkpoint.to_dict(), sort_keys=True, indent=2),
                encoding="utf-8",
            )
        return model_id

    def save_run(
        self,
        group: str,
        *,
        finished: Model,
        checkpoints: tuple[Model, ...] = (),
        metrics: Mapping[str, float] | None = None,
        trained_at: datetime | None = None,
        run_id: str | None = None,
    ) -> str:
        """Record one training run; returns the finished model id."""
        _validate_model_group(group)
        if not isinstance(finished, Model):
            raise TypeError("save_run finished must be a Model")
        checkpoints = tuple(checkpoints)
        if not all(isinstance(item, Model) for item in checkpoints):
            raise TypeError("save_run checkpoints must all be Models")
        if trained_at is None:
            trained_at = datetime.now(timezone.utc)
        if not isinstance(trained_at, datetime) or trained_at.tzinfo is None:
            raise ValueError("save_run trained_at must be a timezone-aware datetime")
        trained_at = trained_at.astimezone(timezone.utc)
        normalized_metrics = dict(_normalize_metrics(dict(metrics or {})))
        if run_id is not None and (not isinstance(run_id, str) or not run_id):
            raise ValueError("save_run run_id must be a non-empty string or None")

        manifest = self.read_manifest(group)
        finished_id = self._write_checkpoint(group, finished)
        for checkpoint in checkpoints:
            self._write_checkpoint(group, checkpoint)

        for run in manifest["runs"]:
            if run["finished_model_id"] == finished_id:
                return finished_id
            if run_id is not None and run["run_id"] == run_id:
                raise ValueError(
                    f"run_id {run_id!r} already records a different finished model"
                )

        manifest["runs"].append(
            {
                "run_id": run_id or finished_id,
                "trained_at": trained_at.isoformat(),
                "finished_model_id": finished_id,
                "metrics": normalized_metrics,
                "checkpoints": [
                    {"checkpoint_id": self.model_id(item)} for item in checkpoints
                ],
            }
        )
        manifest["runs"].sort(key=lambda run: run["trained_at"])
        self._write_manifest(group, manifest)
        return finished_id

    def select_run(self, group: str, at: datetime) -> dict[str, Any]:
        """Return the latest finished run with ``trained_at <= at`` (asof)."""
        if not isinstance(at, datetime):
            raise TypeError("select_run requires a datetime")
        manifest = self.read_manifest(group)
        candidates = [
            run
            for run in manifest["runs"]
            if _manifest_run_trained_at(run) <= at
        ]
        if not candidates:
            earliest = (
                manifest["runs"][0]["trained_at"] if manifest["runs"] else None
            )
            raise LookupError(
                f"No finished model for group {group!r} is available at "
                f"{at.isoformat()}"
                + (f"; earliest trained_at is {earliest}" if earliest else "")
                + ". Train the group first."
            )
        return candidates[-1]

    def latest_run(self, group: str) -> dict[str, Any] | None:
        """Return the newest finished run, or ``None`` when the group is empty."""
        manifest = self.read_manifest(group)
        if not manifest["runs"]:
            return None
        return manifest["runs"][-1]

    def load_model(self, group: str, model_id: str) -> Model:
        """Load one checkpoint artifact by content id."""
        _validate_model_group(group)
        if not isinstance(model_id, str) or not model_id:
            raise ValueError("model_id must be a non-empty string")
        if self._is_s3:
            from pyarrow.fs import FileType

            key = self._s3_key(group, f"{model_id}.json")
            bare = key.removeprefix("s3://")
            filesystem = self._open_s3_filesystem()
            info = filesystem.get_file_info(bare)
            if info.type != FileType.File:
                raise FileNotFoundError(
                    f"Model {model_id!r} for group {group!r} is not in the store. "
                    "Train the group first."
                )
            try:
                with filesystem.open_input_stream(bare) as stream:
                    payload = json.loads(stream.read().decode("utf-8"))
            except (OSError, ValueError) as exc:
                raise ValueError(
                    f"Model artifact {model_id!r} for group {group!r} "
                    f"is unreadable: {exc}"
                ) from exc
            return model_from_dict(payload)
        location = self.group_dir(group)
        assert isinstance(location, Path)
        path = location / f"{model_id}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"Model {model_id!r} for group {group!r} is not in the store. "
                "Train the group first."
            )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"Model artifact {model_id!r} for group {group!r} is unreadable: "
                f"{exc}"
            ) from exc
        return model_from_dict(payload)

    def load_run_model(self, group: str, run: Mapping[str, Any]) -> Model:
        """Load the finished model recorded by one manifest run."""
        if not isinstance(run, Mapping) or "finished_model_id" not in run:
            raise TypeError("load_run_model requires a manifest run mapping")
        model_id = run["finished_model_id"]
        if not isinstance(model_id, str):
            raise TypeError("Manifest run finished_model_id must be a string")
        return self.load_model(group, model_id)


class TrainingTSFN(BatchTSFN, abc.ABC):
    """Batched trainer that persists a versioned checkpoint and emits nothing.

    Trainers are terminal nodes: they declare no value outputs, so no other
    node can bind to them, and they form a branch parallel to the inference
    path over the same data. The graph-visible training result is the
    ``ModelStore`` manifest entry; the frame output is an empty (0-row) frame
    matching the declared output schema. Retraining appends a new finished
    model run rather than replacing an endpoint.
    """

    IS_TRAINER: ClassVar[bool] = True
    IDENTITY_EXCLUDED_PARAMS: ClassVar[frozenset[str]] = frozenset(
        {"model_dir", "run_id"}
    )

    @abc.abstractmethod
    def model_group(self) -> str:
        """Return the store group this trainer publishes to."""
        pass

    @abc.abstractmethod
    def train_frame(self, frame: pl.DataFrame) -> TrainingResult:
        """Fit one finished model over the complete input frame."""
        pass

    def training_run_id(self) -> str | None:
        """Return an explicit run id override, if configured."""
        run_id = getattr(self.parameters, "run_id", None)
        if run_id is not None and (not isinstance(run_id, str) or not run_id):
            raise ValueError("run_id must be a non-empty string or None")
        return run_id

    def training_trained_at(self, frame: pl.DataFrame) -> datetime:
        """Default ``trained_at`` is data time: the end of the training window."""
        time_axis = self.signature[0].time
        if time_axis is None:
            raise ValueError("TrainingTSFN requires an input time axis")
        latest = frame.get_column(time_axis.column).max()
        if latest is None:
            raise ValueError("TrainingTSFN cannot train on an empty frame")
        if not isinstance(latest, datetime):
            raise TypeError("TrainingTSFN time column must hold datetimes")
        if latest.tzinfo is None:
            return latest.replace(tzinfo=timezone.utc)
        return latest

    def model_store(self) -> ModelStore:
        """Resolve the executor-owned store for this training run."""
        return ModelStore.from_context(getattr(self.parameters, "model_dir", None))

    def empty_output_frame(self) -> pl.DataFrame:
        """Return the declared 0-row output frame for a completed training run."""
        return pl.DataFrame(schema=_frame_physical_schema(self.signature[1]))

    def batch(self, frame: pl.DataFrame) -> pl.DataFrame:
        if frame.is_empty():
            return self.empty_output_frame()
        time_axis = self.signature[0].time
        if time_axis is None:
            raise ValueError("TrainingTSFN requires an input time axis")
        ordered = (
            frame
            if frame.get_column(time_axis.column).is_sorted()
            else frame.sort(time_axis.column)
        )
        result = self.train_frame(ordered)
        if not isinstance(result, TrainingResult):
            raise TypeError("TrainingTSFN.train_frame must return a TrainingResult")
        group = _validate_model_group(self.model_group())
        store = self.model_store()
        store.save_run(
            group,
            finished=result.finished,
            checkpoints=result.checkpoints,
            metrics=dict(result.metrics),
            trained_at=self.training_trained_at(ordered),
            run_id=self.training_run_id(),
        )
        return self.empty_output_frame()


class InferenceTSFN(BatchTSFN, abc.ABC):
    """Batched frozen inference resolving a finished model from the store.

    The node pins ``model_id`` for an exact checkpoint, or resolves the latest
    finished run whose ``trained_at`` is at or before the batch end (backward
    asof over training runs). A graph without a trainer for the group is valid
    and simply runs frozen against preexisting store state.
    """

    IS_INFERENCE: ClassVar[bool] = True
    IDENTITY_EXCLUDED_PARAMS: ClassVar[frozenset[str]] = frozenset(
        {"model_dir", "frozen"}
    )

    @abc.abstractmethod
    def model_group(self) -> str:
        """Return the store group this node predicts with."""
        pass

    def pinned_model_id(self) -> str | None:
        """Return an exact checkpoint id override, if configured."""
        model_id = getattr(self.parameters, "model_id", None)
        if model_id is not None and (
            not isinstance(model_id, str) or not model_id
        ):
            raise ValueError("model_id must be a non-empty string or None")
        return model_id

    def is_frozen(self) -> bool:
        """Whether this node never requires a trainer in its graph."""
        return bool(getattr(self.parameters, "frozen", False))

    @abc.abstractmethod
    def predict_with_model(
        self, segment: pl.DataFrame, model: Model
    ) -> pl.DataFrame:
        """Score one frame with a resolved finished model checkpoint."""
        pass

    def model_store(self) -> ModelStore:
        """Resolve the executor-owned store holding this node's finished models."""
        return ModelStore.from_context(getattr(self.parameters, "model_dir", None))

    def _resolve_run(
        self, store: ModelStore, group: str, ordered: pl.DataFrame
    ) -> tuple[dict[str, Any], Model]:
        pinned = self.pinned_model_id()
        if pinned is not None:
            manifest = store.read_manifest(group)
            matches = [
                run
                for run in manifest["runs"]
                if run["finished_model_id"] == pinned
            ]
            if not matches:
                raise LookupError(
                    f"Pinned model {pinned!r} is not a finished model for group "
                    f"{group!r}. Train the group first or check the id."
                )
            return matches[-1], store.load_model(group, pinned)
        time_axis = self.signature[0].time
        if time_axis is None:
            raise ValueError("InferenceTSFN requires an input time axis")
        latest = ordered.get_column(time_axis.column).max()
        if not isinstance(latest, datetime):
            raise TypeError("InferenceTSFN time column must hold datetimes")
        if latest.tzinfo is None:
            latest = latest.replace(tzinfo=timezone.utc)
        run = store.select_run(group, latest)
        return run, store.load_run_model(group, run)

    def batch(self, frame: pl.DataFrame) -> pl.DataFrame:
        output_schema = _frame_physical_schema(self.signature[1])
        if frame.is_empty():
            return pl.DataFrame(schema=output_schema)
        time_axis = self.signature[0].time
        if time_axis is None:
            raise ValueError("InferenceTSFN requires an input time axis")
        ordered = (
            frame
            if frame.get_column(time_axis.column).is_sorted()
            else frame.sort(time_axis.column)
        )
        group = _validate_model_group(self.model_group())
        store = self.model_store()
        _, checkpoint = self._resolve_run(store, group, ordered)
        output = self.predict_with_model(ordered, checkpoint)
        if not isinstance(output, pl.DataFrame):
            raise TypeError(
                f"{self.__class__.__name__}.predict_with_model must return a "
                f"Polars DataFrame, got {type(output).__name__}"
            )
        if output.height != ordered.height:
            raise ValueError("Inference prediction must preserve input row count")
        return output


__all__ = [
    "AnyScheduler",
    "ChronologicalSplitter",
    "Dataset",
    "DatasetSplit",
    "DatasetSplitter",
    "EveryNTicksScheduler",
    "FrameDataset",
    "FrozenScheduler",
    "InferenceTSFN",
    "MetricThresholdScheduler",
    "Model",
    "ModelStore",
    "ScheduleContext",
    "ScheduleDecision",
    "Scheduler",
    "SupervisedModel",
    "SupervisedModelTSFN",
    "TrainingResult",
    "TrainingTSFN",
    "metric_threshold_scheduler_from_declaration",
    "model_from_dict",
    "scheduler_from_declaration",
    "shape_width",
    "splitter_from_declaration",
    "validate_optional_width",
]
