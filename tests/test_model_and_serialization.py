from __future__ import annotations

import abc
import inspect
import json
from dataclasses import FrozenInstanceError, dataclass
from datetime import date, datetime, time, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

from iosislib.core.node import Node
from iosislib.core.model import (
    Dataset,
    DatasetSplit,
    FrameDataset,
    Model,
    SupervisedModel,
)
from iosislib.core.tsfn import (
    FrameSignature,
    TSFN,
    TSFNConfig,
)


class TrainingMode(str, Enum):
    FROZEN = "frozen"
    RETRAIN = "retrain"


@dataclass(frozen=True)
class NestedSettings:
    path: Path
    dtype: pl.DataType
    mode: TrainingMode
    created_at: datetime
    session_date: date
    cutoff: time
    delay: timedelta
    labels: tuple[str, ...]
    groups: frozenset[str]
    numbered: dict[int, str]
    scalar: Any
    payload: bytes


@dataclass(frozen=True)
class NestedConfig(TSFNConfig):
    settings: NestedSettings
    options: dict[str, int]


@dataclass(frozen=True)
class OpaqueConfig(TSFNConfig):
    value: object


@dataclass(frozen=True)
class FloatConfig(TSFNConfig):
    value: float


class NestedSource(TSFN):
    VERSION = "1.0.0"
    CONFIG_CLS = NestedConfig

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        return (
            FrameSignature.empty(),
            FrameSignature(columns=(("value", pl.Int64),)),
        )

    def apply(self) -> pl.LazyFrame:
        return pl.DataFrame(
            {
                "timestamp": [datetime(2026, 1, 1)],
                "value": [len(self.parameters.options)],
            }
        ).lazy()


def nested_settings(*, delay_seconds: int = 30) -> NestedSettings:
    return NestedSettings(
        path=Path("models") / "checkpoint.pt",
        dtype=pl.Datetime(time_unit="ns", time_zone="UTC"),
        mode=TrainingMode.RETRAIN,
        created_at=datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
        session_date=date(2026, 1, 2),
        cutoff=time(16, 30),
        delay=timedelta(seconds=delay_seconds, microseconds=7),
        labels=("feature", "target"),
        groups=frozenset(("beta", "alpha")),
        numbered={2: "two", 1: "one"},
        scalar=np.int64(42),
        payload=b"model",
    )


def test_config_serialization_is_recursive_canonical_and_type_aware() -> None:
    first = NestedConfig(
        settings=nested_settings(),
        options={"second": 2, "first": 1},
    )
    second = NestedConfig(
        settings=nested_settings(),
        options={"first": 1, "second": 2},
    )

    assert str(first) == str(second)
    assert first.to_dict() == json.loads(str(first))

    payload = first.to_dict()
    settings = payload["settings"]
    assert settings["__type__"].endswith(".NestedSettings")
    fields = settings["fields"]
    assert fields["path"] == {
        "__type__": "path",
        "value": str(Path("models") / "checkpoint.pt"),
    }
    assert fields["dtype"] == "Datetime(time_unit='ns', time_zone='UTC')"
    assert fields["mode"] == "retrain"
    assert fields["created_at"] == {
        "__type__": "datetime",
        "value": "2026-01-02T03:04:05+00:00",
    }
    assert fields["session_date"] == {
        "__type__": "date",
        "value": "2026-01-02",
    }
    assert fields["cutoff"] == {"__type__": "time", "value": "16:30:00"}
    assert fields["delay"] == {
        "__type__": "timedelta",
        "days": 0,
        "seconds": 30,
        "microseconds": 7,
    }
    assert fields["labels"] == ["feature", "target"]
    assert fields["groups"] == {
        "__type__": "set",
        "items": ["alpha", "beta"],
    }
    assert fields["numbered"] == {
        "__type__": "mapping",
        "items": [[1, "one"], [2, "two"]],
    }
    assert fields["scalar"] == 42
    assert fields["payload"] == {"__type__": "bytes", "value": "6d6f64656c"}


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_config_serialization_rejects_non_finite_floats(value: float) -> None:
    with pytest.raises(ValueError, match="Non-finite floats"):
        str(FloatConfig(value=value))


def test_config_serialization_rejects_opaque_values() -> None:
    with pytest.raises(TypeError, match="not serializable"):
        str(OpaqueConfig(value=object()))


def test_nested_config_values_participate_in_node_identity() -> None:
    first = Node(
        NestedSource,
        parameters={
            "settings": nested_settings(),
            "options": {"second": 2, "first": 1},
        },
    )
    reordered = Node(
        NestedSource,
        parameters={
            "settings": nested_settings(),
            "options": {"first": 1, "second": 2},
        },
    )
    changed = Node(
        NestedSource,
        parameters={
            "settings": nested_settings(delay_seconds=31),
            "options": {"first": 1, "second": 2},
        },
    )

    assert first.ID == reordered.ID
    assert first.ID != changed.ID


def supervised_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "features": [[1.0, 10.0], [3.0, 30.0]],
            "target": [1.0, 3.0],
        },
        schema={
            "features": pl.Array(pl.Float64, 2),
            "target": pl.Float64,
        },
    )


@dataclass(frozen=True, kw_only=True)
class MeanModel(SupervisedModel):
    VERSION = "1.0.0"
    mean: float = 0.0

    def _fit(
        self,
        train: Dataset,
        validation: Dataset | None,
        *,
        seed: int,
    ) -> SupervisedModel:
        del validation, seed
        targets = pl.concat(
            [batch.get_column("target") for batch in train.batches()]
        )
        return MeanModel(mean=float(targets.mean()))

    def _predict(self, features: pl.Series) -> pl.Series:
        return pl.repeat(self.mean, len(features), dtype=pl.Float64, eager=True)


@dataclass(frozen=True, kw_only=True)
class BrokenModel(SupervisedModel):
    VERSION = "1.0.0"
    behavior: str

    def _fit(
        self,
        train: Dataset,
        validation: Dataset | None,
        *,
        seed: int,
    ) -> SupervisedModel:
        del train, validation, seed
        if self.behavior == "self":
            return self
        if self.behavior == "not-model":
            return object()  # type: ignore[return-value]
        return MeanModel()

    def _predict(self, features: pl.Series) -> pl.Series:
        if self.behavior == "bad-prediction":
            return object()  # type: ignore[return-value]
        if self.behavior == "wrong-length":
            return pl.Series([0.0])
        if self.behavior == "null-prediction":
            return pl.Series([None] * len(features), dtype=pl.Float64)
        return pl.repeat(0.0, len(features), eager=True)


def model_split() -> DatasetSplit:
    return DatasetSplit(FrameDataset(supervised_frame()))


def test_model_is_an_immutable_serializable_inference_checkpoint() -> None:
    model = MeanModel(mean=2.0)

    assert model.version == "1.0.0"
    assert json.loads(str(model)) == model.to_dict()
    assert model.to_dict()["state"] == {"mean": 2.0}
    assert "MeanModel" in repr(model)
    assert not hasattr(Model, "fit")

    with pytest.raises(FrozenInstanceError):
        model.mean = 10.0  # type: ignore[misc]


def test_supervised_fitting_returns_a_new_checkpoint() -> None:
    initial = MeanModel()
    trained = initial.fit(model_split(), seed=7)

    assert trained is not initial
    assert isinstance(trained, MeanModel)
    assert trained.mean == 2.0
    assert initial.mean == 0.0
    assert trained.predict(supervised_frame()["features"]).to_list() == [2.0, 2.0]


def test_supervised_fit_exposes_only_train_and_validation_to_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train = FrameDataset(supervised_frame())
    validation = FrameDataset(supervised_frame())
    test = FrameDataset(supervised_frame())
    datasets = DatasetSplit(train, validation, test)
    received: list[tuple[Dataset, Dataset | None]] = []
    original = MeanModel._fit

    def record(
        self: MeanModel,
        fit_train: Dataset,
        fit_validation: Dataset | None,
        *,
        seed: int,
    ) -> SupervisedModel:
        received.append((fit_train, fit_validation))
        return original(self, fit_train, fit_validation, seed=seed)

    monkeypatch.setattr(MeanModel, "_fit", record)
    MeanModel().fit(datasets, seed=3)

    assert received[0][0] is train
    assert received[0][1] is validation
    assert "test" not in inspect.signature(MeanModel._fit).parameters


def test_model_runtime_contract_rejects_invalid_inputs_and_outputs() -> None:
    features = supervised_frame()["features"]

    with pytest.raises(TypeError, match="requires a DatasetSplit"):
        MeanModel().fit(object(), seed=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="seed must be an integer"):
        MeanModel().fit(model_split(), seed=True)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="must return a SupervisedModel"):
        BrokenModel(behavior="not-model").fit(model_split(), seed=1)
    with pytest.raises(ValueError, match="new checkpoint"):
        BrokenModel(behavior="self").fit(model_split(), seed=1)
    with pytest.raises(TypeError, match="must return a Polars Series"):
        BrokenModel(behavior="bad-prediction").predict(features)
    with pytest.raises(ValueError, match="preserve feature row count"):
        BrokenModel(behavior="wrong-length").predict(features)
    with pytest.raises(ValueError, match="null prediction"):
        BrokenModel(behavior="null-prediction").predict(features)
    with pytest.raises(TypeError, match="requires a Polars Series"):
        MeanModel().predict(object())  # type: ignore[arg-type]

    null_features = pl.Series(
        "features",
        [[1.0, None]],
        dtype=pl.Array(pl.Float64, 2),
    )
    with pytest.raises(ValueError, match="null feature"):
        MeanModel().predict(null_features)


def test_concrete_model_subclasses_must_define_versions() -> None:
    class AbstractModel(Model):
        @abc.abstractmethod
        def extra_contract(self) -> None:
            pass

        def _predict(self, features: pl.Series) -> pl.Series:
            return features

    assert inspect.isabstract(AbstractModel)

    with pytest.raises(TypeError, match="must define VERSION"):

        class MissingVersionModel(Model):
            def _predict(self, features: pl.Series) -> pl.Series:
                return features

    with pytest.raises(TypeError, match="VERSION must be a string"):

        class NonStringVersionModel(Model):
            VERSION = 1  # type: ignore[assignment]

            def _predict(self, features: pl.Series) -> pl.Series:
                return features

    with pytest.raises(ValueError, match="VERSION must be non-empty"):

        class EmptyVersionModel(Model):
            VERSION = " "

            def _predict(self, features: pl.Series) -> pl.Series:
                return features
