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

from src.classes import FrameSignature, Model, Node, TSFN, TSFNConfig


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


@dataclass(frozen=True, kw_only=True)
class MeanModel(Model):
    VERSION = "1.0.0"
    mean: float = 0.0

    def _train(
        self,
        frame: pl.DataFrame,
        *,
        trained_until: datetime,
        available_at: datetime,
        seed: int,
    ) -> Model:
        mean = float(frame.get_column("value").mean())
        return MeanModel(
            ID=f"{self.ID}:{seed}:{mean}",
            trained_until=trained_until,
            available_at=available_at,
            mean=mean,
        )

    def _predict(self, frame: pl.DataFrame) -> pl.DataFrame:
        return frame.select(
            "timestamp",
            (pl.col("value") - self.mean).alias("prediction"),
        )


@dataclass(frozen=True, kw_only=True)
class BrokenModel(Model):
    VERSION = "1.0.0"
    behavior: str

    def _train(
        self,
        frame: pl.DataFrame,
        *,
        trained_until: datetime,
        available_at: datetime,
        seed: int,
    ) -> Model:
        if self.behavior == "self":
            return self
        if self.behavior == "not-model":
            return object()  # type: ignore[return-value]
        return BrokenModel(
            ID=f"{self.ID}:next",
            trained_until=trained_until + timedelta(seconds=1),
            available_at=available_at,
            behavior=self.behavior,
        )

    def _predict(self, frame: pl.DataFrame) -> pl.DataFrame:
        if self.behavior == "bad-prediction":
            return object()  # type: ignore[return-value]
        return frame


def model_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "timestamp": [datetime(2026, 1, 1), datetime(2026, 1, 2)],
            "value": [1.0, 3.0],
        }
    )


def test_model_is_an_immutable_checkpoint_with_serializable_metadata() -> None:
    model = MeanModel(ID="initial")

    assert model.version == "1.0.0"
    assert model.trained_until is None
    assert model.available_at is None
    assert json.loads(str(model)) == model.to_dict()
    assert "MeanModel" in repr(model)
    assert "initial" in repr(model)

    with pytest.raises(FrozenInstanceError):
        model.mean = 10.0  # type: ignore[misc]


def test_model_training_returns_a_new_causally_valid_checkpoint() -> None:
    initial = MeanModel(ID="initial")
    trained_until = datetime(2026, 1, 2)
    available_at = datetime(2026, 1, 2, 0, 1)

    trained = initial.train(
        model_frame(),
        trained_until=trained_until,
        available_at=available_at,
        seed=7,
    )

    assert trained is not initial
    assert isinstance(trained, MeanModel)
    assert trained.mean == 2.0
    assert trained.trained_until == trained_until
    assert trained.available_at == available_at
    assert initial.mean == 0.0
    assert initial.trained_until is None

    predicted = trained.predict(model_frame())
    assert predicted["prediction"].to_list() == [-1.0, 1.0]


def test_model_metadata_rejects_invalid_identity_and_causal_times() -> None:
    with pytest.raises(TypeError, match="ID must be a string"):
        MeanModel(ID=1)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="ID must be non-empty"):
        MeanModel(ID=" ")

    with pytest.raises(ValueError, match="must either both be set"):
        MeanModel(ID="model", trained_until=datetime(2026, 1, 1))

    with pytest.raises(ValueError, match="cannot precede"):
        MeanModel(
            ID="model",
            trained_until=datetime(2026, 1, 2),
            available_at=datetime(2026, 1, 1),
        )

    with pytest.raises(TypeError, match="compatible timezones"):
        MeanModel(
            ID="model",
            trained_until=datetime(2026, 1, 1, tzinfo=timezone.utc),
            available_at=datetime(2026, 1, 1),
        )


def test_model_runtime_contract_rejects_invalid_inputs_and_outputs() -> None:
    initial = MeanModel(ID="initial")
    trained_until = datetime(2026, 1, 2)
    available_at = datetime(2026, 1, 3)

    with pytest.raises(TypeError, match="requires a Polars DataFrame"):
        initial.train(  # type: ignore[arg-type]
            object(),
            trained_until=trained_until,
            available_at=available_at,
            seed=1,
        )

    with pytest.raises(TypeError, match="seed must be an integer"):
        initial.train(
            model_frame(),
            trained_until=trained_until,
            available_at=available_at,
            seed=True,  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="cannot precede"):
        initial.train(
            model_frame(),
            trained_until=available_at,
            available_at=trained_until,
            seed=1,
        )

    with pytest.raises(TypeError, match="Model._train must return a Model"):
        BrokenModel(ID="bad", behavior="not-model").train(
            model_frame(),
            trained_until=trained_until,
            available_at=available_at,
            seed=1,
        )

    with pytest.raises(ValueError, match="must return a new checkpoint"):
        BrokenModel(ID="bad", behavior="self").train(
            model_frame(),
            trained_until=trained_until,
            available_at=available_at,
            seed=1,
        )

    with pytest.raises(ValueError, match="unexpected trained_until"):
        BrokenModel(ID="bad", behavior="wrong-time").train(
            model_frame(),
            trained_until=trained_until,
            available_at=available_at,
            seed=1,
        )

    with pytest.raises(TypeError, match="Model._predict must return a Polars DataFrame"):
        BrokenModel(ID="bad", behavior="bad-prediction").predict(model_frame())

    with pytest.raises(TypeError, match="requires a Polars DataFrame"):
        initial.predict(object())  # type: ignore[arg-type]


def test_concrete_model_subclasses_must_define_versions() -> None:
    class AbstractModel(Model):
        @abc.abstractmethod
        def extra_contract(self) -> None:
            pass

        def _train(
            self,
            frame: pl.DataFrame,
            *,
            trained_until: datetime,
            available_at: datetime,
            seed: int,
        ) -> Model:
            return self

        def _predict(self, frame: pl.DataFrame) -> pl.DataFrame:
            return frame

    assert inspect.isabstract(AbstractModel)

    with pytest.raises(TypeError, match="must define VERSION"):

        class MissingVersionModel(Model):
            def _train(
                self,
                frame: pl.DataFrame,
                *,
                trained_until: datetime,
                available_at: datetime,
                seed: int,
            ) -> Model:
                return self

            def _predict(self, frame: pl.DataFrame) -> pl.DataFrame:
                return frame

    with pytest.raises(TypeError, match="VERSION must be a string"):

        class NonStringVersionModel(Model):
            VERSION = 1  # type: ignore[assignment]

            def _train(
                self,
                frame: pl.DataFrame,
                *,
                trained_until: datetime,
                available_at: datetime,
                seed: int,
            ) -> Model:
                return self

            def _predict(self, frame: pl.DataFrame) -> pl.DataFrame:
                return frame

    with pytest.raises(ValueError, match="VERSION must be non-empty"):

        class EmptyVersionModel(Model):
            VERSION = " "

            def _train(
                self,
                frame: pl.DataFrame,
                *,
                trained_until: datetime,
                available_at: datetime,
                seed: int,
            ) -> Model:
                return self

            def _predict(self, frame: pl.DataFrame) -> pl.DataFrame:
                return frame
