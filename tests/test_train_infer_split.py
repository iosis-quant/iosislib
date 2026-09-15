from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from iosislib.core.graph import Graph, GraphValidationError, LocalExecutor
from iosislib.core.model import (
    InferenceTSFN,
    Model,
    ModelStore,
    TrainingResult,
    TrainingTSFN,
    model_from_dict,
)
from iosislib.core.node import Node
from iosislib.core.tsfn import FrameSignature, TSFNConfig, TimeAxis
from iosislib.core.utils import MODEL_DIR_ENV_VAR, current_model_dir
from iosislib.tsfn.adapters.inline_sources import (
    DataFrameSource,
    DataFrameSourceConfig,
)


@dataclass(frozen=True, kw_only=True)
class MeanModel(Model):
    VERSION = "test-1"

    mean: float
    count: int = 0

    def _predict(self, features: pl.Series) -> pl.Series:
        return pl.Series(
            "prediction", [self.mean] * len(features), dtype=pl.Float64
        )


@dataclass(frozen=True)
class MeanTrainerConfig(TSFNConfig):
    model_group: str
    run_id: str | None = None
    model_dir: str | None = None


class MeanTrainer(TrainingTSFN):
    VERSION = "test-1"
    CONFIG_CLS = MeanTrainerConfig

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        time = TimeAxis("timestamp")
        return (
            FrameSignature(time=time, columns=(("value", pl.Float64),)),
            FrameSignature(time=time, columns=()),
        )

    def model_group(self) -> str:
        return self.parameters.model_group

    def train_frame(self, frame: pl.DataFrame) -> TrainingResult:
        assert current_model_dir() is not None
        mean = float(frame.get_column("value").mean())
        return TrainingResult(
            finished=MeanModel(mean=mean, count=frame.height),
            metrics={"mean": mean},
        )


@dataclass(frozen=True)
class MeanPredictConfig(TSFNConfig):
    model_group: str
    model_id: str | None = None
    frozen: bool = False
    model_dir: str | None = None


class MeanPredict(InferenceTSFN):
    VERSION = "test-1"
    CONFIG_CLS = MeanPredictConfig

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        time = TimeAxis("timestamp")
        return (
            FrameSignature(time=time, columns=(("value", pl.Float64),)),
            FrameSignature(time=time, columns=(("prediction", pl.Float64),)),
        )

    def model_group(self) -> str:
        return self.parameters.model_group

    def predict_with_model(
        self, segment: pl.DataFrame, model: Model
    ) -> pl.DataFrame:
        assert isinstance(model, MeanModel)
        return pl.DataFrame(
            [
                segment.get_column("timestamp"),
                pl.Series(
                    "prediction", [model.mean] * segment.height, dtype=pl.Float64
                ),
            ]
        )


def source_node(rows: int = 6, name: str = "src") -> Node:
    frame = pl.DataFrame(
        {
            "timestamp": [
                datetime(2026, 1, 1) + timedelta(hours=index) for index in range(rows)
            ],
            "value": [float(index) for index in range(rows)],
        },
        schema={"timestamp": pl.Datetime, "value": pl.Float64},
    )
    signature = FrameSignature(
        time=TimeAxis("timestamp"), columns=(("value", pl.Float64),)
    )
    return Node(
        DataFrameSource,
        config=DataFrameSourceConfig.from_frame(frame, signature),
        name=name,
    )


def test_live_graph_trains_then_predicts(tmp_path) -> None:
    src = source_node()
    trainer = Node(
        MeanTrainer,
        bindings={"value": src.output("value")},
        config=MeanTrainerConfig(model_group="means"),
        name="train",
    )
    inference = Node(
        MeanPredict,
        bindings={"value": src.output("value")},
        config=MeanPredictConfig(model_group="means"),
        name="infer",
    )
    graph = Graph([src, trainer, inference])

    assert {node.ID for node in graph.terminal_nodes} == {trainer.ID, inference.ID}

    outputs = graph.execute(LocalExecutor(model_dir=str(tmp_path)))
    assert isinstance(outputs, dict)
    assert set(outputs) == {trainer.ID, inference.ID}

    trained = outputs[trainer.ID]
    assert trained.height == 0
    assert trained.columns == ["timestamp"]

    predicted = outputs[inference.ID]
    assert predicted["prediction"].to_list() == [2.5] * 6

    manifest = ModelStore(tmp_path).read_manifest("means")
    assert len(manifest["runs"]) == 1
    assert manifest["runs"][0]["metrics"] == {"mean": 2.5}


def test_frozen_graph_without_trainer_reuses_store(tmp_path) -> None:
    src = source_node()
    trainer = Node(
        MeanTrainer,
        bindings={"value": src.output("value")},
        config=MeanTrainerConfig(model_group="means"),
        name="train",
    )
    Graph([src, trainer]).execute(LocalExecutor(model_dir=str(tmp_path)))

    inference = Node(
        MeanPredict,
        bindings={"value": src.output("value")},
        config=MeanPredictConfig(model_group="means"),
        name="infer",
    )
    frozen = Graph(inference)
    result = frozen.execute(LocalExecutor(model_dir=str(tmp_path)))
    assert isinstance(result, pl.DataFrame)
    assert result["prediction"].to_list() == [2.5] * 6


def test_pinned_model_id_is_frozen_and_in_identity(tmp_path) -> None:
    src = source_node()
    trainer = Node(
        MeanTrainer,
        bindings={"value": src.output("value")},
        config=MeanTrainerConfig(model_group="means"),
        name="train",
    )
    Graph([src, trainer]).execute(LocalExecutor(model_dir=str(tmp_path)))
    finished_id = ModelStore(tmp_path).latest_run("means")["finished_model_id"]

    pinned = Node(
        MeanPredict,
        bindings={"value": src.output("value")},
        config=MeanPredictConfig(model_group="means", model_id=finished_id),
        name="infer-pinned",
    )
    live = Node(
        MeanPredict,
        bindings={"value": src.output("value")},
        config=MeanPredictConfig(model_group="means"),
        name="infer-live",
    )
    assert pinned.ID != live.ID

    result = Graph(pinned).execute(LocalExecutor(model_dir=str(tmp_path)))
    assert isinstance(result, pl.DataFrame)
    assert result["prediction"].to_list() == [2.5] * 6

    bad = Node(
        MeanPredict,
        bindings={"value": src.output("value")},
        config=MeanPredictConfig(model_group="means", model_id="nope"),
        name="infer-bad",
    )
    with pytest.raises(RuntimeError, match="not a finished model") as exc_info:
        Graph(bad).execute(LocalExecutor(model_dir=str(tmp_path)))
    assert isinstance(exc_info.value.__cause__, LookupError)


def test_volatile_fields_stay_out_of_identity() -> None:
    src = source_node()
    base = Node(
        MeanPredict,
        bindings={"value": src.output("value")},
        config=MeanPredictConfig(model_group="means"),
        name="infer",
    )
    varied = Node(
        MeanPredict,
        bindings={"value": src.output("value")},
        config=MeanPredictConfig(
            model_group="means", frozen=True, model_dir="/elsewhere"
        ),
        name="infer",
    )
    assert base.ID == varied.ID

    trainer_base = Node(
        MeanTrainer,
        bindings={"value": src.output("value")},
        config=MeanTrainerConfig(model_group="means"),
        name="train",
    )
    trainer_varied = Node(
        MeanTrainer,
        bindings={"value": src.output("value")},
        config=MeanTrainerConfig(
            model_group="means", run_id="run-9", model_dir="/elsewhere"
        ),
        name="train",
    )
    assert trainer_base.ID == trainer_varied.ID


def test_duplicate_trainers_rejected() -> None:
    src = source_node()
    other = source_node(rows=4, name="src-other")
    first = Node(
        MeanTrainer,
        bindings={"value": src.output("value")},
        config=MeanTrainerConfig(model_group="means"),
        name="train-a",
    )
    second = Node(
        MeanTrainer,
        bindings={"value": other.output("value")},
        config=MeanTrainerConfig(model_group="means"),
        name="train-b",
    )
    assert first.ID != second.ID
    report = Graph.validate([src, other, first, second])
    assert not report.is_valid
    assert any(issue.code == "MULTIPLE_TRAINERS" for issue in report.issues)
    with pytest.raises(GraphValidationError):
        Graph([src, other, first, second])


def test_invalid_model_group_rejected() -> None:
    src = source_node()
    trainer = Node(
        MeanTrainer,
        bindings={"value": src.output("value")},
        config=MeanTrainerConfig(model_group=""),
        name="train",
    )
    report = Graph.validate([src, trainer])
    assert any(issue.code == "INVALID_MODEL_GROUP" for issue in report.issues)


def test_manifest_asof_selection_and_idempotent_retrain(tmp_path) -> None:
    store = ModelStore(tmp_path)
    first = store.save_run(
        "means",
        finished=MeanModel(mean=1.0, count=2),
        trained_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
    )
    second = store.save_run(
        "means",
        finished=MeanModel(mean=2.0, count=4),
        checkpoints=(MeanModel(mean=1.5, count=1),),
        trained_at=datetime(2026, 1, 5, tzinfo=timezone.utc),
    )
    assert first != second

    assert (
        store.select_run("means", datetime(2026, 1, 3, tzinfo=timezone.utc))[
            "finished_model_id"
        ]
        == first
    )
    assert (
        store.select_run("means", datetime(2026, 1, 9, tzinfo=timezone.utc))[
            "finished_model_id"
        ]
        == second
    )
    with pytest.raises(LookupError):
        store.select_run("means", datetime(2025, 12, 31, tzinfo=timezone.utc))

    rerun = store.save_run(
        "means",
        finished=MeanModel(mean=2.0, count=4),
        trained_at=datetime(2026, 1, 5, tzinfo=timezone.utc),
    )
    assert rerun == second
    assert len(store.read_manifest("means")["runs"]) == 2

    manifest = store.read_manifest("means")
    assert len(manifest["runs"][1]["checkpoints"]) == 1
    loaded = store.load_run_model("means", manifest["runs"][1])
    assert isinstance(loaded, MeanModel) and loaded.mean == 2.0


def test_checkpoint_round_trip_and_version_guard() -> None:
    checkpoint = MeanModel(mean=1.5, count=3)
    payload = model_from_dict(checkpoint.to_dict())
    assert payload == checkpoint

    tampered = dict(checkpoint.to_dict())
    tampered["version"] = "other"
    with pytest.raises(ValueError):
        model_from_dict(tampered)


def test_executor_scopes_model_dir_env(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(MODEL_DIR_ENV_VAR, str(tmp_path))
    src = source_node()
    trainer = Node(
        MeanTrainer,
        bindings={"value": src.output("value")},
        config=MeanTrainerConfig(model_group="means"),
        name="train",
    )
    inference = Node(
        MeanPredict,
        bindings={"value": src.output("value")},
        config=MeanPredictConfig(model_group="means"),
        name="infer",
    )
    assert current_model_dir() is None
    outputs = Graph([src, trainer, inference]).execute(LocalExecutor())
    assert isinstance(outputs, dict)
    assert outputs[inference.ID]["prediction"].to_list() == [2.5] * 6
    assert current_model_dir() is None
    assert __import__("os").environ[MODEL_DIR_ENV_VAR] == str(tmp_path)
