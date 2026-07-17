from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import FrozenInstanceError, dataclass
from datetime import datetime, timedelta

import polars as pl
import pytest

from src.classes import (
    AnyScheduler,
    ChronologicalSplitter,
    Dataset,
    DatasetSplit,
    DatasetSplitter,
    EveryNTicksScheduler,
    FrameDataset,
    FrameSignature,
    FrozenScheduler,
    Graph,
    MetricThresholdScheduler,
    Node,
    NullPolicy,
    ScheduleContext,
    ScheduleDecision,
    Scheduler,
    SupervisedModel,
    SupervisedModelTSFN,
    TSFN,
    TSFNConfig,
    TimeAxis,
)


def supervised_frame(rows: int = 10) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "features": [[float(index), float(index * 10)] for index in range(rows)],
            "target": [float(index * 2) for index in range(rows)],
        },
        schema={
            "features": pl.Array(pl.Float64, 2),
            "target": pl.Float64,
        },
    )


def model_frame(rows: int = 10) -> pl.DataFrame:
    return supervised_frame(rows).insert_column(
        0,
        pl.Series(
            "timestamp",
            [datetime(2026, 1, 1) + timedelta(hours=index) for index in range(rows)],
            pl.Datetime,
        ),
    )


def collect(dataset: Dataset, *, epoch: int = 0, seed: int = 0) -> pl.DataFrame:
    return pl.concat(list(dataset.batches(epoch=epoch, seed=seed)))


def schedule_context(**overrides) -> ScheduleContext:
    values = {
        "total_rows": 20,
        "rows_seen": 10,
        "rows_since_retrain": 5,
        "retrain_count": 1,
        "metrics": (("mse", 2.0),),
    }
    values.update(overrides)
    return ScheduleContext(**values)


def test_frame_dataset_batches_are_repeatable_and_schema_stable() -> None:
    dataset = FrameDataset(supervised_frame(5), batch_size=2)

    first = list(dataset.batches())
    second = list(dataset.batches())

    assert [batch.height for batch in first] == [2, 2, 1]
    assert all(batch.schema == dataset.schema for batch in first)
    assert pl.concat(first).equals(pl.concat(second))
    assert dataset.row_count == 5


def test_frame_dataset_shuffle_is_deterministic_per_seed_and_epoch() -> None:
    dataset = FrameDataset(supervised_frame(10), batch_size=3, shuffle=True)

    first = collect(dataset, epoch=2, seed=7)
    repeated = collect(dataset, epoch=2, seed=7)
    next_epoch = collect(dataset, epoch=3, seed=7)

    assert first.equals(repeated)
    assert first["target"].to_list() != next_epoch["target"].to_list()
    assert sorted(first["target"]) == [float(index * 2) for index in range(10)]


def test_frame_dataset_drop_last_is_explicit() -> None:
    batches = list(
        FrameDataset(
            supervised_frame(5),
            batch_size=2,
            drop_last=True,
        ).batches()
    )

    assert [batch.height for batch in batches] == [2, 2]


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"batch_size": 0}, ValueError),
        ({"batch_size": None, "drop_last": True}, ValueError),
        ({"batch_size": 10, "drop_last": True}, ValueError),
        ({"shuffle": 1}, TypeError),
        ({"drop_last": 1}, TypeError),
    ],
)
def test_frame_dataset_rejects_invalid_batching(kwargs, error) -> None:
    with pytest.raises(error):
        FrameDataset(supervised_frame(3), **kwargs)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"epoch": -1}, ValueError),
        ({"epoch": True}, ValueError),
        ({"seed": True}, TypeError),
    ],
)
def test_dataset_iteration_rejects_invalid_reproducibility_inputs(kwargs, error) -> None:
    with pytest.raises(error):
        list(FrameDataset(supervised_frame(2)).batches(**kwargs))


def test_splitter_requires_only_canonical_supervised_columns() -> None:
    splitter = ChronologicalSplitter()

    with pytest.raises(ValueError, match="canonical"):
        splitter.split(model_frame(3))
    with pytest.raises(ValueError, match="canonical"):
        splitter.split(supervised_frame(3).rename({"target": "label"}))


def test_chronological_splitter_preserves_graph_established_order_and_gaps() -> None:
    datasets = ChronologicalSplitter(
        validation_size=2,
        test_size=2,
        gap=1,
        batch_size=2,
    ).split(supervised_frame(10).reverse())

    assert collect(datasets.train)["target"].to_list() == [18.0, 16.0, 14.0, 12.0]
    assert collect(datasets.validation)["target"].to_list() == [8.0, 6.0]
    assert collect(datasets.test)["target"].to_list() == [2.0, 0.0]


def test_chronological_splitter_supports_fractional_and_absent_holdouts() -> None:
    fractional = ChronologicalSplitter(
        validation_size=0.2,
        test_size=0.2,
    ).split(supervised_frame(10))
    train_only = ChronologicalSplitter().split(supervised_frame(3))

    assert fractional.train.row_count == 6
    assert fractional.validation.row_count == 2
    assert fractional.test.row_count == 2
    assert train_only.train.row_count == 3
    assert train_only.validation is None
    assert train_only.test is None


def test_chronological_splitter_rejects_destructive_splits() -> None:
    with pytest.raises(ValueError, match="no training rows"):
        ChronologicalSplitter(validation_size=2, test_size=2, gap=1).split(
            supervised_frame(5)
        )
    with pytest.raises(ValueError, match="finite batch_size"):
        ChronologicalSplitter(drop_last=True)


class BadSplitter(DatasetSplitter):
    def _split(self, frame: pl.DataFrame, *, seed: int):
        return frame


class DuplicatingSplitter(DatasetSplitter):
    def _split(self, frame: pl.DataFrame, *, seed: int) -> DatasetSplit:
        del seed
        dataset = FrameDataset(frame)
        return DatasetSplit(dataset, test=dataset)


def test_splitter_and_dataset_split_enforce_their_roles() -> None:
    with pytest.raises(TypeError, match="DatasetSplit"):
        BadSplitter().split(supervised_frame(3))

    train = FrameDataset(supervised_frame(2))
    with pytest.raises(TypeError, match="validation"):
        DatasetSplit(train, validation=supervised_frame(1))  # type: ignore[arg-type]

    wrong_schema = FrameDataset(
        supervised_frame(2).with_columns(pl.col("target").cast(pl.Float32))
    )
    with pytest.raises(TypeError, match="schema must match"):
        DatasetSplit(train, validation=wrong_schema)

    with pytest.raises(ValueError, match="more rows"):
        DuplicatingSplitter().split(supervised_frame(3))


def test_schedule_context_is_immutable_normalized_metric_state() -> None:
    context = schedule_context(metrics=(("z", 3), ("mse", 2.0)))

    assert context.metrics == (("mse", 2.0), ("z", 3.0))
    assert context.metric("mse") == 2.0
    assert context.metric("missing") is None
    assert json.loads(str(context))["fields"]["rows_seen"] == 10
    with pytest.raises(FrozenInstanceError):
        context.rows_seen = 11  # type: ignore[misc]


@pytest.mark.parametrize(
    "overrides",
    [
        {"rows_seen": 21},
        {"rows_since_retrain": 11},
        {"metrics": (("mse", float("nan")),)},
        {"metrics": (("mse", 1.0), ("mse", 2.0))},
    ],
)
def test_schedule_context_rejects_incoherent_state(overrides) -> None:
    with pytest.raises((TypeError, ValueError)):
        schedule_context(**overrides)


def test_builtin_schedulers_make_segment_level_decisions() -> None:
    initial = schedule_context(
        rows_seen=0,
        rows_since_retrain=0,
        retrain_count=0,
        metrics=(),
    )

    assert FrozenScheduler().decide(schedule_context()) == ScheduleDecision(False, 20)
    assert EveryNTicksScheduler(5).decide(initial) == ScheduleDecision(False, 5)
    assert EveryNTicksScheduler(5).decide(schedule_context()) == ScheduleDecision(
        True, 15
    )


def test_metric_and_composite_schedulers_use_completed_segment_metrics() -> None:
    metric = MetricThresholdScheduler("mse", threshold=1.5, check_every=2)
    combined = AnyScheduler((EveryNTicksScheduler(5), metric))

    assert metric.decide(schedule_context()).retrain is True
    assert metric.decide(schedule_context(metrics=(("mse", 1.0),))).retrain is False
    assert metric.decide(schedule_context(metrics=())).retrain is False
    assert combined.decide(schedule_context()) == ScheduleDecision(True, 12)


class BrokenScheduler(Scheduler):
    def __init__(self, result) -> None:
        self.result = result

    def _decide(self, context: ScheduleContext):
        return self.result


@pytest.mark.parametrize(
    ("result", "message"),
    [
        (None, "ScheduleDecision"),
        (ScheduleDecision(False, 10), "advance"),
        (ScheduleDecision(False, 21), "exceeds"),
    ],
)
def test_scheduler_rejects_malformed_or_nonprogressing_decisions(result, message) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        BrokenScheduler(result).decide(schedule_context())


def test_scheduler_cannot_retrain_without_history() -> None:
    initial = schedule_context(
        rows_seen=0,
        rows_since_retrain=0,
        retrain_count=0,
        metrics=(),
    )
    with pytest.raises(ValueError, match="without historical rows"):
        BrokenScheduler(ScheduleDecision(True, 5)).decide(initial)


@dataclass(frozen=True)
class ObservationConfig(TSFNConfig):
    timestamps: tuple[datetime, ...]
    factor_a: tuple[float, ...]
    factor_b: tuple[float, ...]
    outcomes: tuple[float, ...]


class ObservationSource(TSFN):
    VERSION = "1.0.0"
    CONFIG_CLS = ObservationConfig

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        return (
            FrameSignature.empty(),
            FrameSignature(
                columns=(
                    ("factor_a", pl.Float64),
                    ("factor_b", pl.Float64),
                    ("outcome", pl.Float64),
                )
            ),
        )

    def apply(self) -> pl.LazyFrame:
        return pl.DataFrame(
            {
                "timestamp": self.parameters.timestamps,
                "factor_a": self.parameters.factor_a,
                "factor_b": self.parameters.factor_b,
                "outcome": self.parameters.outcomes,
            },
            schema={
                "timestamp": pl.Datetime,
                "factor_a": pl.Float64,
                "factor_b": pl.Float64,
                "outcome": pl.Float64,
            },
        ).lazy()


class PackFeatures(TSFN):
    VERSION = "1.0.0"

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        return (
            FrameSignature(
                columns=(
                    ("factor_a", pl.Float64),
                    ("factor_b", pl.Float64),
                )
            ),
            FrameSignature(columns=(("features", pl.Float64, (2,)),)),
        )

    def apply(self, lf: pl.LazyFrame | None = None) -> pl.LazyFrame:
        if lf is None:
            raise ValueError("PackFeatures requires an input frame")
        return lf.select(
            "timestamp",
            pl.concat_arr("factor_a", "factor_b").alias("features"),
        )


class SelectTarget(TSFN):
    VERSION = "1.0.0"

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        return (
            FrameSignature(columns=(("outcome", pl.Float64),)),
            FrameSignature(columns=(("target", pl.Float64),)),
        )

    def apply(self, lf: pl.LazyFrame | None = None) -> pl.LazyFrame:
        if lf is None:
            raise ValueError("SelectTarget requires an input frame")
        return lf.select("timestamp", pl.col("outcome").alias("target"))


def dataset_column(dataset: Dataset, column: str, *, seed: int = 0) -> pl.Series:
    return pl.concat(
        [batch.get_column(column) for batch in dataset.batches(seed=seed)]
    )


@dataclass(frozen=True, kw_only=True)
class MeanCheckpoint(SupervisedModel):
    VERSION = "1.0.0"
    mean: float = 0.0

    def _fit(
        self,
        train: Dataset,
        validation: Dataset | None,
        *,
        seed: int,
    ) -> SupervisedModel:
        del validation
        mean = dataset_column(train, "target", seed=seed).mean()
        assert mean is not None
        return MeanCheckpoint(mean=float(mean))

    def _predict(self, features: pl.Series) -> pl.Series:
        return pl.repeat(
            self.mean,
            len(features),
            dtype=pl.Float64,
            eager=True,
        )


@dataclass(frozen=True)
class MeanModelConfig(TSFNConfig):
    scheduler: Scheduler
    splitter: DatasetSplitter
    seed: int = 17
    initial_mean: float = 0.0


class MeanModelTSFN(SupervisedModelTSFN):
    VERSION = "1.0.0"
    CONFIG_CLS = MeanModelConfig

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        return (
            FrameSignature(
                time=TimeAxis("timestamp"),
                columns=(
                    ("features", pl.Float64, (2,)),
                    ("target", pl.Float64),
                ),
            ),
            FrameSignature(
                time=TimeAxis("timestamp"),
                columns=(("prediction", pl.Float64),),
            ),
        )

    def initial_model(self) -> SupervisedModel:
        return MeanCheckpoint(mean=self.parameters.initial_mean)

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
        return {"mse": float((((target - prediction) ** 2).mean()))}


def example_data() -> ObservationConfig:
    timestamps = tuple(
        datetime(2026, 1, 1) + timedelta(hours=index) for index in range(9)
    )
    return ObservationConfig(
        timestamps=timestamps,
        factor_a=tuple(float(index) for index in range(9)),
        factor_b=tuple(float(index * 10) for index in range(9)),
        outcomes=(1.0, 1.0, 1.0, 1.0, 9.0, 9.0, 9.0, 9.0, 9.0),
    )


def build_graph(
    *,
    data: ObservationConfig | None = None,
    scheduler: Scheduler | None = None,
    splitter: DatasetSplitter | None = None,
) -> Graph:
    data = example_data() if data is None else data
    observations = Node(
        ObservationSource,
        parameters={
            "timestamps": data.timestamps,
            "factor_a": data.factor_a,
            "factor_b": data.factor_b,
            "outcomes": data.outcomes,
        },
        name="observations",
    )
    features = Node(
        PackFeatures,
        bindings={
            "factor_a": observations.factor_a,
            "factor_b": observations.factor_b,
        },
        name="features",
    )
    target = Node(
        SelectTarget,
        bindings={"outcome": observations.outcome},
        name="target",
    )
    model = Node(
        MeanModelTSFN,
        bindings={
            "features": features.features,
            "target": target.target,
        },
        parameters={
            "scheduler": EveryNTicksScheduler(3) if scheduler is None else scheduler,
            "splitter": (
                ChronologicalSplitter(validation_size=1, test_size=1, batch_size=2)
                if splitter is None
                else splitter
            ),
            "seed": 17,
            "initial_mean": 0.0,
        },
        name="walk_forward_mean",
    )
    return Graph(model)


def test_supervised_tsfn_materialization_and_contract_are_implicit() -> None:
    graph = build_graph()
    model_node = graph.root_node

    assert SupervisedModelTSFN.REQUIRES_MATERIALIZATION is True
    assert MeanModelTSFN.DEFAULT_NULL_POLICY is NullPolicy.ERROR
    assert model_node.function.requires_materialization is True
    assert model_node.materialize is True
    assert graph.materialized_node_ids == frozenset({model_node.ID})
    assert graph.verify() is None


def test_model_dataset_is_constructed_by_ordinary_graph_nodes() -> None:
    model_node = build_graph().root_node

    assert set(model_node.bindings) == {"features", "target"}
    assert model_node.bindings["features"][0].function_cls is PackFeatures
    assert model_node.bindings["target"][0].function_cls is SelectTarget


def test_model_verification_does_not_execute_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = build_graph()

    def explode(self, frame: pl.DataFrame) -> pl.DataFrame:
        raise RuntimeError("model batch executed")

    monkeypatch.setattr(MeanModelTSFN, "batch", explode)

    assert graph.verify() is None
    with pytest.raises(RuntimeError, match="model batch executed"):
        graph.execute()


def test_model_execution_is_repeatable_and_returns_only_predictions() -> None:
    first = build_graph().execute()
    repeated = build_graph().execute()

    assert first.equals(repeated)
    assert first.columns == ["timestamp", "prediction"]
    assert first.height == 9


def test_prediction_runs_once_per_scheduler_segment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch_lengths: list[int] = []
    original = MeanCheckpoint._predict

    def record(self: MeanCheckpoint, features: pl.Series) -> pl.Series:
        batch_lengths.append(len(features))
        return original(self, features)

    monkeypatch.setattr(MeanCheckpoint, "_predict", record)
    build_graph().execute()

    assert batch_lengths == [3, 3, 3]


def test_frozen_model_predicts_the_entire_frame_in_one_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[str, pl.DataType, int]] = []
    original = MeanCheckpoint._predict

    def record(self: MeanCheckpoint, features: pl.Series) -> pl.Series:
        seen.append((features.name, features.dtype, len(features)))
        return original(self, features)

    monkeypatch.setattr(MeanCheckpoint, "_predict", record)
    build_graph(scheduler=FrozenScheduler()).execute()

    assert seen == [("features", pl.Array(pl.Float64, 2), 9)]


def test_training_uses_only_rows_before_the_transition() -> None:
    data = example_data()
    altered = ObservationConfig(
        timestamps=data.timestamps,
        factor_a=data.factor_a,
        factor_b=data.factor_b,
        outcomes=(1.0, 1.0, 1.0, 100.0, 9.0, 9.0, 9.0, 9.0, 9.0),
    )

    result = build_graph(
        data=altered,
        splitter=ChronologicalSplitter(),
    ).execute()

    assert result["prediction"].to_list()[:3] == [0.0, 0.0, 0.0]
    assert result["prediction"].to_list()[3] == pytest.approx(1.0)


def test_training_seeds_advance_only_at_fit_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seeds: list[int] = []
    original = MeanCheckpoint._fit

    def record(
        self: MeanCheckpoint,
        train: Dataset,
        validation: Dataset | None,
        *,
        seed: int,
    ) -> SupervisedModel:
        seeds.append(seed)
        return original(self, train, validation, seed=seed)

    monkeypatch.setattr(MeanCheckpoint, "_fit", record)
    build_graph().execute()

    assert seeds == [17, 18]


def test_metric_scheduler_retrains_from_completed_segment_metrics() -> None:
    result = build_graph(
        scheduler=MetricThresholdScheduler("mse", threshold=0.5, check_every=3),
        splitter=ChronologicalSplitter(),
    ).execute()

    predictions = result["prediction"].to_list()
    assert predictions[:3] == [0.0, 0.0, 0.0]
    assert predictions[3:6] == [1.0, 1.0, 1.0]
    assert predictions[6] == pytest.approx(11.0 / 3.0)


def test_nonfinite_metrics_fail_with_node_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        MeanModelTSFN,
        "segment_metrics",
        lambda self, target, prediction: {"mse": float("nan")},
    )

    with pytest.raises(RuntimeError, match="Metric 'mse' must be finite"):
        build_graph().execute()


def test_input_nulls_fail_before_backend_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = example_data()
    null_data = ObservationConfig(
        timestamps=data.timestamps,
        factor_a=(None, *data.factor_a[1:]),  # type: ignore[arg-type]
        factor_b=data.factor_b,
        outcomes=data.outcomes,
    )
    called = False

    def record(self: MeanCheckpoint, features: pl.Series) -> pl.Series:
        nonlocal called
        called = True
        return pl.repeat(0.0, len(features), eager=True)

    monkeypatch.setattr(MeanCheckpoint, "_predict", record)

    with pytest.raises(RuntimeError, match="NullPolicy.ERROR failed.*features"):
        build_graph(data=null_data).execute()
    assert called is False


def test_unsorted_direct_batch_is_sorted_once_before_segmenting() -> None:
    function = MeanModelTSFN(
        {
            "scheduler": FrozenScheduler(),
            "splitter": ChronologicalSplitter(),
        }
    )

    result = function.batch(model_frame(5).reverse())

    assert result["timestamp"].is_sorted()
    assert result.height == 5


def test_empty_batch_preserves_the_declared_output_schema() -> None:
    function = MeanModelTSFN(
        {
            "scheduler": FrozenScheduler(),
            "splitter": ChronologicalSplitter(),
        }
    )

    assert function.batch(model_frame(0)).schema == pl.Schema(
        {"timestamp": pl.Datetime("us"), "prediction": pl.Float64}
    )


class Float32Checkpoint(MeanCheckpoint):
    VERSION = "1.0.0"

    def _predict(self, features: pl.Series) -> pl.Series:
        return pl.repeat(0.0, len(features), dtype=pl.Float32, eager=True)


class Float32ModelTSFN(MeanModelTSFN):
    VERSION = "1.0.0"

    def initial_model(self) -> SupervisedModel:
        return Float32Checkpoint()


def test_prediction_dtype_is_enforced_inside_the_supervised_boundary() -> None:
    function = Float32ModelTSFN(
        {
            "scheduler": FrozenScheduler(),
            "splitter": ChronologicalSplitter(),
        }
    )

    with pytest.raises(TypeError, match="prediction type mismatch"):
        function.batch(model_frame(2))


class WrongContractModelTSFN(MeanModelTSFN):
    VERSION = "1.0.0"

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        input_signature, output_signature = super().type_signature()
        return (
            FrameSignature(
                time=input_signature.time,
                columns=(("feature", pl.Float64), ("target", pl.Float64)),
            ),
            output_signature,
        )


def test_noncanonical_supervised_signatures_fail_at_construction() -> None:
    with pytest.raises(ValueError, match="exactly .*features.*target"):
        WrongContractModelTSFN(
            {
                "scheduler": FrozenScheduler(),
                "splitter": ChronologicalSplitter(),
            }
        )


def test_scheduler_and_splitter_configuration_participate_in_node_identity() -> None:
    every_three = build_graph(scheduler=EveryNTicksScheduler(3))
    repeated = build_graph(scheduler=EveryNTicksScheduler(3))
    every_four = build_graph(scheduler=EveryNTicksScheduler(4))

    assert every_three.ID == repeated.ID
    assert every_three.ID != every_four.ID
