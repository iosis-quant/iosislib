from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import numpy.typing as npt
import polars as pl
import pytest

from iosislib.backtest import (
    BacktestConfig,
    BacktestTSFN,
    FeatureBuffer,
    ModelPolicy,
    Order,
    OrderModelPolicy,
    Venue,
)
from iosislib.backtest.feeds import L1Feed
from iosislib.core.graph import Graph
from iosislib.core.model import (
    ChronologicalSplitter,
    Dataset,
    EveryNTicksScheduler,
    FrozenScheduler,
    MetricThresholdScheduler,
    SupervisedModel,
)
from iosislib.core.node import Node
from iosislib.core.tsfn import FrameSignature, TSFN, TSFNConfig
from iosislib.core.utils import series_to_numpy
from iosislib.models.mlp import DenseMLPModel

Array = npt.NDArray[np.float64]
START = datetime(2026, 1, 1)


def market_frame(
    bid: list[list[float]],
    ask: list[list[float]],
    signal: list[list[float]],
) -> pl.DataFrame:
    width = len(bid[0])
    return pl.DataFrame(
        [
            pl.Series(
                "timestamp",
                [START + timedelta(minutes=row) for row in range(len(bid))],
                dtype=pl.Datetime,
            ),
            pl.Series("bid", bid, dtype=pl.Array(pl.Float64, width)),
            pl.Series("ask", ask, dtype=pl.Array(pl.Float64, width)),
            pl.Series("signal", signal, dtype=pl.Array(pl.Float64, width)),
        ]
    )


@dataclass(frozen=True, kw_only=True)
class VectorWeightModel(SupervisedModel):
    """Small immutable vector checkpoint used to exercise ModelPolicy."""

    VERSION = "1.0.0"

    weights: tuple[float, ...]

    def _fit(
        self,
        train: Dataset,
        validation: Dataset | None,
        *,
        seed: int,
    ) -> SupervisedModel:
        del validation, seed
        target = np.concatenate(
            [
                series_to_numpy(batch.get_column("target"), shape=(len(self.weights),))
                for batch in train.batches()
            ]
        )
        return VectorWeightModel(
            weights=tuple(float(value) for value in target.mean(axis=0))
        )

    def _predict(self, features: pl.Series) -> pl.Series:
        return pl.Series(
            "prediction",
            [list(self.weights)] * len(features),
            dtype=pl.Array(pl.Float64, len(self.weights)),
        )


@dataclass(frozen=True, kw_only=True)
class WrongWidthModel(SupervisedModel):
    VERSION = "1.0.0"

    def _fit(
        self, train: Dataset, validation: Dataset | None, *, seed: int
    ) -> SupervisedModel:
        del train, validation, seed
        return self

    def _predict(self, features: pl.Series) -> pl.Series:
        return pl.Series(
            "prediction",
            [[0.0, 0.0]] * len(features),
            dtype=pl.Array(pl.Float64, 2),
        )


@dataclass(frozen=True, kw_only=True)
class ScalarScoreModel(SupervisedModel):
    VERSION = "1.0.0"

    score: float

    def _fit(
        self, train: Dataset, validation: Dataset | None, *, seed: int
    ) -> SupervisedModel:
        del train, validation, seed
        return self

    def _predict(self, features: pl.Series) -> pl.Series:
        return pl.Series("prediction", [self.score] * len(features), dtype=pl.Float64)


@dataclass(frozen=True)
class ThresholdPolicy(OrderModelPolicy):
    """Interpret a binary-classification score as a one-unit long/short order."""

    def interpret(self, prediction: Array, orders: Array, row: int) -> None:
        orders[row] = np.where(prediction >= 0.5, 1.0, -1.0).astype(
            np.float64, copy=False
        )


@dataclass(frozen=True)
class ModelTapeConfig(TSFNConfig):
    pass


class ModelTape(TSFN[ModelTapeConfig]):
    VERSION = "1.0.0"
    CONFIG_CLS = ModelTapeConfig

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        return (
            FrameSignature.empty(),
            FrameSignature(
                columns=(
                    ("bid", pl.Float64, (1,)),
                    ("ask", pl.Float64, (1,)),
                    ("signal", pl.Float64, (1,)),
                    ("target", pl.Float64, (1,)),
                )
            ),
        )

    def apply(self) -> pl.LazyFrame:
        return labelled_frame([[1.0], [1.0]], [[0.1], [0.1]]).lazy()


def labelled_frame(
    signal: list[list[float]], target: list[list[float]]
) -> pl.DataFrame:
    width = len(signal[0])
    rows = len(signal)
    return market_frame(
        [[10.0] * width for _ in range(rows)],
        [[10.0] * width for _ in range(rows)],
        signal,
    ).with_columns(pl.Series("target", target, dtype=pl.Array(pl.Float64, width)))


def feed(width: int = 1) -> L1Feed:
    return L1Feed(Venue("test", tuple(f"A{index}" for index in range(width))))


def policy(
    model: SupervisedModel,
    scheduler: FrozenScheduler | EveryNTicksScheduler = FrozenScheduler(),
    *,
    feature_shape: tuple[int, ...] = (1,),
    target_shape: tuple[int, ...] = (1,),
    history_rows: int = 8,
) -> OrderModelPolicy:
    return OrderModelPolicy(
        model=model,
        scheduler=scheduler,
        splitter=ChronologicalSplitter(),
        feature_shape=feature_shape,
        target_shape=target_shape,
        history_rows=history_rows,
        seed=11,
    )


def function(model_policy: ModelPolicy, *, width: int = 1) -> BacktestTSFN:
    return BacktestTSFN(
        BacktestConfig(feed=feed(width), policy=model_policy, initial_cash=100.0)
    )


def test_model_policy_declares_and_requires_resolved_vector_targets() -> None:
    model_policy = policy(VectorWeightModel(weights=(0.0,)))
    backtest = function(model_policy)

    assert backtest.signature[0].columns[-1] == ("target", pl.Float64, (1,))
    with pytest.raises(ValueError, match="Missing required input column: 'target'"):
        backtest.batch(market_frame([[10.0]], [[10.0]], [[1.0]]))


def test_model_policy_interprets_regular_vector_predictions_as_orders() -> None:
    backtest = function(
        policy(
            VectorWeightModel(weights=(0.5, -0.25)),
            feature_shape=(2,),
            target_shape=(2,),
        ),
        width=2,
    )
    values = labelled_frame([[1.0, 2.0], [3.0, 4.0]], [[0.0, 0.0], [0.0, 0.0]])

    first = backtest.batch(values)
    second = backtest.batch(values)

    assert first.equals(second)
    assert first.get_column("order").to_list() == [[0.5, -0.25], [0.5, -0.25]]
    assert first.get_column("balance").to_list() == [[0.5, -0.25], [1.0, -0.5]]


def test_model_policy_retrains_only_on_earlier_resolved_labels() -> None:
    backtest = function(
        policy(
            VectorWeightModel(weights=(0.0,)), EveryNTicksScheduler(2), history_rows=2
        )
    )
    values = labelled_frame([[1.0], [1.0], [1.0], [1.0]], [[0.2], [0.2], [99.0], [0.2]])

    result = backtest.batch(values)

    assert result.get_column("order").to_list()[:3] == [[0.0], [0.0], [0.2]]


def test_model_policy_purges_trailing_labels_at_retraining_boundaries() -> None:
    def run(purge_window: int) -> list[list[float]]:
        model_policy = OrderModelPolicy(
            model=VectorWeightModel(weights=(0.0,)),
            scheduler=EveryNTicksScheduler(3),
            splitter=ChronologicalSplitter(purge_window=purge_window),
            feature_shape=(1,),
            target_shape=(1,),
            history_rows=100,
            seed=11,
        )
        values = labelled_frame(
            [[1.0], [1.0], [1.0], [1.0]],
            [[1.0], [2.0], [3.0], [4.0]],
        )
        return function(model_policy).batch(values).get_column("order").to_list()

    # At the row-3 boundary the buffer holds targets [1, 2, 3]. Purge drops the
    # trailing row (label realized only at the boundary) so the fitted mean is
    # 1.5; without a purge the model trains on all three and sizes 2.0.
    assert run(purge_window=1) == [[0.0], [0.0], [0.0], [1.5]]
    assert run(purge_window=0) == [[0.0], [0.0], [0.0], [2.0]]


def test_model_policy_uses_metrics_at_scheduler_check_boundaries() -> None:
    model_policy = OrderModelPolicy(
        model=VectorWeightModel(weights=(0.0,)),
        scheduler=MetricThresholdScheduler("mse", threshold=0.1, check_every=2),
        splitter=ChronologicalSplitter(),
        feature_shape=(1,),
        target_shape=(1,),
        history_rows=2,
        seed=11,
    )
    values = labelled_frame([[1.0], [1.0], [1.0]], [[0.5], [0.5], [99.0]])

    result = function(model_policy).batch(values)

    assert result.get_column("order").to_list() == [[0.0], [0.0], [0.5]]


def test_model_policy_validates_model_output_width() -> None:
    values = labelled_frame([[1.0]], [[0.0]])

    with pytest.raises(ValueError, match="could not broadcast"):
        function(policy(WrongWidthModel())).batch(values)


def test_model_policy_runs_in_graph_with_a_resolved_target_binding() -> None:
    tape = Node(ModelTape)
    model_policy = policy(VectorWeightModel(weights=(0.1,)))
    simulation = Node(
        BacktestTSFN,
        bindings={
            "bid": tape.bid,
            "ask": tape.ask,
            "signal": tape.signal,
            "target": tape.target,
        },
        config=BacktestConfig(feed=feed(), policy=model_policy, initial_cash=100.0),
    )

    result = Graph(simulation).execute()

    assert result.get_column("balance").to_list() == [[0.1], [0.2]]


def test_model_policy_accepts_scalar_dense_mlp_predictions_and_labels() -> None:
    model_policy = policy(
        DenseMLPModel(layers=(1, 1)),
        target_shape=(),
    )
    values = market_frame([[10.0], [10.0]], [[10.0], [10.0]], [[1.0], [2.0]])
    values = values.with_columns(
        pl.Series("target", [0.0, 0.0], dtype=pl.Float64)
    )

    result = function(model_policy).batch(values)

    assert function(model_policy).signature[0].columns[-1] == ("target", pl.Float64)
    assert result.get_column("order").to_list() == [[0.0], [0.0]]


def test_model_policy_interpreter_can_translate_classifier_scores() -> None:
    model_policy = ThresholdPolicy(
        model=ScalarScoreModel(score=0.75),
        scheduler=FrozenScheduler(),
        splitter=ChronologicalSplitter(),
        feature_shape=(1,),
        target_shape=(),
        history_rows=2,
    )
    values = market_frame([[10.0], [10.0]], [[10.0], [10.0]], [[1.0], [2.0]])
    values = values.with_columns(
        pl.Series("target", [1.0, 1.0], dtype=pl.Float64)
    )

    result = function(model_policy).batch(values)

    assert result.get_column("order").to_list() == [[1.0], [1.0]]


def test_feature_buffer_retains_only_configured_history_and_validates_inputs() -> None:
    buffer = FeatureBuffer(2, (1,), (1,))
    for value in (1.0, 2.0, 3.0):
        vector = np.array([value], dtype=np.float64)
        buffer.append(vector, vector)

    assert buffer.frame().get_column("features").to_list() == [[2.0], [3.0]]
    assert buffer.frame().get_column("target").to_list() == [[2.0], [3.0]]
    with pytest.raises(ValueError, match="positive integer"):
        FeatureBuffer(0, (1,), (1,))
    with pytest.raises(ValueError, match="vectors"):
        buffer.append(np.zeros((1, 1), dtype=np.float64), np.zeros(1, dtype=np.float64))
    with pytest.raises(TypeError, match="Float64"):
        buffer.append(np.zeros(1, dtype=np.float32), np.zeros(1, dtype=np.float64))
    with pytest.raises(ValueError, match="feature_shape"):
        buffer.append(np.zeros(2, dtype=np.float64), np.zeros(2, dtype=np.float64))


def test_model_policy_validates_construction_parameters() -> None:
    base = dict(
        model=VectorWeightModel(weights=(0.0,)),
        scheduler=FrozenScheduler(),
        splitter=ChronologicalSplitter(),
        history_rows=2,
        feature_shape=(1,),
        target_shape=(1,),
    )

    with pytest.raises(ValueError, match="positive integer"):
        OrderModelPolicy(**{**base, "history_rows": 0})
    with pytest.raises(TypeError, match="seed must be an integer"):
        OrderModelPolicy(**{**base, "seed": True})
