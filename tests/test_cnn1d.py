from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from iosislib.core.graph import Graph
from iosislib.core.model import DatasetSplit, FrameDataset
from iosislib.core.node import Node
from iosislib.core.tsfn import FrameSignature, TSFN, TSFNConfig, TimeAxis
from iosislib.models.cnn1d import CNN1D, CNN1DConfig, CNN1DModel


def _series_frame(values: list[float]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "features": pl.Series(
                "features",
                [[value] for value in values],
                dtype=pl.Array(pl.Float64, 1),
            ),
            "target": pl.Series(
                "target",
                [[value] for value in values],
                dtype=pl.Array(pl.Float64, 1),
            ),
        }
    )


def _ar_frame(rows: int = 120, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    values = [0.0, 0.0, 0.0]
    for _ in range(3, rows):
        values.append(
            0.5 * values[-1]
            - 0.2 * values[-2]
            + 0.1 * values[-3]
            + float(rng.normal(0, 0.1))
        )
    return _series_frame(values)


def _split(frame: pl.DataFrame, train_rows: int = 80) -> DatasetSplit:
    return DatasetSplit(
        FrameDataset(frame.head(train_rows), batch_size=16),
        validation=FrameDataset(frame.slice(train_rows, frame.height - train_rows)),
    )


def _model(**overrides) -> CNN1DModel:
    params: dict[str, object] = {
        "feature_width": 1,
        "sequence_length": 8,
        "channels": (8,),
        "kernel_sizes": (3,),
        "dilations": (1,),
        "dense_hidden": (16,),
        "output_width": 1,
        "epochs": 5,
        "learning_rate": 0.02,
    }
    params.update(overrides)
    return CNN1DModel(**params)  # type: ignore[arg-type]


def test_cnn1d_learns_ar_signal() -> None:
    frame = _ar_frame()
    fitted = _model(epochs=30).fit(_split(frame), seed=0)
    prediction = fitted.predict(frame["features"]).to_list()
    assert len(prediction) == frame.height
    assert all(math.isnan(row[0]) for row in prediction[:7])
    assert all(math.isfinite(row[0]) for row in prediction[7:])
    tail_target = np.array([row[0] for row in frame["target"].to_list()[80:]])
    tail_pred = np.array([row[0] for row in prediction[80:]])
    assert float(np.mean((tail_target - tail_pred) ** 2)) < 0.05


def test_cnn1d_fit_is_deterministic() -> None:
    frame = _ar_frame()
    initial = _model()
    first = initial.fit(_split(frame), seed=3)
    repeated = initial.fit(_split(frame), seed=3)
    assert first.state == repeated.state
    assert initial.state is None


def test_cnn1d_is_causal() -> None:
    frame = _ar_frame()
    fitted = _model(epochs=10).fit(_split(frame), seed=0)
    values = [row[0] for row in frame["features"].to_list()]
    perturbed = [[999.0] if index == 60 else [value] for index, value in enumerate(values)]
    other = frame.with_columns(
        pl.Series(
            "features",
            perturbed,
            dtype=pl.Array(pl.Float64, 1),
        )
    )
    before = fitted.predict(frame["features"]).to_list()
    after = fitted.predict(other["features"]).to_list()
    assert before[59] == after[59]
    assert before[60] != after[60]


def test_cnn1d_rejects_short_training_frame() -> None:
    frame = _series_frame([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="at least 8 rows"):
        _model().fit(DatasetSplit(FrameDataset(frame)), seed=0)


def test_cnn1d_rejects_shuffled_training_data() -> None:
    frame = _ar_frame()
    split = DatasetSplit(FrameDataset(frame, shuffle=True))
    with pytest.raises(ValueError, match="shuffle_train=False"):
        _model().fit(split, seed=0)


def test_cnn1d_rejects_bad_architecture() -> None:
    with pytest.raises(ValueError, match="same length"):
        CNN1DConfig(channels=(8, 8), kernel_sizes=(3,))
    with pytest.raises(ValueError, match="sequence_length"):
        CNN1DConfig(sequence_length=2, channels=(8,), kernel_sizes=(3,))
    with pytest.raises(ValueError, match="shuffle_train=False"):
        CNN1DConfig(splitter={"shuffle_train": True})


@dataclass(frozen=True)
class SeriesSourceConfig(TSFNConfig):
    rows: int = 40


class SeriesSource(TSFN):
    VERSION = "1.0.0"
    CONFIG_CLS = SeriesSourceConfig

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        return (
            FrameSignature.empty(),
            FrameSignature(
                time=TimeAxis(column="timestamp"),
                columns=(
                    ("features", pl.Float64, (1,)),
                    ("target", pl.Float64, (1,)),
                ),
            ),
        )

    def apply(self) -> pl.LazyFrame:
        values = [float(index % 7) for index in range(self.parameters.rows)]
        return pl.DataFrame(
            {
                "timestamp": [
                    datetime(2026, 1, 1) + timedelta(hours=index)
                    for index in range(self.parameters.rows)
                ],
                "features": pl.Series(
                    "features",
                    [[value] for value in values],
                    dtype=pl.Array(pl.Float64, 1),
                ),
                "target": pl.Series(
                    "target",
                    [[value] for value in values],
                    dtype=pl.Array(pl.Float64, 1),
                ),
            }
        ).lazy()


def test_cnn1d_graph_walk_forward_with_history() -> None:
    src = Node(SeriesSource)
    model = Node(
        CNN1D,
        bindings={"features": src.features, "target": src.target},
        parameters={
            "sequence_length": 4,
            "channels": [4],
            "kernel_sizes": [2],
            "epochs": 1,
            "scheduler": {"every": 20},
        },
        name="cnn1d",
    )
    result = Graph(model).execute()
    assert result.height == 40
    assert result.schema["prediction"] == pl.Array(pl.Float64, 1)
