from __future__ import annotations

import json

import polars as pl
import pytest

from iosislib.core.model import (
    ChronologicalSplitter,
    DatasetSplit,
    EveryNTicksScheduler,
    FrameDataset,
)
from iosislib.models.lightgbm import LightGBM, LightGBMModel
from iosislib.models.mlp import DenseMLP, DenseMLPModel


def regression_frame(rows: int = 20) -> pl.DataFrame:
    features = [
        [float(index), float(index % 3)]
        for index in range(rows)
    ]
    target = [
        2.0 * values[0] - values[1] + 1.0
        for values in features
    ]
    return pl.DataFrame(
        {
            "features": pl.Series(
                "features",
                features,
                dtype=pl.Array(pl.Float64, 2),
            ),
            "target": pl.Series("target", target, dtype=pl.Float64),
        }
    )


def dataset_split(*, validation: bool = False) -> DatasetSplit:
    frame = regression_frame()
    if validation:
        return DatasetSplit(
            FrameDataset(frame.head(16), batch_size=4, shuffle=True),
            validation=FrameDataset(frame.tail(4), batch_size=2),
        )
    return DatasetSplit(FrameDataset(frame, batch_size=5, shuffle=True))


def mse(target: pl.Series, prediction: pl.Series) -> float:
    value = ((target - prediction) ** 2).mean()
    assert value is not None
    return float(value)


def test_dense_mlp_tsfn_accepts_layer_lists_and_declares_scalar_regression() -> None:
    function = DenseMLP(
        {
            "layers": [2, 8, 4, 1],
            "scheduler": EveryNTicksScheduler(5),
            "splitter": ChronologicalSplitter(),
            "epochs": 2,
        }
    )

    input_signature, output_signature = function.signature
    assert function.parameters.layers == (2, 8, 4, 1)
    assert input_signature.columns == (
        ("features", pl.Float64, (2,)),
        ("target", pl.Float64),
    )
    assert output_signature.columns == (("prediction", pl.Float64),)
    assert function.segment_metrics(
        pl.Series([1.0, 3.0]),
        pl.Series([2.0, 1.0]),
    ) == {"mse": 2.5}


def test_dense_mlp_fit_is_deterministic_and_returns_immutable_state() -> None:
    frame = regression_frame()
    initial = DenseMLPModel(
        layers=(2, 8, 1),
        epochs=50,
        learning_rate=0.03,
    )

    first = initial.fit(dataset_split(validation=True), seed=7)
    repeated = initial.fit(dataset_split(validation=True), seed=7)

    assert isinstance(first, DenseMLPModel)
    assert first is not initial
    assert first.state is not None
    assert first.state == repeated.state
    assert initial.state is None
    assert mse(frame["target"], first.predict(frame["features"])) < 1.0
    assert json.loads(str(first))["state"]["layers"] == [2, 8, 1]


def test_lightgbm_fit_is_deterministic_regression_with_serializable_booster() -> None:
    pytest.importorskip("lightgbm")
    frame = regression_frame()
    initial = LightGBMModel(
        feature_width=2,
        num_boost_round=30,
        learning_rate=0.1,
        min_data_in_leaf=1,
    )

    first = initial.fit(dataset_split(), seed=11)
    repeated = initial.fit(dataset_split(), seed=11)

    assert isinstance(first, LightGBMModel)
    assert first is not initial
    assert first.model_text is not None
    assert first.model_text == repeated.model_text
    assert initial.model_text is None
    assert mse(frame["target"], first.predict(frame["features"])) < 10.0
    assert json.loads(str(first))["state"]["model_text"].startswith("tree")


def test_lightgbm_tsfn_declares_width_and_mse_metric() -> None:
    function = LightGBM(
        {
            "feature_width": 3,
            "scheduler": EveryNTicksScheduler(4),
            "splitter": ChronologicalSplitter(),
            "num_boost_round": 2,
        }
    )

    assert function.signature[0].columns[0] == (
        "features",
        pl.Float64,
        (3,),
    )
    assert function.initial_model().predict(
        pl.Series(
            "features",
            [[1.0, 2.0, 3.0]],
            dtype=pl.Array(pl.Float64, 3),
        )
    ).to_list() == [0.0]
    assert function.segment_metrics(
        pl.Series([1.0, 3.0]),
        pl.Series([2.0, 1.0]),
    ) == {"mse": 2.5}


@pytest.mark.parametrize(
    ("layers", "message"),
    [
        ([2], "input and output"),
        ([2, 0, 1], "positive"),
        ([2, 3, 2], "width 1"),
        ([2, True, 1], "integers"),
    ],
)
def test_dense_mlp_rejects_invalid_layer_architectures(
    layers: list[int],
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        DenseMLP(
            {
                "layers": layers,
                "scheduler": EveryNTicksScheduler(2),
                "splitter": ChronologicalSplitter(),
            }
        )


def test_model_checkpoints_reject_wrong_feature_width() -> None:
    wrong_width = pl.Series(
        "features",
        [[1.0, 2.0, 3.0]],
        dtype=pl.Array(pl.Float64, 3),
    )
    trained = DenseMLPModel(
        layers=(2, 1),
        state=((1.0, 1.0), (0.0,)),
    )

    with pytest.raises(ValueError, match="Expected 2 features"):
        trained.predict(wrong_width)
