from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from iosislib.core.graph import Graph
from iosislib.core.model import (
    AnyScheduler,
    ChronologicalSplitter,
    DatasetSplit,
    EveryNTicksScheduler,
    FrameDataset,
    FrozenScheduler,
    MetricThresholdScheduler,
    scheduler_from_declaration,
    splitter_from_declaration,
)
from iosislib.core.node import Node
from iosislib.core.tsfn import FrameSignature, TSFN, TSFNConfig, TimeAxis
from iosislib.models.lightgbm import LightGBM, LightGBMModel
from iosislib.models.mlp import DenseMLP, DenseMLPModel


def regression_frame(rows: int = 20, target_width: int = 1) -> pl.DataFrame:
    features = [
        [float(index), float(index % 3)]
        for index in range(rows)
    ]
    target = [
        [
            2.0 * values[0] - values[1] + 1.0 + float(column)
            for column in range(target_width)
        ]
        for values in features
    ]
    return pl.DataFrame(
        {
            "features": pl.Series(
                "features",
                features,
                dtype=pl.Array(pl.Float64, 2),
            ),
            "target": pl.Series(
                "target",
                target,
                dtype=pl.Array(pl.Float64, target_width),
            ),
        }
    )


def dataset_split(
    *,
    validation: bool = False,
    target_width: int = 1,
) -> DatasetSplit:
    frame = regression_frame(target_width=target_width)
    if validation:
        return DatasetSplit(
            FrameDataset(frame.head(16), batch_size=4, shuffle=True),
            validation=FrameDataset(frame.tail(4), batch_size=2),
        )
    return DatasetSplit(FrameDataset(frame, batch_size=5, shuffle=True))


def mse(target: pl.Series, prediction: pl.Series) -> float:
    target_values = np.asarray(target.to_list(), dtype=np.float64).reshape(-1)
    prediction_values = np.asarray(prediction.to_list(), dtype=np.float64).reshape(-1)
    return float(np.mean((target_values - prediction_values) ** 2))


@dataclass(frozen=True)
class VectorSourceConfig(TSFNConfig):
    rows: int = 12


class VectorSource(TSFN):
    VERSION = "1.0.0"
    CONFIG_CLS = VectorSourceConfig

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        return (
            FrameSignature.empty(),
            FrameSignature(
                time=TimeAxis(column="timestamp"),
                columns=(
                    ("features", pl.Float64, (3,)),
                    ("target", pl.Float64, (2,)),
                ),
            ),
        )

    def apply(self) -> pl.LazyFrame:
        rows = self.parameters.rows
        timestamps = [
            datetime(2026, 1, 1) + timedelta(hours=index)
            for index in range(rows)
        ]
        return pl.DataFrame(
            {
                "timestamp": timestamps,
                "features": pl.Series(
                    "features",
                    [[float(row), float(row % 2), float(row % 5)] for row in range(rows)],
                    dtype=pl.Array(pl.Float64, 3),
                ),
                "target": pl.Series(
                    "target",
                    [[float(row), float(row + 1)] for row in range(rows)],
                    dtype=pl.Array(pl.Float64, 2),
                ),
            }
        ).lazy()


def test_dense_mlp_tsfn_declares_vector_regression() -> None:
    function = DenseMLP(
        {
            "feature_width": 2,
            "target_width": 2,
            "hidden_layers": [8, 4],
            "scheduler": {"every": 5},
            "splitter": {"validation_size": 0.2},
            "epochs": 2,
        }
    )

    input_signature, output_signature = function.signature
    assert function.parameters.hidden_layers == (8, 4)
    assert input_signature.columns == (
        ("features", pl.Float64, (2,)),
        ("target", pl.Float64, (2,)),
    )
    assert output_signature.columns == (("prediction", pl.Float64, (2,)),)
    assert function.parameters.scheduler == EveryNTicksScheduler(5)
    assert function.parameters.splitter == ChronologicalSplitter(validation_size=0.2)
    assert function.segment_metrics(
        pl.Series("target", [[1.0, 2.0]]),
        pl.Series("prediction", [[2.0, 1.0]]),
    ) == {"mse": 1.0}


def test_dense_mlp_fit_is_deterministic_and_returns_immutable_state() -> None:
    frame = regression_frame(target_width=2)
    initial = DenseMLPModel(
        layers=(2, 8, 2),
        epochs=50,
        learning_rate=0.03,
    )

    first = initial.fit(dataset_split(validation=True, target_width=2), seed=7)
    repeated = initial.fit(dataset_split(validation=True, target_width=2), seed=7)

    assert isinstance(first, DenseMLPModel)
    assert first is not initial
    assert first.state is not None
    assert first.state == repeated.state
    assert initial.state is None
    assert mse(frame["target"], first.predict(frame["features"])) < 1.0
    assert json.loads(str(first))["state"]["layers"] == [2, 8, 2]


def test_dense_mlp_scalar_target_widens_to_width_one() -> None:
    initial = DenseMLPModel(layers=(2, 1), epochs=200, learning_rate=0.1)
    checkpoint = initial.fit(dataset_split(), seed=3)

    prediction = checkpoint.predict(
        pl.Series(
            "features",
            [[1.0, 2.0]],
            dtype=pl.Array(pl.Float64, 2),
        )
    )

    assert prediction.dtype == pl.Array(pl.Float64, 1)
    assert prediction.to_list() == [[pytest.approx(1.0, abs=1.0)]]


def test_dense_mlp_rejects_invalid_hidden_layers() -> None:
    for hidden, message in (
        ([0], "positive"),
        ([2, -1], "positive"),
        ([True], "integers"),
    ):
        with pytest.raises((TypeError, ValueError), match=message):
            DenseMLP(
                {
                    "feature_width": 2,
                    "target_width": 1,
                    "hidden_layers": hidden,
                }
            )


def test_dense_mlp_model_rejects_short_architecture() -> None:
    with pytest.raises(ValueError, match="input and output"):
        DenseMLPModel(layers=(2,))


def test_dense_mlp_model_rejects_wrong_feature_width_on_predict() -> None:
    trained = DenseMLPModel(layers=(2, 1), epochs=1)
    fitted = trained.fit(dataset_split(), seed=3)

    with pytest.raises(ValueError, match="Expected 2 features"):
        fitted.predict(
            pl.Series(
                "features",
                [[1.0]],
                dtype=pl.Array(pl.Float64, 1),
            )
        )


def test_model_widths_derive_from_bound_columns() -> None:
    src = Node(VectorSource)
    model = Node(
        DenseMLP,
        bindings={"features": src.features, "target": src.target},
        parameters={
            "hidden_layers": [4],
            "epochs": 1,
            "scheduler": {"every": 4},
            "splitter": {"test_size": 2},
        },
        name="dense_mlp",
    )

    input_columns = {
        entry[0]: (entry[1], entry[2]) for entry in model.function.signature[0].columns
    }
    output_columns = {
        entry[0]: (entry[1], entry[2]) for entry in model.function.signature[1].columns
    }
    assert input_columns["features"] == (pl.Float64, (3,))
    assert input_columns["target"] == (pl.Float64, (2,))
    assert output_columns["prediction"] == (pl.Float64, (2,))

    result = Graph(model).execute()

    assert result.schema["prediction"] == pl.Array(pl.Float64, 2)
    assert result.height == 12


def test_model_rejects_configured_width_mismatch() -> None:
    src = Node(VectorSource)

    with pytest.raises(ValueError, match="does not match the bound width"):
        Node(
            DenseMLP,
            bindings={"features": src.features, "target": src.target},
            parameters={
                "feature_width": 2,
                "scheduler": {"every": 4},
                "splitter": {"test_size": 2},
            },
        )


def test_lightgbm_fit_is_deterministic_with_serializable_booster() -> None:
    pytest.importorskip("lightgbm")
    frame = regression_frame(target_width=1)
    initial = LightGBMModel(
        feature_width=2,
        target_width=1,
        num_boost_round=30,
        learning_rate=0.1,
        min_data_in_leaf=1,
    )

    first = initial.fit(dataset_split(target_width=1), seed=11)
    repeated = initial.fit(dataset_split(target_width=1), seed=11)

    assert isinstance(first, LightGBMModel)
    assert first is not initial
    assert first.model_text is not None
    assert first.model_text == repeated.model_text
    assert initial.model_text is None
    assert mse(frame["target"], first.predict(frame["features"])) < 10.0
    assert json.loads(str(first))["state"]["model_text"].startswith("tree")


def test_lightgbm_rejects_multi_output_targets() -> None:
    with pytest.raises(ValueError, match="single output"):
        LightGBMModel(
            feature_width=2,
            target_width=2,
            num_boost_round=2,
        )


def test_lightgbm_tsfn_declares_widths_and_mse_metric() -> None:
    function = LightGBM(
        {
            "feature_width": 3,
            "target_width": 1,
            "scheduler": {"every": 4},
            "splitter": {"test_size": 3},
            "num_boost_round": 2,
        }
    )

    input_signature, output_signature = function.signature
    assert input_signature.columns == (
        ("features", pl.Float64, (3,)),
        ("target", pl.Float64, (1,)),
    )
    assert output_signature.columns == (("prediction", pl.Float64, (1,)),)
    assert function.parameters.scheduler == EveryNTicksScheduler(4)
    assert function.parameters.splitter == ChronologicalSplitter(test_size=3)

    initial = function.initial_model()
    prediction = initial.predict(
        pl.Series(
            "features",
            [[1.0, 2.0, 3.0]],
            dtype=pl.Array(pl.Float64, 3),
        )
    )
    assert prediction.to_list() == [[0.0]]
    assert function.segment_metrics(
        pl.Series([1.0, 3.0]),
        pl.Series([2.0, 1.0]),
    ) == {"mse": 2.5}


def test_scheduler_declarations_normalize_to_objects() -> None:
    default = EveryNTicksScheduler(5)
    assert scheduler_from_declaration(None, default=default) == default
    assert scheduler_from_declaration(
        {"every": 10}, default=default
    ) == EveryNTicksScheduler(10)
    assert scheduler_from_declaration(
        {"frozen": True}, default=default
    ) == FrozenScheduler()
    assert scheduler_from_declaration(
        {"metric": {"name": "mse", "threshold": 0.5, "check_every": 25}},
        default=default,
    ) == MetricThresholdScheduler("mse", 0.5, 25)
    assert scheduler_from_declaration(
        {"any": [{"every": 2}, {"frozen": True}]},
        default=default,
    ) == AnyScheduler((EveryNTicksScheduler(2), FrozenScheduler()))
    assert scheduler_from_declaration(
        EveryNTicksScheduler(3), default=default
    ) == EveryNTicksScheduler(3)


def test_scheduler_declarations_reject_unknown_keys_and_conflicts() -> None:
    default = EveryNTicksScheduler(5)
    with pytest.raises(ValueError, match="Unexpected keys"):
        scheduler_from_declaration({"every": 2, "bogus": 1}, default=default)
    with pytest.raises(ValueError, match="Unexpected keys"):
        scheduler_from_declaration({"frozen": True, "every": 2}, default=default)
    with pytest.raises(ValueError, match="exactly one of"):
        scheduler_from_declaration({"bogus": 1}, default=default)
    with pytest.raises(ValueError, match="Unexpected keys"):
        scheduler_from_declaration(
            {"metric": {"name": "mse", "threshold": 0.5, "check_every": 1, "bogus": 1}},
            default=default,
        )


def test_splitter_declarations_normalize_to_objects() -> None:
    default = ChronologicalSplitter(validation_size=0.2)
    assert splitter_from_declaration(None, default=default) == default
    assert splitter_from_declaration(
        {"validation_size": 0.2, "gap": 1},
        default=ChronologicalSplitter(),
    ) == ChronologicalSplitter(validation_size=0.2, gap=1)
    assert splitter_from_declaration(
        ChronologicalSplitter(test_size=2),
        default=ChronologicalSplitter(),
    ) == ChronologicalSplitter(test_size=2)
    with pytest.raises(ValueError, match="Unexpected keys"):
        splitter_from_declaration(
            {"validation_size": 0.2, "bogus": 1},
            default=ChronologicalSplitter(),
        )


def test_declarative_and_object_forms_have_identical_identity() -> None:
    declarative = DenseMLP(
        {
            "feature_width": 2,
            "target_width": 1,
            "scheduler": {"every": 100},
            "splitter": {"validation_size": 0.2},
        }
    )
    explicit = DenseMLP(
        {
            "feature_width": 2,
            "target_width": 1,
            "scheduler": EveryNTicksScheduler(100),
            "splitter": ChronologicalSplitter(validation_size=0.2),
        }
    )

    assert str(declarative) == str(explicit)
    assert declarative.parameters.to_dict() == explicit.parameters.to_dict()
