from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from iosislib.core.graph import Graph
from iosislib.core.model import (
    ChronologicalSplitter,
    DatasetSplit,
    EveryNTicksScheduler,
    FrameDataset,
    model_from_dict,
)
from iosislib.core.node import Node
from iosislib.core.tsfn import FrameSignature, TSFN, TSFNConfig, TimeAxis
from iosislib.models import (
    LinearRegression,
    LinearRegressionConfig,
    LinearRegressionModel,
    LogisticRegression,
    LogisticRegressionConfig,
    LogisticRegressionModel,
)
from iosislib.strategy import builtin_registry
from _floats import assert_lists_close


def linear_frame(rows: int = 20) -> pl.DataFrame:
    features = [[float(index), float(index % 3)] for index in range(rows)]
    target = [2.0 * row[0] - row[1] + 1.0 for row in features]
    return pl.DataFrame(
        {
            "features": pl.Series("features", features, dtype=pl.Array(pl.Float64, 2)),
            "target": pl.Series("target", target, dtype=pl.Float64),
        }
    )


def binary_frame(rows: int = 40) -> pl.DataFrame:
    features = [[float(index), float(index % 3)] for index in range(rows)]
    target = [
        1.0 if row[0] + row[1] > 15.5 else 0.0
        for row in features
    ]
    return pl.DataFrame(
        {
            "features": pl.Series("features", features, dtype=pl.Array(pl.Float64, 2)),
            "target": pl.Series("target", target, dtype=pl.Float64),
        }
    )


def linear_split() -> DatasetSplit:
    return DatasetSplit(
        FrameDataset(linear_frame(), batch_size=5, shuffle=True)
    )


def binary_split() -> DatasetSplit:
    return DatasetSplit(
        FrameDataset(binary_frame(), batch_size=8, shuffle=True)
    )


def mse(target: pl.Series, prediction: pl.Series) -> float:
    target_values = np.asarray(target.to_list(), dtype=np.float64).reshape(-1)
    prediction_values = np.asarray(prediction.to_list(), dtype=np.float64).reshape(-1)
    return float(np.mean((target_values - prediction_values) ** 2))


def column_shapes(
    signature: FrameSignature,
) -> dict[str, tuple[pl.DataType, tuple[int, ...]]]:
    return {
        entry[0]: (entry[1], entry[2] if len(entry) == 3 else ())
        for entry in signature.columns
    }


@dataclass(frozen=True)
class SeriesConfig(TSFNConfig):
    values: tuple[float, ...]
    outcomes: tuple[float, ...]


class SeriesSource(TSFN):
    VERSION = "1.0.0"
    CONFIG_CLS = SeriesConfig

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        return (
            FrameSignature.empty(),
            FrameSignature(
                time=TimeAxis(column="timestamp"),
                columns=(
                    ("features", pl.Float64),
                    ("target", pl.Float64),
                ),
            ),
        )

    def apply(self) -> pl.LazyFrame:
        count = len(self.parameters.values)
        return pl.DataFrame(
            {
                "timestamp": [
                    datetime(2026, 1, 1) + timedelta(hours=index)
                    for index in range(count)
                ],
                "features": list(self.parameters.values),
                "target": list(self.parameters.outcomes),
            },
            schema={
                "timestamp": pl.Datetime,
                "features": pl.Float64,
                "target": pl.Float64,
            },
        ).lazy()


LINEAR_VALUES = tuple(float(index) for index in range(12))
LINEAR_TARGETS = tuple(2.0 * value + 1.0 for value in LINEAR_VALUES)
BINARY_TARGETS = tuple(float(index % 2) for index in range(12))


def series_source(values: tuple[float, ...], outcomes: tuple[float, ...]) -> Node:
    return Node(SeriesSource, parameters={"values": values, "outcomes": outcomes})


def walk_forward_graph(
    tsfn: type,
    values: tuple[float, ...],
    outcomes: tuple[float, ...],
    *,
    parameters: dict[str, object] | None = None,
) -> Graph:
    src = series_source(values, outcomes)
    model = Node(
        tsfn,
        bindings={"features": src.features, "target": src.target},
        parameters={"scheduler": {"every": 4}} if parameters is None else parameters,
        name="walk_forward_model",
    )
    return Graph(model)


def test_linear_regression_recovers_known_coefficients() -> None:
    frame = linear_frame()
    initial = LinearRegressionModel(feature_width=2, target_width=1)

    trained = initial.fit(linear_split(), seed=0)

    assert trained.is_trained is True
    assert initial.is_trained is False
    assert initial.coefficients is None
    assert_lists_close(list(trained.coefficients), [2.0, -1.0])
    assert trained.intercept == pytest.approx(1.0)
    prediction = trained.predict(frame.get_column("features"))
    assert prediction.dtype == pl.Array(pl.Float64, 1)
    assert mse(frame.get_column("target"), prediction) < 1e-18


def test_linear_regression_fit_is_deterministic_and_immutable() -> None:
    initial = LinearRegressionModel(feature_width=2, target_width=1)

    first = initial.fit(linear_split(), seed=7)
    repeated = initial.fit(linear_split(), seed=7)

    assert first is not initial
    assert first.coefficients == repeated.coefficients
    assert first.intercept == repeated.intercept
    assert initial.coefficients is None


def test_linear_regression_ridge_shrinks_coefficients() -> None:
    plain = LinearRegressionModel(feature_width=2, target_width=1).fit(
        linear_split(), seed=0
    )
    ridge = LinearRegressionModel(feature_width=2, target_width=1, l2=1000.0).fit(
        linear_split(), seed=0
    )

    assert abs(ridge.coefficients[0]) < abs(plain.coefficients[0])
    assert abs(ridge.coefficients[1]) < abs(plain.coefficients[1])


def test_logistic_regression_learns_probability_direction() -> None:
    frame = binary_frame()
    initial = LogisticRegressionModel(feature_width=2, target_width=1)

    trained = initial.fit(binary_split(), seed=0)

    assert trained.is_trained is True
    assert initial.coefficients is None
    probabilities = np.asarray(
        trained.predict(frame.get_column("features")).to_list(), dtype=np.float64
    ).reshape(-1)
    labels = frame.get_column("target").to_list()
    assert probabilities.min() >= 0.0
    assert probabilities.max() <= 1.0
    assert float(np.mean((probabilities > 0.5) == np.asarray(labels, dtype=bool))) > 0.95

    lower = trained.predict(
        pl.Series("features", [[5.0, 1.0]], dtype=pl.Array(pl.Float64, 2))
    )
    higher = trained.predict(
        pl.Series("features", [[15.0, 1.0]], dtype=pl.Array(pl.Float64, 2))
    )
    assert lower.to_list()[0][0] < higher.to_list()[0][0]


def test_logistic_regression_rejects_non_binary_targets() -> None:
    frame = binary_frame(10).with_columns(
        pl.Series(
            "target",
            [0.0, 1.0, 2.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        )
    )
    split = DatasetSplit(FrameDataset(frame, batch_size=10, shuffle=False))

    with pytest.raises(ValueError, match="only 0 and 1"):
        LogisticRegressionModel(feature_width=2, target_width=1).fit(split, seed=0)


def test_logistic_regression_fit_is_deterministic_and_immutable() -> None:
    initial = LogisticRegressionModel(feature_width=2, target_width=1)

    first = initial.fit(binary_split(), seed=5)
    repeated = initial.fit(binary_split(), seed=5)

    assert first is not initial
    assert first.coefficients == repeated.coefficients
    assert first.intercept == repeated.intercept
    assert initial.coefficients is None


def test_untrained_checkpoints_emit_nan_warmup_predictions() -> None:
    features = pl.Series(
        "features",
        [[1.0, 2.0], [3.0, 4.0]],
        dtype=pl.Array(pl.Float64, 2),
    )
    for cls in (LinearRegressionModel, LogisticRegressionModel):
        model = cls(feature_width=2, target_width=1)
        prediction = model.predict(features)
        assert prediction.dtype == pl.Array(pl.Float64, 1)
        assert all(math.isnan(row[0]) for row in prediction.to_list())


def test_fitted_checkpoint_rejects_wrong_feature_width() -> None:
    fitted = LinearRegressionModel(feature_width=2, target_width=1).fit(
        linear_split(), seed=0
    )

    with pytest.raises(ValueError, match="Expected 2 features"):
        fitted.predict(
            pl.Series("features", [[1.0]], dtype=pl.Array(pl.Float64, 1))
        )


def test_linear_regression_rejects_nonfinite_training_data() -> None:
    frame = linear_frame(5).with_columns(
        pl.Series("target", [float("nan"), 1.0, 2.0, 3.0, 4.0])
    )
    split = DatasetSplit(FrameDataset(frame, batch_size=5, shuffle=False))

    with pytest.raises(ValueError, match="finite values"):
        LinearRegressionModel(feature_width=2, target_width=1).fit(split, seed=0)


def test_logistic_regression_rejects_nonfinite_training_data() -> None:
    frame = binary_frame(5).with_columns(
        pl.Series("target", [float("nan"), 1.0, 0.0, 1.0, 0.0])
    )
    split = DatasetSplit(FrameDataset(frame, batch_size=5, shuffle=False))

    with pytest.raises(ValueError, match="finite values"):
        LogisticRegressionModel(feature_width=2, target_width=1).fit(split, seed=0)


def test_checkpoints_roundtrip_through_json() -> None:
    linear = LinearRegressionModel(feature_width=2, target_width=1).fit(
        linear_split(), seed=0
    )
    payload = json.loads(str(linear))
    assert payload["state"]["coefficients"] == pytest.approx([2.0, -1.0])
    assert payload["state"]["intercept"] == pytest.approx(1.0)
    restored = model_from_dict(payload)
    assert isinstance(restored, LinearRegressionModel)
    assert restored.coefficients == linear.coefficients
    assert restored.intercept == linear.intercept

    logistic = LogisticRegressionModel(feature_width=2, target_width=1).fit(
        binary_split(), seed=0
    )
    restored_logistic = model_from_dict(json.loads(str(logistic)))
    assert isinstance(restored_logistic, LogisticRegressionModel)
    assert restored_logistic.coefficients == logistic.coefficients
    assert restored_logistic.max_iter == logistic.max_iter


def test_regression_models_reject_invalid_checkpoint_fields() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        LinearRegressionModel(feature_width=0, target_width=1)
    with pytest.raises(ValueError, match="single output"):
        LinearRegressionModel(feature_width=2, target_width=2)
    with pytest.raises(ValueError, match="l2 must be finite and non-negative"):
        LinearRegressionModel(feature_width=2, target_width=1, l2=-1.0)
    with pytest.raises(ValueError, match="exactly feature_width values"):
        LinearRegressionModel(feature_width=2, target_width=1, coefficients=(1.0,))
    with pytest.raises(ValueError, match="coefficients must be finite"):
        LinearRegressionModel(
            feature_width=2,
            target_width=1,
            coefficients=(float("inf"), 1.0),
        )
    with pytest.raises(ValueError, match="single output"):
        LogisticRegressionModel(feature_width=2, target_width=2)
    with pytest.raises(ValueError, match="max_iter must be a positive integer"):
        LogisticRegressionModel(feature_width=2, target_width=1, max_iter=0)
    with pytest.raises(ValueError, match="tol must be finite and positive"):
        LogisticRegressionModel(feature_width=2, target_width=1, tol=0.0)


def test_regression_configs_reject_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="single output"):
        LinearRegressionConfig(feature_width=2, target_width=2)
    with pytest.raises(ValueError, match="l2 must be finite and non-negative"):
        LinearRegressionConfig(feature_width=2, target_width=1, l2=-1.0)
    with pytest.raises(TypeError, match="seed must be an integer"):
        LinearRegressionConfig(seed=True)
    with pytest.raises(ValueError, match="positive integer or None"):
        LinearRegressionConfig(feature_width=0)
    with pytest.raises(ValueError, match="single output"):
        LogisticRegressionConfig(feature_width=2, target_width=2)
    with pytest.raises(ValueError, match="max_iter must be a positive integer"):
        LogisticRegressionConfig(feature_width=2, target_width=1, max_iter=0)
    with pytest.raises(ValueError, match="tol must be finite and positive"):
        LogisticRegressionConfig(feature_width=2, target_width=1, tol=0.0)
    with pytest.raises(ValueError, match="l2 must be finite and non-negative"):
        LogisticRegressionConfig(feature_width=2, target_width=1, l2=float("inf"))


def test_regression_checkpoint_predictions_reject_nonfinite_results() -> None:
    model = LinearRegressionModel(
        feature_width=1,
        target_width=1,
        coefficients=(1e308,),
        intercept=1e308,
    )

    with pytest.raises(ValueError, match="non-finite predictions"):
        model.predict(
            pl.Series("features", [[1e308]], dtype=pl.Array(pl.Float64, 1))
        )


def test_linear_regression_tsfn_declares_widths_and_mse_metric() -> None:
    function = LinearRegression(
        {
            "feature_width": 3,
            "target_width": 1,
            "scheduler": {"every": 4},
            "splitter": {"test_size": 3},
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

    prediction = function.initial_model().predict(
        pl.Series(
            "features",
            [[1.0, 2.0, 3.0]],
            dtype=pl.Array(pl.Float64, 3),
        )
    )
    assert math.isnan(prediction.to_list()[0][0])
    assert function.segment_metrics(
        pl.Series([1.0, 3.0]),
        pl.Series([2.0, 1.0]),
    ) == {"mse": 2.5}


def test_logistic_regression_tsfn_declares_widths_and_mse_metric() -> None:
    function = LogisticRegression(
        {
            "feature_width": 2,
            "target_width": 1,
            "scheduler": {"every": 5},
            "splitter": {"validation_size": 0.25},
        }
    )

    input_signature, output_signature = function.signature
    assert input_signature.columns == (
        ("features", pl.Float64, (2,)),
        ("target", pl.Float64, (1,)),
    )
    assert output_signature.columns == (("prediction", pl.Float64, (1,)),)
    assert function.parameters.scheduler == EveryNTicksScheduler(5)
    assert function.parameters.splitter == ChronologicalSplitter(validation_size=0.25)

    prediction = function.initial_model().predict(
        pl.Series(
            "features",
            [[1.0, 2.0]],
            dtype=pl.Array(pl.Float64, 2),
        )
    )
    assert math.isnan(prediction.to_list()[0][0])
    assert function.segment_metrics(
        pl.Series([0.0, 1.0]),
        pl.Series([0.25, 0.75]),
    ) == {"mse": 0.0625}


def test_model_widths_derive_from_bound_scalar_columns() -> None:
    src = series_source(LINEAR_VALUES, LINEAR_TARGETS)
    model = Node(
        LinearRegression,
        bindings={"features": src.features, "target": src.target},
        parameters={"scheduler": {"every": 4}},
        name="linear",
    )

    input_columns = column_shapes(model.function.signature[0])
    output_columns = column_shapes(model.function.signature[1])
    assert input_columns["features"] == (pl.Float64, ())
    assert input_columns["target"] == (pl.Float64, ())
    assert output_columns["prediction"] == (pl.Float64, (1,))

    result = Graph(model).execute()

    assert result.schema["prediction"] == pl.Array(pl.Float64, 1)
    assert result.height == 12


def test_model_rejects_configured_width_mismatch() -> None:
    src = series_source(LINEAR_VALUES, LINEAR_TARGETS)

    with pytest.raises(ValueError, match="does not match the bound width"):
        Node(
            LinearRegression,
            bindings={"features": src.features, "target": src.target},
            parameters={"feature_width": 2, "scheduler": {"every": 4}},
        )


def test_linear_regression_walk_forward_warms_up_then_predicts_exactly() -> None:
    result = walk_forward_graph(
        LinearRegression, LINEAR_VALUES, LINEAR_TARGETS
    ).execute()

    assert result.columns == ["timestamp", "prediction"]
    assert result.schema["prediction"] == pl.Array(pl.Float64, 1)
    assert result.height == 12
    predictions = [row[0] for row in result["prediction"].to_list()]
    assert all(math.isnan(value) for value in predictions[:4])
    assert_lists_close(
        predictions[4:],
        [9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0, 23.0],
    )


def test_logistic_regression_walk_forward_emits_probabilities() -> None:
    result = walk_forward_graph(
        LogisticRegression, LINEAR_VALUES, BINARY_TARGETS
    ).execute()

    assert result.columns == ["timestamp", "prediction"]
    assert result.schema["prediction"] == pl.Array(pl.Float64, 1)
    assert result.height == 12
    predictions = [row[0] for row in result["prediction"].to_list()]
    assert all(math.isnan(value) for value in predictions[:4])
    assert all(0.0 <= value <= 1.0 for value in predictions[4:])


def test_walk_forward_execution_is_repeatable() -> None:
    first = walk_forward_graph(LinearRegression, LINEAR_VALUES, LINEAR_TARGETS)
    repeated = walk_forward_graph(LinearRegression, LINEAR_VALUES, LINEAR_TARGETS)

    assert first.ID == repeated.ID
    assert first.execute().equals(repeated.execute())


def test_typed_and_mapping_parameters_share_node_identity() -> None:
    src = series_source(LINEAR_VALUES, LINEAR_TARGETS)

    def build(parameters=None, config=None) -> Node:
        kwargs: dict[str, object] = {}
        if parameters is not None:
            kwargs["parameters"] = parameters
        if config is not None:
            kwargs["config"] = config
        return Node(
            LinearRegression,
            bindings={"features": src.features, "target": src.target},
            **kwargs,  # type: ignore[arg-type]
        )

    typed = build(config=LinearRegressionConfig(scheduler={"every": 4}))
    mapping = build(parameters={"scheduler": {"every": 4}})
    different = build(parameters={"scheduler": {"every": 4}, "l2": 1.0})

    assert typed.ID == mapping.ID
    assert typed.ID != different.ID
    assert typed.definition == mapping.definition


def test_regression_models_are_discovered_by_the_builtin_registry() -> None:
    registry = builtin_registry()

    assert registry.resolve("model.linear_regression", LinearRegression.VERSION) is (
        LinearRegression
    )
    assert registry.resolve("model.logistic_regression", LogisticRegression.VERSION) is (
        LogisticRegression
    )
