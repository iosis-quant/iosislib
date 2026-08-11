from __future__ import annotations

import math

import polars as pl
import pytest

from iosislib.metrics import (
    MaxDrawdown,
    MeanSquaredError,
    MetricExtractor,
    extract_metrics,
)


def _prediction_target() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "prediction": [1.0, 2.0, 3.0],
            "target": [1.0, 3.0, 3.0],
        }
    )


def test_mse_known_value() -> None:
    result = extract_metrics(_prediction_target(), MeanSquaredError())
    assert result.to_dicts() == [{"mse": 1.0 / 3.0}]


def test_mse_supports_integer_and_float_columns() -> None:
    frame = pl.DataFrame({"signal": [1, 2, 3], "answer": [1, 3, 3]})
    result = extract_metrics(
        frame,
        MeanSquaredError(prediction_column="signal", target_column="answer"),
    )
    assert result.to_dicts() == [{"mse": 1.0 / 3.0}]


def test_mse_requires_at_least_one_row() -> None:
    frame = pl.DataFrame(schema={"prediction": pl.Float64, "target": pl.Float64})
    with pytest.raises(ValueError, match="at least one row"):
        MeanSquaredError().extract(frame)


def test_mse_missing_column_raises() -> None:
    frame = pl.DataFrame({"prediction": [1.0], "answer": [2.0]})
    with pytest.raises(ValueError, match="missing required column"):
        MeanSquaredError().extract(frame)


def test_mse_null_values_raise() -> None:
    frame = pl.DataFrame({"prediction": [1.0, None], "target": [1.0, 2.0]})
    with pytest.raises(ValueError, match="contains null"):
        MeanSquaredError().extract(frame)


def test_mse_non_numeric_column_raises() -> None:
    frame = pl.DataFrame({"prediction": ["a", "b"], "target": [1.0, 2.0]})
    with pytest.raises(TypeError, match="must be numeric"):
        MeanSquaredError().extract(frame)


def test_mse_non_finite_values_raise() -> None:
    frame = pl.DataFrame(
        {"prediction": [1.0, float("inf")], "target": [1.0, 2.0]}
    )
    with pytest.raises(ValueError, match="finite"):
        MeanSquaredError().extract(frame)


def test_mse_rejects_equal_column_names() -> None:
    with pytest.raises(ValueError, match="must differ"):
        MeanSquaredError(
            prediction_column="value", target_column="value"
        )


def test_max_drawdown_known_value() -> None:
    frame = pl.DataFrame({"equity": [100.0, 120.0, 90.0, 110.0]})
    result = extract_metrics(frame, MaxDrawdown())
    assert result.to_dicts() == [{"max_drawdown": 0.25}]


def test_max_drawdown_monotonic_equity_is_zero() -> None:
    frame = pl.DataFrame({"equity": [1.0, 2.0, 3.0, 4.0]})
    assert extract_metrics(frame, MaxDrawdown()).to_dicts() == [
        {"max_drawdown": 0.0}
    ]


def test_max_drawdown_ignores_non_positive_running_peaks() -> None:
    frame = pl.DataFrame({"equity": [0.0, -5.0, -3.0, 1.0, 0.5]})
    result = extract_metrics(frame, MaxDrawdown())
    assert result.to_dicts() == [{"max_drawdown": 0.5}]


def test_max_drawdown_requires_two_rows() -> None:
    frame = pl.DataFrame({"equity": [100.0]})
    with pytest.raises(ValueError, match="at least two rows"):
        MaxDrawdown().extract(frame)


def test_max_drawdown_missing_column_raises() -> None:
    frame = pl.DataFrame({"value": [1.0, 2.0]})
    with pytest.raises(ValueError, match="missing required column"):
        MaxDrawdown().extract(frame)


def test_max_drawdown_null_values_raise() -> None:
    frame = pl.DataFrame({"equity": [1.0, None]})
    with pytest.raises(ValueError, match="contains null"):
        MaxDrawdown().extract(frame)


def test_max_drawdown_non_numeric_column_raises() -> None:
    frame = pl.DataFrame({"equity": ["a", "b"]})
    with pytest.raises(TypeError, match="must be numeric"):
        MaxDrawdown().extract(frame)


def test_extract_metrics_combines_extractors() -> None:
    frame = pl.DataFrame(
        {
            "prediction": [1.0, 2.0],
            "target": [1.0, 4.0],
            "equity": [100.0, 90.0],
        }
    )
    result = extract_metrics(frame, MeanSquaredError(), MaxDrawdown())
    assert result.to_dicts() == [{"mse": 2.0, "max_drawdown": 0.1}]


def test_extract_metrics_resolves_columns_across_frames() -> None:
    prediction_frame = pl.DataFrame({"time": [1, 2], "prediction": [1.0, 2.0]})
    target_frame = pl.DataFrame({"time": [1, 2], "target": [1.0, 4.0]})
    result = extract_metrics(
        (prediction_frame, target_frame),
        MeanSquaredError(),
    )
    assert result.to_dicts() == [{"mse": 2.0}]


def test_extract_metrics_rejects_ragged_frames() -> None:
    prediction_frame = pl.DataFrame({"time": [1, 2], "prediction": [1.0, 2.0]})
    target_frame = pl.DataFrame({"time": [1], "target": [1.0]})
    with pytest.raises(ValueError, match="different row counts"):
        extract_metrics(
            (prediction_frame, target_frame),
            MeanSquaredError(),
        )


def test_extract_metrics_rejects_ambiguous_columns() -> None:
    first = pl.DataFrame({"equity": [1.0, 2.0], "other": [1.0, 2.0]})
    second = pl.DataFrame({"equity": [3.0, 4.0]})
    with pytest.raises(ValueError, match="ambiguous"):
        extract_metrics((first, second), MaxDrawdown())


def test_extract_metrics_all_empty_frames_return_empty_result() -> None:
    empty = pl.DataFrame(schema={"equity": pl.Float64})
    result = extract_metrics(
        empty,
        MaxDrawdown(equity_column="equity"),
        MeanSquaredError(),
    )
    assert result.shape == (0, 2)
    assert set(result.columns) == {"max_drawdown", "mse"}


def test_extract_metrics_single_empty_frame_returns_empty_result() -> None:
    empty = pl.DataFrame(schema={"prediction": pl.Float64, "target": pl.Float64})
    result = extract_metrics(empty, MeanSquaredError())
    assert result.shape == (0, 1)
    assert result.columns == ["mse"]


def test_extract_metrics_requires_extractors() -> None:
    with pytest.raises(ValueError, match="at least one extractor"):
        extract_metrics(_prediction_target())


def test_extract_metrics_rejects_non_extractor() -> None:
    with pytest.raises(TypeError, match="MetricExtractor"):
        extract_metrics(_prediction_target(), object())


def test_extract_metrics_rejects_duplicate_metric_names() -> None:
    frame = pl.DataFrame({"equity": [1.0, 2.0]})
    with pytest.raises(ValueError, match="duplicate metric name"):
        extract_metrics(
            frame,
            MaxDrawdown(equity_column="equity"),
            MaxDrawdown(equity_column="equity"),
        )


def test_extract_metrics_rejects_non_frame_input() -> None:
    with pytest.raises(TypeError, match="pl.DataFrame"):
        extract_metrics(42, MeanSquaredError())


def test_extract_metrics_rejects_empty_frame_sequence() -> None:
    with pytest.raises(ValueError, match="at least one frame"):
        extract_metrics([], MeanSquaredError())


def test_extract_metrics_is_deterministic() -> None:
    frame = _prediction_target()
    first = extract_metrics(frame, MeanSquaredError())
    second = extract_metrics(frame, MeanSquaredError())
    assert first.to_dicts() == second.to_dicts()


class _InfiniteMetric(MetricExtractor):
    VERSION = "1.0.0"

    def required_columns(self) -> tuple[str, ...]:
        return ("equity",)

    def metric_names(self) -> tuple[str, ...]:
        return ("value",)

    def extract(self, frame: pl.DataFrame) -> dict[str, float]:
        del frame
        return {"value": float("inf")}


class _TextMetric(MetricExtractor):
    VERSION = "1.0.0"

    def required_columns(self) -> tuple[str, ...]:
        return ("equity",)

    def metric_names(self) -> tuple[str, ...]:
        return ("value",)

    def extract(self, frame: pl.DataFrame) -> dict[str, float]:
        del frame
        return {"value": "not a number"}  # type: ignore[return-value]


def test_extract_metrics_rejects_non_finite_metric_values() -> None:
    frame = pl.DataFrame({"equity": [1.0, 2.0]})
    with pytest.raises(ValueError, match="finite"):
        extract_metrics(frame, _InfiniteMetric())


def test_extract_metrics_rejects_non_numeric_metric_values() -> None:
    frame = pl.DataFrame({"equity": [1.0, 2.0]})
    with pytest.raises(TypeError, match="must be numeric"):
        extract_metrics(frame, _TextMetric())


def test_metric_names_and_required_columns_contracts() -> None:
    assert MeanSquaredError().required_columns() == ("prediction", "target")
    assert MeanSquaredError().metric_names() == ("mse",)
    assert MaxDrawdown().required_columns() == ("equity",)
    assert MaxDrawdown().metric_names() == ("max_drawdown",)


def test_extractor_to_dict_is_stable() -> None:
    first = MeanSquaredError(prediction_column="signal", target_column="answer")
    second = MeanSquaredError(prediction_column="signal", target_column="answer")
    assert first.to_dict() == second.to_dict()
    assert first.to_dict() == {
        "type": "iosislib.metrics.extractors.MeanSquaredError",
        "version": "1.0.0",
        "prediction_column": "signal",
        "target_column": "answer",
    }
    assert first.__str__() == str(second)


def test_extractor_string_is_canonical_json() -> None:
    import json

    parsed = json.loads(MaxDrawdown(equity_column="value").__str__())
    assert parsed["type"] == "iosislib.metrics.extractors.MaxDrawdown"
    assert parsed["equity_column"] == "value"


def test_mse_zero_rows_frame_without_schema_raises_missing_column() -> None:
    frame = pl.DataFrame(schema={"x": pl.Float64})
    with pytest.raises(ValueError, match="missing required column"):
        MeanSquaredError().extract(frame)


def test_max_drawdown_matches_manual_calculation() -> None:
    equity = [200.0, 160.0, 180.0, 150.0, 190.0]
    frame = pl.DataFrame({"equity": equity})
    peak = -math.inf
    worst = 0.0
    for value in equity:
        peak = max(peak, value)
        worst = max(worst, (peak - value) / peak)
    result = extract_metrics(frame, MaxDrawdown())
    assert result.to_dicts() == [{"max_drawdown": worst}]
