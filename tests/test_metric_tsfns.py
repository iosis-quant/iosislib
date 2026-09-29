from __future__ import annotations

import hashlib
import statistics
from datetime import datetime, timedelta

import polars as pl
import pytest

from iosislib.metrics import (
    Mae,
    MaxDrawdown,
    MetricConfig,
    MetricTSFN,
    Mse,
    Sharpe,
    TotalReturn,
)
from iosislib.strategy import builtin_registry, loads, lower

TIMESTAMPS = [datetime(2026, 1, 1) + timedelta(days=index) for index in range(6)]

METRIC_OPS = {
    ("metrics.mae", "1.0.0"),
    ("metrics.max_drawdown", "1.0.0"),
    ("metrics.mse", "1.0.0"),
    ("metrics.sharpe", "1.0.0"),
    ("metrics.total_return", "1.0.0"),
}


def _frame(columns: dict[str, list[float]], *, time_name: str = "timestamp") -> pl.DataFrame:
    height = len(next(iter(columns.values())))
    return pl.DataFrame(
        {time_name: TIMESTAMPS[:height], **columns},
        schema={time_name: pl.Datetime, **{name: pl.Float64 for name in columns}},
    )


def _run(metric: MetricTSFN, frame: pl.DataFrame) -> pl.DataFrame:
    return metric(frame.lazy()).collect()


def test_registry_registers_exactly_the_five_metric_operations() -> None:
    registry = builtin_registry()
    registered = {key for key in registry.operations if key[0].startswith("metrics.")}
    assert registered == METRIC_OPS


def test_registry_skips_abstract_metric_tsfn() -> None:
    registry = builtin_registry()
    assert not any(key[0] == "metrics.metric_tsfn" for key in registry.operations)


def test_metric_config_rejects_invalid_values() -> None:
    with pytest.raises(TypeError, match="timestamp_column must be a string"):
        MetricConfig(timestamp_column=3)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="timestamp_column must be non-empty"):
        MetricConfig(timestamp_column="")
    with pytest.raises(TypeError, match="drop_nonfinite must be a boolean"):
        MetricConfig(drop_nonfinite="yes")  # type: ignore[arg-type]


def test_mse_matches_manual_computation() -> None:
    frame = _frame({"prediction": [1.0, 2.0, 3.0], "target": [1.5, 2.0, 4.5]})
    result = _run(Mse({}), frame)
    assert result.columns == ["timestamp", "mse"]
    assert result.height == 1
    assert result["mse"][0] == pytest.approx(0.8333333333333334)
    assert result.schema["mse"] == pl.Float64


def test_mae_matches_manual_computation() -> None:
    frame = _frame({"prediction": [1.0, 2.0, 3.0], "target": [1.5, 2.0, 4.5]})
    result = _run(Mae({}), frame)
    assert result.columns == ["timestamp", "mae"]
    assert result["mae"][0] == pytest.approx(2.0 / 3.0)


def test_max_drawdown_measures_peak_relative_decline() -> None:
    frame = _frame({"equity": [100.0, 120.0, 90.0, 110.0]})
    result = _run(MaxDrawdown({}), frame)
    assert result.columns == ["timestamp", "max_drawdown"]
    assert result["max_drawdown"][0] == pytest.approx(0.25)


def test_max_drawdown_of_rising_curve_is_zero() -> None:
    frame = _frame({"equity": [100.0, 101.0, 102.0, 110.0]})
    result = _run(MaxDrawdown({}), frame)
    assert result["max_drawdown"][0] == 0.0


def test_max_drawdown_measures_decline_of_dust_scale_equity() -> None:
    frame = _frame({"equity": [1e-13, 1e-14, 1e-15]})
    result = _run(MaxDrawdown({}), frame)
    assert result["max_drawdown"][0] == pytest.approx(0.99)


@pytest.mark.parametrize(
    "equity",
    ([0.0, -10.0, -20.0], [-100.0, -200.0, -400.0]),
)
def test_max_drawdown_rejects_a_curve_that_never_has_a_positive_peak(
    equity: list[float],
) -> None:
    frame = _frame({"equity": equity})
    with pytest.raises(ValueError, match="positive running peak"):
        _run(MaxDrawdown({}), frame)


def test_sharpe_uses_sample_standard_deviation() -> None:
    returns = [0.01, -0.02, 0.03]
    frame = _frame({"returns": returns})
    result = _run(Sharpe({}), frame)
    expected = statistics.fmean(returns) / statistics.stdev(returns)
    assert result.columns == ["timestamp", "sharpe"]
    assert result["sharpe"][0] == pytest.approx(expected)


def test_sharpe_rejects_zero_variance() -> None:
    frame = _frame({"returns": [0.05, 0.05, 0.05]})
    with pytest.raises(ValueError, match="non-zero standard deviation"):
        _run(Sharpe({}), frame)


def test_sharpe_rejects_standard_deviation_that_underflows_to_zero() -> None:
    frame = _frame({"returns": [1e-200, 2e-200, 3e-200]})
    with pytest.raises(ValueError, match="non-zero standard deviation"):
        _run(Sharpe({}), frame)


def test_total_return_is_last_over_first_minus_one() -> None:
    frame = _frame({"equity": [100.0, 90.0, 105.0]})
    result = _run(TotalReturn({}), frame)
    assert result.columns == ["timestamp", "total_return"]
    assert result["total_return"][0] == pytest.approx(0.05)


def test_total_return_rejects_zero_first_value() -> None:
    frame = _frame({"equity": [0.0, 10.0]})
    with pytest.raises(ValueError, match="positive first equity value"):
        _run(TotalReturn({}), frame)


def test_total_return_rejects_negative_first_value() -> None:
    frame = _frame({"equity": [-100.0, -200.0]})
    with pytest.raises(ValueError, match="positive first equity value"):
        _run(TotalReturn({}), frame)


def test_output_row_is_stamped_with_the_last_sorted_timestamp() -> None:
    frame = _frame({"equity": [100.0, 90.0, 120.0]})
    shuffled = frame.reverse()
    result = _run(TotalReturn({}), shuffled)
    assert result["timestamp"].to_list() == [TIMESTAMPS[2]]


def test_output_row_keeps_the_last_input_timestamp_when_rows_are_dropped() -> None:
    frame = _frame(
        {
            "prediction": [1.0, 2.0, 3.0, float("nan")],
            "target": [1.0, 2.0, 3.0, 99.0],
        }
    )
    result = _run(Mse({"drop_nonfinite": True}), frame)
    assert result["timestamp"].to_list() == [TIMESTAMPS[3]]
    assert result["mse"][0] == pytest.approx(0.0)


def test_equal_timestamps_keep_their_input_order() -> None:
    frame = pl.DataFrame(
        {
            "timestamp": [TIMESTAMPS[0], TIMESTAMPS[1], TIMESTAMPS[1]],
            "equity": [100.0, 90.0, 120.0],
        },
        schema={"timestamp": pl.Datetime, "equity": pl.Float64},
    )
    assert _run(TotalReturn({}), frame)["total_return"][0] == pytest.approx(0.2)
    assert _run(TotalReturn({}), frame.reverse())["total_return"][0] == pytest.approx(
        -0.1
    )


def test_nonfinite_inputs_raise_with_per_column_counts() -> None:
    frame = _frame(
        {"prediction": [1.0, float("nan"), 3.0], "target": [1.0, 2.0, float("inf")]}
    )
    with pytest.raises(ValueError, match=r"non-finite input value\(s\)") as error:
        _run(Mse({}), frame)
    message = str(error.value)
    assert "prediction: 1" in message
    assert "target: 1" in message
    assert "drop_nonfinite" in message


def test_drop_nonfinite_filters_rows_before_the_reduction() -> None:
    frame = _frame(
        {"prediction": [1.0, float("nan"), 3.0], "target": [1.5, 2.0, 3.5]}
    )
    metric = Mse({"drop_nonfinite": True})
    result = _run(metric, frame)
    assert result["mse"][0] == pytest.approx(0.25)


def test_drop_nonfinite_still_enforces_minimum_rows() -> None:
    frame = _frame({"prediction": [float("nan")], "target": [1.0]})
    metric = Mse({"drop_nonfinite": True})
    with pytest.raises(ValueError, match="requires at least 1 row"):
        _run(metric, frame)


def test_minimum_rows_are_enforced() -> None:
    empty = _frame({"prediction": [], "target": []})
    with pytest.raises(ValueError, match="requires at least 1 row"):
        _run(Mse({}), empty)
    one_row = _frame({"equity": [100.0]})
    with pytest.raises(ValueError, match=r"requires at least 2 row"):
        _run(MaxDrawdown({}), one_row)


def test_null_inputs_fail_under_the_default_error_policy() -> None:
    frame = _frame({"prediction": [1.0, None, 3.0], "target": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="NullPolicy.ERROR failed for column 'prediction'"):
        _run(Mse({}), frame)


def test_batch_treats_surviving_nulls_as_nonfinite() -> None:
    frame = _frame({"prediction": [1.0, None], "target": [1.0, 2.0]})
    with pytest.raises(ValueError, match=r"non-finite input value\(s\)"):
        Mse({}).batch(frame)
    dropped = Mse({"drop_nonfinite": True}).batch(frame)
    assert dropped.height == 1


def test_custom_timestamp_column_is_used_for_input_and_output() -> None:
    frame = _frame({"equity": [100.0, 90.0]}, time_name="ts")
    metric = TotalReturn({"timestamp_column": "ts"})
    result = _run(metric, frame)
    assert result.columns == ["ts", "total_return"]
    assert result["ts"].to_list() == [TIMESTAMPS[1]]


def test_metric_name_must_differ_from_the_timestamp_column() -> None:
    with pytest.raises(ValueError, match="must differ from the timestamp column"):
        Mse({"timestamp_column": "mse"})


def test_input_column_must_differ_from_the_timestamp_column() -> None:
    with pytest.raises(ValueError, match="input column 'returns' must differ"):
        Sharpe({"timestamp_column": "returns"})


def test_invalid_metric_classvar_contracts_are_rejected() -> None:
    class EmptyName(MetricTSFN):
        VERSION = "1.0.0"
        METRIC_NAME = ""
        INPUT_COLUMNS = ("value",)

        def metric_value(self, frame: pl.DataFrame) -> float:
            return 0.0

    with pytest.raises(ValueError, match="METRIC_NAME must be non-empty"):
        EmptyName({})

    class NamedWrong(MetricTSFN):
        VERSION = "1.0.0"
        METRIC_NAME = 3  # type: ignore[assignment]
        INPUT_COLUMNS = ("value",)

        def metric_value(self, frame: pl.DataFrame) -> float:
            return 0.0

    with pytest.raises(TypeError, match="METRIC_NAME must be a string"):
        NamedWrong({})

    class DuplicateInputs(MetricTSFN):
        VERSION = "1.0.0"
        METRIC_NAME = "dup"
        INPUT_COLUMNS = ("value", "value")

        def metric_value(self, frame: pl.DataFrame) -> float:
            return 0.0

    with pytest.raises(ValueError, match="contains duplicates"):
        DuplicateInputs({})

    class ColumnNamesWrong(MetricTSFN):
        VERSION = "1.0.0"
        METRIC_NAME = "cols"
        INPUT_COLUMNS = "value"  # type: ignore[assignment]

        def metric_value(self, frame: pl.DataFrame) -> float:
            return 0.0

    with pytest.raises(TypeError, match="INPUT_COLUMNS must be a tuple"):
        ColumnNamesWrong({})

    class BadMinRows(MetricTSFN):
        VERSION = "1.0.0"
        METRIC_NAME = "rows"
        INPUT_COLUMNS = ("value",)
        MIN_ROWS = 0

        def metric_value(self, frame: pl.DataFrame) -> float:
            return 0.0

    with pytest.raises(ValueError, match="MIN_ROWS must be a positive integer"):
        BadMinRows({})

    class MinRowsWrong(MetricTSFN):
        VERSION = "1.0.0"
        METRIC_NAME = "rows"
        INPUT_COLUMNS = ("value",)
        MIN_ROWS = "2"  # type: ignore[assignment]

        def metric_value(self, frame: pl.DataFrame) -> float:
            return 0.0

    with pytest.raises(TypeError, match="MIN_ROWS must be an integer"):
        MinRowsWrong({})


def test_metric_value_must_return_a_finite_number() -> None:
    class Text(MetricTSFN):
        VERSION = "1.0.0"
        METRIC_NAME = "text"
        INPUT_COLUMNS = ("value",)

        def metric_value(self, frame: pl.DataFrame) -> float:
            return "nope"  # type: ignore[return-value]

    frame = _frame({"value": [1.0]})
    with pytest.raises(TypeError, match="must return a float"):
        Text({}).batch(frame)

    class Infinite(MetricTSFN):
        VERSION = "1.0.0"
        METRIC_NAME = "infinite"
        INPUT_COLUMNS = ("value",)

        def metric_value(self, frame: pl.DataFrame) -> float:
            return float("nan")

    with pytest.raises(ValueError, match="produced a non-finite value"):
        Infinite({}).batch(frame)


def test_metadata_metrics_is_rejected_by_strategy_parsing() -> None:
    yaml = """
format: iosis.strategy
version: 0.1.0
name: legacy-metrics
nodes:
  change:
    op: transform.delta
    version: 0.2.0
    params:
      output_column: change
outputs:
  signal: change.change
metadata:
  metrics:
    - name: mse
      column: change
"""
    with pytest.raises(ValueError, match="metadata.metrics is no longer supported"):
        loads(yaml)


def test_metric_node_executes_inside_a_strategy(tmp_path) -> None:
    csv = b"ts,probability\n2026-01-01,0.4\n2026-01-02,0.45\n2026-01-03,0.6\n"
    path = tmp_path / "prices.csv"
    path.write_bytes(csv)
    digest = hashlib.sha256(csv).hexdigest()
    yaml = f"""
format: iosis.strategy
version: 0.1.0
name: metric-e2e
nodes:
  prices:
    op: source.csv_source
    version: 0.2.0
    params:
      path: {path}
      content_sha256: {digest}
      schema:
        time: ts
        columns:
          probability: float64
  change:
    op: transform.delta
    version: 0.2.0
    inputs:
      value: prices.probability
    params:
      timestamp_column: ts
      output_column: change
  mse:
    op: metrics.mse
    version: 1.0.0
    inputs:
      prediction: prices.probability
      target:
        from: change.change
        nulls: drop
    params:
      timestamp_column: ts
      drop_nonfinite: true
outputs:
  signal: change.change
  mse: mse.mse
"""
    lowered = lower(loads(yaml), builtin_registry())
    assert set(lowered.outputs) == {"signal", "mse"}
    result = lowered.graph("mse").execute()
    assert result.columns == ["ts", "mse"]
    assert result.height == 1
    assert result["mse"][0] == pytest.approx(0.18125)


def test_metric_node_with_mismatched_time_column_fails_graph_validation() -> None:
    yaml = """
format: iosis.strategy
version: 0.1.0
name: time-mismatch
nodes:
  change:
    op: transform.delta
    version: 0.2.0
    params:
      output_column: change
  sharpe:
    op: metrics.sharpe
    version: 1.0.0
    inputs:
      returns: change.change
    params:
      timestamp_column: ts
outputs:
  sharpe: sharpe.sharpe
"""
    strategy = loads(yaml)
    with pytest.raises(ValueError, match="Time axis mismatch"):
        lower(strategy, builtin_registry()).graph("sharpe")
