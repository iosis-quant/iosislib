from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import polars as pl
import pytest

from iosislib.core.graph import Graph
from iosislib.core.node import Node
from iosislib.core.tsfn import FrameSignature, NullPolicy, TSFN, TSFNConfig
from iosislib.strategy.lowering import builtin_registry
from iosislib.tsfn.transforms import (
    RollingMax,
    RollingMean,
    RollingMeanConfig,
    RollingMedian,
    RollingMin,
    RollingStd,
    RollingSum,
    RollingZScore,
)


@dataclass(frozen=True)
class RollingSeriesConfig(TSFNConfig):
    minutes: tuple[int, ...]
    values: tuple[float | None, ...]
    output_column: str = "value"


class RollingSeriesSource(TSFN):
    VERSION = "1.0.0"
    CONFIG_CLS = RollingSeriesConfig

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        return (
            FrameSignature.empty(),
            FrameSignature(columns=((self.parameters.output_column, pl.Float64),)),
        )

    def apply(self) -> pl.LazyFrame:
        return pl.DataFrame(
            {
                "timestamp": [dt(minute) for minute in self.parameters.minutes],
                self.parameters.output_column: list(self.parameters.values),
            },
            schema={
                "timestamp": pl.Datetime,
                self.parameters.output_column: pl.Float64,
            },
        ).lazy()


def dt(minute: int) -> datetime:
    return datetime(2026, 1, 1, 0, minute)


def source(values: tuple[float | None, ...]) -> Node:
    return Node(
        RollingSeriesSource,
        parameters={
            "minutes": tuple(range(len(values))),
            "values": values,
        },
    )


def collect(node: Node) -> pl.DataFrame:
    return Graph(node).execute()


def test_rolling_mean_matches_manual_window_and_respects_min_samples() -> None:
    values = (1.0, 2.0, 4.0, 8.0, None, 32.0)
    node = Node(
        RollingMean,
        bindings={"value": source(values).value},
        parameters={"periods": 3, "min_samples": 3},
    )

    result = collect(node)

    assert result["timestamp"].to_list() == [dt(i) for i in range(6)]
    assert result["rolling_mean"].to_list() == pytest.approx(
        [None, None, 7.0 / 3.0, 14.0 / 3.0, None, None]
    )


def test_rolling_family_computes_window_statistics() -> None:
    values = (1.0, 2.0, 3.0, 4.0, 5.0)
    src = source(values)
    transforms = (
        (RollingSum, [1.0, 3.0, 6.0, 9.0, 12.0]),
        (RollingMax, [1.0, 2.0, 3.0, 4.0, 5.0]),
        (RollingMin, [1.0, 1.0, 1.0, 2.0, 3.0]),
        (RollingMedian, [1.0, 1.5, 2.0, 3.0, 4.0]),
    )

    for cls, expected in transforms:
        result = collect(Node(cls, bindings={"value": src.value}))
        assert result.columns == ["timestamp", cls({}).parameters.output_column]
        assert result[result.columns[1]].to_list() == pytest.approx(expected)


def test_rolling_std_is_volatility_and_zscore_nulls_constant_windows() -> None:
    src = source((1.0, 1.0, 1.0, 2.0, 4.0))
    std = collect(Node(RollingStd, bindings={"value": src.value}))
    zscore = collect(Node(RollingZScore, bindings={"value": src.value}))

    assert std["rolling_std"].to_list()[0] is None
    assert zscore["rolling_z_score"].to_list()[1] is None
    assert zscore["rolling_z_score"].to_list()[-1] == pytest.approx(
        (4.0 - 7.0 / 3.0) / std["rolling_std"].to_list()[-1]
    )


def test_rolling_transforms_sort_before_windowing_and_never_look_ahead() -> None:
    lf = pl.DataFrame(
        {
            "timestamp": [dt(3), dt(0), dt(2), dt(1)],
            "value": [40.0, 1.0, 30.0, 20.0],
        },
        schema={"timestamp": pl.Datetime, "value": pl.Float64},
    ).lazy()

    result = RollingMean({"periods": 2, "min_samples": 2})(lf).collect()

    assert result["timestamp"].to_list() == [dt(0), dt(1), dt(2), dt(3)]
    assert result["rolling_mean"].to_list() == pytest.approx(
        [None, 10.5, 25.0, 35.0]
    )


def test_rolling_config_rejects_invalid_windows() -> None:
    with pytest.raises(TypeError, match="periods must be an integer"):
        RollingMeanConfig(periods=True)
    with pytest.raises(ValueError, match="periods must be at least 1"):
        RollingMeanConfig(periods=0)
    with pytest.raises(TypeError, match="min_samples must be an integer"):
        RollingMeanConfig(min_samples=1.5)
    with pytest.raises(ValueError, match="min_samples must be between 1 and periods"):
        RollingMeanConfig(periods=3, min_samples=4)
    with pytest.raises(ValueError, match="Duplicate column names"):
        RollingMeanConfig(input_column="timestamp")
    with pytest.raises(ValueError, match="Duplicate column names"):
        RollingMeanConfig(input_column="rolling_mean")


def test_rolling_mean_ignores_nulls_inside_the_window() -> None:
    node = Node(
        RollingMean,
        bindings={"value": source((1.0, None, 3.0, 4.0)).value},
        parameters={"periods": 3, "min_samples": 1},
    )

    result = collect(node)

    assert result["rolling_mean"].to_list() == pytest.approx(
        [1.0, 1.0, 2.0, 3.5]
    )


def test_rolling_null_policy_drop_removes_rows_before_windowing() -> None:
    node = Node(
        RollingMean,
        bindings={"value": source((1.0, None, 3.0, 4.0)).value},
        parameters={"periods": 3, "min_samples": 2},
        null_policies={"value": NullPolicy.DROP},
    )

    result = collect(node)

    assert result["timestamp"].to_list() == [dt(0), dt(2), dt(3)]
    assert result["rolling_mean"].to_list() == pytest.approx([None, 2.0, 8.0 / 3.0])


def test_rolling_typed_and_mapping_config_share_identity() -> None:
    src = source((1.0, 2.0, 3.0))
    typed = Node(
        RollingMean,
        config=RollingMeanConfig(periods=2, min_samples=1),
        bindings={"value": src.value},
    )
    mapping = Node(
        RollingMean,
        parameters={"periods": 2, "min_samples": 1},
        bindings={"value": src.value},
    )
    different = Node(
        RollingMean,
        parameters={"periods": 3, "min_samples": 1},
        bindings={"value": src.value},
    )

    assert typed.ID == mapping.ID
    assert typed.ID != different.ID
    assert typed.definition == mapping.definition


def test_rolling_transforms_are_discovered_by_the_builtin_registry() -> None:
    registry = builtin_registry()

    expected = {
        "transform.rolling_max": RollingMax,
        "transform.rolling_mean": RollingMean,
        "transform.rolling_median": RollingMedian,
        "transform.rolling_min": RollingMin,
        "transform.rolling_std": RollingStd,
        "transform.rolling_sum": RollingSum,
        "transform.rolling_z_score": RollingZScore,
    }
    for operation, cls in expected.items():
        assert registry.resolve(operation, cls.VERSION) is cls
