from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import log

import polars as pl
import pytest

from iosislib.core.graph import Graph
from iosislib.core.node import Node
from iosislib.core.tsfn import FrameSignature, TSFN, TSFNConfig
from iosislib.tsfn.transforms import Delta, DeltaConfig, Lag, LagConfig, Logit, Ratio, Spread


@dataclass(frozen=True)
class FloatSeriesConfig(TSFNConfig):
    minutes: tuple[int, ...]
    values: tuple[float | None, ...]
    output_column: str = "value"


class FloatSeriesSource(TSFN):
    VERSION = "1.0.0"
    CONFIG_CLS = FloatSeriesConfig

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


def float_source(
    values: tuple[float | None, ...],
    *,
    minutes: tuple[int, ...] | None = None,
    output_column: str = "value",
) -> Node:
    if minutes is None:
        minutes = tuple(range(len(values)))
    return Node(
        FloatSeriesSource,
        parameters={
            "minutes": minutes,
            "values": values,
            "output_column": output_column,
        },
    )


def test_logit_transforms_probabilities_and_nulls_invalid_values() -> None:
    source = float_source((0.2, 0.5, None, 0.0, 1.0, -0.1, 1.1))
    logit = Node(Logit, bindings={"value": source.value})

    result = Graph(logit).execute()

    values = result["logit"].to_list()
    assert values[0] == pytest.approx(log(0.2 / 0.8))
    assert values[1] == pytest.approx(0.0)
    assert values[2:] == [None, None, None, None, None]


def test_spread_subtracts_right_from_left_and_preserves_nulls() -> None:
    left = float_source((0.8, None, 0.5), output_column="left_price")
    right = float_source((0.3, 0.1, None), output_column="right_price")
    spread = Node(
        Spread,
        bindings={
            "left": left.left_price,
            "right": right.right_price,
        },
    )

    result = Graph(spread).execute()

    assert result["timestamp"].to_list() == [dt(0), dt(1), dt(2)]
    assert result["spread"].to_list() == pytest.approx([0.5, None, None])


def test_ratio_divides_and_nulls_zero_or_missing_denominators() -> None:
    left = float_source((0.8, 0.4, None, 0.6), output_column="left_price")
    right = float_source((0.2, 0.0, 0.5, None), output_column="right_price")
    ratio = Node(
        Ratio,
        bindings={
            "left": left.left_price,
            "right": right.right_price,
        },
    )

    result = Graph(ratio).execute()

    assert result["ratio"].to_list() == pytest.approx([4.0, None, None, None])


def test_delta_uses_lag_after_sorting_and_never_looks_forward() -> None:
    lf = pl.DataFrame(
        {
            "timestamp": [dt(3), dt(0), dt(2), dt(1)],
            "value": [7.0, 1.0, None, 3.0],
        },
        schema={"timestamp": pl.Datetime, "value": pl.Float64},
    ).lazy()

    result = Delta({})(lf).collect()

    assert result["timestamp"].to_list() == [dt(0), dt(1), dt(2), dt(3)]
    assert result["delta"].to_list() == pytest.approx([None, 2.0, None, None])


def test_lag_uses_earlier_observations_after_sorting() -> None:
    source = float_source((7.0, 1.0, 5.0), minutes=(3, 0, 2))
    lag = Node(Lag, bindings={"value": source.value}, parameters={"periods": 2})

    result = Graph(lag).execute()

    assert result["timestamp"].to_list() == [dt(0), dt(2), dt(3)]
    assert result["lag"].to_list() == [None, None, 1.0]

def test_transform_configs_support_custom_column_names() -> None:
    source = float_source((0.25, 0.75), output_column="probability")
    logit = Node(
        Logit,
        bindings={"probability": source.probability},
        parameters={
            "input_column": "probability",
            "output_column": "probability_logit",
        },
    )

    result = Graph(logit).execute()

    assert logit.outputs == {"probability_logit": pl.Float64}
    assert result.columns == ["timestamp", "probability_logit"]
    assert result["probability_logit"].to_list() == pytest.approx(
        [log(0.25 / 0.75), log(0.75 / 0.25)]
    )


def test_delta_config_validation_is_explicit() -> None:
    with pytest.raises(TypeError, match="periods must be an integer"):
        DeltaConfig(periods=True)

    with pytest.raises(ValueError, match="periods must be at least 1"):
        DeltaConfig(periods=0)
    with pytest.raises(TypeError, match="periods must be an integer"):
        LagConfig(periods=True)

    with pytest.raises(ValueError, match="periods must be at least 1"):
        LagConfig(periods=0)
