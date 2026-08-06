from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import exp, log

import polars as pl
import pytest

from iosislib.core.graph import Graph
from iosislib.core.node import Node
from iosislib.core.tsfn import FrameSignature, TSFN, TSFNConfig
from iosislib.strategy.lowering import builtin_registry
from iosislib.tsfn.transforms import (
    Exp,
    Log,
    LogRatio,
    LogReturn,
    PctChange,
)


@dataclass(frozen=True)
class ReturnSeriesConfig(TSFNConfig):
    left: tuple[float | None, ...]
    right: tuple[float | None, ...]


class ReturnSeriesSource(TSFN):
    VERSION = "1.0.0"
    CONFIG_CLS = ReturnSeriesConfig

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        return (
            FrameSignature.empty(),
            FrameSignature(
                columns=(
                    ("left", pl.Float64),
                    ("right", pl.Float64),
                )
            ),
        )

    def apply(self) -> pl.LazyFrame:
        return pl.DataFrame(
            {
                "timestamp": [
                    datetime(2026, 1, 1, 0, index) for index in range(len(self.parameters.left))
                ],
                "left": list(self.parameters.left),
                "right": list(self.parameters.right),
            },
            schema={
                "timestamp": pl.Datetime,
                "left": pl.Float64,
                "right": pl.Float64,
            },
        ).lazy()


def source(
    left: tuple[float | None, ...],
    right: tuple[float | None, ...] | None = None,
) -> Node:
    if right is None:
        right = tuple(1.0 for _ in left)
    return Node(ReturnSeriesSource, parameters={"left": left, "right": right})


def test_pct_change_is_relative_return_and_nulls_zero_priors() -> None:
    node = Node(
        PctChange,
        bindings={"value": source((1.0, 2.0, 4.0, 0.0, 8.0)).left},
        parameters={"periods": 1},
    )

    result = Graph(node).execute()

    assert result["pct_change"].to_list() == pytest.approx(
        [None, 1.0, 1.0, -1.0, None]
    )


def test_pct_change_supports_multi_period_windows_and_sorts() -> None:
    node = Node(
        PctChange,
        bindings={"value": source((1.0, 2.0, 4.0, 8.0)).left},
        parameters={"periods": 2},
    )

    result = Graph(node).execute()

    assert result["pct_change"].to_list() == pytest.approx([None, None, 3.0, 3.0])


def test_log_return_is_log_difference_and_nulls_invalid_domains() -> None:
    node = Node(
        LogReturn,
        bindings={"value": source((1.0, 2.0, 4.0, 0.0, -1.0, None)).left},
        parameters={"periods": 1},
    )

    result = Graph(node).execute()

    assert result["log_return"].to_list() == pytest.approx(
        [None, log(2.0), log(2.0), None, None, None]
    )


def test_log_nulls_non_positive_values_and_exp_is_its_inverse() -> None:
    log_node = Node(
        Log,
        bindings={"value": source((0.0, 1.0, None, 2.0)).left},
    )
    exp_node = Node(
        Exp,
        bindings={"value": source((0.0, 1.0, None, 2.0)).left},
    )

    log_result = Graph(log_node).execute()
    exp_result = Graph(exp_node).execute()

    assert log_result["log"].to_list() == pytest.approx([None, 0.0, None, log(2.0)])
    assert exp_result["exp"].to_list() == pytest.approx([1.0, exp(1.0), None, exp(2.0)])


def test_log_ratio_logs_the_ratio_and_nulls_invalid_domains() -> None:
    observations = Node(
        ReturnSeriesSource,
        parameters={
            "left": (0.5, 2.0, 0.0, -1.0, None, 4.0),
            "right": (0.5, 0.5, 1.0, 1.0, 1.0, None),
        },
    )
    node = Node(
        LogRatio,
        bindings={
            "left": observations.left,
            "right": observations.right,
        },
    )

    result = Graph(node).execute()

    assert result["log_ratio"].to_list() == pytest.approx(
        [0.0, log(4.0), None, None, None, None]
    )


def test_pct_change_config_validation_is_explicit() -> None:
    from iosislib.tsfn.transforms import PctChangeConfig

    with pytest.raises(TypeError, match="periods must be an integer"):
        PctChangeConfig(periods=True)
    with pytest.raises(ValueError, match="periods must be at least 1"):
        PctChangeConfig(periods=0)
    with pytest.raises(ValueError, match="Duplicate column names"):
        PctChangeConfig(input_column="timestamp")
    with pytest.raises(ValueError, match="Duplicate column names"):
        PctChangeConfig(input_column="pct_change")


def test_return_transforms_are_discovered_by_the_builtin_registry() -> None:
    registry = builtin_registry()

    expected = {
        "transform.pct_change": PctChange,
        "transform.log_return": LogReturn,
        "transform.log": Log,
        "transform.exp": Exp,
        "transform.log_ratio": LogRatio,
    }
    for operation, cls in expected.items():
        assert registry.resolve(operation, cls.VERSION) is cls
