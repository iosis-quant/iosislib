from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import polars as pl
import pytest

from iosislib.core.graph import Graph
from iosislib.core.node import Node
from iosislib.core.tsfn import FrameSignature, TSFN, TSFNConfig
from iosislib.tsfn.transforms import EwmMean, EwmMeanConfig, EwmStd


@dataclass(frozen=True)
class EwmSeriesConfig(TSFNConfig):
    values: tuple[float | None, ...]


class EwmSeriesSource(TSFN):
    VERSION = "1.0.0"
    CONFIG_CLS = EwmSeriesConfig

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        return (
            FrameSignature.empty(),
            FrameSignature(columns=(("value", pl.Float64),)),
        )

    def apply(self) -> pl.LazyFrame:
        return pl.DataFrame(
            {
                "timestamp": [
                    datetime(2026, 1, 1, 0, index) for index in range(len(self.parameters.values))
                ],
                "value": list(self.parameters.values),
            },
            schema={"timestamp": pl.Datetime, "value": pl.Float64},
        ).lazy()


def source(values: tuple[float | None, ...]) -> Node:
    return Node(EwmSeriesSource, parameters={"values": values})


def test_ewm_mean_matches_polars_reference_for_adjusted_decay() -> None:
    node = Node(
        EwmMean,
        bindings={"value": source((1.0, 2.0, 3.0, 4.0)).value},
        parameters={"alpha": 0.5, "adjust": True},
    )

    result = Graph(node).execute()

    reference = (
        pl.Series("value", [1.0, 2.0, 3.0, 4.0])
        .ewm_mean(alpha=0.5, adjust=True, min_samples=1)
        .to_list()
    )
    assert result["ewm_mean"].to_list() == pytest.approx(reference)


def test_ewm_std_tracks_dispersion_and_starts_null_for_single_sample() -> None:
    node = Node(
        EwmStd,
        bindings={"value": source((5.0, 5.0, 5.0, 6.0, 8.0)).value},
        parameters={"alpha": 0.3},
    )

    result = Graph(node).execute()

    assert result["ewm_std"].to_list()[0] is None
    assert result["ewm_std"].to_list()[-1] > 0.0


def test_ewm_transforms_sort_before_application() -> None:
    lf = pl.DataFrame(
        {
            "timestamp": [
                datetime(2026, 1, 1, 0, 2),
                datetime(2026, 1, 1, 0, 0),
                datetime(2026, 1, 1, 0, 1),
            ],
            "value": [30.0, 10.0, 20.0],
        },
        schema={"timestamp": pl.Datetime, "value": pl.Float64},
    ).lazy()

    result = EwmMean({"alpha": 0.5})(lf).collect()

    reference = (
        pl.Series("value", [10.0, 20.0, 30.0])
        .ewm_mean(alpha=0.5, adjust=True, min_samples=1)
        .to_list()
    )
    assert result["timestamp"].to_list() == [
        datetime(2026, 1, 1, 0, 0),
        datetime(2026, 1, 1, 0, 1),
        datetime(2026, 1, 1, 0, 2),
    ]
    assert result["ewm_mean"].to_list() == pytest.approx(reference)


def test_ewm_config_validation_is_explicit() -> None:
    with pytest.raises(TypeError, match="alpha must be a number"):
        EwmMeanConfig(alpha=True)
    with pytest.raises(ValueError, match="alpha must be between"):
        EwmMeanConfig(alpha=0.0)
    with pytest.raises(ValueError, match="alpha must be between"):
        EwmMeanConfig(alpha=1.5)
    with pytest.raises(TypeError, match="adjust must be a bool"):
        EwmMeanConfig(adjust=1)
    with pytest.raises(TypeError, match="min_samples must be an integer"):
        EwmMeanConfig(min_samples=1.0)
    with pytest.raises(ValueError, match="min_samples must be at least 1"):
        EwmMeanConfig(min_samples=0)


def test_ewm_typed_and_mapping_config_share_identity() -> None:
    src = source((1.0, 2.0, 3.0))
    typed = Node(
        EwmMean,
        config=EwmMeanConfig(alpha=0.2, min_samples=1),
        bindings={"value": src.value},
    )
    mapping = Node(
        EwmMean,
        parameters={"alpha": 0.2, "min_samples": 1},
        bindings={"value": src.value},
    )
    different = Node(
        EwmMean,
        parameters={"alpha": 0.5, "min_samples": 1},
        bindings={"value": src.value},
    )

    assert typed.ID == mapping.ID
    assert typed.ID != different.ID
