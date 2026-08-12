from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import polars as pl
import pytest

from iosislib.core.graph import Graph
from iosislib.core.node import Node
from iosislib.core.tsfn import FrameSignature, TSFN, TSFNConfig, TimeAxis
from iosislib.strategy.lowering import builtin_registry
from iosislib.tsfn.transforms import (
    FeaturePacker,
    FeatureUnpacker,
    FeatureUnpackerConfig,
)


@dataclass(frozen=True)
class PackerSourceConfig(TSFNConfig):
    rows: int = 4


class PackerSource(TSFN):
    VERSION = "1.0.0"
    CONFIG_CLS = PackerSourceConfig

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        return (
            FrameSignature.empty(),
            FrameSignature(
                time=TimeAxis(column="timestamp"),
                columns=(
                    ("first", pl.Float64),
                    ("second", pl.Float64),
                    ("third", pl.Float64),
                ),
            ),
        )

    def apply(self) -> pl.LazyFrame:
        rows = self.parameters.rows
        timestamps = [datetime(2026, 1, 1) + timedelta(hours=index) for index in range(rows)]
        return pl.DataFrame(
            {
                "timestamp": timestamps,
                "first": [float(index) for index in range(rows)],
                "second": [float(index * 2) for index in range(rows)],
                "third": [float(index * 3) for index in range(rows)],
            },
            schema={
                "timestamp": pl.Datetime,
                "first": pl.Float64,
                "second": pl.Float64,
                "third": pl.Float64,
            },
        ).lazy()


@dataclass(frozen=True)
class VectorSourceConfig(TSFNConfig):
    rows: int = 4
    width: int = 3


class VectorSource(TSFN):
    VERSION = "1.0.0"
    CONFIG_CLS = VectorSourceConfig

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        return (
            FrameSignature.empty(),
            FrameSignature(
                time=TimeAxis(column="timestamp"),
                columns=(
                    ("features", pl.Float64, (self.parameters.width,)),
                    ("outcome", pl.Float64),
                ),
            ),
        )

    def apply(self) -> pl.LazyFrame:
        rows = self.parameters.rows
        width = self.parameters.width
        timestamps = [datetime(2026, 1, 1) + timedelta(hours=index) for index in range(rows)]
        return pl.DataFrame(
            {
                "timestamp": timestamps,
                "features": pl.Series(
                    "features",
                    [[float(row + column) for column in range(width)] for row in range(rows)],
                    dtype=pl.Array(pl.Float64, width),
                ),
            },
            schema={
                "timestamp": pl.Datetime,
                "features": pl.Array(pl.Float64, width),
            },
        ).lazy()


def test_feature_unpacker_splits_a_vector_into_scalar_columns() -> None:
    src = Node(VectorSource, parameters={"rows": 3})
    unpacker = Node(FeatureUnpacker, bindings={"features": src.features})

    result = Graph(unpacker).execute()

    assert unpacker.outputs == {
        "feature_0": pl.Float64,
        "feature_1": pl.Float64,
        "feature_2": pl.Float64,
    }
    assert result.columns == ["timestamp", "feature_0", "feature_1", "feature_2"]
    assert result["feature_0"].to_list() == [0.0, 1.0, 2.0]
    assert result["feature_1"].to_list() == [1.0, 2.0, 3.0]
    assert result["feature_2"].to_list() == [2.0, 3.0, 4.0]


def test_feature_unpacker_round_trips_a_feature_packer_graph() -> None:
    src = Node(PackerSource, parameters={"rows": 3})
    packer = Node(
        FeaturePacker,
        bindings={"first": src.first, "second": src.second, "third": src.third},
        parameters={"input_columns": ("first", "second", "third")},
    )
    unpacker = Node(FeatureUnpacker, bindings={"features": packer.features})

    result = Graph(unpacker).execute()

    assert result.schema["feature_0"] == pl.Float64
    assert result.schema["feature_1"] == pl.Float64
    assert result.schema["feature_2"] == pl.Float64
    assert result["feature_0"].to_list() == [0.0, 1.0, 2.0]
    assert result["feature_1"].to_list() == [0.0, 2.0, 4.0]
    assert result["feature_2"].to_list() == [0.0, 3.0, 6.0]


def test_feature_unpacker_accepts_a_scalar_input_as_width_one() -> None:
    src = Node(PackerSource, parameters={"rows": 2})
    unpacker = Node(
        FeatureUnpacker,
        bindings={"features": src.first},
        parameters={"features": 1},
    )

    result = Graph(unpacker).execute()

    assert unpacker.outputs == {"feature_0": pl.Float64}
    assert result["feature_0"].to_list() == [0.0, 1.0]


def test_feature_unpacker_supports_custom_base_name() -> None:
    src = Node(VectorSource, parameters={"rows": 2, "width": 2})
    unpacker = Node(
        FeatureUnpacker,
        bindings={"features": src.features},
        parameters={"base_name": "component"},
    )

    result = Graph(unpacker).execute()

    assert result.columns == ["timestamp", "component_0", "component_1"]


def test_feature_unpacker_rejects_configured_width_mismatch() -> None:
    src = Node(VectorSource, parameters={"rows": 2, "width": 3})

    with pytest.raises(ValueError, match="does not match the bound width"):
        Node(
            FeatureUnpacker,
            bindings={"features": src.features},
            parameters={"features": 2},
        )


def test_feature_unpacker_config_validation_is_explicit() -> None:
    with pytest.raises(TypeError, match="features must be a positive integer"):
        FeatureUnpackerConfig(features=True)
    with pytest.raises(ValueError, match="features must be a positive integer"):
        FeatureUnpackerConfig(features=0)
    with pytest.raises(TypeError, match="input_column"):
        FeatureUnpackerConfig(input_column="")
    with pytest.raises(ValueError, match="distinct"):
        FeatureUnpackerConfig(timestamp_column="features")


def test_feature_unpacker_is_discovered_by_the_builtin_registry() -> None:
    registry = builtin_registry()

    assert (
        registry.resolve("transform.feature_unpacker", FeatureUnpacker.VERSION)
        is FeatureUnpacker
    )
