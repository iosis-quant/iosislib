from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import polars as pl
import pytest

from iosislib.core.graph import Graph
from iosislib.core.model import ChronologicalSplitter, EveryNTicksScheduler
from iosislib.core.node import Node
from iosislib.core.tsfn import FrameSignature, TSFN, TSFNConfig, TimeAxis
from iosislib.models.mlp import DenseMLP
from iosislib.strategy.lowering import builtin_registry
from iosislib.tsfn.transforms import (
    FeaturePacker,
    FeaturePackerConfig,
    RollingMean,
)


@dataclass(frozen=True)
class PackerSourceConfig(TSFNConfig):
    rows: int = 12


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
                    ("outcome", pl.Float64),
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
                "outcome": [1.0 if index % 3 == 0 else 0.0 for index in range(rows)],
            },
            schema={
                "timestamp": pl.Datetime,
                "first": pl.Float64,
                "second": pl.Float64,
                "outcome": pl.Float64,
            },
        ).lazy()


def source() -> Node:
    return Node(PackerSource, parameters={"rows": 12})


def test_feature_packer_emits_exact_array_schema_and_values() -> None:
    src = source()
    packer = Node(
        FeaturePacker,
        bindings={"first": src.first, "second": src.second},
        parameters={"input_columns": ["first", "second"]},
    )

    result = Graph(packer).execute()

    assert packer.outputs == {"features": pl.Array(pl.Float64, 2)}
    assert result.columns == ["timestamp", "features"]
    assert result.schema["features"] == pl.Array(pl.Float64, 2)
    assert result["features"].to_list()[3] == [3.0, 6.0]


def test_feature_packer_propagates_nulls_inside_the_array() -> None:
    lf = pl.DataFrame(
        {
            "timestamp": [datetime(2026, 1, 1) + timedelta(hours=i) for i in range(3)],
            "first": [1.0, None, 3.0],
            "second": [4.0, 5.0, None],
        },
        schema={
            "timestamp": pl.Datetime,
            "first": pl.Float64,
            "second": pl.Float64,
        },
    ).lazy()

    result = FeaturePacker(
        {"input_columns": ("first", "second")}
    )(lf).collect()

    assert result["features"].to_list()[1] == [None, 5.0]
    assert result["features"].to_list()[2] == [3.0, None]


def test_feature_packer_config_validation_is_explicit() -> None:
    with pytest.raises(TypeError, match="input_columns must be a sequence"):
        FeaturePackerConfig(input_columns="first")
    with pytest.raises(ValueError, match="at least one column"):
        FeaturePackerConfig(input_columns=[])
    with pytest.raises(TypeError, match="only strings"):
        FeaturePackerConfig(input_columns=(1, 2))
    with pytest.raises(ValueError, match="non-empty"):
        FeaturePackerConfig(input_columns=("", "second"))
    with pytest.raises(ValueError, match="Duplicate input_columns"):
        FeaturePackerConfig(input_columns=("first", "first"))
    with pytest.raises(ValueError, match="output_column must not be"):
        FeaturePackerConfig(input_columns=("features", "second"))
    with pytest.raises(ValueError, match="timestamp_column must not be"):
        FeaturePackerConfig(input_columns=("first",), timestamp_column="first")
    with pytest.raises(ValueError, match="Unexpected parameters"):
        Node(FeaturePacker, parameters={"input_columns": ("first",), "bogus": 1})


def test_feature_packer_list_and_tuple_columns_share_identity() -> None:
    src = source()
    from_list = Node(
        FeaturePacker,
        bindings={"first": src.first, "second": src.second},
        parameters={"input_columns": ["first", "second"]},
    )
    from_tuple = Node(
        FeaturePacker,
        bindings={"first": src.first, "second": src.second},
        parameters={"input_columns": ("first", "second")},
    )
    reordered = Node(
        FeaturePacker,
        bindings={"first": src.first, "second": src.second},
        parameters={"input_columns": ["second", "first"]},
    )
    different_width = Node(
        FeaturePacker,
        bindings={"first": src.first},
        parameters={"input_columns": ["first"]},
    )

    assert from_list.ID == from_tuple.ID
    assert from_list.definition == from_tuple.definition
    assert from_list.ID != reordered.ID
    assert from_list.ID != different_width.ID


def test_feature_packer_binds_into_a_supervised_model_graph() -> None:
    src = source()
    features = Node(
        FeaturePacker,
        bindings={"first": src.first, "second": src.second},
        parameters={"input_columns": ("first", "second")},
        name="features",
    )
    model = Node(
        DenseMLP,
        bindings={"features": features.features, "target": src.outcome},
        parameters={
            "layers": (2, 1),
            "epochs": 1,
            "scheduler": EveryNTicksScheduler(4),
            "splitter": ChronologicalSplitter(test_size=2),
        },
        name="dense_mlp",
    )

    result = Graph(model).execute()

    assert result.columns == ["timestamp", "prediction"]
    assert result["prediction"].dtype == pl.Float64
    assert result.height == 12
    assert result["prediction"].null_count() < result.height
    assert result["prediction"].drop_nulls().to_list() != []


def test_feature_packer_is_discovered_by_the_builtin_registry() -> None:
    registry = builtin_registry()

    assert registry.resolve("transform.feature_packer", FeaturePacker.VERSION) is FeaturePacker
    assert registry.resolve("transform.rolling_mean", RollingMean.VERSION) is RollingMean
