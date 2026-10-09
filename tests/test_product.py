from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import polars as pl
import pytest

from iosislib.core.graph import Graph
from iosislib.core.node import Node
from iosislib.core.tsfn import FrameSignature, TSFN, TSFNConfig
from iosislib.strategy import builtin_registry
from iosislib.tsfn.transforms import Product, ProductConfig
from _floats import assert_lists_close


@dataclass(frozen=True)
class PairSeriesConfig(TSFNConfig):
    left_values: tuple[float | None, ...]
    right_values: tuple[float | None, ...]


class PairSeriesSource(TSFN):
    VERSION = "1.0.0"
    CONFIG_CLS = PairSeriesConfig

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
                    dt(index) for index in range(len(self.parameters.left_values))
                ],
                "left": list(self.parameters.left_values),
                "right": list(self.parameters.right_values),
            },
            schema={
                "timestamp": pl.Datetime,
                "left": pl.Float64,
                "right": pl.Float64,
            },
        ).lazy()


def dt(minute: int) -> datetime:
    return datetime(2026, 1, 1, 0, minute)


def pair_source(
    left: tuple[float | None, ...],
    right: tuple[float | None, ...],
) -> Node:
    return Node(
        PairSeriesSource,
        parameters={"left_values": left, "right_values": right},
    )


def test_product_multiplies_and_preserves_nulls() -> None:
    src = pair_source((2.0, None, 0.5, 3.0), (3.0, 4.0, None, None))
    node = Node(Product, bindings={"left": src.left, "right": src.right})

    result = Graph(node).execute()

    assert result["product"].to_list() == pytest.approx([6.0, None, None, None])


def test_product_config_rejects_colliding_column_names() -> None:
    with pytest.raises(ValueError, match="Duplicate column names"):
        ProductConfig(left_column="right")
    with pytest.raises(ValueError, match="Duplicate column names"):
        ProductConfig(right_column="timestamp")
    with pytest.raises(ValueError, match="Duplicate column names"):
        ProductConfig(left_column="timestamp")


@dataclass(frozen=True)
class PairVectorConfig(TSFNConfig):
    left_values: tuple[tuple[float, ...], ...]
    right_values: tuple[tuple[float, ...], ...]


class PairVectorSource(TSFN):
    VERSION = "1.0.0"
    CONFIG_CLS = PairVectorConfig

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        return (
            FrameSignature.empty(),
            FrameSignature(
                columns=(
                    ("left", pl.Float64, (2,)),
                    ("right", pl.Float64, (2,)),
                )
            ),
        )

    def apply(self) -> pl.LazyFrame:
        return pl.DataFrame(
            {
                "timestamp": [
                    dt(index) for index in range(len(self.parameters.left_values))
                ],
                "left": pl.Series(
                    "left",
                    list(self.parameters.left_values),
                    dtype=pl.Array(pl.Float64, 2),
                ),
                "right": pl.Series(
                    "right",
                    list(self.parameters.right_values),
                    dtype=pl.Array(pl.Float64, 2),
                ),
            },
            schema={
                "timestamp": pl.Datetime,
                "left": pl.Array(pl.Float64, 2),
                "right": pl.Array(pl.Float64, 2),
            },
        ).lazy()


def test_product_inherits_vector_shape_and_multiplies_cells() -> None:
    src = Node(
        PairVectorSource,
        parameters={
            "left_values": ((1.0, 2.0), (3.0, 4.0)),
            "right_values": ((5.0, 6.0), (7.0, 8.0)),
        },
    )
    node = Node(Product, bindings={"left": src.left, "right": src.right})

    result = Graph(node).execute()

    assert node.outputs == {"product": pl.Array(pl.Float64, 2)}
    assert_lists_close(
        result["product"].to_list(),
        [[5.0, 12.0], [21.0, 32.0]],
    )


def test_product_is_discovered_by_the_builtin_registry() -> None:
    registry = builtin_registry()

    assert registry.resolve("transform.product", Product.VERSION) is Product
