from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import polars as pl
import pytest

from iosislib.core.graph import Graph
from iosislib.core.node import Node
from iosislib.core.tsfn import FrameSignature, TSFN, TSFNConfig
from iosislib.strategy import builtin_registry
from iosislib.tsfn.transforms import BlockPermute, BlockPermuteConfig


@dataclass(frozen=True)
class FrameSourceConfig(TSFNConfig):
    xs: tuple[float, ...]
    ys: tuple[float, ...]


class FrameSource(TSFN):
    VERSION = "1.0.0"
    CONFIG_CLS = FrameSourceConfig

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        return (
            FrameSignature.empty(),
            FrameSignature(
                columns=(("x", pl.Float64), ("y", pl.Float64)),
            ),
        )

    def apply(self) -> pl.LazyFrame:
        n = len(self.parameters.xs)
        return pl.DataFrame(
            {
                "timestamp": [
                    datetime(2026, 1, 1) + timedelta(minutes=i) for i in range(n)
                ],
                "x": list(self.parameters.xs),
                "y": list(self.parameters.ys),
            },
            schema={
                "timestamp": pl.Datetime,
                "x": pl.Float64,
                "y": pl.Float64,
            },
        ).lazy()


def frame_source(xs: tuple[float, ...], ys: tuple[float, ...]) -> Node:
    return Node(FrameSource, parameters={"xs": xs, "ys": ys})


def test_block_permute_preserves_whole_rows() -> None:
    source = frame_source((10.0, 20.0, 30.0, 40.0), (1.0, 2.0, 3.0, 4.0))
    permute = Node(BlockPermute, bindings={"x": source.x, "y": source.y})

    result = Graph(permute).execute()

    input_rows = set(zip((10.0, 20.0, 30.0, 40.0), (1.0, 2.0, 3.0, 4.0)))
    output_rows = set(zip(result["x"].to_list(), result["y"].to_list()))
    assert output_rows == input_rows


def test_block_permute_leaves_timestamp_untouched() -> None:
    source = frame_source((1.0, 2.0, 3.0, 4.0, 5.0, 6.0), (1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
    permute = Node(
        BlockPermute,
        bindings={"x": source.x, "y": source.y},
        parameters={"block_length": 2, "seed": 7},
    )

    result = Graph(permute).execute()

    assert result["timestamp"].to_list() == [
        datetime(2026, 1, 1),
        datetime(2026, 1, 1, 0, 1),
        datetime(2026, 1, 1, 0, 2),
        datetime(2026, 1, 1, 0, 3),
        datetime(2026, 1, 1, 0, 4),
        datetime(2026, 1, 1, 0, 5),
    ]


def test_block_permute_preserves_within_block_order() -> None:
    source = frame_source(
        (0.0, 1.0, 2.0, 3.0, 4.0, 5.0),
        (0.0, 1.0, 2.0, 3.0, 4.0, 5.0),
    )
    permute = Node(
        BlockPermute,
        bindings={"x": source.x, "y": source.y},
        parameters={"block_length": 3, "seed": 42},
    )

    result = Graph(permute).execute()

    xs = result["x"].to_list()
    positions = {value: index for index, value in enumerate(xs)}
    for block_start in (0.0, 3.0):
        assert positions[block_start + 1] == positions[block_start] + 1
        assert positions[block_start + 2] == positions[block_start] + 1 + 1


def test_block_permute_is_deterministic_for_a_given_seed() -> None:
    source = frame_source(
        (0.0, 1.0, 2.0, 3.0, 4.0, 5.0),
        (0.0, 1.0, 2.0, 3.0, 4.0, 5.0),
    )
    params = {"block_length": 2, "seed": 99}

    first = Graph(
        Node(BlockPermute, bindings={"x": source.x, "y": source.y}, parameters=params)
    ).execute()
    second = Graph(
        Node(BlockPermute, bindings={"x": source.x, "y": source.y}, parameters=params)
    ).execute()

    assert first["x"].to_list() == second["x"].to_list()


def test_block_permute_different_seeds_produce_different_orders() -> None:
    xs = tuple(float(i) for i in range(64))
    source = frame_source(xs, xs)

    results = []
    for seed in (0, 1, 2, 3):
        node = Node(
            BlockPermute,
            bindings={"x": source.x, "y": source.y},
            parameters={"block_length": 1, "seed": seed},
        )
        results.append(Graph(node).execute()["x"].to_list())

    assert len({tuple(r) for r in results}) > 1


def test_block_permute_block_length_one_is_an_iid_shuffle() -> None:
    xs = tuple(float(i) for i in range(32))
    source = frame_source(xs, xs)
    permute = Node(
        BlockPermute,
        bindings={"x": source.x, "y": source.y},
        parameters={"block_length": 1, "seed": 5},
    )

    result = Graph(permute).execute()

    assert sorted(result["x"].to_list()) == sorted(xs)
    assert result["x"].to_list() != list(xs)


def test_block_permute_block_length_at_least_height_is_identity() -> None:
    xs = (1.0, 2.0, 3.0, 4.0)
    source = frame_source(xs, xs)
    permute = Node(
        BlockPermute,
        bindings={"x": source.x, "y": source.y},
        parameters={"block_length": 100, "seed": 0},
    )

    result = Graph(permute).execute()

    assert result["x"].to_list() == list(xs)


def test_block_permute_config_validation_is_explicit() -> None:
    with pytest.raises(TypeError, match="block_length must be an integer"):
        BlockPermuteConfig(block_length=True)

    with pytest.raises(ValueError, match="block_length must be at least 1"):
        BlockPermuteConfig(block_length=0)

    with pytest.raises(TypeError, match="seed must be an integer"):
        BlockPermuteConfig(seed=1.5)

    with pytest.raises(ValueError, match="seed must be non-negative"):
        BlockPermuteConfig(seed=-1)


def test_block_permute_passes_through_dtypes_and_columns() -> None:
    source = frame_source((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))
    permute = Node(
        BlockPermute,
        bindings={"x": source.x, "y": source.y},
    )

    assert permute.outputs == {"x": pl.Float64, "y": pl.Float64}


def test_block_permute_is_discovered_by_the_builtin_registry() -> None:
    registry = builtin_registry()

    assert registry.resolve("transform.block_permute", BlockPermute.VERSION) is BlockPermute
