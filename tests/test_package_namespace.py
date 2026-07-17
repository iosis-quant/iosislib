from datetime import datetime
from pathlib import Path

import polars as pl

import iosislib
from iosislib.core.graph import Graph
from iosislib.core.node import Node
from iosislib.core.tsfn import (
    ColumnSignature,
    FrameSignature,
    TSFN,
    TimeAxis,
)
from iosislib.tsfn.adapters import CSVSource, ParquetSource
from iosislib.tsfn.transforms import Logit


class NamespaceSource(TSFN):
    VERSION = "1.0.0"

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        return (
            FrameSignature.empty(),
            FrameSignature(columns=(("value", pl.Float64),)),
        )

    def apply(self) -> pl.LazyFrame:
        return pl.DataFrame(
            {"timestamp": [datetime(2026, 1, 1)], "value": [0.25]},
            schema={"timestamp": pl.Datetime, "value": pl.Float64},
        ).lazy()


def test_top_level_namespace_is_a_small_exact_convenience_surface() -> None:
    assert iosislib.Node is Node
    assert iosislib.Graph is Graph
    assert iosislib.TimeAxis is TimeAxis
    assert iosislib.ColumnSignature is ColumnSignature
    assert iosislib.FrameSignature is FrameSignature


def test_pep561_marker_and_representative_modules_are_installed() -> None:
    assert Path(iosislib.__file__).with_name("py.typed").is_file()
    assert CSVSource.__module__.startswith("iosislib.")
    assert ParquetSource.__module__.startswith("iosislib.")
    assert Logit.__module__.startswith("iosislib.")


def test_post_migration_ids_remain_deterministic() -> None:
    first_source = Node(NamespaceSource)
    second_source = Node(NamespaceSource)
    first = Node(Logit, bindings={"value": first_source.value})
    second = Node(Logit, bindings={"value": second_source.value})

    assert first_source.ID == second_source.ID
    assert first.ID == second.ID
    assert Graph(first).ID == Graph(second).ID
