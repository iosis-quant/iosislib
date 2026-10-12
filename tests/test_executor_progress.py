from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl

from iosislib.core.graph import (
    Executor,
    Graph,
    LocalExecutor,
    NodeProgress,
)
from iosislib.core.node import Node
from iosislib.core.tsfn import FrameSignature, TSFN, TimeAxis


TIMESTAMP_AXIS = TimeAxis(column="timestamp", dtype=pl.Datetime)
VALUE_FRAME = FrameSignature(
    time=TIMESTAMP_AXIS,
    columns=(("value", pl.Float64),),
)


class SimpleSource(TSFN):
    VERSION = "1.0.0"

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        return FrameSignature.empty(), VALUE_FRAME

    def apply(self) -> pl.LazyFrame:
        return pl.DataFrame(
            {
                "timestamp": [
                    datetime(2026, 1, 1),
                    datetime(2026, 1, 2),
                    datetime(2026, 1, 3),
                ],
                "value": [1.0, 2.0, 3.0],
            }
        ).lazy()


class Doubler(TSFN):
    VERSION = "1.0.0"

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        return VALUE_FRAME, VALUE_FRAME

    def apply(self, lf: pl.LazyFrame | None = None) -> pl.LazyFrame:
        assert lf is not None
        return lf.with_columns((pl.col("value") * 2).alias("value"))


class CollectingExecutor(Executor):
    def materialize(self, node: Node, lf: pl.LazyFrame) -> pl.DataFrame:
        return lf.collect()


def _make_transform_graph() -> Graph:
    source = Node(SimpleSource, name="source", materialize=True)
    doubled = Node(
        Doubler,
        bindings={"value": source.output("value")},
        name="doubler",
        materialize=True,
    )
    return Graph(doubled)


def _assert_well_formed(events: list[NodeProgress], graph: Graph) -> None:
    total = len(graph.node_list)
    assert events, "progress callback must fire at least once"
    assert all(event.total == total for event in events)
    assert [event.completed for event in events] == list(range(total + 1))
    pre_node = events[:-1]
    assert all(event.node_id is not None for event in pre_node)
    assert [event.node_id for event in pre_node] == [
        node.ID for node in graph.node_list
    ]
    final = events[-1]
    assert final.node_id is None and final.node_name is None
    assert final.completed == final.total == total


class TestExecutorProgress:
    def test_local_executor_reports_full_order(self) -> None:
        graph = _make_transform_graph()
        events: list[NodeProgress] = []

        graph.execute(
            executor=LocalExecutor(no_cache=True),
            on_progress=events.append,
        )

        _assert_well_formed(events, graph)

    def test_base_executor_reports_full_order(self) -> None:
        graph = _make_transform_graph()
        events: list[NodeProgress] = []

        graph.execute(
            executor=CollectingExecutor(),
            on_progress=events.append,
        )

        _assert_well_formed(events, graph)

    def test_cache_hits_still_reach_total(self, tmp_path: Path) -> None:
        graph = _make_transform_graph()
        executor = LocalExecutor(cache_dir=tmp_path)
        graph.execute(executor=executor)

        events: list[NodeProgress] = []
        result = graph.execute(executor=executor, on_progress=events.append)

        assert result["value"].to_list() == [2.0, 4.0, 6.0]
        _assert_well_formed(events, graph)

    def test_no_callback_is_permitted(self) -> None:
        graph = _make_transform_graph()

        result = graph.execute(executor=LocalExecutor(no_cache=True))

        assert result["value"].to_list() == [2.0, 4.0, 6.0]

    def test_progress_carries_node_names(self) -> None:
        graph = _make_transform_graph()
        events: list[NodeProgress] = []

        graph.execute(
            executor=LocalExecutor(no_cache=True),
            on_progress=events.append,
        )

        assert [event.node_name for event in events[:-1]] == [
            "source",
            "doubler",
        ]

    def test_callback_failure_propagates(self) -> None:
        graph = _make_transform_graph()

        def explode(event: NodeProgress) -> None:
            raise RuntimeError("callback failed")

        try:
            graph.execute(
                executor=LocalExecutor(no_cache=True),
                on_progress=explode,
            )
        except RuntimeError as exc:
            assert str(exc) == "callback failed"
        else:
            raise AssertionError("callback failure must propagate")
