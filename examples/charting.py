"""Create a bounded SVG chart from an iosislib graph.

Run with ``py -3 examples/charting.py``. The example prints a data URI that
can be assigned directly to an HTML ``<img src=...>`` attribute. Charts use
the dark Iosis styleguide by default; pass ``theme="light"`` for the light
preset.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

from iosislib.charting import figure_to_svg_data_uri, plot_graph
from iosislib.core.graph import Graph
from iosislib.core.node import Node
from iosislib.core.tsfn import FrameSignature, TSFN


class ExampleSeries(TSFN):
    VERSION = "1.0.0"

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        output = FrameSignature(columns=(("value", pl.Float64),))
        return FrameSignature.empty(), output

    def apply(self) -> pl.LazyFrame:
        start = datetime(2026, 1, 1)
        return pl.DataFrame(
            {
                "timestamp": [start + timedelta(minutes=i) for i in range(6000)],
                "value": [50.0 + (i % 200) / 10.0 for i in range(6000)],
            }
        ).lazy()


def run_example() -> str:
    graph = Graph(Node(ExampleSeries, name="example-series"))
    figure, _ = plot_graph(
        graph,
        max_points=5000,
        title="Bounded graph output",
        ylabel="Value",
    )
    return figure_to_svg_data_uri(figure)


def main() -> None:
    print(run_example())


if __name__ == "__main__":
    main()
