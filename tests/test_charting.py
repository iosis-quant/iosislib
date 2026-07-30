from datetime import datetime, timedelta

import matplotlib
import polars as pl
import pytest

from iosislib.charting import plot_frame, plot_graph


matplotlib.use("Agg")


def test_plot_frame_uses_first_column_sorts_rows_and_expands_arrays() -> None:
    frame = pl.DataFrame(
        {
            "time": [datetime(2026, 1, 1, 0, 1), datetime(2026, 1, 1, 0, 0)],
            "value": [2.0, 1.0],
            "bands": [[20.0, 21.0], [10.0, 11.0]],
        }
    )

    figure, axes = plot_frame(frame, title="Prices")

    assert axes.get_title(loc="left") == "Prices"
    assert [line.get_label() for line in axes.lines] == [
        "value", "bands[0]", "bands[1]"
    ]
    assert list(axes.lines[0].get_ydata()) == [1.0, 2.0]
    figure.clear()


def test_plot_frame_accepts_lazy_frames_and_selected_columns() -> None:
    frame = pl.DataFrame(
        {
            "timestamp": [datetime(2026, 1, 1), datetime(2026, 1, 2)],
            "left": [1, 2],
            "right": [3, 4],
        }
    ).lazy()

    figure, axes = plot_frame(frame, columns=["right"], legend=False)

    assert len(axes.lines) == 1
    assert axes.lines[0].get_label() == "right"
    figure.clear()


def test_plot_frame_rejects_non_time_first_column() -> None:
    frame = pl.DataFrame({"value": [1], "timestamp": [datetime(2026, 1, 1)]})

    with pytest.raises(TypeError, match="first frame column"):
        plot_frame(frame)


def test_plot_graph_executes_graph_like_object() -> None:
    class FakeGraph:
        def execute(self, *, executor: object = None) -> pl.DataFrame:
            del executor
            return pl.DataFrame(
                {
                    "timestamp": [datetime(2026, 1, 1) + timedelta(days=i) for i in range(2)],
                    "value": [1.0, 2.0],
                }
            )

    figure, axes = plot_graph(FakeGraph())

    assert len(axes.lines) == 1
    figure.clear()

