from datetime import datetime, timedelta

import matplotlib
import polars as pl
import pytest

from iosislib.charting import figure_to_svg, figure_to_svg_data_uri, plot_frame


matplotlib.use("Agg")


def _frame(rows: int) -> pl.DataFrame:
    start = datetime(2026, 1, 1)
    return pl.DataFrame(
        {
            "timestamp": [start + timedelta(minutes=i) for i in range(rows)],
            "value": list(range(rows)),
        }
    )


def test_plot_frame_limits_points_and_preserves_endpoints() -> None:
    figure, axes = plot_frame(_frame(20), max_points=5)

    assert len(axes.lines[0].get_xdata()) == 5
    assert list(axes.lines[0].get_ydata()) == [0.0, 5.0, 10.0, 14.0, 19.0]
    figure.clear()


def test_plot_frame_rejects_invalid_point_budget() -> None:
    with pytest.raises(ValueError, match="greater than or equal to 2"):
        plot_frame(_frame(4), max_points=1)


def test_svg_and_data_uri_exports_are_embeddable() -> None:
    figure, _ = plot_frame(_frame(3), max_points=None)

    svg = figure_to_svg(figure)
    uri = figure_to_svg_data_uri(figure)

    assert svg.startswith("<?xml") or "<svg" in svg
    assert uri.startswith("data:image/svg+xml;charset=utf-8,")
    assert "%3Csvg" in uri or "%3C%3Fxml" in uri
    figure.clear()
