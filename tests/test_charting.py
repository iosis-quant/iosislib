from datetime import datetime, timezone, timedelta

import matplotlib
import polars as pl
import pytest

from iosislib.charting import (
    ChartTheme,
    DARK_THEME,
    LIGHT_THEME,
    plot_frame,
    plot_graph,
)


matplotlib.use("Agg")


def _frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "time": [datetime(2026, 1, 1, 0, 1), datetime(2026, 1, 1, 0, 0)],
            "value": [2.0, 1.0],
            "bands": [[20.0, 21.0], [10.0, 11.0]],
        }
    )


def test_plot_frame_uses_first_column_sorts_rows_and_expands_arrays() -> None:
    figure, axes = plot_frame(_frame(), title="Prices")

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


def test_dark_theme_is_the_default() -> None:
    figure, axes = plot_frame(_frame())

    assert axes.get_facecolor() == matplotlib.colors.to_rgba(DARK_THEME.background)
    assert figure.patch.get_facecolor() == matplotlib.colors.to_rgba(DARK_THEME.background)
    figure.clear()


def test_light_theme_selector_resolves_to_light_preset() -> None:
    figure, axes = plot_frame(_frame(), theme="light")

    assert axes.get_facecolor() == matplotlib.colors.to_rgba(LIGHT_THEME.background)
    assert [line.get_color() for line in axes.lines] == list(LIGHT_THEME.colors[:3])
    figure.clear()


def test_custom_chart_theme_is_respected() -> None:
    theme = ChartTheme(
        colors=("#111111", "#222222", "#333333"),
        background="#000000",
        text_color="#FFFFFF",
        grid_color="#555555",
        grid_alpha=0.2,
        line_width=1.0,
    )
    figure, axes = plot_frame(_frame(), theme=theme)

    assert axes.get_facecolor() == matplotlib.colors.to_rgba("#000000")
    assert [line.get_color() for line in axes.lines] == ["#111111", "#222222", "#333333"]
    assert axes.lines[0].get_linewidth() == 1.0
    assert axes.get_title(loc="left") == "iosislib graph"
    assert axes.get_yticklabels()[0].get_color() == "#FFFFFF"
    figure.clear()


def test_unknown_theme_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="theme must be"):
        plot_frame(_frame(), theme="neon")

    with pytest.raises(TypeError, match="theme must be"):
        plot_frame(_frame(), theme=42)


def test_grid_is_horizontal_only_by_default() -> None:
    figure, axes = plot_frame(_frame())

    assert any(line.get_visible() for line in axes.yaxis.get_gridlines())
    assert all(not line.get_visible() for line in axes.xaxis.get_gridlines())
    figure.clear()


def test_yaxis_uses_thousands_separators_for_large_integral_values() -> None:
    frame = pl.DataFrame(
        {
            "timestamp": [datetime(2026, 1, 1) + timedelta(days=i) for i in range(6)],
            "value": [0, 20_000, 40_000, 60_000, 80_000, 100_000],
        }
    )
    figure, axes = plot_frame(frame, max_points=None)

    assert any("," in tick.get_text() for tick in axes.get_yticklabels())
    figure.clear()


def test_y_tick_format_override_is_applied() -> None:
    frame = pl.DataFrame(
        {
            "timestamp": [datetime(2026, 1, 1) + timedelta(days=i) for i in range(4)],
            "value": [1.0, 2.0, 3.0, 4.0],
        }
    )
    figure, axes = plot_frame(frame, max_points=None, y_tick_format="{x:.1f}%")

    assert all(tick.get_text().endswith("%") for tick in axes.get_yticklabels())
    figure.clear()


def test_ylim_and_xlim_passthrough() -> None:
    figure, axes = plot_frame(_frame(), ylim=(0, 100), xlim=(0, 10))

    assert axes.get_ylim() == (0.0, 100.0)
    assert axes.get_xlim() == (0.0, 10.0)
    figure.clear()


def test_invalid_axis_bounds_are_rejected() -> None:
    with pytest.raises(ValueError, match="pair of numeric bounds"):
        plot_frame(_frame(), ylim=(1, 2, 3))
    with pytest.raises(ValueError, match="pair of numeric bounds"):
        plot_frame(_frame(), xlim=("a", "b"))
    with pytest.raises(ValueError, match="lower bound must be below"):
        plot_frame(_frame(), ylim=(10, 0))


def test_timezone_aware_datetimes_are_normalized_to_naive_utc() -> None:
    start = datetime(2026, 1, 1, tzinfo=timezone(timedelta(hours=2)))
    frame = pl.DataFrame(
        {
            "timestamp": [start + timedelta(hours=i) for i in range(3)],
            "value": [1.0, 2.0, 3.0],
        }
    )
    figure, axes = plot_frame(frame, max_points=None)

    xdata = axes.lines[0].get_xdata()
    assert xdata[0].tzinfo is None
    assert xdata[0] == datetime(2025, 12, 31, 22, 0)
    figure.clear()


def test_infinite_values_render_as_gaps() -> None:
    frame = pl.DataFrame(
        {
            "timestamp": [datetime(2026, 1, 1) + timedelta(days=i) for i in range(3)],
            "value": [1.0, float("inf"), 3.0],
        }
    )
    figure, axes = plot_frame(frame, max_points=None)

    ydata = list(axes.lines[0].get_ydata())
    assert ydata[1] != ydata[1]
    assert ydata[0] == 1.0
    assert ydata[2] == 3.0
    figure.clear()
