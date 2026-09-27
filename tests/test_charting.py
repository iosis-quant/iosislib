from datetime import date, datetime, time, timezone, timedelta

import matplotlib
import matplotlib.collections
import polars as pl
import pytest

from iosislib.charting import (
    ChartTheme,
    DARK_THEME,
    LIGHT_THEME,
    plot_bars,
    plot_equity,
    plot_frame,
    plot_graph,
    plot_scatter,
)


matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


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


def _equity_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "timestamp": [datetime(2026, 1, 1) + timedelta(days=i) for i in range(4)],
            "equity": [100.0, 110.0, 95.0, 105.0],
            "cash": [50.0, 55.0, 45.0, 60.0],
        }
    )


def test_plot_equity_draws_lines_and_fills_to_the_start_value() -> None:
    figure, axes = plot_equity(_equity_frame(), columns=["equity"])

    assert [line.get_label() for line in axes.lines] == ["equity"]
    assert len(axes.collections) == 1
    assert isinstance(axes.collections[0], matplotlib.collections.PolyCollection)
    figure.clear()


def test_plot_equity_fills_each_series_separately() -> None:
    figure, axes = plot_equity(_equity_frame(), columns=["equity", "cash"])

    assert len(axes.lines) == 2
    assert len(axes.collections) == 2
    figure.clear()


def test_plot_equity_requires_numeric_value_columns() -> None:
    frame = pl.DataFrame(
        {
            "timestamp": [datetime(2026, 1, 1), datetime(2026, 1, 2)],
            "value": [None, None],
        }
    )
    with pytest.raises(TypeError, match="plot_equity requires at least one numeric"):
        plot_equity(frame)


def _bar_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "day": [date(2026, 1, 1), date(2026, 1, 2), date(2026, 1, 3)],
            "profit": [1.0, -2.0, 3.0],
            "loss": [-0.5, 1.5, -1.0],
        }
    )


def test_plot_bars_draws_one_grouped_container_per_column() -> None:
    figure, axes = plot_bars(_bar_frame(), columns=["profit", "loss"])

    assert len(axes.containers) == 2
    assert [container.get_label() for container in axes.containers] == ["profit", "loss"]
    assert all(len(container) == 3 for container in axes.containers)
    figure.clear()


def test_plot_bars_rejects_a_time_typed_axis() -> None:
    frame = pl.DataFrame(
        {"when": [time(12, 0), time(13, 0)], "value": [1.0, 2.0]}
    )
    with pytest.raises(TypeError, match="requires a Date or Datetime time column"):
        plot_bars(frame, columns=["value"])


def test_plot_bars_rejects_a_non_temporal_first_column() -> None:
    frame = pl.DataFrame({"index": [1, 2], "value": [1.0, 2.0]})
    with pytest.raises(TypeError, match="date/time values"):
        plot_bars(frame, columns=["value"])


def test_plot_scatter_needs_no_time_axis() -> None:
    frame = pl.DataFrame(
        {"prediction": [1.0, 2.0, 3.0], "target": [1.1, 2.2, 2.9]}
    )
    figure, axes = plot_scatter(frame, x="prediction", y="target")

    assert len(axes.collections) == 1
    assert axes.get_xlabel() == "prediction"
    assert axes.get_ylabel() == "target"
    figure.clear()


def test_plot_scatter_skips_rows_with_non_finite_values() -> None:
    frame = pl.DataFrame(
        {
            "prediction": [1.0, float("nan"), 3.0],
            "target": [1.1, 2.2, float("inf")],
        }
    )
    figure, axes = plot_scatter(frame, x="prediction", y="target")

    offsets = axes.collections[0].get_offsets()
    assert len(offsets) == 1
    figure.clear()


def test_plot_scatter_labels_every_y_series() -> None:
    frame = pl.DataFrame(
        {"a": [1.0, 2.0], "b": [2.0, 1.0], "x": [1.0, 2.0]}
    )
    figure, axes = plot_scatter(frame, x="x", y=["a", "b"])

    assert [collection.get_label() for collection in axes.collections] == ["a", "b"]
    assert axes.get_legend() is not None
    figure.clear()


def test_plot_scatter_legend_can_be_disabled() -> None:
    frame = pl.DataFrame({"a": [1.0, 2.0], "x": [1.0, 2.0]})
    figure, axes = plot_scatter(frame, x="x", y="a", legend=False)

    assert axes.get_legend() is None
    figure.clear()


def test_plot_scatter_rejects_array_valued_columns() -> None:
    frame = pl.DataFrame(
        {"a": [[1.0, 2.0], [3.0, 4.0]], "x": [1.0, 2.0]}
    )
    with pytest.raises(TypeError, match="contains arrays"):
        plot_scatter(frame, x="x", y="a")


def test_plot_scatter_requires_declared_columns() -> None:
    frame = pl.DataFrame({"x": [1.0]})
    with pytest.raises(ValueError, match="Plot columns are not present in the frame"):
        plot_scatter(frame, x="x", y="missing")
    with pytest.raises(ValueError, match="x must be a non-empty column name"):
        plot_scatter(frame, x="", y="x")


def test_plot_scatter_requires_at_least_one_finite_pair() -> None:
    frame = pl.DataFrame(
        {"x": [float("nan")], "y": [float("nan")]}
    )
    with pytest.raises(TypeError, match="at least one row with finite"):
        plot_scatter(frame, x="x", y="y")


def test_plot_scatter_validates_y_max_points_and_bounds() -> None:
    frame = pl.DataFrame({"x": [1.0], "y": [2.0]})
    with pytest.raises(TypeError, match="y must be a column name or a sequence"):
        plot_scatter(frame, x="x", y=3)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must name at least one non-empty column"):
        plot_scatter(frame, x="x", y=[])
    with pytest.raises(ValueError, match="max_points must be"):
        plot_scatter(frame, x="x", y="y", max_points=1)
    with pytest.raises(ValueError, match="lower bound must be below"):
        plot_scatter(frame, x="x", y="y", xlim=(2, 1))


def test_plot_scatter_accepts_lazy_frames() -> None:
    frame = pl.DataFrame({"x": [1.0, 2.0], "y": [2.0, 4.0]}).lazy()
    figure, axes = plot_scatter(frame, x="x", y="y")

    assert len(axes.collections) == 1
    figure.clear()
