"""Matplotlib visualization helpers for Polars time-series frames.

The default :class:`ChartTheme` is the Iosis dark styleguide: a carbon
``#101010`` background, warm ``#F4F2EC`` text, horizontal-only grid lines, and
the colorblind-safe Okabe-Ito series palette (black dropped, blue lightened
for dark backgrounds). Plotting guards normalize timezone-aware timestamps to
naive UTC, render infinite values as gaps, and keep large y-tick labels
readable with thousands separators.
"""

from __future__ import annotations

import base64
import math

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from io import BytesIO
from numbers import Number
from typing import Any
from urllib.parse import quote

import polars as pl


OKABE_ITO = (
    "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"
)

DARK_PALETTE = (
    "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#4095DB", "#D55E00", "#CC79A7"
)


@dataclass(frozen=True, slots=True)
class ChartTheme:
    """Presentation defaults used by :func:`plot_frame`.

    ``DARK_THEME`` is the default styleguide: a carbon ``#101010`` background,
    warm ``#F4F2EC`` text, ``#2B2B2B`` horizontal-only grid, and the
    colorblind-safe Okabe-Ito palette with the blue lightened for dark
    backgrounds. ``LIGHT_THEME`` keeps the white look with the canonical
    Okabe-Ito palette.
    """

    colors: tuple[str, ...] = DARK_PALETTE
    background: str = "#101010"
    text_color: str = "#F4F2EC"
    grid_color: str = "#2B2B2B"
    grid_alpha: float = 0.55
    grid_axis: str = "y"
    grid_linestyle: str = "-"
    grid_linewidth: float = 0.8
    spine_color: str = "#2B2B2B"
    line_width: float = 2.2
    marker_size: float = 4.5


DARK_THEME = ChartTheme()

LIGHT_THEME = ChartTheme(
    colors=OKABE_ITO,
    background="#FFFFFF",
    text_color="#1F2937",
    grid_color="#CBD5E1",
    grid_alpha=0.45,
    grid_axis="y",
    grid_linestyle="-",
    grid_linewidth=0.8,
    spine_color="#CBD5E1",
    line_width=2.2,
    marker_size=4.5,
)


def _resolve_theme(theme: ChartTheme | str | None) -> ChartTheme:
    if theme is None:
        return DARK_THEME
    if isinstance(theme, ChartTheme):
        return theme
    if isinstance(theme, str):
        if theme == "dark":
            return DARK_THEME
        if theme == "light":
            return LIGHT_THEME
        raise ValueError("theme must be 'dark', 'light', or a ChartTheme instance")
    raise TypeError("theme must be a ChartTheme, a theme name, or None")


def plot_graph(graph: Any, *, ax: Any | None = None, **kwargs: Any) -> tuple[Any, Any]:
    """Execute ``graph`` and plot its root output."""

    if not hasattr(graph, "execute"):
        raise TypeError("plot_graph expects a Graph-like object with execute()")
    executor = kwargs.pop("executor", None)
    return plot_frame(graph.execute(executor=executor), ax=ax, **kwargs)


def plot_frame(
    frame: pl.DataFrame | pl.LazyFrame,
    *,
    ax: Any | None = None,
    columns: Sequence[str] | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    time_column: str | None = None,
    theme: ChartTheme | str | None = None,
    legend: bool = True,
    max_points: int | None = 5000,
    xlim: tuple[Number, Number] | None = None,
    ylim: tuple[Number, Number] | None = None,
    y_tick_format: str | Any | None = None,
) -> tuple[Any, Any]:
    """Plot numeric columns from a time-first Polars frame.

    The first frame column is the time axis. Scalar numeric columns become one
    line each; list and fixed-size array columns become one line per element.
    Null values are rendered as gaps and rows are sorted chronologically.
    ``max_points`` bounds the x-axis rows, retaining the first and last points.

    ``theme`` selects the ``"dark"`` (default) or ``"light"`` styleguide preset,
    or accepts a custom :class:`ChartTheme`. ``xlim``/``ylim`` pin the axis
    ranges and ``y_tick_format`` overrides y-tick labels. Timezone-aware
    timestamps are normalized to naive UTC and infinite values render as gaps.
    """

    return _plot_time_frame(
        frame,
        label="plot_frame",
        ax=ax,
        columns=columns,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        time_column=time_column,
        theme=theme,
        legend=legend,
        max_points=max_points,
        xlim=xlim,
        ylim=ylim,
        y_tick_format=y_tick_format,
        draw="line",
        fill=False,
    )


def plot_equity(
    frame: pl.DataFrame | pl.LazyFrame,
    *,
    ax: Any | None = None,
    columns: Sequence[str] | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    time_column: str | None = None,
    theme: ChartTheme | str | None = None,
    legend: bool = True,
    max_points: int | None = 5000,
    xlim: tuple[Number, Number] | None = None,
    ylim: tuple[Number, Number] | None = None,
    y_tick_format: str | Any | None = None,
) -> tuple[Any, Any]:
    """Plot time-series value columns as equity curves.

    Identical to :func:`plot_frame`, except the area between each line and
    that line's first finite value is shaded, giving the standard equity
    presentation where gains and losses read against the starting value.
    """

    return _plot_time_frame(
        frame,
        label="plot_equity",
        ax=ax,
        columns=columns,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        time_column=time_column,
        theme=theme,
        legend=legend,
        max_points=max_points,
        xlim=xlim,
        ylim=ylim,
        y_tick_format=y_tick_format,
        draw="line",
        fill=True,
    )


def plot_bars(
    frame: pl.DataFrame | pl.LazyFrame,
    *,
    ax: Any | None = None,
    columns: Sequence[str] | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    time_column: str | None = None,
    theme: ChartTheme | str | None = None,
    legend: bool = True,
    max_points: int | None = 5000,
    xlim: tuple[Number, Number] | None = None,
    ylim: tuple[Number, Number] | None = None,
    y_tick_format: str | Any | None = None,
) -> tuple[Any, Any]:
    """Plot time-series value columns as bars over a Date/Datetime axis.

    Multiple value columns render side by side within each time slot; the slot
    width follows the median time delta. The time column must be a Date or
    Datetime column.
    """

    return _plot_time_frame(
        frame,
        label="plot_bars",
        ax=ax,
        columns=columns,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        time_column=time_column,
        theme=theme,
        legend=legend,
        max_points=max_points,
        xlim=xlim,
        ylim=ylim,
        y_tick_format=y_tick_format,
        draw="bars",
        fill=False,
    )


def plot_scatter(
    frame: pl.DataFrame | pl.LazyFrame,
    *,
    x: str,
    y: str | Sequence[str],
    ax: Any | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    theme: ChartTheme | str | None = None,
    legend: bool = True,
    max_points: int | None = 5000,
    xlim: tuple[Number, Number] | None = None,
    ylim: tuple[Number, Number] | None = None,
    y_tick_format: str | Any | None = None,
) -> tuple[Any, Any]:
    """Scatter a scalar ``x`` column against one or more scalar ``y`` columns.

    No time axis is required: any frame carrying the declared scalar numeric
    columns works, so any output frame can be plotted. Rows whose ``x`` or
    ``y`` values are null or non-finite are skipped, and ``max_points`` bounds
    the sampled rows. ``xlim``/``ylim`` pin the axis ranges and
    ``y_tick_format`` overrides y-tick labels.
    """

    selected_theme = _resolve_theme(theme)
    if isinstance(frame, pl.LazyFrame):
        frame = frame.collect()
    if not isinstance(frame, pl.DataFrame):
        raise TypeError("plot_scatter expects a Polars DataFrame or LazyFrame")
    if not isinstance(x, str) or not x:
        raise ValueError("x must be a non-empty column name")
    y_names = _y_names(y)
    _validate_max_points(max_points)
    _validate_bounds(xlim, ylim)
    missing = [name for name in (x, *y_names) if name not in frame.columns]
    if missing:
        raise ValueError(f"Plot columns are not present in the frame: {missing}")
    if max_points is not None and frame.height > max_points:
        indices = [
            round(index * (frame.height - 1) / (max_points - 1))
            for index in range(max_points)
        ]
        frame = frame.gather(indices)
    x_values = _scalar_floats(frame, x, label="plot_scatter")
    plotted: list[tuple[str, list[float], list[float]]] = []
    for name in y_names:
        y_values = _scalar_floats(frame, name, label="plot_scatter")
        pairs = [
            (x_value, y_value)
            for x_value, y_value in zip(x_values, y_values)
            if not _is_nan(x_value) and not _is_nan(y_value)
        ]
        if pairs:
            paired_x, paired_y = (list(pair) for pair in zip(*pairs))
            plotted.append((name, paired_x, paired_y))
    if not plotted:
        raise TypeError(
            "plot_scatter requires at least one row with finite x and y values"
        )

    figure, axes = _new_axes(ax)
    for index, (name, paired_x, paired_y) in enumerate(plotted):
        axes.scatter(
            paired_x,
            paired_y,
            s=selected_theme.marker_size**2 * 2,
            label=name,
            color=selected_theme.colors[index % len(selected_theme.colors)],
            edgecolors="none",
            alpha=0.85,
        )
    _style_axes(
        axes,
        figure,
        selected_theme,
        title=title or "iosislib graph",
        xlabel=xlabel or x,
        ylabel=(
            ylabel
            if ylabel is not None
            else (y_names[0] if len(y_names) == 1 else None)
        ),
    )
    if legend:
        axes.legend(
            frameon=False,
            ncol=2 if len(plotted) > 5 else 1,
            labelcolor=selected_theme.text_color,
        )
    if xlim is not None:
        axes.set_xlim(*xlim)
    if ylim is not None:
        axes.set_ylim(*ylim)
    _format_y_ticks(
        axes,
        [value for _, _, values in plotted for value in values],
        y_tick_format,
    )
    return figure, axes


def _plot_time_frame(
    frame: pl.DataFrame | pl.LazyFrame,
    *,
    label: str,
    ax: Any | None,
    columns: Sequence[str] | None,
    title: str | None,
    xlabel: str | None,
    ylabel: str | None,
    time_column: str | None,
    theme: ChartTheme | str | None,
    legend: bool,
    max_points: int | None,
    xlim: tuple[Number, Number] | None,
    ylim: tuple[Number, Number] | None,
    y_tick_format: str | Any | None,
    draw: str,
    fill: bool,
) -> tuple[Any, Any]:
    selected_theme = _resolve_theme(theme)
    time_name, times, series = _prepare_time_frame(
        frame,
        label=label,
        columns=columns,
        time_column=time_column,
        max_points=max_points,
        require_datetime=draw == "bars",
    )
    _validate_bounds(xlim, ylim)
    if not series:
        raise TypeError(f"{label} requires at least one numeric value column")

    figure, axes = _new_axes(ax)
    if draw == "bars":
        _draw_bars(axes, times, series, selected_theme)
    else:
        if fill:
            for index, values in enumerate(series.values()):
                base = next((value for value in values if not _is_nan(value)), None)
                if base is None:
                    continue
                axes.fill_between(
                    times,
                    values,
                    base,
                    color=selected_theme.colors[index % len(selected_theme.colors)],
                    alpha=0.16,
                    linewidth=0,
                    label="_fill",
                )
        marker = "o" if len(times) <= 80 else None
        for index, (name, values) in enumerate(series.items()):
            axes.plot(
                times, values, label=name,
                color=selected_theme.colors[index % len(selected_theme.colors)],
                linewidth=selected_theme.line_width, marker=marker,
                markersize=selected_theme.marker_size, markeredgewidth=0,
            )

    _style_axes(
        axes,
        figure,
        selected_theme,
        title=title or "iosislib graph",
        xlabel=xlabel or time_name,
        ylabel=ylabel,
    )
    if legend:
        axes.legend(
            frameon=False,
            ncol=2 if len(series) > 5 else 1,
            labelcolor=selected_theme.text_color,
        )
    if xlim is not None:
        axes.set_xlim(*xlim)
    if ylim is not None:
        axes.set_ylim(*ylim)
    _format_y_ticks(
        axes,
        [value for values in series.values() for value in values],
        y_tick_format,
    )
    if times and isinstance(times[0], (datetime, date)):
        _apply_date_axis(axes, times)
    return figure, axes


def _prepare_time_frame(
    frame: pl.DataFrame | pl.LazyFrame,
    *,
    label: str,
    columns: Sequence[str] | None,
    time_column: str | None,
    max_points: int | None,
    require_datetime: bool,
) -> tuple[str, list[Any], dict[str, list[float]]]:
    if isinstance(frame, pl.LazyFrame):
        frame = frame.collect()
    if not isinstance(frame, pl.DataFrame):
        raise TypeError(f"{label} expects a Polars DataFrame or LazyFrame")
    if not frame.columns:
        raise ValueError(f"{label} requires a frame with a time column")

    time_name = frame.columns[0] if time_column is None else time_column
    if time_name not in frame.columns:
        raise ValueError(f"Time column {time_name!r} is not present in the frame")
    if time_column is not None and frame.columns[0] != time_column:
        raise ValueError("The time column must be the first frame column")
    if frame.schema[time_name] not in (pl.Date, pl.Datetime, pl.Time):
        raise TypeError("The first frame column must contain date/time values")
    if require_datetime and frame.schema[time_name] == pl.Time:
        raise TypeError(f"{label} requires a Date or Datetime time column")
    _validate_max_points(max_points)

    selected = list(frame.columns[1:] if columns is None else columns)
    missing = [name for name in selected if name not in frame.columns]
    if missing:
        raise ValueError(f"Plot columns are not present in the frame: {missing}")
    if not selected:
        raise ValueError(f"{label} requires at least one value column")

    frame = frame.sort(time_name).drop_nulls(time_name)
    if max_points is not None and frame.height > max_points:
        indices = [
            round(index * (frame.height - 1) / (max_points - 1))
            for index in range(max_points)
        ]
        frame = frame.gather(indices)
    times = _normalize_times(frame.get_column(time_name).to_list())
    series = _numeric_series(frame, selected)
    return time_name, times, series


def _new_axes(ax: Any | None) -> tuple[Any, Any]:
    plt, _ = _matplotlib()
    if ax is None:
        figure, axes = plt.subplots(figsize=(11, 6), constrained_layout=True)
    else:
        axes = ax
        figure = ax.figure
    return figure, axes


def _style_axes(
    axes: Any,
    figure: Any,
    theme: ChartTheme,
    *,
    title: str | None,
    xlabel: str | None,
    ylabel: str | None,
) -> None:
    axes.set_facecolor(theme.background)
    figure.patch.set_facecolor(theme.background)
    axes.grid(
        True,
        color=theme.grid_color,
        alpha=theme.grid_alpha,
        axis=theme.grid_axis,
        linestyle=theme.grid_linestyle,
        linewidth=theme.grid_linewidth,
    )
    axes.tick_params(axis="both", colors=theme.text_color, labelsize=9)
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    axes.spines["left"].set_color(theme.spine_color)
    axes.spines["bottom"].set_color(theme.spine_color)
    axes.margins(x=0.02, y=0.05)
    axes.set_title(
        title or "iosislib graph",
        loc="left",
        pad=16,
        weight="bold",
        fontsize=14,
        color=theme.text_color,
    )
    if xlabel is not None:
        axes.set_xlabel(xlabel, color=theme.text_color, fontsize=11)
    if ylabel is not None:
        axes.set_ylabel(ylabel, color=theme.text_color, fontsize=11)


def _apply_date_axis(axes: Any, times: list[Any]) -> None:
    _, dates = _matplotlib()
    axes.xaxis_date()
    axes.xaxis.set_major_formatter(
        dates.ConciseDateFormatter(axes.xaxis.get_major_locator())
    )


def _draw_bars(
    axes: Any,
    times: list[Any],
    series: dict[str, list[float]],
    theme: ChartTheme,
) -> None:
    slot = _bar_slot_days(times)
    count = max(len(series), 1)
    width = slot * 0.9 / count
    for index, (name, values) in enumerate(series.items()):
        offset = (index - (len(series) - 1) / 2) * width
        positions = [value + timedelta(days=offset) for value in times]
        axes.bar(
            positions,
            values,
            width=width,
            label=name,
            color=theme.colors[index % len(theme.colors)],
            alpha=0.92,
            linewidth=0,
        )


def _bar_slot_days(times: list[Any]) -> float:
    deltas = [
        (right - left).total_seconds()
        for left, right in zip(times, times[1:])
        if isinstance(left, (date, datetime)) and isinstance(right, (date, datetime))
    ]
    positive = sorted(delta for delta in deltas if delta > 0)
    if not positive:
        return 0.5
    return max(positive[len(positive) // 2] / 86400.0, 1e-6)


def _validate_max_points(max_points: int | None) -> None:
    if max_points is not None and (
        not isinstance(max_points, int) or isinstance(max_points, bool) or max_points < 2
    ):
        raise ValueError("max_points must be None or an integer greater than or equal to 2")


def _validate_bounds(
    xlim: tuple[Number, Number] | None,
    ylim: tuple[Number, Number] | None,
) -> None:
    for name, bounds in (("xlim", xlim), ("ylim", ylim)):
        if bounds is not None and (
            not isinstance(bounds, tuple)
            or len(bounds) != 2
            or not all(
                isinstance(value, Number) and not isinstance(value, bool) for value in bounds
            )
        ):
            raise ValueError(f"{name} must be a pair of numeric bounds")
        if bounds is not None and bounds[0] >= bounds[1]:
            raise ValueError(f"{name} lower bound must be below the upper bound")


def _y_names(y: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(y, str):
        names: tuple[str, ...] = (y,)
    elif isinstance(y, Sequence) and not isinstance(y, (bytes, bytearray)):
        names = tuple(y)
    else:
        raise TypeError("y must be a column name or a sequence of column names")
    if not names or not all(isinstance(name, str) and name for name in names):
        raise ValueError("y must name at least one non-empty column")
    return names


def _scalar_floats(frame: pl.DataFrame, name: str, *, label: str) -> list[float]:
    values = frame.get_column(name).to_list()
    if any(isinstance(value, (list, tuple)) for value in values):
        raise TypeError(
            f"{label} requires scalar numeric columns; {name!r} contains arrays"
        )
    return [_as_float(value) for value in values]


def figure_to_svg(figure: Any, *, data_uri: bool = False) -> str:
    """Serialize a Matplotlib figure as SVG or a URI-safe SVG data URI."""

    buffer = BytesIO()
    figure.savefig(
        buffer,
        format="svg",
        bbox_inches="tight",
        pad_inches=0.15,
        facecolor=figure.get_facecolor(),
    )
    svg = buffer.getvalue().decode("utf-8")
    if not data_uri:
        return svg
    return "data:image/svg+xml;charset=utf-8," + quote(svg, safe="~()*!.'-_")


def figure_to_svg_data_uri(figure: Any) -> str:
    """Return ``figure`` as an embeddable ``data:image/svg+xml`` URI."""

    return figure_to_svg(figure, data_uri=True)


def _numeric_series(frame: pl.DataFrame, columns: Sequence[str]) -> dict[str, list[float]]:
    output: dict[str, list[float]] = {}
    for name in columns:
        values = frame.get_column(name).to_list()
        if values and isinstance(values[0], (list, tuple)):
            width = max((len(value) for value in values if value is not None), default=0)
            for element in range(width):
                expanded = [
                    _as_float(value[element])
                    if value is not None and len(value) > element else float("nan")
                    for value in values
                ]
                if any(not _is_nan(value) for value in expanded):
                    output[f"{name}[{element}]"] = expanded
        else:
            expanded = [_as_float(value) for value in values]
            if any(not _is_nan(value) for value in expanded):
                output[name] = expanded
    return output


def _as_float(value: Any) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, bool) or not isinstance(value, Number):
        raise TypeError("Plot value columns must contain numeric scalars or numeric arrays")
    result = float(value)
    return result if math.isfinite(result) else float("nan")


def _is_nan(value: float) -> bool:
    return value != value


def _is_integral(value: float) -> bool:
    return value.is_integer()


def _normalize_times(values: Sequence[Any]) -> list[Any]:
    """Return timestamps with timezone-aware datetimes as naive UTC values."""

    output: list[Any] = []
    for value in values:
        if isinstance(value, datetime):
            offset = value.tzinfo.utcoffset(value) if value.tzinfo is not None else None
            if offset is not None:
                output.append(value.astimezone(timezone.utc).replace(tzinfo=None))
                continue
        output.append(value)
    return output


def _format_y_ticks(axes: Any, values: list[float], y_tick_format: Any | None) -> None:
    if y_tick_format is not None:
        axes.yaxis.set_major_formatter(y_tick_format)
        return
    finite = [value for value in values if not _is_nan(value)]
    if not finite:
        return
    if max(abs(value) for value in finite) >= 10_000 and all(
        _is_integral(value) for value in finite
    ):
        from matplotlib import ticker

        axes.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda value, _: f"{value:,.0f}")
        )


def _matplotlib() -> tuple[Any, Any]:
    try:
        import matplotlib.dates as dates
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Matplotlib is required for iosislib.charting; install matplotlib to use it"
        ) from exc
    return plt, dates


__all__ = [
    "ChartTheme",
    "DARK_THEME",
    "LIGHT_THEME",
    "figure_to_png_data_uri",
    "figure_to_svg",
    "figure_to_svg_data_uri",
    "plot_bars",
    "plot_equity",
    "plot_frame",
    "plot_graph",
    "plot_scatter",
    "to_png",
]


def to_png(source: Any, *, dpi: int = 300, **kwargs: Any) -> bytes:
    """Render a frame or graph to high-resolution PNG bytes.

    ``source`` may be a Polars DataFrame/LazyFrame or any graph-like object
    exposing ``execute()``. Plotting options, including ``max_points``, are
    forwarded to the existing chart helpers.
    """

    if not isinstance(dpi, int) or isinstance(dpi, bool) or dpi <= 0:
        raise ValueError("dpi must be a positive integer")
    plt, _ = _matplotlib()
    owns_figure = kwargs.get("ax") is None
    if hasattr(source, "execute"):
        figure, _ = plot_graph(source, **kwargs)
    else:
        figure, _ = plot_frame(source, **kwargs)
    try:
        buffer = BytesIO()
        figure.savefig(
            buffer,
            format="png",
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.15,
            facecolor=figure.get_facecolor(),
        )
        return buffer.getvalue()
    finally:
        if owns_figure:
            plt.close(figure)


def figure_to_png_data_uri(figure: Any) -> str:
    """Return a high-resolution Matplotlib figure as a PNG data URI."""

    buffer = BytesIO()
    figure.savefig(
        buffer,
        format="png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.15,
        facecolor=figure.get_facecolor(),
    )
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return "data:image/png;base64," + encoded
