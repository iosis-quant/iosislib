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
from datetime import date, datetime, timezone
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

    plt, dates = _matplotlib()
    if isinstance(frame, pl.LazyFrame):
        frame = frame.collect()
    if not isinstance(frame, pl.DataFrame):
        raise TypeError("plot_frame expects a Polars DataFrame or LazyFrame")
    if not frame.columns:
        raise ValueError("plot_frame requires a frame with a time column")

    time_name = frame.columns[0] if time_column is None else time_column
    if time_name not in frame.columns:
        raise ValueError(f"Time column {time_name!r} is not present in the frame")
    if time_column is not None and frame.columns[0] != time_column:
        raise ValueError("The time column must be the first frame column")
    if frame.schema[time_name] not in (pl.Date, pl.Datetime, pl.Time):
        raise TypeError("The first frame column must contain date/time values")
    if max_points is not None and (
        not isinstance(max_points, int) or isinstance(max_points, bool) or max_points < 2
    ):
        raise ValueError("max_points must be None or an integer greater than or equal to 2")
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

    selected = list(frame.columns[1:] if columns is None else columns)
    missing = [name for name in selected if name not in frame.columns]
    if missing:
        raise ValueError(f"Plot columns are not present in the frame: {missing}")
    if not selected:
        raise ValueError("plot_frame requires at least one value column")

    frame = frame.sort(time_name).drop_nulls(time_name)
    if max_points is not None and frame.height > max_points:
        indices = [
            round(index * (frame.height - 1) / (max_points - 1))
            for index in range(max_points)
        ]
        frame = frame.gather(indices)
    times = _normalize_times(frame.get_column(time_name).to_list())
    series = _numeric_series(frame, selected)
    if not series:
        raise TypeError("plot_frame requires at least one numeric value column")

    if ax is None:
        figure, axes = plt.subplots(figsize=(11, 6), constrained_layout=True)
    else:
        axes = ax
        figure = ax.figure
    selected_theme = _resolve_theme(theme)
    axes.set_facecolor(selected_theme.background)
    figure.patch.set_facecolor(selected_theme.background)
    marker = "o" if len(times) <= 80 else None
    for index, (name, values) in enumerate(series.items()):
        axes.plot(
            times, values, label=name,
            color=selected_theme.colors[index % len(selected_theme.colors)],
            linewidth=selected_theme.line_width, marker=marker,
            markersize=selected_theme.marker_size, markeredgewidth=0,
        )

    axes.grid(
        True,
        color=selected_theme.grid_color,
        alpha=selected_theme.grid_alpha,
        axis=selected_theme.grid_axis,
        linestyle=selected_theme.grid_linestyle,
        linewidth=selected_theme.grid_linewidth,
    )
    axes.tick_params(axis="both", colors=selected_theme.text_color, labelsize=9)
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    axes.spines["left"].set_color(selected_theme.spine_color)
    axes.spines["bottom"].set_color(selected_theme.spine_color)
    axes.margins(x=0.02, y=0.05)
    axes.set_title(
        title or "iosislib graph",
        loc="left",
        pad=16,
        weight="bold",
        fontsize=14,
        color=selected_theme.text_color,
    )
    axes.set_xlabel(
        xlabel or time_name,
        color=selected_theme.text_color,
        fontsize=11,
    )
    if ylabel is not None:
        axes.set_ylabel(ylabel, color=selected_theme.text_color, fontsize=11)
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
        axes.xaxis_date()
        axes.xaxis.set_major_formatter(
            dates.ConciseDateFormatter(axes.xaxis.get_major_locator())
        )
    return figure, axes


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
    "plot_frame",
    "plot_graph",
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
