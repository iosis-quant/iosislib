"""Matplotlib visualization helpers for Polars time-series frames."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from numbers import Number
from typing import Any

import polars as pl


@dataclass(frozen=True, slots=True)
class ChartTheme:
    """Presentation defaults used by :func:`plot_frame`."""

    colors: tuple[str, ...] = (
        "#2563EB", "#DB2777", "#059669", "#D97706", "#7C3AED", "#0891B2"
    )
    background: str = "#FFFFFF"
    grid_color: str = "#CBD5E1"
    grid_alpha: float = 0.45
    line_width: float = 2.2
    marker_size: float = 4.5


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
    theme: ChartTheme | None = None,
    legend: bool = True,
) -> tuple[Any, Any]:
    """Plot numeric columns from a time-first Polars frame.

    The first frame column is the time axis. Scalar numeric columns become one
    line each; list and fixed-size array columns become one line per element.
    Null values are rendered as gaps and rows are sorted chronologically.
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

    selected = list(frame.columns[1:] if columns is None else columns)
    missing = [name for name in selected if name not in frame.columns]
    if missing:
        raise ValueError(f"Plot columns are not present in the frame: {missing}")
    if not selected:
        raise ValueError("plot_frame requires at least one value column")

    frame = frame.sort(time_name).drop_nulls(time_name)
    times = frame.get_column(time_name).to_list()
    series = _numeric_series(frame, selected)
    if not series:
        raise TypeError("plot_frame requires at least one numeric value column")

    if ax is None:
        figure, axes = plt.subplots(figsize=(11, 6), constrained_layout=True)
    else:
        axes = ax
        figure = ax.figure
    selected_theme = ChartTheme() if theme is None else theme
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

    axes.grid(True, color=selected_theme.grid_color, alpha=selected_theme.grid_alpha)
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    axes.margins(x=0.02)
    axes.set_title(title or "iosislib graph", loc="left", pad=16, weight="bold")
    axes.set_xlabel(xlabel or time_name)
    if ylabel is not None:
        axes.set_ylabel(ylabel)
    if legend:
        axes.legend(frameon=False, ncol=2 if len(series) > 5 else 1)
    if times and isinstance(times[0], (datetime, date)):
        axes.xaxis_date()
        axes.xaxis.set_major_formatter(
            dates.ConciseDateFormatter(axes.xaxis.get_major_locator())
        )
    return figure, axes


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
    return float(value)


def _is_nan(value: float) -> bool:
    return value != value


def _matplotlib() -> tuple[Any, Any]:
    try:
        import matplotlib.dates as dates
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Matplotlib is required for iosislib.charting; install the 'charting' extra"
        ) from exc
    return plt, dates


__all__ = ["ChartTheme", "plot_frame", "plot_graph"]
