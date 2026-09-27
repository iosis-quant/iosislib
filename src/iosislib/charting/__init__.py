"""Small, polished Matplotlib helpers for visualizing graph results."""

from iosislib.charting.charts import (
    CHART_KINDS,
    ChartDecl,
    parse_chart_decls,
    placeholder_chart_svg,
    render_chart,
    render_chart_decls,
)
from iosislib.charting.plot import (
    ChartTheme,
    DARK_THEME,
    LIGHT_THEME,
    figure_to_png_data_uri,
    figure_to_svg,
    figure_to_svg_data_uri,
    plot_bars,
    plot_equity,
    plot_frame,
    plot_graph,
    plot_scatter,
    to_png,
)

__all__ = [
    "CHART_KINDS",
    "ChartDecl",
    "ChartTheme",
    "DARK_THEME",
    "LIGHT_THEME",
    "figure_to_png_data_uri",
    "figure_to_svg",
    "figure_to_svg_data_uri",
    "parse_chart_decls",
    "placeholder_chart_svg",
    "plot_bars",
    "plot_equity",
    "plot_frame",
    "plot_graph",
    "plot_scatter",
    "render_chart",
    "render_chart_decls",
    "to_png",
]
