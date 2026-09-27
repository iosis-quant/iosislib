"""Parse and render declared charts from a strategy's ``metadata.charts`` block.

Chart declarations are portable YAML/JSON data validated by
:func:`parse_chart_decls`. Rendering takes a plain Polars frame, so any
strategy output frame can be charted: :func:`render_chart` draws one
declaration as SVG and returns a deterministic placeholder SVG when drawing
fails (missing columns, no finite points, Matplotlib errors).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from xml.sax.saxutils import escape

import polars as pl

from iosislib.charting.plot import (
    _matplotlib,
    figure_to_svg,
    plot_bars,
    plot_equity,
    plot_frame,
    plot_scatter,
)

CHART_KINDS = frozenset({"bars", "equity", "line", "scatter"})

_DECL_FIELDS = frozenset({"columns", "kind", "name", "output", "title", "x", "y"})

_ARTIFACT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True, slots=True)
class ChartDecl:
    """One validated chart declaration from ``metadata.charts``."""

    kind: str
    name: str
    output: str | None = None
    columns: tuple[str, ...] | None = None
    x: str | None = None
    y: tuple[str, ...] | None = None
    title: str | None = None


def parse_chart_decls(
    metadata: object,
    *,
    outputs: Sequence[str] | None = None,
) -> tuple[ChartDecl, ...]:
    """Validate the ``charts`` list of a strategy metadata mapping.

    ``outputs`` supplies the strategy's declared output names; when given, each
    chart's ``output`` must reference one of them and may be omitted only when
    the strategy declares exactly one output. Returns ``()`` when no charts are
    declared. Raises ``TypeError``/``ValueError`` for malformed declarations.
    """

    if metadata is None:
        metadata = {}
    if not isinstance(metadata, Mapping):
        raise TypeError("$.metadata must be a mapping")
    if outputs is not None:
        if isinstance(outputs, (str, bytes)) or not isinstance(outputs, Sequence):
            raise TypeError("outputs must be a sequence of output names")
        outputs = tuple(outputs)
        if not all(isinstance(name, str) and name for name in outputs):
            raise TypeError("outputs must contain only non-empty strings")

    raw = metadata.get("charts", ())
    if raw is None:
        raw = ()
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise ValueError("$.metadata.charts must be a list of chart declarations")

    decls: list[ChartDecl] = []
    names: set[str] = set()
    for index, entry in enumerate(raw):
        path = f"$.metadata.charts[{index}]"
        if not isinstance(entry, Mapping):
            raise ValueError(f"{path} must be a mapping")
        if not all(isinstance(key, str) for key in entry):
            raise ValueError(f"{path} field names must be strings")
        extra = sorted(set(entry) - _DECL_FIELDS)
        if extra:
            raise ValueError(f"{path} has unknown field(s): {', '.join(extra)}")

        kind = entry.get("kind")
        if not isinstance(kind, str) or kind not in CHART_KINDS:
            raise ValueError(
                f"{path}.kind must be one of {sorted(CHART_KINDS)}, got {kind!r}"
            )
        output = entry.get("output")
        if output is not None and (not isinstance(output, str) or not output):
            raise ValueError(f"{path}.output must be a non-empty string")
        if output is None and outputs is not None:
            if len(outputs) == 1:
                output = outputs[0]
            elif not outputs:
                raise ValueError(f"{path} cannot declare a chart: the strategy has no outputs")
            else:
                raise ValueError(
                    f"{path}.output is required when the strategy declares "
                    f"multiple outputs: {list(outputs)}"
                )
        if output is not None and outputs is not None and output not in outputs:
            raise ValueError(
                f"{path}.output references unknown output {output!r}; "
                f"declared outputs: {list(outputs)}"
            )

        columns = _string_tuple(entry.get("columns"), f"{path}.columns")
        x = entry.get("x")
        if x is not None and (not isinstance(x, str) or not x):
            raise ValueError(f"{path}.x must be a non-empty column name")
        y = _string_tuple(entry.get("y"), f"{path}.y", allow_scalar=True)
        title = entry.get("title")
        if title is not None and not isinstance(title, str):
            raise ValueError(f"{path}.title must be a string")

        if kind == "scatter":
            if columns is not None:
                raise ValueError(f"{path}: scatter charts declare x/y, not columns")
            if x is None or y is None:
                raise ValueError(f"{path}: scatter charts require x and y")
        elif x is not None or y is not None:
            raise ValueError(f"{path}: x/y are only valid for scatter charts")

        name = entry.get("name")
        if name is None:
            name = f"{output or 'chart'}_{kind}"
        if not isinstance(name, str) or not name:
            raise ValueError(f"{path}.name must be a non-empty string")
        if not _ARTIFACT_NAME.fullmatch(name):
            raise ValueError(
                f"{path}.name must start with a letter and contain only letters, "
                f"digits, '.', '_' or '-'; got {name!r}"
            )
        if name in names:
            raise ValueError(f"{path}.name duplicates chart name {name!r}")
        names.add(name)

        decls.append(
            ChartDecl(
                kind=kind,
                name=name,
                output=output,
                columns=columns,
                x=x,
                y=y,
                title=title,
            )
        )
    return tuple(decls)


def render_chart(
    frame: pl.DataFrame | pl.LazyFrame,
    decl: ChartDecl,
    *,
    theme: Any | None = None,
) -> str:
    """Render one declared chart from an output frame as an SVG string.

    ``frame`` is the plain Polars frame of the chart's output. Any drawing
    failure (missing columns, no finite points, Matplotlib errors) returns a
    deterministic placeholder SVG carrying the failure message instead of
    raising.
    """

    if not isinstance(decl, ChartDecl):
        raise TypeError("decl must be a ChartDecl")
    try:
        plt, _ = _matplotlib()
    except Exception as exc:
        return placeholder_chart_svg(f"{decl.name}: {type(exc).__name__}: {exc}")
    snapshot = set(plt.get_fignums())
    try:
        figure, _ = _draw_chart(frame, decl, theme)
    except Exception as exc:
        _close_new_figures(plt, snapshot)
        return placeholder_chart_svg(f"{decl.name}: {type(exc).__name__}: {exc}")
    try:
        svg = figure_to_svg(figure)
    except Exception as exc:
        svg = placeholder_chart_svg(f"{decl.name}: {type(exc).__name__}: {exc}")
    finally:
        plt.close(figure)
    return svg


def _close_new_figures(plt: Any, snapshot: set[int]) -> None:
    for number in sorted(set(plt.get_fignums()) - snapshot):
        plt.close(number)


def render_chart_decls(
    frames: Mapping[str, pl.DataFrame | pl.LazyFrame],
    decls: Sequence[ChartDecl],
    *,
    theme: Any | None = None,
) -> dict[str, str]:
    """Render declarations against the output frames keyed by output name.

    A declaration whose frame is absent renders as a placeholder SVG so the
    run still succeeds. Returns a mapping of chart name to SVG string.
    """

    if not isinstance(frames, Mapping):
        raise TypeError("frames must be a mapping of output name to frame")
    if isinstance(decls, (str, bytes)) or not isinstance(decls, Sequence):
        raise TypeError("decls must be a sequence of ChartDecl values")
    rendered: dict[str, str] = {}
    for decl in decls:
        if not isinstance(decl, ChartDecl):
            raise TypeError("decls must contain only ChartDecl values")
        if decl.output is None:
            if len(frames) == 1:
                frame = next(iter(frames.values()))
            else:
                rendered[decl.name] = placeholder_chart_svg(
                    f"{decl.name}: chart does not name an output"
                )
                continue
        elif decl.output not in frames:
            rendered[decl.name] = placeholder_chart_svg(
                f"{decl.name}: output {decl.output!r} has no frame"
            )
            continue
        else:
            frame = frames[decl.output]
        rendered[decl.name] = render_chart(frame, decl, theme=theme)
    return rendered


def placeholder_chart_svg(message: str) -> str:
    """Return a deterministic dark placeholder SVG carrying ``message``."""

    lines = _wrap(str(message), 96)[:6]
    tspans = "".join(
        f'<text x="40" y="{120 + index * 24}" fill="#9A968C" '
        f'font-family="monospace" font-size="13">{escape(line)}</text>'
        for index, line in enumerate(lines)
    )
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="800" height="400" '
        'viewBox="0 0 800 400" role="img">'
        '<rect width="800" height="400" fill="#101010"/>'
        '<text x="40" y="64" fill="#F4F2EC" font-family="monospace" '
        'font-size="18" font-weight="bold">chart unavailable</text>'
        f"{tspans}"
        "</svg>"
    )


def _draw_chart(frame: Any, decl: ChartDecl, theme: Any | None) -> tuple[Any, Any]:
    title = decl.title or decl.name
    if decl.kind == "scatter":
        assert decl.x is not None and decl.y is not None
        return plot_scatter(
            frame,
            x=decl.x,
            y=decl.y,
            title=title,
            theme=theme,
            legend=len(decl.y) > 1,
        )
    if decl.kind == "line":
        return plot_frame(frame, columns=decl.columns, title=title, theme=theme)
    if decl.kind == "bars":
        return plot_bars(frame, columns=decl.columns, title=title, theme=theme)
    if decl.kind == "equity":
        return plot_equity(frame, columns=decl.columns, title=title, theme=theme)
    raise ValueError(f"unsupported chart kind {decl.kind!r}")


def _string_tuple(
    value: object,
    path: str,
    *,
    allow_scalar: bool = False,
) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        if not allow_scalar:
            raise ValueError(f"{path} must be a list of column names")
        items: Sequence[object] = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        items = value
    else:
        raise ValueError(f"{path} must be a list of column names")
    names = tuple(items)
    if not names or not all(isinstance(name, str) and name for name in names):
        raise ValueError(f"{path} must contain at least one non-empty column name")
    if len(set(names)) != len(names):
        raise ValueError(f"{path} contains duplicate column names")
    return names


def _wrap(text: str, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines or [""]


__all__ = [
    "CHART_KINDS",
    "ChartDecl",
    "parse_chart_decls",
    "placeholder_chart_svg",
    "render_chart",
    "render_chart_decls",
]
