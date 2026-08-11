"""Post-hoc metric computation over one or more materialized frames.

``extract_metrics`` never executes a graph and never touches node identity: it
purely resolves each extractor's required columns across the supplied frames and
computes named numeric metrics.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite
from typing import Any

import polars as pl

from iosislib.metrics.extractor import MetricExtractor


def _validate_metric_name(name: str) -> None:
    if not isinstance(name, str) or not name:
        raise ValueError("metric names must be non-empty strings")


def _normalize_frames(frames: Any) -> tuple[pl.DataFrame, ...]:
    if isinstance(frames, pl.DataFrame):
        return (frames,)
    if isinstance(frames, Sequence) and not isinstance(frames, (str, bytes)):
        normalized = tuple(frames)
        if not normalized:
            raise ValueError("at least one frame is required")
        for frame in normalized:
            if not isinstance(frame, pl.DataFrame):
                raise TypeError("frames must be pl.DataFrame instances")
        return normalized
    raise TypeError("frames must be a pl.DataFrame or a sequence of pl.DataFrames")


def _collect_metric_names(extractors: Sequence[MetricExtractor]) -> tuple[str, ...]:
    names: list[str] = []
    seen: set[str] = set()
    for extractor in extractors:
        for name in extractor.metric_names():
            _validate_metric_name(name)
            if name in seen:
                raise ValueError(f"duplicate metric name {name!r}")
            seen.add(name)
            names.append(name)
    return tuple(names)


def _extractor_frame(
    frames: tuple[pl.DataFrame, ...],
    extractor: MetricExtractor,
) -> pl.DataFrame:
    providers: dict[str, pl.DataFrame] = {}
    for column in extractor.required_columns():
        if not isinstance(column, str) or not column:
            raise ValueError("required columns must be non-empty strings")
        matches = [frame for frame in frames if column in frame.columns]
        if not matches:
            raise ValueError(f"no frame provides required column {column!r}")
        if len(matches) > 1:
            raise ValueError(
                f"column {column!r} is ambiguous across multiple frames"
            )
        providers[column] = matches[0]

    unique = {id(frame): frame for frame in providers.values()}
    if len(unique) == 1:
        return next(iter(unique.values()))

    heights = {frame.height for frame in unique.values()}
    if len(heights) != 1:
        raise ValueError(
            "extractor columns span frames with different row counts; "
            "align the frames before extracting"
        )
    selected = [
        providers[column].get_column(column)
        for column in extractor.required_columns()
    ]
    return pl.DataFrame(selected)


def extract_metrics(
    frames: pl.DataFrame | Sequence[pl.DataFrame],
    *extractors: MetricExtractor,
) -> pl.DataFrame:
    """Compute ``extractors`` over ``frames`` and return a one-row wide frame.

    ``frames`` may be a single frame or a sequence of frames (for example a
    prediction node output and a target node output that share a timeline).
    Required columns are resolved across frames by name; a column present in
    more than one frame is ambiguous and raises. If every frame is empty, an
    empty typed frame with the extractor metric columns is returned instead of
    calling any extractor.
    """
    normalized = _normalize_frames(frames)
    if not extractors:
        raise ValueError("at least one extractor is required")
    for extractor in extractors:
        if not isinstance(extractor, MetricExtractor):
            raise TypeError("extractors must be MetricExtractor instances")
    metric_names = _collect_metric_names(extractors)

    if all(frame.is_empty() for frame in normalized):
        return pl.DataFrame(schema={name: pl.Float64 for name in metric_names})

    values: dict[str, float] = {}
    for extractor in extractors:
        frame = _extractor_frame(normalized, extractor)
        for name, value in extractor.extract(frame).items():
            _validate_metric_name(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"metric {name!r} must be numeric")
            numeric = float(value)
            if not isfinite(numeric):
                raise ValueError(f"metric {name!r} must be finite")
            if name in values:
                raise ValueError(f"duplicate metric name {name!r}")
            values[name] = numeric
    return pl.DataFrame([values])


__all__ = ["extract_metrics"]
