"""Bulk metric-extraction contract for post-hoc evaluation frames.

Extractors are ordinary value objects, not graph nodes: they consume a
materialized, time-sorted :class:`pl.DataFrame` and return named numeric
metrics. They never run inside the graph and never affect node identity.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import fields, is_dataclass
from typing import Any, ClassVar, Mapping, cast

import polars as pl

from iosislib.core.utils import _canonical_json, _serialize_value


class MetricExtractor(ABC):
    """Compute named metrics from one materialized, time-sorted frame."""

    _SERIALIZE_WITH_TO_DICT: ClassVar[bool] = True
    VERSION: ClassVar[str]

    @abstractmethod
    def required_columns(self) -> tuple[str, ...]:
        """Value columns this extractor needs present in the frame."""

    @abstractmethod
    def metric_names(self) -> tuple[str, ...]:
        """Names of the metrics this extractor emits, in output order."""

    @abstractmethod
    def extract(self, frame: pl.DataFrame) -> Mapping[str, float]:
        """Return each named metric computed over ``frame``.

        ``frame`` is assumed to be materialized and time-sorted. Implementations
        must raise ``ValueError`` for insufficient data and invalid structure,
        and ``TypeError`` for incompatible column types.
        """

    def to_dict(self) -> dict[str, Any]:
        values: dict[str, Any] = (
            {
                item.name: _serialize_value(getattr(self, item.name))
                for item in fields(cast(Any, self))
            }
            if is_dataclass(self)
            else {}
        )
        return {
            "type": f"{type(self).__module__}.{type(self).__qualname__}",
            "version": self.VERSION,
            **values,
        }

    def __str__(self) -> str:
        return _canonical_json(self.to_dict())


__all__ = ["MetricExtractor"]
