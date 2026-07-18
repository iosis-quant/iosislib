"""Stable, backend-independent strategy declarations and YAML parsing."""

import json
from importlib.resources import files
from typing import Any, cast

from iosislib.strategy.ir import (
    Input,
    Node,
    Reference,
    STRATEGY_FORMAT,
    Strategy,
    StrategyValidationError,
    Value,
)
from iosislib.strategy.parser import StrategySyntaxError, dump, dumps, load, loads


def schema() -> dict[str, Any]:
    """Return the JSON Schema for the current strategy format."""
    document = files(__package__).joinpath("strategy.schema.json").read_text(
        encoding="utf-8"
    )
    return cast(dict[str, Any], json.loads(document))


__all__ = [
    "Input",
    "Node",
    "Reference",
    "STRATEGY_FORMAT",
    "Strategy",
    "StrategySyntaxError",
    "StrategyValidationError",
    "Value",
    "dump",
    "dumps",
    "load",
    "loads",
    "schema",
]
