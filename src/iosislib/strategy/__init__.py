"""Stable, backend-independent strategy declarations and YAML parsing."""

import json
from importlib.resources import files
from typing import Any, cast

from iosislib.strategy.expand import (
    EXPANSION_FORMAT,
    EXPANSION_VERSION,
    ExpandedStrategy,
    expand_variants,
)
from iosislib.strategy.ir import (
    Input,
    Node,
    Reference,
    STRATEGY_FORMAT,
    STRATEGY_VERSION,
    Strategy,
    StrategyValidationError,
    Value,
)
from iosislib.strategy.lowering import (
    LoweredStrategy,
    OperationKey,
    OperationRegistry,
    OutputBinding,
    StrategyLoweringError,
    builtin_registry,
    lower,
    registry_from_exports,
)
from iosislib.strategy.parser import (
    StrategySyntaxError,
    dump,
    dumps,
    load,
    load_expansion,
    loads,
    loads_expansion,
)


def schema() -> dict[str, Any]:
    """Return the JSON Schema for the current strategy format."""
    document = files(__package__).joinpath("strategy.schema.json").read_text(
        encoding="utf-8"
    )
    return cast(dict[str, Any], json.loads(document))


def expansion_schema() -> dict[str, Any]:
    """Return the JSON Schema for the current expansion format."""
    document = files(__package__).joinpath("expansion.schema.json").read_text(
        encoding="utf-8"
    )
    return cast(dict[str, Any], json.loads(document))


__all__ = [
    "EXPANSION_FORMAT",
    "EXPANSION_VERSION",
    "ExpandedStrategy",
    "Input",
    "LoweredStrategy",
    "Node",
    "OperationKey",
    "OperationRegistry",
    "OutputBinding",
    "Reference",
    "STRATEGY_FORMAT",
    "STRATEGY_VERSION",
    "Strategy",
    "StrategyLoweringError",
    "StrategySyntaxError",
    "StrategyValidationError",
    "Value",
    "builtin_registry",
    "dump",
    "dumps",
    "expand_variants",
    "expansion_schema",
    "load",
    "load_expansion",
    "lower",
    "loads",
    "loads_expansion",
    "registry_from_exports",
    "schema",
]
