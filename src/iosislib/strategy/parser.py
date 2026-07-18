from __future__ import annotations

import re
from pathlib import Path

import yaml
from yaml.constructor import ConstructorError
from yaml.events import AliasEvent

from iosislib.strategy.ir import Strategy, StrategyValidationError


class StrategySyntaxError(ValueError):
    """Raised when YAML cannot be read unambiguously and safely."""


class _StrategyLoader(yaml.SafeLoader):
    yaml_implicit_resolvers = {
        key: list(value) for key, value in yaml.SafeLoader.yaml_implicit_resolvers.items()
    }

    def compose_node(self, parent: yaml.Node | None, index: int) -> yaml.Node:
        event = self.peek_event()  # type: ignore[no-untyped-call]
        if isinstance(event, AliasEvent) or getattr(event, "anchor", None) is not None:
            raise ConstructorError(
                None,
                None,
                "YAML anchors and aliases are not supported in strategy documents",
                event.start_mark,
            )
        node = super().compose_node(parent, index)
        assert node is not None
        return node

    def construct_mapping(
        self, node: yaml.MappingNode, deep: bool = False
    ) -> dict[object, object]:
        self.flatten_mapping(node)
        result: dict[object, object] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                duplicate = key in result
            except TypeError as exc:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    "found an unhashable mapping key",
                    key_node.start_mark,
                ) from exc
            if duplicate:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found duplicate key {key!r}",
                    key_node.start_mark,
                )
            result[key] = self.construct_object(value_node, deep=deep)
        return result


for first_character, resolvers in tuple(_StrategyLoader.yaml_implicit_resolvers.items()):
    _StrategyLoader.yaml_implicit_resolvers[first_character] = [
        resolver
        for resolver in resolvers
        if resolver[0]
        not in {"tag:yaml.org,2002:bool", "tag:yaml.org,2002:timestamp"}
    ]

_StrategyLoader.add_implicit_resolver(  # type: ignore[no-untyped-call]
    "tag:yaml.org,2002:bool",
    re.compile(r"^(?:true|false)$", re.IGNORECASE),
    list("tTfF"),
)


class _StrategyDumper(yaml.SafeDumper):
    def ignore_aliases(self, data: object) -> bool:
        return True


def loads(text: str) -> Strategy:
    """Parse one strategy from a YAML string."""
    if not isinstance(text, str):
        raise TypeError("strategy YAML must be a string")
    try:
        data = yaml.load(text, Loader=_StrategyLoader)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        location = ""
        if mark is not None:
            location = f" at line {mark.line + 1}, column {mark.column + 1}"
        problem = getattr(exc, "problem", None) or str(exc)
        raise StrategySyntaxError(f"invalid strategy YAML{location}: {problem}") from exc
    if data is None:
        raise StrategyValidationError("$", "strategy document is empty")
    return Strategy.from_data(data)


def dumps(strategy: Strategy) -> str:
    """Serialize a strategy to deterministic, human-readable YAML."""
    if not isinstance(strategy, Strategy):
        raise TypeError("dumps requires a Strategy")
    return str(
        yaml.dump(
            strategy.to_data(),
            Dumper=_StrategyDumper,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
            width=100,
        )
    )


def load(path: str | Path) -> Strategy:
    """Read and parse one UTF-8 strategy file."""
    return loads(Path(path).read_text(encoding="utf-8"))


def dump(strategy: Strategy, path: str | Path) -> None:
    """Write one strategy file as UTF-8 YAML."""
    Path(path).write_text(dumps(strategy), encoding="utf-8")


__all__ = ["StrategySyntaxError", "dump", "dumps", "load", "loads"]
