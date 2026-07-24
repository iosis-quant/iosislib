from __future__ import annotations

from types import MappingProxyType

import pytest

from iosislib.strategy import (
    STRATEGY_FORMAT,
    STRATEGY_VERSION,
    Input,
    Node,
    Reference,
    Strategy,
    StrategySyntaxError,
    StrategyValidationError,
    dumps,
    loads,
    schema,
)


MINIMAL = """\
format: iosis.strategy
version: 0.1.0
name: probability-change
nodes:
  prices:
    op: source.market
    version: 0.1.0
    params:
      market: election
  log_odds:
    op: transform.logit
    version: 0.1.0
    inputs:
      probability: prices.probability
  change:
    op: transform.delta
    version: 0.1.0
    inputs:
      value:
        from: log_odds.log_odds
        tolerance: 5m
        nulls: fill
        fill: 0.0
    params:
      periods: 1
outputs:
  signal: change.change
"""


def test_parse_readable_strategy_and_round_trip_deterministically() -> None:
    strategy = loads(MINIMAL)

    assert strategy.format == STRATEGY_FORMAT
    assert strategy.version == STRATEGY_VERSION
    assert strategy.nodes["prices"].version == "0.1.0"
    assert strategy.topological_order == ("prices", "log_odds", "change")
    assert strategy.nodes["prices"].params["market"] == "election"
    assert strategy.nodes["log_odds"].inputs["probability"] == Input(
        Reference("prices", "probability")
    )
    change_input = strategy.nodes["change"].inputs["value"]
    assert change_input.tolerance == "5m"
    assert change_input.nulls == "fill"
    assert change_input.fill == 0.0

    first_dump = dumps(strategy)
    assert dumps(loads(first_dump)) == first_dump
    assert "probability: prices.probability" in first_dump
    assert "from: log_odds.log_odds" in first_dump


def test_strategy_values_are_immutable_and_fingerprint_is_order_independent() -> None:
    first = Strategy(
        name="stable",
        nodes={
            "source": Node("source.market", "0.1.0", params={"b": 2, "a": [1]}),
        },
        outputs={"result": Reference("source", "value")},
        metadata={"z": True, "a": "first"},
    )
    second = Strategy(
        name="stable",
        nodes={
            "source": Node("source.market", "0.1.0", params={"a": [1], "b": 2}),
        },
        outputs={"result": Reference("source", "value")},
        metadata={"a": "first", "z": True},
    )

    assert first.fingerprint == second.fingerprint
    assert isinstance(first.nodes, MappingProxyType)
    assert first.nodes["source"].params["a"] == (1,)
    with pytest.raises(TypeError):
        first.nodes["new"] = Node("source.other", "0.1.0")  # type: ignore[index]


def test_yaml_uses_unambiguous_boolean_and_string_rules() -> None:
    strategy = loads(
        MINIMAL.replace("market: election", "market: on\n      date: 2026-07-18")
    )

    assert strategy.nodes["prices"].params["market"] == "on"
    assert strategy.nodes["prices"].params["date"] == "2026-07-18"


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (
            MINIMAL.replace("name: probability-change", "name: first\nname: second"),
            "duplicate key",
        ),
        (
            MINIMAL.replace("market: election", "market: &market election\n      copy: *market"),
            "anchors and aliases are not supported",
        ),
    ],
)
def test_ambiguous_yaml_features_have_readable_syntax_errors(
    text: str, message: str
) -> None:
    with pytest.raises(StrategySyntaxError, match=message):
        loads(text)


@pytest.mark.parametrize(
    ("text", "path", "message"),
    [
        (
            MINIMAL.replace("periods: 1", "periods: 1\n    surprise: true"),
            "$.nodes.change",
            "unknown field",
        ),
        (
            MINIMAL.replace("log_odds.log_odds", "missing.value"),
            "$.nodes.change.inputs.value",
            "unknown node",
        ),
        (
            MINIMAL.replace("change.change", "missing.value"),
            "$.outputs.signal",
            "unknown node",
        ),
        (
            MINIMAL.replace("nulls: fill\n        fill: 0.0", "nulls: fill"),
            "$.nodes.change.inputs.value",
            "fill is required",
        ),
        (
            MINIMAL.replace("tolerance: 5m", "tolerance: null"),
            "$.nodes.change.inputs.value.tolerance",
            "omit this field",
        ),
    ],
)
def test_validation_errors_point_to_the_bad_declaration(
    text: str, path: str, message: str
) -> None:
    with pytest.raises(StrategyValidationError) as error:
        loads(text)

    assert error.value.path == path
    assert message in error.value.message


def test_cycles_and_unused_nodes_are_rejected() -> None:
    cycle = """\
format: iosis.strategy
version: 0.1.0
name: cycle
nodes:
  one:
    op: transform.one
    version: 0.1.0
    inputs: {value: two.value}
  two:
    op: transform.two
    version: 0.1.0
    inputs: {value: one.value}
outputs: {result: one.value}
"""
    unused = MINIMAL.replace(
        "outputs:\n",
        "  abandoned:\n    op: source.other\n    version: 0.1.0\noutputs:\n",
    )

    with pytest.raises(StrategyValidationError, match="cycle detected"):
        loads(cycle)
    with pytest.raises(StrategyValidationError, match="not reachable from outputs"):
        loads(unused)


def test_packaged_json_schema_describes_current_semver_format() -> None:
    document = schema()

    assert document["$schema"].endswith("2020-12/schema")
    assert document["properties"]["format"]["const"] == STRATEGY_FORMAT
    assert document["properties"]["version"]["const"] == STRATEGY_VERSION
    assert set(document["required"]) == {
        "format",
        "version",
        "name",
        "nodes",
        "outputs",
    }
    assert set(document["$defs"]["node"]["required"]) == {"op", "version"}


@pytest.mark.parametrize(
    ("old", "new", "path"),
    [
        ("version: 0.1.0", "version: v1", "$.version"),
        (
            "    version: 0.1.0\n    params:",
            "    version: v1\n    params:",
            "$.nodes.prices.version",
        ),
        (
            "    op: source.market\n",
            "    op: source.market/v1\n",
            "$.nodes.prices",
        ),
    ],
)
def test_legacy_or_invalid_versions_are_rejected(
    old: str,
    new: str,
    path: str,
) -> None:
    with pytest.raises(StrategyValidationError) as error:
        loads(MINIMAL.replace(old, new, 1))

    assert error.value.path == path
